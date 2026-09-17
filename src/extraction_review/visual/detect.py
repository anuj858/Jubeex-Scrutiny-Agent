"""OpenRouter page-image detector for legally relevant ink marks."""

from __future__ import annotations

import asyncio
import inspect
import json
import logging
import os
import re
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Any

import httpx

from ..document_parts import family_split_name
from ..llm import (
    DEFAULT_BASE_URL,
    DEFAULT_TIMEOUT_S,
    LLMError,
    _close_truncated_json,
    _extract_content,
    _headers,
    openrouter_api_key,
    openrouter_model,
    parse_openrouter_usage,
)
from ..process_file import FILE_DOWNLOAD_TIMEOUT_S, _require_pdf_bytes
from ..scrutiny.schema import LlmUsage
from .pages import coerce_pages_by_slot, global_page_sources, select_formality_pages
from .prompt import VISION_SYSTEM_PROMPT, user_prompt
from .references import load_reference_assets
from .render import jpeg_data_url, preview_dpi, render_page_jpeg
from .schema import (
    PROMPT_VERSION,
    SIGNATURE_ROLES,
    SUPPORTED_TYPES,
    TYPE_ALIASES,
    VISUAL_SCHEMA,
    VisualMark,
    VisualPageTarget,
    empty_visual_index,
)

logger = logging.getLogger(__name__)

DEFAULT_VISION_CONCURRENCY = 4
DEFAULT_VISION_MAX_TOKENS = 4096
ADVOCATE_LABELS = re.compile(
    r"\b(advocate(?:-on-record)?|aor|counsel|drawn\s*&\s*filed|filed\s+by)\b",
    re.IGNORECASE,
)
DEPONENT_LABELS = re.compile(r"\b(deponent|verification)\b", re.IGNORECASE)
PETITIONER_LABELS = re.compile(r"\b(petitioner|appellant|executant)\b", re.IGNORECASE)
RESPONDENT_LABELS = re.compile(r"\b(respondent|opposite\s+party)\b", re.IGNORECASE)
NOTARY_WORDS = re.compile(
    r"\b(notary|notarial|oath\s+commissioner|before\s+me|reg\.?\s*no)\b",
    re.IGNORECASE,
)


def visual_detection_enabled() -> bool:
    raw = (os.getenv("OPENROUTER_VISION_ENABLED") or "").strip().lower()
    if raw in {"0", "false", "no", "off"}:
        return False
    if raw in {"1", "true", "yes", "on"}:
        return bool(openrouter_api_key())
    return bool(openrouter_api_key())


def vision_model() -> str:
    return (os.getenv("OPENROUTER_VISION_MODEL") or openrouter_model()).strip()


def vision_fallback_model() -> str | None:
    raw = (os.getenv("OPENROUTER_VISION_FALLBACK_MODEL") or "").strip()
    return raw or None


def _int_env(name: str, default: int) -> int:
    raw = (os.getenv(name) or "").strip()
    try:
        return int(raw) if raw else default
    except ValueError:
        return default


def safe_normalized_bbox(value: Any) -> dict[str, float] | None:
    if isinstance(value, (list, tuple)) and len(value) == 4:
        try:
            x, y, width, height = (float(item) for item in value)
        except (TypeError, ValueError):
            return None
        value = {"x": x, "y": y, "width": width, "height": height}
    if not isinstance(value, dict):
        return None
    raw = value
    if "w" in raw and "width" not in raw:
        raw = {**raw, "width": raw.get("w")}
    if "h" in raw and "height" not in raw:
        raw = {**raw, "height": raw.get("h")}
    try:
        bbox = {key: float(raw[key]) for key in ("x", "y", "width", "height")}
    except (KeyError, TypeError, ValueError):
        return None
    tolerance = 0.002
    if any(
        bbox[key] < -tolerance or bbox[key] > 1 + tolerance
        for key in ("x", "y", "width", "height")
    ):
        return None
    bbox = {key: max(0.0, min(1.0, item)) for key, item in bbox.items()}
    if (
        bbox["x"] + bbox["width"] > 1 + tolerance
        or bbox["y"] + bbox["height"] > 1 + tolerance
    ):
        return None
    bbox["width"] = min(bbox["width"], 1 - bbox["x"])
    bbox["height"] = min(bbox["height"], 1 - bbox["y"])
    area = bbox["width"] * bbox["height"]
    if (
        bbox["width"] < 0.005
        or bbox["height"] < 0.005
        or area < 0.00005
        or area >= 0.85
    ):
        return None
    return {
        "x": round(bbox["x"], 6),
        "y": round(bbox["y"], 6),
        "w": round(bbox["width"], 6),
        "h": round(bbox["height"], 6),
    }


def _probable_type(raw: Any) -> str:
    probable_type = re.sub(r"[^a-z0-9]+", "_", str(raw or "").casefold()).strip("_")
    probable_type = TYPE_ALIASES.get(probable_type, probable_type)
    if probable_type not in SUPPORTED_TYPES:
        return "unknown_relevant_visual"
    return probable_type


def _signature_role(raw: Any, probable_type: str) -> str:
    if probable_type != "signature_like_mark":
        return "not_applicable"
    role = re.sub(r"[^a-z0-9]+", "_", str(raw or "").casefold()).strip("_")
    if role in {"aor", "counsel", "advocate_on_record"}:
        return "advocate"
    if role in SIGNATURE_ROLES:
        return role
    return "unknown"


def infer_signature_role(
    role: str,
    *,
    document_type: str,
    page_text: str,
    associated_label: str,
) -> str:
    if role not in {"unknown", "not_applicable"}:
        return role
    blob = f"{associated_label} {page_text}"
    hits = {
        "advocate": bool(ADVOCATE_LABELS.search(blob)),
        "deponent": bool(DEPONENT_LABELS.search(blob)),
        "petitioner": bool(PETITIONER_LABELS.search(blob)),
        "respondent": bool(RESPONDENT_LABELS.search(blob)),
    }
    present = [name for name, found in hits.items() if found]
    if len(present) == 1:
        return present[0]
    if document_type == "Main Petition":
        return "advocate"
    if role == "not_applicable":
        return role
    return "unknown"


def _strip_code_fence(text: str) -> str:
    cleaned = text.strip()
    if cleaned.startswith("```"):
        cleaned = cleaned.strip("`").strip()
        cleaned = cleaned.removeprefix("json").strip()
    return cleaned


def _extract_complete_objects(text: str) -> list[dict[str, Any]]:
    """Keep complete objects from a truncated JSON array; drop the cut-off tail."""
    items: list[dict[str, Any]] = []
    start = text.find("{")
    while start != -1:
        in_string = False
        escape = False
        depth = 0
        end: int | None = None
        for index, char in enumerate(text[start:], start):
            if in_string:
                if escape:
                    escape = False
                elif char == "\\":
                    escape = True
                elif char == '"':
                    in_string = False
                continue
            if char == '"':
                in_string = True
            elif char == "{":
                depth += 1
            elif char == "}":
                depth -= 1
                if depth == 0:
                    end = index
                    break
        if end is None:
            break
        try:
            value = json.loads(text[start : end + 1])
        except json.JSONDecodeError:
            break
        if isinstance(value, dict):
            items.append(value)
        start = text.find("{", end + 1)
    return items


def _parse_vision_json(content: str) -> Any:
    """Parse Gemini vision JSON: object, array, or truncated array of elements."""
    text = _strip_code_fence(content)
    if not text:
        raise LLMError("Model returned empty content")
    candidates = [text]
    repaired = _close_truncated_json(text)
    if repaired and repaired not in candidates:
        candidates.append(repaired)
    last_error: Exception | None = None
    for candidate in candidates:
        try:
            parsed = json.loads(candidate)
        except json.JSONDecodeError as exc:
            last_error = exc
            continue
        if isinstance(parsed, (dict, list)):
            return parsed
        last_error = LLMError(
            f"Response JSON was not a list or object: {content[:400]}"
        )
    complete = _extract_complete_objects(text)
    if complete:
        return complete
    raise LLMError(
        "Response was truncated or not JSON: " + content[:400]
    ) from last_error


def _is_chat_text_part(item: Any) -> bool:
    return (
        isinstance(item, dict)
        and "text" in item
        and "probable_type" not in item
        and item.get("type") in {None, "text"}
    )


def _elements_from_parsed(parsed: Any) -> list[Any]:
    if isinstance(parsed, list):
        if parsed and all(_is_chat_text_part(item) for item in parsed):
            joined = "".join(
                str(item.get("text") or "") for item in parsed if isinstance(item, dict)
            )
            if not joined.strip():
                return []
            return _elements_from_parsed(_parse_vision_json(joined))
        return parsed
    if isinstance(parsed, dict):
        elements = parsed.get("elements")
        if isinstance(elements, list):
            return elements
        if parsed.get("probable_type") is not None:
            return [parsed]
        return []
    if isinstance(parsed, str):
        if not parsed.strip():
            return []
        return _elements_from_parsed(_parse_vision_json(parsed))
    return []


def parse_vision_elements(
    content: Any,
    *,
    page_number: int,
    document_type: str,
    page_text: str = "",
    slot_id: str | None = None,
    local_page: int | None = None,
) -> list[VisualMark]:
    try:
        elements = _elements_from_parsed(content)
    except LLMError:
        elements = (
            _extract_complete_objects(content) if isinstance(content, str) else []
        )
    if not isinstance(elements, list):
        return []
    marks: list[VisualMark] = []
    for item in elements:
        if not isinstance(item, dict):
            continue
        bbox = safe_normalized_bbox(item.get("bbox_normalized") or item.get("bbox"))
        if bbox is None:
            continue
        probable_type = _probable_type(item.get("probable_type"))
        area = bbox["w"] * bbox["h"]
        aspect_ratio = bbox["w"] / max(bbox["h"], 0.000001)
        if (
            probable_type
            in {
                "notary_seal",
                "government_stamp",
                "revenue_stamp",
                "ordinary_seal_or_stamp",
            }
            and area > 0.45
        ):
            continue
        if probable_type == "signature_like_mark" and (
            area > 0.15 or (bbox["w"] > 0.5 and aspect_ratio > 15)
        ):
            continue
        try:
            confidence = float(item.get("confidence", 0.5))
        except (TypeError, ValueError):
            confidence = 0.5
        visible = item.get("visible_text")
        if isinstance(visible, list):
            visible_text = " ".join(
                str(value).strip() for value in visible if str(value).strip()
            )
        else:
            visible_text = str(visible or "")
        associated = str(item.get("associated_label") or item.get("near_label") or "")
        role = infer_signature_role(
            _signature_role(item.get("signature_role"), probable_type),
            document_type=document_type,
            page_text=page_text,
            associated_label=associated,
        )
        marks.append(
            VisualMark(
                page=page_number,
                document_type=document_type,
                marking_type=probable_type,
                signature_role=role,
                bbox=bbox,
                confidence=max(0.0, min(1.0, confidence)),
                visible_text=visible_text[:2000],
                associated_label=associated[:200],
                slot_id=slot_id,
                local_page=local_page,
            )
        )
    return marks


def _document_type_for_page(target: VisualPageTarget) -> str:
    if target.document_types:
        return target.document_types[0]
    return "unknown"


_AOR_SIGN_FAMILIES = frozenset(
    {
        "main petition",
        "advocate's checklist",
        "listing proforma",
        "aor's certificate",
        "application",
        "filing memo",
        "memo of parties",
        "vakalatnama",
        "memo of appearance",
        "annexures",
    }
)


def _folded_families(document_types: Sequence[str]) -> set[str]:
    families: set[str] = set()
    for name in document_types:
        cleaned = str(name).strip()
        if not cleaned:
            continue
        families.add(cleaned.casefold())
        families.add(family_split_name(cleaned).casefold())
    return families


def _references_for_target(
    assets: Sequence[Any], document_types: Sequence[str]
) -> list[Any]:
    names = _folded_families(document_types)
    selected: list[Any] = []
    for asset in assets:
        class_name = getattr(asset, "class_name", "")
        role = getattr(asset, "expected_role", "")
        if class_name == "notary_seal" and names & {"affidavit", "vakalatnama"}:
            selected.append(asset)
        elif class_name == "signature_like_mark" and role == "advocate":
            if names & _AOR_SIGN_FAMILIES:
                selected.append(asset)
        elif (
            class_name == "signature_like_mark"
            and role == "deponent"
            and names & {"affidavit", "vakalatnama"}
        ):
            selected.append(asset)
    return selected


def _expand_marks_for_parts(
    marks: list[VisualMark], target: VisualPageTarget
) -> list[VisualMark]:
    """Duplicate a detected box onto each formality label present on the page."""
    types = target.document_types or [_document_type_for_page(target)]
    expanded: list[VisualMark] = []
    for mark in marks:
        for document_type in types:
            payload = mark.model_dump()
            payload["document_type"] = document_type
            payload["signature_role"] = infer_signature_role(
                mark.signature_role,
                document_type=document_type,
                page_text=target.markdown,
                associated_label=mark.associated_label,
            )
            expanded.append(VisualMark.model_validate(payload))
    return expanded


async def _download_pdf(url: str, http: httpx.AsyncClient) -> bytes:
    response = await http.get(url)
    response.raise_for_status()
    data = response.content
    if not data:
        raise ValueError(f"Downloaded empty body from {url}")
    _require_pdf_bytes(data, url)
    return data


async def analyze_page_image(
    *,
    page_image: bytes,
    target: VisualPageTarget,
    references: Sequence[Any],
    model: str,
    http: httpx.AsyncClient,
) -> tuple[list[VisualMark], LlmUsage]:
    prompt = user_prompt(
        page_number=target.page,
        document_types=target.document_types,
        nearby_text=target.markdown,
    )
    content: list[dict[str, Any]] = [
        {"type": "text", "text": prompt},
        {"type": "image_url", "image_url": {"url": jpeg_data_url(page_image)}},
    ]
    for reference in references:
        content.append(
            {
                "type": "text",
                "text": (
                    f"REFERENCE SHEET {reference.reference_id}: "
                    f"{reference.description} Do not return any element from this image."
                ),
            }
        )
        content.append(
            {
                "type": "image_url",
                "image_url": {"url": jpeg_data_url(reference.content)},
            }
        )
    messages = [
        {"role": "system", "content": VISION_SYSTEM_PROMPT},
        {"role": "user", "content": content},
    ]
    max_tokens = _int_env("VISION_MAX_OUTPUT_TOKENS", DEFAULT_VISION_MAX_TOKENS)
    payload = {
        "model": model,
        "messages": messages,
        "temperature": 0.0,
        "max_tokens": max_tokens,
        "response_format": {"type": "json_object"},
    }
    base_url = os.getenv("OPENROUTER_BASE_URL", DEFAULT_BASE_URL).rstrip("/")
    response = await http.post(
        f"{base_url}/chat/completions",
        headers=_headers(),
        json=payload,
    )
    if response.status_code in (408, 409, 429) or response.status_code >= 500:
        raise LLMError(
            f"Retryable vision status {response.status_code}: {response.text[:300]}"
        )
    response.raise_for_status()
    body = response.json()
    if not isinstance(body, dict):
        raise LLMError("Vision response was not a JSON object")
    usage = parse_openrouter_usage(body, model=model)
    logger.info(
        "Vision page=%s model=%s tokens=%s cost=%s",
        target.page,
        usage.model or model,
        usage.total_tokens,
        usage.cost_usd,
    )
    try:
        parsed = _parse_vision_json(_extract_content(body))
    except LLMError as exc:
        raise LLMError(str(exc), usage=usage) from exc
    marks = parse_vision_elements(
        parsed,
        page_number=target.page,
        document_type=_document_type_for_page(target),
        page_text=target.markdown,
        slot_id=target.slot_id,
        local_page=target.local_page,
    )
    return marks, usage


async def _analyze_with_fallback(
    *,
    page_image: bytes,
    target: VisualPageTarget,
    references: Sequence[Any],
    http: httpx.AsyncClient,
) -> tuple[list[VisualMark], Exception | None, LlmUsage]:
    models = [vision_model()]
    fallback = vision_fallback_model()
    if fallback and fallback not in models:
        models.append(fallback)
    last_error: Exception | None = None
    combined = LlmUsage()
    for model in models:
        try:
            found, usage = await analyze_page_image(
                page_image=page_image,
                target=target,
                references=_references_for_target(references, target.document_types),
                model=model,
                http=http,
            )
            return found, None, combined.plus(usage)
        except LLMError as exc:
            last_error = exc
            combined = combined.plus(exc.usage)
            logger.warning(
                "Vision analysis failed page=%s model=%s: %s",
                target.page,
                model,
                str(exc)[:300],
            )
        except (
            httpx.HTTPError,
            OSError,
            ValueError,
            TypeError,
            KeyError,
        ) as exc:
            last_error = exc
            logger.warning(
                "Vision analysis failed page=%s model=%s: %s",
                target.page,
                model,
                str(exc)[:300],
            )
    if last_error:
        logger.warning(
            "Skipping visual marks for page %s after vision failures",
            target.page,
        )
    return [], last_error, combined


@dataclass(frozen=True)
class _PageVisionResult:
    marks: list[VisualMark]
    reason: str | None = None
    detail: str | None = None
    usage: LlmUsage | None = None


def _usage_payload(usage: LlmUsage | None) -> dict[str, Any] | None:
    if usage is None:
        return None
    if usage.calls <= 0 and usage.cost_usd is None and usage.total_tokens <= 0:
        return None
    return {
        "cost_usd": usage.cost_usd,
        "prompt_tokens": usage.prompt_tokens,
        "completion_tokens": usage.completion_tokens,
        "total_tokens": usage.total_tokens,
        "calls": usage.calls,
        "model": usage.model,
    }


def _openrouter_cost_note(usage: LlmUsage | None) -> str:
    if usage is None or usage.cost_usd is None:
        return ""
    return f", OpenRouter {usage.cost_usd:.6f} USD"


def _failure_row(
    target: VisualPageTarget, reason: str, detail: str | None = None
) -> dict[str, Any]:
    return {
        "page": target.page,
        "slot_id": target.slot_id,
        "document_types": list(target.document_types),
        "reason": reason,
        "detail": (detail or "")[:300] or None,
    }


def _dominant_error(failures: Sequence[Mapping[str, Any]]) -> str:
    counts: dict[str, int] = {}
    for item in failures:
        reason = str(item.get("reason") or "vision_failed")
        counts[reason] = counts.get(reason, 0) + 1
    if not counts:
        return "vision_failed"
    return max(counts.items(), key=lambda item: item[1])[0]


def _target_label(target: VisualPageTarget) -> str:
    types = ", ".join(target.document_types) or "unknown"
    return (
        f"page {target.page} {types} slot={target.slot_id or '-'} "
        f"local={target.local_page} file_id={target.file_id or 'no'} "
        f"url={'yes' if (target.file_url or '').strip() else 'no'}"
    )


async def _emit_log(on_log: Any, message: str) -> None:
    logger.info("%s", message)
    if on_log is None:
        return
    result = on_log(message)
    if inspect.isawaitable(result):
        await result


async def _default_download_file_id(file_id: str) -> bytes:
    from ..clients import get_llama_cloud_client
    from ..process_file import _download_file_bytes

    return await _download_file_bytes(get_llama_cloud_client(), file_id)


async def detect_visual_marks(
    *,
    parts: Sequence[Any],
    pages_by_slot: Mapping[str, Mapping[Any, Any]] | None,
    page_parts: Mapping[int, Sequence[str]],
    page_markdown: Mapping[int, str] | None = None,
    omit_empty: bool = False,
    on_log: Any = None,
    download_file_id: Any = None,
) -> dict[str, Any]:
    """Render selected formality pages and return a visual_index_v1 payload."""
    del page_parts
    enabled = visual_detection_enabled()
    await _emit_log(
        on_log,
        (
            f"Visual detection enabled={enabled} model={vision_model() or '-'} "
            f"api_key={'yes' if openrouter_api_key() else 'no'}"
        ),
    )
    if not enabled:
        return empty_visual_index(status="skipped", error="vision_disabled")
    slot_pages = coerce_pages_by_slot(pages_by_slot)
    sources = global_page_sources(parts, slot_pages, omit_empty=omit_empty)
    targets = select_formality_pages(sources, page_markdown=page_markdown)
    if not targets:
        await _emit_log(on_log, "Visual: no formality last pages present; skipping")
        return {
            "schema": VISUAL_SCHEMA,
            "prompt_version": PROMPT_VERSION,
            "status": "ok",
            "error": None,
            "pages": [],
            "targets": [],
            "failures": [],
            "marks": [],
        }

    await _emit_log(
        on_log,
        f"Visual: {len(targets)} formality page(s) selected",
    )
    for target in targets:
        await _emit_log(on_log, f"Visual target {_target_label(target)}")

    dpi = preview_dpi(os.getenv("VISION_PREVIEW_DPI"))
    concurrency = max(1, _int_env("VISION_CONCURRENCY", DEFAULT_VISION_CONCURRENCY))
    timeout = float(os.getenv("OPENROUTER_TIMEOUT_S", DEFAULT_TIMEOUT_S))
    references = load_reference_assets()
    pdf_by_slot: dict[str, bytes] = {}
    pdf_errors: dict[str, tuple[str, str]] = {}
    fetch_file = download_file_id or _default_download_file_id
    marks: list[VisualMark] = []
    analyzed_pages: list[int] = []
    failures: list[dict[str, Any]] = []

    async with httpx.AsyncClient(
        timeout=httpx.Timeout(max(timeout, FILE_DOWNLOAD_TIMEOUT_S)),
        follow_redirects=True,
    ) as http:
        for target in targets:
            slot = target.slot_id
            if slot in pdf_by_slot or slot in pdf_errors:
                continue
            url = (target.file_url or "").strip()
            file_id = (target.file_id or "").strip()
            if not url and not file_id:
                pdf_errors[slot] = ("missing_pdf", "slot has no file_url or file_id")
                await _emit_log(
                    on_log,
                    f"Visual download failed slot={slot}: missing file_url and file_id",
                )
                continue
            try:
                if url:
                    pdf_by_slot[slot] = await _download_pdf(url, http)
                else:
                    data = await fetch_file(file_id)
                    if not data:
                        raise ValueError(f"Empty PDF for file_id={file_id}")
                    _require_pdf_bytes(data, file_id)
                    pdf_by_slot[slot] = data
                await _emit_log(
                    on_log,
                    (f"Visual downloaded slot={slot} {len(pdf_by_slot[slot])} bytes"),
                )
            except ValueError as exc:
                reason = "not_pdf" if "is not a PDF" in str(exc) else "download_failed"
                pdf_errors[slot] = (reason, str(exc)[:300])
                await _emit_log(
                    on_log,
                    f"Visual download failed slot={slot}: {exc}",
                )
            except (httpx.HTTPError, OSError, RuntimeError, TypeError, KeyError) as exc:
                pdf_errors[slot] = ("download_failed", str(exc)[:300])
                await _emit_log(
                    on_log,
                    f"Visual download failed slot={slot}: {exc}",
                )

        semaphore = asyncio.Semaphore(concurrency)

        async def run_one(target: VisualPageTarget) -> _PageVisionResult:
            pdf_bytes = pdf_by_slot.get(target.slot_id)
            if not pdf_bytes:
                reason, detail = pdf_errors.get(
                    target.slot_id, ("missing_pdf", "PDF not loaded")
                )
                return _PageVisionResult(marks=[], reason=reason, detail=detail)
            try:
                image = await asyncio.to_thread(
                    render_page_jpeg, pdf_bytes, target.local_page, dpi=dpi
                )
            except (ValueError, OSError, RuntimeError) as exc:
                logger.warning(
                    "Could not render page %s (local %s) for vision",
                    target.page,
                    target.local_page,
                    exc_info=True,
                )
                return _PageVisionResult(
                    marks=[],
                    reason="render_failed",
                    detail=str(exc)[:300],
                )
            async with semaphore:
                found, vision_error, usage = await _analyze_with_fallback(
                    page_image=image,
                    target=target,
                    references=references,
                    http=http,
                )
            if vision_error is not None:
                return _PageVisionResult(
                    marks=[],
                    reason="vision_failed",
                    detail=str(vision_error)[:300],
                    usage=usage,
                )
            return _PageVisionResult(
                marks=_expand_marks_for_parts(found, target),
                usage=usage,
            )

        results = await asyncio.gather(
            *(run_one(target) for target in targets),
            return_exceptions=True,
        )

    succeeded = 0
    combined_usage = LlmUsage()
    for target, result in zip(targets, results, strict=True):
        analyzed_pages.append(target.page)
        if isinstance(result, Exception):
            logger.warning(
                "Visual detection failed for page %s: %s",
                target.page,
                result,
            )
            failures.append(_failure_row(target, "vision_failed", str(result)))
            await _emit_log(
                on_log,
                f"Visual page {target.page} failed: {result}",
            )
            continue
        if result.usage:
            combined_usage = combined_usage.plus(result.usage)
        if result.reason:
            failures.append(_failure_row(target, result.reason, result.detail))
            await _emit_log(
                on_log,
                (
                    f"Visual page {target.page} {result.reason}"
                    + (f": {result.detail}" if result.detail else "")
                    + _openrouter_cost_note(result.usage)
                ),
            )
            continue
        succeeded += 1
        marks.extend(result.marks)
        await _emit_log(
            on_log,
            (
                f"Visual page {target.page}: {len(result.marks)} mark(s)"
                + _openrouter_cost_note(result.usage)
            ),
        )

    status = "error" if succeeded == 0 else "ok"
    error = _dominant_error(failures) if status == "error" else None
    usage_payload = _usage_payload(combined_usage)
    await _emit_log(
        on_log,
        (
            f"Visual summary: {len(targets)} targets, {succeeded} ok, "
            f"{len(failures)} failed, {len(marks)} marks"
            + _openrouter_cost_note(combined_usage)
        ),
    )
    payload: dict[str, Any] = {
        "schema": VISUAL_SCHEMA,
        "prompt_version": PROMPT_VERSION,
        "status": status,
        "error": error,
        "pages": analyzed_pages,
        "targets": [
            {"page": target.page, "document_types": list(target.document_types)}
            for target in targets
        ],
        "failures": failures,
        "marks": [mark.model_dump() for mark in marks],
    }
    if usage_payload:
        payload["usage"] = usage_payload
    return payload
