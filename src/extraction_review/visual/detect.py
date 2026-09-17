"""OpenRouter page-image detector for legally relevant ink marks."""

from __future__ import annotations

import asyncio
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
    _extract_content,
    _headers,
    _parse_json,
    openrouter_api_key,
    openrouter_model,
    parse_openrouter_usage,
)
from ..process_file import FILE_DOWNLOAD_TIMEOUT_S, _require_pdf_bytes
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


def parse_vision_elements(
    content: Any,
    *,
    page_number: int,
    document_type: str,
    page_text: str = "",
    slot_id: str | None = None,
    local_page: int | None = None,
) -> list[VisualMark]:
    if isinstance(content, list):
        content = "".join(
            str(part.get("text") or "") for part in content if isinstance(part, dict)
        )
    if isinstance(content, dict):
        parsed = content
    else:
        if not isinstance(content, str) or not content.strip():
            return []
        try:
            parsed = _parse_json(content)
        except LLMError:
            return []
    elements = parsed.get("elements") if isinstance(parsed, dict) else None
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
) -> list[VisualMark]:
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
    parsed = _parse_json(_extract_content(body))
    return parse_vision_elements(
        parsed,
        page_number=target.page,
        document_type=_document_type_for_page(target),
        page_text=target.markdown,
        slot_id=target.slot_id,
        local_page=target.local_page,
    )


async def _analyze_with_fallback(
    *,
    page_image: bytes,
    target: VisualPageTarget,
    references: Sequence[Any],
    http: httpx.AsyncClient,
) -> list[VisualMark]:
    models = [vision_model()]
    fallback = vision_fallback_model()
    if fallback and fallback not in models:
        models.append(fallback)
    last_error: Exception | None = None
    for model in models:
        try:
            return await analyze_page_image(
                page_image=page_image,
                target=target,
                references=_references_for_target(references, target.document_types),
                model=model,
                http=http,
            )
        except (
            LLMError,
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
    return []


@dataclass(frozen=True)
class _PageVisionResult:
    marks: list[VisualMark]
    failed: bool


async def detect_visual_marks(
    *,
    parts: Sequence[Any],
    pages_by_slot: Mapping[str, Mapping[Any, Any]] | None,
    page_parts: Mapping[int, Sequence[str]],
    page_markdown: Mapping[int, str] | None = None,
    omit_empty: bool = False,
) -> dict[str, Any]:
    """Render selected formality pages and return a visual_index_v1 payload."""
    del page_parts
    if not visual_detection_enabled():
        return empty_visual_index(status="skipped", error="vision_disabled")
    slot_pages = coerce_pages_by_slot(pages_by_slot)
    sources = global_page_sources(parts, slot_pages, omit_empty=omit_empty)
    targets = select_formality_pages(sources, page_markdown=page_markdown)
    if not targets:
        return {
            "schema": VISUAL_SCHEMA,
            "prompt_version": PROMPT_VERSION,
            "status": "ok",
            "error": None,
            "pages": [],
            "targets": [],
            "marks": [],
        }

    dpi = preview_dpi(os.getenv("VISION_PREVIEW_DPI"))
    concurrency = max(1, _int_env("VISION_CONCURRENCY", DEFAULT_VISION_CONCURRENCY))
    timeout = float(os.getenv("OPENROUTER_TIMEOUT_S", DEFAULT_TIMEOUT_S))
    references = load_reference_assets()
    pdf_cache: dict[str, bytes] = {}
    marks: list[VisualMark] = []
    analyzed_pages: list[int] = []

    async with httpx.AsyncClient(
        timeout=httpx.Timeout(max(timeout, FILE_DOWNLOAD_TIMEOUT_S)),
        follow_redirects=True,
    ) as http:
        for target in targets:
            url = (target.file_url or "").strip()
            if not url or url in pdf_cache:
                continue
            try:
                pdf_cache[url] = await _download_pdf(url, http)
            except (httpx.HTTPError, ValueError, OSError):
                logger.warning(
                    "Could not download slot PDF for visual detection url=%s",
                    url,
                    exc_info=True,
                )

        semaphore = asyncio.Semaphore(concurrency)

        async def run_one(target: VisualPageTarget) -> _PageVisionResult:
            url = (target.file_url or "").strip()
            pdf_bytes = pdf_cache.get(url)
            if not pdf_bytes:
                return _PageVisionResult(marks=[], failed=True)
            try:
                image = await asyncio.to_thread(
                    render_page_jpeg, pdf_bytes, target.local_page, dpi=dpi
                )
            except (ValueError, OSError, RuntimeError):
                logger.warning(
                    "Could not render page %s (local %s) for vision",
                    target.page,
                    target.local_page,
                    exc_info=True,
                )
                return _PageVisionResult(marks=[], failed=True)
            async with semaphore:
                found = await _analyze_with_fallback(
                    page_image=image,
                    target=target,
                    references=references,
                    http=http,
                )
            return _PageVisionResult(
                marks=_expand_marks_for_parts(found, target), failed=False
            )

        results = await asyncio.gather(
            *(run_one(target) for target in targets),
            return_exceptions=True,
        )

    failed_all = True
    for target, result in zip(targets, results, strict=True):
        analyzed_pages.append(target.page)
        if isinstance(result, Exception):
            logger.warning(
                "Visual detection failed for page %s: %s",
                target.page,
                result,
            )
            continue
        if not result.failed:
            failed_all = False
        marks.extend(result.marks)

    status = "error" if failed_all else "ok"
    return {
        "schema": VISUAL_SCHEMA,
        "prompt_version": PROMPT_VERSION,
        "status": status,
        "error": "vision_failed" if status == "error" else None,
        "pages": analyzed_pages,
        "targets": [
            {"page": target.page, "document_types": list(target.document_types)}
            for target in targets
        ],
        "marks": [mark.model_dump() for mark in marks],
    }
