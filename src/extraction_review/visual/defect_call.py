"""One vision request answers the defects that share its pages.

Each physical page is uploaded once. A defect whose selected pages all fit in
the four-image cap is answered here and is not sent to the text model again.
"""

from __future__ import annotations

import json
import logging
import os
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from typing import Any

import httpx

from ..llm import (
    DEFAULT_BASE_URL,
    DEFAULT_TIMEOUT_S,
    LLMError,
    _extract_content,
    _headers,
    acquire_openrouter_slot,
    generate_vision_json,
    parse_openrouter_usage,
)
from ..scrutiny.rules import Defect
from ..scrutiny.schema import (
    BoundingBox,
    Coverage,
    DefectResponse,
    EvidenceRef,
    LlmUsage,
    VisualLocalization,
    build_finding,
)
from .defect_scope import DefectImagePage, DefectImagePlan
from .detect import (
    safe_normalized_bbox,
    vision_model,
    vision_provider,
    visual_detection_enabled,
)
from .prompt import document_page_label
from .render import jpeg_data_url, preview_dpi, render_page_jpeg

logger = logging.getLogger(__name__)

_STATUS_ALIASES = {
    "defect_found": "defect_found",
    "defect": "defect_found",
    "compliant": "compliant",
    "not_applicable": "not_applicable",
    "not_determined": "not_determined",
    "needs_review": "needs_review",
}
_GAPS = {
    "part_not_in_excerpts",
    "element_page_not_in_excerpts",
    "unreadable",
    "ambiguous",
    "conflict",
    "low_confidence",
    "legal_rule_unverified",
}


@dataclass
class VisionBatch:
    pages: list[DefectImagePage]
    check_ids: list[str]


@dataclass
class VisionAnswer:
    check_id: str
    response: DefectResponse
    marks: list[VisualLocalization] = field(default_factory=list)
    note: str = ""
    usage: LlmUsage = field(default_factory=LlmUsage)
    pages_seen: tuple[int, ...] = ()


def pack_vision_batches(plans: Sequence[DefectImagePlan]) -> list[VisionBatch]:
    """Keep one defect's pages together, and share a page when four images still fit."""
    batches: list[VisionBatch] = []
    for plan in plans:
        if not plan.pages:
            continue
        placed = False
        for batch in batches:
            union = {page.page for page in batch.pages} | {
                page.page for page in plan.pages
            }
            if len(union) > 4:
                continue
            known = {page.page for page in batch.pages}
            for page in plan.pages:
                if page.page not in known:
                    batch.pages.append(page)
            if plan.check_id not in batch.check_ids:
                batch.check_ids.append(plan.check_id)
            placed = True
            break
        if not placed:
            batches.append(
                VisionBatch(pages=list(plan.pages), check_ids=[plan.check_id])
            )
    return batches


def parse_defect_vision_payload(
    raw: str,
    *,
    pages: Sequence[DefectImagePage],
    check_ids: Sequence[str],
) -> dict[str, VisionAnswer]:
    """Turn one vision JSON object into answers keyed by check id."""
    try:
        payload = json.loads(raw)
    except json.JSONDecodeError:
        return {}
    if not isinstance(payload, dict):
        return {}
    rows = payload.get("answers")
    if rows is None and payload.get("check_id"):
        rows = [payload]
    if not isinstance(rows, list):
        return {}
    by_page = {page.page: page for page in pages}
    allowed = set(check_ids)
    found: dict[str, VisionAnswer] = {}
    for row in rows:
        if not isinstance(row, dict):
            continue
        check_id = str(row.get("check_id") or "").strip()
        if not check_id or (allowed and check_id not in allowed):
            continue
        if check_id in found:
            continue
        try:
            response = _response_from_row(row)
        except (TypeError, ValueError):
            continue
        found[check_id] = VisionAnswer(
            check_id=check_id,
            response=response,
            marks=_marks_from_row(row, by_page),
            note=_note_from_row(row),
            pages_seen=tuple(page.page for page in pages),
        )
    return found


def finding_from_vision(
    defect: Defect,
    answer: VisionAnswer,
    *,
    layout: Mapping[int, Mapping[str, object]] | None = None,
) -> Any:
    """Publish the vision answer as the defect finding."""
    pages = list(answer.pages_seen)
    coverage = Coverage(
        pages_reviewed=pages,
        document_found=True,
        required_scope_found=True,
        scope_complete=True,
        evidence_complete=True,
        visual_required=True,
        visual_complete=True,
        visual_pages_reviewed=pages,
    )
    finding = build_finding(
        defect,
        answer.response,
        evidence_ids=[],
        coverage=coverage,
        usage=answer.usage,
        layout=dict(layout or {}),
    )
    if answer.marks:
        finding = finding.model_copy(update={"visual_localizations": answer.marks})
    return finding


def vision_note_for_text(plan: DefectImagePlan, answer: VisionAnswer | None) -> str:
    """What the text call should already know about the imaged pages."""
    if answer is None:
        return ""
    lines: list[str] = []
    if answer.note:
        lines.append(answer.note)
    if plan.pages:
        seen = ", ".join(
            document_page_label([page.document_type], page.local_page)
            for page in plan.pages
        )
        lines.append(f"Page images already read: {seen}.")
    if plan.leftover_pages:
        rest = ", ".join(
            document_page_label([page.document_type], page.local_page)
            for page in plan.leftover_pages
        )
        lines.append(
            "These pages were not in the image request. Decide them from the text: "
            f"{rest}."
        )
    return "\n".join(lines).strip()


async def run_defect_vision(
    plans: Sequence[DefectImagePlan],
    *,
    defects_by_id: Mapping[str, Defect],
    model_tier: str | None,
    request_batch: Any = None,
) -> dict[str, VisionAnswer]:
    """Ask the vision model the defect questions for the packed pages."""
    if not visual_detection_enabled():
        return {}
    batches = pack_vision_batches(plans)
    if not batches:
        return {}
    try:
        model = vision_model(model_tier)
    except ValueError:
        logger.warning("Defect vision skipped because the model tier is missing")
        return {}
    sender = request_batch or _request_batch
    answers: dict[str, VisionAnswer] = {}
    usage_assigned: set[str] = set()
    plans_by_id = {plan.check_id: plan for plan in plans}
    for batch in batches:
        defects = [
            defects_by_id[check_id]
            for check_id in batch.check_ids
            if check_id in defects_by_id
        ]
        if not defects:
            continue
        try:
            raw, usage = await sender(
                pages=batch.pages,
                defects=defects,
                model=model,
            )
        except Exception:
            logger.exception(
                "Defect vision failed for %s",
                ", ".join(batch.check_ids),
            )
            continue
        parsed = parse_defect_vision_payload(
            raw, pages=batch.pages, check_ids=batch.check_ids
        )
        for check_id, answer in parsed.items():
            if check_id in answers:
                continue
            plan = plans_by_id.get(check_id)
            if plan is not None:
                answer.pages_seen = tuple(page.page for page in plan.pages)
            if check_id not in usage_assigned:
                answer.usage = usage
                usage_assigned.add(check_id)
            answers[check_id] = answer
    return answers


def _response_from_row(row: Mapping[str, Any]) -> DefectResponse:
    status = _STATUS_ALIASES.get(str(row.get("status") or "").strip().casefold())
    if status is None:
        raise ValueError("vision status is missing")
    try:
        confidence = float(row.get("confidence") if row.get("confidence") is not None else 0.5)
    except (TypeError, ValueError):
        confidence = 0.5
    confidence = min(1.0, max(0.0, confidence))
    observed = str(row.get("observed") or "").strip()
    summary = str(row.get("summary") or "").strip() or observed or "The page image was inspected."
    conclusion = str(row.get("conclusion") or "").strip()
    gap = str(row.get("gap") or "").strip() or None
    if status != "needs_review" or gap not in _GAPS:
        gap = None
    quotes = []
    evidence = row.get("evidence")
    if isinstance(evidence, list):
        for item in evidence:
            if isinstance(item, str) and item.strip():
                quotes.append(item.strip())
            elif isinstance(item, Mapping):
                quote = str(item.get("quote") or "").strip()
                if quote:
                    quotes.append(quote)
    if not quotes:
        quotes.append(observed or summary)
    return DefectResponse(
        check_id=str(row.get("check_id") or "").strip(),
        status=status,  # type: ignore[arg-type]
        confidence=confidence,
        summary=summary,
        observed=observed,
        conclusion=conclusion,
        gap=gap,  # type: ignore[arg-type]
        gap_detail=str(row.get("gap_detail") or "").strip(),
        review_reason=str(row.get("review_reason") or "").strip(),
        evidence=[EvidenceRef(quote=quote) for quote in quotes],
    )


def _marks_from_row(
    row: Mapping[str, Any],
    pages: Mapping[int, DefectImagePage],
) -> list[VisualLocalization]:
    raw_marks = row.get("marks")
    if not isinstance(raw_marks, list):
        return []
    marks: list[VisualLocalization] = []
    for item in raw_marks:
        if not isinstance(item, Mapping):
            continue
        bbox = safe_normalized_bbox(item.get("bbox_normalized") or item.get("bbox"))
        if bbox is None:
            continue
        try:
            page_number = int(item.get("page"))
        except (TypeError, ValueError):
            page_number = next(iter(pages), 0)
        target = pages.get(page_number)
        local_page = target.local_page if target is not None else page_number
        document_type = target.document_type if target is not None else ""
        probable = str(item.get("probable_type") or item.get("marking_type") or "")
        marking = _marking_type(probable)
        try:
            confidence = float(item.get("confidence")) if item.get("confidence") is not None else None
        except (TypeError, ValueError):
            confidence = None
        marks.append(
            VisualLocalization(
                page=local_page,
                document_type=document_type or None,
                marking_type=marking,
                signature_role=str(item.get("signature_role") or "").strip() or None,
                bounding_boxes=[
                    BoundingBox(
                        page=local_page,
                        x=bbox["x"],
                        y=bbox["y"],
                        w=bbox["w"],
                        h=bbox["h"],
                    )
                ],
                boxes_status="matched",
                confidence=confidence,
            )
        )
    return marks


def _marking_type(probable: str) -> str:
    folded = probable.casefold().replace("-", "_").replace(" ", "_")
    if "notary" in folded or folded in {"seal", "notary_seal"}:
        return "notary_seal"
    if "stamp" in folded:
        return "ordinary_seal_or_stamp"
    return "signature_like_mark"


def _note_from_row(row: Mapping[str, Any]) -> str:
    observed = str(row.get("observed") or "").strip()
    conclusion = str(row.get("conclusion") or "").strip()
    return " ".join(part for part in (observed, conclusion) if part).strip()


async def _request_batch(
    *,
    pages: Sequence[DefectImagePage],
    defects: Sequence[Defect],
    model: str,
) -> tuple[str, LlmUsage]:
    images = await _render_pages(pages)
    if len(images) != len(pages):
        raise LLMError("A selected page could not be rendered")
    user_text = _user_prompt(pages, defects)
    system_text = _system_prompt()
    if vision_provider() == "openrouter":
        return await _request_openrouter(
            pages=pages,
            images=images,
            system_text=system_text,
            user_text=user_text,
            model=model,
        )
    text, usage = await generate_vision_json(
        system_prompt=system_text,
        user_prompt=user_text,
        images=[(image, "image/jpeg") for image in images],
        model=model,
    )
    return text, usage


def _system_prompt() -> str:
    return (
        "You review Indian Supreme Court filing pages from the images in this "
        "request. Answer only the defect questions asked. Decide from the ink "
        "and text visible on those pages. Do not treat nearby OCR as proof when "
        "the image shows a different date or mark. Do not decide that a mark is "
        "genuine. Return one JSON object and nothing else. The object has "
        "`answers`, one item per asked check_id. Each item has check_id, status "
        "(defect_found, compliant, not_applicable, or needs_review), confidence "
        "from 0 to 1, summary, observed, conclusion, gap (null unless "
        "needs_review), gap_detail, review_reason, evidence as a list of "
        '{quote}, and marks. Each mark has page (the physical page number), '
        "probable_type, bbox_normalized with x, y, width, and height from 0 to 1, "
        "visible_text, and confidence. One mark is one signature, seal, or stamp. "
        "Leave marks empty when no such mark is visible."
    )


def _user_prompt(pages: Sequence[DefectImagePage], defects: Sequence[Defect]) -> str:
    inventory: list[str] = []
    for number, page in enumerate(pages, start=1):
        identity = document_page_label([page.document_type], page.local_page)
        inventory.append(
            f"DOCUMENT IMAGE {number}: this page is {identity}; "
            f"physical page={page.page} (return this number in page)"
        )
    questions: list[str] = []
    for defect in defects:
        questions.append(
            f"Check {defect.check_id}: {defect.defect.strip()} "
            f"Requirement: {(defect.requirement or '').strip()}"
        )
    return (
        "Inspect every DOCUMENT IMAGE independently. Answer every check from "
        "these images only.\n\n"
        + "\n".join(inventory)
        + "\n\n"
        + "\n".join(questions)
    )


async def _render_pages(pages: Sequence[DefectImagePage]) -> list[bytes]:
    from .detect import _default_download_file_id, _download_pdf

    dpi = preview_dpi(os.getenv("VISION_PREVIEW_DPI"))
    timeout = float(os.getenv("OPENROUTER_TIMEOUT_S", DEFAULT_TIMEOUT_S))
    pdf_by_slot: dict[str, bytes] = {}
    images: list[bytes] = []
    async with httpx.AsyncClient(timeout=httpx.Timeout(timeout), follow_redirects=True) as http:
        for page in pages:
            slot = page.slot_id or f"page-{page.page}"
            if slot not in pdf_by_slot:
                pdf_by_slot[slot] = await _load_pdf(page, http, _download_pdf, _default_download_file_id)
            images.append(
                render_page_jpeg(pdf_by_slot[slot], page.local_page, dpi=dpi)
            )
    return images


async def _load_pdf(page: DefectImagePage, http: httpx.AsyncClient, download_pdf: Any, download_file_id: Any) -> bytes:
    url = (page.file_url or "").strip()
    file_id = (page.file_id or "").strip()
    if url:
        return await download_pdf(url, http)
    if file_id:
        data = await download_file_id(file_id)
        if not data:
            raise LLMError(f"Empty PDF for file_id={file_id}")
        return data
    raise LLMError(f"Page {page.page} has no file to render")


async def _request_openrouter(
    *,
    pages: Sequence[DefectImagePage],
    images: Sequence[bytes],
    system_text: str,
    user_text: str,
    model: str,
) -> tuple[str, LlmUsage]:
    content: list[dict[str, Any]] = [{"type": "text", "text": user_text}]
    for image in images:
        content.append({"type": "image_url", "image_url": {"url": jpeg_data_url(image)}})
    raw_limit = (os.getenv("VISION_MAX_OUTPUT_TOKENS") or "").strip()
    per_page = int(raw_limit) if raw_limit else 2800
    payload = {
        "model": model,
        "messages": [
            {"role": "system", "content": system_text},
            {"role": "user", "content": content},
        ],
        "temperature": 0.0,
        "max_tokens": min(8000, per_page * max(1, len(pages))),
        "response_format": {"type": "json_object"},
    }
    base_url = os.getenv("OPENROUTER_BASE_URL", DEFAULT_BASE_URL).rstrip("/")
    await acquire_openrouter_slot(
        model_name=model,
        request_label="vision-defects:" + ",".join(str(page.page) for page in pages),
    )
    async with httpx.AsyncClient(timeout=httpx.Timeout(float(os.getenv("OPENROUTER_TIMEOUT_S", DEFAULT_TIMEOUT_S)))) as http:
        response = await http.post(
            f"{base_url}/chat/completions",
            headers=_headers(),
            json=payload,
        )
    if response.status_code >= 400:
        raise LLMError(f"Defect vision status {response.status_code}: {response.text[:300]}")
    body = response.json()
    usage = parse_openrouter_usage(body, model=model)
    return _extract_content(body), usage
