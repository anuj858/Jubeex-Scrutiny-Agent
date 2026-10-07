"""The `scrutiny_finding_v1` response contract.

`DefectResponse` is the narrow shape the model must return for a single defect.
Catalogue fields (defect, requirement, how to cure, rule, source) are copied
onto the finding in Python so they stay aligned with the defect API payload.
"""

from __future__ import annotations

import logging
import os
import re
from collections import Counter
from datetime import UTC, datetime
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field, ValidationError

from ..document_parts import (
    _part_match,
    allows_index_evidence,
    missing_required_parts,
    parts_named_in_where_to_look,
    parts_on_page,
    preferred_parts_for_defect,
)
from ..layout_index import boxes_on_pages, page_entry, part_for_page, quote_in_text
from .prompts import (
    display_cure_steps,
    filing_location,
    validated_reasoning,
    validated_summary,
)
from .rules import Catalogue, CatalogueSource, Defect, get_catalogue, source_match_tokens

logger = logging.getLogger(__name__)

ResultState = Literal[
    "defect_found",
    "compliant",
    "not_applicable",
    "not_determined",
    "needs_review",
]

BoxesStatus = Literal["matched", "page_only", "unavailable"]


class EvidenceRef(BaseModel):
    """A pointer back into the source document for one observation.

    This is the model-facing shape. Do not add coordinates here — the LLM
    schema is derived from DefectResponse. The model returns the excerpt
    id and a quote. The server adds the document page and boxes later.
    """

    model_config = ConfigDict(extra="forbid")

    chunk_id: str | None = Field(
        default=None,
        description=(
            "Excerpt label copied exactly from the header, such as c1. "
            "Null when the quote is not from an excerpt. Do not invent an id."
        ),
    )
    page: int | None = Field(
        default=None,
        description=(
            "Leave null. The document page is taken from chunk_id. "
            "Never copy a page number from Authority or location_source."
        ),
    )
    quote: str = Field(description="Verbatim excerpt supporting the finding")


class BoundingBox(BaseModel):
    """One highlight strip on a PDF page. Coordinates are normalized 0–1."""

    model_config = ConfigDict(extra="forbid")

    page: int
    x: float
    y: float
    w: float
    h: float


class FindingEvidence(BaseModel):
    """Evidence on a server-assembled finding, including highlight boxes."""

    model_config = ConfigDict(extra="forbid")

    page: int | None = None
    quote: str
    bounding_boxes: list[BoundingBox] = Field(default_factory=list)
    boxes_status: BoxesStatus = "unavailable"
    document_part: str | None = None
    slot_id: str | None = None


class VisualLocalization(BaseModel):
    """Ink-mark box attached to a visual catalogue defect after detection."""

    model_config = ConfigDict(
        json_schema_extra={
            "example": {
                "page": 20,
                "document_type": "Main Petition",
                "marking_type": "advocate_on_record_signature",
                "signature_role": "advocate",
                "bounding_boxes": [
                    {"page": 20, "x": 0.62, "y": 0.81, "w": 0.28, "h": 0.08}
                ],
                "boxes_status": "matched",
                "confidence": 0.86,
            }
        }
    )

    page: int | None = None
    document_type: str | None = None
    marking_type: str = Field(
        description=(
            "Attached values: advocate_on_record_signature, "
            "executant_signature, or notary_seal."
        )
    )
    signature_role: str | None = None
    bounding_boxes: list[BoundingBox] = Field(default_factory=list)
    boxes_status: BoxesStatus = "unavailable"
    confidence: float | None = None


class DefectResponse(BaseModel):
    """Exactly what the model returns for one defect."""

    model_config = ConfigDict(extra="forbid")

    check_id: str
    status: ResultState
    confidence: float = Field(ge=0.0, le=1.0)
    summary: str = Field(
        description=(
            "One plain sentence about this filing and this one defect. "
            "Do not copy the Standard paragraph or mention rulebook pages."
        )
    )
    reasoning: str = Field(
        description=(
            "2-4 plain sentences: which filing part and excerpt page were "
            "checked, and why this defect's requirement is met or not. "
            "Quote the filing. Do not mention location_source, handbook "
            "PDF pages, or catalogue check ids."
        )
    )
    evidence: list[EvidenceRef]


class Coverage(BaseModel):
    """How much of the document backed this finding."""

    model_config = ConfigDict(extra="forbid")

    chunks_reviewed: int = 0
    pages_reviewed: list[int] = Field(default_factory=list)
    structured_record_available: bool = False
    evidence_complete: bool = False


class LlmUsage(BaseModel):
    """Spend copied from OpenRouter `usage` on /chat/completions — never from the model JSON."""

    prompt_tokens: int = 0
    completion_tokens: int = 0
    total_tokens: int = 0
    cached_tokens: int = 0
    reasoning_tokens: int = 0
    calls: int = 0
    cost_usd: float | None = None
    model: str | None = None
    generation_id: str | None = None
    generation_ids: list[str] = Field(default_factory=list)

    def plus(self, other: LlmUsage) -> LlmUsage:
        cost: float | None = None
        if self.cost_usd is not None or other.cost_usd is not None:
            cost = (self.cost_usd or 0.0) + (other.cost_usd or 0.0)
        total = self.total_tokens + other.total_tokens
        if not total:
            total = (
                self.prompt_tokens
                + other.prompt_tokens
                + self.completion_tokens
                + other.completion_tokens
            )
        ids = [*(self.generation_ids or []), *(other.generation_ids or [])]
        last = other.generation_id or self.generation_id
        if last and last not in ids:
            ids.append(last)
        return LlmUsage(
            prompt_tokens=self.prompt_tokens + other.prompt_tokens,
            completion_tokens=self.completion_tokens + other.completion_tokens,
            total_tokens=total,
            cached_tokens=self.cached_tokens + other.cached_tokens,
            reasoning_tokens=self.reasoning_tokens + other.reasoning_tokens,
            calls=self.calls + other.calls,
            cost_usd=cost,
            model=other.model or self.model,
            generation_id=last,
            generation_ids=ids,
        )


class UsageByCheck(BaseModel):
    """One row in the cost breakdown, sorted highest charge first."""

    check_id: str
    cost_usd: float | None = None
    prompt_tokens: int = 0
    completion_tokens: int = 0
    total_tokens: int = 0
    llm_calls: int = 0
    share: float | None = None


class UsageSummary(BaseModel):
    """Roll-up of OpenRouter spend for the whole defect check."""

    cost_usd: float | None = None
    prompt_tokens: int = 0
    completion_tokens: int = 0
    total_tokens: int = 0
    cached_tokens: int = 0
    reasoning_tokens: int = 0
    llm_calls: int = 0
    model: str | None = None
    highest_cost_check_id: str | None = None
    highest_cost_usd: float | None = None
    by_check: list[UsageByCheck] = Field(default_factory=list)
    note: str = (
        "Exact OpenRouter usage.cost from each /chat/completions reply, "
        "summed for this check. The model is not asked for cost. LlamaParse, "
        "extract, classify, and Pinecone are billed separately."
    )


class SourceLocation(BaseModel):
    """Official rulebook for this defect: its link and the page in that document."""

    model_config = ConfigDict(extra="forbid")

    url: str | None = None
    page: int | None = None


_SOURCE_PAGE = re.compile(r"\bpages?\s+(\d+)\b", re.IGNORECASE)


def _source_page_number(text: str) -> int | None:
    match = _SOURCE_PAGE.search(text or "")
    if match is None:
        return None
    try:
        return int(match.group(1))
    except ValueError:
        return None


def _sources_named_in(text: str, sources: list[CatalogueSource]) -> list[CatalogueSource]:
    folded = text.casefold()
    cited: list[CatalogueSource] = []
    for source in sources:
        if source.source_id and source.source_id in text:
            cited.append(source)
            continue
        if any(url and url in text for url in source.urls()):
            cited.append(source)
            continue
        if any(token.casefold() in folded for token in source_match_tokens(source)):
            cited.append(source)
    return cited


def official_source_locations(
    defect: Defect, catalogue: Catalogue | None
) -> list[SourceLocation]:
    """Link and page of each official source named on this defect."""
    raw = (defect.location_text or defect.location_source or "").strip()
    if not raw or "did not give" in raw.lower():
        return []
    sources = list(catalogue.sources) if catalogue is not None else []
    found: list[SourceLocation] = []
    seen: set[tuple[str | None, int | None]] = set()
    lines = [line.strip() for line in re.split(r"[\n;]+", raw) if line.strip()]
    for line in lines:
        page = _source_page_number(line)
        matched = _sources_named_in(line, sources)
        if not matched:
            if page is None:
                continue
            key = (None, page)
            if key not in seen:
                seen.add(key)
                found.append(SourceLocation(page=page))
            continue
        for source in matched:
            key = (source.url, page)
            if key in seen:
                continue
            seen.add(key)
            found.append(SourceLocation(url=source.url, page=page))
    return found


class DefectFinding(BaseModel):
    """Server-assembled finding for one catalogue defect."""

    check_id: str
    defect: str = Field(
        description=(
            "Exact catalogue defect text from sci_registry_defects.v1.json. "
            "Copied in Python; the model is not asked for this string."
        )
    )
    requirement: str = Field(
        description=(
            "Exact catalogue requirement text from sci_registry_defects.v1.json. "
            "Copied in Python; the model is not asked for this string."
        )
    )
    main_category: str
    special_category: str | None = None
    status: ResultState
    summary: str
    confidence: float
    reasoning: str
    evidence: list[FindingEvidence] = Field(default_factory=list)
    how_to_cure: list[str] = Field(default_factory=list)
    applicable_rule: str | None = None
    location: str | None = Field(
        default=None,
        description=(
            "Page inside the document type, e.g. 'Vakalatnama, page 1.' "
            "This is not the stitched filing page. If the page is unknown: "
            "'Page missing — …'."
        ),
    )
    location_source: list[SourceLocation] = Field(default_factory=list)
    evidence_ids: list[str] = Field(default_factory=list)
    visual_localizations: list[VisualLocalization] = Field(default_factory=list)
    coverage: Coverage = Field(default_factory=Coverage)
    usage: LlmUsage | None = None
    error: str | None = None
    defect_version: int = 1


DEFAULT_REVIEW_CONFIDENCE = 0.6


def _safe_page_number(value: Any) -> int | None:
    try:
        if value is None or isinstance(value, bool):
            return None
        return int(value)
    except (TypeError, ValueError):
        return None


def review_confidence_threshold() -> float:
    raw = os.getenv("SCRUTINY_REVIEW_CONFIDENCE", "")
    try:
        value = float(raw) if raw else DEFAULT_REVIEW_CONFIDENCE
    except ValueError:
        value = DEFAULT_REVIEW_CONFIDENCE
    return min(1.0, max(0.0, value))


def _retrieved_pages(chunks: list[dict[str, Any]]) -> set[int]:
    pages: set[int] = set()
    for chunk in chunks:
        if chunk.get("chunk_kind") == "summary":
            continue
        start = chunk.get("page")
        if start is None:
            continue
        try:
            start_i = int(start)
        except (TypeError, ValueError):
            continue
        end = chunk.get("page_end")
        try:
            end_i = int(end) if end is not None else start_i
        except (TypeError, ValueError):
            end_i = start_i
        end_i = max(end_i, start_i)
        pages.update(range(start_i, end_i + 1))
    return pages


def _chunk_page_for_quote(quote: str, chunk: dict[str, Any]) -> int | None:
    text = chunk.get("text") or ""
    if not quote_in_text(quote, text):
        return None
    start = chunk.get("page")
    try:
        return int(start) if start is not None else None
    except (TypeError, ValueError):
        return None


def _chunk_part_rank(chunk: dict[str, Any], preferred: list[str]) -> int:
    """Higher is better: target part beat Index listing lines."""
    names = [n.lower() for n in parts_on_page(chunk.get("document_part"))]
    if not names:
        return 0
    if preferred and _part_match(chunk, preferred):
        return 2
    if set(names) <= {"index"}:
        return -1
    return 1


def _page_allowed_for_evidence(
    page_chunks: list[dict[str, Any]],
    page: int | None,
    preferred: list[str],
    allow_index: bool,
) -> bool:
    if page is None:
        return False
    for chunk in page_chunks:
        try:
            chunk_page = int(chunk["page"]) if chunk.get("page") is not None else None
        except (TypeError, ValueError):
            continue
        if chunk_page != page:
            continue
        if allow_index or _chunk_part_rank(chunk, preferred) >= 0:
            return True
    return False


_SNIPPET_MAX = 280


def _snippet_for_boxes(text: str) -> str:
    collapsed = " ".join((text or "").split())
    if len(collapsed) <= _SNIPPET_MAX:
        return collapsed
    return collapsed[-_SNIPPET_MAX:]


def _evidence_from_chunks(
    chunks: list[dict[str, Any]], defect: Defect
) -> list[EvidenceRef]:
    """When the model cites nothing, box the inspected pages we actually retrieved."""
    preferred = parts_named_in_where_to_look(defect) or preferred_parts_for_defect(
        defect
    )
    page_chunks = [chunk for chunk in chunks if chunk.get("chunk_kind") != "summary"]
    ranked = sorted(
        page_chunks,
        key=lambda chunk: (
            -_chunk_part_rank(chunk, preferred),
            _safe_page_number(chunk.get("page")) or 999999,
        ),
    )
    refs: list[EvidenceRef] = []
    seen: set[int] = set()
    for chunk in ranked:
        if preferred and _chunk_part_rank(chunk, preferred) < 2:
            continue
        try:
            page = _safe_page_number(chunk.get("page"))
        except (TypeError, ValueError):
            page = None
        if page is None or page in seen:
            continue
        quote = _snippet_for_boxes(str(chunk.get("text") or ""))
        if not quote:
            continue
        seen.add(page)
        refs.append(
            EvidenceRef(
                page=page,
                quote=quote,
                chunk_id=str(chunk.get("excerpt_id") or "") or None,
            )
        )
        break
    return refs


def _as_bounding_boxes(raw: Any) -> list[BoundingBox]:
    boxes: list[BoundingBox] = []
    if not isinstance(raw, list):
        return boxes
    for item in raw:
        try:
            box = BoundingBox.model_validate(item)
        except ValidationError:
            continue
        x = float(box.x)
        y = float(box.y)
        width = float(box.w)
        height = float(box.h)
        if width <= 0 or height <= 0:
            continue
        max_x = x + width
        max_y = y + height
        if x > 1.5 or y > 1.5 or max_x > 1.5 or max_y > 1.5:
            scale_x = max(max_x, 1.0)
            scale_y = max(max_y, 1.0)
            x /= scale_x
            width /= scale_x
            y /= scale_y
            height /= scale_y
        x = min(max(x, 0.0), 1.0)
        y = min(max(y, 0.0), 1.0)
        width = min(max(width, 0.0), round(1.0 - x, 6))
        height = min(max(height, 0.0), round(1.0 - y, 6))
        if width <= 0:
            width = min(0.001, 1.0 - x) if x < 1.0 else 0.001
            x = min(x, 1.0 - width)
        if height <= 0:
            height = min(0.001, 1.0 - y) if y < 1.0 else 0.001
            y = min(y, 1.0 - height)
        boxes.append(
            BoundingBox(
                page=int(box.page),
                x=round(x, 6),
                y=round(y, 6),
                w=round(width, 6),
                h=round(height, 6),
            )
        )
    return boxes


def _local_pages(
    layout: dict[int, dict[str, Any]] | None,
    global_pages: list[int],
) -> list[int]:
    """Map stitched filing pages to each document's own page. Drop unmapped pages."""
    if not layout:
        return []
    local: list[int] = []
    for global_page in global_pages:
        meta = page_entry(layout, global_page)
        number = _safe_page_number((meta or {}).get("local_page")) if meta else None
        if number is not None and number not in local:
            local.append(number)
    return local


def _document_type_for_page(
    page_meta: dict[str, Any] | None,
    chunks: list[dict[str, Any]] | None,
    global_page: int | None,
) -> str | None:
    """Document type for this layout page. The slot label wins over the chunk."""
    raw = (page_meta or {}).get("document_part")
    if isinstance(raw, list):
        names = [str(name).strip() for name in raw if str(name).strip()]
        if len(names) == 1:
            return names[0]
        if names:
            return ", ".join(names)
    text = str(raw or "").strip()
    if text:
        return text
    return part_for_page(chunks, global_page)


def _publish_local_evidence(
    *,
    local_page: int | None,
    boxes: list[BoundingBox],
    boxes_status: BoxesStatus,
) -> tuple[int | None, list[BoundingBox], BoxesStatus]:
    """API page is the page inside the document type.

    The stitched filing page is never returned. A missing box does not clear
    a known document page. Coordinates are sent only when the quote matched.
    """
    if local_page is None:
        return None, [], "unavailable"
    if boxes_status != "matched" or not boxes:
        return local_page, [], "page_only"
    return (
        local_page,
        [box.model_copy(update={"page": local_page}) for box in boxes],
        "matched",
    )


def _excerpt_label(chunk: dict[str, Any]) -> str:
    return str(chunk.get("excerpt_id") or "").strip()


def _chunk_span(chunk: dict[str, Any]) -> list[int]:
    """The chunk's own page.

    Borrowed overlap is marked ``--- from page N ---`` for retrieval only.
    A quote taken from that margin still cites this page, not the neighbour.
    """
    start = _safe_page_number(chunk.get("page"))
    if start is None:
        return []
    return [start]


def _choose_evidence_chunk(
    ref: EvidenceRef,
    page_chunks: list[dict[str, Any]],
) -> dict[str, Any] | None:
    """Pick the sent chunk that owns this citation.

    A matching quote wins over a wrong excerpt id. A known id is kept when
    the quote is not in any sent chunk, so the document page can still be
    returned. An unknown id with no quote match returns nothing.
    """
    cited_id = re.sub(r"[\[\]]", "", (ref.chunk_id or "")).split("|")[0].strip().lower()
    cited = None
    if cited_id:
        for chunk in page_chunks:
            if _excerpt_label(chunk).strip().lower() == cited_id:
                cited = chunk
                break
    quote = (ref.quote or "").strip()
    quote_hits = [
        chunk
        for chunk in page_chunks
        if quote and quote_in_text(quote, str(chunk.get("text") or ""))
    ]
    if cited is not None and cited in quote_hits:
        return cited
    if quote_hits:
        for chunk in quote_hits:
            if ref.page is not None and _chunk_page_for_quote(quote, chunk) == ref.page:
                return chunk
        return quote_hits[0]
    if cited is not None:
        return cited
    return None


def attach_evidence_boxes(
    refs: list[EvidenceRef],
    *,
    layout: dict[int, dict[str, Any]] | None = None,
    chunks: list[dict[str, Any]] | None = None,
) -> list[FindingEvidence]:
    """Open the cited chunk's pages only. Never scan the rest of the filing."""
    attached: list[FindingEvidence] = []
    page_chunks = [
        chunk for chunk in (chunks or []) if chunk.get("chunk_kind") != "summary"
    ]
    for ref in refs:
        try:
            chunk = _choose_evidence_chunk(ref, page_chunks)
            if chunk is None:
                attached.append(
                    FindingEvidence(
                        page=None,
                        quote=ref.quote or "",
                        bounding_boxes=[],
                        boxes_status="unavailable",
                    )
                )
                continue
            boxes, status, page = boxes_on_pages(
                ref.quote or "",
                _chunk_span(chunk),
                layout,
            )
            boxes_status: BoxesStatus = (
                status
                if status in ("matched", "page_only", "unavailable")
                else "unavailable"
            )
            page_meta = page_entry(layout, page) if layout else None
            local_page = None
            slot_id = None
            if page_meta is not None:
                slot_id = str(page_meta.get("slot_id") or "").strip() or None
                local_page = _safe_page_number(page_meta.get("local_page"))
            document_part = _document_type_for_page(page_meta, chunks, page)
            if not document_part:
                document_part = str(chunk.get("document_part") or "").strip() or None
            published_page, published_boxes, boxes_status = _publish_local_evidence(
                local_page=local_page,
                boxes=_as_bounding_boxes(boxes),
                boxes_status=boxes_status,
            )
            attached.append(
                FindingEvidence(
                    page=published_page,
                    quote=ref.quote or "",
                    bounding_boxes=published_boxes,
                    boxes_status=boxes_status,
                    document_part=document_part,
                    slot_id=slot_id,
                )
            )
        except Exception:
            logger.warning(
                "Failed to attach highlight boxes; leaving citation unboxed",
                exc_info=True,
            )
            attached.append(
                FindingEvidence(
                    page=None,
                    quote=(getattr(ref, "quote", None) or ""),
                    bounding_boxes=[],
                    boxes_status="unavailable",
                )
            )
    return attached


def _one_published_evidence(
    evidence: list[FindingEvidence],
) -> list[FindingEvidence]:
    """The API finding carries one citation. A matched box wins."""
    if len(evidence) <= 1:
        return evidence

    def rank(index: int, item: FindingEvidence) -> tuple[int, int, int]:
        matched = 1 if item.boxes_status == "matched" and item.bounding_boxes else 0
        has_page = 1 if item.page is not None else 0
        return (matched, has_page, -index)

    chosen = max(range(len(evidence)), key=lambda index: rank(index, evidence[index]))
    return [evidence[chosen]]


def apply_evidence_pages(
    response: DefectResponse,
    chunks: list[dict[str, Any]],
    defect: Defect | None = None,
) -> DefectResponse:
    """Snap evidence.page to retrieved filing pages; drop rulebook / Index noise."""
    if not response.evidence:
        return response
    page_chunks = [c for c in chunks if c.get("chunk_kind") != "summary"]
    preferred: list[str] = []
    allow_index = True
    if defect is not None:
        preferred = parts_named_in_where_to_look(defect) or preferred_parts_for_defect(
            defect
        )
        allow_index = allows_index_evidence(defect)
    grounded: list[EvidenceRef] = []
    for ref in response.evidence:
        quote = ref.quote or ""
        matches = [
            chunk
            for chunk in page_chunks
            if _chunk_page_for_quote(quote, chunk) is not None
        ]
        if not allow_index:
            content_matches = [
                chunk for chunk in matches if _chunk_part_rank(chunk, preferred) >= 0
            ]
            # Index listing lines are not proof for content checks — drop them.
            if matches and not content_matches:
                continue
            matches = content_matches
        page: int | None = None
        if matches:
            matches.sort(
                key=lambda chunk: (
                    _chunk_part_rank(chunk, preferred),
                    1 if ref.page == _chunk_page_for_quote(quote, chunk) else 0,
                ),
                reverse=True,
            )
            page = _chunk_page_for_quote(quote, matches[0])
        elif ref.page in _retrieved_pages(chunks) and _page_allowed_for_evidence(
            page_chunks, ref.page, preferred, allow_index
        ):
            # Keep a retrieved page so layout matching can still box the quote
            # when Pinecone OCR differs from the model quote.
            page = ref.page
        grounded.append(
            EvidenceRef(page=page, quote=quote, chunk_id=ref.chunk_id)
        )
    response.evidence = grounded
    return response


def apply_status_policy(response: DefectResponse) -> DefectResponse:
    """Move low-confidence defect/compliant calls to needs_review."""
    if (
        response.status in ("defect_found", "compliant")
        and response.confidence < review_confidence_threshold()
    ):
        response.status = "needs_review"
    return response


def apply_retrieval_policy(
    defect: Defect,
    response: DefectResponse,
    chunks: list[dict[str, Any]],
) -> DefectResponse:
    """Do not treat a missing document part in the excerpt set as a defect."""
    if response.status != "defect_found":
        return response
    missing = missing_required_parts(defect, chunks)
    if missing:
        response.status = "needs_review"
    return response


def apply_undetermined_policy(
    defect: Defect,
    response: DefectResponse,
    chunks: list[dict[str, Any]],
) -> DefectResponse:
    """Stamps, seals, margins and other marks are defects when not found.

    `not_determined` is reserved for checks that never ran. If the model
    used it because a visual/layout requirement was hard to see, treat
    absence in the inspected part as a defect. If that part was never
    retrieved, keep needs_review.
    """
    if response.status != "not_determined":
        return response
    if missing_required_parts(defect, chunks):
        response.status = "needs_review"
        return response
    response.status = "defect_found"
    return response


class ScrutinySummary(BaseModel):
    total_defects: int
    defects_found: int
    compliant: int
    needs_review: int
    not_determined: int
    not_applicable: int
    overall_confidence: float


class ScrutinyReport(BaseModel):
    schema_name: Literal["scrutiny_finding_v1"] = "scrutiny_finding_v1"
    catalogue_id: str
    catalogue_version: str
    agent_data_id: str | None = None
    file_hash: str | None = None
    petition_type: str | None = None
    model: str | None = None
    generated_at: str = Field(default_factory=lambda: datetime.now(UTC).isoformat())
    disclaimer: str | None = None
    findings: list[DefectFinding] = Field(default_factory=list)
    summary: ScrutinySummary
    usage: UsageSummary | None = None
    planned_checks: int | None = None
    stopped_early: bool = False


def _pages_from_chunks(chunks: list[dict[str, Any]] | None) -> list[int]:
    pages: list[int] = []
    for chunk in chunks or []:
        if chunk.get("chunk_kind") == "summary":
            continue
        try:
            page = chunk.get("page")
            if page is None:
                continue
            page_i = int(page)
        except (TypeError, ValueError):
            continue
        if page_i not in pages:
            pages.append(page_i)
    return pages


def _parts_for_pages(
    chunks: list[dict[str, Any]] | None, pages: list[int]
) -> list[str]:
    if not chunks or not pages:
        names: list[str] = []
        for chunk in chunks or []:
            if chunk.get("chunk_kind") == "summary":
                continue
            name = str(chunk.get("document_part") or "").strip()
            if name and name not in names:
                names.append(name)
        return names
    wanted = set(pages)
    names: list[str] = []
    for chunk in chunks:
        if chunk.get("chunk_kind") == "summary":
            continue
        page = chunk.get("page")
        try:
            page_i = int(page) if page is not None else None
        except (TypeError, ValueError):
            page_i = None
        if page_i not in wanted:
            continue
        name = str(chunk.get("document_part") or "").strip()
        if name and name not in names:
            names.append(name)
    return names


def build_finding(
    defect: Defect,
    response: DefectResponse,
    *,
    evidence_ids: list[str],
    coverage: Coverage,
    usage: LlmUsage | None = None,
    chunks: list[dict[str, Any]] | None = None,
    layout: dict[int, dict[str, Any]] | None = None,
    visual_index: dict[str, Any] | None = None,
    page_parts: dict[int, list[str]] | None = None,
    record: dict[str, Any] | None = None,
) -> DefectFinding:
    catalogue = get_catalogue()
    refs = list(response.evidence)
    if not refs and chunks:
        refs = _evidence_from_chunks(chunks, defect)
    evidence = _one_published_evidence(
        attach_evidence_boxes(refs, layout=layout, chunks=chunks)
    )
    evidence_pages = [ref.page for ref in evidence if ref.page is not None]
    reviewed_global = list(coverage.pages_reviewed) or _pages_from_chunks(chunks)
    reviewed_local = _local_pages(layout, reviewed_global)
    pages = reviewed_local
    seen_parts: list[str] = []
    for ref in evidence:
        if ref.document_part and ref.document_part not in seen_parts:
            seen_parts.append(ref.document_part)
    if not seen_parts:
        seen_parts = _parts_for_pages(chunks, reviewed_global)
    location = filing_location(
        evidence_pages=evidence_pages,
        reviewed_pages=reviewed_local or evidence_pages,
        document_parts=seen_parts,
    )
    coverage = coverage.model_copy(update={"pages_reviewed": reviewed_local})
    from ..visual.attach import attach_visual_localizations

    visual_localizations = attach_visual_localizations(
        defect,
        visual_index=visual_index,
        page_parts=page_parts,
        record=record,
    )
    return DefectFinding(
        check_id=defect.check_id,
        defect=defect.defect,
        requirement=defect.requirement,
        main_category=defect.main_category,
        special_category=defect.special_category,
        status=response.status,
        summary=validated_summary(
            defect,
            response.summary,
            response.status,
            pages=pages,
            evidence_pages=evidence_pages,
        ),
        confidence=response.confidence,
        reasoning=validated_reasoning(
            defect,
            response.reasoning,
            response.status,
            pages=pages,
            evidence_pages=evidence_pages,
        ),
        evidence=evidence,
        how_to_cure=display_cure_steps(defect.how_to_cure),
        applicable_rule=defect.applicable_rule,
        location=location,
        location_source=official_source_locations(defect, catalogue),
        evidence_ids=evidence_ids,
        visual_localizations=visual_localizations,
        coverage=coverage,
        usage=usage,
        defect_version=defect.defect_version,
    )


def failed_finding(
    defect: Defect,
    error: str,
    usage: LlmUsage | None = None,
    *,
    visual_index: dict[str, Any] | None = None,
    page_parts: dict[int, list[str]] | None = None,
    record: dict[str, Any] | None = None,
) -> DefectFinding:
    """Placeholder when a defect could not be evaluated at all."""
    catalogue = get_catalogue()
    from ..visual.attach import attach_visual_localizations

    visual_localizations = attach_visual_localizations(
        defect,
        visual_index=visual_index,
        page_parts=page_parts,
        record=record,
    )
    return DefectFinding(
        check_id=defect.check_id,
        defect=defect.defect,
        requirement=defect.requirement,
        main_category=defect.main_category,
        special_category=defect.special_category,
        status="not_determined",
        summary=f"This check could not be completed: {error}",
        confidence=0.0,
        reasoning=f"This check did not run: {error}",
        evidence=[],
        visual_localizations=visual_localizations,
        how_to_cure=display_cure_steps(defect.how_to_cure),
        applicable_rule=defect.applicable_rule,
        location="Filing page missing — this check did not run.",
        location_source=official_source_locations(defect, catalogue),
        error=error,
        usage=usage,
        defect_version=defect.defect_version,
    )


def summarize_usage(
    findings: list[DefectFinding], *, model: str | None
) -> UsageSummary:
    combined = LlmUsage(model=model)
    for finding in findings:
        if finding.usage:
            combined = combined.plus(finding.usage)

    rows: list[UsageByCheck] = []
    for finding in findings:
        usage = finding.usage or LlmUsage()
        share = None
        if combined.cost_usd and usage.cost_usd is not None:
            share = round(usage.cost_usd / combined.cost_usd, 4)
        elif combined.total_tokens and usage.total_tokens:
            share = round(usage.total_tokens / combined.total_tokens, 4)
        rows.append(
            UsageByCheck(
                check_id=finding.check_id,
                cost_usd=usage.cost_usd,
                prompt_tokens=usage.prompt_tokens,
                completion_tokens=usage.completion_tokens,
                total_tokens=usage.total_tokens,
                llm_calls=usage.calls,
                share=share,
            )
        )

    def _rank(row: UsageByCheck) -> tuple[float, int]:
        return (row.cost_usd if row.cost_usd is not None else -1.0, row.total_tokens)

    rows.sort(key=_rank, reverse=True)
    top = rows[0] if rows else None
    has_cost = top is not None and top.cost_usd is not None
    return UsageSummary(
        cost_usd=combined.cost_usd,
        prompt_tokens=combined.prompt_tokens,
        completion_tokens=combined.completion_tokens,
        total_tokens=combined.total_tokens,
        cached_tokens=combined.cached_tokens,
        reasoning_tokens=combined.reasoning_tokens,
        llm_calls=combined.calls,
        model=combined.model or model,
        highest_cost_check_id=top.check_id if has_cost else None,
        highest_cost_usd=top.cost_usd if has_cost else None,
        by_check=rows,
    )


def summarize(findings: list[DefectFinding]) -> ScrutinySummary:
    counts = Counter(f.status for f in findings)
    confidences = [f.confidence for f in findings] or [0.0]
    return ScrutinySummary(
        total_defects=len(findings),
        defects_found=counts.get("defect_found", 0),
        compliant=counts.get("compliant", 0),
        needs_review=counts.get("needs_review", 0),
        not_determined=counts.get("not_determined", 0),
        not_applicable=counts.get("not_applicable", 0),
        overall_confidence=round(sum(confidences) / len(confidences), 3),
    )
