"""The `scrutiny_finding_v1` response contract.

`DefectResponse` is the narrow shape the model must return for a single defect.
Catalogue fields (defect, requirement, how to cure, rule, source) are copied
onto the finding in Python so they stay aligned with the defect API payload.
"""

from __future__ import annotations

import logging
import os
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
from ..layout_index import locate_quote, page_entry, part_for_page, quote_in_text
from .prompts import (
    display_cure_steps,
    filing_location,
    finding_title,
    readable_location_source,
    validated_reasoning,
    validated_summary,
)
from .rules import Defect, get_catalogue

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
    schema is derived from DefectResponse and must stay {page, quote}.
    """

    model_config = ConfigDict(extra="forbid")

    page: int | None = Field(
        description=(
            "1-indexed PDF page of THIS filing, copied from the excerpt "
            "header such as '[Page 12 — Main Petition]'. Null if the quote is "
            "not from an excerpt. Never use a page number from Authority "
            "or location_source (those are official-rulebook locators)."
        )
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
    local_page: int | None = None


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
    suggested_fix: str | None = Field(
        description="What should be there instead. Null unless status is defect_found."
    )
    fix_rationale: str | None = Field(
        description="Why the suggested fix resolves the defect. Null if no fix."
    )


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
    serial_no: int | str
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
    highest_cost_serial_no: int | str | None = None
    highest_cost_usd: float | None = None
    by_check: list[UsageByCheck] = Field(default_factory=list)
    note: str = (
        "Exact OpenRouter usage.cost from each /chat/completions reply, "
        "summed for this check. The model is not asked for cost. LlamaParse, "
        "extract, classify, and Pinecone are billed separately."
    )


class DefectFinding(BaseModel):
    """Server-assembled finding for one catalogue defect."""

    check_id: str
    serial_no: int | str
    title: str
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
    suggested_fix: str | None = None
    fix_rationale: str | None = None
    how_to_cure: list[str] = Field(default_factory=list)
    applicable_rule: str | None = None
    location: str | None = Field(
        default=None,
        description=(
            "Petition/Matter PDF page for this finding (the full filing), "
            "e.g. 'Filing page 12 — Vakalatnama.' If the page is unknown: "
            "'Filing page missing — …'."
        ),
    )
    location_source: str | None = None
    evidence_ids: list[str] = Field(default_factory=list)
    coverage: Coverage = Field(default_factory=Coverage)
    usage: LlmUsage | None = None
    error: str | None = None


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
            _chunk_part_rank(chunk, preferred),
            _safe_page_number(chunk.get("page")) or -1,
        ),
        reverse=True,
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
        refs.append(EvidenceRef(page=page, quote=quote))
        if len(refs) >= 2:
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
        if box.w <= 0 or box.h <= 0:
            continue
        x = min(max(float(box.x), 0.0), 1.0)
        y = min(max(float(box.y), 0.0), 1.0)
        width = min(max(float(box.w), 0.0), round(1.0 - x, 6))
        height = min(max(float(box.h), 0.0), round(1.0 - y, 6))
        if width <= 0 or height <= 0:
            continue
        boxes.append(
            BoundingBox(page=int(box.page), x=x, y=y, w=width, h=height)
        )
    return boxes


def attach_evidence_boxes(
    refs: list[EvidenceRef],
    *,
    layout: dict[int, dict[str, Any]] | None = None,
    chunks: list[dict[str, Any]] | None = None,
) -> list[FindingEvidence]:
    """Attach per-line boxes after page snap. Empty layout → unavailable."""
    attached: list[FindingEvidence] = []
    prefer_pages = sorted(_retrieved_pages(chunks or []))
    for ref in refs:
        try:
            boxes, status, page = locate_quote(
                ref.quote or "",
                ref.page,
                layout,
                prefer_pages=prefer_pages,
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
            attached.append(
                FindingEvidence(
                    page=page,
                    quote=ref.quote or "",
                    bounding_boxes=_as_bounding_boxes(boxes),
                    boxes_status=boxes_status,
                    document_part=part_for_page(chunks, page),
                    slot_id=slot_id,
                    local_page=local_page,
                )
            )
        except Exception:
            logger.warning(
                "Failed to attach highlight boxes; leaving citation unboxed",
                exc_info=True,
            )
            attached.append(
                FindingEvidence(
                    page=getattr(ref, "page", None),
                    quote=(getattr(ref, "quote", None) or ""),
                    bounding_boxes=[],
                    boxes_status="unavailable",
                )
            )
    return attached


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
        grounded.append(EvidenceRef(page=page, quote=quote))
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
    file_name: str | None = None
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
) -> DefectFinding:
    suggested = response.suggested_fix if response.status == "defect_found" else None
    rationale = response.fix_rationale if response.status == "defect_found" else None
    catalogue = get_catalogue()
    refs = list(response.evidence)
    if not refs and chunks:
        refs = _evidence_from_chunks(chunks, defect)
    evidence = attach_evidence_boxes(refs, layout=layout, chunks=chunks)
    evidence_pages = [ref.page for ref in evidence if ref.page is not None]
    pages = list(coverage.pages_reviewed) or _pages_from_chunks(chunks)
    parts = _parts_for_pages(chunks, evidence_pages or pages)
    location = filing_location(
        evidence_pages=evidence_pages,
        reviewed_pages=pages,
        document_parts=parts,
    )
    return DefectFinding(
        check_id=defect.check_id,
        serial_no=defect.serial_no,
        title=finding_title(defect, catalogue),
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
        suggested_fix=suggested,
        fix_rationale=rationale,
        how_to_cure=display_cure_steps(defect.how_to_cure),
        applicable_rule=defect.applicable_rule,
        location=location,
        location_source=readable_location_source(defect, catalogue),
        evidence_ids=evidence_ids,
        coverage=coverage,
        usage=usage,
    )


def failed_finding(
    defect: Defect, error: str, usage: LlmUsage | None = None
) -> DefectFinding:
    """Placeholder when a defect could not be evaluated at all."""
    catalogue = get_catalogue()
    return DefectFinding(
        check_id=defect.check_id,
        serial_no=defect.serial_no,
        title=finding_title(defect, catalogue),
        defect=defect.defect,
        requirement=defect.requirement,
        main_category=defect.main_category,
        special_category=defect.special_category,
        status="not_determined",
        summary=f"This check could not be completed: {error}",
        confidence=0.0,
        reasoning=f"This check did not run: {error}",
        evidence=[],
        suggested_fix=None,
        fix_rationale=None,
        how_to_cure=display_cure_steps(defect.how_to_cure),
        applicable_rule=defect.applicable_rule,
        location="Filing page missing — this check did not run.",
        location_source=readable_location_source(defect, catalogue),
        error=error,
        usage=usage,
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
                serial_no=finding.serial_no,
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
        highest_cost_serial_no=top.serial_no if has_cost else None,
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
