"""The `scrutiny_finding_v1` response contract.

`DefectResponse` is the narrow shape the model must return for a single defect.
Catalogue fields (how to cure, rule, source) are copied onto the finding in
Python so they stay aligned with the defect API payload.
"""

from __future__ import annotations

import os
import re
from collections import Counter
from datetime import UTC, datetime
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field

from .rules import Defect, get_catalogue
from .prompts import (
    finding_title,
    filing_location,
    readable_location_source,
    validated_reasoning,
    validated_summary,
)
from ..document_parts import missing_required_parts

ResultState = Literal[
    "defect_found",
    "compliant",
    "not_applicable",
    "not_determined",
    "needs_review",
]


class EvidenceRef(BaseModel):
    """A pointer back into the source document for one observation."""

    model_config = ConfigDict(extra="forbid")

    page: int | None = Field(
        description=(
            "1-indexed PDF page of THIS filing, copied from the excerpt "
            "header such as '[Page 12 — Petition]'. Null if the quote is "
            "not from an excerpt. Never use a page number from Authority "
            "or location_source (those are official-rulebook locators)."
        )
    )
    quote: str = Field(description="Verbatim excerpt supporting the finding")


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

    def plus(self, other: "LlmUsage") -> "LlmUsage":
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
    serial_no: int
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
    highest_cost_serial_no: int | None = None
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
    serial_no: int
    title: str
    main_category: str
    special_category: str | None = None
    status: ResultState
    summary: str
    confidence: float
    reasoning: str
    evidence: list[EvidenceRef] = Field(default_factory=list)
    suggested_fix: str | None = None
    fix_rationale: str | None = None
    how_to_cure: list[str] = Field(default_factory=list)
    applicable_rule: str | None = None
    location: str | None = Field(
        default=None,
        description=(
            "Petition PDF page for this finding, e.g. 'Filing page 12 — "
            "Vakalatnama.' If the page is unknown: 'Filing page missing — …'."
        ),
    )
    location_source: str | None = None
    evidence_ids: list[str] = Field(default_factory=list)
    coverage: Coverage = Field(default_factory=Coverage)
    usage: LlmUsage | None = None
    error: str | None = None


DEFAULT_REVIEW_CONFIDENCE = 0.6


def review_confidence_threshold() -> float:
    raw = os.getenv("SCRUTINY_REVIEW_CONFIDENCE", "")
    try:
        value = float(raw) if raw else DEFAULT_REVIEW_CONFIDENCE
    except ValueError:
        value = DEFAULT_REVIEW_CONFIDENCE
    return min(1.0, max(0.0, value))


def _norm_quote(text: str) -> str:
    collapsed = re.sub(r"\s+", " ", (text or "").replace("…", " ").replace("...", " "))
    return collapsed.strip().lower()


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
        if end_i < start_i:
            end_i = start_i
        pages.update(range(start_i, end_i + 1))
    return pages


def _quote_in_text(quote: str, text: str) -> bool:
    needle = _norm_quote(quote)
    haystack = _norm_quote(text)
    if not needle or not haystack:
        return False
    if needle in haystack:
        return True
    snippet = needle[:80].strip()
    return len(snippet) >= 12 and snippet in haystack


def _chunk_page_for_quote(quote: str, chunk: dict[str, Any]) -> int | None:
    text = chunk.get("text") or ""
    if not _quote_in_text(quote, text):
        return None
    start = chunk.get("page")
    try:
        return int(start) if start is not None else None
    except (TypeError, ValueError):
        return None


def apply_evidence_pages(
    response: DefectResponse,
    chunks: list[dict[str, Any]],
) -> DefectResponse:
    """Snap evidence.page to retrieved filing pages; drop rulebook pages."""
    if not response.evidence:
        return response
    allowed = _retrieved_pages(chunks)
    page_chunks = [c for c in chunks if c.get("chunk_kind") != "summary"]
    grounded: list[EvidenceRef] = []
    for ref in response.evidence:
        quote = ref.quote or ""
        matched = [
            page
            for chunk in page_chunks
            if (page := _chunk_page_for_quote(quote, chunk)) is not None
        ]
        if matched:
            if ref.page in matched:
                page = ref.page
            else:
                page = matched[0]
        elif ref.page is not None and ref.page in allowed:
            page = ref.page
        else:
            page = None
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
) -> DefectFinding:
    suggested = response.suggested_fix if response.status == "defect_found" else None
    rationale = response.fix_rationale if response.status == "defect_found" else None
    catalogue = get_catalogue()
    evidence_pages = [ref.page for ref in response.evidence if ref.page is not None]
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
        evidence=response.evidence,
        suggested_fix=suggested,
        fix_rationale=rationale,
        how_to_cure=list(defect.how_to_cure),
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
        main_category=defect.main_category,
        special_category=defect.special_category,
        status="not_determined",
        summary=f"This check could not be completed: {error}",
        confidence=0.0,
        reasoning=f"This check did not run: {error}",
        evidence=[],
        suggested_fix=None,
        fix_rationale=None,
        how_to_cure=list(defect.how_to_cure),
        applicable_rule=defect.applicable_rule,
        location="Filing page missing — this check did not run.",
        location_source=readable_location_source(defect, catalogue),
        error=error,
        usage=usage,
    )


def summarize_usage(findings: list[DefectFinding], *, model: str | None) -> UsageSummary:
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
