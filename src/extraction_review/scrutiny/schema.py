"""The `scrutiny_finding_v1` response contract.

`DefectResponse` is the narrow shape the model must return for a single defect.
Catalogue fields (how to cure, rule, source) are copied onto the finding in
Python so they stay aligned with the defect API payload.
"""

from __future__ import annotations

import os
from collections import Counter
from datetime import UTC, datetime
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field

from .rules import Defect
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

    page: int | None = Field(description="1-indexed page number, or null if unknown")
    quote: str = Field(description="Verbatim excerpt supporting the finding")


class DefectResponse(BaseModel):
    """Exactly what the model returns for one defect."""

    model_config = ConfigDict(extra="forbid")

    check_id: str
    status: ResultState
    confidence: float = Field(ge=0.0, le=1.0)
    summary: str = Field(description="One or two sentences covering the whole defect")
    reasoning: str = Field(description="Why this status was chosen, citing evidence")
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


def build_finding(
    defect: Defect,
    response: DefectResponse,
    *,
    evidence_ids: list[str],
    coverage: Coverage,
    usage: LlmUsage | None = None,
) -> DefectFinding:
    suggested = response.suggested_fix if response.status == "defect_found" else None
    rationale = response.fix_rationale if response.status == "defect_found" else None
    return DefectFinding(
        check_id=defect.check_id,
        serial_no=defect.serial_no,
        title=defect.title,
        main_category=defect.main_category,
        special_category=defect.special_category,
        status=response.status,
        summary=response.summary,
        confidence=response.confidence,
        reasoning=response.reasoning,
        evidence=response.evidence,
        suggested_fix=suggested,
        fix_rationale=rationale,
        how_to_cure=list(defect.how_to_cure),
        applicable_rule=defect.applicable_rule,
        location_source=defect.location_source,
        evidence_ids=evidence_ids,
        coverage=coverage,
        usage=usage,
    )


def failed_finding(
    defect: Defect, error: str, usage: LlmUsage | None = None
) -> DefectFinding:
    """Placeholder when a defect could not be evaluated at all."""
    return DefectFinding(
        check_id=defect.check_id,
        serial_no=defect.serial_no,
        title=defect.title,
        main_category=defect.main_category,
        special_category=defect.special_category,
        status="not_determined",
        summary=f"This check could not be completed: {error}",
        confidence=0.0,
        reasoning=f"Check did not run: {error}",
        evidence=[],
        suggested_fix=None,
        fix_rationale=None,
        how_to_cure=list(defect.how_to_cure),
        applicable_rule=defect.applicable_rule,
        location_source=defect.location_source,
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
