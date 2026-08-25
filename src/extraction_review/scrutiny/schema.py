"""The `scrutiny_finding_v1` response contract.

Two layers live here. `DefectResponse` is the narrow shape the model must return
for a single defect; everything else is computed in Python so that roll-up and
severity aggregation stay deterministic and auditable.
"""

from __future__ import annotations

from collections import Counter
from datetime import UTC, datetime
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field

from .rules import Defect, ResultState, Subcheck

# Ordered most severe first. A defect takes the most severe status among its
# applicable subchecks.
RESULT_PRECEDENCE: tuple[ResultState, ...] = (
    "defect_found",
    "needs_review",
    "not_determined",
    "compliant",
    "not_applicable",
)


class EvidenceRef(BaseModel):
    """A pointer back into the source document for one observation."""

    model_config = ConfigDict(extra="forbid")

    page: int | None = Field(description="1-indexed page number, or null if unknown")
    quote: str = Field(description="Verbatim excerpt supporting the finding")


class SubcheckResult(BaseModel):
    """The model's verdict on a single subcheck."""

    model_config = ConfigDict(extra="forbid")

    subcheck_id: str
    status: ResultState
    confidence: float = Field(ge=0.0, le=1.0)
    reasoning: str = Field(description="Why this status was chosen, citing evidence")
    evidence: list[EvidenceRef]
    suggested_fix: str | None = Field(
        description="What should be there instead. Null unless status is defect_found."
    )
    fix_rationale: str | None = Field(
        description="Why the suggested fix resolves the defect. Null if no fix."
    )


class DefectResponse(BaseModel):
    """Exactly what the model returns for one defect."""

    model_config = ConfigDict(extra="forbid")

    check_id: str
    summary: str = Field(description="One or two sentences covering the whole defect")
    subcheck_results: list[SubcheckResult]


class Coverage(BaseModel):
    """How much of the document backed this finding."""

    model_config = ConfigDict(extra="forbid")

    chunks_reviewed: int = 0
    pages_reviewed: list[int] = Field(default_factory=list)
    structured_record_available: bool = False
    evidence_complete: bool = False


class DefectFinding(BaseModel):
    """Server-assembled finding. Field names follow `output_contract`."""

    check_id: str
    title: str
    severity: str
    status: ResultState
    summary: str
    confidence: float
    subcheck_results: list[SubcheckResult]
    evidence_ids: list[str] = Field(default_factory=list)
    coverage: Coverage = Field(default_factory=Coverage)
    authority_refs: list[str] = Field(default_factory=list)
    error: str | None = None


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


def skipped_subcheck(
    subcheck: Subcheck,
    *,
    status: ResultState = "not_determined",
    reason: str,
) -> SubcheckResult:
    """A subcheck the model was never asked about, with the reason recorded."""
    return SubcheckResult(
        subcheck_id=subcheck.subcheck_id,
        status=status,
        confidence=1.0,
        reasoning=reason,
        evidence=[],
        suggested_fix=None,
        fix_rationale=None,
    )


def roll_up_status(results: list[SubcheckResult]) -> ResultState:
    """Most severe subcheck status wins.

    This is computed here rather than asked of the model so aggregation is
    reproducible and reviewable.
    """
    if not results:
        return "not_determined"
    present = {r.status for r in results}
    for state in RESULT_PRECEDENCE:
        if state in present:
            return state
    return "not_determined"


def roll_up_confidence(results: list[SubcheckResult], status: ResultState) -> float:
    """Mean confidence of the subchecks that decided the outcome."""
    deciding = [r.confidence for r in results if r.status == status]
    pool = deciding or [r.confidence for r in results]
    if not pool:
        return 0.0
    return round(sum(pool) / len(pool), 3)


def build_finding(
    defect: Defect,
    response: DefectResponse,
    *,
    evidence_ids: list[str],
    coverage: Coverage,
) -> DefectFinding:
    status = roll_up_status(response.subcheck_results)
    return DefectFinding(
        check_id=defect.check_id,
        title=defect.title,
        severity=defect.severity,
        status=status,
        summary=response.summary,
        confidence=roll_up_confidence(response.subcheck_results, status),
        subcheck_results=response.subcheck_results,
        evidence_ids=evidence_ids,
        coverage=coverage,
        authority_refs=[
            f"{ref.source_id}: {ref.locator}" for ref in defect.authority_refs
        ],
    )


def failed_finding(defect: Defect, error: str) -> DefectFinding:
    """Placeholder when a defect could not be evaluated at all."""
    return DefectFinding(
        check_id=defect.check_id,
        title=defect.title,
        severity=defect.severity,
        status="not_determined",
        summary=f"This check could not be completed: {error}",
        confidence=0.0,
        subcheck_results=[
            skipped_subcheck(sub, reason=f"Check did not run: {error}")
            for sub in defect.subchecks
        ],
        authority_refs=[
            f"{ref.source_id}: {ref.locator}" for ref in defect.authority_refs
        ],
        error=error,
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
