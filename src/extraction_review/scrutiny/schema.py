"""The `scrutiny_finding_v1` response contract.

`DefectResponse` is the narrow shape the model must return for a single defect.
Catalogue fields (how to cure, rule, source) are copied onto the finding in
Python so they stay aligned with the defect API payload.
"""

from __future__ import annotations

from collections import Counter
from datetime import UTC, datetime
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field

from .rules import Defect

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


def build_finding(
    defect: Defect,
    response: DefectResponse,
    *,
    evidence_ids: list[str],
    coverage: Coverage,
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
    )


def failed_finding(defect: Defect, error: str) -> DefectFinding:
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
