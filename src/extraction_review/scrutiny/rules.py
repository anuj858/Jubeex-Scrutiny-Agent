"""Loader for the SCI registry defect catalogue.

The catalogue JSON is authored under `scrutiny_harness/rules/` and mirrors
`sci_registry_defects.schema.v1.json`. It is parsed into Pydantic models once at
first use so malformed rules fail loudly at load time rather than silently
producing a bad prompt.
"""

from __future__ import annotations

import json
import logging
import os
from functools import lru_cache
from pathlib import Path
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field

logger = logging.getLogger(__name__)

CATALOGUE_FILENAME = "sci_registry_defects.v1.json"
SCHEMA_FILENAME = "sci_registry_defects.schema.v1.json"

ResultState = Literal[
    "defect_found",
    "compliant",
    "not_applicable",
    "not_determined",
    "needs_review",
]

# Evidence labels that can only be settled by looking at the rendered page.
# Page markdown cannot support them, so subchecks requiring them are reported as
# `not_determined` instead of being guessed at.
VISUAL_EVIDENCE_LABELS: frozenset[str] = frozenset(
    {
        "aor_signature_slot",
        "application_signature_slots",
        "court_fee_section_or_visual",
        "deponent_signature_slot",
        "listing_proforma_signature_result",
        "listing_proforma_signature_slot",
        "notary_or_attestation_elements",
        "ocr_coverage",
        "page_render_quality",
        "petition_signature_slots",
        "petitioner_signature_slot",
        "physical_and_printed_page_map",
        "typography_summary",
        "visual_coverage",
        "visual_evidence",
        "visual_formality_coverage",
    }
)

DEFAULT_ENABLED_DEFECTS = ("D001", "D004", "D007")


class _Strict(BaseModel):
    model_config = ConfigDict(extra="forbid")


class AuthorityRef(_Strict):
    source_id: str
    locator: str


class Condition(_Strict):
    fact: str
    operator: Literal["equals", "not_equals", "contains", "exists", "greater_than"]
    value: Any = None


class Applicability(_Strict):
    filing_types: list[str]
    conditions: list[Condition] = Field(default_factory=list)
    unknown_condition_result: str


class Subcheck(_Strict):
    subcheck_id: str
    title: str
    criterion: str
    applicability: Applicability
    authority_refs: list[AuthorityRef]
    required_evidence: list[str]
    evaluation_method: Literal["deterministic", "model_assisted", "manual"]
    failure_result: Literal["defect_found", "needs_review", "not_determined"]
    manual_review_when: list[str] = Field(default_factory=list)

    @property
    def requires_visual_evidence(self) -> bool:
        return any(e in VISUAL_EVIDENCE_LABELS for e in self.required_evidence)


class MinimumCoverage(_Strict):
    active_versions_only: bool
    all_relevant_sections: bool
    all_relevant_pages: bool | None = None


class EvidencePlan(_Strict):
    crg_capabilities: list[str]
    required_evidence: list[str]
    optional_evidence: list[str] = Field(default_factory=list)
    minimum_coverage: MinimumCoverage


class Evaluation(_Strict):
    mode: Literal["deterministic", "deterministic_then_model_assisted", "manual"]
    deterministic_validators: list[str] = Field(default_factory=list)
    model_allowed: bool


class DecisionPolicy(_Strict):
    compliant: str
    defect_found: str
    not_applicable: str
    not_determined: str
    needs_review: str


class OutputContract(_Strict):
    schema_name: str
    required_fields: list[str]


class Remediation(_Strict):
    mode: Literal["lawyer_review_required", "manual_only"]
    allowed_operation_codes: list[str] = Field(default_factory=list)
    prohibited_changes: list[str]


class Defect(_Strict):
    check_id: str
    order: int
    title: str
    objective: str
    severity: Literal["critical", "high", "medium", "low"]
    applicability: Applicability
    instruction: str
    authority_refs: list[AuthorityRef]
    subchecks: list[Subcheck]
    evidence_plan: EvidencePlan
    evaluation: Evaluation
    decision_policy: DecisionPolicy
    output_contract: OutputContract
    remediation: Remediation


class Source(_Strict):
    source_id: str
    title: str
    authority_type: str
    url: str
    issued_date: str | None = None
    effective_date: str | None = None
    checksum: str | None = None
    locators: dict[str, str]


class Crosswalk(_Strict):
    authority_ref: str
    disposition: str
    check_ids: list[str]
    reason: str


class Catalogue(_Strict):
    catalogue_id: str
    schema_version: str
    catalogue_version: str
    status: str
    jurisdiction: str
    filing_type: str
    effective_from: str
    last_official_source_reviewed_at: str
    disclaimer: str | None = None
    allowed_result_states: list[str]
    sources: list[Source]
    global_decision_policy: DecisionPolicy
    authority_crosswalk: list[Crosswalk]
    defect_order: list[str]
    defects: list[Defect]

    def defect(self, check_id: str) -> Defect | None:
        return next((d for d in self.defects if d.check_id == check_id), None)


def _candidate_rule_dirs() -> list[Path]:
    """Locations to look for the catalogue, most specific first.

    In a built wheel the rules are force-included at `extraction_review/_rules`.
    When running from a source checkout they live at the repo root.
    """
    override = os.getenv("SCRUTINY_RULES_DIR")
    dirs: list[Path] = [Path(override)] if override else []

    package_dir = Path(__file__).resolve().parent.parent
    dirs.append(package_dir / "_rules")

    # src/extraction_review/scrutiny/rules.py -> repo root is 4 levels up
    repo_root = Path(__file__).resolve().parents[3]
    dirs.append(repo_root / "scrutiny_harness" / "rules")
    dirs.append(Path.cwd() / "scrutiny_harness" / "rules")
    return dirs


def _locate(filename: str) -> Path:
    tried: list[str] = []
    for directory in _candidate_rule_dirs():
        candidate = directory / filename
        tried.append(str(candidate))
        if candidate.is_file():
            return candidate
    raise FileNotFoundError(
        f"Could not locate {filename}. Looked in: {', '.join(tried)}. "
        f"Set SCRUTINY_RULES_DIR to override."
    )


def catalogue_path() -> Path:
    return _locate(CATALOGUE_FILENAME)


def catalogue_schema_path() -> Path:
    return _locate(SCHEMA_FILENAME)


@lru_cache(maxsize=1)
def get_catalogue() -> Catalogue:
    path = catalogue_path()
    with path.open(encoding="utf-8") as fh:
        raw = json.load(fh)
    catalogue = Catalogue.model_validate(raw)
    logger.info(
        "[Scrutiny] Loaded catalogue %s v%s (%s defects) from %s",
        catalogue.catalogue_id,
        catalogue.catalogue_version,
        len(catalogue.defects),
        path,
    )
    return catalogue


def normalize_filing_type(filing_type: str | None) -> str:
    """Map the app's classification labels onto catalogue filing types.

    The pipeline classifies as `SLP_CIVIL`; the catalogue declares `slp_civil`.
    """
    return (filing_type or "").strip().lower()


def enabled_defect_ids() -> tuple[str, ...]:
    """Defect allowlist, so the rollout can be widened without a code change."""
    raw = os.getenv("SCRUTINY_DEFECTS")
    if not raw or not raw.strip():
        return DEFAULT_ENABLED_DEFECTS
    if raw.strip().lower() == "all":
        return tuple(get_catalogue().defect_order)
    ids = tuple(part.strip().upper() for part in raw.split(",") if part.strip())
    return ids or DEFAULT_ENABLED_DEFECTS


def defects_for_filing_type(filing_type: str | None) -> list[Defect]:
    """Enabled defects that apply to this filing type, in catalogue order."""
    catalogue = get_catalogue()
    normalized = normalize_filing_type(filing_type)
    allowed = set(enabled_defect_ids())

    selected = [
        defect
        for defect in catalogue.defects
        if defect.check_id in allowed
        and normalized in {ft.lower() for ft in defect.applicability.filing_types}
    ]
    selected.sort(key=lambda d: d.order)

    unknown = allowed - {d.check_id for d in catalogue.defects}
    if unknown:
        logger.warning(
            "[Scrutiny] SCRUTINY_DEFECTS lists unknown check ids: %s",
            ", ".join(sorted(unknown)),
        )
    return selected


def is_supported_filing_type(filing_type: str | None) -> bool:
    return bool(defects_for_filing_type(filing_type))
