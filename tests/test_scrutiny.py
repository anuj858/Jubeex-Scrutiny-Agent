"""Tests for the registry defect catalogue and finding roll-up."""

import json

import pytest

from extraction_review.llm import strict_json_schema
from extraction_review.scrutiny.prompts import (
    build_defect_prompt,
    build_evidence_queries,
    build_system_prompt,
)
from extraction_review.scrutiny.rules import (
    catalogue_path,
    catalogue_schema_path,
    defects_for_filing_type,
    enabled_defect_ids,
    get_catalogue,
)
from extraction_review.scrutiny.schema import (
    Coverage,
    DefectResponse,
    EvidenceRef,
    SubcheckResult,
    build_finding,
    failed_finding,
    roll_up_confidence,
    roll_up_status,
    summarize,
)


def _subcheck(subcheck_id: str, status: str, confidence: float = 0.9):
    return SubcheckResult(
        subcheck_id=subcheck_id,
        status=status,
        confidence=confidence,
        reasoning="because",
        evidence=[EvidenceRef(page=1, quote="text")],
        suggested_fix=None,
        fix_rationale=None,
    )


class TestCatalogue:
    def test_catalogue_matches_its_json_schema(self):
        jsonschema = pytest.importorskip("jsonschema")

        with catalogue_path().open(encoding="utf-8") as fh:
            catalogue = json.load(fh)
        with catalogue_schema_path().open(encoding="utf-8") as fh:
            schema = json.load(fh)

        jsonschema.validate(instance=catalogue, schema=schema)

    def test_loads_all_eleven_defects(self):
        catalogue = get_catalogue()
        assert catalogue.catalogue_id == "sci_registry_defects"
        assert len(catalogue.defects) == 11
        assert [d.check_id for d in catalogue.defects] == catalogue.defect_order

    def test_every_defect_has_subchecks(self):
        for defect in get_catalogue().defects:
            assert defect.subchecks, f"{defect.check_id} has no subchecks"


class TestDefectSelection:
    def test_default_allowlist_is_the_three_text_evaluable_defects(self, monkeypatch):
        monkeypatch.delenv("SCRUTINY_DEFECTS", raising=False)
        assert enabled_defect_ids() == ("D001", "D004", "D007")

    def test_selection_is_case_insensitive_on_filing_type(self, monkeypatch):
        monkeypatch.delenv("SCRUTINY_DEFECTS", raising=False)
        # The pipeline classifies as SLP_CIVIL; the catalogue says slp_civil.
        assert [d.check_id for d in defects_for_filing_type("SLP_CIVIL")] == [
            "D001",
            "D004",
            "D007",
        ]

    def test_unrelated_filing_types_select_nothing(self, monkeypatch):
        monkeypatch.delenv("SCRUTINY_DEFECTS", raising=False)
        assert defects_for_filing_type("WRIT_PETITION_CIVIL") == []
        assert defects_for_filing_type(None) == []

    def test_env_var_overrides_the_allowlist(self, monkeypatch):
        monkeypatch.setenv("SCRUTINY_DEFECTS", "D001")
        assert [d.check_id for d in defects_for_filing_type("SLP_CIVIL")] == ["D001"]

    def test_all_expands_to_the_full_catalogue(self, monkeypatch):
        monkeypatch.setenv("SCRUTINY_DEFECTS", "all")
        assert len(defects_for_filing_type("SLP_CIVIL")) == 11

    def test_default_defects_need_no_visual_evidence(self, monkeypatch):
        """The starting set must be answerable from extracted text alone."""
        monkeypatch.delenv("SCRUTINY_DEFECTS", raising=False)
        for defect in defects_for_filing_type("SLP_CIVIL"):
            for subcheck in defect.subchecks:
                assert not subcheck.requires_visual_evidence, (
                    f"{subcheck.subcheck_id} needs visual evidence"
                )


class TestRollUp:
    def test_most_severe_subcheck_wins(self):
        results = [
            _subcheck("D001.S01", "compliant"),
            _subcheck("D001.S02", "defect_found"),
            _subcheck("D001.S03", "needs_review"),
        ]
        assert roll_up_status(results) == "defect_found"

    def test_needs_review_outranks_not_determined(self):
        results = [
            _subcheck("D001.S01", "not_determined"),
            _subcheck("D001.S02", "needs_review"),
        ]
        assert roll_up_status(results) == "needs_review"

    def test_all_compliant_is_compliant(self):
        results = [_subcheck("D001.S01", "compliant")]
        assert roll_up_status(results) == "compliant"

    def test_no_results_is_not_determined(self):
        assert roll_up_status([]) == "not_determined"

    def test_confidence_averages_the_deciding_subchecks(self):
        results = [
            _subcheck("D001.S01", "compliant", 0.2),
            _subcheck("D001.S02", "defect_found", 0.8),
            _subcheck("D001.S03", "defect_found", 0.6),
        ]
        assert roll_up_confidence(results, "defect_found") == 0.7


class TestFindings:
    def test_build_finding_carries_catalogue_metadata(self):
        defect = get_catalogue().defect("D001")
        response = DefectResponse(
            check_id="D001",
            summary="Form 28 body is present.",
            subcheck_results=[
                _subcheck(s.subcheck_id, "compliant") for s in defect.subchecks
            ],
        )
        finding = build_finding(
            defect, response, evidence_ids=["abc:page:1:0"], coverage=Coverage()
        )
        assert finding.status == "compliant"
        assert finding.severity == defect.severity
        assert finding.authority_refs

    def test_failed_finding_marks_every_subcheck_undetermined(self):
        defect = get_catalogue().defect("D007")
        finding = failed_finding(defect, "timeout")
        assert finding.status == "not_determined"
        assert finding.error == "timeout"
        assert len(finding.subcheck_results) == len(defect.subchecks)

    def test_summary_counts_by_status(self):
        defect = get_catalogue().defect("D001")
        findings = [
            failed_finding(defect, "boom"),
            build_finding(
                defect,
                DefectResponse(
                    check_id="D001",
                    summary="ok",
                    subcheck_results=[_subcheck("D001.S01", "compliant")],
                ),
                evidence_ids=[],
                coverage=Coverage(),
            ),
        ]
        summary = summarize(findings)
        assert summary.total_defects == 2
        assert summary.not_determined == 1
        assert summary.compliant == 1


class TestPrompts:
    def test_defect_prompt_includes_evidence_and_subcheck_ids(self):
        defect = get_catalogue().defect("D004")
        prompt = build_defect_prompt(
            defect,
            record={"court": "Supreme Court of India", "petitioners": []},
            chunks=[{"chunk_kind": "page", "page": 3, "text": "IN THE SUPREME COURT"}],
            file_name="slp.pdf",
        )
        assert defect.instruction in prompt
        assert "IN THE SUPREME COURT" in prompt
        assert "Page 3" in prompt
        for subcheck in defect.subchecks:
            assert subcheck.subcheck_id in prompt
        # Nulls are pruned so the record reads cleanly.
        assert (
            "petitioners"
            not in prompt.split("## Document excerpts")[0].split(
                "Structured filing record"
            )[1]
        )

    def test_skipped_subchecks_are_excluded_from_the_prompt(self):
        defect = get_catalogue().defect("D001")
        skipped = [defect.subchecks[0]]
        prompt = build_defect_prompt(defect, record={}, chunks=[], skipped=skipped)
        assert "have been excluded" in prompt

    def test_system_prompt_states_the_decision_policy(self):
        prompt = build_system_prompt(get_catalogue())
        for state in (
            "defect_found",
            "compliant",
            "not_applicable",
            "not_determined",
            "needs_review",
        ):
            assert state in prompt

    def test_one_query_per_subcheck_plus_one_for_the_defect(self):
        defect = get_catalogue().defect("D007")
        queries = build_evidence_queries(defect, defect.subchecks)
        assert len(queries) == len(defect.subchecks) + 1


class TestStrictSchema:
    def test_every_object_forbids_extra_properties(self):
        schema = strict_json_schema(DefectResponse)

        def check(node):
            if isinstance(node, dict):
                if "properties" in node:
                    assert node.get("additionalProperties") is False
                    assert set(node["required"]) == set(node["properties"])
                for value in node.values():
                    check(value)
            elif isinstance(node, list):
                for entry in node:
                    check(entry)

        check(schema)
