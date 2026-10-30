from types import SimpleNamespace
from unittest.mock import AsyncMock

import pytest

from extraction_review.scrutiny.schema import (
    DefectResponse,
    LlmUsage,
    finalize_public_response,
)
from extraction_review.scrutiny_workflow import _run_defect, require_successful_checks


def test_failed_defect_check_rejects_completed_scrutiny() -> None:
    findings = [SimpleNamespace(error=None), SimpleNamespace(error="Vertex 429")]
    with pytest.raises(RuntimeError, match="1 of 2 checks failed"):
        require_successful_checks(findings, planned=2)


def test_inconclusive_but_evaluated_check_can_complete() -> None:
    require_successful_checks([SimpleNamespace(error=None)], planned=1)


@pytest.mark.parametrize("internal_status", ["needs_review", "not_determined"])
def test_internal_uncertainty_is_preserved_instead_of_becoming_compliant(
    internal_status: str,
) -> None:
    response = DefectResponse(
        check_id="D-1",
        status=internal_status,
        confidence=0.5,
        summary="Internal uncertainty",
        observed="The evidence did not settle the check.",
        conclusion="A final product outcome is required.",
        evidence=[],
    )

    final = finalize_public_response(response)

    assert final.status == internal_status
    assert final.summary == "Internal uncertainty"


@pytest.mark.parametrize("status", ["defect_found", "compliant", "not_applicable"])
def test_final_public_statuses_are_preserved(status: str) -> None:
    response = DefectResponse(
        check_id="D-1",
        status=status,
        confidence=0.95,
        summary="Final outcome",
        observed="Grounded filing evidence was reviewed.",
        conclusion="The check is resolved.",
        evidence=[],
    )

    assert finalize_public_response(response).status == status


@pytest.mark.asyncio
async def test_uncertain_first_pass_is_not_forced_to_compliant(monkeypatch) -> None:
    import extraction_review.scrutiny_workflow as workflow

    uncertain = DefectResponse(
        check_id="D-1",
        status="needs_review",
        confidence=0.6,
        summary="The first pass is uncertain.",
        observed="The available evidence is inconclusive.",
        conclusion="A final decision is still required.",
        evidence=[],
    )
    call = AsyncMock(return_value=(uncertain, LlmUsage(calls=1)))
    monkeypatch.setattr(workflow, "call_structured", call)
    monkeypatch.setattr(workflow, "manifest_absence_proven", lambda *_args: False)
    monkeypatch.setattr(workflow, "required_parts_for_defect", lambda *_args: [])
    monkeypatch.setattr(workflow, "_retrieval_evidence_complete", lambda *_args: True)
    monkeypatch.setattr(workflow, "build_system_prompt", lambda *_args: "system")
    monkeypatch.setattr(
        workflow, "build_defect_prompt", lambda *_args, **_kwargs: "prompt"
    )
    monkeypatch.setattr(workflow, "_sanitize_response", lambda _d, r, *_args: r)
    monkeypatch.setattr(
        workflow,
        "build_finding",
        lambda _defect, response, **_kwargs: response,
    )
    defect = SimpleNamespace(
        check_id="D-1",
        defect="Example defect",
        requirement="Example requirement",
        ai_note=None,
    )

    result = await _run_defect(
        defect,
        catalogue=SimpleNamespace(),
        record={},
        chunks=[],
        filing_type="SLP_CIVIL",
    )

    assert result.status == "needs_review"
    assert call.await_count == 1
    assert call.await_args.kwargs["request_label"] == "scrutiny:D-1"


@pytest.mark.asyncio
async def test_unverified_rule_is_evaluated_from_filing_evidence(monkeypatch) -> None:
    import extraction_review.scrutiny_workflow as workflow
    from extraction_review.scrutiny.rules import get_catalogue

    response = DefectResponse(
        check_id="D-57",
        status="defect_found",
        confidence=0.95,
        summary="The execution date is blank.",
        observed="The complete Vakalatnama shows a blank execution-date field.",
        conclusion="The Vakalatnama date requirement is not met.",
        evidence=[],
    )
    call = AsyncMock(return_value=(response, LlmUsage(calls=1)))
    monkeypatch.setattr(workflow, "call_structured", call)
    monkeypatch.setattr(workflow, "build_system_prompt", lambda *_args: "system")
    monkeypatch.setattr(
        workflow, "build_defect_prompt", lambda *_args, **_kwargs: "prompt"
    )
    monkeypatch.setattr(workflow, "_sanitize_response", lambda _d, r, *_args: r)
    monkeypatch.setattr(
        workflow,
        "build_finding",
        lambda _defect, result, **_kwargs: result,
    )
    catalogue = get_catalogue()
    defect = catalogue.defect("D-57")

    result = await _run_defect(
        defect,
        catalogue=catalogue,
        record={},
        chunks=[],
        filing_type="SLP_CIVIL",
    )

    assert result.status == "defect_found"
    call.assert_awaited_once()


@pytest.mark.asyncio
async def test_d324_missing_aor_certificate_is_defect_without_legal_gate(
    monkeypatch,
) -> None:
    import extraction_review.scrutiny_workflow as workflow
    from extraction_review.scrutiny.rules import get_catalogue

    call = AsyncMock()
    monkeypatch.setattr(workflow, "call_structured", call)
    catalogue = get_catalogue()
    defect = catalogue.defect("D-324")

    result = await _run_defect(
        defect,
        catalogue=catalogue,
        record={},
        chunks=[],
        filing_type="SLP_CIVIL",
        layout={1: {"document_part": "Main Petition"}},
        absence_defect_parts=["AOR's Certificate"],
    )

    assert result.status == "defect_found"
    assert "AOR's Certificate" in result.summary
    call.assert_not_awaited()


@pytest.mark.asyncio
async def test_d55_is_na_when_affidavit_prerequisite_is_absent(monkeypatch) -> None:
    import extraction_review.scrutiny_workflow as workflow
    from extraction_review.scrutiny.rules import get_catalogue

    call = AsyncMock()
    monkeypatch.setattr(workflow, "call_structured", call)
    catalogue = get_catalogue()
    defect = catalogue.defect("D-55")

    result = await _run_defect(
        defect,
        catalogue=catalogue,
        record={},
        chunks=[],
        filing_type="SLP_CIVIL",
        layout={1: {"document_part": "Main Petition"}},
    )

    assert result.status == "not_applicable"
    assert "Affidavit" in result.summary
    call.assert_not_awaited()
