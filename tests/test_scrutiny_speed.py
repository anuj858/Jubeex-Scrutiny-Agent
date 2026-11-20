from __future__ import annotations

import asyncio
from types import SimpleNamespace

import pytest

from extraction_review.process_file import core_extraction_input_hash
from extraction_review.scrutiny.rules import Defect
from extraction_review.scrutiny.schema import DefectResponse, LlmUsage
from extraction_review.scrutiny_workflow import (
    DefectBatchEnvelope,
    EvidenceWorkspace,
    ScrutinyMicrobatcher,
)


def _defect(check_id: str) -> Defect:
    return Defect(
        check_id=check_id,
        main_category="SLP (Civil)",
        defect="Required text is missing",
        requirement="The Main Petition must contain the required text.",
        where_to_look=["Main Petition"],
        inspect_parts=["Main Petition"],
        how_to_cure=["Add the required text."],
        location_source="",
    )


def _response(check_id: str) -> DefectResponse:
    return DefectResponse(
        check_id=check_id,
        status="compliant",
        confidence=0.9,
        summary="The required text is present.",
        observed="The Main Petition contains the required text.",
        conclusion="The requirement is satisfied.",
        evidence=[],
    )


def test_evidence_workspace_reuses_complete_part_pages() -> None:
    workspace = EvidenceWorkspace.from_layout(
        {
            1: {"document_part": "Cover Page", "text": "cover"},
            2: {"document_part": "Main Petition", "text": "petition one"},
            3: {"document_part": "Main Petition", "text": "petition two"},
        }
    )
    first = workspace.complete_chunks(_defect("D-1"))
    second = workspace.complete_chunks(_defect("D-2"))
    assert [chunk["page"] for chunk in first] == [2, 3]
    assert [chunk["page"] for chunk in second] == [2, 3]
    assert workspace.pages_by_part["Main Petition"] == [2, 3]


def test_core_extraction_hash_changes_with_document_ownership() -> None:
    base = SimpleNamespace(
        slot_id="petition",
        file_hash="abc",
        file_id="dfl-1",
        page_span=[{"start": 10, "end": 20}],
    )
    moved = SimpleNamespace(
        slot_id="petition",
        file_hash="abc",
        file_id="dfl-1",
        page_span=[{"start": 10, "end": 21}],
    )
    assert core_extraction_input_hash("SLP_CIVIL", [base]) == (
        core_extraction_input_hash("SLP_CIVIL", [base])
    )
    assert core_extraction_input_hash("SLP_CIVIL", [base]) != (
        core_extraction_input_hash("SLP_CIVIL", [moved])
    )


@pytest.mark.asyncio
async def test_microbatcher_combines_compatible_checks(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("SCRUTINY_EXECUTION_MODE", "fast")
    monkeypatch.setenv("SCRUTINY_MICROBATCH_SIZE", "2")
    calls: list[str] = []

    async def fake_call_structured(**kwargs):
        calls.append(kwargs["request_label"])
        assert kwargs["response_model"] is DefectBatchEnvelope
        return (
            DefectBatchEnvelope(findings=[_response("D-1"), _response("D-2")]),
            LlmUsage(
                prompt_tokens=100,
                completion_tokens=20,
                total_tokens=120,
                calls=1,
            ),
        )

    monkeypatch.setattr(
        "extraction_review.scrutiny_workflow.call_structured", fake_call_structured
    )
    batcher = ScrutinyMicrobatcher(None)
    results = await asyncio.gather(
        batcher.evaluate(
            defect=_defect("D-1"),
            system_prompt="system",
            user_prompt="first",
            model="test-model",
        ),
        batcher.evaluate(
            defect=_defect("D-2"),
            system_prompt="system",
            user_prompt="second",
            model="test-model",
        ),
    )
    await batcher.close()
    assert len(calls) == 1
    assert [result[0].check_id for result in results] == ["D-1", "D-2"]
    assert sum(result[1].total_tokens for result in results) == 120
    assert sum(result[1].calls for result in results) == 1
