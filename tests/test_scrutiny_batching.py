from __future__ import annotations

from types import SimpleNamespace
from unittest.mock import AsyncMock

import pytest

import extraction_review.scrutiny_workflow as workflow
from extraction_review.llm import LLMError, _complete_structured_fields
from extraction_review.scrutiny.prompts import (
    build_defect_batch_prompt,
    build_defect_prompt,
)
from extraction_review.scrutiny.rules import Catalogue, Defect
from extraction_review.scrutiny.schema import (
    DefectBatchResponse,
    DefectResponse,
    EvidenceRef,
    LlmUsage,
)


def _defect(check_id: str, part: str, *, visual: bool = False) -> Defect:
    wording = "The required signature is missing." if visual else "A field is blank."
    return Defect(
        check_id=check_id,
        main_category="General/Global",
        defect=wording,
        requirement="The required field must be complete.",
        where_to_look=[f"Check the {part}"],
        inspect_parts=[part],
        how_to_cure=["Complete the field."],
        location_source="SCI Handbook",
    )


def _response(check_id: str) -> DefectResponse:
    return DefectResponse(
        check_id=check_id,
        status="compliant",
        confidence=0.95,
        summary="The required field is complete.",
        observed="The document contains the completed field.",
        conclusion="The requirement is satisfied.",
        evidence=[EvidenceRef(chunk_id="c1", quote="completed field")],
    )


def _catalogue(defects: list[Defect]) -> Catalogue:
    return Catalogue(
        schema_version="1",
        catalogue_id="test",
        catalogue_version="1",
        jurisdiction="Supreme Court of India",
        disclaimer="test",
        defects=defects,
    )


def test_partition_defects_groups_primary_family_but_separates_visual_checks() -> None:
    defects = [
        _defect("D-1", "Main Petition"),
        _defect("D-2", "Main Petition"),
        _defect("D-3", "Main Petition"),
        _defect("D-4", "Main Petition", visual=True),
        _defect("D-5", "Affidavit"),
    ]

    batches = workflow.partition_defects(defects, 3)

    assert [[item.check_id for item in batch] for batch in batches] == [
        ["D-1", "D-2", "D-3"],
        ["D-4"],
        ["D-5"],
    ]


def test_batch_concurrency_keeps_request_pipeline_full(monkeypatch) -> None:
    monkeypatch.delenv("SCRUTINY_BATCH_CONCURRENCY", raising=False)
    assert workflow.scrutiny_batch_concurrency(8, 4) == 8
    monkeypatch.setenv("SCRUTINY_BATCH_CONCURRENCY", "3")
    assert workflow.scrutiny_batch_concurrency(8, 4) == 3


def test_split_batch_usage_preserves_exact_totals_and_one_call() -> None:
    usage = LlmUsage(
        prompt_tokens=101,
        completion_tokens=22,
        total_tokens=123,
        cached_tokens=7,
        reasoning_tokens=9,
        calls=1,
        cost_usd=0.0123,
        model="model",
        generation_id="generation",
    )

    shares = workflow.split_batch_usage(usage, 4)
    combined = LlmUsage()
    for share in shares:
        combined = combined.plus(share)

    assert combined.prompt_tokens == 101
    assert combined.completion_tokens == 22
    assert combined.total_tokens == 123
    assert combined.cached_tokens == 7
    assert combined.reasoning_tokens == 9
    assert combined.calls == 1
    assert combined.cost_usd == pytest.approx(0.0123)
    assert combined.generation_ids == ["generation"]


def test_structured_completion_repairs_each_batch_item_without_polluting_envelope() -> (
    None
):
    completed = _complete_structured_fields(
        {
            "findings": [
                {
                    "check_id": "D-1",
                    "status": "needs_review",
                    "confidence": 0.5,
                    "summary": "Evidence is incomplete.",
                }
            ]
        }
    )

    assert set(completed) == {"findings"}
    assert completed["findings"][0]["evidence"] == []
    assert completed["findings"][0]["observed"] == "Evidence is incomplete."


def test_batch_prompt_prints_shared_evidence_once_and_is_smaller() -> None:
    defects = [_defect("D-1", "Main Petition"), _defect("D-2", "Main Petition")]
    chunks = [
        {
            "record_id": "page-1",
            "chunk_kind": "page",
            "page": 1,
            "document_part": "Main Petition",
            "text": "UNIQUE COMPLETED FIELD EVIDENCE",
        }
    ]
    tasks = [(defect, {}, [dict(chunks[0])]) for defect in defects]
    _system, batch_prompt = build_defect_batch_prompt(
        tasks,
        catalogue=_catalogue(defects),
        filing_type="SLP_CIVIL",
    )
    separate = "\n".join(
        build_defect_prompt(
            defect,
            record={},
            chunks=[dict(chunks[0])],
            catalogue=_catalogue(defects),
        )
        for defect in defects
    )

    assert batch_prompt.count("UNIQUE COMPLETED FIELD EVIDENCE") == 1
    assert batch_prompt.count("## Allowed shared excerpts\nc1") == 2
    assert all(task_chunks[0]["excerpt_id"] == "c1" for _, _, task_chunks in tasks)
    assert len(batch_prompt) < len(separate)


@pytest.mark.asyncio
async def test_run_defect_batch_uses_one_call_and_returns_every_check(
    monkeypatch,
) -> None:
    defects = [_defect("D-1", "Main Petition"), _defect("D-2", "Main Petition")]
    usage = LlmUsage(prompt_tokens=100, completion_tokens=20, total_tokens=120, calls=1)
    call = AsyncMock(
        return_value=(
            DefectBatchResponse(findings=[_response("D-1"), _response("D-2")]),
            usage,
        )
    )
    monkeypatch.setattr(workflow, "call_structured", call)
    monkeypatch.setattr(
        workflow,
        "_finding_from_response",
        lambda defect, response, **kwargs: SimpleNamespace(
            check_id=defect.check_id,
            status=response.status,
            usage=kwargs["usage"],
        ),
    )
    chunks = [
        {
            "chunk_kind": "page",
            "page": 1,
            "document_part": "Main Petition",
            "text": "completed field",
        }
    ]

    findings = await workflow._run_defect_batch(
        [(defect, [dict(chunks[0])]) for defect in defects],
        catalogue=_catalogue(defects),
        record={},
        filing_type="SLP_CIVIL",
    )

    assert [finding.check_id for finding in findings] == ["D-1", "D-2"]
    assert sum(finding.usage.calls for finding in findings) == 1
    call.assert_awaited_once()
    assert call.await_args.kwargs["response_model"] is DefectBatchResponse
    assert call.await_args.kwargs["request_label"] == "scrutiny-batch:D-1,D-2"


@pytest.mark.asyncio
async def test_run_defect_batch_rejects_missing_check(monkeypatch) -> None:
    defects = [_defect("D-1", "Main Petition"), _defect("D-2", "Main Petition")]
    monkeypatch.setattr(
        workflow,
        "call_structured",
        AsyncMock(
            return_value=(
                DefectBatchResponse(findings=[_response("D-1")]),
                LlmUsage(calls=1),
            )
        ),
    )

    with pytest.raises(LLMError, match="1 unique result"):
        await workflow._run_defect_batch(
            [(defect, []) for defect in defects],
            catalogue=_catalogue(defects),
            record={},
            filing_type="SLP_CIVIL",
        )


@pytest.mark.asyncio
async def test_batch_collector_publishes_each_finding() -> None:
    defects = [_defect(f"D-{index}", "Main Petition") for index in range(1, 6)]
    published: list[int] = []

    async def runner(batch: list[Defect]) -> list[SimpleNamespace]:
        return [SimpleNamespace(check_id=defect.check_id) for defect in batch]

    async def update(findings: list[SimpleNamespace], stopped: bool) -> None:
        assert stopped is False
        published.append(len(findings))

    findings, stopped = await workflow.collect_defect_batch_findings(
        [defects[:3], defects[3:]],
        runner,
        concurrency=2,
        on_update=update,
    )

    assert stopped is False
    assert len(findings) == 5
    assert published == [1, 2, 3, 4, 5]
