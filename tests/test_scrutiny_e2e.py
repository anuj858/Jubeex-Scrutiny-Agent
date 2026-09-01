"""End-to-end scrutiny-check: workflow, streaming, persist, fail-fast."""

from __future__ import annotations

from types import SimpleNamespace
from typing import Any

import pytest

from extraction_review.llm import LLMError
from extraction_review.scrutiny.rules import get_catalogue
from extraction_review.scrutiny.schema import DefectResponse, LlmUsage
from extraction_review.scrutiny_workflow import (
    ScrutinyEvent,
    ScrutinyPartial,
    ScrutinyResponse,
    ScrutinyWorkflow,
    collect_defect_findings,
)


class FakeAgentData:
    def __init__(self, item: SimpleNamespace) -> None:
        self.item = item
        self.updates: list[dict[str, Any]] = []

    async def get(self, item_id: str) -> SimpleNamespace:
        assert item_id == self.item.id
        return self.item

    async def update(self, item_id: str, data: dict[str, Any] | None = None, **_: Any) -> SimpleNamespace:
        assert item_id == self.item.id
        payload = data or {}
        self.updates.append(payload)
        self.item.data = payload
        return self.item


class FakeLlamaCloud:
    def __init__(self, *args: Any, **kwargs: Any) -> None:
        self.beta = SimpleNamespace(agent_data=FakeAgentData(_approved_item()))


def _approved_item() -> SimpleNamespace:
    return SimpleNamespace(
        id="item-e2e-1",
        data={
            "status": "approved",
            "file_name": "Defect SLP Civil -3 .pdf",
            "file_hash": "hash-e2e",
            "data": {
                "petition_type": "SLP_CIVIL",
                "court": "Supreme Court of India",
            },
            "metadata": {"classification": "SLP_CIVIL"},
        },
    )


def _ok(check_id: str) -> tuple[DefectResponse, LlmUsage]:
    return (
        DefectResponse(
            check_id=check_id,
            status="compliant",
            confidence=0.91,
            summary=f"{check_id} is present in the filing.",
            reasoning=f"The excerpts cover {check_id}.",
            evidence=[],
            suggested_fix=None,
            fix_rationale=None,
        ),
        LlmUsage(calls=1, prompt_tokens=20, completion_tokens=8, total_tokens=28),
    )


def _check_id_from_prompt(user_prompt: str) -> str:
    marker = 'Return JSON with check_id "'
    start = user_prompt.find(marker)
    assert start != -1, user_prompt[:200]
    start += len(marker)
    end = user_prompt.find('"', start)
    return user_prompt[start:end]


@pytest.fixture
def scrutiny_env(monkeypatch: pytest.MonkeyPatch) -> FakeLlamaCloud:
    client = FakeLlamaCloud()
    monkeypatch.setenv("OPENROUTER_API_KEY", "test-key")
    monkeypatch.setenv("SCRUTINY_ENABLED", "true")
    monkeypatch.setenv("SCRUTINY_DEFECTS", "D003,D004,D005,D006")
    monkeypatch.setenv("SCRUTINY_CONCURRENCY", "2")
    monkeypatch.setenv("VECTOR_BACKEND", "off")
    monkeypatch.setattr(
        "extraction_review.clients.AsyncLlamaCloud",
        lambda *a, **k: client,
    )
    monkeypatch.setattr(
        "extraction_review.scrutiny_workflow.pinecone_enabled",
        lambda: False,
    )
    return client


@pytest.mark.asyncio
async def test_scrutiny_runs_enabled_defects_and_streams_partials(
    scrutiny_env: FakeLlamaCloud,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    calls: list[str] = []

    async def fake_llm(**kwargs: Any) -> tuple[DefectResponse, LlmUsage]:
        check_id = _check_id_from_prompt(kwargs["user_prompt"])
        calls.append(check_id)
        return _ok(check_id)

    monkeypatch.setattr(
        "extraction_review.scrutiny_workflow.call_structured",
        fake_llm,
    )

    workflow = ScrutinyWorkflow(timeout=None)
    handler = workflow.run(start_event=ScrutinyEvent(agent_data_id="item-e2e-1"))
    partials: list[ScrutinyPartial] = []
    async for event in handler.stream_events():
        if isinstance(event, ScrutinyPartial):
            partials.append(event)
    result = await handler

    assert isinstance(result, ScrutinyResponse)
    report = result.report
    assert report.stopped_early is False
    assert report.file_name == "Defect SLP Civil -3 .pdf"
    assert report.petition_type == "SLP_CIVIL"
    assert report.planned_checks == 4
    assert {f.check_id for f in report.findings} == {"D003", "D004", "D005", "D006"}
    assert [f.serial_no for f in report.findings] == sorted(
        f.serial_no for f in report.findings
    )
    assert all(f.status == "compliant" for f in report.findings)
    assert set(calls) == {"D003", "D004", "D005", "D006"}

    assert partials
    counts = [p.completed for p in partials]
    assert counts == sorted(counts)
    assert counts[-1] == 4
    assert partials[-1].report.findings[-1].check_id in {
        "D003",
        "D004",
        "D005",
        "D006",
    }

    saved = scrutiny_env.beta.agent_data.updates
    assert saved
    last = saved[-1]["metadata"]["scrutiny_report"]
    assert last["planned_checks"] == 4
    assert last["stopped_early"] is False
    assert len(last["findings"]) == 4


@pytest.mark.asyncio
async def test_scrutiny_all_seventy_four_checks_mocked(
    scrutiny_env: FakeLlamaCloud,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("SCRUTINY_DEFECTS", "all")
    monkeypatch.setenv("SCRUTINY_CONCURRENCY", "6")
    calls: list[str] = []

    async def fake_llm(**kwargs: Any) -> tuple[DefectResponse, LlmUsage]:
        check_id = _check_id_from_prompt(kwargs["user_prompt"])
        calls.append(check_id)
        return _ok(check_id)

    monkeypatch.setattr(
        "extraction_review.scrutiny_workflow.call_structured",
        fake_llm,
    )

    workflow = ScrutinyWorkflow(timeout=None)
    handler = workflow.run(start_event=ScrutinyEvent(agent_data_id="item-e2e-1"))
    partials: list[ScrutinyPartial] = []
    async for event in handler.stream_events():
        if isinstance(event, ScrutinyPartial):
            partials.append(event)
    result = await handler

    assert isinstance(result, ScrutinyResponse)
    report = result.report
    catalogue = get_catalogue()
    assert report.planned_checks == len(catalogue.defects) == 74
    assert report.stopped_early is False
    assert len(report.findings) == 74
    assert len(calls) == 74
    assert len(partials) == 74
    assert [p.completed for p in partials] == list(range(1, 75))
    last = scrutiny_env.beta.agent_data.updates[-1]["metadata"]["scrutiny_report"]
    assert len(last["findings"]) == 74


@pytest.mark.asyncio
async def test_scrutiny_stops_on_llm_error_keeps_completed_results(
    scrutiny_env: FakeLlamaCloud,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    calls: list[str] = []

    async def fake_llm(**kwargs: Any) -> tuple[DefectResponse, LlmUsage]:
        check_id = _check_id_from_prompt(kwargs["user_prompt"])
        calls.append(check_id)
        if check_id == "D003":
            raise LLMError("Response was truncated or not JSON: {")
        return _ok(check_id)

    monkeypatch.setattr(
        "extraction_review.scrutiny_workflow.call_structured",
        fake_llm,
    )

    workflow = ScrutinyWorkflow(timeout=None)
    handler = workflow.run(start_event=ScrutinyEvent(agent_data_id="item-e2e-1"))
    partials: list[ScrutinyPartial] = []
    async for event in handler.stream_events():
        if isinstance(event, ScrutinyPartial):
            partials.append(event)
    result = await handler

    assert isinstance(result, ScrutinyResponse)
    report = result.report
    assert report.stopped_early is True
    assert report.planned_checks == 4
    ids = {f.check_id for f in report.findings}
    assert "D003" in ids
    failed = next(f for f in report.findings if f.check_id == "D003")
    assert failed.status == "not_determined"
    assert failed.error
    assert "D006" not in calls
    assert "D004" not in calls
    assert len(calls) <= 2
    assert partials[-1].stopped_early is True
    assert scrutiny_env.beta.agent_data.updates
    last = scrutiny_env.beta.agent_data.updates[-1]["metadata"]["scrutiny_report"]
    assert last["stopped_early"] is True
    assert len(last["findings"]) == len(report.findings)


@pytest.mark.asyncio
async def test_each_check_retrieves_its_own_pinecone_excerpts(
    scrutiny_env: FakeLlamaCloud,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(
        "extraction_review.scrutiny_workflow.pinecone_enabled",
        lambda: True,
    )
    gathers: list[tuple[str, tuple[str, ...]]] = []

    def fake_gather(
        queries: list[str],
        *,
        file_hash: str,
        top_k: int | None = None,
        max_chunks: int | None = None,
    ) -> list[dict[str, Any]]:
        gathers.append((file_hash, tuple(queries)))
        return [
            {
                "record_id": f"p-{len(gathers)}",
                "chunk_kind": "page",
                "page": 1,
                "document_part": "Petition",
                "text": "SPECIAL LEAVE PETITION",
                "score": 0.9,
            }
        ]

    async def fake_llm(**kwargs: Any) -> tuple[DefectResponse, LlmUsage]:
        check_id = _check_id_from_prompt(kwargs["user_prompt"])
        return _ok(check_id)

    monkeypatch.setattr(
        "extraction_review.scrutiny_workflow.gather_filing_evidence",
        fake_gather,
    )
    monkeypatch.setattr(
        "extraction_review.scrutiny_workflow.call_structured",
        fake_llm,
    )

    workflow = ScrutinyWorkflow(timeout=None)
    handler = workflow.run(start_event=ScrutinyEvent(agent_data_id="item-e2e-1"))
    async for _event in handler.stream_events():
        pass
    result = await handler

    assert isinstance(result, ScrutinyResponse)
    assert len(gathers) == 4
    assert all(file_hash == "hash-e2e" for file_hash, _queries in gathers)
    assert len({queries for _file_hash, queries in gathers}) >= 2


@pytest.mark.asyncio
async def test_low_confidence_finding_is_needs_review(
    scrutiny_env: FakeLlamaCloud,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    async def fake_llm(**kwargs: Any) -> tuple[DefectResponse, LlmUsage]:
        check_id = _check_id_from_prompt(kwargs["user_prompt"])
        response, usage = _ok(check_id)
        if check_id == "D004":
            response = response.model_copy(
                update={
                    "status": "defect_found",
                    "confidence": 0.2,
                    "suggested_fix": "Add the missing heading.",
                    "fix_rationale": "The rule requires it.",
                }
            )
        return response, usage

    monkeypatch.setattr(
        "extraction_review.scrutiny_workflow.call_structured",
        fake_llm,
    )

    workflow = ScrutinyWorkflow(timeout=None)
    handler = workflow.run(start_event=ScrutinyEvent(agent_data_id="item-e2e-1"))
    async for _event in handler.stream_events():
        pass
    result = await handler

    assert isinstance(result, ScrutinyResponse)
    weak = next(f for f in result.report.findings if f.check_id == "D004")
    assert weak.status == "needs_review"
    assert weak.suggested_fix is None
    assert result.report.summary.needs_review == 1


@pytest.mark.asyncio
async def test_all_defects_are_catalogue_sized() -> None:
    catalogue = get_catalogue()
    assert len(catalogue.defects) == 74
    assert catalogue.defect("D018").check_id == "D018"


@pytest.mark.asyncio
async def test_collect_queue_does_not_start_after_abort() -> None:
    started: list[str] = []
    defects = [SimpleNamespace(check_id=f"D{i:03d}") for i in range(1, 13)]

    async def runner(defect: SimpleNamespace) -> SimpleNamespace:
        started.append(defect.check_id)
        if defect.check_id == "D001":
            return SimpleNamespace(check_id="D001", error="stop")
        return SimpleNamespace(check_id=defect.check_id, error=None)

    findings, stopped = await collect_defect_findings(
        defects,  # type: ignore[arg-type]
        runner,
        concurrency=6,
    )
    assert stopped is True
    assert findings[0].check_id == "D001"
    assert len(started) <= 6
    assert "D012" not in started
