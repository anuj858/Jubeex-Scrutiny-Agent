"""Index job nests scrutiny after Pinecone; failure fails the job."""

from __future__ import annotations

import inspect
from types import SimpleNamespace

import pytest

from extraction_review.process_split_files import (
    ProcessSplitFilesWorkflow,
    SplitFilesState,
    SplitPartEvent,
    _complete_index_only,
)
from extraction_review.scrutiny_workflow import (
    SCRUTINY_AFTER_INDEX_ERROR,
    ScrutinyAfterIndexError,
)


class _Store:
    def __init__(self, state: object) -> None:
        self._state = state

    async def get_state(self) -> object:
        return self._state


def _index_state() -> SplitFilesState:
    return SplitFilesState(
        agent_data_id="agd-index",
        parsed_slots=["petition"],
        skip_index=False,
        filename="bundle.pdf",
        filing_type="SLP_CIVIL",
        organization_id="org-1",
        workspace_id="ws-1",
        file_hash="hash-1",
        parts=[SplitPartEvent(slot_id="petition", file_id="dfl-p")],
    )


def _fake_client(payload: dict[str, object] | None = None) -> SimpleNamespace:
    stored = payload if payload is not None else {"data": {}, "metadata": {}}

    class FakeAgentData:
        async def get(self, item_id: str):
            return SimpleNamespace(id=item_id, data=dict(stored))

        async def update(self, item_id: str, data=None, **_):
            stored.clear()
            stored.update(data or {})
            return SimpleNamespace(id=item_id, data=dict(stored))

    return SimpleNamespace(beta=SimpleNamespace(agent_data=FakeAgentData()))


@pytest.mark.asyncio
async def test_complete_index_only_runs_scrutiny_after_pinecone(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    order: list[str] = []

    async def fake_index(**_kwargs: object) -> None:
        order.append("pinecone")

    async def fake_scrutiny(_ctx: object, *, agent_data_id: str) -> dict[str, str]:
        order.append("scrutiny")
        return {"schema_name": "scrutiny_finding_v1", "agent_data_id": agent_data_id}

    monkeypatch.setattr(
        "extraction_review.process_split_files.pinecone_enabled",
        lambda: True,
    )
    monkeypatch.setattr(
        "extraction_review.process_split_files._index_split_upload",
        fake_index,
    )
    monkeypatch.setattr(
        "extraction_review.process_split_files._run_nested_scrutiny",
        fake_scrutiny,
    )
    monkeypatch.setattr(
        "extraction_review.process_split_files.upload_step_json",
        lambda *_args, **_kwargs: None,
    )

    state = _index_state()
    ctx = SimpleNamespace(store=_Store(state), write_event_to_stream=lambda _ev: None)
    payload = await _complete_index_only(
        _fake_client(),
        ctx=ctx,  # type: ignore[arg-type]
        catalog=None,
        parts=[],
        pages_by_slot={},
        parse_job_ids={"petition": "job-1"},
        layouts_by_slot={},
        page_markdown={1: "hello"},
        page_parts={},
        page_layout={},
    )
    assert order == ["pinecone", "scrutiny"]
    assert payload["agent_data_id"] == "agd-index"
    assert payload["report"]["schema_name"] == "scrutiny_finding_v1"


@pytest.mark.asyncio
async def test_complete_index_only_raises_fixed_error_after_pinecone(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    order: list[str] = []

    async def fake_index(**_kwargs: object) -> None:
        order.append("pinecone")

    async def fake_scrutiny(_ctx: object, *, agent_data_id: str) -> dict[str, str]:
        order.append("scrutiny")
        raise ScrutinyAfterIndexError(agent_data_id)

    monkeypatch.setattr(
        "extraction_review.process_split_files.pinecone_enabled",
        lambda: True,
    )
    monkeypatch.setattr(
        "extraction_review.process_split_files._index_split_upload",
        fake_index,
    )
    monkeypatch.setattr(
        "extraction_review.process_split_files._run_nested_scrutiny",
        fake_scrutiny,
    )
    monkeypatch.setattr(
        "extraction_review.process_split_files.upload_step_json",
        lambda *_args, **_kwargs: None,
    )

    state = _index_state()
    ctx = SimpleNamespace(store=_Store(state), write_event_to_stream=lambda _ev: None)
    with pytest.raises(ScrutinyAfterIndexError) as err:
        await _complete_index_only(
            _fake_client(),
            ctx=ctx,  # type: ignore[arg-type]
            catalog=None,
            parts=[],
            pages_by_slot={},
            parse_job_ids={},
            layouts_by_slot={},
            page_markdown={},
            page_parts={},
            page_layout={},
        )
    assert order == ["pinecone", "scrutiny"]
    assert str(err.value) == SCRUTINY_AFTER_INDEX_ERROR
    assert str(err.value) == "Unable to run scrutiny defects. Try after some time."
    assert err.value.agent_data_id == "agd-index"


@pytest.mark.asyncio
async def test_run_nested_scrutiny_wraps_openrouter_failure(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    captured: dict[str, object] = {}

    class FakeHandler:
        def stream_events(self):
            async def _empty():
                if False:
                    yield None

            return _empty()

        def __await__(self):
            return self._boom().__await__()

        async def _boom(self):
            raise RuntimeError("OpenRouter down")

    class FakeWorkflow:
        def __init__(self, timeout: object = None) -> None:
            pass

        def run(self, start_event: object) -> FakeHandler:
            captured["event"] = start_event
            return FakeHandler()

    monkeypatch.setattr(
        "extraction_review.scrutiny_workflow.ScrutinyWorkflow",
        FakeWorkflow,
    )
    from extraction_review.process_split_files import _run_nested_scrutiny

    state = _index_state()
    state.special_category = "Appeal (Armed Forces)"
    streamed: list[object] = []
    ctx = SimpleNamespace(
        store=_Store(state),
        write_event_to_stream=lambda ev: streamed.append(ev),
    )
    with pytest.raises(ScrutinyAfterIndexError) as err:
        await _run_nested_scrutiny(ctx, agent_data_id="agd-index")  # type: ignore[arg-type]
    assert str(err.value) == "Unable to run scrutiny defects. Try after some time."
    assert err.value.agent_data_id == "agd-index"
    assert captured["event"].agent_data_id == "agd-index"
    assert captured["event"].special_category == "Appeal (Armed Forces)"


def test_extract_path_does_not_nest_scrutiny() -> None:
    assert "_run_nested_scrutiny" in inspect.getsource(_complete_index_only)
    assert "_run_nested_scrutiny" not in inspect.getsource(
        ProcessSplitFilesWorkflow.complete_extraction
    )


def test_extract_path_does_not_detect_visual() -> None:
    assert "_detect_visual_for_state" not in inspect.getsource(
        ProcessSplitFilesWorkflow.complete_extraction
    )
    assert "_detect_visual_for_state" in inspect.getsource(_complete_index_only)


@pytest.mark.asyncio
async def test_complete_index_only_writes_visual_marks_before_scrutiny(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    order: list[str] = []
    stored: dict[str, object] = {"data": {}, "metadata": {}}
    mark = {
        "page": 20,
        "document_type": "Main Petition",
        "marking_type": "signature_like_mark",
        "signature_role": "advocate",
        "bbox": {"x": 0.62, "y": 0.81, "width": 0.28, "height": 0.08},
        "confidence": 0.86,
    }

    async def fake_detect(_state: object, _ctx: object) -> dict[str, object]:
        order.append("visual")
        return {
            "status": "ok",
            "marks": [mark],
            "targets": [{"page": 20, "document_types": ["Main Petition"]}],
        }

    def fake_upload(_payload: object, **_kwargs: object) -> dict[str, str]:
        order.append("upload")
        return {"url": "https://s3.example/visual.json", "key": "visualfiles/v.json"}

    async def fake_index(**_kwargs: object) -> None:
        order.append("pinecone")

    async def fake_scrutiny(_ctx: object, *, agent_data_id: str) -> dict[str, str]:
        order.append("scrutiny")
        metadata = stored.get("metadata")
        assert isinstance(metadata, dict)
        summary = metadata["visual_summary"]
        assert summary["marks"][0]["page"] == 20
        assert summary["marks"][0]["marking_type"] == "signature_like_mark"
        assert summary["marks"][0]["bbox"]["x"] == 0.62
        assert metadata["visual_artifact_url"] == "https://s3.example/visual.json"
        return {"schema_name": "scrutiny_finding_v1", "agent_data_id": agent_data_id}

    monkeypatch.setattr(
        "extraction_review.process_split_files._detect_visual_for_state",
        fake_detect,
    )
    monkeypatch.setattr(
        "extraction_review.process_split_files.upload_visual_index",
        fake_upload,
    )
    monkeypatch.setattr(
        "extraction_review.process_split_files.pinecone_enabled",
        lambda: True,
    )
    monkeypatch.setattr(
        "extraction_review.process_split_files._index_split_upload",
        fake_index,
    )
    monkeypatch.setattr(
        "extraction_review.process_split_files._run_nested_scrutiny",
        fake_scrutiny,
    )
    monkeypatch.setattr(
        "extraction_review.process_split_files.upload_step_json",
        lambda *_args, **_kwargs: None,
    )

    state = _index_state()
    ctx = SimpleNamespace(store=_Store(state), write_event_to_stream=lambda _ev: None)
    payload = await _complete_index_only(
        _fake_client(stored),
        ctx=ctx,  # type: ignore[arg-type]
        catalog=None,
        parts=[],
        pages_by_slot={},
        parse_job_ids={"petition": "job-1"},
        layouts_by_slot={},
        page_markdown={1: "hello"},
        page_parts={},
        page_layout={},
    )
    assert order == ["visual", "upload", "pinecone", "scrutiny"]
    assert payload["agent_data_id"] == "agd-index"


@pytest.mark.asyncio
async def test_run_workflow_fails_with_retry_message(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    captured: dict[str, object] = {}

    async def fake_notify(**kwargs: object) -> None:
        captured.update(kwargs)

    monkeypatch.setattr("extraction_review.api.notify_job_finished", fake_notify)
    monkeypatch.setattr("extraction_review.api.recorded_artifacts", lambda _job: [])
    monkeypatch.setattr("extraction_review.api.set_job_context", lambda *_a, **_k: None)

    from extraction_review.api import JobState, _run_workflow

    job = JobState(job_id="job-scr-fail", kind="process_file")

    async def boom() -> None:
        raise ScrutinyAfterIndexError("agd-index")

    await _run_workflow(job, boom())
    assert job.status == "failed"
    assert job.error == "Unable to run scrutiny defects. Try after some time."
    assert job.agent_data_id == "agd-index"
    assert job.result == {"agent_data_id": "agd-index"}
    assert captured["status"] == "failed"
    assert captured["error"] == job.error
    assert captured["agent_data_id"] == "agd-index"


@pytest.mark.asyncio
async def test_run_workflow_unwraps_wrapped_scrutiny_error(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    async def fake_notify(**_kwargs: object) -> None:
        return None

    monkeypatch.setattr("extraction_review.api.notify_job_finished", fake_notify)
    monkeypatch.setattr("extraction_review.api.recorded_artifacts", lambda _job: [])
    monkeypatch.setattr("extraction_review.api.set_job_context", lambda *_a, **_k: None)

    from extraction_review.api import JobState, _run_workflow

    job = JobState(job_id="job-wrap", kind="process_file")

    async def boom() -> None:
        try:
            raise ScrutinyAfterIndexError("agd-index")
        except ScrutinyAfterIndexError as exc:
            raise RuntimeError("workflow failed") from exc

    await _run_workflow(job, boom())
    assert job.status == "failed"
    assert job.error == "Unable to run scrutiny defects. Try after some time."
    assert job.result == {"agent_data_id": "agd-index"}
