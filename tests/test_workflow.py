from importlib.metadata import version
from types import SimpleNamespace

import pytest
from llama_cloud_fake import FakeLlamaCloudServer
from workflows.events import StartEvent

from extraction_review.config import (
    EXTRACTED_DATA_COLLECTION,
    JUBEEX_FILING_TYPES,
    JUBEEX_UPLOAD_FILING_TYPES,
)
from extraction_review.metadata_workflow import DISCRIMINATOR_FIELD, MetadataResponse
from extraction_review.metadata_workflow import workflow as metadata_workflow
from extraction_review.process_file import BundlePrepared, FileEvent, Status
from extraction_review.process_file import workflow as process_file_workflow

FILING_TYPES = set(JUBEEX_FILING_TYPES)
FAKE_HAS_CLASSIFY_V2 = version("llama-cloud-fake") >= "0.1.1"


@pytest.mark.asyncio
async def test_process_file_workflow(
    monkeypatch: pytest.MonkeyPatch,
    fake: FakeLlamaCloudServer,
) -> None:
    monkeypatch.setenv("LLAMA_CLOUD_API_KEY", "fake-api-key")

    called = {"extract": 0}

    async def fake_extract(*_args: object, **_kwargs: object) -> str:
        called["extract"] += 1
        return "agd-compiled-1"

    monkeypatch.setattr(
        "extraction_review.process_file._extract_sliced_parts",
        fake_extract,
    )
    file_id = fake.files.preload(path="tests/files/test.pdf")
    try:
        result = await process_file_workflow.run(start_event=FileEvent(file_id=file_id))
    except Exception:
        result = None
    assert result is not None
    assert isinstance(result, BundlePrepared)
    assert result.filing_type
    assert result.agent_data_id is None
    assert called["extract"] == 0


@pytest.mark.asyncio
@pytest.mark.skipif(
    not FAKE_HAS_CLASSIFY_V2,
    reason="llama-cloud-fake < 0.1.1 does not mock classify v2",
)
async def test_classify_v2_assigns_filing_type(
    monkeypatch: pytest.MonkeyPatch,
    fake: FakeLlamaCloudServer,
) -> None:
    """process_file reports a concrete SEC filing type from classify v2."""
    monkeypatch.setenv("LLAMA_CLOUD_API_KEY", "fake-api-key")
    file_id = fake.files.preload(path="tests/files/test.pdf")

    async def fake_extract(*_args: object, **_kwargs: object) -> str:
        return "agd-compiled-1"

    monkeypatch.setattr(
        "extraction_review.process_file._extract_sliced_parts",
        fake_extract,
    )

    handler = process_file_workflow.run(start_event=FileEvent(file_id=file_id))
    classified_statuses: list[Status] = []
    async for event in handler.stream_events():
        if isinstance(event, Status):
            if event.level == "error":
                raise AssertionError(f"workflow errored: {event.message}")
            if event.message.startswith("Classified as "):
                classified_statuses.append(event)
    await handler

    # A real classify v2 result produces a "Classified as <type>" info status.
    assert classified_statuses, (
        "expected a 'Classified as ...' status from a completed classify v2 job"
    )
    message = classified_statuses[-1].message
    matched = next((t for t in FILING_TYPES if f"Classified as {t} " in message), None)
    assert matched is not None, f"unexpected classification status: {message}"


@pytest.mark.asyncio
async def test_metadata_workflow() -> None:
    result = await metadata_workflow.run(start_event=StartEvent())
    assert isinstance(result, MetadataResponse)
    assert result.extracted_data_collection == EXTRACTED_DATA_COLLECTION
    assert result.discriminator_field == DISCRIMINATOR_FIELD
    assert set(result.schemas.keys()) == FILING_TYPES
    assert DISCRIMINATOR_FIELD in result.json_schema.get("properties", {})
    assert set(result.split_upload_types.keys()) == set(JUBEEX_UPLOAD_FILING_TYPES)
    criminal_ids = [
        slot["id"] for slot in result.split_upload_types["SLP_CRIMINAL"]["slots"]
    ]
    assert "court_fees" not in criminal_ids
    assert result.config["config_id"] == "jubeex_parse"
    assert result.config["config_version"] == "1.0.0"
    assert result.config["classify"]["config_version"] == "1.0.0"
    assert result.config["extract"]["config_version"] == "1.0.0"
    assert result.config["split"]["config_version"] == "1.0.0"
    assert result.upload_sliced_slot_pdfs is True


@pytest.mark.asyncio
async def test_split_petition_skips_classify(
    monkeypatch: pytest.MonkeyPatch,
    fake: FakeLlamaCloudServer,
) -> None:
    monkeypatch.setenv("LLAMA_CLOUD_API_KEY", "fake-api-key")
    monkeypatch.setattr(
        "extraction_review.process_file._extract_sliced_parts",
        lambda *_args, **_kwargs: (_ for _ in ()).throw(AssertionError("extract")),
    )
    file_id = fake.files.preload(path="tests/files/test.pdf")
    handler = process_file_workflow.run(
        start_event=FileEvent(
            job_type="split_petition",
            filing_type="SLP_CIVIL",
            file_id=file_id,
        )
    )
    classified: list[Status] = []
    skipped = False
    async for event in handler.stream_events():
        if isinstance(event, Status) and event.message.startswith("Classified as "):
            classified.append(event)
        if isinstance(event, Status) and "skipping classify" in event.message.lower():
            skipped = True
    result = await handler
    assert skipped is True
    assert classified == []
    assert isinstance(result, BundlePrepared)
    assert result.filing_type == "SLP_CIVIL"
    assert result.agent_data_id is None
    assert result.job_type == "split_petition"


@pytest.mark.asyncio
async def test_extract_only_skips_index_and_limits_parse_scope(
    monkeypatch: pytest.MonkeyPatch,
    fake: FakeLlamaCloudServer,
) -> None:
    monkeypatch.setenv("LLAMA_CLOUD_API_KEY", "fake-api-key")
    captured: dict[str, object] = {}

    async def fake_extract(_ctx: object, **kwargs: object) -> str:
        captured.update(kwargs)
        return "agd-extract"

    monkeypatch.setattr(
        "extraction_review.process_file._extract_sliced_parts",
        fake_extract,
    )
    result = await process_file_workflow.run(
        start_event=FileEvent(
            job_type="extract_only",
            filing_type="SLP_CIVIL",
            documents=[
                {
                    "slot_id": "petition",
                    "file_id": "dfl-petition",
                    "name": "01_Petition.pdf",
                },
                {
                    "slot_id": "annexure_p1",
                    "file_id": "dfl-ann",
                    "name": "annexure_p1.pdf",
                },
            ],
        )
    )
    assert isinstance(result, BundlePrepared)
    assert captured["skip_index"] is True
    assert captured["skip_extract"] is False
    assert captured["parse_scope"] == "extract_sources"
    assert captured["stitch_in_request_order"] is True
    assert "petition" in captured["parsed_slots"]
    assert "annexure_p1" not in captured["parsed_slots"]
    assert result.report is None


@pytest.mark.asyncio
async def test_index_parsed_reuses_slots_when_not_edited(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    captured: dict[str, object] = {}

    async def fake_extract(_ctx: object, **kwargs: object) -> str:
        captured.update(kwargs)
        return "agd-index"

    monkeypatch.setattr(
        "extraction_review.process_file._extract_sliced_parts",
        fake_extract,
    )
    monkeypatch.setattr(
        "extraction_review.s3_artifacts.download_json_object",
        lambda _key: {
            "parsed_slots": ["petition"],
            "document_order": ["petition"],
            "slots": {
                "petition": {
                    "pages": {"1": "kept"},
                    "parse_job_id": "old-parse",
                    "layout": {},
                }
            },
        },
    )

    class FakeAgentData:
        async def get(self, item_id: str):
            return SimpleNamespace(
                id=item_id,
                data={
                    "metadata": {
                        "parsed_slots": ["petition"],
                        "split_files": {"petition": "dfl-petition"},
                        "parse_artifact_key": "org/x/parsefiles/a.json",
                    }
                },
            )

    class FakeClient:
        beta = SimpleNamespace(agent_data=FakeAgentData())

    from extraction_review.process_file import _run_split_from_file_event

    event = FileEvent(
        job_type="index_parsed",
        filing_type="SLP_CIVIL",
        agent_data_id="agd-index",
        edited=False,
        parsed_slots=["petition"],
        documents=[
            {
                "slot_id": "petition",
                "file_id": "dfl-petition",
                "name": "01_Petition.pdf",
            },
            {
                "slot_id": "index",
                "file_id": "dfl-index",
                "name": "index.pdf",
            },
        ],
    )
    ctx = SimpleNamespace(write_event_to_stream=lambda _ev: None)
    result = await _run_split_from_file_event(event, ctx, FakeClient())  # type: ignore[arg-type]
    assert isinstance(result, BundlePrepared)
    assert captured["skip_extract"] is True
    assert captured["skip_index"] is False
    assert captured["parse_scope"] == "unparsed"
    assert captured["parsed_slots"] == ["petition"]
    assert captured["stitch_in_request_order"] is True
    assert "petition" in captured["reuse_pages_by_slot"]
    assert result.report is None


@pytest.mark.asyncio
async def test_index_parsed_includes_nested_scrutiny_report(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    async def fake_extract(_ctx: object, **_kwargs: object) -> dict[str, object]:
        return {
            "agent_data_id": "agd-index",
            "report": {
                "schema_name": "scrutiny_finding_v1",
                "agent_data_id": "agd-index",
            },
        }

    monkeypatch.setattr(
        "extraction_review.process_file._extract_sliced_parts",
        fake_extract,
    )
    monkeypatch.setattr(
        "extraction_review.s3_artifacts.download_json_object",
        lambda _key: {
            "parsed_slots": ["petition"],
            "document_order": ["petition"],
            "slots": {
                "petition": {
                    "pages": {"1": "kept"},
                    "parse_job_id": "old-parse",
                    "layout": {},
                }
            },
        },
    )

    class FakeAgentData:
        async def get(self, item_id: str):
            return SimpleNamespace(
                id=item_id,
                data={
                    "metadata": {
                        "parsed_slots": ["petition"],
                        "split_files": {"petition": "dfl-petition"},
                        "parse_artifact_key": "org/x/parsefiles/a.json",
                    }
                },
            )

    class FakeClient:
        beta = SimpleNamespace(agent_data=FakeAgentData())

    from extraction_review.process_file import _run_split_from_file_event

    event = FileEvent(
        job_type="index_parsed",
        filing_type="SLP_CIVIL",
        agent_data_id="agd-index",
        edited=False,
        parsed_slots=["petition"],
        documents=[
            {
                "slot_id": "petition",
                "file_id": "dfl-petition",
                "name": "01_Petition.pdf",
            }
        ],
    )
    ctx = SimpleNamespace(write_event_to_stream=lambda _ev: None)
    result = await _run_split_from_file_event(event, ctx, FakeClient())  # type: ignore[arg-type]
    assert isinstance(result, BundlePrepared)
    assert result.agent_data_id == "agd-index"
    assert result.report == {
        "schema_name": "scrutiny_finding_v1",
        "agent_data_id": "agd-index",
    }


@pytest.mark.asyncio
async def test_index_edited_reingests_download_urls(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    ingest_calls: list[str] = []

    async def fake_ingest(_client: object, url: str, **kwargs: object) -> tuple[str, str, str]:
        ingest_calls.append(str(url))
        return (f"dfl-new-{len(ingest_calls)}", "hash", str(kwargs.get("filename") or ""))

    async def fake_extract(_ctx: object, **kwargs: object) -> str:
        return "agd-index"

    monkeypatch.setattr(
        "extraction_review.process_file.ingest_remote_file",
        fake_ingest,
    )
    monkeypatch.setattr(
        "extraction_review.process_file._extract_sliced_parts",
        fake_extract,
    )
    monkeypatch.setattr(
        "extraction_review.process_file.upload_step_json",
        lambda *_args, **_kwargs: None,
    )
    monkeypatch.setattr(
        "extraction_review.s3_artifacts.download_json_object",
        lambda _key: {
            "parsed_slots": ["petition"],
            "slots": {"petition": {"pages": {"1": "old"}}},
        },
    )

    class FakeAgentData:
        async def get(self, item_id: str):
            return SimpleNamespace(
                id=item_id,
                data={
                    "metadata": {
                        "split_files": {"petition": "dfl-old-petition"},
                        "parse_artifact_key": "org/x/parsefiles/a.json",
                    }
                },
            )

    class FakeClient:
        beta = SimpleNamespace(agent_data=FakeAgentData())

    from extraction_review.process_file import _run_split_from_file_event

    event = FileEvent(
        job_type="index_parsed",
        filing_type="SLP_CIVIL",
        agent_data_id="agd-index",
        edited=True,
        parsed_slots=["petition"],
        documents=[
            {
                "name": "01_Petition.pdf",
                "download_url": "https://example.com/petition.pdf",
            }
        ],
    )
    ctx = SimpleNamespace(write_event_to_stream=lambda _ev: None)
    result = await _run_split_from_file_event(event, ctx, FakeClient())  # type: ignore[arg-type]
    assert isinstance(result, BundlePrepared)
    assert ingest_calls == ["https://example.com/petition.pdf"]
    assert result.parts[0].file_id == "dfl-new-1"
