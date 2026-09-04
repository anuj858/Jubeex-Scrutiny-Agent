"""Backend file_url ingest for process-file (no FakeLlamaCloudServer)."""

from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock

import httpx
import pytest

from extraction_review.process_file import (
    FileEvent,
    _extract_sliced_parts,
    _filename_from_url,
    _upload_filename,
    compiled_source,
    ingest_remote_file,
    intake_mode,
    slot_id_from_name,
)


def test_file_event_accepts_file_url_without_file_id() -> None:
    event = FileEvent(file_url="https://example.com/a/b/filing.pdf")
    assert event.file_id is None
    assert event.file_url.endswith("filing.pdf")


def test_file_event_requires_file_id_or_url() -> None:
    with pytest.raises(ValueError, match="download_url or file_id"):
        FileEvent()


def test_full_petition_from_parts_slot() -> None:
    event = FileEvent(
        petitiontype="full",
        parts=[
            {
                "slot_id": "fullpetition",
                "file_url": "https://example.com/compiled.pdf",
                "filename": "compiled.pdf",
            }
        ],
    )
    assert intake_mode(event) == "full"
    _id, url, name, _hash = compiled_source(event)
    assert url.endswith("compiled.pdf")
    assert name == "compiled.pdf"


def test_split_parts_require_filing_type() -> None:
    with pytest.raises(ValueError, match="filing_type"):
        FileEvent(
            petitiontype="split",
            parts=[
                {"slot_id": "cover_page", "file_url": "https://example.com/cover.pdf"}
            ],
        )


def test_split_rejects_unknown_filing_type() -> None:
    with pytest.raises(ValueError, match="Unknown filing type"):
        FileEvent(
            job_type="upload_separate",
            filing_type="test123",
            documents=[
                {
                    "slot_id": "petition",
                    "download_url": "https://example.com/petition.pdf",
                }
            ],
        )


def test_swagger_slot_id_string_maps_from_filename() -> None:
    event = FileEvent(
        job_type="upload_separate",
        filing_type="SLP_CIVIL",
        documents=[
            {
                "name": "01_Petition.pdf",
                "document_id": "doc-1",
                "download_url": "https://example.com/01_Petition.pdf",
                "slot_id": "string",
                "file_id": "string",
                "filename": "string",
                "file_url": "string",
            }
        ],
    )
    assert event.documents[0].slot_id == "petition"
    assert event.documents[0].file_id is None
    assert event.documents[0].file_url.endswith("01_Petition.pdf")


def test_split_mode_from_petitiontype() -> None:
    event = FileEvent(
        petitiontype="split",
        filing_type="SLP_CIVIL",
        org_id="org-123",
        parts=[
            {"slot_id": "cover_page", "file_url": "https://example.com/cover.pdf"},
            {"slot_id": "petition", "file_url": "https://example.com/petition.pdf"},
        ],
    )
    assert intake_mode(event) == "split"
    assert event.org_id == "org-123"
    assert len(event.parts) == 2


def test_org_id_blank_string_is_omitted() -> None:
    event = FileEvent(
        petitiontype="split",
        filing_type="SLP_CIVIL",
        org_id="",
        parts=[
            {"slot_id": "cover_page", "file_url": "https://example.com/cover.pdf"},
            {"slot_id": "petition", "file_url": "https://example.com/petition.pdf"},
        ],
    )
    assert event.org_id is None


def test_split_mode_from_slot_ids_without_petitiontype() -> None:
    event = FileEvent(
        filing_type="SLP_CIVIL",
        parts=[
            {"slot_id": "cover_page", "file_url": "https://example.com/cover.pdf"},
            {"slot_id": "petition", "file_url": "https://example.com/petition.pdf"},
        ],
    )
    assert intake_mode(event) == "split"


def test_mixed_full_and_split_slots_require_petitiontype() -> None:
    with pytest.raises(ValueError, match="Mix of fullpetition"):
        FileEvent(
            filing_type="SLP_CIVIL",
            parts=[
                {
                    "slot_id": "fullpetition",
                    "file_url": "https://example.com/compiled.pdf",
                },
                {"slot_id": "cover_page", "file_url": "https://example.com/cover.pdf"},
            ],
        )


def test_process_file_workflow_has_single_stop_event() -> None:
    from extraction_review.process_file import ProcessFileWorkflow

    ProcessFileWorkflow(timeout=None)


def test_slot_id_from_document_name() -> None:
    assert slot_id_from_name("01_Petition.pdf", "SLP_CIVIL") == "petition"
    assert slot_id_from_name("02_List_of_Dates.pdf", "SLP_CIVIL") == "synopsis_lod"
    assert (
        slot_id_from_name("03_Annexure_P1_Impugned_Order.pdf", "SLP_CIVIL")
        == "annexures"
    )
    assert slot_id_from_name("04_Vakalatnama.pdf", "SLP_CIVIL") == "vakalatnama"


def test_upload_separate_documents_payload() -> None:
    event = FileEvent(
        job_type="upload_separate",
        filing_type="SLP_CIVIL",
        organization_id="3f2a9c1e-8b44-4d21-9a70-1c8d4e6b2f11",
        workspace_id="b20c7d91-4e55-48aa-a013-9d6e2f88c104",
        user_id="7b12e4aa-0d55-4c91-b3e8-2a6f19c8d447",
        documents=[
            {
                "name": "01_Petition.pdf",
                "document_id": "11aa22bb-33cc-44dd-85ee-66ff77889900",
                "download_url": "https://example.com/01_Petition.pdf",
            },
            {
                "name": "02_List_of_Dates.pdf",
                "document_id": "22bb33cc-44dd-55ee-96ff-778899001122",
                "download_url": "https://example.com/02_List_of_Dates.pdf",
            },
            {
                "name": "03_Annexure_P1_Impugned_Order.pdf",
                "document_id": "33cc44dd-55ee-66ff-a700-889900112233",
                "download_url": "https://example.com/03_Annexure.pdf",
            },
            {
                "name": "04_Vakalatnama.pdf",
                "document_id": "44dd55ee-66ff-77aa-b811-990011223344",
                "download_url": "https://example.com/04_Vakalatnama.pdf",
            },
        ],
    )
    assert intake_mode(event) == "split"
    assert event.organization_id == "3f2a9c1e-8b44-4d21-9a70-1c8d4e6b2f11"
    assert event.workspace_id == "b20c7d91-4e55-48aa-a013-9d6e2f88c104"
    assert event.user_id == "7b12e4aa-0d55-4c91-b3e8-2a6f19c8d447"
    slots = {item.slot_id for item in event.documents}
    assert slots == {"petition", "synopsis_lod", "annexures", "vakalatnama"}
    petition = next(item for item in event.documents if item.slot_id == "petition")
    assert petition.document_id == "11aa22bb-33cc-44dd-85ee-66ff77889900"
    assert petition.filename == "01_Petition.pdf"


def test_upload_compiled_documents_payload() -> None:
    event = FileEvent(
        job_type="upload_compiled",
        organization_id="org-1",
        workspace_id="ws-1",
        user_id="user-1",
        documents=[
            {
                "name": "Defect_SLP_Civil.pdf",
                "document_id": "doc-compiled-1",
                "download_url": "https://example.com/Defect_SLP_Civil.pdf",
            }
        ],
    )
    assert intake_mode(event) == "full"
    assert event.documents[0].slot_id == "fullpetition"
    _id, url, name, _hash = compiled_source(event)
    assert name == "Defect_SLP_Civil.pdf"
    assert url.endswith("Defect_SLP_Civil.pdf")


def test_filename_from_url_uses_path_basename() -> None:
    assert (
        _filename_from_url(
            "https://bucket.s3.amazonaws.com/uploads/ws/abc-Defect_SLP.pdf?X-Amz-Signature=1"
        )
        == "abc-Defect_SLP.pdf"
    )


def test_upload_filename_strips_unsafe_characters() -> None:
    assert _upload_filename("Cover Page.pdf") == "Cover_Page.pdf"
    assert _upload_filename("AOR's Declaration.pdf") == "AOR_s_Declaration.pdf"
    assert _upload_filename("Defect_SLP_Civil.pdf") == "Defect_SLP_Civil.pdf"


class _EmptyFileList:
    def __aiter__(self):
        async def _gen():
            if False:
                yield None

        return _gen()


class _OneFileList:
    def __init__(self, file_id: str) -> None:
        self._file_id = file_id

    def __aiter__(self):
        async def _gen():
            yield SimpleNamespace(id=self._file_id)

        return _gen()


@pytest.mark.asyncio
async def test_compiled_extract_skips_required_slot_check(
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
            return self._done().__await__()

        async def _done(self):
            return SimpleNamespace(result="agd-compiled-2")

    class FakeWorkflow:
        def __init__(self, timeout: object = None) -> None:
            pass

        def run(self, start_event: object) -> FakeHandler:
            captured["event"] = start_event
            return FakeHandler()

    monkeypatch.setattr(
        "extraction_review.process_split_files.ProcessSplitFilesWorkflow",
        FakeWorkflow,
    )
    ctx = MagicMock()
    ctx.write_event_to_stream = MagicMock()
    agent_data_id = await _extract_sliced_parts(
        ctx,
        filing_type="SLP_CIVIL",
        parts=[],
        echo={
            "job_type": "upload_compiled",
            "organization_id": "org-1",
            "workspace_id": "ws-1",
            "user_id": "user-1",
            "org_id": "org-1",
        },
        require_all_slots=False,
    )
    assert agent_data_id == "agd-compiled-2"
    event = captured["event"]
    assert getattr(event, "require_all_slots") is False
    assert getattr(event, "filing_type") == "SLP_CIVIL"


@pytest.mark.asyncio
async def test_ingest_remote_file_uploads_to_llamacloud(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    pdf_bytes = b"%PDF-1.4 fake"

    class FakeResponse:
        content = pdf_bytes

        def raise_for_status(self) -> None:
            return None

    class FakeAsyncClient:
        def __init__(self, *args: object, **kwargs: object) -> None:
            pass

        async def __aenter__(self) -> "FakeAsyncClient":
            return self

        async def __aexit__(self, *args: object) -> None:
            return None

        async def get(self, url: str) -> FakeResponse:
            assert url.startswith("https://")
            return FakeResponse()

    monkeypatch.setattr(httpx, "AsyncClient", FakeAsyncClient)
    client = MagicMock()
    client.files.list = MagicMock(return_value=_EmptyFileList())
    client.files.create = AsyncMock(return_value=SimpleNamespace(id="dfl-from-url-1"))

    file_id, digest, name = await ingest_remote_file(
        client,
        "https://example.com/path/My_Filing.pdf",
        external_file_id="backend-doc-99",
    )
    assert file_id == "dfl-from-url-1"
    assert name == "My_Filing.pdf"
    assert len(digest) == 64
    kwargs = client.files.create.await_args.kwargs
    assert kwargs["purpose"] == "extract"
    assert kwargs["external_file_id"] == "backend-doc-99"
    assert kwargs["file"][0] == "My_Filing.pdf"


@pytest.mark.asyncio
async def test_ingest_reuses_existing_llamacloud_file(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    class FakeResponse:
        content = b"%PDF-1.4 fake"

        def raise_for_status(self) -> None:
            return None

    class FakeAsyncClient:
        def __init__(self, *args: object, **kwargs: object) -> None:
            pass

        async def __aenter__(self) -> "FakeAsyncClient":
            return self

        async def __aexit__(self, *args: object) -> None:
            return None

        async def get(self, url: str) -> FakeResponse:
            return FakeResponse()

    monkeypatch.setattr(httpx, "AsyncClient", FakeAsyncClient)
    client = MagicMock()
    client.files.list = MagicMock(return_value=_OneFileList("dfl-already-1"))
    client.files.create = AsyncMock()

    file_id, _digest, _name = await ingest_remote_file(
        client,
        "https://example.com/path/My_Filing.pdf",
        external_file_id="backend-doc-99",
    )
    assert file_id == "dfl-already-1"
    client.files.create.assert_not_called()


@pytest.mark.asyncio
async def test_ingest_retries_upload_without_external_id(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from llama_cloud import BadRequestError

    class FakeResponse:
        content = b"%PDF-1.4 fake"

        def raise_for_status(self) -> None:
            return None

    class FakeAsyncClient:
        def __init__(self, *args: object, **kwargs: object) -> None:
            pass

        async def __aenter__(self) -> "FakeAsyncClient":
            return self

        async def __aexit__(self, *args: object) -> None:
            return None

        async def get(self, url: str) -> FakeResponse:
            return FakeResponse()

    monkeypatch.setattr(httpx, "AsyncClient", FakeAsyncClient)
    client = MagicMock()
    client.files.list = MagicMock(return_value=_EmptyFileList())
    err = BadRequestError(
        "Failed to upload file",
        response=MagicMock(status_code=400, headers={}),
        body={"detail": "Failed to upload file"},
    )
    client.files.create = AsyncMock(
        side_effect=[err, SimpleNamespace(id="dfl-retry-1")]
    )

    file_id, _digest, _name = await ingest_remote_file(
        client,
        "https://example.com/path/My_Filing.pdf",
        external_file_id="backend-doc-99",
    )
    assert file_id == "dfl-retry-1"
    assert client.files.create.await_count == 2
    assert "external_file_id" not in client.files.create.await_args.kwargs
