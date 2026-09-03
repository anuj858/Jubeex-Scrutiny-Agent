"""Backend file_url ingest for process-file (no FakeLlamaCloudServer)."""

from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock

import httpx
import pytest

from extraction_review.process_file import (
    FileEvent,
    _filename_from_url,
    ingest_remote_file,
)


def test_file_event_accepts_file_url_without_file_id() -> None:
    event = FileEvent(file_url="https://example.com/a/b/filing.pdf")
    assert event.file_id is None
    assert event.file_url.endswith("filing.pdf")


def test_file_event_requires_file_id_or_url() -> None:
    with pytest.raises(ValueError, match="file_id or file_url"):
        FileEvent()


def test_filename_from_url_uses_path_basename() -> None:
    assert (
        _filename_from_url(
            "https://bucket.s3.amazonaws.com/uploads/ws/abc-Defect_SLP.pdf?X-Amz-Signature=1"
        )
        == "abc-Defect_SLP.pdf"
    )


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
    assert kwargs["file"][1] == pdf_bytes
