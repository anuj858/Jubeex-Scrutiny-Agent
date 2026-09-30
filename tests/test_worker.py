from unittest.mock import patch

import pytest

from extraction_review.worker import run_worker, worker_kinds


@pytest.mark.asyncio
async def test_worker_requires_tesseract_runtime() -> None:
    with (
        patch("extraction_review.worker.sqs_enabled", return_value=True),
        patch("extraction_review.worker.shutil.which", return_value=None),
        pytest.raises(RuntimeError, match="install tesseract-ocr"),
    ):
        await run_worker(once=True)


def test_worker_kinds_default_to_both(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("JUBEEX_WORKER_KIND", raising=False)

    assert worker_kinds() == ("process_file", "scrutiny")


@pytest.mark.parametrize(
    ("configured", "expected"),
    [
        ("ingestion", ("process_file",)),
        ("process-file", ("process_file",)),
        ("scrutiny", ("scrutiny",)),
    ],
)
def test_worker_kinds_select_one_queue(
    monkeypatch: pytest.MonkeyPatch,
    configured: str,
    expected: tuple[str, ...],
) -> None:
    monkeypatch.setenv("JUBEEX_WORKER_KIND", configured)

    assert worker_kinds() == expected


@pytest.mark.asyncio
async def test_ingestion_worker_only_polls_ingestion_queue(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("JUBEEX_WORKER_KIND", "ingestion")
    with (
        patch("extraction_review.worker.sqs_enabled", return_value=True),
        patch("extraction_review.worker.shutil.which", return_value="/usr/bin/tesseract"),
        patch("extraction_review.worker.receive_jobs", return_value=[]) as receive,
    ):
        await run_worker(once=True)

    receive.assert_called_once_with("process_file")


@pytest.mark.asyncio
async def test_scrutiny_worker_does_not_require_tesseract(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("JUBEEX_WORKER_KIND", "scrutiny")
    with (
        patch("extraction_review.worker.sqs_enabled", return_value=True),
        patch("extraction_review.worker.shutil.which", return_value=None),
        patch("extraction_review.worker.receive_jobs", return_value=[]) as receive,
    ):
        await run_worker(once=True)

    receive.assert_called_once_with("scrutiny")
