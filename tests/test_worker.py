from unittest.mock import patch

import pytest

from extraction_review.worker import process_message, run_worker, worker_kinds


@pytest.mark.asyncio
async def test_worker_requires_tesseract_runtime() -> None:
    with (
        patch("extraction_review.worker.sqs_enabled", return_value=True),
        patch("extraction_review.worker.shutil.which", return_value=None),
        pytest.raises(RuntimeError, match="install tesseract-ocr"),
    ):
        await run_worker(once=True)


def test_worker_kinds_can_isolate_ingestion(monkeypatch) -> None:
    monkeypatch.setenv("JUBEEX_WORKER_KINDS", "ingestion")
    assert worker_kinds() == ("process_file",)


def test_worker_kinds_can_isolate_scrutiny(monkeypatch) -> None:
    monkeypatch.setenv("JUBEEX_WORKER_KINDS", "scrutiny,annexure_index")
    assert worker_kinds() == ("scrutiny",)


@pytest.mark.asyncio
async def test_terminal_job_replay_is_deleted_without_running() -> None:
    message = {"job_id": "job-complete", "kind": "process_file"}
    with (
        patch(
            "extraction_review.worker.load_job_status",
            return_value={"status": "completed"},
        ),
        patch("extraction_review.worker.delete_job") as delete,
        patch("extraction_review.worker._run_workflow") as run,
    ):
        await process_message(message)

    delete.assert_called_once_with(message)
    run.assert_not_called()
