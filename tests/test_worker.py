import asyncio
from unittest.mock import patch

import pytest

from extraction_review.worker import (
    _visibility_heartbeat,
    max_job_runtime_seconds,
    process_message,
    run_worker,
    worker_kinds,
)


def test_worker_deadline_is_bounded_and_configurable(monkeypatch) -> None:
    monkeypatch.delenv("JUBEEX_WORKER_MAX_JOB_SECONDS", raising=False)
    assert max_job_runtime_seconds() == 5400
    monkeypatch.setenv("JUBEEX_WORKER_MAX_JOB_SECONDS", "600")
    assert max_job_runtime_seconds() == 600
    monkeypatch.setenv("JUBEEX_WORKER_MAX_JOB_SECONDS", "0")
    with pytest.raises(RuntimeError, match="at least 60"):
        max_job_runtime_seconds()


@pytest.mark.asyncio
async def test_worker_requires_tesseract_runtime() -> None:
    with (
        patch("extraction_review.worker.sqs_enabled", return_value=True),
        patch("extraction_review.worker.shutil.which", return_value=None),
        pytest.raises(RuntimeError, match="install tesseract-ocr"),
    ):
        await run_worker(once=True)


@pytest.mark.asyncio
async def test_worker_rejects_malformed_sqs_job() -> None:
    with pytest.raises(ValueError, match="requires job_id and event"):
        await process_message({"kind": "scrutiny"})


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
        patch("extraction_review.worker.JobLease") as lease_type,
        patch("extraction_review.worker.delete_job") as delete,
        patch("extraction_review.worker._run_workflow") as run,
    ):
        await process_message(message)

    lease_type.return_value.acquire.assert_called_once()
    lease_type.return_value.release.assert_called_once()
    delete.assert_called_once_with(message)
    run.assert_not_called()


@pytest.mark.asyncio
async def test_terminal_replay_retries_undelivered_callback() -> None:
    message = {
        "job_id": "job-callback",
        "kind": "scrutiny",
        "callback_url": "https://backend.example/webhook",
        "event_id": "event-1",
    }
    prior = {"status": "completed", "result": {}, "callback_delivered": False}
    with (
        patch("extraction_review.worker.load_job_status", return_value=prior),
        patch("extraction_review.worker.JobLease"),
        patch(
            "extraction_review.worker.notify_job_finished", return_value=True
        ) as notify,
        patch("extraction_review.worker.save_job_status", return_value=True) as save,
        patch("extraction_review.worker.delete_job") as delete,
    ):
        await process_message(message)

    notify.assert_awaited_once()
    assert save.call_args.args[0]["callback_delivered"] is True
    delete.assert_called_once_with(message)


@pytest.mark.asyncio
async def test_terminal_replay_keeps_message_if_callback_fails() -> None:
    message = {
        "job_id": "job-callback",
        "kind": "scrutiny",
        "callback_url": "https://backend.example/webhook",
    }
    with (
        patch(
            "extraction_review.worker.load_job_status",
            return_value={"status": "failed", "callback_delivered": False},
        ),
        patch("extraction_review.worker.JobLease"),
        patch("extraction_review.worker.notify_job_finished", return_value=False),
        patch("extraction_review.worker.delete_job") as delete,
        pytest.raises(RuntimeError, match="Callback delivery pending"),
    ):
        await process_message(message)

    delete.assert_not_called()


@pytest.mark.asyncio
async def test_terminal_replay_does_not_callback_without_job_lease() -> None:
    message = {
        "job_id": "job-callback",
        "kind": "scrutiny",
        "callback_url": "https://backend.example/webhook",
    }
    with (
        patch("extraction_review.worker.JobLease") as lease_type,
        patch("extraction_review.worker.load_job_status") as load,
        patch("extraction_review.worker.notify_job_finished") as notify,
        patch("extraction_review.worker.delete_job") as delete,
    ):
        lease_type.return_value.acquire.return_value = False
        await process_message(message)

    load.assert_not_called()
    notify.assert_not_called()
    delete.assert_not_called()


@pytest.mark.asyncio
async def test_shutdown_cancels_active_job_and_releases_visibility() -> None:
    stop = asyncio.Event()
    started = asyncio.Event()
    cancelled = asyncio.Event()
    message = {
        "job_id": "job-active",
        "kind": "scrutiny",
        "event": {},
        "_queue_url": "https://sqs/queue",
        "_receipt_handle": "receipt",
    }

    async def active_job(_message) -> None:
        started.set()
        try:
            await asyncio.Event().wait()
        except asyncio.CancelledError:
            cancelled.set()
            raise

    with (
        patch("extraction_review.worker.sqs_enabled", return_value=True),
        patch(
            "extraction_review.worker.shutil.which", return_value="/usr/bin/tesseract"
        ),
        patch("extraction_review.worker.worker_kinds", return_value=("scrutiny",)),
        patch("extraction_review.worker.receive_jobs", return_value=[message]),
        patch("extraction_review.worker.process_message", side_effect=active_job),
        patch("extraction_review.worker.release_job_visibility") as release,
    ):
        worker = asyncio.create_task(run_worker(stop_requested=stop))
        await asyncio.wait_for(started.wait(), timeout=2)
        stop.set()
        await asyncio.wait_for(worker, timeout=2)

    assert cancelled.is_set()
    release.assert_called_once_with(message)


@pytest.mark.asyncio
async def test_claimed_job_is_not_run_by_second_worker() -> None:
    message = {"job_id": "job-active", "kind": "scrutiny", "event": {}}
    lease = patch("extraction_review.worker.JobLease")
    with (
        patch("extraction_review.worker.load_job_status", return_value=None),
        lease as lease_type,
        patch("extraction_review.worker._run_workflow") as run,
        patch("extraction_review.worker.delete_job") as delete,
    ):
        lease_type.return_value.acquire.return_value = False
        await process_message(message)

    run.assert_not_called()
    delete.assert_not_called()


@pytest.mark.asyncio
async def test_lost_visibility_lease_stops_worker() -> None:
    with (
        patch(
            "extraction_review.worker._heartbeat_interval_seconds", return_value=0.01
        ),
        patch(
            "extraction_review.worker.extend_job_visibility",
            side_effect=RuntimeError("SQS unavailable"),
        ),
        pytest.raises(RuntimeError, match="Lost SQS visibility lease"),
    ):
        await _visibility_heartbeat(
            {"job_id": "job-1"},
            asyncio.Event(),
            lease=object(),  # type: ignore[arg-type]
        )
