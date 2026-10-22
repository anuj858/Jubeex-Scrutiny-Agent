"""SQS worker for Agent ingestion and scrutiny jobs."""

from __future__ import annotations

import asyncio
import logging
import os
import shutil
import signal
import uuid
from contextlib import suppress
from typing import Any

from dotenv import load_dotenv

from .annexure_index import AnnexureIndexEvent
from .annexure_index import workflow as annexure_index_workflow
from .api import JOBS, JobState, _run_workflow
from .callbacks import notify_job_finished
from .job_lease import JobLease
from .job_progress import load_job_status, save_job_status
from .logging_utils import silence_http_client_request_logs
from .process_file import FileEvent
from .process_file import workflow as process_file_workflow
from .queue import (
    delete_job,
    extend_job_visibility,
    receive_jobs,
    release_job_visibility,
    sqs_enabled,
    visibility_timeout,
)
from .s3_artifacts import set_job_context
from .scrutiny_workflow import ScrutinyEvent
from .scrutiny_workflow import workflow as scrutiny_workflow

load_dotenv()
logger = logging.getLogger(__name__)

TERMINAL_JOB_STATUSES = frozenset({"completed", "failed"})
DEFAULT_MAX_JOB_SECONDS = 5400


def max_job_runtime_seconds() -> int:
    raw = (os.getenv("JUBEEX_WORKER_MAX_JOB_SECONDS") or "").strip()
    if not raw:
        return DEFAULT_MAX_JOB_SECONDS
    try:
        value = int(raw)
    except ValueError as exc:
        raise RuntimeError("JUBEEX_WORKER_MAX_JOB_SECONDS must be an integer") from exc
    if value < 60:
        raise RuntimeError("JUBEEX_WORKER_MAX_JOB_SECONDS must be at least 60")
    return value


def worker_kinds() -> tuple[str, ...]:
    """Queues consumed by this process.

    Dedicated ECS services should set ``JUBEEX_WORKER_KINDS=process_file`` or
    ``JUBEEX_WORKER_KINDS=scrutiny`` so slow scrutiny cannot starve intake.
    """
    raw = (os.getenv("JUBEEX_WORKER_KINDS") or "").strip()
    if not raw:
        return ("process_file", "scrutiny")
    aliases = {
        "process_file": "process_file",
        "ingestion": "process_file",
        "split": "process_file",
        "extract": "process_file",
        "scrutiny": "scrutiny",
        "annexure_index": "scrutiny",
    }
    kinds: list[str] = []
    invalid: list[str] = []
    for token in (part.strip().lower() for part in raw.split(",")):
        if not token:
            continue
        kind = aliases.get(token)
        if kind is None:
            invalid.append(token)
        elif kind not in kinds:
            kinds.append(kind)
    if invalid or not kinds:
        allowed = ", ".join(sorted(aliases))
        raise RuntimeError(
            f"Invalid JUBEEX_WORKER_KINDS value(s): {', '.join(invalid) or raw}. "
            f"Allowed: {allowed}"
        )
    return tuple(kinds)


def _heartbeat_interval_seconds() -> int:
    raw = (os.getenv("JUBEEX_SQS_HEARTBEAT_SECONDS") or "").strip()
    try:
        if raw:
            return max(1, int(raw))
    except ValueError:
        pass
    return max(30, min(300, visibility_timeout() // 3))


async def _visibility_heartbeat(
    message: dict[str, Any], stop: asyncio.Event, lease: JobLease
) -> None:
    interval = _heartbeat_interval_seconds()
    while True:
        try:
            await asyncio.wait_for(stop.wait(), timeout=interval)
            return
        except TimeoutError:
            try:
                await asyncio.to_thread(extend_job_visibility, message)
                await asyncio.to_thread(lease.renew)
            except Exception:
                logger.warning(
                    "Could not renew SQS visibility for job %s",
                    message.get("job_id"),
                    exc_info=True,
                )
                raise RuntimeError("Lost SQS visibility lease")


def _job_from_message(message: dict[str, Any]) -> JobState:
    job_id = str(message.get("job_id") or uuid.uuid4())
    kind = str(message.get("kind") or "process_file")
    event = _event_from_message(message)
    if kind == "scrutiny":
        job_kind = "scrutiny"
    elif kind == "annexure_index":
        job_kind = "annexure_index"
    else:
        job_kind = "process_file"
    job = JobState(
        job_id=job_id,
        kind=job_kind,
        organization_id=event.get("organization_id") or message.get("organization_id"),
        workspace_id=event.get("workspace_id") or message.get("workspace_id"),
        user_id=event.get("user_id") or message.get("user_id"),
    )
    job.callback_url = message.get("callback_url")
    job.event_id = str(message.get("event_id") or uuid.uuid4())
    callback_kind = str(message.get("callback_kind") or "").strip()
    if callback_kind in {"scrutiny", "process_file", "annexure_index"}:
        job.callback_kind = callback_kind  # type: ignore[assignment]
    prior = load_job_status(job_id)
    if prior:
        created = prior.get("created_at")
        if isinstance(created, str) and created:
            job.created_at = created
        progress = prior.get("progress")
        if isinstance(progress, int) and progress > job.progress:
            job.progress = progress
    JOBS[job_id] = job
    job.persist(force=True)
    return job


def _event_from_message(message: dict[str, Any]) -> dict[str, Any]:
    event = message.get("event")
    return event if isinstance(event, dict) else {}


async def _replay_terminal_message(
    message: dict[str, Any], prior: dict[str, Any]
) -> None:
    job_id = str(message["job_id"])
    if message.get("callback_url") and not prior.get("callback_delivered", False):
        result = prior.get("result")
        result = result if isinstance(result, dict) else None
        delivered = await notify_job_finished(
            callback_url=str(message["callback_url"]),
            job_id=job_id,
            kind=str(message.get("callback_kind") or message.get("kind") or ""),
            status=str(prior["status"]),
            agent_data_id=prior.get("agent_data_id"),
            organization_id=prior.get("organization_id"),
            workspace_id=prior.get("workspace_id"),
            error=prior.get("error"),
            result=result,
            artifacts=result.get("artifacts") if result else None,
            event_id=str(message.get("event_id") or job_id),
        )
        if not delivered:
            raise RuntimeError(f"Callback delivery pending for job {job_id}")
        if not save_job_status({**prior, "callback_delivered": True}, force=True):
            raise RuntimeError(f"Could not persist callback delivery for {job_id}")
    logger.info("Discarding terminal replay for job %s", job_id)
    delete_job(message)


async def process_message(message: dict[str, Any]) -> None:
    job_id = str(message.get("job_id") or "").strip()
    if not job_id:
        raise ValueError("SQS job requires job_id and event object")
    if message.get("kind") not in {"process_file", "scrutiny", "annexure_index"}:
        raise ValueError(f"Unsupported SQS job kind: {message.get('kind')!r}")
    # Terminal replays need the lease too. Otherwise two redeliveries can send
    # the same callback while an active worker is persisting completion.
    lease = JobLease(job_id)
    if not await asyncio.to_thread(lease.acquire):
        logger.info("Job %s is already held by another worker", job_id)
        return
    try:
        prior = load_job_status(job_id)
        if prior and str(prior.get("status") or "").lower() in TERMINAL_JOB_STATUSES:
            await _replay_terminal_message(message, prior)
            return
        if not isinstance(message.get("event"), dict):
            raise ValueError("SQS job requires job_id and event object")  # noqa: TRY004
        await _process_claimed_message(message, lease)
    finally:
        try:
            await asyncio.to_thread(lease.release)
        except Exception:
            logger.warning(
                "Could not release S3 lease for job %s", job_id, exc_info=True
            )
        JOBS.pop(job_id, None)


async def _process_claimed_message(message: dict[str, Any], lease: JobLease) -> None:
    job_id = str(message.get("job_id") or "")
    prior = load_job_status(job_id)
    if prior and str(prior.get("status") or "").lower() in TERMINAL_JOB_STATUSES:
        await _replay_terminal_message(message, prior)
        return
    job = _job_from_message(message)
    event = _event_from_message(message)
    set_job_context(job.job_id, job.organization_id, job.workspace_id)
    if job.kind == "scrutiny":
        handler = scrutiny_workflow.run(start_event=ScrutinyEvent(**event))
    elif job.kind == "annexure_index":
        handler = annexure_index_workflow.run(start_event=AnnexureIndexEvent(**event))
    else:
        handler = process_file_workflow.run(start_event=FileEvent(**event))
    stop_heartbeat = asyncio.Event()
    heartbeat = asyncio.create_task(
        _visibility_heartbeat(message, stop_heartbeat, lease)
    )
    workflow_task = asyncio.create_task(
        _run_workflow(
            job,
            handler,
            require_durable_status=True,
            max_runtime_seconds=max_job_runtime_seconds(),
        )
    )
    try:
        done, _ = await asyncio.wait(
            {workflow_task, heartbeat}, return_when=asyncio.FIRST_COMPLETED
        )
        if heartbeat in done:
            heartbeat.result()
            raise RuntimeError(f"SQS visibility heartbeat stopped for job {job_id}")
        delivered = await workflow_task
        if not delivered:
            raise RuntimeError(f"Callback delivery pending for job {job_id}")
        delete_job(message)
    finally:
        if not workflow_task.done():
            workflow_task.cancel()
            with suppress(asyncio.CancelledError):
                await workflow_task
        stop_heartbeat.set()
        heartbeat.cancel()
        with suppress(asyncio.CancelledError):
            await heartbeat


async def run_worker(
    *, once: bool = False, stop_requested: asyncio.Event | None = None
) -> None:
    if not sqs_enabled():
        raise RuntimeError(
            "SQS is not configured. Set JUBEEX_SQS_ENABLED=true or a queue URL."
        )
    if not shutil.which("tesseract"):
        raise RuntimeError(
            "Split OCR is unavailable: install tesseract-ocr in the worker image."
        )
    kinds = worker_kinds()
    while True:
        if stop_requested is not None and stop_requested.is_set():
            return
        received = False
        for kind in kinds:
            if stop_requested is not None and stop_requested.is_set():
                return
            for message in await asyncio.to_thread(receive_jobs, kind):
                if stop_requested is not None and stop_requested.is_set():
                    await asyncio.to_thread(release_job_visibility, message)
                    return
                received = True
                try:
                    if stop_requested is None:
                        await process_message(message)
                    else:
                        task = asyncio.create_task(process_message(message))
                        stopping = asyncio.create_task(stop_requested.wait())
                        try:
                            done, _ = await asyncio.wait(
                                {task, stopping}, return_when=asyncio.FIRST_COMPLETED
                            )
                            if task in done:
                                await task
                            else:
                                task.cancel()
                                try:
                                    with suppress(asyncio.CancelledError):
                                        await task
                                finally:
                                    await asyncio.to_thread(
                                        release_job_visibility, message
                                    )
                                return
                        finally:
                            stopping.cancel()
                            with suppress(asyncio.CancelledError):
                                await stopping
                except Exception:
                    logger.exception("SQS %s message failed", kind)
        if once:
            return
        if not received:
            await asyncio.sleep(1)


async def _serve_worker() -> None:
    stop_requested = asyncio.Event()
    loop = asyncio.get_running_loop()
    loop.add_signal_handler(signal.SIGTERM, stop_requested.set)
    await run_worker(stop_requested=stop_requested)


def main() -> None:
    logging.basicConfig(level=os.getenv("LOG_LEVEL", "INFO"))
    silence_http_client_request_logs()
    asyncio.run(_serve_worker())


if __name__ == "__main__":
    main()
