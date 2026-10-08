"""SQS worker for Agent ingestion and scrutiny jobs."""

from __future__ import annotations

import asyncio
import logging
import os
import shutil
import uuid
from contextlib import suppress
from typing import Any

from dotenv import load_dotenv

from .annexure_index import AnnexureIndexEvent
from .annexure_index import workflow as annexure_index_workflow
from .api import JOBS, JobState, _run_workflow
from .job_progress import load_job_status
from .process_file import FileEvent
from .process_file import workflow as process_file_workflow
from .queue import (
    delete_job,
    extend_job_visibility,
    receive_jobs,
    sqs_enabled,
    visibility_timeout,
)
from .s3_artifacts import set_job_context
from .scrutiny_workflow import ScrutinyEvent
from .scrutiny_workflow import workflow as scrutiny_workflow

load_dotenv()
logger = logging.getLogger(__name__)

TERMINAL_JOB_STATUSES = frozenset({"completed", "failed"})


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


async def _visibility_heartbeat(message: dict[str, Any], stop: asyncio.Event) -> None:
    interval = _heartbeat_interval_seconds()
    while True:
        try:
            await asyncio.wait_for(stop.wait(), timeout=interval)
            return
        except TimeoutError:
            try:
                await asyncio.to_thread(extend_job_visibility, message)
            except Exception:
                logger.warning(
                    "Could not renew SQS visibility for job %s",
                    message.get("job_id"),
                    exc_info=True,
                )


def _job_from_message(message: dict[str, Any]) -> JobState:
    job_id = str(message.get("job_id") or uuid.uuid4())
    kind = str(message.get("kind") or "process_file")
    event = message.get("event") if isinstance(message.get("event"), dict) else {}
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


async def process_message(message: dict[str, Any]) -> None:
    job_id = str(message.get("job_id") or "").strip()
    prior = load_job_status(job_id) if job_id else None
    if str((prior or {}).get("status") or "").lower() in TERMINAL_JOB_STATUSES:
        logger.info("Discarding terminal replay for job %s", job_id)
        delete_job(message)
        return
    job = _job_from_message(message)
    event = message.get("event") if isinstance(message.get("event"), dict) else {}
    set_job_context(job.job_id, job.organization_id, job.workspace_id)
    if job.kind == "scrutiny":
        handler = scrutiny_workflow.run(start_event=ScrutinyEvent(**event))
    elif job.kind == "annexure_index":
        handler = annexure_index_workflow.run(start_event=AnnexureIndexEvent(**event))
    else:
        handler = process_file_workflow.run(start_event=FileEvent(**event))
    stop_heartbeat = asyncio.Event()
    heartbeat = asyncio.create_task(_visibility_heartbeat(message, stop_heartbeat))
    try:
        await _run_workflow(job, handler)
        delete_job(message)
    finally:
        stop_heartbeat.set()
        heartbeat.cancel()
        with suppress(asyncio.CancelledError):
            await heartbeat


async def run_worker(*, once: bool = False) -> None:
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
        received = False
        for kind in kinds:
            for message in receive_jobs(kind):
                received = True
                try:
                    await process_message(message)
                except Exception:
                    logger.exception("SQS %s message failed", kind)
        if once:
            return
        if not received:
            await asyncio.sleep(1)


def main() -> None:
    logging.basicConfig(level=os.getenv("LOG_LEVEL", "INFO"))
    asyncio.run(run_worker())


if __name__ == "__main__":
    main()
