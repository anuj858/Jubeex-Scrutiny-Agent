"""SQS worker for Agent ingestion and scrutiny jobs."""

from __future__ import annotations

import asyncio
import logging
import os
import uuid
from typing import Any

from dotenv import load_dotenv

from .api import JOBS, JobState, _run_workflow
from .process_file import FileEvent
from .process_file import workflow as process_file_workflow
from .queue import delete_job, receive_jobs, sqs_enabled
from .scrutiny_workflow import ScrutinyEvent
from .scrutiny_workflow import workflow as scrutiny_workflow

load_dotenv()
logger = logging.getLogger(__name__)


def _job_from_message(message: dict[str, Any]) -> JobState:
    job_id = str(message.get("job_id") or uuid.uuid4())
    kind = str(message.get("kind") or "process_file")
    event = message.get("event") if isinstance(message.get("event"), dict) else {}
    job = JobState(
        job_id=job_id,
        kind="scrutiny" if kind == "scrutiny" else "process_file",
        organization_id=event.get("organization_id") or message.get("organization_id"),
        workspace_id=event.get("workspace_id") or message.get("workspace_id"),
        user_id=event.get("user_id") or message.get("user_id"),
    )
    job.callback_url = message.get("callback_url")
    job.event_id = str(message.get("event_id") or uuid.uuid4())
    JOBS[job_id] = job
    return job


async def process_message(message: dict[str, Any]) -> None:
    job = _job_from_message(message)
    event = message.get("event") if isinstance(message.get("event"), dict) else {}
    if job.kind == "scrutiny":
        handler = scrutiny_workflow.run(start_event=ScrutinyEvent(**event))
    else:
        handler = process_file_workflow.run(start_event=FileEvent(**event))
    await _run_workflow(job, handler)
    delete_job(message)


async def run_worker(*, once: bool = False) -> None:
    if not sqs_enabled():
        raise RuntimeError(
            "SQS is not configured. Set JUBEEX_SQS_ENABLED=true or a queue URL."
        )
    kinds = ("process_file", "scrutiny")
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
