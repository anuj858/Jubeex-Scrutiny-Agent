import asyncio
from unittest.mock import AsyncMock, patch

import pytest

from extraction_review.api import JobState, _run_workflow


@pytest.mark.asyncio
async def test_workflow_deadline_fails_job_and_sends_terminal_callback() -> None:
    job = JobState(job_id="job-timeout", kind="scrutiny")
    job.callback_url = "https://backend.example/webhook"
    handler = asyncio.sleep(60)
    with (
        patch.object(job, "persist", return_value=True),
        patch("extraction_review.api.recorded_artifacts", return_value={}),
        patch(
            "extraction_review.api.notify_job_finished",
            new_callable=AsyncMock,
            return_value=True,
        ) as callback,
    ):
        delivered = await _run_workflow(
            job,
            handler,
            require_durable_status=True,
            max_runtime_seconds=0.01,  # type: ignore[arg-type]
        )

    assert delivered is True
    assert job.status == "failed"
    assert job.stage == "failed"
    assert "deadline" in (job.error or "")
    callback.assert_awaited_once()
    assert callback.call_args.kwargs["status"] == "failed"
