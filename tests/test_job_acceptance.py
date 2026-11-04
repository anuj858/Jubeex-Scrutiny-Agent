from unittest.mock import patch

import pytest
from fastapi import HTTPException

from extraction_review.api import JOBS, JobState, _persist_accepted_job


def test_sqs_job_is_not_accepted_without_durable_status() -> None:
    job = JobState(job_id="not-persisted", kind="scrutiny")
    JOBS[job.job_id] = job
    with (
        patch.object(job, "persist", return_value=False),
        patch("extraction_review.api.sqs_enabled", return_value=True),
        pytest.raises(HTTPException) as error,
    ):
        _persist_accepted_job(job)
    assert error.value.status_code == 503
    assert job.job_id not in JOBS


def test_local_job_can_use_in_memory_status() -> None:
    job = JobState(job_id="local-only", kind="scrutiny")
    with (
        patch.object(job, "persist", return_value=False),
        patch("extraction_review.api.sqs_enabled", return_value=False),
    ):
        _persist_accepted_job(job)
