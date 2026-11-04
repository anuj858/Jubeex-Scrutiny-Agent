from unittest.mock import Mock

from extraction_review.job_progress import progress_for_status


def test_progress_for_status_is_monotonic_by_stage() -> None:
    assert progress_for_status("Classifying file petition.pdf")[0] == 28
    assert progress_for_status("Splitting file petition.pdf")[0] == 42
    assert progress_for_status("Parsing affidavit (tier1)")[0] == 55
    assert progress_for_status("Extracting matter fields")[0] == 70
    assert progress_for_status("Pinecone indexing complete")[0] == 82


def test_annexure_index_progress_uses_its_own_stages() -> None:
    pct, stage = progress_for_status(
        "Describing 2 annexure(s) with the LLM",
        kind="annexure_index",
    )
    assert pct == 80
    assert stage == "describe"


def test_scrutiny_progress_uses_defect_stages() -> None:
    pct, stage = progress_for_status("Checking defect D001", kind="scrutiny")
    assert pct == 40
    assert stage == "checking"


def test_incomplete_coverage_warning_does_not_jump_to_report_progress() -> None:
    pct, stage = progress_for_status(
        "Coverage will be incomplete because Pinecone is disabled",
        kind="scrutiny",
    )
    assert pct == 28
    assert stage == "retrieve"


def test_scrutiny_partial_progress_scales_with_completed() -> None:
    from extraction_review.api import JobState, _apply_event_progress

    class _Partial:
        completed = 5
        total = 20

    job = JobState(job_id="test-scrutiny-progress", kind="scrutiny")
    job.progress = 5
    _apply_event_progress(job, _Partial())
    assert job.progress == 20 + int(68 * 5 / 20)
    assert job.stage == "checking"
    assert job.stage_message == "Checked 5/20 defects"


def test_completed_defect_checks_leave_room_for_report_generation() -> None:
    from extraction_review.api import JobState, _apply_event_progress

    partial = type("_Partial", (), {"completed": 20, "total": 20})()
    job = JobState(job_id="test-scrutiny-report-progress", kind="scrutiny")

    _apply_event_progress(job, partial)

    assert job.progress == 88
    assert job.stage == "checking"

    status = type("_Status", (), {"message": "Writing final scrutiny report"})()
    _apply_event_progress(job, status)

    assert job.progress == 92
    assert job.stage == "report"


def test_process_stage_transition_is_persisted_without_throttle() -> None:
    from extraction_review.api import JobState, _apply_event_progress

    job = JobState(job_id="test-stage-transition", kind="process_file")
    job.stage = "running"
    job.persist = Mock()  # type: ignore[method-assign]

    event = type("_Status", (), {"message": "Splitting file petition.pdf"})()
    _apply_event_progress(job, event)

    assert job.stage == "split"
    assert job.stage_message == "Splitting file petition.pdf"
    job.persist.assert_called_once_with(force=True)
