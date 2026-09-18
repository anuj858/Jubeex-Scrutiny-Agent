from extraction_review.job_progress import progress_for_status


def test_progress_for_status_is_monotonic_by_stage() -> None:
    assert progress_for_status("Classifying file petition.pdf")[0] == 28
    assert progress_for_status("Splitting file petition.pdf")[0] == 42
    assert progress_for_status("Parsing affidavit (tier1)")[0] == 55
    assert progress_for_status("Extracting matter fields")[0] == 70
    assert progress_for_status("Pinecone indexing complete")[0] == 82


def test_scrutiny_progress_uses_defect_stages() -> None:
    pct, stage = progress_for_status("Checking defect D001", kind="scrutiny")
    assert pct == 40
    assert stage == "checking"


def test_scrutiny_partial_progress_scales_with_completed() -> None:
    from extraction_review.api import JobState, _apply_event_progress

    class _Partial:
        completed = 5
        total = 20

    job = JobState(job_id="test-scrutiny-progress", kind="scrutiny")
    job.progress = 5
    _apply_event_progress(job, _Partial())
    assert job.progress == 20 + int(72 * 5 / 20)
    assert job.stage == "checking"
    assert job.stage_message == "Checked 5/20 defects"
