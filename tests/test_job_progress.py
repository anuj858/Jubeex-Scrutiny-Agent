from extraction_review.job_progress import progress_for_status


def test_progress_for_status_is_monotonic_by_stage() -> None:
    assert progress_for_status("Classifying file petition.pdf")[0] == 28
    assert progress_for_status("Splitting file petition.pdf")[0] == 42
    assert progress_for_status("Parsing affidavit (tier1)")[0] == 55
    assert progress_for_status("Extracting matter fields")[0] == 70
    assert progress_for_status("Pinecone indexing complete")[0] == 82


def test_scrutiny_progress_uses_defect_stages() -> None:
    pct, stage = progress_for_status("Checking defect D001", kind="scrutiny")
    assert pct == 55
    assert stage == "checking"
