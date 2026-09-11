from extraction_review.job_timing import (
    format_duration,
    timing_payload,
    timing_status_message,
    uploaded_filename,
)


def test_uploaded_filename_uses_the_uploaded_doc_name_only() -> None:
    assert uploaded_filename("Defect_SLP_Civil.pdf") == "Defect_SLP_Civil.pdf"
    assert uploaded_filename("folder/Defect_SLP_Civil.pdf") == "Defect_SLP_Civil.pdf"
    assert uploaded_filename(r"C:\\uploads\\Defect_SLP_Civil.pdf") == "Defect_SLP_Civil.pdf"
    assert uploaded_filename("  ") is None


def test_timing_payload_sums_the_full_journey() -> None:
    payload = timing_payload(
        file_name="folder/Defect_SLP_Civil.pdf",
        classify_split_seconds=12.4,
        parse_extract_seconds=40.6,
    )
    assert payload["file_name"] == "Defect_SLP_Civil.pdf"
    assert payload["classify_split_seconds"] == 12.4
    assert payload["parse_extract_seconds"] == 40.6
    assert payload["total_seconds"] == 53.0
    assert payload["classify_split"] == "12s"
    assert payload["parse_extract"] == "41s"
    assert payload["total"] == "53s"


def test_attach_timing_writes_top_level_and_metadata() -> None:
    from extraction_review.job_timing import attach_timing

    timing = timing_payload(
        file_name="Defect_SLP_Civil.pdf",
        classify_split_seconds=12.4,
    )
    stamped = attach_timing({"filing_type": "SLP_CIVIL"}, timing)
    assert stamped["timing"]["classify_split_seconds"] == 12.4
    assert stamped["metadata"]["timing"]["classify_split"] == "12s"


def test_timing_status_message_names_both_buttons() -> None:
    message = timing_status_message(
        file_name="Defect_SLP_Civil.pdf",
        classify_split_seconds=12.4,
        parse_extract_seconds=40.6,
    )
    assert message.startswith("Defect_SLP_Civil.pdf:")
    assert "bundle upload" in message
    assert "slot submit" in message
    assert "total" in message


def test_timing_status_message_bundle_only() -> None:
    message = timing_status_message(
        file_name="Defect_SLP_Civil.pdf",
        classify_split_seconds=134.0,
    )
    assert message == "Defect_SLP_Civil.pdf: bundle upload 2m 14s"
    assert "slot submit" not in message


def test_format_duration_compact() -> None:
    assert format_duration(3.2) == "3.2s"
    assert format_duration(12) == "12s"
    assert "m" in format_duration(90)
