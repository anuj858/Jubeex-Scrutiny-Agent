"""Regression tests for boundary-first compiled-PDF page ownership."""

import pymupdf
import pytest

from extraction_review.bundle_slicer import (
    slice_bundle_pdf,
    validate_sliced_page_ownership,
)
from extraction_review.process_file import (
    _document_first_page_ownership,
    attach_split_output_trace,
)
from extraction_review.split_audit import audit_compiled_split
from extraction_review.split_repair import (
    _has_leading_high_court_caption,
    _is_lower_court_caption,
)
from extraction_review.split_upload import type_catalog
from extraction_review.structure_split import PageUnit, structure_aware_split


def _proposal(number: int, start: int, end: int) -> dict:
    return {
        "segment_id": f"boundary-{number}",
        "start_page": start,
        "end_page": end,
        "pages": list(range(start, end + 1)),
        "source": "llama_boundary_detection",
    }


def _report(*segments: tuple[str, list[int]]) -> dict:
    return {
        "segments": [
            {
                "category": category,
                "document_type": category,
                "source_pages": pages,
            }
            for category, pages in segments
        ]
    }


def test_internal_cover_page_heading_does_not_split_annexure() -> None:
    page_parts, segments, issues = _document_first_page_ownership(
        [_proposal(1, 1, 3)],
        _report(("Annexure P-1", [1, 2]), ("Cover Page", [3])),
        page_count=3,
    )

    assert page_parts == {page: ["Annexure P-1"] for page in (1, 2, 3)}
    assert segments[0]["pages"] == [1, 2, 3]
    assert segments[0]["review_reasons"] == ["conflicting_categories_inside_boundary"]
    assert issues[0]["code"] == "boundary_classification_conflict"


def test_petition_word_in_annexure_body_does_not_change_owner() -> None:
    page_parts, _, _ = _document_first_page_ownership(
        [_proposal(1, 10, 12)],
        _report(("Annexure P-4", [10, 11]), ("Main Petition", [12])),
        page_count=12,
    )

    assert all(page_parts[page] == ["Annexure P-4"] for page in range(10, 13))


def test_captionless_annexure_continuations_keep_complete_candidate_range() -> None:
    page_parts, segments, _ = _document_first_page_ownership(
        [_proposal(1, 1, 5)],
        _report(("Annexure P-7", [1])),
        page_count=5,
    )

    assert set(page_parts) == set(range(1, 6))
    assert {tuple(labels) for labels in page_parts.values()} == {("Annexure P-7",)}
    assert segments[0]["type_evidence"] == ["consistent_overlapping_classification"]


def test_consecutive_similar_annexures_remain_distinct_by_boundary() -> None:
    page_parts, segments, _ = _document_first_page_ownership(
        [_proposal(1, 1, 3), _proposal(2, 4, 6)],
        _report(("Annexure P-1", [1, 2, 3]), ("Annexure P-2", [4, 5, 6])),
        page_count=6,
    )

    assert [segment["category"] for segment in segments] == [
        "Annexure P-1",
        "Annexure P-2",
    ]
    assert page_parts[3] == ["Annexure P-1"]
    assert page_parts[4] == ["Annexure P-2"]


def test_real_document_after_annexure_uses_next_candidate_boundary() -> None:
    page_parts, _, _ = _document_first_page_ownership(
        [_proposal(1, 1, 3), _proposal(2, 4, 5)],
        _report(("Annexure P-1", [1, 2, 3]), ("Affidavit", [4, 5])),
        page_count=5,
    )

    assert page_parts[3] == ["Annexure P-1"]
    assert page_parts[4] == ["Affidavit"]
    assert page_parts[5] == ["Affidavit"]


def test_boundary_without_title_still_governs_document_ownership() -> None:
    page_parts, _, _ = _document_first_page_ownership(
        [_proposal(1, 1, 2), _proposal(2, 3, 4)],
        _report(("Synopsis", [1, 2]), ("List of Dates & Events", [3, 4])),
        page_count=4,
    )

    assert page_parts == {
        1: ["Synopsis"],
        2: ["Synopsis"],
        3: ["List of Dates & Events"],
        4: ["List of Dates & Events"],
    }


def test_signature_blank_and_verification_pages_follow_candidate_document() -> None:
    page_parts, _, _ = _document_first_page_ownership(
        [_proposal(1, 1, 4)],
        _report(("Affidavit", [1]), ("Main Petition", [2, 3, 4])),
        page_count=4,
    )

    assert all(page_parts[page] == ["Affidavit"] for page in range(1, 5))


def test_incorrect_one_page_llama_segment_cannot_fragment_document() -> None:
    page_parts, segments, _ = _document_first_page_ownership(
        [_proposal(1, 1, 6)],
        _report(
            ("Main Petition", [1, 2, 3]),
            ("Cover Page", [4]),
            ("Main Petition", [5, 6]),
        ),
        page_count=6,
    )

    # The malformed non-contiguous classifier evidence is still only type
    # evidence; the accepted physical boundary remains one complete range.
    assert all(page_parts[page] == ["Main Petition"] for page in range(1, 7))
    assert segments[0]["pages"] == list(range(1, 7))


def test_missing_or_misleading_classification_is_left_for_review() -> None:
    page_parts, segments, issues = _document_first_page_ownership(
        [_proposal(1, 1, 2)],
        _report(),
        page_count=4,
    )

    assert page_parts == {
        1: ["Undefined"],
        2: ["Undefined"],
        3: ["Undefined"],
        4: ["Undefined"],
    }
    assert [segment["pages"] for segment in segments] == [[1, 2], [3, 4]]
    assert {issue["code"] for issue in issues} == {
        "boundary_classification_conflict",
        "unresolved_boundary_pages",
    }


def test_reconciliation_preserves_order_complete_ownership_and_audited_share() -> None:
    page_parts, segments, _ = _document_first_page_ownership(
        [_proposal(1, 1, 3), _proposal(2, 3, 5)],
        _report(("Annexure P-1", [1, 2, 3]), ("Annexure P-2", [3, 4, 5])),
        page_count=5,
    )

    assert [segment["pages"] for segment in segments] == [[1, 2, 3], [3, 4, 5]]
    assert page_parts[3] == ["Annexure P-1", "Annexure P-2"]
    assert set(page_parts) == set(range(1, 6))


def test_annexure_internal_cover_heading_never_reaches_cover_pdf() -> None:
    """Exercise boundary, repair, audit, and actual generated PDF ranges."""
    texts = {
        1: "SUPREME COURT OF INDIA\nPAPERBOOK\nFOR THE INDEX PLEASE SEE INSIDE",
        2: "IN THE SUPREME COURT OF INDIA\nSPECIAL LEAVE PETITION\nMOST RESPECTFULLY SHOWETH",
        3: "Grounds in the petition",
        4: "PRAYER",
        5: "True copy of a lower-court record, first page",
        6: "Lower-court record continues",
        7: (
            "IN THE SUPREME COURT OF INDIA\nCIVIL APPEAL NO. OF 2026\n"
            "PAPERBOOK\n(FOR THE INDEX, PLEASE SEE INSIDE)\n"
            "ADVOCATE FOR THE APPELLANT"
        ),
        8: "Lower-court record continues after its reproduced cover",
        9: "ANNEXURE P-2\nTrue copy of a different order",
        10: "The different order continues",
        11: "AFFIDAVIT\nDEPONENT",
    }
    proposals = [
        _proposal(1, 1, 1), _proposal(2, 2, 4), _proposal(3, 5, 8),
        _proposal(4, 9, 10), _proposal(5, 11, 11),
    ]
    raw_report = _report(
        ("Cover Page", [1]),
        ("Main Petition", [2, 3, 4]),
        ("Annexure P-1", [5, 6]),
        ("Cover Page", [7]),
        ("Annexure P-1", [8]),
        ("Annexure P-2", [9, 10]),
        ("Affidavit", [11]),
    )
    initial, segments, issues = _document_first_page_ownership(
        proposals, raw_report, page_count=11
    )
    assert initial[7] == ["Annexure P-1"]
    assert any(item["code"] == "boundary_classification_conflict" for item in issues)

    with pymupdf.open() as source:
        for page in range(1, 12):
            source.new_page().insert_text((72, 72), f"SOURCE PAGE {page}")
        pdf_bytes = source.tobytes()
    units = [PageUnit(pdf_page=page, text=texts[page]) for page in range(1, 12)]
    result = structure_aware_split(
        pdf_bytes,
        llama_page_parts=initial,
        page_units=units,
        authoritative_boundary_pages=[2, 5, 9, 11],
    )
    assert result.page_parts[7] == ["Annexure P-1"]
    assert all(result.page_parts[page] == ["Annexure P-1"] for page in range(5, 9))
    assert all(result.page_parts[page] == ["Annexure P-2"] for page in (9, 10))
    assert result.page_parts[11] == ["Affidavit"]

    audit = audit_compiled_split(
        result.page_parts,
        texts,
        page_count=11,
        llama_page_parts=initial,
        classifications=result.classifications,
        reconciliation_changes=result.reconciliation_changes,
        reconciliation_operations=result.reconciliation_operations,
        initial_segments=segments,
        raw_classification_segments=[
            {"segment_id": f"classified-{index}", **item}
            for index, item in enumerate(raw_report["segments"], start=1)
        ],
        assignment_events=result.assignment_events,
        page_units=units,
    )
    page_seven = audit["page_manifest"]["pages"][6]
    assert page_seven["effective_owner"] == "Annexure P-1"
    assert page_seven["assignment_trace"][0]["segment_id"] == "boundary-3"
    assert any(
        event.get("proposed_type") == "Cover Page"
        and event.get("status") == "evidence_only"
        for event in page_seven["assignment_trace"]
    )
    assert any(
        event.get("candidate_type") == "Cover Page"
        and event.get("reason")
        == "heading_conflicts_with_reconciled_annexure_owner"
        for event in page_seven["assignment_trace"]
    )
    assert not any(
        event.get("status") == "committed"
        and event.get("to_labels") == ["Cover Page"]
        for event in page_seven["assignment_trace"]
    )

    catalog = type_catalog("SLP_CIVIL")
    slices = slice_bundle_pdf(
        pdf_bytes, catalog, result.page_parts, split_segments=segments
    )
    outputs = {item.slot_id: item for item in slices}
    attach_split_output_trace(audit["page_manifest"]["pages"], slices)
    assert page_seven["output_pdfs"] == [
        {"slot_id": "annexure_p1", "filename": "Annexure P-1.pdf"}
    ]
    assert page_seven["assignment_trace"][-1]["stage"] == "pdf_slice"
    assert outputs["cover_page"].pages == (1,)
    assert outputs["annexure_p1"].pages == (5, 6, 7, 8)
    assert outputs["annexure_p2"].pages == (9, 10)
    assert outputs["affidavit"].pages == (11,)
    assert [item["start"] for item in outputs["annexure_p1"].page_span] == [5]
    assert outputs["annexure_p1"].page_span[0]["end"] == 8
    with pymupdf.open(stream=outputs["annexure_p1"].pdf_bytes, filetype="pdf") as pdf:
        assert pdf.page_count == 4
        assert "SOURCE PAGE 7" in pdf[2].get_text()
    coverage = validate_sliced_page_ownership(
        slices,
        page_count=11,
        page_parts=result.page_parts,
        split_segments=segments,
        catalog=catalog,
    )
    assert coverage["accounted_page_count"] == 11
    assert coverage["missing_pages"] == []
    assert not coverage["overlapping_ownership"]


def test_verified_new_boundary_after_annexure_can_open_cover_document() -> None:
    """The continuity guard does not erase a genuinely separate candidate."""
    texts = {
        1: "True copy of the first record",
        2: "First record continuation",
        3: (
            "IN THE SUPREME COURT OF INDIA\nCIVIL APPEAL NO. OF 2026\n"
            "PAPERBOOK\n(FOR THE INDEX, PLEASE SEE INSIDE)\n"
            "ADVOCATE FOR THE APPELLANT"
        ),
        4: "ANNEXURE P-2\nTrue copy of a second record",
    }
    proposals = [_proposal(1, 1, 2), _proposal(2, 3, 3), _proposal(3, 4, 4)]
    initial, segments, _ = _document_first_page_ownership(
        proposals,
        _report(
            ("Annexure P-1", [1, 2]),
            ("Cover Page", [3]),
            ("Annexure P-2", [4]),
        ),
        page_count=4,
    )
    with pymupdf.open() as source:
        for page in range(1, 5):
            source.new_page().insert_text((72, 72), f"SOURCE PAGE {page}")
        pdf_bytes = source.tobytes()
    result = structure_aware_split(
        pdf_bytes,
        llama_page_parts=initial,
        page_units=[PageUnit(pdf_page=page, text=texts[page]) for page in range(1, 5)],
        authoritative_boundary_pages=[3, 4],
    )
    assert result.page_parts[1] == result.page_parts[2] == ["Annexure P-1"]
    assert result.page_parts[3] == ["Cover Page"]
    assert result.page_parts[4] == ["Annexure P-2"]
    slices = slice_bundle_pdf(
        pdf_bytes, type_catalog("SLP_CIVIL"), result.page_parts,
        split_segments=segments,
    )
    assert {item.slot_id: item.pages for item in slices}["cover_page"] == (3,)


@pytest.mark.parametrize(
    "heading",
    [
        "HON'BLE HIGH COURT OF INDIA",
        "BEFORE THE HON'BLE HIGH COURT OF DELHI",
        "Before the Hon’ble High Court of Maharashtra",
        "BEFORE THE\nHON BLE HIGH-COURT OF KERALA",
        "THE HON. HIGH COURT OF PUNJAB AND HARYANA",
        "IN THE HIGH COURT FOR THE STATE OF TELANGANA",
    ],
)
def test_honble_high_court_petition_inside_annexure_stays_out_of_main_pdf(
    heading: str,
) -> None:
    texts = {
        1: "IN THE SUPREME COURT OF INDIA\nSPECIAL LEAVE PETITION\nPOSITION OF PARTIES",
        2: "GROUNDS OF THE SPECIAL LEAVE PETITION\n49",
        3: "True copy of the earlier writ petition\n50",
        4: (
            f"{heading}\nWRIT PETITION NO. 15 OF 2020\n"
            "MOST RESPECTFULLY SHOWETH\n51"
        ),
        5: "Grounds of the earlier High Court writ petition\n52",
        6: "ANNEXURE P-2\nTrue copy of the next order\n53",
    }
    proposals = [
        _proposal(1, 1, 2), _proposal(2, 3, 3),
        _proposal(3, 4, 5), _proposal(4, 6, 6),
    ]
    initial, segments, _ = _document_first_page_ownership(
        proposals,
        _report(
            ("Main Petition", [1, 2]),
            ("Annexure P-1", [3]),
            ("Main Petition", [4, 5]),
            ("Annexure P-2", [6]),
        ),
        page_count=6,
    )
    with pymupdf.open() as source:
        for page in range(1, 7):
            source.new_page().insert_text((72, 72), f"SOURCE PAGE {page}")
        pdf_bytes = source.tobytes()
    before_outputs = {
        item.slot_id: item.pages
        for item in slice_bundle_pdf(
            pdf_bytes, type_catalog("SLP_CIVIL"), initial,
            split_segments=segments,
        )
    }
    assert before_outputs["petition"] == (1, 2, 4, 5)
    assert before_outputs["annexure_p1"] == (3,)
    result = structure_aware_split(
        pdf_bytes,
        llama_page_parts=initial,
        page_units=[PageUnit(pdf_page=page, text=text) for page, text in texts.items()],
        authoritative_boundary_pages=[3, 4, 6],
    )
    assert all(result.page_parts[page] == ["Annexure P-1"] for page in (3, 4, 5))
    assert any(
        event.get("stage") == "lower_court_petition_ownership"
        and event.get("to_labels") == ["Annexure P-1"]
        and "preceding_annexure_owner" in event.get("evidence", [])
        for event in result.assignment_events[4]
    )
    slices = slice_bundle_pdf(
        pdf_bytes, type_catalog("SLP_CIVIL"), result.page_parts,
        split_segments=segments,
    )
    outputs = {item.slot_id: item for item in slices}
    assert outputs["petition"].pages == (1, 2)
    assert outputs["annexure_p1"].pages == (3, 4, 5)
    assert outputs["annexure_p2"].pages == (6,)
    validate_sliced_page_ownership(
        slices, page_count=6, page_parts=result.page_parts,
        split_segments=segments, catalog=type_catalog("SLP_CIVIL"),
    )


def test_high_court_petition_without_annexure_evidence_is_not_main_petition() -> None:
    texts = {
        1: "IN THE SUPREME COURT OF INDIA\nSPECIAL LEAVE PETITION\nPOSITION OF PARTIES",
        2: "HON'BLE HIGH COURT OF INDIA\nWRIT PETITION NO. 15 OF 2020",
        3: "Continuation of the earlier writ petition",
    }
    with pymupdf.open() as source:
        for page in range(1, 4):
            source.new_page().insert_text((72, 72), f"SOURCE PAGE {page}")
        pdf_bytes = source.tobytes()
    result = structure_aware_split(
        pdf_bytes,
        llama_page_parts={page: ["Main Petition"] for page in range(1, 4)},
        page_units=[PageUnit(pdf_page=page, text=text) for page, text in texts.items()],
        authoritative_boundary_pages=[2],
    )
    assert result.page_parts[1] == ["Main Petition"]
    assert result.page_parts[2] == result.page_parts[3] == ["Undefined"]
    assert "annexure_continuity_unconfirmed" in result.assignment_events[2][-1]["evidence"]
    slices = slice_bundle_pdf(
        pdf_bytes, type_catalog("SLP_CIVIL"), result.page_parts,
    )
    assert {item.slot_id: item.pages for item in slices} == {
        "petition": (1,), "undefined": (2, 3),
    }
    audit = audit_compiled_split(
        result.page_parts, texts, page_count=3,
        classifications=result.classifications,
        assignment_events=result.assignment_events,
        page_units=result.page_units,
    )
    assert "lower_court_petition_owner_unconfirmed" in (
        audit["page_manifest"]["pages"][1]["review_reasons"]
    )
    assert audit["page_manifest"]["pages"][1]["review_status"] == "needs_review"


def test_honble_high_court_heading_is_source_court_not_body_reference() -> None:
    assert _is_lower_court_caption(
        "HON'BLE HIGH COURT OF INDIA\nWRIT PETITION NO. 15 OF 2020"
    )
    assert not _is_lower_court_caption(
        "IN THE SUPREME COURT OF INDIA\nSPECIAL LEAVE PETITION\n"
        "BEFORE HIGH COURT / BEFORE SUPREME COURT"
    )
    assert not _has_leading_high_court_caption(
        "GROUNDS:\nThat the petitioner approached the Hon'ble High Court of Delhi.\n"
        "Before the Hon'ble High Court of Delhi, a writ petition was filed."
    )


@pytest.mark.parametrize(
    "heading",
    [
        "IN THE COURT OF THE DISTRICT AND SESSIONS JUDGE, DELHI",
        "BEFORE THE HON'BLE DISTRICT COURT OF DELHI",
        "BEFORE THE NATIONAL COMPANY LAW TRIBUNAL",
    ],
)
def test_other_lower_court_headings_are_recognized_without_becoming_sci(
    heading: str,
) -> None:
    assert _is_lower_court_caption(f"{heading}\nPETITION NO. 12 OF 2020")
    assert not _has_leading_high_court_caption(heading)
