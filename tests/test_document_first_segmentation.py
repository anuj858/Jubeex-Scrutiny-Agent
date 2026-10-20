"""Regression tests for boundary-first compiled-PDF page ownership."""

import pymupdf
import pytest

from extraction_review.bundle_slicer import (
    slice_bundle_pdf,
    validate_sliced_page_ownership,
)
from extraction_review.document_parts import annexure_ref_in_heading
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
from extraction_review.structure_split import (
    PageUnit,
    classify_page,
    structure_aware_split,
)


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


def test_front_index_continuation_and_lod_reference_override_wrong_model_types() -> (
    None
):
    """Regression for SCR-2026-000072 pages 3 and 9-13."""
    texts = {
        1: "IN THE SUPREME COURT OF INDIA\nPAPER BOOK\nFOR INDEX SEE INSIDE",
        2: (
            "INDEX\nS.NO. PARTICULARS OF DOCUMENTS PAGE NO.\n"
            "1. OFFICE REPORT ON LIMITATION A\n"
            "2. SYNOPSIS AND LIST OF DATES B-Q"
        ),
        3: (
            "11. APPENDIX\nA. Section 2(16), Maharashtra Cooperative "
            "Societies Act 23A\n12. ANNEXURE-P/1\nTrue Typed Copy of "
            "Circular dated 01.01.1993 24-54\n13. ANNEXURE-P/2\n"
            "True Translated Copy of Order 55-62"
        ),
        4: "IN THE HIGH COURT OF BOMBAY\nJUDGMENT\nORDER",
        5: "OFFICE REPORT ON LIMITATION",
        6: "SYNOPSIS\nThe dispute concerns a cooperative housing society.",
        7: "Synopsis narrative continuation.",
        8: "Synopsis narrative conclusion.",
        9: (
            "LIST OF DATES\nS.NO. PARTICULARS\n01.01.2031 Event description\n"
            "Additional wrapped chronology text\nMore wrapped chronology text\n"
            "True Copy of the Circular is annexed with the petition\n"
            "ANNEXURE-P/1\n(Page 24-54)"
        ),
        10: "09.06.2019 Registration certificate was granted.",
        11: "31.01.2030 The appeal was filed.",
        12: "01.02.2030 The authority passed an order.",
        13: "03.02.2030 The petitioner received the order.",
    }
    proposals = [
        _proposal(1, 1, 1),
        _proposal(2, 2, 2),
        _proposal(3, 3, 3),
        _proposal(4, 4, 4),
        _proposal(5, 5, 5),
        _proposal(6, 6, 8),
        _proposal(7, 9, 13),
    ]
    report = _report(
        ("Cover Page", [1]),
        ("Index", [2]),
        ("Cover Page", [3]),
        ("Impugned Order", [4]),
        ("Office Report on Limitation", [5]),
        ("Synopsis", [6, 7, 8]),
        ("Annexure P-1", [9, 10, 11, 12, 13]),
    )

    assert annexure_ref_in_heading(texts[3]) is None
    assert classify_page(PageUnit(pdf_page=3, text=texts[3])).document_type == "Index"

    page_parts, segments, issues = _document_first_page_ownership(
        proposals,
        report,
        page_count=13,
        page_texts=texts,
    )

    assert page_parts[2] == page_parts[3] == ["Index"]
    assert all(page_parts[page] == ["List of Dates & Events"] for page in range(9, 14))
    assert [segment["category"] for segment in segments[1:3]] == ["Index", "Index"]
    assert segments[-1]["category"] == "List of Dates & Events"
    assert any(
        item["code"] == "boundary_classification_conflict"
        and item["segment_id"] == "boundary-3"
        for item in issues
    )


def test_lod_annexure_citation_is_not_a_stamp_but_leading_outer_stamp_is() -> None:
    citation = (
        "LIST OF DATES\nS.NO. PARTICULARS\n01.01.2031 Event description\n"
        "Wrapped event text\nMore chronology text\nTrue Copy is annexed below\n"
        "ANNEXURE-P/1\n(Page 24-54)"
    )
    assert annexure_ref_in_heading(citation) is None
    classification = classify_page(PageUnit(pdf_page=9, text=citation))
    assert classification.document_type == "List of Dates & Events"
    assert "outer_anchor" in classification.signals

    enclosed_lod = "ANNEXURE-P/1\nLIST OF DATES\n01.01.2020 Earlier event"
    assert annexure_ref_in_heading(enclosed_lod).label == "Annexure P-1"


def test_bare_p22_folio_in_testimony_does_not_create_annexure_pdf() -> None:
    """A scanned continuation folio must not override reconciled boundaries."""
    texts = {
        1: "ANNEXURE A-20\nDeposition of witness",
        2: "P22\n_ are any criminal cases against the deceased I do not remember",
        3: "The witness testimony continues on this page",
        4: "ANNEXURE A-21\nJudgment of the Sessions Court",
        5: "Judgment continues",
    }
    assert annexure_ref_in_heading(texts[2]) is None
    assert annexure_ref_in_heading("P22\nTRUE COPY OF THE ORDER").label == "Annexure P-22"
    with pymupdf.open() as source:
        for page in range(1, 6):
            source.new_page().insert_text((72, 72), f"SOURCE PAGE {page}")
        pdf_bytes = source.tobytes()
    result = structure_aware_split(
        pdf_bytes,
        llama_page_parts={
            1: ["Annexure A-20"],
            2: ["Annexure P-22"],
            3: ["Annexure P-22"],
            4: ["Annexure A-21"],
            5: ["Annexure A-21"],
        },
        page_units=[PageUnit(pdf_page=page, text=texts[page]) for page in range(1, 6)],
        authoritative_boundary_pages=[1, 4],
    )
    assert {page: result.page_parts[page] for page in range(1, 6)} == {
        1: ["Annexure A-20"],
        2: ["Annexure A-20"],
        3: ["Annexure A-20"],
        4: ["Annexure A-21"],
        5: ["Annexure A-21"],
    }
    slices = slice_bundle_pdf(pdf_bytes, type_catalog("CRIMINAL_APPEAL"), result.page_parts)
    outputs = {item.slot_id: item for item in slices}
    assert outputs["annexure_a20"].pages == (1, 2, 3)
    assert outputs["annexure_a21"].pages == (4, 5)
    assert "annexure_p22" not in outputs


def test_cover_declaration_rop_form_and_master_index_split_into_distinct_pdfs() -> None:
    """A missed Llama boundary must not make all front matter one Cover/Index."""
    from extraction_review.split_repair import _looks_like_rop_index_form

    texts = {
        1: (
            "IN THE SUPREME COURT OF INDIA\nCRIMINAL APPELLATE JURISDICTION\n"
            "CRIMINAL APPEAL NO. 133 OF 2023\nPAPER BOOK\n"
            "FOR THE INDEX PLEASE SEE INSIDE"
        ),
        2: (
            "DY. NO. 3125/2023\nDECLARATION\n"
            "Whetever has been added/deleted /modified\n"
            "in the petition is the result of curing of defects and nothing else. "
            "Except curing\n.\nthe defects, nothing has been done. "
            "Paperbooks are complete in all respects\n"
            "Signature (S.R.SETIA) 714\nAdvocate on Record for the petitioner"
        ),
        3: "INDEX —\nRECORD OF PROCEEDINGS\nS.NO.  PARTICULARS  PAGES\n1.\n2.",
        4: (
            "INDEX OF PAPERS\nSl. No.\tParticulars of Documents\tPage No.\n"
            "1.\tOffice Report on Limitation\tA\n"
            "2.\tListing Proforma\tA1-A3\n"
            "3.\tIndex of Record of Proceedings\tA5"
        ),
        5: "OFFICE REPORT ON LIMITATION\nFiling details",
    }
    assert _looks_like_rop_index_form(texts[3])
    assert not _looks_like_rop_index_form(texts[4])
    with pymupdf.open() as source:
        for page in range(1, 6):
            source.new_page().insert_text((72, 72), f"SOURCE PAGE {page}")
        pdf_bytes = source.tobytes()
    result = structure_aware_split(
        pdf_bytes,
        llama_page_parts={
            **{page: ["Cover Page"] for page in range(1, 5)},
            5: ["Office Report on Limitation"],
        },
        page_units=[PageUnit(pdf_page=page, text=texts[page]) for page in range(1, 6)],
        authoritative_boundary_pages=[5],
    )
    assert [result.page_parts[page] for page in range(1, 6)] == [
        ["Cover Page"],
        ["AOR's Declaration"],
        ["Record of Proceedings"],
        ["Index"],
        ["Office Report on Limitation"],
    ]
    outputs = {
        item.slot_id: item
        for item in slice_bundle_pdf(
            pdf_bytes, type_catalog("CRIMINAL_APPEAL"), result.page_parts
        )
    }
    assert outputs["cover_page"].pages == (1,)
    assert outputs["aor_s_declaration"].pages == (2,)
    assert outputs["record_of_proceedings"].pages == (3,)
    assert outputs["index"].pages == (4,)


def test_restored_index_rechecks_folio_ranges_after_first_reconciliation_pass() -> None:
    """A Cover-mislabeled Index must still correct later annexure ownership."""
    texts = {
        4: (
            "INDEX OF PAPERS\n"
            "11.\tAppendix\t43-44\n"
            "12.\tAnnexure A-1 FIR\t45-50\n"
            "29.\tAnnexure A-18 Deposition\t125-134\n"
            "32.\tAnnexure A-21 Judgment\t145-200\n"
            "33.\tAnnexure A-22 Appeal\t201-210\n"
            "34.\tApplication for exemption\t211-213"
        ),
        51: "APPENDIX\nSections of the Penal Code",
        137: "Witness testimony",
        140: "P22\n_ are any criminal cases against the deceased I do not remember",
        147: "Deposition continuation\n139",
        219: (
            "IN THE SUPREME COURT OF INDIA\nI.A. NO. OF 2023\n"
            "APPLICATION FOR EXEMPTION"
        ),
    }
    llama = {
        4: ["Cover Page"],
        51: ["Appendix"],
        133: ["Annexures"],
        **{page: ["Record of Proceedings"] for page in range(137, 143)},
        153: ["Annexures"],
        209: ["Main Petition"],
        219: ["Application 1"],
    }
    with pymupdf.open() as source:
        for _ in range(228):
            source.new_page()
        pdf_bytes = source.tobytes()
    result = structure_aware_split(
        pdf_bytes,
        llama_page_parts=llama,
        page_units=[
            PageUnit(pdf_page=page, text=texts.get(page, ""))
            for page in range(1, 229)
        ],
        authoritative_boundary_pages=[4, 51, 53, 133, 137, 153, 209, 219],
    )
    assert result.page_parts[4] == ["Index"]
    assert all(result.page_parts[page] == ["Annexure A-18"] for page in range(133, 143))
    assert result.page_parts[153] == ["Annexure A-21"]
    assert result.page_parts[209] == ["Annexure A-22"]
    outputs = {
        item.slot_id: item
        for item in slice_bundle_pdf(
            pdf_bytes, type_catalog("CRIMINAL_APPEAL"), result.page_parts
        )
    }
    assert outputs["annexure_a18"].pages == tuple(range(133, 143))
    assert "annexure_p22" not in outputs


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


def test_letter_spaced_paperbook_cover_is_not_classified_as_application() -> None:
    """A pending I.A. reference on the paper-book cover is not an application."""
    text = (
        "IN THE SUPREME COURT OF INDIA\n"
        "CRIMINAL APPELLATE JURISDICTION\n"
        "SPECIAL LEAVE PETITION (CRL.) NO. OF 2026\n"
        "WITH PRAYER FOR INTERIM RELIEF\n"
        "IN THE MATTER OF:\nManish ...Petitioner\nVersus\n"
        "State of Rajasthan ...Respondent\n"
        "IA no. /2026 Application for exemption from filing official translation\n"
        "P A P E R B O O K\n"
        "(FOR DETAILED INDEX KINDLY SEE INSIDE)\n"
        "ADVOCATE FOR THE PETITIONER"
    )
    proposals = [_proposal(1, 3, 3)]
    initial, segments, _ = _document_first_page_ownership(
        proposals,
        _report(("Application 1", [3])),
        page_count=3,
    )
    with pymupdf.open() as source:
        for page in range(1, 4):
            source.new_page().insert_text((72, 72), f"SOURCE PAGE {page}")
        pdf_bytes = source.tobytes()
    result = structure_aware_split(
        pdf_bytes,
        llama_page_parts=initial,
        page_units=[
            PageUnit(pdf_page=1, text="AOR declaration"),
            PageUnit(pdf_page=2, text="AOR declaration continuation"),
            PageUnit(pdf_page=3, text=text),
        ],
        authoritative_boundary_pages=[3],
    )
    assert result.page_parts[3] == ["Cover Page"]
    slices = slice_bundle_pdf(
        pdf_bytes,
        type_catalog("SLP_CIVIL"),
        result.page_parts,
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
    assert (
        before_outputs.get("petition") == (1, 2, 4, 5)
        or (
            before_outputs.get("petition_1") == (1, 2)
            and before_outputs.get("petition_2") == (4, 5)
        )
    )
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
