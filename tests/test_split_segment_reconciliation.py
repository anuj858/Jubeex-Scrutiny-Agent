"""Remote boundaries survive local label repairs without restoring stale pages."""

from types import SimpleNamespace

import pymupdf
import pytest

from extraction_review.bundle_slicer import (
    SlotSlice,
    _pixmaps_visually_equivalent,
    extract_pdf_pages,
    slice_bundle_pdf,
    validate_page_manifest_output,
    validate_shared_page_evidence,
    validate_sliced_page_ownership,
)
from extraction_review.split_audit import audit_compiled_split
from extraction_review.split_upload import type_catalog


@pytest.fixture
def source_pdf() -> bytes:
    with pymupdf.open() as pdf:
        for number in range(1, 9):
            page = pdf.new_page()
            page.insert_text((72, 72), f"Physical page {number}")
        return pdf.tobytes()


def test_visual_comparison_allows_harmless_raster_rounding():
    source = SimpleNamespace(
        width=2,
        height=1,
        n=3,
        samples=bytes([0, 10, 20, 30, 40, 50]),
    )
    output = SimpleNamespace(
        width=2,
        height=1,
        n=3,
        samples=bytes([4, 6, 24, 26, 44, 46]),
    )

    assert _pixmaps_visually_equivalent(source, output) is True


def test_visual_comparison_rejects_meaningful_or_different_sized_content():
    source = SimpleNamespace(width=1, height=1, n=3, samples=bytes([0, 10, 20]))
    changed = SimpleNamespace(width=1, height=1, n=3, samples=bytes([9, 10, 20]))
    resized = SimpleNamespace(
        width=2,
        height=1,
        n=3,
        samples=bytes([0, 10, 20, 0, 0, 0]),
    )

    assert _pixmaps_visually_equivalent(source, changed) is False
    assert _pixmaps_visually_equivalent(source, resized) is False


def _slice(source_pdf, page_parts, segments):
    slices = slice_bundle_pdf(
        source_pdf,
        type_catalog("SLP_CIVIL"),
        page_parts,
        split_segments=segments,
    )
    # Verify the PDF payload as well as the metadata, including shared pages.
    for item in slices:
        with pymupdf.open(stream=item.pdf_bytes, filetype="pdf") as pdf:
            assert [page.get_text().strip() for page in pdf] == [
                f"Physical page {number}" for number in item.pages
            ]
    assert {page for item in slices for page in item.pages} == set(range(1, 9))
    return {item.slot_id: item.pages for item in slices}


def _ownership(slices, page_parts, segments, *, page_count=8):
    return validate_sliced_page_ownership(
        slices,
        page_count=page_count,
        page_parts=page_parts,
        split_segments=segments,
        catalog=type_catalog("SLP_CIVIL"),
    )


@pytest.mark.parametrize(
    ("family", "prefix"),
    [("Annexures", "annexure"), ("Application", "application")],
)
def test_adjacent_remote_documents_remain_separate_after_repair(
    source_pdf, family, prefix
):
    segments = [
        {"category": family, "pages": [1, 2, 3]},
        {"category": family, "pages": [4, 5, 6]},
    ]
    repaired = {page: [family] for page in range(1, 7)}
    repaired[1] = ["Index"]
    repaired.pop(6)  # A removed duplicate must stay unassigned.

    assert _slice(source_pdf, repaired, segments) == {
        "index": (1,),
        f"{prefix}_1": (2, 3),
        f"{prefix}_2": (4, 5),
        "undefined": (6, 7, 8),
    }


def test_local_numbered_identities_override_remote_generic_assignments(source_pdf):
    segments = [{"category": "Annexures", "pages": list(range(1, 9))}]
    repaired = {
        1: ["Index"],
        2: ["Main Petition"],
        3: ["Main Petition"],
        4: ["Annexure P-9"],
        5: ["Annexure P-9"],
        6: ["Application 2"],
        7: ["Application 2"],
    }

    assert _slice(source_pdf, repaired, segments) == {
        "index": (1,),
        "petition": (2, 3),
        "annexure_p9": (4, 5),
        "application_2": (6, 7),
        "undefined": (8,),
    }


def test_new_generic_pages_are_retained_without_attaching_to_remote_document(
    source_pdf,
):
    segments = [
        {"category": "Annexures", "pages": [2, 3]},
        {"category": "Annexures", "pages": [6]},
        {"category": "Main Petition", "pages": [7, 8]},
    ]
    repaired = {page: ["Annexures"] for page in (1, 2, 3, 4, 6, 7, 8)}
    repaired[5] = ["Index"]

    assert _slice(source_pdf, repaired, segments) == {
        "index": (5,),
        "annexure_1": (2, 3),
        "annexure_2": (6,),
        "annexure_3": (1,),
        "annexure_4": (4,),
        "annexure_5": (7, 8),
    }


def test_repaired_generic_labels_retain_boundaries_from_numbered_remote_segments(
    source_pdf,
):
    segments = [
        {"category": "Annexure P-5", "pages": [1, 2]},
        {"category": "Annexure P-6", "pages": [3, 4]},
    ]
    repaired = {page: ["Annexures"] for page in range(1, 5)}

    assert _slice(source_pdf, repaired, segments) == {
        "annexure_1": (1, 2),
        "annexure_2": (3, 4),
        "undefined": (5, 6, 7, 8),
    }


def test_remote_segments_keep_shared_pages_and_do_not_restore_reclassified_pages(
    source_pdf,
):
    segments = [
        {"category": "Annexures", "pages": [1, 2, 3]},
        {"category": "Annexures", "pages": [3, 4, 5]},
    ]
    repaired = {page: ["Annexures"] for page in range(1, 6)}
    repaired[4] = ["Affidavit"]

    assert _slice(source_pdf, repaired, segments) == {
        "affidavit": (4,),
        "annexure_1": (1, 2, 3),
        "annexure_2": (3, 5),
        "undefined": (6, 7, 8),
    }


def test_output_ownership_manifest_accounts_for_every_source_page(source_pdf):
    slices = slice_bundle_pdf(
        source_pdf,
        type_catalog("SLP_CIVIL"),
        {1: ["Main Petition"], 2: ["Main Petition"]},
    )

    report = _ownership(
        slices,
        {1: ["Main Petition"], 2: ["Main Petition"]},
        [],
    )

    assert report["source_page_count"] == 8
    assert report["accounted_page_count"] == 8
    assert report["missing_pages"] == []
    assert report["overlapping_ownership"] == []
    assert report["review_status"] == "clear"


def test_output_ownership_rejects_shared_source_pages(source_pdf):
    output_slices = slice_bundle_pdf(
        source_pdf,
        type_catalog("SLP_CIVIL"),
        {page: ["Annexures"] for page in range(1, 7)},
        split_segments=[
            {"category": "Annexures", "pages": [1, 2, 3]},
            {"category": "Annexures", "pages": [3, 4, 5]},
        ],
    )

    overlapping = list(output_slices)
    second = overlapping[1]
    overlapping[1] = SlotSlice(
        slot_id=second.slot_id,
        label=second.label,
        pages=(3, *second.pages),
        page_span=({"start": 3, "end": 3}, *second.page_span),
        pdf_bytes=second.pdf_bytes,
        filename=second.filename,
        file_hash=second.file_hash,
    )
    with pytest.raises(ValueError, match="repeats a source page"):
        _ownership(
            overlapping,
            {page: ["Annexures"] for page in range(1, 6)},
            [
                {"category": "Annexures", "pages": [1, 2, 3]},
                {"category": "Annexures", "pages": [2, 3, 4]},
            ],
        )


def test_output_ownership_rejects_unauthorized_cross_slot_overlap(source_pdf):
    page_parts = {page: ["Annexures"] for page in range(1, 5)}
    segments = [
        {"category": "Annexures", "pages": [1, 2]},
        {"category": "Annexures", "pages": [3, 4]},
    ]
    slices = slice_bundle_pdf(
        source_pdf,
        type_catalog("SLP_CIVIL"),
        page_parts,
        split_segments=segments,
    )
    second = slices[1]
    overlapping = [
        slices[0],
        SlotSlice(
            slot_id=second.slot_id,
            label=second.label,
            pages=(2, *second.pages),
            page_span=second.page_span,
            pdf_bytes=second.pdf_bytes,
            filename=second.filename,
            file_hash=second.file_hash,
        ),
        *slices[2:],
    ]

    with pytest.raises(ValueError, match="page 2 in"):
        _ownership(overlapping, page_parts, segments)


def test_multiple_labels_without_classification_evidence_are_not_authorized(source_pdf):
    page_parts = {1: ["Annexures", "Application"]}
    validation = validate_shared_page_evidence(page_parts, [], page_count=8)

    slices = slice_bundle_pdf(
        source_pdf,
        type_catalog("SLP_CIVIL"),
        page_parts,
        split_segments=[],
    )

    assert validation["authorizations"] == {}
    assert validation["issues"][0]["code"] == (
        "multi_label_page_lacks_classification_evidence"
    )
    assert {item.slot_id for item in slices if 1 in item.pages} == {"annexure_1"}


def test_explicit_types_do_not_authorize_page_with_a_third_conflicting_segment():
    validation = validate_shared_page_evidence(
        {1: ["Affidavit", "Vakalatnama"]},
        [
            {"category": "Affidavit", "pages": [1]},
            {"category": "Vakalatnama", "pages": [1]},
            {"category": "Annexures", "pages": [1, 2]},
        ],
        page_count=8,
    )

    assert validation["authorizations"] == {}
    assert "multi_label_page_lacks_classification_evidence" in {
        issue["code"] for issue in validation["issues"]
    }


def test_cross_family_overlap_requires_matching_adjacent_evidence(source_pdf):
    page_parts = {
        1: ["Annexures"],
        2: ["Annexures", "Application"],
        3: ["Annexures", "Application"],
        4: ["Application"],
    }
    segments = [
        {"category": "Annexures", "pages": [1, 2, 3]},
        {"category": "Application", "pages": [2, 3, 4]},
    ]

    validation = validate_shared_page_evidence(page_parts, segments, page_count=8)

    assert validation["authorizations"] == {}
    assert {item["code"] for item in validation["issues"]} >= {
        "multi_label_page_lacks_classification_evidence",
        "unsupported_classification_overlap",
    }

    slices = slice_bundle_pdf(
        source_pdf,
        type_catalog("SLP_CIVIL"),
        page_parts,
        split_segments=segments,
    )
    page_owners = {item.slot_id for item in slices if 2 in item.pages}
    assert page_owners == {"annexure_1"}
    ownership = _ownership(slices, page_parts, segments)
    assert ownership["review_status"] == "needs_review"
    assert any(
        item["code"] == "multi_label_page_lacks_classification_evidence"
        for item in ownership["sharing_validation"]["issues"]
    )


def test_explicit_shared_boundary_page_is_supported_and_review_flagged(source_pdf):
    slices = slice_bundle_pdf(
        source_pdf,
        type_catalog("SLP_CIVIL"),
        {page: ["Annexures"] for page in range(1, 6)},
        split_segments=[
            {"category": "Annexures", "pages": [1, 2, 3]},
            {"category": "Annexures", "pages": [3, 4, 5]},
        ],
    )

    report = _ownership(
        slices,
        {page: ["Annexures"] for page in range(1, 6)},
        [
            {"category": "Annexures", "pages": [1, 2, 3]},
            {"category": "Annexures", "pages": [3, 4, 5]},
        ],
    )

    assert report["ownership_policy"] == "single_owner_with_explicit_shared_copies"
    assert report["overlapping_ownership"] == []
    shared = report["shared_ownership"][0]
    assert shared["source_page"] == 3
    assert shared["output_slots"] == ["annexure_1", "annexure_2"]
    assert shared["reason"] == "adjacent_classified_segment_boundary"
    assert shared["segment_pages"] == [[1, 2, 3], [3, 4, 5]]
    assert shared["review_status"] == "needs_review"
    assert report["review_status"] == "needs_review"


def test_explicit_mixed_document_page_can_be_copied_to_two_slots(source_pdf):
    slices = slice_bundle_pdf(
        source_pdf,
        type_catalog("SLP_CIVIL"),
        {1: ["Affidavit", "Vakalatnama"]},
        split_segments=[
            {"category": "Affidavit", "pages": [1]},
            {"category": "Vakalatnama", "pages": [1]},
        ],
    )

    report = _ownership(
        slices,
        {1: ["Affidavit", "Vakalatnama"]},
        [
            {"category": "Affidavit", "pages": [1]},
            {"category": "Vakalatnama", "pages": [1]},
        ],
    )

    assert {item.slot_id: item.pages for item in slices} == {
        "affidavit": (1,),
        "vakalatnama_appearance": (1,),
        "undefined": (2, 3, 4, 5, 6, 7, 8),
    }
    assert report["shared_ownership"][0]["source_page"] == 1
    assert report["shared_ownership"][0]["reason"] == (
        "explicit_classified_multi_document_page"
    )
    assert report["review_status"] == "needs_review"


def test_output_ownership_rejects_missing_source_pages():
    with pytest.raises(ValueError, match=r"unassigned source pages \[2\]"):
        validate_sliced_page_ownership(
            [
                SlotSlice(
                    slot_id="petition",
                    label="Main Petition",
                    pages=(1,),
                    page_span=({"start": 1, "end": 1},),
                    pdf_bytes=b"%PDF-test",
                    filename="Main Petition.pdf",
                    file_hash="hash",
                )
            ],
            page_count=2,
            page_parts={1: ["Main Petition"]},
            split_segments=[],
            catalog=type_catalog("SLP_CIVIL"),
        )


@pytest.mark.parametrize(
    ("page_parts", "segments", "expected_code"),
    [
        ({9: ["Annexures"]}, [], "page_assignment_out_of_range"),
        (
            {1: ["Annexures"]},
            [{"category": "Annexures", "pages": [1, 1]}],
            "classified_segment_range_invalid",
        ),
        (
            {1: ["Annexures"]},
            [{"category": "Annexures", "pages": [1, 3]}],
            "classified_segment_range_invalid",
        ),
    ],
)
def test_malformed_sharing_assignments_fail_validation(
    page_parts, segments, expected_code
):
    validation = validate_shared_page_evidence(page_parts, segments, page_count=8)

    assert validation["fatal"] is True
    assert expected_code in {item["code"] for item in validation["issues"]}


def test_manifest_output_validator_detects_gaps_and_slot_mismatch(source_pdf):
    slices = slice_bundle_pdf(
        source_pdf,
        type_catalog("SLP_CIVIL"),
        {1: ["Main Petition"]},
    )
    rows = [
        {"source_page": page, "output_slots": ["wrong"] if page == 1 else ["undefined"]}
        for page in range(1, 8)
    ]

    report = validate_page_manifest_output(
        rows, slices, page_count=8, source_pdf=source_pdf
    )

    codes = {item["code"] for item in report["errors"]}
    assert report["valid"] is False
    assert "manifest_pages_missing" in codes
    assert "manifest_output_slots_mismatch" in codes


def test_manifest_output_validator_detects_duplicate_rows(source_pdf):
    slices = slice_bundle_pdf(source_pdf, type_catalog("SLP_CIVIL"), {})
    rows = [
        {"source_page": page, "output_slots": ["undefined"]}
        for page in range(1, 9)
    ]
    rows.append({"source_page": 1, "output_slots": []})

    report = validate_page_manifest_output(
        rows, slices, page_count=8, source_pdf=source_pdf
    )

    assert {item["code"] for item in report["errors"]} >= {
        "manifest_page_duplicated",
        "manifest_output_slots_mismatch",
    }


def test_manifest_output_validator_detects_wrong_source_page_content(source_pdf):
    slices = slice_bundle_pdf(
        source_pdf,
        type_catalog("SLP_CIVIL"),
        {1: ["Main Petition"]},
    )
    petition = next(item for item in slices if item.slot_id == "petition")
    wrong_content = extract_pdf_pages(source_pdf, [2])
    corrupted = [
        SlotSlice(
            slot_id=item.slot_id,
            label=item.label,
            pages=item.pages,
            page_span=item.page_span,
            pdf_bytes=wrong_content if item.slot_id == petition.slot_id else item.pdf_bytes,
            filename=item.filename,
            file_hash=item.file_hash,
        )
        for item in slices
    ]
    rows = [
        {
            "source_page": page,
            "output_slots": ["petition"] if page == 1 else ["undefined"],
        }
        for page in range(1, 9)
    ]

    report = validate_page_manifest_output(
        rows, corrupted, page_count=8, source_pdf=source_pdf
    )

    assert any(
        error["code"] == "slice_source_content_mismatch"
        and error["source_page"] == 1
        for error in report["errors"]
    )
    assert any(
        error["code"] == "slice_source_visual_mismatch"
        and error["source_page"] == 1
        for error in report["errors"]
    )


def test_manifest_output_validator_checks_scanned_or_textless_visual_content():
    with pymupdf.open() as pdf:
        first = pdf.new_page()
        first.draw_rect(pymupdf.Rect(40, 40, 120, 120), color=(1, 0, 0), fill=(1, 0, 0))
        second = pdf.new_page()
        second.draw_rect(pymupdf.Rect(150, 150, 230, 230), color=(0, 0, 1), fill=(0, 0, 1))
        source_pdf = pdf.tobytes()
    slices = slice_bundle_pdf(
        source_pdf,
        type_catalog("SLP_CIVIL"),
        {1: ["Main Petition"]},
    )
    petition = next(item for item in slices if item.slot_id == "petition")
    wrong_content = extract_pdf_pages(source_pdf, [2])
    corrupted = [
        SlotSlice(
            slot_id=item.slot_id,
            label=item.label,
            pages=item.pages,
            page_span=item.page_span,
            pdf_bytes=wrong_content if item.slot_id == petition.slot_id else item.pdf_bytes,
            filename=item.filename,
            file_hash=item.file_hash,
        )
        for item in slices
    ]
    rows = [
        {
            "source_page": page,
            "output_slots": ["petition"] if page == 1 else ["undefined"],
        }
        for page in range(1, 3)
    ]

    report = validate_page_manifest_output(
        rows, corrupted, page_count=2, source_pdf=source_pdf
    )

    assert "slice_source_content_mismatch" not in {
        error["code"] for error in report["errors"]
    }
    assert any(
        error["code"] == "slice_source_visual_mismatch"
        for error in report["errors"]
    )


def test_audit_manifest_routes_unidentified_pages_to_output_for_review(source_pdf):
    audit = audit_compiled_split(
        {1: ["Main Petition"]},
        {1: "Petition opening"},
        page_count=8,
    )
    rows = audit["page_manifest"]["pages"]
    slices = slice_bundle_pdf(
        source_pdf,
        type_catalog("SLP_CIVIL"),
        {1: ["Main Petition"]},
    )
    slots_by_page = {
        page: sorted(item.slot_id for item in slices if page in item.pages)
        for page in range(1, 9)
    }
    for row in rows:
        row["output_slots"] = slots_by_page[row["source_page"]]

    consistency = validate_page_manifest_output(
        rows, slices, page_count=8, source_pdf=source_pdf
    )

    assert consistency["valid"] is True
    assert rows[1]["effective_owner"] == "Unidentified"
    assert rows[1]["review_status"] == "needs_review"
    assert rows[1]["output_slots"] == ["undefined"]


def test_generic_application_ids_do_not_collide_with_local_numbered_identities(
    source_pdf,
):
    segments = [
        {"category": "Application", "pages": [1, 2, 3, 4]},
        {"category": "Application", "pages": [5, 6]},
    ]
    repaired = {
        1: ["Application 1"],
        2: ["Application 1"],
        3: ["Application"],
        4: ["Application"],
        5: ["Application"],
        6: ["Application"],
        7: ["Application 3"],
    }

    assert _slice(source_pdf, repaired, segments) == {
        "application_1": (1, 2),
        "application_2": (3, 4),
        "application_3": (7,),
        "application_4": (5, 6),
        "undefined": (8,),
    }


def test_empty_segments_keep_all_repaired_generic_runs_without_exhibit_numbers(
    source_pdf,
):
    repaired = {
        1: ["Annexures"],
        2: ["Annexures"],
        3: ["Index"],
        4: ["Annexures"],
        5: ["Application"],
        6: ["Application"],
    }

    assert _slice(source_pdf, repaired, []) == {
        "index": (3,),
        "annexure_1": (1, 2),
        "annexure_2": (4,),
        "application_1": (5, 6),
        "undefined": (7, 8),
    }


def test_combined_catalog_slots_still_combine_their_documents(source_pdf):
    segments = [
        {"category": "Synopsis", "pages": [1, 2]},
        {"category": "List of Dates & Events", "pages": [3, 4]},
        {"category": "Memo of Appearance", "pages": [5]},
        {"category": "Vakalatnama", "pages": [6]},
        {"category": "Annexures", "pages": [7]},
        {"category": "Annexures", "pages": [8]},
    ]
    repaired = {
        page: [segment["category"]] for segment in segments for page in segment["pages"]
    }

    assert _slice(source_pdf, repaired, segments) == {
        "synopsis_lod": (1, 2, 3, 4),
        "vakalatnama_appearance": (5, 6),
        "annexure_1": (7,),
        "annexure_2": (8,),
    }
