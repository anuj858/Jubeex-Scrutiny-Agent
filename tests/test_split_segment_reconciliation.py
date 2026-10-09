"""Remote boundaries survive local label repairs without restoring stale pages."""

import pymupdf
import pytest

from extraction_review.bundle_slicer import slice_bundle_pdf
from extraction_review.split_upload import type_catalog


@pytest.fixture
def source_pdf() -> bytes:
    with pymupdf.open() as pdf:
        for number in range(1, 9):
            page = pdf.new_page()
            page.insert_text((72, 72), f"Physical page {number}")
        return pdf.tobytes()


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
        {"category": "Annexure P-5", "pages": [1, 3]},
        {"category": "Annexure P-6", "pages": [2, 4]},
    ]
    repaired = {page: ["Annexures"] for page in range(1, 5)}

    assert _slice(source_pdf, repaired, segments) == {
        "annexure_1": (1, 3),
        "annexure_2": (2, 4),
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
