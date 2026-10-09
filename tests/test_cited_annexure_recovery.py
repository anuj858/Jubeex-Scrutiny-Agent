"""Narrative ranges need inventory, source and two-sided folio corroboration."""

from __future__ import annotations

import pytest

from extraction_review.split_repair import _apply_cited_annexure_ranges
from extraction_review.structure_split import PageUnit, structure_aware_split


def fixture():
    texts = {
        1: "INDEX\nS.No. Particulars Page No.\n"
        "1.\tANNEXURE P-8 Copy of earlier judgment\t39-40\n"
        "2.\tANNEXURE P-9 Copy of W.A.No.731 of 2014\t\n"
        "3.\tANNEXURE P-10 Copy of later application\t44-45",
        2: "LIST OF DATES & EVENTS\nA true copy is annexed herewith and marked as "
        "ANNEXURE P-9 (Kindly see Pages 41 to 43).\nX",
        3: "ANNEXURE P-8\nEarlier judgment.\n39",
        4: "Earlier judgment ends.\n40",
        5: "WA 731/2014-Affidavit\nBEFORE THE HIGH COURT OF KERALA\n"
        "W.A.No.731 of 2014\nAFFIDAVIT\nI affirm the appeal facts.",
        6: "WA 731/2014-Affidavit\nAffidavit continues.",
        7: "WA 731/2014-Memo of Appearance\nCounsel appears in the same packet.",
        8: "ANNEXURE P-10\nLater exhibit opening.\n44",
        9: "Later exhibit ends.\n45",
    }
    parts = {
        1: ["Index"],
        2: ["List of Dates & Events"],
        **{page: ["Annexure P-8"] for page in range(3, 8)},
        8: ["Annexure P-10"],
        9: ["Annexure P-10"],
    }
    return texts, parts


def test_citation_recovers_only_actual_indexed_packet_not_citing_page():
    texts, parts = fixture()
    result = _apply_cited_annexure_ranges(parts, texts, len(texts))
    assert result == {**parts, **{p: ["Annexure P-9"] for p in range(5, 8)}}


def test_targeted_pipeline_accepts_corroborated_citation_boundary():
    texts, parts = fixture()
    result = structure_aware_split(
        b"",
        llama_page_parts=parts,
        page_units=[
            PageUnit(pdf_page=p, text=t, char_count=len(t)) for p, t in texts.items()
        ],
        reconciliation_mode="targeted",
    )
    assert all(result.page_parts[p] == ["Annexure P-9"] for p in range(5, 8))
    assert result.page_parts[2] == ["List of Dates & Events"]


@pytest.mark.parametrize(
    "change",
    [
        "enclosed_reference",
        "conflicting_references",
        "missing_inventory",
        "wrong_source_case",
        "missing_left_anchor",
        "missing_right_anchor",
        "different_offset",
        "new_stamp",
        "new_sci_document",
        "protected_document",
    ],
)
def test_citation_alone_does_not_reassign_pages(change):
    texts, parts = fixture()
    if change == "enclosed_reference":
        parts[2] = ["Annexure P-1"]
    elif change == "conflicting_references":
        texts[2] += "\nAnnexure P-9 (Kindly see Pages 41 to 44)."
    elif change == "missing_inventory":
        texts[1] = texts[1].replace("ANNEXURE P-9", "ANNEXURE P-19")
    elif change == "wrong_source_case":
        texts[5] = texts[5].replace("731", "732")
    elif change == "missing_left_anchor":
        texts[3] = "Earlier judgment without folio."
    elif change == "missing_right_anchor":
        texts[9] = "Later exhibit without folio."
    elif change == "different_offset":
        texts[8] = texts[8].replace("\n44", "\n45")
        texts[9] = texts[9].replace("\n45", "\n46")
    elif change == "new_stamp":
        texts[6] = "ANNEXURE P-20\nDifferent exhibit."
    elif change == "new_sci_document":
        texts[6] = "IN THE SUPREME COURT OF INDIA\nAPPLICATION FOR EXEMPTION"
    elif change == "protected_document":
        parts[6] = ["Application 2"]
    assert _apply_cited_annexure_ranges(parts, texts, len(texts)) == parts


def test_citation_cannot_override_another_indexed_document():
    texts, parts = fixture()
    texts[1] += "\n4.\tANNEXURE P-11 Other record\t42-43"
    parts[6] = parts[7] = ["Annexure P-11"]
    assert _apply_cited_annexure_ranges(parts, texts, len(texts)) == parts


def test_mislabelled_lower_court_application_is_not_trusted_citation_source():
    texts, parts = fixture()
    texts[2] = "IN THE HIGH COURT OF KERALA\nAPPLICATION FOR LEAVE\n" + texts[2]
    parts[2] = ["Application 1"]
    assert _apply_cited_annexure_ranges(parts, texts, len(texts)) == parts


def test_lower_court_running_header_is_not_trusted_outer_citation_source():
    texts, parts = fixture()
    texts[2] = "WA 731/2014-Petition (Page-10)\n" + texts[2]
    parts[2] = ["Application 1"]
    assert _apply_cited_annexure_ranges(parts, texts, len(texts)) == parts


def test_cited_range_does_not_swallow_a_different_court_case_opening():
    texts, parts = fixture()
    texts[6] = "IN THE HIGH COURT OF KERALA\nW.A.No.999 of 2014\nAFFIDAVIT"
    assert _apply_cited_annexure_ranges(parts, texts, len(texts)) == parts
