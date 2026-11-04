"""Incomplete/stale Index cells must not manufacture document ownership."""

import pytest

from extraction_review.split_audit import aligned_index_printed_rows
from extraction_review.split_repair import _apply_indexed_annexure_ranges


def test_multiple_blank_index_cells_do_not_inherit_one_combined_numeric_gap():
    text = (
        "INDEX\nS.No. Particulars Page No.\n"
        "12.\tANNEXURE P-1 Government letter\t26\n"
        "13.\tANNEXURE P-2 Revocation order\t\n"
        "14.\tANNEXURE P-3 Government communication\t\n"
        "15.\tANNEXURE P-4 Communication\t31\n"
        "16.\tANNEXURE P-5 Notice\t\n"
        "17.\tANNEXURE P-6 Reply\t\n"
        "18.\tANNEXURE P-7 Demolition order\t55-67\n"
        "19.\tANNEXURE P-8 Common judgment\t\n"
        "20.\tANNEXURE P-9 Appeal\t\n"
        "21.\tANNEXURE P-10 Historical application\t200-206\n"
        "22.\tANNEXURE P-11 Order\t207\n"
        "23.\tApplication for certified copy exemption\t208-209\n"
        "24.\tApplication for translation exemption\t210-211\n"
        "25.\tApplication for additional documents\t\n"
        "26.\tANNEXURE P-12 Earlier writ petition\t\n"
        "27.\tFiling Memo\t233"
    )
    rows = aligned_index_printed_rows({1: ["Index"]}, {1: text})
    labels = {row.mapped_part for row in rows}
    assert not labels & {
        "Annexure P-2",
        "Annexure P-3",
        "Annexure P-5",
        "Annexure P-6",
        "Annexure P-8",
        "Annexure P-9",
        "Annexure P-12",
    }
    assert labels >= {"Annexure P-1", "Annexure P-4", "Annexure P-10"}


@pytest.mark.parametrize(
    "extra_row",
    [
        "3.\tApplication for additional documents\t\n",
        "3.\tUnclassified supporting document\t\n",
    ],
)
def test_single_missing_annexure_does_not_steal_another_unresolved_inventory_row(
    extra_row,
):
    text = (
        "INDEX\nS.No. Particulars Page No.\n"
        "1.\tANNEXURE P-1 Notice\t10-11\n"
        "2.\tANNEXURE P-2 Record\t\n" + extra_row + "4.\tANNEXURE P-3 Order\t20-21"
    )
    rows = aligned_index_printed_rows({1: ["Index"]}, {1: text})
    assert "Annexure P-2" not in {row.mapped_part for row in rows}


def test_exactly_one_unresolved_annexure_cell_can_still_be_recovered():
    text = (
        "INDEX\nS.No. Particulars Page No.\n"
        "1.\tANNEXURE P-1 Notice\t10-11\n"
        "2.\tANNEXURE P-2 Record\t\n"
        "3.\tANNEXURE P-3 Order\t20-21"
    )
    rows = aligned_index_printed_rows({1: ["Index"]}, {1: text})
    recovered = next(row for row in rows if row.mapped_part == "Annexure P-2")
    assert (recovered.start, recovered.end) == (12, 19)


def _judgment_packet():
    texts = {
        1: "INDEX\nS.No. Particulars Page No.\n"
        "1.\tANNEXURE P-6 Reply\t\n"
        "2.\tANNEXURE P-7 Demolition order\t55-67\n"
        "3.\tANNEXURE P-8 Common judgment\t",
        34: "ANNEXURE P-6\nReply to notice\n34",
        35: "Reply continues\n35",
        36: "Reply ends\n36",
        37: "ANNEXURE P-7\nDemolition order\n37",
        38: "Demolition order continues\n38",
        39: "Demolition order continues\n39",
        40: "Demolition order ends\n40",
        41: "ANNEXURE P-8\nIN THE HIGH COURT\nCOMMON JUDGMENT\n41",
    }
    texts.update(
        {
            page: f"WP(C) No.1234 of 2012 and connected cases\n"
            f"The common judgment continues.\n{page}"
            for page in range(42, 189)
        }
    )
    parts = {1: ["Index"]}
    parts.update({page: ["Annexure P-6"] for page in range(34, 37)})
    parts.update({page: ["Annexure P-7"] for page in range(37, 41)})
    parts.update({page: ["Annexure P-8"] for page in range(41, 189)})
    return texts, parts


def test_stale_index_range_cannot_cut_into_next_stamped_judgment():
    texts, parts = _judgment_packet()
    assert _apply_indexed_annexure_ranges(parts, texts, 188) == parts


def test_matching_stamp_at_index_start_keeps_index_range_authoritative():
    texts, parts = _judgment_packet()
    texts[55] = "ANNEXURE P-7\nActual exhibit opening\n55"
    result = _apply_indexed_annexure_ranges(parts, texts, 188)
    assert all(result[page] == ["Annexure P-7"] for page in range(55, 68))


def test_missing_stamp_ocr_does_not_invalidate_index_range():
    texts, parts = _judgment_packet()
    for page in (34, 37, 41):
        texts[page] = f"Unreadable document heading\n{page}"
    result = _apply_indexed_annexure_ranges(parts, texts, 188)
    assert all(result[page] == ["Annexure P-7"] for page in range(55, 68))
