"""Index-backed ownership must survive inner folios and repeated document types."""

from __future__ import annotations

import pymupdf

from extraction_review.bundle_slicer import slice_bundle_pdf
from extraction_review.split_audit import aligned_index_printed_rows
from extraction_review.split_repair import (
    _apply_indexed_annexure_ranges,
    _apply_indexed_outer_document_ranges,
)
from extraction_review.split_upload import type_catalog
from extraction_review.structure_split import PageUnit, structure_aware_split


def _filing() -> tuple[dict[int, str], dict[int, list[str]]]:
    # Early folios have +2 physical offset. Later printed 10-48 have +4;
    # two inserted sheets make a single offset for the entire bundle invalid.
    index = (
        "MASTER INDEX\nParticulars of Documents Page No.\n"
        "1.\tImpugned High Court orders\t1-4\n"
        "3.\tAnnexure P-3: earlier exhibit\t6-9\n"
        "8.\tAnnexure P-8: reproduced record and receipts\t10-29\n"
        "9.\tApplication for certified copies\t30-31\n"
        "10.\tApplication for translations\t32-33\n"
        "11.\tApplication for additional documents\t34-36\n"
        "12.\nANNEXURE P-9;\n"
        "True copies of High Court orders\n"
        "13. Filing Memo 43\n14. Vakalatnama 44-45\n"
        "15. Memo of Parties 46-48\n"
    )
    texts = {
        1: index,
        2: "LIST OF DATES & EVENTS\nChronology ends here",
        3: "$~36\n= IN THE HIGH COURT OF DELHI AT NEW DELHI\n"
        "O.M.P. (COMM.) NO. 297 OF 2026",
        4: "The Court orders that notice be issued.",
        5: "$~29\n= IN THE HIGH COURT OF DELHI AT NEW DELHI\n"
        "O.M.P. (COMM.) NO. 297 OF 2026",
        6: "Directions of the Court continue.",
        7: "IN THE SUPREME COURT OF INDIA\nSPECIAL LEAVE PETITION\n"
        "MOST RESPECTFULLY SHOWETH",
        8: "ANNEXURE P-3\nEarlier exhibit\n6",
        11: "Earlier exhibit continued\n9",
        14: "ANNEXURE P-8\nReproduced record\n10",
        15: "Reproduced record continued\n11",
        # The inner receipt has a local 7, which is inside P-3's printed
        # range but must not steal a page from the enclosing P-8.
        20: "RECEIPT VOUCHER\nSAC code 9963\n7",
        33: "Reproduced record continued\n29",
        34: "IN THE SUPREME COURT OF INDIA\nAPPLICATION FOR CERTIFIED COPIES\n30",
        35: "Continuation of the same application\n31",
        36: "IN THE SUPREME COURT OF INDIA\nAPPLICATION FOR TRANSLATIONS\n32",
        37: "Continuation of the same application\n33",
        38: "IN THE SUPREME COURT OF INDIA\nAPPLICATION FOR ADDITIONAL DOCUMENTS\n34",
        40: "PRAYER\nAdvocate-on-Record\n36",
        41: "IN THE HIGH COURT OF DELHI\nORDER\nFirst enclosed order",
        46: "IN THE HIGH COURT OF DELHI\nORDER\nLast enclosed order",
        47: (
            "43\nIN THE SUPREME COURT OF INDIA\nCIVIL APPELLATE JURISDICTION\n"
            "SPECIAL LEAVE PETITION NO. OF 2026\nIN THE MATTER OF:\n"
            "First petitioner\nSecond petitioner\nVERSUS\nFirst respondent\n"
            "Second respondent\nINDEX\n"
            "S.No. Particulars No. of Copies Court Fee\n"
            "1. Copy of Impugned Orders\n2. SLP with Affidavit\n"
            "3. Annexures P-1 to P-9\n4. IA for exemption\n"
            "5. Memo of Parties\nFiled by Advocate\n43"
        ),
        48: "VAKALATNAMA\n44",
        49: "MEMO OF APPEARANCE\n45",
        50: "MEMO OF PARTIES\n46",
        52: "Respondent No. 10\n48",
    }
    wrong = {
        1: ["Index"],
        2: ["List of Dates & Events"],
        3: ["Application 1"],
        4: ["Impugned Order"],
        5: ["Impugned Order"],
        6: ["Impugned Order"],
        7: ["Main Petition"],
    }
    wrong.update({page: ["Annexure P-3"] for page in range(8, 12)})
    wrong.update({page: ["Annexure P-9963"] for page in range(14, 34)})
    wrong.update({page: ["Application 1"] for page in range(34, 41)})
    wrong[35] = ["Application 2"]
    wrong[37] = ["Application 3"]
    wrong.update({page: ["Record of Proceedings"] for page in range(41, 47)})
    wrong[47] = ["Index"]
    wrong[48] = ["Vakalatnama"]
    wrong[49] = ["Index"]
    wrong.update({page: ["Memo of Parties"] for page in range(50, 53)})
    return texts, wrong


def test_page_break_index_row_and_local_folio_do_not_steal_annexure() -> None:
    texts, wrong = _filing()
    rows = aligned_index_printed_rows(wrong, texts)
    assert ("Annexure P-9", 37, 42) in {
        (row.mapped_part, row.start, row.end) for row in rows
    }

    repaired = _apply_indexed_annexure_ranges(wrong, texts, 52)
    assert all(repaired[page] == ["Annexure P-8"] for page in range(14, 34))
    assert all(repaired[page] == ["Annexure P-9"] for page in range(41, 47))
    assert [repaired[page] for page in range(34, 41)] == [
        ["Application 1"],
        ["Application 1"],
        ["Application 2"],
        ["Application 2"],
        ["Application 3"],
        ["Application 3"],
        ["Application 3"],
    ]
    assert repaired[47] == ["Filing Memo"]
    assert repaired[49] == ["Vakalatnama"]


def test_index_bracket_restores_untitled_order_opening() -> None:
    texts = {
        1: "MASTER INDEX\n1.\tImpugned High Court orders\t1-4",
        5: "LIST OF DATES & EVENTS\nLast chronology page",
        6: "$~36\n= IN THE HIGH COURT OF DELHI AT NEW DELHI\n"
        "O.M.P. (COMM.) NO. 297 OF 2026\nPetitioner versus Respondent",
        7: "The Court directs that notice be issued.",
        8: "$~29\n= IN THE HIGH COURT OF DELHI AT NEW DELHI\n"
        "O.M.P. (COMM.) NO. 297 OF 2026",
        9: "The directions of the Court continue.",
        10: "IN THE SUPREME COURT OF INDIA\nSPECIAL LEAVE PETITION\n"
        "MOST RESPECTFULLY SHOWETH",
    }
    wrong = {
        1: ["Index"],
        5: ["List of Dates & Events"],
        6: ["Application 1"],
        7: ["Impugned Order"],
        8: ["Impugned Order"],
        9: ["Impugned Order"],
        10: ["Main Petition"],
    }
    repaired = _apply_indexed_outer_document_ranges(wrong, texts, 10)
    assert [repaired[page] for page in range(6, 10)] == [["Impugned Order"]] * 4


def test_final_slices_follow_reconciled_index_ranges() -> None:
    texts, wrong = _filing()
    with pymupdf.open() as source:
        for _ in range(52):
            source.new_page()
        pdf = source.tobytes()

    result = structure_aware_split(
        pdf,
        page_units=[
            PageUnit(pdf_page=page, text=texts.get(page, "")) for page in range(1, 53)
        ],
        llama_page_parts=wrong,
    )
    outputs = {
        item.slot_id: item
        for item in slice_bundle_pdf(pdf, type_catalog("SLP_CIVIL"), result.page_parts)
    }
    assert outputs["annexure_p8"].pages == tuple(range(14, 34))
    assert outputs["annexure_p9"].pages == tuple(range(41, 47))
    assert outputs["impugned_order"].pages == (3, 4, 5, 6)
    assert outputs["application_1"].pages == (34, 35)
    assert outputs["application_2"].pages == (36, 37)
    assert outputs["application_3"].pages == (38, 39, 40)
    assert outputs["filing_memo"].pages == (47,)
    assert 49 not in outputs["index"].pages
    assert "annexure_p9963" not in outputs
