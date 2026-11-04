import io
import subprocess
from unittest.mock import patch

from pypdf import PdfWriter

from extraction_review.split_ocr import (
    index_table_rows_from_tsv,
    margin_folio_from_tsv,
    ocr_sparse_pages,
    pages_with_large_images,
    split_ocr_concurrency,
)
from extraction_review.structure_split import extract_page_units, structure_aware_split


def blank_pdf(page_count: int = 1):
    writer = PdfWriter()
    for _ in range(page_count):
        writer.add_blank_page(width=300, height=300)
    output = io.BytesIO()
    writer.write(output)
    return output.getvalue()


def _folio_tsv(token: str, *, left: int, top: int) -> str:
    return (
        "level\tpage_num\tblock_num\tpar_num\tline_num\tword_num\tleft\ttop\twidth\theight\tconf\ttext\n"
        "1\t1\t0\t0\t0\t0\t0\t0\t1000\t1400\t-1\t\n"
        f"5\t1\t1\t1\t1\t1\t{left}\t{top}\t40\t30\t95\t{token}\n"
    )


def _scanned_index_tsv() -> str:
    header = (
        "level\tpage_num\tblock_num\tpar_num\tline_num\tword_num\t"
        "left\ttop\twidth\theight\tconf\ttext\n"
    )
    records = ["1\t1\t0\t0\t0\t0\t0\t0\t1000\t1400\t-1\t"]
    # Table borders: serial | particulars | printed page | next column.
    for left in (100, 170, 620, 790):
        records.append(f"5\t1\t1\t1\t1\t1\t{left}\t100\t4\t1100\t0\t ")
    for top in (100, 300):
        records.append(f"5\t1\t1\t1\t1\t1\t100\t{top}\t800\t4\t0\t ")
    words = [
        (120, 160, "16."),
        (190, 160, "Annexure"),
        (330, 160, "P"),
        (355, 160, "-"),
        (380, 160, "4:"),
        (190, 210, "A"),
        (220, 210, "true"),
        (270, 210, "copy"),
        (650, 180, "73-144"),
    ]
    for word_num, (left, top, text) in enumerate(words, 1):
        records.append(f"5\t1\t2\t1\t1\t{word_num}\t{left}\t{top}\t50\t24\t95\t{text}")
    return header + "\n".join(records) + "\n"


def test_margin_folio_accepts_scanned_top_numbers_and_roman_letters():
    assert margin_folio_from_tsv(_folio_tsv("92", left=900, top=35)) == "92"
    assert margin_folio_from_tsv(_folio_tsv("V", left=480, top=35)) == "V"


def test_margin_folio_prefers_outer_center_number_over_reproduced_page_number():
    tsv = _folio_tsv("644", left=480, top=45) + (
        "5\t1\t2\t1\t2\t1\t900\t20\t40\t30\t95\t2\n"
    )

    assert margin_folio_from_tsv(tsv) == "644"


def test_margin_folio_rejects_title_words_and_years():
    assert margin_folio_from_tsv(_folio_tsv("IN", left=60, top=180)) is None
    assert margin_folio_from_tsv(_folio_tsv("2025", left=900, top=35)) is None


def test_scanned_index_geometry_rejoins_annexure_and_printed_range():
    assert index_table_rows_from_tsv(_scanned_index_tsv()) == (
        "16.\tAnnexure P-4: A true copy\t73-144"
    )


def test_scanned_page_uses_ocr_and_clears_unreadable_flag():
    text = "SYNOPSIS\n" + "The petitioner seeks relief in this petition. " * 5
    with patch(
        "extraction_review.structure_split.ocr_sparse_pages", return_value={1: text}
    ):
        unit = extract_page_units(blank_pdf())[0]
    assert unit.text == text.strip()
    assert not unit.requires_ocr
    assert unit.pdf_page == 1


def test_missing_ocr_retains_unreadable_flag():
    with patch("extraction_review.split_ocr.shutil.which", return_value=None):
        assert extract_page_units(blank_pdf())[0].requires_ocr


def test_llamaparse_text_is_authoritative_and_never_invokes_local_ocr():
    parsed = "# INDEX\n\n| S.No. | Particulars | Pages |\n|---|---|---|\n| 1 | Synopsis | 4-6 |"
    with (
        patch("extraction_review.structure_split.extract_split_layout") as layout,
        patch("extraction_review.structure_split.pages_with_large_images") as images,
        patch("extraction_review.structure_split.ocr_sparse_pages") as ocr,
    ):
        # Even a conflicting local text layer must not replace the Parse result.
        layout.return_value = {1: ("LIST OF DATES AND EVENTS", "wrong table", "99")}
        unit = extract_page_units(
            blank_pdf(),
            page_texts={1: parsed},
            use_local_ocr=False,
            supplied_text_source="llamaparse",
        )[0]

    assert unit.text.startswith("INDEX\n")
    assert "| 1 | Synopsis | 4-6 |" in unit.text
    assert unit.extraction_method == "llamaparse"
    images.assert_not_called()
    ocr.assert_not_called()


def test_llamaparse_heading_markup_is_visible_to_boundary_rules():
    from extraction_review.split_repair import _outer_anchor_label

    parsed = "## <u>**LIST OF DATES**</u>\n\n| 5-12-1994 | Event text |"
    unit = extract_page_units(
        blank_pdf(),
        page_texts={1: parsed},
        use_local_ocr=False,
        supplied_text_source="llamaparse",
    )[0]

    assert _outer_anchor_label(unit.text) == "List of Dates & Events"


def test_llamaparse_multisheet_markdown_index_is_normalized_without_title_repeat():
    first = """## INDEX OF PAPERS
| Sl. No. | Particulars of Documents | Page No. |
| --- | --- | --- |
| 12. | Annexure A-1: FIR | 45-50 |
| 13. | Annexure A-2: Deposition of P.W.1 | 51-63 |
"""
    continuation = """| | Deposition continued | |
| --- | --- | --- |
| 14. | Annexure A-3: Deposition of P.W.2 | 64-70 |
| 15. | Annexure A-4: Deposition of P.W.3 | 71-77 |
"""
    units = extract_page_units(
        blank_pdf(2),
        page_texts={1: first, 2: continuation},
        use_local_ocr=False,
        supplied_text_source="llamaparse",
    )

    assert units[0].text.startswith("INDEX\nS.No. Particulars Page No.\n")
    assert "12.\tAnnexure A-1: FIR\t45-50" in units[0].text
    assert units[1].text.startswith("INDEX\nS.No. Particulars Page No.\n")
    assert "14.\tAnnexure A-3: Deposition of P.W.2\t64-70" in units[1].text


def test_llamaparse_folio_metadata_is_appended_as_spatial_evidence():
    unit = extract_page_units(
        blank_pdf(),
        page_texts={1: "Annexure A/16\nDEPOSITION OF WITNESS"},
        page_folios={1: "118"},
        page_confidences={1: 0.952},
        use_local_ocr=False,
        supplied_text_source="llamaparse",
    )[0]

    assert unit.text.endswith("\n118")
    assert unit.printed_page == "118"
    assert unit.parse_confidence == 0.952
    assert unit.annexure_labels == ("Annexure A-16",)


def test_disposed_order_is_not_mistaken_for_an_application():
    from extraction_review.split_repair import _outer_anchor_label

    parsed = """## IN THE SUPREME COURT OF INDIA
### CIVIL APPELLATE JURISDICTION
#### CIVIL APPEAL NO. 3638 OF 2007
APPELLANTS VERSUS RESPONDENTS
#### <u>O R D E R</u>
The civil appeal is, accordingly, dismissed.
Pending interlocutory application(s), if any, is/are disposed of.
[ HEMANT GUPTA ]
[ V. RAMASUBRAMANIAN ]
New Delhi; MARCH 09, 2022.
"""
    unit = extract_page_units(
        blank_pdf(),
        page_texts={1: parsed},
        use_local_ocr=False,
        supplied_text_source="llamaparse",
    )[0]

    assert _outer_anchor_label(unit.text) == "Impugned Order"


def test_markdown_table_date_row_is_lod_continuation():
    from extraction_review.split_repair import _looks_like_lod_continuation

    assert _looks_like_lod_continuation(
        "| 09.09.1999 | The authority issued the challenged notification. |"
    )


def test_ocr_timeout_does_not_invent_text():
    with (
        patch("extraction_review.split_ocr.shutil.which", return_value="tesseract"),
        patch(
            "extraction_review.split_ocr.subprocess.run",
            side_effect=subprocess.TimeoutExpired("tesseract", 30),
        ),
    ):
        assert ocr_sparse_pages(blank_pdf(), [1]) == {1: ""}


def test_split_ocr_concurrency_is_configurable_and_bounded(monkeypatch):
    monkeypatch.setenv("SPLIT_OCR_CONCURRENCY", "4")
    assert split_ocr_concurrency() == 4
    monkeypatch.setenv("SPLIT_OCR_CONCURRENCY", "100")
    assert split_ocr_concurrency() == 8


def test_ocr_keeps_native_annexure_stamp():
    with patch(
        "extraction_review.structure_split.ocr_sparse_pages",
        return_value={
            1: "A scanned judgment with a long body but an unreadable handwritten annexure label."
        },
    ):
        unit = extract_page_units(blank_pdf(), page_texts={1: "ANNEXURE P-9"})[0]
    assert unit.text.startswith("ANNEXURE P-9\n")


def test_large_inserted_page_image_is_selected_even_with_native_text():
    import pymupdf

    # A one-pixel PNG stretched over most of the page models a scanned sheet;
    # native footer text must not cause the scan to bypass OCR.
    pixmap = pymupdf.Pixmap(pymupdf.csRGB, pymupdf.IRect(0, 0, 10, 10), False)
    pixmap.clear_with(255)
    png = pixmap.tobytes("png")
    document = pymupdf.open()
    page = document.new_page(width=600, height=800)
    page.insert_image(pymupdf.Rect(20, 20, 580, 760), stream=png)
    page.insert_text((40, 790), "Digitally added footer with enough searchable text")
    data = document.tobytes()
    document.close()

    assert pages_with_large_images(data) == {1}


def test_image_only_heading_is_merged_with_existing_text_layer():
    native = "Digitally searchable footer and metadata. " * 4
    scanned = "ANNEXURE P-12\nScanned order that starts on this page"
    with (
        patch(
            "extraction_review.structure_split.pages_with_large_images",
            return_value={1},
        ),
        patch(
            "extraction_review.structure_split.ocr_sparse_pages",
            return_value={1: scanned},
        ) as ocr,
    ):
        unit = extract_page_units(blank_pdf(), page_texts={1: native})[0]

    assert unit.text.startswith("ANNEXURE P-12")
    assert native.strip() in unit.text
    ocr.assert_called_once()
    assert ocr.call_args.args[1] == [1]


def test_image_only_vakalatnama_is_not_left_unidentified():
    text = """VAKALATNAMA
IN THE SUPREME COURT OF INDIA, NEW DELHI
ARBITRATION PETITION NO. ___ OF 2025
Easy Handling LLC ... Petitioner
Versus
Pradhaan Air Express Pvt. Ltd. ... Respondent
We hereby appoint and retain the Advocate-on-Record to act and appear.
MEMO OF APPEARANCE
Please enter my appearance on behalf of the Petitioner.
"""
    with patch(
        "extraction_review.structure_split.ocr_sparse_pages", return_value={1: text}
    ):
        result = structure_aware_split(blank_pdf())

    assert result.page_parts == {1: ["Vakalatnama", "Memo of Appearance"]}
