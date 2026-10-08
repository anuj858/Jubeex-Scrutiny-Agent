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


def blank_pdf():
    writer = PdfWriter()
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
