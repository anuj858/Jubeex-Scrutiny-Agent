import io
import subprocess
from unittest.mock import patch

from pypdf import PdfWriter

from extraction_review.split_ocr import ocr_sparse_pages, pages_with_large_images
from extraction_review.structure_split import extract_page_units, structure_aware_split


def blank_pdf():
    writer = PdfWriter()
    writer.add_blank_page(width=300, height=300)
    output = io.BytesIO()
    writer.write(output)
    return output.getvalue()


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


def test_ocr_keeps_native_annexure_stamp():
    with patch("extraction_review.structure_split.ocr_sparse_pages", return_value={1: "A scanned judgment with a long body but an unreadable handwritten annexure label."}):
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

    assert result.page_parts == {1: ["Vakalatnama"]}
