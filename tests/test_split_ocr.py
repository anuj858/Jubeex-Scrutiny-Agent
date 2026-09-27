import io
import subprocess
from unittest.mock import patch

from pypdf import PdfWriter

from extraction_review.split_ocr import ocr_sparse_pages
from extraction_review.structure_split import extract_page_units


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
    assert unit.text == text
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
