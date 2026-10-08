"""Geometry regressions use synthetic PDFs, not private case documents."""

import pymupdf

from extraction_review.split_audit import index_rows_with_printed_pages
from extraction_review.split_pdf_layout import printed_folio
from extraction_review.structure_split import extract_page_units


def _table(page, rows):
    xs = (40, 90, 440, 540)
    ys = [100 + i * 60 for i in range(len(rows) + 1)]
    for x in xs:
        page.draw_line((x, ys[0]), (x, ys[-1]))
    for y in ys:
        page.draw_line((xs[0], y), (xs[-1], y))
    for i, cells in enumerate(rows):
        for j, cell in enumerate(cells):
            page.insert_text((xs[j] + 4, ys[i] + 20), cell, fontsize=9)


def test_index_cells_continue_across_pages_and_do_not_capture_annexed_index():
    with pymupdf.open() as pdf:
        page = pdf.new_page()
        page.insert_text((40, 60), "INDEX")
        _table(
            page,
            [
                ("S.No", "Particulars", "Page No."),
                ("1.", "ANNEXURE P-1 Order in Case No. 765", "32-41"),
                ("2.", "ANNEXURE P-2 Copy of", ""),
            ],
        )
        page = pdf.new_page()
        _table(
            page,
            [
                ("", "order dated 01.02.1982", "42-47"),
                ("3.", "Filing Memo", "48-49"),
                ("4.", "Vakalatnama", "50"),
            ],
        )
        page = pdf.new_page()
        page.insert_text((40, 150), "IN THE HIGH COURT\nBody of reproduced record")
        page.insert_text((500, 30), "32")
        page.insert_text((40, 60), "ANNEXURE P-1")
        page = pdf.new_page()
        page.insert_text((40, 60), "INDEX")
        _table(
            page,
            [
                ("S.No", "Particulars", "Page No."),
                ("1.", "Internal application", "1-2"),
                ("2.", "Internal affidavit", "3-4"),
            ],
        )
        page.insert_text((500, 30), "33")
        page.insert_text((40, 80), "ANNEXURE P-2")
        units = extract_page_units(pdf.tobytes())

    rows = index_rows_with_printed_pages(units[0].text + "\n" + units[1].text)
    assert [(row.start, row.end) for row in rows] == [
        (32, 41),
        (42, 47),
        (48, 49),
        (50, 50),
    ]
    assert "1982" in rows[1].particulars
    assert units[2].printed_page == "32"
    assert units[2].text.rstrip().endswith("\n32")
    assert "ANNEXURE P-1" in units[2].text
    assert "\t" not in units[3].text
    assert "ANNEXURE P-2" in units[3].text


def test_supplied_ocr_is_preserved_and_margin_case_number_is_not_a_folio():
    with pymupdf.open() as pdf:
        page = pdf.new_page()
        page.insert_text((40, 30), "Writ Petition No. 450 of 1986")
        page.insert_text((40, 150), "scan")
        ocr = "Longer OCR body from the scanned page, including the affidavit."
        units = extract_page_units(pdf.tobytes(), page_texts={1: ocr})
    assert units[0].text == ocr
    assert units[0].printed_page is None


def test_last_index_page_can_have_only_one_remaining_row():
    with pymupdf.open() as pdf:
        page = pdf.new_page()
        page.insert_text((40, 60), "INDEX")
        _table(
            page,
            [
                ("S.No", "Particulars", "Page No."),
                ("1.", "ANNEXURE P-1", "1-2"),
                ("2.", "Filing Memo", "3"),
            ],
        )
        page = pdf.new_page()
        _table(page, [("3.", "Vakalatnama", "4")])
        units = extract_page_units(pdf.tobytes())
    assert units[1].text.startswith("INDEX\n")
    assert "3.\tVakalatnama\t4" in units[1].text


def test_index_without_vector_grid_is_rebuilt_from_word_geometry():
    """A visual table may have no detectable ruling lines and vertical text order."""
    with pymupdf.open() as pdf:
        page = pdf.new_page(width=595, height=842)
        page.insert_text((72, 55), "MASTER INDEX")
        page.insert_text((72, 90), "S.No")
        page.insert_text((118, 90), "Particulars")
        page.insert_text((280, 90), "Page No.")
        rows = [
            ("1.", "Synopsis and List of Dates", "B-Y"),
            ("2.", "Special Leave Petition with Affidavit", "14-38"),
            ("3.", "ANNEXURE P-1 Credit letter", "48-63"),
        ]
        for index, (serial, body, span) in enumerate(rows):
            y = 130 + index * 70
            page.insert_text((73, y), serial)
            page.insert_textbox((118, y - 12, 265, y + 48), body, fontsize=9)
            page.insert_text((280, y), span)
        units = extract_page_units(pdf.tobytes())

    parsed = index_rows_with_printed_pages(units[0].text)
    assert [(row.mapped_part, row.kind, row.start, row.end) for row in parsed] == [
        ("Synopsis", "letter", 66, 89),
        ("Main Petition", "number", 14, 38),
        ("Annexure P-1", "number", 48, 63),
    ]


def test_judgment_page_counter_is_a_fallback_folio():
    signed = "Page 1 of 9\nSignature Not Verified\nSigning Date:11.11.2025"
    assert printed_folio(signed) == ("number", 1, "")
    assert printed_folio(signed + "\n92") == ("number", 92, "")
