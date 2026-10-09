"""Inline annexure references retain their visually aligned numeric bounds."""

from __future__ import annotations

import pymupdf
import pytest

from extraction_review.document_parts import (
    annexure_label_from_text,
    annexure_ref_in_heading,
)
from extraction_review.split_pdf_layout import (
    annexure_cited_page_ranges,
    extract_split_layout,
)


def _insert_detached_range(page: pymupdf.Page, *, y: float = 120) -> None:
    """Populate the same visual line in a deliberately wrong insertion order."""
    fragments = [
        ("ANNEXURE P-9 (Kindly see Pages ", False),
        ("189", True),
        (" ___ to ", False),
        ("199", True),
        (")", False),
    ]
    positioned = []
    x = 40.0
    for fragment, inserted_later in fragments:
        positioned.append((x, fragment, inserted_later))
        x += pymupdf.get_text_length(fragment, fontsize=10)
    for later in (False, True):
        for x, fragment, inserted_later in positioned:
            if inserted_later == later:
                page.insert_text((x, y), fragment, fontsize=10)


def test_geometry_recovers_filled_range_without_reordering_native_text() -> None:
    with pymupdf.open() as pdf:
        page = pdf.new_page()
        page.insert_text((40, 70), "The prior appeal is annexed herewith as")
        _insert_detached_range(page)
        page.insert_text((40, 150), "The next chronology paragraph continues.")
        page.insert_text((500, 30), "X")
        native = page.get_text("text")
        assert annexure_cited_page_ranges(native) == []
        layout = extract_split_layout(pdf.tobytes())

    text, index, folio = layout[1]
    assert text.startswith(native)
    assert text[len(native) :] == (
        "Reference to Annexure P-9 (Kindly see Pages 189 to 199)."
    )
    assert annexure_cited_page_ranges(text) == [("Annexure P-9", 189, 199)]
    assert index is None
    assert folio == "X"


def test_geometry_reference_is_not_an_annexure_start() -> None:
    reference = "Reference to Annexure P-9 (Kindly see Pages 189 to 199)."
    assert annexure_label_from_text(reference) is None
    assert annexure_ref_in_heading(reference) is None


def test_native_complete_reference_is_not_appended_twice() -> None:
    with pymupdf.open() as pdf:
        page = pdf.new_page()
        page.insert_text(
            (40, 120), "Reference to Annexure P-12 (Please see Pages 215 to 232)."
        )
        native = page.get_text("text")
        layout = extract_split_layout(pdf.tobytes())
    assert layout[1][0] == native


def test_index_page_does_not_gain_narrative_reference_enrichment() -> None:
    with pymupdf.open() as pdf:
        page = pdf.new_page()
        page.insert_text((40, 60), "INDEX")
        _insert_detached_range(page)
        native = page.get_text("text")
        layout = extract_split_layout(pdf.tobytes())
    assert layout[1][0] == native


@pytest.mark.parametrize(
    ("text", "expected"),
    [
        (
            "marked as ANNEXURE P-8 (Kindly\nsee Pages41to188)",
            [("Annexure P-8", 41, 188)],
        ),
        (
            "ANNEXURE P-9 (Kindly see Pages\n189___ to199)",
            [("Annexure P-9", 189, 199)],
        ),
        (
            "ANNEXURE-P/10 (Please see Pages200___ to206)",
            [("Annexure P-10", 200, 206)],
        ),
        (
            "Annexure p-1 (Kindly see Pages 26 to 26)",
            [("Annexure P-1", 26, 26)],
        ),
        ("ANNEXURE P-8\nPages 41 to 188", []),
        ("ANNEXURE P-8 (Kindly see Pages___ to___)\n41\n188", []),
        ("ANNEXURE P-8 (Kindly see Pages188 to41)", []),
        ("ANNEXURE P-8 (Kindly see Pages0 to188)", []),
        ("ANNEXURE P-8 (Kindly see Pages41 to188", []),
        ("ANNEXURE P-8 (Kindly see Pages41 and188)", []),
        ("ANNEXURE P-8 (Pages41to188)", []),
        ("Kindly see Pages41to188", []),
        ("ANNEXURE P-8 dated 04.03.2024 (Kindly see Pages41to188)", []),
    ],
)
def test_parser_requires_explicit_complete_ordered_inline_reference(
    text: str, expected: list[tuple[str, int, int]]
) -> None:
    assert annexure_cited_page_ranges(text) == expected


def test_multiple_references_are_independent_and_duplicates_are_removed() -> None:
    text = (
        "Annexure P-9 (Kindly see Pages189to199)\n"
        "Annexure P-10 (Kindly see Pages200to206)\n"
        "Reference to Annexure P-9 (Kindly see Pages 189 to 199)."
    )
    assert annexure_cited_page_ranges(text) == [
        ("Annexure P-9", 189, 199),
        ("Annexure P-10", 200, 206),
    ]
