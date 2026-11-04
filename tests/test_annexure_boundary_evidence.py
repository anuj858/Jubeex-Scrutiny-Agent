"""Regressions for real exhibit boundaries versus judgment exhibit lists."""

from __future__ import annotations

import pytest

from extraction_review.document_parts import (
    annexure_label_from_text,
    page_starts_application,
)
from extraction_review.split_repair import _outer_anchor_label
from extraction_review.structure_split import PageUnit, structure_aware_split

JUDGMENT_HEADER = "WP(C) No.28199 of 2012 and conctd cases\n"
NARRATIVE_CITATION = (
    JUDGMENT_HEADER + "55\nThe details of such areas as\n"
    "extracted from the Census of India are appended as\n"
    "Annexure-3.\nIn addition, to reflect the settlement pattern\n95"
)
EXHIBIT_LIST = (
    JUDGMENT_HEADER + "113\nAPPENDIX OF WP(C) 29/2017\nPETITIONER EXHIBITS\n"
    "P1\nTRUE PHOTOCOPY OF THE MINUTES OF THE DECISION\n"
    "P2\nTRUE PHOTOCOPY OF THE LETTER\n153"
)
WRAPPED_APPLICATION_REFERENCE = (
    JUDGMENT_HEADER + "138\nAPPENDIX OF WP(C) 22372/2008\nPETITIONER EXHIBITS\n"
    "EXHIBIT P1: THE TRUE COPY OF THE ORDER\nRESPONDENT EXHIBITS\n"
    "EXHIBIT-R2(B): TRUE COPY OF THE REPORT ALONG WITH THE\n"
    "APPLICATION SUBMITTED ON 13-7-2007.\n178"
)


def _targeted_parts(texts: dict[int, str], label: str) -> dict[int, list[str]]:
    baseline = {page: [label] for page in texts}
    result = structure_aware_split(
        b"",
        llama_page_parts=baseline,
        page_units=[
            PageUnit(pdf_page=page, text=text, char_count=len(text))
            for page, text in texts.items()
        ],
        reconciliation_mode="targeted",
    )
    return result.page_parts


def test_judgment_narrative_annexure_citation_does_not_start_exhibit() -> None:
    assert annexure_label_from_text(NARRATIVE_CITATION) is None


def test_scanned_pleading_exhibit_citation_does_not_start_outer_annexure() -> None:
    # OCR may lose EXHIBIT/marked-as and leave only the P number on its own
    # line; the surviving introduction still establishes a prose citation.
    scanned_pleading = (
        "WA644/2024-Copy Of Wpc (Page-173)\n"
        "13. Accordingly, the petitioner submitted an application.\n"
        "14, Thereafter, the Fire and Rescue Services issued final Fire NOC\n"
        "dated 25-04-2012, produced herewith and\n"
        "P10. |\n"
        "15. It is submitted in this context that the notification applies.\n220"
    )
    assert annexure_label_from_text(scanned_pleading) is None
    texts = {
        1: "IN THE HIGH COURT OF KERALA\nW.P(C) No.28199 of 2012\n215",
        2: scanned_pleading,
        3: "16. The petitioner also completed the electrical works.\n221",
    }
    assert _targeted_parts(texts, "Annexure P-12") == {
        page: ["Annexure P-12"] for page in texts
    }


def test_judgment_exhibit_list_does_not_open_outer_annexure_or_appendix() -> None:
    assert annexure_label_from_text(EXHIBIT_LIST) is None
    assert _outer_anchor_label(EXHIBIT_LIST) is None


def test_wrapped_exhibit_description_does_not_start_application() -> None:
    assert not page_starts_application(WRAPPED_APPLICATION_REFERENCE)
    assert _outer_anchor_label(WRAPPED_APPLICATION_REFERENCE) is None


def test_targeted_repair_keeps_judgment_and_its_exhibit_lists_together() -> None:
    texts = {
        1: JUDGMENT_HEADER + "28\nThe common judgment continues.\n68",
        2: NARRATIVE_CITATION,
        3: JUDGMENT_HEADER + "56\nThe Court considers these submissions.\n96",
        4: EXHIBIT_LIST,
        5: WRAPPED_APPLICATION_REFERENCE,
        6: JUDGMENT_HEADER + "142\nEXHIBIT-P10: TRUE COPY OF THE NOTICE\n182",
    }
    assert _targeted_parts(texts, "Annexure P-8") == {
        page: ["Annexure P-8"] for page in texts
    }


@pytest.mark.parametrize(
    ("texts", "initial", "expected"),
    [
        (
            {
                1: "ANNEXURE P/2\nProceedings of the Secretary\n27",
                2: "The permit is revoked.\n28",
                3: "ANNEXURE P/3\nGOVERNMENT OF KERALA\n29",
                4: "Science and Technology Department\n30",
            },
            "Annexure P-2",
            {
                1: ["Annexure P-2"],
                2: ["Annexure P-2"],
                3: ["Annexure P-3"],
                4: ["Annexure P-3"],
            },
        ),
        (
            {
                1: "ANNEXURE P/5\nNOTICE\n32",
                2: "Notice continues\n33",
                3: "ANNEXURE P/6\nReply to notice\n34",
                4: "Reply continues\n35",
                5: "Reply signature\n36",
                6: "ANNEXURE P/7\nORDER\n37",
                7: "Order continues\n38",
                8: "Order continues\n39",
                9: "Order ends\n40",
                10: "ANNEXURE P/8\nIN THE HIGH COURT OF KERALA\nJUDGMENT\n41",
                11: JUDGMENT_HEADER + "2\nJudgment continues\n42",
            },
            "Annexure P-5",
            {
                **{page: ["Annexure P-5"] for page in (1, 2)},
                **{page: ["Annexure P-6"] for page in (3, 4, 5)},
                **{page: ["Annexure P-7"] for page in (6, 7, 8, 9)},
                **{page: ["Annexure P-8"] for page in (10, 11)},
            },
        ),
    ],
)
def test_targeted_repair_recovers_consecutive_explicit_outer_stamps(
    texts: dict[int, str], initial: str, expected: dict[int, list[str]]
) -> None:
    assert _targeted_parts(texts, initial) == expected


@pytest.mark.parametrize(
    "text",
    [
        "ANNEXURE P/8\n" + EXHIBIT_LIST,
        EXHIBIT_LIST + "\nANNEXURE P/8\n153",
    ],
)
def test_explicit_outer_stamp_survives_reproduced_exhibit_list(text: str) -> None:
    assert annexure_label_from_text(text) == "Annexure P-8"


def test_explicit_outer_appendix_preceding_case_appendix_still_starts() -> None:
    assert _outer_anchor_label("APPENDIX\n" + EXHIBIT_LIST) == "Appendix"
