"""Advisory evidence for a judgment merged with a different source packet."""

from __future__ import annotations

from copy import deepcopy

import pytest

from extraction_review.split_audit import (
    audit_compiled_split,
    check_unresolved_source_transitions,
)


def _merged_judgment_and_appeal() -> tuple[dict[int, list[str]], dict[int, str]]:
    # The judgment's appendices remain part of the judgment. Their Exhibit P-1
    # references are not the outer annexure identity.
    return (
        {page: ["Annexure P-8"] for page in range(1, 6)},
        {
            1: "WP(C) No.28199 of 2012 and conctd cases\nJUDGMENT\nReasons ...",
            2: (
                "WP(C) No.28199 of 2012 and conctd cases\n147\n"
                "APPENDIX OF WP(C) 4659/2010\nPETITIONER EXHIBITS\n"
                "Exhibit-P1: True copy of tax receipts\n187"
            ),
            3: (
                "WP(C) No.28199 of 2012 and conctd cases\n148\n"
                "APPENDIX OF WP(C)13882/2010\nPETITIONER EXHIBITS\n"
                "Exhibit-P1: True copy of tax receipts\n188\n//True Copy//"
            ),
            4: (
                "WA 644/2024-Affidavit\n(Page-290)\n"
                "BEFORE THE HON'BLE HIGH COURT OF JUDICATURE\n"
                "KERALA AT ERNAKULAM\nW.A.No. 644 of 2024\n"
                "(Against the Judgment in WP(C) No.28199/2012)\n"
                "Appellant vs. Respondents\nAFFIDAVIT\n"
                "I am the Appellant in the above Writ Appeal."
            ),
            5: (
                "WA 644/2024-Affidavit (Page-291)\n"
                "4. This affidavit supports my contentions in the Writ Appeal."
            ),
        },
    )


def test_source_transition_is_advisory_and_retains_original_label() -> None:
    parts, texts = _merged_judgment_and_appeal()
    original_parts, original_texts = deepcopy(parts), deepcopy(texts)

    audit = audit_compiled_split(parts, texts, page_count=5)
    flags = [
        flag
        for flag in audit["flags"]
        if flag["code"] == "unresolved_source_transition"
    ]

    assert len(flags) == 1
    flag = flags[0]
    assert flag["severity"] == "warning"
    assert flag["part"] == "Annexure P-8"
    assert flag["details"]["boundary_after_page"] == 3
    assert flag["details"]["boundary_before_page"] == 4
    assert flag["details"]["previous_source_case"] == "WPC 28199/2012"
    assert flag["details"]["next_source_case"] == "WA 644/2024"
    assert flag["details"]["auto_split"] is False
    assert "original paper-book Index and source PDF" in flag["message"]
    assert audit["flag_counts"]["warning"] == len(audit["flags"])
    assert audit["document_spans"] == [
        {"name": "Annexure P-8", "start_page": 1, "end_page": 5}
    ]
    assert parts == original_parts
    assert texts == original_texts


@pytest.mark.parametrize(
    "missing_signal",
    [
        "judgment_title",
        "certification",
        "terminal_certification",
        "repeated_prior_case",
        "repeated_new_case",
        "court_caption",
        "filing_title",
        "different_case",
        "contiguous_part",
        "unresolved_outer_identity",
    ],
)
def test_incomplete_source_evidence_does_not_warn(missing_signal: str) -> None:
    parts, texts = _merged_judgment_and_appeal()
    if missing_signal == "judgment_title":
        texts[1] = texts[1].replace("JUDGMENT", "Memorandum of writ petition")
    elif missing_signal == "certification":
        texts[3] = texts[3].replace("//True Copy//", "")
    elif missing_signal == "terminal_certification":
        texts[3] += "\nExhibit-P5: continuing list of documents"
    elif missing_signal == "repeated_prior_case":
        texts[2] = "Exhibit-P1: True copy of tax receipts"
    elif missing_signal == "repeated_new_case":
        texts[5] = "Text without a reliable source-case header"
    elif missing_signal == "court_caption":
        texts[4] = texts[4].replace("HIGH COURT", "the authority")
    elif missing_signal == "filing_title":
        texts[4] = texts[4].replace(
            "\nAFFIDAVIT\n", "\nExtract refers to an affidavit\n"
        )
    elif missing_signal == "different_case":
        for page in (4, 5):
            texts[page] = texts[page].replace("WA 644/2024", "WP(C) No.28199 of 2012")
    elif missing_signal == "contiguous_part":
        parts[2] = ["Annexure P-7"]
    elif missing_signal == "unresolved_outer_identity":
        texts[4] = "ANNEXURE P-9\n" + texts[4]

    assert check_unresolved_source_transitions(parts, texts, page_count=5) == []


def test_already_separate_outer_documents_do_not_warn() -> None:
    parts, texts = _merged_judgment_and_appeal()
    parts[4] = parts[5] = ["Annexure P-9"]

    assert check_unresolved_source_transitions(parts, texts, page_count=5) == []


def test_judgment_appendices_do_not_warn_at_exhibit_lists() -> None:
    parts, texts = _merged_judgment_and_appeal()
    texts[2] += "\n//True Copy//"
    texts[4] = (
        "WP(C) No.28199 of 2012 and conctd cases\n"
        "APPENDIX OF WP(C) 11469/2010\nPETITIONER EXHIBITS\n"
        "Exhibit-P1: True copy of affidavit before the High Court"
    )
    texts[5] = texts[4]

    assert check_unresolved_source_transitions(parts, texts, page_count=5) == []


def test_body_citation_is_not_a_source_case_header() -> None:
    parts, texts = _merged_judgment_and_appeal()
    texts[4] = texts[4].replace("WA 644/2024-Affidavit", "The affidavit in WA 644/2024")
    texts[4] = texts[4].replace("W.A.No. 644 of 2024\n", "")

    assert check_unresolved_source_transitions(parts, texts, page_count=5) == []
