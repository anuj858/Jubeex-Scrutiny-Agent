"""Split audit: sequence order + Index ↔ file consistency."""

from __future__ import annotations

from extraction_review.split_audit import (
    audit_compiled_split,
    check_document_sequence,
    check_index_consistency,
    map_index_particulars_to_part,
    parse_index_rows,
)


def test_map_index_particulars_to_common_parts() -> None:
    assert map_index_particulars_to_part("Office Report on Limitation") == (
        "Office Report on Limitation"
    )
    assert map_index_particulars_to_part("Listing Proforma") == "Listing Proforma"
    assert map_index_particulars_to_part(
        "Synopsis and List of Dates"
    ) == "Synopsis"
    assert map_index_particulars_to_part("Special Leave Petition") == "Main Petition"
    assert map_index_particulars_to_part("ANNEXURE-P/4") == "Annexure P-4"
    assert map_index_particulars_to_part("ANNEXURE-E/2") == "Annexure P-2"
    assert map_index_particulars_to_part("Annexure E-1") == "Annexure P-1"
    assert map_index_particulars_to_part("Application for condonation") == (
        "Application 1"
    )
    assert map_index_particulars_to_part("Vakalatnama") == "Vakalatnama"


def test_parse_index_rows_extracts_page_spans() -> None:
    text = """
INDEX
S.No. PARTICULARS OF DOCUMENTS Page No. of part
1. Office Report on Limitation 8-9
2. Listing Proforma 10-12
3. Synopsis and List of Dates 13-20
4. Impugned Judgment and Order 21-30
5. Special Leave Petition 31-50
6. Affidavit 51-52
7. ANNEXURE-P/1 53-70
8. ANNEXURE-P/4 71-90
9. Application for exemption 91-92
10. Vakalatnama 93
"""
    rows = parse_index_rows(text)
    by_part = {row.mapped_part: row for row in rows if row.mapped_part}
    assert by_part["Office Report on Limitation"].start_page == 8
    assert by_part["Main Petition"].start_page == 31
    assert by_part["Annexure P-4"].end_page == 90
    assert by_part["Vakalatnama"].start_page == 93


def test_sequence_flags_out_of_order_main_petition_before_index() -> None:
    spans = [
        {"name": "Main Petition", "start_page": 10, "end_page": 20},
        {"name": "Index", "start_page": 21, "end_page": 22},
    ]
    flags = check_document_sequence(spans)
    assert any(flag.code == "out_of_sequence" for flag in flags)
    assert any(flag.part == "Main Petition" for flag in flags)


def test_sequence_ok_when_order_matches() -> None:
    spans = [
        {"name": "Index", "start_page": 5, "end_page": 6},
        {"name": "Office Report on Limitation", "start_page": 7, "end_page": 8},
        {"name": "Main Petition", "start_page": 20, "end_page": 40},
        {"name": "Annexure P-1", "start_page": 41, "end_page": 50},
        {"name": "Vakalatnama", "start_page": 51, "end_page": 51},
    ]
    assert check_document_sequence(spans) == []


def test_index_lists_missing_document() -> None:
    rows = parse_index_rows(
        "1. Special Leave Petition 10-20\n2. ANNEXURE-P/5 30-40\n"
    )
    spans = [
        {"name": "Main Petition", "start_page": 10, "end_page": 20},
    ]
    flags = check_index_consistency(rows, spans, page_count=50)
    codes = {flag.code for flag in flags}
    assert "in_index_missing_in_file" in codes
    missing = next(flag for flag in flags if flag.code == "in_index_missing_in_file")
    assert missing.part == "Annexure P-5"


def test_file_has_document_missing_from_index() -> None:
    rows = parse_index_rows("1. Special Leave Petition 10-20\n")
    spans = [
        {"name": "Main Petition", "start_page": 10, "end_page": 20},
        {"name": "Annexure P-2", "start_page": 30, "end_page": 35},
        {"name": "Cover Page", "start_page": 1, "end_page": 1},
    ]
    flags = check_index_consistency(rows, spans, page_count=40)
    codes = {flag.code: flag for flag in flags}
    assert "in_file_missing_in_index" in codes
    assert codes["in_file_missing_in_index"].part == "Annexure P-2"
    # Cover is optional in Index — not flagged.
    assert all(flag.part != "Cover Page" for flag in flags)


def test_index_page_out_of_range() -> None:
    rows = parse_index_rows("1. Special Leave Petition 90-95\n")
    spans = [{"name": "Main Petition", "start_page": 10, "end_page": 20}]
    flags = check_index_consistency(rows, spans, page_count=50)
    assert any(flag.code == "index_page_out_of_range" for flag in flags)


def test_audit_compiled_split_end_to_end() -> None:
    page_parts = {
        1: ["Cover Page"],
        2: ["Index"],
        3: ["Index"],
        10: ["Main Petition"],
        11: ["Main Petition"],
        20: ["Annexure P-1"],
    }
    texts = {
        2: (
            "INDEX\nS.No. PARTICULARS Page No.\n"
            "1. Special Leave Petition 10-11\n"
            "2. ANNEXURE-P/1 20-25\n"
            "3. ANNEXURE-P/9 40-45\n"
        ),
        3: "continuation of index table",
        10: "IN THE SUPREME COURT OF INDIA\nQUESTIONS OF LAW",
        20: "ANNEXURE P-1\nexhibit",
    }
    for page in range(1, 50):
        texts.setdefault(page, "body")

    audit = audit_compiled_split(page_parts, texts, page_count=49)
    codes = {flag["code"] for flag in audit["flags"]}
    assert "in_index_missing_in_file" in codes  # Annexure P-9 listed, absent
    assert any(row["mapped_part"] == "Main Petition" for row in audit["index_rows"])
    assert audit["flag_counts"]["total"] >= 1
