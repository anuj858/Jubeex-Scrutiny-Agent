"""Split audit: sequence order + Index ↔ file consistency."""

from __future__ import annotations

from extraction_review.split_audit import (
    annexure_labels_from_text,
    audit_compiled_split,
    check_document_sequence,
    check_index_consistency,
    collect_expected_annexures,
    collect_index_annexure_entries,
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
    assert "annexure_mentioned_not_attached" in codes
    assert any(row["mapped_part"] == "Main Petition" for row in audit["index_rows"])
    inv = audit["annexure_inventory"]
    assert inv["mentioned_count"] == 2
    assert "Annexure P-1" in inv["mentioned"]
    assert "Annexure P-9" in inv["mentioned_not_attached"]
    assert inv["attached"] == ["Annexure P-1"]
    assert audit["flag_counts"]["total"] >= 1


def test_collect_expected_annexures_from_index_and_main() -> None:
    page_parts = {
        2: ["Index"],
        10: ["Main Petition"],
        11: ["Main Petition"],
    }
    texts = {
        2: (
            "INDEX\n1. List of dates\n"
            "2. Curative Petition\n"
            "3. ANNEXURE P-1: Writ Petition\n"
            "4. ANNEXURE P-3: High Court order\n"
        ),
        10: "body",
        11: (
            "A copy is annexed and marked as ANNEXURE P-2 "
            "(At page …). Further ANNEXURE P-3 follows."
        ),
    }
    expected = collect_expected_annexures(page_parts, texts)
    assert expected == {"Annexure P-1", "Annexure P-2", "Annexure P-3"}
    entries = collect_index_annexure_entries(page_parts, texts)
    assert [label for label, _ in entries] == ["Annexure P-1", "Annexure P-3"]
    # Bare "Annexure 5" inside annexed HC prose must not invent SCI inventory.
    assert "Annexure P-5" not in annexure_labels_from_text(
        "stated that allocable area in Annexure 5 to IPS-1"
    )

def test_annexure_series_p_petitioner_r_respondent() -> None:
    """P = Petitioner, R = Respondent; E exhibit stamps normalize to P."""
    from extraction_review.document_parts import annexure_label_from_text
    from extraction_review.split_audit import normalize_annexure_part_label

    assert normalize_annexure_part_label("P", 1) == "Annexure P-1"
    assert normalize_annexure_part_label("R", 2) == "Annexure R-2"
    assert normalize_annexure_part_label("E", 3) == "Annexure P-3"
    assert normalize_annexure_part_label("petitioner", 4) == "Annexure P-4"
    assert normalize_annexure_part_label("respondent", 5) == "Annexure R-5"

    assert annexure_label_from_text("ANNEXURE R-1\nIN THE HIGH COURT") == "Annexure R-1"
    assert annexure_label_from_text("ANNEXURE-P/2\nWrit Petition") == "Annexure P-2"
    assert annexure_label_from_text("ANNEXURE-E-3\nBank statement") == "Annexure P-3"

    labels = annexure_labels_from_text(
        "marked as ANNEXURE P-1 and ANNEXURE R-1; also ANNEXURE E-2"
    )
    assert labels == {"Annexure P-1", "Annexure R-1", "Annexure P-2"}

    page_parts = {2: ["Index"], 10: ["Main Petition"]}
    texts = {
        2: (
            "INDEX\n1. ANNEXURE P-1: Petitioner writ\n"
            "2. ANNEXURE R-1: Respondent counter affidavit\n"
        ),
        10: "body citing ANNEXURE R-2",
    }
    expected = collect_expected_annexures(page_parts, texts)
    assert "Annexure P-1" in expected
    assert "Annexure R-1" in expected
    assert "Annexure R-2" in expected
    entries = collect_index_annexure_entries(page_parts, texts)
    assert [label for label, _ in entries] == ["Annexure P-1", "Annexure R-1"]


def test_detached_index_page_column_zips_onto_rows() -> None:
    """Page numbers printed in their own column still attach to each document."""
    from extraction_review.split_audit import index_rows_with_printed_pages

    text = """
INDEX
1. Arbitration Petition with Affidavit
2. Annexure A-1 Board Resolution
3. Annexure A-4 Reminder Notice
4. FILING INDEX
5. V AKALATNAMA
6. Payment receipt for sum of Rs. 15000/-
1-21B
22-24B
36
60
61-63B
64
"""
    rows = index_rows_with_printed_pages(text)
    by_part = {row.mapped_part: row for row in rows}
    assert by_part["Main Petition"].start == 1
    assert by_part["Main Petition"].end == 21
    assert by_part["Main Petition"].end_suffix == "B"
    assert by_part["Annexure A-1"].start == 22
    assert by_part["Annexure A-4"].start == 36
    assert by_part["Filing Memo"].start == 60
    assert by_part["Vakalatnama"].end_suffix == "B"
    assert by_part["Court Fees"].start == 64
