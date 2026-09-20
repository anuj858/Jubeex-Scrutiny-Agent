"""Hybrid compiled split repair: nesting, anchors, duplicates."""

from __future__ import annotations

from extraction_review.split_repair import (
    find_duplicate_split_parts,
    repair_compiled_split,
)


def test_repair_keeps_sci_main_petition_and_nests_high_court_writ() -> None:
    page_parts = {
        1: ["Cover Page"],
        2: ["Index"],
        13: ["Listing Proforma"],
        15: ["Main Petition"],
    }
    for page in range(16, 24):
        page_parts[page] = ["Main Petition"]
    # Llama wrongly labels the High Court annexure as Main Petition.
    for page in range(76, 100):
        page_parts[page] = ["Main Petition"]
    page_parts[28] = ["Advocate's Checklist"]  # state checklist false positive
    texts = {
        1: "IN THE SUPREME COURT OF INDIA\nCIVIL APPELLATE JURISDICTION\nSPECIAL LEAVE PETITION",
        2: "INDEX\nS.No. PARTICULARS Page No. of part\nPart – I",
        13: "PROFORMA FOR FIRST LISTING\nThe case pertains to",
        15: (
            "IN THE SUPREME COURT OF INDIA\nSPECIAL LEAVE PETITION\n"
            "Under Article 136\nQUESTIONS OF LAW"
        ),
        23: "PRAYER FOR INTERIM RELIEF",
        28: (
            "ANNEXURE P-1\nO.No. Housing/D-3/Check List/Year 93\n"
            "Commissioner for Co-operation\nCheck List in registration of Housing Societies"
        ),
        50: "checklist body continuation",
        76: (
            "ANNEXURE-P/4\nIN THE HIGH COURT OF JUDICATURE OF BOMBAY\n"
            "WRIT PETITION NO OF 2015"
        ),
        79: "MOST RESPECTFULLY SHOWETH THAT:\nTHE HUMBLE PETITION",
        99: "prayer of the high court writ",
        100: "foot\nANNEXURE-P/5\n96",
        101: "high court order body",
        103: (
            "IN THE SUPREME COURT OF INDIA\nI.A. NO. ____ of 2025\n"
            "IN SPECIAL LEAVE PETITION"
        ),
        105: "IN THE SUPREME COURT OF INDIA\nVAKALATNAMA",
    }
    for page in range(1, 106):
        texts.setdefault(page, "continuation body")

    repaired, duplicates = repair_compiled_split(page_parts, texts, page_count=105)
    assert repaired[15] == ["Main Petition"]
    assert repaired[23] == ["Main Petition"]
    assert repaired[28] == ["Annexure P-1"]
    assert repaired[50] == ["Annexure P-1"]
    assert repaired[76] == ["Annexure P-4"]
    assert repaired[79] == ["Annexure P-4"]
    assert repaired[99] == ["Annexure P-4"]
    assert repaired[100] == ["Annexure P-5"]
    assert repaired[103][0].startswith("Application")
    assert repaired[105] == ["Vakalatnama"]
    # Longer HC block must not remain Main Petition.
    assert repaired.get(77) != ["Main Petition"]
    assert repaired[76] == ["Annexure P-4"]
    assert repaired[1] == ["Cover Page"]
    # SCI Main Petition first-run survives; HC island was nested away.
    assert all(
        repaired.get(page) != ["Main Petition"] for page in range(76, 100)
    )
    # Pre-nest duplicate signal: Llama labeled HC pages as Main Petition.
    main_dupes = [hit for hit in duplicates if hit.part == "Main Petition"]
    assert main_dupes
    assert main_dupes[0].kept_span[0] <= 15


def test_repair_keeps_first_vakalatnama_and_flags_duplicate() -> None:
    page_parts = {
        40: ["Vakalatnama"],
        41: ["Vakalatnama"],
        90: ["Vakalatnama"],
        91: ["Vakalatnama"],
    }
    texts = {
        40: "IN THE SUPREME COURT OF INDIA\nVAKALATNAMA",
        41: "acceptance body",
        90: "IN THE HIGH COURT OF JUDICATURE\nVAKALATNAMA",
        91: "nested vakalatnama body",
    }
    for page in range(1, 92):
        texts.setdefault(page, "body")
    # Without annexure stamps the HC vakalatnama is a second outer island.
    repaired, duplicates = repair_compiled_split(page_parts, texts, page_count=91)
    assert repaired[40] == ["Vakalatnama"]
    assert repaired[41] == ["Vakalatnama"]
    assert 90 not in repaired
    assert 91 not in repaired
    assert any(hit.part == "Vakalatnama" for hit in duplicates)


def test_repair_keeps_sci_cover_with_article_136_caption() -> None:
    """Real SCI covers print Article 136 + PAPER BOOK / FOR INDEX SEE INSIDE."""
    cover = (
        "IN THE SUPREME COURT OF INDIA\n"
        "CIVIL APPELLATE JURISDICTION\n"
        "[Order XXI Rule 3(1)(a) of SC Rules 2013]\n"
        "SPECIAL LEAVE PETITION (CIVIL) NO. _______OF 2026\n"
        "(Under Article 136 of the Constitution of India)\n"
        "[Against the Impugned Order dated 31.10.2025]\n"
        "IN THE MATTER OF:\nAdarsh Sharma\n...Petitioner\nVersus\nHarsh Sharma\n"
        "...Respondent\nWITH\nIA.NO.___________OF 2026\n"
        "APPLICATION FOR ADDITIONAL DOCUMENTS\n"
        "PAPER BOOK\n(FOR INDEX KINDLY SEE INSIDE)\n"
        "VIJAY KASANA\nADVOCATE FOR THE PETITIONER"
    )
    petition = (
        "IN THE SUPREME COURT OF INDIA\nSPECIAL LEAVE PETITION\n"
        "Under Article 136\nQUESTIONS OF LAW\nMOST RESPECTFULLY SHOWETH"
    )
    page_parts = {1: ["Main Petition"], 10: ["Main Petition"]}
    texts = {1: cover, 10: petition}
    for page in range(1, 12):
        texts.setdefault(page, "body")
    repaired, _ = repair_compiled_split(page_parts, texts, page_count=11)
    assert repaired[1] == ["Cover Page"]
    assert repaired[10] == ["Main Petition"]


def test_repair_index_continuation_and_blank_impugned_before_petition() -> None:
    page_parts: dict[int, list[str]] = {}
    texts = {
        1: (
            "IN THE SUPREME COURT OF INDIA\nCIVIL APPELLATE JURISDICTION\n"
            "SPECIAL LEAVE PETITION\nPAPER BOOK\n(FOR INDEX KINDLY SEE INSIDE)"
        ),
        2: "INDEX\nS.No. PARTICULARS Page No.\n1. Office Report on Limitation 1\n"
        "2. Listing Proforma 2\n3. Synopsis 3",
        3: (
            "11. Special Leave Petition along with Affidavit.\n"
            "12. Appendix: Section 10\n"
            "13. Annexure P-1: A true copy of the plaint\n"
            "14. Annexure P-2: A true copy of the order\n"
            "15. Annexure P-3: A true copy of the order"
        ),
        4: "OFFICE REPORT ON LIMITATION\n1. The Special Leave Petition is within Limitation.",
        5: "PROFORMA FOR FIRST LISTING\nThe case pertains to",
        6: "SYNOPSIS\nThe instant Special Leave Petition",
        7: "LIST OF DATES AND EVENTS\n1964 On or about",
        8: "LOD continuation body with annexure references",
        9: "1",
        10: "2",
        11: "3",
        12: (
            "IN THE SUPREME COURT OF INDIA\nSPECIAL LEAVE PETITION (CIVIL)\n"
            "POSITION OF PARTIES\nMOST RESPECTFULLY SHOWETH:\n1. The instant"
        ),
        13: "Grounds continuation A. Because",
        14: "APPENDIX – A\nCODE OF CIVIL PROCEDURE",
        15: "ANNEXURE-P1\nplaint body",
    }
    for page in range(1, 16):
        texts.setdefault(page, "body")
    repaired, _ = repair_compiled_split(page_parts, texts, page_count=15)
    assert repaired[1] == ["Cover Page"]
    assert repaired[2] == ["Index"]
    assert repaired[3] == ["Index"]
    assert repaired[4] == ["Office Report on Limitation"]
    assert repaired[5] == ["Listing Proforma"]
    assert repaired[6] == ["Synopsis"]
    assert repaired[7] == ["List of Dates & Events"]
    assert repaired[9] == ["Impugned Order"]
    assert repaired[11] == ["Impugned Order"]
    assert repaired[12] == ["Main Petition"]
    assert repaired[14] == ["Appendix"]
    assert repaired[15] == ["Annexure P-1"]


def test_find_duplicate_vakalatnama_spans() -> None:
    page_parts = {
        40: ["Vakalatnama"],
        41: ["Vakalatnama"],
        90: ["Vakalatnama"],
        91: ["Vakalatnama"],
    }
    hits = find_duplicate_split_parts(page_parts)
    assert len(hits) == 1
    assert hits[0].part == "Vakalatnama"
    assert hits[0].kept_span == (40, 41)
    assert hits[0].duplicate_spans == ((90, 91),)
