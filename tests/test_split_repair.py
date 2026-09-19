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
