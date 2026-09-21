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


def test_repair_demotes_cover_tagged_as_main_and_recovers_petition_body() -> None:
    """Llama often labels the cover as Main Petition and only that one page."""
    cover = (
        "IN THE SUPREME COURT OF INDIA\nCIVIL APPELLATE JURISDICTION\n"
        "SPECIAL LEAVE PETITION (CIVIL) NO. _______OF 2026\n"
        "(Under Article 136 of the Constitution of India)\n"
        "PAPER BOOK\n(FOR INDEX KINDLY SEE INSIDE)\nADVOCATE FOR THE PETITIONER"
    )
    petition_start = (
        "IN THE SUPREME COURT OF INDIA\nCIVL APPELLATE JURISDICTION\n"
        "SPECIAL LEAVE PETITION (CIVIL) NO. ______ OF 2026\n"
        "POSITION OF PARTIES\nMOST RESPECTFULLY SHOWETH:\n1. The instant"
    )
    page_parts = {1: ["Main Petition"]}  # wrong Llama label; no other Main pages
    texts = {
        1: cover,
        2: "INDEX\nS.No. PARTICULARS Page No.\n1. Office Report 1",
        3: "OFFICE REPORT ON LIMITATION\n1. Within time.",
        4: "PROFORMA FOR FIRST LISTING\nThe case pertains to",
        5: "SYNOPSIS\nThe instant Special Leave Petition",
        6: "LIST OF DATES AND EVENTS\n1964 event",
        7: "1",
        8: "2",
        9: petition_start,
        10: "Grounds A. Because the High Court erred",
        11: "MAIN PRAYER:\nGrant Special Leave",
        12: (
            "IN THE SUPREME COURT OF INDIA\nCERTIFICATE\n"
            "Certified that the Special Leave Petition is confined only to the pleadings"
        ),
        13: "APPENDIX – A\nCODE OF CIVIL PROCEDURE",
        14: "ANNEXURE-P1\nexhibit",
    }
    for page in range(1, 15):
        texts.setdefault(page, "body")
    repaired, _ = repair_compiled_split(page_parts, texts, page_count=14)
    assert repaired[1] == ["Cover Page"]
    assert repaired[9] == ["Main Petition"]
    assert repaired[10] == ["Main Petition"]
    assert repaired[11] == ["Main Petition"]
    assert repaired[12] == ["AOR's Certificate"]
    assert 1 not in {
        p for p, names in repaired.items() if names == ["Main Petition"]
    }


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


def test_cover_with_spaced_ocr_prayer_and_ia_list_is_not_application() -> None:
    """Defect_File__001-style cover: spaced SCI caption + WITH PRAYER + I.A. list."""
    from extraction_review.split_repair import (
        _looks_like_cover_page,
        _outer_anchor_label,
    )

    cover = (
        "A \n"
        "IN   THE    SUPREME   COURT   OF   INDIA \n"
        "[Order XXI, Rule 3, SCR, 2013] \n"
        "CIVIL APPELLATE JURISDICTION \n"
        "SPECIAL LEAVE PETITION \n"
        "(Under Article 136 of the Constitution of India) \n"
        "SPECIAL LEAVE PETITION (CIVIL) NO.  ________ OF 2025 \n"
        "(WITH PRAYER FOR INTERIM RELIEF) \n"
        "IN THE MATTER OF: \n"
        "COL. PAWAN KUMAR GOPINATH SHARMA \n"
        "…..PETITIONER \n"
        "VERSUS \n"
        "K.D.R. FARMS CO OP HOUSING SOCIETY LTD & ORS. \n"
        "….RESPONDENT \n"
        "W I T H \n"
        "I.A. NO._____OF 2025: APPLICATION FOR EXEMPTION FROM "
        "FILING CERTIFIED COPY OF IMPUGNED ORDER \n"
        "PAPER BOOK \n"
        "{COVER PAGE}\n"
        "(FOR INDEX PLEASE SEE INSIDE) \n"
        "ADVOCATE FOR THE PETITIONER:  Ajit Sharma \n"
    )
    assert _looks_like_cover_page(cover)
    assert _outer_anchor_label(cover) == "Cover Page"

    page_parts = {1: ["Main Petition"], 2: ["Index"], 9: ["Main Petition"]}
    texts = {
        1: cover,
        2: "INDEX\nS.No. PARTICULARS\n1. Office Report on Limitation",
        9: (
            "IN THE SUPREME COURT OF INDIA\nCIVIL APPELLATE JURISDICTION\n"
            "SPECIAL LEAVE PETITION\nQUESTIONS OF LAW\nMOST RESPECTFULLY SHOWETH"
        ),
    }
    for page in range(1, 10):
        texts.setdefault(page, "body")
    repaired, _ = repair_compiled_split(page_parts, texts, page_count=9)
    assert repaired[1] == ["Cover Page"]
    assert repaired[9] == ["Main Petition"]


def test_form28_party_schedule_is_main_petition_not_cover() -> None:
    from extraction_review.split_repair import (
        _looks_like_cover_page,
        _looks_like_sci_main_petition,
        _outer_anchor_label,
    )

    party_start = (
        "IN THE SUPREME COURT OF INDIA\n"
        "CIVIL APPELLATE JURISDICTION\n"
        "SPECIAL LEAVE PETITION (Civil) NO. OF 2025\n"
        "WITH PRAYER FOR INTERIM RELIEF\n"
        "BEFORE HIGH COURT\tBEFORE SUPREME COURT\n"
        "1.Col. Pawan Kumar Gopinath Sharma, s/o Gopinath Sharma\n"
        "Respondent No.\tPetitioner\n"
        "Versus\n"
        "1. K.D.R. Farms Co. op Housing Society Ltd.\n"
        "Petitioner No.\tRespondent No.\n"
    )
    body = (
        "TO, THE HON'BLE THE CHIEF JUSTICE OF INDIA AND HIS COMPANION "
        "JUSTICES OF THE HON'BLE SUPREME COURT OF INDIA.\n"
        "MOST RESPECTFULLY SHOWETH:\n"
        "1. This Special Leave to Appeal is being filed against the impugned order.\n"
    )
    assert not _looks_like_cover_page(party_start)
    assert _looks_like_sci_main_petition(party_start)
    assert _outer_anchor_label(party_start) == "Main Petition"
    assert _looks_like_sci_main_petition(body)
    assert _outer_anchor_label(body) == "Main Petition"


def test_ocr_noisy_listing_notice_main_and_false_impugned() -> None:
    """Defect_005-style OCR: LISTIN.G, S_UPRE1:IE, court notice, CERTIFICA tE."""
    from extraction_review.split_repair import _outer_anchor_label

    listing = (
        "PROFORMA FOR FIRST LISTIN.G\nSECTION - XIV\n"
        "The case pertains to (Please tick/ check the correct box):\n"
        "1. Nature of matter: Civil\n"
    )
    notice = (
        "Delivery_ Mode: Registered\n"
        "IN THE SUPREME COURT OF INDIA\n"
        "EXTRA-ORDINARY APPELLATE JURISDICTION\n"
        "Petition for Special Leave to Appeal (Civil) No. 5253 OF 2021\n"
        "WHEREAS the Petition for SPECIAL LEAVE PETITION was listed for hearing "
        "before this Court on 23rd April, 2021, when the Court was pleased to pass "
        "the following order: Issue notice\n"
    )
    main = (
        "IN THE S_UPRE1:IE COURT OF INDIA\n"
        "CIVIL APPEALLATE JURISDICTION\n"
        "SPECIAL LEA VE PETITION\n"
        "POSITIQN OF PARTIES\n"
        "1. Lalit Mohan Aggarwal\tBEFORE HIGH COURT\tBEFORE THIS COURT\n"
        "Petitioner No.1\tPetitioner No.1\n"
    )
    questions = (
        "QU;gSTIONS OF LAW\n"
        "a) Whether Notice as referred to under Rule 8(6)\n"
        "4. DECLARATION IN TERMS OF RULES\n"
    )
    aor = (
        "IN THE SUPREME COURT OF INDIA\n"
        "CERTIFICA tE\n"
        "Certified that the Special Leave Petition is confined only to the\n"
        "pleadings before The Hon'ble High Court\n"
    )
    assert _outer_anchor_label(listing) == "Listing Proforma"
    assert _outer_anchor_label(notice) == "Record of Proceedings"
    assert _outer_anchor_label(main) == "Main Petition"
    assert _outer_anchor_label(questions) == "Main Petition"
    assert _outer_anchor_label(aor) == "AOR's Certificate"

    page_parts = {
        11: ["Advocate's Checklist"],
        13: ["Filing Memo"],
        23: ["Impugned Order"],
        25: ["Impugned Order"],
        32: ["Main Petition"],
    }
    texts = {11: listing, 13: notice, 23: main, 25: questions, 32: aor}
    for page in range(1, 35):
        texts.setdefault(page, "body")
    repaired, _ = repair_compiled_split(page_parts, texts, page_count=34)
    assert repaired[11] == ["Listing Proforma"]
    assert repaired[13] == ["Record of Proceedings"]
    assert repaired[23] == ["Main Petition"]
    assert repaired[25] == ["Main Petition"]
    assert repaired[32] == ["AOR's Certificate"]


def test_main_petition_extends_through_grounds_not_one_page() -> None:
    """Form-28 start + GROUNDS A/B/C body must stay Main until AOR Certificate."""
    page_parts = {9: ["Main Petition"]}
    texts = {
        9: (
            "IN THE SUPREME COURT OF INDIA\n"
            "CIVIL APPELLATE JURISDICTION\n"
            "SPECIAL LEAVE PETITION\n"
            "POSITION OF PARTIES\n"
            "1. Petitioner\tBEFORE HIGH COURT\tBEFORE THIS COURT\n"
        ),
        10: "3. DECLARATION IN TERMS OF RULE 3 (2):\nThe impugned judgement suffers from grave error.\n",
        11: "5. GROUNDS\nA. Because the High Court erred in law.\n",
        12: "C. Because even by that standard the subsequent suit ought to be stayed.\n",
        13: "D. Because the matter in issue is substantially the same.\n",
        14: "MAIN PRAYER\nGrant special leave to appeal.\n",
        15: (
            "IN THE SUPREME COURT OF INDIA\nCERTIFICATE\n"
            "Certified that the Special Leave Petition is confined only to the pleadings\n"
        ),
    }
    # LOD narrative citations must not become annexure stamps that swallow the petition.
    texts[7] = (
        "LIST OF DATES AND EVENTS\n"
        "2023 order is annexed herewith and marked as\n"
        "ANNEXURE P-10 [Pg ____ to _____].\n"
    )
    for page in range(1, 16):
        texts.setdefault(page, "")
    repaired, _ = repair_compiled_split(page_parts, texts, page_count=15)
    assert repaired[9] == ["Main Petition"]
    assert repaired[12] == ["Main Petition"]
    assert repaired[14] == ["Main Petition"]
    assert repaired[15] == ["AOR's Certificate"]
    assert all(
        repaired.get(page) == ["Main Petition"] for page in range(9, 15)
    )


def test_lod_annexure_page_cites_are_not_stamps() -> None:
    from extraction_review.document_parts import annexure_ref_in_heading

    assert (
        annexure_ref_in_heading(
            "order is annexed herewith and marked as\n"
            "ANNEXURE P-6 [Pg ____ to _____]. A true\n"
        )
        is None
    )
    assert annexure_ref_in_heading("ANNEXURE-P1\nPossession notice") is not None

    """Stamped ANNEXURE-E-n is Index Annexure P-n; later Llama P-7+ must survive."""
    from extraction_review.document_parts import annexure_ref_in_heading

    assert annexure_ref_in_heading(
        "ANNEXURE - E-1 ANDHRA BANK (A Govt of India Undertaking)"
    ).label == "Annexure P-1"
    assert annexure_ref_in_heading("ANNEXURE-:-- E-5\nNOTICE").label == "Annexure P-5"
    assert (
        annexure_ref_in_heading(
            "noise\n·\n@\nANNEXURE - E-4\nMrs. Reema Gupta"
        ).label
        == "Annexure P-4"
    )

    page_parts = {
        10: ["Annexure P-1"],
        14: ["Annexure P-3"],
        18: ["Annexure P-5"],
        22: ["Annexure P-7"],
        24: ["Annexure P-8"],
        5: ["Advocate's Checklist"],
        6: ["Advocate's Checklist"],
    }
    texts = {
        5: (
            "PROFORMA FOR FIRST LISTIN.G\nSECTION - XIV\n"
            "The case pertains to (Please tick/ check the correct box):\n"
            "1. Nature of matter: Civil\n"
        ),
        6: (
            "8. Land Acquisition Matters: N.A.\n"
            "9. Tax Matters: N.A.\n"
            "10. Special Category: N.A.\n"
            "11. Vehicle Number: N.A.\n"
            "ADVOCATE FOR THE PETITIONERS\nCODE: 399\n"
        ),
        10: "ANNEXURE - E-1\nPossession notice",
        11: "body",
        12: "ANNEXURE E-2\nAuction notice",
        13: "body",
        14: "ANNEXURE - E-3 APPENDIX VI\nform body",
        15: "body",
        16: "noise\n@\nANNEXURE - E-4\nGuarantor letter",
        17: "body",
        18: "0\nANNEXURE-:-- E-5\nE-auction",
        19: "body",
        20: "ANNEXURE - E-6\nBank letter",
        21: "",
        22: "",  # image-only P-7 start (Llama)
        23: "",
        24: "",  # image-only P-8
        25: "",
    }
    for page in range(1, 26):
        texts.setdefault(page, "")
    repaired, _ = repair_compiled_split(page_parts, texts, page_count=25)
    assert repaired[5] == ["Listing Proforma"]
    assert repaired[6] == ["Listing Proforma"]
    assert repaired[10] == ["Annexure P-1"]
    assert repaired[12] == ["Annexure P-2"]
    assert repaired[14] == ["Annexure P-3"]
    assert repaired[16] == ["Annexure P-4"]
    assert repaired[18] == ["Annexure P-5"]
    assert repaired[20] == ["Annexure P-6"]
    assert repaired[22] == ["Annexure P-7"]
    assert repaired[24] == ["Annexure P-8"]
    assert "Annexure E-1" not in repaired.get(10, [])
    assert repaired[20] != ["Annexure P-7"]
