"""Hybrid compiled split repair: nesting, anchors, duplicates."""

from __future__ import annotations

from extraction_review.split_repair import (
    find_duplicate_split_parts,
    repair_compiled_split,
)


def test_trailing_index_description_is_not_an_application_start() -> None:
    from extraction_review.split_repair import _outer_anchor_label

    text = (
        "Application seeking exemption\nfrom filing the death certificate\n"
        "50.\nFiling Memo\n51.\nVakalatnama and Power of Attorney\n"
        "52.\nMemo of Parties in High Court\n53.\nLetter/Declaration\n"
        "440-442\n443-444\n445-465\n466-467\n468-469\n"
    )
    assert _outer_anchor_label(text) == "Index"


def test_order_prose_with_application_references_is_not_index() -> None:
    from extraction_review.split_repair import _outer_anchor_label

    assert (
        _outer_anchor_label(
            "The case was dismissed in default.\n"
            "5. Accordingly the present application for recall was filed.\n"
            "6. A perusal of the application shows delay.\n"
            "7. The second application for recall was filed later.\n"
            "8. Objections to the application for condonation were filed.\n2"
        )
        != "Index"
    )


def test_explicit_outer_stamp_wins_over_local_annexure_number() -> None:
    from extraction_review.document_parts import annexure_label_from_text

    text = "IN THE HIGH COURT\nWrit Petition\nAnnexure No. 1\nCopy of the order\n"
    assert annexure_label_from_text(text) is None
    assert annexure_label_from_text(text + "32\nANNEXURE P-2") == "Annexure P-2"


def test_layout_ranges_keep_annexure_body_and_close_at_outer_applications() -> None:
    texts = {
        1: "INDEX\nS.No. Particulars Page No.\n"
        "1.\tANNEXURE P-6 Writ Petition\t65-68\n"
        "2.\tANNEXURE P-7 Order\t69\n"
        "3.\tI.A. NO. Application for exemption\t70-71\n"
        "4.\tI.A. NO. Application for condonation\t72-73\n",
        2: "IN THE SUPREME COURT OF INDIA\nSPECIAL LEAVE PETITION\nQUESTIONS OF LAW",
        3: "ANNEXURE P-6\nIN THE HIGH COURT\nWRIT PETITION\n65",
        4: "IN THE HIGH COURT\nAFFIDAVIT\n66",
        5: "APPLICATION FOR SUBSTITUTION\nIn the writ petition\n67",
        6: "annexure continuation\n68",
        7: "ANNEXURE P-7\nORDER\n69",
        8: "IN THE SUPREME COURT OF INDIA\nAPPLICATION FOR EXEMPTION\n70",
        9: "application body\n71",
        10: "IN THE SUPREME COURT OF INDIA\nAPPLICATION FOR CONDONATION\n72",
        11: "application body\n73",
    }
    repaired, _ = repair_compiled_split(
        {1: ["Index"], 2: ["Main Petition"]}, texts, page_count=11
    )
    assert all(repaired[p] == ["Annexure P-6"] for p in range(3, 7))
    assert repaired[7] == ["Annexure P-7"]
    assert all(repaired[p] == ["Application 1"] for p in (8, 9))
    assert all(repaired[p] == ["Application 2"] for p in (10, 11))


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


def test_repair_keeps_synopsis_out_of_main_petition_when_llama_mislabels() -> None:
    """Synopsis continuation pages often say 'Special Leave Petition'; Llama tags Main."""
    petition_start = (
        "IN THE SUPREME COURT OF INDIA\nCIVL APPELLATE JURISDICTION\n"
        "SPECIAL LEAVE PETITION (CIVIL) NO. ______ OF 2026\n"
        "POSITION OF PARTIES\nMOST RESPECTFULLY SHOWETH:\n1. The instant"
    )
    page_parts = {page: ["Main Petition"] for page in range(5, 16)}
    page_parts[1] = ["Cover Page"]
    page_parts[14] = ["Main Petition"]
    texts = {
        1: (
            "IN THE SUPREME COURT OF INDIA\nCIVIL APPELLATE JURISDICTION\n"
            "SPECIAL LEAVE PETITION (CIVIL) NO. _______OF 2026\n"
            "PAPER BOOK\n(FOR INDEX KINDLY SEE INSIDE)\nADVOCATE FOR THE PETITIONER"
        ),
        2: "INDEX\n1. Office Report 1\n2. Listing Proforma 2\n3. Synopsis 3",
        3: "OFFICE REPORT ON LIMITATION\n1. Within time.",
        4: "PROFORMA FOR FIRST LISTING\nThe case pertains to",
        5: (
            "SYNOPSIS\nThe instant Special Leave Petition under Article 136 of the "
            "Constitution of India is filed being aggrieved by the impugned judgement"
        ),
        6: (
            "and the petitioner herein by virtue of HUF to the exclusion of all "
            "others including respondent herein"
        ),
        7: (
            "Findings of Hon'ble High court as to matter in issue in previously "
            "instituted suit to be upheld"
        ),
        8: "LIST OF DATES AND EVENTS\n1964 On or about the year 1956, the great grandfather",
        9: "27.01.1968 Shri Brij Mohan Sharma separated from his wife",
        10: "1",
        11: "2",
        12: petition_start,
        13: "Grounds A. Because the High Court erred",
        14: "MAIN PRAYER:\nGrant Special Leave",
        15: (
            "IN THE SUPREME COURT OF INDIA\nCERTIFICATE\n"
            "Certified that the Special Leave Petition is confined only to the pleadings"
        ),
    }
    for page in range(1, 16):
        texts.setdefault(page, "body")
    repaired, _ = repair_compiled_split(page_parts, texts, page_count=15)
    assert repaired[5] == ["Synopsis"]
    assert repaired[6] == ["Synopsis"]
    assert repaired[7] == ["Synopsis"]
    assert repaired[8] == ["List of Dates & Events"]
    assert repaired[9] == ["List of Dates & Events"]
    assert repaired[12] == ["Main Petition"]
    assert repaired[13] == ["Main Petition"]
    assert repaired[14] == ["Main Petition"]
    assert repaired[15] == ["AOR's Certificate"]
    assert all(
        repaired.get(page) != ["Main Petition"] for page in range(5, 12)
    )


def test_repair_prefers_form28_island_over_synopsis_main_when_ocr_weak() -> None:
    """Two Main islands: synopsis first, Form-28 second — keep petition, not Unidentified."""
    petition_start = (
        "IN THE SUPREME COURT OF INDIA\nCIVIL APPELLATE JURISDICTION\n"
        "SPECIAL LEAVE PETITION (CIVIL) NO. ______ OF 2026\n"
        "POSITION OF PARTIES\nMOST RESPECTFULLY SHOWETH:\n1. The instant"
    )
    # Llama: synopsis block Main, then a gap, then real petition Main.
    page_parts = {
        1: ["Cover Page"],
        5: ["Main Petition"],
        6: ["Main Petition"],
        7: ["Main Petition"],
        8: ["Main Petition"],
        12: ["Main Petition"],
        13: ["Main Petition"],
        14: ["Main Petition"],
    }
    texts = {
        1: (
            "IN THE SUPREME COURT OF INDIA\nCIVIL APPELLATE JURISDICTION\n"
            "SPECIAL LEAVE PETITION (CIVIL) NO. _______OF 2026\n"
            "PAPER BOOK\n(FOR INDEX KINDLY SEE INSIDE)\nADVOCATE FOR THE PETITIONER"
        ),
        5: (
            "SYNOPSIS\nThe instant Special Leave Petition under Article 136 of the "
            "Constitution of India is filed being aggrieved by the impugned judgement"
        ),
        6: "continuation of synopsis narrative about the Special Leave Petition",
        7: "LIST OF DATES AND EVENTS\n1964 On or about the year 1956",
        8: "27.01.1968 event entry continued",
        9: "scanned blank",
        10: "scanned blank",
        11: "scanned blank",
        12: petition_start,
        13: "Grounds A. Because the High Court erred",
        14: "MAIN PRAYER:\nGrant Special Leave",
    }
    for page in range(1, 15):
        texts.setdefault(page, "body")
    repaired, _ = repair_compiled_split(page_parts, texts, page_count=14)
    assert repaired[5] == ["Synopsis"]
    assert repaired[6] == ["Synopsis"]
    assert repaired[7] == ["List of Dates & Events"]
    assert repaired[12] == ["Main Petition"]
    assert repaired[13] == ["Main Petition"]
    assert repaired[14] == ["Main Petition"]
    assert all(repaired.get(page) != ["Main Petition"] for page in range(5, 12))


def test_repair_ignores_annexed_form28_lookalike_when_picking_main() -> None:
    """Late annexed SLP caption must not demote the outer Main Petition away."""
    petition_start = (
        "IN THE SUPREME COURT OF INDIA\nCIVIL APPELLATE JURISDICTION\n"
        "SPECIAL LEAVE PETITION (CIVIL) NO. ______ OF 2026\n"
        "POSITION OF PARTIES\nMOST RESPECTFULLY SHOWETH:\n1. The instant"
    )
    annexed = (
        "IN THE SUPREME COURT OF INDIA\nCIVIL APPELLATE JURISDICTION\n"
        "SPECIAL LEAVE PETITION (CIVIL) NO. 123 OF 2020\n"
        "POSITION OF PARTIES\nMOST RESPECTFULLY SHOWETH:\n1. Prior SLP"
    )
    page_parts = {
        1: ["Cover Page"],
        5: ["Main Petition"],
        6: ["Main Petition"],
        7: ["Main Petition"],
        10: ["Main Petition"],
        11: ["Main Petition"],
        20: ["Main Petition"],
    }
    texts = {
        1: (
            "IN THE SUPREME COURT OF INDIA\nCIVIL APPELLATE JURISDICTION\n"
            "SPECIAL LEAVE PETITION (CIVIL) NO. _______OF 2026\n"
            "PAPER BOOK\n(FOR INDEX KINDLY SEE INSIDE)\nADVOCATE FOR THE PETITIONER"
        ),
        5: (
            "SYNOPSIS\nThe instant Special Leave Petition under Article 136 "
            "is filed being aggrieved"
        ),
        6: "synopsis continuation about Special Leave Petition",
        7: "LIST OF DATES AND EVENTS\n1964 event",
        8: "impugned blank",
        9: "impugned blank",
        # Outer Form-28 OCR is weak — only party schedule cues on body pages.
        10: "MOST RESPECTFULLY SHOWETH:\n1. The petitioners state",
        11: "MAIN PRAYER:\nGrant Special Leave",
        12: "ANNEXURE P-1\nCertified copy of the High Court order",
        20: annexed,
    }
    for page in range(1, 21):
        texts.setdefault(page, "body")
    repaired, _ = repair_compiled_split(page_parts, texts, page_count=20)
    assert repaired[5] == ["Synopsis"]
    assert repaired[10] == ["Main Petition"]
    assert repaired[11] == ["Main Petition"]
    assert repaired.get(20) != ["Main Petition"]
    assert all(repaired.get(page) != ["Main Petition"] for page in (5, 6, 7))


def test_repair_keeps_annexed_sci_rop_out_of_record_of_proceedings_slot() -> None:
    """SCI ITEM-NO RoP sheets after Form-28 are Annexure copies, not paper-book RoP."""
    rop_form = (
        "RECORD OF PROCEEDINGS\n"
        "SL No. Date of record of proceedings Pages\n"
        "1.\n2.\n3.\n"
    )
    petition = (
        "IN THE SUPREME COURT OF INDIA\nCIVIL APPELLATE JURISDICTION\n"
        "CURATIVE PETITION UNDER ARTICLES 137, 142\n"
        "POSITION OF PARTIES\nMOST RESPECTFULLY SHOWETH:\n1. The petitioner"
    )
    annexed_rop = (
        "ITEM NO.37 COURT NO.5 SECTION IIIA\n"
        "S U P R E M E C O U R T O F I N D I A\n"
        "RECORD OF PROCEEDINGS\n"
        "Petition(s) for Special Leave to Appeal (C) No.7595/2015\n"
        "(Arising out of impugned final judgment)\n"
    )
    annexed_rop_2 = (
        "ITEM NO.205 COURT NO.5 SECTION IIIA\n"
        "SUPREME COURT OF INDIA\n"
        "RECORD OF PROCEEDINGS\n"
        "Petition(s) for Special Leave to Appeal (C) No.7595/2015\n"
    )
    # Llama puts the annexed SCI orders into the RoP slot (longest island).
    page_parts = {
        2: ["Record of Proceedings"],
        8: ["Main Petition"],
        20: ["Record of Proceedings"],
        21: ["Record of Proceedings"],
        22: ["Record of Proceedings"],
        23: ["Record of Proceedings"],
        24: ["Record of Proceedings"],
    }
    texts = {
        1: (
            "IN THE SUPREME COURT OF INDIA\nCIVIL APPELLATE JURISDICTION\n"
            "CURATIVE PETITION (CIVIL) NO. ___ OF 2016\nPAPER BOOK\n"
            "(FOR INDEX KINDLY SEE INSIDE)\nADVOCATE FOR THE PETITIONER"
        ),
        2: rop_form,
        3: "INDEX\n1. List of dates\n2. Curative Petition\n4. ANNEXURE P-1",
        4: "LIST OF DATES\n2007 Development Agreement",
        8: petition,
        9: "Grounds A. Because",
        10: "MAIN PRAYER:\nAllow the curative petition",
        15: "IN THE HIGH COURT OF JUDICATURE AT BOMBAY\nWRIT PETITION NO.60 OF 2015",
        20: annexed_rop,
        21: "Let Rs.10 lacs be deposited within three days before the Registry",
        22: annexed_rop_2,
        23: "Central Board of Direct Taxes (CBDT) and directed for fresh hearing",
        24: (
            "Chamber matter SECTION IIIA\n"
            "S U P R E M E C O U R T O F I N D I A\n"
            "RECORD OF PROCEEDINGS\n"
            "R. P. (C) No. 1512/2016 In SLP (C) No. 7595/2015\n"
        ),
    }
    for page in range(1, 25):
        texts.setdefault(page, "body")
    repaired, _ = repair_compiled_split(page_parts, texts, page_count=24)
    assert repaired[2] == ["Record of Proceedings"]
    assert repaired[8] == ["Main Petition"]
    for page in range(20, 25):
        assert repaired.get(page) != ["Record of Proceedings"], page
    assert all(
        "Record of Proceedings" not in (repaired.get(page) or [])
        for page in range(20, 25)
    )


def test_narrative_filing_vakalatnama_does_not_steal_main_petition() -> None:
    """Grounds that mention 'filing Vakalatnama' must not open a Vakalatnama slot."""
    page_parts = {page: ["Main Petition"] for page in range(5, 12)}
    page_parts[1] = ["Cover Page"]
    texts = {
        1: (
            "IN THE SUPREME COURT OF INDIA\nCIVIL APPELLATE JURISDICTION\n"
            "SPECIAL LEAVE PETITION (CIVIL) NO. _______OF 2026\n"
            "PAPER BOOK\n(FOR INDEX KINDLY SEE INSIDE)\nADVOCATE FOR THE PETITIONER"
        ),
        5: (
            "IN THE SUPREME COURT OF INDIA\nCIVIL APPELLATE JURISDICTION\n"
            "SPECIAL LEAVE PETITION (CIVIL) NO. ______ OF 2026\n"
            "POSITION OF PARTIES\nMOST RESPECTFULLY SHOWETH:\n1. The instant"
        ),
        6: "Grounds A. Because the High Court erred",
        7: (
            "X. Because this Hon'ble Court failed to appreciate that counsel "
            "had also sought four weeks time filing Vakalatnama and affidavit "
            "in reply of the Respondent No. 1."
        ),
        8: "Y. Because this Hon'ble Court failed to appreciate the order",
        9: "MAIN PRAYER:\nGrant Special Leave",
        10: (
            "IN THE SUPREME COURT OF INDIA\nCERTIFICATE\n"
            "Certified that the Special Leave Petition is confined only to the pleadings"
        ),
    }
    for page in range(1, 11):
        texts.setdefault(page, "body")
    repaired, _ = repair_compiled_split(page_parts, texts, page_count=10)
    assert repaired[7] == ["Main Petition"]
    assert repaired.get(7) != ["Vakalatnama"]
    assert all(
        "Vakalatnama" not in (repaired.get(page) or []) for page in range(5, 10)
    )


def test_filing_memo_heading_wins_over_memo_of_appearance_list_item() -> None:
    """Filing Memo sheets list 'vakalatnama and memo of appearance' — keep Filing Memo."""
    page_parts = {12: ["Vakalatnama"], 13: ["Memo of Appearance"]}
    texts = {
        12: (
            "FILING MEMO\n"
            "S. No. Particulars Copies\n"
            "1. SLP with Affidavit 1+3\n"
            "5. Vakalatanama and memo of appearance 1+3\n"
            "IN THE SUPREME COURT OF INDIA\n"
            "CIVIL APPELLATE JURISDICTION\n"
            "SPECIAL LEAVE PETITION (CIVIL) NO. _______OF 2026\n"
        ),
        13: "21 January 2026.\nFiled by",
    }
    repaired, _ = repair_compiled_split(page_parts, texts, page_count=13)
    assert repaired[12] == ["Filing Memo"]
    assert repaired.get(12) != ["Memo of Appearance"]
    assert repaired.get(12) != ["Vakalatnama"]


def test_curative_affidavit_is_not_aor_certificate() -> None:
    """Affidavit pages that say 'confined only to the pleadings' are Affidavit, not AOR."""
    page_parts = {8: ["Main Petition"], 9: ["AOR's Certificate"]}
    texts = {
        8: (
            "IN THE SUPREME COURT OF INDIA\nCIVIL APPELLATE JURISDICTION\n"
            "SPECIAL LEAVE PETITION (CIVIL) NO. ______ OF 2026\n"
            "POSITION OF PARTIES\nMOST RESPECTFULLY SHOWETH:\n1. The instant"
        ),
        9: (
            "IN THE SUPREME COURT OF INDIA\n"
            "CIVIL APPELLATE JURISDICTION\n"
            "CURATIVE PETITION (CIVIL) NO. ___ OF 2016\n"
            "AFFIDAVIT\n"
            "I, Ismail AK. Balwa, do hereby solemnly affirm and state as under:-\n"
            "3. I state that the Curative Petition is confined only to the pleadings "
            "and the grounds taken therein.\n"
        ),
    }
    for page in range(1, 10):
        texts.setdefault(page, "body")
    repaired, _ = repair_compiled_split(page_parts, texts, page_count=9)
    assert repaired[9] == ["Affidavit"]
    assert repaired.get(9) != ["AOR's Certificate"]


def test_impugned_order_does_not_expand_backward_into_writ_body() -> None:
    """Backward fill must not pull HC writ prayer pages into Impugned Order."""
    page_parts = {
        5: ["Main Petition"],
        10: ["Impugned Order"],
    }
    texts = {
        5: (
            "IN THE SUPREME COURT OF INDIA\nCIVIL APPELLATE JURISDICTION\n"
            "SPECIAL LEAVE PETITION (CIVIL) NO. ______ OF 2026\n"
            "POSITION OF PARTIES\nMOST RESPECTFULLY SHOWETH:\n1. The instant"
        ),
        8: (
            "Respondent Nos. 2 & 3 to stay the demand till the disposal of "
            "the present petition by this Hon. Court"
        ),
        9: "VERIFICATION\nI, Ismail, director, do hereby solemnly declare",
        10: (
            "IMPUGNED ORDER\n"
            "IN THE HIGH COURT OF JUDICATURE AT BOMBAY\n"
            "ORDINARY ORIGINAL CIVIL JURISDICTION\n"
            "WRIT PETITION NO.60 OF 2015\n"
            "CORAM: A.A. SAYED & A.S. GADKARI, JJ.\n"
            "DATE: 20th February 2015\n"
            "P.C.\n"
            "1. Heard learned counsel.\n"
        ),
        11: "3. The impugned order dated 22.12.2014 has denied the benefit",
    }
    for page in range(1, 12):
        texts.setdefault(page, "body")
    repaired, _ = repair_compiled_split(page_parts, texts, page_count=11)
    assert repaired[10] == ["Impugned Order"]
    assert repaired.get(8) != ["Impugned Order"]
    assert repaired.get(9) != ["Impugned Order"]


def test_bare_hc_judgment_without_annexure_stamp_is_not_impugned() -> None:
    """Annexed HC orders (no P-n stamp, no Impugned title) must not become Impugned."""
    page_parts = {
        1: ["Cover Page"],
        2: ["Main Petition"],
    }
    texts = {
        1: "IN THE SUPREME COURT OF INDIA\nSPECIAL LEAVE PETITION",
        2: (
            "IN THE SUPREME COURT OF INDIA\nSPECIAL LEAVE PETITION\n"
            "MOST RESPECTFULLY SHOWETH:\n1. The petitioners"
        ),
        20: (
            "IN THE HIGH COURT OF JUDICATURE AT BOMBAY\n"
            "ORDINARY ORIGINAL CIVIL JURISDICTION\n"
            "WRIT PETITION NO.60 OF 2015\n"
            "Balwas Realty & Infrastructure Pvt Ltd. ... Petitioner\n"
            "CORAM: M.S. SANKLECHA & G.S. KULKARNI, JJ.\n"
            "DATE: 20th FEBRUARY, 2015\n"
            "P.C.\n1. This petition under Article 226\n"
        ),
        21: "2. The petitioner on 22.2.2011 applied to the CBDT\n",
        22: "//true copy//\n",
    }
    for page in range(1, 23):
        texts.setdefault(page, "body")
    repaired, _ = repair_compiled_split(page_parts, texts, page_count=22)
    assert repaired.get(20) != ["Impugned Order"]
    assert repaired.get(21) != ["Impugned Order"]
    assert repaired.get(22) != ["Impugned Order"]


def test_explicit_impugned_order_title_still_detected() -> None:
    page_parts = {1: ["Cover Page"]}
    texts = {
        1: "IN THE SUPREME COURT OF INDIA\nSPECIAL LEAVE PETITION",
        5: (
            "IMPUGNED ORDER\n"
            "IN THE HIGH COURT OF JUDICATURE AT BOMBAY\n"
            "WRIT PETITION NO.60 OF 2015\n"
            "CORAM: M.S. SANKLECHA, J.\n"
        ),
    }
    for page in range(1, 6):
        texts.setdefault(page, "body")
    repaired, _ = repair_compiled_split(page_parts, texts, page_count=5)
    assert repaired[5] == ["Impugned Order"]


def test_front_matter_multi_page_index_listing_synopsis_not_one_page_islands() -> None:
    """Index / Listing / Synopsis continuations must not collapse to page 1 only."""
    page_parts: dict[int, list[str]] = {}
    texts = {
        1: (
            "IN THE SUPREME COURT OF INDIA\nCIVIL APPELLATE JURISDICTION\n"
            "SPECIAL LEAVE PETITION (CIVIL) NO. _______OF 2024\n"
            "PAPER BOOK\n(FOR INDEX KINDLY SEE INSIDE)\nADVOCATE FOR THE PETITIONER"
        ),
        2: (
            "RECORD OF PROCEEDINGS\n"
            "SI.No Date of Records of Proceedings Page\n1\n2\n3\n"
        ),
        3: (
            "INDEX\nSl. No. Particulars of documents Page No.\n"
            "1. Court fees\n2. Office Report on Limitation\n"
            "9. Synopsis and List of Dates B – L\n"
            "11. Special Leave Petition with affidavit.\n"
        ),
        4: (
            "Court No. 1, Shimla, H.P. in Rent Petition No. 4-2 of 2013.\n"
            "13. ANNEXURE P-2: A copy of the application dated 30.08.2021\n"
            "14. ANNEXURE P-3: A copy of the reply dated Nil\n"
            "15. ANNEXURE P-4: A copy of the judgment and order\n"
        ),
        5: (
            "17. I.A. No. _____ of 2024 :\n"
            "Application for permission to file Special Leave Petition.\n"
            "19. Filing Memo 89\n"
            "20. Vakalatnama 90\n"
            "21. Memo of Parties. 91\n"
        ),
        6: (
            "PROFORMA FOR FIRST LISTING\nSECTION: XIV (H.P.)\n"
            "The case pertains to (Please tick/check the correct box):\n"
        ),
        7: (
            "6. (a) Similar disposed of matter with citation: No\n"
            "8. Land Acquisition Matters:\n"
            "9. Tax Matters: State the tax effect: N/A\n"
            "10. Special Category: N.A.\n"
            "E-28, Second Floor,\nLajpat Nagar-I,\nNew Delhi-110024\n"
        ),
        8: (
            "SYNOPSIS\n"
            "A. The present Special Leave Petition arises out of the impugned "
            "final Judgment and Order dated 12.07.2024\n"
        ),
        9: (
            "C. That inter alia the following questions of law arise for "
            "consideration of this Hon'ble Court:\n"
            "- Whether the High Court gravely erred\n"
            "Therefore filing the present Special Leave Petition\n"
        ),
        10: "LIST OF DATES AND EVENTS\n30.10.2020 The Rent Controller passed an order",
        11: "31.08.2022 The Appellate Authority passed judgment",
        12: (
            "IN THE SUPREME COURT OF INDIA\n"
            "[Order XXI, Rule 3 (1) (a) of S.C.R., 2013]\n"
            "CIVIL APPELLATE JURISDICTION\n"
            "SPECIAL LEAVE PETITION (CIVIL) NO. OF 2024\n"
            "POSITION OF PARTIES\n"
            "1. Ajay Rawat ... Petitioner\n"
            "MOST RESPECTFULLY SHOWETH:\n1. The petitioners\n"
        ),
        13: "Grounds A. Because the High Court erred",
        14: "MAIN PRAYER:\nGrant Special Leave",
    }
    for page in range(1, 15):
        texts.setdefault(page, "body")
    repaired, _ = repair_compiled_split(page_parts, texts, page_count=14)
    assert repaired[3] == ["Index"]
    assert repaired[4] == ["Index"]
    assert repaired[5] == ["Index"]
    assert repaired[6] == ["Listing Proforma"]
    assert repaired[7] == ["Listing Proforma"]
    assert repaired[7] != ["Annexure P-28"]
    assert repaired[8] == ["Synopsis"]
    assert repaired[9] == ["Synopsis"]
    assert repaired[10] == ["List of Dates & Events"]
    assert repaired[12] == ["Main Petition"]
    assert repaired[13] == ["Main Petition"]
    assert repaired.get(9) != ["Main Petition"]


def test_repair_demotes_annexures_not_mentioned_in_index_or_main() -> None:
    """Stamped annexure absent from Index/LOD/Main → unlabeled (Unidentified)."""
    page_parts = {
        1: ["Index"],
        2: ["Main Petition"],
        10: ["Annexure P-1"],
        11: ["Annexure P-1"],
        20: ["Annexure P-9"],
        21: ["Annexure P-9"],
    }
    texts = {
        1: (
            "INDEX\nSL. NO. PARTICULARS PAGE NO.\n"
            "1. Special Leave Petition\n"
            "2. ANNEXURE P-1: Writ Petition copy\n"
        ),
        2: (
            "IN THE SUPREME COURT OF INDIA\nSPECIAL LEAVE PETITION\n"
            "annexed herewith and marked as ANNEXURE P-1\n"
        ),
        10: "ANNEXURE P-1\nIN THE HIGH COURT",
        11: "writ body",
        20: "ANNEXURE P-9\nstray exhibit not in Index",
        21: "stray body",
    }
    for page in range(1, 22):
        texts.setdefault(page, "body")

    repaired, _ = repair_compiled_split(page_parts, texts, page_count=21)
    assert repaired[10] == ["Annexure P-1"]
    assert repaired[11] == ["Annexure P-1"]
    assert 20 not in repaired
    assert 21 not in repaired


def test_index_places_unstamped_annexure_islands_by_date() -> None:
    """Index P-n rows label HC / SCI RoP islands even without ANNEXURE stamps."""
    page_parts = {
        1: ["Cover Page"],
        2: ["Index"],
        3: ["List of Dates & Events"],
        8: ["Main Petition"],
        12: ["Affidavit"],
    }
    texts = {
        1: (
            "IN THE SUPREME COURT OF INDIA\nCIVIL APPELLATE JURISDICTION\n"
            "CURATIVE PETITION\nPAPER BOOK\n(FOR INDEX KINDLY SEE INSIDE)"
        ),
        2: (
            "INDEX\nSL. NO. PARTICULARS PAGE NO.\n"
            "1. List of dates\n"
            "2. Curative Petition with affidavit.\n"
            "4. ANNEXURE P-1: A copy of the Writ Petition No. 60 of 2015 "
            "dated 9.1.2015.\n"
            "5. ANNEXURE P-3: A copy of the judgment and final order dated "
            "20th February, 2015.\n"
            "7. ANNEXURE P-4: A copy of the order dated 16.03.2015 in "
            "Special Leave Petition (Civil) No. 7595 of 2015.\n"
            "8. ANNEXURE P-5: A copy of the order dated 17.08.2015 in "
            "Special Leave Petition (Civil) No. 7595 of 2015.\n"
            "9. ANNEXURE P-6: A copy of the order dated 12.01.2016 in "
            "Special Leave Petition (Civil) No. 7595 of 2015.\n"
            "10. ANNEXURE P-7: A copy of the order dated 12.04.2016 in "
            "Review Petition (Civil) No. 1512 of 2016.\n"
        ),
        3: "LIST OF DATES\n2015 Writ Petition filed",
        8: (
            "IN THE SUPREME COURT OF INDIA\nCURATIVE PETITION\n"
            "POSITION OF PARTIES\nMOST RESPECTFULLY SHOWETH:\n1. The petitioner"
        ),
        12: (
            "IN THE SUPREME COURT OF INDIA\nAFFIDAVIT\n"
            "I, the deponent, solemnly affirm"
        ),
        13: (
            "IN THE HIGH COURT OF JUDICATURE AT BOMBAY\n"
            "WRIT PETITION No. 60 of 2015\nMOST RESPECTFULLY SHOWETH"
        ),
        14: "writ petition body continuation",
        15: (
            "IN THE HIGH COURT OF JUDICATURE AT BOMBAY\n"
            "WRIT PETITION NO.60 OF 2015\n"
            "CORAM: M.S.SANKLECHA & G.S.KULKARNI, JJ.\n"
            "DATE: 20th FEBRUARY, 2015\nP.C.\n1. This petition is dismissed"
        ),
        16: "judgment body continuation",
        17: (
            "ITEM NO.37 COURT NO.5 SECTION IIIA\n"
            "S U P R E M E C O U R T O F I N D I A\n"
            "RECORD OF PROCEEDINGS\n"
            "Petition(s) for Special Leave to Appeal (C) No.7595/2015\n"
            "Date 16/03/2015 This petition was called on for hearing"
        ),
        18: (
            "ITEM NO.205 COURT NO.5 SECTION IIIA\n"
            "S U P R E M E C O U R T O F I N D I A\n"
            "RECORD OF PROCEEDINGS\n"
            "Petition(s) for Special Leave to Appeal (C) No.7595/2015\n"
            "Date 17/08/2015 This petition was called on for hearing"
        ),
        19: (
            "ITEM NO.19 COURT NO.2 SECTION IIIA\n"
            "S U P R E M E C O U R T O F I N D I A\n"
            "RECORD OF PROCEEDINGS\n"
            "Petition(s) for Special Leave to Appeal (C) No.7595/2015\n"
            "Date 12/01/2016 This petition was called on for hearing"
        ),
        20: (
            "IN THE SUPREME COURT OF INDIA\n"
            "REVIEW PETITION(CIVIL) NO.1512 OF 2016\n"
            "ORDER\nWe have perused the Review Petition"
        ),
        21: (
            "Chamber matter SECTION IIIA\n"
            "S U P R E M E C O U R T O F I N D I A\n"
            "RECORD OF PROCEEDINGS\n"
            "R. P. (C) No. 1512/2016 In SLP (C) No. 7595/2015\n"
            "Date 12/04/2016 This petition was circulated today."
        ),
    }
    for page in range(1, 22):
        texts.setdefault(page, "body")

    repaired, _ = repair_compiled_split(page_parts, texts, page_count=21)
    assert repaired[13] == ["Annexure P-1"]
    assert repaired[14] == ["Annexure P-1"]
    assert repaired[15] == ["Annexure P-3"]
    assert repaired[16] == ["Annexure P-3"]
    assert repaired[17] == ["Annexure P-4"]
    assert repaired[18] == ["Annexure P-5"]
    assert repaired[19] == ["Annexure P-6"]
    assert repaired[20] == ["Annexure P-7"]
    assert repaired[21] == ["Annexure P-7"]
    # Must not land in Impugned Order or paper-book Record of Proceedings.
    for page in range(13, 22):
        assert repaired.get(page) != ["Impugned Order"], page
        assert repaired.get(page) != ["Record of Proceedings"], page


def test_duplicate_stamp_realigns_to_index_particulars() -> None:
    """When ANNEXURE P-4 is stamped twice, Index particulars separate P-3 vs P-4."""
    page_parts = {
        1: ["Cover Page"],
        2: ["Index"],
        5: ["Main Petition"],
        8: ["Affidavit"],
    }
    texts = {
        1: (
            "IN THE SUPREME COURT OF INDIA\nPAPER BOOK\n"
            "(FOR INDEX KINDLY SEE INSIDE)"
        ),
        2: (
            "INDEX\n"
            "1. Main Petition\n"
            "2. ANNEXURE P-3: A true copy of the communication from "
            "Sub-Divisional Officer (Civil) to Deputy Commissioner dated 30.06.2022.\n"
            "3. ANNEXURE P-4: A true copy of FIR No. 177 dated 22.07.2025.\n"
        ),
        5: (
            "IN THE SUPREME COURT OF INDIA\nSPECIAL LEAVE PETITION\n"
            "POSITION OF PARTIES\nMOST RESPECTFULLY SHOWETH:\n1. The petitioner"
        ),
        8: "IN THE SUPREME COURT OF INDIA\nAFFIDAVIT\nI solemnly affirm",
        10: (
            "ANNEXURE P-4\nFrom,\nSub-Divisional Officer (Civil), Narnaul.\n"
            "To,\nDeputy Commissioner\nLetter No 3687 Dated 30-06-2022\n"
            "Subject :- Regarding recovery"
        ),
        11: "SDO communication body continuation",
        12: (
            "ANNEXURE P-4\nFirst information Report\n"
            "{U/S 154 Cr.P-C}\nFIR No.: 0177\nDated 22/07/2025\n"
        ),
        13: "FIR body continuation",
    }
    for page in range(1, 14):
        texts.setdefault(page, "body")
    repaired, _ = repair_compiled_split(page_parts, texts, page_count=13)
    assert repaired[10] == ["Annexure P-3"]
    assert repaired[11] == ["Annexure P-3"]
    assert repaired[12] == ["Annexure P-4"]
    assert repaired[13] == ["Annexure P-4"]


def test_repair_keeps_respondent_annexure_r_series() -> None:
    """Printed ANNEXURE R-n must stay Respondent series, not forced to P."""
    page_parts = {
        1: ["Index"],
        2: ["Main Petition"],
        10: ["Annexure R-1"],
        11: ["Annexure R-1"],
    }
    texts = {
        1: (
            "INDEX\n1. Special Leave Petition\n"
            "2. ANNEXURE P-1: Writ Petition\n"
            "3. ANNEXURE R-1: Counter affidavit of Respondent\n"
        ),
        2: (
            "IN THE SUPREME COURT OF INDIA\nSPECIAL LEAVE PETITION\n"
            "annexed as ANNEXURE P-1 and ANNEXURE R-1\n"
        ),
        5: "ANNEXURE P-1\nIN THE HIGH COURT\nWRIT PETITION",
        6: "writ body",
        10: "ANNEXURE R-1\nCOUNTER AFFIDAVIT ON BEHALF OF THE RESPONDENT",
        11: "respondent annexure body",
    }
    for page in range(1, 12):
        texts.setdefault(page, "body")
    repaired, _ = repair_compiled_split(page_parts, texts, page_count=11)
    assert repaired[5] == ["Annexure P-1"]
    assert repaired[10] == ["Annexure R-1"]
    assert repaired[11] == ["Annexure R-1"]
    assert repaired.get(10) != ["Annexure P-1"]


def test_contiguous_stamped_annexures_survive_incomplete_index() -> None:
    """Index stops at P-7 but printed P-11/P-12 stamps must stay attached."""
    page_parts = {
        1: ["Index"],
        2: ["Main Petition"],
        5: ["Application 1"],
    }
    texts = {
        1: (
            "INDEX\n1. Special Leave Petition\n"
            "2. ANNEXURE P-1: plaint\n"
            "3. ANNEXURE P-7: IA copy\n"
        ),
        2: (
            "IN THE SUPREME COURT OF INDIA\nSPECIAL LEAVE PETITION\n"
            "MOST RESPECTFULLY SHOWETH:\nmarked as ANNEXURE P-1 and ANNEXURE P-7\n"
            "and ANNEXURE P-8 ANNEXURE P-9 ANNEXURE P-10\n"
        ),
        5: (
            "IN THE SUPREME COURT OF INDIA\nI.A. NO. ____ of 2026\n"
            "APPLICATION FOR ADDITIONAL DOCUMENTS\n"
        ),
        6: "application body asking to place Annexures P-11 and P-12",
        7: "ANNEXURE-P11\nAdditional document start",
        8: "P-11 body",
        9: "ANNEXURE-P12\nFurther document",
        10: "P-12 body",
        3: "ANNEXURE P-1\nIN THE HIGH COURT",
        4: "ANNEXURE P-7\nIA body",
    }
    for page in range(1, 11):
        texts.setdefault(page, "body")
    repaired, _ = repair_compiled_split(page_parts, texts, page_count=10)
    assert repaired[7] == ["Annexure P-11"]
    assert repaired[8] == ["Annexure P-11"]
    assert repaired[9] == ["Annexure P-12"]
    assert repaired[5] == ["Application 1"]
    assert repaired[6] == ["Application 1"]


def test_arbitration_preface_is_not_impugned_and_a_series_stamps_stick() -> None:
    """Defect-style book: Synopsis/LOD tagged Impugned Order, A-n tagged P-n / PoA."""
    page_parts = {
        1: ["Index"],
        2: ["Impugned Order"],
        3: ["Impugned Order"],
        4: ["Impugned Order"],
        5: ["Impugned Order"],
        6: ["Impugned Order"],
        7: ["Main Petition"],
        8: ["Affidavit"],
        9: ["List of Dates & Events"],
        10: ["PoA/BR"],
        11: ["Annexure P-4"],
        12: ["Filing Memo"],
    }
    texts = {
        1: (
            "ANNEXURE D\n"
            "PROPOSED ADVOCATE'S CHECK LIST (TO BE CERTIFIED BY ADVOCATE-ON-RECORD)\n"
            "1 SLP (C) has been filed in Form No. 28 with certificate. N/A\n"
            "2 The Petition is as per the provision of Order XV Rule 1. Yes\n"
            "7. List of Dates\n"
            "particulars page no"
        ),
        2: (
            "SYNOPSIS\nThe Petitioner seeks appointment of an Arbitral Tribunal "
            "under Section 11(6) of the Arbitration and Conciliation Act, 1996."
        ),
        3: (
            "end customers and caters to a global clientele. The Respondent "
            "failed to operate the agreed rotations."
        ),
        4: "LIST OF DATES AND EVENTS\n08.03.2024 The parties executed the Agreement.",
        5: (
            "in breach of its obligations; and the Respondent failed to remedy "
            "the breach despite multiple intimations by the Petitioner."
        ),
        6: (
            "IN THE SUPREME COURT OF INDIA\nCIVIL ORIGINAL JURISDICTION\n"
            "ARBITRATION PETITION NO. OF 2025\n"
            "MOST RESPECTFULLY SHOWETH:\n1. The Petitioner is a company."
        ),
        7: "2. The Respondent is a company. Grounds. PRAYER.",
        8: (
            "IN THE SUPREME COURT OF INDIA\nAFFIDAVIT\n"
            "I, Koba Lakia, do hereby solemnly affirm and declare as under."
        ),
        9: (
            "contents of the List of Dates and the present Arbitration petition "
            "which are true and correct.\nDEPONENT\nVERIFICATION\n"
            "Verified at on this day of November, 2025."
        ),
        10: (
            "ANNEXURE A-1\nCERTIFIED TRUE COPY OF THE RESOLUTION PASSED "
            "AT THE MEETING OF THE PARTNERS\nRESOLVED THAT Mr. Koba Lakia "
            "is authorised to sign pleadings."
        ),
        11: (
            "ANNEXURE A-4\nTERMINATION and REFUND NOTICE\n"
            "To: Nipun ANAND\nPRADHAAN AIR EXPRESS PVT LTD"
        ),
        12: (
            "IN THE SUPREME COURT OF INDIA\nFILING INDEX\n"
            "1. Arbitration Petition with affidavit 5000\n"
            "2. Vakalatnama And Memo 10"
        ),
    }
    repaired, _ = repair_compiled_split(page_parts, texts, page_count=12)
    assert repaired[1] == ["Advocate's Checklist"]
    assert repaired[2] == ["Synopsis"]
    assert repaired[3] == ["Synopsis"]
    assert repaired[4] == ["List of Dates & Events"]
    assert repaired[5] == ["List of Dates & Events"]
    assert "Impugned Order" not in {name for names in repaired.values() for name in names}
    assert repaired[8] == ["Affidavit"]
    assert repaired[9] == ["Affidavit"]
    assert repaired[10] == ["Annexure A-1"]
    assert repaired[11] == ["Annexure A-4"]
    assert repaired[12] == ["Filing Memo"]


def test_index_folio_is_checked_and_body_citation_is_not_a_folio() -> None:
    """Index page 36 is the sheet numbered 36, not a 'page 36' cite in the petition."""
    page_parts = {
        1: ["Index"],
        2: ["Main Petition"],
        3: ["Annexure P-4"],
        4: ["Vakalatnama"],
        5: ["Vakalatnama"],
    }
    texts = {
        1: (
            "INDEX\n"
            "1. Annexure A-4 Reminder Notice\n"
            "2. V AKALATNAMA\n"
            "3. Payment receipt for sum of Rs. 15000/-\n"
            "36\n61-63B\n64\n"
        ),
        2: (
            "IN THE SUPREME COURT OF INDIA\n"
            "MOST RESPECTFULLY SHOWETH:\n"
            "marked as Annexure A-4 (page 36).\n"
            "2"
        ),
        3: "ANNEXURE A-4\nReminder Notice\n36",
        4: "VAKALATNAMA\nIN THE SUPREME COURT OF INDIA\n61",
        5: "SUPREME COURT OF INDIA\nCASH & ACCOUNTS\nReceived from the advocate\n64",
    }
    repaired, _ = repair_compiled_split(page_parts, texts, page_count=5)
    assert repaired[2] == ["Main Petition"]
    assert repaired[3] == ["Annexure A-4"]
    assert repaired[4] == ["Vakalatnama"]
    assert repaired[5] == ["Court Fees"]
