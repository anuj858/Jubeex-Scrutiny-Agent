"""First-30-page extract: slice, schema, and a null special category."""

from __future__ import annotations

from types import SimpleNamespace

import pymupdf
import pytest

from extraction_review.config import ExtractConfig, FrontExtractRecord, LegalExtractRecord
from extraction_review.extract_record import (
    finalize_front_extract,
    merge_dropped_party_schedule,
    supplemental_front_party_pages,
)
from extraction_review.process_file import (
    FRONT_EXTRACT_END_PAGE_LIMIT,
    FRONT_EXTRACT_PAGE_LIMIT,
    FileEvent,
    first_pages_pdf,
    front_extract_opening_pages,
)
from extraction_review.split_upload import (
    build_extract_pack_markdown,
    build_front_extract_pack_markdown,
    build_front_extract_system_prompt,
    compiled_index_reuse,
    dump_source_book,
    front_extract_configuration,
    load_source_book,
)


def _pdf_bytes(page_count: int) -> bytes:
    document = pymupdf.open()
    try:
        for _ in range(page_count):
            document.new_page()
        return document.tobytes()
    finally:
        document.close()


def _page_count(pdf_bytes: bytes) -> int:
    document = pymupdf.open(stream=pdf_bytes, filetype="pdf")
    try:
        return document.page_count
    finally:
        document.close()


def _marked_pdf(page_count: int) -> bytes:
    document = pymupdf.open()
    try:
        for number in range(1, page_count + 1):
            page = document.new_page()
            page.insert_text((72, 72), f"PAGE-{number}")
        return document.tobytes()
    finally:
        document.close()


def _page_texts(pdf_bytes: bytes) -> list[str]:
    document = pymupdf.open(stream=pdf_bytes, filetype="pdf")
    try:
        return [page.get_text() for page in document]
    finally:
        document.close()


def test_front_slice_keeps_the_opening_and_the_last_five_pages() -> None:
    sliced = first_pages_pdf(_marked_pdf(40))
    texts = _page_texts(sliced)
    assert len(texts) == FRONT_EXTRACT_PAGE_LIMIT + FRONT_EXTRACT_END_PAGE_LIMIT
    joined = "\n".join(texts)
    assert "PAGE-1" in joined
    assert "PAGE-30" in joined
    assert "PAGE-31" not in joined
    assert "PAGE-36" in joined
    assert "PAGE-40" in joined


def test_front_slice_does_not_repeat_pages_already_in_the_opening() -> None:
    sliced = first_pages_pdf(_marked_pdf(32))
    assert _page_count(sliced) == 32


def test_opening_is_30_pages_or_the_whole_shorter_file() -> None:
    assert front_extract_opening_pages(12) == 12
    assert front_extract_opening_pages(30) == 30
    assert front_extract_opening_pages(62) == 30
    assert front_extract_opening_pages(700) == 30
    assert front_extract_opening_pages(5000) == FRONT_EXTRACT_PAGE_LIMIT


def test_long_book_keeps_the_first_30_pages_and_the_closing_pages() -> None:
    opening = front_extract_opening_pages(700)
    sliced = first_pages_pdf(_marked_pdf(700), opening)
    joined = "\n".join(_page_texts(sliced))
    assert "PAGE-30" in joined
    assert "PAGE-31" not in joined
    assert "PAGE-696" in joined
    assert "PAGE-700" in joined
    assert _page_count(sliced) == opening + FRONT_EXTRACT_END_PAGE_LIMIT


def test_first_pages_pdf_keeps_a_shorter_book() -> None:
    raw = _pdf_bytes(12)
    assert first_pages_pdf(raw) == raw
    assert _page_count(first_pages_pdf(raw)) == 12


def test_special_category_sits_beside_petition_type_on_the_front_schema_only() -> None:
    legal = list(LegalExtractRecord.model_json_schema()["properties"])
    assert "special_category" not in legal
    front = front_extract_configuration(
        ExtractConfig.model_construct(data_schema={}, product_type="extract")
    )
    props = list(front["data_schema"]["properties"])
    assert props.index("special_category") == props.index("petition_type") + 1
    assert "special_category" in FrontExtractRecord.model_fields
    assert "special_category" not in LegalExtractRecord.model_fields
    prompt = build_front_extract_system_prompt()
    assert "Compiled filing" in prompt
    assert "Search every page" in prompt
    assert "petition_date" in prompt
    assert "special_category" in prompt
    assert "Look for:" in prompt
    assert "IN THE MATTER OF" in prompt
    assert "Synopsis" in prompt
    assert "Main Petition" in prompt
    assert "up to 3" in prompt
    assert "stay null" not in prompt
    assert "configuration_id" not in front
    legal_party = LegalExtractRecord.model_json_schema()["$defs"]["Party"]["properties"]
    front_party = front["data_schema"]["$defs"]["FrontParty"]["properties"]
    assert "age" not in legal_party
    assert "age" in front_party
    assert front["data_schema"]["properties"]["petitioners"]["items"]["$ref"].endswith(
        "/FrontParty"
    )


def test_front_pack_keeps_the_middle_of_the_30_page_slice() -> None:
    pages = {page: f"page {page} text" for page in range(1, 36)}
    labels = {page: ["Main Petition"] for page in range(1, 31)}
    normal = build_extract_pack_markdown(pages, labels, {"Main Petition"})
    front = build_front_extract_pack_markdown(pages)
    assert "page 20 text" not in normal
    assert "page 1 text" in front
    assert "page 20 text" in front
    assert "page 30 text" in front
    assert "## [Compiled filing] (p. 20)" in front
    assert "## [Compiled filing] (p. 31)" in front
    assert "## [Cover Page]" not in front
    assert "## [Opening pages]" not in front
    assert "page 35 text" in front
    wide = build_front_extract_pack_markdown(pages, opening_pages=75)
    assert "## [Compiled filing] (p. 31)" in wide
    assert "## [Closing pages]" not in wide


def test_missing_special_category_stays_null_and_later_pages_stay_empty() -> None:
    payload = {
        "court": "Supreme Court of India",
        "petition_type": "Special Leave Petition (Civil)",
        "cause_title": {"formatted_title": "A VS B"},
        "special_category": "N/A",
        "petitioners": [],
        "respondents": [],
        "advocates_on_record": [{"name": "Ravi"}],
        "impugned_orders": [],
        "relief_sort": "Grant leave to appeal",
        "petition_date": "01.01.2024",
        "documents": [{"name": "Main Petition"}],
        "document_counts": {"processed": 1, "failed": 0},
    }
    done = finalize_front_extract(payload)
    keys = list(done)
    assert keys.index("special_category") == keys.index("petition_type") + 1
    assert done["special_category"] is None
    assert done["petition_type"] == "Special Leave Petition (Civil)"
    assert done["relief_sort"] == "Grant leave to appeal"
    assert done["petition_date"] == "01.01.2024"
    assert done["documents"] == []
    assert done["document_counts"] == {"processed": 0, "failed": 0}
    assert done["advocates_on_record"] == [{"name": "Ravi"}]


def test_chamber_line_phone_is_stored_on_the_advocate() -> None:
    """Ph. after the chamber PIN is the advocate's phone, not part of the place."""
    chamber = (
        "Chamber: 320, New Lawyers Chamber, C.K. Daphtary Block, "
        "Supreme Court, New Delhi -110001 Ph.: 23386175"
    )
    done = finalize_front_extract(
        {
            "advocates_on_record": [
                {"name": "Ravi", "office_address": chamber, "mobile": None}
            ]
        }
    )
    advocate = done["advocates_on_record"][0]
    assert advocate["mobile"] == "23386175"
    assert "23386175" not in advocate["office_address"]
    assert "110001" in advocate["office_address"]
    assert "320" in advocate["office_address"]


def test_chamber_phone_is_read_from_the_page_when_the_address_omits_it() -> None:
    done = finalize_front_extract(
        {
            "advocates_on_record": [
                {
                    "name": "Ravi",
                    "office_address": (
                        "320, New Lawyers Chamber, C.K. Daphtary Block, "
                        "Supreme Court, New Delhi -110001"
                    ),
                }
            ]
        },
        page_markdown={
            35: (
                "Chamber: 320, New Lawyers Chamber, C.K. Daphtary Block,\n"
                "Supreme Court, New Delhi -110001 Ph.: 23386175"
            )
        },
    )
    assert done["advocates_on_record"][0]["mobile"] == "23386175"


_MEMO = """
MEMO OF PARTIES
IN THE MATTER OF:-
...Petitioner
Versus
...Respondents
Vinod Kumar Mittal
S/o Late Sh. Kanhaiya Lal Mittal
R/o -869, First Floor, New Rajinder
Nagar, Delhi-110060.
1. State NCT of Delhi
Through SHO PS Economic Offences Wing, New Delhi-110001.
2. Pawan Sachdeva
"""


def test_front_parties_come_from_the_memo_not_the_checklist() -> None:
    payload = {
        "petition_type": "SPECIAL LEAVE PETITION (CRL.)",
        "cause_title": {
            "formatted_title": "VINOD KUMAR MITTAL and Ors. VS STATE NCT OF DELHI and Ors."
        },
        "petitioners": [
            {"name": "1. VINOD KUMAR MITTAL"},
            {
                "name": (
                    "The Petition is barred by time and there is delay of days "
                    "in filing the same"
                )
            },
            {"name": "(a). Main Category classification: 14"},
            {"name": "Criminal Matters: Yes"},
            {"name": "Special Category (First Petitioner/appellant only): NA"},
        ],
        "respondents": [
            {"name": "STATE NCT OF DELHI"},
            {"name": "Learned counsel appearing on behalf of the petitioner submits that"},
            {"name": "The petition is dismissed and disposed of"},
            {"name": "Pending applications, if any, also stand disposed of"},
        ],
    }
    done = finalize_front_extract(payload, page_markdown={338: _MEMO})
    assert [party["name"] for party in done["petitioners"]] == ["Vinod Kumar Mittal"]
    assert [party["name"] for party in done["respondents"]] == [
        "State NCT of Delhi",
        "Pawan Sachdeva",
    ]
    assert done["cause_title"]["formatted_title"] == (
        "Vinod Kumar Mittal VS State NCT of Delhi and Anr."
    )


def test_front_parties_extracted_from_multipage_position_of_parties_table() -> None:
    p27 = """
    <table><tr><th></th><th colspan="3"><u>POSITION OF PARTIES</u></th></tr>
    <tr><th><u>IN THE MATTER</u><br /><u>OF</u></th><th><u>TRIAL COURT</u></th><th><u>HIGH COURT</u></th><th><u>IN THIS COURT</u></th></tr>
    <tr><td>1. Sathish<br /><br />S/o Samuthirapndi<br />South Mavadi,<br />Mavadi post<br />Tirunelveli District<br />Tamil Nadu.<br /><br /><strong>Presently residing at::</strong><br />435, Keela Mavadi,<br />Malaiadiputhur<br />Panchayat,<br />Tirunelveli, T.N.</td><td>Accused No. 1</td><td>Respondent No<br /><br />2</td><td>Appellant</td></tr>
    <tr><td></td><td><u>VERSUS</u></td><td></td><td></td></tr></table>
    """
    p28 = """
    <table><tr><td>1. The state rep by inspector of police, Eruvadi Police station, Tirunelveli District Tamilnadu state Crime No.156/2012</td><td>Complainant</td><td>1<sup>st</sup> respondent</td><td>Contesting respondent No.2</td></tr>
    <tr><td>2. Chandra W/o Ramar South Mavadi Mavadi Post Tirunelveli District Defecto THE HUMBLE APPEAL OF</td><td>Defecto Complainant</td><td>Appellant</td><td>Contesting Respondent No.1</td></tr></table>
    **TO,**
    **THE HON'BLE CHIEF JUSTICE OF INDIA**
    **MOST RESPECTFULLY SHOWETH:**
    """
    payload = {
        "petition_type": "CRIMINAL APPEAL",
        "cause_title": {"formatted_title": "Sathish VS State rep. by The Inspector of police"},
        "petitioners": [{"name": "Sathish"}],
        "respondents": [{"name": "State rep. by The Inspector of police"}],
    }
    done = finalize_front_extract(payload, page_markdown={27: p27, 28: p28})
    assert len(done["petitioners"]) == 1
    pet = done["petitioners"][0]
    assert pet["name"] == "Sathish"
    assert pet["relation"] == "S/o"
    assert pet["guardian"] == "Samuthirapndi"
    assert "South Mavadi" in pet["address"]
    assert "435" in pet["address"]

    assert len(done["respondents"]) == 2
    r1 = done["respondents"][0]
    assert "state rep" in r1["name"].lower()
    assert "inspector of police" in r1["name"].lower()
    assert "Eruvadi" in r1["address"]
    assert r1["acting_through"] == "inspector of police"

    r2 = done["respondents"][1]
    assert r2["name"] == "Chandra"
    assert r2["relation"] == "W/o"
    assert r2["guardian"] == "Ramar"
    assert "South Mavadi" in r2["address"]
    assert r2["address"].rstrip(" .").endswith("Tirunelveli District")
    assert "HUMBLE" not in r2["address"]
    assert done["cause_title"]["formatted_title"] == "Sathish VS The state rep by inspector of police and Anr."


def test_front_party_name_drops_parse_markup_and_the_list_number() -> None:
    done = finalize_front_extract(
        {
            "petition_type": "SPECIAL LEAVE PETITION (CIVIL)",
            "cause_title": {
                "formatted_title": (
                    "**2. JITENDER KUMAR PURI** VS SUNDER LAL GUPTA and Ors."
                )
            },
            "petitioners": [{"name": "**2. JITENDER KUMAR PURI**"}],
            "respondents": [
                {"name": "SUNDER LAL GUPTA"},
                {"name": "**3. ANOTHER PERSON**"},
                {"name": "4. THIRD PERSON"},
            ],
        }
    )
    assert done["petitioners"][0]["name"] == "JITENDER KUMAR PURI"
    assert [party["name"] for party in done["respondents"]] == [
        "SUNDER LAL GUPTA",
        "ANOTHER PERSON",
        "THIRD PERSON",
    ]
    assert done["cause_title"]["formatted_title"] == (
        "JITENDER KUMAR PURI VS SUNDER LAL GUPTA and Ors."
    )


def test_wrapped_party_block_keeps_the_second_petitioner_and_the_address() -> None:
    page = """
    IN THE SUPREME COURT OF INDIA
    SPECIAL LEAVE PETITION (C) NO. OF 2026
    BETWEEN:
    Anoop
    Kumar
    @
    Anup
    Kumar
    S/o Ragya Parsad
    R/o 28 behind Lawrence road
    hotel, Lawrence Road, Amritsar.
    Rajeev Lakhanpal
    S/o Naranjan Dass
    R/o 28 behind Lawrence road
    hotel, Lawrence Road, Amritsar.
    Petitioner No. 1
    Petitioner No. 2
    VERSUS
    Gautam Khanna
    Registered Address at
    IFFCO House, 2 & 3 Floor
    34, Nehru Place
    New Delhi- 110019
    Respondent No. 1
    """
    high_court = """
    IN THE HON'BLE HIGH COURT OF PUNJAB AND HARYANA AT CHANDIGARH
    MEMO OF PARTIES
    1. Anoop Kumar @ Anup Kumar, aged 56 years, son of Ragya Parsad resident of somewhere
    VERSUS
    1. Mr Gautam Khanna resident of House No. A-432 Ranjeet Avenue Amritsar.
    """
    done = finalize_front_extract(
        {
            "petition_type": "SPECIAL LEAVE PETITION (CIVIL)",
            "cause_title": {"formatted_title": "Anoop Kumar @ Anup Kumar & Anr. VS Gautam Khanna"},
            "petitioners": [{"name": "Anoop Kumar @ Anup Kumar"}],
            "respondents": [{"name": "Gautam Khanna"}],
        },
        page_markdown={1: "Anoop Kumar & Anr.\nVersus\nGautam Khanna", 20: page, 33: high_court},
    )
    assert [party["name"] for party in done["petitioners"]] == [
        "Anoop Kumar @ Anup Kumar",
        "Rajeev Lakhanpal",
    ]
    assert done["petitioners"][1]["relation"] == "S/o"
    assert "Lawrence Road" in done["petitioners"][1]["address"]
    assert 20 in done["petitioners"][1]["source_pages"]
    assert "Rajeev" not in done["petitioners"][0]["address"]
    assert "Nehru Place" in done["respondents"][0]["address"]
    assert "110019" in done["respondents"][0]["address"]
    assert 20 in done["respondents"][0]["source_pages"]


def test_same_person_spelled_two_ways_is_one_petitioner() -> None:
    done = finalize_front_extract(
        {
            "petition_type": "SPECIAL LEAVE PETITION (CIVIL)",
            "cause_title": {"main_petitioner": "Anoop Kumar @ Anup Kumar"},
            "petitioners": [
                {
                    "name": "Anoop Kumar @ Anup Kumar",
                    "relation": "S/o",
                    "guardian": "Ragya Parsad",
                },
                {
                    "name": "Rajiv Lakhanpal",
                    "relation": "son of",
                    "guardian": "Naranjan Dass",
                    "address": "a shop at Kucha Tai Khana",
                },
                {
                    "name": "Rajeev Lakhanpal",
                    "relation": "S/o",
                    "guardian": "Naranjan Dass",
                    "address": "R/o 28 behind Lawrence Road, Amritsar",
                    "source_pages": [20],
                },
            ],
            "respondents": [
                {
                    "name": "Gautam Khanna",
                    "address": "IFFCO House, 2 &x26; 3 Floor, Nehru Place THE HON'BLE THE",
                },
                {"name": "2025 passed by the", "source_pages": [3]},
            ],
        }
    )
    assert [party["name"] for party in done["petitioners"]] == [
        "Anoop Kumar @ Anup Kumar",
        "Rajeev Lakhanpal",
    ]
    assert done["petitioners"][1]["relation"] == "S/o"
    assert [party["name"] for party in done["respondents"]] == ["Gautam Khanna"]
    assert done["respondents"][0]["address"] == (
        "IFFCO House, 2 & 3 Floor, Nehru Place"
    )
    assert done["cause_title"]["formatted_title"] == (
        "Anoop Kumar @ Anup Kumar and Anr. VS Gautam Khanna"
    )


def test_petition_date_after_prayer_skips_the_page_number() -> None:
    pages = {
        5: "OFFICE REPORT ON LIMITATION\nDated: 01.01.2020",
        27: (
            "MAIN PRAYER:\n"
            "In the circumstances stated above, the Petitioners pray that this\n"
            "Hon'ble Court may be pleased to grant leave.\n"
            "Dated:\n"
            "14\n"
            "02.03.2026\n"
        ),
    }
    done = finalize_front_extract(
        {"petition_type": "SPECIAL LEAVE PETITION (CIVIL)", "petition_date": None},
        page_markdown=pages,
    )
    assert done["petition_date"] == "02.03.2026"


def test_printed_special_category_is_kept() -> None:
    done = finalize_front_extract(
        {
            "petition_type": "Writ Petition (Civil)",
            "special_category": "Tax Matters",
        }
    )
    assert done["special_category"] == "Tax Matters"


def test_unknown_special_category_and_blank_markers_stay_null() -> None:
    catch_all = finalize_front_extract(
        {
            "special_category": (
                "Any matter arising out of Civil suit/proceedings, "
                "which is not covered in any of the above categories"
            )
        }
    )
    blank = finalize_front_extract({"special_category": "N/A"})
    assert catch_all["special_category"] is None
    assert blank["special_category"] is None


def test_motor_vehicles_act_alias_and_three_category_cap() -> None:
    alias = finalize_front_extract(
        {"special_category": "Motor Vehicles Act, 1988"}
    )
    capped = finalize_front_extract(
        {
            "special_category": (
                "Tax Matters, Eviction Matters, PIL, "
                "Bail Applications/Bail Matters"
            )
        }
    )
    assert alias["special_category"] == "Motor Vehicle Act"
    assert capped["special_category"] == "Tax Matters, Eviction Matters, PIL"


def test_listing_proforma_tax_row_is_not_a_special_category() -> None:
    done = finalize_front_extract(
        {"special_category": None},
        page_markdown={
            3: "9. Tax Matters: State the tax effect: N.A.\n10. Special Category (Petitioner only): N.A."
        },
    )
    assert done["special_category"] is None


def test_plain_petition_has_no_special_category() -> None:
    """A review or transfer petition with no named category stays null."""
    done = finalize_front_extract(
        {
            "petition_type": "REVIEW PETITION (CIVIL)",
            "special_category": "Tax Matters",
        },
        page_markdown={
            8: (
                "SYNOPSIS\n"
                "The petitioners seek review of the judgment dated 09.03.2022.\n"
            ),
            17: (
                "REVIEW PETITION (CIVIL) NO. 572 OF 2022\n"
                "PETITION UNDER ARTICLE 137 OF THE CONSTITUTION OF INDIA\n"
                "MOST RESPECTFULLY SHOWETH:\n"
                "1. That the petitioners are residents of Delhi.\n"
            ),
            18: (
                "TRANSFER PETITION (CIVIL) NO. 12 OF 2024\n"
                "MAIN PETITION\n"
                "The petitioner seeks transfer of the suit.\n"
            ),
        },
    )
    assert done["special_category"] is None


def test_bail_application_in_the_impugned_order_is_a_special_category() -> None:
    done = finalize_front_extract(
        {
            "petition_type": "SPECIAL LEAVE PETITION (CRL.)",
            "special_category": None,
            "impugned_orders": [
                {
                    "case_number": "Misc. Criminal case No. 28663 of 2026",
                    "raw_text": (
                        "Arising from the impugned Judgment and final order dated "
                        "03.08.2026 passed by the Hon'ble High Court of Madhya Pradesh "
                        "at Jabalpur in Bail Application No.- Misc. Criminal case No. "
                        "28663 of 2026"
                    ),
                }
            ],
        },
        page_markdown={
            4: (
                "SYNOPSIS\n"
                "The petitioner seeks leave against the order refusing bail.\n"
                "In a petition (PIL) filed under clause (d) of Rule 12(1), "
                "the petitioner has disclosed the facts in the Public Interest Litigation. | NA\n"
            ),
            9: (
                "LISTING PROFORMA\n"
                "Whether any bail application was preferred earlier and decision thereupon: N.A.\n"
            ),
        },
    )
    assert done["special_category"] == "Bail Applications/Bail Matters"


def test_public_interest_litigation_on_the_main_petition_is_pil() -> None:
    done = finalize_front_extract(
        {"petition_type": "WRIT PETITION (CIVIL)", "special_category": None},
        page_markdown={
            12: (
                "IN THE SUPREME COURT OF INDIA\n"
                "CIVIL WRIT PETITION NO. OF 2024\n"
                "(UNDER ARTICLE 32 OF THE CONSTITUTION OF INDIA)\n"
                "(PUBLIC INTEREST LITIGATION)\n"
                "IN THE MATTER OF:\n"
                "Alakh Pandey\n"
            ),
        },
    )
    assert done["special_category"] == "PIL"


def test_aadhaar_line_is_not_the_petitioner_name() -> None:
    done = finalize_front_extract(
        {
            "petition_type": "WRIT PETITION (CIVIL)",
            "special_category": None,
            "cause_title": {"formatted_title": "Aadhaar No VS National Testing Agency and Anr."},
            "petitioners": [{"name": "Aadhaar No"}],
            "respondents": [{"name": "National Testing Agency"}],
        },
        page_markdown={
            12: """
            IN THE SUPREME COURT OF INDIA
            CIVIL WRIT PETITION NO. OF 2024
            (UNDER ARTICLE 32 OF THE CONSTITUTION OF INDIA)
            (PUBLIC INTEREST LITIGATION)
            IN THE MATTER OF:
            Alakh Pandey
            Aadhaar No. 8562 7987 8438
            S/o Satish Kumar Pandey
            Aged About 32 years
            R/o R K 27, Kalindipuram, Allahabad,
            Uttar Pradesh- 211011
            Email: neet24@singhania.com
            Mob: 9891829172
            ..........Petitioner
            Versus
            1. National Testing Agency
            Through Director General
            First Floor, NSIC-MDBP Building,
            Okhla Industrial Estate, New Delhi,
            Delhi 110020.
            2. Ministry of Education
            Through Secretary,
            Dr Rajendra Prasad Rd, Rajpath Area,
            Windsor Place, New Delhi, Delhi 110001
            ..........Respondents
            """,
        },
    )
    assert [party["name"] for party in done["petitioners"]] == ["Alakh Pandey"]
    petitioner = done["petitioners"][0]
    assert petitioner["relation"] == "S/o"
    assert petitioner["guardian"] == "Satish Kumar Pandey"
    assert petitioner["age"] == "32 years"
    assert "Kalindipuram" in petitioner["address"]
    assert "Aadhaar" not in petitioner["address"]
    assert "neet24" not in petitioner["address"].lower()
    assert done["cause_title"]["formatted_title"].startswith("Alakh Pandey VS")
    assert [party["name"] for party in done["respondents"]] == [
        "National Testing Agency",
        "Ministry of Education",
    ]


def test_synopsis_tax_matters_fills_an_empty_special_category() -> None:
    done = finalize_front_extract(
        {"special_category": "N/A"},
        page_markdown={
            2: "Index\nSynopsis .......... 4\nMain Petition .......... 8",
            4: "SYNOPSIS\nThis Special Leave Petition arises out of tax matters.",
        },
    )
    assert done["special_category"] == "Tax Matters"


@pytest.mark.asyncio
async def test_extract_front_parses_the_compiled_pdf_once(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from extraction_review.process_file import _run_extract_front

    captured: dict[str, object] = {}
    parsed: dict[str, object] = {}

    async def fake_load(*_args: object, **_kwargs: object) -> bytes:
        return _pdf_bytes(45)

    async def fake_parse(_client: object, **kwargs: object) -> tuple:
        parsed.update(kwargs)
        return (
            "pjb-book",
            {1: "page one", 45: "last page"},
            {1: {"items": []}},
            {1: {"words": []}},
            {"product": "parse", "job_id": "pjb-book", "credits": 1.0},
        )

    async def fake_extract(_ctx: object, **kwargs: object) -> str:
        captured.update(kwargs)
        return "agd-front"

    monkeypatch.setattr(
        "extraction_review.process_file._load_bundle_pdf",
        fake_load,
    )
    monkeypatch.setattr(
        "extraction_review.process_file._parse_compiled_book",
        fake_parse,
    )
    monkeypatch.setattr(
        "extraction_review.process_file._extract_sliced_parts",
        fake_extract,
    )
    event = FileEvent(
        job_type="extract_front",
        filing_type="SLP_CIVIL",
        special_category="Tax Matters",
        documents=[
            {
                "name": "compiled.pdf",
                "file_id": "dfl-full",
                "download_url": "https://example.com/compiled.pdf",
            }
        ],
    )
    result = await _run_extract_front(
        event,
        ctx=SimpleNamespace(write_event_to_stream=lambda _event: None),
        client=object(),
    )
    assert parsed["file_id"] == "dfl-full"
    assert captured["front_extract"] is True
    assert captured["skip_index"] is True
    assert captured["parse_scope"] == "supplied"
    assert captured["special_category"] is None
    assert captured["court"] is None
    assert captured["source_parse_job_id"] == "pjb-book"
    assert captured["reuse_parse_job_ids"] == {"petition": "pjb-book"}
    pages = captured["reuse_pages_by_slot"]
    assert pages["petition"]["1"] == "page one"
    assert pages["petition"]["45"] == "last page"
    parts = captured["parts"]
    assert parts[0].file_id == "dfl-full"
    assert result.agent_data_id == "agd-front"
    assert result.source_parse_job_id == "pjb-book"
    assert result.parsed_slots == ["fullpetition"]
    assert len(event.documents) == 1
    assert result.job_type == "extract_front"


def test_defect_file_001_parties_come_from_the_current_court_column() -> None:
    """Pages 15-18 are the petition opening. The High Court memo is not the party list."""
    opening = 'IN THE SUPREME COURT OF INDIA \n[ORDER XXI RULE 3(1)(A)] \nCIVIL APPELLATE JURISDICTION \n(Under Article 136 of Constitution of India) \nSPECIAL LEAVE PETITION (Civil)NO.         \n OF 2025 \nWITH PRAYER FOR INTERIM RELIEF \nIN THE \nMATTER OF \nBEFORE HIGH \nCOURT \nBEFORE \nSUPREME \nCOURT \n1.Col. \nPawan \nKumar \nGopinath \nSharma, \ns/o \nGopinath \nSharma, \nIndian Jew\nRespondent No.  \nPetitioner  \nVersus \n1. K.D.R. Farms Co.\nop Housing Society\nLtd., Gut No. 79,\nSharnaur,\nAurangabad,\nDistrict\nAurangabad,\nthrough \nits\nChairman\nPetitioner No.  \nRespondent No.  \n2. Rajendra\nMadhukarrao\nKulkarni, age 52\nyears, occupation\nBusiness, \nr/o\nK.D.R. Farms Co.\nop. \nHousing\nSociety Ltd. Gut\nNo. 79, Sharnuar,\nAurangabad\nPetitioner No.  \nRespondent No. \n13\n\n3. The \nState \nof\nMaharashtra,\nthrough \nits \nSecretary, \nDepartment \nof \nCooperation \nand \nTextile, \nMantralaya, \nMumbai \nRespondent No.  \nRespondent No. \n4. The\nHon’ble\nMinister Cooperation,\nthrough Principal\nSecretary (Cooperation\nand Marketing),\nGovernment of\nMaharashtra,\nMantralaya, Mumbai,\nRespondent No.  \nRespondent No. \n5. The \nDivisional\nJoint \nRegistrar,\nCooperative\nSocieties,\nAurangabad\nDivision, \nhaving\nOffice at Adalat\nRoad, District and\ncity Aurangabad\nRespondent No. \nRespondent No.  \n6. The Taluka Dy.\nRegistrar,\nCooperative\nSocieties,\nAurangabad,\nTaluka and District\nAurangabad\nRespondent No. \nRespondent No.  \n7. Raibham \ns/o\nMithuji \nMohade,\nage \n68 \nyears,\nRespondent No.  \nRespondent No. \n14\n\noccupation Retired, \nr/o Plot no. \n25, \nK.D.R. Farms,\nSharnaur,\nAurangabad\n8. Colonel Vitthal s/o\nPitamber \nChitte,\nAge \n55 \nyears,\noccupation\nService, r/o Plot\nNo. \n35, \nK.D.R.\nFarms, Sharnapur,\nDistrict and city\nAurangabad\nRespondent No. \nRespondent No.  \n9. Mrs. Chandrakala,\nw.o \nKisanrao\nDeshmukh, age 61\nyears, occupation\nService, r/o Plot\nNo. \n59, \nK.D.R.\nFarms, \nSharnaur,\nAurangabad,\nDistrict\nAurangabad\nRespondent No.  \nRespondent No. \n10. Mr. \nPratap, \ns/o\nHaridas \nDhopte,\nRespondent No. \nRespondent No.  \n15\n\nAurangabad – 431 \n210 \nTO, \nTHE HON’BLE THE CHIEF JUSTICE OF INDIA AND \nHIS COMPANION JUSTICES OF THE HON’BLE \nSUPREME COURT OF INDIA. \nMOST RESPECTFULLY SHOWETH:'
    high_court_memo = 'HON’BLE HIGH COURT OF JUDICATURE OF BOMBAY \nBENCH AT AURANGABAD \nWRIT PETITION NO. 12255 OF 2015 \nMEMO OF PARTIES BEFORE THE HIGH COURT \n1. K.D.R. Farms Co. op Housing Society Ltd.,\nGut No. 79, Sharnaur, Aurangabad,\nDistrict Aurangabad, through its Chairman.\n…PETITIONER 1 \n2. Rajendra Madhukarrao Kulkarni, age 52 years,\noccupation Business, r/o K.D.R. Farms\nCo. op. Housing Society Ltd.\nGut No. 79, Sharnuar, Aurangabad\n …Petitioner 2 \nVERSUS \n1. The State of Maharashtra,\nthrough its Secretary,\nDepartment of Cooperation and Textile,\nMantralaya, Mumbai\n…Respondent No. 1 \n2. The Hon’ble Minister for Cooperation,\nGovernment of Maharashtra,\nMantralaya, Mumbai\n…Respondent No.2 \n3. The Divisional Joint Registrar,\nCooperative Societies,\nAurangabad Division,\nhaving Office at Adalat Road,\nDistrict and city Aurangabad\n…Respondent No. 3 \n4. The Taluka Dy. Registrar,\nCooperative Societies, Aurangabad,\nTaluka and District Aurangabad           …Respondent No. 4 \n5. Col. Pawan Kumar Sharma,\n105\n\ns/o Gopinath Sharma,  \nr/o Plot No. 6, K.D.R. Farms, \n Sharnapur \n             …Respondent No. 5 \n6. Raibham\ns/o Mithuji Mohade, age 68 years,\noccupation Retired,\nr/o Plot no. 25, K.D.R. Farms,\nSharnaur, Aurangabad\n …Respondent No. 6 \n7. Colonel Vitthal\ns/o Pitamber Chitte,\nAge 55 years, occupation Service,\nr/o Plot No. 35, K.D.R. Farms,\nSharnapur, District and city Aurangabad       …Respondent No. 7\n8. Mrs. Chandrakala,\nw.o Kisanrao Deshmukh, age 61 years,\noccupation Service,\nr/o Plot No. 59, K.D.R. Farms,\nSharnaur, Aurangabad, District Aurangabad       …Respondent No.8\n9. Mr. Pratap,\ns/o Haridas Dhopte, age 42 years,\noccupation Service,\nc/o Excelize Software Pvt. Ltd.,\nT-25, Block C and D, M.I.D.C. Chikalthana,       …Respondent No. 9\n106\n'
    done = finalize_front_extract(
        {
            "petitioners": [
                {"serial": 1, "name": "K.D.R. Farms Co-op Housing Society Ltd"},
                {"serial": 2, "name": "Rajendra Madhukarrao Kulkarni"},
            ],
            "respondents": [{"serial": 1, "name": "Col. Pawan Kumar Sharma"}],
            "cause_title": {},
        },
        page_markdown={15: opening, 107: high_court_memo},
    )
    petitioners = done["petitioners"]
    respondents = done["respondents"]
    assert [party["name"] for party in petitioners] == ["Col. Pawan Kumar Gopinath Sharma"]
    assert petitioners[0]["relation"] == "S/o"
    assert petitioners[0]["guardian"] == "Gopinath Sharma"
    assert petitioners[0]["occupation"] == "Indian Jew"
    assert petitioners[0]["address"] is None
    assert "Plot" not in str(petitioners[0].get("address"))
    assert len(respondents) == 10
    names = [party["name"] for party in respondents]
    assert names[0].startswith("K.D.R. Farms")
    assert "Rajendra Madhukarrao Kulkarni" in names[1]
    assert respondents[1]["occupation"] == "Business"
    assert respondents[1]["age"] == "52 years"
    assert "K.D.R. Farms" in respondents[1]["address"]
    assert not respondents[1]["address"].lower().startswith("r/o")
    assert "State of Maharashtra" in names[2]
    assert "Minister" in names[3]
    assert "Divisional Joint Registrar" in names[4]
    assert "Taluka" in names[5]
    assert names[6] == "Raibham"
    assert "Vitthal" in names[7]
    assert "Chandrakala" in names[8]
    assert names[9] == "Mr. Pratap"
    for party in respondents:
        assert party.get("relation") or party.get("address")
    assert respondents[6]["relation"] == "S/o"
    assert respondents[6]["guardian"] == "Mithuji Mohade"
    assert respondents[6]["occupation"] == "Retired"
    assert respondents[6]["age"] == "68 years"
    assert respondents[7]["age"] == "55 years"
    assert "Plot" in respondents[6]["address"]
    assert "431" in respondents[9]["address"]
    assert all(15 in party["source_pages"] for party in [*petitioners, *respondents])
    assert "K.D.R. Farms" not in petitioners[0]["name"]
    assert "Rajendra" not in petitioners[0]["name"]

    filled = merge_dropped_party_schedule(
        {15: "not a party schedule " * 400, 107: high_court_memo},
        {15: opening},
    )
    assert "BEFORE" in filled[15] and "SUPREME" in filled[15]
    restored = finalize_front_extract(
        {
            "petitioners": [
                {"serial": 1, "name": "K.D.R. Farms Co-op Housing Society Ltd"}
            ],
            "respondents": [{"serial": 1, "name": "Col. Pawan Kumar Sharma"}],
            "cause_title": {},
        },
        page_markdown=filled,
    )
    assert len(restored["petitioners"]) == 1
    assert len(restored["respondents"]) == 10
    assert restored["petitioners"][0]["name"] == "Col. Pawan Kumar Gopinath Sharma"


def test_humble_petition_heading_is_not_a_second_respondent() -> None:
    """The salutation after the party table is not another respondent."""
    page = """
    Position of parties
    <table><tr><th></th><th>Before Hon’ble High Court</th><th>Before this Hon’ble Court</th></tr>
    <tr><td><strong>Mubarak Ali</strong><br /><strong>S/o Athar Hussain</strong><br /><strong>R/o Mahuadih</strong><br /><strong>Police Station-Shankargarh</strong><br /><strong>District-Balrampur-Ramanujganj</strong><br /><strong>Chhattisgarh</strong></td><td><strong>Petitioner</strong></td><td><strong>Petitioner</strong></td></tr>
    <tr><td><strong>Versus</strong></td><td></td><td></td></tr>
    <tr><td><strong>State of Chhattisgarh</strong><br /><strong>Through Station House Officer</strong><br /><strong>Police Station Shankargarh</strong><br /><strong>District-Balrampur-Ramanujganj</strong><br /><strong>Chhattisgarh</strong></td><td><strong>Respondent</strong></td><td><strong>Respondent</strong></td></tr></table>
    To,
    Hon’ble the Chief Justice of India
    and his Hon’ble Companion Justices
    of The Hon’ble Supreme Court of India
    The humble Petition of the Petitioner above named:
    """
    done = finalize_front_extract(
        {
            "petition_type": "SPECIAL LEAVE PETITION (CRIMINAL)",
            "cause_title": {
                "formatted_title": "Mubarak Ali VS State of Chhattisgarh and Anr."
            },
            "petitioners": [
                {
                    "name": "Mubarak Ali",
                    "address": (
                        "R/o Mahuadih Police Station-Shankargarh "
                        "District-Balrampur-Ramanujganj Chhattisgarh"
                    ),
                }
            ],
            "respondents": [
                {
                    "name": "State of Chhattisgarh",
                    "address": (
                        "Through Station House Officer Police Station Shankargarh "
                        "District-Balrampur-Ramanujganj Chhattisgarh Hon’ble the"
                    ),
                },
                {"name": "The humble Petition of the"},
            ],
        },
        page_markdown={22: page},
    )
    assert [party["name"] for party in done["petitioners"]] == ["Mubarak Ali"]
    petitioner_address = done["petitioners"][0]["address"]
    assert petitioner_address.startswith("Mahuadih")
    assert not petitioner_address.lower().startswith("r/o")
    assert [party["name"] for party in done["respondents"]] == ["State of Chhattisgarh"]
    address = done["respondents"][0]["address"]
    assert address.startswith("Police Station Shankargarh")
    assert "Chhattisgarh" in address
    assert "through" not in address.lower()
    assert "Hon" not in address
    assert done["respondents"][0]["acting_through"] == "Station House Officer"
    assert done["cause_title"]["formatted_title"] == (
        "Mubarak Ali VS State of Chhattisgarh"
    )


def test_parvez_party_page_has_one_petitioner_and_one_respondent() -> None:
    """Page 37 is the current-court column. A signature line is not a respondent."""
    page = """
    IN THE SUPREME COURT OF INDIA
    CRIMINAL APPELLATE JURISDICTION
    SPECIAL LEAVE PETITION (CRL.) NO. OF 2026
    IN THE MATTER OF:
    BETWEEN
    POSITION OF THE PARTIES
    In the Ld. Sessions Court
    In the Hon'ble High Court
    In this Hon'ble Court
    Parwej Alam @ Parwej Alam
    S/o Nizamuddin Ansari
    R/o VPO Tikaita, PS Turkaulia, Purba Champaran, Bihar
    Applicant
    Petitioner
    Petitioner
    Versus
    STATE OF BIHAR
    Through
    SHO, P.S- Motihari Cyber, Dist. East Champaran, Bihar
    Respondent Respondent
    Respondent
    TO,
    HONBLE THE CHIEF JUSTICE OF INDIA
    AND HIS COMPANION JUDGES OF THE
    HON'BLE SUPREME COURT OF INDIA
    THE SPECIAL LEAVE PETITION OF
    """
    merged = merge_dropped_party_schedule({1: "cover"}, {37: page})
    assert "POSITION OF THE PARTIES" in merged[37]
    done = finalize_front_extract(
        {
            "petition_type": "SPECIAL LEAVE PETITION (CRL)",
            "cause_title": {
                "formatted_title": "PARWEJ ALAM @ PARWEZ ALAM VS STATE OF BIHAR and Ors.",
                "main_petitioner": "PARWEJ ALAM @ PARWEZ ALAM",
                "main_respondent": "STATE OF BIHAR",
            },
            "petitioners": [{"name": "PARWEJ ALAM @ PARWEZ ALAM", "address": None}],
            "respondents": [
                {"name": "STATE OF BIHAR"},
                {"name": "signature: AOR for the", "address": "AOR for the"},
                {"name": "PARWEJ ALAM @ PARWEZ ALAM"},
            ],
            "inconsistencies": {
                "items": [
                    {
                        "id": "1",
                        "label": "Missing acting through",
                        "raw_text": "ORGANIZATION respondent 'STATE OF BIHAR' has no acting_through.",
                    }
                ]
            },
        },
        page_markdown=merged,
    )
    assert [party["name"] for party in done["petitioners"]] == ["Parwej Alam @ Parwej Alam"]
    petitioner = done["petitioners"][0]
    assert petitioner["relation"] == "S/o"
    assert petitioner["guardian"] == "Nizamuddin Ansari"
    assert petitioner["address"] == "VPO Tikaita, PS Turkaulia, Purba Champaran, Bihar"
    assert not petitioner["address"].lower().startswith("r/o")
    assert [party["name"] for party in done["respondents"]] == ["STATE OF BIHAR"]
    respondent = done["respondents"][0]
    assert respondent["acting_through"] == "SHO"
    assert respondent["address"] == "P.S- Motihari Cyber, Dist. East Champaran, Bihar"
    assert "through" not in respondent["address"].lower()
    assert done["cause_title"]["formatted_title"] == "Parwej Alam @ Parwej Alam VS STATE OF BIHAR"
    assert done["inconsistencies"]["items"] == []


def test_review_memo_keeps_every_numbered_respondent_and_the_second_address() -> None:
    """Presently at and Second Address stay with the party. No number is skipped."""
    page_18 = """
    IN THE SUPREME COURT OF INDIA
    IN THE MATTER OF:
    1. Sh. Anang Pal Singh, S/o Shri Ram Narain,
    R/o. A-19, New Police Lines, Kingsway Camp,
    Delhi-110009.
    Presently at:
    H.No.175, Third Floor, Sharda Niketan
    Pitampura, Delhi-110034.
    2. Shri Rajinder Singh, S/o. Shri Ram Narain,
    R/o. B-1, Tehsil Road,
    Najafgarh, New Delhi-110043.
    3. Smt. Shakuntala,
    W/o. Shri Anang Pal Singh,
    R/o. A-19, New Police Lines,
    Kingsway Camp, Delhi-110034.
    Presently at:
    H.No.175, Third Floor, Sharda Niketan, Pitampura, Delhi119934.
    Versus
    1. Union of India,
    Through its Secretary,
    Ministry of Urban Development, New Delhi.
    """
    page_19 = """
    2. Govt. of N.C.T. Delhi,
    Through its Secretary,
    Land Building Department, Vikas Bhawan, New Delhi.
    3. Land Acquisition Collector,
    Delhi, Saraswati Vihar, Kanjhawala, Delhi.
    4. Director of Education, Govt. of NCT, Old secretariat,
    Delhi.
    5. General Manager, Indian Oil Corporation, World Trade Centre,
    Babar Road, New Delhi.
    6. Secretary, Ministry of Petroleum & Natural Gas, Government
    of India, New Delhi.
    7. Sh. K.S.Mahra, Commissioner of MCD, Town Hall, Chandi
    Chowk, Delhi-110006
    8. Master Azad Singh, Counsellor, MCD, S/o Sh. Meer Singh,
    R/o House No.N.S.23, Ground Floor, Mianwali Nagar, Delhi-110087;
    Second Address:
    R/o House No. 555, Vilage & Post Office Mundka, Delhi-110041
    """
    done = finalize_front_extract(
        {
            "petition_type": "REVIEW PETITION (CIVIL)",
            "cause_title": {"formatted_title": "Anang Pal Singh and Ors. VS Union of India and Ors."},
            "petitioners": [{"name": "Sh. Anang Pal Singh"}],
            "respondents": [
                {"name": "Union of India"},
                {"name": "Second Address:"},
            ],
        },
        page_markdown={18: page_18, 19: page_19},
    )
    assert [party["name"] for party in done["petitioners"]] == [
        "Sh. Anang Pal Singh",
        "Shri Rajinder Singh",
        "Smt. Shakuntala",
    ]
    first = done["petitioners"][0]
    assert first["relation"] == "S/o"
    assert first["guardian"] == "Shri Ram Narain"
    assert first["address"].startswith("A-19")
    assert "Presently at" in first["address"]
    assert not first["address"].lower().startswith("r/o")
    assert done["petitioners"][2]["relation"] == "W/o"
    assert done["petitioners"][2]["guardian"] == "Shri Anang Pal Singh"
    names = [party["name"] for party in done["respondents"]]
    assert names == [
        "Union of India",
        "Govt. of N.C.T. Delhi",
        "Land Acquisition Collector",
        "Director of Education, Govt. of NCT",
        "General Manager, Indian Oil Corporation",
        "Secretary, Ministry of Petroleum & Natural Gas, Government of India",
        "Sh. K.S.Mahra, Commissioner of MCD",
        "Master Azad Singh, Counsellor, MCD",
    ]
    assert "Saraswati" in done["respondents"][2]["address"]
    assert done["respondents"][3]["address"] == "Old secretariat, Delhi"
    assert done["respondents"][4]["address"] == (
        "World Trade Centre, Babar Road, New Delhi"
    )
    assert done["respondents"][5]["address"] == "New Delhi"
    assert done["respondents"][6]["address"] == (
        "Town Hall, Chandi Chowk, Delhi-110006"
    )
    assert "Second Address" not in names
    azad = done["respondents"][-1]
    assert "Azad Singh" in azad["name"]
    assert "N.S.23" in azad["address"]
    assert "555" in azad["address"]


def test_position_of_parties_on_later_pages_keeps_both_columns_current_court() -> None:
    """The caption versus is not the list. Column roles and the case number are not parties."""
    page_40 = """
    [Against the Impugned Order in I.A/12/2023 in WRIT-B No. 450 of 1986]
    IN THE MATTER OF:
    Suraj Singh and Ors.
    VERSUS
    Deputy Director of Consolidation, Faizabad and Ors.
    POSITION OF PARTIES
    In the Hon'ble High Court
    In this Hon'ble Court
    1. Mr. Suraj Singh
    Not a Party Appellant No.1
    """
    page_41 = """
    2. Mr. Rameshwar Dutt Singh
    R/o House No. 15, Rampur, Kharahata, P.S. Vazirganj, Tehsil Tarabganj, District Gonda-271124
    Not a Party Appellant No.2
    3. Mr. Sushil Dutt Singh
    R/o House No. 15, Rampur, Kharahata, District Gonda-271124
    Not a Party Appellant No.3
    4. Mr. Akshay Singh
    R/o House No. 15, Rampur, Kharahata, District Gonda-271124
    Not a Party Appellant No.4
    5. Ms. Beauty Singh
    R/o House No. 15, Rampur, Kharahata, District Gonda-271124
    Not a Party Appellant No.5
    6. Mrs. Krishna Kumari
    Not a Party Appellant No. 6
    R/o House No. 15, Rampur, Kharahata, District Gonda-271124
    """
    page_42 = """
    VERSUS
    1.
    Deputy Director of Consolidation, Faizabad
    (now known as Deputy Director of Consolidation, Ayodhya)
    Collectorate Compound, Civil Lines, Ayodhya -224001
    Respondent No.1 Respondent No.1
    2. Mr. Bhagwan Singh (Deceased) through his
    LR's
    Respondent No. 2 Dead
    3. Mr. Vikram Singh (Deceased) through his LR's
    Respondent No. 3 Dead
    4. Mr. Balram Singh (Deceased) through his LR's
    Respondent No. 4 Dead
    5. Mr. Harish Dutt Singh (Deceased) through his LR's
    Petitioner No. 1/2 Dead
    6. Mr. Krishan Dev Singh (Son of Late Mr. Devendra Pratap Singh and missing since last fifteen years)
    Not a Party Proforma Respondent
    """
    done = finalize_front_extract(
        {
            "petition_type": "S.L.P. (CIVIL)",
            "cause_title": {"formatted_title": "A/12/2023 in WRIT-B No. 450 of 1986]"},
            "petitioners": [{"name": "A/12/2023 in WRIT-B No. 450 of 1986]", "address": "Suraj Singh and Ors."}],
            "respondents": [{"name": "and Ors", "address": "Not a Party"}, {"name": "Mr. Suraj Singh"}],
        },
        page_markdown={40: page_40, 41: page_41, 42: page_42},
    )
    petitioners = done["petitioners"]
    respondents = done["respondents"]
    assert [party["name"] for party in petitioners] == [
        "Mr. Suraj Singh",
        "Mr. Rameshwar Dutt Singh",
        "Mr. Sushil Dutt Singh",
        "Mr. Akshay Singh",
        "Ms. Beauty Singh",
        "Mrs. Krishna Kumari",
    ]
    assert petitioners[0]["address"] is None
    assert "271124" in petitioners[1]["address"]
    assert "Not a Party" not in petitioners[1]["address"]
    assert len(respondents) == 6
    assert respondents[0]["name"] == "Deputy Director of Consolidation, Faizabad"
    assert "Ayodhya" in respondents[0]["address"]
    assert "224001" in respondents[0]["address"]
    assert "Bhagwan Singh" in respondents[1]["name"]
    assert "LR" in respondents[1]["acting_through"]
    assert "Krishan Dev Singh" in respondents[5]["name"]
    names = [party["name"] for party in [*petitioners, *respondents]]
    assert not any("A/12/2023" in name or name == "and Ors" or "Not a Party" in name for name in names)


def test_a_middle_party_schedule_is_labelled_opening_pages() -> None:
    pages = supplemental_front_party_pages(
        {
            40: "POSITION OF PARTIES\n1. A\nVERSUS\n1. B",
            41: "2. C",
            42: "3. D",
            43: "MOST RESPECTFULLY SHOWETH",
            44: "later narrative",
        },
        set(range(1, 31)),
    )
    assert sorted(pages) == [40, 41, 42, 43]
    packed = build_front_extract_pack_markdown(
        {1: "opening", 32: "closing vakalatnama", 40: pages[40]},
        opening_pages=30,
    )
    assert "[Compiled filing] (p. 40)" in packed
    assert "[Compiled filing] (p. 32)" in packed
    assert "[Compiled filing] (p. 1)" in packed


def test_index_cuts_the_saved_book_by_source_span() -> None:
    book = {
        "pages": {1: "one", 2: "two", 3: "three"},
        "layout": {2: {"words": ["a"]}, 3: {"words": ["b"]}},
        "structured": {2: {"items": []}},
        "parse_job_id": "pjb-saved",
    }
    pages, layouts, _structured, jobs, slots = compiled_index_reuse(
        book,
        [("petition", [{"start": 2, "end": 3}]), ("annexure_p1", None)],
        edited=False,
        reusable_slots=set(),
    )
    assert slots == ["petition"]
    assert pages["petition"] == {1: "two", 2: "three"}
    assert layouts["petition"][1]["local_page"] == 1
    assert layouts["petition"][1]["slot_id"] == "petition"
    assert layouts["petition"][2]["words"] == ["b"]
    assert jobs["petition"] == "pjb-saved"


def test_unchanged_edited_slot_still_uses_the_saved_book() -> None:
    book = {
        "pages": {1: "one", 2: "two"},
        "layout": {},
        "structured": {},
        "parse_job_id": "pjb",
    }
    pages, _layouts, _structured, jobs, slots = compiled_index_reuse(
        book,
        [("petition", [{"start": 1, "end": 2}])],
        edited=True,
        reusable_slots={"petition"},
    )
    assert slots == ["petition"]
    assert pages["petition"] == {1: "one", 2: "two"}
    assert jobs["petition"] == "pjb"


def test_edited_slot_is_parsed_again_instead_of_using_the_saved_book() -> None:
    book = {"pages": {1: "one", 2: "two"}, "layout": {}, "structured": {}, "parse_job_id": "pjb"}
    _pages, _layouts, _structured, _jobs, slots = compiled_index_reuse(
        book,
        [("petition", [{"start": 1, "end": 2}])],
        edited=True,
        reusable_slots=set(),
    )
    assert slots == []


def test_missing_source_page_is_not_reused() -> None:
    book = {"pages": {1: "one"}, "layout": {}, "structured": {}, "parse_job_id": "pjb"}
    _pages, _layouts, _structured, _jobs, slots = compiled_index_reuse(
        book,
        [("petition", [{"start": 1, "end": 2}])],
        edited=False,
        reusable_slots=set(),
    )
    assert slots == []


def test_source_book_round_trip_keeps_page_text_and_job_id() -> None:
    saved = dump_source_book(
        pages={1: "cover", 2: "petition"},
        layout={1: {"words": []}},
        structured={2: {"items": [{"type": "heading"}]}},
        parse_job_id="pjb-book",
        file_id="dfl-full",
        file_hash="abc",
    )
    loaded = load_source_book({"source_book": saved})
    assert loaded is not None
    assert loaded["pages"] == {1: "cover", 2: "petition"}
    assert loaded["parse_job_id"] == "pjb-book"
    assert loaded["file_id"] == "dfl-full"
    assert loaded["file_hash"] == "abc"
    assert loaded["structured"][2]["items"][0]["type"] == "heading"
