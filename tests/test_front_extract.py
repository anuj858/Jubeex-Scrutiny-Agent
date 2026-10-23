"""First-30-page extract: slice, schema, and a null special category."""

from __future__ import annotations

from types import SimpleNamespace

import pymupdf
import pytest

from extraction_review.config import ExtractConfig, FrontExtractRecord, LegalExtractRecord
from extraction_review.extract_record import (
    finalize_front_extract,
    merge_dropped_party_schedule,
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
    front_extract_configuration,
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
    assert "Opening pages" in prompt
    assert "petition_date" in prompt
    assert "special_category" in prompt
    assert "stay null" not in prompt
    assert "configuration_id" not in front


def test_front_pack_keeps_the_middle_of_the_30_page_slice() -> None:
    pages = {page: f"page {page} text" for page in range(1, 36)}
    labels = {page: ["Main Petition"] for page in range(1, 31)}
    normal = build_extract_pack_markdown(pages, labels, {"Main Petition"})
    front = build_front_extract_pack_markdown(pages)
    assert "page 20 text" not in normal
    assert "page 1 text" in front
    assert "page 20 text" in front
    assert "page 30 text" in front
    assert "## [Opening pages] (p. 20)" in front
    assert "## [Closing pages] (p. 31)" in front
    assert "## [Cover Page]" not in front
    assert "page 35 text" in front
    wide = build_front_extract_pack_markdown(pages, opening_pages=75)
    assert "## [Opening pages] (p. 31)" in wide
    assert "## [Closing pages] (p. 31)" not in wide


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
    <tr><td>2. Chandra W/o Ramar South Mavadi Mavadi Post Tirunelveli District</td><td>Defecto Complainant</td><td>Appellant</td><td>Contesting Respondent No.1</td></tr></table>
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


@pytest.mark.asyncio
async def test_extract_front_uploads_the_30_page_slice_and_skips_the_request_category(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from extraction_review.process_file import _run_extract_front

    uploaded: dict[str, bytes] = {}
    captured: dict[str, object] = {}

    async def fake_load(*_args: object, **_kwargs: object) -> bytes:
        return _pdf_bytes(45)

    async def fake_upload(
        _client: object,
        *,
        filename: str,
        pdf_bytes: bytes,
        external_file_id: str | None = None,
    ) -> str:
        del filename, external_file_id
        uploaded["pdf"] = pdf_bytes
        return "dfl-front-slice"

    async def fake_extract(_ctx: object, **kwargs: object) -> str:
        captured.update(kwargs)
        return "agd-front"

    monkeypatch.setattr(
        "extraction_review.process_file._load_bundle_pdf",
        fake_load,
    )
    monkeypatch.setattr(
        "extraction_review.process_file._upload_slot_pdf",
        fake_upload,
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
    assert _page_count(uploaded["pdf"]) == (
        FRONT_EXTRACT_PAGE_LIMIT + FRONT_EXTRACT_END_PAGE_LIMIT
    )
    assert captured["front_extract"] is True
    assert captured["skip_index"] is True
    assert captured["special_category"] is None
    assert captured["court"] is None
    parts = captured["parts"]
    assert parts[0].file_id == "dfl-front-slice"
    assert result.agent_data_id == "agd-front"
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
    assert len(respondents) == 10
    names = [party["name"] for party in respondents]
    assert names[0].startswith("K.D.R. Farms")
    assert "Rajendra Madhukarrao Kulkarni" in names[1]
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
