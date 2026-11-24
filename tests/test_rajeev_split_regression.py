from __future__ import annotations

from extraction_review.split_repair import (
    _explicit_unmapped_document_heading,
    _is_sci_caption,
    _looks_like_filing_inventory_index,
    _looks_like_judgment_continuation,
    _looks_like_rop_index_form,
    _looks_like_sci_main_petition,
    _outer_anchor_label,
)
from extraction_review.structure_split import PageUnit, structure_aware_split


def test_wrapped_record_of_proceedings_columns_open_the_rop_part() -> None:
    text = (
        "| S.NO. | DATE OF RECORD OF | PAGES |\n"
        "| ----- | ----------------- | ----- |\n"
        "|       | PROCEEDINGS       |       |\n"
    )

    assert _looks_like_rop_index_form(text)
    assert _outer_anchor_label(text) == "Record of Proceedings"


def test_crmp_judgment_continuation_is_not_a_main_petition_opening() -> None:
    text = (
        "4 | (2026:JHHC:26284)\n"
        "4 Cr. M.P. No.2683 of 2026\n"
        "Hon'ble Supreme Court of India in the case of Ram Narayan Popli vs. "
        "Central Bureau of Investigation reported in (2003) 3 SCC 641.\n"
        "11. Now, coming to the facts of the case, there is no entrustment.\n"
        "12. So far as the offence punishable under Section 318(4) is concerned, "
        "this Court considered the submission.\n"
    )

    assert _looks_like_judgment_continuation(text)
    assert not _looks_like_sci_main_petition(text)
    assert _outer_anchor_label(text) is None


def test_judgment_body_sci_citation_is_not_a_new_main_petition_caption() -> None:
    text = (
        "Hon’ble Supreme Court of India in the case of Ram Narayan Popli vs. "
        "Central Bureau of Investigation reported in (2003) 3 SCC 641.\n\n"
        "11. Now, coming to the facts of the case, there is absolutely no "
        "allegation of entrustment of money.\n"
        "The principle of law, as has been held by the Hon’ble Supreme Court "
        "of India in Radheyshyam & Others vs. State of Rajasthan, applies.\n"
        "12. So far as the offence is concerned, this Court considered it.\n"
    )

    assert not _is_sci_caption(text)
    assert not _looks_like_sci_main_petition(text)
    assert _outer_anchor_label(text) is None


def test_neutral_citation_running_header_is_not_an_unmapped_document() -> None:
    text = (
        "**4** | **(2026:JHHC:26284)**\n"
        "Hon’ble Supreme Court of India in the case of Ram Narayan Popli vs. "
        "Central Bureau of Investigation reported in (2003) 3 SCC 641.\n"
        "11. Now, coming to the facts of the case, the Court considered the FIR.\n"
    )

    assert _explicit_unmapped_document_heading(text) is None
    assert not _is_sci_caption(text)
    assert _outer_anchor_label(text) is None


def test_high_court_judgment_opening_is_not_an_unmapped_document() -> None:
    text = (
        "**1** | **(2026:JHHC:26284)**\n"
        "# **IN THE HIGH COURT OF JHARKHAND AT RANCHI**\n"
        "## **Cr.M.P. No.2683 of 2026**\n"
        "Sanjeev Kumar and another — Petitioners\n"
        "Versus\n"
        "The State of Jharkhand — Opposite Party\n"
    )

    assert _explicit_unmapped_document_heading(text) is None
    assert not _is_sci_caption(text)
    assert _outer_anchor_label(text) is None


def test_real_sci_caption_still_opens_main_petition() -> None:
    text = (
        "# IN THE SUPREME COURT OF INDIA\n"
        "CRIMINAL APPELLATE JURISDICTION\n"
        "SPECIAL LEAVE PETITION (CRIMINAL) NO. OF 2026\n"
        "POSITION OF PARTIES\n"
        "Rajeev Ranjan Singh — Petitioner before this Hon'ble Court\n"
    )

    assert _is_sci_caption(text)
    assert _looks_like_sci_main_petition(text)
    assert _outer_anchor_label(text) == "Main Petition"


def test_ocr_split_sci_caption_still_matches() -> None:
    text = (
        "IN THE\n"
        "S_UPRE1:IE COURT OF INDIA\n"
        "SPECIAL LEAVE PETITION (CRIMINAL) NO. OF 2026\n"
    )

    assert _is_sci_caption(text)


def test_unmapped_document_does_not_create_duplicate_known_part() -> None:
    text = (
        "109\nFIR DETAILS\n"
        "1 Diary No /Crime No 303/2024\n"
        "2 Date of Lodgment of FIR/ Complaint 12.09.2024\n"
        "3 Place of Occurrence Ushamatu, District Ranchi\n"
        "4 Police Station address with State PS Ratu, Ranchi\n"
        "5 Date of filing charge sheet & charge sheet: NA\n"
        "6 Whether tried by the Court of Sessions: NA\n"
    )

    assert _explicit_unmapped_document_heading(text)
    # The guarded reconciler must be allowed to reject this particular model
    # label even when the Llama boundary was protected as a singleton segment.
    result = structure_aware_split(
        b"",
        page_units=[PageUnit(pdf_page=1, text=text)],
        llama_page_parts={1: ["Listing Proforma"]},
        authoritative_boundary_pages=(1,),
        authoritative_segments=(
            {
                "start_page": 1,
                "end_page": 1,
                "category": "Listing Proforma",
                "protected": True,
            },
        ),
    )
    assert result.page_parts[1] == ["Undefined"]


def test_any_unmapped_multi_page_document_routes_to_other() -> None:
    opening = (
        "DIGITAL EVIDENCE INVENTORY\n"
        "Device identifier: 42\nHash value: ABC123\nItems enclosed: 7\n"
    )
    continuation = "Inventory item descriptions continue on this page.\n"
    result = structure_aware_split(
        b"",
        page_units=[
            PageUnit(pdf_page=1, text=opening),
            PageUnit(pdf_page=2, text=continuation),
        ],
        llama_page_parts={1: ["Listing Proforma"], 2: ["Listing Proforma"]},
        authoritative_boundary_pages=(1,),
        authoritative_segments=(
            {
                "start_page": 1,
                "end_page": 2,
                "category": "Listing Proforma",
                "protected": True,
            },
        ),
    )

    assert _explicit_unmapped_document_heading(opening)
    assert result.page_parts == {1: ["Undefined"], 2: ["Undefined"]}


def test_index_of_papers_filing_inventory_is_filing_memo() -> None:
    text = (
        "IN THE SUPREME COURT OF INDIA\n"
        "SPECIAL LEAVE PETITION (CRIMINAL) NO. OF 2026\n"
        "INDE\nX OF PAPERS\n"
        "S. NO. PARTICULARS COURT FEE\n"
        "1. Synopsis & List of Dates\n"
        "2. SLP along with affidavit\n"
        "3. Annexure P-1 to P-9\n"
        "4. Application for exemption from official translation\n"
        "5. Application for exemption from certified copy\n"
        "6. Vakalatnama and Memo of Appearance\n"
        "FILED BY: Advocate for the Petitioner\nDATED: 30.10.2026\n"
    )

    assert _looks_like_filing_inventory_index(text)
    assert _outer_anchor_label(text) == "Filing Memo"

