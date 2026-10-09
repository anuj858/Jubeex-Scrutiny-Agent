"""A reproduced e-filing docket owns its internal Index and affidavit."""

from __future__ import annotations

import pytest

from extraction_review.split_repair import (
    _apply_indexed_annexure_ranges,
    _apply_indexed_outer_document_ranges,
    _preserve_historical_efile_annexure_packets,
)
from extraction_review.structure_split import PageUnit, structure_aware_split

OWNER = "Annexure P-10"
CAPTION = "BEFORE THE HONOURABLE HIGH COURT OF KERALA AT ERNAKULAM\n"
PACKET = {
    1: (
        "TA 1/2024 IN WA 644/2024-Docket Presented on 23-05-2024\n"
        "E-IA NO: IA-202425937 E-FILING NO: EF-HCK-2024-040779\n"
        + CAPTION
        + "IA No 1 Of Year 2024\nWA No 644 Of Year 2024\n"
        "SEEKING LEAVE OF THE COURT / PERMISSION\n"
    ),
    2: (
        "JA 1/2024 IN WA 644/2024-Index Presented on 23-05-2024\n"
        + CAPTION
        + "INDEX\nSL Contents Page Nos\n"
        "1 SEEKING LEAVE OF THE COURT / PERMISSION 1-5\n"
    ),
    3: (
        "IA 1/2024 IN WA esarnoza toutlon (Page-1)\n"
        "BEFORE THE HON'BLE HIGH COURT OF KERALA AT ERNAKULAM\n"
        "I A No. 1 of 2024\nW.A. No. 644 of 2024\nAFFIDAVIT\n"
        "I solemnly affirm that I am the appellant.\n"
    ),
    4: (
        "IA 1/2024 IN WA 644/2024-Petition\n(Page-2)\n2\n"
        "The affidavit continues. All facts are true to my knowledge and belief.\n"
        "Solemnly affirmed and signed before me. DEPONENT\n"
    ),
    5: (
        "1A 1/2024 IN WA 644/2024-Petition (Page-3)\n"
        + CAPTION
        + "IA No 1 of 2024\nW.A. No. 644 of 2024\nApplicant/Appellant:\n"
    ),
    6: (
        "IA 1/2024 IN WA 644/2024-Petition\n(Page-4)\n"
        "Respondents: The Secretary, Municipality.\n"
    ),
    7: (
        "IA 1/2024 IN WA 644/2024-Petition\n(Page-5)\n5\n"
        "PETITION FILED UNDER 150 OF HIGH COURT RULES\n"
        "An Affidavit in Support of the Application is filed by the Applicant.\n"
        "PRAYER: Grant leave to file the Writ Appeal.\n"
    ),
}


def _baseline(texts: dict[int, str]) -> dict[int, list[str]]:
    return {page: [OWNER] for page in texts}


def _repair(
    texts: dict[int, str],
    original: dict[int, list[str]] | None = None,
) -> dict[int, list[str]]:
    parts = _baseline(texts)
    parts[2] = ["Index"]
    parts[3] = ["Affidavit"]
    parts[4] = ["Affidavit"]
    return _preserve_historical_efile_annexure_packets(
        parts, original or _baseline(texts), texts, len(texts)
    )


@pytest.mark.parametrize("mode", ["targeted", "legacy_full"])
def test_historical_packet_keeps_internal_index_and_affidavit(mode: str) -> None:
    result = structure_aware_split(
        b"",
        llama_page_parts=_baseline(PACKET),
        page_units=[
            PageUnit(pdf_page=page, text=text, char_count=len(text))
            for page, text in PACKET.items()
        ],
        reconciliation_mode=mode,
    )
    assert result.page_parts == _baseline(PACKET)


def test_running_case_header_repairs_packet_with_ocr_damaged_parent_header() -> None:
    assert _repair(PACKET) == _baseline(PACKET)


@pytest.mark.parametrize(
    "new_document",
    [
        (
            "IN THE SUPREME COURT OF INDIA\nI.A. No. 1 of 2026\n"
            "APPLICATION FOR PERMISSION TO FILE ADDITIONAL DOCUMENTS\n"
            "This relates to IA 1/2024 IN WA 644/2024-Petition.\n"
        ),
        "AFFIDAVIT\nThe deponent previously filed IA 1/2024 IN WA 644/2024.\n",
        "IA 2/2024 IN WA 644/2024-Petition\nAFFIDAVIT\n",
        "IA 1/2024 IN WA 999/2024-Petition\nAFFIDAVIT\n",
        "IA 1/2024 IN WA 644/2024-Petition\nIN THE SUPREME COURT OF INDIA\nAFFIDAVIT\n",
        "IA 1/2024 IN WA 644/2024-Petition\nANNEXURE P-11\nAFFIDAVIT\n",
    ],
)
def test_packet_guard_does_not_absorb_new_document(new_document: str) -> None:
    texts = {**PACKET, 8: new_document}
    parts = _baseline(texts)
    parts[8] = ["Application"]
    repaired = _preserve_historical_efile_annexure_packets(
        parts, _baseline(texts), texts, len(texts)
    )
    assert repaired[8] == ["Application"]


def test_body_case_citation_cannot_bridge_internal_heading() -> None:
    texts = {**PACKET, 3: "AFFIDAVIT\nI filed IA 1/2024 IN WA 644/2024.\n"}
    assert _repair(texts)[3] == ["Affidavit"]


def test_original_document_boundary_remains_authoritative() -> None:
    original = _baseline(PACKET)
    original[3] = ["Affidavit"]
    assert _repair(PACKET, original)[3] == ["Affidavit"]


def test_annexure_identity_recovered_by_index_is_not_overwritten() -> None:
    original = _baseline(PACKET)
    parts = _baseline(PACKET)
    parts[3] = ["Annexure P-11"]
    repaired = _preserve_historical_efile_annexure_packets(
        parts, original, PACKET, len(PACKET)
    )
    assert repaired[3] == ["Annexure P-11"]


def test_case_caption_without_docket_does_not_establish_packet() -> None:
    texts = {**PACKET, 1: CAPTION + "IA 1/2024 IN WA 644/2024\n"}
    assert _repair(texts)[3] == ["Affidavit"]


def test_master_index_range_divides_pages_from_same_historical_filing() -> None:
    texts = {
        1: (
            "INDEX\nS.No. Particulars Page No.\n"
            "1.\tAnnexure P-10: Docket and affidavit\t50-53\n"
            "2.\tAnnexure P-11: Copy of leave petition\t54-56\n"
        ),
        **{page + 1: text + f"\n{49 + page}" for page, text in PACKET.items()},
    }
    original = {1: ["Index"], **{page: [OWNER] for page in range(2, 9)}}
    indexed = _apply_indexed_annexure_ranges(original, texts, len(texts))
    assert indexed[6] == ["Annexure P-11"]
    repaired = _preserve_historical_efile_annexure_packets(
        indexed, original, texts, len(texts)
    )
    assert {page: repaired[page] for page in (6, 7, 8)} == {
        page: ["Annexure P-11"] for page in (6, 7, 8)
    }


def test_outer_index_folios_override_model_owned_historical_source_header() -> None:
    texts = {
        1: (
            "INDEX\nS.No. Particulars Page No.\n"
            "1.\tCopy of the impugned final Order\t54-56\n"
        ),
        **{page + 1: text + f"\n{49 + page}" for page, text in PACKET.items()},
    }
    original = {1: ["Index"], **{page: [OWNER] for page in range(2, 9)}}
    indexed = _apply_indexed_outer_document_ranges(original, texts, len(texts))
    assert indexed[6] == ["Impugned Order"]
    repaired = _preserve_historical_efile_annexure_packets(
        indexed, original, texts, len(texts)
    )
    assert repaired[6] == ["Impugned Order"]
