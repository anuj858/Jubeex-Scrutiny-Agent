"""Outer SCI document anchors cannot be supplied by reproduced court records."""

from __future__ import annotations

import pytest

from extraction_review.split_repair import (
    _is_lower_court_caption,
    _looks_like_court_notice_or_rop,
    _looks_like_sci_main_petition,
    _outer_anchor_label,
    repair_compiled_split,
)

SCI = "IN THE SUPREME COURT OF INDIA\nCIVIL APPELLATE JURISDICTION\n"


@pytest.mark.parametrize(
    "caption",
    [
        "BEFORE THE HON'BLE HIGH COURT OF KERALA AT ERNAKULAM",
        "BEFORE THE HON‘BLE HIGH COURT OF KERALA AT ERNAKULAM",
        "BEFORE THE HONOURABLE HIGH COURT OF\nKERALA AT ERNAKULAM",
        "IN THE HIGH COURT OF DELHI AT NEW DELHI",
        "IN THE DISTRICT COURT AT DELHI",
        "IN THE COURT OF THE ADDITIONAL DISTRICT AND SESSIONS JUDGE",
        "BEFORE THE NATIONAL COMPANY LAW APPELLATE TRIBUNAL",
        "BEFORE THE DEBTS RECOVERY APPELLATE TRIBUNAL AT MUMBAI",
        "BEFORE THE KERALA COASTAL ZONE MANAGEMENT AUTHORITY",
    ],
)
@pytest.mark.parametrize(
    "body",
    [
        "AFFIDAVIT\nI solemnly affirm these facts. DEPONENT",
        "APPLICATION FOR CONDONATION OF DELAY\nGrant leave to file the appeal.",
        "RECORD OF PROCEEDINGS\nThe matter is listed for hearing.",
        (
            "FORM 28\nSPECIAL LEAVE PETITION\nMOST RESPECTFULLY SHOWETH\n"
            "The parties later approached the Supreme Court of India."
        ),
    ],
)
def test_lower_court_caption_below_running_header_is_not_outer_anchor(
    caption: str, body: str
) -> None:
    text = f"IA 1/2024 IN WA 644/2024-Petition\n(Page-1)\n\n{caption}\n{body}"
    assert _is_lower_court_caption(text)
    assert _outer_anchor_label(text) is None
    assert not _looks_like_sci_main_petition(text)
    assert not _looks_like_court_notice_or_rop(text)


@pytest.mark.parametrize(
    "text",
    [
        "The judgment in the High Court of Kerala was challenged.\n",
        "The previous order was issued by the tribunal.\n",
        "1. That the application was made\nbefore the High Court of Kerala.\n",
        (
            "MOST RESPECTFULLY SHOWETH\nThe order was passed\n"
            "IN THE HIGH COURT OF KERALA on 04.03.2024.\n"
        ),
        SCI + "POSITION OF PARTIES\nBEFORE THE TRIBUNAL BEFORE THIS COURT\n",
    ],
)
def test_body_reference_to_lower_court_is_not_a_caption(text: str) -> None:
    assert not _is_lower_court_caption(text)


def test_later_reference_to_supreme_court_does_not_erase_actual_hc_caption() -> None:
    text = (
        "WA 644/2024-Petition\nBEFORE THE HON'BLE HIGH COURT OF KERALA\n"
        "AFFIDAVIT\nThe case was later filed in the Supreme Court of India.\n"
        "I solemnly affirm these facts. DEPONENT\n"
    )
    assert _is_lower_court_caption(text)
    assert _outer_anchor_label(text) is None


@pytest.mark.parametrize(
    ("text", "label"),
    [
        (SCI + "AFFIDAVIT\nI solemnly affirm these facts. DEPONENT", "Affidavit"),
        (
            SCI + "I.A. No. OF 2026\nAPPLICATION FOR PERMISSION TO FILE DOCUMENTS\n"
            "MOST RESPECTFULLY SHOWETH\nThe High Court case supplies the exhibits.",
            "Application 1",
        ),
        (
            SCI + "SPECIAL LEAVE PETITION\nPOSITION OF THE PARTIES\n"
            "BEFORE THE HIGH COURT BEFORE THIS COURT",
            "Main Petition",
        ),
        (
            (
                "RECORD OF PROCEEDINGS\nSL. NO. DATE OF RECORD OF PROCEEDINGS PAGES\n"
                "1.\n2.\n3."
            ),
            "Record of Proceedings",
        ),
        (
            "IN THE HIGH COURT OF KERALA\nIMPUGNED ORDER\nJUDGMENT",
            "Impugned Order",
        ),
        (
            "IN THE HIGH COURT OF KERALA\nMEMO OF PARTIES\nWA 644/2024",
            "Memo of Parties",
        ),
    ],
)
def test_current_sci_anchors_and_explicit_lower_court_exceptions_survive(
    text: str, label: str
) -> None:
    assert _outer_anchor_label(text) == label


def test_affidavit_verifying_application_stays_with_its_sci_application() -> None:
    texts = {
        1: (
            SCI + "I.A. No. OF 2026\nAPPLICATION FOR EXEMPTION FROM TRANSLATION\n"
            "MOST RESPECTFULLY SHOWETH\nThe petitioner seeks exemption.\n210"
        ),
        2: (
            SCI + "AFFIDAVIT\nI solemnly affirm that the contents of the "
            "accompanying application are true.\nDEPONENT\n211"
        ),
    }
    repaired, _ = repair_compiled_split(
        {page: ["Application 1"] for page in texts}, texts, page_count=2
    )
    assert repaired == {page: ["Application 1"] for page in texts}
