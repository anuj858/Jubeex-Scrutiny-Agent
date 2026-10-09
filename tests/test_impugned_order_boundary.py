"""Court neutral citations and delayed judgment titles are not LOD events."""

import pytest

from extraction_review.split_repair import (
    _looks_like_lod_continuation,
    _restore_front_impugned_judgment,
)
from extraction_review.structure_split import PageUnit, structure_aware_split


def judgment_packet():
    return {
        1: "LIST OF DATES & EVENTS\n18.03.2026 The appeal was dismissed.\nY",
        2: "2026:KER:24461\nIN THE HIGH COURT OF KERALA AT ERNAKULAM\n"
        "PRESENT\nTHE HONOURABLE JUDGES\nWA NO. 644 OF 2024\n"
        "AGAINST THE JUDGMENT DATED 04.03.2024 IN WP(C) NO.28199 OF\n"
        "2012 OF HIGH COURT OF KERALA\nAPPELLANT\nRESPONDENTS\n1",
        3: "2026:KER:24461\nWA 644/24\n2\nRESPONDENTS CONTINUED\n"
        "THIS WRIT APPEAL HAVING COME UP FOR ADMISSION ON\n"
        "18.03.2026 THE COURT DELIVERED THE FOLLOWING:\n2",
        4: "2026:KER:24461\nWA 644/24\n3\nJUDGMENT\n"
        "The appellant challenges the earlier decision.\n3",
        5: "2026:KER:24461\nWA 644/24\n4\nThe appeal is dismissed.\nSd/-\nJUDGE\n4",
        6: "IN THE SUPREME COURT OF INDIA\nSPECIAL LEAVE PETITION\n"
        "UNDER ARTICLE 136\nPOSITION OF PARTIES\n5",
    }


def test_neutral_citation_in_court_header_is_not_a_chronology_year():
    for page, text in judgment_packet().items():
        if 2 <= page <= 5:
            assert not _looks_like_lod_continuation(text)
    assert _looks_like_lod_continuation("2026 The High Court decided the appeal.")
    assert _looks_like_lod_continuation(
        "18.03.2026 In the High Court of Kerala the appeal was dismissed."
    )


@pytest.mark.parametrize("seed_order", [True, False])
def test_targeted_front_judgment_does_not_join_list_of_dates(seed_order):
    texts = judgment_packet()
    baseline = {1: ["List of Dates & Events"], 6: ["Main Petition"]}
    if seed_order:
        baseline.update({page: ["Impugned Order"] for page in range(2, 6)})
    result = structure_aware_split(
        b"",
        llama_page_parts=baseline,
        page_units=[
            PageUnit(pdf_page=page, text=text, char_count=len(text))
            for page, text in texts.items()
        ],
        reconciliation_mode="targeted",
    )
    assert result.page_parts == {
        1: ["List of Dates & Events"],
        **{page: ["Impugned Order"] for page in range(2, 6)},
        6: ["Main Petition"],
    }


@pytest.mark.parametrize("change", ["different_source", "missing_folio", "annexure"])
def test_delayed_title_needs_same_source_and_consecutive_pages(change):
    texts = judgment_packet()
    baseline = {1: ["List of Dates & Events"], 6: ["Main Petition"]}
    if change == "different_source":
        texts[4] = texts[4].replace("24461", "99999")
    elif change == "missing_folio":
        texts[3] = texts[3].removesuffix("\n2")
    elif change == "annexure":
        texts[4] = "ANNEXURE P-1\n" + texts[4]
    assert _restore_front_impugned_judgment(baseline, texts, 6) == baseline
