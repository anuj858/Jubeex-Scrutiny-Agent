"""Application body references must not transfer pages to their attachments."""

import pytest

from extraction_review.split_repair import _reclaim_application_continuations
from extraction_review.structure_split import (
    PageUnit,
    classify_pages,
    reconcile_compiled_split,
    structure_aware_split,
)

APPLICATION = (
    "IN THE SUPREME COURT OF INDIA\nCIVIL APPELLATE JURISDICTION\n"
    "I.A. NO. OF 2026\nIN SPECIAL LEAVE PETITION (C) NO. OF 2026\n"
    "AN APPLICATION FOR PERMISSION TO FILE ADDITIONAL DOCUMENTS\n"
    "MOST RESPECTFULLY SHOWETH\n"
    "1. That the present is an application for bringing on record additional documents.\n"
    "2. The petitioner seeks permission to rely on these documents.\n212"
)
CONTINUATION = (
    "3. The respondent filed a writ petition before the High Court.\n"
    "A true copy is annexed herewith and marked as\n"
    "ANNEXURE P-12 (Kindly see Pages 215 to 232).\n"
    "4. The document has material bearing on the special leave petition.\n"
    "5. The present application is made bonafide.\n"
    "6. The Court may allow the applicant to file additional documents.\n"
    "PRAYER\nIt is most respectfully prayed that this Hon'ble Court be pleased to:\n213"
)
CLOSING = (
    "a) Grant permission to bring on record Annexure P/12 as additional documents.\n"
    "b) Pass such further and other orders as may be deemed fit.\n"
    "AND FOR THIS ACT OF KINDNESS THE PETITIONER AS IN\n"
    "DUTY BOUND SHALL EVER PRAY.\nFILED BY\n"
    "ADVOCATE FOR THE PETITIONER\nFILED ON: 21.05.2026\n214"
)


@pytest.mark.parametrize("mode", ["targeted", "legacy_full"])
def test_application_and_prayer_are_recovered_before_the_cited_exhibit(mode):
    texts = {
        1: APPLICATION,
        2: CONTINUATION,
        3: CLOSING,
        4: "BEFORE THE HIGH COURT OF KERALA\nW.P.(C) NO. 1234 OF 2012\n"
        "Petitioner\nRespondents\nWRIT PETITION\n215",
        5: "Continued earlier writ pleading.\n216",
        6: "AFFIDAVIT\nI verify this earlier High Court writ petition.\n217",
    }
    units = [PageUnit(pdf_page=p, text=t, char_count=len(t)) for p, t in texts.items()]
    result = structure_aware_split(
        b"",
        llama_page_parts={p: ["Annexure P-12"] for p in texts},
        page_units=units,
        reconciliation_mode=mode,
    )
    assert result.page_parts == {
        1: ["Application 1"],
        2: ["Application 1"],
        3: ["Application 1"],
        4: ["Annexure P-12"],
        5: ["Annexure P-12"],
        6: ["Annexure P-12"],
    }


@pytest.mark.parametrize(
    "next_text",
    [
        "ANNEXURE P-12\nGovernment notice\n213",
        "IN THE HIGH COURT OF KERALA\nAPPLICATION FOR LEAVE\n213",
        "FILING MEMO\nPapers filed by the advocate\n213",
        "Some unrelated content with no supported continuation evidence.\n213",
        CONTINUATION.replace("\n213", "\n260"),
    ],
)
def test_application_recovery_does_not_swallow_new_or_unsupported_documents(next_text):
    baseline = {1: ["Application 2"], 2: ["Annexure P-12"]}
    assert (
        _reclaim_application_continuations(baseline, {1: APPLICATION, 2: next_text}, 2)
        == baseline
    )


def test_lower_court_application_packet_remains_enclosed():
    texts = {
        1: APPLICATION.replace("SUPREME COURT OF INDIA", "HIGH COURT OF KERALA"),
        2: CONTINUATION,
        3: CLOSING,
    }
    baseline = {p: ["Annexure P-10"] for p in texts}
    assert _reclaim_application_continuations(baseline, texts, 3) == baseline


def test_application_recovery_stops_after_its_signed_prayer():
    texts = {1: APPLICATION, 2: CONTINUATION, 3: CLOSING, 4: CONTINUATION}
    baseline = {p: ["Annexure P-12"] for p in texts}
    result = _reclaim_application_continuations(baseline, texts, 4)
    assert result[1] == result[2] == result[3] == ["Application 1"]
    assert result[4] == ["Annexure P-12"]


def test_targeted_recovery_recognizes_second_application_and_its_body():
    texts = {
        1: APPLICATION.replace("ADDITIONAL DOCUMENTS", "EXEMPTION FROM CERTIFIED COPY"),
        2: CLOSING.replace("\n214", "\n213"),
        3: APPLICATION.replace("\n212", "\n214"),
        4: CONTINUATION.replace("\n213", "\n215"),
        5: CLOSING.replace("\n214", "\n216"),
        6: "ANNEXURE P-12\nIN THE HIGH COURT OF KERALA\nWRIT PETITION\n217",
    }
    baseline = {
        page: ["Application 1"] if page <= 2 else ["Annexure P-12"] for page in texts
    }
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
        1: ["Application 1"],
        2: ["Application 1"],
        3: ["Application 2"],
        4: ["Application 2"],
        5: ["Application 2"],
        6: ["Annexure P-12"],
    }


@pytest.mark.parametrize("proposed_identity", ["Application 1", "Application 3"])
def test_generic_application_heading_does_not_renumber_existing_identity(
    proposed_identity,
):
    texts = {1: APPLICATION, 2: CONTINUATION}
    baseline = {page: ["Application 2"] for page in texts}
    proposed = {page: [proposed_identity] for page in texts}
    units = [
        PageUnit(pdf_page=page, text=text, char_count=len(text))
        for page, text in texts.items()
    ]
    classifications = classify_pages(units, llama_page_parts=baseline)
    assert classifications[0].document_type == "Application 1"
    result, changes = reconcile_compiled_split(
        baseline, proposed, classifications, texts, page_count=2
    )
    assert result == baseline
    assert not changes
