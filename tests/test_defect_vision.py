"""Defect-scoped vision selects pages from the defect text, not a fixed id list."""

from __future__ import annotations

from types import SimpleNamespace

import pytest

from extraction_review.process_split_files import _detect_visual_for_state
from extraction_review.scrutiny.rules import Defect
from extraction_review.visual.defect_call import (
    pack_vision_batches,
    parse_defect_vision_payload,
)
from extraction_review.visual.defect_scope import (
    defect_needs_ink,
    plan_defect_images,
    selector_for_document,
)
from extraction_review.visual.detect import detect_visual_marks


def _defect(
    check_id: str,
    title: str,
    *,
    requirement: str = "The named document must show the mark.",
    inspect_parts: list[str],
    ai_note: str | None = None,
) -> Defect:
    return Defect(
        check_id=check_id,
        main_category="General/Global",
        defect=title,
        requirement=requirement,
        where_to_look=list(inspect_parts),
        inspect_parts=inspect_parts,
        how_to_cure=["Replace the document"],
        location_source="SCI Handbook",
        ai_note=ai_note,
    )


def _layout(rows: list[tuple[int, str, str, int, str]]) -> dict[int, dict]:
    layout: dict[int, dict] = {}
    for page, part, slot, local, text in rows:
        layout[page] = {
            "document_part": part,
            "slot_id": slot,
            "local_page": local,
            "words": [{"t": token} for token in text.split() if token],
        }
    return layout


def test_ink_comes_from_the_defect_wording_not_a_fixed_list() -> None:
    sworn = _defect("NEW-1", "The affidavit is not duly sworn", inspect_parts=["Affidavit"])
    signed = _defect("NEW-2", "The vakalatnama is not signed", inspect_parts=["Vakalatnama"])
    plain = _defect(
        "NEW-3",
        "The party address is incomplete",
        requirement="Each party needs a city and PIN.",
        inspect_parts=["Main Petition"],
        ai_note="NEED REVIEW when a signature cannot be retrieved.",
    )
    landmark = _defect(
        "NEW-4",
        "The date below the Judge's signature is blank",
        requirement="State the date of the order.",
        inspect_parts=["Impugned Order"],
    )
    assert defect_needs_ink(sworn)
    assert defect_needs_ink(signed)
    assert not defect_needs_ink(plain)
    assert not defect_needs_ink(landmark)


def test_vakalatnama_of_two_or_three_pages_is_sent_whole() -> None:
    defect = _defect(
        "VX-1",
        "Vakalatnama is not signed by the petitioner",
        inspect_parts=["Vakalatnama"],
    )
    two = plan_defect_images(
        [defect],
        _layout(
            [
                (1, "Vakalatnama", "vak", 1, "do hereby appoint"),
                (2, "Vakalatnama", "vak", 2, "accepted identified"),
            ]
        ),
    )[0]
    assert [page.local_page for page in two.pages] == [1, 2]
    assert two.complete

    three = plan_defect_images(
        [defect],
        _layout(
            [
                (4, "Vakalatnama", "vak", 1, "appoint"),
                (5, "Vakalatnama", "vak", 2, "accepted"),
                (6, "Vakalatnama", "vak", 3, "identified"),
            ]
        ),
    )[0]
    assert [page.local_page for page in three.pages] == [1, 2, 3]
    assert three.complete


def test_four_image_cap_leaves_the_rest_for_the_text_call() -> None:
    defect = _defect(
        "CAP-1",
        "The petition is not duly signed at the required places",
        inspect_parts=[
            "Advocate's Checklist",
            "AOR's Certificate",
            "Listing Proforma",
            "Filing Memo",
            "Vakalatnama",
        ],
    )
    layout = _layout(
        [
            (1, "Advocate's Checklist", "a", 1, "checklist"),
            (2, "AOR's Certificate", "b", 1, "certificate"),
            (3, "Listing Proforma", "c", 1, "proforma"),
            (4, "Filing Memo", "d", 1, "memo"),
            (5, "Vakalatnama", "e", 1, "vakalatnama"),
        ]
    )
    plan = plan_defect_images([defect], layout)[0]
    assert [page.document_type for page in plan.pages] == [
        "Advocate's Checklist",
        "AOR's Certificate",
        "Listing Proforma",
        "Filing Memo",
    ]
    assert [page.document_type for page in plan.leftover_pages] == ["Vakalatnama"]
    assert not plan.complete


def test_a_new_document_name_uses_every_page_until_the_cap() -> None:
    assert selector_for_document("Custody Receipt") == "all"
    defect = _defect(
        "FUTURE-1",
        "The custody receipt is not signed",
        inspect_parts=["Custody Receipt"],
    )
    plan = plan_defect_images(
        [defect],
        _layout(
            [
                (8, "Custody Receipt", "cust", 1, "received"),
                (9, "Custody Receipt", "cust", 2, "signed"),
            ]
        ),
    )[0]
    assert [page.local_page for page in plan.pages] == [1, 2]
    assert plan.complete


def test_a_shared_page_is_one_batch() -> None:
    layout = _layout([(3, "Vakalatnama", "vak", 1, "accepted")])
    plans = plan_defect_images(
        [
            _defect("A-1", "Vakalatnama is not signed", inspect_parts=["Vakalatnama"]),
            _defect(
                "A-2",
                "The acceptance is not attested",
                inspect_parts=["Vakalatnama"],
            ),
        ],
        layout,
    )
    batches = pack_vision_batches(plans)
    assert len(batches) == 1
    assert len(batches[0].pages) == 1
    assert batches[0].pages[0].page == 3
    assert batches[0].check_ids == ["A-1", "A-2"]


def test_vision_payload_keeps_the_answer_and_the_mark() -> None:
    page = plan_defect_images(
        [_defect("VX-9", "Vakalatnama is not signed", inspect_parts=["Vakalatnama"])],
        _layout([(3, "Vakalatnama", "vak", 2, "accepted")]),
    )[0].pages[0]
    parsed = parse_defect_vision_payload(
        """
        {"answers":[{
          "check_id":"VX-9",
          "status":"defect_found",
          "confidence":0.8,
          "summary":"The acceptance line has no signature.",
          "observed":"The acceptance line is blank.",
          "conclusion":"The vakalatnama is unsigned.",
          "evidence":[{"quote":"Satisfied, certified, identified and accepted"}],
          "marks":[{
            "page":3,
            "probable_type":"signature_like_mark",
            "bbox_normalized":{"x":0.62,"y":0.81,"width":0.2,"height":0.06},
            "confidence":0.7
          }]
        }]}
        """,
        pages=[page],
        check_ids=["VX-9"],
    )
    answer = parsed["VX-9"]
    assert answer.response.status == "defect_found"
    assert answer.marks[0].page == 2
    assert answer.marks[0].bounding_boxes[0].x == 0.62


def test_short_affidavit_is_sent_whole() -> None:
    defect = _defect(
        "AF-1",
        "The affidavit is not duly sworn",
        inspect_parts=["Affidavit"],
    )
    plan = plan_defect_images(
        [defect],
        _layout(
            [
                (10, "Affidavit", "aff", 1, "solemnly affirm"),
                (11, "Affidavit", "aff", 2, "notary deponent"),
            ]
        ),
    )[0]
    assert [page.local_page for page in plan.pages] == [1, 2]


@pytest.mark.asyncio
async def test_ingest_defers_the_broad_formality_scan() -> None:
    assert detect_visual_marks.__name__ == "detect_visual_marks"
    events: list[object] = []
    index = await _detect_visual_for_state(
        SimpleNamespace(),
        SimpleNamespace(write_event_to_stream=events.append),
    )
    assert index["status"] == "skipped"
    assert index["error"] == "deferred_to_scrutiny"
    assert index["marks"] == []
    assert events
