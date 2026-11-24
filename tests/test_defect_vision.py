"""Defect-scoped vision selects pages from the defect text, not a fixed id list."""

from __future__ import annotations

from types import SimpleNamespace

import pytest

from extraction_review.process_split_files import _detect_visual_for_state
from extraction_review.scrutiny.rules import Catalogue, Defect
from extraction_review.visual.defect_call import (
    finding_from_vision,
    pack_vision_batches,
    parse_defect_vision_payload,
)
from extraction_review.visual.defect_scope import (
    DefectImagePage,
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
    rendered_quality = _defect(
        "NEW-5",
        "The annexure is dim and not accompanied by a true typed copy",
        requirement="Every annexure must be clear, sharp and legible.",
        inspect_parts=["Annexures"],
    )
    pagination = _defect(
        "NEW-6",
        "The Index does not use running pagination",
        requirement="Index page ranges must remain in pagination order.",
        inspect_parts=["Index"],
    )
    assert defect_needs_ink(sworn)
    assert defect_needs_ink(signed)
    assert not defect_needs_ink(plain)
    assert defect_needs_ink(landmark)
    assert defect_needs_ink(rendered_quality)
    assert defect_needs_ink(pagination)


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


def test_date_check_keeps_the_closing_page_inside_the_cap() -> None:
    defect = _defect(
        "DATE-4",
        "The enclosed document is not of the same or subsequent date",
        requirement="The document must bear a date that is the same as, or later than, the date of drafting.",
        inspect_parts=["Affidavit", "Main Petition"],
    )
    affidavit = [
        (index, "Affidavit", "aff", index, "verification deponent")
        for index in range(1, 4)
    ]
    petition = [
        (10, "Main Petition", "pet", 1, "cause title"),
        (11, "Main Petition", "pet", 2, "facts"),
        (12, "Main Petition", "pet", 3, "Drawn on 10.10.2025 Filed on 12.12.2025"),
    ]
    plan = plan_defect_images([defect], _layout(affidavit + petition))[0]
    petition_pages = [
        page.local_page for page in plan.pages if page.document_type == "Main Petition"
    ]
    assert petition_pages == [3]
    assert all(page.local_page != 1 or page.document_type != "Main Petition" for page in plan.pages)


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


def _catalogue_for(defect: Defect) -> Catalogue:
    return Catalogue(
        catalogue_id="sci_registry_defects",
        schema_version="test",
        catalogue_version="test",
        jurisdiction="sci",
        defects=[defect],
    )


def test_a_vision_quote_keeps_the_inspected_document(monkeypatch) -> None:
    defect = _defect(
        "D-116",
        "Memo of Appearance does not mention the date of execution/filing.",
        requirement=(
            "The Memo of Appearance must mention the date on which it is "
            "executed/signed."
        ),
        inspect_parts=["Vakalatnama"],
    )
    monkeypatch.setattr(
        "extraction_review.scrutiny.schema.get_catalogue",
        lambda: _catalogue_for(defect),
    )
    layout = {
        216: {
            "document_part": "Vakalatnama",
            "slot_id": "vakalatnama",
            "local_page": 1,
            "words": [
                {"t": "MEMO", "x": 0.30, "y": 0.42, "w": 0.08, "h": 0.02, "line": 1},
                {"t": "OF", "x": 0.39, "y": 0.42, "w": 0.04, "h": 0.02, "line": 1},
                {
                    "t": "APPEARANCE",
                    "x": 0.44,
                    "y": 0.42,
                    "w": 0.16,
                    "h": 0.02,
                    "line": 1,
                },
            ],
        }
    }
    page = DefectImagePage(
        page=216,
        local_page=1,
        slot_id="vakalatnama",
        document_type="Vakalatnama",
    )
    parsed = parse_defect_vision_payload(
        """
        {"answers":[{
          "check_id":"D-116",
          "status":"defect_found",
          "confidence":0.98,
          "summary":"The Memo of Appearance has a blank date field",
          "observed":"The field labelled Dated is unfilled.",
          "conclusion":"The memo does not mention its date.",
          "evidence":[{"quote":"MEMO OF APPEARANCE"},{"quote":"Dated:"}]
        }]}
        """,
        pages=[page],
        check_ids=["D-116"],
    )
    finding = finding_from_vision(defect, parsed["D-116"], layout=layout)
    by_quote = {item.quote: item for item in finding.evidence}
    memo = by_quote["MEMO OF APPEARANCE"]
    assert memo.document_part == "Vakalatnama"
    assert memo.slot_id == "vakalatnama"
    assert memo.page == 1
    assert memo.boxes_status == "matched"
    assert memo.bounding_boxes
    dated = by_quote["Dated:"]
    assert dated.document_part == "Vakalatnama"
    assert dated.slot_id == "vakalatnama"
    assert dated.page == 1
    assert dated.boxes_status == "page_only"


def test_a_vision_quote_is_not_assigned_across_two_documents(monkeypatch) -> None:
    defect = _defect(
        "D-116",
        "Memo of Appearance does not mention the date of execution/filing.",
        requirement="The memo must be signed.",
        inspect_parts=["Vakalatnama"],
    )
    monkeypatch.setattr(
        "extraction_review.scrutiny.schema.get_catalogue",
        lambda: _catalogue_for(defect),
    )
    layout = _layout(
        [
            (10, "Affidavit", "affidavit", 1, "solemnly affirm"),
            (216, "Vakalatnama", "vakalatnama", 1, "advocate"),
        ]
    )
    pages = [
        DefectImagePage(
            page=10,
            local_page=1,
            slot_id="affidavit",
            document_type="Affidavit",
        ),
        DefectImagePage(
            page=216,
            local_page=1,
            slot_id="vakalatnama",
            document_type="Vakalatnama",
        ),
    ]
    parsed = parse_defect_vision_payload(
        """
        {"answers":[{
          "check_id":"D-116",
          "status":"defect_found",
          "confidence":0.9,
          "summary":"The date is blank.",
          "observed":"Dated is blank.",
          "conclusion":"No date.",
          "evidence":[{"quote":"MEMO OF APPEARANCE"}]
        }]}
        """,
        pages=pages,
        check_ids=["D-116"],
    )
    finding = finding_from_vision(defect, parsed["D-116"], layout=layout)
    assert finding.evidence[0].document_part is None
    assert finding.evidence[0].slot_id is None


def test_a_date_on_any_document_is_imaged_from_its_own_parts() -> None:
    affidavit = _defect(
        "DATE-1",
        "The enclosed document is not of the same or subsequent date",
        requirement="The document must bear a date that is the same as, or later than, the other date.",
        inspect_parts=["Affidavit"],
    )
    vakalatnama = _defect(
        "DATE-2",
        "The execution date is not stated",
        requirement="The document must mention the date of execution.",
        inspect_parts=["Vakalatnama"],
    )
    order_date = _defect(
        "DATE-3",
        "The appeal does not state the date of the impugned order",
        requirement="State the date of the impugned judgment.",
        inspect_parts=["Impugned Order"],
    )
    assert defect_needs_ink(affidavit)
    assert defect_needs_ink(vakalatnama)
    assert not defect_needs_ink(order_date)
    plan = plan_defect_images(
        [affidavit],
        _layout(
            [
                (10, "Affidavit", "aff", 1, "solemnly affirm"),
                (11, "Affidavit", "aff", 2, "Verified on this day of the stated month"),
            ]
        ),
    )[0]
    assert [page.local_page for page in plan.pages] == [1, 2]
    vak_plan = plan_defect_images(
        [vakalatnama],
        _layout(
            [
                (3, "Vakalatnama", "vak", 1, "appoint"),
                (4, "Vakalatnama", "vak", 2, "dated"),
            ]
        ),
    )[0]
    assert [page.local_page for page in vak_plan.pages] == [1, 2]


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
async def test_supplied_pages_use_the_notary_detector(monkeypatch: pytest.MonkeyPatch) -> None:
    from extraction_review.visual.schema import VisualPageTarget

    def refuse_policy(*_args: object, **_kwargs: object) -> list[object]:
        raise AssertionError("the standing page policy must not choose these pages")

    monkeypatch.setattr(
        "extraction_review.visual.detect.visual_detection_enabled",
        lambda: True,
    )
    monkeypatch.setattr(
        "extraction_review.visual.detect.vision_model",
        lambda _tier=None: "test-model",
    )
    monkeypatch.setattr(
        "extraction_review.visual.detect.select_formality_pages",
        refuse_policy,
    )
    index = await detect_visual_marks(
        parts=[],
        pages_by_slot={},
        page_parts={},
        model_tier="low",
        selected_targets=[
            VisualPageTarget(
                page=11,
                slot_id="affidavit",
                local_page=2,
                document_types=["Affidavit"],
            )
        ],
    )
    assert index["targets"] == [{"page": 11, "document_types": ["Affidavit"]}]
    assert index["status"] == "error"
    assert index["failures"][0]["page"] == 11


def test_signature_check_sends_the_last_page_and_its_text() -> None:
    defect = _defect(
        "SIGN-1",
        "The annexure is not signed",
        requirement="Each annexure must bear a signature on its closing page.",
        inspect_parts=["Annexures"],
    )
    rows = [
        (index, f"Annexure P-1", "ann", index, f"body {index}")
        for index in range(1, 6)
    ]
    rows[-1] = (5, "Annexure P-1", "ann", 5, "Signed by the deponent")
    plan = plan_defect_images([defect], _layout(rows))[0]
    assert any(page.local_page == 5 and "Signed" in page.text for page in plan.pages)
    from extraction_review.visual.defect_scope import visual_targets_for_plans

    targets = visual_targets_for_plans([plan])
    closing = next(item for item in targets if item.local_page == 5)
    assert "Signed by the deponent" in closing.markdown


def test_selected_pages_become_one_notary_target_each() -> None:
    from extraction_review.visual.defect_scope import (
        DefectImagePage,
        DefectImagePlan,
        visual_targets_for_plans,
    )

    plans = [
        DefectImagePlan(
            check_id="A",
            pages=(
                DefectImagePage(11, 2, "aff", "Affidavit", file_id="file-a"),
            ),
            leftover_pages=(),
        ),
        DefectImagePlan(
            check_id="B",
            pages=(
                DefectImagePage(11, 2, "aff", "Affidavit", file_id="file-a"),
                DefectImagePage(20, 9, "pet", "Main Petition", file_id="file-b"),
            ),
            leftover_pages=(),
        ),
    ]
    targets = visual_targets_for_plans(plans)
    assert [(item.page, item.local_page, item.file_id) for item in targets] == [
        (11, 2, "file-a"),
        (20, 9, "file-b"),
    ]


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
