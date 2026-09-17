"""Ink-mark detector schema, formality page targeting, and defect attachment."""

from types import SimpleNamespace

import pytest

from extraction_review.document_parts import MAIN_PETITION_PART
from extraction_review.llm import LLMError
from extraction_review.scrutiny.rules import Defect
from extraction_review.scrutiny.schema import (
    Coverage,
    DefectResponse,
    EvidenceRef,
    LlmUsage,
    apply_status_policy,
    apply_undetermined_policy,
    build_finding,
)
from extraction_review.visual.attach import (
    attach_visual_localizations,
    visual_needs_for_defect,
)
from extraction_review.visual.detect import (
    _parse_vision_json,
    detect_visual_marks,
    parse_vision_elements,
    safe_normalized_bbox,
)
from extraction_review.visual.pages import (
    global_page_sources,
    select_formality_pages,
)
from extraction_review.visual.references import ReferenceCatalog
from extraction_review.visual.render import render_page_jpeg
from extraction_review.visual.schema import empty_visual_index
from extraction_review.visual.store import (
    coerce_visual_index,
    dump_visual_index,
    visual_summary,
)


def _part(
    slot_id: str,
    names: list[str],
    *,
    file_url: str | None = None,
    file_id: str | None = None,
) -> SimpleNamespace:
    return SimpleNamespace(
        slot_id=slot_id,
        document_parts=tuple(names),
        file_url=file_url,
        file_id=file_id,
    )


def _defect(
    *,
    check_id: str = "D466",
    defect: str,
    requirement: str,
    inspect_parts: list[str],
    where_to_look: list[str] | None = None,
) -> Defect:
    return Defect(
        check_id=check_id,
        serial_no=int(check_id[1:]) if check_id[1:].isdigit() else 1,
        main_category="General/Global",
        defect=defect,
        requirement=requirement,
        where_to_look=where_to_look or inspect_parts,
        inspect_parts=inspect_parts,
        how_to_cure=["Cure the defect"],
        location_source="SCI Handbook",
    )


def test_formality_pages_last_page_only_when_present() -> None:
    parts = [
        _part("checklist", ["Advocate's Checklist"]),
        _part("petition", [MAIN_PETITION_PART]),
        _part("affidavit", ["Affidavit"]),
        _part("annexure_p1", ["Annexure P-1"]),
        _part("annexure_p2", ["Annexure P-2"]),
        _part("vakalatnama", ["Vakalatnama"]),
        _part("fees", ["Court Fees"]),
        _part("order", ["Impugned Order"]),
    ]
    pages_by_slot = {
        "checklist": {1: "a", 2: "b"},
        "petition": {1: "p1", 2: "p2", 3: "p3"},
        "affidavit": {1: "aff1", 2: "aff2"},
        "annexure_p1": {1: "a1", 2: "a2", 3: "a3"},
        "annexure_p2": {1: "b1", 2: "b2"},
        "vakalatnama": {1: "vak"},
        "fees": {1: "fee"},
        "order": {1: "o1", 2: "o2"},
    }
    sources = global_page_sources(parts, pages_by_slot)
    targets = select_formality_pages(sources)
    pages = {target.page: target.document_types for target in targets}
    # checklist 1-2, petition 3-5, affidavit 6-7, P-1 8-10, P-2 11-12,
    # vakalatnama 13, fees 14, order 15-16
    assert 2 in pages
    assert 1 not in pages
    assert 5 in pages
    assert 3 not in pages and 4 not in pages
    assert 7 in pages
    assert 6 not in pages
    assert 10 in pages and 8 not in pages and 9 not in pages
    assert 12 in pages and 11 not in pages
    assert 13 in pages
    assert 14 not in pages
    assert 15 not in pages and 16 not in pages
    assert MAIN_PETITION_PART in pages[5]
    assert "Affidavit" in pages[7]
    assert "Annexure P-1" in pages[10]
    assert "Annexure P-2" in pages[12]
    assert "Vakalatnama" in pages[13]


def test_missing_d093_part_is_not_sent() -> None:
    parts = [
        _part("petition", [MAIN_PETITION_PART]),
        _part("affidavit", ["Affidavit"]),
    ]
    pages_by_slot = {"petition": {1: "p1", 2: "p2"}, "affidavit": {1: "aff"}}
    targets = select_formality_pages(global_page_sources(parts, pages_by_slot))
    names = {name for target in targets for name in target.document_types}
    assert "Vakalatnama" not in names
    assert "Listing Proforma" not in names
    assert MAIN_PETITION_PART in names
    assert "Affidavit" in names


def test_safe_bbox_rejects_full_page_and_accepts_tight_box() -> None:
    assert safe_normalized_bbox({"x": 0, "y": 0, "width": 1, "height": 1}) is None
    box = safe_normalized_bbox({"x": 0.62, "y": 0.81, "w": 0.28, "h": 0.08})
    assert box == {"x": 0.62, "y": 0.81, "w": 0.28, "h": 0.08}
    listed = safe_normalized_bbox([0.565, 0.041, 0.156, 0.222])
    assert listed == {"x": 0.565, "y": 0.041, "w": 0.156, "h": 0.222}


def test_parse_vision_elements_keeps_signature_and_drops_page_wide_box() -> None:
    marks = parse_vision_elements(
        {
            "elements": [
                {
                    "probable_type": "signature",
                    "bbox_normalized": {
                        "x": 0.62,
                        "y": 0.81,
                        "width": 0.28,
                        "height": 0.08,
                    },
                    "signature_role": "advocate",
                    "confidence": 0.86,
                    "associated_label": "DRAWN & FILED BY",
                },
                {
                    "probable_type": "signature_like_mark",
                    "bbox_normalized": {
                        "x": 0.0,
                        "y": 0.0,
                        "width": 0.9,
                        "height": 0.4,
                    },
                    "signature_role": "advocate",
                    "confidence": 0.99,
                },
            ]
        },
        page_number=42,
        document_type=MAIN_PETITION_PART,
    )
    assert len(marks) == 1
    assert marks[0].marking_type == "signature_like_mark"
    assert marks[0].signature_role == "advocate"
    assert marks[0].bbox["x"] == 0.62


def test_petition_signature_defect_gets_advocate_mark_on_last_page() -> None:
    defect = _defect(
        defect="Name/signature of the Advocate for the petitioner not provided at required places",
        requirement="Form No. 28 shall contain the name/signature of the Advocate for the petitioner",
        inspect_parts=[MAIN_PETITION_PART],
    )
    needs = visual_needs_for_defect(defect)
    assert any(need.marking_type == "advocate_on_record_signature" for need in needs)
    visual_index = {
        "marks": [
            {
                "page": 20,
                "document_type": MAIN_PETITION_PART,
                "marking_type": "signature_like_mark",
                "signature_role": "advocate",
                "bbox": {"x": 0.62, "y": 0.81, "w": 0.28, "h": 0.08},
                "confidence": 0.86,
            },
            {
                "page": 40,
                "document_type": "Vakalatnama",
                "marking_type": "signature_like_mark",
                "signature_role": "deponent",
                "bbox": {"x": 0.2, "y": 0.7, "w": 0.2, "h": 0.06},
                "confidence": 0.9,
            },
        ],
        "targets": [
            {"page": 20, "document_types": [MAIN_PETITION_PART]},
            {"page": 40, "document_types": ["Vakalatnama"]},
        ],
    }
    localizations = attach_visual_localizations(defect, visual_index=visual_index)
    assert len(localizations) == 1
    found = localizations[0]
    assert found.page == 20
    assert found.document_type == MAIN_PETITION_PART
    assert found.marking_type == "advocate_on_record_signature"
    assert found.boxes_status == "matched"
    assert found.bounding_boxes[0].x == 0.62
    assert found.signature_role == "advocate"


def test_vakalatnama_deponent_mark_is_not_attached_to_petition_defect() -> None:
    defect = _defect(
        defect="The Advocate-on-Record signature is missing on the Main Petition",
        requirement="The last page of the Main Petition must bear the AOR signature",
        inspect_parts=[MAIN_PETITION_PART],
    )
    visual_index = {
        "marks": [
            {
                "page": 40,
                "document_type": "Vakalatnama",
                "marking_type": "signature_like_mark",
                "signature_role": "deponent",
                "bbox": {"x": 0.2, "y": 0.7, "w": 0.2, "h": 0.06},
                "confidence": 0.99,
            }
        ],
        "targets": [{"page": 20, "document_types": [MAIN_PETITION_PART]}],
    }
    localizations = attach_visual_localizations(defect, visual_index=visual_index)
    assert localizations[0].page == 20
    assert localizations[0].boxes_status == "page_only"
    assert localizations[0].bounding_boxes == []


def test_text_only_defect_gets_empty_visual_localizations() -> None:
    defect = _defect(
        defect="The cause title does not match the memo of parties",
        requirement="The cause title must match the memo of parties",
        inspect_parts=["Cover Page", "Memo of Parties"],
    )
    assert visual_needs_for_defect(defect) == []
    assert attach_visual_localizations(defect, visual_index={"marks": []}) == []


def test_notary_defect_matches_seal_or_page_only_without_inventing_box() -> None:
    defect = _defect(
        check_id="D080",
        defect="The affidavit is not attested with a notary seal",
        requirement="The affidavit must be sworn before a notary or oath commissioner",
        inspect_parts=["Affidavit"],
    )
    empty = attach_visual_localizations(
        defect,
        visual_index={
            "targets": [
                {"page": 12, "document_types": ["Affidavit"]},
                {"page": 13, "document_types": ["Affidavit"]},
            ]
        },
    )
    assert empty[0].document_type == "Affidavit"
    assert empty[0].marking_type == "notary_seal"
    assert empty[0].page == 13
    assert empty[0].boxes_status == "page_only"
    assert empty[0].bounding_boxes == []

    matched = attach_visual_localizations(
        defect,
        visual_index={
            "marks": [
                {
                    "page": 13,
                    "document_type": "Affidavit",
                    "marking_type": "ordinary_seal_or_stamp",
                    "signature_role": "not_applicable",
                    "bbox": {"x": 0.7, "y": 0.75, "w": 0.18, "h": 0.12},
                    "confidence": 0.7,
                    "visible_text": "NOTARY GOVERNMENT OF INDIA",
                    "associated_label": "BEFORE ME",
                }
            ],
            "targets": [{"page": 13, "document_types": ["Affidavit"]}],
        },
    )
    assert matched[0].boxes_status == "matched"
    assert matched[0].bounding_boxes[0].w == 0.18


def test_build_finding_does_not_change_status_when_attaching_marks() -> None:
    defect = _defect(
        defect="Name/signature of the Advocate for the petitioner not provided",
        requirement="The Main Petition must bear the Advocate signature",
        inspect_parts=[MAIN_PETITION_PART],
    )
    response = apply_status_policy(
        DefectResponse(
            check_id=defect.check_id,
            status="defect_found",
            confidence=0.91,
            summary="The last page of the Main Petition has no AOR signature.",
            reasoning="Page 20 was checked and the signature block is blank.",
            evidence=[EvidenceRef(page=20, quote="DRAWN AND FILED BY")],
            suggested_fix="Sign the last page",
            fix_rationale="The AOR must sign Form 28",
        )
    )
    finding = build_finding(
        defect,
        response,
        evidence_ids=[],
        coverage=Coverage(chunks_reviewed=1, pages_reviewed=[20]),
        visual_index={
            "marks": [
                {
                    "page": 20,
                    "document_type": MAIN_PETITION_PART,
                    "marking_type": "signature_like_mark",
                    "signature_role": "advocate",
                    "bbox": {"x": 0.6, "y": 0.8, "w": 0.2, "h": 0.07},
                    "confidence": 0.8,
                }
            ],
            "targets": [{"page": 20, "document_types": [MAIN_PETITION_PART]}],
        },
    )
    assert finding.status == "defect_found"
    assert finding.visual_localizations[0].boxes_status == "matched"
    undetermined = apply_undetermined_policy(
        defect,
        DefectResponse(
            check_id=defect.check_id,
            status="not_determined",
            confidence=0.2,
            summary="Cannot see the signature.",
            reasoning="The page is visual.",
            evidence=[],
            suggested_fix=None,
            fix_rationale=None,
        ),
        [
            {
                "page": 20,
                "document_part": MAIN_PETITION_PART,
                "text": "DRAWN AND FILED BY",
            }
        ],
    )
    finding2 = build_finding(
        defect,
        undetermined,
        evidence_ids=[],
        coverage=Coverage(chunks_reviewed=1, pages_reviewed=[20]),
        visual_index={
            "targets": [{"page": 20, "document_types": [MAIN_PETITION_PART]}],
        },
    )
    assert finding2.status == undetermined.status
    assert finding2.visual_localizations[0].boxes_status == "page_only"


def test_dump_and_coerce_visual_index_round_trip() -> None:
    payload = dump_visual_index(
        {
            "marks": [
                {
                    "page": "42",
                    "document_type": MAIN_PETITION_PART,
                    "marking_type": "signature_like_mark",
                    "signature_role": "advocate",
                    "bbox": {"x": 0.1, "y": 0.2, "w": 0.3, "h": 0.05},
                    "confidence": 0.5,
                }
            ],
            "pages": ["42"],
            "targets": [{"page": "42", "document_types": [MAIN_PETITION_PART]}],
        }
    )
    coerced = coerce_visual_index(payload)
    assert coerced["schema"] == "visual_index_v1"
    assert coerced["marks"][0]["page"] == 42
    assert coerced["targets"][0]["page"] == 42


def test_reference_catalog_loads_checksum_pinned_sheets() -> None:
    assets = ReferenceCatalog().all_assets()
    names = {asset.class_name for asset in assets}
    roles = {asset.expected_role for asset in assets}
    assert "notary_seal" in names
    assert "signature_like_mark" in names
    assert {"advocate", "deponent"} <= roles
    assert all(asset.content.startswith(b"\x89PNG") for asset in assets)


def test_render_page_jpeg_returns_jpeg() -> None:
    import pymupdf

    document = pymupdf.open()
    page = document.new_page()
    page.insert_text((72, 72), "Formality page")
    pdf_bytes = document.tobytes()
    document.close()
    image = render_page_jpeg(pdf_bytes, 1, dpi=72)
    assert image.startswith(b"\xff\xd8\xff")


D093_INSPECT = [
    "Advocate's Checklist",
    "Listing Proforma",
    MAIN_PETITION_PART,
    "AOR's Certificate",
    "Application",
    "Memo of Parties",
    "Memo of Appearance",
    "Vakalatnama",
    "Filing Memo",
]


def test_certified_copy_defects_are_ocr_only() -> None:
    stamp_index = {
        "marks": [
            {
                "page": 4,
                "document_type": "Impugned Order",
                "marking_type": "government_stamp",
                "signature_role": "not_applicable",
                "bbox": {"x": 0.7, "y": 0.7, "w": 0.15, "h": 0.12},
                "confidence": 0.9,
            }
        ],
        "targets": [{"page": 4, "document_types": ["Impugned Order"]}],
    }
    for check_id, defect, requirement in (
        (
            "D248",
            "Certified copy of impugned order is not filed. Application seeking exemption from filing certified copy of the impugned order not filed",
            "Certified copy of impugned order must be filed. Where such copy is not filed an application seeking exemption from filing certified copy of the impugned order must be filed",
        ),
        (
            "D267",
            "Certified copy of impugned order/judgement is not filed and an Application seeking exemption from filing certified copy of the impugned order/judgement is not filed.",
            "Where certified copy of impugned order/judgement is not filed and an Application seeking exemption from filing certified copy of the impugned order/judgement must be filed.",
        ),
    ):
        item = _defect(
            check_id=check_id,
            defect=defect,
            requirement=requirement,
            inspect_parts=["Impugned Order", "Application"],
            where_to_look=[
                "Check whether the impugned order/judgement is certified or not (Look for certification stamp on the judgement)",
                "If Certification is missing, Check Application seeking exemption from filing the certified copy",
            ],
        )
        assert visual_needs_for_defect(item) == []
        assert attach_visual_localizations(item, visual_index=stamp_index) == []


def test_court_fee_and_vakalatnama_seal_are_not_stamp_needs() -> None:
    fees = _defect(
        defect="Court fee stamps are insufficient",
        requirement="The prescribed court-fee stamp must be affixed",
        inspect_parts=["Court Fees"],
    )
    assert visual_needs_for_defect(fees) == []
    seal = _defect(
        defect="The Vakalatnama does not bear the seal of the advocate",
        requirement="The Vakalatnama must bear the seal",
        inspect_parts=["Vakalatnama"],
    )
    assert visual_needs_for_defect(seal) == []


def test_d093_attaches_aor_marks_on_present_last_pages() -> None:
    defect = _defect(
        check_id="D093",
        defect=(
            "The petition is not duly signed by the Advocate-on-Record/"
            "Party-in-Person at the required places, including the Declaration, "
            "Advocate's Checklist, Listing Proforma/Proforma for First Listing, "
            "Main Petition, Advocate's Certificate, Application(s), Filing Memo, "
            "Memo of Parties, Vakalatnama, Memo of Appearance, and the last page "
            "of each Annexure."
        ),
        requirement=(
            "AOR to sign each page-digitally (Adobe sign) at the required places, "
            "including the Declaration, Advocate's Checklist, Listing Proforma/"
            "Proforma for First Listing, Main Petition, Advocate's Certificate, "
            "Application(s), Filing Memo, Memo of Parties, Vakalatnama, Memo of "
            "Appearance, and the last page of each Annexure."
        ),
        inspect_parts=D093_INSPECT,
    )
    visual_index = {
        "marks": [
            {
                "page": 20,
                "document_type": MAIN_PETITION_PART,
                "marking_type": "signature_like_mark",
                "signature_role": "advocate",
                "bbox": {"x": 0.62, "y": 0.81, "w": 0.28, "h": 0.08},
                "confidence": 0.86,
            },
            {
                "page": 2,
                "document_type": "Advocate's Checklist",
                "marking_type": "signature_like_mark",
                "signature_role": "advocate",
                "bbox": {"x": 0.5, "y": 0.8, "w": 0.2, "h": 0.07},
                "confidence": 0.8,
            },
            {
                "page": 99,
                "document_type": "Court Fees",
                "marking_type": "government_stamp",
                "signature_role": "not_applicable",
                "bbox": {"x": 0.1, "y": 0.1, "w": 0.2, "h": 0.2},
                "confidence": 0.99,
            },
        ],
        "targets": [
            {"page": 2, "document_types": ["Advocate's Checklist"]},
            {"page": 20, "document_types": [MAIN_PETITION_PART]},
        ],
    }
    localizations = attach_visual_localizations(defect, visual_index=visual_index)
    matched = {
        item.document_type: item
        for item in localizations
        if item.boxes_status == "matched"
    }
    assert MAIN_PETITION_PART in matched
    assert "Advocate's Checklist" in matched
    assert all(item.document_type != "Court Fees" for item in localizations)
    assert all(item.marking_type != "court_fee_stamp" for item in localizations)


def test_d121_notary_uses_affidavit_last_page_not_first() -> None:
    defect = _defect(
        check_id="D121",
        defect=(
            "Document such as affidavit, affidavit in opposition, rejoinder "
            "affidavit, etc., do not mention the date on which they are sworn "
            "before the notary or oath commissioner"
        ),
        requirement=(
            "the date on which it is sworn before the notary or oath commissioner "
            "must be mentioned on the affidavit."
        ),
        inspect_parts=["Affidavit"],
        where_to_look=[
            "Check the first page of the affidavit",
            "If the date is not found on the first page, check the subsequent/final page",
        ],
    )
    needs = visual_needs_for_defect(defect)
    assert {need.marking_type for need in needs} == {"notary_seal"}
    assert all(need.document_type == "Affidavit" for need in needs)
    localizations = attach_visual_localizations(
        defect,
        visual_index={
            "targets": [{"page": 13, "document_types": ["Affidavit"]}],
        },
    )
    assert localizations[0].page == 13
    assert localizations[0].marking_type == "notary_seal"
    assert localizations[0].boxes_status == "page_only"


def test_d077_vakalatnama_does_not_attach_affidavit_notary() -> None:
    defect = _defect(
        check_id="D077",
        defect=(
            "The AOR fails to:- (i) either certify that the Vakalatnama was "
            "executed in his presence or (ii) Make an endorsement that he has "
            "satisfied himself about it's due execution before a notary or advocate"
        ),
        requirement=(
            "The AOR shall:- (i) either certify that the Vakalatnama was executed "
            "in his presence or (ii) Make an endorsement that he has satisfied "
            "himself about it's due execution before a notary or advocate"
        ),
        inspect_parts=["Vakalatnama"],
        where_to_look=[
            "Go to the Vakalatnama and verify that it is signed/executed by the petitioner(s)",
            "Check that it has been accepted by the AOR",
        ],
    )
    needs = visual_needs_for_defect(defect)
    assert {need.document_type for need in needs} == {"Vakalatnama"}
    types = {need.marking_type for need in needs}
    assert "advocate_on_record_signature" in types
    assert "notary_seal" in types
    assert "executant_signature" not in types
    localizations = attach_visual_localizations(
        defect,
        visual_index={
            "marks": [
                {
                    "page": 12,
                    "document_type": "Affidavit",
                    "marking_type": "notary_seal",
                    "signature_role": "not_applicable",
                    "bbox": {"x": 0.7, "y": 0.7, "w": 0.1, "h": 0.1},
                    "confidence": 0.99,
                },
                {
                    "page": 40,
                    "document_type": "Vakalatnama",
                    "marking_type": "signature_like_mark",
                    "signature_role": "advocate",
                    "bbox": {"x": 0.2, "y": 0.8, "w": 0.2, "h": 0.06},
                    "confidence": 0.8,
                },
            ],
            "targets": [
                {"page": 12, "document_types": ["Affidavit"]},
                {"page": 40, "document_types": ["Vakalatnama"]},
            ],
        },
    )
    assert all(item.document_type == "Vakalatnama" for item in localizations)
    assert any(
        item.marking_type == "advocate_on_record_signature"
        and item.boxes_status == "matched"
        for item in localizations
    )


def test_d115_and_d127_vakalatnama_executant_marks() -> None:
    d115 = _defect(
        check_id="D115",
        defect=(
            "The Vakalatnama not properly executed by the petitioners/appellants "
            "or not accepted and identified by the Advocate"
        ),
        requirement=(
            "The Vakalatnama has been properly executed by the petitioners/"
            "appellants and accepted and identified by the Advocate"
        ),
        inspect_parts=["Vakalatnama"],
        where_to_look=[
            "signature or thumb impression of each executant",
            "Advocate's endorsement accepting the Vakalatnama",
        ],
    )
    types_115 = {need.marking_type for need in visual_needs_for_defect(d115)}
    assert types_115 == {
        "executant_signature",
        "advocate_on_record_signature",
    }
    d127 = _defect(
        check_id="D127",
        defect=(
            "Where several persons sign a single Vakalatnama, failure to affix "
            "the signature seriatim without mentioning their serial numbers or "
            "names in brackets"
        ),
        requirement=(
            "Where several persons sign a single Vakalatnama, affix the signature "
            "seriatim and mention their serial numbers or names in brackets"
        ),
        inspect_parts=["Vakalatnama"],
    )
    types_127 = {need.marking_type for need in visual_needs_for_defect(d127)}
    assert types_127 == {"executant_signature"}
    localizations = attach_visual_localizations(
        d127,
        visual_index={
            "marks": [
                {
                    "page": 40,
                    "document_type": "Vakalatnama",
                    "marking_type": "signature_like_mark",
                    "signature_role": "petitioner",
                    "bbox": {"x": 0.2, "y": 0.6, "w": 0.15, "h": 0.05},
                    "confidence": 0.7,
                },
                {
                    "page": 40,
                    "document_type": "Vakalatnama",
                    "marking_type": "signature_like_mark",
                    "signature_role": "petitioner",
                    "bbox": {"x": 0.2, "y": 0.7, "w": 0.15, "h": 0.05},
                    "confidence": 0.6,
                },
            ],
            "targets": [{"page": 40, "document_types": ["Vakalatnama"]}],
        },
    )
    assert len(localizations) == 2
    assert all(item.marking_type == "executant_signature" for item in localizations)
    assert all(item.page == 40 for item in localizations)


def test_d174_and_d131_advocate_signature_pages() -> None:
    d174 = _defect(
        check_id="D174",
        defect="Name/signature of the Advocate for the petitioner not provided at required places",
        requirement="Form No. 28 shall contain the name/signature of the Advocate for the petitioner at the required places",
        inspect_parts=[MAIN_PETITION_PART, "Listing Proforma", "Application"],
        where_to_look=[
            "Check the execution details at the end of Form 28",
            "Check the Listing Proforma for the name/signature",
            "Check the Certificate for the name/signature of the Advocate",
            "Check all Applications filed with the SLP",
        ],
    )
    types = {
        (need.document_type, need.marking_type)
        for need in visual_needs_for_defect(d174)
    }
    assert (MAIN_PETITION_PART, "advocate_on_record_signature") in types
    assert ("Listing Proforma", "advocate_on_record_signature") in types
    assert ("AOR's Certificate", "advocate_on_record_signature") in types
    assert ("Application", "advocate_on_record_signature") in types
    d131 = _defect(
        check_id="D131",
        defect="Complete Listing proforma/Proforma for First Listing has not been filled in or signed or not included altogether",
        requirement="Complete Listing proforma/Proforma for First Listing must be filled in, signed and be included",
        inspect_parts=["Listing Proforma"],
        where_to_look=[
            "Check that every column is filled and that it bears the signature of the AOR with the date."
        ],
    )
    needs_131 = visual_needs_for_defect(d131)
    assert {need.document_type for need in needs_131} == {"Listing Proforma"}
    assert {need.marking_type for need in needs_131} == {"advocate_on_record_signature"}
    localizations = attach_visual_localizations(
        d131,
        visual_index={
            "targets": [{"page": 3, "document_types": ["Listing Proforma"]}],
        },
    )
    assert localizations[0].page == 3
    assert localizations[0].boxes_status == "page_only"


def test_visual_index_error_is_structured() -> None:
    skipped = empty_visual_index(status="skipped", error="vision_disabled")
    assert skipped["schema"] == "visual_index_v1"
    assert skipped["status"] == "skipped"
    assert skipped["marks"] == []
    assert skipped["failures"] == []
    dumped = dump_visual_index(
        {
            "status": "error",
            "error": "missing_pdf",
            "marks": [],
            "failures": [
                {
                    "page": 1,
                    "slot_id": "petition",
                    "document_types": [MAIN_PETITION_PART],
                    "reason": "missing_pdf",
                    "detail": "slot has no file_url or file_id",
                }
            ],
        }
    )
    coerced = coerce_visual_index(dumped)
    assert coerced["status"] == "error"
    assert coerced["error"] == "missing_pdf"
    assert coerced["marks"] == []
    assert coerced["failures"][0]["reason"] == "missing_pdf"
    summary = visual_summary(coerced)
    assert summary["target_count"] == 0
    assert summary["failures"][0]["reason"] == "missing_pdf"
    missing = empty_visual_index(status="missing")
    assert missing["status"] == "skipped"


def test_missing_listing_proforma_omitted_from_targets() -> None:
    parts = [_part("petition", [MAIN_PETITION_PART])]
    targets = select_formality_pages(global_page_sources(parts, {"petition": {1: "p"}}))
    names = [name for target in targets for name in target.document_types]
    assert "Listing Proforma" not in names
    assert MAIN_PETITION_PART in names


def test_formality_targets_carry_file_id() -> None:
    parts = [_part("petition", [MAIN_PETITION_PART], file_id="dfl-petition-1")]
    targets = select_formality_pages(global_page_sources(parts, {"petition": {1: "p"}}))
    assert targets[0].file_id == "dfl-petition-1"
    assert targets[0].file_url is None


def _one_page_pdf() -> bytes:
    import pymupdf

    document = pymupdf.open()
    page = document.new_page()
    page.insert_text((72, 72), "Formality page")
    pdf_bytes = document.tobytes()
    document.close()
    return pdf_bytes


def _enable_vision(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(
        "extraction_review.visual.detect.visual_detection_enabled",
        lambda: True,
    )

    async def fake_analyze(**_kwargs):
        return [], None, LlmUsage()

    monkeypatch.setattr(
        "extraction_review.visual.detect._analyze_with_fallback",
        fake_analyze,
    )


@pytest.mark.asyncio
async def test_detect_missing_pdf_records_failure(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _enable_vision(monkeypatch)
    logs: list[str] = []
    parts = [_part("petition", [MAIN_PETITION_PART])]
    index = await detect_visual_marks(
        parts=parts,
        pages_by_slot={"petition": {1: "p"}},
        page_parts={1: [MAIN_PETITION_PART]},
        on_log=logs.append,
    )
    assert index["status"] == "error"
    assert index["error"] == "missing_pdf"
    assert index["marks"] == []
    assert index["failures"][0]["reason"] == "missing_pdf"
    assert index["targets"][0]["page"] == 1
    assert any("file_id=no" in line for line in logs)
    assert any("missing_pdf" in line for line in logs)


@pytest.mark.asyncio
async def test_detect_uses_file_id_when_url_missing(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _enable_vision(monkeypatch)
    pdf_bytes = _one_page_pdf()
    calls: list[str] = []

    async def fake_download(file_id: str) -> bytes:
        calls.append(file_id)
        return pdf_bytes

    monkeypatch.setattr(
        "extraction_review.visual.detect._default_download_file_id",
        fake_download,
    )
    parts = [_part("petition", [MAIN_PETITION_PART], file_id="dfl-petition-1")]
    index = await detect_visual_marks(
        parts=parts,
        pages_by_slot={"petition": {1: "p"}},
        page_parts={1: [MAIN_PETITION_PART]},
    )
    assert calls == ["dfl-petition-1"]
    assert index["status"] == "ok"
    assert index["error"] is None
    assert index["failures"] == []
    assert index["targets"][0]["document_types"] == [MAIN_PETITION_PART]


@pytest.mark.asyncio
async def test_detect_mixed_success_keeps_status_ok(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _enable_vision(monkeypatch)
    pdf_bytes = _one_page_pdf()
    calls: list[str] = []

    async def fake_download(file_id: str) -> bytes:
        calls.append(file_id)
        return pdf_bytes

    parts = [
        _part("petition", [MAIN_PETITION_PART], file_id="dfl-petition-1"),
        _part("affidavit", ["Affidavit"]),
    ]
    index = await detect_visual_marks(
        parts=parts,
        pages_by_slot={"petition": {1: "p"}, "affidavit": {1: "a"}},
        page_parts={1: [MAIN_PETITION_PART], 2: ["Affidavit"]},
        download_file_id=fake_download,
    )
    assert calls == ["dfl-petition-1"]
    assert index["status"] == "ok"
    assert index["error"] is None
    reasons = {item["reason"] for item in index["failures"]}
    assert reasons == {"missing_pdf"}
    assert len(index["failures"]) == 1
    assert index["failures"][0]["slot_id"] == "affidavit"
    assert len(index["targets"]) == 2
    summary = visual_summary(index)
    assert summary["status"] == "ok"
    assert summary["target_count"] == 2
    assert summary["failures"][0]["reason"] == "missing_pdf"


@pytest.mark.asyncio
async def test_detect_api_error_is_vision_failed(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(
        "extraction_review.visual.detect.visual_detection_enabled",
        lambda: True,
    )
    monkeypatch.setattr(
        "extraction_review.visual.detect.vision_fallback_model",
        lambda: None,
    )

    async def boom(**_kwargs):
        raise LLMError("401 unauthorized")

    monkeypatch.setattr(
        "extraction_review.visual.detect.analyze_page_image",
        boom,
    )

    async def fake_download(_file_id: str) -> bytes:
        return _one_page_pdf()

    index = await detect_visual_marks(
        parts=[_part("petition", [MAIN_PETITION_PART], file_id="dfl-petition-1")],
        pages_by_slot={"petition": {1: "p"}},
        page_parts={1: [MAIN_PETITION_PART]},
        download_file_id=fake_download,
    )
    assert index["status"] == "error"
    assert index["error"] == "vision_failed"
    assert index["marks"] == []
    assert index["failures"][0]["reason"] == "vision_failed"
    assert "401" in (index["failures"][0]["detail"] or "")


def test_parse_vision_empty_array_is_zero_marks() -> None:
    assert _parse_vision_json("[]") == []
    assert (
        parse_vision_elements("[]", page_number=31, document_type="Annexure P-3") == []
    )
    assert (
        parse_vision_elements(
            {"elements": []}, page_number=31, document_type="Annexure P-3"
        )
        == []
    )


def test_parse_vision_top_level_array_list_bbox() -> None:
    marks = parse_vision_elements(
        [
            {
                "probable_type": "notary_seal",
                "bbox_normalized": [0.565, 0.041, 0.156, 0.222],
                "confidence": 0.8,
            }
        ],
        page_number=26,
        document_type="Affidavit",
    )
    assert len(marks) == 1
    assert marks[0].marking_type == "notary_seal"
    assert marks[0].bbox["x"] == 0.565


def test_parse_vision_truncated_array_keeps_complete_object() -> None:
    truncated = (
        '[ { "probable_type": "notary_seal", "bbox_normalized": [0.565, 0.041, 0.156, 0.222], '
        '"confidence": 0.9, "associated_label": "NOTARY" }, { "probable_type": "table", '
        '"bbox_normalized": [0.1, 0.1'
    )
    parsed = _parse_vision_json(truncated)
    assert isinstance(parsed, list)
    marks = parse_vision_elements(truncated, page_number=26, document_type="Affidavit")
    assert [mark.marking_type for mark in marks] == ["notary_seal"]


def _openrouter_body(content: str, cost: float) -> dict:
    return {
        "id": f"gen-{cost}",
        "model": "google/gemini-3.8-flash",
        "usage": {
            "prompt_tokens": 10,
            "completion_tokens": 4,
            "total_tokens": 14,
            "cost": cost,
        },
        "choices": [{"message": {"content": content}}],
    }


def _stub_openrouter_post(monkeypatch: pytest.MonkeyPatch, bodies: list[dict]) -> None:
    queue = list(bodies)

    async def fake_post(self, url, **_kwargs):
        del self, url
        payload = queue.pop(0)
        return SimpleNamespace(
            status_code=200,
            text="",
            json=lambda: payload,
            raise_for_status=lambda: None,
        )

    monkeypatch.setattr(
        "extraction_review.visual.detect.visual_detection_enabled",
        lambda: True,
    )
    monkeypatch.setattr(
        "extraction_review.visual.detect.vision_fallback_model",
        lambda: None,
    )
    monkeypatch.setattr(
        "extraction_review.visual.detect._headers",
        lambda: {
            "Authorization": "Bearer test-key",
            "Content-Type": "application/json",
        },
    )
    monkeypatch.setattr("httpx.AsyncClient.post", fake_post)


@pytest.mark.asyncio
async def test_detect_empty_array_body_is_ok_not_vision_failed(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _stub_openrouter_post(monkeypatch, [_openrouter_body("[]", 0.01)])

    async def fake_download(_file_id: str) -> bytes:
        return _one_page_pdf()

    index = await detect_visual_marks(
        parts=[_part("petition", [MAIN_PETITION_PART], file_id="dfl-petition-1")],
        pages_by_slot={"petition": {1: "p"}},
        page_parts={1: [MAIN_PETITION_PART]},
        download_file_id=fake_download,
    )
    assert index["status"] == "ok"
    assert index["error"] is None
    assert index["failures"] == []
    assert index["marks"] == []
    assert index["usage"]["cost_usd"] == pytest.approx(0.01)


@pytest.mark.asyncio
async def test_detect_sums_openrouter_cost_across_pages(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _stub_openrouter_post(
        monkeypatch,
        [
            _openrouter_body("[]", 0.01),
            _openrouter_body("[]", 0.02),
        ],
    )

    async def fake_download(_file_id: str) -> bytes:
        return _one_page_pdf()

    index = await detect_visual_marks(
        parts=[
            _part("petition", [MAIN_PETITION_PART], file_id="dfl-petition-1"),
            _part("affidavit", ["Affidavit"], file_id="dfl-affidavit-1"),
        ],
        pages_by_slot={"petition": {1: "p"}, "affidavit": {1: "a"}},
        page_parts={1: [MAIN_PETITION_PART], 2: ["Affidavit"]},
        download_file_id=fake_download,
    )
    assert index["status"] == "ok"
    assert index["usage"]["cost_usd"] == pytest.approx(0.03)
    assert index["usage"]["calls"] == 2
    summary = visual_summary(index)
    assert summary["usage"]["cost_usd"] == pytest.approx(0.03)
