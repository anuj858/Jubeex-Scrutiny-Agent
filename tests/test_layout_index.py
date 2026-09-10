"""Layout index: sidecar compacting, page remap, quote-to-line boxes."""

import pytest

from extraction_review.layout_index import (
    boxes_for_quote,
    compact_sidecar_pages,
    dump_layout_index,
    grounded_items_url,
    load_layout_index,
    locate_quote,
    merge_granular_bboxes,
    parse_sidecar_jsonl,
    quote_in_text,
    stitch_slot_layouts,
)
from extraction_review.llm import strict_json_schema
from extraction_review.scrutiny.rules import get_catalogue
from extraction_review.scrutiny.schema import (
    Coverage,
    DefectResponse,
    EvidenceRef,
    _as_bounding_boxes,
    apply_evidence_pages,
    attach_evidence_boxes,
    build_finding,
)
from extraction_review.split_upload import SplitPartInput, UploadSlot, UploadTypeCatalog


def _rows_to_sidecar(rows: list[list[tuple[str, float, float, float, float]]]) -> dict:
    md_parts: list[str] = []
    lines: list[dict] = []
    cursor = 0
    for row in rows:
        words: list[dict] = []
        for token, x, y, w, h in row:
            if md_parts:
                cursor += 1
            start = cursor
            cursor += len(token)
            md_parts.append(token)
            words.append(
                {
                    "span": [start, cursor],
                    "bbox": {"x": x, "y": y, "w": w, "h": h},
                }
            )
        lines.append(
            {
                "span": [words[0]["span"][0], words[-1]["span"][1]],
                "bbox": {
                    "x": words[0]["bbox"]["x"],
                    "y": words[0]["bbox"]["y"],
                    "w": words[-1]["bbox"]["x"]
                    + words[-1]["bbox"]["w"]
                    - words[0]["bbox"]["x"],
                    "h": words[0]["bbox"]["h"],
                },
                "words": words,
            }
        )
    return {
        "page_number": 1,
        "page_width": 1.0,
        "page_height": 1.0,
        "success": True,
        "items": [
            {
                "type": "text",
                "md": " ".join(md_parts),
                "grounding": {"source": "md", "lines": lines},
            }
        ],
    }


MID_LINE_ROWS = [
    [
        ("the", 0.05, 0.61, 0.06, 0.028),
        ("petitioner", 0.12, 0.61, 0.28, 0.028),
        ("hereby", 0.42, 0.61, 0.14, 0.028),
        ("undertakes", 0.57, 0.61, 0.18, 0.028),
        ("that", 0.76, 0.61, 0.07, 0.028),
    ],
    [
        ("the", 0.12, 0.64, 0.06, 0.028),
        ("facts", 0.19, 0.64, 0.10, 0.028),
        ("stated", 0.30, 0.64, 0.12, 0.028),
        ("are", 0.43, 0.64, 0.07, 0.028),
    ],
    [
        ("true", 0.12, 0.70, 0.08, 0.028),
        ("to", 0.21, 0.70, 0.04, 0.028),
        ("my", 0.26, 0.70, 0.05, 0.028),
        ("knowledge", 0.32, 0.70, 0.20, 0.028),
    ],
]

QUOTE = "hereby undertakes that the facts stated are true to my"


def _mid_line_layout(global_page: int = 12) -> dict[int, dict]:
    compact = compact_sidecar_pages(
        [_rows_to_sidecar(MID_LINE_ROWS)], slot_id="vakalatnama"
    )
    page = compact[1]
    page["slot_id"] = "vakalatnama"
    page["local_page"] = 1
    return {global_page: page}


def test_compact_normalizes_pdf_points_to_unit_square() -> None:
    row = _rows_to_sidecar([[("hello", 42.0, 61.0, 14.0, 2.8)]])
    row["page_width"] = 100
    row["page_height"] = 100
    layout = compact_sidecar_pages([row])
    word = layout[1]["words"][0]
    assert word["t"] == "hello"
    assert word["x"] == 0.42
    assert word["y"] == 0.61
    assert word["w"] == 0.14
    assert word["h"] == 0.028


def test_merge_granular_bboxes_adds_word_and_line() -> None:
    kwargs = merge_granular_bboxes({"file_id": "abc", "tier": "agentic"})
    assert kwargs["output_options"]["granular_bboxes"] == ["word", "line"]


def test_merge_granular_bboxes_skips_hosted_configuration() -> None:
    kwargs = merge_granular_bboxes({"configuration_id": "cfg-1", "file_id": "abc"})
    assert "output_options" not in kwargs


def test_grounded_items_url_reads_metadata() -> None:
    url = grounded_items_url(
        {
            "result_content_metadata": {
                "grounded_items": {
                    "exists": True,
                    "presigned_url": "https://example/sidecar.jsonl",
                }
            }
        }
    )
    assert url == "https://example/sidecar.jsonl"
    assert (
        grounded_items_url(
            {"result_content_metadata": {"grounded_items": {"exists": False}}}
        )
        is None
    )


def test_mid_line_quote_returns_three_clipped_boxes() -> None:
    layout = _mid_line_layout(12)
    boxes, status = boxes_for_quote(QUOTE, 12, layout)
    assert status == "matched"
    assert len(boxes) == 3
    first, middle, last = boxes
    assert first["page"] == 12
    assert first["x"] == 0.42
    assert first["x"] > 0.12
    assert first["w"] == round(0.76 + 0.07 - 0.42, 6)
    assert middle["x"] == 0.12
    assert middle["w"] == round(0.43 + 0.07 - 0.12, 6)
    assert last["x"] == 0.12
    assert last["x"] + last["w"] == round(0.26 + 0.05, 6)
    assert last["x"] + last["w"] < 0.32 + 0.20


def test_unknown_quote_is_page_only() -> None:
    boxes, status = boxes_for_quote(
        "this text is not on the page", 12, _mid_line_layout()
    )
    assert boxes == []
    assert status == "page_only"


def test_missing_layout_is_unavailable() -> None:
    boxes, status = boxes_for_quote(QUOTE, 12, None)
    assert boxes == []
    assert status == "unavailable"
    boxes, status = boxes_for_quote(QUOTE, 12, {})
    assert status == "unavailable"


def test_slot_local_page_stitches_to_global_page_12() -> None:
    catalog = UploadTypeCatalog(
        filing_type="SLP_CIVIL",
        label="Test",
        slots=(
            UploadSlot(id="cover_page", label="Cover", parts=("Cover Page",)),
            UploadSlot(
                id="vakalatnama_appearance", label="Vakalatnama", parts=("Vakalatnama",)
            ),
        ),
    )
    parts = [
        SplitPartInput(slot_id="cover_page", file_id="f-cover"),
        SplitPartInput(slot_id="vakalatnama_appearance", file_id="f-vak"),
    ]
    pages_by_slot = {
        "cover_page": {index: f"cover {index}" for index in range(1, 12)},
        "vakalatnama_appearance": {1: "vakalatnama"},
    }
    local = compact_sidecar_pages(
        [_rows_to_sidecar(MID_LINE_ROWS)], slot_id="vakalatnama_appearance"
    )
    stitched = stitch_slot_layouts(
        catalog,
        parts,
        pages_by_slot,
        {"vakalatnama_appearance": local},
    )
    assert 12 in stitched
    assert stitched[12]["local_page"] == 1
    assert stitched[12]["slot_id"] == "vakalatnama_appearance"
    boxes, status = boxes_for_quote(QUOTE, 12, stitched)
    assert status == "matched"
    assert boxes[0]["page"] == 12


def test_parse_sidecar_jsonl_skips_blank_lines() -> None:
    pages = parse_sidecar_jsonl(
        '\n{"page_number": 1, "success": true, "items": []}\n\n'
    )
    assert len(pages) == 1
    assert pages[0]["page_number"] == 1


def test_dump_layout_index_uses_string_page_keys() -> None:
    payload = dump_layout_index(_mid_line_layout(12))
    assert payload["schema"] == "layout_index_v1"
    assert "12" in payload["pages"]


def test_defect_response_schema_has_no_bounding_boxes() -> None:
    schema = strict_json_schema(DefectResponse)
    dumped = str(schema)
    defs = schema.get("$defs") or schema.get("definitions") or {}
    properties = (defs.get("EvidenceRef") or {}).get("properties") or {}
    assert set(properties) == {"page", "quote"}
    assert "FindingEvidence" not in dumped
    assert "BoundingBox" not in dumped
    assert "bounding_boxes" not in dumped
    assert "boxes_status" not in dumped


def test_apply_evidence_pages_still_snaps_page_from_chunk() -> None:
    response = DefectResponse(
        check_id="D003",
        status="compliant",
        confidence=0.9,
        summary="Checklist is present.",
        reasoning="The Advocate's Check List is on the excerpt page.",
        evidence=[EvidenceRef(page=99, quote="ADVOCATE'S CHECK LIST")],
        suggested_fix=None,
        fix_rationale=None,
    )
    grounded = apply_evidence_pages(
        response,
        [
            {
                "chunk_kind": "page",
                "page": 2,
                "document_part": "Advocate's Checklist",
                "text": "ADVOCATE'S CHECK LIST yes no",
            }
        ],
    )
    assert grounded.evidence[0].page == 2
    assert grounded.evidence[0].model_dump() == {
        "page": 2,
        "quote": "ADVOCATE'S CHECK LIST",
    }


def test_build_finding_attaches_boxes_from_layout() -> None:
    catalogue = get_catalogue()
    layout = _mid_line_layout(12)
    finding = build_finding(
        catalogue.defect("D003"),
        DefectResponse(
            check_id="D003",
            status="defect_found",
            confidence=0.9,
            summary="Vakalatnama undertaking is incomplete.",
            reasoning="The Vakalatnama quote is on filing page 12.",
            evidence=[EvidenceRef(page=12, quote=QUOTE)],
            suggested_fix="Complete the undertaking.",
            fix_rationale="Required.",
        ),
        evidence_ids=["p12"],
        coverage=Coverage(chunks_reviewed=1, pages_reviewed=[12]),
        chunks=[
            {
                "record_id": "p12",
                "chunk_kind": "page",
                "page": 12,
                "document_part": "Vakalatnama",
                "text": QUOTE + " knowledge",
            }
        ],
        layout=layout,
    )
    ref = finding.evidence[0]
    assert ref.boxes_status == "matched"
    assert len(ref.bounding_boxes) == 3
    assert ref.document_part == "Vakalatnama"
    assert ref.slot_id == "vakalatnama"
    assert ref.local_page == 1
    assert ref.bounding_boxes[0].x == 0.42


def test_build_finding_without_layout_is_unavailable() -> None:
    catalogue = get_catalogue()
    finding = build_finding(
        catalogue.defect("D003"),
        DefectResponse(
            check_id="D003",
            status="compliant",
            confidence=0.9,
            summary="Checklist is present.",
            reasoning="The Advocate's Check List is on filing page 2.",
            evidence=[EvidenceRef(page=2, quote="ADVOCATE'S CHECK LIST")],
            suggested_fix=None,
            fix_rationale=None,
        ),
        evidence_ids=[],
        coverage=Coverage(chunks_reviewed=1, pages_reviewed=[2]),
        chunks=[
            {
                "chunk_kind": "page",
                "page": 2,
                "document_part": "Advocate's Checklist",
                "text": "ADVOCATE'S CHECK LIST",
            }
        ],
    )
    assert finding.evidence[0].boxes_status == "unavailable"
    assert finding.evidence[0].bounding_boxes == []
    assert finding.evidence[0].document_part == "Advocate's Checklist"


def _sentence_layout(page: int, text: str, slot_id: str = "affidavit") -> dict:
    tokens = text.split()
    row = [(token, 0.08 + index * 0.03, 0.62, 0.028, 0.018) for index, token in enumerate(tokens)]
    compact = compact_sidecar_pages([_rows_to_sidecar([row])], slot_id=slot_id)
    payload = compact[1]
    payload["slot_id"] = slot_id
    payload["local_page"] = 1
    return {page: payload}


def test_null_page_quote_is_found_on_layout_page() -> None:
    layout = _sentence_layout(
        26,
        "Verified at Una on 17th day of April 2026 that the contents of Para 1 to 3 of the affidavit are true and correct to the best of my knowledge and belief.",
    )
    boxes, status, page = locate_quote(
        "Verified at Una on 17/4 day of April, 2026 that the contents of Para 1 to 3 of the affidavit are true and correct to the best of my knowledge and belief.",
        None,
        layout,
        prefer_pages=[21, 25, 26],
    )
    assert status == "matched"
    assert page == 26
    assert boxes
    assert boxes[0]["page"] == 26


def test_build_finding_fills_null_page_from_layout() -> None:
    catalogue = get_catalogue()
    layout = _sentence_layout(
        26,
        "Verified at Una on 17th day of April 2026 that the contents of Para 1 to 3 of the affidavit are true and correct to the best of my knowledge and belief.",
        slot_id="affidavit",
    )
    finding = build_finding(
        catalogue.defect("D077"),
        DefectResponse(
            check_id="D077",
            status="defect_found",
            confidence=0.9,
            summary="Affidavit date cannot be checked against drafting date.",
            reasoning="The affidavit verification is on filing page 26.",
            evidence=[
                EvidenceRef(
                    page=None,
                    quote=(
                        "Verified at Una on 17/4 day of April, 2026 that the "
                        "contents of Para 1 to 3 of the affidavit are true and "
                        "correct to the best of my knowledge and belief."
                    ),
                )
            ],
            suggested_fix="State the drafting date.",
            fix_rationale="Required to compare dates.",
        ),
        evidence_ids=["p26"],
        coverage=Coverage(chunks_reviewed=1, pages_reviewed=[26]),
        chunks=[
            {
                "record_id": "p26",
                "chunk_kind": "page",
                "page": 26,
                "document_part": "Affidavit",
                "text": "AFFIDAVIT truncated pinecone excerpt",
            }
        ],
        layout=layout,
    )
    ref = finding.evidence[0]
    assert ref.page == 26
    assert ref.boxes_status == "matched"
    assert ref.bounding_boxes
    assert ref.document_part == "Affidavit"
    assert ref.slot_id == "affidavit"
    assert ref.local_page == 1


def test_build_finding_empty_evidence_uses_retrieved_inspect_pages() -> None:
    catalogue = get_catalogue()
    layout = _sentence_layout(
        25,
        "Drawn By AOR Filed on 15.04.2026 Filed By the petitioner",
        slot_id="petition",
    )
    finding = build_finding(
        catalogue.defect("D010"),
        DefectResponse(
            check_id="D010",
            status="needs_review",
            confidence=0.5,
            summary="The drafting date page is not in the excerpts.",
            reasoning="The Main Petition end was not retrieved.",
            evidence=[],
            suggested_fix=None,
            fix_rationale=None,
        ),
        evidence_ids=["p25"],
        coverage=Coverage(chunks_reviewed=1, pages_reviewed=[25]),
        chunks=[
            {
                "record_id": "p25",
                "chunk_kind": "page",
                "page": 25,
                "document_part": "Main Petition",
                "text": "Drawn By AOR Filed on 15.04.2026 Filed By the petitioner",
            }
        ],
        layout=layout,
    )
    assert finding.evidence
    ref = finding.evidence[0]
    assert ref.page == 25
    assert ref.boxes_status == "matched"
    assert ref.bounding_boxes
    assert "Filing page 25" in (finding.location or "")


def test_locate_quote_does_not_raise_on_garbage_layout() -> None:
    boxes, status, _page = locate_quote("any quote at all", 12, ["not-a-mapping"])  # type: ignore[arg-type]
    assert boxes == []
    assert status == "unavailable"
    boxes, status, _page = locate_quote(
        "any quote at all",
        12,
        {12: {"words": "broken"}},
    )
    assert status in {"page_only", "unavailable"}
    assert boxes == []


def test_empty_quote_is_unavailable_without_page() -> None:
    boxes, status, page = locate_quote("   ", None, _mid_line_layout())
    assert boxes == []
    assert status == "unavailable"
    assert page is None


def test_prefer_pages_wins_when_quote_is_on_two_pages() -> None:
    page_25 = _sentence_layout(25, QUOTE, slot_id="petition")
    page_26 = _sentence_layout(26, QUOTE, slot_id="affidavit")
    layout = {**page_25, **page_26}
    _boxes, status, page = locate_quote(QUOTE, None, layout, prefer_pages=[26, 25])
    assert status == "matched"
    assert page == 26


def test_date_ocr_variants_count_as_the_same_quote() -> None:
    ocr = (
        "Verified at Una on 17th day of April 2026 that the contents of Para 1 "
        "to 3 of the affidavit are true and correct"
    )
    model = (
        "Verified at Una on 17/4 day of April, 2026 that the contents of Para 1 "
        "to 3 of the affidavit are true and correct"
    )
    assert quote_in_text(model, ocr)
    layout = _sentence_layout(26, ocr)
    boxes, status, page = locate_quote(model, None, layout)
    assert status == "matched"
    assert page == 26
    assert boxes


def test_unrelated_quote_does_not_false_match() -> None:
    layout = _sentence_layout(26, "Verified at Una on 17th day of April 2026")
    boxes, status, page = locate_quote(
        "Advocate's Check List whether the petition is in Form 28",
        None,
        layout,
    )
    assert boxes == []
    assert status == "unavailable"
    assert page is None


def test_attach_evidence_boxes_survives_broken_citation() -> None:
    layout = _sentence_layout(
        25,
        "Drawn By AOR Filed on 15.04.2026 Filed By the petitioner",
        slot_id="petition",
    )
    attached = attach_evidence_boxes(
        [
            EvidenceRef(page=None, quote=""),
            EvidenceRef(
                page=25,
                quote="Drawn By AOR Filed on 15.04.2026 Filed By the petitioner",
            ),
        ],
        layout=layout,
        chunks=[
            {
                "chunk_kind": "page",
                "page": 25,
                "document_part": "Main Petition",
                "text": "Drawn By AOR Filed on 15.04.2026",
            }
        ],
    )
    assert attached[0].boxes_status in {"unavailable", "page_only"}
    assert attached[0].bounding_boxes == []
    assert attached[1].boxes_status == "matched"
    assert attached[1].page == 25
    assert attached[1].bounding_boxes
    for box in attached[1].bounding_boxes:
        assert 0.0 <= box.x <= 1.0
        assert 0.0 <= box.y <= 1.0
        assert box.w > 0
        assert box.h > 0
        assert box.x + box.w <= 1.000001
        assert box.y + box.h <= 1.000001


def test_build_finding_mixed_d077_citations() -> None:
    catalogue = get_catalogue()
    petition = _sentence_layout(
        25,
        "Drawn By AOR Filed on 15.04.2026 Filed By the petitioner",
        slot_id="petition",
    )
    affidavit = _sentence_layout(
        26,
        "Verified at Una on 17th day of April 2026 that the contents of Para 1 to 3 of the affidavit are true and correct to the best of my knowledge and belief.",
        slot_id="affidavit",
    )
    finding = build_finding(
        catalogue.defect("D077"),
        DefectResponse(
            check_id="D077",
            status="defect_found",
            confidence=0.95,
            summary="Drafting date is missing so the affidavit date cannot be checked.",
            reasoning="Page 25 of the Main Petition and page 26 of the Affidavit were inspected.",
            evidence=[
                EvidenceRef(
                    page=25,
                    quote="Drawn By:\n\nFiled on: 15.04.2026\n\nFiled By",
                ),
                EvidenceRef(
                    page=None,
                    quote=(
                        "Verified at Una on 17/4 day of April, 2026 that the "
                        "contents of Para 1 to 3 of the affidavit are true and "
                        "correct to the best of my knowledge and belief."
                    ),
                ),
            ],
            suggested_fix="State the drafting date.",
            fix_rationale="Required to compare dates.",
        ),
        evidence_ids=["p25", "p26"],
        coverage=Coverage(chunks_reviewed=2, pages_reviewed=[25, 26]),
        chunks=[
            {
                "record_id": "p25",
                "chunk_kind": "page",
                "page": 25,
                "document_part": "Main Petition",
                "text": "prayer only, no drafting date line",
            },
            {
                "record_id": "p26",
                "chunk_kind": "page",
                "page": 26,
                "document_part": "Affidavit",
                "text": "AFFIDAVIT truncated pinecone excerpt",
            },
        ],
        layout={**petition, **affidavit},
    )
    assert len(finding.evidence) == 2
    assert finding.evidence[0].page == 25
    assert finding.evidence[0].boxes_status == "matched"
    assert finding.evidence[0].bounding_boxes
    assert finding.evidence[1].page == 26
    assert finding.evidence[1].boxes_status == "matched"
    assert finding.evidence[1].bounding_boxes
    assert finding.evidence[1].document_part == "Affidavit"


def test_empty_evidence_ignores_non_numeric_chunk_pages() -> None:
    catalogue = get_catalogue()
    finding = build_finding(
        catalogue.defect("D010"),
        DefectResponse(
            check_id="D010",
            status="needs_review",
            confidence=0.5,
            summary="Drafting date could not be checked.",
            reasoning="The Main Petition excerpts are incomplete.",
            evidence=[],
            suggested_fix=None,
            fix_rationale=None,
        ),
        evidence_ids=[],
        coverage=Coverage(chunks_reviewed=1, pages_reviewed=[]),
        chunks=[
            {
                "chunk_kind": "page",
                "page": "unknown",
                "document_part": "Main Petition",
                "text": "Drawn By AOR Filed on 15.04.2026",
            }
        ],
        layout=_sentence_layout(25, "Drawn By AOR Filed on 15.04.2026", slot_id="petition"),
    )
    assert finding.error is None
    assert finding.evidence == []


@pytest.mark.asyncio
async def test_load_layout_index_falls_back_to_s3_key(monkeypatch) -> None:
    dumped = dump_layout_index(
        _sentence_layout(34, "DRAWN ON 11.04.2026", slot_id="petition")
    )

    def fake_download(key: str | None):
        assert key == "org/llamacloud/filing-workspace/default/layoutfiles/job.json"
        return dumped

    monkeypatch.setattr(
        "extraction_review.s3_artifacts.download_json_object",
        fake_download,
    )
    loaded = await load_layout_index(
        None,
        key="org/llamacloud/filing-workspace/default/layoutfiles/job.json",
    )
    assert 34 in loaded
    assert loaded[34]["words"]


@pytest.mark.asyncio
async def test_load_layout_index_uses_key_when_url_fails(monkeypatch) -> None:
    dumped = dump_layout_index(
        _sentence_layout(12, "hereby undertakes that the facts stated", slot_id="vakalatnama")
    )

    class FakeResponse:
        def raise_for_status(self) -> None:
            raise RuntimeError("expired presign")

        def json(self):
            return {}

    class FakeClient:
        def __init__(self, *args, **kwargs):
            del args, kwargs

        async def __aenter__(self):
            return self

        async def __aexit__(self, *args):
            return None

        async def get(self, url: str):
            del url
            return FakeResponse()

    monkeypatch.setattr("extraction_review.layout_index.httpx.AsyncClient", FakeClient)
    monkeypatch.setattr(
        "extraction_review.s3_artifacts.download_json_object",
        lambda key: dumped if key == "layout-key-1" else None,
    )
    loaded = await load_layout_index(
        "https://s3.example/expired",
        key="layout-key-1",
    )
    assert 12 in loaded


def test_build_finding_d077_drawn_on_quote_matches_boxes() -> None:
    catalogue = get_catalogue()
    layout = _sentence_layout(34, "DRAWN ON 11.04.2026", slot_id="petition")
    finding = build_finding(
        catalogue.defect("D077"),
        DefectResponse(
            check_id="D077",
            status="defect_found",
            confidence=0.9,
            summary="Drafting date is after the affidavit date.",
            reasoning="The drafting date is on page 34 of the Main Petition.",
            evidence=[
                EvidenceRef(page=34, quote="**DRAWN ON**: 11.04.2026"),
            ],
            suggested_fix="Correct the drafting date.",
            fix_rationale="Required to compare dates.",
        ),
        evidence_ids=["p34"],
        coverage=Coverage(chunks_reviewed=1, pages_reviewed=[34]),
        chunks=[
            {
                "record_id": "p34",
                "chunk_kind": "page",
                "page": 34,
                "document_part": "Main Petition",
                "text": "prayer only",
            }
        ],
        layout=layout,
    )
    assert finding.evidence[0].page == 34
    assert finding.evidence[0].boxes_status == "matched"
    assert finding.evidence[0].bounding_boxes
    assert finding.evidence[0].slot_id == "petition"
    assert finding.evidence[0].document_part == "Main Petition"


def test_as_bounding_boxes_keeps_in_range_unit_boxes() -> None:
    boxes = _as_bounding_boxes(
        [{"page": 34, "x": 0.42, "y": 0.61, "w": 0.14, "h": 0.028}]
    )
    assert len(boxes) == 1
    assert boxes[0].page == 34
    assert boxes[0].x == 0.42
    assert boxes[0].y == 0.61
    assert boxes[0].w == 0.14
    assert boxes[0].h == 0.028


def test_as_bounding_boxes_normalizes_pixel_coords_instead_of_dropping() -> None:
    boxes = _as_bounding_boxes(
        [{"page": 12, "x": 42.0, "y": 61.0, "w": 14.0, "h": 2.8}]
    )
    assert len(boxes) == 1
    assert 0.0 <= boxes[0].x < 1.0
    assert 0.0 < boxes[0].w <= 1.0
    assert 0.0 <= boxes[0].y < 1.0
    assert 0.0 < boxes[0].h <= 1.0
