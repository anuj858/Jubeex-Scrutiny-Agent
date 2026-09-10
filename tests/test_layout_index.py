"""Layout index: sidecar compacting, page remap, quote-to-line boxes."""

from extraction_review.layout_index import (
    boxes_for_quote,
    compact_sidecar_pages,
    dump_layout_index,
    grounded_items_url,
    merge_granular_bboxes,
    parse_sidecar_jsonl,
    stitch_slot_layouts,
)
from extraction_review.llm import strict_json_schema
from extraction_review.scrutiny.rules import get_catalogue
from extraction_review.scrutiny.schema import (
    Coverage,
    DefectResponse,
    EvidenceRef,
    apply_evidence_pages,
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
