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
    merge_pdf_text_words,
    parse_sidecar_jsonl,
    quote_in_text,
    stitch_slot_layouts,
)
from extraction_review.llm import strict_json_schema
from extraction_review.scrutiny.prompts import _format_evidence
from extraction_review.scrutiny.schema import (
    DefectResponse,
    EvidenceRef,
    _as_bounding_boxes,
    apply_evidence_pages,
    attach_evidence_boxes,
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


def test_merge_granular_bboxes_adds_word_line_and_cell() -> None:
    kwargs = merge_granular_bboxes({"file_id": "abc", "tier": "agentic"})
    assert kwargs["output_options"]["granular_bboxes"] == ["word", "line", "cell"]


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
        SplitPartInput(
            slot_id="vakalatnama_appearance",
            file_id="f-vak",
            document_parts=("Vakalatnama",),
        ),
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
    assert stitched[12]["document_part"] == "Vakalatnama"
    assert stitched[1]["local_page"] == 1
    assert stitched[1]["slot_id"] == "cover_page"
    assert stitched[1]["words"] == []
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


def test_excerpts_are_labeled_with_short_chunk_ids() -> None:
    chunks = [
        {
            "chunk_kind": "page",
            "page": 14,
            "document_part": "Annexure P-4",
            "text": "hello",
        },
        {
            "chunk_kind": "page",
            "page": 15,
            "page_end": 16,
            "document_part": "Annexure P-4",
            "text": "next",
        },
    ]
    text = _format_evidence(chunks)
    assert "[c1 | Annexure P-4, page 14]" in text
    assert "[c2 | Annexure P-4, page 15]" in text
    assert chunks[0]["excerpt_id"] == "c1"
    assert chunks[1]["excerpt_id"] == "c2"


def test_defect_response_schema_has_no_bounding_boxes() -> None:
    schema = strict_json_schema(DefectResponse)
    dumped = str(schema)
    defs = schema.get("$defs") or schema.get("definitions") or {}
    properties = (defs.get("EvidenceRef") or {}).get("properties") or {}
    assert set(properties) == {"chunk_id", "page", "quote"}
    assert "bounding_boxes" not in properties
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
        evidence=[EvidenceRef(page=99, quote="ADVOCATE'S CHECK LIST", chunk_id="c1")],
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
    assert grounded.evidence[0].chunk_id == "c1"
    assert grounded.evidence[0].model_dump() == {
        "chunk_id": "c1",
        "page": 2,
        "quote": "ADVOCATE'S CHECK LIST",
    }


def _sentence_layout(page: int, text: str, slot_id: str = "affidavit") -> dict:
    tokens = text.split()
    row = [
        (token, 0.08 + index * 0.03, 0.62, 0.028, 0.018)
        for index, token in enumerate(tokens)
    ]
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
            EvidenceRef(page=99, quote=""),
            EvidenceRef(
                chunk_id="c1",
                page=25,
                quote="Drawn By AOR Filed on 15.04.2026 Filed By the petitioner",
            ),
        ],
        layout=layout,
        chunks=[
            {
                "excerpt_id": "c1",
                "chunk_kind": "page",
                "page": 25,
                "document_part": "Main Petition",
                "text": "Drawn By AOR Filed on 15.04.2026",
            }
        ],
    )
    assert attached[0].page is None
    assert attached[0].boxes_status == "unavailable"
    assert attached[0].bounding_boxes == []
    assert attached[1].boxes_status == "matched"
    assert attached[1].page == 1
    assert attached[1].document_part == "Main Petition"
    assert attached[1].bounding_boxes
    assert all(box.page == 1 for box in attached[1].bounding_boxes)
    for box in attached[1].bounding_boxes:
        assert 0.0 <= box.x <= 1.0
        assert 0.0 <= box.y <= 1.0
        assert box.w > 0
        assert box.h > 0
        assert box.x + box.w <= 1.000001
        assert box.y + box.h <= 1.000001


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
        _sentence_layout(
            12, "hereby undertakes that the facts stated", slot_id="vakalatnama"
        )
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


def test_known_chunk_id_returns_that_documents_page() -> None:
    layout = _sentence_layout(14, QUOTE, slot_id="annexure")
    layout[14]["local_page"] = 1
    layout[14]["document_part"] = "Annexure P-4"
    attached = attach_evidence_boxes(
        [EvidenceRef(chunk_id="c1", page=99, quote=QUOTE)],
        layout=layout,
        chunks=[
            {
                "excerpt_id": "c1",
                "chunk_kind": "page",
                "page": 14,
                "document_part": "Annexure P-4",
                "text": QUOTE,
            }
        ],
    )
    assert attached[0].page == 1
    assert attached[0].document_part == "Annexure P-4"
    assert attached[0].boxes_status == "matched"
    assert attached[0].bounding_boxes
    assert all(box.page == 1 for box in attached[0].bounding_boxes)


def test_quote_in_a_different_chunk_uses_that_chunk() -> None:
    cited = _sentence_layout(10, "alpha words only", slot_id="index")
    actual = _sentence_layout(20, QUOTE, slot_id="annexure")
    cited[10]["local_page"] = 2
    cited[10]["document_part"] = "Index"
    actual[20]["local_page"] = 1
    actual[20]["document_part"] = "Annexure P-4"
    attached = attach_evidence_boxes(
        [EvidenceRef(chunk_id="c1", page=10, quote=QUOTE)],
        layout={**cited, **actual},
        chunks=[
            {
                "excerpt_id": "c1",
                "chunk_kind": "page",
                "page": 10,
                "document_part": "Index",
                "text": "alpha words only",
            },
            {
                "excerpt_id": "c2",
                "chunk_kind": "page",
                "page": 20,
                "document_part": "Annexure P-4",
                "text": QUOTE,
            },
        ],
    )
    assert attached[0].page == 1
    assert attached[0].document_part == "Annexure P-4"
    assert attached[0].boxes_status == "matched"


def test_page_without_word_boxes_keeps_document_page() -> None:
    layout = {
        14: {
            "words": [],
            "local_page": 1,
            "slot_id": "annexure",
            "document_part": "Annexure P-4",
        }
    }
    attached = attach_evidence_boxes(
        [
            EvidenceRef(
                chunk_id="c1",
                page=14,
                quote="a sentence the page never boxed",
            )
        ],
        layout=layout,
        chunks=[
            {
                "excerpt_id": "c1",
                "chunk_kind": "page",
                "page": 14,
                "document_part": "Annexure P-4",
                "text": "pinecone text that does not contain the quote",
            }
        ],
    )
    assert attached[0].page == 1
    assert attached[0].document_part == "Annexure P-4"
    assert attached[0].bounding_boxes == []
    assert attached[0].boxes_status == "page_only"


def test_overlap_quote_stays_on_the_chunk_page() -> None:
    start = _sentence_layout(14, "opening words of the annexure", slot_id="annexure")
    end = _sentence_layout(15, QUOTE, slot_id="annexure")
    start[14]["local_page"] = 1
    start[14]["document_part"] = "Annexure P-4"
    end[15]["local_page"] = 2
    end[15]["document_part"] = "Annexure P-4"
    attached = attach_evidence_boxes(
        [EvidenceRef(chunk_id="c1", page=14, quote=QUOTE)],
        layout={**start, **end},
        chunks=[
            {
                "excerpt_id": "c1",
                "chunk_kind": "page",
                "page": 14,
                "page_end": 15,
                "document_part": "Annexure P-4",
                "text": f"opening words of the annexure\n--- from page 15 ---\n{QUOTE}",
            }
        ],
    )
    assert attached[0].page == 1
    assert attached[0].document_part == "Annexure P-4"
    assert attached[0].bounding_boxes == []
    assert attached[0].boxes_status == "page_only"


def _bare_layout(words: list[dict]) -> dict:
    return {1: {"words": words, "width": 1.0, "height": 1.0, "local_page": 1}}


def test_markdown_and_curly_quotes_match_plain_words() -> None:
    layout = _sentence_layout(3, "Form 28 Yes ANNEXURE court's order is filed")
    quotes = (
        "| Form 28 | Yes |\n|---|---|",
        "**ANNEXURE**",
        "court’s order",
    )
    for quote in quotes:
        boxes, status = boxes_for_quote(quote, 3, layout)
        assert status == "matched", quote
        assert boxes, quote


def test_ellipsis_matches_both_sides_and_skips_the_middle() -> None:
    layout = _bare_layout(
        [
            {"t": "the", "x": 0.10, "y": 0.20, "w": 0.05, "h": 0.02, "line": 0},
            {"t": "petitioner", "x": 0.16, "y": 0.20, "w": 0.14, "h": 0.02, "line": 0},
            {"t": "hereby", "x": 0.10, "y": 0.55, "w": 0.10, "h": 0.02, "line": 1},
            {"t": "undertakes", "x": 0.32, "y": 0.20, "w": 0.14, "h": 0.02, "line": 0},
        ]
    )
    boxes, status = boxes_for_quote("the petitioner ... undertakes", 1, layout)
    assert status == "matched"
    assert boxes
    assert all(box["y"] + box["h"] < 0.40 for box in boxes)


def test_same_baseline_columns_do_not_share_one_box() -> None:
    layout = _bare_layout(
        [
            {"t": "left", "x": 0.05, "y": 0.40, "w": 0.12, "h": 0.02, "line": 0},
            {"t": "column", "x": 0.18, "y": 0.40, "w": 0.12, "h": 0.02, "line": 0},
            {"t": "right", "x": 0.70, "y": 0.402, "w": 0.10, "h": 0.02, "line": 0},
            {"t": "side", "x": 0.81, "y": 0.402, "w": 0.10, "h": 0.02, "line": 0},
        ]
    )
    boxes, status = boxes_for_quote("left column right side", 1, layout)
    assert status == "matched"
    assert len(boxes) == 2
    assert boxes[0]["x"] + boxes[0]["w"] < 0.50
    assert boxes[1]["x"] >= 0.70


def test_vertical_jitter_joins_and_tight_lines_stay_apart() -> None:
    layout = _bare_layout(
        [
            {"t": "alpha", "x": 0.10, "y": 0.200, "w": 0.08, "h": 0.020, "line": 0},
            {"t": "beta", "x": 0.20, "y": 0.205, "w": 0.08, "h": 0.020, "line": 1},
            {"t": "gamma", "x": 0.10, "y": 0.600, "w": 0.08, "h": 0.004, "line": 2},
            {"t": "delta", "x": 0.20, "y": 0.605, "w": 0.08, "h": 0.004, "line": 3},
        ]
    )
    boxes, status = boxes_for_quote("alpha beta gamma delta", 1, layout)
    assert status == "matched"
    assert len(boxes) == 3
    assert boxes[0]["w"] == round(0.28 - 0.10, 6)
    assert boxes[1]["y"] == 0.6
    assert boxes[1]["w"] == 0.08
    assert boxes[2]["y"] == 0.605
    assert boxes[2]["w"] == 0.08


def _table_sidecar(*, with_cells: bool) -> dict:
    md = "| Form 28 | Yes |"
    raw = md.encode("utf-8")
    form_at = raw.find(b"Form")
    number_at = raw.find(b"28")
    grounding: dict = {"source": "md"}
    if with_cells:
        grounding["rows"] = [
            [
                {
                    "bbox": [{"x": 10.0, "y": 20.0, "w": 30.0, "h": 12.0}],
                    "lines": [
                        {
                            "span": [form_at, number_at + 2],
                            "bbox": {"x": 10.0, "y": 20.0, "w": 30.0, "h": 12.0},
                            "words": [
                                {
                                    "span": [form_at, form_at + 4],
                                    "bbox": {
                                        "x": 10.0,
                                        "y": 20.0,
                                        "w": 12.0,
                                        "h": 12.0,
                                    },
                                },
                                {
                                    "span": [number_at, number_at + 2],
                                    "bbox": {
                                        "x": 24.0,
                                        "y": 20.0,
                                        "w": 10.0,
                                        "h": 12.0,
                                    },
                                },
                            ],
                        }
                    ],
                },
                {"bbox": [{"x": 50.0, "y": 20.0, "w": 16.0, "h": 12.0}]},
            ]
        ]
    return {
        "page_number": 1,
        "page_width": 100.0,
        "page_height": 100.0,
        "success": True,
        "items": [
            {
                "type": "table",
                "md": md,
                "rows": [["Form 28", "Yes"]],
                "bbox": [{"x": 10.0, "y": 20.0, "w": 80.0, "h": 40.0}],
                "grounding": grounding,
            }
        ],
    }


def test_table_cell_quote_is_cell_sized() -> None:
    layout = compact_sidecar_pages([_table_sidecar(with_cells=True)])
    assert [word["t"] for word in layout[1]["words"]] == ["Form", "28", "Yes"]
    boxes, status = boxes_for_quote("| Form 28 | Yes |", 1, layout)
    assert status == "matched"
    assert boxes
    assert all(box["w"] < 0.5 for box in boxes)
    assert (
        max(box["x"] + box["w"] for box in boxes) - min(box["x"] for box in boxes) < 0.7
    )


def test_table_without_cell_boxes_stays_page_only() -> None:
    layout = compact_sidecar_pages([_table_sidecar(with_cells=False)])
    assert layout[1]["words"] == []
    boxes, status = boxes_for_quote("Form 28 Yes", 1, layout)
    assert boxes == []
    assert status == "page_only"


def test_utf8_byte_spans_keep_the_following_word() -> None:
    md = "“Hello” world"
    raw = md.encode("utf-8")
    hello_at = raw.find(b"Hello")
    world_at = raw.find(b"world")
    layout = compact_sidecar_pages(
        [
            {
                "page_number": 1,
                "page_width": 100.0,
                "page_height": 100.0,
                "success": True,
                "items": [
                    {
                        "type": "text",
                        "md": md,
                        "grounding": {
                            "source": "md",
                            "lines": [
                                {
                                    "span": [0, len(raw)],
                                    "bbox": {"x": 10.0, "y": 20.0, "w": 40.0, "h": 4.0},
                                    "words": [
                                        {
                                            "span": [
                                                hello_at,
                                                hello_at + len(b"Hello"),
                                            ],
                                            "bbox": {
                                                "x": 10.0,
                                                "y": 20.0,
                                                "w": 15.0,
                                                "h": 4.0,
                                            },
                                        },
                                        {
                                            "span": [
                                                world_at,
                                                world_at + len(b"world"),
                                            ],
                                            "bbox": {
                                                "x": 28.0,
                                                "y": 20.0,
                                                "w": 15.0,
                                                "h": 4.0,
                                            },
                                        },
                                    ],
                                }
                            ],
                        },
                    },
                    {
                        "type": "list",
                        "items": [
                            {
                                "type": "text",
                                "md": "nested",
                                "grounding": {
                                    "source": "md",
                                    "lines": [
                                        {
                                            "span": [0, 6],
                                            "bbox": {
                                                "x": 10.0,
                                                "y": 40.0,
                                                "w": 20.0,
                                                "h": 4.0,
                                            },
                                            "words": [
                                                {
                                                    "span": [0, 6],
                                                    "bbox": {
                                                        "x": 10.0,
                                                        "y": 40.0,
                                                        "w": 20.0,
                                                        "h": 4.0,
                                                    },
                                                }
                                            ],
                                        }
                                    ],
                                },
                            }
                        ],
                    },
                ],
            }
        ]
    )
    assert [word["t"] for word in layout[1]["words"]] == ["Hello", "world", "nested"]
    boxes, status = boxes_for_quote("Hello", 1, layout)
    assert status == "matched"
    assert boxes[0]["x"] == 0.1
    assert boxes[0]["w"] == 0.15


def _placed(text: str, x: float, y: float, w: float = 0.06) -> dict:
    return {"t": text, "x": x, "y": y, "w": w, "h": 0.02, "line": 0}


_PAWAN_QUOTE = (
    "Column 1: 1.Col. Pawan Kumar Gopinath Sharma, s/o Gopinath Sharma, Indian Jew\n"
    "Column 2: Respondent No.\n"
    "Column 3: Petitioner\n"
    "\n"
    "Column 1: Versus"
)

_PRATAP_QUOTE = (
    "Column 1: 10.Mr. Pratap, s/o\n"
    "Haridas Dhopte,\n"
    "Column 2: Respondent No.\n"
    "Column 3: Respondent No."
)


def _party_word_layout() -> dict:
    """Cell order, with role words on the first name line and an earlier role."""
    words = [
        _placed("Respondent", 0.50, 0.10, 0.10),
        _placed("No.", 0.61, 0.10, 0.04),
        _placed("1.Col.", 0.06, 0.30, 0.06),
        _placed("Pawan", 0.13, 0.30, 0.06),
        _placed("Kumar", 0.20, 0.30, 0.06),
        _placed("Gopinath", 0.06, 0.33, 0.08),
        _placed("Sharma,", 0.15, 0.33, 0.08),
        _placed("s/o", 0.24, 0.33, 0.04),
        _placed("Gopinath", 0.06, 0.36, 0.08),
        _placed("Sharma,", 0.15, 0.36, 0.08),
        _placed("Indian", 0.06, 0.39, 0.06),
        _placed("Jew", 0.13, 0.39, 0.04),
        _placed("Respondent", 0.50, 0.30, 0.10),
        _placed("No.", 0.61, 0.30, 0.04),
        _placed("Petitioner", 0.78, 0.30, 0.12),
        _placed("Versus", 0.30, 0.50, 0.08),
        _placed("Versus", 0.30, 0.80, 0.08),
        _placed("10.Mr.", 0.06, 0.62, 0.07),
        _placed("Pratap,", 0.14, 0.62, 0.07),
        _placed("s/o", 0.22, 0.62, 0.04),
        _placed("Haridas", 0.06, 0.65, 0.08),
        _placed("Dhopte,", 0.15, 0.65, 0.08),
        _placed("Respondent", 0.50, 0.62, 0.10),
        _placed("No.", 0.61, 0.62, 0.04),
        _placed("Respondent", 0.78, 0.62, 0.10),
        _placed("No.", 0.89, 0.62, 0.04),
    ]
    return _bare_layout(words)


def test_column_labels_box_the_party_row_and_the_versus_below() -> None:
    boxes, status = boxes_for_quote(_PAWAN_QUOTE, 1, _party_word_layout())
    assert status == "matched"
    assert boxes
    assert all(box["y"] > 0.20 for box in boxes)
    assert all(box["y"] < 0.70 for box in boxes)
    name = [box for box in boxes if box["x"] < 0.40 and box["y"] < 0.45]
    assert len(name) >= 2
    assert any(box["y"] == 0.30 for box in name)
    roles = [box for box in boxes if box["x"] >= 0.45 and box["y"] == 0.30]
    assert len(roles) == 2
    assert roles[0]["x"] < 0.70
    assert roles[1]["x"] >= 0.75
    versus = [box for box in boxes if box["y"] >= 0.45]
    assert len(versus) == 1
    assert versus[0]["y"] == 0.50


def test_column_labels_box_both_role_cells_on_the_same_row() -> None:
    boxes, status = boxes_for_quote(_PRATAP_QUOTE, 1, _party_word_layout())
    assert status == "matched"
    assert all(box["y"] > 0.20 for box in boxes)
    name = [box for box in boxes if box["x"] < 0.40]
    assert len(name) == 2
    assert name[0]["y"] == 0.62
    assert name[1]["y"] == 0.65
    roles = [box for box in boxes if box["x"] >= 0.45]
    assert len(roles) == 2
    assert roles[0]["y"] == 0.62
    assert roles[1]["y"] == 0.62
    assert roles[0]["x"] < roles[1]["x"]
    assert roles[0]["x"] + roles[0]["w"] < roles[1]["x"]


def test_parenthetical_column_label_is_removed_before_matching() -> None:
    layout = _bare_layout(
        [
            _placed("Respondent", 0.50, 0.20, 0.10),
            _placed("No.", 0.61, 0.20, 0.04),
        ]
    )
    boxes, status = boxes_for_quote(
        "BEFORE HIGH COURT (column 2): Respondent No.", 1, layout
    )
    assert status == "matched"
    assert len(boxes) == 1
    assert boxes[0]["x"] == 0.50


def test_column_word_inside_a_sentence_stays_on_the_normal_path() -> None:
    layout = _bare_layout(
        [
            _placed("the", 0.10, 0.20, 0.04),
            _placed("column", 0.15, 0.20, 0.08),
            _placed("of", 0.24, 0.20, 0.03),
            _placed("names", 0.28, 0.20, 0.06),
            _placed("stays", 0.35, 0.20, 0.06),
        ]
    )
    boxes, status = boxes_for_quote("the column of names stays", 1, layout)
    assert status == "matched"
    assert len(boxes) == 1
    assert boxes[0]["x"] == 0.10


def test_unmatched_party_row_does_not_box_a_later_versus() -> None:
    layout = _bare_layout([_placed("Versus", 0.30, 0.50, 0.08)])
    boxes, status = boxes_for_quote(_PAWAN_QUOTE, 1, layout)
    assert boxes == []
    assert status == "page_only"


def _reading_order(lines: list[tuple[float, list[tuple[str, float, float]]]]) -> list[dict]:
    words: list[dict] = []
    for y, items in lines:
        for text, x, width in items:
            words.append(_placed(text, x, y, width))
    return words


_PAWAN_LINES = (
    "1.Col. Pawan Kumar\nGopinath Sharma, s/o\nGopinath Sharma,\nIndian Jew"
)


def _interrupted_party_layout() -> dict:
    """Each name line is broken by the next column, so no five words sit together."""
    return _bare_layout(
        _reading_order(
            [
                (
                    0.30,
                    [
                        ("1.", 0.06, 0.03),
                        ("Col.", 0.10, 0.05),
                        ("Pawan", 0.16, 0.07),
                        ("Kumar", 0.24, 0.07),
                        ("Respondent", 0.50, 0.12),
                        ("No.", 0.63, 0.04),
                        ("Petitioner", 0.78, 0.12),
                    ],
                ),
                (
                    0.33,
                    [
                        ("Gopinath", 0.06, 0.09),
                        ("Sharma,", 0.16, 0.08),
                        ("s/o", 0.25, 0.04),
                        ("role", 0.50, 0.06),
                    ],
                ),
                (
                    0.36,
                    [
                        ("Gopinath", 0.06, 0.09),
                        ("Sharma,", 0.16, 0.08),
                        ("role", 0.50, 0.06),
                    ],
                ),
                (0.39, [("Indian", 0.06, 0.07), ("Jew", 0.14, 0.05), ("role", 0.50, 0.06)]),
            ]
        )
    )


def test_party_lines_split_by_other_columns_still_box_indian_jew() -> None:
    layout = _interrupted_party_layout()
    boxes, status = boxes_for_quote(_PAWAN_LINES, 1, layout)
    assert status == "matched"
    jew = [box for box in boxes if box["y"] == 0.39 and box["x"] < 0.30]
    assert jew
    assert jew[0]["x"] < 0.10
    name = [box for box in boxes if box["y"] == 0.30 and box["x"] < 0.40]
    assert name
    assert name[0]["x"] < 0.10

    boxed, boxed_status = boxes_for_quote(_PAWAN_QUOTE, 1, layout)
    assert boxed_status == "matched"
    assert any(box["y"] == 0.39 and box["x"] < 0.30 for box in boxed)
    roles = [box for box in boxed if box["y"] == 0.30 and box["x"] >= 0.45]
    assert len(roles) == 2
    assert roles[0]["x"] + roles[0]["w"] < roles[1]["x"]


def test_index_note_sheet_row_is_boxed_without_the_wrapped_header() -> None:
    quote = (
        "S.No.: 7.\n"
        "PARTICULARS: Note Sheet\n"
        "Page No. of part to which it belongs — Part – II\n"
        "(Contents\n"
        "of Paper\n"
        "Book): NS1 to"
    )
    words = _reading_order(
        [
            (
                0.08,
                [
                    ("Page", 0.40, 0.06),
                    ("No.", 0.47, 0.04),
                    ("PARTICULARS", 0.55, 0.12),
                    ("of", 0.68, 0.03),
                    ("part", 0.72, 0.05),
                    ("to", 0.78, 0.03),
                ],
            ),
            (
                0.42,
                [
                    ("7.", 0.06, 0.04),
                    ("Note", 0.18, 0.06),
                    ("Sheet", 0.25, 0.07),
                    ("NS1", 0.55, 0.05),
                    ("to", 0.61, 0.03),
                ],
            ),
        ]
    )
    boxes, status = boxes_for_quote(quote, 1, _bare_layout(words))
    assert status == "matched"
    assert boxes
    assert all(box["y"] > 0.20 for box in boxes)
    assert any(box["x"] < 0.12 for box in boxes)
    assert any(box["x"] >= 0.50 for box in boxes)
    note = [box for box in boxes if 0.15 <= box["x"] < 0.45]
    assert note
    assert note[0]["y"] == 0.42


def test_index_office_report_row_boxes_both_page_letters() -> None:
    quote = (
        "S.No.: 1.\n"
        "PARTICULARS: Office Report on Limitation\n"
        "Page No. of part to which it belongs — Part – I\n"
        "(Contents\n"
        "of Paper\n"
        "Book): A\n"
        "Page No. of part to which it belongs — Part – II\n"
        "(Contents\n"
        "of Paper\n"
        "Book): A"
    )
    words = _reading_order(
        [
            (
                0.40,
                [
                    ("1.", 0.06, 0.04),
                    ("Office", 0.16, 0.08),
                    ("Report", 0.25, 0.08),
                    ("on", 0.34, 0.03),
                    ("Limitation", 0.38, 0.10),
                    ("A", 0.62, 0.03),
                    ("A", 0.78, 0.03),
                ],
            )
        ]
    )
    boxes, status = boxes_for_quote(quote, 1, _bare_layout(words))
    assert status == "matched"
    letters = [box for box in boxes if box["x"] >= 0.60]
    assert len(letters) == 2
    assert letters[0]["x"] < letters[1]["x"]
    assert any(box["x"] < 0.12 for box in boxes)


def test_list_of_dates_keeps_the_left_date_on_its_own_row() -> None:
    quote = (
        "S.NO.: 01.01.2031\n"
        "PARTICULARS: That in exercise of power under Section 165 of the Act "
        "for getting the necessary documents"
    )
    words = _reading_order(
        [
            (0.05, [("01.01.1993", 0.08, 0.12)]),
            (
                0.20,
                [
                    ("01.01.2031", 0.08, 0.12),
                    ("That", 0.36, 0.05),
                    ("in", 0.42, 0.03),
                    ("exercise", 0.46, 0.08),
                    ("of", 0.55, 0.03),
                    ("power", 0.59, 0.06),
                    ("under", 0.66, 0.06),
                    ("Section", 0.73, 0.08),
                    ("165", 0.82, 0.04),
                ],
            ),
            (
                0.24,
                [
                    ("of", 0.36, 0.03),
                    ("the", 0.40, 0.04),
                    ("Act", 0.45, 0.04),
                    ("for", 0.50, 0.04),
                    ("getting", 0.55, 0.08),
                    ("the", 0.64, 0.04),
                    ("necessary", 0.69, 0.09),
                    ("documents", 0.79, 0.10),
                ],
            ),
        ]
    )
    boxes, status = boxes_for_quote(quote, 1, _bare_layout(words))
    assert status == "matched"
    left = [box for box in boxes if box["x"] < 0.20]
    assert len(left) == 1
    assert left[0]["y"] == 0.20
    assert left[0]["x"] + left[0]["w"] < 0.30
    assert any(box["x"] >= 0.30 and box["y"] == 0.20 for box in boxes)
    assert all(box["y"] > 0.10 for box in boxes)


def test_two_shared_words_do_not_box_a_longer_quote() -> None:
    layout = _bare_layout(
        [_placed("the", 0.10, 0.20, 0.04), _placed("court", 0.16, 0.20, 0.06)]
    )
    boxes, status = boxes_for_quote(
        "the court granted the petitioner interim relief", 1, layout
    )
    assert boxes == []
    assert status == "page_only"


def _pdf_bytes() -> bytes:
    import pymupdf

    document = pymupdf.open()
    try:
        page = document.new_page(width=1400, height=800)
        page.insert_text(
            (36, 220),
            "1.Col. Pawan Kumar Gopinath Sharma, s/o Gopinath Sharma, Indian Jew",
            fontsize=11,
        )
        page.insert_text((780, 220), "Respondent No.", fontsize=11)
        page.insert_text((1100, 220), "Petitioner", fontsize=11)
        return document.tobytes()
    finally:
        document.close()


def _empty_table_layout() -> dict:
    return compact_sidecar_pages(
        [
            {
                "page_number": 1,
                "page_width": 1400.0,
                "page_height": 800.0,
                "success": True,
                "items": [
                    {
                        "type": "table",
                        "md": "party table",
                        "grounding": {
                            "source": "table",
                            "bbox": {"x": 0, "y": 0, "w": 1400, "h": 400},
                        },
                    }
                ],
            }
        ]
    )


def test_pdf_words_box_a_table_page_that_has_no_cell_words() -> None:
    layout = _empty_table_layout()
    assert layout[1]["words"] == []
    merged = merge_pdf_text_words(layout, _pdf_bytes())
    boxes, status = boxes_for_quote(_PAWAN_QUOTE, 1, merged)
    assert status == "matched"
    name = [box for box in boxes if box["x"] < 0.45]
    roles = [box for box in boxes if box["x"] >= 0.50]
    assert name
    assert len(roles) >= 2
    assert roles[0]["x"] + roles[0]["w"] <= roles[1]["x"] + 0.01


def test_a_sidecar_word_is_not_added_again_from_the_pdf() -> None:
    hello_pdf = _hello_world_pdf()
    discovered = merge_pdf_text_words({1: {"words": [], "local_page": 1}}, hello_pdf)
    hello = next(word for word in discovered[1]["words"] if word["t"] == "Hello")
    layout = {1: {"words": [hello], "local_page": 1}}
    merged = merge_pdf_text_words(layout, hello_pdf)
    texts = [word["t"] for word in merged[1]["words"]]
    assert merged[1]["words"][0] is hello
    assert texts.count("Hello") == 1
    assert "World" in texts


def _hello_world_pdf() -> bytes:
    import pymupdf

    document = pymupdf.open()
    try:
        page = document.new_page(width=600, height=400)
        page.insert_text((40, 120), "Hello", fontsize=12)
        page.insert_text((240, 120), "World", fontsize=12)
        return document.tobytes()
    finally:
        document.close()


def test_a_scan_page_does_not_receive_pdf_words() -> None:
    import pymupdf

    pixmap = pymupdf.Pixmap(pymupdf.csRGB, pymupdf.IRect(0, 0, 10, 10), False)
    pixmap.clear_with(255)
    document = pymupdf.open()
    try:
        page = document.new_page(width=600, height=800)
        page.insert_image(pymupdf.Rect(20, 20, 580, 760), stream=pixmap.tobytes("png"))
        page.insert_text((40, 400), "Pawan Kumar stale ocr layer", fontsize=12)
        pdf = document.tobytes()
    finally:
        document.close()
    layout = {1: {"words": [], "local_page": 1}}
    merged = merge_pdf_text_words(layout, pdf)
    assert merged[1]["words"] == []
