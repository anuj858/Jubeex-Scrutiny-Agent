"""Compact LlamaParse word/line boxes and map evidence quotes onto them.

Parse already ran. This module downloads the grounded-items sidecar from that
same job, stitches slot-local pages onto global filing pages, and later maps a
quote to per-line rectangles. Scrutiny does not call LlamaParse or LlamaExtract.
"""

from __future__ import annotations

import json
import logging
import re
from collections.abc import Mapping, Sequence
from itertools import pairwise
from typing import Any

import httpx

from .document_parts import parts_on_page
from .logging_utils import safe_url_for_log
from .process_file import FILE_DOWNLOAD_TIMEOUT_S
from .split_upload import SplitPartInput, UploadTypeCatalog, ordered_parts

logger = logging.getLogger(__name__)

LAYOUT_SCHEMA = "layout_index_v1"
LAYOUT_ARTIFACT_URL_KEY = "layout_artifact_url"
LAYOUT_ARTIFACT_KEY_KEY = "layout_artifact_key"
GRANULAR_BBOXES = ("word", "line", "cell")
# A gap wider than this, and wider than a few normal word spaces, is a
# column gutter rather than a space inside the quote.
_COLUMN_GAP = 0.045
_GAP_MULTIPLE = 4
_LINE_OVERLAP = 0.5

_WS = re.compile(r"\s+")
_PUNCT = re.compile(r"[^\w./-]+")
_HTML = re.compile(r"<[^>]+>")
_MD_LINK = re.compile(r"\[([^\]]*)\]\([^)]*\)")
_MD_MARK = re.compile(r"(\*\*|__|~~|`)")
_ELLIPSIS = re.compile(r"\[\s*(?:\.{3,}|…+)\s*\]|\(\s*(?:\.{3,}|…+)\s*\)|\.{3,}|…+")
_TABLE_RULE = re.compile(r"(?:\|?\s*:?-{3,}:?\s*)+")
# Excerpt rows are "Column 1: cell" or "Header (column 2): cell". The label is
# not printed beside the cell. Only a whole line is a label, so a real word
# "column" in a sentence stays in the quote.
_COLUMN_LINE = re.compile(
    r"^(?:column\s+\d+|.+?\(\s*column\s+\d+\s*\))\s*:\s*(.*)$",
    re.IGNORECASE,
)
# Table chunks prefix each cell with its header: "S.NO.: 01.01.2031".
# The header is printed once, above the column, not on the row.
_HEADER_LINE = re.compile(r"^[A-Z][A-Z0-9 .'/().-]{0,40}\s*:\s*(.*)$")
# "S.No.:" is mixed case, so the all-caps header rule does not see it.
_SNO_LINE = re.compile(
    r"^(?:s\.?\s*no\.?|sr\.?\s*no\.?)\s*:\s*(.*)$",
    re.IGNORECASE,
)
# A wrapped index header keeps its colon on the last physical line:
# "Page No. of part …\n(Contents\nof Paper\nBook): NS1 to".
_WRAPPED_HEADER_START = re.compile(r"^page no\.?\s+of\s+part\b", re.IGNORECASE)
_FOLD = str.maketrans(
    {
        "\u2018": "'",
        "\u2019": "'",
        "\u201a": "'",
        "\u201b": "'",
        "\u2032": "'",
        "\u201c": '"',
        "\u201d": '"',
        "\u201e": '"',
        "\u201f": '"',
        "\u00ab": '"',
        "\u00bb": '"',
        "\u2013": "-",
        "\u2014": "-",
        "\u2015": "-",
        "\u2212": "-",
        "\u00a0": " ",
        "\u2007": " ",
        "\u2009": " ",
        "\u202f": " ",
        "\ufeff": "",
    }
)
# Survives punctuation stripping so an ellipsis can split the quote into
# ordered segments. The skipped middle is not part of either segment.
_GAP_TOKEN = "qxellipsisgapqx"

PageLayout = dict[str, Any]
LayoutIndex = dict[int, PageLayout]


_ORDINAL = re.compile(r"^(\d{1,2})(st|nd|rd|th)$")
_SLASH_DATE = re.compile(r"^\d{1,4}([./-]\d{1,4}){1,2}$")
_MIN_TOKEN_RUN = 5


def _norm_text(text: str) -> str:
    """Plain words, with an ellipsis left as a gap token."""
    raw = (text or "").translate(_FOLD)
    raw = _HTML.sub(" ", raw)
    raw = _MD_LINK.sub(r"\1", raw)
    raw = _ELLIPSIS.sub(f" {_GAP_TOKEN} ", raw)
    raw = _MD_MARK.sub(" ", raw)
    raw = raw.replace("_", " ")
    raw = _TABLE_RULE.sub(" ", raw)
    collapsed = _PUNCT.sub(" ", raw.lower())
    return _WS.sub(" ", collapsed).strip()


def _canon_token(token: str) -> str:
    """Collapse 17th / 17/4 / 17.04.2026 and Cooperative / Co-operative."""
    ordinal = _ORDINAL.match(token)
    if ordinal:
        return ordinal.group(1)
    if _SLASH_DATE.match(token):
        return token.split("/")[0].split(".")[0].split("-")[0]
    return token.replace("-", "").rstrip(".")


def quote_in_text(quote: str, text: str) -> bool:
    """True when the quote (or a stable span of it) appears in excerpt/OCR text."""
    return _match_span(_norm_text(quote), _norm_text(text)) is not None


def _as_mapping(value: Any) -> Mapping[str, Any] | None:
    if isinstance(value, Mapping):
        return value
    dump = getattr(value, "model_dump", None)
    if callable(dump):
        payload = dump()
        return payload if isinstance(payload, Mapping) else None
    return None


def _output_options_dict(value: Any) -> dict[str, Any]:
    mapping = _as_mapping(value)
    if mapping is None:
        return {}
    return {key: item for key, item in mapping.items() if item is not None}


def merge_granular_bboxes(create_kwargs: dict[str, Any]) -> dict[str, Any]:
    """Ensure the existing parse job requests word, line, and cell grounding."""
    if create_kwargs.get("configuration_id"):
        return create_kwargs
    options = _output_options_dict(create_kwargs.get("output_options"))
    current = options.get("granular_bboxes") or []
    if not isinstance(current, list):
        current = list(current) if isinstance(current, tuple) else []
    merged = list(current)
    for kind in GRANULAR_BBOXES:
        if kind not in merged:
            merged.append(kind)
    options["granular_bboxes"] = merged
    create_kwargs["output_options"] = options
    return create_kwargs


def grounded_items_url(parse_result: Any) -> str | None:
    """Presigned URL for the grounded-items JSONL sidecar, if present."""
    payload = _as_mapping(parse_result) or {}
    meta = payload.get("result_content_metadata")
    if meta is None:
        meta = getattr(parse_result, "result_content_metadata", None)
    meta_map = _as_mapping(meta)
    sidecar: Any = None
    if meta_map is not None:
        sidecar = meta_map.get("grounded_items")
    elif meta is not None:
        sidecar = getattr(meta, "grounded_items", None)
    sidecar_map = _as_mapping(sidecar)
    if sidecar_map is None and sidecar is not None:
        url = getattr(sidecar, "presigned_url", None)
        exists = getattr(sidecar, "exists", True)
    else:
        url = (sidecar_map or {}).get("presigned_url")
        exists = (sidecar_map or {}).get("exists", True)
    if exists is False:
        return None
    text = str(url or "").strip()
    return text or None


def parse_sidecar_jsonl(text: str) -> list[dict[str, Any]]:
    pages: list[dict[str, Any]] = []
    for line in (text or "").splitlines():
        raw = line.strip()
        if not raw:
            continue
        try:
            row = json.loads(raw)
        except json.JSONDecodeError:
            logger.warning("Skipping invalid grounded-items sidecar line")
            continue
        if isinstance(row, dict):
            pages.append(row)
    return pages


def _norm_coord(value: Any, size: float) -> float:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return 0.0
    if size > 1.5:
        number = number / size
    return max(0.0, min(1.0, number))


def _bbox_dict(raw: Any) -> dict[str, Any] | None:
    mapping = _as_mapping(raw)
    if mapping is None:
        return None
    if not any(key in mapping for key in ("x", "y", "w", "h", "width", "height")):
        return None
    return dict(mapping)


def _bbox_list(raw: Any) -> list[dict[str, Any]]:
    """A word box is one object. A cell box is a list of objects."""
    if isinstance(raw, Mapping):
        box = _bbox_dict(raw)
        return [box] if box is not None else []
    if isinstance(raw, Sequence) and not isinstance(raw, (str, bytes)):
        boxes: list[dict[str, Any]] = []
        for item in raw:
            box = _bbox_dict(item)
            if box is not None:
                boxes.append(box)
        return boxes
    return []


def _box_edges(box: Mapping[str, Any]) -> tuple[float, float, float, float]:
    x = float(box.get("x") or 0)
    y = float(box.get("y") or 0)
    width = float(box.get("w", box.get("width")) or 0)
    height = float(box.get("h", box.get("height")) or 0)
    return x, y, width, height


def _union_bbox(boxes: Sequence[Mapping[str, Any]]) -> dict[str, Any] | None:
    edges: list[tuple[float, float, float, float]] = []
    for box in boxes:
        try:
            edges.append(_box_edges(box))
        except (TypeError, ValueError):
            continue
    if not edges:
        return None
    x = min(item[0] for item in edges)
    y = min(item[1] for item in edges)
    right = max(item[0] + item[2] for item in edges)
    bottom = max(item[1] + item[3] for item in edges)
    return {"x": x, "y": y, "w": right - x, "h": bottom - y}


def _span_bounds(span: Any) -> tuple[int, int]:
    try:
        start, end = int(span[0]), int(span[1])
    except (TypeError, ValueError, IndexError):
        return 0, 0
    if start < 0 or end <= start:
        return 0, 0
    return start, end


def _utf8_slice(source: str, span: Any) -> str:
    """Sidecar spans are UTF-8 byte offsets, not Python character indexes."""
    start, end = _span_bounds(span)
    if not source or end <= start:
        return ""
    raw = source.encode("utf-8")
    if start >= len(raw) or end > len(raw):
        return ""
    return raw[start:end].decode("utf-8", errors="replace").strip()


def _text_from_span(span: Any, *sources: str) -> str:
    for source in sources:
        token = _utf8_slice(source, span)
        if token:
            return token
    return ""


def _append_word(
    words: list[dict[str, Any]],
    token: str,
    box: Mapping[str, Any] | None,
    width: float,
    height: float,
    line_id: int,
) -> bool:
    text = str(token or "").strip()
    if not text or box is None:
        return False
    try:
        x, y, box_w, box_h = _box_edges(box)
    except (TypeError, ValueError):
        return False
    words.append(
        {
            "t": text,
            "x": round(_norm_coord(x, width), 6),
            "y": round(_norm_coord(y, height), 6),
            "w": round(_norm_coord(box_w, width), 6),
            "h": round(_norm_coord(box_h, height), 6),
            "line": line_id,
        }
    )
    return True


def _emit_lines(
    words: list[dict[str, Any]],
    lines: Sequence[Any],
    width: float,
    height: float,
    line_id: int,
    *sources: str,
) -> int:
    for line in lines:
        if not isinstance(line, Mapping):
            continue
        emitted = 0
        for word in line.get("words") or []:
            if not isinstance(word, Mapping):
                continue
            box = _union_bbox(_bbox_list(word.get("bbox")))
            token = _text_from_span(word.get("span"), *sources) or str(
                word.get("t") or ""
            )
            if _append_word(words, token, box, width, height, line_id):
                emitted += 1
        if not emitted:
            token = _text_from_span(line.get("span"), *sources)
            box = _union_bbox(_bbox_list(line.get("bbox")))
            if _append_word(words, token, box, width, height, line_id):
                emitted = 1
        line_id += 1
    return line_id


def _cell_plain_text(text: str) -> str:
    return _WS.sub(" ", _HTML.sub(" ", text or "")).strip()


def _emit_cell(
    words: list[dict[str, Any]],
    cell: Mapping[str, Any],
    cell_text: str,
    source: str,
    width: float,
    height: float,
    line_id: int,
) -> int:
    lines = cell.get("lines") or []
    if isinstance(lines, Sequence) and not isinstance(lines, (str, bytes)) and lines:
        before = len(words)
        line_id = _emit_lines(words, lines, width, height, line_id, source, cell_text)
        if len(words) > before:
            return line_id
    box = _union_bbox(_bbox_list(cell.get("bbox")))
    plain = _cell_plain_text(cell_text)
    if box is None or not plain:
        return line_id
    emitted = False
    for token in plain.split():
        if _append_word(words, token, box, width, height, line_id):
            emitted = True
    if emitted:
        line_id += 1
    return line_id


def _row_text(rows: Any, row_index: int, col_index: int) -> str:
    if not isinstance(rows, Sequence) or isinstance(rows, (str, bytes)):
        return ""
    if row_index >= len(rows) or not isinstance(rows[row_index], Sequence):
        return ""
    if isinstance(rows[row_index], (str, bytes)) or col_index >= len(rows[row_index]):
        return ""
    return str(rows[row_index][col_index] or "")


def _emit_table(
    words: list[dict[str, Any]],
    item: Mapping[str, Any],
    grounding: Mapping[str, Any],
    source: str,
    width: float,
    height: float,
    line_id: int,
) -> int:
    rows = grounding.get("rows") or []
    if not isinstance(rows, Sequence) or isinstance(rows, (str, bytes)):
        return line_id
    text_rows = item.get("rows")
    for row_index, row in enumerate(rows):
        if not isinstance(row, Sequence) or isinstance(row, (str, bytes)):
            continue
        for col_index, cell in enumerate(row):
            if not isinstance(cell, Mapping):
                continue
            line_id = _emit_cell(
                words,
                cell,
                _row_text(text_rows, row_index, col_index),
                source,
                width,
                height,
                line_id,
            )
    return line_id


def _grounding_source_text(
    item: Mapping[str, Any], grounding: Mapping[str, Any]
) -> str:
    if grounding.get("source") == "caption":
        return str(item.get("caption") or item.get("md") or "")
    return str(item.get("md") or item.get("value") or "")


def _emit_grounding(
    words: list[dict[str, Any]],
    item: Mapping[str, Any],
    grounding: Mapping[str, Any],
    width: float,
    height: float,
    line_id: int,
) -> int:
    source = _grounding_source_text(item, grounding)
    rows = grounding.get("rows")
    if isinstance(rows, Sequence) and not isinstance(rows, (str, bytes)) and rows:
        return _emit_table(words, item, grounding, source, width, height, line_id)
    # An item-level table rectangle is the whole table. Without cell geometry
    # there is nothing honest to highlight, so the quote stays page-only.
    if grounding.get("source") not in (None, "md", "caption"):
        return line_id
    lines = grounding.get("lines") or []
    if not isinstance(lines, Sequence) or isinstance(lines, (str, bytes)):
        return line_id
    return _emit_lines(words, lines, width, height, line_id, source)


def _words_from_item(
    item: Any,
    width: float,
    height: float,
    line_id: int,
) -> tuple[list[dict[str, Any]], int]:
    if not isinstance(item, Mapping):
        return [], line_id
    words: list[dict[str, Any]] = []
    grounding = item.get("grounding")
    if isinstance(grounding, Mapping):
        line_id = _emit_grounding(words, item, grounding, width, height, line_id)
    for child in item.get("items") or []:
        child_words, line_id = _words_from_item(child, width, height, line_id)
        words.extend(child_words)
    return words, line_id


def compact_sidecar_pages(
    sidecar_pages: Sequence[Mapping[str, Any]],
    *,
    slot_id: str | None = None,
) -> LayoutIndex:
    """Turn sidecar JSONL pages into normalized word boxes, keyed by local page."""
    layout: LayoutIndex = {}
    for row in sidecar_pages:
        if not row.get("success", True):
            continue
        try:
            local_page = int(row.get("page_number"))
        except (TypeError, ValueError):
            continue
        try:
            width = float(row.get("page_width") or 0) or 1.0
            height = float(row.get("page_height") or 0) or 1.0
        except (TypeError, ValueError):
            width, height = 1.0, 1.0
        words: list[dict[str, Any]] = []
        line_id = 0
        for item in row.get("items") or []:
            item_words, line_id = _words_from_item(item, width, height, line_id)
            words.extend(item_words)
        page: PageLayout = {
            "width": width,
            "height": height,
            "local_page": local_page,
            "words": words,
        }
        if slot_id:
            page["slot_id"] = slot_id
        layout[local_page] = page
    return layout


def _word_center_inside(word: Mapping[str, Any], host: Mapping[str, Any]) -> bool:
    center_x = _num(word.get("x")) + _num(word.get("w")) / 2
    center_y = _num(word.get("y")) + _num(word.get("h")) / 2
    left = _num(host.get("x"))
    top = _num(host.get("y"))
    return left <= center_x <= left + _num(
        host.get("w")
    ) and top <= center_y <= top + _num(host.get("h"))


def _pdf_page_words(page: Any) -> list[dict[str, Any]]:
    """Normalized visual word boxes from the page text layer."""
    import pymupdf

    visual = page.rect
    width = float(visual.width) or 1.0
    height = float(visual.height) or 1.0
    matrix = page.rotation_matrix
    words: list[dict[str, Any]] = []
    for item in page.get_text("words") or []:
        if len(item) < 7:
            continue
        x0, y0, x1, y1, text, block, line = item[:7]
        token = str(text or "").strip()
        if not token:
            continue
        rect = pymupdf.Rect(float(x0), float(y0), float(x1), float(y1)) * matrix
        if rect.width <= 0 or rect.height <= 0:
            continue
        words.append(
            {
                "t": token,
                "x": round(max(0.0, min(1.0, rect.x0 / width)), 6),
                "y": round(max(0.0, min(1.0, rect.y0 / height)), 6),
                "w": round(max(0.0, min(1.0, rect.width / width)), 6),
                "h": round(max(0.0, min(1.0, rect.height / height)), 6),
                "line": int(block) * 1000 + int(line),
            }
        )
    return words


def merge_pdf_text_words(layout: LayoutIndex, pdf_bytes: bytes) -> LayoutIndex:
    """Add text-layer words the sidecar did not already box.

    Scan-sized pages are left alone. A PDF word whose center sits inside a
    sidecar word is not added again.
    """
    if not pdf_bytes:
        return layout
    import pymupdf

    from .split_ocr import pages_with_large_images

    scans = pages_with_large_images(pdf_bytes)
    document = pymupdf.open(stream=pdf_bytes, filetype="pdf")
    try:
        for index in range(document.page_count):
            local_page = index + 1
            if local_page in scans:
                continue
            page = document[index]
            incoming = _pdf_page_words(page)
            entry = page_entry(layout, local_page)
            if entry is None:
                layout[local_page] = {
                    "width": float(page.rect.width) or 1.0,
                    "height": float(page.rect.height) or 1.0,
                    "local_page": local_page,
                    "words": incoming,
                }
                continue
            existing = entry.get("words")
            if not isinstance(existing, list):
                existing = []
                entry["words"] = existing
            for word in incoming:
                if any(
                    isinstance(host, Mapping) and _word_center_inside(word, host)
                    for host in existing
                ):
                    continue
                existing.append(word)
    finally:
        document.close()
    return layout


async def download_sidecar_pages(url: str | None) -> list[dict[str, Any]]:
    if not url:
        return []
    timeout = httpx.Timeout(FILE_DOWNLOAD_TIMEOUT_S)
    try:
        async with httpx.AsyncClient(timeout=timeout, follow_redirects=True) as client:
            response = await client.get(url)
            response.raise_for_status()
            return parse_sidecar_jsonl(response.text)
    except Exception:
        logger.warning("Failed to download grounded-items sidecar", exc_info=True)
        return []


def stitch_slot_layouts(
    catalog: UploadTypeCatalog,
    parts: Sequence[SplitPartInput],
    pages_by_slot: Mapping[str, Mapping[int, Any]],
    layouts_by_slot: Mapping[str, LayoutIndex],
    *,
    part_order: Sequence[SplitPartInput] | None = None,
    omit_empty: bool = False,
) -> LayoutIndex:
    """Remap slot-local layout pages onto the same global pages as markdown."""
    stitched: LayoutIndex = {}
    next_page = 1
    sequence = (
        list(part_order) if part_order is not None else ordered_parts(catalog, parts)
    )
    for item in sequence:
        local = pages_by_slot.get(item.slot_id) or {}
        local_numbers = sorted(int(page) for page in local)
        slot_layout = layouts_by_slot.get(item.slot_id) or {}
        part_names = [
            str(name).strip()
            for name in (item.document_parts or ())
            if str(name).strip()
        ]
        document_part = (
            part_names[0]
            if len(part_names) == 1
            else ", ".join(part_names)
            if part_names
            else None
        )
        if not local_numbers:
            if omit_empty:
                continue
            stitched[next_page] = _layout_page_stub(
                local_page=1,
                slot_id=item.slot_id,
                document_part=document_part,
            )
            next_page += 1
            continue
        for local_page in local_numbers:
            page = page_entry(slot_layout, local_page)
            if isinstance(page, dict):
                copied = dict(page)
                words = copied.get("words")
                copied["words"] = list(words) if isinstance(words, list) else []
            else:
                copied = {"words": []}
            copied["slot_id"] = item.slot_id
            copied["local_page"] = local_page
            if document_part:
                copied["document_part"] = document_part
            stitched[next_page] = copied
            next_page += 1
    return stitched


def _layout_page_stub(
    *,
    local_page: int,
    slot_id: str,
    document_part: str | None,
) -> dict[str, Any]:
    page: dict[str, Any] = {
        "words": [],
        "local_page": local_page,
        "slot_id": slot_id,
    }
    if document_part:
        page["document_part"] = document_part
    return page


def page_entry(layout: LayoutIndex | None, page: int | None) -> PageLayout | None:
    if layout is None or page is None:
        return None
    value = layout.get(page)
    if isinstance(value, dict):
        return value
    return None


def coerce_page_layout(raw: Mapping[Any, Any] | None) -> LayoutIndex:
    pages: LayoutIndex = {}
    blob = raw or {}
    if "pages" in blob and isinstance(blob.get("pages"), Mapping):
        blob = blob["pages"]
    for key, value in blob.items():
        try:
            number = int(key)
        except (TypeError, ValueError):
            continue
        if isinstance(value, Mapping):
            pages[number] = dict(value)
    return pages


def dump_layout_index(pages: LayoutIndex) -> dict[str, Any]:
    return {
        "schema": LAYOUT_SCHEMA,
        "pages": {str(page): payload for page, payload in sorted(pages.items())},
    }


def _join_words(
    words: Sequence[Mapping[str, Any]],
) -> tuple[str, list[tuple[int, int, int]]]:
    """Normalized haystack plus (word_index, start, end) spans into it."""
    pieces: list[str] = []
    spans: list[tuple[int, int, int]] = []
    cursor = 0
    for index, word in enumerate(words):
        token = _norm_text(str(word.get("t") or ""))
        if not token:
            continue
        if pieces:
            cursor += 1
        start = cursor
        cursor += len(token)
        pieces.append(token)
        spans.append((index, start, cursor))
    return " ".join(pieces), spans


def _token_starts(tokens: Sequence[str]) -> list[int]:
    starts: list[int] = []
    cursor = 0
    for index, token in enumerate(tokens):
        starts.append(cursor)
        cursor += len(token) + (1 if index < len(tokens) - 1 else 0)
    return starts


def _span_from_tokens(
    haystack_tokens: Sequence[str], starts: Sequence[int], hay_at: int, run: int
) -> tuple[int, int]:
    last = hay_at + run - 1
    return starts[hay_at], starts[last] + len(haystack_tokens[last])


def _required_run(nlen: int) -> int:
    """Short quotes must match in full. A partial window is at least 5 tokens."""
    if nlen < _MIN_TOKEN_RUN:
        return nlen
    return _MIN_TOKEN_RUN


def _hyphen_join(left: str, right: str) -> str | None:
    """A line break stores Co- and operative as two words."""
    if not left.endswith("-") or not right.isalpha():
        return None
    return _canon_token(left[:-1] + right)


def _same_canon(left: str, right: str) -> bool:
    if left == right:
        return True
    stripped = left.rstrip(".")
    return bool(stripped) and stripped == right.rstrip(".")


def _punct_glue(
    needle_canon: Sequence[str],
    hay_tokens: Sequence[str],
    needle_at: int,
    hay_at: int,
) -> int:
    """1. + Col. is one token in the quote, and s + / + o is s/o."""
    needle = needle_canon[needle_at]
    if "." not in needle and "/" not in needle:
        return 0
    acc = ""
    limit = min(3, len(hay_tokens) - hay_at)
    for step in range(1, limit + 1):
        acc += hay_tokens[hay_at + step - 1]
        if step > 1 and _same_canon(_canon_token(acc), needle):
            return step
    return 0


def _token_step(
    needle_canon: Sequence[str],
    hay_tokens: Sequence[str],
    hay_canon: Sequence[str],
    needle_at: int,
    hay_at: int,
) -> int:
    if _same_canon(needle_canon[needle_at], hay_canon[hay_at]):
        return 1
    if hay_at + 1 < len(hay_tokens):
        joined = _hyphen_join(hay_tokens[hay_at], hay_tokens[hay_at + 1])
        if joined is not None and _same_canon(joined, needle_canon[needle_at]):
            return 2
    return _punct_glue(needle_canon, hay_tokens, needle_at, hay_at)


def _cover_token_runs(
    needle_tokens: Sequence[str],
    hay_tokens: Sequence[str],
    *,
    min_run: int,
) -> list[tuple[int, int, int]]:
    """In-order haystack runs that cover the quote.

    Each run is (hay index, hay length, needle tokens covered). One mismatched
    word does not drop the rest of the quote. A short quote still has to
    match in full.
    """
    if min_run < 1 or not needle_tokens or not hay_tokens:
        return []
    needle_canon = [_canon_token(token) for token in needle_tokens]
    hay_canon = [_canon_token(token) for token in hay_tokens]
    covered: list[tuple[int, int, int]] = []
    needle_at = 0
    hay_cursor = 0
    nlen = len(needle_canon)
    hlen = len(hay_tokens)
    while needle_at < nlen:
        found: tuple[int, int, int] | None = None
        for hay_at in range(hay_cursor, hlen):
            hay_index = hay_at
            needle_index = needle_at
            hay_run = 0
            needle_run = 0
            while needle_index < nlen and hay_index < hlen:
                step = _token_step(
                    needle_canon, hay_tokens, hay_canon, needle_index, hay_index
                )
                if step == 0:
                    break
                hay_index += step
                hay_run += step
                needle_index += 1
                needle_run += 1
            if needle_run >= min_run:
                found = (hay_at, hay_run, needle_run)
                break
        if found is None:
            needle_at += 1
            continue
        hay_at, hay_run, needle_run = found
        covered.append((hay_at, hay_run, needle_run))
        needle_at += needle_run
        hay_cursor = hay_at + hay_run
    return covered


def _fragment_limits(nlen: int) -> tuple[int, int]:
    """A wrapped column may break every few words. Cover the quote in pieces.

    Two words together can count. A quote shorter than five words still has
    to be covered in full, and a longer quote still needs five words so a
    shared pair does not highlight the page.
    """
    if nlen <= 2:
        return nlen, nlen
    if nlen < _MIN_TOKEN_RUN:
        return 2, nlen
    return 2, _MIN_TOKEN_RUN


def _token_run_spans(
    needle: str, haystack: str, *, fragmented: bool = False
) -> list[tuple[int, int]]:
    needle_tokens = needle.split()
    hay_tokens = haystack.split()
    if not needle_tokens or not hay_tokens:
        return []
    nlen = len(needle_tokens)
    if fragmented:
        min_run, need = _fragment_limits(nlen)
    else:
        min_run = _required_run(nlen)
        need = min_run
    runs = _cover_token_runs(needle_tokens, hay_tokens, min_run=min_run)
    if sum(needle_run for _hay_at, _hay_run, needle_run in runs) < need:
        return []
    starts = _token_starts(hay_tokens)
    return [
        _span_from_tokens(hay_tokens, starts, hay_at, hay_run)
        for hay_at, hay_run, _needle_run in runs
    ]


def _match_segment(
    segment: str, haystack: str, cursor: int, *, fragmented: bool = False
) -> list[tuple[int, int]]:
    suffix = haystack[cursor:]
    lead = len(suffix) - len(suffix.lstrip(" "))
    suffix = suffix[lead:]
    base = cursor + lead
    return [
        (base + start, base + end)
        for start, end in _token_run_spans(segment, suffix, fragmented=fragmented)
    ]


def _quote_segments(needle: str) -> list[str]:
    return [part.strip() for part in needle.split(_GAP_TOKEN) if part.strip()]


def _match_spans(
    needle: str, haystack: str, *, fragmented: bool = False
) -> list[tuple[int, int]]:
    """Character spans of each ellipsis segment. Empty when any segment misses."""
    if not needle or not haystack:
        return []
    segments = _quote_segments(needle)
    if not segments:
        return []
    spans: list[tuple[int, int]] = []
    cursor = 0
    for segment in segments:
        found = _match_segment(segment, haystack, cursor, fragmented=fragmented)
        if not found:
            return []
        spans.extend(found)
        cursor = found[-1][1]
    return spans


def _match_span(needle: str, haystack: str) -> tuple[int, int] | None:
    spans = _match_spans(needle, haystack)
    if not spans:
        return None
    return spans[0]


def _num(value: Any) -> float:
    try:
        return float(value or 0)
    except (TypeError, ValueError):
        return 0.0


def _vertical_span(word: Mapping[str, Any]) -> tuple[float, float]:
    top = _num(word.get("y"))
    return top, top + max(_num(word.get("h")), 0.0)


def _same_visual_line(word: Mapping[str, Any], top: float, bottom: float) -> bool:
    """True when the word overlaps this line by half of the shorter height."""
    word_top, word_bottom = _vertical_span(word)
    overlap = min(word_bottom, bottom) - max(word_top, top)
    if overlap <= 0:
        return False
    shorter = min(word_bottom - word_top, bottom - top)
    if shorter <= 0:
        return abs(word_top - top) <= 0.004
    return overlap >= _LINE_OVERLAP * shorter


def _cluster_visual_lines(
    words: Sequence[Mapping[str, Any]],
) -> list[list[Mapping[str, Any]]]:
    ordered = sorted(words, key=lambda word: (_num(word.get("y")), _num(word.get("x"))))
    clusters: list[list[Mapping[str, Any]]] = []
    bands: list[tuple[float, float]] = []
    for word in ordered:
        placed = False
        for index, (top, bottom) in enumerate(bands):
            if not _same_visual_line(word, top, bottom):
                continue
            clusters[index].append(word)
            word_top, word_bottom = _vertical_span(word)
            bands[index] = (min(top, word_top), max(bottom, word_bottom))
            placed = True
            break
        if not placed:
            clusters.append([word])
            bands.append(_vertical_span(word))
    return clusters


def _median(values: Sequence[float]) -> float:
    ordered = sorted(values)
    mid = len(ordered) // 2
    if len(ordered) % 2:
        return ordered[mid]
    return (ordered[mid - 1] + ordered[mid]) / 2


def _horizontal_gap(prev: Mapping[str, Any], nxt: Mapping[str, Any]) -> float:
    return _num(nxt.get("x")) - (_num(prev.get("x")) + _num(prev.get("w")))


def _split_horizontal(
    line_words: Sequence[Mapping[str, Any]],
) -> list[list[Mapping[str, Any]]]:
    ordered = sorted(line_words, key=lambda word: _num(word.get("x")))
    if len(ordered) < 2:
        return [list(ordered)]
    gaps = [_horizontal_gap(prev, nxt) for prev, nxt in pairwise(ordered)]
    word_gaps = [gap for gap in gaps if 0 < gap <= _COLUMN_GAP]
    threshold = max(
        _COLUMN_GAP, _GAP_MULTIPLE * (_median(word_gaps) if word_gaps else 0.0)
    )
    segments: list[list[Mapping[str, Any]]] = [[ordered[0]]]
    for gap, nxt in zip(gaps, ordered[1:], strict=True):
        if gap > threshold:
            segments.append([nxt])
        else:
            segments[-1].append(nxt)
    return segments


def _word_rect(
    line_words: Sequence[Mapping[str, Any]], page: int
) -> dict[str, float | int]:
    xs = [_num(word.get("x")) for word in line_words]
    ys = [_num(word.get("y")) for word in line_words]
    rights = [_num(word.get("x")) + _num(word.get("w")) for word in line_words]
    bottoms = [_num(word.get("y")) + _num(word.get("h")) for word in line_words]
    x = min(xs)
    y = min(ys)
    return {
        "page": page,
        "x": round(x, 6),
        "y": round(y, 6),
        "w": round(max(rights) - x, 6),
        "h": round(max(bottoms) - y, 6),
    }


def union_line_boxes(
    words: Sequence[Mapping[str, Any]], page: int
) -> list[dict[str, float | int]]:
    """One rectangle per visual line segment, clipped to the matched words.

    Words group by vertical overlap, then split on a column gutter so a
    shared baseline cannot paint one bar across both columns.
    """
    boxes: list[dict[str, float | int]] = []
    for cluster in _cluster_visual_lines(words):
        for segment in _split_horizontal(cluster):
            boxes.append(_word_rect(segment, page))
    boxes.sort(key=lambda box: (_num(box.get("y")), _num(box.get("x"))))
    return boxes


def _column_label_cell(line: str) -> str | None:
    """Cell text when this line is a pipeline column label, else None."""
    text = line.strip()
    match = (
        _COLUMN_LINE.match(text) or _HEADER_LINE.match(text) or _SNO_LINE.match(text)
    )
    if match is None:
        return None
    return match.group(1).strip()


def _wrapped_header_cell(lines: Sequence[str]) -> str | None:
    """Cell under a header whose colon sits on the last wrapped line."""
    if not lines or ":" not in lines[-1]:
        return None
    head, cell = lines[-1].split(":", 1)
    label = " ".join([*lines[:-1], head]).casefold()
    if "page no" not in label or "part" not in label:
        return None
    return cell.strip()


def _quote_rows(quote: str) -> list[list[str]] | None:
    """Table-chunk rows as cell values, with column titles removed.

    None when the quote has no column label, so prose stays on the normal path.
    A blank line starts the next row. Wrapped particulars stay in their cell.
    """
    lines = [line.strip() for line in (quote or "").splitlines()]
    if not any(
        _column_label_cell(line) is not None or _WRAPPED_HEADER_START.match(line)
        for line in lines
    ):
        return None
    rows: list[list[str]] = []
    cells: list[str] = []
    pending: list[str] = []

    def flush_row() -> None:
        nonlocal cells
        if cells:
            rows.append(cells)
            cells = []

    def absorb(extra: list[str]) -> None:
        if not extra:
            return
        text = "\n".join(extra)
        if cells:
            cells[-1] = f"{cells[-1]}\n{text}".strip()
        else:
            cells.append(text)

    for line in lines:
        if not line:
            absorb(pending)
            pending = []
            flush_row()
            continue
        if pending:
            pending.append(line)
            if ":" not in line:
                continue
            cell = _wrapped_header_cell(pending)
            if cell is not None:
                if cell:
                    cells.append(cell)
                pending = []
                continue
            absorb(pending[:-1])
            pending = []
        cell = _column_label_cell(line)
        if cell is not None:
            if cell:
                cells.append(cell)
            continue
        if _WRAPPED_HEADER_START.match(line):
            wrapped = _wrapped_header_cell([line])
            if wrapped is not None:
                if wrapped:
                    cells.append(wrapped)
            else:
                pending = [line]
            continue
        if cells:
            cells[-1] = f"{cells[-1]}\n{line}".strip()
        else:
            cells.append(line)
    absorb(pending)
    flush_row()
    return rows or None


def quote_body(quote: str) -> str:
    """Quote text after pipeline column labels are removed."""
    rows = _quote_rows(quote or "")
    if not rows:
        return quote or ""
    return "\n\n".join(
        "\n".join(cell for cell in row if cell) for row in rows if any(row)
    )


def _matched_words(
    quote: str, words: Sequence[Mapping[str, Any]]
) -> list[Mapping[str, Any]]:
    """Words that carry this quote. Column gaps may split a long quote."""
    if not words or not (quote or "").strip():
        return []
    haystack, spans = _join_words(words)
    found = _match_spans(_norm_text(quote), haystack, fragmented=True)
    if not found:
        return []
    return [
        words[index]
        for index, span_start, span_end in spans
        if any(span_end > start and span_start < end for start, end in found)
    ]


def _boxes_for_words(
    quote: str, page: int, words: Sequence[Mapping[str, Any]]
) -> tuple[list[dict[str, float | int]], str]:
    matched = _matched_words(quote, words)
    if not matched:
        return [], "page_only"
    return union_line_boxes(matched, page), "matched"


def _words_at_or_below(
    words: Sequence[Mapping[str, Any]], floor: float
) -> list[Mapping[str, Any]]:
    return [word for word in words if _num(word.get("y")) >= floor]


def _can_anchor(cell: str) -> bool:
    """A lone letter is too common to choose the row by itself."""
    tokens = _norm_text(cell).split()
    if len(tokens) >= 2:
        return True
    if not tokens:
        return False
    token = tokens[0]
    if any(char.isdigit() for char in token):
        return True
    return len(token) >= 3


def _top_line_band(words: Sequence[Mapping[str, Any]]) -> tuple[float, float]:
    top = min(_num(word.get("y")) for word in words)
    line = [word for word in words if abs(_num(word.get("y")) - top) <= 0.008]
    bottom = max(_num(word.get("y")) + _num(word.get("h")) for word in line)
    return top, max(bottom, top + 0.012)


def _words_on_band(
    words: Sequence[Mapping[str, Any]], top: float, bottom: float
) -> list[Mapping[str, Any]]:
    return [word for word in words if _same_visual_line(word, top, bottom)]


def _row_words(
    cells: Sequence[str], words: Sequence[Mapping[str, Any]]
) -> list[Mapping[str, Any]]:
    """Match one table row. Short cells stay on the anchor cell's line."""
    lengths = [len(_norm_text(cell).split()) for cell in cells]
    ranked = sorted(range(len(cells)), key=lambda index: lengths[index], reverse=True)
    longest = lengths[ranked[0]] if ranked else 0
    # A long party cell has to be the one that places the row. "Respondent No."
    # must not highlight a later "Versus" when the name itself is missing.
    if longest >= _MIN_TOKEN_RUN:
        ranked = [index for index in ranked if lengths[index] >= _MIN_TOKEN_RUN]
    anchor: list[Mapping[str, Any]] = []
    anchor_at: int | None = None
    for index in ranked:
        if not _can_anchor(cells[index]):
            continue
        hit = _matched_words(cells[index], words)
        if not hit:
            continue
        anchor = hit
        anchor_at = index
        break
    if anchor_at is None:
        return []
    chosen = {id(word) for word in anchor}
    top, bottom = _top_line_band(anchor)
    band = _words_on_band(words, top, bottom)
    for index, cell in enumerate(cells):
        if index == anchor_at or not cell.strip():
            continue
        pool = [word for word in band if id(word) not in chosen]
        hit = _matched_words(cell, pool)
        if not hit:
            continue
        anchor.extend(hit)
        chosen.update(id(word) for word in hit)
    return anchor


def _boxes_on_page(
    quote: str, page: int, layout: LayoutIndex
) -> tuple[list[dict[str, float | int]], str]:
    page_layout = layout.get(page)
    if not isinstance(page_layout, Mapping):
        return [], "page_only"
    words = page_layout.get("words") or []
    if not isinstance(words, Sequence) or isinstance(words, (str, bytes)):
        return [], "page_only"
    if not words:
        return [], "page_only"
    rows = _quote_rows(quote)
    if not rows:
        return _boxes_for_words(quote, page, words)
    collected: list[dict[str, float | int]] = []
    floor: float | None = None
    for index, row in enumerate(rows):
        if not any(row):
            continue
        scoped = words if floor is None else _words_at_or_below(words, floor)
        # A one-word row such as "Versus" must sit under the party row.
        if index > 0 and floor is None:
            return [], "page_only"
        hit = _row_words(row, scoped)
        if not hit:
            if index == 0:
                return [], "page_only"
            continue
        boxes = union_line_boxes(hit, page)
        collected.extend(boxes)
        bottom = max(_num(box.get("y")) + _num(box.get("h")) for box in boxes)
        floor = bottom if floor is None else max(floor, bottom)
    if not collected:
        return [], "page_only"
    collected.sort(key=lambda box: (_num(box.get("y")), _num(box.get("x"))))
    return collected, "matched"


def _page_search_order(
    layout: LayoutIndex,
    page: int | None,
    prefer_pages: Sequence[int] | None,
) -> list[int]:
    order: list[int] = []
    seen: set[int] = set()
    for candidate in (*(prefer_pages or ()), page, *sorted(layout)):
        if candidate is None:
            continue
        try:
            number = int(candidate)
        except (TypeError, ValueError):
            continue
        if number in seen:
            continue
        seen.add(number)
        order.append(number)
    return order


def locate_quote(
    quote: str,
    page: int | None,
    layout: LayoutIndex | None,
    *,
    prefer_pages: Sequence[int] | None = None,
) -> tuple[list[dict[str, float | int]], str, int | None]:
    """Map a quote to per-line boxes. Never invents coordinates.

    When ``page`` is missing, search retrieved pages then the rest of the
    layout. Returns (boxes, boxes_status, resolved_page). Never raises.
    """
    resolved_page = page if isinstance(page, int) else None
    try:
        if page is not None and not isinstance(page, int):
            resolved_page = int(page)
    except (TypeError, ValueError):
        resolved_page = None
    try:
        if not isinstance(layout, Mapping) or not layout:
            return [], "unavailable", resolved_page
        needle = (quote or "").strip()
        if not needle:
            if resolved_page is None:
                return [], "unavailable", None
            return [], "page_only", resolved_page
        for candidate in _page_search_order(layout, resolved_page, prefer_pages):
            boxes, status = _boxes_on_page(needle, candidate, layout)
            if status == "matched":
                return boxes, "matched", candidate
        if resolved_page is not None:
            return [], "page_only", resolved_page
        return [], "unavailable", None
    except Exception:
        logger.warning("locate_quote failed", exc_info=True)
        return [], "unavailable", resolved_page


def boxes_on_pages(
    quote: str,
    pages: Sequence[int],
    layout: LayoutIndex | None,
) -> tuple[list[dict[str, float | int]], str, int | None]:
    """Match a quote on the given pages only. Does not scan the rest of the layout.

    Returns (boxes, boxes_status, matched_page). ``matched_page`` is the page
    whose words contain the quote, or the first requested page when none do.
    """
    ordered: list[int] = []
    for page in pages:
        try:
            number = int(page)
        except (TypeError, ValueError):
            continue
        if number not in ordered:
            ordered.append(number)
    if not ordered or not isinstance(layout, Mapping) or not layout:
        return [], "unavailable", ordered[0] if ordered else None
    needle = (quote or "").strip()
    if not needle:
        return [], "page_only", ordered[0]
    for page in ordered:
        boxes, status = _boxes_on_page(needle, page, layout)
        if status == "matched":
            return boxes, "matched", page
    return [], "page_only", ordered[0]


def boxes_for_quote(
    quote: str,
    page: int | None,
    layout: LayoutIndex | None,
    *,
    prefer_pages: Sequence[int] | None = None,
) -> tuple[list[dict[str, float | int]], str]:
    """Map a quote to per-line boxes. Never invents coordinates.

    Returns (boxes, boxes_status) where status is matched, page_only, or
    unavailable.
    """
    boxes, status, _resolved = locate_quote(
        quote, page, layout, prefer_pages=prefer_pages
    )
    return boxes, status


def part_for_page(
    chunks: Sequence[Mapping[str, Any]] | None, page: int | None
) -> str | None:
    if page is None or not chunks:
        return None
    names: list[str] = []
    for chunk in chunks:
        if chunk.get("chunk_kind") == "summary":
            continue
        try:
            chunk_page = int(chunk["page"]) if chunk.get("page") is not None else None
        except (TypeError, ValueError):
            continue
        if chunk_page != page:
            continue
        for name in parts_on_page(chunk.get("document_part")):
            if name not in names:
                names.append(name)
    if not names:
        return None
    return names[0] if len(names) == 1 else " / ".join(names)


async def load_layout_index(
    url: str | None,
    *,
    key: str | None = None,
) -> LayoutIndex:
    """Load a stored compact layout index. Empty on any failure — never re-parse.

    Tries the presigned URL first, then the durable S3 object key.
    """
    from .s3_artifacts import download_json_object

    payload: Any = None
    cleaned_url = (url or "").strip()
    cleaned_key = (key or "").strip()
    if cleaned_url:
        timeout = httpx.Timeout(FILE_DOWNLOAD_TIMEOUT_S)
        try:
            async with httpx.AsyncClient(
                timeout=timeout, follow_redirects=True
            ) as client:
                response = await client.get(cleaned_url)
                response.raise_for_status()
                payload = response.json()
        except Exception:
            logger.warning(
                "Failed to load layout index from URL; trying S3 key",
                exc_info=True,
            )
            payload = None
    if payload is None and cleaned_key:
        payload = download_json_object(cleaned_key)
    if not isinstance(payload, Mapping):
        logger.warning(
            "Layout index empty pages=0 url=%s key=%s",
            safe_url_for_log(cleaned_url) if cleaned_url else None,
            cleaned_key or None,
        )
        return {}
    pages = coerce_page_layout(payload)
    if not pages:
        logger.warning(
            "Layout index empty pages=0 url=%s key=%s",
            safe_url_for_log(cleaned_url) if cleaned_url else None,
            cleaned_key or None,
        )
    return pages
