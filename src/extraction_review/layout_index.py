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
from typing import Any

import httpx

from .process_file import FILE_DOWNLOAD_TIMEOUT_S
from .split_upload import SplitPartInput, UploadTypeCatalog, ordered_parts

logger = logging.getLogger(__name__)

LAYOUT_SCHEMA = "layout_index_v1"
LAYOUT_ARTIFACT_URL_KEY = "layout_artifact_url"
LAYOUT_ARTIFACT_KEY_KEY = "layout_artifact_key"
GRANULAR_BBOXES = ("word", "line")
LINE_Y_EPSILON = 0.012

_WS = re.compile(r"\s+")
_PUNCT = re.compile(r"[^\w./-]+")

PageLayout = dict[str, Any]
LayoutIndex = dict[int, PageLayout]


_ORDINAL = re.compile(r"^(\d{1,2})(st|nd|rd|th)$")
_SLASH_DATE = re.compile(r"^\d{1,4}([./-]\d{1,4}){1,2}$")
_MIN_TOKEN_RUN = 5
_MAX_NEEDLE_TOKENS = 48


def _norm_text(text: str) -> str:
    collapsed = _WS.sub(" ", (text or "").replace("…", " ").replace("...", " "))
    collapsed = _PUNCT.sub(" ", collapsed.lower())
    return _WS.sub(" ", collapsed).strip()


def _canon_token(token: str) -> str:
    """Collapse 17th / 17/4 / 17.04.2026 so OCR and model quotes still align."""
    ordinal = _ORDINAL.match(token)
    if ordinal:
        return ordinal.group(1)
    if _SLASH_DATE.match(token):
        return token.split("/")[0].split(".")[0].split("-")[0]
    return token


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
    """Ensure the existing parse job requests word and line grounding."""
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
            if not isinstance(item, Mapping):
                continue
            md = str(item.get("md") or "")
            grounding = item.get("grounding")
            if not isinstance(grounding, Mapping):
                continue
            if grounding.get("source") not in (None, "md", "caption"):
                continue
            for line in grounding.get("lines") or []:
                if not isinstance(line, Mapping):
                    continue
                line_words = line.get("words") or []
                emitted = 0
                for word in line_words:
                    if not isinstance(word, Mapping):
                        continue
                    box = _bbox_dict(word.get("bbox"))
                    if box is None:
                        continue
                    span = word.get("span") or [0, 0]
                    try:
                        start, end = int(span[0]), int(span[1])
                    except (TypeError, ValueError, IndexError):
                        start, end = 0, 0
                    token = (
                        md[start:end]
                        if md and end > start
                        else str(word.get("t") or "")
                    )
                    token = token.strip()
                    if not token:
                        continue
                    words.append(
                        {
                            "t": token,
                            "x": round(_norm_coord(box.get("x"), width), 6),
                            "y": round(_norm_coord(box.get("y"), height), 6),
                            "w": round(
                                _norm_coord(box.get("w", box.get("width")), width), 6
                            ),
                            "h": round(
                                _norm_coord(box.get("h", box.get("height")), height), 6
                            ),
                            "line": line_id,
                        }
                    )
                    emitted += 1
                if not emitted:
                    box = _bbox_dict(line.get("bbox"))
                    span = line.get("span") or [0, 0]
                    try:
                        start, end = int(span[0]), int(span[1])
                    except (TypeError, ValueError, IndexError):
                        start, end = 0, 0
                    token = md[start:end].strip() if md and end > start else ""
                    if box is not None and token:
                        words.append(
                            {
                                "t": token,
                                "x": round(_norm_coord(box.get("x"), width), 6),
                                "y": round(_norm_coord(box.get("y"), height), 6),
                                "w": round(
                                    _norm_coord(box.get("w", box.get("width")), width),
                                    6,
                                ),
                                "h": round(
                                    _norm_coord(
                                        box.get("h", box.get("height")), height
                                    ),
                                    6,
                                ),
                                "line": line_id,
                            }
                        )
                line_id += 1
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
) -> LayoutIndex:
    """Remap slot-local layout pages onto the same global pages as markdown."""
    stitched: LayoutIndex = {}
    next_page = 1
    for item in ordered_parts(catalog, parts):
        local = pages_by_slot.get(item.slot_id) or {}
        local_numbers = sorted(int(page) for page in local)
        slot_layout = layouts_by_slot.get(item.slot_id) or {}
        if not local_numbers:
            next_page += 1
            continue
        for local_page in local_numbers:
            page = page_entry(slot_layout, local_page)
            if isinstance(page, dict) and page.get("words"):
                copied = dict(page)
                copied["slot_id"] = item.slot_id
                copied["local_page"] = local_page
                stitched[next_page] = copied
            next_page += 1
    return stitched


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


def _token_run_span(needle: str, haystack: str) -> tuple[int, int] | None:
    needle_tokens = needle.split()[:_MAX_NEEDLE_TOKENS]
    haystack_tokens = haystack.split()
    if not needle_tokens or not haystack_tokens:
        return None
    needle_canon = [_canon_token(token) for token in needle_tokens]
    haystack_canon = [_canon_token(token) for token in haystack_tokens]
    starts = _token_starts(haystack_tokens)
    nlen = len(needle_canon)
    hlen = len(haystack_canon)
    if nlen >= 2 and hlen >= nlen:
        for hay_at in range(hlen - nlen + 1):
            if haystack_canon[hay_at : hay_at + nlen] == needle_canon:
                return _span_from_tokens(haystack_tokens, starts, hay_at, nlen)
    min_run = min(_MIN_TOKEN_RUN, nlen) if nlen >= 3 else nlen
    if min_run < 2:
        return None
    for run in range(nlen - 1, min_run - 1, -1):
        for needle_at in range(nlen - run + 1):
            window = needle_canon[needle_at : needle_at + run]
            for hay_at in range(hlen - run + 1):
                if haystack_canon[hay_at : hay_at + run] != window:
                    continue
                return _span_from_tokens(haystack_tokens, starts, hay_at, run)
    return None


def _match_span(needle: str, haystack: str) -> tuple[int, int] | None:
    if not needle or not haystack:
        return None
    at = haystack.find(needle)
    if at >= 0:
        return at, at + len(needle)
    snippet = needle[:80].strip()
    if len(snippet) >= 12:
        at = haystack.find(snippet)
        if at >= 0:
            return at, at + len(snippet)
    return _token_run_span(needle, haystack)


def _group_line_key(word: Mapping[str, Any]) -> tuple[int, float]:
    line = word.get("line")
    try:
        return (0, float(int(line)))
    except (TypeError, ValueError):
        try:
            return (1, round(float(word.get("y") or 0) / LINE_Y_EPSILON))
        except (TypeError, ValueError):
            return (1, 0.0)


def union_line_boxes(
    words: Sequence[Mapping[str, Any]], page: int
) -> list[dict[str, float | int]]:
    """One rectangle per visual line, clipped to the matched words."""
    grouped: dict[tuple[int, float], list[Mapping[str, Any]]] = {}
    order: list[tuple[int, float]] = []
    for word in words:
        key = _group_line_key(word)
        if key not in grouped:
            grouped[key] = []
            order.append(key)
        grouped[key].append(word)
    boxes: list[dict[str, float | int]] = []
    for key in order:
        line_words = grouped[key]
        xs = [float(w.get("x") or 0) for w in line_words]
        ys = [float(w.get("y") or 0) for w in line_words]
        rights = [float(w.get("x") or 0) + float(w.get("w") or 0) for w in line_words]
        bottoms = [float(w.get("y") or 0) + float(w.get("h") or 0) for w in line_words]
        x = min(xs)
        y = min(ys)
        boxes.append(
            {
                "page": page,
                "x": round(x, 6),
                "y": round(y, 6),
                "w": round(max(rights) - x, 6),
                "h": round(max(bottoms) - y, 6),
            }
        )
    return boxes


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
    haystack, spans = _join_words(words)
    found = _match_span(_norm_text(quote), haystack)
    if found is None:
        return [], "page_only"
    start, end = found
    matched = [
        words[index]
        for index, span_start, span_end in spans
        if span_end > start and span_start < end
    ]
    if not matched:
        return [], "page_only"
    return union_line_boxes(matched, page), "matched"


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
        name = str(chunk.get("document_part") or "").strip()
        if name and name not in names:
            names.append(name)
    return names[0] if names else None


async def load_layout_index(url: str | None) -> LayoutIndex:
    """GET a stored compact layout index. Empty on any failure — never re-parse."""
    if not (url or "").strip():
        return {}
    timeout = httpx.Timeout(FILE_DOWNLOAD_TIMEOUT_S)
    try:
        async with httpx.AsyncClient(timeout=timeout, follow_redirects=True) as client:
            response = await client.get(url)
            response.raise_for_status()
            payload = response.json()
    except Exception:
        logger.warning("Failed to load layout index from %s", url, exc_info=True)
        return {}
    if not isinstance(payload, Mapping):
        return {}
    return coerce_page_layout(payload)
