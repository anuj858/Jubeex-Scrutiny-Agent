"""Compact LlamaParse pages and turn their structured items into legal chunks.

LlamaParse exposes the same page in several representations.  This module keeps
Markdown as the audit/fallback form, but prefers typed items for retrieval text:
headings/text/lists use ``md`` and tables use ``rows``.  It deliberately drops
HTML/CSV duplicates and presigned URLs before workflow state or S3 persistence.
"""

from __future__ import annotations

import html
import re
from collections.abc import Mapping, Sequence
from typing import Any

from llama_index.core.utils import get_tokenizer

TARGET_CHUNK_TOKENS = 500
MAX_CHUNK_TOKENS = 900
CHUNK_OVERLAP_TOKENS = 80

_BR_RE = re.compile(r"<br\s*/?>", re.IGNORECASE)
_TAG_RE = re.compile(r"<[^>]+>")
_HEADING_MARK_RE = re.compile(r"^\s{0,3}#{1,6}\s+")
_EMPHASIS_RE = re.compile(r"(?<!\\)(?:\*\*|__)")
_ROMAN_COLUMN_RE = re.compile(r"^\([ivxlcdm]+\)$", re.IGNORECASE)
_NUMBERED_START_RE = re.compile(
    r"^(?:\(?\d+[.)]|\d{1,2}[./-]\d{1,2}[./-]\d{2,4}\b)", re.IGNORECASE
)


def _as_mapping(value: Any) -> dict[str, Any]:
    if isinstance(value, Mapping):
        return dict(value)
    dump = getattr(value, "model_dump", None)
    if callable(dump):
        payload = dump(mode="json")
        return dict(payload) if isinstance(payload, Mapping) else {}
    return {}


def _page_number(value: Any) -> int | None:
    try:
        number = int(value)
    except (TypeError, ValueError):
        return None
    return number if number > 0 else None


def _pages_by_number(container: Any) -> dict[int, dict[str, Any]]:
    data = _as_mapping(container)
    raw_pages = data.get("pages")
    if raw_pages is None:
        raw_pages = getattr(container, "pages", None)
    pages: dict[int, dict[str, Any]] = {}
    for raw in raw_pages or []:
        page = _as_mapping(raw)
        number = _page_number(page.get("page_number"))
        if number is not None:
            pages[number] = page
    return pages


def _compact_bbox(raw: Any) -> list[dict[str, Any]]:
    boxes: list[dict[str, Any]] = []
    for value in raw or []:
        box = _as_mapping(value)
        compact = {
            key: box[key]
            for key in ("x", "y", "w", "h", "confidence", "label")
            if box.get(key) is not None
        }
        if compact:
            boxes.append(compact)
    return boxes


def _compact_item(raw: Any, item_index: int) -> dict[str, Any] | None:
    item = _as_mapping(raw)
    kind = str(item.get("type") or "").strip().lower()
    if not kind:
        return None
    compact: dict[str, Any] = {"type": kind, "item_index": item_index}
    for key in (
        "md",
        "value",
        "level",
        "ordered",
        "merged_from_pages",
        "merged_into_page",
    ):
        value = item.get(key)
        if value is not None and value != "":
            compact[key] = value
    if kind == "table":
        rows: list[list[str]] = []
        for raw_row in item.get("rows") or []:
            if isinstance(raw_row, Sequence) and not isinstance(raw_row, (str, bytes)):
                rows.append([str(cell or "") for cell in raw_row])
        if rows:
            compact["rows"] = rows
        concerns = item.get("parse_concerns")
        if isinstance(concerns, list):
            compact["parse_concerns"] = [
                _as_mapping(concern) for concern in concerns if _as_mapping(concern)
            ]
    boxes = _compact_bbox(item.get("bbox"))
    if boxes:
        compact["bbox"] = boxes
    return compact


def extract_structured_pages(parse_result: Any) -> dict[int, dict[str, Any]]:
    """Return compact page objects from a LlamaCloud ParsingGetResponse.

    The result is JSON-safe and intentionally contains no table HTML/CSV or
    result-content URLs.  Pages are joined by their explicit ``page_number``.
    """

    payload = _as_mapping(parse_result)
    markdown_pages = _pages_by_number(
        payload.get("markdown") or getattr(parse_result, "markdown", None)
    )
    text_pages = _pages_by_number(
        payload.get("text") or getattr(parse_result, "text", None)
    )
    item_pages = _pages_by_number(
        payload.get("items") or getattr(parse_result, "items", None)
    )
    metadata_pages = _pages_by_number(
        payload.get("metadata") or getattr(parse_result, "metadata", None)
    )
    numbers = sorted(
        set(markdown_pages) | set(text_pages) | set(item_pages) | set(metadata_pages)
    )
    pages: dict[int, dict[str, Any]] = {}
    for number in numbers:
        markdown_page = markdown_pages.get(number, {})
        text_page = text_pages.get(number, {})
        item_page = item_pages.get(number, {})
        metadata_page = metadata_pages.get(number, {})
        metadata = {
            key: metadata_page[key]
            for key in (
                "confidence",
                "original_orientation_angle",
                "printed_page_number",
                "cost_optimized",
                "triggered_auto_mode",
            )
            if metadata_page.get(key) is not None
        }
        items = [
            compact
            for index, raw in enumerate(item_page.get("items") or [])
            if (compact := _compact_item(raw, index)) is not None
        ]
        pages[number] = {
            "page_number": number,
            "success": bool(
                item_page.get("success", markdown_page.get("success", True))
            ),
            "markdown": str(markdown_page.get("markdown") or ""),
            "text": str(text_page.get("text") or ""),
            "metadata": metadata,
            "items": items,
        }
    return pages


def coerce_structured_pages(raw: Mapping[Any, Any] | None) -> dict[int, dict[str, Any]]:
    pages: dict[int, dict[str, Any]] = {}
    for key, value in (raw or {}).items():
        number = _page_number(key)
        page = _as_mapping(value)
        if number is None or not page:
            continue
        page["page_number"] = number
        pages[number] = page
    return pages


def clean_legal_text(value: Any) -> str:
    """Remove presentation noise without rewriting legal content."""

    text = html.unescape(str(value or "")).replace("\r\n", "\n").replace("\r", "\n")
    text = _BR_RE.sub("\n", text)
    text = _TAG_RE.sub("", text)
    text = _HEADING_MARK_RE.sub("", text)
    text = _EMPHASIS_RE.sub("", text)
    text = re.sub(r"[ \t]+", " ", text)
    text = re.sub(r" *\n *", "\n", text)
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip()


def _unique_headers(raw: list[str], width: int) -> list[str]:
    headers: list[str] = []
    for index in range(width):
        value = clean_legal_text(raw[index] if index < len(raw) else "")
        base = value or f"Column {index + 1}"
        headers.append(base)
    totals: dict[str, int] = {}
    for header in headers:
        totals[header.casefold()] = totals.get(header.casefold(), 0) + 1
    seen: dict[str, int] = {}
    out: list[str] = []
    for index, header in enumerate(headers):
        folded = header.casefold()
        seen[folded] = seen.get(folded, 0) + 1
        out.append(f"{header} (column {index + 1})" if totals[folded] > 1 else header)
    return out


def _looks_like_header(row: list[str]) -> bool:
    cells = [clean_legal_text(cell) for cell in row]
    joined = " ".join(cells).casefold()
    if not any(cells) or _NUMBERED_START_RE.match(cells[0] or ""):
        return False
    cues = ("s.no", "serial", "particular", "remarks", "date", "page no", "description")
    return any(cue in joined for cue in cues)


def _table_blocks(item: Mapping[str, Any]) -> list[dict[str, Any]]:
    rows = [list(row) for row in item.get("rows") or [] if isinstance(row, list)]
    if not rows:
        fallback = clean_legal_text(item.get("md") or item.get("value"))
        return (
            [{"text": fallback, "kind": "table", "row_start": 0, "row_end": 0}]
            if fallback
            else []
        )
    width = max((len(row) for row in rows), default=0)
    has_header = _looks_like_header(rows[0])
    raw_headers = list(rows[0]) if has_header else []
    start = 1 if has_header else 0
    if has_header:
        # LlamaParse commonly emits a second line for grouped table headings,
        # followed by a roman-number column guide. Fold that second line into
        # the field names instead of indexing it as if it were a legal row.
        while start < len(rows):
            cells = [clean_legal_text(cell) for cell in rows[start]]
            nonempty = [cell for cell in cells if cell]
            if nonempty and all(_ROMAN_COLUMN_RE.fullmatch(cell) for cell in nonempty):
                start += 1
                continue
            if (
                cells
                and not cells[0]
                and any("content" in cell.casefold() for cell in nonempty)
            ):
                raw_headers = [
                    " — ".join(
                        part
                        for part in (
                            clean_legal_text(
                                raw_headers[index] if index < len(raw_headers) else ""
                            ),
                            cells[index] if index < len(cells) else "",
                        )
                        if part
                    )
                    for index in range(width)
                ]
                start += 1
                continue
            break
    headers = _unique_headers(raw_headers, width)
    blocks: list[dict[str, Any]] = []
    for row_index, row in enumerate(rows[start:], start=start):
        cleaned = [clean_legal_text(cell) for cell in row]
        if cleaned and all(
            _ROMAN_COLUMN_RE.fullmatch(cell or "") for cell in cleaned if cell
        ):
            continue
        fields = [
            f"{headers[index]}: {cell}" for index, cell in enumerate(cleaned) if cell
        ]
        text = "\n".join(fields).strip()
        if text:
            blocks.append(
                {
                    "text": text,
                    "kind": "table_row",
                    "row_start": row_index,
                    "row_end": row_index,
                }
            )
    return blocks


def page_semantic_blocks(page: Mapping[str, Any]) -> list[dict[str, Any]]:
    """Normalize one compact page into ordered, provenance-carrying blocks."""

    blocks: list[dict[str, Any]] = []
    for raw in page.get("items") or []:
        item = _as_mapping(raw)
        kind = str(item.get("type") or "").strip().lower()
        if kind == "header":
            continue
        item_index = int(item.get("item_index") or 0)
        if kind == "table":
            item_blocks = _table_blocks(item)
        else:
            text = clean_legal_text(item.get("md") or item.get("value"))
            item_blocks = [{"text": text, "kind": kind}] if text else []
        concerns = item.get("parse_concerns") or []
        for block in item_blocks:
            block.update(
                {
                    "item_index": item_index,
                    "item_type": kind,
                    "bbox": item.get("bbox") or [],
                    "parse_concerns": concerns,
                }
            )
            if kind == "heading":
                block["heading_level"] = int(item.get("level") or 1)
            blocks.append(block)
    if blocks:
        return blocks
    fallback = clean_legal_text(page.get("markdown") or page.get("text"))
    if not fallback:
        return []
    return [
        {
            "text": fallback,
            "kind": "markdown_fallback",
            "item_type": "markdown_fallback",
            "item_index": 0,
            "bbox": [],
            "parse_concerns": [],
        }
    ]


def _token_count(text: str) -> int:
    return len(get_tokenizer()(text or ""))


def _split_oversized(text: str, *, max_tokens: int, overlap_tokens: int) -> list[str]:
    tokenizer = get_tokenizer()
    tokens = tokenizer(text)
    if len(tokens) <= max_tokens:
        return [text]
    # Token ids from LlamaIndex's default tokenizer are decoded through tiktoken
    # only when available; otherwise use conservative character windows.
    try:
        import tiktoken

        encoding = tiktoken.get_encoding("cl100k_base")
        windows: list[str] = []
        start = 0
        while start < len(tokens):
            end = min(start + max_tokens, len(tokens))
            windows.append(encoding.decode(tokens[start:end]).strip())
            if end >= len(tokens):
                break
            start = max(0, end - overlap_tokens)
        return [window for window in windows if window]
    except (ImportError, KeyError, ValueError):
        max_chars = max_tokens * 3
        overlap_chars = overlap_tokens * 3
        return [
            text[start : start + max_chars].strip()
            for start in range(0, len(text), max(1, max_chars - overlap_chars))
            if text[start : start + max_chars].strip()
        ]


def build_semantic_chunks(
    page: Mapping[str, Any],
    *,
    target_tokens: int = TARGET_CHUNK_TOKENS,
    max_tokens: int = MAX_CHUNK_TOKENS,
    overlap_tokens: int = CHUNK_OVERLAP_TOKENS,
) -> list[dict[str, Any]]:
    """Group semantic blocks without splitting ordinary clauses or table rows."""

    if not 0 <= overlap_tokens < max_tokens:
        raise ValueError("overlap_tokens must be non-negative and below max_tokens")
    if target_tokens <= 0 or target_tokens > max_tokens:
        raise ValueError("target_tokens must be positive and no larger than max_tokens")
    source = page_semantic_blocks(page)
    expanded: list[dict[str, Any]] = []
    for block in source:
        text = str(block.get("text") or "").strip()
        for piece_index, piece in enumerate(
            _split_oversized(text, max_tokens=max_tokens, overlap_tokens=overlap_tokens)
        ):
            copied = dict(block)
            copied["text"] = piece
            copied["piece_index"] = piece_index
            expanded.append(copied)

    chunks: list[dict[str, Any]] = []
    current: list[dict[str, Any]] = []
    headings: list[tuple[int, str]] = []

    def flush(*, keep_overlap: bool = True) -> None:
        nonlocal current
        if not current:
            return
        text = "\n\n".join(str(block["text"]) for block in current).strip()
        concerns = [
            str(concern.get("type") or concern.get("details") or concern)
            for block in current
            for concern in block.get("parse_concerns") or []
            if isinstance(concern, Mapping)
        ]
        rows = [
            int(block[key])
            for block in current
            for key in ("row_start", "row_end")
            if block.get(key) is not None
        ]
        chunks.append(
            {
                "text": text,
                "content_kind": (
                    "table_rows"
                    if all(block.get("kind") == "table_row" for block in current)
                    else "structured_text"
                ),
                "item_start": min(
                    int(block.get("item_index") or 0) for block in current
                ),
                "item_end": max(int(block.get("item_index") or 0) for block in current),
                "item_types": sorted(
                    {str(block.get("item_type") or "text") for block in current}
                ),
                "heading_path": list(current[-1].get("heading_path") or []),
                "table_row_start": min(rows) if rows else None,
                "table_row_end": max(rows) if rows else None,
                "parse_concerns": sorted(set(concerns)),
                "token_count": _token_count(text),
            }
        )
        previous = current[-1:] if keep_overlap else []
        current = (
            previous
            if previous and _token_count(str(previous[0]["text"])) <= overlap_tokens
            else []
        )

    for block in expanded:
        if block.get("kind") == "heading":
            if current:
                flush(keep_overlap=False)
            level = int(block.get("heading_level") or 1)
            headings = [
                (old_level, text) for old_level, text in headings if old_level < level
            ]
            headings.append((level, str(block["text"])))
        block["heading_path"] = [heading for _, heading in headings]
        candidate = current + [block]
        candidate_text = "\n\n".join(str(item["text"]) for item in candidate)
        if current and _token_count(candidate_text) > target_tokens:
            flush()
        current.append(block)
        if (
            _token_count("\n\n".join(str(item["text"]) for item in current))
            >= max_tokens
        ):
            flush()
    flush(keep_overlap=False)
    chunks = [chunk for chunk in chunks if chunk.get("text")]

    # Consecutive parser headings and isolated folio-like fragments create
    # poor embeddings. Merge them with adjacent content on the same source
    # page; no text or provenance is discarded and page boundaries are never
    # crossed because this function operates on one page at a time.
    merged: list[dict[str, Any]] = []
    for chunk in chunks:
        text = str(chunk.get("text") or "").strip()
        low_information = len(text) < 40 or (
            set(chunk.get("item_types") or []) <= {"heading", "text"}
            and _token_count(text) < 10
        )
        if low_information and merged:
            previous = merged[-1]
            combined = f"{previous['text']}\n\n{text}".strip()
            if _token_count(combined) <= max_tokens:
                previous["text"] = combined
                previous["item_end"] = max(previous["item_end"], chunk["item_end"])
                previous["item_types"] = sorted(
                    set(previous.get("item_types") or [])
                    | set(chunk.get("item_types") or [])
                )
                previous["parse_concerns"] = sorted(
                    set(previous.get("parse_concerns") or [])
                    | set(chunk.get("parse_concerns") or [])
                )
                previous["token_count"] = _token_count(combined)
                continue
        merged.append(dict(chunk))

    if len(merged) > 1:
        first = merged[0]
        first_text = str(first.get("text") or "").strip()
        if (
            len(first_text) < 40
            and _token_count(f"{first_text}\n\n{merged[1]['text']}") <= max_tokens
        ):
            second = merged[1]
            second["text"] = f"{first_text}\n\n{second['text']}".strip()
            second["item_start"] = min(first["item_start"], second["item_start"])
            second["item_types"] = sorted(
                set(first.get("item_types") or []) | set(second.get("item_types") or [])
            )
            second["token_count"] = _token_count(second["text"])
            merged.pop(0)
    return merged
