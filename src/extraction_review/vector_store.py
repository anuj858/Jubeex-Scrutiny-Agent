"""Pinecone vector store with integrated embeddings (llama-text-embed-v2).

Two jobs, kept separate:

- Write (`process-file`): `build_page_records` / `upsert_records` store page text
  plus metadata (`file_hash`, `document_part`, `chunk_kind`, page numbers).
- Read (`scrutiny-check`): `search_filing_chunks` / `gather_filing_evidence`
  return hits for one filing. Each concurrent defect queries Pinecone with
  its own captions. They do not decide what the LLM sees.

What to send the model (log-gap cutoff, per-defect page budget) lives in
`document_parts.select_chunks_for_defect`, not here.

This project does not run a separate embedding model. When VECTOR_BACKEND=pinecone,
text is upserted via Pinecone's integrated embedding index; Pinecone embeds server-side.
"""

from __future__ import annotations

import hashlib
import logging
import os
import re
import threading
from concurrent.futures import ThreadPoolExecutor
from typing import Any, cast

from pinecone import Pinecone

from .document_parts import (
    EvidenceQuery,
    _part_match,
    canonical_document_part_id,
    document_subtype_id as infer_document_subtype_id,
    family_split_name,
    parts_on_page,
    pool_search_queries,
)
from .parsed_document import build_semantic_chunks, coerce_structured_pages

logger = logging.getLogger(__name__)

# Record field that Pinecone embeds (must match index field_map.text).
# Existing `legal-document-knowledge` index uses `normalized_text`.
DEFAULT_TEXT_FIELD = "normalized_text"
MAX_CHUNK_CHARS = 3500
CHUNK_OVERLAP = 200
VECTOR_SCHEMA_VERSION = 4
EMPTY_PAGE_MARKER = "[No extractable text on this page; visual review required.]"
LOW_INFORMATION_CHARS = 40
_PAGE_SEGMENT_RE = re.compile(
    r"^--- page (?P<page>\d+) \((?P<role>primary|overlap)\) ---\n",
    re.MULTILINE,
)
_FOLIO_ONLY_RE = re.compile(
    r"^(?:\d{1,4}|[A-Z](?:[-/]?\d{1,4})?|[IVXLCDM]{1,8})$",
    re.IGNORECASE,
)
_NON_SEMANTIC_HEADINGS = {
    "SUPREME COURT OF INDIA",
    "IN THE SUPREME COURT OF INDIA",
    "IN THE HON BLE SUPREME COURT OF INDIA",
    "IN THE HONBLE SUPREME COURT OF INDIA",
    "IN THE HONOURABLE SUPREME COURT OF INDIA",
}

pinecone_api_key = os.getenv("PINECONE_API_KEY")
pinecone_index_name = os.getenv("PINECONE_INDEX", "legal-document-knowledge")
pinecone_cloud = os.getenv("PINECONE_CLOUD", "aws")
pinecone_region = os.getenv("PINECONE_REGION", "us-east-1")
pinecone_embed_model = os.getenv("PINECONE_EMBED_MODEL", "llama-text-embed-v2")
pinecone_namespace = os.getenv("PINECONE_NAMESPACE", "jubeex-filings")
pinecone_text_field = os.getenv("PINECONE_TEXT_FIELD", DEFAULT_TEXT_FIELD)
vector_backend = (os.getenv("VECTOR_BACKEND") or "pinecone").strip().lower()

_cached_client: Pinecone | None = None
_cached_index: Any = None
_cached_text_field: str | None = None
_cache_lock = threading.RLock()
_query_semaphore: threading.Semaphore | None = None

# Scrutiny retrieval. Env overrides these; do not duplicate the numbers elsewhere.
DEFAULT_TOP_K = 8
DEFAULT_MAX_CHUNKS = 12
DEFAULT_POOL_MAX_CHUNKS = 60
# urllib3 default pool is 10; scrutiny fans out many queries (defects × captions).
DEFAULT_POOL_THREADS = 32
DEFAULT_QUERY_CONCURRENCY = 10


def _int_env(name: str, default: int) -> int:
    try:
        return int(os.getenv(name, "") or default)
    except ValueError:
        return default


def scrutiny_top_k() -> int:
    return _int_env("SCRUTINY_TOP_K", DEFAULT_TOP_K)


def scrutiny_max_chunks() -> int:
    return _int_env("SCRUTINY_MAX_CHUNKS", DEFAULT_MAX_CHUNKS)


def scrutiny_pool_max_chunks() -> int:
    return _int_env("SCRUTINY_POOL_MAX_CHUNKS", DEFAULT_POOL_MAX_CHUNKS)


def pinecone_pool_threads() -> int:
    return max(1, _int_env("PINECONE_POOL_THREADS", DEFAULT_POOL_THREADS))


def pinecone_query_concurrency() -> int:
    return max(1, _int_env("PINECONE_QUERY_CONCURRENCY", DEFAULT_QUERY_CONCURRENCY))


def pinecone_enabled() -> bool:
    return vector_backend == "pinecone" and bool(pinecone_api_key)


def scoped_record_base_id(
    *,
    file_hash: str,
    organization_id: str,
    workspace_id: str,
    index_generation: str,
) -> str:
    """Opaque, tenant-scoped prefix for every record in one completed run."""
    values = [file_hash, organization_id, workspace_id, index_generation]
    if any(not str(value or "").strip() for value in values):
        raise ValueError(
            "file_hash, organization_id, workspace_id, and index_generation are "
            "required for vector indexing"
        )
    material = "\x1f".join(str(value).strip() for value in values)
    return hashlib.sha256(material.encode("utf-8")).hexdigest()


def _pinecone_query_gate() -> threading.Semaphore:
    """Cap concurrent Pinecone HTTP calls across all scrutiny threads."""
    global _query_semaphore
    if _query_semaphore is None:
        with _cache_lock:
            if _query_semaphore is None:
                _query_semaphore = threading.Semaphore(pinecone_query_concurrency())
    return _query_semaphore


def get_pinecone_client() -> Pinecone:
    global _cached_client
    if not pinecone_api_key:
        raise ValueError("PINECONE_API_KEY is not set")
    if _cached_client is None:
        with _cache_lock:
            if _cached_client is None:
                _cached_client = Pinecone(
                    api_key=pinecone_api_key,
                    pool_threads=pinecone_pool_threads(),
                )
    return _cached_client


def resolve_text_field(pc: Pinecone | None = None) -> str:
    """Read the index field_map when available; fall back to PINECONE_TEXT_FIELD."""
    global _cached_text_field
    if _cached_text_field:
        return _cached_text_field
    client = pc or get_pinecone_client()
    try:
        desc = client.describe_index(pinecone_index_name)
        embed = getattr(desc, "embed", None) or (
            desc.get("embed") if isinstance(desc, dict) else None
        )
        if embed:
            field_map = getattr(embed, "field_map", None) or (
                embed.get("field_map") if isinstance(embed, dict) else None
            )
            if isinstance(field_map, dict) and field_map.get("text"):
                _cached_text_field = str(field_map["text"])
                return _cached_text_field
    except Exception as e:
        logger.debug("Could not resolve Pinecone text field from index: %s", e)
    with _cache_lock:
        if _cached_text_field is None:
            _cached_text_field = pinecone_text_field
    return _cached_text_field


def _index_embed_config(description: Any) -> tuple[str | None, str | None]:
    """Return the integrated model and text field from an SDK response."""
    embed = getattr(description, "embed", None) or (
        description.get("embed") if isinstance(description, dict) else None
    )
    if not embed:
        return None, None
    model = getattr(embed, "model", None) or (
        embed.get("model") if isinstance(embed, dict) else None
    )
    field_map = getattr(embed, "field_map", None) or (
        embed.get("field_map") if isinstance(embed, dict) else None
    )
    text_field = field_map.get("text") if isinstance(field_map, dict) else None
    return (
        str(model).strip() if model else None,
        str(text_field).strip() if text_field else None,
    )


def validate_index_contract(description: Any) -> str:
    """Fail fast when an existing Pinecone index cannot embed our records."""
    model, text_field = _index_embed_config(description)
    if not model or not text_field:
        raise RuntimeError(
            f"Pinecone index {pinecone_index_name!r} is not an integrated-embedding "
            "index with a text field_map"
        )
    if model != pinecone_embed_model:
        raise RuntimeError(
            f"Pinecone index {pinecone_index_name!r} uses embedding model {model!r}; "
            f"configured model is {pinecone_embed_model!r}"
        )
    if pinecone_text_field and text_field != pinecone_text_field:
        raise RuntimeError(
            f"Pinecone index {pinecone_index_name!r} embeds field {text_field!r}; "
            f"configured field is {pinecone_text_field!r}"
        )
    return text_field


def ensure_index(pc: Pinecone | None = None) -> Any:
    """Create an integrated-embedding index if missing; return a data-plane Index."""
    global _cached_index
    if _cached_index is not None:
        return _cached_index
    with _cache_lock:
        if _cached_index is not None:
            return _cached_index
        client = pc or get_pinecone_client()
        name = pinecone_index_name
        if not client.has_index(name):
            logger.info(
                "[Pinecone] Creating index %s with embed model %s (%s/%s, field=%s)",
                name,
                pinecone_embed_model,
                pinecone_cloud,
                pinecone_region,
                pinecone_text_field,
            )
            client.create_index_for_model(
                name=name,
                cloud=pinecone_cloud,
                region=pinecone_region,
                embed=cast(
                    Any,
                    {
                        "model": pinecone_embed_model,
                        "field_map": {"text": pinecone_text_field},
                    },
                ),
            )
            logger.info("[Pinecone] Index %s created", name)
        else:
            logger.info("[Pinecone] Using existing index %s", name)
        description = client.describe_index(name)
        resolved_field = validate_index_contract(description)
        global _cached_text_field
        _cached_text_field = resolved_field
        _cached_index = client.Index(name)
        return _cached_index


def _as_dict(value: Any) -> dict[str, Any]:
    if value is None:
        return {}
    if isinstance(value, dict):
        return value
    if hasattr(value, "model_dump"):
        return value.model_dump(mode="json")
    return {}


def _named_value(value: Any) -> str | None:
    if isinstance(value, str):
        text = value.strip()
        return text or None
    data = _as_dict(value)
    for key in ("name", "full_name"):
        item = data.get(key)
        if isinstance(item, str) and item.strip():
            return item.strip()
    return None


def _first_dict(value: Any) -> dict[str, Any]:
    if isinstance(value, list) and value:
        return _as_dict(value[0])
    return _as_dict(value)


def _party_name(party: Any) -> str | None:
    data = _as_dict(party)
    return _named_value(data.get("name") or data.get("full_name") or data)


def build_filing_chunk_text(
    filing: dict[str, Any] | Any,
    *,
    filename: str | None = None,
    filing_type: str | None = None,
) -> str:
    """Turn a Core Filing Record into searchable plain text for embedding."""
    data = _as_dict(filing)
    cause = _as_dict(data.get("cause_title"))
    matter = _as_dict(data.get("classification") or data.get("matter_classification"))
    impugned = _first_dict(data.get("impugned_orders") or data.get("impugned_order"))
    aor = _first_dict(data.get("advocates_on_record") or data.get("advocate_on_record"))

    petitioners = data.get("petitioners") or []
    respondents = data.get("respondents") or []

    petitioner_names = [name for name in (_party_name(p) for p in petitioners) if name]
    respondent_names = [name for name in (_party_name(r) for r in respondents) if name]

    special = None
    specials = matter.get("special_categories")
    if isinstance(specials, list) and specials:
        special = ", ".join(str(item) for item in specials if item)
    if not special:
        special = data.get("special_category") or matter.get("special_category")

    petition_type = filing_type or _named_value(data.get("petition_type"))
    court = _named_value(data.get("court"))
    parts = [
        f"Filename: {filename}" if filename else None,
        f"Petition type: {petition_type}" if petition_type else None,
        f"Court: {court}" if court else None,
        f"Special category: {special}" if special else None,
        f"Cause title: {cause.get('title') or cause.get('formatted_title') or cause.get('raw_text')}",
        f"Petitioners: {', '.join(petitioner_names)}" if petitioner_names else None,
        f"Respondents: {', '.join(respondent_names)}" if respondent_names else None,
        f"Matter: {matter.get('main_category_name') or matter.get('main_category')} / "
        f"{matter.get('sub_category_name') or matter.get('sub_category')}",
        f"PIL: {matter.get('is_pil')}",
        f"Impugned order: {impugned.get('case_number')} "
        f"({impugned.get('Forum') or impugned.get('forum') or impugned.get('court_name') or impugned.get('earlier_court')}) "
        f"dated {impugned.get('order_date') or impugned.get('date_of_impugned_order')}",
        f"AOR: {aor.get('name')} ({aor.get('registration_number')})",
        f"Summary title: {cause.get('formatted_title') or cause.get('title')}",
    ]
    return "\n".join(
        p for p in parts if p and not p.endswith(": None") and not p.endswith("/ None")
    )


def split_text_windows(
    text: str,
    *,
    max_chars: int = MAX_CHUNK_CHARS,
    overlap: int = CHUNK_OVERLAP,
) -> list[str]:
    """Split on the strongest nearby text boundary, with bounded overlap."""
    if max_chars <= 0:
        raise ValueError("max_chars must be positive")
    if overlap < 0 or overlap >= max_chars:
        raise ValueError("overlap must be non-negative and smaller than max_chars")
    cleaned = re.sub(r"\n{3,}", "\n\n", (text or "").strip())
    if not cleaned:
        return []
    if len(cleaned) <= max_chars:
        return [cleaned]

    windows: list[str] = []
    start = 0
    while start < len(cleaned):
        hard_end = min(start + max_chars, len(cleaned))
        end = hard_end
        if hard_end < len(cleaned):
            minimum = start + max((max_chars * 3) // 4, 1)
            for separator in ("\n\n", "\n", ". ", " "):
                boundary = cleaned.rfind(separator, minimum, hard_end)
                if boundary >= minimum:
                    end = boundary + len(separator)
                    break
        windows.append(cleaned[start:end].strip())
        if end >= len(cleaned):
            break
        start = max(0, end - overlap)
    return [w for w in windows if w]


def _same_split_part(left: Any, right: Any) -> bool:
    """True when Split labelled both pages with at least one shared filing part."""
    left_names = set(parts_on_page(left))
    right_names = set(parts_on_page(right))
    return bool(left_names) and bool(left_names & right_names)


def _page_margin(text: str, *, from_end: bool, size: int = CHUNK_OVERLAP) -> str:
    cleaned = (text or "").strip()
    if not cleaned:
        return ""
    if from_end:
        return cleaned[-size:]
    return cleaned[:size]


def _labelled_page_segment(page_num: int, snippet: str, *, role: str) -> str:
    snippet = snippet.strip()
    if not snippet:
        return ""
    return f"--- page {page_num} ({role}) ---\n{snippet}"


def _normalized_fragment(text: str) -> str:
    """Collapse formatting before deciding whether text carries semantics."""
    return " ".join((text or "").split()).strip()


def _folio_marker(text: str) -> str | None:
    """Return a standalone printed folio, never a sentence containing one."""
    normalized = _normalized_fragment(text).strip("[](){}.: ")
    return normalized if _FOLIO_ONLY_RE.fullmatch(normalized) else None


def _is_repeated_court_heading(text: str) -> bool:
    normalized = re.sub(r"[^A-Z ]+", " ", _normalized_fragment(text).upper())
    return " ".join(normalized.split()) in _NON_SEMANTIC_HEADINGS


def is_nonsemantic_vector_text(text: str) -> bool:
    """True only for standalone noise that must never consume an embedding."""
    return _folio_marker(text) is not None or _is_repeated_court_heading(text)


def _record_primary_page(record: dict[str, Any]) -> int | None:
    value = (record.get("metadata") or {}).get("primary_page")
    try:
        return int(value) if value is not None else None
    except (TypeError, ValueError):
        return None


def _record_parts(record: dict[str, Any]) -> set[str]:
    return {
        name.casefold()
        for name in parts_on_page((record.get("metadata") or {}).get("document_part"))
    }


def _same_record_part(left: dict[str, Any], right: dict[str, Any]) -> bool:
    left_parts = _record_parts(left)
    right_parts = _record_parts(right)
    return bool(left_parts) and bool(left_parts & right_parts)


def _record_source_pages(record: dict[str, Any]) -> list[int]:
    metadata = record.get("metadata") or {}
    values = metadata.get("source_pages") or []
    if isinstance(values, str):
        values = [values]
    pages: set[int] = set()
    for value in values:
        try:
            pages.add(int(value))
        except (TypeError, ValueError):
            continue
    primary = _record_primary_page(record)
    if primary is not None:
        pages.add(primary)
    return sorted(pages)


def _can_merge_short_record(target: dict[str, Any], fragment: dict[str, Any]) -> bool:
    target_meta = target.get("metadata") or {}
    fragment_meta = fragment.get("metadata") or {}
    target_tokens = int(target_meta.get("token_count") or 0)
    fragment_tokens = int(fragment_meta.get("token_count") or 0)
    if target_tokens and fragment_tokens:
        return target_tokens + fragment_tokens + 12 <= 900
    return (
        len(str(target.get("chunk_text") or ""))
        + len(str(fragment.get("chunk_text") or ""))
        + 80
        <= MAX_CHUNK_CHARS
    )


def _merge_short_record(
    target: dict[str, Any], fragment: dict[str, Any]
) -> dict[str, Any]:
    """Merge a useful short page into adjacent same-part evidence."""
    merged = {
        **target,
        "metadata": dict(target.get("metadata") or {}),
    }
    target_meta = merged["metadata"]
    fragment_meta = fragment.get("metadata") or {}
    target_page = _record_primary_page(target)
    fragment_page = _record_primary_page(fragment)
    target_text = str(target.get("chunk_text") or "").strip()
    fragment_text = str(fragment.get("chunk_text") or "").strip()
    if target_page is None or fragment_page is None:
        return merged

    labelled_target = (
        target_text
        if _PAGE_SEGMENT_RE.search(target_text)
        else _labelled_page_segment(target_page, target_text, role="primary")
    )
    labelled_fragment = _labelled_page_segment(
        fragment_page, fragment_text, role="overlap"
    )
    pieces = (
        [labelled_fragment, labelled_target]
        if fragment_page < target_page
        else [labelled_target, labelled_fragment]
    )
    combined = "\n\n".join(piece for piece in pieces if piece)
    pages = sorted(
        set(_record_source_pages(target)) | set(_record_source_pages(fragment))
    )
    merged["chunk_text"] = combined
    target_meta["source_pages"] = [str(page) for page in pages]
    target_meta["page_start"] = min(pages)
    target_meta["page_end"] = max(pages)
    target_meta["pages"] = (
        str(pages[0]) if len(pages) == 1 else f"{min(pages)}-{max(pages)}"
    )
    target_meta["token_count"] = (
        int(target_meta.get("token_count") or 0)
        + int(fragment_meta.get("token_count") or 0)
        + 12
    )
    target_meta["item_types"] = sorted(
        set(target_meta.get("item_types") or [])
        | set(fragment_meta.get("item_types") or [])
    )
    target_meta["parse_concerns"] = sorted(
        set(target_meta.get("parse_concerns") or [])
        | set(fragment_meta.get("parse_concerns") or [])
    )
    target_meta["merged_short_pages"] = sorted(
        set(target_meta.get("merged_short_pages") or []) | {str(fragment_page)}
    )
    target_meta["content_hash"] = hashlib.sha256(combined.encode("utf-8")).hexdigest()
    return merged


def _clean_page_records(records: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Remove vector noise and merge useful short pages without crossing parts."""
    page_records = [
        record
        for record in records
        if (record.get("metadata") or {}).get("chunk_kind") == "page"
    ]
    substantive = [
        record
        for record in page_records
        if len(_normalized_fragment(str(record.get("chunk_text") or "")))
        >= LOW_INFORMATION_CHARS
        and not is_nonsemantic_vector_text(str(record.get("chunk_text") or ""))
    ]
    replacements: dict[str, dict[str, Any]] = {}
    suppressed_ids: set[str] = set()

    for record in page_records:
        record_id = str(record.get("record_id") or "")
        text = str(record.get("chunk_text") or "").strip()
        normalized = _normalized_fragment(text)
        if not record_id or not normalized:
            continue
        page = _record_primary_page(record)
        if is_nonsemantic_vector_text(text):
            suppressed_ids.add(record_id)
            marker = _folio_marker(text)
            if marker and page is not None:
                candidates = [
                    candidate
                    for candidate in substantive
                    if _same_record_part(record, candidate)
                    and _record_primary_page(candidate) is not None
                ]
                if candidates:
                    target = min(
                        candidates,
                        key=lambda candidate: abs(
                            (_record_primary_page(candidate) or page) - page
                        ),
                    )
                    target_id = str(target.get("record_id") or "")
                    updated = replacements.get(target_id, target)
                    updated = {
                        **updated,
                        "metadata": dict(updated.get("metadata") or {}),
                    }
                    metadata = updated["metadata"]
                    metadata["folio_markers"] = sorted(
                        set(metadata.get("folio_markers") or []) | {f"{page}:{marker}"}
                    )
                    replacements[target_id] = updated
            continue
        if len(normalized) >= LOW_INFORMATION_CHARS:
            continue

        candidates: list[dict[str, Any]] = []
        for candidate in substantive:
            candidate_id = str(candidate.get("record_id") or "")
            current_candidate = replacements.get(candidate_id, candidate)
            candidate_page = _record_primary_page(current_candidate)
            if (
                _same_record_part(record, current_candidate)
                and candidate_page is not None
                and page is not None
                and abs(candidate_page - page) <= 1
                and _can_merge_short_record(current_candidate, record)
            ):
                candidates.append(current_candidate)
        if not candidates:
            # A short legal phrase can be material. Retain it when no safe,
            # adjacent same-part destination exists rather than losing evidence.
            continue
        target = min(
            candidates,
            key=lambda candidate: (
                abs((_record_primary_page(candidate) or page) - page),
                0 if (_record_primary_page(candidate) or page) < page else 1,
            ),
        )
        target_id = str(target.get("record_id") or "")
        replacements[target_id] = _merge_short_record(
            replacements.get(target_id, target), record
        )
        suppressed_ids.add(record_id)

    cleaned = [
        replacements.get(str(record.get("record_id") or ""), record)
        for record in records
        if str(record.get("record_id") or "") not in suppressed_ids
    ]
    if suppressed_ids:
        logger.info(
            "[Pinecone] Suppressed or merged %s low-information page chunk(s)",
            len(suppressed_ids),
        )
    return cleaned


def _neighbor_margin(
    *,
    page_num: int,
    neighbor: int,
    page_markdown: dict[int, str],
    parts: dict[int, Any],
    from_end: bool,
) -> str:
    """200 chars from an adjacent page, only when Split says it is the same part."""
    if neighbor not in page_markdown:
        return ""
    neighbor_text = (page_markdown.get(neighbor) or "").strip()
    if not neighbor_text:
        return ""
    if is_nonsemantic_vector_text(neighbor_text):
        return ""
    if not _same_split_part(parts.get(page_num), parts.get(neighbor)):
        return ""
    return _labelled_page_segment(
        neighbor,
        _page_margin(page_markdown.get(neighbor) or "", from_end=from_end),
        role="overlap",
    )


def _structured_page_records(
    *,
    base_id: str,
    page_num: int,
    page: dict[str, Any],
    base_meta: dict[str, Any],
    page_meta: dict[str, Any] | None,
    document_part: Any,
) -> list[dict[str, Any]]:
    """Build compatible evidence records from compact LlamaParse items."""

    chunks = build_semantic_chunks(page)
    if not chunks:
        return []
    parser_meta: dict[str, Any] = {}
    raw_parser_meta = page.get("metadata")
    if isinstance(raw_parser_meta, dict):
        parser_meta = raw_parser_meta
    records: list[dict[str, Any]] = []
    for chunk_index, chunk in enumerate(chunks):
        text = str(chunk.get("text") or "").strip()
        if not text:
            continue
        content_hash = hashlib.sha256(text.encode("utf-8")).hexdigest()
        metadata = {
            **base_meta,
            **_flatten_metadata(page_meta),
            "chunk_kind": "page",
            "content_kind": str(chunk.get("content_kind") or "structured_text"),
            "parser_source": (
                "markdown_fallback"
                if "markdown_fallback" in (chunk.get("item_types") or [])
                else "items"
            ),
            "page_start": page_num,
            "page_end": page_num,
            "primary_page": page_num,
            "source_pages": [str(page_num)],
            "pages": str(page_num),
            "chunk_index": chunk_index,
            "item_start": int(chunk.get("item_start") or 0),
            "item_end": int(chunk.get("item_end") or 0),
            "item_types": [str(value) for value in chunk.get("item_types") or []],
            "heading_path": [str(value) for value in chunk.get("heading_path") or []],
            "token_count": int(chunk.get("token_count") or 0),
            "content_hash": content_hash,
            "ocr_status": "text",
            "requires_visual_review": False,
            "parse_success": bool(page.get("success", True)),
            "vector_schema_version": VECTOR_SCHEMA_VERSION,
        }
        for source, target in (
            ("confidence", "parse_confidence"),
            ("printed_page_number", "printed_page_number"),
            ("original_orientation_angle", "original_orientation_angle"),
        ):
            if parser_meta.get(source) is not None:
                metadata[target] = parser_meta[source]
        for key in ("table_row_start", "table_row_end"):
            if chunk.get(key) is not None:
                metadata[key] = int(chunk[key])
        concerns = [str(value) for value in chunk.get("parse_concerns") or []]
        if concerns:
            metadata["parse_concerns"] = concerns
            metadata["requires_visual_review"] = True
        if document_part:
            metadata["document_part"] = document_part
        records.append(
            {
                "record_id": f"{base_id}:page:{page_num}:{chunk_index}",
                "chunk_text": text,
                "metadata": metadata,
            }
        )
    return records


def build_page_records(
    *,
    base_id: str,
    page_markdown: dict[int, str],
    metadata: dict[str, Any] | None = None,
    page_parts: dict[int, Any] | None = None,
    page_metadata: dict[int, dict[str, Any]] | None = None,
    page_structures: dict[int, dict[str, Any]] | None = None,
) -> list[dict[str, Any]]:
    """
    Build Pinecone records from structured parse pages, with Markdown fallback.

    Expanded LlamaParse items are grouped into token-aware semantic chunks.
    Legacy/reused artifacts without structured items retain the prior character
    window behavior, including same-document neighbour context.
    `page_parts` is the Split map (page number → Listing Proforma, …).
    The first/last window of a page also takes CHUNK_OVERLAP characters from
    the previous/next page when Split labelled both as the same part. Cover
    Page is never glued onto Main Petition. Every borrowed and primary segment
    in an overlap record is labelled so a returned quote resolves to its true page.
    """
    base_meta = metadata or {}
    parts = page_parts or {}
    per_page = page_metadata or {}
    structured = coerce_structured_pages(page_structures)
    records: list[dict[str, Any]] = []
    empty_pages = 0

    # Application headings are commonly present only on the first page. Infer
    # the subtype from the complete classified run and propagate it to its
    # continuation pages without treating the generic word "Application" as a
    # subtype.
    subtype_by_part: dict[str, str] = {}
    text_by_part: dict[str, list[str]] = {}
    for page_num, value in parts.items():
        for part in parts_on_page(value):
            if family_split_name(part) == "Application":
                text_by_part.setdefault(part, []).append(
                    str(page_markdown.get(page_num) or "")
                )
    for part, texts in text_by_part.items():
        subtype = infer_document_subtype_id(part, "\n".join(texts))
        if subtype:
            subtype_by_part[part] = subtype

    for page_num in sorted(page_markdown.keys()):
        page_text = (page_markdown.get(page_num) or "").strip()
        names = parts_on_page(parts.get(page_num))
        document_part: Any = names[0] if len(names) == 1 else names
        if page_num in structured:
            structured_records = _structured_page_records(
                base_id=base_id,
                page_num=page_num,
                page=structured[page_num],
                base_meta=base_meta,
                page_meta=per_page.get(page_num),
                document_part=document_part,
            )
            if structured_records:
                records.extend(structured_records)
                logger.info(
                    "[Pinecone] Page %s → %s structured chunk(s)",
                    page_num,
                    len(structured_records),
                )
                continue
        if not page_text:
            empty_pages += 1
            meta = {
                **base_meta,
                **_flatten_metadata(per_page.get(page_num)),
                "chunk_kind": "page_manifest",
                "page_start": page_num,
                "page_end": page_num,
                "primary_page": page_num,
                "source_pages": [str(page_num)],
                "pages": str(page_num),
                "chunk_index": 0,
                "ocr_status": "empty",
                "requires_visual_review": True,
                "vector_schema_version": VECTOR_SCHEMA_VERSION,
            }
            if document_part:
                meta["document_part"] = document_part
            records.append(
                {
                    "record_id": f"{base_id}:manifest:{page_num}",
                    "chunk_text": EMPTY_PAGE_MARKER,
                    "metadata": meta,
                }
            )
            continue

        # Do not inflate a folio or repeated court caption with neighbour
        # overlap. Keep it as a temporary page record so the shared cleanup
        # pass can suppress it and retain a folio marker on nearby same-part
        # metadata when possible.
        if is_nonsemantic_vector_text(page_text):
            meta = {
                **base_meta,
                **_flatten_metadata(per_page.get(page_num)),
                "chunk_kind": "page",
                "content_kind": "markdown_fallback",
                "parser_source": "markdown_fallback",
                "page_start": page_num,
                "page_end": page_num,
                "primary_page": page_num,
                "source_pages": [str(page_num)],
                "pages": str(page_num),
                "chunk_index": 0,
                "ocr_status": "low_information",
                "requires_visual_review": False,
                "vector_schema_version": VECTOR_SCHEMA_VERSION,
            }
            if document_part:
                meta["document_part"] = document_part
            records.append(
                {
                    "record_id": f"{base_id}:page:{page_num}:0",
                    "chunk_text": page_text,
                    "metadata": meta,
                }
            )
            continue

        windows = split_text_windows(page_text)
        logger.info(
            "[Pinecone] Page %s → %s char(s) → %s window(s)",
            page_num,
            len(page_text),
            len(windows),
        )
        prev_margin = _neighbor_margin(
            page_num=page_num,
            neighbor=page_num - 1,
            page_markdown=page_markdown,
            parts=parts,
            from_end=True,
        )
        next_margin = _neighbor_margin(
            page_num=page_num,
            neighbor=page_num + 1,
            page_markdown=page_markdown,
            parts=parts,
            from_end=False,
        )
        last_idx = len(windows) - 1
        for chunk_idx, window in enumerate(windows):
            pieces = []
            source_pages: list[int] = []
            include_prev = chunk_idx == 0 and bool(prev_margin)
            include_next = chunk_idx == last_idx and bool(next_margin)
            has_cross_page_context = include_prev or include_next
            if include_prev:
                pieces.append(prev_margin)
                source_pages.append(page_num - 1)
            pieces.append(
                _labelled_page_segment(page_num, window, role="primary")
                if has_cross_page_context
                else window
            )
            source_pages.append(page_num)
            if include_next:
                pieces.append(next_margin)
                source_pages.append(page_num + 1)
            record_id = f"{base_id}:page:{page_num}:{chunk_idx}"
            meta = {
                **base_meta,
                **_flatten_metadata(per_page.get(page_num)),
                "chunk_kind": "page",
                "page_start": page_num,
                "page_end": max(source_pages),
                "primary_page": page_num,
                "source_pages": [str(page) for page in source_pages],
                "pages": (
                    str(source_pages[0])
                    if len(source_pages) == 1
                    else f"{min(source_pages)}-{max(source_pages)}"
                ),
                "chunk_index": chunk_idx,
                "ocr_status": "text",
                "requires_visual_review": False,
                "vector_schema_version": VECTOR_SCHEMA_VERSION,
            }
            if document_part:
                meta["document_part"] = document_part
            records.append(
                {
                    "record_id": record_id,
                    "chunk_text": "\n\n".join(pieces),
                    "metadata": meta,
                }
            )

    for record in records:
        record_meta = record.get("metadata") or {}
        names = parts_on_page(record_meta.get("document_part"))
        if not names:
            continue
        part_ids = list(
            dict.fromkeys(canonical_document_part_id(name) for name in names)
        )
        record_meta["document_part_id"] = part_ids[0]
        record_meta["document_part_ids"] = part_ids
        subtype_ids = list(
            dict.fromkeys(
                subtype
                for name in names
                if (
                    subtype := subtype_by_part.get(name)
                    or infer_document_subtype_id(
                        name, str(record.get("chunk_text") or "")
                    )
                )
            )
        )
        if subtype_ids:
            record_meta["document_subtype_id"] = subtype_ids[0]
            record_meta["document_subtype_ids"] = subtype_ids
        record["metadata"] = record_meta

    records = _clean_page_records(records)
    logger.info(
        "[Pinecone] Page records ready: %s chunk(s) from %s page(s) (%s empty)",
        len(records),
        len(page_markdown),
        empty_pages,
    )
    return records


def _flatten_metadata(metadata: dict[str, Any] | None) -> dict[str, Any]:
    flat: dict[str, Any] = {}
    if not metadata:
        return flat
    for key, value in metadata.items():
        if value is None:
            continue
        if isinstance(value, (str, int, float, bool)):
            flat[key] = value
        elif isinstance(value, list) and all(isinstance(v, str) for v in value):
            flat[key] = value
        else:
            flat[key] = str(value)
    return flat


def upsert_records(items: list[dict[str, Any]], *, batch_size: int = 50) -> int:
    """Batch-upsert text records for integrated embedding."""
    if not items:
        logger.info("[Pinecone] No records to upsert (empty items list)")
        return 0

    logger.info(
        "[Pinecone] Preparing upsert: %s item(s) → index=%s namespace=%s "
        "embed_model=%s (server-side integrated embeddings)",
        len(items),
        pinecone_index_name,
        pinecone_namespace,
        pinecone_embed_model,
    )

    client = get_pinecone_client()
    index = ensure_index(client)
    text_field = resolve_text_field(client)
    logger.info(
        "[Pinecone] Connected to index=%s text_field=%s (Pinecone embeds this field)",
        pinecone_index_name,
        text_field,
    )

    prepared: list[dict[str, Any]] = []
    skipped = 0
    total_chars = 0
    for item in items:
        chunk_text = (item.get("chunk_text") or "").strip()
        record_id = item.get("record_id")
        if not chunk_text or not record_id:
            skipped += 1
            continue
        metadata = item.get("metadata") or {}
        if metadata.get("chunk_kind") == "page" and is_nonsemantic_vector_text(
            chunk_text
        ):
            skipped += 1
            logger.warning(
                "[Pinecone] Skipped non-semantic page record %s before embedding",
                record_id,
            )
            continue
        record: dict[str, Any] = {
            "_id": str(record_id),
            text_field: chunk_text,
            **_flatten_metadata(metadata),
        }
        prepared.append(record)
        total_chars += len(chunk_text)

    if skipped:
        logger.warning(
            "[Pinecone] Skipped %s item(s) with empty text or missing record_id",
            skipped,
        )

    logger.info(
        "[Pinecone] Embedding + upserting %s record(s) (~%s chars total) "
        "via %s in batches of %s",
        len(prepared),
        total_chars,
        pinecone_embed_model,
        batch_size,
    )

    for i in range(0, len(prepared), batch_size):
        batch = prepared[i : i + batch_size]
        batch_num = (i // batch_size) + 1
        batch_total = (len(prepared) + batch_size - 1) // batch_size
        logger.info(
            "[Pinecone] Upserting batch %s/%s (%s records)...",
            batch_num,
            batch_total,
            len(batch),
        )
        index.upsert_records(namespace=pinecone_namespace, records=batch)
        logger.info(
            "[Pinecone] Batch %s/%s upserted (embeddings generated server-side)",
            batch_num,
            batch_total,
        )

    logger.info(
        "[Pinecone] Done: upserted %s record(s) into %s/%s (model=%s, text_field=%s)",
        len(prepared),
        pinecone_index_name,
        pinecone_namespace,
        pinecone_embed_model,
        text_field,
    )
    return len(prepared)


def upsert_filing_record(
    *,
    record_id: str,
    chunk_text: str,
    metadata: dict[str, Any] | None = None,
) -> str:
    """
    Upsert one filing into Pinecone.

    Pinecone embeds `chunk_text` server-side using the index's integrated model
    (configured via PINECONE_EMBED_MODEL, default llama-text-embed-v2).
    """
    upsert_records(
        [
            {
                "record_id": record_id,
                "chunk_text": chunk_text,
                "metadata": metadata,
            }
        ]
    )
    return record_id


def delete_workspace_records(workspace_id: str) -> None:
    """Delete every Pinecone record stored for one workspace."""
    workspace_id = (workspace_id or "").strip()
    if not workspace_id:
        raise ValueError("workspace_id is required")
    if not pinecone_enabled():
        logger.info("[Pinecone] Skipped workspace delete; Pinecone is disabled")
        return

    logger.info(
        "[Pinecone] Deleting records for workspace_id=%s index=%s namespace=%s",
        workspace_id,
        pinecone_index_name,
        pinecone_namespace,
    )
    index = ensure_index()
    index.delete(
        filter={"workspace_id": {"$eq": workspace_id}},
        namespace=pinecone_namespace,
    )
    logger.info("[Pinecone] Deleted records for workspace_id=%s", workspace_id)


def delete_vector_generation(
    generation: str,
    *,
    organization_id: str | None = None,
    workspace_id: str | None = None,
) -> None:
    """Delete one inactive generation without touching another filing."""
    generation = (generation or "").strip()
    if not generation:
        return
    metadata_filter: dict[str, Any] = {"index_generation": {"$eq": generation}}
    if organization_id:
        metadata_filter["organization_id"] = {"$eq": organization_id}
    if workspace_id:
        metadata_filter["workspace_id"] = {"$eq": workspace_id}
    ensure_index().delete(filter=metadata_filter, namespace=pinecone_namespace)


def _parse_page_segments(text: str) -> dict[int, str]:
    """Recover per-page provenance from an embedded overlap record."""
    matches = list(_PAGE_SEGMENT_RE.finditer(text or ""))
    segments: dict[int, str] = {}
    for index, match in enumerate(matches):
        end = matches[index + 1].start() if index + 1 < len(matches) else len(text)
        body = text[match.end() : end].strip()
        if body:
            segments[int(match.group("page"))] = body
    return segments


def _raw_hits(result: Any) -> list[Any]:
    hits = (
        result.get("result", result)
        if isinstance(result, dict)
        else getattr(result, "result", result)
    )
    matches = getattr(hits, "hits", None)
    if matches is None and isinstance(hits, dict):
        matches = hits.get("hits", [])
    return list(matches or [])


def search_filings(query: str, *, top_k: int = 5) -> list[dict[str, Any]]:
    """Semantic search over indexed filings (uses the same integrated embed model)."""
    index = ensure_index()
    with _pinecone_query_gate():
        result = index.search(
            namespace=pinecone_namespace,
            query={"top_k": top_k, "inputs": {"text": query}},
        )
    out: list[dict[str, Any]] = []
    for hit in _raw_hits(result):
        if isinstance(hit, dict):
            out.append(hit)
        elif hasattr(hit, "to_dict"):
            out.append(hit.to_dict())
        else:
            out.append(
                {
                    "id": getattr(hit, "_id", None),
                    "score": getattr(hit, "_score", None),
                }
            )
    return out


def _hit_fields(hit: Any) -> tuple[str | None, float | None, dict[str, Any]]:
    """Normalize a Pinecone hit into (id, score, fields) across SDK shapes."""
    if hasattr(hit, "to_dict"):
        hit = hit.to_dict()
    if isinstance(hit, dict):
        record_id = hit.get("_id") or hit.get("id")
        score = hit.get("_score") or hit.get("score")
        fields = hit.get("fields") or hit.get("metadata") or {}
        return record_id, score, dict(fields) if fields else {}
    fields = getattr(hit, "fields", None) or getattr(hit, "metadata", None) or {}
    return (
        getattr(hit, "_id", None) or getattr(hit, "id", None),
        getattr(hit, "_score", None) or getattr(hit, "score", None),
        dict(fields) if fields else {},
    )


def _to_chunk(hit: Any, text_field: str) -> dict[str, Any] | None:
    record_id, score, fields = _hit_fields(hit)
    text = (fields.get(text_field) or "").strip()
    if not text:
        return None

    page = fields.get("primary_page", fields.get("page_start"))
    page_end = fields.get("page_end")
    try:
        page = int(page) if page is not None else None
    except (TypeError, ValueError):
        page = None
    try:
        page_end = int(page_end) if page_end is not None else page
    except (TypeError, ValueError):
        page_end = page

    return {
        "record_id": record_id,
        "score": score,
        "text": text,
        "page": page,
        "page_end": page_end,
        "chunk_kind": fields.get("chunk_kind"),
        "document_part": fields.get("document_part"),
        "document_part_id": fields.get("document_part_id"),
        "document_part_ids": fields.get("document_part_ids") or [],
        "document_subtype_id": fields.get("document_subtype_id"),
        "document_subtype_ids": fields.get("document_subtype_ids") or [],
        "source_pages": fields.get("source_pages") or [],
        "page_texts": _parse_page_segments(text),
        "ocr_status": fields.get("ocr_status"),
        "requires_visual_review": bool(fields.get("requires_visual_review")),
        "index_generation": fields.get("index_generation"),
        "content_kind": fields.get("content_kind"),
        "parser_source": fields.get("parser_source"),
        "heading_path": fields.get("heading_path") or [],
        "item_types": fields.get("item_types") or [],
        "item_start": fields.get("item_start"),
        "item_end": fields.get("item_end"),
        "table_row_start": fields.get("table_row_start"),
        "table_row_end": fields.get("table_row_end"),
        "parse_confidence": fields.get("parse_confidence"),
        "parse_concerns": fields.get("parse_concerns") or [],
        "printed_page_number": fields.get("printed_page_number"),
        "original_page": fields.get("original_page"),
        "content_hash": fields.get("content_hash"),
        "vector_schema_version": fields.get("vector_schema_version"),
    }


def search_filing_chunks(
    query: str,
    *,
    file_hash: str,
    top_k: int | None = None,
    chunk_kind: str | None = None,
    document_part: str | None = None,
    document_subtype_id: str | None = None,
    organization_id: str | None = None,
    workspace_id: str | None = None,
    index_generation: str | None = None,
    vector_schema_version: int | str | None = VECTOR_SCHEMA_VERSION,
) -> list[dict[str, Any]]:
    """Return Pinecone hits for one filing. No LLM packing or score cutoff.

    Unlike `search_filings`, this filters on `file_hash` metadata so scrutiny of
    one document never pulls evidence out of a different filing. `top_k` is
    how many neighbours Pinecone fills; far hits are still returned. Dropping
    them is `select_chunks_for_defect`, not this function.
    """
    if top_k is None:
        top_k = scrutiny_top_k()
    if not file_hash:
        raise ValueError("file_hash is required to scope a filing search")

    client = get_pinecone_client()
    index = ensure_index(client)
    text_field = resolve_text_field(client)

    metadata_filter: dict[str, Any] = {"file_hash": {"$eq": file_hash}}
    if organization_id:
        metadata_filter["organization_id"] = {"$eq": organization_id}
    if workspace_id:
        metadata_filter["workspace_id"] = {"$eq": workspace_id}
    if index_generation:
        metadata_filter["index_generation"] = {"$eq": index_generation}
    if vector_schema_version is not None:
        metadata_filter["vector_schema_version"] = {"$eq": int(vector_schema_version)}
    if chunk_kind:
        metadata_filter["chunk_kind"] = {"$eq": chunk_kind}
    if document_part:
        metadata_filter["document_part_id"] = {
            "$eq": canonical_document_part_id(document_part)
        }
    if document_subtype_id:
        metadata_filter["document_subtype_id"] = {"$eq": document_subtype_id}

    fields = [
        text_field,
        "chunk_kind",
        "page_start",
        "page_end",
        "document_part",
        "document_part_id",
        "document_part_ids",
        "document_subtype_id",
        "document_subtype_ids",
        "source_pages",
        "primary_page",
        "ocr_status",
        "requires_visual_review",
        "index_generation",
        "content_kind",
        "parser_source",
        "heading_path",
        "item_types",
        "item_start",
        "item_end",
        "table_row_start",
        "table_row_end",
        "parse_confidence",
        "parse_concerns",
        "printed_page_number",
        "original_page",
        "content_hash",
        "vector_schema_version",
    ]

    def _search(active_filter: dict[str, Any]) -> Any:
        with _pinecone_query_gate():
            return index.search(
                namespace=pinecone_namespace,
                query={
                    "top_k": top_k,
                    "inputs": {"text": query},
                    "filter": active_filter,
                },
                fields=fields,
            )

    result = _search(metadata_filter)
    chunks = [c for c in (_to_chunk(h, text_field) for h in _raw_hits(result)) if c]

    # Schema-4 records created before canonical part metadata was introduced
    # remain readable during a controlled reindex. They are never mixed with
    # schema-3 records, and post-filtering is used only when the canonical
    # filtered search returned no candidates.
    if document_part and not chunks:
        legacy_filter = dict(metadata_filter)
        legacy_filter.pop("document_part_id", None)
        legacy_filter.pop("document_subtype_id", None)
        result = _search(legacy_filter)
        chunks = [c for c in (_to_chunk(h, text_field) for h in _raw_hits(result)) if c]

    if document_part:
        wanted = document_part.strip()
        chunks = [c for c in chunks if _part_match(c, [wanted])]
    if document_subtype_id:
        subtype_matches: list[dict[str, Any]] = []
        for chunk in chunks:
            stored = chunk.get("document_subtype_id")
            stored_ids = chunk.get("document_subtype_ids") or []
            inferred = None
            if not stored and not stored_ids:
                names = parts_on_page(chunk.get("document_part"))
                inferred = next(
                    (
                        value
                        for part in names
                        if (
                            value := infer_document_subtype_id(
                                part, str(chunk.get("text") or "")
                            )
                        )
                    ),
                    None,
                )
            if stored == document_subtype_id or document_subtype_id in stored_ids:
                subtype_matches.append(chunk)
            elif inferred == document_subtype_id:
                annotated = dict(chunk)
                annotated["document_subtype_id"] = inferred
                annotated["document_subtype_inferred"] = True
                subtype_matches.append(annotated)
        chunks = subtype_matches
    logger.info(
        "[Pinecone] Filing search returned %s chunk(s) for file_hash=%s (top_k=%s)",
        len(chunks),
        file_hash,
        top_k,
    )
    return chunks


def gather_filing_evidence(
    queries: list[str],
    *,
    file_hash: str,
    top_k: int | None = None,
    max_chunks: int | None = None,
    document_parts: list[str] | None = None,
    organization_id: str | None = None,
    workspace_id: str | None = None,
    index_generation: str | None = None,
    query_plan: list[EvidenceQuery] | tuple[EvidenceQuery, ...] | None = None,
    vector_schema_version: int | str | None = VECTOR_SCHEMA_VERSION,
) -> list[dict[str, Any]]:
    """Union the results of several queries into one deduped evidence set.

    Absence-type criteria ("the declaration is missing") are sensitive to
    retrieval recall, so each where-to-look / trigger-word query contributes
    rather than relying on a single top-k over the whole defect.

    `document_parts` runs an extra search per Split label filtered to that
    metadata value so Cover Page hits cannot stand in for the Checklist.
    """
    if top_k is None:
        top_k = scrutiny_top_k()
    if max_chunks is None:
        max_chunks = scrutiny_max_chunks()

    seen: dict[str, dict[str, Any]] = {}

    def _absorb(chunk: dict[str, Any]) -> None:
        record_id = chunk.get("record_id")
        if not record_id:
            return
        existing = seen.get(record_id)
        if existing is None or (chunk.get("score") or 0) > (existing.get("score") or 0):
            seen[record_id] = chunk

    jobs: list[tuple[str, dict[str, Any]]] = []
    scope = {
        "file_hash": file_hash,
        "top_k": top_k,
        "organization_id": organization_id,
        "workspace_id": workspace_id,
        "index_generation": index_generation,
        "vector_schema_version": vector_schema_version,
    }
    if query_plan:
        for planned in query_plan:
            text = planned.text.strip()
            if not text:
                continue
            kwargs = dict(scope)
            if planned.document_part:
                kwargs.update(
                    chunk_kind="page",
                    document_part=planned.document_part,
                    document_subtype_id=planned.document_subtype_id,
                )
            jobs.append((text, kwargs))
    else:
        for query in queries:
            if query and query.strip():
                jobs.append((query.strip(), dict(scope)))
        for part in document_parts or []:
            label = (part or "").strip()
            if label:
                jobs.append(
                    (
                        label,
                        {
                            **scope,
                            "chunk_kind": "page",
                            "document_part": label,
                        },
                    )
                )

    def _run_one(query: str, kwargs: dict[str, Any]) -> list[dict[str, Any]]:
        try:
            return search_filing_chunks(query, **kwargs)
        except Exception as e:
            kind = "Part-filtered query" if kwargs.get("document_part") else "Query"
            logger.warning("[Pinecone] %s failed (%s): %s", kind, query[:60], e)
            return []

    if len(jobs) <= 1:
        hits = [_run_one(query, kwargs) for query, kwargs in jobs]
    else:
        # Outer scrutiny already fans out defects; keep inner fan-out small so
        # urllib3 does not blow past the Pinecone connection pool.
        workers = min(4, len(jobs), pinecone_query_concurrency())
        with ThreadPoolExecutor(max_workers=workers) as pool:
            hits = list(pool.map(lambda job: _run_one(job[0], job[1]), jobs))
    for chunks in hits:
        for chunk in chunks:
            _absorb(chunk)

    chunks = sorted(
        seen.values(),
        key=lambda c: (c.get("chunk_kind") != "summary", -(c.get("score") or 0)),
    )
    if len(chunks) > max_chunks:
        logger.info(
            "[Pinecone] Trimming evidence from %s to %s chunk(s)",
            len(chunks),
            max_chunks,
        )
        reserved: list[dict[str, Any]] = []
        used: set[str] = set()
        for part in document_parts or []:
            want = (part or "").strip()
            if not want:
                continue
            for chunk in chunks:
                record_id = str(chunk.get("record_id") or "")
                if not record_id or record_id in used:
                    continue
                if not _part_match(chunk, [want]):
                    continue
                reserved.append(chunk)
                used.add(record_id)
                break
        leftover = max(0, max_chunks - len(reserved))
        rest = [c for c in chunks if str(c.get("record_id") or "") not in used]
        chunks = reserved + rest[:leftover]

    # Present page chunks in reading order; keep the summary chunk first.
    summary = [c for c in chunks if c.get("chunk_kind") == "summary"]
    pages = sorted(
        (c for c in chunks if c.get("chunk_kind") != "summary"),
        key=lambda c: (c.get("page") is None, c.get("page") or 0),
    )
    return summary + pages


def gather_filing_evidence_pool(
    *,
    file_hash: str,
    top_k: int | None = None,
    max_chunks: int | None = None,
    organization_id: str | None = None,
    workspace_id: str | None = None,
    index_generation: str | None = None,
) -> list[dict[str, Any]]:
    """Caption-wide gather for a filing (tests and fallbacks).

    Live scrutiny queries per defect via `gather_filing_evidence` instead of
    this shared dump, so 74 checks do not share one generic excerpt set.
    """
    cap = max_chunks if max_chunks is not None else scrutiny_pool_max_chunks()
    queries = pool_search_queries()
    chunks = gather_filing_evidence(
        queries,
        file_hash=file_hash,
        top_k=top_k,
        max_chunks=cap,
        organization_id=organization_id,
        workspace_id=workspace_id,
        index_generation=index_generation,
    )
    logger.info(
        "[Pinecone] Shared evidence pool: %s chunk(s) from %s caption queries "
        "for file_hash=%s",
        len(chunks),
        len(queries),
        file_hash,
    )
    return chunks


THIN_PAGE_CHARS = 500
MAX_PAGE_WINDOWS = 4
_FETCH_BATCH = 100


def page_record_id(base_id: str, page: int, window: int) -> str:
    """Pinecone id written by ``build_page_records``."""
    return f"{base_id}:page:{int(page)}:{int(window)}"


def _vectors_from_fetch(result: Any) -> dict[str, Any]:
    if isinstance(result, dict):
        vectors = result.get("vectors") or result.get("records") or {}
        return dict(vectors) if isinstance(vectors, dict) else {}
    vectors = getattr(result, "vectors", None) or getattr(result, "records", None) or {}
    return dict(vectors) if isinstance(vectors, dict) else {}


def _record_fields(record: Any) -> dict[str, Any]:
    fields: dict[str, Any] = {}
    if isinstance(record, dict):
        fields.update(record)
        meta = record.get("metadata") or record.get("fields")
        if isinstance(meta, dict):
            fields.update(meta)
        return fields
    meta = getattr(record, "metadata", None)
    if isinstance(meta, dict):
        fields.update(meta)
    if hasattr(record, "to_dict"):
        dumped = record.to_dict()
        if isinstance(dumped, dict):
            fields.update(dumped)
            inner = dumped.get("metadata")
            if isinstance(inner, dict):
                fields.update(inner)
    return fields


def _text_from_fields(fields: dict[str, Any], text_field: str) -> str:
    for key in (text_field, "chunk_text", "normalized_text", "text"):
        value = fields.get(key)
        if isinstance(value, str) and value.strip():
            return value.strip()
    return ""


def _fetch_id_batch(ids: list[str]) -> dict[str, Any]:
    if not ids:
        return {}
    client = get_pinecone_client()
    index = ensure_index(client)
    found: dict[str, Any] = {}
    with _pinecone_query_gate():
        for start in range(0, len(ids), _FETCH_BATCH):
            batch = ids[start : start + _FETCH_BATCH]
            result = index.fetch(ids=batch, namespace=pinecone_namespace)
            found.update(_vectors_from_fetch(result))
    return found


def fetch_page_texts(
    *,
    base_id: str,
    pages: list[int],
    max_windows: int = MAX_PAGE_WINDOWS,
    thin_chars: int = THIN_PAGE_CHARS,
) -> dict[int, str]:
    """Read page text by id ``{base_id}:page:{page}:{window}``.

    Window 0 is always requested. Later windows are read only while the page
    text is still shorter than ``thin_chars``.
    """
    wanted = sorted({int(page) for page in pages if int(page) > 0})
    if not base_id or not wanted:
        return {}
    text_field = pinecone_text_field
    try:
        text_field = resolve_text_field()
    except ValueError as exc:
        logger.debug("Using configured Pinecone text field: %s", exc)

    windows = {page: 0 for page in wanted}
    parts: dict[int, list[str]] = {page: [] for page in wanted}
    open_pages = set(wanted)
    while open_pages:
        ids = [
            page_record_id(base_id, page, windows[page]) for page in sorted(open_pages)
        ]
        try:
            records = _fetch_id_batch(ids)
        except Exception:
            logger.exception(
                "[Pinecone] Page fetch failed for %s page(s) of %s",
                len(ids),
                base_id,
            )
            raise
        next_open: set[int] = set()
        for page in list(open_pages):
            record_id = page_record_id(base_id, page, windows[page])
            record = records.get(record_id)
            text = (
                _text_from_fields(_record_fields(record), text_field) if record else ""
            )
            if text:
                parts[page].append(text)
            joined = "\n\n".join(parts[page]).strip()
            if text and len(joined) < thin_chars and windows[page] + 1 < max_windows:
                windows[page] += 1
                next_open.add(page)
        open_pages = next_open

    loaded = {
        page: "\n\n".join(chunks).strip()
        for page, chunks in parts.items()
        if any(chunk.strip() for chunk in chunks)
    }
    logger.info(
        "[Pinecone] Fetched %s/%s page(s) for %s",
        len(loaded),
        len(wanted),
        base_id,
    )
    return loaded
