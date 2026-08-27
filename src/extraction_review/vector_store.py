"""Pinecone vector store with integrated embeddings (llama-text-embed-v2).

This project does not run a separate embedding model. When VECTOR_BACKEND=pinecone,
text is upserted via Pinecone's integrated embedding index; Pinecone embeds server-side.
"""

from __future__ import annotations

import logging
import os
import re
from typing import Any

from pinecone import Pinecone

logger = logging.getLogger(__name__)

# Record field that Pinecone embeds (must match index field_map.text).
# Existing `legal-document-knowledge` index uses `normalized_text`.
DEFAULT_TEXT_FIELD = "normalized_text"
MAX_CHUNK_CHARS = 3500
CHUNK_OVERLAP = 200

pinecone_api_key = os.getenv("PINECONE_API_KEY")
pinecone_index_name = os.getenv("PINECONE_INDEX", "legal-document-knowledge")
pinecone_cloud = os.getenv("PINECONE_CLOUD", "aws")
pinecone_region = os.getenv("PINECONE_REGION", "us-east-1")
pinecone_embed_model = os.getenv("PINECONE_EMBED_MODEL", "llama-text-embed-v2")
pinecone_namespace = os.getenv("PINECONE_NAMESPACE", "jubeex-filings")
pinecone_text_field = os.getenv("PINECONE_TEXT_FIELD", DEFAULT_TEXT_FIELD)
vector_backend = (os.getenv("VECTOR_BACKEND") or "pinecone").strip().lower()


def pinecone_enabled() -> bool:
    return vector_backend == "pinecone" and bool(pinecone_api_key)


def get_pinecone_client() -> Pinecone:
    if not pinecone_api_key:
        raise ValueError("PINECONE_API_KEY is not set")
    return Pinecone(api_key=pinecone_api_key)


def resolve_text_field(pc: Pinecone | None = None) -> str:
    """Read the index field_map when available; fall back to PINECONE_TEXT_FIELD."""
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
                return str(field_map["text"])
    except Exception as e:
        logger.debug("Could not resolve Pinecone text field from index: %s", e)
    return pinecone_text_field


def ensure_index(pc: Pinecone | None = None) -> Any:
    """Create an integrated-embedding index if missing; return a data-plane Index."""
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
            embed={
                "model": pinecone_embed_model,
                "field_map": {"text": pinecone_text_field},
            },
        )
        logger.info("[Pinecone] Index %s created", name)
    else:
        logger.info("[Pinecone] Using existing index %s", name)
    return client.Index(name)


def _as_dict(value: Any) -> dict[str, Any]:
    if value is None:
        return {}
    if isinstance(value, dict):
        return value
    if hasattr(value, "model_dump"):
        return value.model_dump(mode="json")
    return {}


def build_filing_chunk_text(
    filing: dict[str, Any] | Any,
    *,
    filename: str | None = None,
    filing_type: str | None = None,
) -> str:
    """Turn a Core Filing Record into searchable plain text for embedding."""
    data = _as_dict(filing)
    cause = _as_dict(data.get("cause_title"))
    matter = _as_dict(data.get("matter_classification"))
    impugned = _as_dict(data.get("impugned_order"))
    aor = _as_dict(data.get("advocate_on_record"))
    summary = _as_dict(data.get("filing_summary"))

    petitioners = data.get("petitioners") or []
    respondents = data.get("respondents") or []

    petitioner_names = [
        p.get("full_name")
        for p in petitioners
        if isinstance(p, dict) and p.get("full_name")
    ]
    respondent_names = [
        r.get("full_name")
        for r in respondents
        if isinstance(r, dict) and r.get("full_name")
    ]

    parts = [
        f"Filename: {filename}" if filename else None,
        f"Petition type: {filing_type or data.get('petition_type')}",
        f"Court: {data.get('court')}",
        f"Special category: {data.get('special_category')}",
        f"Cause title: {cause.get('formatted_title') or cause.get('raw_text')}",
        f"Petitioners: {', '.join(petitioner_names)}" if petitioner_names else None,
        f"Respondents: {', '.join(respondent_names)}" if respondent_names else None,
        f"Matter: {matter.get('main_category')} / {matter.get('sub_category')}",
        f"PIL: {matter.get('is_pil')}",
        f"Impugned order: {impugned.get('case_number')} "
        f"({impugned.get('earlier_court')}) dated {impugned.get('date_of_impugned_order')}",
        f"AOR: {aor.get('name')} ({aor.get('registration_number')})",
        f"Summary title: {summary.get('matter_title')}",
    ]
    return "\n".join(p for p in parts if p and not p.endswith(": None"))


def split_text_windows(
    text: str,
    *,
    max_chars: int = MAX_CHUNK_CHARS,
    overlap: int = CHUNK_OVERLAP,
) -> list[str]:
    """Split long section text into overlapping windows for embedding."""
    cleaned = re.sub(r"\n{3,}", "\n\n", (text or "").strip())
    if not cleaned:
        return []
    if len(cleaned) <= max_chars:
        return [cleaned]

    windows: list[str] = []
    start = 0
    while start < len(cleaned):
        end = min(start + max_chars, len(cleaned))
        windows.append(cleaned[start:end].strip())
        if end >= len(cleaned):
            break
        start = max(0, end - overlap)
    return [w for w in windows if w]


def build_page_records(
    *,
    base_id: str,
    page_markdown: dict[int, str],
    metadata: dict[str, Any] | None = None,
) -> list[dict[str, Any]]:
    """
    Build Pinecone records from parse page markdown.

    Each page is window-chunked if longer than MAX_CHUNK_CHARS.
    """
    base_meta = metadata or {}
    records: list[dict[str, Any]] = []
    empty_pages = 0

    for page_num in sorted(page_markdown.keys()):
        page_text = (page_markdown.get(page_num) or "").strip()
        if not page_text:
            empty_pages += 1
            continue

        windows = split_text_windows(page_text)
        logger.info(
            "[Pinecone] Page %s → %s char(s) → %s window(s)",
            page_num,
            len(page_text),
            len(windows),
        )
        for chunk_idx, window in enumerate(windows):
            record_id = f"{base_id}:page:{page_num}:{chunk_idx}"
            records.append(
                {
                    "record_id": record_id,
                    "chunk_text": window,
                    "metadata": {
                        **base_meta,
                        "chunk_kind": "page",
                        "page_start": page_num,
                        "page_end": page_num,
                        "pages": str(page_num),
                        "chunk_index": chunk_idx,
                    },
                }
            )

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
        record: dict[str, Any] = {
            "_id": str(record_id),
            text_field: chunk_text,
            **_flatten_metadata(item.get("metadata")),
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


def _raw_hits(result: Any) -> list[Any]:
    hits = getattr(result, "result", result)
    matches = getattr(hits, "hits", None)
    if matches is None and isinstance(hits, dict):
        matches = hits.get("hits", [])
    return list(matches or [])


def search_filings(query: str, *, top_k: int = 5) -> list[dict[str, Any]]:
    """Semantic search over indexed filings (uses the same integrated embed model)."""
    index = ensure_index()
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

    page = fields.get("page_start")
    try:
        page = int(page) if page is not None else None
    except (TypeError, ValueError):
        page = None

    return {
        "record_id": record_id,
        "score": score,
        "text": text,
        "page": page,
        "chunk_kind": fields.get("chunk_kind"),
        "file_name": fields.get("file_name"),
    }


def search_filing_chunks(
    query: str,
    *,
    file_hash: str,
    top_k: int = 8,
    chunk_kind: str | None = None,
) -> list[dict[str, Any]]:
    """Semantic search scoped to a single filing.

    Unlike `search_filings`, this filters on `file_hash` metadata so scrutiny of
    one document never pulls evidence out of a different filing.
    """
    if not file_hash:
        raise ValueError("file_hash is required to scope a filing search")

    client = get_pinecone_client()
    index = ensure_index(client)
    text_field = resolve_text_field(client)

    metadata_filter: dict[str, Any] = {"file_hash": {"$eq": file_hash}}
    if chunk_kind:
        metadata_filter["chunk_kind"] = {"$eq": chunk_kind}

    result = index.search(
        namespace=pinecone_namespace,
        query={
            "top_k": top_k,
            "inputs": {"text": query},
            "filter": metadata_filter,
        },
        fields=[text_field, "chunk_kind", "page_start", "page_end", "file_name"],
    )

    chunks = [c for c in (_to_chunk(h, text_field) for h in _raw_hits(result)) if c]
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
    top_k: int = 8,
    max_chunks: int = 24,
) -> list[dict[str, Any]]:
    """Union the results of several queries into one deduped evidence set.

    Absence-type criteria ("the declaration is missing") are sensitive to
    retrieval recall, so each where-to-look / trigger-word query contributes
    rather than relying on a single top-k over the whole defect.
    """
    seen: dict[str, dict[str, Any]] = {}
    for query in queries:
        if not query or not query.strip():
            continue
        try:
            for chunk in search_filing_chunks(query, file_hash=file_hash, top_k=top_k):
                record_id = chunk.get("record_id")
                if not record_id:
                    continue
                existing = seen.get(record_id)
                if existing is None or (chunk.get("score") or 0) > (
                    existing.get("score") or 0
                ):
                    seen[record_id] = chunk
        except Exception as e:
            logger.warning("[Pinecone] Query failed (%s): %s", query[:60], e)

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
        chunks = chunks[:max_chunks]

    # Present page chunks in reading order; keep the summary chunk first.
    summary = [c for c in chunks if c.get("chunk_kind") == "summary"]
    pages = sorted(
        (c for c in chunks if c.get("chunk_kind") != "summary"),
        key=lambda c: (c.get("page") is None, c.get("page") or 0),
    )
    return summary + pages
