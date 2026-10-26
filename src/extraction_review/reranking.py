"""Defect-aware Pinecone reranking for the retrieved filing evidence pool."""

from __future__ import annotations

import logging
import os
import threading
from typing import Any

from .document_parts import (
    parts_named_in_where_to_look,
    preferred_parts_for_defect,
)
from .scrutiny.rules import Defect
from .vector_store import get_pinecone_client

logger = logging.getLogger(__name__)

DEFAULT_RERANK_MODEL = "bge-reranker-v2-m3"
DEFAULT_MAX_CANDIDATES = 60
DEFAULT_CONCURRENCY = 4

_rerank_semaphore: threading.Semaphore | None = None
_semaphore_lock = threading.Lock()


def _bool_env(name: str, default: bool) -> bool:
    raw = os.getenv(name)
    if raw is None:
        return default
    return raw.strip().lower() in {"1", "true", "yes", "on"}


def _int_env(name: str, default: int) -> int:
    try:
        return int(os.getenv(name, "") or default)
    except ValueError:
        return default


def reranking_enabled() -> bool:
    return _bool_env("SCRUTINY_RERANK_ENABLED", True)


def rerank_model() -> str:
    return os.getenv("PINECONE_RERANK_MODEL", DEFAULT_RERANK_MODEL).strip()


def rerank_max_candidates() -> int:
    return max(1, _int_env("SCRUTINY_RERANK_MAX_CANDIDATES", DEFAULT_MAX_CANDIDATES))


def _rerank_gate() -> threading.Semaphore:
    global _rerank_semaphore
    if _rerank_semaphore is None:
        with _semaphore_lock:
            if _rerank_semaphore is None:
                concurrency = max(
                    1, _int_env("PINECONE_RERANK_CONCURRENCY", DEFAULT_CONCURRENCY)
                )
                _rerank_semaphore = threading.Semaphore(concurrency)
    return _rerank_semaphore


def build_rerank_query(defect: Defect) -> str:
    """Build one focused query from the catalogue's defect-specific evidence."""

    parts = parts_named_in_where_to_look(defect) or preferred_parts_for_defect(defect)
    values = [
        defect.defect,
        defect.requirement,
        *(defect.where_to_look or []),
        *(parts or []),
        defect.trigger_words or "",
        defect.ai_note or "",
    ]
    seen: set[str] = set()
    lines: list[str] = []
    for value in values:
        text = " ".join(str(value or "").split()).strip()
        folded = text.casefold()
        if text and folded not in seen:
            seen.add(folded)
            lines.append(text)
    return "\n".join(lines)


def _result_rows(result: Any) -> list[Any]:
    if isinstance(result, dict):
        return list(result.get("data") or [])
    return list(getattr(result, "data", None) or [])


def _value(row: Any, key: str) -> Any:
    if isinstance(row, dict):
        return row.get(key)
    return getattr(row, key, None)


def rerank_chunks_for_defect(
    chunks: list[dict[str, Any]],
    defect: Defect,
) -> list[dict[str, Any]]:
    """Annotate page chunks with cross-encoder scores; fail open on errors.

    Summary/manifest records are not sent to the reranker. Candidate order is
    preserved here because the downstream selector handles part coverage and
    final reading order.
    """

    if not reranking_enabled() or not chunks:
        return chunks
    candidate_indexes = [
        index
        for index, chunk in enumerate(chunks)
        if chunk.get("chunk_kind") == "page" and str(chunk.get("text") or "").strip()
    ][: rerank_max_candidates()]
    if len(candidate_indexes) < 2:
        return chunks

    query = build_rerank_query(defect)
    if not query:
        return chunks
    documents = [str(chunks[index]["text"]) for index in candidate_indexes]
    model = rerank_model()
    try:
        with _rerank_gate():
            result = get_pinecone_client().inference.rerank(
                model=model,
                query=query,
                documents=documents,
                top_n=len(documents),
                return_documents=False,
                parameters={"truncate": "END"},
            )
    except Exception as exc:  # noqa: BLE001 - external inference must fail open
        logger.warning("[Scrutiny] Rerank failed for %s: %s", defect.check_id, exc)
        return chunks

    output = [dict(chunk) for chunk in chunks]
    for rank, row in enumerate(_result_rows(result), start=1):
        relative_index = _value(row, "index")
        score = _value(row, "score")
        if not isinstance(relative_index, int) or not 0 <= relative_index < len(
            candidate_indexes
        ):
            continue
        target = output[candidate_indexes[relative_index]]
        try:
            target["rerank_score"] = float(score)
        except (TypeError, ValueError):
            continue
        target["rerank_rank"] = rank
        target["rerank_model"] = model
    return output
