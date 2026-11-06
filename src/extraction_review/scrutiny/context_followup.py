"""One extra context hop for an undecided Need Review defect.

This is the tool loop for a single check. It is not a network server. The
first model answer may name one of two tools:

- ``adjacent`` reads at most two stored chunks beside the cited excerpt,
  staying inside that excerpt's inspect part.
- ``search`` embeds one short filing phrase and keeps hits from the inspect
  parts only.

Python runs the tool, then makes at most one more model call with the new
text and the earlier answer. A decided or high-confidence row never enters
this module's second call. Grep ``[ContextFollowup]`` to see the tool, the
ids it fetched, and whether the second call ran.
"""

from __future__ import annotations

import asyncio
import logging
import re
from dataclasses import dataclass
from typing import Any

from ..document_parts import (
    catalogue_context_parts,
    chunks_cover_part,
    distinctive_trigger_phrases,
    required_parts_for_defect,
    slice_record_for_defect,
)
from ..llm import call_structured
from ..vector_store import (
    VECTOR_SCHEMA_VERSION,
    fetch_chunks_by_id,
    page_record_id,
    scoped_record_base_id,
    search_filing_chunks,
)
from .prompts import MAX_EVIDENCE_CHARS, build_defect_prompt
from .rules import Catalogue, Defect, check_requirement
from .schema import (
    DefectResponse,
    EvidenceRequest,
    LlmUsage,
    review_confidence_threshold,
)

logger = logging.getLogger(__name__)

_CONTEXT_GAPS = frozenset(
    {
        "part_not_in_excerpts",
        "element_page_not_in_excerpts",
        "ambiguous",
        "low_confidence",
    }
)
_MAX_NEIGHBOR_PAGES = 4
# A stored page is split into windows of about 3,500 characters. Sixteen
# windows covers a long parsed page. "Previous" must read the last window
# that exists, which can sit above index 3.
_MAX_PAGE_CHUNK_INDEX = 15
_SEARCH_FOLLOWUP_HITS = 5
_TOOLS = frozenset({"adjacent", "search"})
_QUERY_LIMIT = 200
_QUERY_STOP = frozenset(
    {
        "that",
        "this",
        "with",
        "from",
        "have",
        "been",
        "were",
        "will",
        "shall",
        "must",
        "into",
        "such",
        "than",
        "then",
        "them",
        "they",
        "their",
        "there",
        "these",
        "those",
        "which",
        "while",
        "where",
        "when",
        "what",
        "does",
        "not",
        "and",
        "the",
        "for",
        "are",
        "was",
        "has",
        "had",
        "its",
        "any",
        "all",
        "each",
        "every",
        "only",
        "also",
        "under",
        "over",
        "after",
        "before",
        "between",
        "within",
        "without",
        "against",
        "whether",
        "required",
        "requirement",
        "filing",
        "document",
        "page",
    }
)


@dataclass
class ContextFollowupResult:
    """The second raw answer and the chunks the tool actually fetched."""

    action: str
    added: list[dict[str, Any]]
    response: DefectResponse
    usage: LlmUsage
    query: str | None


def _part_name_in(name: str | None, inspect_parts: list[str]) -> str | None:
    """Return the inspect part a label belongs to, if it belongs to one."""
    if not name or not str(name).strip():
        return None
    probe = {"document_part": name, "chunk_kind": "page"}
    for part in inspect_parts:
        if chunks_cover_part([probe], part):
            return part
    return None


def _chunk_part_in(chunk: dict[str, Any], inspect_parts: list[str]) -> str | None:
    for part in inspect_parts:
        if chunks_cover_part([chunk], part):
            return part
    return None


def _chunk_index_of(chunk: dict[str, Any]) -> int | None:
    raw = chunk.get("chunk_index")
    if raw is not None and not isinstance(raw, bool):
        try:
            return int(raw)
        except (TypeError, ValueError):
            return None
    record_id = str(chunk.get("record_id") or "")
    match = re.search(r":page:\d+:(\d+)$", record_id)
    if match is None:
        return None
    return int(match.group(1))


def _record_base_from_id(record_id: str) -> str | None:
    marker = ":page:"
    if marker not in record_id:
        return None
    return record_id.rsplit(marker, 1)[0] or None


def _layout_document_parts(value: Any) -> list[str]:
    from ..scrutiny_workflow import _layout_document_parts as read_parts

    return read_parts(value)


def _layout_page_text(page: dict[str, Any]) -> str:
    from ..scrutiny_workflow import _layout_page_text as read_text

    return read_text(page)


def _content_words(text: str) -> set[str]:
    return {
        word
        for word in re.findall(r"[a-z][a-z'-]{3,}", text.casefold())
        if word not in _QUERY_STOP
    }


def search_phrase_for_followup(defect: Defect, model_text: str) -> str:
    """Phrase embedded against normalized_text.

    A short clause the model copied from the filing stays first. A restatement
    of the Standard is replaced by the catalogue phrases that are printed on
    the page. Up to two of those phrases are added when the model's clause is
    short, so the embedding still names the surrounding form language.
    """
    text = re.sub(r"\s+", " ", model_text or "").strip()
    triggers = distinctive_trigger_phrases(defect)
    words = _content_words(text)
    source = _content_words(
        f"{getattr(defect, 'defect', '')} {check_requirement(defect)}"
    )
    overlap = (len(words & source) / len(words)) if words else 0.0
    restatement = overlap >= 0.7 and len(text) >= 80
    if restatement or not text:
        phrase = ". ".join(triggers)
    else:
        phrase = text
        if len(text) <= 120:
            added = 0
            for trigger in triggers:
                if trigger.casefold() in phrase.casefold():
                    continue
                candidate = f"{phrase}. {trigger}"
                if len(candidate) > _QUERY_LIMIT:
                    break
                phrase = candidate
                added += 1
                if added >= 2:
                    break
    phrase = re.sub(r"\s+", " ", phrase).strip(" .")
    if not phrase:
        phrase = text
    return phrase[:_QUERY_LIMIT].rsplit(" ", 1)[0] if len(phrase) > _QUERY_LIMIT else phrase


def resolve_context_request(
    *,
    model_status: str,
    response: DefectResponse,
    chunks: list[dict[str, Any]],
    inspect_parts: list[str],
) -> tuple[EvidenceRequest | None, str]:
    """Return a tool call, or a skip reason when the model asked and was refused.

    An empty reason means the first answer did not ask for a tool. Those rows
    are not logged.
    """
    request = response.evidence_request
    asked = request is not None and request.action in _TOOLS
    if model_status != "needs_review" or response.status != "needs_review":
        return None, "status_not_needs_review" if asked else ""
    if response.confidence >= review_confidence_threshold():
        return None, "high_confidence" if asked else ""
    if response.gap not in _CONTEXT_GAPS:
        return None, f"gap_{response.gap or 'none'}" if asked else ""
    if not asked or request is None:
        return None, ""
    if not inspect_parts:
        return None, "no_inspect_parts"
    named = _part_name_in(request.document_part, inspect_parts)
    if request.action == "search":
        text = re.sub(r"\s+", " ", request.search_text or "").strip()
        if not text or len(text) > 200:
            return None, "search_text_unusable"
        return (
            request.model_copy(
                update={
                    "search_text": text,
                    "document_part": named,
                    "direction": None,
                    "count": None,
                    "anchor_chunk_id": None,
                }
            ),
            "",
        )
    anchor_id = (request.anchor_chunk_id or "").strip()
    anchor = next(
        (chunk for chunk in chunks if chunk.get("excerpt_id") == anchor_id),
        None,
    )
    if anchor is None or request.direction not in {"previous", "next"}:
        return None, "anchor_or_direction_invalid"
    if request.count not in {1, 2}:
        return None, "count_invalid"
    anchor_part = _chunk_part_in(anchor, inspect_parts)
    if anchor_part is None or _chunk_index_of(anchor) is None:
        return None, "anchor_part_or_index_missing"
    if anchor.get("page") is None:
        return None, "anchor_page_missing"
    return (
        request.model_copy(
            update={
                "anchor_chunk_id": anchor_id,
                "document_part": anchor_part,
                "search_text": None,
            }
        ),
        "",
    )


def _pages_for_part(
    layout: dict[int, dict[str, Any]] | None,
    part: str,
) -> list[int]:
    if not layout:
        return []
    pages: list[int] = []
    for number, page in sorted(layout.items()):
        if not isinstance(page, dict):
            continue
        names = _layout_document_parts(page.get("document_part"))
        if chunks_cover_part(
            [{"document_part": names, "chunk_kind": "page"}],
            part,
        ):
            pages.append(int(number))
    return pages


def _adjacent_slots(
    *,
    page: int,
    index: int,
    direction: str,
    count: int,
    part_pages: list[int],
) -> list[tuple[int, int]]:
    """Reading-order slots. At most a few same-part pages, never another part."""
    slots: list[tuple[int, int]] = []
    if direction == "next":
        for offset in range(1, count + 1):
            slots.append((page, index + offset))
        later = [item for item in part_pages if item > page][:_MAX_NEIGHBOR_PAGES]
        for next_page in later:
            for slot_index in range(count):
                slots.append((next_page, slot_index))
    else:
        for offset in range(1, count + 1):
            if index - offset >= 0:
                slots.append((page, index - offset))
        earlier = [item for item in part_pages if item < page][-_MAX_NEIGHBOR_PAGES:]
        for prev_page in reversed(earlier):
            for slot_index in range(_MAX_PAGE_CHUNK_INDEX, -1, -1):
                slots.append((prev_page, slot_index))
    unique: list[tuple[int, int]] = []
    seen: set[tuple[int, int]] = set()
    for slot in slots:
        if slot in seen:
            continue
        seen.add(slot)
        unique.append(slot)
    return unique


def _already_shown(chunk: dict[str, Any], existing: list[dict[str, Any]]) -> bool:
    record_id = str(chunk.get("record_id") or "")
    if record_id and any(
        str(item.get("record_id") or "") == record_id for item in existing
    ):
        return True
    page = chunk.get("page")
    if page is None:
        return False
    index = chunk.get("chunk_index")
    for item in existing:
        if item.get("page") != page:
            continue
        other = item.get("chunk_index")
        if index is None or other is None or index == other:
            return True
    return False


def _layout_context_chunk(
    page_number: int,
    layout: dict[int, dict[str, Any]],
    part: str,
) -> dict[str, Any] | None:
    page = layout.get(page_number)
    if not isinstance(page, dict):
        return None
    names = _layout_document_parts(page.get("document_part"))
    probe = {"document_part": names, "chunk_kind": "page"}
    if not chunks_cover_part([probe], part):
        return None
    text = _layout_page_text(page)
    if not text:
        return None
    return {
        "record_id": f"layout-context:{page_number}",
        "score": 0.0,
        "chunk_kind": "page",
        "page": page_number,
        "page_end": page_number,
        "document_part": names,
        "text": text,
        "chunk_index": 0,
        "context_followup": True,
    }


def _stored_chunk(
    record_id: str,
    fetched: dict[str, dict[str, Any]],
    part: str,
) -> dict[str, Any] | None:
    chunk = fetched.get(record_id)
    if not chunk or not _chunk_part_in(chunk, [part]):
        return None
    copied = dict(chunk)
    copied["context_followup"] = True
    return copied


def load_adjacent_chunks(
    request: EvidenceRequest,
    *,
    chunks: list[dict[str, Any]],
    layout: dict[int, dict[str, Any]] | None,
    base_id: str | None,
) -> list[dict[str, Any]]:
    """Read at most count real chunks beside the anchor, same inspect part."""
    anchor = next(
        (
            chunk
            for chunk in chunks
            if chunk.get("excerpt_id") == request.anchor_chunk_id
        ),
        None,
    )
    part = request.document_part
    count = request.count if request.count in {1, 2} else None
    if anchor is None or not part or count is None:
        return []
    page = anchor.get("page")
    index = _chunk_index_of(anchor)
    direction = request.direction
    if page is None or index is None or direction not in {"previous", "next"}:
        return []
    page = int(page)
    part_pages = _pages_for_part(layout, part)
    slots = _adjacent_slots(
        page=page,
        index=index,
        direction=direction,
        count=count,
        part_pages=part_pages,
    )
    identity = _record_base_from_id(str(anchor.get("record_id") or "")) or base_id
    ids = (
        [page_record_id(identity, slot_page, slot_index) for slot_page, slot_index in slots]
        if identity
        else []
    )
    logger.info(
        "[ContextFollowup] tool=adjacent anchor=%s direction=%s count=%s part=%s asked=%s",
        request.anchor_chunk_id,
        direction,
        count,
        part,
        ids,
    )
    fetched: dict[str, dict[str, Any]] = {}
    if ids:
        try:
            fetched = fetch_chunks_by_id(ids)
        except Exception:
            logger.warning(
                "[ContextFollowup] tool=adjacent fetch_failed anchor=%s",
                request.anchor_chunk_id,
                exc_info=True,
            )
            fetched = {}
    selected: list[dict[str, Any]] = []
    pages_with_text: set[int] = set()
    for record_id in fetched:
        match = re.search(r":page:(\d+):\d+$", str(record_id))
        if match:
            pages_with_text.add(int(match.group(1)))
    for slot_page, slot_index in slots:
        if len(selected) >= count:
            break
        record_id = page_record_id(identity, slot_page, slot_index) if identity else ""
        stored = _stored_chunk(record_id, fetched, part) if record_id else None
        if stored is None or _already_shown(stored, [*chunks, *selected]):
            continue
        selected.append(stored)
    if layout and len(selected) < count:
        for slot_page, _slot_index in slots:
            if len(selected) >= count:
                break
            if slot_page in pages_with_text:
                continue
            fallback = _layout_context_chunk(slot_page, layout, part)
            if fallback is None or _already_shown(fallback, [*chunks, *selected]):
                pages_with_text.add(slot_page)
                continue
            selected.append(fallback)
            pages_with_text.add(slot_page)
    return selected[:count]


def load_search_chunks(
    request: EvidenceRequest,
    *,
    chunks: list[dict[str, Any]],
    inspect_parts: list[str],
    file_hash: str | None,
    organization_id: str | None,
    workspace_id: str | None,
    index_generation: str | None,
    use_pinecone: bool,
) -> list[dict[str, Any]]:
    """Embed the model's phrase and keep hits from the inspect parts only."""
    text = (request.search_text or "").strip()
    if not text or not use_pinecone or not file_hash:
        logger.info(
            "[ContextFollowup] tool=search skipped=not_indexed query=%r",
            text,
        )
        return []
    parts = [request.document_part] if request.document_part else list(inspect_parts)
    parts = [part for part in parts if part]
    if not parts:
        return []
    kwargs: dict[str, Any] = {
        "file_hash": file_hash,
        "top_k": max(_SEARCH_FOLLOWUP_HITS * 2, 6),
        "chunk_kind": "page",
        "organization_id": organization_id,
        "workspace_id": workspace_id,
        "index_generation": index_generation,
        "vector_schema_version": VECTOR_SCHEMA_VERSION,
    }
    if len(parts) == 1:
        kwargs["document_part"] = parts[0]
    else:
        kwargs["part_labels"] = parts
    logger.info(
        "[ContextFollowup] tool=search query=%r parts=%s",
        text,
        parts,
    )
    try:
        hits = search_filing_chunks(text, **kwargs)
    except Exception:
        logger.warning("[ContextFollowup] tool=search fetch_failed", exc_info=True)
        return []
    selected: list[dict[str, Any]] = []
    for hit in hits:
        if not _chunk_part_in(hit, parts):
            continue
        if _already_shown(hit, [*chunks, *selected]):
            continue
        copied = dict(hit)
        copied["context_followup"] = True
        selected.append(copied)
        if len(selected) >= _SEARCH_FOLLOWUP_HITS:
            break
    return selected


def load_context_chunks(
    request: EvidenceRequest,
    *,
    chunks: list[dict[str, Any]],
    inspect_parts: list[str],
    layout: dict[int, dict[str, Any]] | None,
    file_hash: str | None,
    organization_id: str | None,
    workspace_id: str | None,
    index_generation: str | None,
    use_pinecone: bool,
) -> list[dict[str, Any]]:
    """Dispatch ``adjacent`` or ``search`` and return only new inspect-part text."""
    base_id = None
    if file_hash and organization_id and workspace_id and index_generation:
        try:
            base_id = scoped_record_base_id(
                file_hash=file_hash,
                organization_id=organization_id,
                workspace_id=workspace_id,
                index_generation=index_generation,
            )
        except ValueError:
            base_id = None
    if request.action == "adjacent":
        return load_adjacent_chunks(
            request,
            chunks=chunks,
            layout=layout,
            base_id=base_id,
        )
    if request.action == "search":
        return load_search_chunks(
            request,
            chunks=chunks,
            inspect_parts=inspect_parts,
            file_hash=file_hash,
            organization_id=organization_id,
            workspace_id=workspace_id,
            index_generation=index_generation,
            use_pinecone=use_pinecone,
        )
    return []


def evidence_chunks_for_rerun(
    defect: Defect,
    chunks: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    """Keep only the inspect parts and context parts for a second look.

    The first retrieval can include a neighbouring document. A rerun must
    judge the check from the parts it was written to inspect, not from that
    extra text.
    """
    allowed = list(
        dict.fromkeys(
            [
                *required_parts_for_defect(defect),
                *catalogue_context_parts(defect),
            ]
        )
    )
    if not allowed:
        return list(chunks)
    kept: list[dict[str, Any]] = []
    for chunk in chunks:
        if chunk.get("chunk_kind") == "summary":
            kept.append(chunk)
            continue
        if any(chunks_cover_part([chunk], part) for part in allowed):
            kept.append(chunk)
    return kept


def _additional_chunks_fit(
    chunks: list[dict[str, Any]],
    added: list[dict[str, Any]],
) -> bool:
    """A second call needs the new text to fit in the evidence budget."""
    used = sum(len(str(chunk.get("text") or "")) + 300 for chunk in chunks)
    extra = sum(len(str(chunk.get("text") or "")) + 300 for chunk in added)
    return bool(added) and used < MAX_EVIDENCE_CHARS and used + extra <= MAX_EVIDENCE_CHARS


async def serve_context_followup(
    *,
    defect: Defect,
    catalogue: Catalogue,
    record: dict[str, Any] | None,
    chunks: list[dict[str, Any]],
    layout: dict[int, dict[str, Any]] | None,
    visual_index: dict[str, Any] | None,
    model: str,
    system_prompt: str,
    model_status: str,
    response: DefectResponse,
    prior_decision: dict[str, Any],
    prior_generation_id: str | None,
    file_hash: str | None,
    organization_id: str | None,
    workspace_id: str | None,
    index_generation: str | None,
    use_pinecone: bool,
) -> ContextFollowupResult | None:
    """Run one tool and, when it returns new text, one follow-up model call.

    The returned response is the raw second answer. The caller still applies
    the same sanitizers as the first call. ``None`` means no second call.
    """
    inspect_parts = required_parts_for_defect(defect)
    request, reason = resolve_context_request(
        model_status=model_status,
        response=response,
        chunks=chunks,
        inspect_parts=inspect_parts,
    )
    if request is None:
        if reason:
            logger.info(
                "[ContextFollowup] check=%s skipped=%s",
                defect.check_id,
                reason,
            )
        return None
    if request.action == "search":
        embedded = search_phrase_for_followup(defect, request.search_text or "")
        if embedded and embedded != (request.search_text or ""):
            logger.info(
                "[ContextFollowup] check=%s search_phrase=%r model_phrase=%r",
                defect.check_id,
                embedded,
                request.search_text,
            )
            request = request.model_copy(update={"search_text": embedded})
    prior_call_id = f"scrutiny:{defect.check_id}"
    logger.info(
        "[ContextFollowup] check=%s tool=%s prior_call=%s generation=%s",
        defect.check_id,
        request.action,
        prior_call_id,
        prior_generation_id,
    )
    added = await asyncio.to_thread(
        load_context_chunks,
        request,
        chunks=chunks,
        inspect_parts=inspect_parts,
        layout=layout,
        file_hash=file_hash,
        organization_id=organization_id,
        workspace_id=workspace_id,
        index_generation=index_generation,
        use_pinecone=use_pinecone,
    )
    if not _additional_chunks_fit(chunks, added):
        logger.info(
            "[ContextFollowup] check=%s skipped=%s fetched=%s",
            defect.check_id,
            "no_new_chunk" if not added else "evidence_budget",
            [chunk.get("record_id") for chunk in added],
        )
        return None
    shown = evidence_chunks_for_rerun(defect, chunks)
    follow_prompt = build_defect_prompt(
        defect,
        record=slice_record_for_defect(record, defect),
        chunks=shown,
        catalogue=catalogue,
        visual_index=visual_index,
        layout=layout,
        extra_chunks=added,
        context_followup=True,
        prior_call_id=prior_call_id,
        prior_generation_id=prior_generation_id,
        prior_decision=prior_decision,
    )
    second_call_id = f"scrutiny:{defect.check_id}:context-followup"
    logger.info(
        "[ContextFollowup] check=%s second_call=%s ids=%s",
        defect.check_id,
        second_call_id,
        [chunk.get("record_id") for chunk in added],
    )
    follow_raw, follow_usage = await call_structured(
        system_prompt=system_prompt,
        user_prompt=follow_prompt,
        response_model=DefectResponse,
        model=model,
        request_label=second_call_id,
    )
    return ContextFollowupResult(
        action=request.action,
        added=added,
        response=follow_raw,
        usage=follow_usage,
        query=request.search_text,
    )
