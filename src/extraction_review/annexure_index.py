"""Annexure index descriptions, after Pinecone indexing and defect scrutiny.

One call asks for one or more annexures. Each row is the particulars sentence
for the new index page, plus where that sentence was taken from.
"""

from __future__ import annotations

import asyncio
import logging
import os
import re
from collections.abc import Callable
from typing import Annotated, Any, Literal

from llama_cloud import AsyncLlamaCloud
from pydantic import BaseModel, Field, field_validator
from workflows import Context, Workflow, step
from workflows.events import Event, StartEvent, StopEvent
from workflows.resource import Resource

from .clients import get_llama_cloud_client
from .document_parts import AnnexureMark, numbered_part_slot_id
from .llm import call_structured
from .vector_store import (
    fetch_page_texts,
    scoped_record_base_id,
    search_filing_chunks,
)

logger = logging.getLogger(__name__)

DEFAULT_CONCURRENCY = 4
_SLOT_RE = re.compile(r"^annexure_([a-z])(\d+)$", re.IGNORECASE)
_LABEL_PREFIX_RE = re.compile(
    r"^\s*(?:annexure|annx\.?)\s*[-–—:.\s]*[a-z]\s*[-–—/]?\s*\d+\s*[:.\-–—]\s*",
    re.IGNORECASE,
)
_PAGE_CITE_RE = re.compile(
    r"(?:\s*[,;]?\s*)?[\(\[]?\s*(?:pg\.?|pgs\.?|pages?|pp\.?)\s*"
    r"\d{1,4}(?:\s*(?:[-–—]|to)\s*\d{1,4})?\s*[\)\]]?",
    re.IGNORECASE,
)
_PAREN_RANGE_RE = re.compile(r"\s*[\(\[]\s*\d{1,4}\s*[-–—]\s*\d{1,4}\s*[\)\]]")
_PART_LABEL_RE = re.compile(
    r"^(?:list of dates(?:\s*(?:&|and)\s*events)?|"
    r"annexure\s+[a-z]-?\d+(?:,\s*(?:first|last)\s+page)?)$",
    re.IGNORECASE,
)

SYSTEM_PROMPT = """You write one index description for a Supreme Court paper-book annexure.

Follow this order:
1. If the List of Dates has an entry for this annexure, take the description from that entry.
2. If that entry already says True Copy, Certified Copy, or Copy, keep that wording.
3. If the copy type is missing, look at the annexure first page, then the last page, for a true-copy or certified-copy stamp or certification.
4. If the List of Dates has no description, write one from the annexure. The first page is usually enough (document name, date, and parties or case number when they are printed). Use the last page only when the first page does not identify the document or the copy type.
5. Parties and the case number are optional. Include them only when the source actually names them. Do not invent a party or a case.

description rules:
- Return only the particulars sentence.
- Do not start it with "ANNEXURE P-1:" or any annexure label.
- Do not include a page number or a page range, even if the List of Dates line has one.

source is "list_of_dates" when the sentence comes from the List of Dates, otherwise "annexure".
source_location is the exact wording you took the sentence from, copied from the evidence. It is not a part name such as "List of Dates & Events" or "Annexure P-1, first page". Keep a page range inside source_location when that range is part of the original wording.
confidence is a number from 0 to 1.
"""


class Status(Event):
    level: Literal["info", "warning", "error"]
    message: str


class AnnexureDraft(BaseModel):
    source: Literal["list_of_dates", "annexure"]
    source_location: str
    confidence: float = Field(ge=0, le=1)
    description: str

    @field_validator("confidence", mode="before")
    @classmethod
    def _scale_percent(cls, value: object) -> object:
        if (
            isinstance(value, (int, float))
            and not isinstance(value, bool)
            and value > 1
        ):
            return float(value) / 100.0
        return value


class AnnexureDescription(BaseModel):
    annexure: str
    source: Literal["list_of_dates", "annexure"]
    source_location: str
    confidence: float
    description: str


class AnnexureIndexEvent(StartEvent):
    agent_data_id: str
    annexures: list[str]


class AnnexureIndexResponse(StopEvent):
    agent_data_id: str
    annexures: list[AnnexureDescription]


class AnnexureIndexState(BaseModel):
    agent_data_id: str | None = None


def annexure_index_concurrency() -> int:
    raw = (os.getenv("ANNEXURE_INDEX_CONCURRENCY") or "").strip()
    try:
        return max(1, int(raw)) if raw else DEFAULT_CONCURRENCY
    except ValueError:
        return DEFAULT_CONCURRENCY


def resolve_annexure_token(raw: str) -> tuple[str, str, list[str]]:
    """Return ``(slot_id, label, names that may appear on the filing)``."""
    text = (raw or "").strip()
    if not text:
        raise ValueError("annexure id is empty")
    slot_match = _SLOT_RE.fullmatch(text.replace(" ", ""))
    if slot_match:
        letter = slot_match.group(1).upper()
        number = int(slot_match.group(2))
        mark = AnnexureMark(number=number, series=letter)
        names = [mark.label]
        if letter == "E":
            names.append(f"Annexure E-{number}")
        return mark.slot_id, mark.label, names
    slot = numbered_part_slot_id(text)
    if slot and _SLOT_RE.fullmatch(slot):
        parsed = _SLOT_RE.fullmatch(slot)
        assert parsed is not None
        letter = parsed.group(1).upper()
        number = int(parsed.group(2))
        mark = AnnexureMark(number=number, series=letter)
        names = [mark.label, text]
        if letter == "E":
            names.append(f"Annexure E-{number}")
        return mark.slot_id, mark.label, names
    raise ValueError(f"Unknown annexure id {raw!r}. Use annexure_p1 or Annexure P-1.")


def clean_index_description(text: str) -> str:
    """Drop an annexure label and any page number from the particulars sentence."""
    cleaned = (text or "").strip()
    cleaned = _LABEL_PREFIX_RE.sub("", cleaned)
    cleaned = _PAGE_CITE_RE.sub("", cleaned)
    cleaned = _PAREN_RANGE_RE.sub("", cleaned)
    cleaned = re.sub(r"\s{2,}", " ", cleaned)
    cleaned = re.sub(r"\s+([,.;])", r"\1", cleaned)
    cleaned = cleaned.strip(" \t-–—,;")
    if cleaned and cleaned[-1] not in ".!?":
        cleaned = f"{cleaned}."
    return cleaned


def _norm_name(name: str) -> str:
    return re.sub(r"[^a-z0-9]+", "", (name or "").casefold())


def document_spans(payload: dict[str, Any]) -> list[dict[str, Any]]:
    record = payload.get("data") if isinstance(payload.get("data"), dict) else payload
    docs = record.get("documents") if isinstance(record, dict) else None
    if not isinstance(docs, list):
        docs = payload.get("documents")
    if not isinstance(docs, list):
        return []
    spans: list[dict[str, Any]] = []
    for item in docs:
        if not isinstance(item, dict):
            continue
        name = str(item.get("name") or "").strip()
        if not name:
            continue
        spans.append(
            {
                "name": name,
                "start_page": _page_num(item.get("start_page")),
                "end_page": _page_num(item.get("end_page")),
            }
        )
    return spans


def _page_num(value: object) -> int | None:
    try:
        number = int(value)  # type: ignore[arg-type]
    except (TypeError, ValueError):
        return None
    return number if number > 0 else None


def _span_pages(span: dict[str, Any] | None) -> list[int]:
    if not span:
        return []
    start = span.get("start_page")
    end = span.get("end_page") or start
    if not isinstance(start, int) or not isinstance(end, int):
        return []
    if end < start:
        start, end = end, start
    return list(range(start, end + 1))


def _lod_span(documents: list[dict[str, Any]]) -> dict[str, Any] | None:
    matches = [
        doc
        for doc in documents
        if "listofdates" in _norm_name(str(doc.get("name") or ""))
    ]
    if not matches:
        return None
    exact = [
        doc
        for doc in matches
        if _norm_name(str(doc.get("name") or "")) == "listofdatesevents"
    ]
    return exact[0] if exact else matches[0]


def _annexure_span(
    documents: list[dict[str, Any]], names: list[str]
) -> dict[str, Any] | None:
    wanted = {_norm_name(name) for name in names}
    for doc in documents:
        if _norm_name(str(doc.get("name") or "")) in wanted:
            return doc
    return None


def _is_part_label(text: str) -> bool:
    return bool(_PART_LABEL_RE.fullmatch(re.sub(r"\s+", " ", (text or "").strip())))


def _excerpt_around(label: str, text: str) -> str:
    body = (text or "").strip()
    if not body:
        return ""
    match = re.search(re.escape(label), body, re.IGNORECASE)
    if match is None:
        compact = re.sub(r"[^a-z0-9]+", "", label.casefold())
        for line in body.splitlines():
            if compact and compact in re.sub(r"[^a-z0-9]+", "", line.casefold()):
                return line.strip()
        return body[:1200].strip()
    line_start = body.rfind("\n", 0, match.start())
    start = 0 if line_start < 0 else line_start + 1
    line_end = body.find("\n", match.end())
    end = len(body) if line_end < 0 else line_end
    quote = body[start:end].strip()
    return quote or body[match.start() : match.end() + 400].strip()


def _join_pages(page_text: dict[int, str], pages: list[int]) -> str:
    return "\n\n".join(
        page_text[page].strip() for page in pages if page_text.get(page, "").strip()
    ).strip()


def _user_prompt(
    *,
    slot_id: str,
    label: str,
    lod_text: str,
    first_page: int | None,
    first_text: str,
    last_page: int | None,
    last_text: str,
) -> str:
    sections = [
        f"Annexure id: {slot_id}",
        f"Annexure label: {label}",
        "",
        "List of Dates:",
        lod_text.strip() or "(no List of Dates text)",
        "",
        f"Annexure first page{f' {first_page}' if first_page else ''}:",
        first_text.strip() or "(no first-page text)",
    ]
    if last_page and last_page != first_page:
        sections.extend(
            [
                "",
                f"Annexure last page {last_page}:",
                last_text.strip() or "(no last-page text)",
            ]
        )
    return "\n".join(sections)


async def describe_annexures(
    *,
    annexures: list[str],
    documents: list[dict[str, Any]],
    file_hash: str,
    fetch_pages: Callable[..., dict[int, str]] | None = None,
) -> list[AnnexureDescription]:
    """Describe each requested annexure, in request order."""
    if not annexures:
        raise ValueError("annexures must contain at least one annexure id")
    if not file_hash:
        raise ValueError(
            "This filing has no file_hash, so Pinecone pages cannot be read."
        )

    resolved = [resolve_annexure_token(item) for item in annexures]
    lod = _lod_span(documents)
    lod_pages = _span_pages(lod)
    needed: set[int] = set(lod_pages)
    spans: list[dict[str, Any] | None] = []
    for _slot, _label, names in resolved:
        span = _annexure_span(documents, names)
        spans.append(span)
        pages = _span_pages(span)
        if pages:
            needed.add(pages[0])
            needed.add(pages[-1])
    missing = [
        slot
        for (slot, _label, _names), span in zip(resolved, spans, strict=True)
        if span is None
    ]
    if missing:
        joined = ", ".join(missing)
        raise ValueError(f"Annexure not found in this filing: {joined}")

    reader = fetch_pages or fetch_page_texts
    page_text = reader(base_id=file_hash, pages=sorted(needed))
    lod_text = _join_pages(page_text, lod_pages)
    semaphore = asyncio.Semaphore(annexure_index_concurrency())

    async def _one(
        slot_id: str,
        label: str,
        span: dict[str, Any],
    ) -> AnnexureDescription:
        pages = _span_pages(span)
        first = pages[0] if pages else None
        last = pages[-1] if pages else None
        first_text = page_text.get(first, "") if first else ""
        last_text = page_text.get(last, "") if last else ""
        if not lod_text and not first_text.strip() and not last_text.strip():
            raise ValueError(f"No Pinecone text for {slot_id}.")
        async with semaphore:
            draft, _usage = await call_structured(
                system_prompt=SYSTEM_PROMPT,
                user_prompt=_user_prompt(
                    slot_id=slot_id,
                    label=label,
                    lod_text=lod_text,
                    first_page=first,
                    first_text=first_text,
                    last_page=last,
                    last_text=last_text,
                ),
                response_model=AnnexureDraft,
            )
        description = clean_index_description(draft.description)
        if not description:
            raise ValueError(f"Empty description for {slot_id}.")
        source_location = (draft.source_location or "").strip()
        if not source_location or _is_part_label(source_location):
            source_location = (
                _excerpt_around(label, lod_text)
                if draft.source == "list_of_dates"
                else (first_text or last_text).strip()
            )
        return AnnexureDescription(
            annexure=slot_id,
            source=draft.source,
            source_location=source_location,
            confidence=draft.confidence,
            description=description,
        )

    return list(
        await asyncio.gather(
            *[
                _one(slot, label, span)
                for (slot, label, _names), span in zip(resolved, spans, strict=True)
                if span is not None
            ]
        )
    )


def _vector_index(payload: dict[str, Any]) -> dict[str, Any]:
    metadata = payload.get("metadata") if isinstance(payload.get("metadata"), dict) else {}
    vector = metadata.get("vector_index")
    return vector if isinstance(vector, dict) else {}


def _scoped_base_id(
    *,
    file_hash: str,
    organization_id: str,
    workspace_id: str,
    generation: str,
) -> str | None:
    if not all((file_hash, organization_id, workspace_id, generation)):
        return None
    return scoped_record_base_id(
        file_hash=file_hash,
        organization_id=organization_id,
        workspace_id=workspace_id,
        index_generation=generation,
    )


def _record_base_id(record_id: str) -> str:
    for marker in (":page:", ":manifest:", ":summary"):
        if marker in record_id:
            return record_id.split(marker, 1)[0]
    return ""


def probe_pages_for(documents: list[dict[str, Any]]) -> list[int]:
    """A few document edges, enough to tell whether a Pinecone id has this filing."""
    pages: list[int] = []
    for span in documents:
        for key in ("start_page", "end_page"):
            number = span.get(key)
            if isinstance(number, int) and number > 0 and number not in pages:
                pages.append(number)
            if len(pages) >= 8:
                return pages
    return pages


def _pages_have_text(
    base_id: str,
    pages: list[int],
    fetch_pages: Callable[..., dict[int, str]],
) -> bool:
    wanted = [page for page in pages if page > 0][:8]
    if not base_id or not wanted:
        return False
    found = fetch_pages(base_id=base_id, pages=wanted)
    return any(str(text or "").strip() for text in found.values())


def _search_page_hits(
    *,
    file_hash: str,
    organization_id: str | None,
    workspace_id: str | None,
) -> list[dict[str, Any]]:
    return search_filing_chunks(
        "annexure",
        file_hash=file_hash,
        organization_id=organization_id,
        workspace_id=workspace_id,
        top_k=3,
        vector_schema_version=None,
    )


def resolve_page_base_id(
    payload: dict[str, Any],
    *,
    probe_pages: list[int],
    fetch_pages: Callable[..., dict[int, str]] | None = None,
    search_pages: Callable[..., list[dict[str, Any]]] | None = None,
) -> str:
    """Pinecone id prefix for this filing's page text.

    Indexing stores pages under a scoped id, not the raw file hash. When the
    active vector generation has no pages, use the id on the records that do.
    """
    file_hash = str(payload.get("file_hash") or "").strip()
    vector = _vector_index(payload)
    organization_id = str(
        vector.get("organization_id") or payload.get("organization_id") or ""
    ).strip()
    workspace_id = str(
        vector.get("workspace_id") or payload.get("workspace_id") or ""
    ).strip()
    reader = fetch_pages or fetch_page_texts
    candidates: list[str] = []
    active = _scoped_base_id(
        file_hash=str(vector.get("file_hash") or file_hash).strip(),
        organization_id=organization_id,
        workspace_id=workspace_id,
        generation=str(vector.get("generation") or "").strip(),
    )
    if active:
        candidates.append(active)
    metadata = payload.get("metadata") if isinstance(payload.get("metadata"), dict) else {}
    report = metadata.get("scrutiny_report")
    report_generation = (
        str(report.get("index_generation") or "").strip()
        if isinstance(report, dict)
        else ""
    )
    reported = _scoped_base_id(
        file_hash=file_hash,
        organization_id=organization_id,
        workspace_id=workspace_id,
        generation=report_generation,
    )
    if reported and reported not in candidates:
        candidates.append(reported)
    if file_hash and file_hash not in candidates:
        candidates.append(file_hash)

    for base_id in candidates:
        if _pages_have_text(base_id, probe_pages, reader):
            return base_id

    finder = search_pages or _search_page_hits
    if file_hash:
        hits = finder(
            file_hash=file_hash,
            organization_id=organization_id or None,
            workspace_id=workspace_id or None,
        )
        for hit in hits:
            prefix = _record_base_id(str(hit.get("record_id") or ""))
            if prefix and prefix not in candidates and _pages_have_text(
                prefix, probe_pages, reader
            ):
                logger.warning(
                    "Annexure pages for workspace %s are stored under a Pinecone "
                    "generation other than the active vector index",
                    workspace_id or file_hash[:12],
                )
                return prefix
    return candidates[0] if candidates else file_hash


def _payload_dict(item: Any) -> dict[str, Any]:
    data = getattr(item, "data", None)
    if hasattr(data, "model_dump"):
        dumped = data.model_dump(mode="json")
        return dumped if isinstance(dumped, dict) else {}
    if isinstance(data, dict):
        return dict(data)
    return {}


class AnnexureIndexWorkflow(Workflow):
    """Read Pinecone pages and describe the annexures named on the event."""

    @step()
    async def describe(
        self,
        event: AnnexureIndexEvent,
        ctx: Context[AnnexureIndexState],
        llama_cloud_client: Annotated[
            AsyncLlamaCloud, Resource(get_llama_cloud_client)
        ],
    ) -> AnnexureIndexResponse:
        ctx.write_event_to_stream(
            Status(level="info", message="Loading Agent Data for annexure index")
        )
        item = await llama_cloud_client.beta.agent_data.get(event.agent_data_id)
        payload = _payload_dict(item)
        metadata = (
            payload.get("metadata") if isinstance(payload.get("metadata"), dict) else {}
        )
        if not metadata.get("scrutiny_report"):
            raise ValueError(
                "Annexure index runs after scrutiny. Run index or scrutiny first."
            )
        documents = document_spans(payload)
        page_base_id = resolve_page_base_id(
            payload, probe_pages=probe_pages_for(documents)
        )
        async with ctx.store.edit_state() as state:
            state.agent_data_id = event.agent_data_id
        ctx.write_event_to_stream(
            Status(
                level="info",
                message="Fetching Pinecone pages for List of Dates and annexures",
            )
        )
        ctx.write_event_to_stream(
            Status(
                level="info",
                message=f"Describing {len(event.annexures)} annexure(s) with the LLM",
            )
        )
        rows = await describe_annexures(
            annexures=list(event.annexures),
            documents=documents,
            file_hash=page_base_id,
        )
        ctx.write_event_to_stream(
            Status(level="info", message="Annexure descriptions complete")
        )
        return AnnexureIndexResponse(
            agent_data_id=event.agent_data_id,
            annexures=rows,
        )


workflow = AnnexureIndexWorkflow(timeout=None)
