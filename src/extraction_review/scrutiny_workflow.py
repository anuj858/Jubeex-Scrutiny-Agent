"""Registry defect scrutiny for an extracted filing.

Runs the enabled defects from the SCI catalogue against a document that has
already been parsed, extracted and indexed. Evidence comes from the structured
record in Agent Data plus page chunks retrieved from Pinecone for that document.
"""

from __future__ import annotations

import asyncio
import hashlib
import logging
import os
import re
from typing import Annotated, Any, Literal

import httpx
from llama_cloud import AsyncLlamaCloud
from pydantic import BaseModel
from workflows import Context, Workflow, step
from workflows.events import Event, StartEvent, StopEvent
from workflows.resource import Resource

from .clients import agent_name, get_llama_cloud_client
from .config import EXTRACTED_DATA_COLLECTION
from .document_parts import (
    APPLICATION_FAMILY,
    chunks_cover_part,
    expand_parts_for_retrieval,
    family_split_name,
    filing_type_label,
    max_chunks_for_defect,
    missing_required_parts,
    parts_named_in_where_to_look,
    parts_on_page,
    preferred_parts_for_defect,
    required_parts_for_defect,
    select_chunks_for_defect,
    slice_record_for_defect,
)
from .extract_record import stamp_review_status
from .layout_index import (
    LAYOUT_ARTIFACT_KEY_KEY,
    LAYOUT_ARTIFACT_URL_KEY,
    load_layout_index,
)
from .llm import (
    LLMError,
    call_structured,
    llm_provider,
    mask_openrouter_api_key,
    openrouter_enabled,
    openrouter_model,
    openrouter_requests_per_minute,
)
from .process_file import FILE_DOWNLOAD_TIMEOUT_S, _require_pdf_bytes
from .s3_artifacts import STEP_DEFECTS, upload_step_json
from .scrutiny.prompts import (
    build_defect_prompt,
    build_evidence_queries,
    build_system_prompt,
)
from .scrutiny.rules import (
    Catalogue,
    Defect,
    check_id_sort_key,
    defects_for_filing_type,
    enabled_defect_ids,
    normalize_filing_type,
    normalize_special_category,
    refresh_catalogue,
)
from .scrutiny.schema import (
    Coverage,
    DefectFinding,
    DefectResponse,
    ScrutinyReport,
    apply_evidence_pages,
    apply_retrieval_policy,
    apply_status_policy,
    apply_undetermined_policy,
    build_finding,
    failed_finding,
    summarize,
    summarize_usage,
)
from .vector_store import (
    gather_filing_evidence,
    pinecone_enabled,
    scrutiny_max_chunks,
)
from .visual.store import (
    VISUAL_ARTIFACT_KEY_KEY,
    VISUAL_ARTIFACT_URL_KEY,
    load_visual_index,
)

logger = logging.getLogger(__name__)

DEFAULT_CONCURRENCY = 8
DEFAULT_PERSIST_EVERY = 10
SCRUTINY_AFTER_INDEX_ERROR = "Unable to run scrutiny defects. Try after some time."


class ScrutinyAfterIndexError(Exception):
    """Pinecone index finished; nested defect checks could not run."""

    def __init__(self, agent_data_id: str) -> None:
        super().__init__(SCRUTINY_AFTER_INDEX_ERROR)
        self.agent_data_id = agent_data_id


class ScrutinyEvent(StartEvent):
    agent_data_id: str | None = None
    file_hash: str | None = None
    file_url: str | None = None
    organization_id: str | None = None
    workspace_id: str | None = None
    special_category: str | None = None
    filing_type: str | None = None
    court: str | None = None


class Status(Event):
    level: Literal["info", "warning", "error"]
    message: str


class ScrutinyResponse(StopEvent):
    report: ScrutinyReport


class ScrutinyPartial(Event):
    """Live snapshot so the UI can show findings as each check finishes."""

    report: ScrutinyReport
    completed: int
    total: int
    stopped_early: bool = False


class ScrutinyState(BaseModel):
    agent_data_id: str | None = None
    file_hash: str | None = None
    file_url: str | None = None


def _int_env(name: str, default: int) -> int:
    try:
        return int(os.getenv(name, "") or default)
    except ValueError:
        return default


def scrutiny_enabled() -> bool:
    return (os.getenv("SCRUTINY_ENABLED", "true").strip().lower()) not in (
        "false",
        "0",
        "no",
    )


def assert_filing_ready_for_scrutiny(
    review_status: object, file_name: object = None
) -> None:
    """Allow extract-complete filings. Block only rejected records.

    LlamaExtract stores job status on the same ``status`` field (``error``,
    ``success``, …). That is not a user rejection and must not block scrutiny.
    """
    status = str(review_status or "").strip().lower()
    label = str(file_name or "").strip() or "this document"
    if status == "rejected":
        raise ValueError(
            f"Scrutiny cannot run on a rejected filing; {label} is 'rejected'."
        )


async def _load_item(
    client: AsyncLlamaCloud,
    *,
    agent_data_id: str | None,
    file_hash: str | None,
) -> Any:
    """Fetch the Agent Data record by id, falling back to a file_hash lookup."""
    deployment = agent_name or "_public"

    if agent_data_id:
        return await client.beta.agent_data.get(agent_data_id)

    if file_hash:
        paginator = client.beta.agent_data.search(
            deployment_name=deployment,
            collection=EXTRACTED_DATA_COLLECTION,
            filter={"file_hash": {"eq": file_hash}},
            page_size=1,
        )
        async for item in paginator:
            return item

    raise ValueError(
        "Could not load the filing record. Provide a valid agent_data_id, "
        "file_hash, or file_url."
    )


async def _sha256_from_url(file_url: str) -> str:
    url = (file_url or "").strip()
    if not url:
        raise ValueError("file_url is empty")
    timeout = httpx.Timeout(FILE_DOWNLOAD_TIMEOUT_S)
    async with httpx.AsyncClient(timeout=timeout, follow_redirects=True) as http:
        response = await http.get(url)
        response.raise_for_status()
        data = response.content
    if not data:
        raise ValueError(f"Downloaded empty body from {url}")
    _require_pdf_bytes(data, url)
    return hashlib.sha256(data).hexdigest()


def _sanitize_response(
    defect: Defect,
    response: DefectResponse,
    chunks: list[dict[str, Any]],
) -> DefectResponse:
    """Keep the catalogue check_id and drop fixes unless a defect was found."""
    if response.check_id != defect.check_id:
        logger.warning(
            "[Scrutiny] Model returned check_id %s for %s; correcting",
            response.check_id,
            defect.check_id,
        )
    response.check_id = defect.check_id
    response = apply_evidence_pages(response, chunks, defect)
    response = apply_status_policy(response)
    response = apply_undetermined_policy(defect, response, chunks)
    response = apply_retrieval_policy(defect, response, chunks)
    return response


async def _run_defect(
    defect: Defect,
    *,
    catalogue: Catalogue,
    record: dict[str, Any] | None,
    chunks: list[dict[str, Any]],
    filing_type: str | None,
    layout: dict[int, dict[str, Any]] | None = None,
    visual_index: dict[str, Any] | None = None,
) -> DefectFinding:
    pages = sorted({c["page"] for c in chunks if c.get("page") is not None})
    coverage = Coverage(
        chunks_reviewed=len(chunks),
        pages_reviewed=pages,
        structured_record_available=bool(record),
        evidence_complete=bool(record) and not missing_required_parts(defect, chunks),
    )

    raw, usage = await call_structured(
        system_prompt=build_system_prompt(catalogue, filing_type),
        user_prompt=build_defect_prompt(
            defect,
            record=slice_record_for_defect(record, defect),
            chunks=chunks,
            catalogue=catalogue,
            visual_index=visual_index,
        ),
        response_model=DefectResponse,
    )
    response = _sanitize_response(defect, raw, chunks)

    return build_finding(
        defect,
        response,
        evidence_ids=[c["record_id"] for c in chunks if c.get("record_id")],
        coverage=coverage,
        usage=usage,
        chunks=chunks,
        layout=layout,
        visual_index=visual_index,
        record=record if isinstance(record, dict) else None,
    )


async def _chunks_for_defect(
    defect: Defect,
    *,
    file_hash: str | None,
    max_chunks: int,
    use_pinecone: bool,
    layout: dict[int, dict[str, Any]] | None = None,
) -> list[dict[str, Any]]:
    """Fetch this defect's excerpts. Runs inside the concurrency semaphore."""
    page_budget = max_chunks_for_defect(defect, ceiling=max_chunks)
    if not use_pinecone or not file_hash:
        return _add_layout_fallback_chunks(
            defect, [], layout=layout, max_chunks=page_budget
        )

    queries = build_evidence_queries(defect)
    targets = expand_parts_for_retrieval(
        parts_named_in_where_to_look(defect) or preferred_parts_for_defect(defect)
    )
    gather_cap = max(
        max_chunks,
        len(queries) * 3,
        len(targets) * 6,
    )
    try:
        pool = await asyncio.to_thread(
            gather_filing_evidence,
            queries,
            file_hash=file_hash,
            max_chunks=gather_cap,
            document_parts=targets,
        )
    except Exception as e:
        logger.warning(
            "[Scrutiny] Pinecone retrieve failed for %s: %s",
            defect.check_id,
            e,
        )
        return _add_layout_fallback_chunks(
            defect, [], layout=layout, max_chunks=page_budget
        )

    selected = select_chunks_for_defect(pool, defect, max_chunks=page_budget)
    return _add_layout_fallback_chunks(
        defect,
        selected,
        layout=layout,
        max_chunks=page_budget,
    )


def _layout_document_parts(value: Any) -> list[str]:
    """Read both current list metadata and legacy comma-joined layout labels."""
    if isinstance(value, str) and "," in value:
        found: list[str] = []
        for item in value.split(","):
            for name in parts_on_page(item.strip()):
                if name not in found:
                    found.append(name)
        return found
    return parts_on_page(value)


def _layout_page_text(page: dict[str, Any]) -> str:
    """Rebuild conservative OCR lines from the stored grounded word boxes."""
    words = page.get("words") or []
    if not isinstance(words, list):
        return ""
    lines: list[str] = []
    current: list[str] = []
    current_line: Any = object()
    for word in words:
        if not isinstance(word, dict):
            continue
        token = str(word.get("t") or "").strip()
        if not token:
            continue
        line = word.get("line")
        if current and line != current_line:
            lines.append(" ".join(current))
            current = []
        current.append(token)
        current_line = line
    if current:
        lines.append(" ".join(current))
    return "\n".join(lines).strip()


def _add_layout_fallback_chunks(
    defect: Defect,
    chunks: list[dict[str, Any]],
    *,
    layout: dict[int, dict[str, Any]] | None,
    max_chunks: int,
) -> list[dict[str, Any]]:
    """Use stored split/layout OCR to guarantee required document boundaries.

    Pinecone excerpts remain preferred when they cover the same page. Stored
    OCR guarantees that opening captions and closing dates/signatures are not
    lost merely because semantic retrieval selected an interior page.
    """
    if not layout or max_chunks <= 0:
        return chunks

    summary = [chunk for chunk in chunks if chunk.get("chunk_kind") == "summary"][:1]
    pages = [chunk for chunk in chunks if chunk.get("chunk_kind") != "summary"]
    required = required_parts_for_defect(defect)
    if not required:
        return chunks

    candidates: dict[str, list[dict[str, Any]]] = {}
    for part in required:
        found: list[dict[str, Any]] = []
        for page_number, page in sorted(layout.items()):
            if not isinstance(page, dict):
                continue
            names = _layout_document_parts(page.get("document_part"))
            if not names or not chunks_cover_part(
                [{"document_part": names, "chunk_kind": "page"}], part
            ):
                continue
            text = _layout_page_text(page)
            if not text:
                continue
            found.append(
                {
                    "record_id": f"layout:{page_number}:{part}",
                    "score": 0.0,
                    "chunk_kind": "page",
                    "page": page_number,
                    "page_end": page_number,
                    "document_part": names,
                    "text": text,
                    "layout_fallback": True,
                }
            )
        # A filing can contain a petition affidavit and a later application
        # affidavit under the same Split label. Main-petition checks must use
        # the first affidavit run immediately after the petition, not the last
        # affidavit in the paper book.
        if part == "Affidavit" and "Main Petition" in required and found:
            main_pages = [
                page_number
                for page_number, page in sorted(layout.items())
                if isinstance(page, dict)
                and chunks_cover_part(
                    [
                        {
                            "document_part": _layout_document_parts(
                                page.get("document_part")
                            ),
                            "chunk_kind": "page",
                        }
                    ],
                    "Main Petition",
                )
            ]
            if main_pages:
                after = [row for row in found if int(row["page"]) > max(main_pages)]
                if after:
                    first_run = [after[0]]
                    for row in after[1:]:
                        if int(row["page"]) != int(first_run[-1]["page"]) + 1:
                            break
                        first_run.append(row)
                    found = first_run
        if found:
            candidates[part] = found

    if not candidates:
        return chunks

    target_pages = [
        chunk
        for chunk in pages
        if any(chunks_cover_part([chunk], part) for part in required)
    ]
    other_pages = [chunk for chunk in pages if chunk not in target_pages]
    chosen: list[dict[str, Any]] = []
    capacity = max(0, max_chunks - len(summary))

    wording = " ".join(
        [
            str(getattr(defect, "defect", "") or ""),
            str(getattr(defect, "requirement", "") or ""),
            *list(getattr(defect, "where_to_look", None) or []),
        ]
    ).casefold()
    last_first = bool(
        re.search(
            r"last\s+page|end\s+of|below\s+the\s+prayer|date\s+of\s+drafting|"
            r"execution\s+date|signature",
            wording,
        )
    )
    boundaries = ("last", "first") if last_first else ("first", "last")

    # First guarantee the most relevant boundary for every required part,
    # then add the opposite boundary when the evidence budget permits.
    for boundary in boundaries:
        for part in required:
            if len(chosen) >= capacity:
                break
            if chunks_cover_part(chosen, part) and boundary == boundaries[0]:
                continue
            options = candidates.get(part) or []
            if not options:
                continue
            fallback = options[0] if boundary == "first" else options[-1]
            candidate = next(
                (
                    chunk
                    for chunk in target_pages
                    if chunk.get("page") == fallback.get("page")
                    and chunks_cover_part([chunk], part)
                ),
                fallback,
            )
            if any(
                chunk.get("page") == candidate.get("page")
                and chunks_cover_part([chunk], part)
                for chunk in chosen
            ):
                continue
            chosen.append(candidate)

    for chunk in target_pages:
        if len(chosen) >= capacity:
            break
        if chunk not in chosen:
            chosen.append(chunk)

    for chunk in other_pages:
        if len(chosen) >= capacity:
            break
        chosen.append(chunk)

    chosen.sort(key=lambda chunk: (chunk.get("page") is None, chunk.get("page") or 0))
    return summary + chosen


def _special_categories_for_filing(
    *,
    supplied: str | None,
    metadata: dict[str, Any] | None,
    record: dict[str, Any] | None,
    layout: dict[int, dict[str, Any]] | None,
    filing_type: str | None,
) -> list[str]:
    """Combine explicit and structurally certain special-category overlays."""
    found: list[str] = []

    def add(value: Any) -> None:
        text = str(value or "").strip()
        if not normalize_special_category(text):
            return
        if text not in found:
            found.append(text)

    add(supplied)
    add((metadata or {}).get("special_category"))
    add((record or {}).get("special_category"))

    pages = layout or {}
    part_names = {
        name
        for page in pages.values()
        if isinstance(page, dict)
        for name in _layout_document_parts(page.get("document_part"))
    }
    if any(family_split_name(name) == APPLICATION_FAMILY for name in part_names):
        add("Interlocutory Applications")

    filing = normalize_filing_type(filing_type)
    if filing in {"writ_petition_civil", "writ_petition_criminal"}:
        for page_number, page in sorted(pages.items()):
            if page_number > 30 or not isinstance(page, dict):
                continue
            if "public interest litigation" in _layout_page_text(page).casefold():
                add("PIL")
                break
    return found


def _defects_for_special_categories(
    filing_type: str | None,
    *,
    special_categories: list[str],
    court: str | None,
) -> list[Defect]:
    """Union base checks with every applicable filing overlay."""
    categories: list[str | None] = special_categories or [None]
    found: list[Defect] = []
    seen: set[str] = set()
    for category in categories:
        for defect in defects_for_filing_type(
            filing_type,
            special_category=category,
            court=court,
        ):
            if defect.check_id in seen:
                continue
            seen.add(defect.check_id)
            found.append(defect)
    return found


async def collect_defect_findings(
    defects: list[Defect],
    runner: Any,
    *,
    concurrency: int,
    stop_on_error: bool = True,
    on_update: Any = None,
) -> tuple[list[DefectFinding], bool]:
    """Run defect checks with a concurrency cap.

    Completed findings are published immediately via `on_update`. On the first
    failed check, remaining queued calls are cancelled so they do not spend
    more tokens. In-flight calls that already finished are still kept.
    """
    semaphore = asyncio.Semaphore(max(1, concurrency))
    abort = asyncio.Event()
    findings: list[DefectFinding] = []
    seen: set[str] = set()
    stopped_early = False

    async def guarded(defect: Defect) -> DefectFinding:
        if abort.is_set():
            raise asyncio.CancelledError
        async with semaphore:
            if abort.is_set():
                raise asyncio.CancelledError
            finding = await runner(defect)
            if stop_on_error and getattr(finding, "error", None):
                abort.set()
            return finding

    tasks = [asyncio.create_task(guarded(defect)) for defect in defects]
    added_after_cancel = False
    try:
        for finished in asyncio.as_completed(tasks):
            try:
                finding = await finished
            except asyncio.CancelledError:
                continue
            except Exception:
                stopped_early = True
                abort.set()
                for task in tasks:
                    if not task.done():
                        task.cancel()
                break
            if finding.check_id in seen:
                continue
            seen.add(finding.check_id)
            findings.append(finding)
            if on_update is not None:
                await on_update(list(findings), False)
            if stop_on_error and finding.error:
                stopped_early = True
                abort.set()
                for task in tasks:
                    if not task.done():
                        task.cancel()
                break
    finally:
        leftovers = await asyncio.gather(*tasks, return_exceptions=True)
        for result in leftovers:
            if isinstance(result, DefectFinding) and result.check_id not in seen:
                seen.add(result.check_id)
                findings.append(result)
                added_after_cancel = True
        if on_update is not None and (stopped_early or added_after_cancel):
            await on_update(list(findings), stopped_early)
    return findings, stopped_early


class ScrutinyWorkflow(Workflow):
    """Check an extracted filing against the SCI registry defect catalogue."""

    @step()
    async def run_scrutiny(
        self,
        event: ScrutinyEvent,
        ctx: Context[ScrutinyState],
        llama_cloud_client: Annotated[
            AsyncLlamaCloud, Resource(get_llama_cloud_client)
        ],
    ) -> ScrutinyResponse:
        if not scrutiny_enabled():
            raise ValueError("Scrutiny is disabled (SCRUTINY_ENABLED=false)")
        if not openrouter_enabled():
            if llm_provider() == "vertex":
                raise ValueError(
                    "LLM_PROVIDER=vertex but GOOGLE_CLOUD_PROJECT is not set"
                )
            raise ValueError(
                "OPENROUTER_API_KEY is not set, so defect checks cannot run"
            )
        logger.info(
            "LLM provider=%s key=%s model=%s rpm=%s",
            llm_provider(),
            mask_openrouter_api_key(),
            openrouter_model(),
            openrouter_requests_per_minute() or "unlimited",
        )

        async with ctx.store.edit_state() as state:
            state.agent_data_id = event.agent_data_id
            state.file_hash = event.file_hash
            state.file_url = event.file_url

        item = await _load_item(
            llama_cloud_client,
            agent_data_id=event.agent_data_id,
            file_hash=event.file_hash,
        )

        payload: dict[str, Any] = dict(getattr(item, "data", None) or {})
        extract_status = payload.get("status")
        payload = stamp_review_status(payload)
        if payload.get("status") != extract_status:
            item_id = str(getattr(item, "id", "") or event.agent_data_id or "")
            if item_id:
                try:
                    await llama_cloud_client.beta.agent_data.update(
                        item_id, data=payload
                    )
                    logger.info(
                        "Normalized Agent Data %s status %r → pending_review",
                        item_id,
                        extract_status,
                    )
                except Exception:
                    logger.warning(
                        "Could not persist pending_review on Agent Data %s",
                        item_id,
                        exc_info=True,
                    )
        review_status = payload.get("status")
        file_name = payload.get("file_name")
        file_hash = payload.get("file_hash") or event.file_hash
        if not file_hash and event.file_url:
            ctx.write_event_to_stream(
                Status(
                    level="info",
                    message="Computing file_hash from file_url",
                )
            )
            file_hash = await _sha256_from_url(event.file_url)
            async with ctx.store.edit_state() as state:
                state.file_hash = file_hash
        record = payload.get("data") or {}
        metadata = payload.get("metadata") or {}
        filing_type = (
            event.filing_type
            or metadata.get("classification")
            or record.get("petition_type")
        )
        layout_url = (
            metadata.get(LAYOUT_ARTIFACT_URL_KEY)
            if isinstance(metadata, dict)
            else None
        )
        layout_key = (
            metadata.get(LAYOUT_ARTIFACT_KEY_KEY)
            if isinstance(metadata, dict)
            else None
        )
        layout = await load_layout_index(
            layout_url if isinstance(layout_url, str) else None,
            key=layout_key if isinstance(layout_key, str) else None,
        )
        visual_url = (
            metadata.get(VISUAL_ARTIFACT_URL_KEY)
            if isinstance(metadata, dict)
            else None
        )
        visual_key = (
            metadata.get(VISUAL_ARTIFACT_KEY_KEY)
            if isinstance(metadata, dict)
            else None
        )
        visual_index = await load_visual_index(
            visual_url if isinstance(visual_url, str) else None,
            key=visual_key if isinstance(visual_key, str) else None,
        )
        if not layout:
            logger.warning(
                "Layout index empty pages=0; defect findings will have no "
                "highlight coordinates url=%s key=%s",
                layout_url if isinstance(layout_url, str) else None,
                layout_key if isinstance(layout_key, str) else None,
            )
            ctx.write_event_to_stream(
                Status(
                    level="warning",
                    message=(
                        "No page layout is stored for this filing, so defect "
                        "findings will not include highlight coordinates"
                    ),
                )
            )

        assert_filing_ready_for_scrutiny(review_status, file_name)

        catalogue = refresh_catalogue()
        logger.info(
            "[Scrutiny] Using catalogue %s for %s",
            catalogue.catalogue_version,
            event.agent_data_id,
        )
        special_categories = _special_categories_for_filing(
            supplied=event.special_category,
            metadata=metadata if isinstance(metadata, dict) else None,
            record=record if isinstance(record, dict) else None,
            layout=layout,
            filing_type=filing_type,
        )
        defects = _defects_for_special_categories(
            filing_type,
            special_categories=special_categories,
            court=event.court,
        )

        if not defects:
            covered = sorted({d.main_category for d in catalogue.defects})
            raise ValueError(
                f"No defect checks apply to petition type '{filing_type}'. "
                f"The catalogue currently covers {', '.join(covered) or 'no categories'} "
                f"(enabled checks: {', '.join(enabled_defect_ids())})."
            )

        concurrency = _int_env("SCRUTINY_CONCURRENCY", DEFAULT_CONCURRENCY)
        persist_every = max(
            1, _int_env("SCRUTINY_PERSIST_EVERY", DEFAULT_PERSIST_EVERY)
        )
        max_chunks = scrutiny_max_chunks()
        use_pinecone = pinecone_enabled() and bool(file_hash)

        if not pinecone_enabled():
            ctx.write_event_to_stream(
                Status(
                    level="warning",
                    message=(
                        "Pinecone is disabled, so checks will use the extracted "
                        "record plus stored split-page OCR. Semantic context may "
                        "be less complete."
                    ),
                )
            )
        elif not file_hash:
            ctx.write_event_to_stream(
                Status(
                    level="warning",
                    message=(
                        "No file hash is available, so checks will use the "
                        "extracted record plus stored split-page OCR. Semantic "
                        "context may be less complete."
                    ),
                )
            )
        else:
            ctx.write_event_to_stream(
                Status(
                    level="info",
                    message=(
                        "Each concurrent check will retrieve its own excerpts "
                        f"from Pinecone ({concurrency} at a time)"
                    ),
                )
            )

        category_labels = ", ".join(
            sorted({cat for d in defects for cat in d.main_categories})
        )
        ctx.write_event_to_stream(
            Status(
                level="info",
                message=(
                    f"Checking {file_name or 'filing'} "
                    f"({filing_type_label(filing_type)}) against "
                    f"{len(defects)} defect(s) in {category_labels} "
                    f"(overlays: {', '.join(special_categories) or 'none'}; "
                    f"{concurrency} at a time)"
                ),
            )
        )

        planned = len(defects)

        def build_report(
            current: list[DefectFinding], *, stopped_early: bool
        ) -> ScrutinyReport:
            snapshot = sorted(current, key=lambda f: check_id_sort_key(f.check_id))
            return ScrutinyReport(
                catalogue_id=catalogue.catalogue_id,
                catalogue_version=catalogue.catalogue_version,
                agent_data_id=str(getattr(item, "id", "") or "") or event.agent_data_id,
                file_hash=file_hash,
                petition_type=filing_type,
                model=openrouter_model(),
                disclaimer=catalogue.disclaimer,
                findings=snapshot,
                summary=summarize(snapshot),
                usage=summarize_usage(snapshot, model=openrouter_model()),
                planned_checks=planned,
                stopped_early=stopped_early,
            )

        persist_lock = asyncio.Lock()
        persist_tasks: set[asyncio.Task[None]] = set()

        async def persist_report(report: ScrutinyReport) -> None:
            async with persist_lock:
                await self._persist(llama_cloud_client, item, payload, report, ctx)

        def schedule_persist(report: ScrutinyReport) -> None:
            # Do not queue a remote Agent Data write for every interval while
            # an earlier write is still running. With a large catalogue that
            # produced a long serialized backlog after all checks had already
            # completed. The final report is always persisted below.
            if any(not task.done() for task in persist_tasks):
                return
            task = asyncio.create_task(persist_report(report))
            persist_tasks.add(task)
            task.add_done_callback(persist_tasks.discard)

        async def publish(current: list[DefectFinding], stopped_early: bool) -> None:
            report = build_report(current, stopped_early=stopped_early)
            ctx.write_event_to_stream(
                ScrutinyPartial(
                    report=report,
                    completed=len(current),
                    total=planned,
                    stopped_early=stopped_early,
                )
            )
            done = stopped_early or len(current) >= planned
            if not done and len(current) % persist_every == 0:
                schedule_persist(report)

        async def run_one(defect: Defect) -> DefectFinding:
            ctx.write_event_to_stream(
                Status(
                    level="info",
                    message=f"Checking {defect.check_id}",
                )
            )
            try:
                chunks = await _chunks_for_defect(
                    defect,
                    file_hash=file_hash,
                    max_chunks=max_chunks,
                    use_pinecone=use_pinecone,
                    layout=layout,
                )
                finding = await _run_defect(
                    defect,
                    catalogue=catalogue,
                    record=record,
                    chunks=chunks,
                    filing_type=filing_type,
                    layout=layout,
                    visual_index=visual_index,
                )
            except asyncio.CancelledError:
                raise
            except LLMError as e:
                logger.exception("[Scrutiny] %s failed", defect.check_id)
                ctx.write_event_to_stream(
                    Status(
                        level="error",
                        message=f"{defect.check_id} failed; continuing remaining checks. {e}",
                    )
                )
                return failed_finding(
                    defect,
                    str(e),
                    usage=e.usage,
                    visual_index=visual_index,
                    record=record if isinstance(record, dict) else None,
                )
            except Exception as e:
                logger.exception("[Scrutiny] %s failed", defect.check_id)
                ctx.write_event_to_stream(
                    Status(
                        level="error",
                        message=f"{defect.check_id} failed; continuing remaining checks. {e}",
                    )
                )
                return failed_finding(
                    defect,
                    str(e),
                    visual_index=visual_index,
                    record=record if isinstance(record, dict) else None,
                )

            ctx.write_event_to_stream(
                Status(
                    level="info",
                    message=(
                        f"{defect.check_id} → {finding.status} "
                        f"({finding.confidence:.0%} confidence)"
                    ),
                )
            )
            return finding

        findings, stopped_early = await collect_defect_findings(
            defects,
            run_one,
            concurrency=concurrency,
            stop_on_error=False,
            on_update=publish,
        )
        report = build_report(findings, stopped_early=stopped_early)
        ctx.write_event_to_stream(
            Status(level="info", message="Writing final scrutiny report")
        )
        if persist_tasks:
            await asyncio.gather(*persist_tasks, return_exceptions=True)
        await persist_report(report)
        defects_payload = report.model_dump(mode="json")
        if isinstance(defects_payload, dict):
            defects_payload.setdefault("organization_id", event.organization_id)
            defects_payload.setdefault("workspace_id", event.workspace_id)
        upload_step_json(
            STEP_DEFECTS,
            defects_payload,
            organization_id=event.organization_id,
            workspace_id=event.workspace_id,
        )

        cost_note = ""
        if report.usage and report.usage.cost_usd is not None:
            cost_note = f", OpenRouter {report.usage.cost_usd:.6f} USD"
        elif report.usage and report.usage.total_tokens:
            cost_note = f", {report.usage.total_tokens} tokens"
        if stopped_early:
            ctx.write_event_to_stream(
                Status(
                    level="warning",
                    message=(
                        f"Stopped early after {len(findings)} of {planned} "
                        f"checks{cost_note}. Results below are what completed."
                    ),
                )
            )
        else:
            ctx.write_event_to_stream(
                Status(
                    level="info",
                    message=(
                        f"Scrutiny complete: {report.summary.defects_found} defect(s), "
                        f"{report.summary.needs_review} needing review, "
                        f"{report.summary.not_determined} undetermined{cost_note}"
                    ),
                )
            )
        return ScrutinyResponse(report=report)

    async def _persist(
        self,
        client: AsyncLlamaCloud,
        item: Any,
        payload: dict[str, Any],
        report: ScrutinyReport,
        ctx: Context[ScrutinyState],
    ) -> None:
        """Write the report onto the same Agent Data item LlamaExtract created.

        That keeps one record per filing (no extra collection) and lets a later
        'View last check' read it back from the extraction item.
        """
        item_id = str(getattr(item, "id", "") or "") or report.agent_data_id
        if not item_id:
            ctx.write_event_to_stream(
                Status(
                    level="warning",
                    message="Results could not be saved: the filing has no Agent Data id.",
                )
            )
            return

        updated = dict(payload)
        metadata = dict(updated.get("metadata") or {})
        metadata["scrutiny_report"] = report.model_dump(mode="json")
        updated["metadata"] = metadata

        try:
            await client.beta.agent_data.update(item_id, data=updated)
            logger.info(
                "[Scrutiny] Saved %s/%s finding(s) on extraction item %s%s",
                len(report.findings),
                report.planned_checks or len(report.findings),
                item_id,
                " (stopped early)" if report.stopped_early else "",
            )
        except Exception as e:
            # A storage failure should not lose the results the user is waiting on.
            logger.exception("[Scrutiny] Could not store report")
            ctx.write_event_to_stream(
                Status(
                    level="warning",
                    message=f"Results could not be saved for later: {e}",
                )
            )


workflow = ScrutinyWorkflow(timeout=None)
