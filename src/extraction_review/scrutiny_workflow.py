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
import time
from dataclasses import dataclass, field
from typing import Annotated, Any, Literal

import httpx
from llama_cloud import AsyncLlamaCloud
from pydantic import BaseModel, Field
from workflows import Context, Workflow, step
from workflows.events import Event, StartEvent, StopEvent
from workflows.resource import Resource

from .clients import agent_name, get_llama_cloud_client
from .config import EXTRACTED_DATA_COLLECTION
from .document_parts import (
    ANNEXURE_FAMILY,
    APPLICATION_FAMILY,
    _is_numbered_annexure,
    _requires_annexure_collection_coverage,
    _requires_complete_part_coverage,
    catalogue_context_parts,
    chunks_cover_part,
    distinctive_trigger_phrases,
    expand_parts_for_retrieval,
    family_split_name,
    filing_type_label,
    max_chunks_for_defect,
    missing_required_parts,
    parts_named_in_where_to_look,
    parts_on_page,
    preferred_parts_for_defect,
    required_parts_for_defect,
    retrieval_plan_for_defect,
    select_chunks_for_defect,
    slice_record_for_defect,
)
from .extract_record import match_front_special_categories, stamp_review_status
from .layout_index import (
    LAYOUT_ARTIFACT_KEY_KEY,
    LAYOUT_ARTIFACT_URL_KEY,
    load_layout_index,
)
from .llm import (
    LLMError,
    OpenRouterBatchError,
    StructuredBatchRequest,
    StructuredBatchResult,
    call_structured,
    call_structured_batch,
    create_openrouter_http_client,
    delete_openrouter_batches,
    llm_enabled,
    llm_provider,
    llm_requests_per_minute,
    mask_llm_credential,
    openrouter_batch_enabled,
    openrouter_batch_max_requests,
    openrouter_batch_sync_fallback_enabled,
    scrutiny_model,
)
from .logging_utils import safe_url_for_log
from .process_file import FILE_DOWNLOAD_TIMEOUT_S, _require_pdf_bytes
from .remote_download import download_remote_pdf
from .reranking import rerank_chunks_for_defect
from .s3_artifacts import STEP_DEFECTS, upload_step_json
from .scrutiny.context_followup import (
    evidence_chunks_for_rerun,
    serve_context_followup,
)
from .scrutiny.prompts import (
    MAX_EVIDENCE_CHARS,
    build_defect_prompt,
    build_system_prompt,
)
from .scrutiny.rules import (
    Catalogue,
    Defect,
    check_id_sort_key,
    check_requirement,
    defects_for_filing_type,
    enabled_defect_ids,
    normalize_filing_type,
    normalize_special_category,
    refresh_catalogue,
    split_special_category_list,
)
from .scrutiny.schema import (
    Coverage,
    DefectFinding,
    DefectResponse,
    LlmUsage,
    ScrutinyReport,
    apply_catalogue_gap_policy,
    apply_conditional_manifest_policy,
    apply_deterministic_postclassification_policy,
    apply_element_status_contract_policy,
    apply_evidence_capability_policy,
    apply_evidence_pages,
    apply_final_evidence_contract_policy,
    apply_grounded_defect_policy,
    apply_internal_contradiction_policy,
    apply_margin_layout_policy,
    apply_presence_only_policy,
    apply_retrieval_policy,
    apply_review_reason_consistency_policy,
    apply_reviewed_document_policy,
    apply_scope_coverage_policy,
    apply_status_policy,
    apply_supported_evidence_policy,
    apply_undetermined_policy,
    apply_unreviewed_handwriting_policy,
    apply_visual_conflict_policy,
    apply_visual_field_policy,
    build_finding,
    failed_finding,
    manifest_absence_proven,
    missing_dependency_finding,
    missing_dependency_parts_in_layout,
    missing_document_finding,
    missing_parts_in_layout,
    summarize,
    summarize_usage,
)
from .vector_store import (
    VECTOR_SCHEMA_VERSION,
    gather_filing_evidence,
    pinecone_enabled,
    scrutiny_max_chunks,
    scrutiny_top_k,
)
from .visual.defect_scope import (
    plan_defect_images,
    visual_targets_for_plans,
    with_slot_files,
)
from .visual.detect import detect_visual_marks, visual_detection_enabled
from .visual.store import (
    VISUAL_ARTIFACT_KEY_KEY,
    VISUAL_ARTIFACT_URL_KEY,
    load_visual_index,
)

logger = logging.getLogger(__name__)

DEFAULT_CONCURRENCY = 8
DEFAULT_PERSIST_EVERY = 10
SCRUTINY_AFTER_INDEX_ERROR = "Unable to run scrutiny defects. Try after some time."
_VISUAL_TERMS = (
    "signature",
    "signed",
    "stamp",
    "seal",
    "notary",
    "handwritten",
    "blank date",
    "legible",
    "illegible",
    "dim",
    "blurred",
    "margin",
    "font size",
    "paper size",
    "spacing",
)

_AI_NOTE_SECTION = re.compile(
    r"(?ms)^(SCOPE|TEST|EVIDENCE|DECISION|GUARDRAILS|RECORD)\s*$\n"
    r"(.*?)(?=^[A-Z][A-Z ]+\s*$\n|\Z)"
)


def _defect_requires_visual_evidence(defect: Defect) -> bool:
    """Return whether the substantive rule needs rendered-page evidence.

    Catalogue notes share generic review wording containing terms such as
    ``illegible`` and ``visual inspection``. Scanning the whole note made
    text-only rules visual, increasing cost and allowing missing image coverage
    to block otherwise decisive text. Only the rule wording and substantive
    TEST, EVIDENCE and GUARDRAILS sections participate here.
    """
    sections = {
        name: body
        for name, body in _AI_NOTE_SECTION.findall(str(defect.ai_note or ""))
    }
    substantive_note = " ".join(
        sections.get(name, "") for name in ("TEST", "EVIDENCE", "GUARDRAILS")
    )
    wording = (
        f"{defect.defect} {check_requirement(defect)} {substantive_note}"
    ).casefold()
    return any(term in wording for term in _VISUAL_TERMS)


def _usable_visual_detection(detected: dict[str, Any] | None) -> bool:
    """Accept completed visual evidence even when no ink mark was detected."""
    if not isinstance(detected, dict) or detected.get("status") != "ok":
        return False
    return any(
        bool(detected.get(key))
        for key in (
            "pages",
            "targets",
            "marks",
            "field_assessments",
            "quality_assessments",
        )
    )


def _visual_evidence_complete(
    check_id: str, visual_index: dict[str, Any] | None
) -> bool:
    """True only when every planned page for this check was rendered."""
    if not isinstance(visual_index, dict) or visual_index.get("status") != "ok":
        return False
    if check_id in set(visual_index.get("incomplete_check_ids") or []):
        return False
    target_pages = {
        int(target["page"])
        for target in visual_index.get("targets") or []
        if isinstance(target, dict) and target.get("page") is not None
    }
    reviewed_pages = {int(page) for page in visual_index.get("pages") or []}
    return bool(
        target_pages
        and target_pages.issubset(reviewed_pages)
        and not visual_index.get("failures")
    )


@dataclass
class EvidenceWorkspace:
    """Immutable filing evidence compiled once and shared by every check."""

    layout: dict[int, dict[str, Any]]
    pages_by_part: dict[str, list[int]] = field(default_factory=dict)
    _complete_cache: dict[tuple[str, ...], list[dict[str, Any]]] = field(
        default_factory=dict
    )

    @classmethod
    def from_layout(cls, layout: dict[int, dict[str, Any]] | None) -> EvidenceWorkspace:
        normalized = layout or {}
        pages_by_part: dict[str, list[int]] = {}
        for page_number, page in sorted(normalized.items()):
            if not isinstance(page, dict):
                continue
            for part in _layout_document_parts(page.get("document_part")):
                pages_by_part.setdefault(part, []).append(int(page_number))
        return cls(layout=normalized, pages_by_part=pages_by_part)

    def complete_chunks(self, defect: Defect) -> list[dict[str, Any]]:
        parts = tuple(
            dict.fromkeys(
                [
                    *required_parts_for_defect(defect),
                    *(getattr(defect, "context_parts", None) or []),
                ]
            )
        )
        if not parts:
            return []
        key = tuple(sorted(parts, key=str.casefold))
        cached = self._complete_cache.get(key)
        if cached is not None:
            return [dict(chunk) for chunk in cached]
        chunks: list[dict[str, Any]] = []
        for page_number, page in sorted(self.layout.items()):
            if not isinstance(page, dict):
                continue
            names = _layout_document_parts(page.get("document_part"))
            if not names or not any(
                chunks_cover_part(
                    [{"document_part": names, "chunk_kind": "page"}], part
                )
                for part in parts
            ):
                continue
            text = _layout_page_text(page)
            if text:
                chunks.append(
                    {
                        "record_id": f"layout-complete:{page_number}",
                        "score": 0.0,
                        "chunk_kind": "page",
                        "page": page_number,
                        "page_end": page_number,
                        "document_part": names,
                        "text": text,
                        "layout_fallback": True,
                        "complete_part_retry": True,
                    }
                )
        self._complete_cache[key] = [dict(chunk) for chunk in chunks]
        return chunks


class DefectBatchEnvelope(BaseModel):
    """One independently keyed result for every check in a micro-batch."""

    findings: list[DefectResponse]


@dataclass
class _PendingEvaluation:
    defect: Defect
    system_prompt: str
    user_prompt: str
    model: str
    future: asyncio.Future[tuple[DefectResponse, LlmUsage]]


class ScrutinyMicrobatcher:
    """Coalesce compatible synchronous checks without changing their outputs."""

    def __init__(self, client: httpx.AsyncClient | None) -> None:
        mode = (os.getenv("SCRUTINY_EXECUTION_MODE") or "fast").strip().casefold()
        configured = int(os.getenv("SCRUTINY_MICROBATCH_SIZE", "3") or "3")
        self.size = 1 if mode == "legacy" else max(1, min(configured, 5))
        self.window_seconds = max(
            0.0,
            min(
                0.5,
                float(os.getenv("SCRUTINY_MICROBATCH_WINDOW_MS", "75") or "75")
                / 1000.0,
            ),
        )
        self.max_chars = max(
            10_000, int(os.getenv("SCRUTINY_MICROBATCH_MAX_CHARS", "90000"))
        )
        self.client = client
        self._lock = asyncio.Lock()
        self._queues: dict[tuple[str, ...], list[_PendingEvaluation]] = {}
        self._flush_tasks: set[asyncio.Task[None]] = set()

    @staticmethod
    def _key(defect: Defect, model: str) -> tuple[str, ...]:
        parts = [
            *required_parts_for_defect(defect),
            *(getattr(defect, "context_parts", None) or []),
        ]
        visual = "visual" if _defect_requires_visual_evidence(defect) else "text"
        return (model, visual, *sorted(set(parts), key=str.casefold))

    async def evaluate(
        self,
        *,
        defect: Defect,
        system_prompt: str,
        user_prompt: str,
        model: str,
    ) -> tuple[DefectResponse, LlmUsage]:
        if self.size <= 1:
            return await call_structured(
                system_prompt=system_prompt,
                user_prompt=user_prompt,
                response_model=DefectResponse,
                model=model,
                client=self.client,
                request_label=f"scrutiny:{defect.check_id}",
            )
        loop = asyncio.get_running_loop()
        future: asyncio.Future[tuple[DefectResponse, LlmUsage]] = loop.create_future()
        pending = _PendingEvaluation(
            defect=defect,
            system_prompt=system_prompt,
            user_prompt=user_prompt,
            model=model,
            future=future,
        )
        key = self._key(defect, model)
        flush_now: list[_PendingEvaluation] | None = None
        async with self._lock:
            queue = self._queues.setdefault(key, [])
            queue.append(pending)
            if len(queue) >= self.size:
                flush_now = queue[: self.size]
                del queue[: self.size]
            elif len(queue) == 1:
                task = asyncio.create_task(self._delayed_flush(key))
                self._flush_tasks.add(task)
                task.add_done_callback(self._flush_tasks.discard)
        if flush_now:
            await self._flush(flush_now)
        return await future

    async def _delayed_flush(self, key: tuple[str, ...]) -> None:
        await asyncio.sleep(self.window_seconds)
        async with self._lock:
            pending = self._queues.pop(key, [])
        if pending:
            await self._flush(pending)

    @staticmethod
    def _usage_shares(usage: LlmUsage, count: int) -> list[LlmUsage]:
        count = max(1, count)
        fields = (
            "prompt_tokens",
            "completion_tokens",
            "total_tokens",
            "cached_tokens",
            "reasoning_tokens",
        )
        shares = [LlmUsage(model=usage.model) for _ in range(count)]
        for name in fields:
            total = int(getattr(usage, name) or 0)
            quotient, remainder = divmod(total, count)
            for index, share in enumerate(shares):
                setattr(share, name, quotient + (1 if index < remainder else 0))
        if usage.cost_usd is not None:
            for share in shares:
                share.cost_usd = usage.cost_usd / count
        shares[0].calls = usage.calls
        shares[0].generation_id = usage.generation_id
        shares[0].generation_ids = list(usage.generation_ids or [])
        return shares

    async def _flush(self, pending: list[_PendingEvaluation]) -> None:
        if not pending:
            return
        if (
            len(pending) == 1
            or sum(len(item.user_prompt) for item in pending) > self.max_chars
        ):
            await asyncio.gather(*(self._single(item) for item in pending))
            return
        system_prompt = pending[0].system_prompt + (
            "\n\nEvaluate every numbered check independently. Return one finding "
            "for each supplied check_id, in the same order, inside the findings array."
        )
        user_prompt = "\n\n".join(
            f"# Independent check {index}: {item.defect.check_id}\n{item.user_prompt}"
            for index, item in enumerate(pending, start=1)
        )
        try:
            envelope, usage = await call_structured(
                system_prompt=system_prompt,
                user_prompt=user_prompt,
                response_model=DefectBatchEnvelope,
                model=pending[0].model,
                client=self.client,
                request_label="scrutiny:microbatch:"
                + ",".join(item.defect.check_id for item in pending),
            )
            by_id = {finding.check_id: finding for finding in envelope.findings}
            if len(envelope.findings) != len(pending) or set(by_id) != {
                item.defect.check_id for item in pending
            }:
                raise ValueError("micro-batch response check_ids did not match request")
            shares = self._usage_shares(usage, len(pending))
            for item, share in zip(pending, shares, strict=True):
                if not item.future.done():
                    item.future.set_result((by_id[item.defect.check_id], share))
        except Exception:
            logger.warning(
                "Micro-batch failed; retrying checks individually", exc_info=True
            )
            await asyncio.gather(*(self._single(item) for item in pending))

    async def _single(self, item: _PendingEvaluation) -> None:
        try:
            result = await call_structured(
                system_prompt=item.system_prompt,
                user_prompt=item.user_prompt,
                response_model=DefectResponse,
                model=item.model,
                client=self.client,
                request_label=f"scrutiny:{item.defect.check_id}",
            )
            if not item.future.done():
                item.future.set_result(result)
        except Exception as exc:
            if not item.future.done():
                item.future.set_exception(exc)

    async def close(self) -> None:
        async with self._lock:
            queued = [item for queue in self._queues.values() for item in queue]
            self._queues.clear()
        if queued:
            await self._flush(queued)
        if self._flush_tasks:
            await asyncio.gather(*list(self._flush_tasks), return_exceptions=True)


class ScrutinyAfterIndexError(Exception):
    """Pinecone index finished; nested defect checks could not run."""

    def __init__(self, agent_data_id: str) -> None:
        super().__init__(SCRUTINY_AFTER_INDEX_ERROR)
        self.agent_data_id = agent_data_id


def require_successful_checks(findings: list[DefectFinding], planned: int) -> None:
    """Fail only when no check produced a usable result.

    One model failure stays on that finding and the other findings are
    returned. Every check failing means the model produced nothing to send.
    """
    usable = sum(not finding.error for finding in findings)
    if usable:
        return
    failed = sum(bool(finding.error) for finding in findings)
    raise RuntimeError(
        f"Scrutiny incomplete: {failed} of {planned} checks failed; rerun scrutiny"
    )


class ScrutinyEvent(StartEvent):
    agent_data_id: str | None = None
    file_hash: str | None = None
    file_url: str | None = None
    organization_id: str | None = None
    workspace_id: str | None = None
    special_category: str | None = None
    special_categories: list[str] | None = None
    filing_type: str | None = None
    court: str | None = None
    model_tier: str | None = None
    version: int | None = None


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
    openrouter_batch_ids: list[str] = Field(default_factory=list)


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
    data = await download_remote_pdf(url, timeout_seconds=FILE_DOWNLOAD_TIMEOUT_S)
    if not data:
        raise ValueError(f"Downloaded empty body from {url}")
    _require_pdf_bytes(data, url)
    return hashlib.sha256(data).hexdigest()


def _sanitize_response(
    defect: Defect,
    response: DefectResponse,
    chunks: list[dict[str, Any]],
    visual_index: dict[str, Any] | None = None,
    layout: dict[int, dict[str, Any]] | None = None,
    *,
    enforce_retrieval: bool = True,
) -> DefectResponse:
    """Keep the catalogue check_id and drop fixes unless a defect was found."""
    if response.check_id != defect.check_id:
        logger.warning(
            "[Scrutiny] Model returned check_id %s for %s; correcting",
            response.check_id,
            defect.check_id,
        )
    response.check_id = defect.check_id
    response = apply_catalogue_gap_policy(response)
    response = apply_evidence_pages(response, chunks, defect)
    response = apply_presence_only_policy(defect, response, chunks, layout)
    response = apply_status_policy(response)
    response = apply_review_reason_consistency_policy(response)
    response = apply_undetermined_policy(defect, response, chunks, layout)
    if enforce_retrieval:
        response = apply_retrieval_policy(defect, response, chunks, layout)
    response = apply_reviewed_document_policy(defect, response, chunks)
    response = apply_margin_layout_policy(defect, response, layout)
    response = apply_supported_evidence_policy(defect, response)
    response = apply_grounded_defect_policy(response)
    response = apply_internal_contradiction_policy(defect, response)
    response = apply_visual_conflict_policy(defect, response, visual_index)
    response = apply_visual_field_policy(defect, response, visual_index)
    response = apply_evidence_capability_policy(defect, response, visual_index)
    response = apply_unreviewed_handwriting_policy(response, visual_index)
    return response


def _retrieval_evidence_complete(
    defect: Defect,
    chunks: list[dict[str, Any]],
    layout: dict[int, dict[str, Any]] | None,
) -> bool:
    """Do not equate a successful semantic sample with exhaustive inspection."""

    if not layout:
        return False

    strategy = retrieval_plan_for_defect(defect).strategy

    reviewed = {int(chunk["page"]) for chunk in chunks if chunk.get("page") is not None}
    for part in required_parts_for_defect(defect):
        part_pages = [
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
                part,
            )
        ]
        if not part_pages:
            # The complete layout proves this optional inspection target is
            # absent. Dependency and presence-owner outcomes are handled
            # before evaluation; absence here is not a retrieval failure.
            continue
        complete_retry = any(chunk.get("complete_part_retry") for chunk in chunks)
        if complete_retry or strategy == "exhaustive_part":
            required_pages = set(part_pages)
        elif strategy == "boundary_and_neighbors":
            required_pages = {part_pages[0], part_pages[-1]}
        else:
            # A semantic hit proves that a document was found, not that the
            # exact clause or field needed for this check was inspected.
            return False
        if not required_pages.issubset(reviewed):
            return False
    return True


def _prompt_evidence_truncated(chunks: list[dict[str, Any]]) -> bool:
    """Conservative preflight for the prompt's evidence-character ceiling."""
    used = 0
    for chunk in chunks:
        text = str(chunk.get("text") or "")
        used += len(text) + 300
        if used > MAX_EVIDENCE_CHARS:
            return True
    return False


def _complete_evidence_batches(
    chunks: list[dict[str, Any]],
) -> list[list[dict[str, Any]]]:
    """Partition complete evidence without dropping a parsed page.

    Summary chunks are repeated as context. Oversized single pages are split
    into page-preserving slices so every character reaches an evaluator.
    """
    summaries = [
        dict(chunk) for chunk in chunks if chunk.get("chunk_kind") == "summary"
    ]
    evidence: list[dict[str, Any]] = []
    slice_limit = max(4_000, MAX_EVIDENCE_CHARS - 1_000)
    for chunk in chunks:
        if chunk.get("chunk_kind") == "summary":
            continue
        text = str(chunk.get("text") or "")
        if len(text) <= slice_limit:
            evidence.append(dict(chunk))
            continue
        for offset in range(0, len(text), slice_limit):
            piece = dict(chunk)
            piece["text"] = text[offset : offset + slice_limit]
            piece["complete_page_slice"] = offset // slice_limit + 1
            evidence.append(piece)
    batches: list[list[dict[str, Any]]] = []
    current = [dict(item) for item in summaries]
    used = sum(len(str(item.get("text") or "")) + 300 for item in current)
    for chunk in evidence:
        cost = len(str(chunk.get("text") or "")) + 300
        if len(current) > len(summaries) and used + cost > MAX_EVIDENCE_CHARS:
            batches.append(current)
            current = [dict(item) for item in summaries]
            used = sum(len(str(item.get("text") or "")) + 300 for item in current)
        current.append(chunk)
        used += cost
    if len(current) > len(summaries) or not batches:
        batches.append(current)
    return batches


def _aggregate_complete_batch_responses(
    responses: list[DefectResponse],
) -> DefectResponse:
    """Aggregate exhaustive batches without treating a partial view as global.

    A batch can positively prove a local blank, mismatch, or other defect, but
    it cannot prove a collection-wide absence.  Mixed outcomes therefore need
    one compact adjudication over *all* batch findings.  The caller performs
    that adjudication; this helper returns ``needs_review`` as the safe interim
    state instead of publishing whichever batch happened to be most confident.
    """
    defects = [item for item in responses if item.status == "defect_found"]
    if defects and len(defects) == len(responses):
        return max(defects, key=lambda item: item.confidence)
    if responses and all(item.status == "compliant" for item in responses):
        chosen = max(responses, key=lambda item: item.confidence)
        chosen.summary = "Every complete-evidence batch satisfied the requirement."
        return chosen
    if responses and all(item.status == "not_applicable" for item in responses):
        return max(responses, key=lambda item: item.confidence)
    chosen = max(responses, key=lambda item: item.confidence)
    chosen.status = "needs_review"
    chosen.gap = chosen.gap or "ambiguous"
    chosen.gap_detail = chosen.gap_detail or (
        "The complete evidence batches produced different filing outcomes and "
        "must be reconciled against their combined evidence."
    )
    return chosen


def _complete_batch_outcomes_conflict(responses: list[DefectResponse]) -> bool:
    """True when page-preserving batches need a combined-evidence decision."""
    return len({item.status for item in responses}) > 1


def _complete_batch_adjudication_prompt(
    defect: Defect,
    responses: list[DefectResponse],
) -> str:
    """Build a compact second-pass prompt from every batch's grounded result."""
    findings = "\n\n".join(
        f"BATCH {number}\n{response.model_dump_json(exclude_none=True)}"
        for number, response in enumerate(responses, start=1)
    )
    return (
        f"Reconcile the complete-evidence batch findings for {defect.check_id}.\n"
        f"Requirement: {defect.requirement}\n"
        f"Operational note:\n{defect.ai_note or ''}\n\n"
        "The batches together cover the complete relevant evidence scope. Treat "
        "their evidence references and element results as one filing. A statement "
        "that an item is absent from one batch does not prove it is absent from the "
        "filing when another batch shows it. A visible blank, mismatch, prohibited "
        "content, or other positive non-compliance remains a Defect. Return "
        "Compliant only when the combined findings establish every applicable "
        "element. Return Needs Review only for a genuine unreadable region or a "
        "material conflict that the combined findings cannot resolve. Do not use "
        "confidence as evidence and do not invent pages or text. Preserve supporting "
        "evidence references from the batch findings.\n\n"
        f"{findings}"
    )


def _absence_owner_check_ids(
    defects: list[Defect],
    layout: dict[int, dict[str, Any]] | None,
) -> dict[str, list[str]]:
    """Choose exactly one catalogue row to report each missing document.

    Only rows explicitly marked ``manifest_absence_proves_defect`` may own a
    missing document.  Wording is deliberately not used as a fallback: phrases
    such as "not filed" commonly occur in conditional checks (caveats,
    substitution applications, powers of attorney, etc.) and choosing one of
    those rows would manufacture a defect whose prerequisites were never
    established.

    The policy is catalogue-driven rather than a hard-coded check-id map.  If a
    document has no applicable, explicitly marked presence row, all checks
    which depend on that absent document are skipped as not applicable.  This
    is safer than relabelling an arbitrary content check as a document-presence
    defect.
    """
    candidates: dict[str, list[Defect]] = {}
    for defect in defects:
        if not defect.manifest_absence_proves_defect:
            continue
        missing = missing_parts_in_layout(defect, layout)
        if not missing:
            continue
        owned_parts = required_parts_for_defect(defect)
        for part in missing:
            if part not in owned_parts:
                continue
            candidates.setdefault(part, []).append(defect)

    owners: dict[str, list[str]] = {}
    for part, options in candidates.items():
        # Prefer the narrowest explicit row when two applicable catalogue rows
        # own the same part; the stable check-id tie-break keeps runs
        # reproducible.  Both candidates have already opted into deterministic
        # manifest-absence handling.
        owner = min(
            options,
            key=lambda item: (
                # Prefer the filing-specific presence check over a broad
                # catalogue row.  For example, D-181 owns a missing
                # supporting Affidavit in SLP (Civil), while D-64 remains the
                # fallback for filing types without a narrower check.
                len(getattr(item, "agent_filing_types", []) or []),
                len(required_parts_for_defect(item)),
                check_id_sort_key(item.check_id),
            ),
        )
        owners.setdefault(owner.check_id, []).append(part)
    return owners


def _visual_index_for_defect(
    defect: Defect,
    visual_index: dict[str, Any] | None,
) -> dict[str, Any] | None:
    """Restrict visual evidence to pages belonging to this check's documents."""
    if not isinstance(visual_index, dict):
        return visual_index
    required = required_parts_for_defect(defect)
    if not required:
        return dict(visual_index)

    targets: list[dict[str, Any]] = []
    for target in visual_index.get("targets") or []:
        if not isinstance(target, dict):
            continue
        names = target.get("document_types") or target.get("document_parts") or []
        probe = {"document_part": names, "chunk_kind": "page"}
        if any(chunks_cover_part([probe], part) for part in required):
            targets.append(target)
    target_pages = {
        int(target["page"]) for target in targets if target.get("page") is not None
    }
    filtered = dict(visual_index)
    filtered["targets"] = targets
    filtered["pages"] = [
        int(page)
        for page in visual_index.get("pages") or []
        if int(page) in target_pages
    ]
    filtered["marks"] = [
        mark
        for mark in visual_index.get("marks") or []
        if isinstance(mark, dict) and int(mark.get("page") or -1) in target_pages
    ]
    filtered["field_assessments"] = [
        item
        for item in visual_index.get("field_assessments") or []
        if isinstance(item, dict) and int(item.get("page") or -1) in target_pages
    ]
    filtered["failures"] = [
        failure
        for failure in visual_index.get("failures") or []
        if not isinstance(failure, dict)
        or failure.get("page") is None
        or int(failure.get("page")) in target_pages
    ]
    return filtered


def _complete_layout_chunks_for_defect(
    defect: Defect,
    chunks: list[dict[str, Any]],
    layout: dict[int, dict[str, Any]] | None,
    evidence_workspace: EvidenceWorkspace | None = None,
) -> list[dict[str, Any]]:
    """Return every parsed page belonging to this check's required parts.

    This is the automatic second-pass evidence set.  It is intentionally built
    from the immutable page manifest rather than another semantic search, so a
    paragraph, blank field, signature block, or application cannot remain
    "not in excerpts" when its document was successfully parsed.
    """
    if not layout:
        return chunks
    if evidence_workspace is not None:
        complete = evidence_workspace.complete_chunks(defect)
        if complete:
            summary = [
                chunk for chunk in chunks if chunk.get("chunk_kind") == "summary"
            ][:1]
            return summary + complete
    required = required_parts_for_defect(defect)
    evidence_parts = list(
        dict.fromkeys([*required, *(getattr(defect, "context_parts", None) or [])])
    )
    if not evidence_parts:
        return chunks

    summary = [chunk for chunk in chunks if chunk.get("chunk_kind") == "summary"][:1]
    complete: list[dict[str, Any]] = []
    for page_number, page in sorted(layout.items()):
        if not isinstance(page, dict):
            continue
        names = _layout_document_parts(page.get("document_part"))
        if not names or not any(
            chunks_cover_part([{"document_part": names, "chunk_kind": "page"}], part)
            for part in evidence_parts
        ):
            continue
        text = _layout_page_text(page)
        if not text:
            continue
        stored = next(
            (
                chunk
                for chunk in chunks
                if chunk.get("chunk_kind") != "summary"
                and chunk.get("page") == page_number
                and chunk.get("record_id")
            ),
            None,
        )
        # Keep the Pinecone id when this page was already retrieved. The
        # follow-up addresses the next stored chunk from that id.
        record_id = (
            str(stored.get("record_id"))
            if stored is not None
            else f"layout-complete:{page_number}"
        )
        raw_index = stored.get("chunk_index") if stored is not None else 0
        try:
            chunk_index = int(raw_index) if raw_index is not None else 0
        except (TypeError, ValueError):
            chunk_index = 0
        complete.append(
            {
                "record_id": record_id,
                "score": 0.0,
                "chunk_kind": "page",
                "page": page_number,
                "page_end": page_number,
                "document_part": names,
                "text": text,
                "chunk_index": chunk_index,
                "layout_fallback": True,
                "complete_part_retry": True,
            }
        )
    return summary + complete if complete else chunks


@dataclass
class _PreparedDefectEvaluation:
    defect: Defect
    catalogue: Catalogue
    record: dict[str, Any] | None
    chunks: list[dict[str, Any]]
    layout: dict[int, dict[str, Any]] | None
    check_visual_index: dict[str, Any] | None
    model: str
    coverage: Coverage
    system_prompt: str
    user_prompt: str
    vision_note: str | None = None
    llm_client: httpx.AsyncClient | None = None
    evidence_workspace: EvidenceWorkspace | None = None
    file_hash: str | None = None
    organization_id: str | None = None
    workspace_id: str | None = None
    index_generation: str | None = None
    use_pinecone: bool = False


def _prepare_defect_evaluation(
    defect: Defect,
    *,
    catalogue: Catalogue,
    record: dict[str, Any] | None,
    chunks: list[dict[str, Any]],
    filing_type: str | None,
    layout: dict[int, dict[str, Any]] | None = None,
    visual_index: dict[str, Any] | None = None,
    absence_defect_parts: list[str] | None = None,
    model: str,
    llm_client: httpx.AsyncClient | None = None,
    evidence_workspace: EvidenceWorkspace | None = None,
    file_hash: str | None = None,
    organization_id: str | None = None,
    workspace_id: str | None = None,
    index_generation: str | None = None,
    use_pinecone: bool = False,
    vision_note: str | None = None,
) -> DefectFinding | _PreparedDefectEvaluation:
    # Legal-source fields are catalogue-governance metadata.  They must not
    # short-circuit retrieval or replace the filing outcome.  The operational
    # requirement, applicability, manifest and filing evidence determine the
    # result; source verification can be audited separately.
    presence_missing = missing_parts_in_layout(defect, layout)
    dependency_missing = missing_dependency_parts_in_layout(defect, layout)
    required_now = required_parts_for_defect(defect)
    certificate_check = (
        "AOR's Certificate" in required_now
        and "Advocate's Checklist" not in required_now
    )
    # A certificate printed at the end of the Main Petition is still the
    # certificate. A checklist signature on a different document is not.
    if (
        certificate_check
        and presence_missing
        and "AOR's Certificate" in presence_missing
        and _layout_contains_aor_certificate(layout)
    ):
        presence_missing = [
            part for part in presence_missing if part != "AOR's Certificate"
        ]
    if (
        getattr(defect, "manifest_absence_proves_defect", False)
        and certificate_check
        and layout
        and not _layout_contains_aor_certificate(layout)
        and not _layout_has_blank_unlabeled_page(layout)
    ):
        return missing_document_finding(
            defect,
            ["AOR's Certificate"],
            visual_index=visual_index,
            record=record if isinstance(record, dict) else None,
        )
    if presence_missing and absence_defect_parts:
        return missing_document_finding(
            defect,
            [part for part in presence_missing if part in absence_defect_parts],
            visual_index=visual_index,
            record=record if isinstance(record, dict) else None,
        )
    if dependency_missing:
        return missing_dependency_finding(
            defect,
            dependency_missing,
            record=record if isinstance(record, dict) else None,
        )
    if manifest_absence_proven(defect, layout) and not (
        certificate_check and _layout_contains_aor_certificate(layout)
    ):
        return missing_document_finding(
            defect,
            presence_missing or required_parts_for_defect(defect),
            visual_index=visual_index,
            record=record if isinstance(record, dict) else None,
        )

    # Complete the required evidence set before the first evaluation. Semantic
    # retrieval is only a locator; it must not decide which pages the model is
    # allowed to inspect. Oversized complete sets are handled by the existing
    # page-preserving batch retry below.
    if layout and not _retrieval_evidence_complete(defect, chunks, layout):
        complete_chunks = _complete_layout_chunks_for_defect(
            defect, chunks, layout, evidence_workspace
        )
        if complete_chunks:
            chunks = complete_chunks

    check_visual_index = _visual_index_for_defect(defect, visual_index)
    # The model, the follow-up, and the full-part retry all see the same
    # documents: the inspect parts and any context parts. A neighbouring
    # document cannot decide the row.
    chunks = evidence_chunks_for_rerun(defect, chunks)

    pages = sorted({c["page"] for c in chunks if c.get("page") is not None})
    visual_required = _defect_requires_visual_evidence(defect)
    visual_pages = (
        [int(page) for page in check_visual_index.get("pages") or []]
        if isinstance(check_visual_index, dict)
        else []
    )
    visual_complete = bool(
        visual_required
        and _visual_evidence_complete(defect.check_id, check_visual_index)
    )
    document_found = bool(chunks) and not missing_required_parts(defect, chunks)
    prompt_truncated = _prompt_evidence_truncated(chunks)
    scope_complete = (
        _retrieval_evidence_complete(defect, chunks, layout) and not prompt_truncated
    )
    coverage = Coverage(
        chunks_reviewed=len(chunks),
        pages_reviewed=pages,
        structured_record_available=bool(record),
        document_found=document_found,
        required_scope_found=scope_complete,
        scope_complete=scope_complete,
        prompt_truncated=prompt_truncated,
        evidence_complete=scope_complete,
        visual_required=visual_required,
        visual_complete=visual_complete,
        visual_pages_reviewed=visual_pages,
    )

    system_prompt = build_system_prompt(catalogue, filing_type)
    user_prompt = build_defect_prompt(
        defect,
        record=slice_record_for_defect(record, defect),
        chunks=chunks,
        catalogue=catalogue,
        visual_index=check_visual_index,
        layout=layout,
        vision_note=vision_note,
    )
    return _PreparedDefectEvaluation(
        defect=defect,
        catalogue=catalogue,
        record=record,
        chunks=chunks,
        layout=layout,
        check_visual_index=check_visual_index,
        model=model,
        coverage=coverage,
        system_prompt=system_prompt,
        user_prompt=user_prompt,
        vision_note=vision_note,
        llm_client=llm_client,
        evidence_workspace=evidence_workspace,
        file_hash=file_hash,
        organization_id=organization_id,
        workspace_id=workspace_id,
        index_generation=index_generation,
        use_pinecone=use_pinecone,
    )


async def _finalize_defect_evaluation(
    prepared: _PreparedDefectEvaluation,
    raw: DefectResponse,
    usage: LlmUsage,
) -> DefectFinding:
    """Apply post-processing and any evidence follow-up to a model result."""
    defect = prepared.defect
    catalogue = prepared.catalogue
    record = prepared.record
    chunks = prepared.chunks
    layout = prepared.layout
    check_visual_index = prepared.check_visual_index
    model = prepared.model
    coverage = prepared.coverage
    system_prompt = prepared.system_prompt
    vision_note = prepared.vision_note
    llm_client = prepared.llm_client
    evidence_workspace = prepared.evidence_workspace
    scope_complete = coverage.scope_complete
    prompt_truncated = coverage.prompt_truncated
    file_hash = prepared.file_hash
    organization_id = prepared.organization_id
    workspace_id = prepared.workspace_id
    index_generation = prepared.index_generation
    use_pinecone = prepared.use_pinecone
    model_status = raw.status
    prior_decision = {
        "status": raw.status,
        "confidence": raw.confidence,
        "gap": raw.gap,
        "gap_detail": raw.gap_detail,
        "summary": raw.summary,
        "observed": raw.observed,
        "conclusion": raw.conclusion,
    }
    response = _sanitize_response(defect, raw, chunks, check_visual_index, layout)
    response = apply_scope_coverage_policy(
        response,
        scope_complete=scope_complete,
        prompt_truncated=prompt_truncated,
    )

    # At most one extra model call, owned by the context-followup tools.
    # A confident or decided first answer is final. The follow-up sees only
    # chunks those tools fetched, and it does not run with the full-part retry.
    followed_up = False
    complete_document_retried = False
    followup = await serve_context_followup(
        defect=defect,
        catalogue=catalogue,
        record=record,
        chunks=chunks,
        layout=layout,
        visual_index=check_visual_index,
        model=model,
        system_prompt=system_prompt,
        model_status=model_status,
        response=response,
        prior_decision=prior_decision,
        prior_generation_id=usage.generation_id,
        file_hash=file_hash,
        organization_id=organization_id,
        workspace_id=workspace_id,
        index_generation=index_generation,
        use_pinecone=use_pinecone,
    )
    if followup is not None:
        added = followup.added
        # Same inspect-part rule as the first call. Pages from another
        # document stay out of the second answer.
        chunks = [*evidence_chunks_for_rerun(defect, chunks), *added]
        usage = usage.plus(followup.usage)
        response = _sanitize_response(
            defect, followup.response, chunks, check_visual_index, layout
        )
        follow_complete = _retrieval_evidence_complete(defect, chunks, layout)
        # Same scope rule as the first call. One or two extra chunks of the
        # right document can support a grounded defect. They do not make the
        # inspection complete, so they cannot publish compliant.
        response = apply_scope_coverage_policy(
            response,
            scope_complete=follow_complete,
            prompt_truncated=False,
        )
        pages = sorted({c["page"] for c in chunks if c.get("page") is not None})
        added_pages = sorted(
            {int(chunk["page"]) for chunk in added if chunk.get("page") is not None}
        )
        coverage = coverage.model_copy(
            update={
                "chunks_reviewed": len(chunks),
                "pages_reviewed": pages,
                "document_found": bool(chunks)
                and not missing_required_parts(defect, chunks),
                "required_scope_found": follow_complete,
                "scope_complete": follow_complete,
                "prompt_truncated": False,
                "evidence_complete": follow_complete,
                "context_followup": True,
                "context_followup_action": followup.action,
                "context_followup_pages": added_pages,
                "context_followup_query": followup.query,
            }
        )
        followed_up = True

    # A needs-review result caused by sampled evidence is an internal retry
    # signal, not a final user journey.  Retry once with every parsed page of
    # the required document parts.  Genuine unreadable/contradictory evidence
    # remains unresolved; missing semantic excerpts do not.
    if (
        not followed_up
        and response.status == "needs_review"
        and response.gap
        in {
            "part_not_in_excerpts",
            "element_page_not_in_excerpts",
            "unreadable",
            "ambiguous",
        }
        and layout
    ):
        complete_chunks = _complete_layout_chunks_for_defect(
            defect, chunks, layout, evidence_workspace
        )
        complete_pages = {
            int(chunk["page"])
            for chunk in complete_chunks
            if chunk.get("page") is not None
        }
        original_signature = {
            (str(chunk.get("record_id") or ""), str(chunk.get("text") or ""))
            for chunk in chunks
            if chunk.get("chunk_kind") != "summary"
        }
        complete_signature = {
            (str(chunk.get("record_id") or ""), str(chunk.get("text") or ""))
            for chunk in complete_chunks
            if chunk.get("chunk_kind") != "summary"
        }
        if complete_signature and complete_signature != original_signature:
            retry_usage = LlmUsage()
            retry_responses: list[DefectResponse] = []
            complete_batches = _complete_evidence_batches(complete_chunks)
            retry_concurrency = max(
                1, int(os.getenv("SCRUTINY_EVIDENCE_RETRY_CONCURRENCY", "3"))
            )
            retry_semaphore = asyncio.Semaphore(retry_concurrency)

            async def evaluate_complete_batch(
                batch_number: int, batch: list[dict[str, Any]]
            ) -> tuple[DefectResponse, LlmUsage]:
                retry_prompt = build_defect_prompt(
                    defect,
                    record=slice_record_for_defect(record, defect),
                    chunks=batch,
                    catalogue=catalogue,
                    visual_index=check_visual_index,
                    layout=layout,
                    vision_note=vision_note,
                )
                async with retry_semaphore:
                    retry_raw, batch_usage = await call_structured(
                        system_prompt=system_prompt,
                        user_prompt=retry_prompt,
                        response_model=DefectResponse,
                        model=model,
                        client=llm_client,
                        request_label=(
                            f"scrutiny:{defect.check_id}:complete-evidence-retry:"
                            f"{batch_number}-of-{len(complete_batches)}"
                        ),
                    )
                return (
                    _sanitize_response(
                        defect,
                        retry_raw,
                        batch,
                        check_visual_index,
                        layout,
                        enforce_retrieval=False,
                    ),
                    batch_usage,
                )

            retry_results = await asyncio.gather(
                *(
                    evaluate_complete_batch(number, batch)
                    for number, batch in enumerate(complete_batches, start=1)
                )
            )
            for retry_response, batch_usage in retry_results:
                retry_responses.append(retry_response)
                retry_usage = retry_usage.plus(batch_usage)
            complete_document_retried = bool(retry_responses)
            if _complete_batch_outcomes_conflict(retry_responses):
                adjudicated, adjudication_usage = await call_structured(
                    system_prompt=system_prompt,
                    user_prompt=_complete_batch_adjudication_prompt(
                        defect, retry_responses
                    ),
                    response_model=DefectResponse,
                    model=model,
                    client=llm_client,
                    request_label=(
                        f"scrutiny:{defect.check_id}:complete-evidence-adjudication"
                    ),
                )
                retry_usage = retry_usage.plus(adjudication_usage)
                response = _sanitize_response(
                    defect,
                    adjudicated,
                    complete_chunks,
                    check_visual_index,
                    layout,
                    enforce_retrieval=False,
                )
            else:
                response = _aggregate_complete_batch_responses(retry_responses)
            chunks = complete_chunks
            usage = usage.plus(retry_usage)
            pages = sorted(complete_pages)
            retry_truncated = False
            retry_complete = _retrieval_evidence_complete(defect, chunks, layout)
            # Each batch skipped the retrieval check because one batch may not
            # hold every inspect part. The combined pages are checked the same
            # way as the first call.
            response = apply_retrieval_policy(defect, response, chunks)
            response = apply_reviewed_document_policy(defect, response, chunks)
            response = apply_scope_coverage_policy(
                response,
                scope_complete=retry_complete,
                prompt_truncated=retry_truncated,
            )
            coverage = coverage.model_copy(
                update={
                    "chunks_reviewed": len(chunks),
                    "pages_reviewed": pages,
                    "document_found": bool(chunks)
                    and not missing_required_parts(defect, chunks),
                    "required_scope_found": retry_complete,
                    "scope_complete": retry_complete,
                    "prompt_truncated": retry_truncated,
                    "evidence_complete": retry_complete,
                }
            )

    if response.status != "needs_review" or response.gap not in {
        "part_not_in_excerpts",
        "element_page_not_in_excerpts",
    }:
        coverage = coverage.model_copy(update={"required_scope_found": True})

    response = apply_deterministic_postclassification_policy(
        defect,
        response,
        chunks,
        check_visual_index,
        scope_complete=coverage.scope_complete,
    )
    response = apply_conditional_manifest_policy(defect, response, layout)
    response = apply_element_status_contract_policy(
        response,
        scope_complete=coverage.scope_complete,
        visual_required=coverage.visual_required,
        visual_complete=coverage.visual_complete,
    )
    # A context follow-up is one extra chunk, not the complete-document retry.
    # The contract turns a leftover retrieval gap into not_determined only
    # after that retry has called the model. Otherwise the answer stays in
    # review, including when the first call already saw every saved page.
    response = apply_final_evidence_contract_policy(
        response,
        complete_document_retried=complete_document_retried,
    )

    return build_finding(
        defect,
        response,
        evidence_ids=[c["record_id"] for c in chunks if c.get("record_id")],
        coverage=coverage,
        usage=usage,
        chunks=chunks,
        layout=layout,
        visual_index=check_visual_index,
        record=record if isinstance(record, dict) else None,
    )


async def _run_defect(
    defect: Defect,
    *,
    catalogue: Catalogue,
    record: dict[str, Any] | None,
    chunks: list[dict[str, Any]],
    filing_type: str | None,
    layout: dict[int, dict[str, Any]] | None = None,
    visual_index: dict[str, Any] | None = None,
    absence_defect_parts: list[str] | None = None,
    model: str,
    llm_client: httpx.AsyncClient | None = None,
    evidence_workspace: EvidenceWorkspace | None = None,
    evaluator: ScrutinyMicrobatcher | None = None,
    file_hash: str | None = None,
    organization_id: str | None = None,
    workspace_id: str | None = None,
    index_generation: str | None = None,
    use_pinecone: bool = False,
    vision_note: str | None = None,
) -> DefectFinding:
    """Synchronous-call compatibility wrapper used outside batch scrutiny."""
    prepared = _prepare_defect_evaluation(
        defect,
        catalogue=catalogue,
        record=record,
        chunks=chunks,
        filing_type=filing_type,
        layout=layout,
        visual_index=visual_index,
        absence_defect_parts=absence_defect_parts,
        model=model,
        llm_client=llm_client,
        evidence_workspace=evidence_workspace,
        file_hash=file_hash,
        organization_id=organization_id,
        workspace_id=workspace_id,
        index_generation=index_generation,
        use_pinecone=use_pinecone,
        vision_note=vision_note,
    )
    if isinstance(prepared, DefectFinding):
        return prepared
    if evaluator is not None:
        raw, usage = await evaluator.evaluate(
            defect=defect,
            system_prompt=prepared.system_prompt,
            user_prompt=prepared.user_prompt,
            model=model,
        )
    else:
        raw, usage = await call_structured(
            system_prompt=prepared.system_prompt,
            user_prompt=prepared.user_prompt,
            response_model=DefectResponse,
            model=model,
            client=llm_client,
            request_label=f"scrutiny:{defect.check_id}",
        )
    return await _finalize_defect_evaluation(prepared, raw, usage)


async def _chunks_for_defect(
    defect: Defect,
    *,
    file_hash: str | None,
    max_chunks: int,
    use_pinecone: bool,
    organization_id: str | None = None,
    workspace_id: str | None = None,
    index_generation: str | None = None,
    layout: dict[int, dict[str, Any]] | None = None,
    evidence_workspace: EvidenceWorkspace | None = None,
) -> list[dict[str, Any]]:
    """Fetch this defect's excerpts. Runs inside the concurrency semaphore."""
    retrieval_plan = retrieval_plan_for_defect(defect)
    logger.info(
        "[Scrutiny] Retrieval plan check=%s strategy=%s queries=%s parts=%s",
        defect.check_id,
        retrieval_plan.strategy,
        len(retrieval_plan.queries),
        list(retrieval_plan.include_parts),
    )
    for planned_query in retrieval_plan.queries:
        logger.debug(
            "[Scrutiny] Retrieval query check=%s part=%s subtype=%s text=%s",
            defect.check_id,
            planned_query.document_part,
            planned_query.document_subtype_id,
            planned_query.text,
        )
    page_budget = max_chunks_for_defect(defect, ceiling=max_chunks)
    if retrieval_plan.strategy == "exhaustive_part" and layout:
        required = required_parts_for_defect(defect)
        exhaustive_pages = {
            page_number
            for page_number, page in layout.items()
            if isinstance(page, dict)
            and any(
                chunks_cover_part(
                    [
                        {
                            "document_part": _layout_document_parts(
                                page.get("document_part")
                            ),
                            "chunk_kind": "page",
                        }
                    ],
                    part,
                )
                for part in required
            )
        }
        page_budget = max(page_budget, len(exhaustive_pages))
        direct_chunks = (
            evidence_workspace.complete_chunks(defect)
            if evidence_workspace is not None
            else []
        )
        # Every labeled page already has stored text. A sample from Pinecone
        # cannot add a page the manifest does not have.
        if direct_chunks and len(direct_chunks) >= len(exhaustive_pages):
            logger.info(
                "[Scrutiny] Direct evidence hit check=%s pages=%s",
                defect.check_id,
                len(direct_chunks),
            )
            return direct_chunks
    if not use_pinecone or not file_hash:
        return _add_layout_fallback_chunks(
            defect, [], layout=layout, max_chunks=page_budget
        )

    queries = [query.text for query in retrieval_plan.queries]
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
            organization_id=organization_id,
            workspace_id=workspace_id,
            index_generation=index_generation,
            query_plan=retrieval_plan.queries,
            vector_schema_version=VECTOR_SCHEMA_VERSION,
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

    missing = missing_required_parts(defect, pool)
    manifest_missing = missing_parts_in_layout(defect, layout)
    if manifest_missing is not None:
        missing = [part for part in missing if part not in manifest_missing]
    if missing:
        missing_plan = tuple(
            query
            for query in retrieval_plan.queries
            if query.document_part is None or query.document_part in missing
        )
        broader = await asyncio.to_thread(
            gather_filing_evidence,
            queries,
            file_hash=file_hash,
            top_k=max(scrutiny_top_k() * 2, page_budget),
            max_chunks=gather_cap,
            document_parts=missing,
            organization_id=organization_id,
            workspace_id=workspace_id,
            index_generation=index_generation,
            query_plan=missing_plan,
            vector_schema_version=VECTOR_SCHEMA_VERSION,
        )
        by_id = {
            str(chunk.get("record_id")): chunk
            for chunk in [*pool, *broader]
            if chunk.get("record_id")
        }
        pool = list(by_id.values())
    pool = await asyncio.to_thread(rerank_chunks_for_defect, pool, defect)
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
    """Return the parsed page text stored with the complete page manifest.

    LlamaParse-backed layouts can carry page markdown/text without word boxes.
    Prefer that source verbatim, then rebuild text from grounded boxes for older
    artifacts.  This makes the manifest a real retrieval fallback instead of
    silently producing an empty excerpt when Pinecone returns no match.
    """
    direct = str(page.get("text") or page.get("markdown") or "").strip()
    if direct:
        return direct
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


def _layout_anchor_phrases(defect: Defect) -> list[str]:
    """Return headings/paragraph markers that identify the page under review.

    Semantic retrieval is not reliable for short, repeated form labels such as
    ``Paragraph 3``.  These anchors are intentionally conservative: they come
    only from quoted headings and explicit paragraph references authored in
    the catalogue, plus a few standard Form 28 headings.
    """
    wording = " ".join(
        [
            check_requirement(defect),
            *list(getattr(defect, "where_to_look", None) or []),
        ]
    )
    anchors: list[str] = []

    def add(value: str) -> None:
        value = re.sub(r"\s+", " ", value).strip(" .,:;\"'[]()")
        if len(value) >= 4 and value.casefold() not in {
            item.casefold() for item in anchors
        }:
            anchors.append(value)

    quoted_heading = re.compile(
        r"[\"'\u2018\u2019\u201c\u201d]([^\"'\u2018\u2019\u201c\u201d]{4,120})"
        r"[\"'\u2018\u2019\u201c\u201d]"
    )
    for quoted in quoted_heading.findall(wording):
        add(quoted)

    for phrase in distinctive_trigger_phrases(defect):
        add(phrase)

    for number in re.findall(r"\bparagraph\s+(\d+[A-Z]?)\b", wording, re.IGNORECASE):
        add(f"{number}.")

    folded = wording.casefold()
    standard_headings = (
        "declaration in terms of rule 3(2)",
        "declaration in terms of rule 5",
        "question of law",
        "main prayer",
    )
    for heading in standard_headings:
        if heading in folded:
            add(heading)
    if "declaration in terms" in folded:
        # OCR and drafting errors often corrupt the rule number (the reviewed
        # filing says "Rule 105"), while the declaration heading remains a
        # reliable locator for the paragraph whose substance must be checked.
        add("declaration in terms")
    return anchors


def _anchor_score(text: str, anchors: list[str]) -> int:
    folded = re.sub(r"\s+", " ", text).casefold()
    return sum(1 for anchor in anchors if anchor.casefold() in folded)


def _paragraph_numbers_for_defect(defect: Defect) -> list[str]:
    wording = " ".join(
        [
            check_requirement(defect),
            *list(getattr(defect, "where_to_look", None) or []),
        ]
    )
    return list(
        dict.fromkeys(
            re.findall(r"\bparagraph\s+(\d+[A-Z]?)\b", wording, re.IGNORECASE)
        )
    )


def _has_paragraph_marker(text: str, number: str) -> bool:
    return bool(
        re.search(
            rf"(?im)^\s*{re.escape(number)}\s*[.)]\s+\S",
            text,
        )
    )


_AOR_CERTIFICATE_HEADING = re.compile(
    r"(?m)^\s*[^A-Za-z0-9\s]{0,3}"
    r"(?:c\s+e\s+r\s+t\s+i\s+f\s+i\s+c\s+a\s+t\s+e|certificate)\b",
    re.IGNORECASE,
)
_AOR_CERTIFIED_THAT = re.compile(r"\bcertif(?:i|l)ed\s+that\b", re.IGNORECASE)
_AOR_CONFINED_PLEADINGS = re.compile(
    r"confined\s+on(?:ly|iy).{0,80}pleadings",
    re.IGNORECASE,
)


def _folded_page_text(text: str) -> str:
    return re.sub(r"\s+", " ", text or "").casefold()


def _looks_like_aor_certificate_text(text: str) -> bool:
    """The AOR filing certificate, not the checklist declaration or an affidavit."""
    window = (text or "")[:2500]
    if not window.strip():
        return False
    folded = _folded_page_text(window)
    opening = folded[:400]
    if "affidavit" in opening and "certificate" not in opening:
        return False
    if "check list" in opening or "checklist" in opening:
        return False
    heading = _AOR_CERTIFICATE_HEADING.search(window) is not None
    certified = _AOR_CERTIFIED_THAT.search(folded) is not None
    confined = _AOR_CONFINED_PLEADINGS.search(folded) is not None
    advocate = "advocate-on-record" in folded or "advocate on record" in folded
    if heading and (certified or confined or advocate):
        return True
    return certified and confined


def _aor_certificate_text_pages(
    layout: dict[int, dict[str, Any]] | None,
) -> list[dict[str, Any]]:
    """Certificate pages Split left on, or immediately after, the Main Petition.

    The Advocate's Checklist has its own AOR signature. That page is not the
    certificate filed after the petition.
    """
    if not layout:
        return []
    main_pages = [
        int(number)
        for number, page in layout.items()
        if isinstance(page, dict)
        and "Main Petition" in _layout_document_parts(page.get("document_part"))
    ]
    last_main = max(main_pages) if main_pages else None
    found: list[dict[str, Any]] = []
    for number, page in sorted(layout.items()):
        if not isinstance(page, dict):
            continue
        names = _layout_document_parts(page.get("document_part"))
        if "Advocate's Checklist" in names or "AOR's Certificate" in names:
            continue
        if any(name.startswith("Annexure") or name == "Appendix" for name in names):
            continue
        text = _layout_page_text(page)
        if not _looks_like_aor_certificate_text(text):
            continue
        near_petition = last_main is not None and abs(int(number) - last_main) <= 2
        on_petition = "Main Petition" in names or not names
        if not on_petition and not near_petition:
            continue
        found.append(
            {
                "record_id": f"layout:{number}:AOR's Certificate",
                "score": 0.0,
                "chunk_kind": "page",
                "page": int(number),
                "page_end": int(number),
                "document_part": ["AOR's Certificate"],
                "text": text,
                "layout_fallback": True,
            }
        )
    return found


def _layout_contains_aor_certificate(
    layout: dict[int, dict[str, Any]] | None,
) -> bool:
    """True when a page is labeled as the certificate or carries its text."""
    if not layout:
        return False
    for page in layout.values():
        if not isinstance(page, dict):
            continue
        names = _layout_document_parts(page.get("document_part"))
        if "AOR's Certificate" in names:
            return True
        if "Advocate's Checklist" in names:
            continue
        if _looks_like_aor_certificate_text(_layout_page_text(page)):
            return True
    return False


def _layout_has_blank_unlabeled_page(
    layout: dict[int, dict[str, Any]] | None,
) -> bool:
    """A blank unlabeled page can still be the unscanned certificate."""
    if not layout:
        return False
    for page in layout.values():
        if not isinstance(page, dict):
            return True
        if _layout_document_parts(page.get("document_part")):
            continue
        if not _layout_page_text(page):
            return True
    return False


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
    # The checklist names the Advocate-on-Record and prints a code. It is not
    # the certificate filed after the Main Petition, so it must not fill this
    # check's evidence when that certificate is the document under review.
    if "AOR's Certificate" in required and "Advocate's Checklist" not in required:
        pages = [
            chunk
            for chunk in pages
            if "Advocate's Checklist"
            not in _layout_document_parts(chunk.get("document_part"))
        ]
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
        if part == "AOR's Certificate":
            seen_pages = {int(row["page"]) for row in found}
            for extra in _aor_certificate_text_pages(layout):
                if int(extra["page"]) not in seen_pages:
                    found.append(extra)
                    seen_pages.add(int(extra["page"]))
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
            check_requirement(defect),
            *list(getattr(defect, "where_to_look", None) or []),
        ]
    ).casefold()
    last_first = bool(
        re.search(
            r"last\s+page|end\s+of|below\s+the\s+prayer|date\s+of\s+drafting|"
            r"execution\s+date|signature|true\s+cop|certif|"
            r"verified\s+at|drawn\s+on|filed\s+on|deponent",
            wording,
        )
    )
    # The date and the signature sit on the closing page. Put that page in
    # before any opening-page anchor can use up the excerpt budget.
    if last_first:
        for part in required:
            options = candidates.get(part) or []
            if not options:
                continue
            closing = options[-1]
            if any(chunk.get("page") == closing.get("page") for chunk in chosen):
                continue
            chosen.append(closing)
        capacity = max(capacity, len(chosen))

    # Paragraph-level Form 28 checks need the page containing the authored
    # heading, not merely the opening and closing pages of a long petition.
    # Prefer grounded layout OCR here; Pinecone still supplies any remaining
    # pages after the anchors and boundaries have been reserved.
    anchors = _layout_anchor_phrases(defect)
    if anchors:
        for part in required:
            if len(chosen) >= capacity:
                break
            options = candidates.get(part) or []
            # Form 28's opening numbered paragraph can begin on the page after
            # "MOST RESPECTFULLY SHOWETH".  Prefer that paragraph page over a
            # party-list page that happens to contain the same number.
            paragraph_numbers = _paragraph_numbers_for_defect(defect)
            if part == "Main Petition" and paragraph_numbers:
                showeth_positions = [
                    index
                    for index, option in enumerate(options)
                    if "most respectfully showeth"
                    in str(option.get("text") or "").casefold()
                ]
                paragraph_page = None
                for start in showeth_positions:
                    for option in options[start : start + 3]:
                        if any(
                            _has_paragraph_marker(str(option.get("text") or ""), number)
                            for number in paragraph_numbers
                        ):
                            paragraph_page = option
                            break
                    if paragraph_page is not None:
                        break
                if paragraph_page is not None:
                    candidate = next(
                        (
                            chunk
                            for chunk in target_pages
                            if chunk.get("page") == paragraph_page.get("page")
                            and chunks_cover_part([chunk], part)
                        ),
                        paragraph_page,
                    )
                    if not any(
                        chunk.get("page") == candidate.get("page") for chunk in chosen
                    ):
                        chosen.append(candidate)
                    continue
            ranked_anchors = sorted(
                (
                    (_anchor_score(str(option.get("text") or ""), anchors), option)
                    for option in options
                ),
                key=lambda item: (-item[0], int(item[1].get("page") or 0)),
            )
            if not ranked_anchors or ranked_anchors[0][0] <= 0:
                continue
            fallback = ranked_anchors[0][1]
            candidate = next(
                (
                    chunk
                    for chunk in target_pages
                    if chunk.get("page") == fallback.get("page")
                    and chunks_cover_part([chunk], part)
                ),
                fallback,
            )
            if not any(chunk.get("page") == candidate.get("page") for chunk in chosen):
                chosen.append(candidate)

    # Exact stamps, signatures and closing clauses frequently split across a
    # page boundary. Expand only to immediate neighbors carrying the same
    # classified part; never cross into another document merely because its
    # page number is adjacent.
    if retrieval_plan_for_defect(defect).strategy == "boundary_and_neighbors":
        anchors_chosen = list(chosen)
        for anchor in anchors_chosen:
            if len(chosen) >= capacity:
                break
            anchor_page = int(anchor.get("page") or 0)
            anchor_parts = _layout_document_parts(anchor.get("document_part"))
            for part in required:
                if part not in anchor_parts and not chunks_cover_part([anchor], part):
                    continue
                options = candidates.get(part) or []
                for neighbor_page in (anchor_page - 1, anchor_page + 1):
                    fallback = next(
                        (
                            option
                            for option in options
                            if int(option.get("page") or 0) == neighbor_page
                        ),
                        None,
                    )
                    if fallback is None or any(
                        chunk.get("page") == neighbor_page
                        and chunks_cover_part([chunk], part)
                        for chunk in chosen
                    ):
                        continue
                    candidate = next(
                        (
                            chunk
                            for chunk in target_pages
                            if chunk.get("page") == neighbor_page
                            and chunks_cover_part([chunk], part)
                        ),
                        fallback,
                    )
                    chosen.append(candidate)
                    if len(chosen) >= capacity:
                        break

    boundaries = ("last", "first") if last_first else ("first", "last")

    if _requires_complete_part_coverage(defect):
        for part in required:
            for fallback in candidates.get(part) or []:
                if len(chosen) >= capacity:
                    break
                candidate = next(
                    (
                        chunk
                        for chunk in target_pages
                        if chunk.get("page") == fallback.get("page")
                        and chunks_cover_part([chunk], part)
                    ),
                    fallback,
                )
                if not any(
                    chunk.get("page") == candidate.get("page")
                    and chunks_cover_part([chunk], part)
                    for chunk in chosen
                ):
                    chosen.append(candidate)

    # Collection-wide Annexure checks cannot rely on semantic ranking alone.
    # For example, a "TRUE COPY" search ranks compliant pages above the
    # Annexure that actually omits the endorsement.  Ground one boundary from
    # every numbered Annexure first, then the opposite boundary when the
    # evidence budget permits.  This also lets vernacular/translation checks
    # see each distinct exhibit instead of a single high-scoring sample.
    if _requires_annexure_collection_coverage(defect):
        annexure_runs: dict[str, list[dict[str, Any]]] = {}
        for options in candidates.values():
            for option in options:
                numbered = next(
                    (
                        name
                        for name in _layout_document_parts(option.get("document_part"))
                        if _is_numbered_annexure(name)
                        and family_split_name(name) == ANNEXURE_FAMILY
                    ),
                    None,
                )
                if numbered:
                    annexure_runs.setdefault(numbered.casefold(), []).append(option)

        for boundary in boundaries:
            for options in annexure_runs.values():
                if len(chosen) >= capacity:
                    break
                fallback = options[0] if boundary == "first" else options[-1]
                candidate = next(
                    (
                        chunk
                        for chunk in target_pages
                        if chunk.get("page") == fallback.get("page")
                        and any(
                            _is_numbered_annexure(name)
                            for name in _layout_document_parts(
                                chunk.get("document_part")
                            )
                        )
                    ),
                    fallback,
                )
                if any(chunk.get("page") == candidate.get("page") for chunk in chosen):
                    continue
                chosen.append(candidate)

    # First guarantee the most relevant boundary for every required part,
    # then add the opposite boundary when the evidence budget permits.
    for boundary in boundaries:
        for part in required:
            if len(chosen) >= capacity:
                break
            options = candidates.get(part) or []
            if not options:
                continue
            fallback = options[0] if boundary == "first" else options[-1]
            if any(
                chunk.get("page") == fallback.get("page")
                and chunks_cover_part([chunk], part)
                for chunk in chosen
            ):
                continue
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

    context_parts = catalogue_context_parts(defect)
    for chunk in other_pages:
        if len(chosen) >= capacity or not context_parts:
            break
        if any(chunks_cover_part([chunk], part) for part in context_parts):
            chosen.append(chunk)

    chosen.sort(key=lambda chunk: (chunk.get("page") is None, chunk.get("page") or 0))
    return summary + chosen


def _special_categories_for_filing(
    *,
    supplied: str | None,
    supplied_categories: list[str] | None = None,
    metadata: dict[str, Any] | None,
    record: dict[str, Any] | None,
    layout: dict[int, dict[str, Any]] | None,
    filing_type: str | None,
) -> list[str]:
    """Combine explicit and structurally certain special-category overlays.

    Index and reindex pass ``supplied_categories``, the array from the
    request. The comma-separated ``special_category`` string is only the
    value stored on the matter, used when the request did not send the array.
    """
    found: list[str] = []

    def add(value: Any) -> None:
        if isinstance(value, list):
            for item in value:
                add(item)
            return
        text = str(value or "").strip()
        names: list[str] = []
        for name in split_special_category_list(text):
            if name not in names:
                names.append(name)
        for name in match_front_special_categories(text, limit=None):
            if name not in names:
                names.append(name)
        if not names and normalize_special_category(text):
            names = [text]
        for name in names:
            if name not in found:
                found.append(name)

    if supplied_categories is not None:
        add(supplied_categories)
    else:
        add(supplied)
        add((metadata or {}).get("special_categories"))
        add((record or {}).get("special_categories"))
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
    categories: list[str | None] = (
        [*special_categories] if special_categories else [None]
    )
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

    # These catalogue rows express the same global tests.  Publishing both
    # produces duplicate findings (and can even produce two different statuses
    # for the same blank field), so keep the older canonical row whenever both
    # are selected.  If a caller explicitly enables only the alias, preserve it.
    duplicate_of = {"D-84": "D-55", "D-85": "D-57"}
    selected_ids = {defect.check_id for defect in found}
    return [
        defect
        for defect in found
        if not (
            defect.check_id in duplicate_of
            and duplicate_of[defect.check_id] in selected_ids
        )
    ]


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
        scrutiny_started = time.perf_counter()
        if not scrutiny_enabled():
            raise ValueError("Scrutiny is disabled (SCRUTINY_ENABLED=false)")
        if not llm_enabled():
            if llm_provider() == "vertex":
                raise ValueError(
                    "LLM_PROVIDER=vertex but GOOGLE_CLOUD_PROJECT is not set"
                )
            if llm_provider() == "gemini":
                raise ValueError("LLM_PROVIDER=gemini but GEMINI_API_KEY is not set")
            raise ValueError(
                "OPENROUTER_API_KEY is not set, so defect checks cannot run"
            )
        try:
            defect_model = scrutiny_model(event.model_tier or "")
        except ValueError as exc:
            raise ValueError(str(exc)) from exc
        logger.info(
            "LLM provider=%s key=%s model=%s tier=%s rpm=%s",
            llm_provider(),
            mask_llm_credential(),
            defect_model,
            event.model_tier,
            llm_requests_per_minute() or "unlimited",
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
        vector_meta = (
            metadata.get("vector_index") if isinstance(metadata, dict) else None
        )
        if not isinstance(vector_meta, dict):
            vector_meta = {}
        index_generation = str(vector_meta.get("generation") or "").strip() or None
        vector_organization_id = (
            str(
                vector_meta.get("organization_id")
                or (
                    metadata.get("organization_id")
                    if isinstance(metadata, dict)
                    else ""
                )
                or event.organization_id
                or ""
            ).strip()
            or None
        )
        vector_workspace_id = (
            str(
                vector_meta.get("workspace_id")
                or (metadata.get("workspace_id") if isinstance(metadata, dict) else "")
                or event.workspace_id
                or ""
            ).strip()
            or None
        )
        vector_file_hash = str(vector_meta.get("file_hash") or "").strip()
        if (
            index_generation
            and vector_file_hash
            and vector_file_hash != str(file_hash or "")
        ):
            raise ValueError(
                "Active Pinecone generation belongs to a different filing hash; reindex "
                "this workspace before running scrutiny"
            )
        if event.organization_id and vector_organization_id != event.organization_id:
            raise ValueError(
                "Active Pinecone generation belongs to another organization"
            )
        if event.workspace_id and vector_workspace_id != event.workspace_id:
            raise ValueError("Active Pinecone generation belongs to another workspace")
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
        evidence_workspace = EvidenceWorkspace.from_layout(layout)
        if not layout:
            logger.warning(
                "Layout index empty pages=0; defect findings will have no "
                "highlight coordinates url=%s key=%s",
                safe_url_for_log(layout_url) if isinstance(layout_url, str) else None,
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
            supplied_categories=event.special_categories,
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

        absence_owners = _absence_owner_check_ids(defects, layout)
        if absence_owners:
            logger.info(
                "[Scrutiny] Manifest absence owners: %s",
                ", ".join(
                    f"{check_id}={'+'.join(parts)}"
                    for check_id, parts in sorted(absence_owners.items())
                ),
            )

        concurrency = _int_env("SCRUTINY_CONCURRENCY", DEFAULT_CONCURRENCY)
        persist_every = max(
            1, _int_env("SCRUTINY_PERSIST_EVERY", DEFAULT_PERSIST_EVERY)
        )
        max_chunks = scrutiny_max_chunks()
        use_pinecone = pinecone_enabled() and bool(file_hash)

        active_schema = vector_meta.get("schema_version")
        if (
            use_pinecone
            and index_generation
            and str(active_schema) != str(VECTOR_SCHEMA_VERSION)
        ):
            raise ValueError(
                "Active Pinecone generation uses vector schema "
                f"{active_schema or 'unknown'}; schema {VECTOR_SCHEMA_VERSION} is "
                "required. Reindex this filing before running scrutiny."
            )

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
        image_plans = with_slot_files(
            plan_defect_images(defects, layout),
            metadata if isinstance(metadata, dict) else None,
        )
        incomplete_visual_checks = sorted(
            plan.check_id for plan in image_plans if plan.leftover_pages
        )
        if isinstance(visual_index, dict) and incomplete_visual_checks:
            visual_index = dict(visual_index)
            visual_index["incomplete_check_ids"] = incomplete_visual_checks
        if visual_detection_enabled() and any(plan.pages for plan in image_plans):
            ctx.write_event_to_stream(
                Status(
                    level="info",
                    message=(
                        "Detecting notary seals, signatures, and stamps "
                        f"for {len(image_plans)} defect(s)"
                    ),
                )
            )
            detected = await detect_visual_marks(
                parts=[],
                pages_by_slot={},
                page_parts={},
                model_tier=event.model_tier,
                selected_targets=visual_targets_for_plans(image_plans),
            )
            # Preserve completed page-quality and labelled-field assessments
            # even when the page contains no signature, seal, or stamp. An
            # empty marks list is affirmative visual evidence for blank-field
            # and legibility checks, not a failed scan.
            if _usable_visual_detection(detected):
                if incomplete_visual_checks:
                    detected["incomplete_check_ids"] = incomplete_visual_checks
                visual_index = detected

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
                model=defect_model,
                index_generation=index_generation,
                vector_schema_version=str(VECTOR_SCHEMA_VERSION),
                disclaimer=catalogue.disclaimer,
                findings=snapshot,
                summary=summarize(snapshot),
                usage=summarize_usage(snapshot, model=defect_model),
                planned_checks=planned,
                stopped_early=stopped_early,
            )

        persist_lock = asyncio.Lock()
        persist_tasks: set[asyncio.Task[None]] = set()
        terminal_openrouter_batch_ids: set[str] = set()

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

        llm_http_client = (
            create_openrouter_http_client() if llm_provider() == "openrouter" else None
        )
        microbatcher = ScrutinyMicrobatcher(llm_http_client)

        async def run_one(defect: Defect) -> DefectFinding:
            ctx.write_event_to_stream(
                Status(
                    level="info",
                    message=f"Checking {defect.check_id}",
                )
            )
            try:
                retrieval_started = time.perf_counter()
                manifest_missing = (
                    missing_parts_in_layout(defect, layout)
                    if defect.manifest_absence_proves_defect
                    else missing_dependency_parts_in_layout(defect, layout)
                )
                chunks = (
                    []
                    if manifest_missing
                    else await _chunks_for_defect(
                        defect,
                        file_hash=file_hash,
                        max_chunks=max_chunks,
                        use_pinecone=use_pinecone,
                        organization_id=vector_organization_id,
                        workspace_id=vector_workspace_id,
                        index_generation=index_generation,
                        layout=layout,
                        evidence_workspace=evidence_workspace,
                    )
                )
                retrieval_seconds = time.perf_counter() - retrieval_started
                evaluation_started = time.perf_counter()
                finding = await _run_defect(
                    defect,
                    catalogue=catalogue,
                    record=record,
                    chunks=chunks,
                    filing_type=filing_type,
                    layout=layout,
                    visual_index=visual_index,
                    absence_defect_parts=absence_owners.get(defect.check_id),
                    model=defect_model,
                    llm_client=llm_http_client,
                    evidence_workspace=evidence_workspace,
                    evaluator=microbatcher,
                    file_hash=file_hash,
                    organization_id=vector_organization_id,
                    workspace_id=vector_workspace_id,
                    index_generation=index_generation,
                    use_pinecone=use_pinecone,
                )
                logger.info(
                    "[ScrutinyTiming] check=%s retrieval=%.2fs evaluation=%.2fs",
                    defect.check_id,
                    retrieval_seconds,
                    time.perf_counter() - evaluation_started,
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

        async def run_batch_checks() -> tuple[list[DefectFinding], bool]:
            """Retrieve first, then submit model-ready checks in bulk.

            Retrieval remains bounded by ``SCRUTINY_CONCURRENCY``. It is not
            placed inside the OpenRouter request batch because Pinecone/layout
            evidence must be complete before a model request can be built.
            """
            retrieval_semaphore = asyncio.Semaphore(max(1, concurrency))

            async def prepare_one(
                defect: Defect,
            ) -> DefectFinding | _PreparedDefectEvaluation:
                ctx.write_event_to_stream(
                    Status(level="info", message=f"Preparing {defect.check_id}")
                )
                try:
                    started = time.perf_counter()
                    manifest_missing = (
                        missing_parts_in_layout(defect, layout)
                        if defect.manifest_absence_proves_defect
                        else missing_dependency_parts_in_layout(defect, layout)
                    )
                    async with retrieval_semaphore:
                        chunks = (
                            []
                            if manifest_missing
                            else await _chunks_for_defect(
                                defect,
                                file_hash=file_hash,
                                max_chunks=max_chunks,
                                use_pinecone=use_pinecone,
                                organization_id=vector_organization_id,
                                workspace_id=vector_workspace_id,
                                index_generation=index_generation,
                                layout=layout,
                                evidence_workspace=evidence_workspace,
                            )
                        )
                    prepared = _prepare_defect_evaluation(
                        defect,
                        catalogue=catalogue,
                        record=record,
                        chunks=chunks,
                        filing_type=filing_type,
                        layout=layout,
                        visual_index=visual_index,
                        absence_defect_parts=absence_owners.get(defect.check_id),
                        model=defect_model,
                        llm_client=llm_http_client,
                        evidence_workspace=evidence_workspace,
                        file_hash=file_hash,
                        organization_id=vector_organization_id,
                        workspace_id=vector_workspace_id,
                        index_generation=index_generation,
                        use_pinecone=use_pinecone,
                    )
                    logger.info(
                        "[ScrutinyTiming] check=%s batch_prepare=%.2fs",
                        defect.check_id,
                        time.perf_counter() - started,
                    )
                    return prepared
                except asyncio.CancelledError:
                    raise
                except Exception as exc:
                    logger.exception(
                        "[Scrutiny] %s preparation failed", defect.check_id
                    )
                    return failed_finding(
                        defect,
                        str(exc),
                        usage=exc.usage if isinstance(exc, LLMError) else None,
                        visual_index=visual_index,
                        record=record if isinstance(record, dict) else None,
                    )

            prepared_items = await asyncio.gather(
                *(prepare_one(defect) for defect in defects)
            )
            completed: list[DefectFinding] = [
                item for item in prepared_items if isinstance(item, DefectFinding)
            ]
            pending: list[_PreparedDefectEvaluation] = [
                item
                for item in prepared_items
                if isinstance(item, _PreparedDefectEvaluation)
            ]
            if completed:
                await publish(completed, False)
            if not pending:
                return completed, False

            max_batch = openrouter_batch_max_requests()
            groups = [
                pending[index : index + max_batch]
                for index in range(0, len(pending), max_batch)
            ]
            ctx.write_event_to_stream(
                Status(
                    level="info",
                    message=(
                        f"Submitting {len(pending)} checks in {len(groups)} "
                        "discounted OpenRouter batch(es)"
                    ),
                )
            )

            async def submit_group(
                group: list[_PreparedDefectEvaluation],
            ) -> tuple[list[_PreparedDefectEvaluation], Any]:
                async def remember_batch(batch_id: str) -> None:
                    async with ctx.store.edit_state() as state:
                        if batch_id not in state.openrouter_batch_ids:
                            state.openrouter_batch_ids.append(batch_id)
                    ctx.write_event_to_stream(
                        Status(
                            level="info",
                            message=f"OpenRouter batch accepted: {batch_id}",
                        )
                    )

                try:
                    result = await call_structured_batch(
                        requests=[
                            StructuredBatchRequest(
                                custom_id=item.defect.check_id,
                                system_prompt=item.system_prompt,
                                user_prompt=item.user_prompt,
                                request_label=f"scrutiny:{item.defect.check_id}",
                            )
                            for item in group
                        ],
                        response_model=DefectResponse,
                        model=defect_model,
                        client=llm_http_client,
                        on_batch_accepted=remember_batch,
                    )
                    return group, result
                except Exception as exc:
                    return group, exc

            grouped_results = await asyncio.gather(
                *(submit_group(group) for group in groups)
            )
            finalization_semaphore = asyncio.Semaphore(max(1, concurrency))

            async def finalize_item(
                item: _PreparedDefectEvaluation,
                batch_result: StructuredBatchResult,
            ) -> DefectFinding:
                if batch_result.error is not None or batch_result.value is None:
                    error = batch_result.error or LLMError("Batch result was empty")
                    return failed_finding(
                        item.defect,
                        str(error),
                        usage=batch_result.usage,
                        visual_index=visual_index,
                        record=record if isinstance(record, dict) else None,
                    )
                async with finalization_semaphore:
                    try:
                        return await _finalize_defect_evaluation(
                            item,
                            DefectResponse.model_validate(
                                batch_result.value.model_dump()
                            ),
                            batch_result.usage,
                        )
                    except asyncio.CancelledError:
                        raise
                    except Exception as exc:
                        logger.exception(
                            "[Scrutiny] %s batch finalization failed",
                            item.defect.check_id,
                        )
                        return failed_finding(
                            item.defect,
                            str(exc),
                            usage=(
                                exc.usage
                                if isinstance(exc, LLMError)
                                else batch_result.usage
                            ),
                            visual_index=visual_index,
                            record=record if isinstance(record, dict) else None,
                        )

            async def synchronous_fallback(
                group: list[_PreparedDefectEvaluation],
            ) -> list[DefectFinding]:
                fallback_semaphore = asyncio.Semaphore(max(1, concurrency))

                async def evaluate(item: _PreparedDefectEvaluation) -> DefectFinding:
                    async with fallback_semaphore:
                        try:
                            raw, usage = await call_structured(
                                system_prompt=item.system_prompt,
                                user_prompt=item.user_prompt,
                                response_model=DefectResponse,
                                model=item.model,
                                client=llm_http_client,
                                request_label=(
                                    f"scrutiny:{item.defect.check_id}:batch-fallback"
                                ),
                            )
                            return await _finalize_defect_evaluation(item, raw, usage)
                        except Exception as exc:
                            return failed_finding(
                                item.defect,
                                str(exc),
                                usage=exc.usage if isinstance(exc, LLMError) else None,
                                visual_index=visual_index,
                                record=record if isinstance(record, dict) else None,
                            )

                return await asyncio.gather(*(evaluate(item) for item in group))

            for group, result in grouped_results:
                if isinstance(result, Exception):
                    if getattr(result, "batch_id", None) and getattr(
                        result, "terminal", False
                    ):
                        terminal_openrouter_batch_ids.add(str(result.batch_id))
                    safe_fallback = (
                        isinstance(result, OpenRouterBatchError)
                        and not result.accepted
                        and openrouter_batch_sync_fallback_enabled()
                    )
                    if safe_fallback:
                        logger.warning(
                            "[Scrutiny] Batch rejected before acceptance; "
                            "using synchronous fallback: %s",
                            result,
                        )
                        completed.extend(await synchronous_fallback(group))
                    else:
                        # Never re-submit accepted remote work: that can double
                        # charge the same filing. The batch id is kept in the
                        # finding error/log for operational recovery.
                        batch_id = getattr(result, "batch_id", None)
                        detail = f"{result}"
                        if batch_id:
                            detail = f"{detail} (OpenRouter batch {batch_id})"
                        for item in group:
                            completed.append(
                                failed_finding(
                                    item.defect,
                                    detail,
                                    usage=(
                                        result.usage
                                        if isinstance(result, LLMError)
                                        else None
                                    ),
                                    visual_index=visual_index,
                                    record=(
                                        record if isinstance(record, dict) else None
                                    ),
                                )
                            )
                    await publish(completed, False)
                    continue

                finalized = await asyncio.gather(
                    *(
                        finalize_item(item, result[item.defect.check_id])
                        for item in group
                    )
                )
                terminal_openrouter_batch_ids.update(
                    batch_result.batch_id
                    for batch_result in result.values()
                    if batch_result.batch_id
                )
                completed.extend(finalized)
                await publish(completed, False)
            return completed, False

        try:
            if openrouter_batch_enabled() and llm_provider() == "openrouter":
                findings, stopped_early = await run_batch_checks()
            else:
                findings, stopped_early = await collect_defect_findings(
                    defects,
                    run_one,
                    concurrency=concurrency,
                    stop_on_error=False,
                    on_update=publish,
                )
        finally:
            await microbatcher.close()
            if llm_http_client is not None:
                await llm_http_client.aclose()
        logger.info(
            "[ScrutinyTiming] total=%.1fs checks=%s concurrency=%s rpm=%s",
            time.perf_counter() - scrutiny_started,
            len(findings),
            concurrency,
            llm_requests_per_minute() or "unlimited",
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
            version=event.version,
        )
        # OpenRouter retains batch inputs/results for up to 30 days. Cleanup is
        # intentionally after both report stores have succeeded, never merely
        # after the HTTP result was read into worker memory.
        await delete_openrouter_batches(sorted(terminal_openrouter_batch_ids))
        if terminal_openrouter_batch_ids:
            async with ctx.store.edit_state() as state:
                state.openrouter_batch_ids = [
                    batch_id
                    for batch_id in state.openrouter_batch_ids
                    if batch_id not in terminal_openrouter_batch_ids
                ]
        require_successful_checks(findings, planned)

        cost_note = ""
        if report.usage and report.usage.cost_usd is not None:
            cost_note = f", OpenRouter {report.usage.cost_usd:.6f} USD"
        elif report.usage and report.usage.total_tokens:
            cost_note = f", {report.usage.total_tokens} tokens"
        failed_checks = sum(bool(finding.error) for finding in findings)
        failed_note = (
            f", {failed_checks} check(s) could not be completed"
            if failed_checks
            else ""
        )
        if stopped_early:
            ctx.write_event_to_stream(
                Status(
                    level="warning",
                    message=(
                        f"Stopped early after {len(findings)} of {planned} "
                        f"checks{failed_note}{cost_note}. "
                        "Results below are what completed."
                    ),
                )
            )
        else:
            ctx.write_event_to_stream(
                Status(
                    level="warning" if failed_checks else "info",
                    message=(
                        f"Scrutiny complete: {report.summary.defects_found} defect(s), "
                        f"{report.summary.needs_review} needing review, "
                        f"{report.summary.not_determined} undetermined"
                        f"{failed_note}{cost_note}"
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
        metadata["scrutiny_report_status"] = "current"
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
