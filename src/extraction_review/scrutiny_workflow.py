"""Registry defect scrutiny for an approved filing.

Runs the enabled defects from the SCI catalogue against a document that has
already been parsed, extracted and indexed. Evidence comes from the structured
record in Agent Data plus page chunks retrieved from Pinecone for that document.
"""

from __future__ import annotations

import asyncio
import logging
import os
from typing import Annotated, Any, Literal

from llama_cloud import AsyncLlamaCloud
from pydantic import BaseModel
from workflows import Context, Workflow, step
from workflows.events import Event, StartEvent, StopEvent
from workflows.resource import Resource

from .clients import agent_name, get_llama_cloud_client
from .config import EXTRACTED_DATA_COLLECTION
from .llm import LLMError, call_structured, openrouter_enabled, openrouter_model
from .scrutiny.prompts import (
    build_defect_prompt,
    build_evidence_queries,
    build_system_prompt,
)
from .scrutiny.rules import (
    Catalogue,
    Defect,
    defects_for_filing_type,
    enabled_defect_ids,
    get_catalogue,
)
from .scrutiny.schema import (
    Coverage,
    DefectFinding,
    DefectResponse,
    ScrutinyReport,
    build_finding,
    failed_finding,
    summarize,
    summarize_usage,
)
from .vector_store import gather_filing_evidence, pinecone_enabled

logger = logging.getLogger(__name__)

DEFAULT_CONCURRENCY = 3
DEFAULT_TOP_K = 8
DEFAULT_MAX_CHUNKS = 24


class ScrutinyEvent(StartEvent):
    agent_data_id: str | None = None
    file_hash: str | None = None


class Status(Event):
    level: Literal["info", "warning", "error"]
    message: str


class ScrutinyResponse(StopEvent):
    report: ScrutinyReport


class ScrutinyState(BaseModel):
    agent_data_id: str | None = None
    file_hash: str | None = None


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
        "Could not load the filing record. Provide a valid agent_data_id or file_hash."
    )


def _sanitize_response(defect: Defect, response: DefectResponse) -> DefectResponse:
    """Keep the catalogue check_id and drop fixes unless a defect was found."""
    if response.check_id != defect.check_id:
        logger.warning(
            "[Scrutiny] Model returned check_id %s for %s; correcting",
            response.check_id,
            defect.check_id,
        )
        response.check_id = defect.check_id
    if response.status != "defect_found":
        response.suggested_fix = None
        response.fix_rationale = None
    return response


async def _run_defect(
    defect: Defect,
    *,
    catalogue: Catalogue,
    record: dict[str, Any] | None,
    file_hash: str | None,
    file_name: str | None,
    top_k: int,
    max_chunks: int,
) -> DefectFinding:
    chunks: list[dict[str, Any]] = []
    if file_hash and pinecone_enabled():
        queries = build_evidence_queries(defect)
        chunks = await asyncio.to_thread(
            gather_filing_evidence,
            queries,
            file_hash=file_hash,
            top_k=top_k,
            max_chunks=max_chunks,
        )

    pages = sorted({c["page"] for c in chunks if c.get("page") is not None})
    coverage = Coverage(
        chunks_reviewed=len(chunks),
        pages_reviewed=pages,
        structured_record_available=bool(record),
        evidence_complete=bool(chunks) and bool(record),
    )

    raw, usage = await call_structured(
        system_prompt=build_system_prompt(catalogue, defect),
        user_prompt=build_defect_prompt(
            defect,
            record=record,
            chunks=chunks,
            file_name=file_name,
            catalogue=catalogue,
        ),
        response_model=DefectResponse,
    )
    response = _sanitize_response(defect, raw)

    return build_finding(
        defect,
        response,
        evidence_ids=[c["record_id"] for c in chunks if c.get("record_id")],
        coverage=coverage,
        usage=usage,
    )


class ScrutinyWorkflow(Workflow):
    """Check an approved filing against the SCI registry defect catalogue."""

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
            raise ValueError(
                "OPENROUTER_API_KEY is not set, so defect checks cannot run"
            )

        async with ctx.store.edit_state() as state:
            state.agent_data_id = event.agent_data_id
            state.file_hash = event.file_hash

        item = await _load_item(
            llama_cloud_client,
            agent_data_id=event.agent_data_id,
            file_hash=event.file_hash,
        )

        payload: dict[str, Any] = dict(getattr(item, "data", None) or {})
        review_status = payload.get("status")
        file_name = payload.get("file_name")
        file_hash = payload.get("file_hash") or event.file_hash
        record = payload.get("data") or {}
        metadata = payload.get("metadata") or {}
        filing_type = metadata.get("classification") or record.get("petition_type")

        if review_status != "approved":
            raise ValueError(
                f"Scrutiny only runs on approved filings; "
                f"{file_name or 'this document'} is '{review_status}'."
            )

        catalogue = get_catalogue()
        defects = defects_for_filing_type(filing_type)

        if not defects:
            covered = sorted({d.main_category for d in catalogue.defects})
            raise ValueError(
                f"No defect checks apply to petition type '{filing_type}'. "
                f"The catalogue currently covers {', '.join(covered) or 'no categories'} "
                f"(enabled checks: {', '.join(enabled_defect_ids())})."
            )

        if not pinecone_enabled():
            ctx.write_event_to_stream(
                Status(
                    level="warning",
                    message=(
                        "Pinecone is disabled, so checks will run against the "
                        "extracted record only. Coverage will be incomplete."
                    ),
                )
            )

        ctx.write_event_to_stream(
            Status(
                level="info",
                message=(
                    f"Checking {file_name or 'filing'} against "
                    f"{len(defects)} defect(s): "
                    f"{', '.join(d.check_id for d in defects)}"
                ),
            )
        )

        semaphore = asyncio.Semaphore(
            _int_env("SCRUTINY_CONCURRENCY", DEFAULT_CONCURRENCY)
        )
        top_k = _int_env("SCRUTINY_TOP_K", DEFAULT_TOP_K)
        max_chunks = _int_env("SCRUTINY_MAX_CHUNKS", DEFAULT_MAX_CHUNKS)

        async def guarded(defect: Defect) -> DefectFinding:
            async with semaphore:
                ctx.write_event_to_stream(
                    Status(
                        level="info",
                        message=f"Checking {defect.check_id} — S.No. {defect.serial_no}",
                    )
                )
                try:
                    finding = await _run_defect(
                        defect,
                        catalogue=catalogue,
                        record=record,
                        file_hash=file_hash,
                        file_name=file_name,
                        top_k=top_k,
                        max_chunks=max_chunks,
                    )
                except LLMError as e:
                    logger.exception("[Scrutiny] %s failed", defect.check_id)
                    ctx.write_event_to_stream(
                        Status(
                            level="error",
                            message=f"{defect.check_id} could not be completed: {e}",
                        )
                    )
                    return failed_finding(defect, str(e), usage=e.usage)
                except Exception as e:
                    logger.exception("[Scrutiny] %s failed", defect.check_id)
                    ctx.write_event_to_stream(
                        Status(
                            level="error",
                            message=f"{defect.check_id} could not be completed: {e}",
                        )
                    )
                    return failed_finding(defect, str(e))

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

        findings = list(await asyncio.gather(*(guarded(d) for d in defects)))
        findings.sort(key=lambda f: f.serial_no)

        report = ScrutinyReport(
            catalogue_id=catalogue.catalogue_id,
            catalogue_version=catalogue.catalogue_version,
            agent_data_id=str(getattr(item, "id", "") or "") or event.agent_data_id,
            file_hash=file_hash,
            file_name=file_name,
            petition_type=filing_type,
            model=openrouter_model(),
            disclaimer=catalogue.disclaimer,
            findings=findings,
            summary=summarize(findings),
            usage=summarize_usage(findings, model=openrouter_model()),
        )

        await self._persist(llama_cloud_client, item, payload, report, ctx)

        cost_note = ""
        if report.usage and report.usage.cost_usd is not None:
            cost_note = f", OpenRouter {report.usage.cost_usd:.6f} USD"
        elif report.usage and report.usage.total_tokens:
            cost_note = f", {report.usage.total_tokens} tokens"
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
                "[Scrutiny] Saved report on extraction item %s (%s)",
                item_id,
                report.file_name,
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
