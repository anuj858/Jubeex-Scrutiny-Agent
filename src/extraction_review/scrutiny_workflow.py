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
    apply_retrieval_policy,
    apply_status_policy,
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
from .document_parts import (
    max_chunks_for_defect,
    missing_required_parts,
    parts_named_in_where_to_look,
    preferred_parts_for_defect,
    select_chunks_for_defect,
    slice_record_for_defect,
)

logger = logging.getLogger(__name__)

DEFAULT_CONCURRENCY = 3


class ScrutinyEvent(StartEvent):
    agent_data_id: str | None = None
    file_hash: str | None = None


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
    response = apply_status_policy(response)
    response = apply_retrieval_policy(defect, response, chunks)
    if response.status != "defect_found":
        response.suggested_fix = None
        response.fix_rationale = None
    return response


async def _run_defect(
    defect: Defect,
    *,
    catalogue: Catalogue,
    record: dict[str, Any] | None,
    chunks: list[dict[str, Any]],
    file_name: str | None,
    filing_type: str | None,
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
            file_name=file_name,
            catalogue=catalogue,
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
    )


async def _chunks_for_defect(
    defect: Defect,
    *,
    file_hash: str | None,
    max_chunks: int,
    use_pinecone: bool,
) -> list[dict[str, Any]]:
    """Fetch this defect's excerpts. Runs inside the concurrency semaphore."""
    if not use_pinecone or not file_hash:
        return []

    queries = build_evidence_queries(defect)
    page_budget = max_chunks_for_defect(defect, ceiling=max_chunks)
    gather_cap = max(
        max_chunks,
        len(queries) * 3,
        len(parts_named_in_where_to_look(defect) or preferred_parts_for_defect(defect))
        * 4,
    )
    try:
        pool = await asyncio.to_thread(
            gather_filing_evidence,
            queries,
            file_hash=file_hash,
            max_chunks=gather_cap,
        )
    except Exception as e:
        logger.warning(
            "[Scrutiny] Pinecone retrieve failed for %s: %s",
            defect.check_id,
            e,
        )
        return []

    return select_chunks_for_defect(pool, defect, max_chunks=page_budget)


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
            if (
                isinstance(result, DefectFinding)
                and result.check_id not in seen
            ):
                seen.add(result.check_id)
                findings.append(result)
                added_after_cancel = True
        if on_update is not None and (stopped_early or added_after_cancel):
            await on_update(list(findings), stopped_early)
    return findings, stopped_early


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

        concurrency = _int_env("SCRUTINY_CONCURRENCY", DEFAULT_CONCURRENCY)
        max_chunks = scrutiny_max_chunks()
        use_pinecone = pinecone_enabled() and bool(file_hash)

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
        elif not file_hash:
            ctx.write_event_to_stream(
                Status(
                    level="warning",
                    message=(
                        "No file hash is available, so checks will run against "
                        "the extracted record only. Coverage will be incomplete."
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

        ctx.write_event_to_stream(
            Status(
                level="info",
                message=(
                    f"Checking {file_name or 'filing'} against "
                    f"{len(defects)} defect(s) "
                    f"({concurrency} at a time)"
                ),
            )
        )

        planned = len(defects)

        def build_report(
            current: list[DefectFinding], *, stopped_early: bool
        ) -> ScrutinyReport:
            snapshot = sorted(current, key=lambda f: f.serial_no)
            return ScrutinyReport(
                catalogue_id=catalogue.catalogue_id,
                catalogue_version=catalogue.catalogue_version,
                agent_data_id=str(getattr(item, "id", "") or "")
                or event.agent_data_id,
                file_hash=file_hash,
                file_name=file_name,
                petition_type=filing_type,
                model=openrouter_model(),
                disclaimer=catalogue.disclaimer,
                findings=snapshot,
                summary=summarize(snapshot),
                usage=summarize_usage(snapshot, model=openrouter_model()),
                planned_checks=planned,
                stopped_early=stopped_early,
            )

        async def publish(
            current: list[DefectFinding], stopped_early: bool
        ) -> None:
            report = build_report(current, stopped_early=stopped_early)
            ctx.write_event_to_stream(
                ScrutinyPartial(
                    report=report,
                    completed=len(current),
                    total=planned,
                    stopped_early=stopped_early,
                )
            )
            await self._persist(llama_cloud_client, item, payload, report, ctx)

        async def run_one(defect: Defect) -> DefectFinding:
            ctx.write_event_to_stream(
                Status(
                    level="info",
                    message=f"Checking {defect.check_id} — S.No. {defect.serial_no}",
                )
            )
            try:
                chunks = await _chunks_for_defect(
                    defect,
                    file_hash=file_hash,
                    max_chunks=max_chunks,
                    use_pinecone=use_pinecone,
                )
                finding = await _run_defect(
                    defect,
                    catalogue=catalogue,
                    record=record,
                    chunks=chunks,
                    file_name=file_name,
                    filing_type=filing_type,
                )
            except asyncio.CancelledError:
                raise
            except LLMError as e:
                logger.exception("[Scrutiny] %s failed", defect.check_id)
                ctx.write_event_to_stream(
                    Status(
                        level="error",
                        message=(
                            f"{defect.check_id} failed; stopping remaining "
                            f"checks so no more tokens are used. {e}"
                        ),
                    )
                )
                return failed_finding(defect, str(e), usage=e.usage)
            except Exception as e:
                logger.exception("[Scrutiny] %s failed", defect.check_id)
                ctx.write_event_to_stream(
                    Status(
                        level="error",
                        message=(
                            f"{defect.check_id} failed; stopping remaining "
                            f"checks so no more tokens are used. {e}"
                        ),
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

        findings, stopped_early = await collect_defect_findings(
            defects,
            run_one,
            concurrency=concurrency,
            on_update=publish,
        )
        report = build_report(findings, stopped_early=stopped_early)

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
