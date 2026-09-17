"""Already-split filing upload: parse labeled PDFs, extract, index. No LlamaSplit."""

from __future__ import annotations

import asyncio
import io
import json
import logging
import os
from collections.abc import Mapping
from typing import Annotated, Any, cast

from llama_cloud import AsyncLlamaCloud
from llama_cloud.types.beta.extracted_data import ExtractedData, InvalidExtractionData
from pydantic import BaseModel, Field, field_validator, model_validator
from workflows import Context, Workflow, step
from workflows.events import StartEvent, StopEvent
from workflows.resource import Resource, ResourceConfig

from .clients import agent_name, get_llama_cloud_client, project_id
from .config import (
    EXTRACTED_DATA_COLLECTION,
    ExtractConfig,
    LegalExtractRecord,
    ParseConfig,
    config_identity,
    with_config_identity,
)
from .document_parts import overlay_split_documents
from .extract_record import (
    apply_extract_envelope,
    field_confidence_from_job,
    overall_confidence_from_job,
    stamp_review_status,
    stamp_source_pages,
    unwrap_extracted_record,
)
from .job_timing import (
    attach_timing,
    elapsed_seconds,
    start_timer,
    timing_payload,
    timing_status_message,
    uploaded_filename,
)
from .layout_index import (
    LAYOUT_ARTIFACT_KEY_KEY,
    LAYOUT_ARTIFACT_URL_KEY,
    coerce_page_layout,
    compact_sidecar_pages,
    download_sidecar_pages,
    dump_layout_index,
    grounded_items_url,
    merge_granular_bboxes,
    stitch_slot_layouts,
)
from .llama_usage import (
    collect_extract_usage,
    collect_parse_usage,
    summarize_llamacloud_usage,
    usage_status_message,
)
from .process_file import (
    ExtractedEvent,
    ExtractedInvalidEvent,
    ExtractJobStartedEvent,
    ParsedEvent,
    Status,
    _extract_page_markdown,
    _wait_for_parse,
    ingest_remote_file,
    normalize_org_id,
    positive_int_env,
)
from .s3_artifacts import (
    STEP_EXTRACT,
    STEP_LAYOUT,
    STEP_PARSE,
    artifact_bucket,
    set_job_context,
    upload_step_json,
)
from .split_upload import (
    PARSE_ARTIFACT_KEY_KEY,
    PARSE_ARTIFACT_URL_KEY,
    PETITION_SLOT_ID,
    SplitPartInput,
    SplitUploadError,
    build_extract_pack_markdown,
    bundle_file_hash,
    coerce_page_markdown,
    coerce_page_parts,
    display_filename,
    dump_parse_artifact,
    extract_configuration,
    extract_source_parts,
    find_part,
    parse_action_for_slot,
    slot_needs_precise_parse,
    stitch_parsed_parts,
    stitch_parsed_parts_in_order,
    validate_parts,
)
from .vector_store import (
    build_filing_chunk_text,
    build_page_records,
    pinecone_enabled,
    upsert_records,
)
from .visual.detect import detect_visual_marks
from .visual.schema import empty_visual_index
from .visual.store import (
    VISUAL_ARTIFACT_KEY_KEY,
    VISUAL_ARTIFACT_URL_KEY,
    VISUAL_SUMMARY_KEY,
    upload_visual_index,
    visual_summary,
)

logger = logging.getLogger(__name__)

DEFAULT_PARSE_CONCURRENCY = 8
DEFAULT_INGEST_CONCURRENCY = 8
DEFAULT_FAST_PARSE_TIER = "fast"
PARSE_CONCURRENCY = DEFAULT_PARSE_CONCURRENCY


def parse_concurrency() -> int:
    return positive_int_env("PARSE_CONCURRENCY", DEFAULT_PARSE_CONCURRENCY)


def ingest_concurrency() -> int:
    return positive_int_env("INGEST_CONCURRENCY", DEFAULT_INGEST_CONCURRENCY)


def fast_parse_tier() -> str:
    raw = (os.getenv("FAST_PARSE_TIER") or "").strip()
    return raw or DEFAULT_FAST_PARSE_TIER


def parse_create_kwargs(
    parse_config: ParseConfig,
    *,
    file_id: str,
    precise: bool,
) -> dict[str, Any]:
    create_kwargs: dict[str, Any] = {
        "file_id": file_id,
        "project_id": project_id,
    }
    if parse_config.configuration_id:
        create_kwargs["configuration_id"] = parse_config.configuration_id
        return merge_granular_bboxes(create_kwargs)
    create_kwargs.update(
        parse_config.model_dump(
            exclude={"configuration_id", "product_type"},
            exclude_none=True,
        )
    )
    if not precise:
        create_kwargs["tier"] = fast_parse_tier()
    return merge_granular_bboxes(create_kwargs)


class SplitPartEvent(BaseModel):
    slot_id: str
    document_parts: list[str] = Field(default_factory=list)
    file_id: str | None = None
    file_url: str | None = None
    file_hash: str | None = None
    filename: str | None = None
    document_id: str | None = None


class SplitFilesEvent(StartEvent):
    filing_type: str
    job_type: str | None = None
    org_id: str | None = None
    organization_id: str | None = None
    workspace_id: str | None = None
    user_id: str | None = None
    parts: list[SplitPartEvent]
    require_all_slots: bool = False
    fallback_file_id: str | None = None
    filename: str | None = None
    classify_split_seconds: float | None = None
    parse_scope: str = "all"
    skip_index: bool = False
    skip_extract: bool = False
    stitch_in_request_order: bool = False
    edited: bool = False
    parsed_slots: list[str] = Field(default_factory=list)
    agent_data_id: str | None = None
    special_category: str | None = None
    reuse_pages_by_slot: dict[str, dict[str, str]] = Field(default_factory=dict)
    reuse_layouts_by_slot: dict[str, dict[str, Any]] = Field(default_factory=dict)
    reuse_parse_job_ids: dict[str, str] = Field(default_factory=dict)

    @field_validator(
        "org_id", "organization_id", "workspace_id", "user_id", mode="before"
    )
    @classmethod
    def _blank_ids(cls, value: object) -> str | None:
        if value is None:
            return None
        return normalize_org_id(str(value))

    @model_validator(mode="after")
    def _sync_organization_id(self) -> SplitFilesEvent:
        org_id = self.organization_id or self.org_id
        self.organization_id = org_id
        self.org_id = org_id
        return self


class SplitFilesState(BaseModel):
    filing_type: str | None = None
    job_type: str | None = None
    org_id: str | None = None
    organization_id: str | None = None
    workspace_id: str | None = None
    user_id: str | None = None
    require_all_slots: bool = False
    parts: list[SplitPartEvent] = Field(default_factory=list)
    filename: str | None = None
    file_hash: str | None = None
    petition_file_id: str | None = None
    fallback_file_id: str | None = None
    extract_pack_file_id: str | None = None
    extract_job_id: str | None = None
    parse_job_ids: dict[str, str] = Field(default_factory=dict)
    page_markdown: dict[int, str] = Field(default_factory=dict)
    page_parts: dict[int, list[str]] = Field(default_factory=dict)
    page_layout: dict[int, dict[str, Any]] = Field(default_factory=dict)
    started_at: float | None = None
    classify_split_seconds: float | None = None
    llamacloud_jobs: list[dict[str, Any]] = Field(default_factory=list)
    parse_scope: str = "all"
    skip_index: bool = False
    skip_extract: bool = False
    stitch_in_request_order: bool = False
    edited: bool = False
    parsed_slots: list[str] = Field(default_factory=list)
    agent_data_id: str | None = None
    special_category: str | None = None
    pages_by_slot: dict[str, dict[int, str]] = Field(default_factory=dict)
    layouts_by_slot: dict[str, dict[int, dict[str, Any]]] = Field(default_factory=dict)
    parse_artifact_url: str | None = None
    parse_artifact_key: str | None = None


class ProcessSplitFilesWorkflow(Workflow):
    """Parse labeled PDFs, extract a CoreFilingRecord, store Agent Data and vectors."""

    @step()
    async def parse_files(
        self,
        event: SplitFilesEvent,
        ctx: Context[SplitFilesState],
        llama_cloud_client: Annotated[
            AsyncLlamaCloud, Resource(get_llama_cloud_client)
        ],
        parse_config: Annotated[
            ParseConfig,
            ResourceConfig(
                config_file="configs/config.json",
                path_selector="parse",
                label="Parse Settings",
                description="LlamaParse settings for JubeeX filings",
            ),
        ],
    ) -> ParsedEvent | StopEvent:
        started_at = start_timer()
        ctx.write_event_to_stream(
            Status(
                level="info",
                message=(
                    f"Ingesting {len(event.parts)} labeled file_url part(s) "
                    f"(concurrency {ingest_concurrency()})"
                ),
            )
        )
        ingested = await _ingest_labeled_parts(
            llama_cloud_client,
            event.parts,
            concurrency=ingest_concurrency(),
        )

        try:
            catalog, parts = validate_parts(
                event.filing_type,
                _as_part_inputs(ingested),
                require_all_slots=event.require_all_slots,
            )
        except SplitUploadError as exc:
            ctx.write_event_to_stream(Status(level="error", message=str(exc)))
            raise

        filename = uploaded_filename(event.filename) or display_filename(
            catalog.filing_type, parts, original=event.filename
        )
        file_hash = bundle_file_hash(parts)
        petition = find_part(parts, PETITION_SLOT_ID)
        source_parts = extract_source_parts(catalog)
        reuse_pages, reuse_layouts, reuse_jobs = _reuse_parse_maps(event, parts=parts)
        ctx.write_event_to_stream(
            Status(
                level="info",
                message=(
                    f"Parsing {len(parts)} labeled document(s) for {catalog.label} "
                    f"(concurrency {parse_concurrency()}; "
                    f"extract sources {parse_config.tier or 'agentic'}, "
                    f"others {fast_parse_tier()})"
                ),
            )
        )

        (
            pages_by_slot,
            parse_job_ids,
            layouts_by_slot,
            parse_usage,
        ) = await _parse_labeled_files(
            llama_cloud_client,
            parse_config=parse_config,
            parts=parts,
            source_parts=source_parts,
            catalog=catalog,
            parse_scope=event.parse_scope,
            reuse_slots=set(event.parsed_slots),
            reuse_pages=reuse_pages,
            reuse_layouts=reuse_layouts,
            reuse_job_ids=reuse_jobs,
            ctx=ctx,
        )
        if event.stitch_in_request_order:
            omit_empty = (event.parse_scope or "").strip().lower() == "extract_sources"
            page_markdown, page_parts = stitch_parsed_parts_in_order(
                parts, pages_by_slot, omit_empty=omit_empty
            )
            page_layout = stitch_slot_layouts(
                catalog,
                parts,
                pages_by_slot,
                layouts_by_slot,
                part_order=parts,
                omit_empty=omit_empty,
            )
        else:
            page_markdown, page_parts = stitch_parsed_parts(
                catalog, parts, pages_by_slot
            )
            page_layout = stitch_slot_layouts(
                catalog, parts, pages_by_slot, layouts_by_slot
            )
        parsed_slots = [
            item.slot_id
            for item in parts
            if pages_by_slot.get(item.slot_id) or item.slot_id in parse_job_ids
        ]

        async with ctx.store.edit_state() as state:
            state.filing_type = catalog.filing_type
            state.job_type = event.job_type
            state.require_all_slots = event.require_all_slots
            state.organization_id = event.organization_id or event.org_id
            state.org_id = state.organization_id
            state.workspace_id = event.workspace_id
            state.user_id = event.user_id
            ingested_by_slot = {item.slot_id: item for item in ingested}
            state.parts = [
                SplitPartEvent(
                    slot_id=item.slot_id,
                    document_parts=list(item.document_parts),
                    file_id=item.file_id,
                    file_hash=item.file_hash,
                    filename=item.filename,
                    document_id=(
                        item.document_id
                        or getattr(
                            ingested_by_slot.get(item.slot_id), "document_id", None
                        )
                    ),
                    file_url=getattr(
                        ingested_by_slot.get(item.slot_id), "file_url", None
                    ),
                )
                for item in parts
            ]
            state.filename = filename
            state.file_hash = file_hash
            state.petition_file_id = petition.file_id if petition else None
            state.fallback_file_id = event.fallback_file_id
            state.parse_job_ids = parse_job_ids
            state.page_markdown = page_markdown
            state.page_parts = page_parts
            state.page_layout = page_layout
            state.started_at = started_at
            state.classify_split_seconds = event.classify_split_seconds
            state.llamacloud_jobs = list(parse_usage)
            state.parse_scope = event.parse_scope
            state.skip_index = event.skip_index
            state.skip_extract = event.skip_extract
            state.stitch_in_request_order = event.stitch_in_request_order
            state.edited = event.edited
            state.parsed_slots = parsed_slots
            state.agent_data_id = event.agent_data_id
            state.special_category = event.special_category
            state.pages_by_slot = pages_by_slot
            state.layouts_by_slot = layouts_by_slot

        set_job_context(
            file_hash or filename,
            event.organization_id or event.org_id,
            event.workspace_id,
            reset=False,
        )
        parse_record = _upload_parse_artifact(
            parsed_slots=parsed_slots,
            document_order=[item.slot_id for item in parts],
            pages_by_slot=pages_by_slot,
            parse_job_ids=parse_job_ids,
            layouts_by_slot=layouts_by_slot,
            organization_id=event.organization_id or event.org_id,
            workspace_id=event.workspace_id,
            job_id=file_hash,
        )
        if parse_record:
            async with ctx.store.edit_state() as state:
                state.parse_artifact_url = parse_record.get("url")
                state.parse_artifact_key = parse_record.get("key")
        if not page_layout:
            logger.error(
                "Layout index is empty after parse for %s (%s slot(s))",
                filename,
                len(parts),
            )
            ctx.write_event_to_stream(
                Status(
                    level="warning",
                    message=(
                        "No page layout was built from parse; scrutiny "
                        "will not have highlight coordinates"
                    ),
                )
            )

        ctx.write_event_to_stream(
            Status(
                level="info",
                message=(
                    f"Parsed {len(page_markdown)} page(s) across "
                    f"{len(page_parts)} labeled part(s)"
                ),
            )
        )
        if event.skip_extract:
            payload = await _complete_index_only(
                llama_cloud_client,
                ctx=ctx,
                catalog=catalog,
                parts=parts,
                pages_by_slot=pages_by_slot,
                parse_job_ids=parse_job_ids,
                layouts_by_slot=layouts_by_slot,
                page_markdown=page_markdown,
                page_parts=page_parts,
                page_layout=page_layout,
            )
            return StopEvent(result=payload)
        return ParsedEvent()

    @step()
    async def start_extraction(
        self,
        event: ParsedEvent,
        ctx: Context[SplitFilesState],
        llama_cloud_client: Annotated[
            AsyncLlamaCloud, Resource(get_llama_cloud_client)
        ],
        extract_config: Annotated[
            ExtractConfig,
            ResourceConfig(
                config_file="configs/config.json",
                path_selector="extract-jubeex",
                label="Default Extraction Settings",
                description="Extraction config for JubeeX core filing record",
            ),
        ],
    ) -> ExtractJobStartedEvent:
        state = await ctx.store.get_state()
        if not state.filing_type:
            raise ValueError("Filing type is not set")
        catalog, _parts = validate_parts(
            state.filing_type,
            _as_part_inputs(state.parts),
            require_all_slots=state.require_all_slots,
        )
        page_markdown = coerce_page_markdown(state.page_markdown)
        page_parts = coerce_page_parts(state.page_parts)

        pack_text = build_extract_pack_markdown(
            page_markdown,
            page_parts,
            extract_source_parts(catalog),
            catalog=catalog,
        )
        extract_file_id = state.petition_file_id
        pack_file_id: str | None = None
        if pack_text:
            pack_name = f"{state.filename or catalog.filing_type}-extract-pack.md"
            uploaded = await llama_cloud_client.files.create(
                file=(
                    pack_name,
                    io.BytesIO(pack_text.encode("utf-8")),
                    "text/markdown",
                ),
                purpose="extract",
                project_id=project_id,
            )
            pack_file_id = uploaded.id
            extract_file_id = pack_file_id
            ctx.write_event_to_stream(
                Status(
                    level="info",
                    message="Extracting from labeled document parts (no LlamaSplit)",
                )
            )
        else:
            extract_file_id = extract_input_file_id(state)
            if not extract_file_id:
                raise RuntimeError(
                    "No extract pack and no source PDF are available for extraction"
                )
            ctx.write_event_to_stream(
                Status(
                    level="warning",
                    message=(
                        "Extract pack was empty; extracting from a source PDF "
                        f"({extract_file_id})"
                    ),
                )
            )

        configuration = extract_configuration(extract_config, catalog)
        if extract_config.configuration_id:
            extract_job = await llama_cloud_client.extract.create(
                file_input=extract_file_id,
                configuration_id=extract_config.configuration_id,
                project_id=project_id,
            )
        else:
            extract_job = await llama_cloud_client.extract.create(
                file_input=extract_file_id,
                configuration=cast(Any, configuration),
                project_id=project_id,
            )

        async with ctx.store.edit_state() as state:
            state.extract_pack_file_id = pack_file_id
            state.extract_job_id = extract_job.id

        return ExtractJobStartedEvent()

    @step()
    async def complete_extraction(
        self,
        event: ExtractJobStartedEvent,
        ctx: Context[SplitFilesState],
        llama_cloud_client: Annotated[
            AsyncLlamaCloud, Resource(get_llama_cloud_client)
        ],
        extract_jubeex: Annotated[
            ExtractConfig,
            ResourceConfig(
                config_file="configs/config.json",
                path_selector="extract-jubeex",
                label="JubeeX Extraction",
            ),
        ],
    ) -> StopEvent:
        state = await ctx.store.get_state()
        set_job_context(
            state.extract_job_id or state.file_hash or state.filename,
            state.organization_id or state.org_id,
            state.workspace_id,
            reset=False,
        )
        if state.extract_job_id is None:
            raise ValueError("Job ID cannot be null when waiting for its completion")
        if not state.filing_type:
            raise ValueError("Filing type is not set")
        filing_type = state.filing_type
        del extract_jubeex
        page_markdown = coerce_page_markdown(state.page_markdown)
        page_parts = coerce_page_parts(state.page_parts)
        visual_task = asyncio.create_task(_detect_visual_for_state(state, ctx))

        await llama_cloud_client.extract.wait_for_completion(
            state.extract_job_id,
            project_id=project_id,
        )
        job = await llama_cloud_client.extract.get(
            state.extract_job_id,
            expand=["extract_metadata", "usage"],
            project_id=project_id,
        )
        extract_usage = await collect_extract_usage(
            llama_cloud_client,
            state.extract_job_id,
            payload=job,
        )
        usage_summary = summarize_llamacloud_usage(
            [*(state.llamacloud_jobs or []), extract_usage]
        )
        logger.info("[Usage] %s", usage_status_message(usage_summary))
        ctx.write_event_to_stream(
            Status(level="info", message=usage_status_message(usage_summary))
        )
        visual_index = await _await_visual_index(visual_task)

        record_file_id = state.petition_file_id or state.extract_pack_file_id
        extracted_event: ExtractedEvent | ExtractedInvalidEvent
        try:
            logger.info(
                "Extracted split-upload data: %s",
                json.dumps(job.model_dump(mode="json"), indent=2, default=str),
            )
            data = ExtractedData.from_extract_job(
                job=job,
                schema=LegalExtractRecord,
                file_name=state.filename,
                file_id=record_file_id,
                file_hash=state.file_hash,
            )
            if data.metadata is None:
                data.metadata = {}
            overall = overall_confidence_from_job(job)
            data.metadata["classification"] = filing_type
            data.metadata["parse_job_ids"] = state.parse_job_ids
            data.metadata["parsed_slots"] = list(state.parsed_slots)
            data.metadata["document_order"] = [item.slot_id for item in state.parts]
            data.metadata["page_count"] = len(page_markdown)
            data.metadata["split_upload"] = True
            data.metadata["usage"] = {"llamacloud": usage_summary}
            data.metadata["extract_pack_file_id"] = state.extract_pack_file_id
            data.metadata["split_files"] = {
                item.slot_id: item.file_id for item in state.parts
            }
            data.metadata["extract_confidence"] = {
                "overall": overall,
                "fields": field_confidence_from_job(job),
            }
            data.metadata["documents"] = [
                {
                    "slot_id": item.slot_id,
                    "name": item.filename,
                    "document_id": item.document_id,
                    "file_id": item.file_id,
                }
                for item in state.parts
            ]
            if state.job_type:
                data.metadata["job_type"] = state.job_type
            if state.organization_id or state.org_id:
                org_id = state.organization_id or state.org_id
                data.metadata["organization_id"] = org_id
                data.metadata["org_id"] = org_id
            if state.workspace_id:
                data.metadata["workspace_id"] = state.workspace_id
            if state.user_id:
                data.metadata["user_id"] = state.user_id
            parse_extract_seconds = elapsed_seconds(state.started_at)
            timing = timing_payload(
                file_name=state.filename,
                classify_split_seconds=state.classify_split_seconds,
                parse_extract_seconds=parse_extract_seconds,
            )
            data.metadata["timing"] = timing
            data.metadata["config"] = config_identity()
            ctx.write_event_to_stream(
                Status(
                    level="info",
                    message=timing_status_message(
                        file_name=state.filename,
                        classify_split_seconds=state.classify_split_seconds,
                        parse_extract_seconds=parse_extract_seconds,
                    ),
                )
            )
            layout_record = upload_step_json(
                STEP_LAYOUT,
                dump_layout_index(coerce_page_layout(state.page_layout)),
                organization_id=state.organization_id or state.org_id,
                workspace_id=state.workspace_id,
            )
            if layout_record:
                data.metadata[LAYOUT_ARTIFACT_URL_KEY] = layout_record["url"]
                data.metadata[LAYOUT_ARTIFACT_KEY_KEY] = layout_record["key"]
            visual_record = upload_visual_index(
                visual_index,
                organization_id=state.organization_id or state.org_id,
                workspace_id=state.workspace_id,
                job_id=state.extract_job_id or state.file_hash,
            )
            _persist_visual_metadata(data.metadata, visual_index, visual_record)
            if visual_record:
                ctx.write_event_to_stream(
                    Status(
                        level="info",
                        message=(
                            "Visual marks JSON saved to S3. Open the filing "
                            "to download the link and verify signatures, "
                            "seals, and stamps."
                        ),
                    )
                )
            if state.parse_artifact_url:
                data.metadata[PARSE_ARTIFACT_URL_KEY] = state.parse_artifact_url
            if state.parse_artifact_key:
                data.metadata[PARSE_ARTIFACT_KEY_KEY] = state.parse_artifact_key
            elif artifact_bucket():
                logger.error(
                    "Layout artifact upload failed for %s; scrutiny will "
                    "not have highlight coordinates",
                    state.filename,
                )
            elif state.page_layout:
                logger.error(
                    "Layout index built for %s but AWS_S3_BUCKET is unset; "
                    "scrutiny will not have highlight coordinates",
                    state.filename,
                )
            extracted_event = ExtractedEvent(data=data)
        except InvalidExtractionData as exc:
            logger.exception("Error validating extracted data")
            extracted_event = ExtractedInvalidEvent(data=exc.invalid_item)
        except Exception as exc:
            logger.exception(
                "Error extracting split-upload data from %s",
                state.filename,
            )
            ctx.write_event_to_stream(
                Status(
                    level="error",
                    message=f"Error extracting data from {state.filename}: {exc}",
                )
            )
            raise

        ctx.write_event_to_stream(extracted_event)
        extracted_data = extracted_event.data
        data_dict = extracted_data.model_dump()
        meta = (
            data_dict.get("metadata")
            if isinstance(data_dict.get("metadata"), dict)
            else {}
        )
        if not isinstance(data_dict.get("metadata"), dict):
            data_dict["metadata"] = dict(meta)
            meta = data_dict["metadata"]
        meta.setdefault("usage", {"llamacloud": usage_summary})
        inner = unwrap_extracted_record(data_dict)
        stamp_source_pages(inner)
        meta = (
            data_dict.get("metadata")
            if isinstance(data_dict.get("metadata"), dict)
            else {}
        )
        src_meta = (
            extracted_data.metadata
            if isinstance(getattr(extracted_data, "metadata", None), dict)
            else {}
        )
        if src_meta:
            if not isinstance(data_dict.get("metadata"), dict):
                data_dict["metadata"] = dict(meta)
                meta = data_dict["metadata"]
            for key in (
                LAYOUT_ARTIFACT_URL_KEY,
                LAYOUT_ARTIFACT_KEY_KEY,
                PARSE_ARTIFACT_URL_KEY,
                PARSE_ARTIFACT_KEY_KEY,
            ):
                if src_meta.get(key):
                    meta[key] = src_meta[key]
        confidence = (meta.get("extract_confidence") or {}).get("overall")
        inner = apply_extract_envelope(
            inner,
            page_parts=page_parts,
            filing_type=filing_type,
            overall_confidence=confidence,
            field_confidence=(meta.get("extract_confidence") or {}).get("fields"),
        )
        data_dict["data"] = inner
        if page_parts:
            overlay_split_documents(data_dict, page_parts)
        org_id = state.organization_id or state.org_id
        if isinstance(data_dict, dict):
            data_dict.setdefault("organization_id", org_id)
            data_dict.setdefault("workspace_id", state.workspace_id)
            data_dict = stamp_review_status(data_dict)
            data_dict = attach_timing(
                data_dict,
                timing_payload(
                    file_name=state.filename,
                    classify_split_seconds=state.classify_split_seconds,
                    parse_extract_seconds=elapsed_seconds(state.started_at),
                ),
            )
        upload_step_json(
            STEP_EXTRACT,
            with_config_identity(data_dict),
            organization_id=org_id,
            workspace_id=state.workspace_id,
        )

        if extracted_data.file_hash is not None:
            delete_result = await llama_cloud_client.beta.agent_data.delete_by_query(
                deployment_name=agent_name or "_public",
                collection=EXTRACTED_DATA_COLLECTION,
                filter={"file_hash": {"eq": extracted_data.file_hash}},
            )
            if delete_result.deleted_count > 0:
                logger.info(
                    "Removed %s existing record(s) for %s",
                    delete_result.deleted_count,
                    extracted_data.file_name,
                )
        item = await llama_cloud_client.beta.agent_data.create(
            data=data_dict,
            deployment_name=agent_name or "_public",
            collection=EXTRACTED_DATA_COLLECTION,
        )
        ctx.write_event_to_stream(
            Status(
                level="info",
                message=f"Recorded extracted data for {extracted_data.file_name or ''}",
            )
        )

        if pinecone_enabled() and not state.skip_index:
            try:
                await _index_split_upload(
                    extracted_data=extracted_data,
                    item_id=str(item.id),
                    state=state,
                    filing_type=filing_type,
                    page_markdown=page_markdown,
                    page_parts=page_parts,
                    ctx=ctx,
                )
            except Exception as exc:
                logger.exception(
                    "[Pinecone] Indexing failed for %s",
                    state.filename,
                )
                ctx.write_event_to_stream(
                    Status(level="warning", message=f"Pinecone indexing failed: {exc}")
                )
        else:
            logger.info(
                "[Pinecone] Skipped indexing for %s "
                "(VECTOR_BACKEND=%s, PINECONE_API_KEY set=%s)",
                state.filename,
                os.getenv("VECTOR_BACKEND") or "pinecone",
                bool(os.getenv("PINECONE_API_KEY")),
            )

        return StopEvent(result=item.id)


async def _ingest_one_part(
    client: AsyncLlamaCloud,
    part: SplitPartEvent,
) -> SplitPartEvent:
    file_id = (part.file_id or "").strip() or None
    filename = (part.filename or "").strip() or None
    file_hash = part.file_hash
    if not file_id:
        if not (part.file_url or "").strip():
            raise SplitUploadError(f"Slot {part.slot_id!r} needs file_url or file_id")
        file_id, digest, filename = await ingest_remote_file(
            client,
            part.file_url or "",
            filename=filename,
            external_file_id=part.document_id or file_hash,
        )
        file_hash = file_hash or digest
    return part.model_copy(
        update={
            "file_id": file_id,
            "filename": filename,
            "file_hash": file_hash,
        }
    )


async def _ingest_labeled_parts(
    client: AsyncLlamaCloud,
    parts: list[SplitPartEvent],
    *,
    concurrency: int,
) -> list[SplitPartEvent]:
    semaphore = asyncio.Semaphore(max(1, concurrency))

    async def _one(part: SplitPartEvent) -> SplitPartEvent:
        async with semaphore:
            return await _ingest_one_part(client, part)

    return list(await asyncio.gather(*[_one(part) for part in parts]))


async def _parse_one_file(
    client: AsyncLlamaCloud,
    *,
    parse_config: ParseConfig,
    part: SplitPartInput,
    source_parts: set[str],
    semaphore: asyncio.Semaphore,
    ctx: Context[SplitFilesState],
    action: str = "parse",
    reuse_pages: dict[int, str] | None = None,
    reuse_layout: dict[int, dict[str, Any]] | None = None,
    reuse_job_id: str | None = None,
) -> tuple[
    str,
    dict[int, str],
    str | None,
    dict[int, dict[str, Any]],
    dict[str, Any] | None,
]:
    async with semaphore:
        label = part.filename or part.slot_id
        if action == "skip":
            ctx.write_event_to_stream(
                Status(
                    level="info",
                    message=f"Skipping parse for {label} (not an extract source)",
                )
            )
            return part.slot_id, {}, None, {}, None
        if action == "reuse":
            pages = dict(reuse_pages or {})
            layout = dict(reuse_layout or {})
            ctx.write_event_to_stream(
                Status(
                    level="info",
                    message=f"Reusing S3 markdown for {label} ({len(pages)} page(s))",
                )
            )
            return part.slot_id, pages, reuse_job_id, layout, None
        try:
            precise = slot_needs_precise_parse(part, source_parts)
            create_kwargs = parse_create_kwargs(
                parse_config,
                file_id=part.file_id or "",
                precise=precise,
            )
            tier = create_kwargs.get("tier") or "hosted"
            ctx.write_event_to_stream(
                Status(level="info", message=f"Parsing {label} ({tier})")
            )
            logger.info(
                "[Parse] %s slot=%s tier=%s",
                label,
                part.slot_id,
                tier,
            )
            parse_job = await client.parsing.create(**create_kwargs)
            await _wait_for_parse(client, parse_job.id)
            parse_result = await client.parsing.get(
                parse_job.id,
                expand=["markdown", "usage"],
                project_id=project_id,
            )
            pages = _extract_page_markdown(parse_result)
            usage_row = await collect_parse_usage(
                client,
                parse_job.id,
                payload=parse_result,
                slot_id=part.slot_id,
                pages=len(pages),
            )
            sidecar_pages = await download_sidecar_pages(
                grounded_items_url(parse_result)
            )
            layout = compact_sidecar_pages(sidecar_pages, slot_id=part.slot_id)
            ctx.write_event_to_stream(
                Status(
                    level="info",
                    message=f"Parsed {len(pages)} page(s) from {label}",
                )
            )
            return part.slot_id, pages, parse_job.id, layout, usage_row
        except Exception as exc:
            logger.exception("Parse failed for %s", label)
            ctx.write_event_to_stream(
                Status(
                    level="warning",
                    message=f"Parse failed for {label}; continuing without page text: {exc}",
                )
            )
            return part.slot_id, {}, None, {}, None


async def _parse_labeled_files(
    client: AsyncLlamaCloud,
    *,
    parse_config: ParseConfig,
    parts: list[SplitPartInput],
    source_parts: set[str],
    ctx: Context[SplitFilesState],
    catalog: Any | None = None,
    parse_scope: str = "all",
    reuse_slots: set[str] | None = None,
    reuse_pages: dict[str, dict[int, str]] | None = None,
    reuse_layouts: dict[str, dict[int, dict[str, Any]]] | None = None,
    reuse_job_ids: dict[str, str] | None = None,
) -> tuple[
    dict[str, dict[int, str]],
    dict[str, str],
    dict[str, dict[int, dict[str, Any]]],
    list[dict[str, Any]],
]:
    semaphore = asyncio.Semaphore(parse_concurrency())
    tasks = []
    for part in parts:
        action = "parse"
        if catalog is not None:
            action = parse_action_for_slot(
                part,
                catalog=catalog,
                parse_scope=parse_scope,
                reuse_slots=reuse_slots,
            )
        if action == "reuse" and not (reuse_pages or {}).get(part.slot_id):
            action = "parse"
        tasks.append(
            _parse_one_file(
                client,
                parse_config=parse_config,
                part=part,
                source_parts=source_parts,
                semaphore=semaphore,
                ctx=ctx,
                action=action,
                reuse_pages=(reuse_pages or {}).get(part.slot_id),
                reuse_layout=(reuse_layouts or {}).get(part.slot_id),
                reuse_job_id=(reuse_job_ids or {}).get(part.slot_id),
            )
        )
    results = await asyncio.gather(*tasks)
    pages_by_slot: dict[str, dict[int, str]] = {}
    parse_job_ids: dict[str, str] = {}
    layouts_by_slot: dict[str, dict[int, dict[str, Any]]] = {}
    usage_jobs: list[dict[str, Any]] = []
    for slot_id, pages, job_id, layout, usage_row in results:
        pages_by_slot[slot_id] = pages
        layouts_by_slot[slot_id] = layout
        if job_id:
            parse_job_ids[slot_id] = job_id
        if usage_row:
            usage_jobs.append(usage_row)
    return pages_by_slot, parse_job_ids, layouts_by_slot, usage_jobs


async def _index_split_upload(
    *,
    extracted_data: ExtractedData,
    item_id: str,
    state: SplitFilesState,
    filing_type: str,
    page_markdown: dict[int, str],
    page_parts: dict[int, list[str]],
    ctx: Context[SplitFilesState],
) -> None:
    base_id = extracted_data.file_hash or state.petition_file_id or item_id
    shared_meta = {
        "agent_data_id": item_id,
        "file_id": state.petition_file_id,
        "file_name": state.filename,
        "file_hash": extracted_data.file_hash,
        "petition_type": filing_type,
        "split_upload": True,
    }
    if state.organization_id or state.org_id:
        org_id = state.organization_id or state.org_id
        shared_meta["organization_id"] = org_id
        shared_meta["org_id"] = org_id
    if state.workspace_id:
        shared_meta["workspace_id"] = state.workspace_id
    if state.user_id:
        shared_meta["user_id"] = state.user_id
    ctx.write_event_to_stream(
        Status(
            level="info",
            message=(
                f"Indexing vectors in Pinecone for {state.filename} "
                "(integrated embeddings)"
            ),
        )
    )
    pinecone_items: list[dict[str, Any]] = []
    filing_payload = getattr(extracted_data, "data", None)
    summary_text = build_filing_chunk_text(
        filing_payload,
        filename=state.filename,
        filing_type=filing_type,
    )
    if summary_text.strip():
        pinecone_items.append(
            {
                "record_id": f"{base_id}:summary",
                "chunk_text": summary_text,
                "metadata": {**shared_meta, "chunk_kind": "summary"},
            }
        )
    page_records = build_page_records(
        base_id=base_id,
        page_markdown=page_markdown,
        metadata=shared_meta,
        page_parts=page_parts,
    )
    pinecone_items.extend(page_records)
    count = upsert_records(pinecone_items)
    ctx.write_event_to_stream(
        Status(
            level="info",
            message=(
                f"Indexed {count} vector(s) in Pinecone "
                f"(summary + {len(page_records)} page chunks)"
            ),
        )
    )


def extract_input_file_id(state: SplitFilesState) -> str | None:
    """Petition, compiled original, or any sliced PDF we can send to extract."""
    if state.petition_file_id:
        return state.petition_file_id
    if state.fallback_file_id:
        return state.fallback_file_id
    preferred = (
        PETITION_SLOT_ID,
        "synopsis_lod",
        "impugned_order",
        "listing_proforma",
        "cover_page",
    )
    by_slot = {item.slot_id: item.file_id for item in state.parts if item.file_id}
    for slot in preferred:
        file_id = by_slot.get(slot)
        if file_id:
            return file_id
    for item in state.parts:
        if item.file_id:
            return item.file_id
    return None


def _payload_dict(data: Any) -> dict[str, Any]:
    if hasattr(data, "model_dump"):
        dumped = data.model_dump(mode="json")
        return dumped if isinstance(dumped, dict) else {}
    if isinstance(data, dict):
        return dict(data)
    return {}


class _StoredExtract:
    def __init__(self, payload: Mapping[str, Any]) -> None:
        self.file_hash = payload.get("file_hash")
        self.file_name = payload.get("file_name")
        self.data = payload.get("data")
        meta = payload.get("metadata")
        self.metadata = dict(meta) if isinstance(meta, dict) else {}


def _reuse_parse_maps(
    event: SplitFilesEvent,
    *,
    parts: list[SplitPartInput],
) -> tuple[
    dict[str, dict[int, str]],
    dict[str, dict[int, dict[str, Any]]],
    dict[str, str],
]:
    del parts
    if event.edited or (event.parse_scope or "all") != "unparsed":
        return {}, {}, {}
    pages = {
        slot: coerce_page_markdown(raw if isinstance(raw, dict) else {})
        for slot, raw in (event.reuse_pages_by_slot or {}).items()
    }
    layouts: dict[str, dict[int, dict[str, Any]]] = {}
    for slot, raw in (event.reuse_layouts_by_slot or {}).items():
        slot_layout: dict[int, dict[str, Any]] = {}
        if isinstance(raw, dict):
            for page, payload in raw.items():
                try:
                    number = int(page)
                except (TypeError, ValueError):
                    continue
                if isinstance(payload, dict):
                    slot_layout[number] = payload
        layouts[slot] = slot_layout
    return pages, layouts, dict(event.reuse_parse_job_ids or {})


def _persist_visual_metadata(
    meta: dict[str, Any],
    visual_index: Mapping[str, Any],
    visual_record: dict[str, str] | None,
) -> None:
    meta[VISUAL_SUMMARY_KEY] = visual_summary(visual_index)
    if not visual_record:
        return
    meta[VISUAL_ARTIFACT_URL_KEY] = visual_record["url"]
    meta[VISUAL_ARTIFACT_KEY_KEY] = visual_record["key"]


async def _detect_visual_for_state(
    state: SplitFilesState,
    ctx: Context[SplitFilesState],
) -> dict[str, Any]:
    async def on_log(message: str) -> None:
        ctx.write_event_to_stream(Status(level="info", message=message))

    ctx.write_event_to_stream(
        Status(
            level="info",
            message="Detecting signatures, seals, and stamps on formality pages",
        )
    )
    try:
        index = await detect_visual_marks(
            parts=state.parts,
            pages_by_slot=state.pages_by_slot,
            page_parts=coerce_page_parts(state.page_parts),
            page_markdown=coerce_page_markdown(state.page_markdown),
            omit_empty=(state.parse_scope or "").strip().lower() == "extract_sources",
            on_log=on_log,
        )
    except Exception:
        logger.exception("Visual ink detection failed for %s", state.filename)
        ctx.write_event_to_stream(
            Status(
                level="warning",
                message="Visual ink detection failed; scrutiny will have no mark boxes",
            )
        )
        return empty_visual_index(status="error", error="visual_detection_failed")
    mark_count = len((index or {}).get("marks") or [])
    page_count = len((index or {}).get("pages") or [])
    failed_count = len((index or {}).get("failures") or [])
    ctx.write_event_to_stream(
        Status(
            level="info",
            message=(
                f"Stored {mark_count} visual mark(s) from {page_count} "
                f"formality page(s)"
                + (f", {failed_count} failed" if failed_count else "")
            ),
        )
    )
    return index


async def _await_visual_index(task: asyncio.Task[dict[str, Any]]) -> dict[str, Any]:
    try:
        return await task
    except Exception:
        logger.exception("Visual ink detection task failed")
        return empty_visual_index(status="error", error="visual_detection_failed")


def _upload_parse_artifact(
    *,
    parsed_slots: list[str],
    document_order: list[str],
    pages_by_slot: dict[str, dict[int, str]],
    parse_job_ids: dict[str, str],
    layouts_by_slot: dict[str, dict[int, dict[str, Any]]],
    organization_id: str | None,
    workspace_id: str | None,
    job_id: str | None,
) -> dict[str, str] | None:
    payload = dump_parse_artifact(
        parsed_slots=parsed_slots,
        document_order=document_order,
        pages_by_slot=pages_by_slot,
        parse_job_ids=parse_job_ids,
        layouts_by_slot=layouts_by_slot,
    )
    if organization_id:
        payload["organization_id"] = organization_id
    if workspace_id:
        payload["workspace_id"] = workspace_id
    return upload_step_json(
        STEP_PARSE,
        with_config_identity(payload),
        organization_id=organization_id,
        workspace_id=workspace_id,
        job_id=job_id,
    )


async def _complete_index_only(
    client: AsyncLlamaCloud,
    *,
    ctx: Context[SplitFilesState],
    catalog: Any,
    parts: list[SplitPartInput],
    pages_by_slot: dict[str, dict[int, str]],
    parse_job_ids: dict[str, str],
    layouts_by_slot: dict[str, dict[int, dict[str, Any]]],
    page_markdown: dict[int, str],
    page_parts: dict[int, list[str]],
    page_layout: dict[int, dict[str, Any]],
) -> dict[str, Any]:
    del catalog, pages_by_slot, layouts_by_slot
    state = await ctx.store.get_state()
    agent_data_id = (state.agent_data_id or "").strip()
    if not agent_data_id:
        raise ValueError("agent_data_id is required to index without re-extracting")
    item = await client.beta.agent_data.get(agent_data_id)
    data_dict = _payload_dict(getattr(item, "data", None))
    meta = data_dict.get("metadata")
    if not isinstance(meta, dict):
        meta = {}
        data_dict["metadata"] = meta
    meta["parse_job_ids"] = parse_job_ids
    meta["parsed_slots"] = list(state.parsed_slots)
    meta["document_order"] = [item.slot_id for item in parts]
    meta["page_count"] = len(page_markdown)
    meta["split_files"] = {
        part.slot_id: part.file_id for part in state.parts if part.file_id
    }
    if state.parse_artifact_url:
        meta[PARSE_ARTIFACT_URL_KEY] = state.parse_artifact_url
    if state.parse_artifact_key:
        meta[PARSE_ARTIFACT_KEY_KEY] = state.parse_artifact_key
    layout_record = upload_step_json(
        STEP_LAYOUT,
        dump_layout_index(coerce_page_layout(page_layout)),
        organization_id=state.organization_id or state.org_id,
        workspace_id=state.workspace_id,
        job_id=state.file_hash,
    )
    if layout_record:
        meta[LAYOUT_ARTIFACT_URL_KEY] = layout_record["url"]
        meta[LAYOUT_ARTIFACT_KEY_KEY] = layout_record["key"]
    visual_index = await _detect_visual_for_state(state, ctx)
    visual_record = upload_visual_index(
        visual_index,
        organization_id=state.organization_id or state.org_id,
        workspace_id=state.workspace_id,
        job_id=state.file_hash,
    )
    _persist_visual_metadata(meta, visual_index, visual_record)
    if visual_record:
        ctx.write_event_to_stream(
            Status(
                level="info",
                message=(
                    "Visual marks JSON saved to S3. Open the filing to "
                    "download the link and verify signatures, seals, and stamps."
                ),
            )
        )
    if page_parts:
        overlay_split_documents(data_dict, page_parts)
    await client.beta.agent_data.update(agent_data_id, data=data_dict)
    extracted = _StoredExtract(data_dict)
    if pinecone_enabled() and not state.skip_index:
        try:
            await _index_split_upload(
                extracted_data=extracted,  # type: ignore[arg-type]
                item_id=agent_data_id,
                state=state,
                filing_type=state.filing_type or "",
                page_markdown=page_markdown,
                page_parts=page_parts,
                ctx=ctx,
            )
        except Exception as exc:
            logger.exception("[Pinecone] Indexing failed for %s", state.filename)
            ctx.write_event_to_stream(
                Status(level="warning", message=f"Pinecone indexing failed: {exc}")
            )
    else:
        logger.info(
            "[Pinecone] Skipped indexing for %s on index-only path",
            state.filename,
        )
    ctx.write_event_to_stream(
        Status(
            level="info",
            message=f"Indexed filing {agent_data_id} without re-extracting",
        )
    )
    report = await _run_nested_scrutiny(ctx, agent_data_id=agent_data_id)
    return {"agent_data_id": agent_data_id, "report": report}


async def _run_nested_scrutiny(
    ctx: Context[SplitFilesState],
    *,
    agent_data_id: str,
) -> dict[str, Any]:
    from .scrutiny_workflow import (
        ScrutinyAfterIndexError,
        ScrutinyEvent,
        ScrutinyWorkflow,
    )

    state = await ctx.store.get_state()
    ctx.write_event_to_stream(
        Status(
            level="info",
            message=f"Running scrutiny defects for {agent_data_id}",
        )
    )
    nested = ScrutinyWorkflow(timeout=None)
    handler = nested.run(
        start_event=ScrutinyEvent(
            agent_data_id=agent_data_id,
            file_hash=state.file_hash,
            organization_id=state.organization_id or state.org_id,
            workspace_id=state.workspace_id,
            special_category=state.special_category,
        )
    )
    try:
        async for ev in handler.stream_events():
            ctx.write_event_to_stream(ev)
        result = await handler
    except ScrutinyAfterIndexError:
        raise
    except Exception as exc:
        logger.exception("Scrutiny after index failed for %s", agent_data_id)
        raise ScrutinyAfterIndexError(agent_data_id) from exc
    report = getattr(result, "report", None)
    if report is None and isinstance(result, dict):
        report = result.get("report")
    if hasattr(report, "model_dump"):
        dumped = report.model_dump(mode="json")
        if isinstance(dumped, dict):
            return dumped
    if isinstance(report, dict):
        return report
    logger.error("Scrutiny after index returned no report for %s", agent_data_id)
    raise ScrutinyAfterIndexError(agent_data_id)


def _as_part_inputs(parts: list[SplitPartEvent]) -> list[SplitPartInput]:
    return [
        SplitPartInput(
            slot_id=item.slot_id,
            file_id=item.file_id or "",
            document_parts=tuple(item.document_parts),
            file_hash=item.file_hash,
            filename=item.filename,
            document_id=item.document_id,
        )
        for item in parts
    ]


workflow = ProcessSplitFilesWorkflow(timeout=None)
