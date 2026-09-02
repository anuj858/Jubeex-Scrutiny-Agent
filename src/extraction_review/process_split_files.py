"""Already-split filing upload: parse labeled PDFs, extract, index. No LlamaSplit."""

from __future__ import annotations

import asyncio
import io
import json
import logging
import os
from typing import Annotated, Any, cast

from llama_cloud import AsyncLlamaCloud
from llama_cloud.types.beta.extracted_data import ExtractedData, InvalidExtractionData
from llama_cloud.types.configuration_response import ExtractV2Parameters
from pydantic import BaseModel, Field
from workflows import Context, Workflow, step
from workflows.events import StartEvent, StopEvent
from workflows.resource import Resource, ResourceConfig

from .clients import agent_name, get_llama_cloud_client, project_id
from .config import (
    EXTRACTED_DATA_COLLECTION,
    ExtractConfig,
    ParseConfig,
    get_extraction_schema,
)
from .document_parts import overlay_split_documents
from .process_file import (
    DISCRIMINATOR_FIELD,
    ExtractedEvent,
    ExtractedInvalidEvent,
    ExtractJobStartedEvent,
    ParsedEvent,
    Status,
    _extract_page_markdown,
    _wait_for_parse,
)
from .split_upload import (
    PETITION_SLOT_ID,
    SplitPartInput,
    SplitUploadError,
    build_extract_pack_markdown,
    bundle_file_hash,
    coerce_page_markdown,
    coerce_page_parts,
    display_filename,
    extract_configuration,
    extract_source_parts,
    find_part,
    stitch_parsed_parts,
    validate_parts,
)
from .vector_store import (
    build_filing_chunk_text,
    build_page_records,
    pinecone_enabled,
    upsert_records,
)

logger = logging.getLogger(__name__)

PARSE_CONCURRENCY = 4


class SplitPartEvent(BaseModel):
    slot_id: str
    document_parts: list[str] = Field(default_factory=list)
    file_id: str
    file_hash: str | None = None
    filename: str | None = None


class SplitFilesEvent(StartEvent):
    filing_type: str
    parts: list[SplitPartEvent]


class SplitFilesState(BaseModel):
    filing_type: str | None = None
    parts: list[SplitPartEvent] = Field(default_factory=list)
    filename: str | None = None
    file_hash: str | None = None
    petition_file_id: str | None = None
    extract_pack_file_id: str | None = None
    extract_job_id: str | None = None
    parse_job_ids: dict[str, str] = Field(default_factory=dict)
    page_markdown: dict[int, str] = Field(default_factory=dict)
    page_parts: dict[int, list[str]] = Field(default_factory=dict)


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
    ) -> ParsedEvent:
        try:
            catalog, parts = validate_parts(
                event.filing_type, _as_part_inputs(event.parts)
            )
        except SplitUploadError as exc:
            ctx.write_event_to_stream(Status(level="error", message=str(exc)))
            raise

        filename = display_filename(catalog.filing_type, parts)
        file_hash = bundle_file_hash(parts)
        petition = find_part(parts, PETITION_SLOT_ID)
        ctx.write_event_to_stream(
            Status(
                level="info",
                message=(
                    f"Parsing {len(parts)} labeled document(s) for {catalog.label}"
                ),
            )
        )

        pages_by_slot, parse_job_ids = await _parse_labeled_files(
            llama_cloud_client,
            parse_config=parse_config,
            parts=parts,
            ctx=ctx,
        )
        page_markdown, page_parts = stitch_parsed_parts(catalog, parts, pages_by_slot)

        async with ctx.store.edit_state() as state:
            state.filing_type = catalog.filing_type
            state.parts = [
                SplitPartEvent(
                    slot_id=item.slot_id,
                    document_parts=list(item.document_parts),
                    file_id=item.file_id,
                    file_hash=item.file_hash,
                    filename=item.filename,
                )
                for item in parts
            ]
            state.filename = filename
            state.file_hash = file_hash
            state.petition_file_id = petition.file_id if petition else None
            state.parse_job_ids = parse_job_ids
            state.page_markdown = page_markdown
            state.page_parts = page_parts

        ctx.write_event_to_stream(
            Status(
                level="info",
                message=(
                    f"Parsed {len(page_markdown)} page(s) across "
                    f"{len(page_parts)} labeled part(s)"
                ),
            )
        )
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
            state.filing_type, _as_part_inputs(state.parts)
        )
        page_markdown = coerce_page_markdown(state.page_markdown)
        page_parts = coerce_page_parts(state.page_parts)

        pack_text = build_extract_pack_markdown(
            page_markdown,
            page_parts,
            extract_source_parts(catalog),
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
        elif extract_file_id:
            ctx.write_event_to_stream(
                Status(
                    level="warning",
                    message="Extract pack was empty; extracting from the Petition PDF",
                )
            )
        else:
            raise RuntimeError(
                "No extract pack and no Petition file are available for extraction"
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
        if state.extract_job_id is None:
            raise ValueError("Job ID cannot be null when waiting for its completion")
        filing_type = state.filing_type or "other"
        extract_config = extract_jubeex
        page_markdown = coerce_page_markdown(state.page_markdown)
        page_parts = coerce_page_parts(state.page_parts)

        await llama_cloud_client.extract.wait_for_completion(
            state.extract_job_id,
            project_id=project_id,
        )
        job = await llama_cloud_client.extract.get(
            state.extract_job_id,
            expand=["extract_metadata"],
            project_id=project_id,
        )

        record_file_id = state.petition_file_id or state.extract_pack_file_id
        extracted_event: ExtractedEvent | ExtractedInvalidEvent
        try:
            logger.info(
                "Extracted split-upload data: %s",
                json.dumps(job.model_dump(mode="json"), indent=2, default=str),
            )
            if extract_config.configuration_id:
                config_resp = await llama_cloud_client.configurations.retrieve(
                    extract_config.configuration_id,
                    project_id=project_id,
                )
                params = config_resp.parameters
                if not isinstance(params, ExtractV2Parameters):
                    raise ValueError(
                        f"Configuration {extract_config.configuration_id} is not extract_v2"
                    )
                schema_class = get_extraction_schema(
                    dict(params.data_schema),
                    discriminator_field=DISCRIMINATOR_FIELD,
                    discriminator_value=filing_type,
                )
            else:
                schema_class = get_extraction_schema(
                    dict(extract_config.data_schema),
                    discriminator_field=DISCRIMINATOR_FIELD,
                    discriminator_value=filing_type,
                )
            data = ExtractedData.from_extract_job(
                job=job,
                schema=schema_class,
                file_name=state.filename,
                file_id=record_file_id,
                file_hash=state.file_hash,
            )
            if data.metadata is None:
                data.metadata = {}
            data.metadata["classification"] = filing_type
            data.metadata["parse_job_ids"] = state.parse_job_ids
            data.metadata["page_count"] = len(page_markdown)
            data.metadata["split_upload"] = True
            data.metadata["extract_pack_file_id"] = state.extract_pack_file_id
            data.metadata["split_files"] = {
                item.slot_id: item.file_id for item in state.parts
            }
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
        if page_parts:
            overlay_split_documents(data_dict, page_parts)

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

        if pinecone_enabled():
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


async def _parse_one_file(
    client: AsyncLlamaCloud,
    *,
    parse_config: ParseConfig,
    part: SplitPartInput,
    semaphore: asyncio.Semaphore,
    ctx: Context[SplitFilesState],
) -> tuple[str, dict[int, str], str | None]:
    async with semaphore:
        label = part.filename or part.slot_id
        try:
            ctx.write_event_to_stream(Status(level="info", message=f"Parsing {label}"))
            create_kwargs: dict[str, Any] = {
                "file_id": part.file_id,
                "project_id": project_id,
            }
            if parse_config.configuration_id:
                create_kwargs["configuration_id"] = parse_config.configuration_id
            else:
                create_kwargs.update(
                    parse_config.model_dump(
                        exclude={"configuration_id", "product_type"},
                        exclude_none=True,
                    )
                )
            parse_job = await client.parsing.create(**create_kwargs)
            await _wait_for_parse(client, parse_job.id)
            parse_result = await client.parsing.get(
                parse_job.id,
                expand=["markdown"],
                project_id=project_id,
            )
            pages = _extract_page_markdown(parse_result)
            ctx.write_event_to_stream(
                Status(
                    level="info",
                    message=f"Parsed {len(pages)} page(s) from {label}",
                )
            )
            return part.slot_id, pages, parse_job.id
        except Exception as exc:
            logger.exception("Parse failed for %s", label)
            ctx.write_event_to_stream(
                Status(
                    level="warning",
                    message=f"Parse failed for {label}; continuing without page text: {exc}",
                )
            )
            return part.slot_id, {}, None


async def _parse_labeled_files(
    client: AsyncLlamaCloud,
    *,
    parse_config: ParseConfig,
    parts: list[SplitPartInput],
    ctx: Context[SplitFilesState],
) -> tuple[dict[str, dict[int, str]], dict[str, str]]:
    semaphore = asyncio.Semaphore(PARSE_CONCURRENCY)
    results = await asyncio.gather(
        *[
            _parse_one_file(
                client,
                parse_config=parse_config,
                part=part,
                semaphore=semaphore,
                ctx=ctx,
            )
            for part in parts
        ]
    )
    pages_by_slot: dict[str, dict[int, str]] = {}
    parse_job_ids: dict[str, str] = {}
    for slot_id, pages, job_id in results:
        pages_by_slot[slot_id] = pages
        if job_id:
            parse_job_ids[slot_id] = job_id
    return pages_by_slot, parse_job_ids


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


def _as_part_inputs(parts: list[SplitPartEvent]) -> list[SplitPartInput]:
    return [
        SplitPartInput(
            slot_id=item.slot_id,
            file_id=item.file_id,
            document_parts=tuple(item.document_parts),
            file_hash=item.file_hash,
            filename=item.filename,
        )
        for item in parts
    ]


workflow = ProcessSplitFilesWorkflow(timeout=None)
