import asyncio
import json
import logging
import os
from typing import Annotated, Any, Literal

from llama_cloud import AsyncLlamaCloud
from llama_cloud.types.beta.extracted_data import ExtractedData, InvalidExtractionData
from llama_cloud.types.configuration_response import ExtractV2Parameters
from pydantic import BaseModel, Field
from workflows import Context, Workflow, step
from workflows.events import Event, StartEvent, StopEvent
from workflows.resource import Resource, ResourceConfig

from .clients import agent_name, get_llama_cloud_client, project_id
from .config import (
    EXTRACTED_DATA_COLLECTION,
    ClassifyConfig,
    ExtractConfig,
    ParseConfig,
    SplitConfig,
    get_extraction_schema,
)
from .document_parts import overlay_split_documents, page_parts_from_split
from .vector_store import (
    build_filing_chunk_text,
    build_page_records,
    pinecone_enabled,
    upsert_records,
)

logger = logging.getLogger(__name__)

DISCRIMINATOR_FIELD = "petition_type"

CLASSIFY_POLL_INTERVAL_S = 1.0
CLASSIFY_POLL_MAX_S = 300.0
PARSE_POLL_INTERVAL_S = 2.0
PARSE_POLL_MAX_S = 300.0
SPLIT_POLL_INTERVAL_S = 2.0
SPLIT_POLL_MAX_S = 180.0
PARSE_DONE_STATUSES = frozenset({"COMPLETED", "SUCCESS"})
PARSE_FAILED_STATUSES = frozenset({"FAILED", "CANCELLED", "CANCELED", "ERROR"})
SPLIT_DONE_STATUSES = frozenset({"COMPLETED", "SUCCESS"})
SPLIT_FAILED_STATUSES = frozenset({"FAILED", "CANCELLED", "CANCELED", "ERROR"})


class FileEvent(StartEvent):
    file_id: str
    file_hash: str | None = None


class ParsedEvent(Event):
    """Parse finished (or soft-failed); extraction may proceed."""

    pass


class FileClassifiedEvent(Event):
    filing_type: str
    confidence: float | None = None
    reasoning: str | None = None


class Status(Event):
    level: Literal["info", "warning", "error"]
    message: str


class ExtractJobStartedEvent(Event):
    pass


class ExtractedEvent(Event):
    data: ExtractedData


class ExtractedInvalidEvent(Event):
    data: ExtractedData[dict[str, Any]]


class ExtractionState(BaseModel):
    file_id: str | None = None
    filename: str | None = None
    file_hash: str | None = None
    extract_job_id: str | None = None
    parse_job_id: str | None = None
    filing_type: str | None = None
    classification_confidence: float | None = None
    classification_reasoning: str | None = None
    # page_number (1-indexed) -> markdown text
    page_markdown: dict[int, str] = Field(default_factory=dict)


def _job_status(payload: Any) -> str:
    job = getattr(payload, "job", payload)
    return str(getattr(job, "status", "") or "").upper()


async def _wait_for_classify(client: AsyncLlamaCloud, job_id: str) -> Any:
    """Poll `classify.get` until the job reaches a terminal state."""
    elapsed = 0.0
    while elapsed < CLASSIFY_POLL_MAX_S:
        job = await client.classify.get(job_id, project_id=project_id)
        if job.status in ("COMPLETED", "FAILED"):
            return job
        await asyncio.sleep(CLASSIFY_POLL_INTERVAL_S)
        elapsed += CLASSIFY_POLL_INTERVAL_S
    raise TimeoutError(
        f"Classify job {job_id} did not complete within {CLASSIFY_POLL_MAX_S}s"
    )


async def _wait_for_parse(client: AsyncLlamaCloud, job_id: str) -> None:
    """Poll parse status until done, failed, or timed out.

    Do not use `parsing.parse()` / `wait_for_completion()` here: those poll for up
    to 2 hours, treat only `COMPLETED` as done, and will keep GET-ing forever if
    the API returns `SUCCESS`.
    """
    elapsed = 0.0
    last_status: str | None = None
    while elapsed <= PARSE_POLL_MAX_S:
        result = await client.parsing.get(job_id, project_id=project_id)
        status = _job_status(result)
        if status != last_status:
            logger.info("[Parse] job %s status=%s", job_id, status)
            last_status = status
        if status in PARSE_DONE_STATUSES:
            return
        if status in PARSE_FAILED_STATUSES:
            job = getattr(result, "job", result)
            detail = getattr(job, "error_message", None) or status
            raise RuntimeError(f"Parse job {job_id} failed: {detail}")
        await asyncio.sleep(PARSE_POLL_INTERVAL_S)
        elapsed += PARSE_POLL_INTERVAL_S
    raise TimeoutError(
        f"Parse job {job_id} did not complete within {PARSE_POLL_MAX_S}s"
    )


async def _wait_for_split(client: AsyncLlamaCloud, job_id: str) -> Any:
    elapsed = 0.0
    last_status: str | None = None
    while elapsed <= SPLIT_POLL_MAX_S:
        job = await client.split.get(job_id, project_id=project_id)
        status = str(getattr(job, "status", "") or "").upper()
        if status != last_status:
            logger.info("[Split] job %s status=%s", job_id, status)
            last_status = status
        if status in SPLIT_DONE_STATUSES:
            return job
        if status in SPLIT_FAILED_STATUSES:
            detail = getattr(job, "error_message", None) or status
            raise RuntimeError(f"Split job {job_id} failed: {detail}")
        await asyncio.sleep(SPLIT_POLL_INTERVAL_S)
        elapsed += SPLIT_POLL_INTERVAL_S
    raise TimeoutError(
        f"Split job {job_id} did not complete within {SPLIT_POLL_MAX_S}s"
    )


async def _split_page_parts(
    client: AsyncLlamaCloud,
    *,
    file_id: str | None,
    split_config: SplitConfig | None,
    filename: str | None = None,
) -> dict[int, str]:
    """Label pages with Split categories. Required when indexing Pinecone."""
    label = filename or "filing"
    if split_config is None:
        raise RuntimeError(
            f"Split config is missing, so document parts cannot be labelled for {label}."
        )
    if not getattr(client, "split", None):
        raise RuntimeError(
            f"LlamaCloud client has no split API, so document parts cannot be labelled for {label}."
        )
    if not file_id:
        raise RuntimeError(f"No file id is available to split {label}.")
    if not split_config.configuration_id and not split_config.categories:
        raise RuntimeError(
            f"Split categories are empty, so document parts cannot be labelled for {label}."
        )

    if split_config.configuration_id:
        job = await client.split.create(
            file_input=file_id,
            configuration_id=split_config.configuration_id,
            project_id=project_id,
        )
    else:
        job = await client.split.create(
            file_input=file_id,
            configuration=split_config.model_dump(
                exclude={"configuration_id", "product_type"},
                exclude_none=True,
            ),
            project_id=project_id,
        )
    completed = await _wait_for_split(client, job.id)
    mapping = page_parts_from_split(completed)
    if not mapping:
        raise RuntimeError(
            f"Split finished for {label} but labelled no pages. "
            "Scrutiny cannot filter Listing Proforma / Petition / checklist parts."
        )
    return mapping


def _extract_page_markdown(parse_result: Any) -> dict[int, str]:
    """Build page_number -> markdown map from a ParsingGetResponse."""
    pages: dict[int, str] = {}
    markdown = getattr(parse_result, "markdown", None)
    md_pages = getattr(markdown, "pages", None) if markdown else None
    if md_pages:
        for page in md_pages:
            page_number = getattr(page, "page_number", None)
            text = getattr(page, "markdown", None) or ""
            if page_number is not None and text:
                pages[int(page_number)] = str(text)
        return pages

    # Fallback: single blob if only markdown_full/text_full is available
    full = getattr(parse_result, "markdown_full", None) or getattr(
        parse_result, "text_full", None
    )
    if full:
        pages[1] = str(full)
    return pages


class ProcessFileWorkflow(Workflow):
    """Parse, classify, and extract a JubeeX filing."""

    @step()
    async def parse_file(
        self,
        event: FileEvent,
        ctx: Context[ExtractionState],
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
        """Parse the PDF to markdown pages for Pinecone indexing."""
        file_id = event.file_id
        logger.info(f"Running file {file_id}")

        try:
            file_metadata = None
            async for f in llama_cloud_client.files.list(file_ids=[file_id]):
                file_metadata = f
                break
            if file_metadata is None:
                raise ValueError(f"File {file_id} not found")
            filename = file_metadata.name
        except Exception as e:
            logger.error(f"Error fetching file metadata {file_id}: {e}", exc_info=True)
            ctx.write_event_to_stream(
                Status(
                    level="error",
                    message=f"Error fetching file metadata {file_id}: {e}",
                )
            )
            raise e

        file_hash = event.file_hash or file_metadata.external_file_id
        async with ctx.store.edit_state() as state:
            state.file_id = file_id
            state.filename = filename
            state.file_hash = file_hash

        # --- Parse ---
        page_markdown: dict[int, str] = {}
        parse_job_id: str | None = None
        try:
            ctx.write_event_to_stream(
                Status(level="info", message=f"Parsing file {filename}")
            )
            create_kwargs: dict[str, Any] = {
                "file_id": file_id,
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

            parse_job = await llama_cloud_client.parsing.create(**create_kwargs)
            parse_job_id = parse_job.id
            await _wait_for_parse(llama_cloud_client, parse_job.id)
            parse_result = await llama_cloud_client.parsing.get(
                parse_job.id,
                expand=["markdown"],
                project_id=project_id,
            )

            page_markdown = _extract_page_markdown(parse_result)
            ctx.write_event_to_stream(
                Status(
                    level="info",
                    message=f"Parsed {len(page_markdown)} page(s) from {filename}",
                )
            )
        except Exception as e:
            logger.error(f"Parse failed for {filename}: {e}", exc_info=True)
            ctx.write_event_to_stream(
                Status(
                    level="warning",
                    message=f"Parse failed; continuing without page text: {e}",
                )
            )

        async with ctx.store.edit_state() as state:
            state.parse_job_id = parse_job_id
            state.page_markdown = page_markdown

        return ParsedEvent()

    @step()
    async def start_extraction(
        self,
        event: ParsedEvent,
        ctx: Context[ExtractionState],
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
        """Start extraction job for the document."""
        state = await ctx.store.get_state()
        if state.file_id is None or state.filename is None:
            raise ValueError("File ID or filename is not set")

        logger.info(f"Extracting data from file {state.filename}")
        ctx.write_event_to_stream(
            Status(
                level="info",
                message=f"Extracting data from file {state.filename}",
            )
        )

        if extract_config.configuration_id:
            extract_job = await llama_cloud_client.extract.create(
                file_input=state.file_id,
                configuration_id=extract_config.configuration_id,
                project_id=project_id,
            )
        else:
            extract_job = await llama_cloud_client.extract.create(
                file_input=state.file_id,
                configuration=extract_config.model_dump(
                    exclude={"configuration_id", "product_type"},
                    exclude_none=True,
                ),
                project_id=project_id,
            )

        async with ctx.store.edit_state() as state:
            state.extract_job_id = extract_job.id

        return ExtractJobStartedEvent()

    @step()
    async def classify_file(
        self,
        event: ExtractJobStartedEvent,
        ctx: Context[ExtractionState],
        llama_cloud_client: Annotated[
            AsyncLlamaCloud, Resource(get_llama_cloud_client)
        ],
        classify_config: Annotated[
            ClassifyConfig,
            ResourceConfig(
                config_file="configs/config.json",
                path_selector="classify",
                label="Classification Rules",
                description="Rules for classifying JubeeX filing types",
            ),
        ],
    ) -> FileClassifiedEvent:
        """Classify the JubeeX filing document type in parallel with extraction."""
        state = await ctx.store.get_state()
        if state.file_id is None or state.filename is None:
            raise ValueError("File ID or filename is not set")

        try:
            logger.info(f"Classifying file {state.filename}")
            ctx.write_event_to_stream(
                Status(level="info", message=f"Classifying file {state.filename}")
            )

            if classify_config.configuration_id:
                classify_job = await llama_cloud_client.classify.create(
                    file_input=state.file_id,
                    configuration_id=classify_config.configuration_id,
                    project_id=project_id,
                )
            else:
                classify_job = await llama_cloud_client.classify.create(
                    file_input=state.file_id,
                    configuration=classify_config.model_dump(
                        exclude={"configuration_id", "product_type"},
                        exclude_none=True,
                    ),
                    project_id=project_id,
                )

            completed = await _wait_for_classify(llama_cloud_client, classify_job.id)

            if completed.status == "FAILED" or completed.result is None:
                logger.warning(
                    f"Classification did not resolve for {state.filename}, defaulting to 'other'"
                )
                ctx.write_event_to_stream(
                    Status(
                        level="warning",
                        message="Classification uncertain, using default schema",
                    )
                )
                async with ctx.store.edit_state() as state:
                    state.filing_type = "other"
                return FileClassifiedEvent(filing_type="other")

            result = completed.result
            filing_type = result.type or "other"
            confidence = result.confidence
            reasoning = result.reasoning

            logger.info(
                f"Classified {state.filename} as {filing_type} "
                f"(confidence: {confidence}, reasoning: {reasoning})"
            )
            ctx.write_event_to_stream(
                Status(
                    level="info",
                    message=f"Classified as {filing_type} JubeeX filing",
                )
            )

            async with ctx.store.edit_state() as state:
                state.filing_type = filing_type
                state.classification_confidence = confidence
                state.classification_reasoning = reasoning

            return FileClassifiedEvent(
                filing_type=filing_type,
                confidence=confidence,
                reasoning=reasoning,
            )

        except Exception as e:
            logger.error(f"Error classifying file {state.filename}: {e}", exc_info=True)
            ctx.write_event_to_stream(
                Status(
                    level="warning",
                    message=f"Classification failed, using default schema: {e}",
                )
            )
            async with ctx.store.edit_state() as state:
                state.filing_type = "other"
            return FileClassifiedEvent(filing_type="other")

    @step()
    async def complete_extraction(
        self,
        event: FileClassifiedEvent,
        ctx: Context[ExtractionState],
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
        split_config: Annotated[
            SplitConfig,
            ResourceConfig(
                config_file="configs/config.json",
                path_selector="split",
                label="Document Parts",
                description="LlamaSplit categories for petition bundle parts",
            ),
        ],
    ) -> StopEvent:
        """Wait for extraction, save Agent Data, and index pages in Pinecone."""
        state = await ctx.store.get_state()
        if state.extract_job_id is None:
            raise ValueError("Job ID cannot be null when waiting for its completion")

        filing_type = state.filing_type or "other"
        extract_config = extract_jubeex

        await llama_cloud_client.extract.wait_for_completion(
            state.extract_job_id,
            project_id=project_id,
        )
        job = await llama_cloud_client.extract.get(
            state.extract_job_id,
            expand=["extract_metadata"],
            project_id=project_id,
        )

        extracted_event: ExtractedEvent | ExtractedInvalidEvent
        try:
            logger.info(
                f"Extracted data: {json.dumps(job.model_dump(mode='json'), indent=2, default=str)}"
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
                file_id=state.file_id,
                file_hash=state.file_hash,
            )
            if data.metadata is None:
                data.metadata = {}
            data.metadata["classification"] = filing_type
            data.metadata["classification_confidence"] = state.classification_confidence
            data.metadata["classification_reasoning"] = state.classification_reasoning
            data.metadata["parse_job_id"] = state.parse_job_id
            data.metadata["page_count"] = len(state.page_markdown or {})
            extracted_event = ExtractedEvent(data=data)
        except InvalidExtractionData as e:
            logger.error(f"Error validating extracted data: {e}", exc_info=True)
            extracted_event = ExtractedInvalidEvent(data=e.invalid_item)
        except Exception as e:
            logger.error(
                f"Error extracting data from file {state.filename}: {e}", exc_info=True
            )
            ctx.write_event_to_stream(
                Status(
                    level="error",
                    message=f"Error extracting data from file {state.filename}: {e}",
                )
            )
            raise e

        ctx.write_event_to_stream(extracted_event)

        extracted_data = extracted_event.data
        page_parts: dict[int, str] = {}
        if pinecone_enabled():
            try:
                logger.info(f"Splitting file {state.filename}")
                ctx.write_event_to_stream(
                    Status(level="info", message=f"Splitting file {state.filename}")
                )
                page_parts = await _split_page_parts(
                    llama_cloud_client,
                    file_id=state.file_id,
                    split_config=split_config,
                    filename=state.filename,
                )
                parts = sorted(set(page_parts.values()))
                logger.info(
                    f"Split {state.filename} into {len(page_parts)} page(s) "
                    f"(parts: {', '.join(parts)})"
                )
                ctx.write_event_to_stream(
                    Status(
                        level="info",
                        message=(
                            f"Split {state.filename} into {len(parts)} document part(s)"
                        ),
                    )
                )
            except Exception as e:
                logger.error(
                    f"Error splitting file {state.filename}: {e}",
                    exc_info=True,
                )
                ctx.write_event_to_stream(
                    Status(
                        level="error",
                        message=f"Error splitting file {state.filename}: {e}",
                    )
                )
                raise

        data_dict = extracted_data.model_dump()
        if page_parts:
            overlay_split_documents(data_dict, page_parts)
        if extracted_data.file_hash is not None:
            delete_result = await llama_cloud_client.beta.agent_data.delete_by_query(
                deployment_name=agent_name or "_public",
                collection=EXTRACTED_DATA_COLLECTION,
                filter={
                    "file_hash": {
                        "eq": extracted_data.file_hash,
                    },
                },
            )
            if delete_result.deleted_count > 0:
                logger.info(
                    f"Removed {delete_result.deleted_count} existing record(s) "
                    f"for file {extracted_data.file_name}"
                )
        item = await llama_cloud_client.beta.agent_data.create(
            data=data_dict,
            deployment_name=agent_name or "_public",
            collection=EXTRACTED_DATA_COLLECTION,
        )
        logger.info(
            f"Recorded extracted data for file {extracted_data.file_name or ''}"
        )
        ctx.write_event_to_stream(
            Status(
                level="info",
                message=f"Recorded extracted data for file {extracted_data.file_name or ''}",
            )
        )

        if pinecone_enabled():
            try:
                base_id = extracted_data.file_hash or state.file_id or str(item.id)
                shared_meta = {
                    "agent_data_id": str(item.id),
                    "file_id": state.file_id,
                    "file_name": state.filename,
                    "file_hash": extracted_data.file_hash,
                    "petition_type": filing_type,
                }

                logger.info(
                    "[Pinecone] Indexing start for %s (base_id=%s, pages=%s)",
                    state.filename,
                    base_id,
                    len(state.page_markdown or {}),
                )
                ctx.write_event_to_stream(
                    Status(
                        level="info",
                        message=(
                            f"Indexing vectors in Pinecone for {state.filename} "
                            f"(integrated embeddings)"
                        ),
                    )
                )

                pinecone_items: list[dict[str, Any]] = []

                # 1) Filing-level summary vector
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
                            "metadata": {
                                **shared_meta,
                                "chunk_kind": "summary",
                            },
                        }
                    )
                    logger.info(
                        "[Pinecone] Built summary chunk (%s chars)",
                        len(summary_text),
                    )
                else:
                    logger.warning("[Pinecone] Summary chunk empty; skipping")

                # 2) Page vectors from Parse markdown
                page_records = build_page_records(
                    base_id=base_id,
                    page_markdown=state.page_markdown,
                    metadata=shared_meta,
                    page_parts=page_parts,
                )
                pinecone_items.extend(page_records)
                logger.info(
                    "[Pinecone] Built %s page chunk(s) from %s page(s)",
                    len(page_records),
                    len(state.page_markdown or {}),
                )
                if not page_records:
                    logger.warning(
                        "[Pinecone] No page markdown available — "
                        "only summary vector will be indexed"
                    )

                count = upsert_records(pinecone_items)
                logger.info(
                    "[Pinecone] Indexed %s vector(s) for %s "
                    "(1 summary + %s page chunks)",
                    count,
                    state.filename,
                    len(page_records),
                )
                ctx.write_event_to_stream(
                    Status(
                        level="info",
                        message=(
                            f"Indexed {count} vector(s) in Pinecone "
                            f"(summary + {len(page_records)} page chunks)"
                        ),
                    )
                )
            except Exception as e:
                logger.error(
                    "[Pinecone] Indexing failed for %s: %s",
                    state.filename,
                    e,
                    exc_info=True,
                )
                ctx.write_event_to_stream(
                    Status(
                        level="warning",
                        message=f"Pinecone indexing failed: {e}",
                    )
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


workflow = ProcessFileWorkflow(timeout=None)

if __name__ == "__main__":
    from pathlib import Path

    from dotenv import load_dotenv

    load_dotenv()
    logging.basicConfig(level=logging.INFO)

    async def main():
        file = await get_llama_cloud_client().files.create(
            file=Path("test.pdf").open("rb"),
            purpose="extract",
        )
        await workflow.run(start_event=FileEvent(file_id=file.id))

    asyncio.run(main())
