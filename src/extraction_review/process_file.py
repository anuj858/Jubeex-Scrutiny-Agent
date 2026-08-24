import asyncio
import json
import logging
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
from .vector_store import (
    build_filing_chunk_text,
    build_section_records,
    pinecone_enabled,
    upsert_records,
)

logger = logging.getLogger(__name__)

DISCRIMINATOR_FIELD = "petition_type"

CLASSIFY_POLL_INTERVAL_S = 1.0
CLASSIFY_POLL_MAX_S = 300.0


class FileEvent(StartEvent):
    file_id: str
    file_hash: str | None = None


class ParsedSplitEvent(Event):
    """Parse + split finished (or soft-failed); extraction may proceed."""

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
    # [{category, pages, confidence_category}, ...]
    split_segments: list[dict[str, Any]] = Field(default_factory=list)


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
    """Parse, split, classify, and extract a JubeeX filing."""

    @step()
    async def parse_and_split(
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
        split_config: Annotated[
            SplitConfig,
            ResourceConfig(
                config_file="configs/config.json",
                path_selector="split",
                label="Split Categories",
                description="Section categories for petition bundles",
            ),
        ],
    ) -> ParsedSplitEvent:
        """Parse the PDF to markdown pages, then split into petition sections."""
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
            parse_kwargs: dict[str, Any] = {
                "file_id": file_id,
                "expand": ["markdown"],
                "project_id": project_id,
            }
            if parse_config.configuration_id:
                parse_kwargs["configuration_id"] = parse_config.configuration_id
            else:
                dumped = parse_config.model_dump(
                    exclude={"configuration_id", "product_type"},
                    exclude_none=True,
                )
                parse_kwargs.update(dumped)

            # Prefer convenience parse(); fall back to create + wait + get
            if hasattr(llama_cloud_client.parsing, "parse") and not parse_config.configuration_id:
                parse_result = await llama_cloud_client.parsing.parse(**parse_kwargs)
                parse_job_id = getattr(getattr(parse_result, "job", None), "id", None)
            else:
                create_kwargs = {
                    k: v for k, v in parse_kwargs.items() if k != "expand"
                }
                parse_job = await llama_cloud_client.parsing.create(**create_kwargs)
                parse_job_id = parse_job.id
                await llama_cloud_client.parsing.wait_for_completion(
                    parse_job.id,
                    project_id=project_id,
                )
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

        # --- Split ---
        segments: list[dict[str, Any]] = []
        try:
            categories = [
                {"name": c.name, "description": c.description}
                for c in (split_config.categories or [])
            ]
            if not categories:
                raise ValueError("split.categories is empty in configs/config.json")

            ctx.write_event_to_stream(
                Status(level="info", message=f"Splitting file {filename} into sections")
            )
            document_input: dict[str, str]
            if parse_job_id:
                document_input = {"type": "parse_job_id", "value": parse_job_id}
            else:
                document_input = {"type": "file_id", "value": file_id}

            split_kwargs: dict[str, Any] = {
                "document_input": document_input,
                "categories": categories,
                "project_id": project_id,
            }
            if split_config.splitting_strategy is not None:
                split_kwargs["splitting_strategy"] = split_config.model_dump(
                    include={"splitting_strategy"},
                    exclude_none=True,
                ).get("splitting_strategy")

            if split_config.configuration_id:
                completed_split = await llama_cloud_client.beta.split.create(
                    document_input=document_input,
                    configuration_id=split_config.configuration_id,
                    project_id=project_id,
                )
                completed_split = (
                    await llama_cloud_client.beta.split.wait_for_completion(
                        completed_split.id,
                        project_id=project_id,
                    )
                )
            else:
                completed_split = await llama_cloud_client.beta.split.split(
                    **split_kwargs
                )

            result = getattr(completed_split, "result", None)
            raw_segments = getattr(result, "segments", None) or []
            for seg in raw_segments:
                if hasattr(seg, "model_dump"):
                    segments.append(seg.model_dump(mode="json"))
                elif isinstance(seg, dict):
                    segments.append(seg)
                else:
                    segments.append(
                        {
                            "category": getattr(seg, "category", "uncategorized"),
                            "pages": list(getattr(seg, "pages", []) or []),
                            "confidence_category": getattr(
                                seg, "confidence_category", None
                            ),
                        }
                    )

            ctx.write_event_to_stream(
                Status(
                    level="info",
                    message=f"Split into {len(segments)} section(s)",
                )
            )
        except Exception as e:
            logger.error(f"Split failed for {filename}: {e}", exc_info=True)
            ctx.write_event_to_stream(
                Status(
                    level="warning",
                    message=f"Split failed; continuing without sections: {e}",
                )
            )

        async with ctx.store.edit_state() as state:
            state.parse_job_id = parse_job_id
            state.page_markdown = page_markdown
            state.split_segments = segments

        return ParsedSplitEvent()

    @step()
    async def start_extraction(
        self,
        event: ParsedSplitEvent,
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
    ) -> StopEvent:
        """Wait for extraction, save Agent Data, and index sections in Pinecone."""
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
            data.metadata["split_segment_count"] = len(state.split_segments)
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
        data_dict = extracted_data.model_dump()
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

                # 2) Section vectors from Parse pages × Split segments
                pinecone_items.extend(
                    build_section_records(
                        base_id=base_id,
                        page_markdown=state.page_markdown,
                        segments=state.split_segments,
                        metadata=shared_meta,
                    )
                )

                count = upsert_records(pinecone_items)
                ctx.write_event_to_stream(
                    Status(
                        level="info",
                        message=(
                            f"Indexed {count} vector(s) in Pinecone "
                            f"(summary + section chunks)"
                        ),
                    )
                )
            except Exception as e:
                logger.error(
                    "Pinecone indexing failed for %s: %s",
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
