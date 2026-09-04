"""Classify and LlamaSplit a bundled PDF, then slice it into labeled files.

Extract, overlay, Agent Data, and Pinecone run only in process-split-files
after the user Submits the same split-upload form.
"""

from __future__ import annotations

import asyncio
import io
import logging
from typing import Annotated, Any, Literal

import httpx
from llama_cloud import AsyncLlamaCloud
from llama_cloud.types.beta.extracted_data import ExtractedData
from pydantic import BaseModel, Field
from workflows import Context, Workflow, step
from workflows.events import Event, StartEvent, StopEvent
from workflows.resource import Resource, ResourceConfig

from .bundle_slicer import slice_bundle_pdf
from .clients import get_llama_cloud_client, project_id
from .config import ClassifyConfig, SplitConfig
from .document_parts import page_parts_from_split, parts_on_page
from .split_upload import SplitUploadError, type_catalog

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
FILE_DOWNLOAD_TIMEOUT_S = 120.0


class FileEvent(StartEvent):
    file_id: str
    file_hash: str | None = None


class ParsedEvent(Event):
    """Parse finished (or soft-failed); extraction may proceed."""


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


class PreparedPart(BaseModel):
    slot_id: str
    file_id: str
    file_hash: str | None = None
    filename: str | None = None


class BundlePrepared(StopEvent):
    filing_type: str
    parts: list[PreparedPart] = Field(default_factory=list)
    slot_pages: dict[str, str] = Field(default_factory=dict)


class PrepareState(BaseModel):
    file_id: str | None = None
    filename: str | None = None
    file_hash: str | None = None
    filing_type: str | None = None
    classification_confidence: float | None = None
    classification_reasoning: str | None = None


# Kept for callers that still type the old bundled-extract store.
ExtractionState = PrepareState


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


def _split_api_configuration(split_config: SplitConfig) -> dict[str, Any]:
    """LlamaSplit accepts only category name + description."""
    dumped = split_config.model_dump(
        exclude={"configuration_id", "product_type"},
        exclude_none=True,
    )
    dumped["categories"] = [
        {
            "name": item["name"],
            **({"description": item["description"]} if item.get("description") else {}),
        }
        for item in dumped.get("categories") or []
        if isinstance(item, dict) and item.get("name")
    ]
    return dumped


async def _split_page_parts(
    client: AsyncLlamaCloud,
    *,
    file_id: str | None,
    split_config: SplitConfig | None,
    filename: str | None = None,
) -> dict[int, list[str]]:
    """Label pages with Split categories."""
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
            configuration=_split_api_configuration(split_config),
            project_id=project_id,
        )
    completed = await _wait_for_split(client, job.id)
    mapping = page_parts_from_split(completed)
    if not mapping:
        raise RuntimeError(
            f"Split finished for {label} but labelled no pages. "
            "Scrutiny cannot filter Listing Proforma / Main Petition / checklist parts."
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


async def _download_file_bytes(client: AsyncLlamaCloud, file_id: str) -> bytes:
    """Download original PDF bytes via the files content presigned URL."""
    presigned = await client.files.content(file_id, project_id=project_id)
    url = getattr(presigned, "url", None)
    if not url:
        raise RuntimeError(f"No download URL for file {file_id}")
    timeout = httpx.Timeout(FILE_DOWNLOAD_TIMEOUT_S)
    async with httpx.AsyncClient(timeout=timeout, follow_redirects=True) as http:
        response = await http.get(str(url))
        response.raise_for_status()
        return response.content


async def _upload_slot_pdf(
    client: AsyncLlamaCloud,
    *,
    filename: str,
    pdf_bytes: bytes,
) -> str:
    uploaded = await client.files.create(
        file=(filename, io.BytesIO(pdf_bytes), "application/pdf"),
        purpose="extract",
        project_id=project_id,
    )
    file_id = getattr(uploaded, "id", None)
    if not file_id:
        raise RuntimeError(f"Upload of {filename} did not return a file id")
    return str(file_id)


class ProcessFileWorkflow(Workflow):
    """Classify and split a bundled PDF into catalog-slot files. Does not extract."""

    @step()
    async def classify_file(
        self,
        event: FileEvent,
        ctx: Context[PrepareState],
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
        file_id = event.file_id
        logger.info("Preparing bundled file %s", file_id)

        try:
            file_metadata = await llama_cloud_client.files.retrieve(
                file_id, project_id=project_id
            )
            filename = file_metadata.name
        except Exception as exc:
            logger.exception("Error fetching file metadata %s", file_id)
            ctx.write_event_to_stream(
                Status(
                    level="error",
                    message=f"Error fetching file metadata {file_id}: {exc}",
                )
            )
            raise

        file_hash = event.file_hash or file_metadata.external_file_id
        async with ctx.store.edit_state() as state:
            state.file_id = file_id
            state.filename = filename
            state.file_hash = file_hash

        ctx.write_event_to_stream(
            Status(level="info", message=f"Classifying file {filename}")
        )
        if classify_config.configuration_id:
            classify_job = await llama_cloud_client.classify.create(
                file_input=file_id,
                configuration_id=classify_config.configuration_id,
                project_id=project_id,
            )
        else:
            classify_job = await llama_cloud_client.classify.create(
                file_input=file_id,
                configuration=classify_config.model_dump(
                    exclude={"configuration_id", "product_type"},
                    exclude_none=True,
                ),
                project_id=project_id,
            )

        completed = await _wait_for_classify(llama_cloud_client, classify_job.id)
        if completed.status == "FAILED" or completed.result is None:
            message = f"Classification did not resolve for {filename}"
            ctx.write_event_to_stream(Status(level="error", message=message))
            raise RuntimeError(message)

        result = completed.result
        filing_type = result.type or "other"
        confidence = result.confidence
        reasoning = result.reasoning

        try:
            catalog = type_catalog(filing_type)
        except SplitUploadError:
            message = (
                f"No split-upload catalog for {filing_type}; "
                "cannot prepare sliced files"
            )
            ctx.write_event_to_stream(Status(level="error", message=message))
            raise RuntimeError(message)

        logger.info(
            "Classified %s as %s (confidence: %s, reasoning: %s)",
            filename,
            filing_type,
            confidence,
            reasoning,
        )
        ctx.write_event_to_stream(
            Status(
                level="info",
                message=f"Classified as {filing_type} JubeeX filing",
            )
        )
        ctx.write_event_to_stream(
            Status(
                level="info",
                message=f"Using {catalog.label} document slots",
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

    @step()
    async def prepare_bundle(
        self,
        event: FileClassifiedEvent,
        ctx: Context[PrepareState],
        llama_cloud_client: Annotated[
            AsyncLlamaCloud, Resource(get_llama_cloud_client)
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
    ) -> BundlePrepared:
        state = await ctx.store.get_state()
        if not state.file_id or not state.filename:
            raise ValueError("File ID or filename is not set")
        filing_type = event.filing_type or state.filing_type
        if not filing_type:
            raise ValueError("Filing type is not set")
        catalog = type_catalog(filing_type)

        ctx.write_event_to_stream(
            Status(level="info", message=f"Splitting file {state.filename}")
        )
        page_parts = await _split_page_parts(
            llama_cloud_client,
            file_id=state.file_id,
            split_config=split_config,
            filename=state.filename,
        )
        parts_found = sorted(
            {name for names in page_parts.values() for name in parts_on_page(names)}
        )
        ctx.write_event_to_stream(
            Status(
                level="info",
                message=(
                    f"Split {state.filename} into {len(parts_found)} document part(s)"
                ),
            )
        )

        ctx.write_event_to_stream(
            Status(
                level="info",
                message=f"Downloading {state.filename} to slice labeled pages",
            )
        )
        pdf_bytes = await _download_file_bytes(llama_cloud_client, state.file_id)
        slices = slice_bundle_pdf(pdf_bytes, catalog, page_parts)
        ctx.write_event_to_stream(
            Status(
                level="info",
                message=(
                    f"Sliced {len(slices)} document file(s) from {state.filename}"
                ),
            )
        )

        prepared: list[PreparedPart] = []
        slot_pages: dict[str, str] = {}
        for item in slices:
            file_id = await _upload_slot_pdf(
                llama_cloud_client,
                filename=item.filename,
                pdf_bytes=item.pdf_bytes,
            )
            prepared.append(
                PreparedPart(
                    slot_id=item.slot_id,
                    file_id=file_id,
                    file_hash=item.file_hash,
                    filename=item.filename,
                )
            )
            if item.page_span:
                slot_pages[item.slot_id] = item.page_span
            ctx.write_event_to_stream(
                Status(
                    level="info",
                    message=f"Ready {item.label} ({item.page_span or 'pages unknown'})",
                )
            )

        return BundlePrepared(
            filing_type=catalog.filing_type,
            parts=prepared,
            slot_pages=slot_pages,
        )


workflow = ProcessFileWorkflow(timeout=None)

if __name__ == "__main__":
    from pathlib import Path

    from dotenv import load_dotenv

    load_dotenv()
    logging.basicConfig(level=logging.INFO)

    async def main():
        file = await get_llama_cloud_client().files.create(
            file=Path("test.pdf"),
            purpose="extract",
        )
        await workflow.run(start_event=FileEvent(file_id=file.id))

    asyncio.run(main())
