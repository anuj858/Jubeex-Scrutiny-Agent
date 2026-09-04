"""Classify and LlamaSplit a bundled PDF, slice it, then extract.

``upload_compiled`` classifies, splits, and slices, then runs
process-split-files (parse, extract, Agent Data, Pinecone).
``upload_separate`` skips classify/split and runs process-split-files only.
"""

from __future__ import annotations

import asyncio
import hashlib
import io
import logging
import re
from pathlib import PurePosixPath
from typing import Annotated, Any, Literal
from urllib.parse import unquote, urlparse

import httpx
from llama_cloud import AsyncLlamaCloud, BadRequestError
from llama_cloud.types.beta.extracted_data import ExtractedData
from pydantic import (
    AliasChoices,
    BaseModel,
    ConfigDict,
    Field,
    field_validator,
    model_validator,
)
from workflows import Context, Workflow, step
from workflows.events import Event, StartEvent, StopEvent
from workflows.resource import Resource, ResourceConfig

from .bundle_slicer import slice_bundle_pdf
from .clients import get_llama_cloud_client, project_id
from .config import ClassifyConfig, SplitConfig
from .document_parts import page_parts_from_split, parts_on_page
from .split_upload import SplitUploadError, type_catalog, ui_catalog

logger = logging.getLogger(__name__)

DISCRIMINATOR_FIELD = "petition_type"

CLASSIFY_POLL_INTERVAL_S = 1.0
CLASSIFY_POLL_MAX_S = 600.0
PARSE_POLL_INTERVAL_S = 2.0
PARSE_POLL_MAX_S = 600.0
SPLIT_POLL_INTERVAL_S = 2.0
SPLIT_POLL_MAX_S = 600.0
PARSE_DONE_STATUSES = frozenset({"COMPLETED", "SUCCESS"})
PARSE_FAILED_STATUSES = frozenset({"FAILED", "CANCELLED", "CANCELED", "ERROR"})
SPLIT_DONE_STATUSES = frozenset({"COMPLETED", "SUCCESS"})
SPLIT_FAILED_STATUSES = frozenset({"FAILED", "CANCELLED", "CANCELED", "ERROR"})
FILE_DOWNLOAD_TIMEOUT_S = 120.0
FULL_PETITION_SLOTS = frozenset({"fullpetition", "full_petition", "compiled"})
FULL_JOB_TYPES = frozenset(
    {
        "full",
        "compiled",
        "bundle",
        "upload_compiled",
        "upload_combined",
        "upload_full",
        "upload_bundle",
    }
)
SPLIT_JOB_TYPES = frozenset(
    {"split", "parts", "upload_separate", "upload_split"}
)
DEFAULT_COMPILED_FILING_TYPE = "SLP_CIVIL"
SWAGGER_PLACEHOLDERS = frozenset({"string", "str", "none", "null"})
_SLOT_NAME_ALIASES = {
    "list_of_dates": "synopsis_lod",
    "list_of_dates_events": "synopsis_lod",
    "list_of_dates_and_events": "synopsis_lod",
    "synopsis": "synopsis_lod",
    "lod": "synopsis_lod",
    "annexure": "annexures",
    "full_petition": "fullpetition",
    "advocates_check_list": "advocates_checklist",
    "advocate_checklist": "advocates_checklist",
    "aor_declaration": "aors_declaration",
    "office_report_limitation": "office_report_on_limitation",
}


class FilingPartIn(BaseModel):
    """One PDF in a backend intake payload (compiled or already-split)."""

    model_config = ConfigDict(populate_by_name=True)

    slot_id: str | None = None
    document_id: str | None = None
    filename: str | None = Field(
        default=None,
        validation_alias=AliasChoices("name", "filename"),
    )
    file_url: str | None = Field(
        default=None,
        validation_alias=AliasChoices("download_url", "file_url"),
    )
    file_id: str | None = None
    file_hash: str | None = None

    @field_validator(
        "slot_id",
        "document_id",
        "filename",
        "file_url",
        "file_id",
        "file_hash",
        mode="before",
    )
    @classmethod
    def _drop_swagger_placeholders(cls, value: object) -> str | None:
        return blank_or_placeholder(value)

    @model_validator(mode="after")
    def _require_file_id_or_url(self) -> FilingPartIn:
        file_id = (self.file_id or "").strip()
        if file_id and not file_id.lower().startswith("dfl-"):
            # Backend document UUIDs / Swagger "test123" are not LlamaCloud file ids.
            self.file_id = None
        if not (self.file_id or "").strip() and not (self.file_url or "").strip():
            label = self.slot_id or self.filename or self.document_id or "document"
            raise ValueError(f"Part {label!r} needs download_url or file_id")
        return self


def blank_or_placeholder(value: object) -> str | None:
    if value is None:
        return None
    raw = str(value).strip()
    if not raw or raw.lower() in SWAGGER_PLACEHOLDERS:
        return None
    return raw


def normalize_id(value: str | None) -> str | None:
    return blank_or_placeholder(value)


def normalize_org_id(value: str | None) -> str | None:
    return normalize_id(value)


def normalize_job_type(value: str | None) -> str | None:
    raw = (value or "").strip().lower()
    if raw in FULL_JOB_TYPES:
        return "full"
    if raw in SPLIT_JOB_TYPES:
        return "split"
    return None


def normalize_petitiontype(value: str | None) -> str | None:
    return normalize_job_type(value)


def normalize_document_stem(name: str) -> str:
    stem = PurePosixPath((name or "").replace("\\", "/")).stem
    stem = re.sub(r"^\d+[_.\-\s]+", "", stem)
    return re.sub(r"[^a-z0-9]+", "_", stem.lower()).strip("_")


def _known_slot_ids(filing_type: str | None) -> list[str]:
    if filing_type:
        try:
            return [slot.id for slot in type_catalog(filing_type).slots]
        except SplitUploadError:
            pass
    ids: list[str] = []
    seen: set[str] = set()
    for entry in ui_catalog().values():
        for slot in entry.get("slots") or []:
            slot_id = str(slot.get("id") or "").strip()
            if slot_id and slot_id not in seen:
                seen.add(slot_id)
                ids.append(slot_id)
    return ids


def slot_id_from_name(name: str, filing_type: str | None = None) -> str | None:
    """Map `01_Petition.pdf` / `List of Dates` onto a split-upload slot id."""
    stem = normalize_document_stem(name)
    if not stem:
        return None
    if stem in FULL_PETITION_SLOTS or stem == "full_petition":
        return "fullpetition"
    if stem.startswith("annexure"):
        return "annexures"
    known = _known_slot_ids(filing_type)
    if stem in known:
        return stem
    alias = _SLOT_NAME_ALIASES.get(stem)
    if alias and (not known or alias in known or alias == "fullpetition"):
        return alias
    matches = [slot_id for slot_id in known if slot_id in stem or stem in slot_id]
    if matches:
        return max(matches, key=len)
    return None


def resolve_document_slot(
    item: FilingPartIn,
    *,
    mode: str,
    filing_type: str | None,
) -> str:
    explicit = blank_or_placeholder(item.slot_id)
    if explicit:
        return explicit
    if mode == "full":
        return "fullpetition"
    names = [
        item.filename or "",
        _filename_from_url(item.file_url or "") if item.file_url else "",
    ]
    for name in names:
        resolved = slot_id_from_name(name, filing_type)
        if resolved and resolved not in FULL_PETITION_SLOTS:
            return resolved
    label = item.filename or item.document_id or item.file_url or "document"
    raise ValueError(
        f"Could not map {label!r} to a document slot. "
        "Do not send Swagger placeholders like slot_id='string'. "
        "Set name to something like 01_Petition.pdf, or slot_id to "
        "petition, cover_page, vakalatnama, … "
        "For one compiled PDF use job_type upload_compiled."
    )


def resolve_compiled_filing_type(
    classified: str,
    requested: str | None,
) -> tuple[str, Any]:
    """Pick a slice catalog. Prefer a classified type that we can slice."""
    classified_key = (classified or "").strip()
    requested_key = (requested or "").strip()
    try:
        return classified_key, type_catalog(classified_key)
    except SplitUploadError:
        pass
    if requested_key:
        try:
            return requested_key, type_catalog(requested_key)
        except SplitUploadError:
            pass
    try:
        return DEFAULT_COMPILED_FILING_TYPE, type_catalog(
            DEFAULT_COMPILED_FILING_TYPE
        )
    except SplitUploadError:
        allowed = ", ".join(sorted(ui_catalog()))
    raise SplitUploadError(
        f"Classified as {classified_key or 'other'}, which cannot be sliced. "
        f"Send filing_type as one of: {allowed}"
    )


def compiled_catalog_override(filing_type: str | None) -> Any | None:
    """Return a slice catalog when the caller already sent a usable filing_type."""
    key = (filing_type or "").strip()
    if not key:
        return None
    try:
        return type_catalog(key)
    except SplitUploadError:
        return None


def intake_mode(event: FileEvent) -> str:
    """full → process-file bundle path; split → process-split-files."""
    typed = normalize_job_type(event.job_type)
    if typed:
        return typed
    slots = [(p.slot_id or "").strip().lower() for p in event.documents]
    full_slots = [s for s in slots if s in FULL_PETITION_SLOTS]
    other_slots = [s for s in slots if s and s not in FULL_PETITION_SLOTS]
    if full_slots and other_slots:
        raise ValueError(
            "Mix of fullpetition and split slots requires job_type "
            "upload_compiled or upload_separate"
        )
    if full_slots:
        return "full"
    if other_slots:
        return "split"
    if len(event.documents) > 1:
        return "split"
    return "full"


def compiled_source(
    event: FileEvent,
) -> tuple[str | None, str | None, str | None, str | None]:
    """file_id, file_url, filename, file_hash for a compiled petition."""
    for part in event.documents:
        if (part.slot_id or "").strip().lower() in FULL_PETITION_SLOTS:
            return (
                (part.file_id or "").strip() or None,
                (part.file_url or "").strip() or None,
                (part.filename or "").strip() or None,
                part.file_hash,
            )
    if event.documents:
        part = event.documents[0]
        return (
            (part.file_id or "").strip() or None,
            (part.file_url or "").strip() or None,
            (part.filename or "").strip() or None,
            part.file_hash,
        )
    return (
        (event.file_id or "").strip() or None,
        (event.file_url or "").strip() or None,
        (event.filename or "").strip() or None,
        event.file_hash,
    )


def source_document_from_part(
    item: FilingPartIn,
    *,
    file_id: str | None = None,
    file_hash: str | None = None,
) -> SourceDocument:
    return SourceDocument(
        name=item.filename,
        document_id=item.document_id,
        download_url=item.file_url,
        slot_id=item.slot_id,
        file_id=file_id or item.file_id,
        file_hash=file_hash or item.file_hash,
    )


def intake_echo(event: FileEvent) -> dict[str, str | None]:
    org_id = event.organization_id
    return {
        "job_type": event.job_type,
        "organization_id": org_id,
        "workspace_id": event.workspace_id,
        "user_id": event.user_id,
        "org_id": org_id,
    }


class FileEvent(StartEvent):
    """Start from LlamaCloud file_id, a remote URL, or labeled documents.

    Backend (no Llama UI): POST this to ``process-file``.

    - ``job_type=upload_compiled`` (or ``full``): classify, slice, then extract.
    - ``job_type=upload_separate`` (or ``split``) plus ``documents[]``:
      already-split files; runs process-split-files internally.
    """

    model_config = ConfigDict(populate_by_name=True)

    job_type: str | None = Field(
        default=None,
        validation_alias=AliasChoices("job_type", "petitiontype", "petition_type"),
    )
    filing_type: str | None = None
    organization_id: str | None = Field(
        default=None,
        validation_alias=AliasChoices("organization_id", "org_id"),
    )
    workspace_id: str | None = None
    user_id: str | None = None
    file_id: str | None = None
    file_url: str | None = Field(
        default=None,
        validation_alias=AliasChoices("download_url", "file_url"),
    )
    file_hash: str | None = None
    filename: str | None = Field(
        default=None,
        validation_alias=AliasChoices("name", "filename"),
    )
    documents: list[FilingPartIn] = Field(
        default_factory=list,
        validation_alias=AliasChoices("documents", "parts"),
    )

    def __init__(self, **params: Any) -> None:
        # workflows.Event only forwards exact field names, not validation aliases.
        if "job_type" not in params:
            if "petitiontype" in params:
                params["job_type"] = params.pop("petitiontype")
            elif "petition_type" in params:
                params["job_type"] = params.pop("petition_type")
        params.pop("petitiontype", None)
        params.pop("petition_type", None)
        if "organization_id" not in params and "org_id" in params:
            params["organization_id"] = params.pop("org_id")
        else:
            params.pop("org_id", None)
        if "documents" not in params and "parts" in params:
            params["documents"] = params.pop("parts")
        else:
            params.pop("parts", None)
        if "file_url" not in params and "download_url" in params:
            params["file_url"] = params.pop("download_url")
        if "filename" not in params and "name" in params:
            params["filename"] = params.pop("name")
        super().__init__(**params)

    @property
    def parts(self) -> list[FilingPartIn]:
        return self.documents

    @property
    def org_id(self) -> str | None:
        return self.organization_id

    @property
    def petitiontype(self) -> str | None:
        return self.job_type

    @field_validator("organization_id", "workspace_id", "user_id", mode="before")
    @classmethod
    def _blank_ids(cls, value: object) -> str | None:
        if value is None:
            return None
        return normalize_id(str(value))

    @model_validator(mode="after")
    def _require_file_id_or_url(self) -> FileEvent:
        mode = intake_mode(self)
        resolved: list[FilingPartIn] = []
        for item in self.documents:
            resolved.append(
                item.model_copy(
                    update={
                        "slot_id": resolve_document_slot(
                            item, mode=mode, filing_type=self.filing_type
                        )
                    }
                )
            )
        self.documents = resolved
        if mode == "split":
            if not self.documents:
                raise ValueError(
                    "documents[] is required when job_type is upload_separate"
                )
            if not (self.filing_type or "").strip():
                raise ValueError(
                    "filing_type is required when job_type is upload_separate"
                )
            try:
                catalog = type_catalog(self.filing_type)
            except SplitUploadError as exc:
                allowed = ", ".join(sorted(ui_catalog()))
                raise ValueError(f"{exc}. Use one of: {allowed}") from exc
            allowed_slots = catalog.slot_by_id()
            for item in self.documents:
                slot = (item.slot_id or "").strip()
                if slot not in allowed_slots:
                    raise ValueError(
                        f"Unknown slot {slot!r} for {catalog.filing_type}. "
                        f"Use one of: {', '.join(allowed_slots)}"
                    )
            return self
        file_id, file_url, _name, _hash = compiled_source(self)
        if not file_id and not file_url:
            raise ValueError(
                "Provide download_url or file_id, or documents[] with the compiled PDF"
            )
        if (self.filing_type or "").strip():
            try:
                type_catalog(self.filing_type)
            except SplitUploadError as exc:
                allowed = ", ".join(sorted(ui_catalog()))
                raise ValueError(f"{exc}. Use one of: {allowed}") from exc
        return self


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
    document_id: str | None = None
    download_url: str | None = None
    name: str | None = None


class SourceDocument(BaseModel):
    name: str | None = None
    document_id: str | None = None
    download_url: str | None = None
    slot_id: str | None = None
    file_id: str | None = None
    file_hash: str | None = None


class BundlePrepared(StopEvent):
    filing_type: str
    parts: list[PreparedPart] = Field(default_factory=list)
    documents: list[SourceDocument] = Field(default_factory=list)
    slot_pages: dict[str, str] = Field(default_factory=dict)
    agent_data_id: str | None = None
    job_type: str | None = None
    organization_id: str | None = None
    workspace_id: str | None = None
    user_id: str | None = None
    org_id: str | None = None


class PrepareState(BaseModel):
    file_id: str | None = None
    filename: str | None = None
    file_hash: str | None = None
    filing_type: str | None = None
    job_type: str | None = None
    organization_id: str | None = None
    workspace_id: str | None = None
    user_id: str | None = None
    org_id: str | None = None
    source_documents: list[SourceDocument] = Field(default_factory=list)
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


def _filename_from_url(url: str) -> str:
    path = unquote(urlparse(url).path or "")
    name = PurePosixPath(path).name
    if name and "." in name:
        return name
    return "filing.pdf"


def _upload_filename(name: str) -> str:
    raw = PurePosixPath((name or "").replace("\\", "/")).name.strip()
    if not raw:
        return "filing.pdf"
    stem, dot, suffix = raw.rpartition(".")
    if not dot or suffix.lower() != "pdf":
        stem = raw
    cleaned = re.sub(r"[^A-Za-z0-9._-]+", "_", stem).strip("._") or "filing"
    return f"{cleaned[:80]}.pdf"


def _require_pdf_bytes(data: bytes, url: str) -> None:
    if data.startswith(b"%PDF"):
        return
    preview = data[:180].decode("utf-8", errors="replace").replace("\n", " ")
    raise ValueError(
        f"Downloaded body from {url} is not a PDF "
        f"({len(data)} bytes, starts with {preview!r})"
    )


async def ingest_remote_file(
    client: AsyncLlamaCloud,
    file_url: str,
    *,
    filename: str | None = None,
    external_file_id: str | None = None,
) -> tuple[str, str, str]:
    """Download a remote PDF and upload it to LlamaCloud.

    Returns ``(file_id, content_sha256, filename)``.
    """
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

    name = _upload_filename((filename or "").strip() or _filename_from_url(url))
    content_hash = hashlib.sha256(data).hexdigest()
    file_id = await _upload_slot_pdf(
        client,
        filename=name,
        pdf_bytes=data,
        external_file_id=external_file_id,
    )
    logger.info(
        "[process-file] Ingested remote file url=%s → file_id=%s name=%s bytes=%s",
        url[:120],
        file_id,
        name,
        len(data),
    )
    return file_id, content_hash, name


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
    external_file_id: str | None = None,
) -> str:
    name = _upload_filename(filename)
    external = (external_file_id or "").strip() or None

    create_kwargs: dict[str, Any] = {
        "file": (name, io.BytesIO(pdf_bytes), "application/pdf"),
        "purpose": "extract",
        "project_id": project_id,
    }
    if external:
        create_kwargs["external_file_id"] = external
    try:
        uploaded = await client.files.create(**create_kwargs)
    except BadRequestError:
        if not external:
            raise
        logger.warning(
            "LlamaCloud rejected upload with external_file_id=%s; retrying without it",
            external,
        )
        uploaded = await client.files.create(
            file=(name, io.BytesIO(pdf_bytes), "application/pdf"),
            purpose="extract",
            project_id=project_id,
        )
    file_id = getattr(uploaded, "id", None)
    if not file_id:
        raise RuntimeError(f"Upload of {name} did not return a file id")
    return str(file_id)


async def _extract_sliced_parts(
    ctx: Context[PrepareState],
    *,
    filing_type: str,
    parts: list[Any],
    echo: dict[str, str | None],
    require_all_slots: bool,
    fallback_file_id: str | None = None,
) -> str | None:
    """Run process-split-files (parse, extract, Agent Data, Pinecone)."""
    from .process_split_files import ProcessSplitFilesWorkflow, SplitFilesEvent

    nested = ProcessSplitFilesWorkflow(timeout=None)
    handler = nested.run(
        start_event=SplitFilesEvent(
            filing_type=filing_type,
            org_id=echo["organization_id"],
            organization_id=echo["organization_id"],
            workspace_id=echo["workspace_id"],
            user_id=echo["user_id"],
            job_type=echo["job_type"],
            parts=parts,
            require_all_slots=require_all_slots,
            fallback_file_id=fallback_file_id,
        )
    )
    async for ev in handler.stream_events():
        ctx.write_event_to_stream(ev)
    result = await handler
    item_id = getattr(result, "result", result)
    return str(item_id) if item_id else None


async def _run_split_from_file_event(
    event: FileEvent,
    ctx: Context[PrepareState],
    client: AsyncLlamaCloud,
) -> BundlePrepared:
    """Ingest labeled URLs and run process-split-files inside this handler."""
    from .process_split_files import SplitPartEvent

    filing_type = (event.filing_type or "").strip()
    ctx.write_event_to_stream(
        Status(
            level="info",
            message=(
                f"job_type=upload_separate: ingesting {len(event.documents)} file(s)"
            ),
        )
    )
    split_parts: list[SplitPartEvent] = []
    source_docs: list[SourceDocument] = []
    for item in event.documents:
        slot = (item.slot_id or "").strip()
        if slot.lower() in FULL_PETITION_SLOTS:
            raise ValueError(
                f"slot_id {slot!r} belongs to job_type upload_compiled, not upload_separate"
            )
        file_id = (item.file_id or "").strip() or None
        filename = (item.filename or "").strip() or None
        file_hash = item.file_hash
        if not file_id:
            file_id, digest, filename = await ingest_remote_file(
                client,
                item.file_url or "",
                filename=filename,
                external_file_id=item.document_id or file_hash,
            )
            file_hash = file_hash or digest
        split_parts.append(
            SplitPartEvent(
                slot_id=slot,
                file_id=file_id,
                filename=filename,
                file_hash=file_hash,
                file_url=item.file_url,
                document_id=item.document_id,
            )
        )
        source_docs.append(
            source_document_from_part(item, file_id=file_id, file_hash=file_hash)
        )

    echo = intake_echo(event)
    agent_data_id = await _extract_sliced_parts(
        ctx,
        filing_type=filing_type,
        parts=split_parts,
        echo=echo,
        require_all_slots=True,
    )
    prepared = [
        PreparedPart(
            slot_id=part.slot_id,
            file_id=part.file_id,
            file_hash=part.file_hash,
            filename=part.filename,
            document_id=part.document_id,
            download_url=part.file_url,
            name=part.filename,
        )
        for part in split_parts
        if part.file_id
    ]
    return BundlePrepared(
        result=agent_data_id,
        filing_type=filing_type,
        parts=prepared,
        documents=source_docs,
        agent_data_id=agent_data_id,
        **echo,
    )


class ProcessFileWorkflow(Workflow):
    """Classify and split a bundled PDF, or ingest already-split parts."""

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
    ) -> FileClassifiedEvent | BundlePrepared:
        if intake_mode(event) == "split":
            return await _run_split_from_file_event(event, ctx, llama_cloud_client)

        file_id, file_url, filename, file_hash_hint = compiled_source(event)
        content_hash: str | None = None

        if not file_id and file_url:
            ctx.write_event_to_stream(
                Status(
                    level="info",
                    message="Downloading filing from file_url and uploading to LlamaCloud",
                )
            )
            try:
                compiled_external = (
                    event.documents[0].document_id if event.documents else None
                )
                file_id, content_hash, filename = await ingest_remote_file(
                    llama_cloud_client,
                    file_url,
                    filename=filename,
                    external_file_id=compiled_external or file_hash_hint,
                )
            except Exception as exc:
                logger.exception("Failed to ingest file_url %s", file_url)
                ctx.write_event_to_stream(
                    Status(
                        level="error",
                        message=f"Failed to ingest file_url: {exc}",
                    )
                )
                raise

        if not file_id:
            raise ValueError("Provide file_id or file_url in start_event")

        logger.info("Preparing bundled file %s", file_id)

        try:
            file_metadata = await llama_cloud_client.files.retrieve(
                file_id, project_id=project_id
            )
            filename = filename or file_metadata.name
        except Exception as exc:
            logger.exception("Error fetching file metadata %s", file_id)
            ctx.write_event_to_stream(
                Status(
                    level="error",
                    message=f"Error fetching file metadata {file_id}: {exc}",
                )
            )
            raise

        file_hash = file_hash_hint or content_hash or file_metadata.external_file_id
        echo = intake_echo(event)
        source_docs = [
            source_document_from_part(
                item,
                file_id=item.file_id or file_id,
                file_hash=item.file_hash or file_hash,
            )
            for item in event.documents
        ]
        if not source_docs:
            source_docs = [
                SourceDocument(
                    name=filename,
                    download_url=event.file_url,
                    file_id=file_id,
                    file_hash=file_hash,
                    slot_id="fullpetition",
                )
            ]
        async with ctx.store.edit_state() as state:
            state.file_id = file_id
            state.filename = filename
            state.file_hash = file_hash
            state.job_type = echo["job_type"]
            state.organization_id = echo["organization_id"]
            state.workspace_id = echo["workspace_id"]
            state.user_id = echo["user_id"]
            state.org_id = echo["org_id"]
            state.source_documents = source_docs

        override = compiled_catalog_override(event.filing_type)
        if override is not None:
            ctx.write_event_to_stream(
                Status(
                    level="info",
                    message=(
                        f"Using provided filing_type {override.filing_type}; "
                        "skipping classify"
                    ),
                )
            )
            async with ctx.store.edit_state() as state:
                state.filing_type = override.filing_type
            return FileClassifiedEvent(filing_type=override.filing_type)

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
        classified = result.type or "other"
        confidence = result.confidence
        reasoning = result.reasoning
        try:
            filing_type, catalog = resolve_compiled_filing_type(
                classified, event.filing_type
            )
        except SplitUploadError as exc:
            ctx.write_event_to_stream(Status(level="error", message=str(exc)))
            raise RuntimeError(str(exc)) from exc
        if filing_type != classified:
            ctx.write_event_to_stream(
                Status(
                    level="warning",
                    message=(
                        f"Classified as {classified}; using filing_type "
                        f"{filing_type} for slicing"
                    ),
                )
            )

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

        from .process_split_files import SplitPartEvent

        ctx.write_event_to_stream(
            Status(
                level="info",
                message=(
                    "Extracting sliced compiled petition "
                    f"({len(prepared)} document part(s))"
                ),
            )
        )
        echo = {
            "job_type": state.job_type,
            "organization_id": state.organization_id,
            "workspace_id": state.workspace_id,
            "user_id": state.user_id,
            "org_id": state.org_id,
        }
        agent_data_id = await _extract_sliced_parts(
            ctx,
            filing_type=catalog.filing_type,
            fallback_file_id=state.file_id,
            parts=[
                SplitPartEvent(
                    slot_id=item.slot_id,
                    file_id=item.file_id,
                    file_hash=item.file_hash,
                    filename=item.filename,
                    document_id=item.document_id,
                    file_url=item.download_url,
                )
                for item in prepared
                if item.file_id
            ],
            echo=echo,
            require_all_slots=False,
        )
        return BundlePrepared(
            result=agent_data_id,
            filing_type=catalog.filing_type,
            parts=prepared,
            documents=list(state.source_documents),
            slot_pages=slot_pages,
            agent_data_id=agent_data_id,
            job_type=state.job_type,
            organization_id=state.organization_id,
            workspace_id=state.workspace_id,
            user_id=state.user_id,
            org_id=state.org_id,
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
