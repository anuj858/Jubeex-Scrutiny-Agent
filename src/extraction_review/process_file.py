"""Classify and LlamaSplit a bundled PDF, then stop.

``upload_compiled`` classifies, splits, and slices slot PDFs. Parse/extract
runs only after Submit (UI ``process-split-files``) or backend
``upload_separate``.
``upload_separate`` skips classify/split and runs process-split-files only.
"""

from __future__ import annotations

import asyncio
import hashlib
import io
import logging
import math
import os
import re
from collections.abc import Mapping, Sequence
from pathlib import PurePosixPath
from typing import TYPE_CHECKING, Annotated, Any, Literal
from urllib.parse import unquote, urlparse

import httpx
import pymupdf
from llama_cloud import AsyncLlamaCloud, BadRequestError, NotFoundError
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

from .bundle_slicer import (
    map_slot_pages,
    slice_bundle_pdf,
    validate_page_manifest_output,
    validate_shared_page_evidence,
    validate_sliced_page_ownership,
)
from .clients import get_llama_cloud_client, project_id
from .config import (
    ClassifyConfig,
    SplitConfig,
    dump_api_configuration,
    with_config_identity,
)
from .document_parts import (
    ANNEXURE_FAMILY,
    family_split_name,
    format_page_span,
    normalize_part_name,
    page_parts_from_split,
    page_ranges,
    parts_on_page,
)
from .job_timing import (
    attach_timing,
    elapsed_seconds,
    start_timer,
    timing_payload,
    timing_status_message,
    uploaded_filename,
)
from .llama_usage import (
    collect_optional_usage,
    summarize_llamacloud_usage,
    usage_status_message,
)
from .logging_utils import safe_url_for_log
from .s3_artifacts import STEP_SPLIT, upload_step_json
from .split_upload import (
    UNDEFINED_SLOT_ID,
    SplitUploadError,
    dynamic_upload_slot,
    type_catalog,
    ui_catalog,
)

if TYPE_CHECKING:
    from .structure_split import PageUnit

logger = logging.getLogger(__name__)

DISCRIMINATOR_FIELD = "petition_type"
_TRUTHY = frozenset({"1", "true", "yes"})
_SPLIT_RECONCILIATION_MODES = frozenset({"llama_only", "targeted", "legacy_full"})
DEFAULT_SLOT_UPLOAD_CONCURRENCY = 8
_FILE_BYTES: dict[str, bytes] = {}


def positive_int_env(name: str, default: int) -> int:
    raw = (os.getenv(name) or "").strip()
    if not raw:
        return default
    try:
        value = int(raw)
    except ValueError:
        return default
    return value if value > 0 else default


def slot_upload_concurrency() -> int:
    return positive_int_env("SLOT_UPLOAD_CONCURRENCY", DEFAULT_SLOT_UPLOAD_CONCURRENCY)


def split_reconciliation_mode() -> str:
    """Resolve the split mode; the staged boundary/review flow is the default.

    ``SPLIT_RECONCILIATION_MODE`` is authoritative when valid. The legacy
    boolean remains an explicit compatibility switch, while ``llama_only``
    and ``legacy_full`` remain available for controlled rollback/comparison.
    """
    configured = (os.getenv("SPLIT_RECONCILIATION_MODE") or "").strip().lower()
    if configured in _SPLIT_RECONCILIATION_MODES:
        return configured
    legacy_value = os.getenv("SPLIT_PYTHON_RULES_ENABLED")
    if legacy_value is None:
        return "targeted"
    legacy_enabled = legacy_value.strip().lower() in _TRUTHY
    return "targeted" if legacy_enabled else "llama_only"


def split_python_rules_enabled() -> bool:
    """Compatibility predicate for local OCR, reconciliation and audit."""
    return split_reconciliation_mode() != "llama_only"


def split_output_selection(value: str | None = None) -> dict[str, Any]:
    """Resolve the optional output filter without changing page classification.

    ``SPLIT_OUTPUT_DOCUMENTS`` accepts ``Annexures, Cover Page`` (the default),
    ``All``, or a comma/semicolon/pipe-separated list of canonical document
    names. Parsing and auditing always cover the
    complete source PDF; this setting only selects which sliced PDFs are
    emitted.
    """
    raw = os.getenv("SPLIT_OUTPUT_DOCUMENTS") if value is None else value
    configured = (raw or "Annexures, Cover Page").strip()
    if not configured:
        configured = "Annexures, Cover Page"
    requested = [
        token.strip() for token in re.split(r"[,;|\n]+", configured) if token.strip()
    ]
    if not requested:
        requested = ["Annexures", "Cover Page"]
    all_tokens = [token for token in requested if token.casefold() == "all"]
    if all_tokens and len(requested) != 1:
        raise ValueError("SPLIT_OUTPUT_DOCUMENTS cannot combine 'All' with other names")
    if all_tokens:
        return {"mode": "all", "configured_value": configured, "requested": ["All"]}

    normalized = list(dict.fromkeys(normalize_part_name(token) for token in requested))
    if any(not token for token in normalized):
        raise ValueError("SPLIT_OUTPUT_DOCUMENTS contains an empty document name")
    return {
        "mode": "selected",
        "configured_value": configured,
        "requested": normalized,
    }


def _split_output_matches(item: Any, selection: Mapping[str, Any]) -> bool:
    if selection.get("mode") == "all":
        return True
    wanted = set(selection.get("requested") or ())
    candidates = {
        normalize_part_name(str(getattr(item, "label", "") or "")),
        normalize_part_name(str(getattr(item, "slot_id", "") or "").replace("_", " ")),
    }
    for candidate in candidates - {""}:
        family = family_split_name(candidate)
        for requested in wanted:
            if candidate == requested:
                return True
            if requested == ANNEXURE_FAMILY and family == ANNEXURE_FAMILY:
                return True
            if requested == family and family in {ANNEXURE_FAMILY, "Application"}:
                return True
    return False


def filter_split_outputs(
    items: Sequence[Any], selection: Mapping[str, Any]
) -> tuple[list[Any], dict[str, Any]]:
    """Select output slices while retaining a trace of omitted source pages."""
    selected = [item for item in items if _split_output_matches(item, selection)]
    selected_slots = {str(item.slot_id) for item in selected}
    all_slots = {str(item.slot_id) for item in items}
    requested = list(selection.get("requested") or ())
    matched = [
        name
        for name in requested
        if name == "All"
        or any(
            _split_output_matches(item, {"mode": "selected", "requested": [name]})
            for item in items
        )
    ]
    selected_pages = {int(page) for item in selected for page in item.pages}
    excluded_pages = {
        int(page)
        for item in items
        if str(item.slot_id) not in selected_slots
        for page in item.pages
    } - selected_pages
    report = {
        **dict(selection),
        "matched": matched,
        "unmatched": [name for name in requested if name not in matched],
        "selected_slots": sorted(selected_slots),
        "excluded_slots": sorted(all_slots - selected_slots),
        "source_pages_excluded_from_outputs": sorted(excluded_pages),
        "status": "all" if selection.get("mode") == "all" else "filtered",
    }
    return selected, report


def attach_split_output_trace(
    page_manifest: list[dict[str, Any]], slices: Sequence[Any]
) -> None:
    """Finish each page's audit with the actual generated PDF destination."""
    destinations: dict[int, list[dict[str, str]]] = {}
    for sliced in slices:
        for page in sliced.pages:
            destinations.setdefault(int(page), []).append(
                {"slot_id": str(sliced.slot_id), "filename": str(sliced.filename)}
            )
    for row in page_manifest:
        outputs = destinations.get(int(row["source_page"]), [])
        row["output_pdfs"] = outputs
        trace = row.setdefault("assignment_trace", [])
        if not any(event.get("stage") == "final_page_assignment" for event in trace):
            trace.append(
                {
                    "stage": "final_page_assignment",
                    "status": "committed",
                    "segment_id": row.get("segment_id"),
                    "canonical_type": row.get("effective_owner"),
                    "labels": list(row.get("assigned_labels") or ()),
                    "reason": "final_page_map_with_unavailable_stage_history",
                    "evidence": ["final_page_map"],
                }
            )
        trace.append(
            {
                "stage": "pdf_slice",
                "status": row.get(
                    "output_disposition", "included" if outputs else "unassigned"
                ),
                "segment_id": row.get("segment_id"),
                "canonical_type": row.get("effective_owner"),
                "output_pdfs": outputs,
                "reason": (
                    "slice_uses_final_page_ownership"
                    if outputs
                    else "no_output_slot_for_final_page_assignment"
                ),
                "evidence": ["final_page_map", "catalog_slot_mapping"],
            }
        )


def remember_file_bytes(file_id: str | None, data: bytes | None) -> None:
    key = (file_id or "").strip()
    if not key or not data:
        return
    _FILE_BYTES[key] = data


def take_file_bytes(file_id: str | None) -> bytes | None:
    key = (file_id or "").strip()
    if not key:
        return None
    return _FILE_BYTES.pop(key, None)


def clear_file_bytes_cache() -> None:
    _FILE_BYTES.clear()


def upload_sliced_slot_pdfs() -> bool:
    """Upload sliced slot PDFs to LlamaCloud for the split-upload form.

    The UI needs a LlamaCloud ``file_id`` per slot to fill Upload, enable
    Download, and Submit parse/extract. Skip only when a backend explicitly
    sets ``SKIP_SLOT_PDF_UPLOAD``.
    """
    skip = (os.getenv("SKIP_SLOT_PDF_UPLOAD") or "").strip().lower()
    return skip not in _TRUTHY


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
PETITION_SPLIT_JOB_TYPES = frozenset({"split_petition"})
VERIFY_JOB_TYPES = frozenset({"verify_document"})
EXTRACT_ONLY_JOB_TYPES = frozenset({"extract_only"})
EXTRACT_FRONT_JOB_TYPES = frozenset({"extract_front"})
FRONT_EXTRACT_PAGE_LIMIT = 30
FRONT_EXTRACT_END_PAGE_LIMIT = 5
FRONT_EXTRACT_SMALL_BOOK_PAGES = 200
FRONT_EXTRACT_OPENING_CAP = 120


def front_extract_opening_pages(total: int) -> int:
    """How many opening pages to parse from a compiled PDF.

    A book of 200 pages or fewer uses 30 pages. A longer book starts from
    20 percent and that share shrinks as the book grows, so 700 pages keeps
    75 opening pages and 1000 pages keeps 90, never more than 120.
    """
    if total <= FRONT_EXTRACT_SMALL_BOOK_PAGES:
        return min(FRONT_EXTRACT_PAGE_LIMIT, max(total, 0))
    grown = math.ceil(
        0.20 * math.sqrt(FRONT_EXTRACT_SMALL_BOOK_PAGES * total)
    )
    return min(FRONT_EXTRACT_OPENING_CAP, total, grown)


def pdf_page_count(pdf_bytes: bytes) -> int:
    source = pymupdf.open(stream=pdf_bytes, filetype="pdf")
    try:
        return source.page_count
    finally:
        source.close()


def first_pages_pdf(
    pdf_bytes: bytes,
    page_limit: int = FRONT_EXTRACT_PAGE_LIMIT,
    end_page_limit: int = FRONT_EXTRACT_END_PAGE_LIMIT,
) -> bytes:
    """Keep the opening pages and the last pages, without repeating the overlap.

    The last pages are where the Vakalatnama sits, so the advocate on record
    can be read from there.
    """
    if page_limit < 1:
        raise ValueError("page_limit must be at least 1")
    if end_page_limit < 0:
        raise ValueError("end_page_limit must be at least 0")
    source = pymupdf.open(stream=pdf_bytes, filetype="pdf")
    try:
        total = source.page_count
        if total < 1:
            raise ValueError("Compiled PDF has no pages")
        opening = min(page_limit, total)
        end_start = max(opening, total - end_page_limit)
        if end_start >= total:
            if opening == total:
                return pdf_bytes
            sliced = pymupdf.open()
            try:
                sliced.insert_pdf(source, from_page=0, to_page=opening - 1)
                return sliced.tobytes()
            finally:
                sliced.close()
        sliced = pymupdf.open()
        try:
            sliced.insert_pdf(source, from_page=0, to_page=opening - 1)
            sliced.insert_pdf(source, from_page=end_start, to_page=total - 1)
            return sliced.tobytes()
        finally:
            sliced.close()
    finally:
        source.close()
INDEX_JOB_TYPES = frozenset({"index_parsed"})
FULL_JOB_TYPES = (
    frozenset(
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
    | PETITION_SPLIT_JOB_TYPES
)
CLASSIFY_COMPILED_JOB_TYPES = FULL_JOB_TYPES - PETITION_SPLIT_JOB_TYPES
SPLIT_JOB_TYPES = (
    frozenset({"split", "parts", "upload_separate", "upload_split"})
    | EXTRACT_ONLY_JOB_TYPES
    | INDEX_JOB_TYPES
)
CATALOG_JOB_TYPES = [
    "upload_compiled",
    "upload_separate",
    "split_petition",
    "verify_document",
    "extract_only",
    "extract_front",
    "index_parsed",
]
DEFAULT_COMPILED_FILING_TYPE = "SLP_CIVIL"
SWAGGER_PLACEHOLDERS = frozenset({"string", "str", "none", "null"})
_ANNEXURE_SLOT_RE = re.compile(r"annexure_([a-z])_?(\d+)")
_APPLICATION_SLOT_RE = re.compile(r"application_?(\d+)")
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
    "aor_declaration": "aor_s_declaration",
    # Historical compiled-split payloads used this ambiguous id for the
    # Certificate.  Keep that meaning while exposing the Declaration under
    # the new, explicit aor_s_declaration id.
    "aors_declaration": "aor_s_certificate",
    "aors_certificate": "aor_s_certificate",
    "aor_certificate": "aor_s_certificate",
    "office_report_limitation": "office_report_on_limitation",
    "vakalatnama": "vakalatnama_appearance",
    "memo_of_appearance": "vakalatnama_appearance",
    "filing_memorandum": "filing_memo",
    "aor_s_certificate": "aor_s_certificate",
    "aor_s_declaration": "aor_s_declaration",
    "listing_proforma": "listing_proforma",
    "custody_certificate": "custody_certificate",
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
    if raw in EXTRACT_FRONT_JOB_TYPES:
        return "extract_front"
    if raw in SPLIT_JOB_TYPES:
        return "split"
    if raw in VERIFY_JOB_TYPES:
        return "verify"
    return None


def parse_edited_flag(value: object) -> bool:
    """Accept true/false, yes/no, 1/0 for the index ``edited`` flag."""
    if value is None or value == "":
        return False
    if isinstance(value, bool):
        return value
    if isinstance(value, (int, float)) and not isinstance(value, bool):
        return value != 0
    return str(value).strip().lower() in _TRUTHY | {"yes"}


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


def _numbered_slot_id(stem: str, known: list[str]) -> str | None:
    annexure = _ANNEXURE_SLOT_RE.search(stem)
    if annexure:
        series = annexure.group(1)
        number = int(annexure.group(2))
        candidate = f"annexure_{series}{number}"
        if 1 <= number <= 999 and (
            not known
            or candidate in known
            or "annexures" in known
            or "annexure_p1" in known
            or any(kid.startswith(f"annexure_{series}") for kid in known)
        ):
            return candidate
    application = _APPLICATION_SLOT_RE.search(stem)
    if application:
        number = int(application.group(1))
        candidate = f"application_{number}"
        if 1 <= number <= 999 and (
            not known
            or candidate in known
            or "applications" in known
            or "application_1" in known
        ):
            return candidate
    return None


def slot_id_from_name(name: str, filing_type: str | None = None) -> str | None:
    """Map `01_Petition.pdf` / `List of Dates` onto a split-upload slot id."""
    stem = normalize_document_stem(name)
    if not stem:
        return None
    if stem in FULL_PETITION_SLOTS or stem == "full_petition":
        return "fullpetition"
    known = _known_slot_ids(filing_type)
    if stem in known:
        return stem
    numbered = _numbered_slot_id(stem, known)
    if numbered:
        return numbered
    if stem.startswith("annexure") and (not known or "annexures" in known):
        return "annexures"
    if stem.startswith("application") and (not known or "applications" in known):
        return "applications"
    alias = _SLOT_NAME_ALIASES.get(stem)
    if alias and (not known or alias in known or alias == "fullpetition"):
        return alias
    matches = [slot_id for slot_id in known if slot_id in stem or stem in slot_id]
    if matches:
        return max(matches, key=len)
    return None


def coerce_slot_id(slot: str, *, filing_type: str | None) -> str:
    known = set(_known_slot_ids(filing_type))
    key = (slot or "").strip()
    if key in known:
        return key
    underscored = key.replace("-", "_").lower()
    if underscored in known:
        return underscored
    alias = _SLOT_NAME_ALIASES.get(underscored)
    if alias in known:
        return alias
    dynamic = dynamic_upload_slot(underscored) or dynamic_upload_slot(key)
    if dynamic and (
        not known
        or "annexures" in known
        or "applications" in known
        or dynamic.id == "custody_certificate"
        or dynamic.id in known
    ):
        return dynamic.id
    from_name = slot_id_from_name(key, filing_type)
    if from_name in known or (from_name and dynamic_upload_slot(from_name)):
        return from_name
    return UNDEFINED_SLOT_ID


def resolve_document_slot(
    item: FilingPartIn,
    *,
    mode: str,
    filing_type: str | None,
) -> str:
    explicit = blank_or_placeholder(item.slot_id)
    if explicit:
        return (
            coerce_slot_id(explicit, filing_type=filing_type)
            if mode == "split"
            else explicit
        )
    if mode == "full":
        return "fullpetition"
    names = [
        item.filename or "",
        _filename_from_url(item.file_url or "") if item.file_url else "",
    ]
    for name in names:
        resolved = slot_id_from_name(name, filing_type)
        if resolved and resolved not in FULL_PETITION_SLOTS:
            return coerce_slot_id(resolved, filing_type=filing_type)
    if mode == "split":
        return UNDEFINED_SLOT_ID
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
        return DEFAULT_COMPILED_FILING_TYPE, type_catalog(DEFAULT_COMPILED_FILING_TYPE)
    except SplitUploadError:
        allowed = ", ".join(sorted(ui_catalog()))
    raise SplitUploadError(
        f"Classified as {classified_key or '(unknown)'}, which cannot be sliced. "
        f"Send filing_type as one of: {allowed}"
    )


def split_labels_match_expected_slot(
    *,
    expected_slot: str | None,
    catalog: Any,
    page_parts: Mapping[int, Sequence[str] | str | None],
) -> bool:
    """True when LlamaSplit's dominant known slot is the named document type."""
    expected = (expected_slot or "").strip()
    if not expected or expected == UNDEFINED_SLOT_ID:
        return False
    pages_by_slot = map_slot_pages(catalog, page_parts)
    counts = {
        slot_id: len(pages)
        for slot_id, pages in pages_by_slot.items()
        if pages and slot_id != UNDEFINED_SLOT_ID
    }
    if not counts:
        return False
    dominant = max(counts, key=lambda slot_id: (counts[slot_id], slot_id))
    expected_dynamic = dynamic_upload_slot(expected)
    if expected_dynamic and expected_dynamic.parts == ("Annexures",):
        # Generic upload ordinals are not printed exhibit identities. Verify
        # the category returned by LlamaCloud, not an invented P/A number.
        actual_dynamic = dynamic_upload_slot(dominant)
        return dominant == "annexures" or bool(
            actual_dynamic
            and all(
                family_split_name(part) == "Annexures" for part in actual_dynamic.parts
            )
        )
    if expected_dynamic and all(
        family_split_name(part) == "Application" for part in expected_dynamic.parts
    ):
        if dominant == "applications":
            return True
        if dominant == "application_1":
            # The catalog mapper promotes an unnumbered Application to slot 1.
            # That synthetic ordinal cannot disprove the expected upload ID.
            # An explicit Application 1 label must still match only slot 1.
            unpromoted = map_slot_pages(catalog, page_parts, promote_repeatable=False)
            if "applications" in unpromoted and "application_1" not in unpromoted:
                return True
    return dominant == expected


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
    _file_id, _url, filename, _hash = compiled_source(event)
    if not filename and event.documents:
        filename = next(
            (
                (item.filename or "").strip()
                for item in event.documents
                if (item.filename or "").strip()
            ),
            None,
        )
    return {
        "job_type": event.job_type,
        "organization_id": org_id,
        "workspace_id": event.workspace_id,
        "user_id": event.user_id,
        "org_id": org_id,
        "filename": uploaded_filename(filename),
    }


class FileEvent(StartEvent):
    """Start from LlamaCloud file_id, a remote URL, or labeled documents.

    Backend (no Llama UI): POST this to ``process-file``.

    - ``job_type=upload_compiled`` (or ``full``): classify, slice, and upload
      slot PDFs, then stop. Parse/extract wait for Submit.
    - ``job_type=upload_separate`` (or ``split``) plus ``documents[]``:
      already-split files; runs process-split-files (parse, extract).
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
    edited: bool | None = None
    parsed_slots: list[str] = Field(default_factory=list)
    agent_data_id: str | None = None
    special_category: str | None = None
    court: str | None = None

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
        if "edited" in params:
            params["edited"] = parse_edited_flag(params.get("edited"))
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
        if mode == "extract_front":
            if len(self.documents) != 1:
                raise ValueError(
                    "documents[] must contain the one compiled PDF"
                )
            slot_mode = "full"
        elif mode == "verify":
            slot_mode = "split"
        else:
            slot_mode = mode
        resolved: list[FilingPartIn] = []
        for item in self.documents:
            resolved.append(
                item.model_copy(
                    update={
                        "slot_id": resolve_document_slot(
                            item, mode=slot_mode, filing_type=self.filing_type
                        )
                    }
                )
            )
        self.documents = resolved
        job = (self.job_type or "").strip().lower()
        if mode in {"split", "verify"}:
            label = "upload_separate" if mode == "split" else "verify_document"
            if not self.documents:
                raise ValueError(f"documents[] is required when job_type is {label}")
            if not (self.filing_type or "").strip():
                raise ValueError(f"filing_type is required when job_type is {label}")
            try:
                catalog = type_catalog(self.filing_type)
            except SplitUploadError as exc:
                allowed = ", ".join(sorted(ui_catalog()))
                raise ValueError(f"{exc}. Use one of: {allowed}") from exc
            allowed_slots = catalog.slot_by_id()
            coerced: list[FilingPartIn] = []
            for item in self.documents:
                slot = coerce_slot_id(
                    (item.slot_id or "").strip(),
                    filing_type=self.filing_type,
                )
                if slot not in allowed_slots and dynamic_upload_slot(slot) is None:
                    slot = UNDEFINED_SLOT_ID
                coerced.append(item.model_copy(update={"slot_id": slot}))
            self.documents = coerced
            if mode == "verify":
                return self
            if job in INDEX_JOB_TYPES and not (self.agent_data_id or "").strip():
                raise ValueError(
                    "agent_data_id is required when job_type is index_parsed"
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
    file_id: str | None = None
    file_hash: str | None = None
    filename: str | None = None
    document_id: str | None = None
    download_url: str | None = None
    name: str | None = None
    label: str | None = None
    page_span: list[dict[str, int]] | None = None
    duplicate: bool = False
    reason: str | None = None


class SourceDocument(BaseModel):
    name: str | None = None
    document_id: str | None = None
    download_url: str | None = None
    slot_id: str | None = None
    file_id: str | None = None
    file_hash: str | None = None


class VerifiedDocument(BaseModel):
    name: str | None = None
    match: bool


class BundlePrepared(StopEvent):
    filing_type: str
    parts: list[PreparedPart] = Field(default_factory=list)
    has_duplicates: bool = False
    documents: list[SourceDocument] = Field(default_factory=list)
    slot_pages: dict[str, list[dict[str, int]]] = Field(default_factory=dict)
    agent_data_id: str | None = None
    job_type: str | None = None
    organization_id: str | None = None
    workspace_id: str | None = None
    user_id: str | None = None
    org_id: str | None = None
    filename: str | None = None
    classify_split_seconds: float | None = None
    timing: dict[str, Any] | None = None
    usage: dict[str, Any] | None = None
    llama_split: dict[str, Any] | None = None
    parsed_slots: list[str] = Field(default_factory=list)
    match: bool | None = None
    verified_documents: list[VerifiedDocument] = Field(default_factory=list)
    report: dict[str, Any] | None = None


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
    started_at: float | None = None
    llamacloud_jobs: list[dict[str, Any]] = Field(default_factory=list)


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


# Official Split API field limits; reject invalid rules instead of cutting them.
# https://developers.llamaindex.ai/reference/resources/split/methods/create/
_SPLIT_CUSTOM_INSTRUCTIONS_MAX = 5000
_SPLIT_CATEGORY_NAME_MAX = 200
_SPLIT_CATEGORY_DESCRIPTION_MAX = 2000


def _split_strategy_payload(raw: Any) -> dict[str, Any] | None:
    if raw is None:
        return None
    if hasattr(raw, "model_dump"):
        dumped = raw.model_dump(exclude_none=True)
    elif isinstance(raw, dict):
        dumped = {key: value for key, value in raw.items() if value is not None}
    else:
        return None
    instructions = dumped.get("custom_instructions")
    if instructions is not None:
        if not isinstance(instructions, str):
            raise ValueError(
                "split.splitting_strategy.custom_instructions must be a string."
            )
        if len(instructions) > _SPLIT_CUSTOM_INSTRUCTIONS_MAX:
            raise ValueError(
                "split.splitting_strategy.custom_instructions has "
                f"{len(instructions)} characters; the Split API limit is "
                f"{_SPLIT_CUSTOM_INSTRUCTIONS_MAX}. Shorten the rules before submitting; "
                "instructions will not be truncated."
            )
    return dumped or None


def _split_api_configuration(split_config: SplitConfig) -> dict[str, Any]:
    """Validate inline Split rules and preserve their complete supported payload."""
    dumped = dump_api_configuration(split_config)
    categories: list[dict[str, str]] = []
    for index, item in enumerate(dumped.get("categories") or []):
        name = item.get("name") if isinstance(item, dict) else None
        field = f"split.categories[{index}]"
        if not isinstance(name, str) or not name.strip():
            raise ValueError(f"{field}.name must be a non-empty string.")
        if len(name) > _SPLIT_CATEGORY_NAME_MAX:
            raise ValueError(
                f"{field}.name has {len(name)} characters; the Split API limit is "
                f"{_SPLIT_CATEGORY_NAME_MAX}. Shorten the category name before submitting."
            )
        category = {"name": name}
        description = item.get("description")
        if description is not None:
            if not isinstance(description, str):
                raise ValueError(f"{field}.description must be a string.")
            if len(description) > _SPLIT_CATEGORY_DESCRIPTION_MAX:
                raise ValueError(
                    f"{field}.description for {name!r} has {len(description)} characters; "
                    f"the Split API limit is {_SPLIT_CATEGORY_DESCRIPTION_MAX}. "
                    "Shorten the description before submitting; rules will not be truncated."
                )
            if description:
                category["description"] = description
        categories.append(category)
    dumped["categories"] = categories
    strategy = _split_strategy_payload(
        dumped.get("splitting_strategy")
        or getattr(split_config, "splitting_strategy", None)
    )
    if strategy:
        dumped["splitting_strategy"] = strategy
    else:
        dumped.pop("splitting_strategy", None)
    return dumped


def _json_value(value: Any) -> Any:
    """SDK object as JSON data. Does not rename or reformat fields."""
    if hasattr(value, "model_dump"):
        return value.model_dump(mode="json")
    if isinstance(value, dict):
        return {key: _json_value(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_value(item) for item in value]
    if hasattr(value, "__dict__"):
        return {
            key: _json_value(item)
            for key, item in vars(value).items()
            if not key.startswith("_")
        }
    return value


async def _split_page_parts(
    client: AsyncLlamaCloud,
    *,
    file_id: str | None,
    split_config: SplitConfig | None,
    filename: str | None = None,
    api_configuration: dict[str, Any] | None = None,
) -> tuple[dict[int, list[str]], str, dict[str, Any]]:
    """Label pages with Split categories.

    Returns page labels, the split job id, and the request/response pair
    exactly as sent to and returned by LlamaSplit.
    """
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

    if api_configuration is not None:
        sent = {
            "file_input": file_id,
            "configuration": api_configuration,
            "project_id": project_id,
        }
        job = await client.split.create(
            file_input=file_id,
            configuration=api_configuration,
            project_id=project_id,
        )
    elif split_config.configuration_id:
        sent: dict[str, Any] = {
            "file_input": file_id,
            "configuration_id": split_config.configuration_id,
            "project_id": project_id,
        }
        job = await client.split.create(
            file_input=file_id,
            configuration_id=split_config.configuration_id,
            project_id=project_id,
        )
    else:
        configuration = _split_api_configuration(split_config)
        sent = {
            "file_input": file_id,
            "configuration": configuration,
            "project_id": project_id,
        }
        job = await client.split.create(
            file_input=file_id,
            configuration=configuration,
            project_id=project_id,
        )
    completed = await _wait_for_split(client, job.id)
    exchange = {"sent": sent, "returned": _json_value(completed)}
    mapping = page_parts_from_split(completed)
    segments = (exchange["returned"].get("result") or {}).get("segments") or []
    if not mapping and not any(segment.get("pages") for segment in segments):
        raise RuntimeError(
            f"Split finished for {label} but labelled no pages. "
            "Scrutiny cannot filter Listing Proforma / Main Petition / checklist parts."
        )
    # An entirely uncategorized result is still usable: preserve the pages in
    # Unidentified rather than manufacturing local labels or dropping the PDF.
    return mapping, str(getattr(completed, "id", None) or job.id), exchange


def _boundary_detection_configuration(
    page_units: Sequence[PageUnit], *, page_count: int
) -> dict[str, Any]:
    """Build a compact, boundary-only Split request from parsed page evidence."""
    evidence: list[tuple[int, int, str]] = []
    selected = list(page_units)
    for unit in selected:
        headings = getattr(unit, "headings", ())
        annexure_labels = getattr(unit, "annexure_labels", ())
        paragraph_starts = getattr(unit, "paragraph_starts", ())
        closing_section = getattr(unit, "closing_section", None)
        text_quality = getattr(unit, "text_quality", "usable")
        cues = [*headings[:2], *annexure_labels[:2]]
        if paragraph_starts:
            cues.append("paragraphs=" + ",".join(paragraph_starts[:3]))
        if closing_section:
            cues.append("closing=" + closing_section)
        if getattr(unit, "signature_detected", False):
            cues.append("signature_block")
        if getattr(unit, "sentence_continues_previous", False):
            cues.append("continues_sentence_from_previous")
        if getattr(unit, "paragraph_continues_previous", False):
            cues.append("continues_paragraph_from_previous")
        if getattr(unit, "printed_page", None):
            cues.append("folio=" + unit.printed_page)
        if text_quality != "usable":
            cues.append("extraction=" + text_quality)
        if not cues:
            excerpt = re.sub(r"\s+", " ", getattr(unit, "text", "")).strip()[:100]
            if excerpt:
                cues.append("text=" + excerpt)
        # Strong starts must compete on evidence, not on their position in the
        # file. Long bundles used to lose every cue after the first ~50 pages.
        priority = (
            10 * bool(annexure_labels)
            + 8 * bool(headings)
            + 3 * bool(closing_section)
            + 2 * bool(paragraph_starts)
            + bool(getattr(unit, "signature_detected", False))
        )
        evidence.append(
            (unit.pdf_page, priority, f"PDF page {unit.pdf_page}: " + "; ".join(cues))
        )
    prefix = (
        "BOUNDARY DETECTION ONLY. Do not classify legal document types. "
        f"Analyze all {page_count} physical PDF pages and propose document starts, "
        "continuations, ends, and uncertain transitions. Use the page evidence below "
        "and neighboring-page context. Preserve page order; do not omit pages. Return "
        "one segment per plausible complete document, and do not merge merely because "
        "a continuation page lacks a repeated heading. Mark ambiguous transitions for review.\n"
        "Parsed page cues (physical page order):\n"
    )
    # Select across the entire file, with high-evidence pages first. Reserve
    # space for the first/last page and distribute the rest by physical-page
    # bucket so late annexures are not silently excluded by prompt truncation.
    omission = "[Page cues are sampled across the PDF; inspect uncued pages in the PDF itself.]"
    budget = _SPLIT_CUSTOM_INSTRUCTIONS_MAX - len(prefix) - len(omission) - 2
    bucket_width = max(1, (page_count + 15) // 16)
    buckets: dict[int, list[tuple[int, int, str]]] = {}
    chosen: list[tuple[int, int, str]] = []
    used = 0
    endpoints = {1, page_count}
    for item in evidence:
        if item[0] in endpoints and used + len(item[2]) + 1 <= budget:
            chosen.append(item)
            used += len(item[2]) + 1
            continue
        buckets.setdefault((item[0] - 1) // bucket_width, []).append(item)
    for bucket in buckets.values():
        bucket.sort(key=lambda item: (-item[1], item[0]))
    while any(buckets.values()):
        progressed = False
        for bucket in buckets.values():
            if not bucket:
                continue
            item = bucket.pop(0)
            line = item[2]
            if used + len(line) + 1 > budget:
                continue
            chosen.append(item)
            used += len(line) + 1
            progressed = True
        if not progressed:
            break
    chosen.sort(key=lambda item: item[0])
    cue_lines = [item[2] for item in chosen]
    instruction = prefix + "\n".join(cue_lines)
    if len(chosen) < len(evidence):
        instruction += "\n" + omission
    if len(instruction) > _SPLIT_CUSTOM_INSTRUCTIONS_MAX:
        raise ValueError(
            "Boundary-only instructions exceed the Split API character limit"
        )
    return {
        "product_type": "split_v1",
        "categories": [
            {
                "name": "Document Segment",
                "description": "A complete contiguous document segment; this stage detects boundaries only.",
            }
        ],
        "splitting_strategy": {"custom_instructions": instruction},
    }


def _validated_segment_pages(
    segment: Any, *, page_count: int, segment_id: str
) -> tuple[list[int], list[dict[str, Any]]]:
    """Validate a model's original page list before deriving any range from it."""
    issues: list[dict[str, Any]] = []

    def add(code: str, **details: Any) -> None:
        issues.append({"code": code, "segment_id": segment_id, **details})

    if not isinstance(segment, Mapping):
        add("segment_is_not_an_object")
        return [], issues
    raw_pages = segment.get("pages")
    if not isinstance(raw_pages, (list, tuple)) or not raw_pages:
        add("segment_pages_missing_or_invalid")
        return [], issues

    pages: list[int] = []
    for value in raw_pages:
        if type(value) is not int:
            add("invalid_page_number", value_type=type(value).__name__)
            continue
        pages.append(value)
    if len(set(pages)) != len(pages):
        add("duplicate_page_within_segment", pages=pages)
    if pages != sorted(pages):
        add("pages_out_of_order", pages=pages)
    if any(page < 1 or page > page_count for page in pages):
        add(
            "page_out_of_range",
            pages=[page for page in pages if page < 1 or page > page_count],
            page_count=page_count,
        )
    if (
        pages
        and pages == sorted(set(pages))
        and pages != list(range(pages[0], pages[-1] + 1))
    ):
        add("segment_pages_not_contiguous", pages=pages)

    has_start = "start_page" in segment
    has_end = "end_page" in segment
    if has_start != has_end:
        add("segment_range_incomplete")
    elif has_start and has_end:
        start, end = segment.get("start_page"), segment.get("end_page")
        if (
            type(start) is not int
            or type(end) is not int
            or start < 1
            or end < start
            or end > page_count
        ):
            add("segment_range_invalid")
        elif pages and (start != pages[0] or end != pages[-1]):
            add("segment_range_conflicts_with_pages", start_page=start, end_page=end)

    return pages, issues


def _boundary_proposals_from_exchange(
    exchange: Mapping[str, Any], *, page_count: int
) -> dict[str, Any]:
    """Validate boundary page lists and report gaps/shares instead of guessing ranges."""
    response = exchange.get("returned") or {}
    result = response.get("result") or {}
    raw_segments = result.get("segments") or []
    issues: list[dict[str, Any]] = []
    proposals: list[dict[str, Any]] = []
    if not isinstance(raw_segments, (list, tuple)):
        raw_segments = []
        issues.append({"code": "boundary_segments_invalid"})

    accepted_pages: list[list[int]] = []
    accepted_segment_indexes: list[int] = []
    pages_by_segment_index: dict[int, list[int]] = {}
    for index, segment in enumerate(raw_segments, start=1):
        segment_id = f"boundary-{index}"
        pages, segment_issues = _validated_segment_pages(
            segment, page_count=page_count, segment_id=segment_id
        )
        issues.extend(segment_issues)
        if not pages or segment_issues:
            continue
        accepted_pages.append(pages)
        accepted_segment_indexes.append(index)
        pages_by_segment_index[index] = pages
        proposals.append(
            {
                "segment_id": segment_id,
                "start_page": pages[0],
                "end_page": pages[-1],
                "pages": pages,
                "start_proposal": True,
                "end_proposal": True,
                "continuation_pages": pages[1:],
                "source": "llama_boundary_detection",
                "final": False,
            }
        )

    if not raw_segments:
        issues.append({"code": "boundary_segments_missing"})
    if [pages[0] for pages in accepted_pages] != sorted(
        pages[0] for pages in accepted_pages
    ):
        issues.append({"code": "boundary_segments_out_of_order"})

    page_segments: dict[int, list[int]] = {}
    for segment_index, pages in zip(
        accepted_segment_indexes, accepted_pages, strict=True
    ):
        for page in pages:
            page_segments.setdefault(page, []).append(segment_index)
    shared_boundary_pages: list[int] = []
    for page, segment_indexes in sorted(page_segments.items()):
        if len(segment_indexes) < 2:
            continue
        adjacent_edge_share = (
            len(segment_indexes) == 2
            and segment_indexes[1] == segment_indexes[0] + 1
            and pages_by_segment_index[segment_indexes[0]][-1] == page
            and pages_by_segment_index[segment_indexes[1]][0] == page
        )
        if adjacent_edge_share:
            shared_boundary_pages.append(page)
            issues.append({"code": "shared_boundary_page", "source_page": page})
        else:
            issues.append(
                {
                    "code": "contradictory_boundary_overlap",
                    "source_page": page,
                    "segment_indexes": segment_indexes,
                }
            )

    covered = set(page_segments)
    missing_pages = sorted(set(range(1, page_count + 1)) - covered)
    if missing_pages:
        issues.append({"code": "boundary_pages_missing", "pages": missing_pages})
    return {
        "proposals": proposals,
        "issues": issues,
        "missing_pages": missing_pages,
        "shared_boundary_pages": shared_boundary_pages,
        "status": "clear" if not issues else "needs_review",
    }


def _classification_configuration(
    split_config: SplitConfig,
    proposals: Sequence[Mapping[str, Any]],
    *,
    proposal_issues: Sequence[Mapping[str, Any]] = (),
) -> tuple[dict[str, Any] | None, str]:
    """Apply candidate ranges to the canonical split_v1 classifier if they fit."""
    if getattr(split_config, "configuration_id", None):
        return None, "remote_configuration_cannot_be_augmented"
    if not proposals:
        return None, "no_valid_boundary_guidance"
    blocking_issues = [
        issue
        for issue in proposal_issues
        if issue.get("code") != "shared_boundary_page"
    ]
    if blocking_issues:
        return None, "boundary_proposals_invalid"
    for item in proposals:
        if not isinstance(item, Mapping):
            return None, "boundary_proposals_invalid"
        pages = item.get("pages")
        if (
            not isinstance(pages, list)
            or not pages
            or any(type(page) is not int for page in pages)
            or pages != sorted(set(pages))
            or pages != list(range(pages[0], pages[-1] + 1))
            or item.get("start_page") != pages[0]
            or item.get("end_page") != pages[-1]
            or not item.get("segment_id")
        ):
            return None, "boundary_proposals_invalid"
    configuration = _split_api_configuration(split_config)
    page_lists = "; ".join(
        f"{item['segment_id']}={item['pages']}" for item in proposals
    )
    directive = (
        "\nCLASSIFICATION PASS: classify each proposed complete segment using the "
        "canonical split_v1 categories and rules. Keep candidate boundaries unless "
        "page evidence clearly shows the proposal is wrong. These are exact physical "
        "PDF page lists; do not fill gaps between listed pages: " + page_lists
    )
    strategy = dict(configuration.get("splitting_strategy") or {})
    instructions = str(strategy.get("custom_instructions") or "")
    if len(instructions) + len(directive) > _SPLIT_CUSTOM_INSTRUCTIONS_MAX:
        # Large paper-books can produce dozens of valid candidates. Repeating
        # every physical page number alongside the normal boundary-discovery
        # prompt exceeded SplitV1's 5,000-character limit, silently causing the
        # second pass to run without first-pass guidance. The classification
        # pass already has the canonical per-category rules, so replace the
        # boundary-discovery prompt with a compact range-only directive.
        compact_ranges = ";".join(
            f"{item['segment_id']}:{item['start_page']}"
            + (f"-{item['end_page']}" if item["end_page"] != item["start_page"] else "")
            for item in proposals
        )
        compact_directive = (
            "CLASSIFICATION PASS. Treat each listed physical PDF range as one "
            "candidate document and assign one canonical category to the complete "
            "range. Preserve these boundaries unless page evidence clearly proves "
            "one is wrong; do not infer pages outside a listed range. RANGES:"
            + compact_ranges
        )
        if len(compact_directive) > _SPLIT_CUSTOM_INSTRUCTIONS_MAX:
            return None, "candidate_ranges_exceed_custom_instruction_limit"
        strategy["custom_instructions"] = compact_directive
    else:
        strategy["custom_instructions"] = instructions + directive
    configuration["splitting_strategy"] = strategy
    status = (
        "sent_with_shared_boundary_pages"
        if any(issue.get("code") == "shared_boundary_page" for issue in proposal_issues)
        else "sent"
    )
    return configuration, status


def _classification_call_evidence(
    exchange: Mapping[str, Any], *, boundary_guidance_status: str
) -> dict[str, Any]:
    """Persist which kind of classification configuration was actually sent."""
    sent = exchange.get("sent") or {}
    if not isinstance(sent, Mapping):
        sent = {}
    inline = sent.get("configuration") or {}
    if not isinstance(inline, Mapping):
        inline = {}
    strategy = inline.get("splitting_strategy") or {}
    if not isinstance(strategy, Mapping):
        strategy = {}
    instructions = str(strategy.get("custom_instructions") or "")
    configuration_id = sent.get("configuration_id")
    return {
        "kind": "remote_configuration_id"
        if configuration_id
        else "inline_configuration",
        "configuration_id": configuration_id,
        "boundary_guidance_sent": (
            boundary_guidance_status.startswith("sent")
            and "These are exact physical PDF page lists" in instructions
        ),
        "boundary_guidance_status": boundary_guidance_status,
    }


def _split_page_review_reasons(
    boundary_validation: Mapping[str, Any],
    classification_report: Mapping[str, Any],
    sharing_validation: Mapping[str, Any] | None = None,
) -> dict[int, list[str]]:
    """Map explicit boundary/classification concerns back to physical pages."""
    reasons: dict[int, list[str]] = {}

    def add_pages(values: Any, code: str) -> None:
        if not isinstance(values, (list, tuple, set)):
            return
        for value in values:
            if type(value) is int:
                reasons.setdefault(value, []).append(code)

    add_pages(boundary_validation.get("missing_pages"), "boundary_page_missing")
    add_pages(classification_report.get("missing_pages"), "classification_page_missing")
    add_pages(
        classification_report.get("overlapping_pages"), "classification_page_overlap"
    )
    add_pages(
        classification_report.get("out_of_range_pages"),
        "classification_page_out_of_range",
    )
    for issue in boundary_validation.get("issues", ()):
        if not isinstance(issue, Mapping):
            continue
        code = str(issue.get("code") or "boundary_uncertain")
        add_pages(issue.get("pages"), f"boundary:{code}")
        add_pages([issue.get("source_page")], f"boundary:{code}")
    for issue in (sharing_validation or {}).get("issues", ()):
        if not isinstance(issue, Mapping):
            continue
        code = str(issue.get("code") or "shared_page_uncertain")
        add_pages(issue.get("pages"), f"sharing:{code}")
        add_pages([issue.get("source_page")], f"sharing:{code}")
    return reasons


def _segment_classification_report(
    exchange: Mapping[str, Any], *, page_count: int
) -> dict[str, Any]:
    """Validate canonical category output and exact source-page coverage."""
    response = exchange.get("returned") or {}
    result = response.get("result") or {}
    raw_segments = result.get("segments") or []
    segments: list[dict[str, Any]] = []
    owners: dict[int, list[int]] = {}
    issues: list[str] = []
    if not isinstance(raw_segments, (list, tuple)):
        raw_segments = []
        issues.append("classified_segments_invalid")
    normalized_segments: list[list[int]] = []
    observed_segments: list[list[int]] = []
    valid_segment_indexes: set[int] = set()
    for index, segment in enumerate(raw_segments, start=1):
        pages, page_issues = _validated_segment_pages(
            segment, page_count=page_count, segment_id=f"classified-{index}"
        )
        observed_segments.append(pages)
        issues.extend(issue["code"] for issue in page_issues)
        if not pages:
            continue
        for page in pages:
            owners.setdefault(page, []).append(index)
        if page_issues:
            continue
        valid_segment_indexes.add(index)
        category = str(segment.get("category") or "").strip()
        if not category:
            issues.append(f"segment_{index}_has_no_category")
        normalized_segments.append(pages)
        segments.append(
            {
                "segment_id": f"classified-{index}",
                "document_type": category or None,
                "start_page": pages[0],
                "end_page": pages[-1],
                "source_pages": pages,
                "category": category or None,
                "pages": pages,
            }
        )
    observed_in_range = {page for page in owners if 1 <= page <= page_count}
    missing = sorted(set(range(1, page_count + 1)) - observed_in_range)
    overlap = sorted(page for page, indexes in owners.items() if len(indexes) > 1)
    out_of_range = sorted(page for page in owners if not 1 <= page <= page_count)
    if missing:
        issues.append("classified_segments_missing_source_pages")
    if overlap:
        issues.append("classified_segments_overlap_source_pages")
    if out_of_range:
        issues.append("classified_segments_out_of_range")
    shared_boundary_pages: list[int] = []
    unsupported_overlaps: list[int] = []
    for page in overlap:
        indexes = owners[page]
        is_adjacent_edge_share = (
            len(indexes) == 2
            and indexes[1] == indexes[0] + 1
            and indexes[0] in valid_segment_indexes
            and indexes[1] in valid_segment_indexes
            and observed_segments[indexes[0] - 1][-1] == page
            and observed_segments[indexes[1] - 1][0] == page
        )
        if is_adjacent_edge_share:
            shared_boundary_pages.append(page)
        else:
            unsupported_overlaps.append(page)
    if [pages[0] for pages in normalized_segments] != sorted(
        pages[0] for pages in normalized_segments
    ):
        issues.append("classified_segments_out_of_order")
    return {
        "status": "clear" if not issues else "needs_review",
        "segments": segments,
        "source_page_count": page_count,
        "missing_pages": missing,
        "overlapping_pages": overlap,
        "shared_boundary_pages": shared_boundary_pages,
        "unsupported_overlapping_pages": unsupported_overlaps,
        "out_of_range_pages": out_of_range,
        "issues": issues,
        "boundary_segments": [
            {"category": item["category"], "pages": item["pages"]} for item in segments
        ],
    }


def _document_first_page_ownership(
    boundary_proposals: Sequence[Mapping[str, Any]],
    classification_report: Mapping[str, Any],
    *,
    page_count: int,
) -> tuple[dict[int, list[str]], list[dict[str, Any]], list[dict[str, Any]]]:
    """Make detected document boundaries authoritative over type classification.

    The first Split pass owns page segmentation.  The second pass may name a
    complete candidate segment, but its returned page lists cannot split,
    reorder, omit, or steal pages from that candidate.  When the classifier
    fragments a candidate, the category covering its opening page wins and the
    conflict is retained as review evidence.  Pages not covered by any valid
    boundary are explicitly routed to ``Undefined`` instead of being guessed.

    The returned segment dictionaries deliberately retain the existing
    ``category``/``pages`` contract used by the slicer while adding boundary,
    classification, confidence, and review evidence for the audit artifact.
    """
    classified = [
        item
        for item in classification_report.get("segments", ())
        if isinstance(item, Mapping)
        and item.get("category")
        and isinstance(item.get("source_pages"), list)
    ]
    page_parts: dict[int, list[str]] = {}
    segments: list[dict[str, Any]] = []
    issues: list[dict[str, Any]] = []
    covered: set[int] = set()

    for index, proposal in enumerate(boundary_proposals, start=1):
        raw_pages = proposal.get("pages") if isinstance(proposal, Mapping) else None
        if (
            not isinstance(raw_pages, list)
            or not raw_pages
            or any(
                type(page) is not int or not 1 <= page <= page_count
                for page in raw_pages
            )
            or raw_pages != sorted(set(raw_pages))
            or raw_pages != list(range(raw_pages[0], raw_pages[-1] + 1))
        ):
            issues.append(
                {
                    "code": "invalid_boundary_ignored",
                    "boundary_index": index,
                }
            )
            continue

        pages = list(raw_pages)
        page_set = set(pages)
        exact = [item for item in classified if item.get("source_pages") == pages]
        overlapping = [
            item
            for item in classified
            if page_set.intersection(item.get("source_pages") or ())
        ]
        opening = [
            item for item in overlapping if pages[0] in (item.get("source_pages") or ())
        ]
        categories = list(
            dict.fromkeys(str(item["category"]).strip() for item in overlapping)
        )
        review_reasons: list[str] = []
        type_evidence: list[str] = []

        if len(exact) == 1:
            category = str(exact[0]["category"]).strip()
            confidence = 1.0
            type_evidence.append("exact_complete_segment_classification")
        elif categories and len(categories) == 1:
            category = categories[0]
            confidence = 0.9
            type_evidence.append("consistent_overlapping_classification")
            if not exact:
                review_reasons.append("classifier_range_differs_from_boundary")
        elif len(opening) == 1:
            # The boundary pass established one document.  A category on an
            # internal page is body-text evidence, not a new owner.  Prefer the
            # opening page's type while making the disagreement reviewable.
            category = str(opening[0]["category"]).strip()
            confidence = 0.65
            type_evidence.append("opening_page_classification")
            review_reasons.append("conflicting_categories_inside_boundary")
        else:
            category = "Undefined"
            confidence = 0.0
            type_evidence.append("classification_unresolved")
            review_reasons.append("no_unambiguous_segment_classification")

        for page in pages:
            labels = page_parts.setdefault(page, [])
            if category not in labels:
                labels.append(category)
            covered.add(page)
        segment = {
            "segment_id": str(proposal.get("segment_id") or f"boundary-{index}"),
            "category": category,
            "pages": pages,
            "start_page": pages[0],
            "end_page": pages[-1],
            "boundary_evidence": [
                str(proposal.get("source") or "boundary_detection"),
                "contiguous_physical_source_pages",
            ],
            "type_evidence": type_evidence,
            "confidence": confidence,
            "review_reasons": review_reasons,
            "reported_categories": categories,
        }
        segments.append(segment)
        if review_reasons:
            issues.append(
                {
                    "code": "boundary_classification_conflict",
                    "segment_id": segment["segment_id"],
                    "pages": pages,
                    "selected_category": category,
                    "reported_categories": categories,
                    "reasons": review_reasons,
                }
            )

    # A missing boundary is uncertainty, not permission for the classifier to
    # claim the page.  Keep physical ownership complete and visibly reviewable.
    missing = sorted(set(range(1, page_count + 1)) - covered)
    for run_index, span in enumerate(page_ranges(missing), start=1):
        pages = list(range(span["start"], span["end"] + 1))
        for page in pages:
            page_parts[page] = ["Undefined"]
        segments.append(
            {
                "segment_id": f"unresolved-{run_index}",
                "category": "Undefined",
                "pages": pages,
                "start_page": pages[0],
                "end_page": pages[-1],
                "boundary_evidence": ["boundary_gap"],
                "type_evidence": ["classification_not_authoritative_for_ownership"],
                "confidence": 0.0,
                "review_reasons": ["no_valid_boundary_assignment"],
                "reported_categories": [],
            }
        )
    if missing:
        issues.append({"code": "unresolved_boundary_pages", "pages": missing})

    segments.sort(key=lambda item: (item["start_page"], item["end_page"]))
    return page_parts, segments, issues


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


def prefixed_slot_filename(slot_filename: str, bundle_name: str | None) -> str:
    """Keep LlamaCloud slot PDFs unique across runs of the same paper book."""
    slot = _upload_filename(slot_filename)
    from .job_timing import uploaded_filename

    bundle = uploaded_filename(bundle_name)
    if not bundle:
        return slot
    stem = (
        re.sub(r"[^A-Za-z0-9._-]+", "_", PurePosixPath(bundle).stem).strip("._")
        or "filing"
    )
    slot_stem = PurePosixPath(slot).stem
    lowered = slot_stem.lower()
    prefix = stem.lower()
    if lowered == prefix or lowered.startswith(f"{prefix}-"):
        return slot
    return _upload_filename(f"{stem}-{slot_stem}.pdf")


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
    cache_bytes: bool = False,
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
    if cache_bytes:
        remember_file_bytes(file_id, data)
    logger.info(
        "[process-file] Ingested remote file url=%s → file_id=%s name=%s bytes=%s",
        safe_url_for_log(url),
        file_id,
        name,
        len(data),
    )
    return file_id, content_hash, name


async def _download_file_bytes(client: AsyncLlamaCloud, file_id: str) -> bytes:
    """Download original PDF bytes via the files content presigned URL."""
    fid = file_id[4:] if file_id.startswith("dfl-") else file_id
    presigned = await client.files.content(fid, project_id=project_id)
    url = getattr(presigned, "url", None)
    if not url:
        raise RuntimeError(f"No download URL for file {file_id}")
    timeout = httpx.Timeout(FILE_DOWNLOAD_TIMEOUT_S)
    async with httpx.AsyncClient(timeout=timeout, follow_redirects=True) as http:
        response = await http.get(str(url))
        response.raise_for_status()
        return response.content


async def _load_bundle_pdf(
    client: AsyncLlamaCloud,
    *,
    file_id: str,
    source_documents: list[SourceDocument] | None = None,
) -> bytes:
    """Prefer in-process ingest bytes, then the original S3 URL, then LlamaCloud."""
    cached = take_file_bytes(file_id)
    if cached:
        logger.info(
            "[process-file] Using cached compiled PDF file_id=%s bytes=%s",
            file_id,
            len(cached),
        )
        return cached
    for doc in source_documents or []:
        url = (doc.download_url or "").strip()
        if not url:
            continue
        try:
            timeout = httpx.Timeout(FILE_DOWNLOAD_TIMEOUT_S)
            async with httpx.AsyncClient(
                timeout=timeout, follow_redirects=True
            ) as http:
                response = await http.get(url)
                response.raise_for_status()
                data = response.content
            if data.startswith(b"%PDF"):
                logger.info(
                    "[process-file] Reloaded compiled PDF from source URL "
                    "file_id=%s bytes=%s",
                    file_id,
                    len(data),
                )
                return data
        except Exception as exc:
            logger.warning(
                "[process-file] Source URL reload failed for %s: %s",
                file_id,
                exc,
            )
    return await _download_file_bytes(client, file_id)


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


async def _upload_sliced_slots(
    client: AsyncLlamaCloud,
    slices: list[Any],
    *,
    enabled: bool,
    original_filename: str | None = None,
) -> list[str | None]:
    """Upload sliced slot PDFs in parallel. Skip when the backend re-slices."""
    if not enabled or not slices:
        return [None] * len(slices)
    semaphore = asyncio.Semaphore(slot_upload_concurrency())

    async def _one(item: Any) -> str:
        async with semaphore:
            filename = prefixed_slot_filename(item.filename, original_filename)
            return await _upload_slot_pdf(
                client,
                filename=filename,
                pdf_bytes=item.pdf_bytes,
            )

    return list(await asyncio.gather(*[_one(item) for item in slices]))


async def _verify_one_document(
    client: AsyncLlamaCloud,
    *,
    item: FilingPartIn,
    catalog: Any,
    split_config: SplitConfig,
    semaphore: asyncio.Semaphore,
) -> VerifiedDocument:
    name = item.filename or item.document_id or item.slot_id
    expected = (item.slot_id or "").strip()
    try:
        async with semaphore:
            file_id = (item.file_id or "").strip() or None
            filename = (item.filename or "").strip() or None
            if not file_id:
                file_id, _digest, filename = await ingest_remote_file(
                    client,
                    item.file_url or "",
                    filename=filename,
                    external_file_id=item.document_id or item.file_hash,
                )
            page_parts, _job_id, _exchange = await _split_page_parts(
                client,
                file_id=file_id,
                split_config=split_config,
                filename=filename or name,
            )
        matched = split_labels_match_expected_slot(
            expected_slot=expected,
            catalog=catalog,
            page_parts=page_parts,
        )
        return VerifiedDocument(name=name, match=matched)
    except Exception:
        logger.exception("Verify failed for %s", name)
        return VerifiedDocument(name=name, match=False)


async def _run_verify_from_file_event(
    event: FileEvent,
    ctx: Context[PrepareState],
    client: AsyncLlamaCloud,
    split_config: SplitConfig,
) -> BundlePrepared:
    filing_type = (event.filing_type or "").strip()
    catalog = type_catalog(filing_type)
    ctx.write_event_to_stream(
        Status(
            level="info",
            message=f"Verifying {len(event.documents)} named document(s)",
        )
    )
    semaphore = asyncio.Semaphore(slot_upload_concurrency())
    results = await asyncio.gather(
        *[
            _verify_one_document(
                client,
                item=item,
                catalog=catalog,
                split_config=split_config,
                semaphore=semaphore,
            )
            for item in event.documents
        ]
    )
    documents = list(results)
    overall = all(item.match for item in documents) if documents else False
    echo = intake_echo(event)
    ctx.write_event_to_stream(
        Status(
            level="info",
            message=(f"Verified {len(documents)} document(s); overall match={overall}"),
        )
    )
    return BundlePrepared(
        filing_type=filing_type,
        match=overall,
        verified_documents=documents,
        job_type=echo["job_type"],
        organization_id=echo["organization_id"],
        workspace_id=echo["workspace_id"],
        user_id=echo["user_id"],
        result={
            "match": overall,
            "documents": [item.model_dump() for item in documents],
        },
    )


def _nested_split_payload(result: Any) -> dict[str, Any]:
    item = getattr(result, "result", result)
    if isinstance(item, dict):
        agent_data_id = item.get("agent_data_id") or item.get("result")
        report = item.get("report")
        if hasattr(report, "model_dump"):
            report = report.model_dump(mode="json")
        return {
            "agent_data_id": str(agent_data_id) if agent_data_id else None,
            "report": report if isinstance(report, dict) else None,
        }
    return {"agent_data_id": str(item) if item else None, "report": None}


def _split_result_fields(nested: Any) -> tuple[str | None, dict[str, Any] | None]:
    if isinstance(nested, dict):
        agent_data_id = nested.get("agent_data_id") or nested.get("result")
        report = nested.get("report")
        if hasattr(report, "model_dump") and not isinstance(report, dict):
            report = report.model_dump(mode="json")
        return (
            str(agent_data_id) if agent_data_id else None,
            report if isinstance(report, dict) else None,
        )
    return (str(nested) if nested else None, None)


async def _extract_sliced_parts(
    ctx: Context[PrepareState],
    *,
    filing_type: str,
    parts: list[Any],
    echo: dict[str, str | None],
    require_all_slots: bool,
    fallback_file_id: str | None = None,
    classify_split_seconds: float | None = None,
    parse_scope: str = "all",
    skip_index: bool = False,
    skip_extract: bool = False,
    stitch_in_request_order: bool = False,
    front_extract: bool = False,
    front_opening_pages: int = FRONT_EXTRACT_PAGE_LIMIT,
    edited: bool = False,
    parsed_slots: list[str] | None = None,
    agent_data_id: str | None = None,
    special_category: str | None = None,
    court: str | None = None,
    reuse_pages_by_slot: dict[str, dict[str, str]] | None = None,
    reuse_layouts_by_slot: dict[str, dict[str, Any]] | None = None,
    reuse_parse_job_ids: dict[str, str] | None = None,
) -> dict[str, Any]:
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
            filename=echo.get("filename") if isinstance(echo, dict) else None,
            classify_split_seconds=classify_split_seconds,
            parse_scope=parse_scope,
            skip_index=skip_index,
            skip_extract=skip_extract,
            stitch_in_request_order=stitch_in_request_order,
            front_extract=front_extract,
            front_opening_pages=front_opening_pages,
            edited=edited,
            parsed_slots=list(parsed_slots or []),
            agent_data_id=agent_data_id,
            special_category=special_category,
            court=court,
            reuse_pages_by_slot=reuse_pages_by_slot or {},
            reuse_layouts_by_slot=reuse_layouts_by_slot or {},
            reuse_parse_job_ids=reuse_parse_job_ids or {},
        )
    )
    async for ev in handler.stream_events():
        ctx.write_event_to_stream(ev)
    result = await handler
    return _nested_split_payload(result)


async def _run_split_from_file_event(
    event: FileEvent,
    ctx: Context[PrepareState],
    client: AsyncLlamaCloud,
) -> BundlePrepared:
    """Ingest labeled URLs and run process-split-files inside this handler."""
    from .process_split_files import SplitPartEvent
    from .s3_artifacts import download_json_object
    from .split_upload import (
        PARSE_ARTIFACT_KEY_KEY,
        SplitPartInput,
        load_parse_artifact,
        resolve_upload_slot,
        slot_is_extract_source,
    )

    filing_type = (event.filing_type or "").strip()
    job = (event.job_type or "").strip().lower()
    prior_file_ids: dict[str, str] = {}
    prior_file_hashes: dict[str, str] = {}
    parsed_slots = [slot for slot in (event.parsed_slots or []) if slot]
    reuse_pages: dict[str, dict[str, str]] = {}
    reuse_layouts: dict[str, dict[str, Any]] = {}
    reuse_jobs: dict[str, str] = {}
    agent_data_id = (event.agent_data_id or "").strip() or None
    if agent_data_id:
        try:
            item = await client.beta.agent_data.get(agent_data_id)
        except NotFoundError as exc:
            if job in INDEX_JOB_TYPES:
                raise ValueError(f"Filing not found: {agent_data_id}") from exc
            # Re-extraction is the recovery path for a stale Agent Data id.
            # Keep the requested id on nested state so its upsert can attempt
            # the update, observe the 404, and return a replacement id.
            logger.warning(
                "Agent Data %s is missing; re-extraction will recreate it",
                agent_data_id,
            )
            item = None
        data = (
            item.data
            if item is not None and isinstance(getattr(item, "data", None), dict)
            else {}
        )
        if (
            item is not None
            and hasattr(item.data, "model_dump")
            and not isinstance(item.data, dict)
        ):
            dumped = item.data.model_dump(mode="json")
            data = dumped if isinstance(dumped, dict) else {}
        meta = data.get("metadata") if isinstance(data.get("metadata"), dict) else {}
        split_files = meta.get("split_files")
        if isinstance(split_files, dict):
            prior_file_ids = {
                str(slot): str(file_id)
                for slot, file_id in split_files.items()
                if slot and file_id
            }
        stored_hashes = meta.get("document_hashes")
        if isinstance(stored_hashes, dict):
            prior_file_hashes = {
                str(slot): str(file_hash)
                for slot, file_hash in stored_hashes.items()
                if slot and file_hash
            }
        if not prior_file_hashes:
            for document in meta.get("documents") or []:
                if not isinstance(document, dict):
                    continue
                slot = str(document.get("slot_id") or "").strip()
                file_hash = str(document.get("file_hash") or "").strip()
                if slot and file_hash:
                    prior_file_hashes[slot] = file_hash
        if not parsed_slots:
            parsed_slots = [
                str(slot).strip()
                for slot in (meta.get("parsed_slots") or [])
                if str(slot).strip()
            ]
        artifact = download_json_object(str(meta.get(PARSE_ARTIFACT_KEY_KEY) or ""))
        _slots, _order, pages_by_slot, job_ids, layouts_by_slot = load_parse_artifact(
            artifact
        )
        reuse_pages = {
            slot: {str(page): text for page, text in pages.items()}
            for slot, pages in pages_by_slot.items()
        }
        reuse_layouts = {
            slot: {str(page): payload for page, payload in layout.items()}
            for slot, layout in layouts_by_slot.items()
        }
        reuse_jobs = dict(job_ids)
        if not parsed_slots:
            parsed_slots = list(_slots)

    ctx.write_event_to_stream(
        Status(
            level="info",
            message=(
                f"job_type={job or 'upload_separate'}: ingesting "
                f"{len(event.documents)} file(s)"
            ),
        )
    )
    split_parts: list[SplitPartEvent] = []
    source_docs: list[SourceDocument] = []
    reusable_slots: set[str] = set(parsed_slots) if not event.edited else set()
    for item in event.documents:
        slot = (item.slot_id or "").strip()
        if slot.lower() in FULL_PETITION_SLOTS:
            raise ValueError(
                f"slot_id {slot!r} belongs to job_type upload_compiled, not upload_separate"
            )
        item_file_id = (item.file_id or "").strip() or None
        prior_id = prior_file_ids.get(slot) or None
        edited = bool(event.edited)
        current_hash = (item.file_hash or "").strip()
        prior_hash = prior_file_hashes.get(slot, "")
        unchanged = bool(
            edited and current_hash and prior_hash and current_hash == prior_hash
        )
        if unchanged and prior_id:
            reusable_slots.add(slot)
        # Invalidate only documents whose content changed. A matching checksum
        # proves that the prior file id and parse artifact are still current.
        file_id = (
            (item_file_id or prior_id) if (not edited or unchanged) else item_file_id
        )
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

    parsed_slots = [slot for slot in parsed_slots if slot in reusable_slots]
    reuse_pages = {
        slot: pages for slot, pages in reuse_pages.items() if slot in reusable_slots
    }
    reuse_layouts = {
        slot: layout for slot, layout in reuse_layouts.items() if slot in reusable_slots
    }
    reuse_jobs = {
        slot: parse_job_id
        for slot, parse_job_id in reuse_jobs.items()
        if slot in reusable_slots
    }

    echo = intake_echo(event)
    upload_step_json(
        STEP_SPLIT,
        with_config_identity(
            {
                "job_type": echo["job_type"],
                "filing_type": filing_type,
                "organization_id": echo["organization_id"],
                "workspace_id": echo["workspace_id"],
                "parts": [item.model_dump(mode="json") for item in split_parts],
                "slot_pages": {},
            }
        ),
        organization_id=echo["organization_id"],
        workspace_id=echo["workspace_id"],
    )
    catalog = type_catalog(filing_type)
    extract_only = job in EXTRACT_ONLY_JOB_TYPES
    index_only = job in INDEX_JOB_TYPES
    edited = bool(event.edited)
    if extract_only:
        parsed_slots = []
        for part in split_parts:
            slot = resolve_upload_slot(catalog, part.slot_id)
            names = slot.parts if slot else ()
            probe = SplitPartInput(
                slot_id=part.slot_id,
                file_id=part.file_id or "",
                document_parts=names,
            )
            if slot_is_extract_source(probe, catalog):
                parsed_slots.append(part.slot_id)
    parse_scope = "all"
    if extract_only:
        parse_scope = "extract_sources"
    elif index_only:
        parse_scope = "unparsed"
    nested_payload = await _extract_sliced_parts(
        ctx,
        filing_type=filing_type,
        parts=split_parts,
        echo=echo,
        require_all_slots=False,
        parse_scope=parse_scope,
        skip_index=extract_only,
        skip_extract=index_only,
        stitch_in_request_order=extract_only or index_only,
        edited=edited,
        parsed_slots=parsed_slots,
        agent_data_id=agent_data_id,
        special_category=event.special_category,
        court=event.court,
        reuse_pages_by_slot=reuse_pages if index_only else {},
        reuse_layouts_by_slot=reuse_layouts if index_only else {},
        reuse_parse_job_ids=reuse_jobs if index_only else {},
    )
    agent_data_id, report = _split_result_fields(nested_payload)
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
        parsed_slots=parsed_slots,
        report=report,
        **echo,
    )


async def _run_extract_front(
    event: FileEvent,
    ctx: Context[PrepareState],
    client: AsyncLlamaCloud,
) -> BundlePrepared:
    """Parse pages 1–30 of one compiled PDF, then extract. No split or index."""
    from .process_split_files import SplitPartEvent

    filing_type = (event.filing_type or "").strip() or DEFAULT_COMPILED_FILING_TYPE
    file_id, file_url, filename, file_hash = compiled_source(event)
    if not file_id and file_url:
        ctx.write_event_to_stream(
            Status(
                level="info",
                message="Downloading the compiled PDF for the first 30 pages",
            )
        )
        file_id, content_hash, filename = await ingest_remote_file(
            client,
            file_url,
            filename=filename,
            external_file_id=(
                event.documents[0].document_id if event.documents else None
            )
            or file_hash,
            cache_bytes=True,
        )
        file_hash = file_hash or content_hash
    if not file_id:
        raise ValueError(
            "Provide download_url or file_id, or documents[] with the compiled PDF"
        )

    pdf_bytes = await _load_bundle_pdf(client, file_id=file_id)
    total_pages = pdf_page_count(pdf_bytes)
    opening_pages = front_extract_opening_pages(total_pages)
    sliced = first_pages_pdf(pdf_bytes, opening_pages)
    slice_name = filename or "compiled.pdf"
    ctx.write_event_to_stream(
        Status(
            level="info",
            message=(
                f"extract_front: parsing the first {opening_pages} pages "
                f"of {total_pages} and the last {FRONT_EXTRACT_END_PAGE_LIMIT} "
                "pages. Split, classify, and Pinecone are skipped."
            ),
        )
    )
    slice_id = await _upload_slot_pdf(
        client,
        filename=(
            f"front-{opening_pages}-end-{FRONT_EXTRACT_END_PAGE_LIMIT}-{slice_name}"
        ),
        pdf_bytes=sliced,
    )
    source = event.documents[0]
    part = SplitPartEvent(
        slot_id="petition",
        file_id=slice_id,
        filename=slice_name,
        file_hash=hashlib.sha256(sliced).hexdigest(),
        file_url=file_url,
        document_id=source.document_id,
    )
    echo = intake_echo(event)
    echo["job_type"] = "extract_front"
    nested_payload = await _extract_sliced_parts(
        ctx,
        filing_type=filing_type,
        parts=[part],
        echo=echo,
        require_all_slots=False,
        parse_scope="all",
        skip_index=True,
        stitch_in_request_order=True,
        front_extract=True,
        front_opening_pages=opening_pages,
        special_category=None,
        court=None,
    )
    agent_data_id, report = _split_result_fields(nested_payload)
    prepared = PreparedPart(
        slot_id="fullpetition",
        file_id=part.file_id,
        file_hash=part.file_hash,
        filename=source.filename or slice_name,
        document_id=source.document_id,
        download_url=source.file_url,
        name=source.filename or slice_name,
    )
    source_docs = [
        source_document_from_part(
            item,
            file_id=item.file_id,
            file_hash=item.file_hash,
        )
        for item in event.documents
    ]
    return BundlePrepared(
        result=agent_data_id,
        filing_type=filing_type,
        parts=[prepared],
        documents=source_docs,
        agent_data_id=agent_data_id,
        parsed_slots=["fullpetition"],
        report=report,
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
        split_config: Annotated[
            SplitConfig,
            ResourceConfig(
                config_file="configs/config.json",
                path_selector="split",
                label="Document Parts",
                description="LlamaSplit categories for petition bundle parts",
            ),
        ],
    ) -> FileClassifiedEvent | BundlePrepared:
        mode = intake_mode(event)
        if mode == "extract_front":
            return await _run_extract_front(event, ctx, llama_cloud_client)
        if mode == "split":
            return await _run_split_from_file_event(event, ctx, llama_cloud_client)
        if mode == "verify":
            return await _run_verify_from_file_event(
                event, ctx, llama_cloud_client, split_config
            )

        file_id, file_url, filename, file_hash_hint = compiled_source(event)
        content_hash: str | None = None
        started_at = start_timer()

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
                    cache_bytes=True,
                )
            except Exception as exc:
                logger.exception(
                    "Failed to ingest file_url %s", safe_url_for_log(file_url)
                )
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
            filename = uploaded_filename(filename) or uploaded_filename(
                getattr(file_metadata, "name", None)
            )
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
            state.started_at = started_at

        job = (event.job_type or "").strip().lower()
        # Compiled bundles must be classified from the PDF. A caller-supplied
        # filing_type is only a fallback when classify cannot be sliced.  The
        # split_petition endpoint is the exception: it is used after the user
        # has selected a petition type and intentionally avoids another
        # classification call.
        override = (
            None
            if job in CLASSIFY_COMPILED_JOB_TYPES
            else compiled_catalog_override(event.filing_type)
        )
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
                configuration=dump_api_configuration(classify_config),
                project_id=project_id,
            )

        completed = await _wait_for_classify(llama_cloud_client, classify_job.id)
        classify_usage = await collect_optional_usage(
            llama_cloud_client,
            product="classify",
            job_id=str(classify_job.id),
            payload=completed,
        )
        async with ctx.store.edit_state() as state:
            state.llamacloud_jobs = [
                *(state.llamacloud_jobs or []),
                classify_usage,
            ]
        if completed.status == "FAILED" or completed.result is None:
            message = f"Classification did not resolve for {filename}"
            ctx.write_event_to_stream(Status(level="error", message=message))
            raise RuntimeError(message)

        result = completed.result
        classified = (result.type or "").strip()
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
        # Snapshot once so all stages of this job use the same mode.
        split_mode = split_reconciliation_mode()
        output_selection = split_output_selection()
        use_python_rules = split_mode != "llama_only"
        logger.info("[SplitMode] mode=%s file=%s", split_mode, state.filename)

        async def run_llama_split(
            *, api_configuration: dict[str, Any] | None = None
        ) -> tuple[dict[int, list[str]], str, dict[str, Any]]:
            return await _split_page_parts(
                llama_cloud_client,
                file_id=state.file_id,
                split_config=split_config,
                filename=state.filename,
                api_configuration=api_configuration,
            )

        async def load_and_prepare_pdf() -> tuple[bytes, list[PageUnit], float]:
            pdf_bytes = await _load_bundle_pdf(
                llama_cloud_client,
                file_id=state.file_id,
                source_documents=state.source_documents,
            )
            # Parsing and OCR are mandatory for every accepted result, including
            # the explicit legacy mode. That mode disables local repair only.
            from .structure_split import extract_page_units

            local_started = start_timer()
            units = await asyncio.to_thread(
                extract_page_units,
                pdf_bytes,
                source_pdf=state.filename,
            )
            return pdf_bytes, units, elapsed_seconds(local_started) or 0.0

        # Establish the immutable physical-page map before either model pass.
        pdf_bytes, page_units, local_read_seconds = await load_and_prepare_pdf()
        with pymupdf.open(stream=pdf_bytes, filetype="pdf") as pdf:
            pdf_page_count = pdf.page_count
        boundary_started = start_timer()
        boundary_config = _boundary_detection_configuration(
            page_units, page_count=pdf_page_count
        )
        _boundary_pages, boundary_job_id, boundary_exchange = await run_llama_split(
            api_configuration=boundary_config
        )
        boundary_validation = _boundary_proposals_from_exchange(
            boundary_exchange, page_count=pdf_page_count
        )
        boundary_proposals = boundary_validation["proposals"]
        boundary_seconds = elapsed_seconds(boundary_started) or 0.0
        # Classification is a distinct second SplitV1 pass with the canonical
        # configured categories/rules; boundary labels are never reused as types.
        classification_config, boundary_guidance_status = _classification_configuration(
            split_config,
            boundary_proposals,
            proposal_issues=boundary_validation["issues"],
        )
        classification_started = start_timer()
        split_result = await run_llama_split(api_configuration=classification_config)
        page_parts, split_job_id, llama_split = split_result
        raw_classification_page_parts = page_parts
        classification_report = _segment_classification_report(
            llama_split, page_count=pdf_page_count
        )
        raw_classified_ranges = {
            tuple(item["source_pages"]) for item in classification_report["segments"]
        }
        proposed_ranges = {tuple(item["pages"]) for item in boundary_proposals}
        if proposed_ranges != raw_classified_ranges:
            classification_report["issues"].append(
                "classification_ranges_differ_from_boundary_proposals"
            )
            classification_report["status"] = "needs_review"
        (
            llama_page_parts,
            authoritative_boundary_segments,
            ownership_reconciliation_issues,
        ) = _document_first_page_ownership(
            boundary_proposals,
            classification_report,
            page_count=pdf_page_count,
        )
        # From this point onward, page ownership comes only from the boundary
        # pass. The classification response supplies document types, never new
        # page ranges. Keep the raw map in the audit for diagnosis.
        page_parts = llama_page_parts
        classification_report["boundary_segments"] = authoritative_boundary_segments
        classification_report["ownership_reconciliation"] = {
            "policy": "boundary_ranges_authoritative",
            "issues": ownership_reconciliation_issues,
            "raw_classification_page_parts": raw_classification_page_parts,
        }
        if ownership_reconciliation_issues:
            classification_report["status"] = "needs_review"
            classification_report["issues"].extend(
                f"ownership:{item['code']}" for item in ownership_reconciliation_issues
            )
        classification_seconds = elapsed_seconds(classification_started) or 0.0
        logger.info(
            "[SplitTiming] mode=%s boundary=%.1fs classification=%.1fs "
            "local_page_read=%.1fs pages=%s",
            split_mode,
            boundary_seconds,
            classification_seconds,
            local_read_seconds,
            pdf_page_count,
        )
        boundary_usage = await collect_optional_usage(
            llama_cloud_client, product="split", job_id=boundary_job_id
        )
        split_usage = await collect_optional_usage(
            llama_cloud_client,
            product="split",
            job_id=split_job_id,
        )
        usage_jobs = [*(state.llamacloud_jobs or []), boundary_usage, split_usage]
        usage_summary = summarize_llamacloud_usage(usage_jobs)
        logger.info("[Usage] %s", usage_status_message(usage_summary))

        ctx.write_event_to_stream(
            Status(
                level="info",
                message=f"Loading {state.filename} to slice labeled pages",
            )
        )
        # Structural ownership and page-level extraction are mandatory for
        # every output. llama_only preserves legacy labels and skips repair,
        # but its output remains explicitly unvalidated semantically.
        split_duplicates: list[dict[str, Any]] = []
        split_audit: dict[str, Any] = {
            "index_status": "not_checked",
            "reconciliation_mode": split_mode,
            "index_rows": [],
            "document_spans": [],
            "flags": [],
            "flag_counts": {"error": 0, "warning": 0, "total": 0},
            "llama_boundary_detection": {
                "job_id": boundary_job_id,
                "proposals": boundary_proposals,
                "seconds": round(boundary_seconds, 3),
                "evidence_page_count": len(page_units),
            },
        }
        from .split_audit import audit_compiled_split
        from .structure_split import structure_aware_split

        ctx.write_event_to_stream(
            Status(
                level="info",
                message=(
                    "Applying split structure rules and audit"
                    if use_python_rules
                    else "Running mandatory page and ownership audit in legacy mode"
                ),
            )
        )
        rules_started = start_timer()
        structured = await asyncio.to_thread(
            structure_aware_split,
            pdf_bytes,
            llama_page_parts=page_parts,
            page_units=page_units,
            source_pdf=state.filename,
            run_hybrid_repair=use_python_rules,
            reconciliation_mode=split_mode,
            authoritative_boundary_pages=[
                item["start_page"] for item in authoritative_boundary_segments[1:]
            ],
        )
        page_parts = structured.page_parts
        reconciliation_changes = getattr(structured, "reconciliation_changes", [])
        logger.info(
            "[SplitReconciliation] mode=%s committed_pages=%s",
            split_mode,
            len(reconciliation_changes),
        )
        split_duplicates = [
            hit.as_dict() if hasattr(hit, "as_dict") else hit
            for hit in structured.duplicates
        ]
        split_audit = await asyncio.to_thread(
            audit_compiled_split,
            page_parts,
            {unit.pdf_page: unit.text for unit in structured.page_units},
            page_count=pdf_page_count,
            llama_page_parts=llama_page_parts,
            classifications=getattr(structured, "classifications", ()),
            segment_classifications=getattr(structured, "segment_classifications", ()),
            llama_boundary_proposals=getattr(
                structured, "llama_boundary_proposals", ()
            ),
            reconciliation_changes=reconciliation_changes,
            reconciliation_operations=getattr(
                structured, "reconciliation_operations", ()
            ),
            initial_segments=authoritative_boundary_segments,
            raw_classification_segments=classification_report["segments"],
            assignment_events=getattr(structured, "assignment_events", {}),
            boundaries=getattr(structured, "boundaries", ()),
            page_units=structured.page_units,
            source_document_id=state.file_id,
            source_pdf_sha256=hashlib.sha256(pdf_bytes).hexdigest(),
        )
        split_audit["llama_boundary_detection"] = {
            "job_id": boundary_job_id,
            "proposals": boundary_proposals,
            "validation": boundary_validation,
            "seconds": round(boundary_seconds, 3),
            "evidence_page_count": len(page_units),
            "evidence_complete": "Page cues are sampled"
            not in boundary_config["splitting_strategy"]["custom_instructions"],
        }
        split_audit["llama_segment_classification"] = {
            "job_id": split_job_id,
            "boundary_guidance_status": boundary_guidance_status,
            "configuration_used": _classification_call_evidence(
                llama_split, boundary_guidance_status=boundary_guidance_status
            ),
            **classification_report,
        }
        if (
            split_audit["llama_segment_classification"]["status"] != "clear"
            or boundary_guidance_status != "sent"
            or boundary_validation["status"] != "clear"
        ):
            split_audit.setdefault("flags", []).append(
                {
                    "code": "llama_segment_classification_incomplete",
                    "severity": "warning",
                    "message": "The complete-segment classification could not be fully validated or was not supplied all candidate boundaries.",
                    "details": [
                        *split_audit["llama_segment_classification"]["issues"],
                        *[
                            f"boundary:{issue['code']}"
                            for issue in boundary_validation["issues"]
                        ],
                        *(
                            []
                            if boundary_guidance_status == "sent"
                            else [boundary_guidance_status]
                        ),
                    ],
                }
            )
            split_audit["review_status"] = "needs_review"
            split_audit.setdefault("flag_counts", {})["warning"] = (
                split_audit.get("flag_counts", {}).get("warning", 0) + 1
            )
            split_audit["flag_counts"]["total"] = (
                split_audit.get("flag_counts", {}).get("total", 0) + 1
            )
        split_audit["structure"] = structured.report()
        split_audit["reconciliation"] = {
            "summary": getattr(structured, "reconciliation_summary", {}),
            "operations": [
                item.as_dict() if hasattr(item, "as_dict") else item
                for item in getattr(structured, "reconciliation_operations", ())
            ],
        }
        unresolved_reconciliation = [
            item
            for item in getattr(structured, "reconciliation_operations", ())
            if getattr(item, "status", None)
            in {"left_for_review", "rolled_back", "rejected"}
            and getattr(item, "operation_type", None) != "reconciliation_stop"
        ]
        if unresolved_reconciliation:
            split_audit.setdefault("flags", []).append(
                {
                    "code": "reconciliation_proposal_needs_review",
                    "severity": "warning",
                    "message": "One or more proposed page-map corrections were unresolved or rolled back; review the reconciliation audit before accepting the split.",
                    "details": [
                        {
                            "operation_id": item.operation_id,
                            "pass_index": item.pass_index,
                            "status": item.status,
                            "reason": item.reason,
                            "pages": format_page_span(item.proposed_pages),
                        }
                        for item in unresolved_reconciliation
                    ],
                }
            )
            split_audit["review_status"] = "needs_review"
            split_audit.setdefault("flag_counts", {})["warning"] = (
                split_audit.get("flag_counts", {}).get("warning", 0) + 1
            )
            split_audit["flag_counts"]["total"] = (
                split_audit.get("flag_counts", {}).get("total", 0) + 1
            )
        split_audit["reconciliation_mode"] = split_mode
        if not use_python_rules:
            split_audit["semantic_validation"] = {
                "status": "not_assessed_legacy_mode",
                "reason": "llama_only compatibility mode disables local reconciliation",
            }
        logger.info(
            "[SplitTiming] mode=%s repair_and_audit=%.1fs",
            split_mode,
            elapsed_seconds(rules_started) or 0.0,
        )
        ctx.write_event_to_stream(
            Status(
                level="info",
                message=(
                    f"Structure-aware split found {len(structured.logical_documents)} "
                    "logical document(s) after local processing"
                ),
            )
        )
        audit_total = int((split_audit.get("flag_counts") or {}).get("total") or 0)
        if audit_total:
            ctx.write_event_to_stream(
                Status(
                    level="info", message=f"Split audit raised {audit_total} flag(s)"
                )
            )
        index_status = split_audit.get("index_status")
        if index_status in {"missing", "unreadable"}:
            ctx.write_event_to_stream(
                Status(
                    level="warning",
                    message=(
                        "No usable Index was available; boundaries were assessed from "
                        "page headings, folios, and document cues. Review audit flags."
                    ),
                )
            )
        boundary_flags = [
            flag
            for flag in split_audit.get("flags", [])
            if flag.get("code") in {"index_page_mismatch", "index_end_page_mismatch"}
        ]
        if boundary_flags:
            examples = []
            for flag in boundary_flags[:3]:
                part = flag.get("part") or "document"
                expected = flag.get("expected_pages") or {}
                found = flag.get("found_pages") or {}
                examples.append(
                    f"{part}: Index {expected.get('start_page', '?')}–"
                    f"{expected.get('end_page', '?')}, split "
                    f"{found.get('start_page', '?')}–{found.get('end_page', '?')}"
                )
            ctx.write_event_to_stream(
                Status(
                    level="warning",
                    message="Index/split boundary mismatch; verify: "
                    + "; ".join(examples),
                )
            )
        parts_found = sorted(
            {name for names in page_parts.values() for name in parts_on_page(names)}
        )
        labeled_pages = len(page_parts)
        ctx.write_event_to_stream(
            Status(
                level="info",
                message=(
                    f"Classified {len(parts_found)} document part(s) across "
                    f"{labeled_pages}/{pdf_page_count or labeled_pages} page(s)"
                ),
            )
        )
        # Keep remote document identities in every mode. The slicer intersects
        # these boundaries with final labels, so local repairs stay authoritative.
        split_segments = classification_report["boundary_segments"]
        sharing_validation = validate_shared_page_evidence(
            page_parts,
            split_segments,
            page_count=pdf_page_count,
        )
        classified_slices = slice_bundle_pdf(
            pdf_bytes,
            catalog,
            page_parts,
            split_segments=split_segments,
        )
        ownership = validate_sliced_page_ownership(
            classified_slices,
            page_count=pdf_page_count,
            page_parts=page_parts,
            split_segments=split_segments,
            catalog=catalog,
        )
        slices, output_filter_report = filter_split_outputs(
            classified_slices, output_selection
        )
        selected_output_slots = {item.slot_id for item in slices}
        classified_output_slots = {item.slot_id for item in classified_slices}
        output_filter_report["environment_variable"] = "SPLIT_OUTPUT_DOCUMENTS"
        output_filter_report["selected_slots"] = sorted(selected_output_slots)
        output_filter_report["excluded_slots"] = sorted(
            classified_output_slots - selected_output_slots
        )
        split_audit["output_filter"] = {
            **output_filter_report,
        }
        if output_filter_report["unmatched"]:
            split_audit.setdefault("flags", []).append(
                {
                    "code": "split_output_filter_unmatched",
                    "severity": "warning",
                    "message": "One or more configured output document names were not found in this split.",
                    "details": output_filter_report["unmatched"],
                }
            )
            split_audit["review_status"] = "needs_review"
            counts = split_audit.setdefault("flag_counts", {})
            counts["warning"] = counts.get("warning", 0) + 1
            counts["total"] = counts.get("total", 0) + 1
        split_audit["shared_page_validation"] = sharing_validation
        if sharing_validation["issues"]:
            split_audit.setdefault("flags", []).append(
                {
                    "code": "shared_page_authorization_needs_review",
                    "severity": "warning",
                    "message": "Some page sharing lacked sufficient classification evidence; the page remains single-owner.",
                    "details": sharing_validation["issues"],
                }
            )
            split_audit["review_status"] = "needs_review"
            split_audit.setdefault("flag_counts", {})["warning"] = (
                split_audit.get("flag_counts", {}).get("warning", 0) + 1
            )
            split_audit["flag_counts"]["total"] = (
                split_audit.get("flag_counts", {}).get("total", 0) + 1
            )
        owners_by_page: dict[int, list[str]] = {}
        for sliced in classified_slices:
            for source_page in sliced.pages:
                owners_by_page.setdefault(source_page, []).append(sliced.slot_id)
        selected_owners_by_page: dict[int, list[str]] = {}
        for sliced in slices:
            for source_page in sliced.pages:
                selected_owners_by_page.setdefault(source_page, []).append(
                    sliced.slot_id
                )

        page_manifest = (split_audit.get("page_manifest") or {}).get("pages")
        if not isinstance(page_manifest, list):
            page_manifest = [
                {
                    "source_page": page,
                    "source_document_id": state.file_id,
                    "assigned_labels": parts_on_page(page_parts.get(page)),
                    "effective_owner": (
                        parts_on_page(page_parts.get(page)) or ["Unidentified"]
                    )[0],
                    "segment_id": None,
                    "extraction": None,
                    "boundary_before": None,
                    "classification": None,
                    "reconciliation": None,
                    "review_status": "needs_review",
                    "review_reasons": [
                        "page_manifest_fallback_used",
                        *(
                            ["unassigned_source_page"]
                            if not parts_on_page(page_parts.get(page))
                            else []
                        ),
                    ],
                    "assignment_trace": [
                        {
                            "stage": "llama_boundary_ownership",
                            "status": "unavailable",
                            "segment_id": None,
                            "proposed_type": None,
                            "reason": "page_manifest_fallback_used",
                            "evidence": [],
                        }
                    ],
                }
                for page in range(1, pdf_page_count + 1)
            ]
            # The fallback path has no local extraction evidence, but still
            # records one manifest row per immutable source page.
            for row in page_manifest:
                row["review_reasons"] = list(row["review_reasons"])

        page_review_reasons = _split_page_review_reasons(
            boundary_validation, classification_report, sharing_validation
        )
        for row in page_manifest:
            page = int(row["source_page"])
            reasons_for_page = page_review_reasons.get(page, [])
            if reasons_for_page:
                reasons = row.setdefault("review_reasons", [])
                reasons.extend(
                    reason for reason in reasons_for_page if reason not in reasons
                )
                row["review_status"] = "needs_review"
            classified_slots = sorted(set(owners_by_page.get(page, [])))
            slots = sorted(set(selected_owners_by_page.get(page, [])))
            row["classified_slots"] = classified_slots
            row["output_slots"] = slots
            row["output_disposition"] = (
                "included"
                if slots
                else "excluded_by_split_output_filter"
                if classified_slots and output_selection["mode"] != "all"
                else "unassigned"
            )
            if not slots and classified_slots and output_selection["mode"] != "all":
                reasons = row.setdefault("review_reasons", [])
                if "excluded_by_split_output_filter" not in reasons:
                    reasons.append("excluded_by_split_output_filter")
            if len(slots) > 1:
                reasons = row.setdefault("review_reasons", [])
                if "shared_source_page" not in reasons:
                    reasons.append("shared_source_page")
                row["review_status"] = "needs_review"
                row["shared_page_evidence"] = next(
                    (
                        item
                        for item in ownership["shared_ownership"]
                        if item["source_page"] == page
                    ),
                    None,
                )
        attach_split_output_trace(page_manifest, slices)
        manifest_consistency = validate_page_manifest_output(
            page_manifest,
            slices,
            page_count=pdf_page_count,
            source_pdf=pdf_bytes,
        )
        if not manifest_consistency["valid"]:
            raise ValueError(
                "Split page manifest does not match final output slices: "
                + ", ".join(
                    str(item["code"]) for item in manifest_consistency["errors"]
                )
            )
        ownership["review_status"] = (
            "needs_review"
            if ownership["overlapping_ownership"]
            or ownership["shared_ownership"]
            or split_audit.get("review_status") == "needs_review"
            else "clear"
            if split_audit.get("reconciliation_mode") != "llama_only"
            else "needs_review"
        )
        split_audit["structural_validation"] = ownership
        split_audit["manifest_consistency"] = manifest_consistency
        split_audit["page_manifest"] = {
            "schema_version": "1.0",
            "ownership_policy": "single_owner_with_explicit_shared_copies",
            "output_filter": split_audit["output_filter"],
            "shared_source_pages_require_review": True,
            "source_document_id": state.file_id,
            "source_sha256": hashlib.sha256(pdf_bytes).hexdigest(),
            "source_page_count": pdf_page_count,
            "pages": page_manifest,
        }
        split_audit["review_status"] = ownership["review_status"]
        ctx.write_event_to_stream(
            Status(
                level="info",
                message=(
                    f"Sliced {len(slices)} selected document file(s) from {state.filename}"
                ),
            )
        )

        upload_slots = upload_sliced_slot_pdfs()
        file_ids = await _upload_sliced_slots(
            llama_cloud_client,
            slices,
            enabled=upload_slots,
            original_filename=state.filename,
        )
        prepared: list[PreparedPart] = []
        slot_pages: dict[str, list[dict[str, int]]] = {}
        for item, file_id in zip(slices, file_ids, strict=True):
            filename = (
                prefixed_slot_filename(item.filename, state.filename)
                if upload_slots
                else item.filename
            )
            unidentified_reason = None
            if item.slot_id == "undefined" or item.slot_id.startswith("undefined_"):
                reasons = split_audit.get("unidentified_reasons") or []
                item_pages = set(item.pages)
                matching = [
                    row.get("reason")
                    for row in reasons
                    if isinstance(row, dict)
                    and isinstance(row.get("page_span"), dict)
                    and item_pages.intersection(
                        range(
                            int(row["page_span"].get("start", 0)),
                            int(row["page_span"].get("end", -1)) + 1,
                        )
                    )
                    and row.get("reason")
                ]
                unidentified_reason = " ".join(dict.fromkeys(matching)) or (
                    "These pages did not match a document section with enough confidence "
                    "to assign them safely."
                )
            prepared.append(
                PreparedPart(
                    slot_id=item.slot_id,
                    file_id=file_id,
                    file_hash=item.file_hash,
                    filename=filename,
                    label=item.label,
                    page_span=list(item.page_span),
                    duplicate=item.duplicate,
                    reason=unidentified_reason,
                )
            )
            if item.page_span:
                slot_pages[item.slot_id] = list(item.page_span)
            readable_span = format_page_span(list(item.pages))
            ctx.write_event_to_stream(
                Status(
                    level="info",
                    message=f"Ready {item.label} ({readable_span or 'pages unknown'})",
                )
            )

        has_duplicates = any(part.duplicate for part in prepared)
        classify_split_seconds = elapsed_seconds(state.started_at)
        file_name = uploaded_filename(state.filename)
        timing = timing_payload(
            file_name=file_name,
            classify_split_seconds=classify_split_seconds,
        )
        upload_step_json(
            STEP_SPLIT,
            attach_timing(
                with_config_identity(
                    {
                        "job_type": state.job_type,
                        "filing_type": catalog.filing_type,
                        "filename": state.filename,
                        "file_id": state.file_id,
                        "organization_id": state.organization_id,
                        "workspace_id": state.workspace_id,
                        "usage": {"llamacloud": usage_summary},
                        "has_duplicates": has_duplicates,
                        "parts": [
                            {
                                "slot_id": part.slot_id,
                                "label": part.label,
                                "filename": part.filename,
                                "page_span": part.page_span,
                                "duplicate": part.duplicate,
                                "reason": part.reason,
                                "file_hash": part.file_hash,
                                "file_id": part.file_id,
                            }
                            for part in prepared
                        ],
                        "slot_pages": slot_pages,
                        "duplicate_parts": split_duplicates,
                        "split_audit": split_audit,
                        "split_review": {
                            "status": ownership["review_status"],
                            "error_count": int(
                                (split_audit.get("flag_counts") or {}).get("error") or 0
                            ),
                            "warning_count": int(
                                (split_audit.get("flag_counts") or {}).get("warning")
                                or 0
                            ),
                            "overlap_count": len(ownership["overlapping_ownership"]),
                            "source_page_count": pdf_page_count,
                        },
                    }
                ),
                timing,
            ),
            organization_id=state.organization_id,
            workspace_id=state.workspace_id,
        )
        if upload_slots:
            ready_message = (
                "Split ready. Review the document slots, then Submit to "
                f"parse and extract ({len(slices)} document part(s))"
            )
        else:
            ready_message = f"Split JSON ready ({len(slices)} document part(s))"
        ctx.write_event_to_stream(
            Status(
                level="info",
                message=ready_message,
            )
        )
        ctx.write_event_to_stream(
            Status(
                level="info",
                message=timing_status_message(
                    file_name=file_name,
                    classify_split_seconds=classify_split_seconds,
                ),
            )
        )
        return BundlePrepared(
            filing_type=catalog.filing_type,
            parts=prepared,
            has_duplicates=has_duplicates,
            documents=list(state.source_documents),
            slot_pages=slot_pages,
            agent_data_id=None,
            job_type=state.job_type,
            organization_id=state.organization_id,
            workspace_id=state.workspace_id,
            user_id=state.user_id,
            org_id=state.org_id,
            filename=file_name,
            classify_split_seconds=classify_split_seconds,
            timing=timing,
            usage={"llamacloud": usage_summary},
            llama_split=llama_split,
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
