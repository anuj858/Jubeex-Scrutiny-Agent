"""Catalog, validation, and extract-pack helpers for already-split uploads.

LlamaSplit is not used. `document_part` comes from labeled upload slots in
`configs/config.json` `split_upload.types`.
"""

from __future__ import annotations

import copy
import hashlib
import json
import re
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass, field
from functools import lru_cache
from pathlib import Path
from typing import Any

from .document_parts import MAIN_PETITION_PART, parts_on_page

_CONFIG_PATH = Path(__file__).resolve().parents[2] / "configs" / "config.json"

# Never sent to LlamaExtract. Scrutiny still parses these slots.
EXTRACT_PACK_EXCLUDED_PARTS = frozenset(
    {
        "Annexures",
        "Appendix",
        "Application",
        "Index",
        "Synopsis",
        "List of Dates & Events",
        "Filing Memo",
        "Record of Proceedings",
        "Court Fees",
        "Office Report on Limitation",
    }
)


def _excluded_from_extract_pack(name: str) -> bool:
    folded = (name or "").strip().lower()
    if (name or "").strip() in EXTRACT_PACK_EXCLUDED_PARTS:
        return True
    return folded.startswith("annexure") or folded.startswith("application")


# Long bodies stay out of Extract. Cover Page has one name per side.
# The Main Petition party list can run 3-4 pages or more; keep those
# opening pages plus the closing prayer pages.
PETITION_PACK_FIRST_PAGES = 8
PETITION_PACK_LAST_PAGES = 3
IMPUGNED_PACK_FIRST_PAGES = 2
IMPUGNED_PACK_LAST_PAGES = 2
VAKALATNAMA_PACK_FIRST_PAGES = 1
VAKALATNAMA_PACK_LAST_PAGES = 2
EXTRACT_PACK_PAGE_WINDOWS: dict[str, tuple[int, int]] = {
    MAIN_PETITION_PART: (PETITION_PACK_FIRST_PAGES, PETITION_PACK_LAST_PAGES),
    "Impugned Order": (IMPUGNED_PACK_FIRST_PAGES, IMPUGNED_PACK_LAST_PAGES),
    "Vakalatnama": (VAKALATNAMA_PACK_FIRST_PAGES, VAKALATNAMA_PACK_LAST_PAGES),
}
LOOK_ONLY_SUFFIX = " Ignore other document parts."
AOR_EXTRACT_RULES = (
    "Copy the printed AOR footer only. Leave a field null if it is not printed; "
    "do not invent name, code, email, mobile, firm, chamber, or PIN. "
    "Do not copy petitioner or respondent names or addresses. "
    "Prefer Vakalatnama. If Vakalatnama is not in this pack or prints no AOR, "
    "use the Memo of Appearance footer, then the last page of the Main Petition "
    "(very end: Drawn By, Filed on, DRAWN & FILED BY, Advocate for "
    "Petitioner/Respondent, Chamber lines under the name). "
    "Do not use Main Petition opening or party-list pages. "
    "If office_address is still blank, last page of Listing Proforma / "
    "Proforma for First Listing. Remaining blanks: AOR's Certificate signature "
    "/ DRAWN & FILED BY and Advocate's Checklist. Never override Vakalatnama."
)
PETITION_SLOT_ID = "petition"
UNDEFINED_SLOT_ID = "undefined"
_ANNEXURE_SLOT_RE = re.compile(r"^annexure_([a-z])(\d{1,3})$")
_APPLICATION_SLOT_RE = re.compile(r"^application_(\d{1,3})$")
_PARSE_STUB_PREFIX = "(No parse text for"
PARTY_FIELDS = frozenset({"petitioners", "respondents"})


class SplitUploadError(ValueError):
    """Invalid filing type or slot mapping for a split upload."""


@dataclass(frozen=True)
class UploadSlot:
    id: str
    label: str
    parts: tuple[str, ...]
    required: bool = False


def dynamic_upload_slot(slot_id: str) -> UploadSlot | None:
    """Annexure X-n / Application n slots are created when those files appear."""
    key = (slot_id or "").strip()
    match = _ANNEXURE_SLOT_RE.fullmatch(key)
    if match:
        series = match.group(1).upper()
        number = int(match.group(2))
        return UploadSlot(
            id=key,
            label=f"Annexure {series}-{number}",
            parts=(f"Annexure {series}-{number}",),
            required=False,
        )
    match = _APPLICATION_SLOT_RE.fullmatch(key)
    if match:
        number = int(match.group(1))
        return UploadSlot(
            id=key,
            label=f"Application {number}",
            parts=(f"Application {number}",),
            required=False,
        )
    return None


def resolve_upload_slot(catalog: UploadTypeCatalog, slot_id: str) -> UploadSlot | None:
    return catalog.slot_by_id().get(slot_id) or dynamic_upload_slot(slot_id)


def _numbered_slot_sort_key(slot_id: str) -> tuple[int, int, int]:
    match = _ANNEXURE_SLOT_RE.fullmatch(slot_id)
    if match:
        series = match.group(1)
        # Keep P first, then R, then E, then other letters.
        series_rank = {"p": 0, "r": 1, "e": 2}.get(series, 3)
        return (0, series_rank, int(match.group(2)))
    match = _APPLICATION_SLOT_RE.fullmatch(slot_id)
    if match:
        return (1, 0, int(match.group(1)))
    return (2, 0, 0)


@dataclass(frozen=True)
class FieldSources:
    """Where one extract field is filled from, and where spelling is checked."""

    fill: tuple[str, ...] = ()
    verify: tuple[str, ...] = ()

    def all_parts(self) -> tuple[str, ...]:
        names: list[str] = list(self.fill)
        for name in self.verify:
            if name not in names:
                names.append(name)
        return tuple(names)


@dataclass(frozen=True)
class UploadTypeCatalog:
    filing_type: str
    label: str
    slots: tuple[UploadSlot, ...]
    extract_field_sources: dict[str, FieldSources] = field(default_factory=dict)

    def slot_by_id(self) -> dict[str, UploadSlot]:
        return {slot.id: slot for slot in self.slots}


@dataclass(frozen=True)
class SplitPartInput:
    slot_id: str
    file_id: str
    document_parts: tuple[str, ...] = ()
    file_hash: str | None = None
    filename: str | None = None
    document_id: str | None = None


@lru_cache(maxsize=1)
def _read_config() -> dict[str, Any]:
    with _CONFIG_PATH.open(encoding="utf-8") as fh:
        payload = json.load(fh)
    if not isinstance(payload, dict):
        return {}
    return payload


def load_split_upload_config(
    payload: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    data = payload if payload is not None else _read_config()
    block = data.get("split_upload") if isinstance(data, Mapping) else None
    return dict(block) if isinstance(block, Mapping) else {}


def _as_part_names(value: Any) -> tuple[str, ...]:
    names = parts_on_page(value)
    return tuple(names)


def _parse_slot(raw: Mapping[str, Any]) -> UploadSlot | None:
    slot_id = str(raw.get("id") or "").strip()
    label = str(raw.get("label") or slot_id).strip()
    parts = _as_part_names(raw.get("parts") or label)
    if not slot_id or not parts:
        return None
    return UploadSlot(
        id=slot_id,
        label=label or slot_id,
        parts=parts,
        required=bool(raw.get("required", False)),
    )


def _parse_field_sources(raw: Any) -> FieldSources | None:
    if isinstance(raw, Mapping):
        fill = _as_part_names(raw.get("fill") or raw.get("parts") or [])
        verify = _as_part_names(raw.get("verify") or [])
        if fill or verify:
            return FieldSources(fill=fill, verify=verify)
        return None
    names = _as_part_names(raw)
    if not names:
        return None
    return FieldSources(fill=names)


def _parse_sources(raw: Any) -> dict[str, FieldSources]:
    if not isinstance(raw, Mapping):
        return {}
    sources: dict[str, FieldSources] = {}
    for field_name, spec in raw.items():
        key = str(field_name).strip()
        parsed = _parse_field_sources(spec)
        if key and parsed is not None:
            sources[key] = parsed
    return sources


def type_catalog(
    filing_type: str,
    payload: Mapping[str, Any] | None = None,
) -> UploadTypeCatalog:
    block = load_split_upload_config(payload)
    types = block.get("types") if isinstance(block.get("types"), Mapping) else {}
    key = str(filing_type or "").strip()
    raw = types.get(key) if isinstance(types, Mapping) else None
    if not isinstance(raw, Mapping):
        raise SplitUploadError(f"Unknown filing type: {key or '(empty)'}")
    slots: list[UploadSlot] = []
    seen: set[str] = set()
    for item in raw.get("slots") or []:
        if not isinstance(item, Mapping):
            continue
        slot = _parse_slot(item)
        if slot is None or slot.id in seen:
            continue
        seen.add(slot.id)
        slots.append(slot)
    if not slots:
        raise SplitUploadError(f"No upload slots configured for {key}")
    shared = _parse_sources(block.get("extract_field_sources"))
    override = _parse_sources(raw.get("extract_field_sources"))
    sources = {**shared, **override}
    label = str(raw.get("label") or key).strip() or key
    return UploadTypeCatalog(
        filing_type=key,
        label=label,
        slots=tuple(slots),
        extract_field_sources=sources,
    )


def _ui_slot_dict(slot: UploadSlot) -> dict[str, Any]:
    return {
        "id": slot.id,
        "label": slot.label,
        "parts": list(slot.parts),
        "required": slot.required,
    }


def _repeatable_ui_slot(
    group: str, number: int = 1, *, series: str = "P"
) -> dict[str, Any]:
    if group == "annexures":
        letter = (series or "P").strip().upper()[:1] or "P"
        if letter == "E":
            letter = "P"
        return {
            "id": f"annexure_{letter.lower()}{number}",
            "label": f"Annexure {letter}-{number}",
            "parts": [f"Annexure {letter}-{number}"],
            "required": False,
            "repeatable": True,
            "repeat_group": "annexures",
            "series": letter,
            "series_label": (
                "Petitioner" if letter == "P" else ("Respondent" if letter == "R" else letter)
            ),
        }
    return {
        "id": f"application_{number}",
        "label": f"Application {number}",
        "parts": [f"Application {number}"],
        "required": False,
        "repeatable": True,
        "repeat_group": "applications",
    }


def ui_catalog(payload: Mapping[str, Any] | None = None) -> dict[str, Any]:
    """Filing-type labels and slots for the UI. No extract internals.

    Catch-all Annexures / Applications slots become the first numbered upload
    (Annexure P-1 / Application 1) with ``repeatable`` so the form can Add
    P-2, P-3, … and Respondent series Annexure R-1, R-2, ….
    ``P`` = Petitioner, ``R`` = Respondent.
    """
    block = load_split_upload_config(payload)
    types = block.get("types") if isinstance(block.get("types"), Mapping) else {}
    catalog: dict[str, Any] = {}
    if not isinstance(types, Mapping):
        return catalog
    for filing_type, raw in types.items():
        if not isinstance(raw, Mapping):
            continue
        try:
            entry = type_catalog(str(filing_type), payload)
        except SplitUploadError:
            continue
        ui_slots: list[dict[str, Any]] = []
        for slot in entry.slots:
            if slot.id == "annexures":
                ui_slots.append(_repeatable_ui_slot("annexures", 1))
                continue
            if slot.id == "applications":
                ui_slots.append(_repeatable_ui_slot("applications", 1))
                continue
            ui_slots.append(_ui_slot_dict(slot))
        catalog[entry.filing_type] = {
            "label": entry.label,
            "slots": ui_slots,
        }
    return catalog


def _part_from_mapping(raw: Mapping[str, Any] | SplitPartInput) -> SplitPartInput:
    if isinstance(raw, SplitPartInput):
        return raw
    slot_id = str(raw.get("slot_id") or "").strip()
    file_id = str(raw.get("file_id") or "").strip()
    parts = _as_part_names(raw.get("document_parts"))
    file_hash = raw.get("file_hash")
    filename = raw.get("filename")
    document_id = raw.get("document_id")
    return SplitPartInput(
        slot_id=slot_id,
        file_id=file_id,
        document_parts=parts,
        file_hash=str(file_hash) if file_hash else None,
        filename=str(filename) if filename else None,
        document_id=str(document_id) if document_id else None,
    )


def validate_parts(
    filing_type: str,
    parts: Sequence[Mapping[str, Any] | SplitPartInput],
    payload: Mapping[str, Any] | None = None,
    *,
    require_all_slots: bool = False,
) -> tuple[UploadTypeCatalog, list[SplitPartInput]]:
    catalog = type_catalog(filing_type, payload)
    allowed = catalog.slot_by_id()
    parsed: list[SplitPartInput] = []
    seen: set[str] = set()
    for raw in parts:
        item = _part_from_mapping(raw)
        if not item.slot_id:
            raise SplitUploadError("Each uploaded file must include slot_id")
        seen.add(item.slot_id)
        slot = allowed.get(item.slot_id) or dynamic_upload_slot(item.slot_id)
        if slot is None:
            raise SplitUploadError(
                f"Unknown slot {item.slot_id!r} for {catalog.filing_type}"
            )
        if not item.file_id:
            raise SplitUploadError(f"No file uploaded for {slot.label}")
        parsed.append(
            SplitPartInput(
                slot_id=slot.id,
                file_id=item.file_id,
                document_parts=slot.parts,
                file_hash=item.file_hash,
                filename=item.filename,
                document_id=item.document_id,
            )
        )

    missing = [
        slot.label for slot in catalog.slots if slot.required and slot.id not in seen
    ]
    if require_all_slots and missing:
        raise SplitUploadError("Missing required documents: " + ", ".join(missing))
    if not parsed:
        raise SplitUploadError(
            "No labeled documents were sliced from the compiled PDF"
        )
    return catalog, parsed


def ordered_parts(
    catalog: UploadTypeCatalog, parts: Sequence[SplitPartInput]
) -> list[SplitPartInput]:
    grouped: dict[str, list[SplitPartInput]] = {}
    for item in parts:
        grouped.setdefault(item.slot_id, []).append(item)
    ordered: list[SplitPartInput] = []
    consumed: set[str] = set()
    annexure_ids = sorted(
        (slot_id for slot_id in grouped if _ANNEXURE_SLOT_RE.fullmatch(slot_id)),
        key=_numbered_slot_sort_key,
    )
    application_ids = sorted(
        (slot_id for slot_id in grouped if _APPLICATION_SLOT_RE.fullmatch(slot_id)),
        key=_numbered_slot_sort_key,
    )
    for slot in catalog.slots:
        if slot.id == "annexures":
            for slot_id in annexure_ids:
                ordered.extend(grouped[slot_id])
                consumed.add(slot_id)
            ordered.extend(grouped.get("annexures", []))
            consumed.add("annexures")
            continue
        if slot.id == "applications":
            for slot_id in application_ids:
                ordered.extend(grouped[slot_id])
                consumed.add(slot_id)
            ordered.extend(grouped.get("applications", []))
            consumed.add("applications")
            continue
        ordered.extend(grouped.get(slot.id, []))
        consumed.add(slot.id)
    for slot_id, items in grouped.items():
        if slot_id not in consumed:
            ordered.extend(items)
    return ordered


def bundle_file_hash(parts: Sequence[SplitPartInput]) -> str:
    payload = [
        [item.slot_id, item.file_hash or item.file_id]
        for item in sorted(parts, key=lambda item: item.slot_id)
    ]
    encoded = json.dumps(payload, separators=(",", ":"), ensure_ascii=True)
    return hashlib.sha256(encoded.encode("utf-8")).hexdigest()


def stitch_parsed_parts_in_order(
    parts: Sequence[SplitPartInput],
    pages_by_slot: Mapping[str, Mapping[int, str]],
    *,
    omit_empty: bool = False,
) -> tuple[dict[int, str], dict[int, list[str]]]:
    """Concatenate per-file pages in the given list order. Global pages are 1-indexed."""
    page_markdown: dict[int, str] = {}
    page_parts: dict[int, list[str]] = {}
    next_page = 1
    for item in parts:
        local = pages_by_slot.get(item.slot_id) or {}
        local_numbers = sorted(int(page) for page in local)
        if not local_numbers:
            if omit_empty:
                continue
            page_markdown[next_page] = (
                f"{_PARSE_STUB_PREFIX} {item.filename or item.slot_id})"
            )
            page_parts[next_page] = list(item.document_parts)
            next_page += 1
            continue
        for local_page in local_numbers:
            text = str(local.get(local_page) or local.get(str(local_page)) or "")
            page_markdown[next_page] = text
            page_parts[next_page] = list(item.document_parts)
            next_page += 1
    return page_markdown, page_parts


def stitch_parsed_parts(
    catalog: UploadTypeCatalog,
    parts: Sequence[SplitPartInput],
    pages_by_slot: Mapping[str, Mapping[int, str]],
) -> tuple[dict[int, str], dict[int, list[str]]]:
    """Concatenate per-file pages in catalog order. Global pages are 1-indexed."""
    return stitch_parsed_parts_in_order(
        ordered_parts(catalog, parts), pages_by_slot
    )


def coerce_page_markdown(raw: Mapping[Any, Any] | None) -> dict[int, str]:
    """JSON round-trips can turn page numbers into strings."""
    pages: dict[int, str] = {}
    for key, value in (raw or {}).items():
        try:
            number = int(key)
        except (TypeError, ValueError):
            continue
        pages[number] = str(value or "")
    return pages


def coerce_page_parts(raw: Mapping[Any, Any] | None) -> dict[int, list[str]]:
    pages: dict[int, list[str]] = {}
    for key, value in (raw or {}).items():
        try:
            number = int(key)
        except (TypeError, ValueError):
            continue
        names = parts_on_page(value)
        if names:
            pages[number] = names
    return pages


def extract_source_parts(catalog: UploadTypeCatalog) -> set[str]:
    """Document parts LlamaExtract needs to fill the record.

    Verify-only parts (Affidavit, Office Report, Index, …) stay out of the
    pack. They are still parsed for scrutiny.
    """
    names = {
        part
        for spec in catalog.extract_field_sources.values()
        for part in spec.fill
    }
    return names - {part for part in names if _excluded_from_extract_pack(part)}


def page_is_extract_source(names: Iterable[str], source_parts: set[str]) -> bool:
    labels = [name for name in names if name]
    if not labels:
        return False
    if all(_excluded_from_extract_pack(name) for name in labels):
        return False
    if not source_parts:
        return not any(_excluded_from_extract_pack(name) for name in labels)
    return any(name in source_parts for name in labels)


PRECISE_PARSE_SLOT_IDS = frozenset(
    {
        "petition",
        "cover_page",
        "memo_of_parties",
        "vakalatnama_appearance",
        "aor_certificate",
        "aors_declaration",
        "impugned_order",
        "affidavit",
        "listing_proforma",
        "advocates_checklist",
        "office_report_limitation",
        "office_report_on_limitation",
    }
)
FAST_PARSE_SLOT_IDS = frozenset(
    {"undefined", "annexures", "applications", "appendix"}
)


def slot_is_extract_source(
    part: SplitPartInput, catalog: UploadTypeCatalog
) -> bool:
    """True when this upload slot is needed to fill LlamaExtract fields."""
    sources = extract_source_parts(catalog)
    names = [name for name in part.document_parts if name]
    if not names:
        slot = resolve_upload_slot(catalog, part.slot_id)
        names = list(slot.parts) if slot else []
    return page_is_extract_source(names, sources)


def parse_action_for_slot(
    part: SplitPartInput,
    *,
    catalog: UploadTypeCatalog,
    parse_scope: str,
    reuse_slots: set[str] | None = None,
) -> str:
    """Return ``parse``, ``reuse``, or ``skip`` for one labeled PDF."""
    slot = (part.slot_id or "").strip()
    scope = (parse_scope or "all").strip().lower()
    reused = reuse_slots or set()
    if scope == "extract_sources":
        if not slot_is_extract_source(part, catalog):
            return "skip"
        return "parse"
    if scope == "unparsed":
        if slot in reused:
            return "reuse"
        return "parse"
    return "parse"


PARSE_ARTIFACT_URL_KEY = "parse_artifact_url"
PARSE_ARTIFACT_KEY_KEY = "parse_artifact_key"


def _int_keyed_mapping(raw: Mapping[Any, Any] | None) -> dict[int, Any]:
    pages: dict[int, Any] = {}
    for key, value in (raw or {}).items():
        try:
            pages[int(key)] = value
        except (TypeError, ValueError):
            continue
    return pages


def dump_parse_artifact(
    *,
    parsed_slots: Sequence[str],
    document_order: Sequence[str],
    pages_by_slot: Mapping[str, Mapping[int, str]],
    parse_job_ids: Mapping[str, str] | None = None,
    layouts_by_slot: Mapping[str, Mapping[int, Any]] | None = None,
) -> dict[str, Any]:
    """Per-slot markdown JSON so a later job can re-stitch in a new send order."""
    job_ids = dict(parse_job_ids or {})
    layouts = layouts_by_slot or {}
    slot_ids = list(document_order)
    for slot_id in pages_by_slot:
        if slot_id not in slot_ids:
            slot_ids.append(slot_id)
    slots: dict[str, Any] = {}
    for slot_id in slot_ids:
        pages = pages_by_slot.get(slot_id) or {}
        layout = layouts.get(slot_id) or {}
        slots[slot_id] = {
            "pages": {
                str(page): str(text or "")
                for page, text in sorted(
                    (
                        (int(number), value)
                        for number, value in pages.items()
                    ),
                    key=lambda item: item[0],
                )
            },
            "parse_job_id": job_ids.get(slot_id),
            "layout": {
                str(page): payload
                for page, payload in sorted(
                    _int_keyed_mapping(layout).items(),
                    key=lambda item: item[0],
                )
                if isinstance(payload, Mapping)
            },
        }
    return {
        "parsed_slots": [slot for slot in parsed_slots if slot],
        "document_order": [slot for slot in document_order if slot],
        "slots": slots,
    }


def load_parse_artifact(
    payload: Mapping[str, Any] | None,
) -> tuple[
    list[str],
    list[str],
    dict[str, dict[int, str]],
    dict[str, str],
    dict[str, dict[int, dict[str, Any]]],
]:
    """Restore per-slot markdown, parse job ids, and layouts from S3 JSON."""
    data = payload if isinstance(payload, Mapping) else {}
    parsed_slots = [
        str(slot).strip()
        for slot in (data.get("parsed_slots") or [])
        if str(slot).strip()
    ]
    document_order = [
        str(slot).strip()
        for slot in (data.get("document_order") or [])
        if str(slot).strip()
    ]
    raw_slots = data.get("slots") if isinstance(data.get("slots"), Mapping) else {}
    pages_by_slot: dict[str, dict[int, str]] = {}
    parse_job_ids: dict[str, str] = {}
    layouts_by_slot: dict[str, dict[int, dict[str, Any]]] = {}
    for slot_id, raw in raw_slots.items():
        key = str(slot_id).strip()
        if not key or not isinstance(raw, Mapping):
            continue
        pages_by_slot[key] = coerce_page_markdown(
            raw.get("pages") if isinstance(raw.get("pages"), Mapping) else {}
        )
        job_id = str(raw.get("parse_job_id") or "").strip()
        if job_id:
            parse_job_ids[key] = job_id
        layout_raw = raw.get("layout")
        layout_pages: dict[int, dict[str, Any]] = {}
        for page, payload_page in _int_keyed_mapping(
            layout_raw if isinstance(layout_raw, Mapping) else {}
        ).items():
            if isinstance(payload_page, Mapping):
                layout_pages[page] = dict(payload_page)
        layouts_by_slot[key] = layout_pages
    if not parsed_slots:
        parsed_slots = [slot for slot, pages in pages_by_slot.items() if pages]
    return parsed_slots, document_order, pages_by_slot, parse_job_ids, layouts_by_slot


def slot_needs_precise_parse(
    part: SplitPartInput, source_parts: set[str]
) -> bool:
    """True when LlamaParse should keep the extract-quality (agentic) tier."""
    slot = (part.slot_id or "").strip().lower()
    if slot in FAST_PARSE_SLOT_IDS or _ANNEXURE_SLOT_RE.fullmatch(slot):
        return False
    if slot.startswith("application_") and slot != "applications":
        return False
    if part.document_parts:
        return page_is_extract_source(part.document_parts, source_parts)
    return slot in PRECISE_PARSE_SLOT_IDS


def _part_pages_to_keep(
    part: str,
    page_markdown: Mapping[int, str],
    page_parts: Mapping[int, Any],
    first: int,
    last: int,
) -> set[int]:
    pages = [
        page
        for page in sorted(page_markdown)
        if part in parts_on_page(page_parts.get(page))
        and not (page_markdown.get(page) or "").strip().startswith(_PARSE_STUB_PREFIX)
    ]
    if not pages:
        return set()
    if len(pages) <= first + last:
        return set(pages)
    keep = set(pages[:first])
    keep.update(pages[-last:])
    return keep


def _windowed_pages_to_keep(
    page_markdown: Mapping[int, str],
    page_parts: Mapping[int, Any],
) -> dict[str, set[int]]:
    return {
        part: _part_pages_to_keep(part, page_markdown, page_parts, first, last)
        for part, (first, last) in EXTRACT_PACK_PAGE_WINDOWS.items()
    }


def _keep_extract_page(
    page: int,
    names: Sequence[str],
    source_parts: set[str],
    window_keep: Mapping[str, set[int]],
) -> bool:
    sources = [name for name in names if name in source_parts]
    if not sources:
        return False
    unwindowed = [name for name in sources if name not in EXTRACT_PACK_PAGE_WINDOWS]
    if unwindowed:
        return True
    return any(page in window_keep.get(name, set()) for name in sources)


def _section_use_notes(catalog: UploadTypeCatalog | None) -> dict[str, str]:
    fill_of: dict[str, list[str]] = {}
    verify_of: dict[str, list[str]] = {}
    if catalog is not None:
        for field_name, spec in catalog.extract_field_sources.items():
            for part in spec.fill:
                fill_of.setdefault(part, []).append(field_name)
            for part in spec.verify:
                verify_of.setdefault(part, []).append(field_name)
    notes: dict[str, str] = {}
    for part, fields in fill_of.items():
        notes[part] = (
            f"Fill {', '.join(fields)} from this section when listed as a fill source."
        )
    for part, fields in verify_of.items():
        extra = (
            f" Check spelling for {', '.join(fields)}; "
            "do not overwrite fill values with text from this section."
        )
        notes[part] = (notes.get(part) or "").rstrip() + extra
    notes.setdefault(
        "Memo of Appearance",
        "Use only for advocates_on_record. AOR footer under the name: Chamber "
        "lines, often not labelled Address. Copy printed text only; do not invent.",
    )
    notes.setdefault(
        "Vakalatnama",
        "Use only for advocates_on_record. Prefer this over Main Petition last page.",
    )
    notes.setdefault(
        "AOR's Certificate",
        "Cause title at the top, then the word CERTIFICATE (or C E R T I F I C A T E). "
        "Both are required. Body starts Certified that / CERTIFIED that the petition "
        "is confined only to the pleadings. Use the signature or DRAWN & FILED BY "
        "block to fill blank advocates_on_record fields only; never override Vakalatnama. "
        "Use bracketed impugned-order particulars only when Impugned Order is not in "
        "this pack.",
    )
    for part, extra in (
        (
            "Listing Proforma",
            " Use only to fill blank advocates_on_record fields. Never override "
            "Vakalatnama. office_address is the Chamber lines on the last page, "
            "under the AOR name / (ADVOCATE-ON-RECORD). Copy printed text only; "
            "do not invent.",
        ),
        (
            "Advocate's Checklist",
            " Use only to fill blank advocates_on_record fields. Never override Vakalatnama.",
        ),
        (
            "Affidavit",
            " Use bracketed impugned-order particulars only when Impugned Order is "
            "not in this pack. Do not copy petitioner or respondent names.",
        ),
        (
            "AOR's Certificate",
            " Use the signature or DRAWN & FILED BY block to fill blank "
            "advocates_on_record fields only; never override Vakalatnama. "
            "Use bracketed impugned-order particulars only when Impugned Order is "
            "not in this pack.",
        ),
        (
            "Vakalatnama",
            " Prefer this over Main Petition last page. Never override Vakalatnama "
            "values from later sections.",
        ),
        (
            "Memo of Appearance",
            " Prefer this over Main Petition last page when Vakalatnama is missing "
            "or prints no AOR. Fill blank office_address from Chamber lines under "
            "the name. Copy printed text only; never override Vakalatnama.",
        ),
    ):
        current = notes.get(part) or ""
        if extra.strip() not in current:
            notes[part] = (current.rstrip() + extra).strip() if current else extra.strip()
    for part in ("Memo of Appearance", "Vakalatnama", "AOR's Certificate"):
        if "Do not copy petitioner or respondent names" not in notes[part]:
            notes[part] = (
                notes[part].rstrip()
                + " Do not copy petitioner or respondent names."
            )
    cert_note = notes.get("AOR's Certificate") or ""
    if "cause title" not in cert_note.lower():
        notes["AOR's Certificate"] = (
            cert_note.rstrip()
            + " This page always has the cause title at the top, then the word "
            "CERTIFICATE (or C E R T I F I C A T E), then Certified that the "
            "petition is confined only to the pleadings. Use the signature or "
            "DRAWN & FILED BY block for advocates_on_record only. "
            "Cause title without CERTIFICATE is Cover Page or Main Petition."
        )
    petition_note = notes.get("Main Petition") or ""
    extras: list[str] = []
    petition_fill = fill_of.get("Main Petition", [])
    if "advocates_on_record" in petition_fill:
        extras.append(
            " For advocates_on_record, use only the very last page: Drawn By, "
            "Filed on, DRAWN & FILED BY, Advocate for Petitioner/Respondent, then "
            "Chamber lines under the name. Copy printed text only; do not invent. "
            "Fill blanks if Vakalatnama is missing, prints no AOR, or leaves that "
            "field blank. Do not take advocate names or addresses from the opening "
            "party-list pages."
        )
    if "cause_title" in petition_fill:
        extras.append(
            " For cause_title, use this caption only when Cover Page is not in this pack."
        )
    if "petition_type" in petition_fill:
        extras.append(
            " For petition_type, fill from this cause title first; Cover Page is the fallback."
        )
    if "impugned_orders" in petition_fill:
        extras.append(
            " For impugned_orders, use bracketed particulars only when Impugned Order "
            "is not in this pack."
        )
    if "petition_date" in petition_fill:
        extras.append(
            " For petition_date, use only the last page after Main Prayer or Prayer. "
            "If Date Drafted and Date Filed on are both printed, use Date Drafted only."
        )
    for extra in extras:
        if extra.strip() not in petition_note:
            petition_note = petition_note.rstrip() + extra
    if petition_note:
        notes["Main Petition"] = petition_note
    cover_note = notes.get("Cover Page") or ""
    cover_fill = fill_of.get("Cover Page", [])
    cover_bits: list[str] = []
    if "cause_title" in cover_fill:
        cover_bits.append(" Prefer this page for cause_title.")
    if "petition_type" in cover_fill:
        cover_bits.append(
            " For petition_type, use this cause title only if Main Petition does not print it."
        )
    if "impugned_orders" in cover_fill:
        cover_bits.append(
            " For impugned_orders, use bracketed particulars only when Impugned Order "
            "is not in this pack."
        )
    for extra in cover_bits:
        if extra.strip() not in cover_note:
            cover_note = cover_note.rstrip() + extra
    if cover_note:
        notes["Cover Page"] = cover_note
    return notes


def extract_pack_preamble(catalog: UploadTypeCatalog | None = None) -> str:
    lines = [
        "# Extraction rules",
        "Copy printed text only. Do not invent names, addresses, dates, or "
        "categories. If a fill-source document part is not in this pack, or the "
        "value is not printed there, write N/A. Do not guess from other parts.",
        "",
    ]
    sources = catalog.extract_field_sources if catalog is not None else {}
    for field_name, spec in sources.items():
        fill = ", ".join(f"[{p}]" for p in spec.fill) or "(none)"
        bit = f"- {field_name}: fill from {fill}"
        if spec.verify:
            verify = ", ".join(f"[{p}]" for p in spec.verify)
            bit += f". Check spelling against {verify}; never overwrite fill text"
        if field_name in PARTY_FIELDS:
            bit += (
                ". List every petitioner and respondent from the starting pages of "
                "the Main Petition; that party list can run 3-4 pages or more. "
                "Cover Page prints only one petitioner and one respondent (the main "
                "names, plus And Anr/Ors if extras exist). If a field is blank on "
                "those starting pages or Main Petition is missing, use the Cover Page "
                "if it is in this pack. Merge blank particulars between those two only. "
                "Never copy party names or addresses from Vakalatnama or Memo of Parties. "
                "Use Cover Page to mark is_primary from the cover cause-title names. "
                "Petitioner 1 on the Main Petition starting pages must be the same "
                "person as cause_title.main_petitioner; respondent 1 must be the same "
                "person as cause_title.main_respondent. Flag a different name, a "
                "missing name, or a side swap (Cause Title petitioner listed as a "
                "respondent, or the reverse). And Anr/Ors on Cover Page means extra "
                "parties exist; list those names from the Main Petition starting pages"
            )
        if field_name == "cause_title":
            bit += (
                ". Prefer Cover Page. If Cover Page is not in this pack, fill from the "
                "Main Petition caption and draft formatted_title from party counts"
            )
        if field_name == "petition_type":
            bit += (
                ". Prefer the Main Petition cause title. If not printed there, use the "
                "Cover Page cause title"
            )
        if field_name == "advocates_on_record":
            bit += f". {AOR_EXTRACT_RULES.rstrip('.')}"
        if field_name == "impugned_orders":
            bit += (
                ". Prefer the Impugned Order document. If it is not in this pack, use "
                "bracketed particulars on Main Petition, Cover Page, AOR's Certificate, "
                "and Affidavit. Keep one consistent set"
            )
        lines.append(bit + ".")
    lines.extend(
        [
            "- formatted_title: Cover Page names are main_petitioner and main_respondent "
            "when Cover Page is present. If Cover Page is missing, use Main Petition "
            "petitioner 1 and respondent 1. "
            "Store those names without And Anr, And Ors, Petitioner, or Respondent. "
            "Each side is 'MainName' (1 party), 'MainName and Anr.' (exactly 2), "
            "'MainName and Ors.' (3 or more). Join with ' VS '. Never use square "
            "brackets: write 'and Anr.' and 'and Ors.', not '[And ors.]' or '[and Anr.]'. "
            "Do not append and Anr. or and Ors. if that suffix is already on the name.",
            "- kind: INDIVIDUAL or ORGANIZATION from the printed name. Organization "
            "prefixes: M/s, M/s., Messrs, The, Union, Government of, Ministry of, "
            "Department of. Suffixes: Pvt Ltd, Pvt. Ltd., Private Limited, Ltd, Limited, "
            "LLP, LLC, Inc., Corp., Corporation, Co., Company, Foundation, Trust, "
            "Society, Association. 'The' is a name prefix (The State of …), not every 'the'.",
            "- acting_through: look under the party for 'acting through' / 'through'. "
            "ORGANIZATION almost always has it; if missing, add an inconsistencies item. "
            "INDIVIDUAL: optional.",
            "- relief_sort: copy only the prayer body under Main Prayer / Prayer on the "
            "last 2-3 pages of the Main Petition. Do not include the heading, number, "
            "or markdown such as '7. MAIN PRAYER:' or '<u>**MAIN PRAYER**</u>:'.",
            "- petition_date: on the last page of the Main Petition, after Main Prayer "
            "/ Prayer, copy the date next to Date, Date Drafted, or Date Filed on. "
            "If both Date Drafted and Date Filed on are printed, use only Date Drafted. "
            "Return the date only, without the label.",
            "- confidence: percentage string such as 95% or 65% on each object.",
            "- inconsistencies: record spelling or value mismatches between fill and "
            "verify sources. Always record a letter-level mismatch of the Cause Title "
            "main petitioner or main respondent versus that same person's name on "
            "the Main Petition starting pages (example: Shalija vs Shailja). "
            "The Cause Title petitioner must be petitioner 1 on those starting pages; "
            "the Cause Title respondent must be respondent 1. Flag when they are a "
            "different person, missing, or listed on the other side. Cover Page has "
            "only one name per side. Do not skip the main-name spelling because Cover "
            "Page also says And Anr/Ors. "
            "Do not invent extra parties to resolve a mismatch. "
            "Do not flag Cover Page / caption role labels Petitioner, Petitioner(s), "
            "Respondent, or Respondent(s), with or without leading dots (... or …). "
            "Those marks are not part of the party name. "
            "Do not flag ALL CAPS vs title case as a spelling mismatch "
            "(Impugned Order names are often printed in capitals). "
            "Do not compare party names against the Impugned Order. "
            "One item per distinct name spelling; do not repeat the same two names. "
            "Skip only extra serials (petitioner/respondent 2, 3, …) compared against "
            "Cover Page And Anr/Ors; those extra people are listed on the Main "
            "Petition starting pages, not on the cover shorthand. "
            "Party-name raw_text must be: Cause Title: \"Name\"; Main Petition: "
            "\"Name\". Quote the person's name only: no And Anr/Ors, "
            "no Petitioner/Respondent, and do not list Vakalatnama, Affidavit, or "
            "AOR's Certificate as party-name sources. "
            "Also record Impugned Order case number, date, or forum mismatches "
            "across Impugned Order, Main Petition, Cover Page, AOR's Certificate, "
            "and Affidavit. Also record printed AOR chamber/office address "
            "mismatches across Vakalatnama, Memo of Appearance, Main Petition last "
            "page, Listing Proforma last page, and AOR's Certificate. Quote both "
            "printed strings. Do not invent an address. Ignore extra Ph. on the "
            "same chamber line. "
            "items[].id is '1', '2', …; use raw_text, not detail.",
        ]
    )
    return "\n".join(lines).strip()


def build_extract_pack_markdown(
    page_markdown: Mapping[int, str],
    page_parts: Mapping[int, Any],
    source_parts: set[str],
    catalog: UploadTypeCatalog | None = None,
) -> str:
    sections: list[str] = []
    window_keep = _windowed_pages_to_keep(page_markdown, page_parts)
    notes = _section_use_notes(catalog)
    for page in sorted(page_markdown):
        names = parts_on_page(page_parts.get(page))
        if not page_is_extract_source(names, source_parts):
            continue
        if not _keep_extract_page(page, names, source_parts, window_keep):
            continue
        label = " / ".join(names) or "Unknown"
        body = (page_markdown.get(page) or "").strip()
        if body.startswith(_PARSE_STUB_PREFIX):
            continue
        note = next((notes[n] for n in names if n in notes), None)
        if note:
            sections.append(f"## [{label}] (p. {page})\n\n> {note}\n\n{body}".rstrip())
        else:
            sections.append(f"## [{label}] (p. {page})\n\n{body}".rstrip())
    packed = "\n\n".join(sections).strip()
    if not packed:
        return ""
    return f"{extract_pack_preamble(catalog)}\n\n{packed}"


def _look_only_text(field_name: str, spec: FieldSources) -> str:
    fill = ", ".join(spec.fill) or "no fill source"
    extra = f"Fill only from {fill}."
    if spec.verify:
        extra += (
            f" Check spelling against {', '.join(spec.verify)}. "
            "If spellings differ, keep the fill value and add an inconsistencies item."
        )
    if field_name in PARTY_FIELDS:
        extra += (
            " List every petitioner and respondent from the starting pages of the "
            "Main Petition; that party list can run 3-4 pages or more. Cover Page "
            "prints only one petitioner and one respondent. If a field is blank on "
            "those starting pages or Main Petition is missing, use the Cover Page if "
            "it is in this pack. If a field is blank in one of those parts, fill it "
            "from the other. If neither Main Petition nor Cover Page is in this pack, "
            "set party names to N/A. Never copy party names or addresses from Vakalatnama, PoA/BR, "
            "Memo of Appearance, AOR's Certificate, or Memo of Parties. "
            "Use Cover Page to decide which already-listed party is primary and to "
            "fill blanks. Petitioner 1 must be the same person as "
            "cause_title.main_petitioner; respondent 1 must be the same person as "
            "cause_title.main_respondent. Flag a different name, a missing name, or "
            "a side swap. Extra petitioners and respondents continue on later starting "
            "pages of the Main Petition; Cover Page And Anr/Ors is not the second "
            "party's name. Do not invent parties. Write N/A if a field is not printed "
            "on a fill source that is present. "
            "kind is INDIVIDUAL or ORGANIZATION from name prefixes/suffixes. "
            "ORGANIZATION without acting_through is an inconsistencies item."
        )
    if field_name == "cause_title":
        extra += (
            " main_petitioner and main_respondent are the names on the Cover Page "
            "cause-title line if Cover Page is in this pack, without And Anr / And Ors / "
            "Petitioner / Respondent. If Cover Page is missing, use petitioner 1 and "
            "respondent 1 from the Main Petition caption. "
            "formatted_title uses those main names plus and Anr. for exactly "
            "one extra party on that side and and Ors. for two or more extras. "
            "Never wrap and Anr. or and Ors. in square brackets. "
            "Do not treat trailing Petitioner / Petitioner(s) / Respondent / "
            "Respondent(s), with or without dots, as a spelling mismatch. "
            "Do not write 'and Anr. and Anr.' "
            "Cover Page has only one petitioner and one respondent. "
            "Cross-check Cause Title with the starting pages of the Main Petition "
            "(the party list can run 3-4 pages or more). "
            "The Cause Title petitioner must be the same person as petitioner 1; "
            "the Cause Title respondent must be respondent 1. "
            "If the Cover Page / Cause Title main name differs in letters from "
            "petitioner 1 or respondent 1 on those starting pages (Shalija vs "
            "Shailja), that is an inconsistencies item. Also flag a different "
            "person, a missing name, or a side swap (Cause Title petitioner listed "
            "as a respondent, or the reverse). Extra parties on later starting "
            "pages are not a spelling mismatch against And Anr/Ors."
        )
    if field_name == "petition_type":
        extra += (
            " Fill from the cause title of the Main Petition first "
            "(e.g. Special Leave Petition (Civil)). If that heading is not printed "
            "there, use the Cover Page cause title."
        )
    if field_name == "advocates_on_record":
        extra += f" {AOR_EXTRACT_RULES}"
    if field_name == "impugned_orders":
        extra += (
            " If the Impugned Order document is in this pack, fill from it. If it is "
            "not attached, fill from bracketed particulars on the Main Petition cause "
            "title, Cover Page, AOR's Certificate, and Affidavit. Cross-check those "
            "documents. Keep Impugned Order PDF values when present; otherwise keep "
            "one consistent set and add an inconsistencies item for mismatches."
        )
    if field_name == "relief_sort":
        extra += (
            " Copy only the prayer body under Main Prayer or Prayer on the last 2-3 pages "
            "of the Main Petition. Omit the heading, clause number, HTML, and markdown."
        )
    if field_name == "petition_date":
        extra += (
            " Read the last page of the Main Petition, after Main Prayer or Prayer. "
            "Use the date next to Date, Date Drafted, or Date Filed on. If both "
            "Date Drafted and Date Filed on are printed, use only the Date Drafted "
            "date. Return the date only, without the label. Do not use a date inside "
            "the prayer or from any other document."
        )
    return f"{extra}{LOOK_ONLY_SUFFIX}"


def inject_where_to_look(
    schema: Mapping[str, Any],
    sources: Mapping[str, FieldSources | Sequence[str]],
) -> dict[str, Any]:
    """Copy extract JSON schema and append look-only guidance to field descriptions."""
    updated = copy.deepcopy(dict(schema))
    props = updated.get("properties")
    if not isinstance(props, dict):
        return updated
    for field_name, spec in sources.items():
        node = props.get(field_name)
        if not isinstance(node, dict):
            continue
        parsed = spec if isinstance(spec, FieldSources) else _parse_field_sources(spec)
        if parsed is None or not parsed.all_parts():
            continue
        extra = _look_only_text(field_name, parsed)
        existing = str(node.get("description") or "").rstrip()
        if extra in existing:
            continue
        node["description"] = f"{existing} {extra}".strip() if existing else extra
    return updated


def build_extract_system_prompt(catalog: UploadTypeCatalog) -> str:
    lines = [
        "You are extracting a compiled Supreme Court filing record from an already-split paper book.",
        "Each section is labelled with its document part, for example ## [Cover Page] (p. 1).",
        "Copy printed text only. Do not invent or complete a field from a document "
        "part that is not a fill source for that field. If that fill-source document is "
        "not in this pack, or the value is not printed there, write N/A.",
        "source_part must be the labelled Split name (Cover Page, Main Petition, …). "
        "source_pages must be the integer page numbers in the headings, for example (p. 6).",
        "Ignore Annexures, Appendix, applications, Index, "
        "Synopsis, List of Dates, Filing Memo, and "
        "Office Report on Limitation. Those pages are not in this pack.",
        "",
    ]
    for field_name, spec in catalog.extract_field_sources.items():
        line = f"- {field_name}: fill {', '.join(spec.fill) or '(none)'}"
        if spec.verify:
            line += f"; verify {', '.join(spec.verify)}"
        lines.append(line)
    lines.extend(
        [
            "- formatted_title: MainName / MainName and Anr. / MainName and Ors. per side, joined by VS. Main names from Cover Page if present, else Main Petition petitioner 1 / respondent 1, without And Anr / And Ors. Never use [And ors.] or other square brackets.",
            "- kind: INDIVIDUAL or ORGANIZATION from name prefixes/suffixes on Main Petition.",
            "- acting_through: required for ORGANIZATION (missing is an inconsistency); optional for INDIVIDUAL.",
            "- petition_type: Main Petition cause title first; Cover Page cause title if not printed there.",
            "- advocates_on_record: Vakalatnama first. Copy the printed AOR footer only; leave a field null if it is not printed; do not invent name, code, email, mobile, firm, chamber, or PIN. If Vakalatnama is missing or prints no AOR, Memo of Appearance footer, then last page of the Main Petition (very end: Drawn By / Filed on / DRAWN & FILED BY / Advocate for Petitioner / Chamber). Do not use petition opening pages. If office_address is still blank, last page of Listing Proforma / Proforma for First Listing. Remaining blanks: AOR's Certificate signature / DRAWN & FILED BY and Advocate's Checklist. Never override Vakalatnama.",
            "- impugned_orders: Impugned Order PDF first. If missing, bracketed particulars on Main Petition, Cover Page, AOR's Certificate, and Affidavit. Keep one consistent set.",
            "- relief_sort: prayer body only under Main Prayer / Prayer on the last 2-3 pages of the Main Petition. Do not include the heading or markdown.",
            "- petition_date: last page of the Main Petition, after Main Prayer / Prayer. Use Date, Date Drafted, or Date Filed on. If both Date Drafted and Date Filed on are printed, use only Date Drafted. Return the date only.",
            "- confidence: percentage strings such as 95% or 65%.",
            "- inconsistencies: one item per spelling or value mismatch between fill and verify sources. Always keep the Cause Title main petitioner/respondent letter mismatch versus petitioner 1 / respondent 1 on the Main Petition starting pages (Shalija vs Shailja). Cover Page has only one name per side; extra parties continue on later starting pages. Flag a different person, a missing name, or a side swap. Also record Impugned Order case number, date, or forum mismatches across Impugned Order, Main Petition, Cover Page, AOR's Certificate, and Affidavit. Also record printed AOR chamber/office address mismatches across Vakalatnama, Memo of Appearance, Main Petition last page, Listing Proforma last page, and AOR's Certificate. Quote both printed strings. Do not invent an address. Ignore extra Ph. on the same chamber line. id is '1', '2', …; use raw_text as Cause Title: \"Name\"; Main Petition: \"Name\". Do not list Vakalatnama, Affidavit, Memo of Parties, or AOR's Certificate as party-name sources. Do not flag Petitioner / Respondent caption labels, with or without dots. Do not flag ALL CAPS vs title case. Do not compare party names against the Impugned Order. Do not repeat the same name pair. Extra serials on the Main Petition starting pages are not spelling errors against Cover Page And Anr/Ors.",
        ]
    )
    return "\n".join(lines).strip()


def extract_configuration(
    extract_config: Any,
    catalog: UploadTypeCatalog,
) -> dict[str, Any]:
    from .config import LegalExtractRecord, dump_api_configuration

    dumped = dump_api_configuration(extract_config)
    dumped["data_schema"] = inject_where_to_look(
        LegalExtractRecord.model_json_schema(),
        catalog.extract_field_sources,
    )
    dumped["system_prompt"] = build_extract_system_prompt(catalog)
    return dumped


def find_part(parts: Sequence[SplitPartInput], slot_id: str) -> SplitPartInput | None:
    for item in parts:
        if item.slot_id == slot_id:
            return item
    return None


def display_filename(
    filing_type: str,
    parts: Sequence[SplitPartInput],
    original: str | None = None,
) -> str:
    from .job_timing import uploaded_filename

    named = uploaded_filename(original)
    if named:
        return named
    cover = find_part(parts, "cover_page")
    if cover and cover.filename:
        return uploaded_filename(cover.filename) or cover.filename
    petition = find_part(parts, PETITION_SLOT_ID)
    if petition and petition.filename:
        return uploaded_filename(petition.filename) or petition.filename
    for item in parts:
        if item.filename:
            return uploaded_filename(item.filename) or item.filename
    return f"{filing_type} split upload"
