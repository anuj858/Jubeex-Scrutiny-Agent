"""Catalog, validation, and extract-pack helpers for already-split uploads.

LlamaSplit is not used. `document_part` comes from labeled upload slots in
`configs/config.json` `split_upload.types`.
"""

from __future__ import annotations

import copy
import hashlib
import json
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass, field
from functools import lru_cache
from pathlib import Path
from typing import Any

from .document_parts import parts_on_page

_CONFIG_PATH = Path(__file__).resolve().parents[2] / "configs" / "config.json"

EXTRACT_PACK_EXCLUDED_PARTS = frozenset({"Annexures", "Appendix"})
LOOK_ONLY_SUFFIX = " Ignore other document parts."
PETITION_SLOT_ID = "petition"
UNDEFINED_SLOT_ID = "undefined"
_PARSE_STUB_PREFIX = "(No parse text for"


class SplitUploadError(ValueError):
    """Invalid filing type or slot mapping for a split upload."""


@dataclass(frozen=True)
class UploadSlot:
    id: str
    label: str
    parts: tuple[str, ...]
    required: bool = True


@dataclass(frozen=True)
class UploadTypeCatalog:
    filing_type: str
    label: str
    slots: tuple[UploadSlot, ...]
    extract_field_sources: dict[str, tuple[str, ...]] = field(default_factory=dict)

    def slot_by_id(self) -> dict[str, UploadSlot]:
        return {slot.id: slot for slot in self.slots}


@dataclass(frozen=True)
class SplitPartInput:
    slot_id: str
    file_id: str
    document_parts: tuple[str, ...] = ()
    file_hash: str | None = None
    filename: str | None = None


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
        required=bool(raw.get("required", True)),
    )


def _parse_sources(raw: Any) -> dict[str, tuple[str, ...]]:
    if not isinstance(raw, Mapping):
        return {}
    sources: dict[str, tuple[str, ...]] = {}
    for field_name, parts in raw.items():
        key = str(field_name).strip()
        names = _as_part_names(parts)
        if key and names:
            sources[key] = names
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


def ui_catalog(payload: Mapping[str, Any] | None = None) -> dict[str, Any]:
    """Filing-type labels and slots for the UI. No extract internals."""
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
        catalog[entry.filing_type] = {
            "label": entry.label,
            "slots": [
                {
                    "id": slot.id,
                    "label": slot.label,
                    "parts": list(slot.parts),
                    "required": slot.required,
                }
                for slot in entry.slots
            ],
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
    return SplitPartInput(
        slot_id=slot_id,
        file_id=file_id,
        document_parts=parts,
        file_hash=str(file_hash) if file_hash else None,
        filename=str(filename) if filename else None,
    )


def validate_parts(
    filing_type: str,
    parts: Sequence[Mapping[str, Any] | SplitPartInput],
    payload: Mapping[str, Any] | None = None,
) -> tuple[UploadTypeCatalog, list[SplitPartInput]]:
    catalog = type_catalog(filing_type, payload)
    allowed = catalog.slot_by_id()
    parsed: list[SplitPartInput] = []
    seen: set[str] = set()
    for raw in parts:
        item = _part_from_mapping(raw)
        if not item.slot_id:
            raise SplitUploadError("Each uploaded file must include slot_id")
        if item.slot_id in seen:
            raise SplitUploadError(f"Duplicate slot: {item.slot_id}")
        seen.add(item.slot_id)
        slot = allowed.get(item.slot_id)
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
            )
        )

    missing = [
        slot.label for slot in catalog.slots if slot.required and slot.id not in seen
    ]
    if missing:
        raise SplitUploadError("Missing required documents: " + ", ".join(missing))
    return catalog, parsed


def ordered_parts(
    catalog: UploadTypeCatalog, parts: Sequence[SplitPartInput]
) -> list[SplitPartInput]:
    by_id = {item.slot_id: item for item in parts}
    return [by_id[slot.id] for slot in catalog.slots if slot.id in by_id]


def bundle_file_hash(parts: Sequence[SplitPartInput]) -> str:
    payload = [
        [item.slot_id, item.file_hash or item.file_id]
        for item in sorted(parts, key=lambda item: item.slot_id)
    ]
    encoded = json.dumps(payload, separators=(",", ":"), ensure_ascii=True)
    return hashlib.sha256(encoded.encode("utf-8")).hexdigest()


def stitch_parsed_parts(
    catalog: UploadTypeCatalog,
    parts: Sequence[SplitPartInput],
    pages_by_slot: Mapping[str, Mapping[int, str]],
) -> tuple[dict[int, str], dict[int, list[str]]]:
    """Concatenate per-file pages in catalog order. Global pages are 1-indexed."""
    page_markdown: dict[int, str] = {}
    page_parts: dict[int, list[str]] = {}
    next_page = 1
    for item in ordered_parts(catalog, parts):
        local = pages_by_slot.get(item.slot_id) or {}
        local_numbers = sorted(int(page) for page in local)
        if not local_numbers:
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
    names = {part for parts in catalog.extract_field_sources.values() for part in parts}
    return names - EXTRACT_PACK_EXCLUDED_PARTS


def page_is_extract_source(names: Iterable[str], source_parts: set[str]) -> bool:
    labels = [name for name in names if name]
    if not labels:
        return False
    if all(name in EXTRACT_PACK_EXCLUDED_PARTS for name in labels):
        return False
    if not source_parts:
        return not any(name in EXTRACT_PACK_EXCLUDED_PARTS for name in labels)
    return any(name in source_parts for name in labels)


def build_extract_pack_markdown(
    page_markdown: Mapping[int, str],
    page_parts: Mapping[int, Any],
    source_parts: set[str],
) -> str:
    sections: list[str] = []
    for page in sorted(page_markdown):
        names = parts_on_page(page_parts.get(page))
        if not page_is_extract_source(names, source_parts):
            continue
        label = " / ".join(names) or "Unknown"
        body = (page_markdown.get(page) or "").strip()
        if body.startswith(_PARSE_STUB_PREFIX):
            continue
        sections.append(f"## [{label}] (p. {page})\n\n{body}".rstrip())
    return "\n\n".join(sections).strip()


def _look_only_text(parts: Sequence[str]) -> str:
    joined = ", ".join(parts)
    return f"Look only in {joined}.{LOOK_ONLY_SUFFIX}"


def inject_where_to_look(
    schema: Mapping[str, Any],
    sources: Mapping[str, Sequence[str]],
) -> dict[str, Any]:
    """Copy extract JSON schema and append look-only guidance to field descriptions."""
    updated = copy.deepcopy(dict(schema))
    props = updated.get("properties")
    if not isinstance(props, dict):
        return updated
    for field_name, parts in sources.items():
        node = props.get(field_name)
        if not isinstance(node, dict) or not parts:
            continue
        extra = _look_only_text(parts)
        existing = str(node.get("description") or "").rstrip()
        if extra in existing:
            continue
        node["description"] = f"{existing} {extra}".strip() if existing else extra
    return updated


def build_extract_system_prompt(catalog: UploadTypeCatalog) -> str:
    lines = [
        "You are extracting a Core Filing Record from an already-split Supreme Court filing.",
        "Each section is labelled with its document part, for example ## [Listing Proforma] (p. 3).",
        "Extract each field only from the parts listed below. Ignore Annexures and Appendix.",
        "",
    ]
    for field_name, parts in catalog.extract_field_sources.items():
        lines.append(f"- {field_name}: {', '.join(parts)}")
    return "\n".join(lines).strip()


def extract_configuration(
    extract_config: Any,
    catalog: UploadTypeCatalog,
) -> dict[str, Any]:
    dumped = extract_config.model_dump(
        exclude={"configuration_id", "product_type"},
        exclude_none=True,
    )
    schema = dumped.get("data_schema") or {}
    if isinstance(schema, Mapping):
        dumped["data_schema"] = inject_where_to_look(
            schema, catalog.extract_field_sources
        )
    dumped["system_prompt"] = build_extract_system_prompt(catalog)
    return dumped


def find_part(parts: Sequence[SplitPartInput], slot_id: str) -> SplitPartInput | None:
    for item in parts:
        if item.slot_id == slot_id:
            return item
    return None


def display_filename(filing_type: str, parts: Sequence[SplitPartInput]) -> str:
    cover = find_part(parts, "cover_page")
    if cover and cover.filename:
        return cover.filename
    petition = find_part(parts, PETITION_SLOT_ID)
    if petition and petition.filename:
        return petition.filename
    for item in parts:
        if item.filename:
            return item.filename
    return f"{filing_type} split upload"
