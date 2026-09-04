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

from .document_parts import MAIN_PETITION_PART, parts_on_page

_CONFIG_PATH = Path(__file__).resolve().parents[2] / "configs" / "config.json"

EXTRACT_PACK_EXCLUDED_PARTS = frozenset({"Annexures", "Appendix"})
# Main Petition grounds stay out of Extract. Keep page 1 (parties) and the last
# pages (prayer / relief).
PETITION_PACK_FIRST_PAGES = 1
PETITION_PACK_LAST_PAGES = 3
LOOK_ONLY_SUFFIX = " Ignore other document parts."
PETITION_SLOT_ID = "petition"
UNDEFINED_SLOT_ID = "undefined"
_PARSE_STUB_PREFIX = "(No parse text for"
PARTY_FIELDS = frozenset({"petitioners", "respondents"})


class SplitUploadError(ValueError):
    """Invalid filing type or slot mapping for a split upload."""


@dataclass(frozen=True)
class UploadSlot:
    id: str
    label: str
    parts: tuple[str, ...]
    required: bool = True


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
    names = {
        part
        for spec in catalog.extract_field_sources.values()
        for part in spec.all_parts()
    }
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


def _petition_pages_to_keep(
    page_markdown: Mapping[int, str],
    page_parts: Mapping[int, Any],
) -> set[int]:
    petition_pages = [
        page
        for page in sorted(page_markdown)
        if MAIN_PETITION_PART in parts_on_page(page_parts.get(page))
        and not (page_markdown.get(page) or "").strip().startswith(_PARSE_STUB_PREFIX)
    ]
    if not petition_pages:
        return set()
    keep = set(petition_pages[:PETITION_PACK_FIRST_PAGES])
    keep.update(petition_pages[-PETITION_PACK_LAST_PAGES:])
    return keep


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
        "Vakalatnama",
        "Use only for advocates_on_record.",
    )
    notes.setdefault(
        "AOR's Declaration",
        "Use only for advocates_on_record.",
    )
    for part in ("Vakalatnama", "AOR's Declaration"):
        if "Do not copy petitioner or respondent names" not in notes[part]:
            notes[part] = (
                notes[part].rstrip()
                + " Do not copy petitioner or respondent names."
            )
    return notes


def extract_pack_preamble(catalog: UploadTypeCatalog | None = None) -> str:
    lines = [
        "# Extraction rules",
        "Copy printed text only. Do not invent names, addresses, dates, or "
        "categories. If a value is not printed in the allowed fill section, leave it null.",
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
                ". Prefer Memo of Parties; if it is missing, use the first page of "
                "the Main Petition. Merge blank particulars between those two only. "
                "Never copy party names or addresses from Vakalatnama or Cover Page. "
                "Use Cover Page only to mark is_primary from the cover cause-title names"
            )
        lines.append(bit + ".")
    lines.extend(
        [
            "- formatted_title: Cover Page names are main_petitioner and main_respondent. "
            "Store those names without And Anr, And Ors, Petitioner, or Respondent. "
            "Each side is 'MainName' (1 party), 'MainName and Anr.' (exactly 2), "
            "'MainName and Ors.' (3 or more). Join with ' VS '. Do not append and Anr. "
            "or and Ors. if that suffix is already on the name.",
            "- kind: INDIVIDUAL or ORGANIZATION from the printed name. Organization "
            "prefixes: M/s, M/s., Messrs, The, Union, Government of, Ministry of, "
            "Department of. Suffixes: Pvt Ltd, Pvt. Ltd., Private Limited, Ltd, Limited, "
            "LLP, LLC, Inc., Corp., Corporation, Co., Company, Foundation, Trust, "
            "Society, Association. 'The' is a name prefix (The State of …), not every 'the'.",
            "- acting_through: look under the party for 'acting through' / 'through'. "
            "ORGANIZATION almost always has it; if missing, add an inconsistencies item. "
            "INDIVIDUAL: optional.",
            "- relief_sort: Main Prayer / Prayer heading on the last 2-3 pages of the "
            "Main Petition only.",
            "- confidence: percentage string such as 95% or 65% on each object.",
            "- inconsistencies: record spelling or value mismatches between fill and "
            "verify sources. Do not invent extra parties to resolve a mismatch. "
            "Do not flag Cover Page / caption role labels Petitioner, Petitioner(s), "
            "Respondent, or Respondent(s), with or without leading dots (... or …). "
            "Those marks are not part of the party name. "
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
    petition_keep = _petition_pages_to_keep(page_markdown, page_parts)
    notes = _section_use_notes(catalog)
    for page in sorted(page_markdown):
        names = parts_on_page(page_parts.get(page))
        if not page_is_extract_source(names, source_parts):
            continue
        label = " / ".join(names) or "Unknown"
        body = (page_markdown.get(page) or "").strip()
        if body.startswith(_PARSE_STUB_PREFIX):
            continue
        if MAIN_PETITION_PART in names:
            others = [n for n in names if n in source_parts and n != MAIN_PETITION_PART]
            if page not in petition_keep and not others:
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
            " Prefer Memo of Parties; if it is missing, use the first page of the "
            "Main Petition. If a field is blank in one of those parts, fill it from the "
            "other. Never copy party names or addresses from Vakalatnama, PoA/BR, "
            "Memo of Appearance, AOR's Declaration, or Cover Page. Use Cover Page "
            "only to decide which already-listed party is primary. Do not invent "
            "parties. Leave a field null if it is not printed on a fill source. "
            "kind is INDIVIDUAL or ORGANIZATION from name prefixes/suffixes. "
            "ORGANIZATION without acting_through is an inconsistencies item."
        )
    if field_name == "cause_title":
        extra += (
            " main_petitioner and main_respondent are the names on the Cover Page "
            "cause-title line without And Anr / And Ors / Petitioner / Respondent. "
            "formatted_title uses and Anr. for exactly one extra "
            "party on that side and and Ors. for two or more extras. "
            "Do not treat trailing Petitioner / Petitioner(s) / Respondent / "
            "Respondent(s), with or without dots, as a spelling mismatch. "
            "Do not write 'and Anr. and Anr.'"
        )
    if field_name == "relief_sort":
        extra += (
            " Copy only the block under Main Prayer or Prayer on the last 2-3 pages "
            "of the Main Petition."
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
        "part that is not a fill source for that field. If it is not printed there, leave it null.",
        "source_part must be the labelled Split name (Memo of Parties, Cover Page, Main Petition, …). "
        "source_pages must be the integer page numbers in the headings, for example (p. 6).",
        "Ignore Annexures and Appendix.",
        "",
    ]
    for field_name, spec in catalog.extract_field_sources.items():
        line = f"- {field_name}: fill {', '.join(spec.fill) or '(none)'}"
        if spec.verify:
            line += f"; verify {', '.join(spec.verify)}"
        lines.append(line)
    lines.extend(
        [
            "- formatted_title: MainName / MainName and Anr. / MainName and Ors. per side, joined by VS. Main names from Cover Page without And Anr / And Ors.",
            "- kind: INDIVIDUAL or ORGANIZATION from name prefixes/suffixes on Main Petition.",
            "- acting_through: required for ORGANIZATION (missing is an inconsistency); optional for INDIVIDUAL.",
            "- relief_sort: Main Prayer / Prayer on the last 2-3 pages of the Main Petition.",
            "- confidence: percentage strings such as 95% or 65%.",
            "- inconsistencies: one item per spelling or value mismatch between fill and verify sources. id is '1', '2', …; use raw_text. Do not flag Petitioner / Respondent / Petitioner(s) / Respondent(s) caption labels, with or without dots, as a mismatch.",
        ]
    )
    return "\n".join(lines).strip()


def extract_configuration(
    extract_config: Any,
    catalog: UploadTypeCatalog,
) -> dict[str, Any]:
    from .config import LegalExtractRecord

    dumped = extract_config.model_dump(
        exclude={"configuration_id", "product_type"},
        exclude_none=True,
    )
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
