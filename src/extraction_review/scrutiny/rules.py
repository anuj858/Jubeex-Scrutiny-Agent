"""Loader for the SCI registry defect catalogue.

Defects match the API payload shape (S.No., Main Category, Defect/Objection,
Requirement, Where to Look, How to cure, rule, source). The JSON is validated
into Pydantic models at first use.
"""

from __future__ import annotations

import json
import logging
import os
import re
from functools import lru_cache
from pathlib import Path

from pydantic import BaseModel, ConfigDict, Field, field_validator

logger = logging.getLogger(__name__)

CATALOGUE_FILENAME = "sci_registry_defects.v1.json"
SCHEMA_FILENAME = "sci_registry_defects.schema.v1.json"

DEFAULT_ENABLED_DEFECTS = ("D003", "D004", "D005", "D006")

_DRIVE_FILE_MARKER = "/file/d/"


def _drive_file_id(url: str | None) -> str | None:
    if not url or _DRIVE_FILE_MARKER not in url:
        return None
    return url.split(_DRIVE_FILE_MARKER, 1)[1].split("/", 1)[0] or None


# Pipeline classify labels (SLP_CIVIL) vs API Main Category ("SLP (Civil)").
_CATEGORY_ALIASES = {
    "slp (civil)": "slp_civil",
    "slp civil": "slp_civil",
    "slp_civil": "slp_civil",
    "slp (criminal)": "slp_criminal",
    "slp criminal": "slp_criminal",
    "slp_criminal": "slp_criminal",
    "general/global": "global",
    "general": "global",
    "global": "global",
}


class _Strict(BaseModel):
    model_config = ConfigDict(extra="forbid")


class DefectCategory(_Strict):
    """One prompt shared by every defect in the same scrutiny area."""

    id: str
    label: str
    prompt: str


class CatalogueSource(_Strict):
    """Official document the Location/Source field can point at."""

    source_id: str
    title: str
    authority_type: str
    url: str | None = None
    issued_date: str | None = None
    effective_date: str | None = None
    checksum: str | None = None
    locators: dict[str, str] = Field(default_factory=dict)
    alternate_urls: list[str] = Field(default_factory=list)

    def urls(self) -> list[str]:
        return [u for u in [self.url, *self.alternate_urls] if u]


class Defect(_Strict):
    check_id: str
    serial_no: int
    main_category: str
    special_category: str | None = None
    category_id: str | None = None
    parent_check_id: str | None = None
    overlap_note: str | None = None
    defect: str
    requirement: str
    trigger_words: str | None = None
    where_to_look: list[str]
    how_to_cure: list[str]
    applicable_rule: str | None = None
    location_source: str

    @field_validator(
        "special_category",
        "trigger_words",
        "category_id",
        "parent_check_id",
        "overlap_note",
        "applicable_rule",
        mode="before",
    )
    @classmethod
    def _blank_to_none(cls, value: object) -> object:
        if isinstance(value, str) and not value.strip():
            return None
        return value

    @field_validator("serial_no", mode="before")
    @classmethod
    def _int_serial(cls, value: object) -> object:
        if isinstance(value, str) and value.strip().isdigit():
            return int(value.strip())
        return value

    @field_validator("where_to_look", "how_to_cure", mode="before")
    @classmethod
    def _as_string_list(cls, value: object) -> object:
        if isinstance(value, str):
            parts = [p.strip() for p in re.split(r"\n+", value) if p.strip()]
            return parts or [value.strip()]
        return value

    @property
    def title(self) -> str:
        return self.defect


class Catalogue(_Strict):
    catalogue_id: str
    schema_version: str
    catalogue_version: str
    jurisdiction: str
    disclaimer: str | None = None
    sources: list[CatalogueSource] = Field(default_factory=list)
    categories: list[DefectCategory] = Field(default_factory=list)
    defects: list[Defect] = Field(min_length=1)

    @property
    def defect_order(self) -> list[str]:
        return [d.check_id for d in self.defects]

    def defect_by_id(self, check_id: str) -> Defect | None:
        return next((d for d in self.defects if d.check_id == check_id), None)

    def defect(self, check_id: str) -> Defect:
        found = self.defect_by_id(check_id)
        if found is None:
            raise KeyError(f"Unknown defect {check_id}")
        return found

    def category_for(self, defect: Defect) -> DefectCategory | None:
        if defect.category_id:
            found = next((c for c in self.categories if c.id == defect.category_id), None)
            if found:
                return found
        if defect.special_category:
            key = defect.special_category.strip().lower()
            return next(
                (
                    c
                    for c in self.categories
                    if c.label.lower() == key or c.id.replace("_", " ") == key
                ),
                None,
            )
        return None

    def source(self, source_id: str) -> CatalogueSource | None:
        return next((s for s in self.sources if s.source_id == source_id), None)

    def sources_cited_by(self, defect: Defect) -> list[CatalogueSource]:
        text = defect.location_source
        cited: list[CatalogueSource] = []
        for source in self.sources:
            if source.source_id and source.source_id in text:
                cited.append(source)
                continue
            if any(url in text for url in source.urls()):
                cited.append(source)
                continue
            if any(
                (file_id := _drive_file_id(url)) and file_id in text
                for url in source.urls()
            ):
                cited.append(source)
        return cited


def _candidate_rule_dirs() -> list[Path]:
    override = os.getenv("SCRUTINY_RULES_DIR")
    dirs: list[Path] = [Path(override)] if override else []
    package_dir = Path(__file__).resolve().parent.parent
    dirs.append(package_dir / "_rules")
    repo_root = Path(__file__).resolve().parents[3]
    dirs.append(repo_root / "scrutiny_harness" / "rules")
    dirs.append(Path.cwd() / "scrutiny_harness" / "rules")
    return dirs


def _locate(filename: str) -> Path:
    tried: list[str] = []
    for directory in _candidate_rule_dirs():
        candidate = directory / filename
        tried.append(str(candidate))
        if candidate.is_file():
            return candidate
    raise FileNotFoundError(
        f"Could not locate {filename}. Looked in: {', '.join(tried)}. "
        f"Set SCRUTINY_RULES_DIR to override."
    )


def catalogue_path() -> Path:
    return _locate(CATALOGUE_FILENAME)


def catalogue_schema_path() -> Path:
    return _locate(SCHEMA_FILENAME)


@lru_cache(maxsize=1)
def get_catalogue() -> Catalogue:
    path = catalogue_path()
    with path.open(encoding="utf-8") as fh:
        raw = json.load(fh)
    catalogue = Catalogue.model_validate(raw)
    logger.info(
        "[Scrutiny] Loaded catalogue %s v%s (%s defects) from %s",
        catalogue.catalogue_id,
        catalogue.catalogue_version,
        len(catalogue.defects),
        path,
    )
    return catalogue


def normalize_filing_type(filing_type: str | None) -> str:
    """Map classify labels and API Main Category onto one key."""
    raw = (filing_type or "").strip().lower()
    raw = re.sub(r"\s+", " ", raw)
    return _CATEGORY_ALIASES.get(raw, raw.replace(" ", "_").replace("(", "").replace(")", ""))


def enabled_defect_ids() -> tuple[str, ...]:
    raw = os.getenv("SCRUTINY_DEFECTS")
    if not raw or not raw.strip():
        return DEFAULT_ENABLED_DEFECTS
    if raw.strip().lower() == "all":
        return tuple(get_catalogue().defect_order)
    ids = tuple(part.strip().upper() for part in raw.split(",") if part.strip())
    return ids or DEFAULT_ENABLED_DEFECTS


def _applies_to_filing(defect: Defect, normalized_filing_type: str) -> bool:
    if not normalized_filing_type:
        return False
    category = normalize_filing_type(defect.main_category)
    return category == "global" or category == normalized_filing_type


def order_parent_then_children(defects: list[Defect]) -> list[Defect]:
    """Run each parent immediately before its children.

    Consecutive OpenRouter calls then share a longer prompt prefix (same
    petition-type system prompt, then the same category block) so the child
    can reuse the cached parent prefix.
    """
    by_id = {d.check_id: d for d in defects}
    children: dict[str, list[Defect]] = {}
    for defect in defects:
        parent_id = defect.parent_check_id
        if parent_id and parent_id in by_id:
            children.setdefault(parent_id, []).append(defect)
    for kids in children.values():
        kids.sort(key=lambda d: (d.serial_no, d.check_id))

    roots = [
        d
        for d in defects
        if not d.parent_check_id or d.parent_check_id not in by_id
    ]
    roots.sort(key=lambda d: (d.serial_no, d.check_id))

    ordered: list[Defect] = []
    seen: set[str] = set()

    def emit(defect: Defect) -> None:
        if defect.check_id in seen:
            return
        seen.add(defect.check_id)
        ordered.append(defect)
        for child in children.get(defect.check_id, []):
            emit(child)

    for root in roots:
        emit(root)
    for defect in defects:
        emit(defect)
    return ordered


def defects_for_filing_type(filing_type: str | None) -> list[Defect]:
    catalogue = get_catalogue()
    normalized = normalize_filing_type(filing_type)
    allowed = set(enabled_defect_ids())

    selected = [
        defect
        for defect in catalogue.defects
        if defect.check_id in allowed and _applies_to_filing(defect, normalized)
    ]
    selected = order_parent_then_children(selected)

    unknown = allowed - {d.check_id for d in catalogue.defects}
    if unknown:
        logger.warning(
            "[Scrutiny] SCRUTINY_DEFECTS lists unknown check ids: %s",
            ", ".join(sorted(unknown)),
        )
    return selected


def is_supported_filing_type(filing_type: str | None) -> bool:
    return bool(defects_for_filing_type(filing_type))
