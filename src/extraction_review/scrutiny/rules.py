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
from urllib.parse import unquote, urlparse

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


def _url_filename(url: str | None) -> str | None:
    """Basename of a source URL when it is a real file, not a Drive viewer path."""
    if not url:
        return None
    name = unquote(urlparse(url).path).rstrip("/").rsplit("/", 1)[-1].strip()
    if not name or "." not in name:
        return None
    lowered = name.lower()
    if lowered in {"view", "preview", "edit", "usp=sharing"}:
        return None
    return name


def source_match_tokens(source: CatalogueSource) -> tuple[str, ...]:
    """Filenames and shorthands that mean this catalogue source."""
    tokens: list[str] = []
    seen: set[str] = set()

    def add(raw: str | None) -> None:
        text = (raw or "").strip()
        if not text:
            return
        key = text.casefold()
        if key in seen:
            return
        seen.add(key)
        tokens.append(text)

    for alias in source.filename_aliases:
        add(alias)
    for url in source.urls():
        add(_url_filename(url))
    return tuple(tokens)


def rewrite_location_source(text: str, sources: list[CatalogueSource]) -> str:
    """Replace opaque PDF names with catalogue source_ids. Longest alias first."""
    raw = text or ""
    replacements: list[tuple[str, str]] = []
    for source in sources:
        for token in source_match_tokens(source):
            if token.casefold() == source.source_id.casefold():
                continue
            replacements.append((token, source.source_id))
    replacements.sort(key=lambda item: len(item[0]), reverse=True)
    rewritten = raw
    for token, source_id in replacements:
        rewritten = re.sub(re.escape(token), source_id, rewritten, flags=re.IGNORECASE)
    return rewritten


# Pipeline classify labels (SLP_CIVIL) vs catalogue Main Category ("SLP (Civil)").
# Family keys (slp, transfer_petition) apply to both civil and criminal of that family.
_CATEGORY_ALIASES = {
    "slp": "slp",
    "special leave petition": "slp",
    "slp (civil)": "slp_civil",
    "slp civil": "slp_civil",
    "slp_civil": "slp_civil",
    "special leave petition (civil)": "slp_civil",
    "special leave petition civil": "slp_civil",
    "slp (criminal)": "slp_criminal",
    "slp criminal": "slp_criminal",
    "slp_criminal": "slp_criminal",
    "special leave petition (criminal)": "slp_criminal",
    "special leave petition criminal": "slp_criminal",
    "transfer petition": "transfer_petition",
    "transfer_petition": "transfer_petition",
    "transfer petition (civil)": "transfer_petition_civil",
    "transfer petition civil": "transfer_petition_civil",
    "transfer_petition_civil": "transfer_petition_civil",
    "tp (civil)": "transfer_petition_civil",
    "tp civil": "transfer_petition_civil",
    "tp_civil": "transfer_petition_civil",
    "transfer petition (criminal)": "transfer_petition_criminal",
    "transfer petition criminal": "transfer_petition_criminal",
    "transfer_petition_criminal": "transfer_petition_criminal",
    "tp (criminal)": "transfer_petition_criminal",
    "tp criminal": "transfer_petition_criminal",
    "tp_criminal": "transfer_petition_criminal",
    "writ petition": "writ_petition",
    "writ_petition": "writ_petition",
    "writ petition (civil)": "writ_petition_civil",
    "writ petition civil": "writ_petition_civil",
    "writ_petition_civil": "writ_petition_civil",
    "writ petition (criminal)": "writ_petition_criminal",
    "writ petition criminal": "writ_petition_criminal",
    "writ_petition_criminal": "writ_petition_criminal",
    "arbitration petition": "arbitration_petition",
    "arbitration_petition": "arbitration_petition",
    "civil appeal": "civil_appeal",
    "civil_appeal": "civil_appeal",
    "criminal appeal": "criminal_appeal",
    "criminal_appeal": "criminal_appeal",
    "review petition": "review_petition",
    "review_petition": "review_petition",
    "review petition (civil)": "review_petition_civil",
    "review petition civil": "review_petition_civil",
    "review_petition_civil": "review_petition_civil",
    "review petition (criminal)": "review_petition_criminal",
    "review petition criminal": "review_petition_criminal",
    "review_petition_criminal": "review_petition_criminal",
    "contempt petition": "contempt_petition",
    "contempt_petition": "contempt_petition",
    "contempt petition (civil)": "contempt_petition_civil",
    "contempt petition civil": "contempt_petition_civil",
    "contempt_petition_civil": "contempt_petition_civil",
    "contempt petition (criminal)": "contempt_petition_criminal",
    "contempt petition criminal": "contempt_petition_criminal",
    "contempt_petition_criminal": "contempt_petition_criminal",
    "election petition": "election_petition",
    "election_petition": "election_petition",
    "election petition (civil)": "election_petition_civil",
    "election petition civil": "election_petition_civil",
    "election_petition_civil": "election_petition_civil",
    "curative petition": "curative_petition",
    "curative_petition": "curative_petition",
    "curative petition (civil)": "curative_petition_civil",
    "curative petition civil": "curative_petition_civil",
    "curative_petition_civil": "curative_petition_civil",
    "curative petition (criminal)": "curative_petition_criminal",
    "curative petition criminal": "curative_petition_criminal",
    "curative_petition_criminal": "curative_petition_criminal",
    "original suit": "original_suit",
    "original_suit": "original_suit",
    "original suit (civil)": "original_suit_civil",
    "original suit civil": "original_suit_civil",
    "original_suit_civil": "original_suit_civil",
    "general/global": "global",
    "global/general": "global",
    "global / general": "global",
    "general / global": "global",
    "general": "global",
    "global": "global",
}

_SIDE_SUFFIXES = ("_civil", "_criminal")


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
    filename_aliases: list[str] = Field(default_factory=list)

    def urls(self) -> list[str]:
        return [u for u in [self.url, *self.alternate_urls] if u]


class Defect(_Strict):
    check_id: str
    serial_no: int | str
    main_category: str
    special_category: str | None = None
    category_id: str | None = None
    parent_check_id: str | None = None
    overlap_note: str | None = None
    defect: str
    requirement: str
    trigger_words: str | None = None
    where_to_look: list[str]
    inspect_parts: list[str] = Field(default_factory=list)
    context_parts: list[str] = Field(default_factory=list)
    exclude_parts: list[str] | None = None
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

    @field_validator("main_category", mode="before")
    @classmethod
    def _main_category_as_string(cls, value: object) -> object:
        if isinstance(value, (list, tuple)):
            parts = [str(item).strip() for item in value if str(item).strip()]
            return ", ".join(parts)
        return value

    @property
    def main_categories(self) -> tuple[str, ...]:
        return split_main_categories(self.main_category)

    @field_validator("serial_no", mode="before")
    @classmethod
    def _serial_no(cls, value: object) -> object:
        if isinstance(value, str):
            text = value.strip().upper()
            if text.isdigit():
                return int(text)
            if re.fullmatch(r"\d+[A-Z]", text):
                return text
        return value

    @field_validator(
        "where_to_look",
        "how_to_cure",
        "inspect_parts",
        "context_parts",
        "exclude_parts",
        mode="before",
    )
    @classmethod
    def _as_string_list(cls, value: object) -> object:
        if value is None:
            return value
        if isinstance(value, str):
            parts = [p.strip() for p in re.split(r"\n+", value) if p.strip()]
            return parts or [value.strip()]
        return value

    @property
    def title(self) -> str:
        return self.defect


def split_main_categories(main_category: str | list[str] | None) -> tuple[str, ...]:
    """Split a catalogue Main Category into one or more petition-type labels.

    Accepts a JSON list, or a string split on commas and on slashes that follow
    a closing parenthesis (e.g. "SLP (Civil)/SLP (Criminal)"). Leaves
    "General/Global" as a single label.
    """
    if main_category is None:
        return ()
    if isinstance(main_category, (list, tuple)):
        parts = [str(item).strip() for item in main_category]
    else:
        cleaned = re.sub(
            r"\s*-\s*leave it for the user to select\s*$",
            "",
            str(main_category).strip(),
            flags=re.IGNORECASE,
        )
        parts = [
            part.strip()
            for part in re.split(r"\s*,\s*|(?<=\))\s*/\s*", cleaned)
            if part.strip()
        ]
    return tuple(part for part in parts if part)


class Catalogue(_Strict):
    catalogue_id: str
    schema_version: str
    catalogue_version: str
    jurisdiction: str
    disclaimer: str | None = None
    sources: list[CatalogueSource] = Field(default_factory=list)
    categories: list[DefectCategory] = Field(default_factory=list)
    defects: list[Defect] = Field(default_factory=list)

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
            found = next(
                (c for c in self.categories if c.id == defect.category_id), None
            )
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
        folded = text.casefold()
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
                continue
            if any(token.casefold() in folded for token in source_match_tokens(source)):
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
    if not raw:
        return ""
    if raw in _CATEGORY_ALIASES:
        return _CATEGORY_ALIASES[raw]
    # "SLP(Civil)" → "slp (civil)" so the spaced aliases hit.
    spaced = re.sub(r"\s*\(\s*", " (", raw)
    spaced = re.sub(r"\s*\)\s*", ")", spaced).strip()
    if spaced in _CATEGORY_ALIASES:
        return _CATEGORY_ALIASES[spaced]
    collapsed = (
        spaced.replace(" ", "_").replace("-", "_").replace("(", "").replace(")", "")
    )
    collapsed = re.sub(r"_+", "_", collapsed).strip("_")
    return _CATEGORY_ALIASES.get(collapsed, collapsed)


def categories_for_filing_type(filing_type: str | None) -> frozenset[str]:
    """Normalized main_category keys that run for this petition type.

    SLP_CIVIL → global + slp + slp_civil
    SLP_CRIMINAL → global + slp + slp_criminal

    Later types follow the same family pattern: Transfer Petition (Civil)
    runs Global/General + Transfer Petition + Transfer Petition (Civil).
    """
    normalized = normalize_filing_type(filing_type)
    if not normalized:
        return frozenset()
    keys = {"global", normalized}
    for suffix in _SIDE_SUFFIXES:
        if normalized.endswith(suffix):
            parent = normalized[: -len(suffix)]
            if parent:
                keys.add(parent)
            break
    return frozenset(keys)


def serial_sort_key(serial_no: int | str | None) -> tuple[int, str]:
    """Sort 92 before 96A before 96B; letter suffixes follow the number."""
    text = str(serial_no or "").strip().upper()
    match = re.fullmatch(r"(\d+)([A-Z]*)", text)
    if not match:
        return (10**9, text)
    return (int(match.group(1)), match.group(2))


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
    applicable = categories_for_filing_type(normalized_filing_type)
    for category in defect.main_categories:
        key = normalize_filing_type(category)
        if key and key in applicable:
            return True
    return False


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
        kids.sort(key=lambda d: (serial_sort_key(d.serial_no), d.check_id))

    roots = [
        d for d in defects if not d.parent_check_id or d.parent_check_id not in by_id
    ]
    roots.sort(key=lambda d: (serial_sort_key(d.serial_no), d.check_id))

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
    logger.info(
        "[Scrutiny] Petition type %s selected %s defect(s) from main_category {%s}",
        filing_type or "(missing)",
        len(selected),
        ", ".join(sorted(categories_for_filing_type(filing_type))),
    )
    return selected


def is_supported_filing_type(filing_type: str | None) -> bool:
    return bool(defects_for_filing_type(filing_type))
