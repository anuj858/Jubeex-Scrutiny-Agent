"""Build the normalized defect-catalogue seed from the JSON catalogue.

The seed is what the JubeeX backend loads into ``hub_defects``. Applicability
is already resolved (petition type codes, special-category mode) so the
database does not have to interpret ``main_category`` strings.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from extraction_review.config import JUBEEX_FILING_TYPES

from .applicability import (
    PART_DOCUMENT_TYPES,
    PETITION_TYPES_TO_CREATE,
    catalog_special_name,
    category_code,
    defect_applies,
    derive_applicability,
    petition_code_for_filing_type,
)
from .rules import (
    Defect,
    allowed_special_keys,
    get_case_types,
    get_catalogue,
    normalize_filing_type,
    normalize_special_category,
)

_PETITION_NAMES: dict[str, str] = {
    "slp_civil": "SLP (Civil)",
    "slp_criminal": "SLP (Criminal)",
    "civil_appeal": "Civil Appeal",
    "criminal_appeal": "Criminal Appeal",
    "writ_petition_civil": "Writ Petition (Civil)",
    "writ_petition_criminal": "Writ Petition (Criminal)",
    "transfer_petition_civil": "Transfer Petition (Civil)",
    "transfer_petition_criminal": "Transfer Petition (Criminal)",
    "review_petition_civil": "Review Petition (Civil)",
    "review_petition_criminal": "Review Petition (Criminal)",
    "original_suit_civil": "Original Suit (Civil)",
    "contempt_petition_civil": "Contempt Petition (Civil)",
    "contempt_petition_criminal": "Contempt Petition (Criminal)",
    "election_petition_civil": "Election Petition (Civil)",
    "arbitration_petition": "Arbitration Petition",
    "curative_petition_civil": "Curative Petition (Civil)",
    "curative_petition_criminal": "Curative Petition (Criminal)",
    "miscellaneous_application": "Miscellaneous Application",
}


def split_trigger_words(value: str | None) -> list[str]:
    parts = [part.strip() for part in (value or "").split(",")]
    seen: set[str] = set()
    words: list[str] = []
    for part in parts:
        if not part:
            continue
        key = part.casefold()
        if key in seen:
            continue
        seen.add(key)
        words.append(part)
    return words


def _part_codes(labels: list[str] | None, *, check_id: str, role: str) -> list[str]:
    codes: list[str] = []
    for label in labels or []:
        spec = PART_DOCUMENT_TYPES.get(label)
        if spec is None:
            raise ValueError(
                f"{check_id} {role} part {label!r} has no document type mapping"
            )
        codes.append(spec["code"])
    return codes


def _special_label(defect: Defect) -> str | None:
    """Catalogue wording for a tagged or overlay special, when this defect has one."""
    applicability = derive_applicability(defect)
    if not applicability.special_key:
        return None
    label = defect.special_category
    if applicability.special_key == "refiling_defect" and not label:
        label = "Re-filing defect"
    if applicability.special_key == "miscellaneous_application" and not label:
        label = "Miscellaneous Application"
    if not label:
        raise ValueError(
            f"{defect.check_id} special {applicability.special_key} has no label"
        )
    return label


def _defect_row(defect: Defect, source_codes: list[str]) -> dict[str, Any]:
    applicability = derive_applicability(defect)
    label = _special_label(defect)
    special_code = category_code(catalog_special_name(label)) if label else None
    return {
        "code": defect.check_id,
        "title": defect.defect,
        "requirement": defect.requirement,
        "rule_number": defect.applicable_rule,
        "citation": defect.location_text or defect.location_source,
        "main_category": defect.main_category,
        "where_to_look": list(defect.where_to_look),
        "how_to_cure": list(defect.how_to_cure),
        "trigger_words": split_trigger_words(defect.trigger_words),
        "notes": defect.notes,
        "ivan_comment": defect.ivan_comment,
        "notes_2": defect.notes_2,
        "overlap_note": defect.overlap_note,
        "applies_to_all_petition_types": applicability.applies_to_all,
        "petition_type_codes": applicability.petition_type_codes(),
        "special_category_code": special_code,
        "special_category_mode": applicability.special_mode,
        "parent_code": defect.parent_check_id,
        "is_enabled": True,
        "inspect_part_codes": _part_codes(
            defect.inspect_parts, check_id=defect.check_id, role="inspect"
        ),
        "context_part_codes": _part_codes(
            defect.context_parts, check_id=defect.check_id, role="context"
        ),
        "exclude_part_codes": _part_codes(
            defect.exclude_parts, check_id=defect.check_id, role="exclude"
        ),
        "source_codes": source_codes,
        "status": "reviewed_by_legal",
        "severity": "p2",
    }


def build_seed() -> dict[str, Any]:
    catalogue = get_catalogue()
    case_types = get_case_types()

    petition_types: list[dict[str, Any]] = []
    for filing_type in JUBEEX_FILING_TYPES:
        normalized = normalize_filing_type(filing_type)
        code = petition_code_for_filing_type(filing_type)
        spec = case_types.get(normalized) or {}
        petition_types.append(
            {
                "code": code,
                "name": str(spec.get("label") or _PETITION_NAMES[normalized]),
                "agent_filing_type": filing_type,
                "create": code in PETITION_TYPES_TO_CREATE,
            }
        )

    special_names: list[str] = []
    seen_specials: set[str] = set()
    links: list[dict[str, str]] = []
    for filing_type in JUBEEX_FILING_TYPES:
        normalized = normalize_filing_type(filing_type)
        spec = case_types.get(normalized) or {}
        petition_code = petition_code_for_filing_type(filing_type)
        for label in spec.get("special_categories") or []:
            name = str(label).strip()
            if not name:
                continue
            if normalize_special_category(name) == "":
                raise ValueError(f"Case type special {name!r} does not normalize")
            catalog_name = catalog_special_name(name)
            key = catalog_name.casefold()
            if key not in seen_specials:
                seen_specials.add(key)
                special_names.append(catalog_name)
            links.append(
                {
                    "petition_type_code": petition_code,
                    "special_category_code": category_code(catalog_name),
                }
            )

    for defect in catalogue.defects:
        label = _special_label(defect)
        if not label:
            continue
        catalog_name = catalog_special_name(label)
        key = catalog_name.casefold()
        if key not in seen_specials:
            seen_specials.add(key)
            special_names.append(catalog_name)

    document_types: list[dict[str, Any]] = []
    seen_docs: set[str] = set()
    for spec in PART_DOCUMENT_TYPES.values():
        if spec["code"] in seen_docs:
            continue
        seen_docs.add(spec["code"])
        document_types.append(
            {
                "code": spec["code"],
                "name": spec["name"],
                "split_label": spec["split_label"],
                "create": spec.get("create") == "true",
                "document_group": spec.get("document_group") or "supporting",
            }
        )

    sources = [
        {
            "code": source.source_id,
            "title": source.title,
            "authority_type": source.authority_type,
            "url": source.url,
            "alternate_urls": list(source.alternate_urls),
            "filename_aliases": list(source.filename_aliases),
            "issued_date": source.issued_date,
            "effective_date": source.effective_date,
            "checksum": source.checksum,
            "locators": dict(source.locators),
        }
        for source in catalogue.sources
    ]

    defects = [
        _defect_row(
            defect,
            [source.source_id for source in catalogue.sources_cited_by(defect)],
        )
        for defect in catalogue.defects
    ]

    return {
        "catalogue_id": catalogue.catalogue_id,
        "schema_version": catalogue.schema_version,
        "catalogue_version": catalogue.catalogue_version,
        "jurisdiction": catalogue.jurisdiction,
        "disclaimer": catalogue.disclaimer,
        "court_code": "sci",
        "petition_types": petition_types,
        "special_categories": [
            {"code": category_code(name), "name": name} for name in special_names
        ],
        "petition_special_links": links,
        "document_types": document_types,
        "sources": sources,
        "defects": defects,
    }


def write_seed(path: Path) -> dict[str, Any]:
    payload = build_seed()
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(payload, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    return payload


def legacy_selected_ids(
    filing_type: str | None,
    special_category: str | None,
) -> list[str]:
    """Check ids the current string-based selector would run.

    Kept beside the new selector so a test can prove they match after
    ``defects_for_filing_type`` starts using explicit applicability.
    """
    from .rules import (
        _applies_to_filing,
        _applies_to_special,
        _overlay_special_key,
        enabled_defect_ids,
        order_parent_then_children,
    )

    catalogue = get_catalogue()
    normalized = normalize_filing_type(filing_type)
    allowed = set(enabled_defect_ids())
    overlay_requested = normalize_special_category(special_category)
    selected = [
        defect
        for defect in catalogue.defects
        if defect.check_id in allowed
        and (
            _applies_to_filing(defect, normalized)
            or (overlay_requested and _overlay_special_key(defect) == overlay_requested)
        )
        and _applies_to_special(defect, filing_type, special_category)
    ]
    ordered = order_parent_then_children(selected)
    return [defect.check_id for defect in ordered]


def explicit_selected_ids(
    filing_type: str | None,
    special_category: str | None,
) -> list[str]:
    from .rules import enabled_defect_ids, order_parent_then_children

    catalogue = get_catalogue()
    allowed = set(enabled_defect_ids())
    allowed_specials = allowed_special_keys(filing_type)
    selected = [
        defect
        for defect in catalogue.defects
        if defect.check_id in allowed
        and defect.is_enabled
        and defect_applies(
            derive_applicability(defect),
            filing_type,
            special_category,
            allowed_specials,
        )
    ]
    ordered = order_parent_then_children(selected)
    return [defect.check_id for defect in ordered]
