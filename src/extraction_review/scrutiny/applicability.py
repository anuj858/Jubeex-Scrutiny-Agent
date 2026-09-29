"""Explicit petition-type applicability for a catalogue defect.

Today a defect says which filings it runs on with free-text ``main_category``
and ``special_category`` strings. ``defects_for_filing_type`` interprets those
strings (family labels, Global/General, Refiling, Miscellaneous Application).

The database stores the result of that interpretation instead of the strings:

* ``applies_to_all`` — Global/General defects that run on every petition type
* ``filing_types`` — normalized petition types (``slp_civil``, …)
* ``special_key`` + ``special_mode`` — ``required`` (runs only when that
  special is selected) or ``overlay`` (runs on its petition types always, and
  on other types only when that special is selected)

``derive_applicability`` uses the same helpers as the live selector, so the
seed matches current behaviour. ``defect_applies`` is what both the file
loader and the database loader select with.
"""

from __future__ import annotations

from dataclasses import dataclass

from extraction_review.config import JUBEEX_FILING_TYPES

from .rules import (
    Defect,
    _applies_to_filing,
    _overlay_special_key,
    allowed_special_keys,
    normalize_filing_type,
    normalize_special_category,
)

SPECIAL_MODE_REQUIRED = "required"
SPECIAL_MODE_OVERLAY = "overlay"

# Normalized filing type (slp_civil) -> petition_types.code in the JubeeX catalog.
FILING_TYPE_TO_PETITION_CODE: dict[str, str] = {
    "slp_civil": "slp-civil",
    "slp_criminal": "slp-criminal",
    "civil_appeal": "civil-appeal",
    "criminal_appeal": "criminal-appeal",
    "writ_petition_civil": "writ-petition",
    "writ_petition_criminal": "writ-petition-criminal",
    "transfer_petition_civil": "transfer-petition",
    "transfer_petition_criminal": "transfer-petition-criminal",
    "review_petition_civil": "review-petition",
    "review_petition_criminal": "review-petition-criminal",
    "original_suit_civil": "original-suit",
    "contempt_petition_civil": "contempt-petition-civil",
    "contempt_petition_criminal": "contempt-petition-criminal",
    "election_petition_civil": "election-petition-civil",
    "arbitration_petition": "arbitration-petition",
    "curative_petition_civil": "curative-petition",
    "curative_petition_criminal": "curative-petition-criminal",
    "miscellaneous_application": "miscellaneous-application",
}

# Petition codes that are not in the catalog yet and must be inserted by the seed.
PETITION_TYPES_TO_CREATE: frozenset[str] = frozenset({"election-petition-civil"})

# inspect_parts / context_parts label -> document_types.code.
# split_label is the string the Parse splitter emits; it is not always the
# catalog display name (for example "Main Petition" vs "Petition").
PART_DOCUMENT_TYPES: dict[str, dict[str, str]] = {
    "Advocate's Checklist": {
        "code": "advocate-s-checklist",
        "name": "Advocate's Checklist",
        "split_label": "Advocate's Checklist",
    },
    "Cover Page": {
        "code": "cover-page",
        "name": "Cover Page",
        "split_label": "Cover Page",
    },
    "Record of Proceedings": {
        "code": "record-of-proceedings",
        "name": "Record of Proceedings",
        "split_label": "Record of Proceedings",
    },
    "AOR's Certificate": {
        "code": "aor-s-certificate",
        "name": "AOR's Certificate",
        "split_label": "AOR's Certificate",
    },
    "Index": {"code": "index", "name": "Index", "split_label": "Index"},
    "Office Report on Limitation": {
        "code": "office-report-on-limitation",
        "name": "Office Report on Limitation",
        "split_label": "Office Report on Limitation",
    },
    "Listing Proforma": {
        "code": "listing-proforma",
        "name": "Listing Proforma",
        "split_label": "Listing Proforma",
    },
    "Synopsis": {"code": "synopsis", "name": "Synopsis", "split_label": "Synopsis"},
    "List of Dates & Events": {
        "code": "list-of-dates",
        "name": "List of Dates",
        "split_label": "List of Dates & Events",
    },
    "Main Petition": {
        "code": "petition",
        "name": "Petition",
        "split_label": "Main Petition",
    },
    "Affidavit": {
        "code": "affidavit",
        "name": "Affidavit",
        "split_label": "Affidavit",
    },
    "Annexures": {
        "code": "annexures",
        "name": "Annexures",
        "split_label": "Annexures",
    },
    "Appendix": {"code": "appendix", "name": "Appendix", "split_label": "Appendix"},
    "Memo of Parties": {
        "code": "memo-of-parties",
        "name": "Memo of Parties",
        "split_label": "Memo of Parties",
    },
    "Memo of Appearance": {
        "code": "memo-of-appearance",
        "name": "Memo of Appearance",
        "split_label": "Memo of Appearance",
    },
    "Impugned Order": {
        "code": "impugned-order",
        "name": "Impugned Order",
        "split_label": "Impugned Order",
    },
    "Vakalatnama": {
        "code": "vakalatnama",
        "name": "Vakalatnama",
        "split_label": "Vakalatnama",
    },
    "Filing Memo": {
        "code": "filing-memo",
        "name": "Filing Memo",
        "split_label": "Filing Memo",
    },
    "Court Fees": {
        "code": "proof-of-court-fees",
        "name": "Proof of Court Fees",
        "split_label": "Court Fees",
    },
    "Application": {
        "code": "applications",
        "name": "Application",
        "split_label": "Application",
        "create": "true",
        "document_group": "supporting",
    },
}


def petition_code_for_filing_type(filing_type: str) -> str:
    normalized = normalize_filing_type(filing_type)
    code = FILING_TYPE_TO_PETITION_CODE.get(normalized)
    if code is None:
        raise KeyError(f"No petition_types.code for filing type {filing_type!r}")
    return code


# Catalogue labels that already exist in the filing catalog under another spelling.
SPECIAL_LABEL_TO_CATALOG_NAME = {
    "motor vehical act": "Motor Vehicle Act",
}


def catalog_special_name(label: str) -> str:
    text = (label or "").strip()
    return SPECIAL_LABEL_TO_CATALOG_NAME.get(text.casefold(), text)


def category_code(name: str) -> str:
    """Same slug as the JubeeX filing-catalog special-category seeder."""
    import re

    slug = re.sub(r"[^a-z0-9]+", "-", (name or "").lower().strip()).strip("-")
    return slug[:128]


@dataclass(frozen=True)
class ExplicitApplicability:
    applies_to_all: bool
    filing_types: frozenset[str]
    special_key: str
    special_mode: str | None

    def petition_type_codes(self) -> list[str]:
        codes = [petition_code_for_filing_type(filing) for filing in self.filing_types]
        return sorted(set(codes))


def _all_filing_keys() -> frozenset[str]:
    return frozenset(normalize_filing_type(filing) for filing in JUBEEX_FILING_TYPES)


def _matching_filing_keys(defect: Defect) -> frozenset[str]:
    return frozenset(
        normalize_filing_type(filing)
        for filing in JUBEEX_FILING_TYPES
        if _applies_to_filing(defect, normalize_filing_type(filing))
    )


def derive_applicability(defect: Defect) -> ExplicitApplicability:
    """Resolve one defect the way ``defects_for_filing_type`` will."""
    if getattr(defect, "use_explicit_applicability", False):
        return ExplicitApplicability(
            applies_to_all=bool(defect.applies_to_all_petition_types),
            filing_types=frozenset(
                normalize_filing_type(filing)
                for filing in defect.agent_filing_types
                if normalize_filing_type(filing)
            ),
            special_key=(
                defect.special_category_key
                or normalize_special_category(defect.special_category)
            ),
            special_mode=defect.special_category_mode,
        )

    tagged = normalize_special_category(defect.special_category)
    overlay = _overlay_special_key(defect)
    matching = _matching_filing_keys(defect)

    if tagged:
        allowed_types = frozenset(
            filing for filing in matching if tagged in allowed_special_keys(filing)
        )
        return ExplicitApplicability(
            applies_to_all=False,
            filing_types=allowed_types,
            special_key=tagged,
            special_mode=SPECIAL_MODE_REQUIRED,
        )

    if overlay == "refiling_defect":
        allowed_types = frozenset(
            normalize_filing_type(filing)
            for filing in JUBEEX_FILING_TYPES
            if "refiling_defect" in allowed_special_keys(filing)
        )
        return ExplicitApplicability(
            applies_to_all=False,
            filing_types=allowed_types,
            special_key="refiling_defect",
            special_mode=SPECIAL_MODE_REQUIRED,
        )

    if overlay == "miscellaneous_application":
        return ExplicitApplicability(
            applies_to_all=False,
            filing_types=matching,
            special_key="miscellaneous_application",
            special_mode=SPECIAL_MODE_OVERLAY,
        )

    return ExplicitApplicability(
        applies_to_all=matching == _all_filing_keys() and bool(matching),
        filing_types=matching,
        special_key="",
        special_mode=None,
    )


def defect_applies(
    applicability: ExplicitApplicability,
    filing_type: str | None,
    special_category: str | None,
    allowed_specials: frozenset[str],
) -> bool:
    """Whether this defect runs for one petition type and special category."""
    normalized = normalize_filing_type(filing_type)
    if not normalized:
        return False
    requested = normalize_special_category(special_category)
    filing_ok = applicability.applies_to_all or normalized in applicability.filing_types

    if applicability.special_mode == SPECIAL_MODE_REQUIRED:
        return bool(
            filing_ok
            and requested
            and requested == applicability.special_key
            and requested in allowed_specials
        )

    if applicability.special_mode == SPECIAL_MODE_OVERLAY:
        if filing_ok:
            return True
        return bool(
            requested
            and requested == applicability.special_key
            and requested in allowed_specials
        )

    return filing_ok
