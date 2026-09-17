"""Attach stored ink marks onto existing visual catalogue defects only."""

from __future__ import annotations

import re
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, replace
from typing import Any

from pydantic import ValidationError

from ..document_parts import (
    ANNEXURE_FAMILY,
    APPLICATION_FAMILY,
    MAIN_PETITION_PART,
    catalogue_inspect_parts,
    family_split_name,
    parts_named_in_text,
)
from ..scrutiny.rules import Defect
from ..scrutiny.schema import BoundingBox, VisualLocalization
from .pages import pages_for_part, policy_selector
from .schema import VisualMark
from .store import coerce_visual_index

_SIGNATURE = re.compile(
    r"\b(signatur\w*|signed|wet[-\s]?ink|digital\s+sign|thumb\s+impression)\b",
    re.IGNORECASE,
)
_NOTARY = re.compile(
    r"\b(notary|notarial|notar\w*|oath\s+commissioner|before\s+me|attestation)\b",
    re.IGNORECASE,
)
_NOTARY_TEXT = re.compile(
    r"\b(notary|notarial|oath\s+commissioner|before\s+me|reg\.?\s*no)\b",
    re.IGNORECASE,
)
_EXECUTANT = re.compile(
    r"\b(thumb|seriatim|properly executed|executant)\b",
    re.IGNORECASE,
)
_ADVOCATE = re.compile(
    r"\b(advocate|aor|endors|certif|identified|digital\s+sign|accepted)\b",
    re.IGNORECASE,
)
_CERTIFIED_COPY = re.compile(r"certified\s+copy", re.IGNORECASE)

ADVOCATE_ROLES = frozenset({"advocate", "unknown"})
EXECUTANT_ROLES = frozenset({"petitioner", "respondent", "deponent", "unknown"})
AOR_SIGNATURE_PARTS = frozenset(
    {
        MAIN_PETITION_PART,
        "Advocate's Checklist",
        "Listing Proforma",
        "AOR's Certificate",
        "Application",
        "Filing Memo",
        "Memo of Parties",
        "Memo of Appearance",
        ANNEXURE_FAMILY,
    }
)
NOTARY_PARTS = frozenset({"Affidavit", "Vakalatnama"})
NOTARY_MARK_TYPES = (
    "notary_seal",
    "ordinary_seal_or_stamp",
    "government_stamp",
)


@dataclass(frozen=True)
class VisualNeed:
    document_type: str
    marking_types: tuple[str, ...]
    signature_roles: tuple[str, ...] | None
    page_selector: str
    marking_type: str


def _defect_text(defect: Defect) -> str:
    where = " ".join(defect.where_to_look or [])
    return " ".join(part for part in (defect.defect, defect.requirement, where) if part)


def _in_policy(name: str) -> bool:
    return policy_selector(name) is not None


def _named_parts(defect: Defect) -> list[str]:
    names = [name for name in catalogue_inspect_parts(defect) if _in_policy(name)]
    for step in defect.where_to_look or []:
        for name in parts_named_in_text(step):
            if _in_policy(name) and name not in names:
                names.append(name)
    text = _defect_text(defect)
    lowered = text.casefold()
    extras: list[str] = []
    if re.search(r"\bannexure", lowered) and not any(
        family_split_name(name) == ANNEXURE_FAMILY for name in names
    ):
        extras.append(ANNEXURE_FAMILY)
    if (
        re.search(r"\bcertificate\b", lowered)
        and not _CERTIFIED_COPY.search(text)
        and "AOR's Certificate" not in names
    ):
        extras.append("AOR's Certificate")
    if not names:
        if "affidavit" in lowered:
            extras.append("Affidavit")
        if "vakalatnama" in lowered:
            extras.append("Vakalatnama")
        if "memo of appearance" in lowered:
            extras.append("Memo of Appearance")
        if "checklist" in lowered or "check list" in lowered:
            extras.append("Advocate's Checklist")
        if "listing proforma" in lowered or "proforma for first listing" in lowered:
            extras.append("Listing Proforma")
        if "filing memo" in lowered:
            extras.append("Filing Memo")
        if "memo of parties" in lowered:
            extras.append("Memo of Parties")
        if "petition" in lowered or "form 28" in lowered or "form no. 28" in lowered:
            extras.append(MAIN_PETITION_PART)
    for name in extras:
        if name not in names and _in_policy(name):
            names.append(name)
    return names


def _is_aor_signature_part(name: str) -> bool:
    return family_split_name(name) in AOR_SIGNATURE_PARTS or name in AOR_SIGNATURE_PARTS


def visual_needs_for_defect(defect: Defect) -> list[VisualNeed]:
    """Which stored marks this catalogue defect should receive.

    Returns an empty list for text-only defects. Does not invent new defects.
    """
    text = _defect_text(defect)
    wants_signature = bool(_SIGNATURE.search(text))
    wants_notary = bool(_NOTARY.search(text))
    if not (wants_signature or wants_notary):
        return []

    parts = _named_parts(defect)
    needs: list[VisualNeed] = []
    seen: set[tuple[str, str]] = set()

    def add(need: VisualNeed) -> None:
        key = (need.document_type, need.marking_type)
        if key in seen:
            return
        seen.add(key)
        needs.append(need)

    def add_advocate(name: str) -> None:
        selector = policy_selector(name) or "last"
        add(
            VisualNeed(
                document_type=name,
                marking_types=("signature_like_mark",),
                signature_roles=tuple(ADVOCATE_ROLES),
                page_selector=selector,
                marking_type="advocate_on_record_signature",
            )
        )

    def add_executant(name: str) -> None:
        selector = policy_selector(name) or "last"
        add(
            VisualNeed(
                document_type=name,
                marking_types=("signature_like_mark",),
                signature_roles=tuple(EXECUTANT_ROLES),
                page_selector=selector,
                marking_type="executant_signature",
            )
        )

    def add_notary(name: str) -> None:
        selector = policy_selector(name) or "last"
        add(
            VisualNeed(
                document_type=name,
                marking_types=NOTARY_MARK_TYPES,
                signature_roles=None,
                page_selector=selector,
                marking_type="notary_seal",
            )
        )

    formality_parts = [name for name in parts if _in_policy(name)]
    if wants_signature:
        for name in formality_parts:
            if name == "Affidavit":
                continue
            if name == "Vakalatnama":
                executant = bool(_EXECUTANT.search(text))
                advocate = bool(_ADVOCATE.search(text))
                if not executant and not advocate:
                    advocate = True
                if advocate:
                    add_advocate(name)
                if executant:
                    add_executant(name)
                continue
            if _is_aor_signature_part(name):
                add_advocate(name)
    if wants_notary:
        notary_parts = [name for name in formality_parts if name in NOTARY_PARTS]
        if not notary_parts:
            lowered = text.casefold()
            if "affidavit" in lowered:
                notary_parts.append("Affidavit")
            elif "vakalatnama" in lowered:
                notary_parts.append("Vakalatnama")
        for name in notary_parts:
            add_notary(name)
    return needs


def _marks_from_index(visual_index: Mapping[str, Any] | None) -> list[VisualMark]:
    payload = coerce_visual_index(visual_index)
    marks: list[VisualMark] = []
    for item in payload.get("marks") or []:
        try:
            marks.append(VisualMark.model_validate(item))
        except ValidationError:
            continue
    return marks


def _consider_family_name(found: list[str], name: str, family: str) -> None:
    cleaned = str(name).strip()
    if not cleaned:
        return
    if family_split_name(cleaned) != family and cleaned != family:
        return
    if cleaned not in found:
        found.append(cleaned)


def _family_member_names(
    document_type: str,
    *,
    page_parts: Mapping[int, Sequence[str]],
    visual_index: Mapping[str, Any] | None,
    record: Mapping[str, Any] | None,
) -> list[str]:
    family = family_split_name(document_type)
    if family not in {ANNEXURE_FAMILY, APPLICATION_FAMILY}:
        return [document_type]
    found: list[str] = []
    for names in page_parts.values():
        for name in names:
            _consider_family_name(found, str(name), family)
    payload = coerce_visual_index(visual_index)
    for item in payload.get("targets") or []:
        if not isinstance(item, Mapping):
            continue
        for name in item.get("document_types") or []:
            _consider_family_name(found, str(name), family)
    for span in (record or {}).get("documents") or []:
        if not isinstance(span, Mapping):
            continue
        _consider_family_name(found, str(span.get("name") or ""), family)
    return found or [document_type]


def _expand_needs(
    needs: Sequence[VisualNeed],
    *,
    page_parts: Mapping[int, Sequence[str]],
    visual_index: Mapping[str, Any] | None,
    record: Mapping[str, Any] | None,
) -> list[VisualNeed]:
    expanded: list[VisualNeed] = []
    seen: set[tuple[str, str]] = set()
    for need in needs:
        for name in _family_member_names(
            need.document_type,
            page_parts=page_parts,
            visual_index=visual_index,
            record=record,
        ):
            item = replace(
                need,
                document_type=name,
                page_selector=policy_selector(name) or need.page_selector,
            )
            key = (item.document_type, item.marking_type)
            if key in seen:
                continue
            seen.add(key)
            expanded.append(item)
    return expanded


def _pages_from_targets(
    visual_index: Mapping[str, Any] | None,
    document_type: str,
    *,
    selector: str,
) -> list[int]:
    wanted = document_type.casefold()
    pages: list[int] = []
    payload = coerce_visual_index(visual_index)
    for item in payload.get("targets") or []:
        if not isinstance(item, Mapping):
            continue
        names = [str(name).casefold() for name in (item.get("document_types") or [])]
        if wanted not in names:
            continue
        try:
            pages.append(int(item.get("page")))
        except (TypeError, ValueError):
            continue
    pages = sorted(set(pages))
    if selector == "last" and pages:
        return [pages[-1]]
    return pages


def _pages_from_record(
    record: Mapping[str, Any] | None,
    document_type: str,
    *,
    selector: str,
) -> list[int]:
    wanted = document_type.casefold()
    pages: list[int] = []
    for span in (record or {}).get("documents") or []:
        if not isinstance(span, Mapping):
            continue
        name = str(span.get("name") or "")
        if wanted not in name.casefold() and name.casefold() not in wanted:
            continue
        try:
            start = int(span.get("start_page"))
            end = int(span.get("end_page") or start)
        except (TypeError, ValueError):
            continue
        if end < start:
            start, end = end, start
        pages.extend(range(start, end + 1))
    pages = sorted(set(pages))
    if selector == "last" and pages:
        return [pages[-1]]
    return pages


def _notary_wording(mark: VisualMark) -> bool:
    blob = f"{mark.visible_text} {mark.associated_label}"
    return bool(_NOTARY_TEXT.search(blob))


def _mark_matches(
    need: VisualNeed, mark: VisualMark, target_pages: Sequence[int]
) -> bool:
    if mark.document_type.casefold() != need.document_type.casefold():
        return False
    if target_pages and mark.page not in target_pages:
        return False
    if mark.marking_type not in need.marking_types:
        return False
    if need.marking_type == "notary_seal":
        if mark.marking_type == "notary_seal":
            return True
        return _notary_wording(mark)
    if need.signature_roles is not None:
        return mark.signature_role in need.signature_roles
    return True


def _page_only(need: VisualNeed, page: int | None) -> VisualLocalization:
    return VisualLocalization(
        page=page,
        document_type=need.document_type,
        marking_type=need.marking_type,
        signature_role=(
            need.signature_roles[0]
            if need.signature_roles and need.signature_roles[0] != "unknown"
            else None
        ),
        bounding_boxes=[],
        boxes_status="page_only" if page is not None else "unavailable",
        confidence=None,
    )


def _from_mark(need: VisualNeed, mark: VisualMark) -> VisualLocalization:
    box = mark.bbox
    bounding = BoundingBox(
        page=mark.page,
        x=float(box.get("x") or 0),
        y=float(box.get("y") or 0),
        w=float(box.get("w") or 0),
        h=float(box.get("h") or 0),
    )
    return VisualLocalization(
        page=mark.page,
        document_type=need.document_type,
        marking_type=need.marking_type,
        signature_role=(
            mark.signature_role
            if mark.signature_role not in {"not_applicable", "unknown"}
            else None
        ),
        bounding_boxes=[bounding],
        boxes_status="matched",
        confidence=mark.confidence,
    )


def attach_visual_localizations(
    defect: Defect,
    *,
    visual_index: Mapping[str, Any] | None,
    page_parts: Mapping[int, Sequence[str]] | None = None,
    record: Mapping[str, Any] | None = None,
) -> list[VisualLocalization]:
    """Matching stored marks for this defect, or page-only placeholders.

    Never invents a box. Text-only defects return an empty list.
    """
    needs = visual_needs_for_defect(defect)
    if not needs:
        return []
    parts_map = {int(page): list(names) for page, names in (page_parts or {}).items()}
    needs = _expand_needs(
        needs,
        page_parts=parts_map,
        visual_index=visual_index,
        record=record,
    )
    marks = _marks_from_index(visual_index)
    localizations: list[VisualLocalization] = []
    for need in needs:
        target_pages = pages_for_part(
            parts_map, need.document_type, selector=need.page_selector
        )
        if not target_pages:
            target_pages = _pages_from_targets(
                visual_index, need.document_type, selector=need.page_selector
            )
        if not target_pages:
            target_pages = _pages_from_record(
                record, need.document_type, selector=need.page_selector
            )
        matched = [mark for mark in marks if _mark_matches(need, mark, target_pages)]
        matched.sort(key=lambda item: item.confidence, reverse=True)
        if matched:
            best_page = matched[0].page
            for mark in matched:
                if mark.page != best_page:
                    continue
                localizations.append(_from_mark(need, mark))
            continue
        page = target_pages[-1] if target_pages else None
        localizations.append(_page_only(need, page))
    return localizations
