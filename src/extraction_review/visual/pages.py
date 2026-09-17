"""Select formality pages from Jubeex Split document_part spans."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Any

from ..document_parts import MAIN_PETITION_PART, family_split_name
from .schema import VisualPageTarget

# Last page of each D093 Split type that is present, plus Affidavit last
# for notary. Impugned Order and Court Fees are OCR-only.
FORMALITY_PAGE_POLICY: dict[str, str] = {
    "Advocate's Checklist": "last",
    "Listing Proforma": "last",
    MAIN_PETITION_PART: "last",
    "AOR's Certificate": "last",
    "Application": "last",
    "Filing Memo": "last",
    "Memo of Parties": "last",
    "Vakalatnama": "last",
    "Memo of Appearance": "last",
    "Annexures": "last",
    "Affidavit": "last",
}


@dataclass(frozen=True)
class PageSource:
    page: int
    slot_id: str
    local_page: int
    document_parts: tuple[str, ...]
    file_url: str | None = None
    file_id: str | None = None


def policy_selector(name: str, policy: Mapping[str, str] | None = None) -> str | None:
    """Return ``last``/``all`` when ``name`` or its Split family is in policy."""
    rules = policy or FORMALITY_PAGE_POLICY
    if name in rules:
        return rules[name]
    family = family_split_name(name)
    if family != name:
        return rules.get(family)
    return None


def coerce_pages_by_slot(
    raw: Mapping[str, Mapping[Any, Any]] | None,
) -> dict[str, dict[int, str]]:
    pages: dict[str, dict[int, str]] = {}
    for slot_id, local in (raw or {}).items():
        slot = str(slot_id)
        mapped: dict[int, str] = {}
        for key, value in (local or {}).items():
            try:
                number = int(key)
            except (TypeError, ValueError):
                continue
            mapped[number] = str(value or "")
        if mapped:
            pages[slot] = mapped
    return pages


def global_page_sources(
    parts: Sequence[Any],
    pages_by_slot: Mapping[str, Mapping[int, str]],
    *,
    omit_empty: bool = False,
) -> list[PageSource]:
    """Mirror stitch_parsed_parts_in_order: catalog/request order, 1-indexed global pages."""
    sources: list[PageSource] = []
    next_page = 1
    for item in parts:
        slot_id = str(getattr(item, "slot_id", "") or "")
        document_parts = tuple(
            str(name).strip()
            for name in (getattr(item, "document_parts", None) or ())
            if str(name).strip()
        )
        file_url = str(getattr(item, "file_url", None) or "").strip() or None
        file_id = str(getattr(item, "file_id", None) or "").strip() or None
        local = pages_by_slot.get(slot_id) or {}
        local_numbers = sorted(int(page) for page in local)
        if not local_numbers:
            if omit_empty:
                continue
            next_page += 1
            continue
        for local_page in local_numbers:
            sources.append(
                PageSource(
                    page=next_page,
                    slot_id=slot_id,
                    local_page=local_page,
                    document_parts=document_parts,
                    file_url=file_url,
                    file_id=file_id,
                )
            )
            next_page += 1
    return sources


def pages_for_part(
    page_parts: Mapping[int, Sequence[str]],
    document_type: str,
    *,
    selector: str = "all",
) -> list[int]:
    wanted = document_type.casefold()
    matched = sorted(
        page
        for page, names in page_parts.items()
        if any(str(name).casefold() == wanted for name in names)
    )
    if selector == "last" and matched:
        return [matched[-1]]
    return matched


def select_formality_pages(
    sources: Sequence[PageSource],
    *,
    page_markdown: Mapping[int, str] | None = None,
    policy: Mapping[str, str] | None = None,
) -> list[VisualPageTarget]:
    """Return unique last pages of present D093 types (and Affidavit last)."""
    rules = dict(policy or FORMALITY_PAGE_POLICY)
    last_by_part: dict[str, PageSource] = {}
    all_hits: dict[int, PageSource] = {}
    types_by_page: dict[int, list[str]] = {}

    for source in sources:
        for name in source.document_parts:
            selector = policy_selector(name, rules)
            if selector is None:
                continue
            if selector == "last":
                last_by_part[name] = source
                continue
            all_hits[source.page] = source
            if name not in types_by_page.setdefault(source.page, []):
                types_by_page[source.page].append(name)

    for name, source in last_by_part.items():
        all_hits[source.page] = source
        if name not in types_by_page.setdefault(source.page, []):
            types_by_page[source.page].append(name)

    markdown = page_markdown or {}
    targets: list[VisualPageTarget] = []
    for page in sorted(all_hits):
        source = all_hits[page]
        if source.local_page < 1:
            continue
        targets.append(
            VisualPageTarget(
                page=source.page,
                slot_id=source.slot_id,
                local_page=source.local_page,
                document_types=types_by_page.get(page, list(source.document_parts)),
                file_url=source.file_url,
                file_id=source.file_id,
                markdown=str(markdown.get(page) or ""),
            )
        )
    return targets
