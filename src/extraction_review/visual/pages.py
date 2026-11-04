"""Select formality pages from Jubeex Split document_part spans."""

from __future__ import annotations

import math
import os
import re
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Any

from ..document_parts import MAIN_PETITION_PART, family_split_name
from .schema import VisualPageTarget

# Visual inspection is deliberately selective.  OCR/structured extraction covers
# text on every page; vision is reserved for pages likely to contain handwriting,
# signatures, seals, stamps, blank form fields, or representative layout evidence.
#
# Selectors:
# - all_short: all pages up to SHORT_DOCUMENT_PAGE_LIMIT, otherwise boundaries/cues
# - first_last_cues: first/last plus cue-bearing pages (bounded per document)
# - boundary_cues: first/middle/last plus cue-bearing pages
# - first_cues: first plus cue-bearing pages (used for annexure markings)
FORMALITY_PAGE_POLICY: dict[str, str] = {
    "Index": "all",
    # Synopsis and List of Dates are text-comparison documents. Their checks
    # use parsed text/layout and do not justify a standing vision scan.
    "Impugned Order": "first_last_cues",
    "Advocate's Checklist": "all",
    "Listing Proforma": "all",
    MAIN_PETITION_PART: "boundary_cues",
    "AOR's Certificate": "all",
    "Application": "first_last_cues",
    "Filing Memo": "all",
    "Memo of Parties": "first_last_cues",
    "Vakalatnama": "all",
    "Memo of Appearance": "all",
    "Annexures": "first_cues",
    "Affidavit": "all",
}

SHORT_DOCUMENT_PAGE_LIMIT = 3
MAX_CUE_PAGES_PER_DOCUMENT = 2
DEFAULT_MAX_VISUAL_PAGE_FRACTION = 0.30
FULL_SCAN_PAGE_LIMIT = 20

VISUAL_CUE_RE = re.compile(
    r"\b(?:signature|signed|signatory|advocate(?:-on-record)?|aor|counsel|"
    r"deponent|notary|notarial|oath\s+commissioner|attest(?:ed|ation)?|"
    r"verification|verified|seal|stamp|thumb\s+impression|executant|"
    r"certified\s+(?:true\s+)?copy|true\s+copy|drawn\s*&\s*filed|"
    r"filed\s+by|identified\s+by|dated)\b",
    re.IGNORECASE,
)


@dataclass(frozen=True)
class PageSource:
    page: int
    slot_id: str
    local_page: int
    document_parts: tuple[str, ...]
    file_url: str | None = None
    file_id: str | None = None


def policy_selector(name: str, policy: Mapping[str, str] | None = None) -> str | None:
    """Return the selector for ``name`` or its normalized Split family."""
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
    if not matched:
        return []
    if selector == "first":
        return [matched[0]]
    if selector == "last":
        return [matched[-1]]
    if selector in {"first_last", "first_last_cues"}:
        return sorted({matched[0], matched[-1]})
    if selector == "boundary_cues":
        return sorted({matched[0], matched[len(matched) // 2], matched[-1]})
    if selector == "first_cues":
        # Cue pages require markdown and are therefore taken from visual targets
        # by callers that have the stored visual index.
        return []
    return matched


def _fraction_from_env() -> float:
    raw = os.getenv("VISION_MAX_PAGE_FRACTION", str(DEFAULT_MAX_VISUAL_PAGE_FRACTION))
    try:
        return min(1.0, max(0.05, float(raw)))
    except (TypeError, ValueError):
        return DEFAULT_MAX_VISUAL_PAGE_FRACTION


def _cue_pages(
    sources: Sequence[PageSource], markdown: Mapping[int, str]
) -> list[PageSource]:
    hits = [
        source
        for source in sources
        if VISUAL_CUE_RE.search(markdown.get(source.page, ""))
    ]
    if len(hits) <= MAX_CUE_PAGES_PER_DOCUMENT:
        return hits
    # Prefer cues nearest execution boundaries; they most often contain the
    # signature/attestation blocks relevant to visual catalogue checks.
    return sorted(hits, key=lambda item: (-item.local_page, item.page))[
        :MAX_CUE_PAGES_PER_DOCUMENT
    ]


def _selected_for_document(
    sources: Sequence[PageSource],
    *,
    selector: str,
    markdown: Mapping[int, str],
) -> list[tuple[PageSource, int]]:
    ordered = sorted(sources, key=lambda item: (item.local_page, item.page))
    if not ordered:
        return []
    first = ordered[0]
    last = ordered[-1]
    cues = _cue_pages(ordered, markdown)

    selected: dict[int, tuple[PageSource, int]] = {}

    def add(source: PageSource, priority: int) -> None:
        current = selected.get(source.page)
        if current is None or priority > current[1]:
            selected[source.page] = (source, priority)

    if selector == "all":
        for source in ordered:
            add(source, 100)
    elif selector == "all_short" and len(ordered) <= SHORT_DOCUMENT_PAGE_LIMIT:
        for source in ordered:
            add(source, 90)
        add(first, 95)
        add(last, 100)
    elif selector == "first":
        add(first, 95)
    elif selector == "last":
        add(last, 100)
    elif selector == "first_cues":
        add(first, 100)
        for source in cues:
            add(source, 95)
    elif selector == "boundary_cues":
        add(first, 95)
        add(ordered[len(ordered) // 2], 80)
        add(last, 100)
        for source in cues:
            add(source, 98)
    else:  # first_last, first_last_cues, and long all_short documents
        add(first, 95)
        add(last, 100)
        for source in cues:
            add(source, 98)
    return list(selected.values())


def eligible_formality_page_count(
    sources: Sequence[PageSource], policy: Mapping[str, str] | None = None
) -> int:
    """Number of pages in document families eligible for visual inspection."""
    rules = policy or FORMALITY_PAGE_POLICY
    return len(
        {
            source.page
            for source in sources
            if any(
                policy_selector(name, rules) is not None
                for name in source.document_parts
            )
        }
    )


def select_formality_pages(
    sources: Sequence[PageSource],
    *,
    page_markdown: Mapping[int, str] | None = None,
    policy: Mapping[str, str] | None = None,
    max_page_fraction: float | None = None,
) -> list[VisualPageTarget]:
    """Return a bounded, document-aware set of pages for visual inspection."""
    rules = dict(policy or FORMALITY_PAGE_POLICY)
    markdown = {
        int(page): str(text or "") for page, text in (page_markdown or {}).items()
    }
    sources_by_part: dict[tuple[str, str], list[PageSource]] = {}
    selector_by_part: dict[tuple[str, str], str] = {}
    for source in sources:
        for name in source.document_parts:
            selector = policy_selector(name, rules)
            if selector is None:
                continue
            # The same family label can appear in more than one uploaded file
            # (for example, Annexure P-7 split into two source PDFs). Preserve
            # per-slot coverage instead of merging those files by label.
            key = (source.slot_id, name)
            sources_by_part.setdefault(key, []).append(source)
            selector_by_part[key] = selector

    candidates: dict[int, tuple[PageSource, int]] = {}
    types_by_page: dict[int, list[str]] = {}
    candidates_by_part: dict[tuple[str, str], list[tuple[PageSource, int]]] = {}
    for key, part_sources in sources_by_part.items():
        _slot_id, name = key
        selected = _selected_for_document(
            part_sources,
            selector=selector_by_part[key],
            markdown=markdown,
        )
        candidates_by_part[key] = selected
        for source, priority in selected:
            current = candidates.get(source.page)
            if current is None or priority > current[1]:
                candidates[source.page] = (source, priority)
            if name not in types_by_page.setdefault(source.page, []):
                types_by_page[source.page].append(name)

    legacy_count = eligible_formality_page_count(sources, rules)
    fraction = _fraction_from_env() if max_page_fraction is None else max_page_fraction
    fraction = min(1.0, max(0.05, float(fraction)))
    if legacy_count <= FULL_SCAN_PAGE_LIMIT:
        budget = legacy_count
    else:
        budget = max(1, math.ceil(legacy_count * fraction))

    # Preserve at least one high-value page for every document type before
    # spending the remaining budget on boundary samples and cue pages.
    essential_pages: set[int] = set()
    for key, selected in candidates_by_part.items():
        if not selected:
            continue
        if selector_by_part[key] == "all":
            essential_pages.update(source.page for source, _priority in selected)
            continue
        source, _priority = max(selected, key=lambda item: (item[1], -item[0].page))
        essential_pages.add(source.page)
    budget = max(budget, len(essential_pages))

    chosen = set(essential_pages)
    remaining = sorted(
        (
            (page, source, priority)
            for page, (source, priority) in candidates.items()
            if page not in chosen
        ),
        key=lambda item: (-item[2], item[0]),
    )
    for page, _source, _priority in remaining:
        if len(chosen) >= budget:
            break
        chosen.add(page)

    targets: list[VisualPageTarget] = []
    for page in sorted(chosen):
        source = candidates[page][0]
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
                markdown=markdown.get(page, ""),
            )
        )
    return targets
