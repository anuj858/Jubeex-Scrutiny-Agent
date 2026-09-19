"""Hybrid compiled-PDF split repair: anchors, court hierarchy, duplicates.

LlamaSplit is treated as a hint. Final page labels come from printed headings,
annexure stamps, and nesting rules so High Court exhibits never overwrite the
Supreme Court Main Petition / Vakalatnama slots.
"""

from __future__ import annotations

import re
from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any

from .document_parts import (
    ANNEXURE_FAMILY,
    MAIN_PETITION_PART,
    PagePartMap,
    _CARRY_BLOCKING_PARTS,
    _contiguous_groups,
    _heading_window,
    _is_real_split_label,
    _is_sci_application_start,
    _looks_like_index_table,
    _memo_of_parties_heading,
    _vakalatnama_heading,
    annexure_mark_in_heading,
    collapse_repeated_split_pages,
    explode_repeating_split_parts,
    family_split_name,
    page_starts_application,
    parts_on_page,
)

_SCI_CAPTION_RE = re.compile(r"in the supreme court of india", re.IGNORECASE)
_HC_CAPTION_RE = re.compile(
    r"in the (?:hon'?ble\s+)?high court|"
    r"high court of judicature|"
    r"bench at\s+\w+",
    re.IGNORECASE,
)
_TRIBUNAL_CAPTION_RE = re.compile(
    r"before the (?:hon'?ble\s+)?(?:minister|tribunal|authority|registrar)|"
    r"revisional authority|"
    r"divisional joint registrar",
    re.IGNORECASE,
)
_OFFICE_REPORT_RE = re.compile(r"office report on limitation|o/?r on limitation", re.I)
_LISTING_RE = re.compile(
    r"proforma for first listing|listing proforma|listed proforma", re.I
)
_SYNOPSIS_RE = re.compile(r"(?m)^\s*synopsis\b", re.I)
_LOD_RE = re.compile(r"list of dates", re.I)
_APPENDIX_RE = re.compile(r"(?m)^\s*appendix\b", re.I)
_RECORD_RE = re.compile(r"record of proceedings?", re.I)
_FILING_MEMO_RE = re.compile(r"filing memo|index of filing|filing index", re.I)
_AOR_CERT_RE = re.compile(
    r"confined only to the pleadings|"
    r"(?:^|\n)\s*(?:c\s+e\s+r\s+t\s+i\s+f\s+i\s+c\s+a\s+t\s+e|certificate)\b",
    re.IGNORECASE,
)
_SCI_CHECKLIST_RE = re.compile(
    r"advocate'?s?\s+check[\s\-]*list|check[\s\-]*list.*supreme court",
    re.I,
)
_STATE_CHECKLIST_RE = re.compile(
    r"check[\s\-]*list.*(?:housing societ|co-?operative|registration of)|"
    r"commissioner for co-?operation",
    re.I,
)
# Cover pages also say "SPECIAL LEAVE PETITION"; require Form-28 body cues.
_FORM28_RE = re.compile(
    r"form\s*28|questions of law|"
    r"prayer for interim relief|under article\s*136",
    re.I,
)
_IMPUGNED_ORDER_RE = re.compile(
    r"(?m)^\s*impugned\s+(?:final\s+)?(?:order|judgment)\b|"
    r"true copy of (?:the\s+)?(?:impugned\s+)?(?:order|judgment)",
    re.I,
)
_APPEARANCE_RE = re.compile(r"memo of appearance", re.I)

# Outer SCI paper-book parts: keep the first contiguous run, not the longest.
# (A later High Court writ mislabeled Main Petition must not win.)
# Mirrored in document_parts.collapse_repeated_split_pages.
_FIRST_RUN_PARTS = frozenset(
    {
        MAIN_PETITION_PART,
        "Cover Page",
        "Index",
        "Advocate's Checklist",
        "Office Report on Limitation",
        "Listing Proforma",
        "Synopsis",
        "List of Dates & Events",
        "Impugned Order",
        "AOR's Certificate",
        "Affidavit",
        "Appendix",
        "Vakalatnama",
        "Memo of Appearance",
        "Memo of Parties",
        "Filing Memo",
        "Court Fees",
        "PoA/BR",
    }
)

_NESTED_STEAL_PARTS = frozenset(
    {
        MAIN_PETITION_PART,
        "Affidavit",
        "Vakalatnama",
        "Memo of Appearance",
        "Memo of Parties",
        "AOR's Certificate",
        "Advocate's Checklist",
        "Impugned Order",
        "Cover Page",
    }
)


@dataclass(frozen=True)
class DuplicateSplitHit:
    part: str
    kept_span: tuple[int, int]
    duplicate_spans: tuple[tuple[int, int], ...]

    def as_dict(self) -> dict[str, Any]:
        return {
            "part": self.part,
            "kept_span": {"start_page": self.kept_span[0], "end_page": self.kept_span[1]},
            "duplicate_spans": [
                {"start_page": start, "end_page": end}
                for start, end in self.duplicate_spans
            ],
        }


def _fold(text: str) -> str:
    return re.sub(r"\s+", " ", (text or "").casefold()).strip()


def _is_sci_caption(text: str) -> bool:
    return bool(_SCI_CAPTION_RE.search(_heading_window(text, lines=10)))


def _is_lower_court_caption(text: str) -> bool:
    head = _heading_window(text, lines=14)
    if _is_sci_caption(text):
        return False
    return bool(_HC_CAPTION_RE.search(head) or _TRIBUNAL_CAPTION_RE.search(head))


def _looks_like_sci_checklist(text: str) -> bool:
    head = text[:1200]
    if _STATE_CHECKLIST_RE.search(head):
        return False
    return bool(_SCI_CHECKLIST_RE.search(head))


def _looks_like_cover_page(text: str) -> bool:
    if not _is_sci_caption(text):
        return False
    head = _fold(_heading_window(text, lines=16))
    if _FORM28_RE.search(head):
        return False
    if _is_sci_application_start(text):
        return False
    if _OFFICE_REPORT_RE.search(text[:900]):
        return False
    if _LISTING_RE.search(text[:900]):
        return False
    return "for index please see inside" in head or (
        "civil appellate jurisdiction" in head and "questions of law" not in head
    )


def _looks_like_sci_main_petition(text: str) -> bool:
    if not _is_sci_caption(text):
        return False
    if _looks_like_cover_page(text):
        return False
    if _is_sci_application_start(text):
        return False
    if _OFFICE_REPORT_RE.search(text[:900]):
        return False
    if _is_lower_court_caption(text):
        return False
    head = _heading_window(text, lines=20)
    return bool(_FORM28_RE.search(head))


def _outer_anchor_label(text: str) -> str | None:
    """Return a strong outer-document label from page text, or None."""
    if not (text or "").strip():
        return None
    if _looks_like_index_table(text):
        return "Index"
    if _OFFICE_REPORT_RE.search(_heading_window(text, lines=12)):
        return "Office Report on Limitation"
    if _LISTING_RE.search(_heading_window(text, lines=10)):
        return "Listing Proforma"
    if _RECORD_RE.search(_heading_window(text, lines=10)):
        return "Record of Proceedings"
    if _SYNOPSIS_RE.search(_heading_window(text, lines=8)):
        return "Synopsis"
    if _LOD_RE.search(_heading_window(text, lines=8)) and not _looks_like_index_table(
        text
    ):
        return "List of Dates & Events"
    if _APPENDIX_RE.search(_heading_window(text, lines=6)):
        return "Appendix"
    if _FILING_MEMO_RE.search(_heading_window(text, lines=10)):
        return "Filing Memo"
    if _looks_like_sci_checklist(text):
        return "Advocate's Checklist"
    if _AOR_CERT_RE.search(_heading_window(text, lines=16)) and _is_sci_caption(text):
        return "AOR's Certificate"
    if page_starts_application(text):
        return "Application 1"
    if _vakalatnama_heading(text) and (
        _is_sci_caption(text) or not _is_lower_court_caption(text)
    ):
        return "Vakalatnama"
    if _APPEARANCE_RE.search(_heading_window(text, lines=10)):
        return "Memo of Appearance"
    if _memo_of_parties_heading(text):
        # High Court memo of parties stays nested when under annexures;
        # outer assignment is applied only outside annexure runs.
        return "Memo of Parties"
    if _IMPUGNED_ORDER_RE.search(_heading_window(text, lines=12)) and not annexure_mark_in_heading(
        text
    ):
        return "Impugned Order"
    # Cover before Main Petition: covers print SCI + SLP without Form-28 body cues.
    if _looks_like_cover_page(text):
        return "Cover Page"
    if _looks_like_sci_main_petition(text):
        return MAIN_PETITION_PART
    return None


def _annexure_run_bounds(
    page_text: Mapping[int, str], page_count: int
) -> list[tuple[int, int, int]]:
    """(start_page, end_page, mark) for each printed annexure run."""
    starts: list[tuple[int, int]] = []
    for page in range(1, page_count + 1):
        mark = annexure_mark_in_heading(page_text.get(page, ""))
        if mark:
            starts.append((page, mark))
    if not starts:
        return []

    runs: list[tuple[int, int, int]] = []
    for index, (start, mark) in enumerate(starts):
        if index + 1 < len(starts):
            end = starts[index + 1][0] - 1
        else:
            end = page_count
            for page in range(start + 1, page_count + 1):
                text = page_text.get(page, "")
                if page_starts_application(text):
                    end = page - 1
                    break
                if _vakalatnama_heading(text) and _is_sci_caption(text):
                    end = page - 1
                    break
                if _FILING_MEMO_RE.search(_heading_window(text, lines=8)):
                    end = page - 1
                    break
        if end >= start:
            runs.append((start, end, mark))
    return runs


def _force_annexure_nesting(
    page_parts: PagePartMap,
    page_text: Mapping[int, str],
    page_count: int,
) -> PagePartMap:
    """Pages inside Annexure P-n keep that label even if they look like Main Petition."""
    updated = {page: list(names) for page, names in page_parts.items()}
    for start, end, mark in _annexure_run_bounds(page_text, page_count):
        label = f"Annexure P-{mark}"
        for page in range(start, end + 1):
            text = page_text.get(page, "")
            names = parts_on_page(updated.get(page))
            # Never steal Index / OR / Listing that somehow overlaps (shouldn't).
            if names and all(name in _CARRY_BLOCKING_PARTS for name in names):
                if not annexure_mark_in_heading(text):
                    continue
            if not names or any(name in _NESTED_STEAL_PARTS for name in names):
                updated[page] = [label]
                continue
            if any(family_split_name(name) == ANNEXURE_FAMILY for name in names):
                updated[page] = [label]
                continue
            if _is_lower_court_caption(text) or annexure_mark_in_heading(text):
                updated[page] = [label]
    return updated


def _apply_outer_anchors(
    page_parts: PagePartMap,
    page_text: Mapping[int, str],
    page_count: int,
) -> PagePartMap:
    """Stamp strong outer headings onto pages outside annexure runs."""
    updated = {page: list(names) for page, names in page_parts.items()}
    annexure_pages: set[int] = set()
    for start, end, _mark in _annexure_run_bounds(page_text, page_count):
        annexure_pages.update(range(start, end + 1))

    for page in range(1, page_count + 1):
        if page in annexure_pages:
            continue
        text = page_text.get(page, "")
        label = _outer_anchor_label(text)
        if not label:
            continue
        # High Court memo of parties after annexures: keep as Memo of Parties
        # but do not treat HC caption pages as Main Petition (handled by nesting).
        if label == MAIN_PETITION_PART and _is_lower_court_caption(text):
            continue
        if label == "Advocate's Checklist" and _STATE_CHECKLIST_RE.search(text[:1200]):
            continue
        names = parts_on_page(updated.get(page))
        if not names or any(
            name in _NESTED_STEAL_PARTS
            or family_split_name(name) == ANNEXURE_FAMILY
            or not _is_real_split_label(name)
            for name in names
        ):
            # Never overwrite a Cover Page with Main Petition from a weak cue;
            # Cover is only replaced when the page itself is a Form-28 start.
            if (
                label == MAIN_PETITION_PART
                and "Cover Page" in names
                and not _looks_like_sci_main_petition(text)
            ):
                continue
            updated[page] = [label]
            continue
        # Upgrade weak/wrong labels when a strong outer heading is present.
        if label in {
            "Office Report on Limitation",
            "Listing Proforma",
            "Index",
            "Appendix",
            "Record of Proceedings",
            MAIN_PETITION_PART,
            "Cover Page",
        }:
            if label == MAIN_PETITION_PART and "Cover Page" in names:
                if not _looks_like_sci_main_petition(text):
                    continue
            updated[page] = [label]
    return updated


def _fill_gaps(
    page_parts: PagePartMap,
    page_text: Mapping[int, str],
    page_count: int,
) -> PagePartMap:
    updated = {page: list(names) for page, names in page_parts.items() if parts_on_page(names)}
    last_label: list[str] | None = None
    for page in range(1, page_count + 1):
        names = parts_on_page(updated.get(page))
        if names:
            last_label = list(names)
            continue
        text = page_text.get(page, "")
        if not last_label:
            continue
        if any(name in _CARRY_BLOCKING_PARTS for name in last_label):
            continue
        if _looks_like_index_table(text):
            continue
        if page_starts_application(text) or _vakalatnama_heading(text):
            continue
        if _outer_anchor_label(text):
            continue
        # Do not bridge two islands of the same outer part (e.g. two Vakalatnamas).
        next_labeled: list[str] | None = None
        for ahead in range(page + 1, page_count + 1):
            ahead_names = parts_on_page(updated.get(ahead))
            if ahead_names:
                next_labeled = ahead_names
                break
        if next_labeled and (set(last_label) & set(next_labeled) & _FIRST_RUN_PARTS):
            continue
        updated[page] = list(last_label)

    for page in range(page_count, 0, -1):
        if parts_on_page(updated.get(page)):
            continue
        nxt = parts_on_page(updated.get(page + 1))
        if not nxt:
            continue
        # Front-matter slots must not expand backward into unlabeled gaps.
        if any(name in _CARRY_BLOCKING_PARTS for name in nxt):
            continue
        text = page_text.get(page, "")
        if _looks_like_index_table(text) or _outer_anchor_label(text):
            continue
        prev_labeled: list[str] | None = None
        for back in range(page - 1, 0, -1):
            back_names = parts_on_page(updated.get(back))
            if back_names:
                prev_labeled = back_names
                break
        if prev_labeled and (set(nxt) & set(prev_labeled) & _FIRST_RUN_PARTS):
            continue
        updated[page] = list(nxt)
    return updated


def _demote_false_advocate_checklist(
    page_parts: PagePartMap, page_text: Mapping[int, str]
) -> PagePartMap:
    updated = {page: list(names) for page, names in page_parts.items()}
    for page, names in list(updated.items()):
        if "Advocate's Checklist" not in parts_on_page(names):
            continue
        text = page_text.get(page, "")
        if _STATE_CHECKLIST_RE.search(text[:1500]) or annexure_mark_in_heading(text):
            mark = annexure_mark_in_heading(text)
            updated[page] = [f"Annexure P-{mark}"] if mark else ["Annexure P-1"]
    return updated


def collapse_repeated_split_pages_first_outer(page_parts: PagePartMap) -> PagePartMap:
    """Delegate to document_parts collapse (first-run outer parts)."""
    return collapse_repeated_split_pages(page_parts)


def find_duplicate_split_parts(page_parts: PagePartMap) -> list[DuplicateSplitHit]:
    """Outer parts with more than one contiguous island before collapse."""
    pages_by_part: dict[str, list[int]] = {}
    for page, names in page_parts.items():
        for part in parts_on_page(names):
            pages_by_part.setdefault(part, []).append(page)

    hits: list[DuplicateSplitHit] = []
    for part, pages in sorted(pages_by_part.items()):
        groups = _contiguous_groups(sorted(set(pages)))
        if len(groups) < 2:
            continue
        if part not in _FIRST_RUN_PARTS and not re.fullmatch(
            r"(?i)annexure p-?\d{1,3}|application \d{1,3}", part
        ):
            continue
        kept = groups[0]
        dupes = tuple(groups[1:])
        hits.append(DuplicateSplitHit(part=part, kept_span=kept, duplicate_spans=dupes))
    return hits


def repair_compiled_split(
    page_parts: PagePartMap,
    page_text: Mapping[int, str],
    *,
    page_count: int,
) -> tuple[PagePartMap, list[DuplicateSplitHit]]:
    """Full hybrid repair used after LlamaSplit for compiled petitions."""
    if page_count < 1:
        cleaned = {
            int(page): list(names)
            for page, names in page_parts.items()
            if parts_on_page(names)
        }
        return cleaned, []

    updated: PagePartMap = {}
    for page, names in page_parts.items():
        try:
            number = int(page)
        except (TypeError, ValueError):
            continue
        kept = [name for name in parts_on_page(names) if _is_real_split_label(name)]
        if kept:
            updated[number] = kept

    updated = _demote_false_advocate_checklist(updated, page_text)
    updated = _apply_outer_anchors(updated, page_text, page_count)
    # Capture Llama wrong-label duplicates before nesting absorbs HC exhibits.
    duplicates = find_duplicate_split_parts(updated)
    updated = _force_annexure_nesting(updated, page_text, page_count)
    updated = _fill_gaps(updated, page_text, page_count)
    updated = _force_annexure_nesting(updated, page_text, page_count)
    updated = _demote_false_advocate_checklist(updated, page_text)

    exploded = explode_repeating_split_parts(updated, page_text)
    exploded = _force_annexure_nesting(exploded, page_text, page_count)
    more = find_duplicate_split_parts(exploded)
    seen = {(hit.part, hit.kept_span) for hit in duplicates}
    for hit in more:
        key = (hit.part, hit.kept_span)
        if key not in seen:
            duplicates.append(hit)
            seen.add(key)
    repaired = collapse_repeated_split_pages(exploded)
    repaired = _force_annexure_nesting(repaired, page_text, page_count)
    repaired = collapse_repeated_split_pages(repaired)
    return repaired, duplicates
