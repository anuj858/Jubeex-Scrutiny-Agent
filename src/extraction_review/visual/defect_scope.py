"""Choose vision pages from the defects in a run.

No check id is special-cased. A defect is imaged when its title or requirement
asks for a signature, seal, stamp, sworn execution, or a date written on the
document. The pages come from that defect's ``inspect_parts`` using the
selector already stored for the document. A later catalogue row is picked up
the same way.
"""

from __future__ import annotations

import re
from collections.abc import Mapping, Sequence
from dataclasses import dataclass

from ..document_parts import family_split_name, parts_on_page
from ..scrutiny.rules import Defect
from .pages import (
    PageSource,
    _selected_for_document,
    policy_selector,
)

MAX_DEFECT_IMAGES = 4

_INK_RE = re.compile(
    r"\b(?:signatur\w*|signed|notary|notarial|notar\w*|seal|stamp|"
    r"handwritten|thumb\s+impression|duly\s+sworn|attested|attestation)\b",
    re.IGNORECASE,
)
_DATE_ROLE_WORDS = frozenset(
    {
        "same",
        "subsequent",
        "prior",
        "later",
        "earlier",
        "blank",
        "missing",
        "mention",
        "mentions",
        "mentioned",
        "bear",
        "bears",
        "execution",
        "verification",
        "attestation",
        "drafting",
        "signing",
    }
)
_NOT_PAGE_DATE_RE = re.compile(r"\b(?:limitation|impugned)\b", re.IGNORECASE)
# A signature or stamp named only as the place to read a date or a filing
# time is not a request to inspect the ink.
_LANDMARK_RE = re.compile(
    r"(?:below|beside|under|above|near)\s+the\s+(?:judge['’]s\s+)?signature"
    r"|filing(?:\s+and\s+refiling)?\s+(?:history|timestamps?)"
    r"|refiling\s+timestamps?",
    re.IGNORECASE,
)


@dataclass(frozen=True)
class DefectImagePage:
    """One filing page chosen for a defect."""

    page: int
    local_page: int
    slot_id: str
    document_type: str
    file_id: str | None = None
    file_url: str | None = None
    text: str = ""


@dataclass(frozen=True)
class DefectImagePlan:
    """Pages a defect will see, and pages past the four-image cap."""

    check_id: str
    pages: tuple[DefectImagePage, ...]
    leftover_pages: tuple[DefectImagePage, ...]

    @property
    def complete(self) -> bool:
        """True when every selected page fits in one four-image budget."""
        return bool(self.pages) and not self.leftover_pages


def _needs_page_date(text: str) -> bool:
    """True when a sentence asks for a date written on the document itself.

    The date word and the role word must sit near each other. A date of an
    impugned order or a limitation period is not a date filled on the filing.
    """
    for sentence in re.split(r"[\n.]+", text):
        if _NOT_PAGE_DATE_RE.search(sentence):
            continue
        words = [word.lower() for word in re.findall(r"[A-Za-z']+", sentence)]
        for index, word in enumerate(words):
            if word not in {"date", "dates"}:
                continue
            window = words[max(0, index - 8) : index + 9]
            if any(item in _DATE_ROLE_WORDS for item in window):
                return True
            if any(item in {"state", "states", "stated"} for item in words[max(0, index - 3) : index]):
                return True
    return False


def defect_needs_ink(defect: Defect) -> bool:
    """True when this defect's own title or requirement asks to see the page."""
    text = _LANDMARK_RE.sub(" ", f"{defect.defect or ''}\n{defect.requirement or ''}")
    return bool(_INK_RE.search(text) or _needs_page_date(text))


def selector_for_document(name: str) -> str:
    """Selector for a catalogue part, or every page when the part is new."""
    return policy_selector(name) or policy_selector(family_split_name(name)) or "all"


def with_slot_files(
    plans: Sequence[DefectImagePlan],
    metadata: Mapping[str, object] | None,
) -> list[DefectImagePlan]:
    """Copy file ids from the filing metadata onto pages that do not have one."""
    files = _slot_files(metadata or {})
    if not files:
        return list(plans)
    updated: list[DefectImagePlan] = []
    for plan in plans:
        updated.append(
            DefectImagePlan(
                check_id=plan.check_id,
                pages=tuple(_with_file(page, files) for page in plan.pages),
                leftover_pages=tuple(
                    _with_file(page, files) for page in plan.leftover_pages
                ),
            )
        )
    return updated


def _slot_files(
    metadata: Mapping[str, object],
) -> dict[str, tuple[str | None, str | None]]:
    found: dict[str, tuple[str | None, str | None]] = {}
    documents = metadata.get("documents")
    if isinstance(documents, list):
        for item in documents:
            if not isinstance(item, Mapping):
                continue
            slot = str(item.get("slot_id") or "").strip()
            if not slot:
                continue
            file_id = str(item.get("file_id") or "").strip() or None
            file_url = str(item.get("file_url") or "").strip() or None
            found[slot] = (file_id, file_url)
    split_files = metadata.get("split_files")
    if isinstance(split_files, Mapping):
        for slot, file_id in split_files.items():
            current = found.get(str(slot), (None, None))
            stored = str(file_id or "").strip() or current[0]
            found[str(slot)] = (stored, current[1])
    return found


def _with_file(
    page: DefectImagePage,
    files: Mapping[str, tuple[str | None, str | None]],
) -> DefectImagePage:
    file_id, file_url = files.get(page.slot_id, (None, None))
    if page.file_id or page.file_url or (not file_id and not file_url):
        return page
    return DefectImagePage(
        page=page.page,
        local_page=page.local_page,
        slot_id=page.slot_id,
        document_type=page.document_type,
        file_id=file_id or page.file_id,
        file_url=file_url or page.file_url,
    )


def visual_targets_for_plans(plans: Sequence[DefectImagePlan]) -> list[Any]:
    """One notary-scan target per physical page the defects selected."""
    from .schema import VisualPageTarget

    by_page: dict[int, VisualPageTarget] = {}
    for plan in plans:
        for page in plan.pages:
            current = by_page.get(page.page)
            if current is None:
                by_page[page.page] = VisualPageTarget(
                    page=page.page,
                    slot_id=page.slot_id or f"page-{page.page}",
                    local_page=page.local_page,
                    document_types=[page.document_type] if page.document_type else [],
                    file_url=page.file_url,
                    file_id=page.file_id,
                    markdown=page.text,
                )
                continue
            if page.document_type and page.document_type not in current.document_types:
                current.document_types.append(page.document_type)
    return [by_page[number] for number in sorted(by_page)]


def plan_defect_images(
    defects: Sequence[Defect],
    layout: Mapping[int, Mapping[str, object]] | None,
) -> list[DefectImagePlan]:
    """Page plans for the ink defects in this run, in catalogue order."""
    pages = _sources_from_layout(layout or {})
    markdown = _markdown_from_layout(layout or {})
    plans: list[DefectImagePlan] = []
    for defect in defects:
        if not defect_needs_ink(defect):
            continue
        plans.append(_plan_one(defect, pages, markdown))
    return plans


def _plan_one(
    defect: Defect,
    sources: Sequence[PageSource],
    markdown: Mapping[int, str],
) -> DefectImagePlan:
    chosen: list[tuple[int, DefectImagePage]] = []
    seen: set[int] = set()
    for name in defect.inspect_parts or []:
        label = str(name).strip()
        if not label:
            continue
        selector = selector_for_document(label)
        for group in _groups_for_part(label, sources):
            picked = _selected_for_document(
                group, selector=selector, markdown=markdown
            )
            for source, priority in picked:
                if source.page in seen:
                    continue
                seen.add(source.page)
                chosen.append(
                    (priority, _image_page(source, label, markdown))
                )
            # A signature, seal, or execution date is on the closing page.
            # Keep that page even when the selector prefers the first page.
            closing = max(group, key=lambda source: (source.local_page, source.page))
            if closing.page not in seen:
                seen.add(closing.page)
                chosen.append((120, _image_page(closing, label, markdown)))
    # The closing page carries the drafting, filing, and execution date.
    # When only four images fit, keep that page ahead of an earlier page.
    chosen.sort(key=lambda item: (-item[0], -item[1].local_page, item[1].page))
    kept = [page for _priority, page in chosen[:MAX_DEFECT_IMAGES]]
    rest = [page for _priority, page in chosen[MAX_DEFECT_IMAGES:]]
    kept.sort(key=lambda page: (page.page, page.local_page))
    rest.sort(key=lambda page: (page.page, page.local_page))
    return DefectImagePlan(
        check_id=defect.check_id,
        pages=tuple(kept),
        leftover_pages=tuple(rest),
    )


def _image_page(
    source: PageSource,
    catalogue_name: str,
    markdown: Mapping[int, str] | None = None,
) -> DefectImagePage:
    concrete = _concrete_label(catalogue_name, source.document_parts)
    return DefectImagePage(
        page=source.page,
        local_page=source.local_page,
        slot_id=source.slot_id,
        document_type=concrete or catalogue_name,
        file_id=source.file_id,
        file_url=source.file_url,
        text=str((markdown or {}).get(source.page) or "")[:2500],
    )


def _sources_from_layout(
    layout: Mapping[int, Mapping[str, object]],
) -> list[PageSource]:
    sources: list[PageSource] = []
    for number in sorted(layout):
        page = layout.get(number)
        if not isinstance(page, Mapping):
            continue
        try:
            global_page = int(number)
        except (TypeError, ValueError):
            continue
        local_raw = page.get("local_page")
        try:
            local_page = int(local_raw) if local_raw is not None else global_page
        except (TypeError, ValueError):
            local_page = global_page
        sources.append(
            PageSource(
                page=global_page,
                slot_id=str(page.get("slot_id") or ""),
                local_page=local_page,
                document_parts=tuple(parts_on_page(page.get("document_part"))),
                file_url=str(page.get("file_url") or "").strip() or None,
                file_id=str(page.get("file_id") or "").strip() or None,
            )
        )
    return sources


def _markdown_from_layout(
    layout: Mapping[int, Mapping[str, object]],
) -> dict[int, str]:
    text: dict[int, str] = {}
    for number, page in layout.items():
        if not isinstance(page, Mapping):
            continue
        try:
            global_page = int(number)
        except (TypeError, ValueError):
            continue
        words = page.get("words") or []
        pieces: list[str] = []
        if isinstance(words, list):
            for word in words:
                if not isinstance(word, Mapping):
                    continue
                token = str(word.get("t") or word.get("text") or "").strip()
                if token:
                    pieces.append(token)
        text[global_page] = " ".join(pieces)
    return text


def _part_matches(catalogue_name: str, labels: tuple[str, ...]) -> bool:
    wanted = catalogue_name.casefold()
    family = family_split_name(catalogue_name).casefold()
    for label in labels:
        if label.casefold() == wanted:
            return True
        if family_split_name(label).casefold() == family:
            return True
    return False


def _concrete_label(catalogue_name: str, labels: tuple[str, ...]) -> str:
    for label in labels:
        if label.casefold() == catalogue_name.casefold():
            return label
    family = family_split_name(catalogue_name)
    for label in labels:
        if family_split_name(label) == family:
            return label
    return catalogue_name


def _groups_for_part(
    catalogue_name: str, sources: Sequence[PageSource]
) -> list[list[PageSource]]:
    """One group per filed document, in filing order.

    Annexures and Applications are one catalogue name and many files. Each
    file keeps its own selector.
    """
    buckets: dict[tuple[str, str], list[PageSource]] = {}
    for source in sources:
        if not _part_matches(catalogue_name, source.document_parts):
            continue
        label = _concrete_label(catalogue_name, source.document_parts)
        buckets.setdefault((source.slot_id, label), []).append(source)
    ordered = sorted(buckets, key=lambda key: buckets[key][0].page)
    return [buckets[key] for key in ordered]
