"""Slice a bundled filing PDF into catalog-slot PDFs using LlamaSplit labels.

LlamaSplit only labels pages. This module is the code that actually cuts the
uploaded PDF into per-slot files the UI can show and Submit.
"""

from __future__ import annotations

import hashlib
import io
from collections.abc import Mapping, Sequence
from dataclasses import dataclass

from pypdf import PdfReader, PdfWriter

from .document_parts import format_page_span, parts_on_page
from .split_upload import UNDEFINED_SLOT_ID, UploadSlot, UploadTypeCatalog


@dataclass(frozen=True)
class SlotSlice:
    slot_id: str
    label: str
    pages: tuple[int, ...]
    page_span: str
    pdf_bytes: bytes
    filename: str
    file_hash: str


def _labels_match_slot(labels: Sequence[str], slot: UploadSlot) -> bool:
    slot_parts = set(slot.parts)
    return any(name in slot_parts for name in labels)


def map_slot_pages(
    catalog: UploadTypeCatalog,
    page_parts: Mapping[int, Sequence[str] | str | None],
) -> dict[str, list[int]]:
    """Map 1-indexed LlamaSplit pages onto catalog slots.

    A page is copied into every matching slot. Pages that match no known slot
    are leftover; ``slice_bundle_pdf`` puts them in the optional Undefined
    slot. A leftover combined ``Vakalatnama + PoA/BR`` LlamaSplit label is
    expanded to Vakalatnama and PoA/BR so both slots receive those pages.
    """
    pages_by_slot: dict[str, list[int]] = {slot.id: [] for slot in catalog.slots}
    for page, raw_labels in page_parts.items():
        try:
            number = int(page)
        except (TypeError, ValueError):
            continue
        labels = parts_on_page(raw_labels)
        if not labels:
            continue
        for slot in catalog.slots:
            if slot.id == UNDEFINED_SLOT_ID:
                continue
            if _labels_match_slot(labels, slot):
                pages_by_slot[slot.id].append(number)
    return {
        slot_id: sorted(set(pages)) for slot_id, pages in pages_by_slot.items() if pages
    }


def leftover_pages(
    page_count: int,
    pages_by_slot: Mapping[str, Sequence[int]],
) -> list[int]:
    """1-indexed PDF pages that were not copied into any known document slot."""
    assigned: set[int] = set()
    for slot_id, pages in pages_by_slot.items():
        if slot_id == UNDEFINED_SLOT_ID:
            continue
        for page in pages:
            try:
                assigned.add(int(page))
            except (TypeError, ValueError):
                continue
    return [number for number in range(1, int(page_count) + 1) if number not in assigned]


def extract_pdf_pages(pdf_bytes: bytes, pages: Sequence[int]) -> bytes:
    """Copy 1-indexed pages into a new PDF. pypdf indexes pages from 0."""
    if not pages or not pdf_bytes:
        return b""
    reader = PdfReader(io.BytesIO(pdf_bytes))
    writer = PdfWriter()
    total = len(reader.pages)
    added = 0
    for page in pages:
        index = int(page) - 1
        if index < 0 or index >= total:
            continue
        writer.add_page(reader.pages[index])
        added += 1
    if added == 0:
        return b""
    buffer = io.BytesIO()
    writer.write(buffer)
    return buffer.getvalue()


def slice_bundle_pdf(
    pdf_bytes: bytes,
    catalog: UploadTypeCatalog,
    page_parts: Mapping[int, Sequence[str] | str | None],
) -> list[SlotSlice]:
    """Cut the bundle into one PDF per catalog slot that LlamaSplit found.

    Pages that match no known slot are copied into Undefined when that slot
    exists and leftover pages remain.
    """
    pages_by_slot = dict(map_slot_pages(catalog, page_parts))
    if pdf_bytes and any(slot.id == UNDEFINED_SLOT_ID for slot in catalog.slots):
        leftover = leftover_pages(
            len(PdfReader(io.BytesIO(pdf_bytes)).pages), pages_by_slot
        )
        if leftover:
            pages_by_slot[UNDEFINED_SLOT_ID] = leftover
    slices: list[SlotSlice] = []
    for slot in catalog.slots:
        pages = pages_by_slot.get(slot.id) or []
        if not pages:
            continue
        chunk = extract_pdf_pages(pdf_bytes, pages)
        if not chunk:
            continue
        filename = f"{slot.label}.pdf"
        slices.append(
            SlotSlice(
                slot_id=slot.id,
                label=slot.label,
                pages=tuple(pages),
                page_span=format_page_span(pages),
                pdf_bytes=chunk,
                filename=filename,
                file_hash=hashlib.sha256(chunk).hexdigest(),
            )
        )
    return slices


def slot_page_spans(slices: Sequence[SlotSlice]) -> dict[str, str]:
    return {item.slot_id: item.page_span for item in slices if item.page_span}
