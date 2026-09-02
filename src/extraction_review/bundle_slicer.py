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
from .split_upload import UploadSlot, UploadTypeCatalog

COMBINED_VAKALATNAMA = "Vakalatnama + PoA/BR"


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
    for name in labels:
        if name in slot_parts:
            return True
        # Combined LlamaSplit label fills the Vakalatnama slot only.
        if name == COMBINED_VAKALATNAMA and "Vakalatnama" in slot_parts:
            return True
    return False


def map_slot_pages(
    catalog: UploadTypeCatalog,
    page_parts: Mapping[int, Sequence[str] | str | None],
) -> dict[str, list[int]]:
    """Map 1-indexed LlamaSplit pages onto catalog slots.

    A page is copied into every matching slot. Pages that match no slot are
    dropped. Combined ``Vakalatnama + PoA/BR`` maps to the Vakalatnama slot
    only.
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
            if _labels_match_slot(labels, slot):
                pages_by_slot[slot.id].append(number)
    return {
        slot_id: sorted(set(pages)) for slot_id, pages in pages_by_slot.items() if pages
    }


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
    """Cut the bundle into one PDF per catalog slot that LlamaSplit found."""
    pages_by_slot = map_slot_pages(catalog, page_parts)
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
