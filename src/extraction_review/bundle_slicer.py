"""Slice a bundled filing PDF into catalog-slot PDFs using LlamaSplit labels.

LlamaSplit only labels pages. This module is the code that actually cuts the
uploaded PDF into per-slot files the UI can show and Submit.
"""

from __future__ import annotations

import hashlib
import io
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Any

import pymupdf
from PIL import Image, ImageChops
from pypdf import PdfReader

from .document_parts import (
    ANNEXURE_FAMILY,
    APPLICATION_FAMILY,
    MAIN_PETITION_PART,
    family_split_name,
    numbered_part_slot_id,
    page_ranges,
    parts_on_page,
)
from .split_upload import (
    _ANNEXURE_SLOT_RE,
    _GENERIC_ANNEXURE_SLOT_RE,
    UNDEFINED_SLOT_ID,
    UploadSlot,
    UploadTypeCatalog,
    _numbered_slot_sort_key,
    resolve_upload_slot,
)

# MuPDF can rasterize a page with channel values a few levels apart after
# insert_pdf rewrites otherwise equivalent image/color-space resources. Exact
# byte hashes therefore reject valid slices. A delta above this small
# rasterization allowance still catches changed, missing, or swapped content.
_VISUAL_CHANNEL_TOLERANCE = 8


def _pixmaps_visually_equivalent(
    source: pymupdf.Pixmap,
    output: pymupdf.Pixmap,
) -> bool:
    """Compare RGB page renders while tolerating harmless raster rounding."""
    if (
        source.width != output.width
        or source.height != output.height
        or source.n != output.n
    ):
        return False
    if source.samples == output.samples:
        return True

    mode = "RGB" if source.n == 3 else "L" if source.n == 1 else None
    if mode is None:
        return False
    size = (source.width, source.height)
    source_image = Image.frombytes(mode, size, source.samples)
    output_image = Image.frombytes(mode, size, output.samples)
    extrema = ImageChops.difference(source_image, output_image).getextrema()
    if mode == "L":
        return extrema[1] <= _VISUAL_CHANNEL_TOLERANCE
    return all(high <= _VISUAL_CHANNEL_TOLERANCE for _low, high in extrema)


def _is_numbered_annexure_slot(slot_id: str) -> bool:
    return bool(
        _ANNEXURE_SLOT_RE.fullmatch(slot_id or "")
        or _GENERIC_ANNEXURE_SLOT_RE.fullmatch(slot_id or "")
    )


@dataclass(frozen=True)
class SlotSlice:
    slot_id: str
    label: str
    pages: tuple[int, ...]
    page_span: tuple[dict[str, int], ...]
    pdf_bytes: bytes
    filename: str
    file_hash: str


def _labels_match_slot(labels: Sequence[str], slot: UploadSlot) -> bool:
    slot_parts = set(slot.parts)
    return any(name in slot_parts for name in labels)


def _slot_ids_for_labels(labels: Sequence[str], catalog: UploadTypeCatalog) -> set[str]:
    ids: set[str] = set()
    for slot in catalog.slots:
        if slot.id == UNDEFINED_SLOT_ID:
            continue
        if _labels_match_slot(labels, slot):
            ids.add(slot.id)
    for name in labels:
        slot_id = numbered_part_slot_id(name)
        if slot_id:
            ids.add(slot_id)
        elif name == "Custody Certificate":
            ids.add("custody_certificate")
    if any(_is_numbered_annexure_slot(slot_id) for slot_id in ids):
        ids.discard("annexures")
    if any(
        slot_id.startswith("application_") and slot_id != "applications"
        for slot_id in ids
    ):
        ids.discard("applications")
    return ids


def map_slot_pages(
    catalog: UploadTypeCatalog,
    page_parts: Mapping[int, Sequence[str] | str | None],
    *,
    promote_repeatable: bool = True,
) -> dict[str, list[int]]:
    """Map 1-indexed LlamaSplit pages onto catalog slots.

    A page is copied into every matching slot. Pages that match no known slot
    are     leftover; ``slice_bundle_pdf`` puts them in the optional Undefined
    slot.     A leftover combined ``Vakalatnama + PoA/BR`` LlamaSplit label is
    expanded to Vakalatnama and PoA/BR so the combined Vakalatnama slot and
    the PoA/BR slot both receive those pages.

    Numbered annexures and applications (P-1…P-n, Application 1…n) become
    their own slots even when those ids are not listed in the catalog. A
    merged Annexures / Applications run with no P-n labels is promoted to
    P-1 / Application 1 so the UI row receives the sliced PDF.
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
        for slot_id in _slot_ids_for_labels(labels, catalog):
            pages_by_slot.setdefault(slot_id, []).append(number)
    mapped = {
        slot_id: sorted(set(pages)) for slot_id, pages in pages_by_slot.items() if pages
    }
    return _promote_repeatable_catchall(mapped) if promote_repeatable else mapped


def _map_repeatable_segments(
    pages_by_slot: dict[str, list[int]],
    segments: Sequence[Mapping[str, Any]],
    *,
    allowed_shared_pages: set[int] | None = None,
) -> dict[str, list[int]]:
    """Preserve remote document boundaries within the final page assignments.

    Reconciliation may change a page's category, give it a numbered identity,
    or remove a duplicate assignment. Raw segments supply boundaries only:
    they must never restore pages that the final map no longer assigns to the
    generic family. Explicit numbered identities remain authoritative.

    A generic page absent from matching remote segments is retained in its own
    contiguous run, rather than lost or attached to an unrelated remote
    document. Ordinal upload IDs do not assert printed exhibit numbers.
    Singleton/combined slots keep their configured catalog mapping.
    """
    mapped = dict(pages_by_slot)
    families = {
        ANNEXURE_FAMILY: ("annexures", "annexure"),
        APPLICATION_FAMILY: ("applications", "application"),
    }
    generic_pages = {
        family: set(mapped.pop(catchall, []))
        for family, (catchall, _prefix) in families.items()
    }
    assigned: dict[str, set[int]] = {family: set() for family in families}
    counters = {family: 0 for family in families}
    shared_pages = allowed_shared_pages or set()

    def add_segment(family: str, pages: list[int]) -> None:
        prefix = families[family][1]
        counters[family] += 1
        slot_id = f"{prefix}_{counters[family]}"
        while slot_id in mapped:
            counters[family] += 1
            slot_id = f"{prefix}_{counters[family]}"
        mapped[slot_id] = pages

    for segment in segments:
        labels = parts_on_page(segment.get("category"))
        if len(labels) != 1:
            continue
        family = family_split_name(labels[0])
        if family not in families:
            continue
        pages = sorted(
            {int(page) for page in segment.get("pages") or []} & generic_pages[family]
        )
        if not pages:
            continue
        repeated = set(pages) & assigned[family]
        unexpected_repeats = repeated - shared_pages
        if unexpected_repeats:
            # Deterministically retain the earlier physical segment as owner;
            # malformed/contradictory boundaries are separately flagged by
            # classification validation and must not duplicate output pages.
            pages = [page for page in pages if page not in unexpected_repeats]
        if not pages:
            continue
        add_segment(family, pages)
        assigned[family].update(pages)

    for family in families:
        remaining = sorted(generic_pages[family] - assigned[family])
        for span in page_ranges(remaining):
            add_segment(family, list(range(span["start"], span["end"] + 1)))
    return mapped


def validate_shared_page_evidence(
    page_parts: Mapping[int, Sequence[str] | str | None],
    segments: Sequence[Mapping[str, Any]],
    *,
    page_count: int,
) -> dict[str, Any]:
    """Authorize copies only when final labels and complete-segment evidence agree.

    An explicit multi-document page must be classified as exactly those
    document types on that physical page. An adjacent boundary share must be
    the sole overlap of two consecutive, contiguous classified segments at
    the left segment's end and the right segment's start. Every other overlap
    is unsupported and remains single-owner/review-needed.
    """
    issues: list[dict[str, Any]] = []
    normalized_parts: dict[int, list[str]] = {}
    for raw_page, labels in page_parts.items():
        if type(raw_page) is not int or not 1 <= raw_page <= page_count:
            issues.append({"code": "page_assignment_out_of_range", "page": raw_page})
            continue
        normalized_parts[raw_page] = parts_on_page(labels)

    segment_pages: list[list[int]] = []
    segment_labels: list[list[str]] = []
    page_segments: dict[int, list[int]] = {}
    for index, segment in enumerate(segments):
        raw_pages = segment.get("pages") if isinstance(segment, Mapping) else None
        category = segment.get("category") if isinstance(segment, Mapping) else None
        if (
            not isinstance(raw_pages, (list, tuple))
            or not raw_pages
            or any(type(page) is not int for page in raw_pages)
        ):
            issues.append({"code": "classified_segment_pages_invalid", "segment_index": index})
            segment_pages.append([])
            segment_labels.append([])
            continue
        pages = list(raw_pages)
        if (
            pages != sorted(set(pages))
            or any(page < 1 or page > page_count for page in pages)
            or pages != list(range(pages[0], pages[-1] + 1))
        ):
            issues.append({"code": "classified_segment_range_invalid", "segment_index": index})
            segment_pages.append([])
            segment_labels.append([])
            continue
        labels = parts_on_page(category)
        if not labels:
            issues.append({"code": "classified_segment_category_invalid", "segment_index": index})
            segment_pages.append([])
            segment_labels.append([])
            continue
        segment_pages.append(pages)
        segment_labels.append(labels)
        for page in pages:
            page_segments.setdefault(page, []).append(index)

    authorizations: dict[int, dict[str, Any]] = {}
    fatal_codes = {
        "page_assignment_out_of_range",
        "classified_segment_pages_invalid",
        "classified_segment_range_invalid",
    }
    candidate_pages = set(page_segments)
    candidate_pages.update(
        page for page, labels in normalized_parts.items() if len(labels) > 1
    )
    for page in sorted(candidate_pages):
        labels = normalized_parts.get(page, [])
        indexes = page_segments.get(page, [])
        if len(labels) > 1:
            singleton_matches = [
                index
                for index in indexes
                if segment_pages[index] == [page]
                and set(segment_labels[index]).issubset(set(labels))
            ]
            covered_labels = {
                label for index in singleton_matches for label in segment_labels[index]
            }
            if (
                covered_labels == set(labels)
                and len(singleton_matches) == len(indexes)
                and len(singleton_matches) <= 2
            ):
                authorizations[page] = {
                    "source_page": page,
                    "reason": "explicit_classified_multi_document_page",
                    "labels": labels,
                    "segment_indexes": singleton_matches,
                    "segment_categories": [segment_labels[i] for i in singleton_matches],
                    "segment_pages": [segment_pages[i] for i in singleton_matches],
                    "review_status": "needs_review",
                }
                continue

        if len(indexes) == 2 and indexes[1] == indexes[0] + 1:
            left, right = indexes
            left_pages, right_pages = segment_pages[left], segment_pages[right]
            category_labels = set(segment_labels[left]) | set(segment_labels[right])
            is_valid_edge = (
                left_pages
                and right_pages
                and left_pages[-1] == page
                and right_pages[0] == page
                and len(indexes) == 2
                and set(labels) == category_labels
            )
            if is_valid_edge:
                authorizations[page] = {
                    "source_page": page,
                    "reason": "adjacent_classified_segment_boundary",
                    "labels": labels,
                    "segment_indexes": [left, right],
                    "segment_categories": [segment_labels[left], segment_labels[right]],
                    "segment_pages": [left_pages, right_pages],
                    "review_status": "needs_review",
                }
                continue

        if len(labels) > 1:
            issues.append(
                {
                    "code": "multi_label_page_lacks_classification_evidence",
                    "source_page": page,
                    "labels": labels,
                    "segment_indexes": indexes,
                }
            )
        if len(indexes) > 1 and page not in authorizations:
            issues.append(
                {
                    "code": "unsupported_classification_overlap",
                    "source_page": page,
                    "segment_indexes": indexes,
                }
            )

    return {
        "status": "needs_review" if issues or authorizations else "clear",
        "fatal": any(issue.get("code") in fatal_codes for issue in issues),
        "authorizations": authorizations,
        "issues": issues,
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
    return [
        number for number in range(1, int(page_count) + 1) if number not in assigned
    ]


def _promote_repeatable_catchall(
    pages_by_slot: dict[str, list[int]],
) -> dict[str, list[int]]:
    """Map a lone Annexures/Applications blob onto P-1 / Application 1.

    The UI lists those under the Annexures and Applications headings as
    ``annexure_p1`` / ``application_1``. Keep the catch-all only when
    numbered siblings already exist (leftover "Other annexures").
    """
    promoted = dict(pages_by_slot)
    if "annexures" in promoted and not any(
        _is_numbered_annexure_slot(key) for key in promoted
    ):
        promoted["annexure_p1"] = promoted.pop("annexures")
    has_numbered_apps = any(
        key.startswith("application_") and key != "applications" for key in promoted
    )
    if "applications" in promoted and not has_numbered_apps:
        promoted["application_1"] = promoted.pop("applications")
    return promoted


def extract_pdf_pages(
    pdf_bytes: bytes,
    pages: Sequence[int],
    *,
    reader: PdfReader | None = None,
) -> bytes:
    """Copy 1-indexed pages without flattening or losing layered resources."""
    if not pages or not pdf_bytes:
        return b""
    # ``reader`` remains in the public signature for existing callers. Page
    # copying itself uses MuPDF because pypdf's add_page can lose inherited
    # form/image resources in scanned filings with layered OCR content.
    del reader
    with pymupdf.open(stream=pdf_bytes, filetype="pdf") as source:
        valid = [
            int(page) - 1 for page in pages if 0 <= int(page) - 1 < source.page_count
        ]
        if not valid:
            return b""
        with pymupdf.open() as output:
            for index in valid:
                output.insert_pdf(
                    source,
                    from_page=index,
                    to_page=index,
                    links=True,
                    annots=True,
                    widgets=True,
                )
            return output.tobytes(garbage=4, deflate=True)


def _pages_needing_family_text(
    normalized: Mapping[int, Sequence[str]],
) -> list[int]:
    """Annexure, application, Main Petition, and unlabeled pages need heading OCR."""
    families = {ANNEXURE_FAMILY, APPLICATION_FAMILY, MAIN_PETITION_PART}
    return [
        page
        for page, names in normalized.items()
        if not names or any(family_split_name(name) in families for name in names)
    ]


def pdf_page_texts(
    pdf_bytes: bytes,
    pages: Sequence[int] | None = None,
    *,
    reader: PdfReader | None = None,
) -> dict[int, str]:
    """Extract plain text for 1-indexed PDF pages (all pages when ``pages`` is None)."""
    if not pdf_bytes:
        return {}
    local = reader or PdfReader(io.BytesIO(pdf_bytes))
    total = len(local.pages)
    if pages is None:
        targets = range(1, total + 1)
    else:
        targets = pages
    texts: dict[int, str] = {}
    for page in targets:
        index = int(page) - 1
        if index < 0 or index >= total:
            continue
        try:
            texts[int(page)] = local.pages[index].extract_text() or ""
        except Exception:
            texts[int(page)] = ""
    return texts


# Older call sites / tests.
_pdf_page_texts = pdf_page_texts


def _slice_slot_order(
    catalog: UploadTypeCatalog, pages_by_slot: Mapping[str, Sequence[int]]
) -> list[str]:
    ordered: list[str] = []
    consumed: set[str] = set()
    annexure_ids = sorted(
        (slot_id for slot_id in pages_by_slot if _is_numbered_annexure_slot(slot_id)),
        key=_numbered_slot_sort_key,
    )
    application_ids = sorted(
        (
            slot_id
            for slot_id in pages_by_slot
            if slot_id.startswith("application_") and slot_id != "applications"
        ),
        key=lambda slot_id: int(slot_id.rsplit("_", 1)[-1]),
    )
    for slot in catalog.slots:
        if slot.id == "annexures":
            ordered.extend(annexure_ids)
            consumed.update(annexure_ids)
            ordered.append(slot.id)
            consumed.add(slot.id)
            continue
        if slot.id == "applications":
            ordered.extend(application_ids)
            consumed.update(application_ids)
            ordered.append(slot.id)
            consumed.add(slot.id)
            continue
        ordered.append(slot.id)
        consumed.add(slot.id)
    for slot_id in pages_by_slot:
        if slot_id not in consumed:
            ordered.append(slot_id)
    return ordered


def slice_bundle_pdf(
    pdf_bytes: bytes,
    catalog: UploadTypeCatalog,
    page_parts: Mapping[int, Sequence[str] | str | None],
    *,
    split_segments: Sequence[Mapping[str, Any]] | None = None,
) -> list[SlotSlice]:
    """Cut the bundle into one PDF per catalog slot that LlamaSplit found.

    Pages that match no known slot are copied into Undefined when that slot
    exists and leftover pages remain. Annexure P-n and Application n names
    are used as LlamaSplit returned them. Printed headings are not used to
    invent or renumber those labels. When raw segments are provided, generic
    Annexures/Application segments receive distinct ordinal upload IDs within
    the final page labels. Repaired assignments take precedence over raw
    categories; raw boundaries keep otherwise indistinguishable documents
    separate. The ordinals are not printed exhibit numbers.
    """
    reader = PdfReader(io.BytesIO(pdf_bytes)) if pdf_bytes else None
    page_count = len(reader.pages) if reader is not None else 0
    segments = list(split_segments or [])
    sharing = validate_shared_page_evidence(
        page_parts, segments, page_count=page_count
    )
    if sharing["fatal"]:
        raise ValueError(
            "Invalid page-sharing evidence: "
            + ", ".join(issue["code"] for issue in sharing["issues"])
        )
    authorized_pages = set(sharing["authorizations"])
    normalized: dict[int, list[str]] = {}
    for page, raw in page_parts.items():
        try:
            number = int(page)
        except (TypeError, ValueError):
            continue
        labels = parts_on_page(raw)
        if labels:
            if len(labels) > 1 and number not in authorized_pages:
                # Preserve one deterministic owner while keeping the rejected
                # authorization visible through validate_shared_page_evidence.
                labels = labels[:1]
            normalized[number] = labels
    pages_by_slot = dict(
        map_slot_pages(catalog, normalized, promote_repeatable=split_segments is None)
    )
    if split_segments is not None:
        pages_by_slot = _map_repeatable_segments(
            pages_by_slot,
            segments,
            allowed_shared_pages=authorized_pages,
        )
    if reader is not None:
        invalid_pages = {
            page
            for pages in pages_by_slot.values()
            for page in pages
            if page < 1 or page > len(reader.pages)
        }
        if split_segments is not None:
            invalid_pages.update(
                int(page)
                for segment in split_segments
                for page in segment.get("pages") or []
                if int(page) < 1 or int(page) > len(reader.pages)
            )
        if invalid_pages:
            raise ValueError(
                f"Split returned out-of-range PDF pages: {sorted(invalid_pages)}"
            )
    if reader is not None and any(
        slot.id == UNDEFINED_SLOT_ID for slot in catalog.slots
    ):
        leftover = leftover_pages(len(reader.pages), pages_by_slot)
        if leftover:
            pages_by_slot[UNDEFINED_SLOT_ID] = leftover
    slices: list[SlotSlice] = []
    for slot_id in _slice_slot_order(catalog, pages_by_slot):
        pages = pages_by_slot.get(slot_id) or []
        if not pages:
            continue
        slot = resolve_upload_slot(catalog, slot_id)
        if slot is None:
            continue
        chunk = extract_pdf_pages(pdf_bytes, pages, reader=reader)
        if not chunk:
            continue
        filename = f"{slot.label}.pdf"
        slices.append(
            SlotSlice(
                slot_id=slot.id,
                label=slot.label,
                pages=tuple(pages),
                page_span=tuple(page_ranges(pages)),
                pdf_bytes=chunk,
                filename=filename,
                file_hash=hashlib.sha256(chunk).hexdigest(),
            )
        )
    return slices


def slot_page_spans(
    slices: Sequence[SlotSlice],
) -> dict[str, list[dict[str, int]]]:
    return {item.slot_id: list(item.page_span) for item in slices if item.page_span}


def validate_sliced_page_ownership(
    slices: Sequence[SlotSlice],
    *,
    page_count: int,
    page_parts: Mapping[int, Sequence[str] | str | None],
    split_segments: Sequence[Mapping[str, Any]],
    catalog: UploadTypeCatalog,
) -> dict[str, Any]:
    """Check complete source coverage; only explicitly evidenced copies may share.

    A source page normally belongs to one output slot. A copy may appear in
    multiple slots only when a multi-document page label or a validated
    adjacent segment boundary explicitly authorized it.
    """
    owners: dict[int, list[str]] = {}
    for item in slices:
        if len(set(item.pages)) != len(item.pages):
            raise ValueError(f"Output slot {item.slot_id} repeats a source page")
        for page in item.pages:
            owners.setdefault(int(page), []).append(item.slot_id)

    invalid_pages = sorted(page for page in owners if not 1 <= page <= page_count)
    missing_pages = sorted(set(range(1, page_count + 1)) - set(owners))
    if invalid_pages or missing_pages:
        details = []
        if invalid_pages:
            details.append(f"out-of-range pages {invalid_pages}")
        if missing_pages:
            details.append(f"unassigned source pages {missing_pages}")
        raise ValueError("Invalid split output ownership: " + "; ".join(details))

    sharing = validate_shared_page_evidence(
        page_parts, split_segments, page_count=page_count
    )
    if sharing["fatal"]:
        raise ValueError(
            "Invalid page-sharing evidence: "
            + ", ".join(issue["code"] for issue in sharing["issues"])
        )
    authorizations = sharing["authorizations"]
    shared = []
    for page, slot_ids in sorted(owners.items()):
        actual_slots = sorted(set(slot_ids))
        if len(actual_slots) <= 1:
            continue
        authorization = authorizations.get(page)
        if authorization is None:
            continue
        if not _authorization_matches_slots(authorization, actual_slots, catalog):
            raise ValueError(
                f"Shared page {page} output slots do not match classified evidence"
            )
        shared.append(
            {
                **authorization,
                "source_page": page,
                "output_slots": actual_slots,
                "review_status": "needs_review",
            }
        )
    overlaps = [
        {"source_page": page, "output_slots": sorted(set(slot_ids))}
        for page, slot_ids in sorted(owners.items())
        if len(set(slot_ids)) > 1 and page not in authorizations
    ]
    if overlaps:
        details = ", ".join(
            f"page {item['source_page']} in {item['output_slots']}" for item in overlaps
        )
        raise ValueError(
            "Invalid split output ownership: source pages have multiple owners: "
            + details
        )
    return {
        "ownership_policy": "single_owner_with_explicit_shared_copies",
        "source_page_count": page_count,
        "accounted_page_count": len(owners),
        "missing_pages": [],
        "out_of_range_pages": [],
        "overlapping_ownership": overlaps,
        "shared_ownership": shared,
        "shared_pages_require_review": bool(shared),
        "sharing_validation": sharing,
        "review_status": "needs_review"
        if shared or sharing["issues"]
        else "clear",
    }


def _authorization_matches_slots(
    authorization: Mapping[str, Any],
    actual_slots: Sequence[str],
    catalog: UploadTypeCatalog,
) -> bool:
    categories = {
        label
        for category in authorization.get("segment_categories", ())
        for label in category
    }
    expected_slots: set[str] = set()
    generic_families: set[str] = set()
    for label in categories:
        family = family_split_name(label)
        if family in {ANNEXURE_FAMILY, APPLICATION_FAMILY}:
            generic_families.add(family)
        else:
            expected_slots.update(_slot_ids_for_labels([label], catalog))
    actual = set(actual_slots)
    if not generic_families:
        return actual == expected_slots
    for family in generic_families:
        prefix = "annexure_" if family == ANNEXURE_FAMILY else "application_"
        family_slots = {slot for slot in actual if slot.startswith(prefix)}
        expected_count = sum(
            any(family_split_name(label) == family for label in categories_for_segment)
            for categories_for_segment in authorization.get("segment_categories", ())
        )
        if len(family_slots) != expected_count:
            return False
        if family == ANNEXURE_FAMILY and any(
            not _is_numbered_annexure_slot(slot) for slot in family_slots
        ):
            return False
        if family == APPLICATION_FAMILY and any(
            not slot.startswith("application_") for slot in family_slots
        ):
            return False
        expected_slots.update(family_slots)
    return actual == expected_slots


def validate_page_manifest_output(
    page_manifest: Sequence[Mapping[str, Any]],
    slices: Sequence[SlotSlice],
    *,
    page_count: int,
    source_pdf: bytes,
) -> dict[str, Any]:
    """Require exact manifest/slot/range agreement and verify sliced page text."""
    errors: list[dict[str, Any]] = []
    rows_by_page: dict[int, list[Mapping[str, Any]]] = {}
    for row in page_manifest:
        page = row.get("source_page")
        if type(page) is not int:
            errors.append({"code": "manifest_page_number_invalid"})
            continue
        rows_by_page.setdefault(page, []).append(row)
    for page, rows in sorted(rows_by_page.items()):
        if page < 1 or page > page_count:
            errors.append({"code": "manifest_page_out_of_range", "page": page})
        if len(rows) > 1:
            errors.append({"code": "manifest_page_duplicated", "page": page})
    missing = sorted(set(range(1, page_count + 1)) - set(rows_by_page))
    if missing:
        errors.append({"code": "manifest_pages_missing", "pages": missing})

    source_reader = PdfReader(io.BytesIO(source_pdf))
    if len(source_reader.pages) != page_count:
        errors.append(
            {
                "code": "source_pdf_page_count_mismatch",
                "expected_page_count": page_count,
                "pdf_page_count": len(source_reader.pages),
            }
        )
    actual_slots: dict[int, list[str]] = {}
    with pymupdf.open(stream=source_pdf, filetype="pdf") as original_doc:
        for item in slices:
            if tuple(item.page_span) != tuple(page_ranges(item.pages)):
                errors.append(
                    {"code": "slice_page_span_mismatch", "slot_id": item.slot_id}
                )
            sliced_reader = PdfReader(io.BytesIO(item.pdf_bytes))
            sliced_page_count = len(sliced_reader.pages)
            if sliced_page_count != len(item.pages):
                errors.append(
                    {
                        "code": "slice_pdf_page_count_mismatch",
                        "slot_id": item.slot_id,
                        "metadata_page_count": len(item.pages),
                        "pdf_page_count": sliced_page_count,
                    }
                )
            for page in item.pages:
                actual_slots.setdefault(int(page), []).append(item.slot_id)
            with pymupdf.open(stream=item.pdf_bytes, filetype="pdf") as sliced_doc:
                for output_index, (source_page, output_page) in enumerate(
                    zip(item.pages, sliced_reader.pages, strict=False)
                ):
                    if not 1 <= source_page <= len(source_reader.pages):
                        continue
                    original_page = source_reader.pages[source_page - 1]
                    source_text = " ".join((original_page.extract_text() or "").split())
                    output_text = " ".join((output_page.extract_text() or "").split())
                    if source_text != output_text:
                        errors.append(
                            {
                                "code": "slice_source_content_mismatch",
                                "slot_id": item.slot_id,
                                "source_page": source_page,
                            }
                        )
                    original_pixmap = original_doc[source_page - 1].get_pixmap(
                        matrix=pymupdf.Matrix(1.5, 1.5),
                        colorspace=pymupdf.csRGB,
                        alpha=False,
                    )
                    output_pixmap = sliced_doc[output_index].get_pixmap(
                        matrix=pymupdf.Matrix(1.5, 1.5),
                        colorspace=pymupdf.csRGB,
                        alpha=False,
                    )
                    if not _pixmaps_visually_equivalent(
                        original_pixmap, output_pixmap
                    ):
                        errors.append(
                            {
                                "code": "slice_source_visual_mismatch",
                                "slot_id": item.slot_id,
                                "source_page": source_page,
                            }
                        )

    for page in range(1, page_count + 1):
        rows = rows_by_page.get(page) or []
        if not rows:
            continue
        expected_slots = sorted(set(actual_slots.get(page, [])))
        for row in rows:
            recorded_slots = row.get("output_slots")
            if (
                not isinstance(recorded_slots, list)
                or any(not isinstance(slot, str) for slot in recorded_slots)
                or sorted(set(recorded_slots)) != expected_slots
            ):
                errors.append(
                    {
                        "code": "manifest_output_slots_mismatch",
                        "page": page,
                        "expected_slots": expected_slots,
                    }
                )
    return {"valid": not errors, "errors": errors}
