"""Structure-aware compiled PDF split: page units → classify → boundaries → slice.

Legal paper-books are not split by generic chunking. Flow:

    PDF
      → page extraction (text + layout cues + OCR-need flag)
      → page classification (controlled taxonomy + confidence)
      → boundary detection (multi-signal scores)
      → logical documents (page ranges; source PDF stays intact)
      → hybrid repair (annexure nesting / gap fill)
      → physical slice (bundle_slicer)

LlamaSplit is a warm-start hint, not the sole boundary authority.
"""

from __future__ import annotations

import io
import re
from collections.abc import Callable, Mapping, Sequence
from dataclasses import asdict, dataclass, field
from typing import Any

from pypdf import PdfReader

from .document_parts import (
    ANNEXURE_FAMILY,
    MAIN_PETITION_PART,
    PagePartMap,
    annexure_label_from_text,
    family_split_name,
    normalize_part_name,
    page_ranges,
    page_starts_application,
    parts_on_page,
)
from .split_ocr import ocr_sparse_pages, pages_with_large_images
from .split_pdf_layout import extract_split_layout
from .split_repair import (
    _apply_cited_annexure_ranges,
    _apply_indexed_annexure_ranges,
    _apply_indexed_outer_document_ranges,
    _heading_window,
    _is_lower_court_caption,
    _is_near_blank_page,
    _looks_like_sci_main_petition,
    _outer_anchor_label,
    _restore_index_bridge_pages,
    _restore_master_volume_sections,
    repair_compiled_split,
)

# Controlled taxonomy (canonical Split part names — do not invent aliases).
DOCUMENT_TYPES: tuple[str, ...] = (
    "Cover Page",
    "Index",
    "Advocate's Checklist",
    "Office Report on Limitation",
    "Listing Proforma",
    "Record of Proceedings",
    "Synopsis",
    "List of Dates & Events",
    "Impugned Order",
    MAIN_PETITION_PART,
    "AOR's Declaration",
    "AOR's Certificate",
    "Affidavit",
    "Appendix",
    "Filing Memo",
    "Custody Certificate",
    "Vakalatnama",
    "Memo of Appearance",
    "Memo of Parties",
    "Application 1",
    "Annexure P-1",
    "other",
)

_AUTO_SPLIT_THRESHOLD = 0.75
_VERIFY_THRESHOLD = 0.55
_OCR_TEXT_MIN = 50
_SUSPICIOUS_CONTROL_RE = re.compile(r"[\x00-\x08\x0b\x0c\x0e-\x1f\x7f]")

_ANNEXURE_HEADING_RE = re.compile(
    r"(?m)^\s*(?:annexure|exhibit)\s*[-–—:]?\s*[a-z]?\s*-?\s*p?-?\s*\d+\b",
    re.I,
)
_CENTERED_TITLE_RE = re.compile(
    r"(?m)^\s{0,40}(ANNEXURE|EXHIBIT|VAKALATNAMA|AFFIDAVIT|SYNOPSIS|"
    r"LIST OF DATES|MEMO OF PARTIES|FILING MEMO|IMPUGNED)\b",
    re.I,
)


@dataclass(frozen=True)
class PageUnit:
    """Immutable PDF page unit — logical docs are ranges over these."""

    pdf_page: int
    text: str
    requires_ocr: bool = False
    char_count: int = 0
    printed_page: str | None = None
    header: str = ""
    footer: str = ""
    source_pdf: str = "bundle.pdf"
    native_text: str = ""
    ocr_text: str = ""
    extraction_method: str = "native"
    text_quality: str = "usable"
    headings: tuple[str, ...] = ()
    paragraph_starts: tuple[str, ...] = ()
    sentence_continues_previous: bool = False
    paragraph_continues_previous: bool = False
    signature_detected: bool = False
    closing_section: str | None = None
    annexure_labels: tuple[str, ...] = ()
    layout_fingerprint: tuple[str, ...] = ()

    def as_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class PageClassification:
    page: int
    document_type: str
    confidence: float
    signals: tuple[str, ...] = ()
    llama_hint: str | None = None

    def as_dict(self) -> dict[str, Any]:
        return {
            "page": self.page,
            "document_type": self.document_type,
            "confidence": round(self.confidence, 3),
            "signals": list(self.signals),
            "llama_hint": self.llama_hint,
        }


@dataclass(frozen=True)
class BoundaryCandidate:
    """Proposed start of a new logical document at ``page``."""

    page: int
    document_type: str
    score: float
    signals: tuple[str, ...] = ()
    action: str = "auto"  # auto | verify | skip

    def as_dict(self) -> dict[str, Any]:
        return {
            "page": self.page,
            "document_type": self.document_type,
            "score": round(self.score, 3),
            "signals": list(self.signals),
            "action": self.action,
        }


@dataclass(frozen=True)
class LlamaBoundaryProposal:
    """A boundary proposal derived from LlamaSplit's segment transitions."""

    page: int
    previous_labels: tuple[str, ...] = ()
    proposed_labels: tuple[str, ...] = ()

    def as_dict(self) -> dict[str, Any]:
        return {
            "page": self.page,
            "previous_labels": list(self.previous_labels),
            "proposed_labels": list(self.proposed_labels),
            "source": "llama_split_segment_transition",
        }


@dataclass(frozen=True)
class SegmentClassification:
    """Classification evidence for a complete, contiguous proposed segment."""

    segment_id: str
    start_page: int
    end_page: int
    document_type: str
    confidence: float
    signals: tuple[str, ...] = ()

    def as_dict(self) -> dict[str, Any]:
        return {
            "segment_id": self.segment_id,
            "start_page": self.start_page,
            "end_page": self.end_page,
            "document_type": self.document_type,
            "confidence": round(self.confidence, 3),
            "signals": list(self.signals),
        }


@dataclass(frozen=True)
class LogicalDocument:
    """Confirmed page range — physical PDF is created only after this exists."""

    document_id: str
    document_type: str
    start_page: int
    end_page: int
    confidence: float
    signals: tuple[str, ...] = ()

    def as_dict(self) -> dict[str, Any]:
        return {
            "document_id": self.document_id,
            "type": self.document_type,
            "start_page": self.start_page,
            "end_page": self.end_page,
            "confidence": round(self.confidence, 3),
            "signals": list(self.signals),
        }


@dataclass(frozen=True)
class ReconciliationChange:
    """Internal provenance for one committed targeted page correction."""

    page: int
    old_labels: tuple[str, ...]
    new_labels: tuple[str, ...]
    evidence: tuple[str, ...]

    def as_dict(self) -> dict[str, Any]:
        return {
            "page": self.page,
            "old_labels": list(self.old_labels),
            "new_labels": list(self.new_labels),
            "evidence": list(self.evidence),
        }


@dataclass(frozen=True)
class ReconciliationOperation:
    operation_id: str
    operation_type: str
    original_pages: tuple[int, ...]
    proposed_pages: tuple[int, ...]
    original_labels: tuple[str, ...]
    proposed_labels: tuple[str, ...]
    original_segment_range: tuple[int, int] | None
    proposed_segment_range: tuple[int, int] | None
    evidence: tuple[str, ...]
    classification_before: str | None
    classification_after: str | None
    validation: dict[str, Any]
    status: str
    reason: str | None = None
    pass_index: int = 1

    def as_dict(self) -> dict[str, Any]:
        return {
            "operation_id": self.operation_id,
            "operation_type": self.operation_type,
            "original_page_ranges": page_ranges(self.original_pages),
            "proposed_page_ranges": page_ranges(self.proposed_pages),
            "original_labels": list(self.original_labels),
            "proposed_labels": list(self.proposed_labels),
            "original_segment_range": (
                {"start": self.original_segment_range[0], "end": self.original_segment_range[1]}
                if self.original_segment_range
                else None
            ),
            "proposed_segment_range": (
                {"start": self.proposed_segment_range[0], "end": self.proposed_segment_range[1]}
                if self.proposed_segment_range
                else None
            ),
            "evidence": list(self.evidence),
            "classification_before": self.classification_before,
            "classification_after": self.classification_after,
            "validation": dict(self.validation),
            "status": self.status,
            "reason": self.reason,
            "pass_index": self.pass_index,
        }


@dataclass
class StructureSplitResult:
    page_units: list[PageUnit]
    classifications: list[PageClassification]
    boundaries: list[BoundaryCandidate]
    logical_documents: list[LogicalDocument]
    page_parts: PagePartMap
    duplicates: list[Any] = field(default_factory=list)
    ocr_needed_pages: list[int] = field(default_factory=list)
    # Kept internal so existing split artifact/API schemas remain unchanged.
    reconciliation_changes: list[ReconciliationChange] = field(default_factory=list)
    llama_boundary_proposals: list[LlamaBoundaryProposal] = field(default_factory=list)
    segment_classifications: list[SegmentClassification] = field(default_factory=list)
    reconciliation_operations: list[ReconciliationOperation] = field(default_factory=list)
    reconciliation_summary: dict[str, Any] = field(default_factory=dict)

    def report(self) -> dict[str, Any]:
        return {
            "architecture": "structure_aware_v1",
            "page_count": len(self.page_units),
            "ocr_needed_pages": list(self.ocr_needed_pages),
            "ocr_needed_count": len(self.ocr_needed_pages),
            "classifications": [c.as_dict() for c in self.classifications],
            "boundaries": [b.as_dict() for b in self.boundaries],
            "llama_boundary_proposals": [
                item.as_dict() for item in self.llama_boundary_proposals
            ],
            "segment_classifications": [
                item.as_dict() for item in self.segment_classifications
            ],
            "reconciliation_operations": [
                item.as_dict() for item in self.reconciliation_operations
            ],
            "reconciliation_summary": dict(self.reconciliation_summary),
            "logical_documents": [d.as_dict() for d in self.logical_documents],
            "auto_boundaries": sum(1 for b in self.boundaries if b.action == "auto"),
            "verify_boundaries": sum(
                1 for b in self.boundaries if b.action == "verify"
            ),
        }


def extract_page_units(
    pdf_bytes: bytes,
    *,
    source_pdf: str = "bundle.pdf",
    page_texts: Mapping[int, str] | None = None,
) -> list[PageUnit]:
    """Extract immutable page units. Prefer supplied texts; enrich via PyMuPDF."""
    if not pdf_bytes:
        return []
    reader = PdfReader(io.BytesIO(pdf_bytes))
    total = len(reader.pages)
    layout = extract_split_layout(pdf_bytes)
    sparse_pages = {
        number
        for number in range(1, total + 1)
        if max(
            len(layout.get(number, ("", None, None))[0].strip()),
            len(((page_texts or {}).get(number) or "").strip()),
        )
        < _OCR_TEXT_MIN
    }
    image_pages = pages_with_large_images(pdf_bytes)
    ocr_pages = sorted(sparse_pages | image_pages)
    ocr_texts = ocr_sparse_pages(pdf_bytes, ocr_pages)
    units: list[PageUnit] = []
    for number in range(1, total + 1):
        text = layout.get(number, ("", None, None))[0]
        if page_texts is not None and number in page_texts:
            text = page_texts.get(number) or ""
        if not (text or "").strip():
            try:
                text = reader.pages[number - 1].extract_text() or ""
            except Exception:
                text = ""
        alt, index_table, folio = layout.get(number, ("", None, None))
        # Some PDFs use a broken character map for digits.  The upstream text
        # can then be longer than the layout extraction but contain C0 control
        # bytes where the visible date is printed (for example
        # ``Dated:\x15\x14.\x13\x14.202\x19``).  Prefer the clean layout text in
        # that case; length alone would preserve the corrupt/wrong date.
        if len(alt.strip()) > len((text or "").strip()) or (
            _SUSPICIOUS_CONTROL_RE.search(text or "")
            and not _SUSPICIOUS_CONTROL_RE.search(alt)
            and len(alt.strip()) >= len((text or "").strip()) * 0.5
        ):
            text = alt
        native_text = (text or "").strip()
        ocr_text = ocr_texts.get(number, "").strip()
        if ocr_text:
            native_stamp = annexure_label_from_text(text)
            ocr_stamp = annexure_label_from_text(ocr_text)
            native_anchor = _outer_anchor_label(text)
            ocr_anchor = _outer_anchor_label(ocr_text)
            # Scanned Index tables can have a stale/native text layer that
            # reads every column vertically.  OCR is only slightly longer in
            # that case, but it restores serial-numbered rows and their page
            # spans. Prefer that row-shaped OCR so the Index can drive ranges.
            native_index_rows = len(re.findall(r"(?m)^\s*\d{1,3}[.)|]\s+", text))
            ocr_index_rows = len(re.findall(r"(?m)^\s*\d{1,3}[.)|]\s+", ocr_text))
            ocr_index_spans = len(
                re.findall(
                    r"(?i)(?<![A-Za-z0-9])(?:A\d{1,2}|[A-Z]{1,2}|\d{1,4})"
                    r"\s*[-–—]\s*(?:A\d{1,2}|[A-Z]{1,2}|\d{1,4})(?![A-Za-z0-9])",
                    ocr_text,
                )
            )
            ocr_has_better_index = bool(
                re.search(r"(?mi)^\s*index\s*$", ocr_text)
                and "particulars" in ocr_text.casefold()
                and ocr_index_rows >= 3
                and ocr_index_spans >= 3
                and ocr_index_rows > native_index_rows
            )
            if ocr_has_better_index:
                text = ocr_text
            elif len(ocr_text) > len(text.strip()) * 1.15:
                text = (text.rstrip() + "\n" if native_stamp else "") + ocr_text
            elif (ocr_stamp and not native_stamp) or (ocr_anchor and not native_anchor):
                # Preserve exact born-digital text while making a heading that
                # exists only inside the inserted scan visible to repair.
                text = ocr_text + "\n" + text.lstrip()
        if index_table:
            text = index_table
        elif folio and text.rstrip().splitlines()[-1:] != [folio]:
            # Put the geometrically verified margin folio last for repair's
            # text-only reader; stamps and body text keep their full content.
            text = text.rstrip() + "\n" + folio
        stripped = (text or "").strip()
        requires_ocr = len(stripped) < _OCR_TEXT_MIN
        extraction_method = (
            "native+ocr"
            if native_text and ocr_text
            else "ocr"
            if ocr_text
            else "native"
            if native_text
            else "unavailable"
        )
        text_quality = (
            "empty"
            if not stripped
            else "sparse"
            if requires_ocr
            else "usable"
        )
        lines = [line.strip() for line in (text or "").splitlines() if line.strip()]
        header = "\n".join(lines[:3])
        footer = "\n".join(lines[-2:]) if len(lines) > 3 else ""
        headings = tuple(
            line[:180]
            for line in lines[:12]
            if _CENTERED_TITLE_RE.search(line) or line.isupper() and 4 <= len(line) <= 100
        )[:5]
        paragraph_starts = tuple(
            match.group(0)[:80]
            for line in lines
            if (
                match := re.match(
                    r"\s*(?:\(?\d{1,3}[.)]|[a-z][.)]|\([a-z]\))\s+",
                    line,
                    re.IGNORECASE,
                )
            )
        )[:12]
        lower_text = stripped.casefold()
        signature_detected = bool(
            re.search(r"\b(?:signed|signature|deponent|advocate|aor)\b", lower_text)
            and re.search(r"\b(?:place|dated|date|deponent|advocate|aor)\b", lower_text)
        )
        closing_match = re.search(
            r"(?i)\b(prayer|wherefore|verification|solemnly affirmed|sworn|deponent)\b",
            stripped[-900:],
        )
        annexure_labels = tuple(
            dict.fromkeys(
                f"Annexure {match.group(1).upper()}-{match.group(2)}"
                for match in re.finditer(
                    r"(?i)\bannexure\s*[-–—:]?\s*([a-z])\s*[-–—/]\s*(\d+)\b",
                    stripped,
                )
            )
        )
        layout_fingerprint = tuple(
            name
            for name, pattern in (
                ("table", r"\S.{0,60}\s{2,}\S"),
                ("numbered_paragraphs", r"(?m)^\s*\(?\d{1,3}[.)]"),
                ("signature_block", r"(?i)\b(?:signature|signed|deponent)\b"),
                ("all_caps_heading", r"(?m)^\s*[A-Z][A-Z .,/()'-]{5,}$"),
            )
            if re.search(pattern, stripped)
        )
        printed = folio
        if not printed and lines and re.fullmatch(r"\d{1,4}[A-Za-z]?", lines[-1]):
            printed = lines[-1]
        units.append(
            PageUnit(
                pdf_page=number,
                text=text or "",
                requires_ocr=requires_ocr,
                char_count=len(stripped),
                printed_page=printed,
                header=header,
                footer=footer,
                source_pdf=source_pdf,
                native_text=native_text,
                ocr_text=ocr_text,
                extraction_method=extraction_method,
                text_quality=text_quality,
                headings=headings,
                paragraph_starts=paragraph_starts,
                sentence_continues_previous=bool(
                    number > 1
                    and stripped
                    and not re.match(r"\s*(?:[A-Z][A-Z\s]{5,}|\d+[.)])", stripped)
                    and not re.search(r"[.!?][\"')\]]?\s*$", (units[-1].text or "").strip())
                ),
                paragraph_continues_previous=bool(
                    number > 1
                    and stripped
                    and not paragraph_starts
                    and not headings
                ),
                signature_detected=signature_detected,
                closing_section=closing_match.group(1).casefold() if closing_match else None,
                annexure_labels=annexure_labels,
                layout_fingerprint=layout_fingerprint,
            )
        )
    return units


def _normalize_taxonomy(label: str | None) -> str:
    if not label:
        return "other"
    name = normalize_part_name(label)
    if name in DOCUMENT_TYPES:
        return name
    family = family_split_name(name)
    if family == ANNEXURE_FAMILY:
        return name if name.startswith("Annexure") else "Annexure P-1"
    if family == "Application":
        return name if name.startswith("Application") else "Application 1"
    if name == MAIN_PETITION_PART:
        return MAIN_PETITION_PART
    return name or "other"


def classify_page(
    unit: PageUnit,
    *,
    llama_hint: str | None = None,
) -> PageClassification:
    """Stage A — cheap per-page classification with confidence."""
    text = unit.text or ""
    signals: list[str] = []
    score = 0.0
    doc_type = "other"

    if unit.requires_ocr and not text.strip():
        if llama_hint:
            return PageClassification(
                page=unit.pdf_page,
                document_type=_normalize_taxonomy(llama_hint),
                confidence=0.55,
                signals=("ocr_needed", "llama_hint"),
                llama_hint=llama_hint,
            )
        return PageClassification(
            page=unit.pdf_page,
            document_type="other",
            confidence=0.2,
            signals=("ocr_needed", "empty_text"),
            llama_hint=llama_hint,
        )

    mark_label = annexure_label_from_text(text)
    if mark_label:
        doc_type = mark_label
        score = 0.95
        signals.append("annexure_stamp")
    else:
        anchor = _outer_anchor_label(text)
        if anchor:
            doc_type = anchor
            score = 0.88
            signals.append("outer_anchor")
            if _CENTERED_TITLE_RE.search(_heading_window(text, lines=8)):
                score = min(0.97, score + 0.07)
                signals.append("centered_title")
        elif _looks_like_sci_main_petition(text):
            doc_type = MAIN_PETITION_PART
            score = 0.9
            signals.append("form28_lookalike")
        elif page_starts_application(text) and not _is_lower_court_caption(text):
            doc_type = "Application 1"
            score = 0.85
            signals.append("application_start")
        elif _is_near_blank_page(text):
            doc_type = "other"
            score = 0.35
            signals.append("near_blank")
        elif llama_hint:
            doc_type = _normalize_taxonomy(llama_hint)
            score = 0.5
            signals.append("llama_hint_only")
        else:
            doc_type = "other"
            score = 0.25
            signals.append("no_strong_cue")

    if llama_hint:
        hint = _normalize_taxonomy(llama_hint)
        if hint == _normalize_taxonomy(doc_type):
            score = min(0.99, score + 0.05)
            signals.append("llama_agree")
        elif doc_type == "other" and hint != "other":
            doc_type = hint
            score = max(score, 0.58)
            signals.append("llama_fill")
        elif score < 0.7 and hint != "other":
            # Weak local cue — keep local type but note disagreement.
            signals.append(f"llama_disagree:{hint}")

    return PageClassification(
        page=unit.pdf_page,
        document_type=_normalize_taxonomy(doc_type),
        confidence=round(min(0.99, score), 3),
        signals=tuple(signals),
        llama_hint=llama_hint,
    )


def classify_pages(
    units: Sequence[PageUnit],
    *,
    llama_page_parts: PagePartMap
    | Mapping[int, Sequence[str] | str | None]
    | None = None,
) -> list[PageClassification]:
    hints: dict[int, str] = {}
    if llama_page_parts:
        for page, names in llama_page_parts.items():
            parts = parts_on_page(names)
            if parts:
                hints[int(page)] = parts[0]
    return [classify_page(unit, llama_hint=hints.get(unit.pdf_page)) for unit in units]


def propose_llama_boundaries(
    llama_page_parts: PagePartMap
    | Mapping[int, Sequence[str] | str | None]
    | None,
    *,
    page_count: int,
) -> list[LlamaBoundaryProposal]:
    """Record LlamaSplit segment transitions before local classification runs.

    These are proposals, not final ownership decisions. Missing/unclassified
    pages are represented as empty labels so a gap cannot hide a transition.
    """
    if page_count < 1:
        return []
    labels_by_page = {
        page: tuple(parts_on_page((llama_page_parts or {}).get(page)))
        for page in range(1, page_count + 1)
    }
    proposals: list[LlamaBoundaryProposal] = []
    previous: tuple[str, ...] | None = None
    for page in range(1, page_count + 1):
        labels = labels_by_page[page]
        if page == 1 or labels != previous:
            proposals.append(
                LlamaBoundaryProposal(
                    page=page,
                    previous_labels=previous or (),
                    proposed_labels=labels,
                )
            )
        previous = labels
    return proposals


def classify_complete_segments(
    page_parts: Mapping[int, Sequence[str] | str | None],
    units: Sequence[PageUnit],
) -> list[SegmentClassification]:
    """Classify each complete contiguous assigned segment from its full text."""
    units_by_page = {unit.pdf_page: unit for unit in units}
    pages = sorted(page_parts)
    result: list[SegmentClassification] = []
    index = 0
    number = 0
    while index < len(pages):
        start = pages[index]
        labels = tuple(parts_on_page(page_parts.get(start)))
        end_index = index
        while end_index + 1 < len(pages):
            next_page = pages[end_index + 1]
            if next_page != pages[end_index] + 1:
                break
            if tuple(parts_on_page(page_parts.get(next_page))) != labels:
                break
            end_index += 1
        end = pages[end_index]
        text = "\n".join(
            units_by_page[page].text
            for page in range(start, end + 1)
            if page in units_by_page
        )
        evidence = classify_page(PageUnit(pdf_page=start, text=text))
        number += 1
        result.append(
            SegmentClassification(
                segment_id=f"segment_{number:04d}",
                start_page=start,
                end_page=end,
                document_type=evidence.document_type,
                confidence=evidence.confidence,
                signals=evidence.signals,
            )
        )
        index = end_index + 1
    return result


def _boundary_score(
    prev: PageClassification | None,
    curr: PageClassification,
    unit: PageUnit,
    prev_unit: PageUnit | None,
) -> tuple[float, list[str]]:
    """Multi-signal boundary score for starting a new doc at ``curr.page``."""
    if prev is None:
        return 1.0, ["document_start"]

    score = 0.0
    signals: list[str] = []
    prev_type = _normalize_taxonomy(prev.document_type)
    curr_type = _normalize_taxonomy(curr.document_type)

    if curr_type != prev_type and curr_type != "other":
        score += 0.35
        signals.append("heading_change")
    if curr.confidence >= 0.85 and curr_type != "other":
        score += 0.30
        signals.append("strong_classification")
    if "annexure_stamp" in curr.signals or _ANNEXURE_HEADING_RE.search(
        _heading_window(unit.text, lines=10)
    ):
        score += 0.20
        signals.append("annexure_marker")
    if "centered_title" in curr.signals or _CENTERED_TITLE_RE.search(
        _heading_window(unit.text, lines=6)
    ):
        score += 0.15
        signals.append("layout_title")
    if prev_unit and unit.requires_ocr != prev_unit.requires_ocr:
        score += 0.05
        signals.append("ocr_transition")
    if prev_unit and unit.printed_page and prev_unit.printed_page:
        try:
            if int(unit.printed_page) < int(prev_unit.printed_page):
                score += 0.05
                signals.append("page_number_reset")
        except ValueError:
            pass
    # Continuation of same type is not a boundary.
    if curr_type == prev_type and curr_type != "other":
        score = min(score, 0.35)
        signals.append("same_type_continuation")
    # Weak "other" after a labeled page — do not split.
    if curr_type == "other" and prev_type != "other":
        score = min(score, 0.4)
        signals.append("weak_other")
    return score, signals


def detect_boundaries(
    classifications: Sequence[PageClassification],
    units: Sequence[PageUnit],
    *,
    llama_boundary_proposals: Sequence[LlamaBoundaryProposal] = (),
) -> list[BoundaryCandidate]:
    """Stage B — candidate boundaries with auto / verify / skip actions."""
    by_page = {unit.pdf_page: unit for unit in units}
    llama_pages = {item.page for item in llama_boundary_proposals}
    boundaries: list[BoundaryCandidate] = []
    prev: PageClassification | None = None
    prev_unit: PageUnit | None = None
    for curr in classifications:
        unit = by_page.get(curr.page) or PageUnit(pdf_page=curr.page, text="")
        score, signals = _boundary_score(prev, curr, unit, prev_unit)
        if curr.page in llama_pages and prev is not None:
            score = min(1.0, score + 0.15)
            signals.append("llama_segment_transition")
        if prev is None:
            action = "auto"
        elif score >= _AUTO_SPLIT_THRESHOLD:
            action = "auto"
        elif score >= _VERIFY_THRESHOLD:
            action = "verify"
        else:
            action = "skip"
        if action != "skip" or prev is None:
            boundaries.append(
                BoundaryCandidate(
                    page=curr.page,
                    document_type=curr.document_type
                    if curr.document_type != "other"
                    else (prev.document_type if prev else "other"),
                    score=score,
                    signals=tuple(signals),
                    action=action if prev is not None else "auto",
                )
            )
        prev = curr
        prev_unit = unit
    return boundaries


def _resolve_page_labels(
    classifications: Sequence[PageClassification],
    boundaries: Sequence[BoundaryCandidate],
) -> dict[int, str]:
    """Assign each page a document type from accepted boundaries + carry."""
    starts = {
        b.page: b.document_type
        for b in boundaries
        if b.action in {"auto", "verify"} and b.document_type != "other"
    }
    # Prefer high-confidence page classifications at boundary pages.
    for c in classifications:
        if c.page in starts and c.confidence >= 0.8 and c.document_type != "other":
            starts[c.page] = c.document_type

    short_slots = {
        "Affidavit",
        "AOR's Declaration",
        "AOR's Certificate",
        "Filing Memo",
        "Memo of Parties",
        "Cover Page",
        "Office Report on Limitation",
        "Custody Certificate",
    }

    labels: dict[int, str] = {}
    current: str | None = None
    for c in classifications:
        if c.page in starts:
            current = starts[c.page]
        elif (
            c.document_type != "other"
            and c.confidence >= 0.85
            and (
                current is None
                or family_split_name(c.document_type)
                != family_split_name(current or "")
            )
        ):
            # Strong mid-run cue (e.g. Annexure stamp) opens a new label.
            current = c.document_type

        if current and current in short_slots:
            # Affidavit / AOR / Filing Memo are typically 1–2 pages — do not
            # paint subsequent unlabeled annexure body as the same slot.
            if c.document_type == current and c.confidence >= 0.7:
                labels[c.page] = current
            elif c.page in starts and starts[c.page] == current:
                labels[c.page] = current
                current = None
            else:
                current = None
            continue

        if current and current != "other":
            if (
                c.document_type != "other"
                and c.confidence >= 0.9
                and family_split_name(c.document_type) == family_split_name(current)
            ):
                current = c.document_type
            labels[c.page] = current
        elif c.document_type != "other" and c.confidence >= 0.7:
            labels[c.page] = c.document_type
            current = c.document_type
    return labels


def build_logical_documents(
    page_labels: Mapping[int, str],
    classifications: Sequence[PageClassification],
) -> list[LogicalDocument]:
    """Collapse labeled pages into contiguous logical documents."""
    if not page_labels:
        return []
    conf_by_page = {c.page: c.confidence for c in classifications}
    sig_by_page = {c.page: c.signals for c in classifications}
    pages = sorted(page_labels)
    docs: list[LogicalDocument] = []
    run_start = pages[0]
    run_type = page_labels[pages[0]]
    prev = pages[0]
    seq = 0

    def _flush(start: int, end: int, doc_type: str) -> None:
        nonlocal seq
        if end < start:
            return
        seq += 1
        confs = [conf_by_page.get(p, 0.5) for p in range(start, end + 1)]
        signals: list[str] = []
        for p in range(start, min(start + 3, end + 1)):
            signals.extend(sig_by_page.get(p, ()))
        docs.append(
            LogicalDocument(
                document_id=f"doc_{seq:03d}",
                document_type=doc_type,
                start_page=start,
                end_page=end,
                confidence=sum(confs) / max(len(confs), 1),
                signals=tuple(dict.fromkeys(signals)),
            )
        )

    for page in pages[1:]:
        label = page_labels[page]
        if label != run_type or page != prev + 1:
            _flush(run_start, prev, run_type)
            run_start = page
            run_type = label
        prev = page
    _flush(run_start, prev, run_type)
    return docs


def logical_documents_to_page_parts(
    documents: Sequence[LogicalDocument],
) -> PagePartMap:
    page_parts: PagePartMap = {}
    for doc in documents:
        if doc.document_type in {"other", ""}:
            continue
        for page in range(doc.start_page, doc.end_page + 1):
            page_parts[page] = [doc.document_type]
    return page_parts


_STRONG_RECONCILIATION_SIGNALS = frozenset(
    {
        "annexure_stamp",
        "outer_anchor",
        "form28_lookalike",
        "centered_title",
        "application_start",
    }
)
_UNIDENTIFIED_LABELS = frozenset({"Undefined", "Unidentified", "other"})


def _copy_page_parts(page_parts: Mapping[int, Sequence[str] | str | None]) -> PagePartMap:
    copied: PagePartMap = {}
    for page, names in page_parts.items():
        labels = parts_on_page(names)
        if labels:
            copied[int(page)] = list(labels)
    return copied


def _same_document_type(left: str, right: str) -> bool:
    """Compare canonical types without equating different numbered exhibits."""
    if left == right:
        return True
    left_family = family_split_name(left)
    right_family = family_split_name(right)
    if left_family != right_family:
        return False
    return left_family not in {ANNEXURE_FAMILY, "Application"}


def _labels_match_type(labels: Sequence[str], document_type: str) -> bool:
    return any(_same_document_type(label, document_type) for label in labels)


def _segment_classification_supports_labels(
    segment: SegmentClassification, labels: Sequence[str]
) -> bool:
    if _labels_match_type(labels, segment.document_type):
        return True
    return bool(
        segment.document_type == "Application 1"
        and set(segment.signals) & {"outer_anchor", "application_start"}
        and any(family_split_name(label) == "Application" for label in labels)
    )


def _index_supported_map(
    baseline: PagePartMap,
    page_text: Mapping[int, str],
    page_count: int,
) -> PagePartMap:
    """Build only the deterministic master-Index/folio reconciliation map."""
    indexed = _restore_master_volume_sections(baseline, page_text, page_count)
    indexed = _restore_index_bridge_pages(indexed, page_text)
    indexed = _apply_indexed_outer_document_ranges(indexed, page_text, page_count)
    indexed = _apply_indexed_annexure_ranges(indexed, page_text, page_count)
    indexed = _apply_cited_annexure_ranges(indexed, page_text, page_count)
    return indexed


def validate_reconciliation_page_map(
    page_parts: Mapping[int, Sequence[str] | str | None], *, page_count: int
) -> dict[str, Any]:
    """Validate map structure independently of legal-document semantics.

    Unlabelled in-range source pages are represented by the downstream
    ``Unidentified`` output slot; they are not silently dropped.
    """
    normalized_pages: list[int] = []
    invalid: list[Any] = []
    for raw_page in page_parts:
        try:
            page = int(raw_page)
        except (TypeError, ValueError):
            invalid.append(raw_page)
            continue
        normalized_pages.append(page)
        if not 1 <= page <= page_count:
            invalid.append(page)
    multi_owned = sorted(
        int(page)
        for page, labels in page_parts.items()
        if str(page).isdigit() and len(parts_on_page(labels)) > 1
    )
    missing_labels = sorted(set(range(1, page_count + 1)) - set(normalized_pages))
    return {
        # A single physical page may intentionally carry a combined page label
        # that maps to one combined upload slot. Exact slot ownership is
        # verified after slicing by validate_sliced_page_ownership.
        "valid": not invalid,
        "source_page_count": page_count,
        "invalid_pages": invalid,
        "multiple_label_pages": multi_owned,
        "unlabelled_pages_routed_to_unidentified": missing_labels,
        "single_physical_owner": not invalid,
        "combined_labels_require_slot_validation": bool(multi_owned),
    }


def _operation_type_for_page(
    page: int,
    old: tuple[str, ...],
    new: tuple[str, ...],
    before: Mapping[int, Sequence[str] | str | None],
    after: Mapping[int, Sequence[str] | str | None],
    unit: PageUnit | None = None,
) -> str:
    if not old or all(label in _UNIDENTIFIED_LABELS for label in old):
        return "relabel_segment"
    left_old = tuple(parts_on_page(before.get(page - 1))) if page > 1 else ()
    right_old = tuple(parts_on_page(before.get(page + 1)))
    left_new = tuple(parts_on_page(after.get(page - 1))) if page > 1 else ()
    right_new = tuple(parts_on_page(after.get(page + 1)))
    if old and new and new != old and (left_old == old or right_old == old):
        return "split_segment"
    if unit and (
        unit.sentence_continues_previous
        or unit.paragraph_continues_previous
        or unit.signature_detected
        or unit.closing_section
    ):
        return "boundary_adjustment"
    if old and new and old != new and (left_new == new or right_new == new):
        return "merge_segments"
    if old and new and old != new:
        return "boundary_adjustment"
    return "relabel_segment"


def _contiguous_segment_range(
    page: int,
    page_parts: Mapping[int, Sequence[str] | str | None],
    labels: Sequence[str],
    *,
    page_count: int,
) -> tuple[int, int] | None:
    if not labels or tuple(parts_on_page(page_parts.get(page))) != tuple(labels):
        return None
    start = page
    end = page
    while start > 1 and tuple(parts_on_page(page_parts.get(start - 1))) == tuple(labels):
        start -= 1
    while end < page_count and tuple(parts_on_page(page_parts.get(end + 1))) == tuple(labels):
        end += 1
    return start, end


def _record_reconciliation_state(state: str, seen_states: set[str]) -> bool:
    """Return False for a repeated page map, which stops oscillating repairs."""
    if state in seen_states:
        return False
    seen_states.add(state)
    return True


def reconcile_compiled_split(
    baseline: PagePartMap,
    proposed: PagePartMap,
    classifications: Sequence[PageClassification],
    page_text: Mapping[int, str],
    *,
    page_count: int,
    operation_log: list[ReconciliationOperation] | None = None,
    max_iterations: int = 3,
    max_model_calls: int = 0,
    proposal_generator: Callable[[PagePartMap, int], PagePartMap] | None = None,
    page_units: Sequence[PageUnit] = (),
) -> tuple[PagePartMap, list[ReconciliationChange]]:
    """Apply repair proposals only to missing or strongly evidenced ranges.

    ``baseline`` is the immutable Llama result plus anchors for pages Llama did
    not label. ``proposed`` may be produced by the legacy repair pipeline, but
    disagreement alone is not authority to overwrite Llama.  A proposal is
    committed only for an unassigned/ambiguous page, an Index+folio-supported
    page, or a continuous proposed span with a verified outer opening.
    """
    original = _copy_page_parts(baseline)
    candidate = _copy_page_parts(proposed)
    operations = operation_log if operation_log is not None else []
    before_validation = validate_reconciliation_page_map(original, page_count=page_count)
    candidate_validation = validate_reconciliation_page_map(candidate, page_count=page_count)
    if not before_validation["valid"] or not candidate_validation["valid"]:
        operations.append(
            ReconciliationOperation(
                operation_id="reconcile-0001",
                operation_type="global_candidate_map",
                original_pages=tuple(sorted(original)),
                proposed_pages=tuple(sorted(candidate)),
                original_labels=tuple(
                    sorted({label for labels in original.values() for label in labels})
                ),
                proposed_labels=tuple(
                    sorted({label for labels in candidate.values() for label in labels})
                ),
                original_segment_range=None,
                proposed_segment_range=None,
                evidence=("complete_map_preflight",),
                classification_before=None,
                classification_after=None,
                validation={
                    "before": before_validation,
                    "proposed": candidate_validation,
                },
                status="rejected",
                reason="Page map failed structural preflight; no correction applied",
            )
        )
        return original, []
    indexed = _index_supported_map(original, page_text, page_count)
    result = _copy_page_parts(original)
    changes: list[ReconciliationChange] = []

    strong_by_page = {
        item.page: item
        for item in classifications
        if item.confidence >= 0.85
        and item.document_type != "other"
        and set(item.signals) & _STRONG_RECONCILIATION_SIGNALS
    }
    units_by_page = {unit.pdf_page: unit for unit in page_units}

    def classification_supports_labels(
        item: PageClassification, labels: Sequence[str]
    ) -> bool:
        # Heading classification uses Application 1 as a taxonomy placeholder;
        # it has no knowledge of this opening's order in the complete filing.
        # Let that real opening support a repaired Application 2+ span, while
        # retaining any explicit numbered identity already supplied by Llama.
        if (
            item.document_type == "Application 1"
            and set(item.signals) & {"outer_anchor", "application_start"}
            and page_starts_application(page_text.get(item.page, ""))
        ):
            seeded_identities = {
                name
                for name in parts_on_page(result.get(item.page))
                if re.fullmatch(r"Application \d+", name)
            }
            return any(
                family_split_name(label) == "Application"
                and (not seeded_identities or label in seeded_identities)
                for label in labels
            )
        return _labels_match_type(labels, item.document_type)

    seen_states = {repr(sorted((p, tuple(v)) for p, v in result.items()))}
    iteration_limit = max(1, min(int(max_iterations), 3))
    operation_limit = 500
    iteration = 0
    model_call_budget = max(0, min(int(max_model_calls), 1))
    model_calls = 0
    stop_reason = "no_further_supported_mutations"
    for pass_number in range(1, iteration_limit + 1):
        iteration = pass_number
        pass_input = _copy_page_parts(result)
        pass_start = len(changes)
        changed_in_iteration = False
        candidate_validation = validate_reconciliation_page_map(
            candidate, page_count=page_count
        )
        if not candidate_validation["valid"]:
            operations.append(
                ReconciliationOperation(
                    operation_id=f"reconcile-{len(operations) + 1:04d}",
                    operation_type="reconciliation_pass",
                    original_pages=tuple(sorted(result)),
                    proposed_pages=tuple(sorted(candidate)),
                    original_labels=tuple(
                        sorted({label for labels in result.values() for label in labels})
                    ),
                    proposed_labels=tuple(
                        sorted({label for labels in candidate.values() for label in labels})
                    ),
                    original_segment_range=None,
                    proposed_segment_range=None,
                    evidence=("fresh_global_reassessment",),
                    classification_before=None,
                    classification_after=None,
                    validation={"proposed": candidate_validation},
                    status="rejected",
                    reason="Fresh proposal failed structural preflight; current map retained",
                    pass_index=pass_number,
                )
            )
            stop_reason = "invalid_fresh_proposal"
            break

        # An explicit opening may support captionless continuations in its
        # proposed span, but only if it replaces at most one existing family.
        span_policy: dict[int, bool] = {}
        pages = sorted(candidate)
        start_index = 0
        while start_index < len(pages):
            start = pages[start_index]
            labels = tuple(parts_on_page(candidate[start]))
            end_index = start_index
            while end_index + 1 < len(pages):
                next_page = pages[end_index + 1]
                if next_page != pages[end_index] + 1:
                    break
                if tuple(parts_on_page(candidate[next_page])) != labels:
                    break
                end_index += 1
            span_pages = pages[start_index : end_index + 1]
            anchored = any(
                classification_supports_labels(item, labels)
                for page in span_pages
                if (item := strong_by_page.get(page))
            )
            replaced_families = {
                family_split_name(label)
                for page in span_pages
                if tuple(parts_on_page(result.get(page))) != labels
                for label in parts_on_page(result.get(page))
                if label not in _UNIDENTIFIED_LABELS
            }
            allow_continuation = anchored and len(replaced_families) <= 1
            for page in span_pages:
                span_policy[page] = allow_continuation
            start_index = end_index + 1

        for page in range(1, page_count + 1):
            old = tuple(parts_on_page(result.get(page)))
            new = tuple(parts_on_page(candidate.get(page)))
            indexed_labels = tuple(parts_on_page(indexed.get(page)))
            if indexed_labels and indexed_labels != old and indexed_labels != new:
                # Master-Index + folio ranges override ambiguous page proposals.
                new = indexed_labels
            if not new or new == old:
                continue
            if len(operations) >= operation_limit:
                stop_reason = "operation_limit_reached"
                break

            evidence: list[str] = []
            if not old or all(label in _UNIDENTIFIED_LABELS for label in old):
                evidence.append("missing_or_unidentified")
            if len(old) > 1:
                evidence.append("overlapping_labels")
            if indexed_labels == new and indexed_labels != old:
                evidence.append("master_index_and_folio")
            direct = strong_by_page.get(page)
            if direct and classification_supports_labels(direct, new):
                evidence.append("explicit_outer_boundary")
            elif span_policy.get(page):
                evidence.append("anchored_continuation")

            op_type = _operation_type_for_page(
                page, old, new, result, candidate, units_by_page.get(page)
            )
            op_id = f"reconcile-{len(operations) + 1:04d}"
            original_segment_range = _contiguous_segment_range(
                page, result, old, page_count=page_count
            )
            proposed_segment_range = _contiguous_segment_range(
                page, candidate, new, page_count=page_count
            )
            direct_type = direct.document_type if direct else None
            if not evidence:
                operations.append(
                    ReconciliationOperation(
                        operation_id=op_id,
                        operation_type=op_type,
                        original_pages=(page,),
                        proposed_pages=(page,),
                        original_labels=old,
                        proposed_labels=new,
                        original_segment_range=original_segment_range,
                        proposed_segment_range=proposed_segment_range,
                        evidence=(),
                        classification_before=direct_type,
                        classification_after=direct_type,
                        validation={"structural": before_validation},
                        status="left_for_review",
                        reason="No independent boundary, Index, or continuity evidence supports this change",
                        pass_index=pass_number,
                    )
                )
                continue

            proposed_state = _copy_page_parts(result)
            proposed_state[page] = list(new)
            structural = validate_reconciliation_page_map(
                proposed_state, page_count=page_count
            )
            affected_segments = classify_complete_segments(proposed_state, page_units)
            affected = [
                segment
                for segment in affected_segments
                if segment.start_page <= page <= segment.end_page
            ]
            semantic_conflicts = [
                segment.as_dict()
                for segment in affected
                if segment.confidence >= 0.85
                and not _segment_classification_supports_labels(segment, new)
            ]
            semantic_ok = not (
                direct
                and direct.confidence >= 0.85
                and not classification_supports_labels(direct, new)
            ) and not semantic_conflicts
            validation = {
                "structural": structural,
                "semantic": {
                    "valid": semantic_ok,
                    "classification": direct_type,
                    "proposed_labels": list(new),
                    "affected_segment_classifications": [
                        segment.as_dict() for segment in affected
                    ],
                    "conflicts": semantic_conflicts,
                },
            }
            if not structural["valid"] or not semantic_ok:
                operations.append(
                    ReconciliationOperation(
                        operation_id=op_id,
                        operation_type=op_type,
                        original_pages=(page,),
                        proposed_pages=(page,),
                        original_labels=old,
                        proposed_labels=new,
                        original_segment_range=original_segment_range,
                        proposed_segment_range=proposed_segment_range,
                        evidence=tuple(evidence),
                        classification_before=direct_type,
                        classification_after=direct_type,
                        validation=validation,
                        status="rolled_back",
                        reason="Mutation failed structural or semantic revalidation",
                        pass_index=pass_number,
                    )
                )
                continue

            result = proposed_state
            changed_in_iteration = True
            changes.append(
                ReconciliationChange(
                    page=page,
                    old_labels=old,
                    new_labels=new,
                    evidence=tuple(evidence),
                )
            )
            operations.append(
                ReconciliationOperation(
                    operation_id=op_id,
                    operation_type=op_type,
                    original_pages=(page,),
                    proposed_pages=(page,),
                    original_labels=old,
                    proposed_labels=new,
                    original_segment_range=original_segment_range,
                    proposed_segment_range=_contiguous_segment_range(
                        page, proposed_state, new, page_count=page_count
                    ),
                    evidence=tuple(evidence),
                    classification_before=direct_type,
                    classification_after=direct_type,
                    validation=validation,
                    status="accepted",
                    pass_index=pass_number,
                )
            )
        operations.append(
            ReconciliationOperation(
                operation_id=f"reconcile-{len(operations) + 1:04d}",
                operation_type="reconciliation_pass",
                original_pages=tuple(sorted(pass_input)),
                proposed_pages=tuple(sorted(result)),
                original_labels=(),
                proposed_labels=(),
                original_segment_range=None,
                proposed_segment_range=None,
                evidence=("deterministic_global_reassessment",),
                classification_before=None,
                classification_after=None,
                validation={
                    "structural": validate_reconciliation_page_map(
                        result, page_count=page_count
                    )
                },
                status="evaluated",
                reason=(
                    f"accepted_mutations={len(changes) - pass_start}; "
                    f"model_calls={model_calls}/{model_call_budget}"
                ),
                pass_index=pass_number,
            )
        )
        state = repr(sorted((p, tuple(v)) for p, v in result.items()))
        is_new_state = _record_reconciliation_state(state, seen_states)
        if not is_new_state and changed_in_iteration:
            stop_reason = "repeated_state_oscillation_guard"
            break
        if stop_reason == "operation_limit_reached":
            break
        if not changed_in_iteration:
            stop_reason = "no_further_supported_mutations"
            break
        if pass_number >= iteration_limit:
            stop_reason = "iteration_budget_reached"
            break
        if proposal_generator is None:
            stop_reason = "no_fresh_proposal_generator"
            break
        try:
            fresh_candidate = _copy_page_parts(
                proposal_generator(_copy_page_parts(result), pass_number + 1)
            )
        except Exception as exc:  # noqa: BLE001 — record provider failures for review.
            operations.append(
                ReconciliationOperation(
                    operation_id=f"reconcile-{len(operations) + 1:04d}",
                    operation_type="reconciliation_pass",
                    original_pages=tuple(sorted(result)),
                    proposed_pages=tuple(sorted(result)),
                    original_labels=(),
                    proposed_labels=(),
                    original_segment_range=None,
                    proposed_segment_range=None,
                    evidence=("fresh_proposal_generation_failed",),
                    classification_before=None,
                    classification_after=None,
                    validation={},
                    status="left_for_review",
                    reason=f"Fresh reassessment failed: {type(exc).__name__}",
                    pass_index=pass_number + 1,
                )
            )
            stop_reason = "fresh_proposal_generation_failed"
            break
        fresh_state = repr(sorted((p, tuple(v)) for p, v in fresh_candidate.items()))
        if fresh_state == state:
            stop_reason = "no_actionable_fresh_proposals"
            break
        if fresh_state in seen_states:
            operations.append(
                ReconciliationOperation(
                    operation_id=f"reconcile-{len(operations) + 1:04d}",
                    operation_type="reconciliation_pass",
                    original_pages=tuple(sorted(result)),
                    proposed_pages=tuple(sorted(fresh_candidate)),
                    original_labels=(),
                    proposed_labels=(),
                    original_segment_range=None,
                    proposed_segment_range=None,
                    evidence=("repeated_state_guard",),
                    classification_before=None,
                    classification_after=None,
                    validation={"structural": candidate_validation},
                    status="rejected",
                    reason="Fresh proposal repeated a previous page-map state",
                    pass_index=pass_number + 1,
                )
            )
            stop_reason = "repeated_state_oscillation_guard"
            break
        candidate = fresh_candidate
    operations.append(
        ReconciliationOperation(
            operation_id=f"reconcile-{len(operations) + 1:04d}",
            operation_type="reconciliation_stop",
            original_pages=tuple(range(1, page_count + 1)),
            proposed_pages=tuple(sorted(result)),
            original_labels=(),
            proposed_labels=(),
            original_segment_range=None,
            proposed_segment_range=None,
            evidence=("bounded_iterations",),
            classification_before=None,
            classification_after=None,
            validation={
                "structural": validate_reconciliation_page_map(
                    result, page_count=page_count
                )
            },
            status="stopped",
            reason=f"iterations={iteration}; max_iterations={iteration_limit}; max_operations={operation_limit}; {stop_reason}",
        )
    )

    return result, changes


def structure_aware_split(
    pdf_bytes: bytes,
    *,
    llama_page_parts: PagePartMap
    | Mapping[int, Sequence[str] | str | None]
    | None = None,
    page_texts: Mapping[int, str] | None = None,
    page_units: Sequence[PageUnit] | None = None,
    source_pdf: str = "bundle.pdf",
    run_hybrid_repair: bool = True,
    reconciliation_mode: str = "targeted",
) -> StructureSplitResult:
    """Full structure-aware split. Physical slicing stays outside this function.

    Classification + boundaries are recorded for audit. Label authority is:
    LlamaSplit warm-start (if any) → strong local anchors on empty pages →
    hybrid repair (fill gaps / demote / nest). Do not paint every classified
    page into ``page_parts`` before repair — that created one-page Index /
    Listing / Synopsis islands when a continuation page was mis-typed.
    """
    units = (
        list(page_units)
        if page_units is not None
        else extract_page_units(pdf_bytes, source_pdf=source_pdf, page_texts=page_texts)
    )
    texts = {unit.pdf_page: unit.text for unit in units}

    # LlamaSplit returns candidate segments; preserve those transitions first.
    # Local content classification is evidence for validating/refining them,
    # not the source of the initial boundary proposal.
    llama_boundary_proposals = propose_llama_boundaries(
        llama_page_parts,
        page_count=len(units),
    )
    # Do not feed Llama's category labels back into final local classification;
    # only its segment transitions are boundary evidence at this stage.
    classifications = classify_pages(units)
    boundaries = detect_boundaries(
        classifications,
        units,
        llama_boundary_proposals=llama_boundary_proposals,
    )

    # Seed: Llama first, then only high-confidence anchors on still-empty pages.
    page_parts: PagePartMap = {}
    if llama_page_parts:
        for page, names in llama_page_parts.items():
            parts = parts_on_page(names)
            if parts:
                page_parts[int(page)] = list(parts)

    strong_signals = {
        "annexure_stamp",
        "outer_anchor",
        "form28_lookalike",
        "centered_title",
        "application_start",
    }
    for c in classifications:
        if c.page in page_parts:
            continue
        if c.document_type == "other" or c.confidence < 0.88:
            continue
        if not (set(c.signals) & strong_signals):
            continue
        page_parts[c.page] = [c.document_type]

    duplicates: list[Any] = []
    reconciliation_changes: list[ReconciliationChange] = []
    reconciliation_operations: list[ReconciliationOperation] = []
    reconciliation_summary: dict[str, Any] = {
        "status": "not_run_legacy_mode",
        "max_iterations": 0,
        "iterations": 0,
        "stop_reason": "compatibility mode skips semantic reconciliation",
    }
    if run_hybrid_repair and units:
        baseline = _copy_page_parts(page_parts)
        proposed, duplicates = repair_compiled_split(
            page_parts,
            texts,
            page_count=len(units),
        )

        def reassess_current_map(
            current: PagePartMap, _pass_index: int
        ) -> PagePartMap:
            # Deterministic global repair is recomputed against the current
            # accepted map. No Llama call is warranted here: only the already
            # extracted page evidence and initial Llama classification are
            # available; any unresolved semantic conflict stays review-only.
            fresh, _fresh_duplicates = repair_compiled_split(
                current,
                texts,
                page_count=len(units),
            )
            return fresh

        page_parts, reconciliation_changes = reconcile_compiled_split(
            baseline,
            proposed,
            classifications,
            texts,
            page_count=len(units),
            operation_log=reconciliation_operations,
            proposal_generator=reassess_current_map,
            page_units=units,
        )
        stop_operation = next(
            (
                item
                for item in reversed(reconciliation_operations)
                if item.operation_type == "reconciliation_stop"
            ),
            None,
        )
        reconciliation_summary = {
            "status": "completed",
            "max_iterations": 3,
            "iterations": sum(
                item.operation_type == "reconciliation_pass"
                and item.status == "evaluated"
                for item in reconciliation_operations
            ),
            "model_call_budget": 0,
            "model_calls": 0,
            "stop_reason": stop_operation.reason if stop_operation else "no_operations",
            "accepted": sum(item.status == "accepted" for item in reconciliation_operations),
            "rejected": sum(item.status in {"rejected", "rolled_back"} for item in reconciliation_operations),
            "left_for_review": sum(item.status == "left_for_review" for item in reconciliation_operations),
        }
    # With repair disabled, preserve the supplied Llama ownership map. The
    # extracted page analysis and structural audit still run; this is the
    # compatibility path, not a way to bypass mandatory parsing.

    repaired_labels = {
        page: parts_on_page(names)[0]
        for page, names in page_parts.items()
        if parts_on_page(names)
    }
    logical_docs = build_logical_documents(repaired_labels, classifications)
    segment_classifications = classify_complete_segments(page_parts, units)

    ocr_pages = [unit.pdf_page for unit in units if unit.requires_ocr]
    return StructureSplitResult(
        page_units=list(units),
        classifications=classifications,
        boundaries=boundaries,
        logical_documents=logical_docs,
        page_parts=page_parts,
        duplicates=duplicates,
        ocr_needed_pages=ocr_pages,
        reconciliation_changes=reconciliation_changes,
        llama_boundary_proposals=llama_boundary_proposals,
        segment_classifications=segment_classifications,
        reconciliation_operations=reconciliation_operations,
        reconciliation_summary=reconciliation_summary,
    )
