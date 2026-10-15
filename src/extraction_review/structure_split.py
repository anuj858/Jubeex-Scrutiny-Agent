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
    APPLICATION_FAMILY,
    MAIN_PETITION_PART,
    PagePartMap,
    annexure_label_from_text,
    annexure_ref_in_heading,
    family_split_name,
    normalize_part_name,
    page_ranges,
    page_starts_application,
    parts_on_page,
)
from .split_ocr import ocr_sparse_pages, pages_with_large_images
from .split_pdf_layout import extract_split_layout
from .split_pdf_layout import printed_folio as _printed_folio
from .split_repair import (
    _apply_cited_annexure_ranges,
    _apply_indexed_annexure_ranges,
    _apply_indexed_outer_document_ranges,
    _apply_outer_anchors,
    _apply_strong_index_content_spans,
    _force_annexure_nesting,
    _has_leading_high_court_caption,
    _heading_window,
    _is_lower_court_caption,
    _is_near_blank_page,
    _is_sci_caption,
    _looks_like_sci_main_petition,
    _outer_anchor_label,
    _restore_early_record_of_proceedings,
    _restore_front_impugned_judgment,
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
                {
                    "start": self.original_segment_range[0],
                    "end": self.original_segment_range[1],
                }
                if self.original_segment_range
                else None
            ),
            "proposed_segment_range": (
                {
                    "start": self.proposed_segment_range[0],
                    "end": self.proposed_segment_range[1],
                }
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
    reconciliation_operations: list[ReconciliationOperation] = field(
        default_factory=list
    )
    reconciliation_summary: dict[str, Any] = field(default_factory=dict)
    # Compact, text-free provenance; persisted only inside the additive audit.
    assignment_events: dict[int, list[dict[str, Any]]] = field(default_factory=dict)

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


def _assignment_run_ranges(
    page_parts: Mapping[int, Sequence[str] | str | None], page_count: int
) -> dict[int, dict[str, int]]:
    """Contiguous same-owner ranges used to explain merges and splits."""
    ranges: dict[int, dict[str, int]] = {}
    start = 1
    while start <= page_count:
        labels = tuple(parts_on_page(page_parts.get(start)))
        end = start
        while (
            end < page_count
            and tuple(parts_on_page(page_parts.get(end + 1))) == labels
        ):
            end += 1
        for page in range(start, end + 1):
            ranges[page] = {"start_page": start, "end_page": end}
        start = end + 1
    return ranges


def _trace_assignment_stage(
    events: dict[int, list[dict[str, Any]]],
    stage: str,
    before: Mapping[int, Sequence[str] | str | None],
    after: Mapping[int, Sequence[str] | str | None],
    page_count: int,
    *,
    reason: str,
    evidence: Sequence[str] = (),
    status: str = "committed",
    evidence_by_page: Mapping[int, Sequence[str]] | None = None,
) -> None:
    """Record changed labels *and* neighboring range changes without page text."""
    old_ranges = _assignment_run_ranges(before, page_count)
    new_ranges = _assignment_run_ranges(after, page_count)
    for page in range(1, page_count + 1):
        old_labels = parts_on_page(before.get(page))
        new_labels = parts_on_page(after.get(page))
        if old_labels == new_labels and old_ranges[page] == new_ranges[page]:
            continue
        events.setdefault(page, []).append(
            {
                "stage": stage,
                "status": status,
                "reason": reason,
                "evidence": list(
                    (evidence_by_page or {}).get(page, evidence)
                ),
                "from_labels": old_labels,
                "to_labels": new_labels,
                "from_range": old_ranges[page],
                "to_range": new_ranges[page],
            }
        )


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
            "empty" if not stripped else "sparse" if requires_ocr else "usable"
        )
        lines = [line.strip() for line in (text or "").splitlines() if line.strip()]
        header = "\n".join(lines[:3])
        footer = "\n".join(lines[-2:]) if len(lines) > 3 else ""
        headings = tuple(
            line[:180]
            for line in lines[:12]
            if _CENTERED_TITLE_RE.search(line)
            or line.isupper()
            and 4 <= len(line) <= 100
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
                    and not re.search(
                        r"[.!?][\"')\]]?\s*$", (units[-1].text or "").strip()
                    )
                ),
                paragraph_continues_previous=bool(
                    number > 1 and stripped and not paragraph_starts and not headings
                ),
                signature_detected=signature_detected,
                closing_section=closing_match.group(1).casefold()
                if closing_match
                else None,
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
        if anchor == "Application 1" and not page_starts_application(text):
            # IA body/prayer language identifies the document family, not a
            # new outer opening. Treating it as an outer_anchor protects a
            # false Llama boundary before a captionless continuation page.
            doc_type = anchor
            score = 0.65
            signals.append("application_body")
        elif anchor:
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
    llama_page_parts: PagePartMap | Mapping[int, Sequence[str] | str | None] | None,
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


def _copy_page_parts(
    page_parts: Mapping[int, Sequence[str] | str | None],
) -> PagePartMap:
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


_DIRECT_DOCUMENT_START_SIGNALS = frozenset(
    {
        "annexure_stamp",
        "outer_anchor",
        "form28_lookalike",
        "application_start",
    }
)


def _classification_opens_labels(
    classification: PageClassification | None,
    labels: Sequence[str],
) -> bool:
    """Return whether this page independently opens ``labels``.

    A Llama segment transition is deliberately not sufficient here.  The page
    itself must carry a document-start cue.  This prevents a continuation page
    from leaving its current document merely because its body text resembles a
    different configured category.

    Generic ``Annexures`` / ``Application`` labels are allowed to match a
    numbered local opening in the same family.  Different numbered annexures
    and applications are never treated as the same identity.
    """
    if classification is None or classification.confidence < 0.85:
        return False
    if not (set(classification.signals) & _DIRECT_DOCUMENT_START_SIGNALS):
        return False
    if _labels_match_type(labels, classification.document_type):
        return True
    classified_family = family_split_name(classification.document_type)
    return any(
        label in {ANNEXURE_FAMILY, "Application", "Applications"}
        and family_split_name(label) == classified_family
        for label in labels
    )


def _stabilize_document_continuity(
    page_parts: PagePartMap,
    classifications: Sequence[PageClassification],
    *,
    authoritative_boundary_pages: Sequence[int] = (),
) -> PagePartMap:
    """Reject unsupported page-by-page label changes inside an open document.

    The first independently headed page opens a document state.  Its following
    assigned pages keep that owner until another page independently proves a
    new document start.  Index/folio reconciliation and the existing repair
    pipeline can still replace these ranges later when they have stronger
    evidence.

    This rule is document-type agnostic: it protects annexures, applications,
    petitions, judgments and front/back-matter documents alike.  It does not
    infer an ending from phrases such as ``True Copy`` because those phrases
    are optional, OCR-sensitive and also occur inside reproduced records.
    """
    updated = _copy_page_parts(page_parts)
    by_page = {item.page: item for item in classifications}
    authoritative_starts = {int(page) for page in authoritative_boundary_pages}
    active_labels: tuple[str, ...] = ()
    active_has_opening = False
    previous_page: int | None = None
    first_page = min(updated, default=0)
    last_page = max(updated, default=0)

    # Walk the physical page sequence, not only pages which Llama assigned.
    # Otherwise one omitted continuation page resets the state and the next
    # weak page-level guess can open a false document.  Unassigned pages after
    # a verified opening inherit that owner until a verified next heading.
    for page in range(first_page, last_page + 1):
        labels = tuple(parts_on_page(updated.get(page)))
        if not labels:
            if active_has_opening and active_labels:
                updated[page] = list(active_labels)
            previous_page = page
            continue

        if previous_page is None or page != previous_page + 1:
            active_labels = labels
            active_has_opening = _classification_opens_labels(
                by_page.get(page), labels
            )
            previous_page = page
            continue

        if labels == active_labels:
            active_has_opening = active_has_opening or _classification_opens_labels(
                by_page.get(page), labels
            )
            previous_page = page
            continue

        if page in authoritative_starts:
            # The dedicated boundary pass has already established a new
            # physical document here. Classification may name that complete
            # range, but page-level semantic cues must not erase the boundary.
            active_labels = labels
            active_has_opening = _classification_opens_labels(by_page.get(page), labels)
            previous_page = page
            continue

        classification = by_page.get(page)
        candidate_has_opening = _classification_opens_labels(classification, labels)
        structurally_distinct_opening = bool(
            classification
            and classification.confidence >= 0.85
            and (
                set(classification.signals) & {"application_start", "annexure_stamp"}
                or (
                    "outer_anchor" in classification.signals
                    and family_split_name(classification.document_type) == "Application"
                )
            )
        )
        if (
            candidate_has_opening
            or structurally_distinct_opening
            or not active_has_opening
        ):
            # A verified heading starts the next document.  When neither side
            # has a verified heading, retain Llama's proposal and leave it for
            # audit/review rather than inventing continuity.
            active_labels = labels
            active_has_opening = (
                candidate_has_opening or structurally_distinct_opening
            )
        else:
            # The current document has a verified opening, while the proposed
            # new label has only page-body similarity.  Keep the page with the
            # open document until a real next-document boundary appears.
            updated[page] = list(active_labels)
        previous_page = page

    return updated


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


def _confirmed_application_continuation(
    page: int,
    old: Sequence[str],
    new: Sequence[str],
    current: PagePartMap,
    page_text: Mapping[int, str],
) -> bool:
    """Permit correcting a false IA boundary only with physical continuity.

    A preceding application assignment, consecutive printed folios and a
    body/prayer opening establish continuation. A fresh application caption,
    another outer heading or missing folio evidence leaves the model boundary
    for human review rather than merging two possible applications.
    """
    if (
        page <= 1
        or len(old) != 1
        or len(new) != 1
        or old == new
        or any(family_split_name(label) != "Application" for label in (*old, *new))
        or tuple(parts_on_page(current.get(page - 1))) != tuple(new)
    ):
        return False
    text = page_text.get(page, "")
    previous = page_text.get(page - 1, "")
    if (
        not text.strip()
        or page_starts_application(text)
        or annexure_label_from_text(text)
        or _is_lower_court_caption(text)
        or _looks_like_sci_main_petition(text)
        or _outer_anchor_label(text) not in {None, "Application 1"}
    ):
        return False
    prior_folio = _printed_folio(previous)
    folio = _printed_folio(text)
    if not (
        prior_folio
        and folio
        and prior_folio[0] == folio[0] == "number"
        and folio[1] == prior_folio[1] + 1
    ):
        return False
    return bool(
        re.search(
            r"(?mi)^\s*(?:(?:[2-9]|\d{2,3})[.)]\s+|PRAYER\b|FILED\s+BY\b|"
            r"VERIFICATION\b|AND\s+FOR\s+THIS\s+ACT\b)",
            text,
        )
    )


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
    while start > 1 and tuple(parts_on_page(page_parts.get(start - 1))) == tuple(
        labels
    ):
        start -= 1
    while end < page_count and tuple(parts_on_page(page_parts.get(end + 1))) == tuple(
        labels
    ):
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
    authoritative_boundary_pages: Sequence[int] = (),
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
    before_validation = validate_reconciliation_page_map(
        original, page_count=page_count
    )
    candidate_validation = validate_reconciliation_page_map(
        candidate, page_count=page_count
    )
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
    authoritative_starts = {int(page) for page in authoritative_boundary_pages}

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
                        sorted(
                            {label for labels in result.values() for label in labels}
                        )
                    ),
                    proposed_labels=tuple(
                        sorted(
                            {label for labels in candidate.values() for label in labels}
                        )
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
            if (
                len(old) == len(new) == 1
                and re.fullmatch(r"Application \d+", old[0])
                and new[0] == "Application"
            ):
                # A fresh repair pass may emit the family-only placeholder
                # after an earlier pass has established the numbered IA from
                # the Index. Never degrade that confirmed identity.
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
            if _confirmed_application_continuation(page, old, new, result, page_text):
                evidence.append("application_continuation")

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
            protected_boundary = bool(
                original_segment_range
                and original_segment_range[0] in authoritative_starts
            )
            annexure_to_other_document = bool(
                old
                and all(family_split_name(label) == ANNEXURE_FAMILY for label in old)
                and any(family_split_name(label) != ANNEXURE_FAMILY for label in new)
            )
            independent_annexure_exit = bool(
                page in authoritative_starts
                or "master_index_and_folio" in evidence
                or (
                    any(family_split_name(label) == "Application" for label in new)
                    and _is_sci_caption(page_text.get(page, ""))
                    and page_starts_application(page_text.get(page, ""))
                )
                or (
                    any(family_split_name(label) == "Application" for label in new)
                    and page > 1
                    and tuple(parts_on_page(result.get(page - 1))) == new
                    and span_policy.get(page)
                )
            )
            if annexure_to_other_document and not independent_annexure_exit:
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
                        classification_before=direct.document_type if direct else None,
                        classification_after=direct.document_type if direct else None,
                        validation={"structural": before_validation},
                        status="left_for_review",
                        reason=(
                            "A heading inside an established Annexure is not "
                            "independent evidence of a new document boundary"
                        ),
                        pass_index=pass_number,
                    )
                )
                continue
            if protected_boundary and not {
                "master_index_and_folio",
                "explicit_outer_boundary",
                "application_continuation",
            }.intersection(evidence):
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
                        classification_before=direct.document_type if direct else None,
                        classification_after=direct.document_type if direct else None,
                        validation={"structural": before_validation},
                        status="left_for_review",
                        reason=(
                            "Proposal would erase a dedicated boundary without "
                            "independent boundary evidence"
                        ),
                        pass_index=pass_number,
                    )
                )
                continue
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
            semantic_ok = (
                not (
                    direct
                    and direct.confidence >= 0.85
                    and not classification_supports_labels(direct, new)
                )
                and not semantic_conflicts
            )
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


def _restore_annexed_lower_court_petitions(
    page_parts: PagePartMap,
    page_text: Mapping[int, str],
    page_count: int,
) -> tuple[PagePartMap, dict[int, tuple[str, ...]]]:
    """Keep a reproduced High Court petition out of the SCI Main Petition slot.

    A lower-court caption is evidence about the source court, not by itself
    evidence of which annexure owns the page. Reuse the preceding annexure
    only when a following annexure boundary or consecutive printed folio
    corroborates that continuity; otherwise leave the ownership unresolved.
    """
    updated = _copy_page_parts(page_parts)
    evidence_by_page: dict[int, tuple[str, ...]] = {}
    page = 1
    while page <= page_count:
        opening = _heading_window(page_text.get(page, ""), lines=10)
        if (
            parts_on_page(updated.get(page)) != [MAIN_PETITION_PART]
            or not _has_leading_high_court_caption(page_text.get(page, ""))
            or not re.search(r"\bpetition\b", opening, re.IGNORECASE)
        ):
            page += 1
            continue
        start = page
        end = start
        while (
            end < page_count
            and parts_on_page(updated.get(end + 1)) == [MAIN_PETITION_PART]
            and not _looks_like_sci_main_petition(page_text.get(end + 1, ""))
            and not annexure_label_from_text(page_text.get(end + 1, ""))
        ):
            end += 1

        preceding = parts_on_page(updated.get(start - 1))
        annexure = next(
            (label for label in preceding if family_split_name(label) == ANNEXURE_FAMILY),
            None,
        )
        following = parts_on_page(updated.get(end + 1))
        next_annexure = any(
            family_split_name(label) == ANNEXURE_FAMILY for label in following
        )
        prior_folio = _printed_folio(page_text.get(start - 1, ""))
        opening_folio = _printed_folio(page_text.get(start, ""))
        folio_continues = bool(
            prior_folio
            and opening_folio
            and prior_folio[0] == opening_folio[0] == "number"
            and opening_folio[1] == prior_folio[1] + 1
        )
        owner = (
            annexure
            if annexure and (next_annexure or folio_continues)
            else "Undefined"
        )
        evidence = ["leading_lower_court_caption", "not_sci_main_petition"]
        if owner != "Undefined":
            evidence.append("preceding_annexure_owner")
            evidence.append(
                "next_annexure_boundary"
                if next_annexure
                else "consecutive_printed_folio"
            )
        else:
            evidence.append("annexure_continuity_unconfirmed")
        for source_page in range(start, end + 1):
            updated[source_page] = [owner]
            evidence_by_page[source_page] = tuple(evidence)
        page = end + 1
    return updated, evidence_by_page


def _stabilize_annexure_zone_and_single_continuation(
    page_parts: PagePartMap,
    page_text: Mapping[int, str],
    page_count: int,
) -> tuple[PagePartMap, dict[int, tuple[str, ...]]]:
    """Ensure outer documents have a single contiguous run, and annexures remain continuous.

    1. Outer petition documents (Checklist, Cover, Index, Office Report, Listing,
       Synopsis, LOD, Impugned Order, Main Petition, Certificate, Affidavit, Appendix,
       and front Record of Proceedings) keep strictly their first contiguous run.
    2. Any later occurrence of these outer doc types inside the Annexure Zone
       (between petition closing and back matter) is an enclosed exhibit/lower court
       filing and is demoted to ANNEXURE_FAMILY.
    3. Trailing Index (immediately preceding Vakalatnama in back matter) is renamed
       to Filing Memo.
    4. Scanned/unassigned/uncategorized pages inside the Annexure Zone inherit the
       active Annexure label instead of dropping into Undefined.
    5. Genuine outer Applications (e.g. Supreme Court I.A.s) between whole annexures
       or in back matter are preserved.
    """
    updated = _copy_page_parts(page_parts)
    evidence_by_page: dict[int, tuple[str, ...]] = {}

    main_pages = sorted(
        page
        for page, names in updated.items()
        if MAIN_PETITION_PART in parts_on_page(names)
    )
    if not main_pages:
        return updated, evidence_by_page

    has_any_annexure = any(
        family_split_name(name) == ANNEXURE_FAMILY
        for names in updated.values()
        for name in parts_on_page(names)
    )
    if not has_any_annexure:
        return updated, evidence_by_page

    outer_main_start = main_pages[0]
    shell_parts = {
        MAIN_PETITION_PART,
        "AOR's Certificate",
        "AOR's Declaration",
        "Affidavit",
        "Appendix",
    }
    outer_shell_end = outer_main_start
    for p in range(outer_main_start, page_count + 1):
        names = parts_on_page(updated.get(p))
        if any(name in shell_parts for name in names):
            outer_shell_end = p
        elif names:
            break

    annexure_zone_start = outer_shell_end + 1
    if annexure_zone_start > page_count:
        return updated, evidence_by_page

    outer_parts = {
        "Advocate's Checklist",
        "Cover Page",
        "Index",
        "Office Report on Limitation",
        "Listing Proforma",
        "Synopsis",
        "List of Dates & Events",
        "Impugned Order",
        MAIN_PETITION_PART,
        "AOR's Certificate",
        "AOR's Declaration",
        "Affidavit",
        "Appendix",
        "Record of Proceedings",
    }

    tail_closing_markers = {
        "Vakalatnama",
        "Filing Memo",
        "Memo of Appearance",
        "PoA/BR",
        "Memo of Parties",
    }
    last_page_names = parts_on_page(updated.get(page_count))
    if "Affidavit" in last_page_names:
        tail_closing_markers.add("Affidavit")

    back_matter_start = page_count + 1
    for p in range(annexure_zone_start, page_count + 1):
        names = parts_on_page(updated.get(p))
        if any(name in tail_closing_markers for name in names):
            back_matter_start = p
            break

    # If 1-2 pages immediately before back_matter_start is Index or Proof of Service, include in back matter
    if back_matter_start <= page_count:
        if back_matter_start > annexure_zone_start:
            prev_names = parts_on_page(updated.get(back_matter_start - 1))
            if "Index" in prev_names:
                back_matter_start -= 1
                updated[back_matter_start] = ["Filing Memo"]
                evidence_by_page[back_matter_start] = ("trailing_index_is_filing_memo",)
            elif "Proof of Service" in prev_names:
                back_matter_start -= 1

    active_annexure = "Annexures"
    for p in range(annexure_zone_start, min(annexure_zone_start + 10, page_count + 1)):
        for name in parts_on_page(updated.get(p)):
            if family_split_name(name) == ANNEXURE_FAMILY:
                active_annexure = name
                break

    # Determine qualified outer applications in post-petition zone:
    # A genuine outer Application must be >= 2 pages and not followed by a large block of annexures (> 15 pages)
    app_pages = sorted(
        p for p in range(annexure_zone_start, back_matter_start)
        if any(family_split_name(n) == APPLICATION_FAMILY for n in parts_on_page(updated.get(p)))
    )
    keep_apps: set[int] = set()
    if app_pages:
        from .document_parts import _contiguous_groups
        for st, en in _contiguous_groups(app_pages):
            count = en - st + 1
            annex_after = sum(
                1 for p in range(en + 1, back_matter_start)
                if any(family_split_name(n) == ANNEXURE_FAMILY for n in parts_on_page(updated.get(p)))
                or not parts_on_page(updated.get(p))
                or parts_on_page(updated.get(p)) in (["uncategorized"], ["Undefined"], ["Unidentified"])
            )
            if count >= 2 and annex_after <= 15:
                keep_apps.update(range(st, en + 1))

    for p in range(annexure_zone_start, back_matter_start):
        names = parts_on_page(updated.get(p))
        text = page_text.get(p, "")

        is_outer_app = (
            p in keep_apps
            or (
                page_starts_application(text)
                and _is_sci_caption(text)
                and not _is_lower_court_caption(text)
            )
        )
        if is_outer_app:
            continue

        stamp = annexure_ref_in_heading(text)
        if stamp:
            active_annexure = stamp.label
            updated[p] = [active_annexure]
            evidence_by_page[p] = ("stamped_annexure_opening",)
            continue

        has_outer = any(n in outer_parts or n == "Proof of Service" for n in names)
        if has_outer:
            updated[p] = [active_annexure]
            evidence_by_page[p] = ("demoted_outer_document_in_annexure_zone",)
            continue

        existing_annex = [n for n in names if family_split_name(n) == ANNEXURE_FAMILY]
        if existing_annex:
            active_annexure = existing_annex[0]
            continue

        updated[p] = [active_annexure]
        evidence_by_page[p] = ("annexure_zone_continuation",)

    # In back matter:
    for p in range(back_matter_start, page_count + 1):
        names = parts_on_page(updated.get(p))
        if "Index" in names:
            updated[p] = ["Filing Memo"]
            evidence_by_page[p] = ("trailing_index_is_filing_memo",)
        elif not names or names == ["uncategorized"] or names == ["Undefined"] or names == ["Unidentified"]:
            prev = parts_on_page(updated.get(p - 1))
            if prev and (
                family_split_name(prev[0]) in {ANNEXURE_FAMILY, APPLICATION_FAMILY}
                or prev[0] in {"Proof of Service", "Filing Memo", "Vakalatnama", "Memo of Appearance"}
            ):
                updated[p] = prev
                evidence_by_page[p] = ("back_matter_continuation",)

    return updated, evidence_by_page


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
    authoritative_boundary_pages: Sequence[int] = (),
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
    assignment_events: dict[int, list[dict[str, Any]]] = {}

    def apply_map(
        stage: str,
        candidate: PagePartMap,
        *,
        reason: str,
        evidence: Sequence[str] = (),
        evidence_by_page: Mapping[int, Sequence[str]] | None = None,
    ) -> None:
        nonlocal page_parts
        _trace_assignment_stage(
            assignment_events,
            stage,
            page_parts,
            candidate,
            len(units),
            reason=reason,
            evidence=evidence,
            evidence_by_page=evidence_by_page,
        )
        page_parts = candidate

    before_classification_seed = _copy_page_parts(page_parts)
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
    _trace_assignment_stage(
        assignment_events,
        "high_confidence_empty_page_seed",
        before_classification_seed,
        page_parts,
        len(units),
        reason="previously_unassigned_page_has_direct_document_start",
        evidence_by_page={c.page: c.signals for c in classifications},
    )

    # Llama remains the warm-start, but an isolated page-level label change is
    # not a legal-document boundary.  Once a directly headed document opens,
    # preserve its continuous pages until the next independently supported
    # opening.  Keep llama_only byte-for-byte compatible by applying this gate
    # only when semantic reconciliation is enabled.
    if run_hybrid_repair:
        apply_map(
            "document_continuity",
            _stabilize_document_continuity(
                page_parts,
                classifications,
                authoritative_boundary_pages=authoritative_boundary_pages,
            ),
            reason="unsupported_page_level_transition_inside_open_document",
            evidence=("page_classification_signals", "authoritative_boundary_starts"),
        )

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
        _trace_assignment_stage(
            assignment_events,
            "repair_proposal",
            baseline,
            proposed,
            len(units),
            reason="deterministic_repair_candidate_not_yet_committed",
            evidence=("local_structure_rules",),
            status="proposed",
        )

        def reassess_current_map(current: PagePartMap, _pass_index: int) -> PagePartMap:
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

        reconciled, reconciliation_changes = reconcile_compiled_split(
            baseline,
            proposed,
            classifications,
            texts,
            page_count=len(units),
            operation_log=reconciliation_operations,
            proposal_generator=reassess_current_map,
            page_units=units,
            authoritative_boundary_pages=authoritative_boundary_pages,
        )
        apply_map(
            "segment_reconciliation",
            reconciled,
            reason="validated_reconciliation_operation",
            evidence=("reconciliation_operation_log",),
            evidence_by_page={
                change.page: change.evidence for change in reconciliation_changes
            },
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
            "accepted": sum(
                item.status == "accepted" for item in reconciliation_operations
            ),
            "rejected": sum(
                item.status in {"rejected", "rolled_back"}
                for item in reconciliation_operations
            ),
            "left_for_review": sum(
                item.status == "left_for_review" for item in reconciliation_operations
            ),
        }
        # Reconciliation is intentionally conservative and may reject a
        # multi-page correction when several Llama page labels disagree at
        # once. Reapply only the hard structural invariants afterwards:
        # explicit outer headings, the challenged order's bounded position,
        # and stamp-to-next-stamp annexure ownership. These are deterministic
        # document boundaries, not another model guess.
        blocked_outer_decisions: list[dict[str, Any]] = []
        apply_map(
            "post_reconciliation_outer_anchors",
            _apply_outer_anchors(
                page_parts,
                texts,
                len(units),
                authoritative_boundary_pages=authoritative_boundary_pages,
                protect_established_annexures=True,
                decision_log=blocked_outer_decisions,
            ),
            reason="explicit_outer_heading_outside_established_annexure_continuation",
            evidence=("outer_document_heading", "boundary_start_or_non_annexure_context"),
        )
        for decision in blocked_outer_decisions:
            assignment_events.setdefault(decision["source_page"], []).append(decision)
        apply_map(
            "front_impugned_judgment",
            _restore_front_impugned_judgment(page_parts, texts, len(units)),
            reason="bounded_challenged_judgment_before_petition",
            evidence=("judgment_opening", "following_petition_boundary"),
        )
        index_supported_pages = [
            change.page
            for change in reconciliation_changes
            if "master_index_and_folio" in change.evidence
        ]
        apply_map(
            "annexure_nesting",
            _force_annexure_nesting(
                page_parts,
                texts,
                len(units),
                strict_continuity=True,
                preserve_pages=index_supported_pages,
            ),
            reason="stamped_annexure_continues_until_independent_boundary",
            evidence=("annexure_stamp", "next_document_boundary"),
        )
        # A scanned Index can identify an exact document span even when the
        # pages' own folio OCR is unreadable. Apply only unique date/case or
        # purpose-matched openings with a verified following boundary.
        apply_map(
            "strong_index_content_spans",
            _apply_strong_index_content_spans(page_parts, texts, len(units)),
            reason="unique_index_row_matches_document_opening_and_next_boundary",
            evidence=("index_row", "document_opening", "following_boundary"),
        )
        apply_map(
            "early_record_of_proceedings",
            _restore_early_record_of_proceedings(page_parts, texts, len(units)),
            reason="record_of_proceedings_continuation_before_next_heading",
            evidence=("record_of_proceedings_opening", "next_document_heading"),
        )
        lower_court_map, lower_court_evidence = _restore_annexed_lower_court_petitions(
            page_parts, texts, len(units)
        )
        apply_map(
            "lower_court_petition_ownership",
            lower_court_map,
            reason="lower_court_petition_cannot_be_sci_main_petition",
            evidence_by_page=lower_court_evidence,
        )
        stabilized_map, stabilized_evidence = _stabilize_annexure_zone_and_single_continuation(
            page_parts, texts, len(units)
        )
        apply_map(
            "annexure_zone_and_single_continuation",
            stabilized_map,
            reason="single_continuation_outer_docs_and_annexure_continuity",
            evidence_by_page=stabilized_evidence,
        )
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
        assignment_events=assignment_events,
    )
