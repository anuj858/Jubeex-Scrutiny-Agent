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
from collections.abc import Mapping, Sequence
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
    page_starts_application,
    parts_on_page,
)
from .split_pdf_layout import extract_split_layout
from .split_repair import (
    _heading_window,
    _is_near_blank_page,
    _looks_like_sci_main_petition,
    _outer_anchor_label,
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
    "AOR's Certificate",
    "Affidavit",
    "Appendix",
    "Filing Memo",
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


@dataclass
class StructureSplitResult:
    page_units: list[PageUnit]
    classifications: list[PageClassification]
    boundaries: list[BoundaryCandidate]
    logical_documents: list[LogicalDocument]
    page_parts: PagePartMap
    duplicates: list[Any] = field(default_factory=list)
    ocr_needed_pages: list[int] = field(default_factory=list)

    def report(self) -> dict[str, Any]:
        return {
            "architecture": "structure_aware_v1",
            "page_count": len(self.page_units),
            "ocr_needed_pages": list(self.ocr_needed_pages),
            "ocr_needed_count": len(self.ocr_needed_pages),
            "classifications": [c.as_dict() for c in self.classifications],
            "boundaries": [b.as_dict() for b in self.boundaries],
            "logical_documents": [d.as_dict() for d in self.logical_documents],
            "auto_boundaries": sum(1 for b in self.boundaries if b.action == "auto"),
            "verify_boundaries": sum(1 for b in self.boundaries if b.action == "verify"),
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
    units: list[PageUnit] = []
    for number in range(1, total + 1):
        text = ""
        if page_texts is not None and number in page_texts:
            text = page_texts.get(number) or ""
        if not (text or "").strip():
            try:
                text = reader.pages[number - 1].extract_text() or ""
            except Exception:
                text = ""
        alt, index_table, folio = layout.get(number, ("", None, None))
        if len(alt.strip()) > len((text or "").strip()):
            text = alt
        if index_table:
            text = index_table
        elif folio and text.rstrip().splitlines()[-1:] != [folio]:
            # Put the geometrically verified margin folio last for repair's
            # text-only reader; stamps and body text keep their full content.
            text = text.rstrip() + "\n" + folio
        stripped = (text or "").strip()
        requires_ocr = len(stripped) < _OCR_TEXT_MIN
        lines = [line.strip() for line in (text or "").splitlines() if line.strip()]
        header = "\n".join(lines[:3])
        footer = "\n".join(lines[-2:]) if len(lines) > 3 else ""
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
        elif page_starts_application(text):
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
    llama_page_parts: PagePartMap | Mapping[int, Sequence[str] | str | None] | None = None,
) -> list[PageClassification]:
    hints: dict[int, str] = {}
    if llama_page_parts:
        for page, names in llama_page_parts.items():
            parts = parts_on_page(names)
            if parts:
                hints[int(page)] = parts[0]
    return [
        classify_page(unit, llama_hint=hints.get(unit.pdf_page)) for unit in units
    ]


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
) -> list[BoundaryCandidate]:
    """Stage B — candidate boundaries with auto / verify / skip actions."""
    by_page = {unit.pdf_page: unit for unit in units}
    boundaries: list[BoundaryCandidate] = []
    prev: PageClassification | None = None
    prev_unit: PageUnit | None = None
    for curr in classifications:
        unit = by_page.get(curr.page) or PageUnit(pdf_page=curr.page, text="")
        score, signals = _boundary_score(prev, curr, unit, prev_unit)
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
        "AOR's Certificate",
        "Filing Memo",
        "Memo of Parties",
        "Cover Page",
        "Office Report on Limitation",
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
                and family_split_name(c.document_type)
                == family_split_name(current)
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


def structure_aware_split(
    pdf_bytes: bytes,
    *,
    llama_page_parts: PagePartMap | Mapping[int, Sequence[str] | str | None] | None = None,
    page_texts: Mapping[int, str] | None = None,
    source_pdf: str = "bundle.pdf",
    run_hybrid_repair: bool = True,
) -> StructureSplitResult:
    """Full structure-aware split. Physical slicing stays outside this function.

    Classification + boundaries are recorded for audit. Label authority is:
    LlamaSplit warm-start (if any) → strong local anchors on empty pages →
    hybrid repair (fill gaps / demote / nest). Do not paint every classified
    page into ``page_parts`` before repair — that created one-page Index /
    Listing / Synopsis islands when a continuation page was mis-typed.
    """
    units = extract_page_units(
        pdf_bytes, source_pdf=source_pdf, page_texts=page_texts
    )
    texts = {unit.pdf_page: unit.text for unit in units}

    classifications = classify_pages(units, llama_page_parts=llama_page_parts)
    boundaries = detect_boundaries(classifications, units)

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
    if run_hybrid_repair and units:
        page_parts, duplicates = repair_compiled_split(
            page_parts,
            texts,
            page_count=len(units),
        )
    elif not run_hybrid_repair:
        # Offline path: resolve carries without hybrid repair.
        page_labels = _resolve_page_labels(classifications, boundaries)
        page_parts = logical_documents_to_page_parts(
            build_logical_documents(page_labels, classifications)
        )

    repaired_labels = {
        page: parts_on_page(names)[0]
        for page, names in page_parts.items()
        if parts_on_page(names)
    }
    logical_docs = build_logical_documents(repaired_labels, classifications)

    ocr_pages = [unit.pdf_page for unit in units if unit.requires_ocr]
    return StructureSplitResult(
        page_units=list(units),
        classifications=classifications,
        boundaries=boundaries,
        logical_documents=logical_docs,
        page_parts=page_parts,
        duplicates=duplicates,
        ocr_needed_pages=ocr_pages,
    )
