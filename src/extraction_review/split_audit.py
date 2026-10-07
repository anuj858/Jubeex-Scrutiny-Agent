"""Post-split audit: paper-book sequence + Index ↔ file consistency.

SCI compiled petitions only for now. Flags are advisory (stored on STEP_SPLIT);
they do not block slicing.
"""

from __future__ import annotations

import re
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, replace
from typing import Any

from .document_parts import (
    PagePartMap,
    document_spans_from_page_parts,
    family_split_name,
    parts_on_page,
)
from .split_pdf_layout import letter_folio_number, printed_folio

# Parts that commonly sit outside the Part-I Index table.
_INDEX_OPTIONAL_PARTS = frozenset(
    {
        "Index",
        "Cover Page",
        "Advocate's Checklist",
        "Record of Proceedings",
        "Filing Memo",
        "Court Fees",
        "PoA/BR",
        "Undefined",
        "AOR's Certificate",  # often omitted from Part-I Index table
    }
)

# Expected outer order for SCI paper books (D-225 / catalog-aligned).
_SEQUENCE_ORDER: tuple[str, ...] = (
    "AOR's Declaration",
    "Advocate's Checklist",
    "Cover Page",
    "Record of Proceedings",
    "Index",
    "Office Report on Limitation",
    "Listing Proforma",
    "Synopsis",
    "List of Dates & Events",
    "Impugned Order",
    "Main Petition",
    "Affidavit",
    "AOR's Certificate",
    "Appendix",
    "Annexures",  # placeholder rank for any Annexure P-n
    "Applications",  # placeholder rank for any Application n
    "Memo of Parties",
    "Memo of Appearance",
    "Vakalatnama",
    "PoA/BR",
    "Court Fees",
    "Filing Memo",
)

_SEQUENCE_RANK = {name: index for index, name in enumerate(_SEQUENCE_ORDER)}

_COMBINED_SEQUENCE_ALIASES: dict[str, tuple[str, ...]] = {
    "Synopsis + List of Dates & Events": ("Synopsis", "List of Dates & Events"),
    "Memo of Appearance + Vakalatnama": ("Memo of Appearance", "Vakalatnama"),
}

_INDEX_ROW_RE = re.compile(
    r"(?m)^\s*(?:(?P<sno>\d{1,3})[.)]\s*)(?P<body>.+?)\s+"
    r"(?P<start>\d{1,4})(?:\s*[-–—/]\s*(?P<end>\d{1,4}))?\s*$"
)
# Softer row: serial + particulars, page number may be on the same line mid-OCR.
_INDEX_SOFT_ROW_RE = re.compile(
    r"(?m)^\s*(?P<sno>\d{1,3})[.)]\s+(?P<body>.+?)"
    r"(?:\s+(?P<start>\d{1,4})(?:\s*[-–—/]\s*(?P<end>\d{1,4}))?)?\s*$"
)

_ANNEXURE_IN_INDEX_RE = re.compile(
    r"annexure[\s\-]*([a-z])?[\s\-/\.]*(\d+)", re.IGNORECASE
)
# Some outer Index tables put the exhibit identifier in its own column, so a
# row may read ``Copy of order ... P-6 65-92`` without the word Annexure.
# This is applied only to Index particulars, never to body prose.
_SERIES_LABEL_IN_INDEX_RE = re.compile(
    r"(?<![A-Za-z0-9./])([A-Z])\s*[-/]\s*(\d+)\b(?!/\d)"
)

# SCI paper-book cites use a letter series: P (Petitioner) / R (Respondent) /
# E (exhibit → P). Bare "Annexure 5" inside an annexed HC writ is not an
# inventory row for the outer petition.
_SCI_ANNEXURE_MENTION_RE = re.compile(
    r"\bannexure\s*[-–—:/\s]*(petitioner|respondent|[per])\s*[-–—./\s]*(\d+)\b",
    re.IGNORECASE,
)
_APPLICATION_IN_INDEX_RE = re.compile(
    r"(?:application|i\.?\s*a\.?)[\s\-]*(?:no\.?\s*)?(\d+)|"
    r"\bi\.?\s*a\.?\b",
    re.IGNORECASE,
)

# Parts whose body text is trusted for annexure inventory (plus Index rows).
_ANNEXURE_INVENTORY_PARTS = frozenset(
    {
        "Index",
        "List of Dates & Events",
        "Synopsis + List of Dates & Events",
        "Main Petition",
    }
)


@dataclass(frozen=True)
class IndexRow:
    particulars: str
    start_page: int
    end_page: int
    mapped_part: str | None
    serial: int | None = None

    def as_dict(self) -> dict[str, Any]:
        return {
            "particulars": self.particulars,
            "start_page": self.start_page,
            "end_page": self.end_page,
            "mapped_part": self.mapped_part,
            "serial": self.serial,
        }


@dataclass(frozen=True)
class SplitAuditFlag:
    code: str
    message: str
    severity: str = "warning"
    part: str | None = None
    index_particulars: str | None = None
    expected_pages: dict[str, int] | None = None
    found_pages: dict[str, int] | None = None
    details: dict[str, Any] | None = None

    def as_dict(self) -> dict[str, Any]:
        payload: dict[str, Any] = {
            "code": self.code,
            "message": self.message,
            "severity": self.severity,
        }
        if self.part is not None:
            payload["part"] = self.part
        if self.index_particulars is not None:
            payload["index_particulars"] = self.index_particulars
        if self.expected_pages is not None:
            payload["expected_pages"] = self.expected_pages
        if self.found_pages is not None:
            payload["found_pages"] = self.found_pages
        if self.details:
            payload["details"] = self.details
        return payload


def _fold(text: str) -> str:
    return re.sub(r"\s+", " ", (text or "").casefold()).strip()


def map_index_particulars_to_part(particulars: str) -> str | None:
    """Map an Index row's particulars text onto a Split part name."""
    text = _fold(particulars)
    if not text:
        return None
    # Skip section headers / blank rows.
    if text in {"part i", "part – i", "part - i", "part ii", "part – ii", "part - ii"}:
        return None
    if text.startswith("s.no") or text.startswith("particulars"):
        return None

    annex = _ANNEXURE_IN_INDEX_RE.search(particulars)
    if annex:
        return normalize_annexure_part_label(annex.group(1), int(annex.group(2)))
    series_label = _SERIES_LABEL_IN_INDEX_RE.search(particulars or "")
    if series_label:
        return normalize_annexure_part_label(
            series_label.group(1), int(series_label.group(2))
        )

    # An IA description can mention the SLP, its affidavit or impugned order.
    # Its own title determines the document type, not those references.
    if re.match(
        r"^[^a-z0-9]{0,3}(?:[il]\.?\s*a\.?\s*(?:no\.?|\b)|"
        r"(?:an?\s+)?application\b)",
        text,
    ) or "application for" in text:
        return "Application 1"

    if "fresh case" in text and "report" in text:
        return None
    if "office report on limitation" in text or "o/r on limitation" in text or (
        "limitation" in text and "report" in text
    ):
        return "Office Report on Limitation"
    if "listing" in text and "proforma" in text:
        return "Listing Proforma"
    if "proforma for first listing" in text:
        return "Listing Proforma"
    if "synopsis" in text and ("list of date" in text or "dates" in text):
        # Combined row — prefer Synopsis as primary; LOD checked via siblings.
        return "Synopsis"
    if text.startswith("synopsis") or " synopsis" in f" {text}":
        return "Synopsis"
    if "list of date" in text or "list of dates" in text:
        return "List of Dates & Events"
    if "impugned" in text and ("order" in text or "judgment" in text or "judgement" in text):
        return "Impugned Order"
    if re.search(r"\b(?:criminal|civil)\s+appeal\b", text):
        # The Index names an appeal pleading directly (often "Criminal Appeal
        # with affidavit") rather than calling it a petition or Form 28.
        return "Main Petition"
    if "special leave" in text or "form 28" in text or (
        "petition" in text
        and "transfer" not in text
        and "writ" not in text
        and "application" not in text
    ):
        # "SLP along with Affidavit" is still the petition slot in the Index.
        return "Main Petition"
    if "affidavit" in text:
        return "Affidavit"
    if "appendix" in text:
        return "Appendix"
    if "vakalatnama" in text or "vakalatnama" in re.sub(r"\s+", "", text):
        return "Vakalatnama"
    if "memo of appearance" in text or "memorandum of appearance" in text:
        return "Memo of Appearance"
    if "memo of parties" in text or "memorandum of parties" in text:
        return "Memo of Parties"
    if "declaration" in text and ("aor" in text or "defect" in text or "refiling" in text):
        return "AOR's Declaration"
    if "certificate" in text and ("aor" in text or "advocate" in text or "confined" in text):
        return "AOR's Certificate"
    if "advocate" in text and "check" in text:
        return "Advocate's Checklist"
    if "cover" in text and "page" in text:
        return "Cover Page"
    if "record of proceeding" in text:
        return "Record of Proceedings"
    if "filing memo" in text or "index of filing" in text or "filing index" in text:
        return "Filing Memo"
    if "court fee" in text or "payment receipt" in text:
        return "Court Fees"
    if "application" in text or re.search(r"\bi\.?\s*a\.?\b", text):
        match = _APPLICATION_IN_INDEX_RE.search(particulars)
        if match and match.lastindex and match.group(1):
            return f"Application {int(match.group(1))}"
        return "Application 1"
    return None


def parse_index_rows(index_text: str) -> list[IndexRow]:
    """Parse PARTICULARS + page span rows from Index page text."""
    # Flatten OCR line-breaks inside a row: join lines that don't start a new serial.
    # Tab-separated rows come from verified PDF table cells. Keep the page
    # column separate from dates, case numbers and section numbers in prose.
    if any(re.match(r"^\d+[.)]\t", line) for line in index_text.splitlines()):
        rows = []
        for line in index_text.splitlines():
            cells = line.split("\t")
            if len(cells) != 3 or not re.fullmatch(r"\d+[.)]", cells[0]):
                continue
            spans = _span_only_line(cells[2])
            span = spans[0] if len(spans) == 1 and spans[0].kind == "number" else None
            rows.append(
                IndexRow(
                    particulars=cells[1],
                    start_page=span.start if span else 0,
                    end_page=span.end if span else 0,
                    mapped_part=map_index_particulars_to_part(cells[1]),
                    serial=int(cells[0][:-1]),
                )
            )
        return rows
    raw_lines = (index_text or "").splitlines()
    merged: list[str] = []
    for line in raw_lines:
        stripped = line.strip()
        if not stripped:
            continue
        if re.match(r"^\d{1,3}[.)](?:\s+|$)", stripped) or not merged:
            merged.append(stripped)
        else:
            merged[-1] = f"{merged[-1]} {stripped}"
    normalized = "\n".join(merged)

    rows: list[IndexRow] = []
    seen: set[tuple[str, int, int]] = set()
    for match in _INDEX_SOFT_ROW_RE.finditer(normalized):
        body = (match.group("body") or "").strip()
        original_body = re.sub(r"^\s*\d+[.)]\s*", "", match.group(0)).strip()
        if not body or len(body) < 3:
            continue
        folded = _fold(body)
        if "page no" in folded or folded.startswith("particulars"):
            continue
        start_raw = match.group("start")
        if not start_raw:
            # Try to pull a trailing page span out of the body (OCR glue).
            trailing = re.search(
                r"(\d{1,4})(?:\s*[-–—/]\s*(\d{1,4}))?\s*$", body
            )
            # In scanned tables OCR commonly reads across the page-number
            # column and then resumes the wrapped particulars, for example
            # ``Annexure P-1: 24-46 A copy of order ...``.
            embedded = re.search(
                r"(?<![.\d])(\d{1,4})\s*[-–—]\s*(\d{1,4})(?![.\d])",
                body,
            )
            if embedded:
                start = int(embedded.group(1))
                end = int(embedded.group(2))
                body = (body[: embedded.start()] + " " + body[embedded.end() :]).strip()
                trailing = None
            elif trailing:
                start = int(trailing.group(1))
                end = int(trailing.group(2) or start)
                body = body[: trailing.start()].strip(" -–—/")
            else:
                # Still keep mapped rows without pages for missing-in-file checks.
                mapped = map_index_particulars_to_part(body)
                if not mapped:
                    continue
                serial = int(match.group("sno")) if match.group("sno") else None
                rows.append(
                    IndexRow(
                        particulars=body,
                        start_page=0,
                        end_page=0,
                        mapped_part=mapped,
                        serial=serial,
                    )
                )
                continue
        else:
            start = int(start_raw)
            end_raw = match.group("end")
            end = int(end_raw) if end_raw else start
        # A final year, date component, annexure ID or case number is part of
        # the description, not an implicit page column.
        if start == end and (
            1900 <= start <= 2099
            or re.search(
                r"(?:\bno\.?|\bsection|\bof|\d[./]|[A-Z][- /])\s*$", body, re.IGNORECASE
            )
            or _ANNEXURE_IN_INDEX_RE.fullmatch(original_body)
        ):
            body = original_body
            start = end = 0
        if end < start:
            # Do not turn a damaged OCR value such as 55-43 into an
            # authoritative 43-55 range. Keep the Index row for inventory
            # matching, but discard its unsafe page coordinates.
            start = end = 0
        serial = int(match.group("sno")) if match.group("sno") else None
        mapped = map_index_particulars_to_part(body)
        key = (_fold(body), start, end)
        if key in seen:
            continue
        seen.add(key)
        rows.append(
            IndexRow(
                particulars=body,
                start_page=start,
                end_page=end,
                mapped_part=mapped,
                serial=serial,
            )
        )
    return rows


@dataclass(frozen=True)
class IndexPrintedRow:
    """Index document tied to the printed folio numbers on the sheets.

    ``kind="number"`` is 36 or 61–63B. ``kind="letter"`` is B–I.
    ``end_suffix`` is the letter on the last folio (63B → ``B``).
    """

    mapped_part: str | None
    particulars: str
    kind: str
    start: int
    end: int
    end_suffix: str = ""


_SPAN_TOKEN_RE = re.compile(
    r"^(?:"
    r"(?P<n1>\d{1,4})(?:(?P<s1>[A-Za-z])?(?:\s*[-–—]\s*(?P<n2>\d{1,4})(?P<s2>[A-Za-z])?)?)?"
    r"|(?P<L1>[A-Za-z]{1,2})\s*[-–—]\s*(?P<L2>[A-Za-z]{1,2})"
    r"|A\s*(?P<a1>\d{1,2})\s*[-–—]\s*A\s*-?\s*(?P<a2>\d{1,2})"
    r"|A(?P<aonly>\d{1,2})"
    r")$",
    re.IGNORECASE,
)
_LEADING_SPAN_RE = re.compile(
    r"^(?P<n1>\d{1,4})(?:\s*[-–—]\s*(?P<n2>\d{1,4}))?(?P<s2>[A-Za-z])?(?P<rest>\D.+)$"
)


def _printed_span_from_token(token: str) -> IndexPrintedRow | None:
    if re.fullmatch(r"[A-Za-z]{1,2}", token.strip()):
        number = letter_folio_number(token)
        return IndexPrintedRow(None, token, "letter", number, number)
    match = _SPAN_TOKEN_RE.fullmatch(token.strip())
    if not match:
        return None
    if match.group("a1") or match.group("aonly"):
        start = int(match.group("a1") or match.group("aonly"))
        end = int(match.group("a2") or start)
        return IndexPrintedRow(
            mapped_part=None,
            particulars=token,
            kind="prefixed",
            start=start,
            end=end,
        )
    if match.group("L1"):
        return IndexPrintedRow(
            mapped_part=None,
            particulars=token,
            kind="letter",
            start=letter_folio_number(match.group("L1")),
            end=letter_folio_number(match.group("L2")),
        )
    start = int(match.group("n1"))
    end = int(match.group("n2") or start)
    if end < start:
        # A legal paper-book range is ascending. Reversing OCR corruption such
        # as ``55-43`` creates a believable but incorrect Annexure range.
        return None
    suffix = (match.group("s2") or match.group("s1") or "").upper()
    return IndexPrintedRow(
        mapped_part=None,
        particulars=token,
        kind="number",
        start=start,
        end=end,
        end_suffix=suffix,
    )


def _span_only_line(line: str) -> list[IndexPrintedRow]:
    whole = _printed_span_from_token(
        re.sub(r"\s+to\s+", "-", line.strip(), flags=re.IGNORECASE)
    )
    if whole:
        return [whole]
    tokens = [token for token in line.split() if token]
    if not tokens:
        return []
    spans: list[IndexPrintedRow] = []
    for token in tokens:
        span = _printed_span_from_token(token.strip(".,"))
        if span is None:
            return []
        spans.append(span)
    return spans


def _dedupe_spans(spans: Sequence[IndexPrintedRow]) -> list[IndexPrintedRow]:
    seen: set[tuple[str, int, int, str]] = set()
    unique: list[IndexPrintedRow] = []
    for span in spans:
        key = (span.kind, span.start, span.end, span.end_suffix)
        if key in seen:
            continue
        seen.add(key)
        unique.append(span)
    return unique


def index_rows_with_printed_pages(index_text: str) -> list[IndexPrintedRow]:
    """Attach Index page numbers, including a column the OCR split off the rows.

    SCI indexes print the page span in its own column. Those spans come out as
    lines like ``37-48`` and ``61-63B``. They are zipped onto the rows that
    lost their page numbers, from the bottom, and only when the counts match.
    """
    tabbed_lines = [
        line
        for line in index_text.splitlines()
        if re.match(r"^\d+[.)]\t", line)
    ]
    if tabbed_lines:
        result: list[IndexPrintedRow] = []
        for line in tabbed_lines:
            cells = line.split("\t")
            if len(cells) != 3 or not re.fullmatch(r"\d+[.)]", cells[0]):
                continue
            spans = _span_only_line(cells[2])
            if len(spans) == 1:
                span = spans[0]
                result.append(
                    IndexPrintedRow(
                        mapped_part=map_index_particulars_to_part(cells[1]),
                        particulars=cells[1],
                        kind=span.kind,
                        start=span.start,
                        end=span.end,
                        end_suffix=span.end_suffix,
                    )
                )
        # Geometry-rebuilt rows have a verified page-number column. Do not
        # mix the remaining free-form OCR back into folio ranges: dates, case
        # numbers and broken cells can look like plausible page spans. The
        # plain OCR is still consumed separately for document inventory by
        # ``collect_index_annexure_entries``.
        return result
    # Parse the untouched text as well.  ``parse_index_rows`` can join a page
    # span printed on the next line to its wrapped particulars.  The orphan
    # alignment below is still needed for true table-column OCR, but removing
    # every span-only line first used to discard otherwise unambiguous rows
    # such as ``Copy of impugned judgment ...`` followed by ``1–15``.
    direct_rows = parse_index_rows(index_text)

    kept: list[str] = []
    orphans: list[IndexPrintedRow] = []
    for line in (index_text or "").splitlines():
        stripped = line.strip()
        if not stripped:
            continue
        spans = _span_only_line(stripped)
        if spans:
            orphans.extend(spans)
            continue
        kept.append(stripped)
    rows = parse_index_rows("\n".join(kept))
    if not rows:
        return []

    printed: list[IndexPrintedRow | None] = [None] * len(rows)
    unassigned: list[int] = []
    for index, row in enumerate(rows):
        start, end, suffix = row.start_page, row.end_page, ""
        if start <= 0:
            lead = _LEADING_SPAN_RE.match(row.particulars.strip())
            if lead:
                start = int(lead.group("n1"))
                end = int(lead.group("n2") or start)
                suffix = (lead.group("s2") or "").upper()
                if end < start:
                    start, end = end, start
        if start > 0:
            printed[index] = IndexPrintedRow(
                mapped_part=row.mapped_part,
                particulars=row.particulars,
                kind="number",
                start=start,
                end=end,
                end_suffix=suffix,
            )
        else:
            unassigned.append(index)

    orphans = _dedupe_spans(orphans)
    numeric = [span for span in orphans if span.kind == "number"]
    if numeric and len(numeric) == len(unassigned):
        targets = unassigned[-len(numeric) :]
        # Only the trailing rows. Leave a gap of unassigned rows in front
        # when the letter-page column is a different list.
        if targets == unassigned[-len(targets) :]:
            for index, span in zip(targets, numeric, strict=True):
                row = rows[index]
                printed[index] = IndexPrintedRow(
                    mapped_part=row.mapped_part,
                    particulars=row.particulars,
                    kind="number",
                    start=span.start,
                    end=span.end,
                    end_suffix=span.end_suffix,
                )
                unassigned.remove(index)

    aligned = [row for row in printed if row is not None and row.start > 0]
    seen_serials = {row.serial for row in rows if row.serial is not None}
    for line in kept:
        serial_match = re.match(r"^(\d{1,3})[.)]\s+(.*)$", line)
        if not serial_match:
            continue
        serial = int(serial_match.group(1))
        if serial in seen_serials:
            continue
        lead = _LEADING_SPAN_RE.match(serial_match.group(2).strip())
        if not lead:
            continue
        start = int(lead.group("n1"))
        end = int(lead.group("n2") or start)
        if end < start:
            start, end = end, start
        suffix = (lead.group("s2") or "").upper()
        rest = lead.group("rest") or ""
        # "67Undertaking" is a word, not folio suffix 67U.
        if suffix and rest[:1].isalpha():
            suffix = ""
            rest = f"{lead.group('s2')}{rest}"
        aligned.append(
            IndexPrintedRow(
                mapped_part=map_index_particulars_to_part(rest),
                particulars=line,
                kind="number",
                start=start,
                end=end,
                end_suffix=suffix,
            )
        )
    seen = {
        (_fold(row.particulars), row.start, row.end, row.mapped_part)
        for row in aligned
    }
    for row in direct_rows:
        if row.start_page <= 0:
            continue
        # Direct parsing is a narrow fallback for wrapped outer-document rows.
        # Annexure/application tables with a detached page column can contain
        # several candidate rows followed by one range; assigning that range
        # to the last row would be a guess.
        if row.mapped_part not in {
            "Synopsis",
            "List of Dates & Events",
            "Impugned Order",
            "Main Petition",
        }:
            continue
        candidate = IndexPrintedRow(
            mapped_part=row.mapped_part,
            particulars=row.particulars,
            kind="number",
            start=row.start_page,
            end=row.end_page,
            end_suffix="",
        )
        key = (
            _fold(candidate.particulars),
            candidate.start,
            candidate.end,
            candidate.mapped_part,
        )
        if key in seen:
            continue
        seen.add(key)
        aligned.append(candidate)
    return aligned


def aligned_index_printed_rows(
    page_parts: PagePartMap,
    page_text: Mapping[int, str],
) -> list[IndexPrintedRow]:
    """Printed-folio rows from the paper-book Index, when that Index is present."""
    index_text = _index_pages_text(page_parts, page_text)
    if not index_text.strip():
        return []
    return index_rows_with_printed_pages(index_text)


def normalize_annexure_part_label(series: str | None, number: int) -> str:
    """Normalize Index/body annexure cites to ``Annexure P-n`` / ``Annexure R-n``.

    - ``P`` / petitioner → Petitioner series
    - ``R`` / respondent → Respondent series
    - ``E`` / exhibit → ``P`` (paper-book alias)
    """
    raw = (series or "P").strip().lower()
    if raw.startswith("pet") or raw in {"p", "e"} or raw.startswith("exh"):
        letter = "P"
    elif raw.startswith("res") or raw == "r":
        letter = "R"
    else:
        letter = raw[:1].upper() if raw else "P"
        if letter == "E":
            letter = "P"
        if not letter.isalpha():
            letter = "P"
    return f"Annexure {letter}-{int(number)}"


def annexure_labels_from_text(text: str) -> set[str]:
    """Extract SCI ``Annexure P-n`` / ``Annexure R-n`` labels from prose."""
    found: set[str] = set()
    for match in _SCI_ANNEXURE_MENTION_RE.finditer(text or ""):
        found.add(normalize_annexure_part_label(match.group(1), int(match.group(2))))
    return found


def collect_index_annexure_entries(
    page_parts: PagePartMap,
    page_text: Mapping[int, str],
) -> list[tuple[str, str]]:
    """Ordered ``(Annexure P-n, particulars)`` rows from the Index only.

    Used to place unstamped exhibit islands after Affidavit / Certificate.
    Prefers Index serial order; falls back to annexure number.
    """
    index_text = _index_pages_text(page_parts, page_text)
    if not index_text.strip():
        return []

    # Scanned Index rows often lose serial punctuation or put the serial on
    # its own line. Explicit Annexure headings still delimit the particulars.
    # Do not mistake A-4/A-5 front-matter folios for exhibit identifiers.
    # Geometry OCR may rebuild only a few rows and append them as tab-separated
    # lines after otherwise useful plain OCR. Do not let one rebuilt row hide
    # the remaining readable Annexure headings (Defect File 014 rebuilt only
    # P-2 while its plain OCR still contained P-1 through P-8).
    # OCR geometry rows are appended to the end of each source page. If an
    # Index row wraps onto the next page, those synthetic tab rows sit in the
    # middle of its particulars and make the continuation look like a new
    # serial row. Remove them for free-text inventory parsing; they remain
    # available to the separate, page-column parser.
    explicit_text = "\n".join(
        line
        for line in index_text.splitlines()
        if not re.match(r"^\d+[.)]\t", line)
    )
    headings = list(
        re.finditer(
            r"(?im)^\s*(?:\d{1,3}[.,)]?\s*[|]?\s*)?[|]?\s*"
            r"[\[(]?\s*annexure\s*[-:~]?\s*[PR]\s*[/~–—-]?\s*\d+\b",
            explicit_text,
        )
    )
    if headings:
        explicit: list[tuple[str, str]] = []
        seen_explicit: set[str] = set()
        for i, heading in enumerate(headings):
            end = (
                headings[i + 1].start()
                if i + 1 < len(headings)
                else len(explicit_text)
            )
            labels = annexure_labels_from_text(heading.group())
            if len(labels) != 1:
                continue
            label = next(iter(labels))
            if label in seen_explicit:
                continue
            particulars = explicit_text[heading.start():end].strip()
            # The final exhibit may be followed by applications and
            # filing-list rows, which are not its particulars.
            following_row = re.search(
                r"\n\s*\d{1,3}[.)]\s*(?:\n|[A-Z])", particulars
            )
            if following_row:
                particulars = particulars[: following_row.start()].strip()
            explicit.append((label, particulars))
            seen_explicit.add(label)
        if explicit:
            return explicit

    entries: list[tuple[int, tuple[int, int], str, str]] = []
    seen: set[str] = set()
    for row_index, row in enumerate(parse_index_rows(index_text)):
        labels: set[str] = set()
        part = row.mapped_part
        if part and family_split_name(part) == "Annexures":
            labels.add(part)
        labels.update(annexure_labels_from_text(row.particulars))
        for label in sorted(labels, key=_annexure_sort_key):
            if label in seen:
                continue
            seen.add(label)
            serial = row.serial if row.serial is not None else 10_000 + row_index
            entries.append((serial, _annexure_sort_key(label), label, row.particulars))

    if not entries:
        # Free-text Index without parseable rows — keep number order.
        for label in sorted(annexure_labels_from_text(index_text), key=_annexure_sort_key):
            entries.append((10_000, _annexure_sort_key(label), label, ""))

    entries.sort(key=lambda item: (item[0], item[1], item[2]))
    return [(label, particulars) for _serial, _num, label, particulars in entries]


def _annexure_sort_key(label: str) -> tuple[int, int]:
    """Sort Petitioner (P) before Respondent (R), then by number."""
    match = re.fullmatch(r"(?i)annexure\s+([a-z])-?(\d+)", (label or "").strip())
    if not match:
        return (99, 9999)
    series = match.group(1).upper()
    series_rank = 0 if series == "P" else (1 if series == "R" else 2)
    return (series_rank, int(match.group(2)))


def collect_expected_annexures(
    page_parts: PagePartMap,
    page_text: Mapping[int, str],
) -> set[str]:
    """Annexures mentioned in Index, List of Dates, and/or Main Petition.

    Used to decide which stamped annexures belong to the paper-book versus
    stray / nested exhibits that should fall through to Unidentified.
    """
    expected: set[str] = set()

    for label, _particulars in collect_index_annexure_entries(page_parts, page_text):
        expected.add(label)

    index_text = _index_pages_text(page_parts, page_text)
    if index_text.strip():
        expected.update(annexure_labels_from_text(index_text))

    for page, names in page_parts.items():
        labels = parts_on_page(names)
        # Index free-text is handled above; only LOD + Main add more cites.
        inventory_labels = [
            label
            for label in labels
            if label in _ANNEXURE_INVENTORY_PARTS and label != "Index"
        ]
        if not inventory_labels:
            continue
        expected.update(annexure_labels_from_text(page_text.get(int(page), "")))

    return expected


def attached_annexures(page_parts: PagePartMap) -> set[str]:
    """Annexure part labels currently present in the split map."""
    found: set[str] = set()
    for names in page_parts.values():
        for part in parts_on_page(names):
            if family_split_name(part) == "Annexures":
                found.add(part)
    return found


def _index_pages_text(page_parts: PagePartMap, page_text: Mapping[int, str]) -> str:
    pages = sorted(
        page
        for page, names in page_parts.items()
        if "Index" in parts_on_page(names)
    )
    if not pages:
        pages = [
            page
            for page, text in page_text.items()
            if "index" in _fold(text[:400])
            and ("particulars" in _fold(text[:800]) or "page no" in _fold(text[:800]))
        ]
    if not pages:
        return ""
    runs: list[list[int]] = []
    for page in pages:
        if runs and page == runs[-1][-1] + 1:
            runs[-1].append(page)
        else:
            runs.append([page])

    # The first master Index is authoritative. Also include a later Index run
    # when it immediately follows an explicit paper-book Volume cover. This
    # keeps Volume II/III ranges while excluding nested lower-court indexes and
    # the final Filing Index.
    selected = list(runs[0])
    for run in runs[1:]:
        start = run[0]
        prior_text = _fold(page_text.get(start - 1, "")[:3000])
        prior_parts = parts_on_page(page_parts.get(start - 1))
        if (
            "Cover Page" in prior_parts
            and "paper book" in prior_text
            and re.search(r"\bvolume\s*[-–—]?\s*(?:i{1,3}|[1-3])\b", prior_text)
        ):
            selected.extend(run)
    return "\n".join(page_text.get(page, "") for page in selected)


def _sequence_family_rank(part: str) -> tuple[int, int]:
    """Return (block_rank, sub_rank) for ordering comparisons."""
    aliases = _COMBINED_SEQUENCE_ALIASES.get(part)
    if aliases:
        return (min(_SEQUENCE_RANK[name] for name in aliases), 0)
    folded = _fold(part)
    annex = re.fullmatch(r"annexure ([a-z])-?(\d+)", folded)
    if annex:
        series = annex.group(1)
        number = int(annex.group(2))
        # Petitioner (P) before Respondent (R) before other series.
        series_rank = 0 if series == "p" else (1 if series == "r" else 2)
        return (_SEQUENCE_RANK["Annexures"], series_rank * 1000 + number)
    app = re.fullmatch(r"application (\d+)", folded)
    if app:
        return (_SEQUENCE_RANK["Applications"], int(app.group(1)))
    if part in _SEQUENCE_RANK:
        return (_SEQUENCE_RANK[part], 0)
    family = family_split_name(part)
    if family in _SEQUENCE_RANK:
        return (_SEQUENCE_RANK[family], 0)
    return (len(_SEQUENCE_ORDER) + 10, 0)


def check_document_sequence(
    spans: Sequence[Mapping[str, Any]],
) -> list[SplitAuditFlag]:
    """Flag when consecutive found documents invert expected SCI order."""
    items: list[tuple[str, int, tuple[int, int]]] = []
    for span in spans:
        name = str(span.get("name") or "").strip()
        if not name:
            continue
        try:
            start = int(span["start_page"])
        except (KeyError, TypeError, ValueError):
            continue
        items.append((name, start, _sequence_family_rank(name)))

    items.sort(key=lambda row: row[1])
    flags: list[SplitAuditFlag] = []
    for index in range(1, len(items)):
        earlier_name, earlier_start, earlier_rank = items[index - 1]
        name, start, rank = items[index]
        if earlier_rank <= rank:
            continue
        flags.append(
            SplitAuditFlag(
                code="out_of_sequence",
                severity="warning",
                part=earlier_name,
                message=(
                    f"{earlier_name} (p.{earlier_start}) appears before "
                    f"{name} (p.{start}), but expected paper-book order "
                    f"places {name} before {earlier_name}"
                ),
                details={
                    "out_of_order_part": earlier_name,
                    "should_precede_part": name,
                    "out_of_order_start_page": earlier_start,
                    "should_precede_start_page": start,
                },
            )
        )
    return flags


def _spans_by_part(spans: Sequence[Mapping[str, Any]]) -> dict[str, tuple[int, int]]:
    by_part: dict[str, tuple[int, int]] = {}
    for span in spans:
        name = str(span.get("name") or "").strip()
        if not name:
            continue
        try:
            start = int(span["start_page"])
            end = int(span["end_page"])
        except (KeyError, TypeError, ValueError):
            continue
        by_part[name] = (start, end)
    return by_part


def check_index_consistency(
    rows: Sequence[IndexRow],
    spans: Sequence[Mapping[str, Any]],
    *,
    page_count: int,
) -> list[SplitAuditFlag]:
    """Compare Index rows to repaired split spans."""
    by_part = _spans_by_part(spans)
    flags: list[SplitAuditFlag] = []
    mentioned: set[str] = set()

    for row in rows:
        part = row.mapped_part
        if not part:
            continue
        mentioned.add(part)
        if part == "Synopsis" and "list of date" in _fold(row.particulars):
            mentioned.add("List of Dates & Events")

        # Part-II letter pages (A/A1) often OCR as tiny integers — skip mismatch.
        if (
            row.start_page > 0
            and row.start_page <= 20
            and part
            in {
                "Cover Page",
                "Listing Proforma",
                "Record of Proceedings",
                "Office Report on Limitation",
            }
        ):
            found_early = by_part.get(part)
            if found_early and abs(found_early[0] - row.start_page) >= 2:
                continue

        # Absurd OCR page numbers (advocate codes, etc.).
        if row.start_page > page_count * 2:
            continue

        if row.start_page > page_count or (
            row.end_page > page_count and row.start_page > 0
        ):
            if row.start_page <= 0:
                continue
            flags.append(
                SplitAuditFlag(
                    code="index_page_out_of_range",
                    severity="error",
                    part=part,
                    index_particulars=row.particulars,
                    expected_pages={
                        "start_page": row.start_page,
                        "end_page": row.end_page,
                    },
                    message=(
                        f"Index lists {part!r} at pp. {row.start_page}–{row.end_page} "
                        f"but the PDF only has {page_count} page(s)"
                    ),
                )
            )
            continue

        found = by_part.get(part)
        if found is None and part == "Synopsis":
            found = by_part.get("Synopsis + List of Dates & Events")
        if found is None and part == "List of Dates & Events":
            found = by_part.get("Synopsis + List of Dates & Events")
        if found is None and part in {"Vakalatnama", "Memo of Appearance"}:
            found = by_part.get("Memo of Appearance + Vakalatnama")

        if found is None:
            # Court Fee / registry Part-II rows are often not separate split docs.
            if part in {"Court Fees", "Cover Page", "Record of Proceedings"}:
                continue
            flags.append(
                SplitAuditFlag(
                    code="in_index_missing_in_file",
                    severity="error",
                    part=part,
                    index_particulars=row.particulars,
                    expected_pages=(
                        {
                            "start_page": row.start_page,
                            "end_page": row.end_page,
                        }
                        if row.start_page > 0
                        else None
                    ),
                    message=(
                        f"Index lists {part!r} ({row.particulars})"
                        + (
                            f" at pp. {row.start_page}–{row.end_page}"
                            if row.start_page > 0
                            else ""
                        )
                        + ", but that document was not found in the split file"
                    ),
                )
            )
            continue

        if row.start_page <= 0:
            continue

        found_start, found_end = found
        if not (found_start - 1 <= row.start_page <= found_end + 1):
            flags.append(
                SplitAuditFlag(
                    code="index_page_mismatch",
                    severity="warning",
                    part=part,
                    index_particulars=row.particulars,
                    expected_pages={
                        "start_page": row.start_page,
                        "end_page": row.end_page,
                    },
                    found_pages={"start_page": found_start, "end_page": found_end},
                    message=(
                        f"Index lists {part!r} at pp. {row.start_page}–{row.end_page}, "
                        f"but split found it at pp. {found_start}–{found_end}"
                    ),
                )
            )

    for part, (start, end) in sorted(by_part.items(), key=lambda item: item[1][0]):
        if part in _INDEX_OPTIONAL_PARTS:
            continue
        if part in mentioned:
            continue
        if part == "Synopsis + List of Dates & Events" and (
            "Synopsis" in mentioned or "List of Dates & Events" in mentioned
        ):
            continue
        if part == "Memo of Appearance + Vakalatnama" and (
            "Vakalatnama" in mentioned or "Memo of Appearance" in mentioned
        ):
            continue
        flags.append(
            SplitAuditFlag(
                code="in_file_missing_in_index",
                severity="warning",
                part=part,
                found_pages={"start_page": start, "end_page": end},
                message=(
                    f"{part} is present in the file (pp. {start}–{end}) "
                    f"but was not listed in the Index"
                ),
            )
        )
    return flags


def audit_compiled_split(
    page_parts: PagePartMap,
    page_text: Mapping[int, str],
    *,
    page_count: int,
) -> dict[str, Any]:
    """Run sequence + Index consistency checks on a repaired split map."""
    spans = document_spans_from_page_parts(page_parts)
    index_text = _index_pages_text(page_parts, page_text)
    rows = parse_index_rows(index_text) if index_text.strip() else []

    flags: list[SplitAuditFlag] = []
    flags.extend(check_document_sequence(spans))
    if not rows and any(
        "Index" in parts_on_page(names) for names in page_parts.values()
    ):
        flags.append(
            SplitAuditFlag(
                code="index_unparsed",
                severity="warning",
                part="Index",
                message=(
                    "Index pages were found but no particulars/page rows "
                    "could be parsed for consistency checks"
                ),
            )
        )
    elif not rows:
        flags.append(
            SplitAuditFlag(
                code="index_missing",
                severity="warning",
                part="Index",
                message="No Index document was found; Index consistency was skipped",
            )
        )
    else:
        checked_rows = rows
        if "\t" in index_text:
            # Geometry-backed Index numbers are printed folios, while split
            # spans use physical PDF pages. Compare in the same coordinate
            # system; keep the original printed rows in the audit payload.
            printed_rows = index_rows_with_printed_pages(index_text)
            by_description = {row.particulars: row for row in printed_rows}
            numeric_start = min(
                (
                    page
                    for page, names in page_parts.items()
                    if any(
                        name in {"Main Petition", "Impugned Order"}
                        or family_split_name(name) == "Annexures"
                        for name in parts_on_page(names)
                    )
                ),
                default=1,
            )
            checked_rows = []
            app_number = 0
            for row in rows:
                part = row.mapped_part
                if part and part.startswith("Application "):
                    app_number += 1
                    part = f"Application {app_number}"
                printed_row = by_description.get(row.particulars)
                pages = []
                if printed_row:
                    for page, text in page_text.items():
                        if printed_row.kind == "number" and page < numeric_start:
                            continue
                        folio = printed_folio(text)
                        if (
                            folio
                            and folio[0] == printed_row.kind
                            and printed_row.start <= folio[1] <= printed_row.end
                        ):
                            if (
                                folio[1] == printed_row.end
                                and folio[2] > printed_row.end_suffix
                            ):
                                continue
                            pages.append(page)
                checked_rows.append(
                    replace(
                        row,
                        mapped_part=part,
                        start_page=min(pages) if pages else 0,
                        end_page=max(pages) if pages else 0,
                    )
                )
        flags.extend(
            check_index_consistency(checked_rows, spans, page_count=page_count)
        )

    expected = collect_expected_annexures(page_parts, page_text)
    attached = attached_annexures(page_parts)
    missing_attached = sorted(expected - attached, key=_annexure_sort_key)
    extra_attached = sorted(attached - expected, key=_annexure_sort_key)
    for part in missing_attached:
        flags.append(
            SplitAuditFlag(
                code="annexure_mentioned_not_attached",
                severity="error",
                part=part,
                message=(
                    f"{part} is listed in Index / List of Dates / Main Petition "
                    "but was not found as an attached annexure in the split"
                ),
            )
        )
    for part in extra_attached:
        flags.append(
            SplitAuditFlag(
                code="annexure_attached_not_mentioned",
                severity="warning",
                part=part,
                message=(
                    f"{part} is attached in the PDF but is not mentioned in "
                    "Index, List of Dates, or Main Petition "
                    "(should move to Unidentified)"
                ),
            )
        )

    unidentified_reasons: list[dict[str, Any]] = []
    unlabeled = [page for page in range(1, page_count + 1) if page not in page_parts]
    if unlabeled:
        starts = [unlabeled[0]]
        ends: list[int] = []
        for previous, current in zip(unlabeled, unlabeled[1:]):
            if current != previous + 1:
                ends.append(previous)
                starts.append(current)
        ends.append(unlabeled[-1])
        for start, end in zip(starts, ends, strict=True):
            excerpt = "\n".join(page_text.get(page, "") for page in range(start, end + 1))
            if re.search(r"(?i)\bannexure\s*(?:no\.?\s*)\d+\b", excerpt) and re.search(
                r"(?i)\b(?:high\s+court|district\s+court|tribunal|writ\s+petition)\b",
                excerpt,
            ):
                unidentified_reasons.append(
                    {
                        "page_span": {"start": start, "end": end},
                        "reason": (
                            "This continuous document is Unidentified because its reproduced "
                            "lower-court pages use a local Annexure No. label that may conflict "
                            "with the Supreme Court paper-book Index. The full run was kept "
                            "together because the outer exhibit boundary could not be confirmed "
                            "without risking a split inside the document."
                        ),
                    }
                )

    return {
        "index_rows": [row.as_dict() for row in rows],
        "document_spans": list(spans),
        "unidentified_reasons": unidentified_reasons,
        "annexure_inventory": {
            "mentioned": sorted(expected, key=_annexure_sort_key),
            "attached": sorted(attached, key=_annexure_sort_key),
            "mentioned_count": len(expected),
            "attached_count": len(attached),
            "mentioned_not_attached": missing_attached,
            "attached_not_mentioned": extra_attached,
        },
        "flags": [flag.as_dict() for flag in flags],
        "flag_counts": {
            "error": sum(1 for flag in flags if flag.severity == "error"),
            "warning": sum(1 for flag in flags if flag.severity == "warning"),
            "total": len(flags),
        },
    }
