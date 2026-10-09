"""Post-split audit: paper-book sequence + Index ↔ file consistency.

SCI compiled petitions only for now. Flags are advisory (stored on STEP_SPLIT);
they do not block slicing.
"""

from __future__ import annotations

import re
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, replace
from itertools import pairwise
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
        "AOR's Declaration",  # commonly precedes the indexed paper book
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

# These document families have no single universal order in SCI paper books.
# Treat them as order-equivalent instead of producing false audit warnings.
_SEQUENCE_FLEXIBLE_GROUPS: tuple[frozenset[str], ...] = (
    frozenset({"Affidavit", "AOR's Certificate"}),
    frozenset({"Annexures", "Applications"}),
    frozenset(
        {
            "Memo of Parties",
            "Memo of Appearance",
            "Vakalatnama",
            "PoA/BR",
            "Court Fees",
            "Filing Memo",
        }
    ),
)

_COMBINED_SEQUENCE_ALIASES: dict[str, tuple[str, ...]] = {
    "Synopsis + List of Dates & Events": ("Synopsis", "List of Dates & Events"),
    "Memo of Appearance + Vakalatnama": ("Memo of Appearance", "Vakalatnama"),
}

_INDEX_ROW_RE = re.compile(
    r"(?m)^\s*(?:(?P<sno>\d{1,3})[.),]\s*)(?P<body>.+?)\s+"
    r"(?P<start>\d{1,4})(?:\s*[-–—/]\s*(?P<end>\d{1,4}))?\s*$"
)
# Softer row: serial + particulars, page number may be on the same line mid-OCR.
_INDEX_SOFT_ROW_RE = re.compile(
    r"(?m)^\s*(?P<sno>\d{1,3})[.),]\s+(?P<body>.+?)"
    r"(?:\s+(?P<start>\d{1,4})(?:\s*[-–—/]\s*(?P<end>\d{1,4}))?)?\s*$"
)

_ANNEXURE_IN_INDEX_RE = re.compile(
    r"annexure[\s\-–—]*([a-z])?[\s\-–—/\.]*(\d+)", re.IGNORECASE
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

    # Common SCI Index abbreviations.  These appear as standalone particulars
    # in the master Index and must be resolved before the generic word tests
    # below.  Keeping them document-type based also covers spelling/layout
    # variants without tying the split to a particular filing.
    compact = re.sub(r"[^a-z]", "", text)
    if compact == "fm":
        return "Filing Memo"
    if compact == "va":
        return "Vakalatnama"

    # An Annexure description can itself be an application or affidavit. Its
    # leading outer-paper-book label is authoritative (for example
    # ``ANNEXURE P-11: Anticipatory Bail Application ...``).
    leading_annex = re.match(
        r"^[^a-z0-9]{0,3}annexure[\s\-–—]*([a-z])?"
        r"[\s\-–—/\.]*([0-9]+)\b",
        particulars,
        re.IGNORECASE,
    )
    if leading_annex:
        return normalize_annexure_part_label(
            leading_annex.group(1), int(leading_annex.group(2))
        )

    # An IA description can mention the SLP, its affidavit or impugned order.
    # It can also request copies/translations of Annexure P-1/P-8. Its own
    # title determines the row type, so test Application before references.
    if re.match(
        r"^[^a-z0-9]{0,3}(?:[il]\.?\s*a\.?\s*(?:no\.?|\b)|"
        r"crl\.?\s*m\.?\s*p\.?\s*(?:no\.?|\b)|"
        r"(?:an?\s+)?application\b)",
        text,
    ) or re.search(r"\bapplication\s+(?:for|seeking)\b", text):
        return "Application 1"

    annex = _ANNEXURE_IN_INDEX_RE.search(particulars)
    if annex:
        return normalize_annexure_part_label(annex.group(1), int(annex.group(2)))
    series_label = _SERIES_LABEL_IN_INDEX_RE.search(particulars or "")
    if series_label:
        return normalize_annexure_part_label(
            series_label.group(1), int(series_label.group(2))
        )

    if "fresh case" in text and "report" in text:
        return None
    if "office report on limitation" in text or "o/r on limitation" in text or (
        "limitation" in text and "report" in text
    ):
        return "Office Report on Limitation"
    if "listing" in text and ("proforma" in text or "performa" in text):
        return "Listing Proforma"
    if (
        "proforma for first listing" in text
        or "performa for first listing" in text
    ):
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
    if (
        "special leave" in text
        or re.search(r"\bs\.?\s*l\.?\s*p\.?\b", text)
        or "form 28" in text
        or (
            "petition" in text
            and "transfer" not in text
            and "writ" not in text
            and "application" not in text
        )
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
    if (
        "filing memo" in text
        or "filling memo" in text
        or "index of filing" in text
        or "filing index" in text
    ):
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
    if any(re.match(r"^\d+[.),]\t", line) for line in index_text.splitlines()):
        rows = []
        for line in index_text.splitlines():
            cells = line.split("\t")
            if len(cells) != 3 or not re.fullmatch(r"\d+[.),]", cells[0]):
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
        if re.match(r"^\d{1,3}[.),](?:\s+|$)", stripped) or not merged:
            merged.append(stripped)
        else:
            merged[-1] = f"{merged[-1]} {stripped}"
    normalized = "\n".join(merged)

    rows: list[IndexRow] = []
    seen: set[tuple[str, int, int]] = set()
    for match in _INDEX_SOFT_ROW_RE.finditer(normalized):
        body = (match.group("body") or "").strip()
        original_body = re.sub(r"^\s*\d+[.),]\s*", "", match.group(0)).strip()
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
        if re.match(r"^\d+[.),]\t", line)
    ]
    if tabbed_lines:
        result: list[IndexPrintedRow] = []
        for line in tabbed_lines:
            cells = line.split("\t")
            if len(cells) != 3 or not re.fullmatch(r"\d+[.),]", cells[0]):
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
        # Geometry-rebuilt rows have a verified page-number column. Scanned
        # continuation sheets may nevertheless lack enough surviving ruling
        # lines for TSV reconstruction (typically the short final Index page
        # containing Filing Memo / Vakalatnama / Memo of Parties). Recover
        # only those strongly named filing rows from free OCR; dates and case
        # numbers in ordinary particulars remain excluded.
        safe_plain_parts = {
            "Filing Memo",
            "Vakalatnama",
            "Memo of Parties",
            "Memo of Appearance",
        }
        seen = {
            (row.mapped_part, row.kind, row.start, row.end, row.end_suffix)
            for row in result
        }
        plain_index_text = "\n".join(
            line
            for line in index_text.splitlines()
            if not re.match(r"^\d+[.),]\t", line)
        )
        for row in parse_index_rows(plain_index_text):
            if row.mapped_part not in safe_plain_parts or row.start_page <= 0:
                continue
            candidate = IndexPrintedRow(
                mapped_part=row.mapped_part,
                particulars=row.particulars,
                kind="number",
                start=row.start_page,
                end=row.end_page,
            )
            key = (
                candidate.mapped_part,
                candidate.kind,
                candidate.start,
                candidate.end,
                candidate.end_suffix,
            )
            if key not in seen:
                result.append(candidate)
                seen.add(key)
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
        serial_match = re.match(r"^(\d{1,3})[.),]\s+(.*)$", line)
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
    rows = index_rows_with_printed_pages(index_text)

    # Preserve every inventory row, including blank page-column cells and
    # mixed tabular/plain continuations. One numeric gap can contain several
    # documents: P-1=26, P-2/P-3 blank, P-4=31 does not prove P-2=27-30.
    # An unnumbered application row before an annexure is equally material.
    serial_starts = list(
        re.finditer(r"(?m)^\s*\d{1,3}[.),](?:\s+|$)", index_text)
    )
    inventory: list[IndexRow] = []
    for index, match in enumerate(serial_starts):
        stop = (
            serial_starts[index + 1].start()
            if index + 1 < len(serial_starts)
            else len(index_text)
        )
        inventory.extend(parse_index_rows(index_text[match.start() : stop]))

    def matches_explicit_row(item: IndexRow, row: IndexPrintedRow) -> bool:
        return bool(
            item.mapped_part == row.mapped_part
            and item.start_page == row.start
            and item.end_page == row.end
            and item.start_page > 0
            and row.kind == "number"
        )

    def single_unresolved_inventory_row(
        previous: IndexPrintedRow, following: IndexPrintedRow, label: str
    ) -> bool:
        prior_positions = [
            index for index, item in enumerate(inventory)
            if matches_explicit_row(item, previous)
        ]
        later_positions = [
            index for index, item in enumerate(inventory)
            if matches_explicit_row(item, following)
        ]
        if len(prior_positions) != 1 or len(later_positions) != 1:
            return False
        start, stop = prior_positions[0], later_positions[0]
        if stop <= start:
            return False
        unresolved = [
            item for item in inventory[start + 1 : stop]
            if not any(matches_explicit_row(item, row) for row in rows)
        ]
        return len(unresolved) == 1 and unresolved[0].mapped_part == label

    # A long particulars cell can cross an Index page boundary. In that case
    # OCR may leave ``ANNEXURE P-6`` at the bottom of one sheet and put only
    # its particulars plus ``120-149`` on the next sheet, so neither fragment
    # is a complete parseable row. Recover only a single, sequential gap whose
    # label is explicitly present in the Index and whose printed-page gap is
    # bounded by the neighbouring annexures. This deliberately does not infer
    # arbitrary missing inventory entries.
    def annexure_key(row: IndexPrintedRow) -> tuple[str, int] | None:
        match = re.fullmatch(
            r"Annexure\s+([A-Z])-(\d+)", row.mapped_part or "", re.IGNORECASE
        )
        if not match or row.kind != "number" or row.end_suffix:
            return None
        return match.group(1).upper(), int(match.group(2))

    unique: dict[tuple[str | None, str, int, int, str], IndexPrintedRow] = {
        (row.mapped_part, row.kind, row.start, row.end, row.end_suffix): row
        for row in rows
    }
    annexure_rows = sorted(
        (row for row in unique.values() if annexure_key(row)),
        key=lambda row: (annexure_key(row) or ("", 0), row.start, row.end),
    )
    for previous, following in pairwise(annexure_rows):
        previous_key = annexure_key(previous)
        following_key = annexure_key(following)
        if not previous_key or not following_key:
            continue
        series, number = previous_key
        if following_key != (series, number + 2):
            continue
        start = previous.end + 1
        end = following.start - 1
        if end < start:
            continue
        missing_label = f"Annexure {series}-{number + 1}"
        explicit = re.search(
            rf"\bANNEXURE\s*{re.escape(series)}\s*[-/]\s*{number + 1}\b",
            index_text,
            re.IGNORECASE,
        )
        if not explicit:
            continue
        if not single_unresolved_inventory_row(previous, following, missing_label):
            continue
        recovered = IndexPrintedRow(
            mapped_part=missing_label,
            particulars=missing_label,
            kind="number",
            start=start,
            end=end,
        )
        unique[(missing_label, "number", start, end, "")] = recovered

    # A scanned Index can lose one page-column cell while retaining the row
    # title and both neighbouring ranges. Recover the petition range only
    # when the Index explicitly names it and Impugned Order / Appendix leave
    # one exact contiguous gap (for example 1-4, SLP 5-21, Appendix 22-25).
    has_main = any(row.mapped_part == "Main Petition" for row in unique.values())
    impugned = [
        row
        for row in unique.values()
        if row.mapped_part == "Impugned Order" and row.kind == "number"
    ]
    appendices = [
        row
        for row in unique.values()
        if row.mapped_part == "Appendix" and row.kind == "number"
    ]
    if (
        not has_main
        and len(impugned) == 1
        and len(appendices) == 1
        and impugned[0].end + 1 <= appendices[0].start - 1
        and re.search(
            r"\bspecial\s+leave\s+petition\s+with\s+affidavit\b",
            index_text,
            re.IGNORECASE,
        )
    ):
        start = impugned[0].end + 1
        end = appendices[0].start - 1
        recovered = IndexPrintedRow(
            mapped_part="Main Petition",
            particulars="Special Leave Petition with Affidavit",
            kind="number",
            start=start,
            end=end,
        )
        unique[("Main Petition", "number", start, end, "")] = recovered

    # The final Annexure row may be split between two scanned Index sheets:
    # its label remains at the bottom of one page while its particulars/range
    # start the next. Recover it only when (a) the label is explicit, (b) the
    # previous sequential Annexure exists, and (c) parsed rows leave exactly
    # one bounded numeric gap before the next Annexure or filing-back-matter
    # row. This fixes P-9 637-646 without inventing omitted exhibits.
    annexure_keys = {
        key
        for row in unique.values()
        if (key := annexure_key(row)) is not None
    }
    explicit_keys = {
        ((match.group(1) or "P").upper(), int(match.group(2)))
        for match in _ANNEXURE_IN_INDEX_RE.finditer(index_text)
    }
    back_matter = {
        "Filing Memo",
        "Vakalatnama",
        "Memo of Parties",
        "Memo of Appearance",
    }
    for series, number in sorted(explicit_keys - annexure_keys):
        previous = next(
            (
                row
                for row in unique.values()
                if annexure_key(row) == (series, number - 1)
            ),
            None,
        )
        if previous is None:
            continue
        following_annexure = [
            row
            for row in unique.values()
            if (key := annexure_key(row)) is not None
            and key[0] == series
            and key[1] > number
        ]
        following_back = [
            row
            for row in unique.values()
            if row.kind == "number"
            and row.mapped_part in back_matter
            and row.start > previous.end
        ]
        following = min(
            following_annexure + following_back,
            key=lambda row: row.start,
            default=None,
        )
        upper = following.start if following else 0
        if upper <= previous.end + 1:
            continue
        label = f"Annexure {series}-{number}"
        if following is None or not single_unresolved_inventory_row(
            previous, following, label
        ):
            continue
        covered = sorted(
            (
                row.start,
                row.end,
            )
            for row in unique.values()
            if row.kind == "number"
            and row.start > previous.end
            and row.start < upper
        )
        gaps: list[tuple[int, int]] = []
        cursor = previous.end + 1
        for start, end in covered:
            if start > cursor:
                gaps.append((cursor, start - 1))
            cursor = max(cursor, end + 1)
        if cursor < upper:
            gaps.append((cursor, upper - 1))
        if len(gaps) != 1:
            continue
        start, end = gaps[0]
        unique[(label, "number", start, end, "")] = IndexPrintedRow(
            mapped_part=label,
            particulars=label,
            kind="number",
            start=start,
            end=end,
        )

    return list(unique.values())


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
    labeled_pages = {
        page
        for page, names in page_parts.items()
        if "Index" in parts_on_page(names)
    }
    # Every geometry-rebuilt master-Index continuation repeats the synthetic
    # ``INDEX / S.No. / Particulars / Page No.`` header.  Always union those
    # pages with the mutable split labels.  Repair/collapse passes may retain
    # only the first Index page; relying on that mutated map then drops the
    # remaining inventory and makes later reconciliation non-idempotent.
    detected_pages = {
        page
        for page, text in page_text.items()
        if "index" in _fold(text[:400])
        and ("particulars" in _fold(text[:800]) or "page no" in _fold(text[:800]))
        and (
            "\t" in text
            or len(re.findall(r"(?m)^\s*\d{1,3}[.),]\s+", text)) >= 2
        )
    }
    pages = sorted(labeled_pages | detected_pages)
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
            and re.search(
                r"\bvol(?:ume)?\.?\s*[-–—]?\s*(?:i{1,3}|[1-3])\b",
                prior_text,
            )
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


def _sequence_family_name(part: str) -> str:
    """Return the stable family used for flexible-order comparisons."""
    folded = _fold(part)
    if re.fullmatch(r"annexure [a-z]-?\d+", folded):
        return "Annexures"
    if re.fullmatch(r"application \d+", folded):
        return "Applications"
    return family_split_name(part)


def _sequence_order_is_flexible(left: str, right: str) -> bool:
    families = {_sequence_family_name(left), _sequence_family_name(right)}
    return any(families <= group for group in _SEQUENCE_FLEXIBLE_GROUPS)


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
        if earlier_rank <= rank or _sequence_order_is_flexible(earlier_name, name):
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
        if part == "Main Petition" and "affidavit" in _fold(row.particulars):
            mentioned.add("Affidavit")

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


_SOURCE_CASE_HEADER_RE = re.compile(
    r"(?mi)^[ \t]*(?P<kind>W\s*\.?\s*(?:P\s*\.?\s*"
    r"(?:\(\s*[A-Z]+\s*\))?|A\s*\.?))\s*(?:No\.?\s*)?"
    r"(?P<number>\d{1,7})\s*(?:/\s*|of\s+)(?P<year>(?:19|20)\d{2})\b"
)
_JUDGMENT_TITLE_RE = re.compile(
    r"(?mi)^[ \t]*(?:COMMON\s+)?J\s*U\s*D\s*G\s*E?\s*M\s*E\s*N\s*T[ \t]*$"
)
_CERTIFIED_END_RE = re.compile(
    r"(?mi)^[ \t/\\|]*(?:true|certified)[ \t]+copy[ \t/\\|]*"
    r"(?:\r?\n[ \t]*\d+)?\s*\Z"
)
_COURT_FILING_TITLE_RE = re.compile(
    r"(?mi)^[ \t]*(?:AFFIDAVIT|MEMO(?:RANDUM)?\s+OF\s+(?:APPEAL|APPEARANCE)|"
    r"PETITION\s+FILED\s+UNDER\b[^\n]*)[ \t]*$"
)
_STANDALONE_OUTER_ANNEXURE_RE = re.compile(
    r"(?mi)^[ \t]*ANNEXURE[ \t:-]*[PR][ \t/-]*\d+[ \t]*$"
)


def _source_case_header(text: str) -> tuple[str, str, str] | None:
    """Read a source-case running header, never a case cited in body prose."""
    match = _SOURCE_CASE_HEADER_RE.search(text[:400])
    if not match:
        return None
    return (
        re.sub(r"[^A-Z]", "", match.group("kind").upper()),
        match.group("number"),
        match.group("year"),
    )


def check_unresolved_source_transitions(
    page_parts: PagePartMap,
    page_text: Mapping[int, str],
    *,
    page_count: int,
) -> list[SplitAuditFlag]:
    """Warn about a completed judgment merged with a different court packet.

    A new affidavit, application, exhibit list or court caption alone is not
    an outer boundary. Require the judgment's own heading and repeated source
    header, its terminal certification, and a different source header repeated
    over a newly captioned filing. These signals justify review, not a guessed
    annexure identity or an automatic split of a historical court bundle.
    """
    judgment_cases: dict[str, tuple[str, str, str]] = {}
    flags: list[SplitAuditFlag] = []
    previous_parts: set[str] = set()
    for page in range(1, page_count + 1):
        current_parts = {
            name
            for name in parts_on_page(page_parts.get(page))
            if family_split_name(name) in {"Annexures", "Impugned Order"}
        }
        judgment_cases = {
            part: case
            for part, case in judgment_cases.items()
            if part in current_parts and part in previous_parts
        }
        text = page_text.get(page, "")
        source_case = _source_case_header(text)
        if source_case and _JUDGMENT_TITLE_RE.search(text[:1200]):
            judgment_cases.update({part: source_case for part in current_parts})

        prior_text = page_text.get(page - 1, "")
        prior_case = _source_case_header(prior_text)
        older_case = _source_case_header(page_text.get(page - 2, ""))
        next_case = _source_case_header(page_text.get(page + 1, ""))
        next_parts = set(parts_on_page(page_parts.get(page + 1)))
        if (
            prior_case
            and source_case
            and prior_case != source_case
            and older_case == prior_case
            and next_case == source_case
            and _CERTIFIED_END_RE.search(prior_text[-300:])
            and _COURT_FILING_TITLE_RE.search(text[:1600])
            and re.search(r"(?i)\bBEFORE\b.{0,60}\bHIGH\s+COURT\b", _fold(text[:800]))
            and not _STANDALONE_OUTER_ANNEXURE_RE.search(text[:800])
        ):
            for part in sorted(current_parts & previous_parts & next_parts):
                if judgment_cases.get(part) != prior_case:
                    continue
                flags.append(
                    SplitAuditFlag(
                        code="unresolved_source_transition",
                        part=part,
                        message=(
                            f"{part} contains a completed judgment at p.{page - 1} "
                            f"followed by a different court filing packet at p.{page}. "
                            "Check the original paper-book Index and source PDF to "
                            "confirm the outer boundary and annexure identity; "
                            "these source signals do not authorize an automatic split."
                        ),
                        details={
                            "boundary_after_page": page - 1,
                            "boundary_before_page": page,
                            "previous_source_case": " ".join(
                                (prior_case[0], "/".join(prior_case[1:]))
                            ),
                            "next_source_case": " ".join(
                                (source_case[0], "/".join(source_case[1:]))
                            ),
                            "evidence": [
                                "judgment_heading_in_same_contiguous_part",
                                "repeated_judgment_source_header",
                                "terminal_copy_certification",
                                "new_court_caption_and_filing_title",
                                "different_repeated_source_case_header",
                            ],
                            "resolution": "original_index_and_source_required",
                            "auto_split": False,
                        },
                    )
                )
                judgment_cases.pop(part, None)
        previous_parts = current_parts
    return flags


def audit_compiled_split(
    page_parts: PagePartMap,
    page_text: Mapping[int, str],
    *,
    page_count: int,
) -> dict[str, Any]:
    """Run sequence, Index and unresolved source checks on a repaired split map."""
    spans = document_spans_from_page_parts(page_parts)
    index_text = _index_pages_text(page_parts, page_text)
    rows = parse_index_rows(index_text) if index_text.strip() else []

    flags: list[SplitAuditFlag] = []
    flags.extend(check_document_sequence(spans))
    flags.extend(
        check_unresolved_source_transitions(
            page_parts, page_text, page_count=page_count
        )
    )
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
                printed_row = by_description.get(row.particulars)
                # The geometry-backed row contains the complete table-cell
                # text. Prefer its document classification over a flattened
                # OCR row, which can lose leading words such as "SLP WITH".
                part = (
                    printed_row.mapped_part
                    if printed_row and printed_row.mapped_part
                    else row.mapped_part
                )
                if part and part.startswith("Application "):
                    app_number += 1
                    part = f"Application {app_number}"
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
