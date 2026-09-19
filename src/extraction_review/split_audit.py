"""Post-split audit: paper-book sequence + Index ↔ file consistency.

SCI compiled petitions only for now. Flags are advisory (stored on STEP_SPLIT);
they do not block slicing.
"""

from __future__ import annotations

import re
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Any

from .document_parts import (
    PagePartMap,
    document_spans_from_page_parts,
    family_split_name,
    parts_on_page,
)

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
    }
)

# Expected outer order for SCI paper books (D-225 / catalog-aligned).
_SEQUENCE_ORDER: tuple[str, ...] = (
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
    "Annexures",  # placeholder rank for any Annexure P-n
    "Appendix",
    "Applications",  # placeholder rank for any Application n
    "Memo of Parties",
    "Memo of Appearance",
    "Vakalatnama",
    "PoA/BR",
    "Court Fees",
    "Filing Memo",
)

_SEQUENCE_RANK = {name: index for index, name in enumerate(_SEQUENCE_ORDER)}

_INDEX_ROW_RE = re.compile(
    r"(?m)^\s*(?:(?P<sno>\d{1,3})[.)]\s*)?(?P<body>.+?)\s+"
    r"(?P<start>\d{1,4})(?:\s*[-–—/]\s*(?P<end>\d{1,4}))?\s*$"
)

_ANNEXURE_IN_INDEX_RE = re.compile(
    r"annexure[\s\-]*p[\s\-/\.]*(\d{1,3})", re.IGNORECASE
)
_APPLICATION_IN_INDEX_RE = re.compile(
    r"(?:application|i\.?\s*a\.?)[\s\-]*(?:no\.?\s*)?(\d{1,3})|"
    r"\bi\.?\s*a\.?\b",
    re.IGNORECASE,
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
        return f"Annexure P-{int(annex.group(1))}"

    if "office report" in text or "o/r on limitation" in text or (
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
    if "special leave" in text or "form 28" in text or (
        "petition" in text
        and "transfer" not in text
        and "writ" not in text
        and "application" not in text
    ):
        if "affidavit" in text:
            return "Affidavit"
        return "Main Petition"
    if "affidavit" in text:
        return "Affidavit"
    if "appendix" in text:
        return "Appendix"
    if "vakalatnama" in text:
        return "Vakalatnama"
    if "memo of appearance" in text or "memorandum of appearance" in text:
        return "Memo of Appearance"
    if "memo of parties" in text or "memorandum of parties" in text:
        return "Memo of Parties"
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
    if "court fee" in text:
        return "Court Fees"
    if "application" in text or re.search(r"\bi\.?\s*a\.?\b", text):
        match = _APPLICATION_IN_INDEX_RE.search(particulars)
        if match and match.lastindex and match.group(1):
            return f"Application {int(match.group(1))}"
        return "Application 1"
    return None


def parse_index_rows(index_text: str) -> list[IndexRow]:
    """Parse PARTICULARS + page span rows from Index page text."""
    rows: list[IndexRow] = []
    seen: set[tuple[str, int, int]] = set()
    for match in _INDEX_ROW_RE.finditer(index_text or ""):
        body = (match.group("body") or "").strip()
        if not body or len(body) < 3:
            continue
        # Drop header-ish lines without a real particular.
        folded = _fold(body)
        if "page no" in folded or folded in {"particulars", "particulars of documents"}:
            continue
        try:
            start = int(match.group("start"))
        except (TypeError, ValueError):
            continue
        end_raw = match.group("end")
        end = int(end_raw) if end_raw else start
        if end < start:
            start, end = end, start
        serial = None
        if match.group("sno"):
            try:
                serial = int(match.group("sno"))
            except ValueError:
                serial = None
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


def _index_pages_text(page_parts: PagePartMap, page_text: Mapping[int, str]) -> str:
    pages = sorted(
        page
        for page, names in page_parts.items()
        if "Index" in parts_on_page(names)
    )
    if not pages:
        # Fallback: any page that looks like an index heading.
        pages = [
            page
            for page, text in page_text.items()
            if "index" in _fold(text[:400])
            and ("particulars" in _fold(text[:800]) or "page no" in _fold(text[:800]))
        ]
    return "\n".join(page_text.get(page, "") for page in sorted(set(pages)))


def _sequence_family_rank(part: str) -> tuple[int, int]:
    """Return (block_rank, sub_rank) for ordering comparisons."""
    folded = _fold(part)
    annex = re.fullmatch(r"annexure p-?(\d{1,3})", folded)
    if annex:
        return (_SEQUENCE_RANK["Annexures"], int(annex.group(1)))
    app = re.fullmatch(r"application (\d{1,3})", folded)
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
    """Flag when found documents appear out of expected SCI paper-book order."""
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
    for index, (name, start, rank) in enumerate(items):
        for earlier_name, earlier_start, earlier_rank in items[:index]:
            if earlier_start >= start:
                continue
            if earlier_rank <= rank:
                continue
            # Document that should come later in the paper book starts earlier.
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
    # Deduplicate noisy pairwise flags: keep one per out-of-order part pair.
    deduped: list[SplitAuditFlag] = []
    seen_pairs: set[tuple[str, str]] = set()
    for flag in flags:
        details = flag.details or {}
        key = (
            str(details.get("out_of_order_part") or ""),
            str(details.get("should_precede_part") or ""),
        )
        if key in seen_pairs:
            continue
        seen_pairs.add(key)
        deduped.append(flag)
    return deduped


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
        # Combined Synopsis+LOD row maps to Synopsis; also credit LOD.
        if part == "Synopsis" and "list of date" in _fold(row.particulars):
            mentioned.add("List of Dates & Events")

        if row.start_page > page_count or row.end_page > page_count:
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
            # Accept combined documents row.
            found = by_part.get("Synopsis + List of Dates & Events")
        if found is None and part == "List of Dates & Events":
            found = by_part.get("Synopsis + List of Dates & Events")
        if found is None and part in {"Vakalatnama", "Memo of Appearance"}:
            found = by_part.get("Memo of Appearance + Vakalatnama")

        if found is None:
            flags.append(
                SplitAuditFlag(
                    code="in_index_missing_in_file",
                    severity="error",
                    part=part,
                    index_particulars=row.particulars,
                    expected_pages={
                        "start_page": row.start_page,
                        "end_page": row.end_page,
                    },
                    message=(
                        f"Index lists {part!r} ({row.particulars}) at "
                        f"pp. {row.start_page}–{row.end_page}, but that document "
                        f"was not found in the split file"
                    ),
                )
            )
            continue

        found_start, found_end = found
        # Soft page mismatch: Index start not inside the found span (and not adjacent).
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
        # Numbered annexure/application: check exact name in mentioned.
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
        flags.extend(
            check_index_consistency(rows, spans, page_count=page_count)
        )

    return {
        "index_rows": [row.as_dict() for row in rows],
        "document_spans": list(spans),
        "flags": [flag.as_dict() for flag in flags],
        "flag_counts": {
            "error": sum(1 for flag in flags if flag.severity == "error"),
            "warning": sum(1 for flag in flags if flag.severity == "warning"),
            "total": len(flags),
        },
    }
