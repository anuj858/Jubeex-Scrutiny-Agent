"""Document-part labels, record slicing, and local chunk selection.

Pinecone stores `document_part` using LlamaSplit names from
`configs/config.json` `split.categories`. This module does not invent extra
parts: a new document type is added only in that Split list. Catalogue
`where_to_look` is matched against those names (and nicknames written in the
category description, e.g. `(V/A)`).
"""

from __future__ import annotations

import json
import math
import re
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path
from typing import Any

from .scrutiny.rules import Defect, normalize_filing_type

_CONFIG_PATH = Path(__file__).resolve().parents[2] / "configs" / "config.json"

# Captions and headings as they appear on SCI forms — used as search text
# (not the long legal objection sentences).
FILING_CAPTION_QUERIES: tuple[str, ...] = (
    "IN THE SUPREME COURT OF INDIA CIVIL APPELLATE JURISDICTION",
    "SPECIAL LEAVE PETITION UNDER ARTICLE 136 Form 28",
    "QUESTIONS OF LAW GROUNDS MAIN PRAYER INTERIM RELIEF",
    "Listing Proforma Proforma for First Listing",
    "Advocate's Check List Advocate-on-Record certificate",
    "CERTIFICATE confined only to the pleadings",
    "cause title CERTIFICATE CERTIFIED confined only to the pleadings",
    "DECLARATION IN TERMS OF RULE 3(2) Affidavit",
    "Cover Page Index Office Report on Limitation",
    "Vakalatnama AOR Certificate Memo of Parties",
)

PINECONE_QUERY_MAX_CHARS = 110

# One PDF page can carry more than one Split label (e.g. Affidavit + Vakalatnama).
PagePartMap = dict[int, list[str]]

# Split label for the Form 28 body only. Slot id stays `petition`.
# "Petition" / "Matter" in product language means the full filing PDF.
MAIN_PETITION_PART = "Main Petition"
_LEGACY_PART_NAMES = {
    "petition": MAIN_PETITION_PART,
    "aor's declaration": "AOR's Certificate",
    "aors declaration": "AOR's Certificate",
    "aor declaration": "AOR's Certificate",
}

# Extra catalogue phrases → Split labels (beyond the config name/description).
_PART_ALIASES: dict[str, tuple[str, ...]] = {
    # Do not alias bare "declaration" / "check the declaration": those words
    # on the Advocate's Check List are not this page.
    "AOR's Certificate": (
        "advocate's certificate",
        "advocate-on-record certificate",
        "advocate on record certificate",
        "aor's certificate",
        "certificate after the main prayer",
        "certificate after main prayer",
        "certified that the special leave petition is confined",
        "certified that the special leave petition is confined only",
        "confined only to the pleadings",
        "court/tribunal whose order is challenged",
        "this certificate is given on the basis of the instructions",
        "c e r t i f i c a t e",
        "aor's declaration",
        "aors declaration",
    ),
    "Affidavit": (
        "a f f i d a v i t",
        "deponent",
        "verification",
    ),
    "Impugned Order": (
        "impugned judgment",
        "impugned order",
        "judgment under challenge",
    ),
    "Main Petition": (
        "form 28",
        "special leave petition body",
    ),
}

# Paper-book Index lists document names; quoting those lines is not proof
# that the named document was filed or signed.
_NON_CONTENT_PARTS = frozenset({"Index"})

ALWAYS_RECORD_FIELDS: tuple[str, ...] = ("court", "petition_type")

# Ceiling on page excerpts sent to the LLM (summary is extra). Narrow checks
# do not need the global SCRUTINY_MAX_CHUNKS dump. Multi-part checks then
# raise this to PAGES_PER_TARGET_PART each.

# At least this many page excerpts per document part the check must open.
PAGES_PER_TARGET_PART = 3

# Placement phrases in where_to_look ("before the Cover Page") name other
# documents as landmarks, not as extra parts this check must retrieve.
_LANDMARK_PREP = re.compile(
    r"\b(?:before|after|following|preceding|followed by)\s+(?:the\s+)?",
    re.IGNORECASE,
)
_LANDMARK_STOP = re.compile(
    r"\s+for\s+"
    r"|,\s*(?:and\s+)?(?:check|look|verify|confirm|go\s+to|review)\b"
    r"|[.;]",
    re.IGNORECASE,
)
_PLACEMENT_PAREN = re.compile(
    r"\((?:placed|found|located|situated)\b[^)]{0,240}\)", re.I
)

# Keep a page chunk only if ln(best_score / this_score) is below this.
# 0.36 ≈ keep scores at least ~70% of the best hit; farther neighbours are dropped.
MAX_SCORE_LOG_GAP = 0.36

FILING_TYPE_LABELS: dict[str, str] = {
    "slp_civil": "Special Leave Petition (Civil)",
    "slp_criminal": "Special Leave Petition (Criminal)",
    "civil_appeal": "Civil Appeal",
    "criminal_appeal": "Criminal Appeal",
    "writ_petition_civil": "Writ Petition (Civil)",
    "writ_petition_criminal": "Writ Petition (Criminal)",
    "transfer_petition_civil": "Transfer Petition (Civil)",
    "transfer_petition_criminal": "Transfer Petition (Criminal)",
    "review_petition_civil": "Review Petition (Civil)",
    "review_petition_criminal": "Review Petition (Criminal)",
    "original_suit_civil": "Original Suit (Civil)",
    "contempt_petition_civil": "Contempt Petition (Civil)",
    "contempt_petition_criminal": "Contempt Petition (Criminal)",
    "election_petition_civil": "Election Petition (Civil)",
    "arbitration_petition": "Arbitration Petition",
    "curative_petition_civil": "Curative Petition (Civil)",
    "curative_petition_criminal": "Curative Petition (Criminal)",
    "miscellaneous_application": "Miscellaneous Application",
}


@lru_cache(maxsize=1)
def _split_categories() -> tuple[tuple[str, str], ...]:
    """LlamaSplit (name, description) — same labels Pinecone stores."""
    with _CONFIG_PATH.open(encoding="utf-8") as fh:
        payload = json.load(fh)
    categories = (payload.get("split") or {}).get("categories") or []
    return tuple(
        (str(item.get("name") or "").strip(), str(item.get("description") or ""))
        for item in categories
        if isinstance(item, dict) and str(item.get("name") or "").strip()
    )


def split_part_names() -> tuple[str, ...]:
    return tuple(name for name, _ in _split_categories())


def _fold(text: str) -> str:
    return (
        re.sub(r"\s+", " ", (text or "").replace("\u2019", "'").replace("\u2018", "'"))
        .strip()
        .lower()
    )


def _needles_for_part(name: str, description: str = "") -> tuple[str, ...]:
    """Search phrases for one Split part: official name plus description nicknames."""
    needles: list[str] = []

    def add(raw: str) -> None:
        folded = _fold(raw)
        if len(folded) >= 3 and folded not in needles:
            needles.append(folded)

    add(name)
    if re.search(r"checklist", name, re.IGNORECASE):
        add(re.sub(r"checklist", "Check List", name, flags=re.IGNORECASE))
    left = name.split("+", 1)[0].strip()
    if left != name:
        add(left)
    for nick in re.findall(r"\(([^)]{2,48})\)", description):
        add(nick)
    for alias in _PART_ALIASES.get(name, ()):
        add(alias)
    return tuple(needles)


def normalize_part_name(name: str | None) -> str:
    if not name:
        return ""
    text = re.sub(
        r"\s+", " ", name.replace("\u2019", "'").replace("\u2018", "'")
    ).strip()
    folded = _fold(text)
    if folded in _LEGACY_PART_NAMES:
        return _LEGACY_PART_NAMES[folded]
    for canonical, _description in _split_categories():
        if folded == _fold(canonical):
            return canonical
    for canonical, description in _split_categories():
        for needle in _needles_for_part(canonical, description):
            if folded == needle:
                return canonical
    return text


# Old LlamaSplit jobs used one label for both documents. Expand it so
# bundled pages fill the Vakalatnama and PoA/BR slots, not a combined slot.
_LEGACY_COMBINED_VAKALATNAMA = "vakalatnama + poa/br"


def _expanded_parts(part: str) -> tuple[str, ...]:
    if _fold(part) == _LEGACY_COMBINED_VAKALATNAMA:
        return ("Vakalatnama", "PoA/BR")
    return (part,)


def parts_on_page(value: Any) -> list[str]:
    """Normalise a page's Split labels to a unique list."""
    if not value:
        return []
    if isinstance(value, str):
        names = [value]
    elif isinstance(value, (list, tuple, set)):
        names = [str(item) for item in value]
    else:
        names = [str(value)]
    found: list[str] = []
    for name in names:
        part = normalize_part_name(name)
        for item in _expanded_parts(part):
            if item and item not in found:
                found.append(item)
    return found


def format_document_parts(value: Any) -> str:
    names = parts_on_page(value)
    return " / ".join(names)


def _contiguous_groups(pages: list[int]) -> list[tuple[int, int]]:
    ordered = sorted(set(pages))
    if not ordered:
        return []
    groups: list[tuple[int, int]] = []
    start = prev = ordered[0]
    for number in ordered[1:]:
        if number == prev + 1:
            prev = number
            continue
        groups.append((start, prev))
        start = prev = number
    groups.append((start, prev))
    return groups


def format_page_span(pages: list[int]) -> str:
    """Human-readable page span, e.g. 'p. 3' or 'pp. 3–5, 9'."""
    groups = _contiguous_groups(pages)
    if not groups:
        return ""
    bits = [str(a) if a == b else f"{a}–{b}" for a, b in groups]
    prefix = "p. " if sum(b - a + 1 for a, b in groups) == 1 else "pp. "
    return prefix + ", ".join(bits)


def _format_page_span(pages: list[int]) -> str:
    return format_page_span(pages)


# Outer SCI paper-book parts keep the *first* contiguous run. A later High Court
# writ mislabeled as Main Petition must not replace the real Form 28 body.
_FIRST_RUN_OUTER_PARTS = frozenset(
    {
        MAIN_PETITION_PART,
        "Cover Page",
        "Index",
        "Advocate's Checklist",
        "Office Report on Limitation",
        "Listing Proforma",
        "Synopsis",
        "List of Dates & Events",
        "Impugned Order",
        "AOR's Certificate",
        "Affidavit",
        "Appendix",
        "Vakalatnama",
        "Memo of Appearance",
        "Memo of Parties",
        "Filing Memo",
        "Court Fees",
        "PoA/BR",
        # Record of Proceedings uses longest-run: a stray early page is common noise.
    }
)


def collapse_repeated_split_pages(page_parts: PagePartMap) -> PagePartMap:
    """Keep one contiguous run of each Split part.

    Index at 5–7 is kept; Index at 55–57 is dropped. Main Petition at 15–23 is
    kept; a later Main Petition island (often a High Court writ) is dropped —
    first run wins for outer paper-book parts. Record of Proceedings keeps the
    longest run (stray early pages are common noise). Annexure P-2 at 29–30 is
    kept; later P-2 islands at 100–102 and 104 are dropped. Generic unnumbered
    Annexures stay until they are labelled P-n.
    """
    pages_by_part: dict[str, list[int]] = {}
    for page, names in page_parts.items():
        for part in parts_on_page(names):
            pages_by_part.setdefault(part, []).append(page)

    keep: set[tuple[int, str]] = set()
    for part, pages in pages_by_part.items():
        unique = sorted(set(pages))
        if family_split_name(part) == ANNEXURE_FAMILY and not _is_numbered_annexure(
            part
        ):
            keep.update((page, part) for page in unique)
            continue
        groups = _contiguous_groups(unique)
        if not groups:
            continue
        folded = _fold(part)
        numbered_app = bool(re.fullmatch(r"application \d{1,3}", folded))
        if (
            _is_numbered_annexure(part)
            or numbered_app
            or part in _FIRST_RUN_OUTER_PARTS
        ):
            start, end = groups[0]
        else:
            start, end = max(
                groups, key=lambda group: (group[1] - group[0] + 1, -group[0])
            )
        keep.update((page, part) for page in unique if start <= page <= end)

    collapsed: PagePartMap = {}
    for page, names in page_parts.items():
        kept = [part for part in parts_on_page(names) if (page, part) in keep]
        if kept:
            collapsed[page] = kept
    return collapsed


# Upload slots that are one paper-book file with two LlamaSplit names.
# documents[] uses the slot label, not two rows with the same page span.
COMBINED_DOCUMENT_GROUPS: tuple[tuple[frozenset[str], str], ...] = (
    (
        frozenset({"Synopsis", "List of Dates & Events"}),
        "Synopsis + List of Dates & Events",
    ),
    (
        frozenset({"Memo of Appearance", "Vakalatnama"}),
        "Memo of Appearance + Vakalatnama",
    ),
)


def _pages_by_part(
    page_parts: PagePartMap | dict[int, str],
) -> tuple[list[str], dict[str, list[int]]]:
    order: list[str] = []
    pages_by_part: dict[str, list[int]] = {}
    for page in sorted(page_parts):
        for name in parts_on_page(page_parts.get(page)):
            if name not in pages_by_part:
                pages_by_part[name] = []
                order.append(name)
            pages_by_part[name].append(page)
    return order, pages_by_part


def _collapse_combined_document_parts(
    order: list[str],
    pages_by_part: dict[str, list[int]],
) -> tuple[list[str], dict[str, list[int]]]:
    """Merge slot pairs (Synopsis+LOD, Memo+Vakalatnama) into one documents[] row."""
    merged_pages = {name: list(pages) for name, pages in pages_by_part.items()}
    merged_order = list(order)
    for members, label in COMBINED_DOCUMENT_GROUPS:
        present = [name for name in merged_order if name in members]
        if not present:
            continue
        pages: list[int] = []
        for name in present:
            pages.extend(merged_pages.pop(name, []))
        collapsed: list[str] = []
        inserted = False
        for name in merged_order:
            if name in members:
                if not inserted:
                    collapsed.append(label)
                    inserted = True
            else:
                collapsed.append(name)
        merged_order = collapsed
        merged_pages[label] = sorted(set(pages))
    return merged_order, merged_pages


def documents_from_page_parts(
    page_parts: PagePartMap | dict[int, str],
) -> dict[str, Any]:
    """Build a count/items list of Split parts with page spans."""
    order, pages_by_part = _collapse_combined_document_parts(
        *_pages_by_part(page_parts)
    )
    items = [f"{name} ({_format_page_span(pages_by_part[name])})" for name in order]
    return {"count": len(items), "items": items}


def document_spans_from_page_parts(
    page_parts: PagePartMap | dict[int, str],
) -> list[dict[str, Any]]:
    """Build documents[] spans from Split labels and global page numbers."""
    order, pages_by_part = _collapse_combined_document_parts(
        *_pages_by_part(page_parts)
    )
    return [
        {
            "name": name,
            "start_page": min(pages_by_part[name]),
            "end_page": max(pages_by_part[name]),
        }
        for name in order
        if pages_by_part[name]
    ]


def overlay_split_documents(
    payload: dict[str, Any], page_parts: PagePartMap | dict[int, str]
) -> None:
    """Stamp stitch document spans onto the stored record. Drop filing_summary."""
    spans = document_spans_from_page_parts(page_parts)
    if not spans:
        return
    record = payload
    while (
        isinstance(record, dict)
        and "petition_type" not in record
        and "documents" not in record
        and isinstance(record.get("data"), dict)
    ):
        record = record["data"]
    if not isinstance(record, dict):
        return
    record.pop("filing_summary", None)
    record["documents"] = spans
    record["document_counts"] = {
        "processed": len(spans),
        "failed": 0,
    }


def filing_type_label(filing_type: str | None) -> str:
    key = normalize_filing_type(filing_type)
    if key in FILING_TYPE_LABELS:
        return FILING_TYPE_LABELS[key]
    raw = (filing_type or "").strip()
    return raw or "this filing"


MAX_NUMBERED_PART = 999
ANNEXURE_FAMILY = "Annexures"
APPLICATION_FAMILY = "Application"


@dataclass(frozen=True)
class AnnexureMark:
    """Printed exhibit stamp: Annexure P-1 / E-2 / R-3.

    On SCI paper-books, bank/HC exhibits often still print ``ANNEXURE-E-n``.
    The Index and filing profile use petitioner series ``P-n`` for those same
    pages, so ``E`` is normalized to ``P`` for labels/slots.
    """

    number: int
    series: str = "P"

    def __post_init__(self) -> None:
        letter = (self.series or "P").strip().upper()[:1] or "P"
        if not letter.isalpha():
            letter = "P"
        # Underlying exhibit stamps use E-n; paper-book Index uses P-n.
        if letter == "E":
            letter = "P"
        object.__setattr__(self, "series", letter)

    @property
    def label(self) -> str:
        return f"Annexure {self.series}-{self.number}"

    @property
    def slot_id(self) -> str:
        return f"annexure_{self.series.lower()}{self.number}"


# OCR: ANNEXURE - E-1, ANNEXURE-:-- E-5, ANNEXURE-P/4, ANNEXURE P-1.
_ANNEXURE_HEADING_RE = re.compile(
    r"(?:annexure|annx\.?)\s*[-–—:.\s]*?(?:no\.?\s*)?"
    r"(?:(?P<series>[A-Za-z]|petitioner|respondent)\s*[-–—/:.\s]*)?"
    r"(?P<num>\d{1,3})\b"
    r"|(?:^|\n)\s*(?:marked\s+)?(?:as\s+)?"
    r"(?P<bare_series>[PREpre])[-\s]?(?P<bare_num>\d{1,3})\b",
    re.IGNORECASE,
)
# Paper-book title or stamp line (not an Index row like "15. ANNEXURE-P/4").
_ANNEXURE_TITLE_LINE_RE = re.compile(
    r"^(?:\d{1,4}\s+)?"
    r"(?:"
    r"(?:annexure|annx\.?)\s*[-–—:.\s]*?(?:no\.?\s*)?"
    r"(?:(?P<series>[A-Za-z]|petitioner|respondent)\s*[-–—/:.\s]*)?"
    r"(?P<num>\d{1,3})"
    r"|(?P<bare_series>[PREpre])[-\s]?(?P<bare_num>\d{1,3})"
    r")\b",
    re.IGNORECASE,
)
_INDEX_ROW_ANNEXURE_RE = re.compile(
    r"^\d{1,3}[.\)]\s*(?:annexure|annx\.?)\b",
    re.IGNORECASE,
)
_ANNEXURE_PAGE_CITE_RE = re.compile(
    r"^[\[\(]?\s*(?:pg|pgs?|pages?|pp)\b|"
    r"^\d{1,4}\s*[-–—]\s*\d{1,4}\b|"
    r"\[?\s*pg\s*[_\.\-–—\s]*to\b|"
    r"\ba\s+true\b|\btrue\s+cop",
    re.IGNORECASE,
)
_ANNEXURE_CITATION_PREV_RE = re.compile(
    r"annexed\s+herewith|marked\s+as|true\s+cop(?:y|ies)\s+of",
    re.IGNORECASE,
)
_APPLICATION_CAUSE_RE = re.compile(r"in the supreme court of india", re.IGNORECASE)
_AFFIDAVIT_HEADING_RE = re.compile(
    r"(?m)^(?:A\s+F\s+F\s+I\s+D\s+A\s+V\s+I\s+T|AFFIDAVIT)\b",
    re.IGNORECASE,
)
_CERTIFICATE_HEADING_RE = re.compile(
    r"(?m)^(?:C\s+E\s+R\s+T\s+I\s+F\s+I\s+C\s+A\s+T\s+E|CERTIFICATE)\b",
    re.IGNORECASE,
)
_PROTECTED_HEADING_PARTS = frozenset(
    {"Cover Page", "Index", "AOR's Certificate", "Affidavit"}
)
_RECLASSIFY_FAMILIES = frozenset({MAIN_PETITION_PART, APPLICATION_FAMILY})


def family_split_name(name: str) -> str:
    """Collapse Annexure P-n / Application n onto the LlamaSplit family name."""
    folded = _fold(name)
    if folded.startswith("annexure"):
        return ANNEXURE_FAMILY
    if folded.startswith("application"):
        return APPLICATION_FAMILY
    return name


_NUMBERED_ANNEXURE_RE = re.compile(r"^annexure [a-z]-?\d{1,3}$")


def _is_numbered_annexure(name: str) -> bool:
    return bool(_NUMBERED_ANNEXURE_RE.fullmatch(_fold(name)))


def _normalize_annexure_series(raw: str | None) -> str:
    token = (raw or "").strip().lower()
    if token.startswith("pet"):
        return "P"
    if token.startswith("res"):
        return "R"
    if token and token[0].isalpha():
        return token[0].upper()
    return "P"


def _annexure_mark_from_match(match: re.Match[str]) -> AnnexureMark | None:
    groups = match.groupdict()
    number_raw = groups.get("num") or groups.get("bare_num")
    if not number_raw:
        # Legacy positional groups (older callers / tests).
        number_raw = next((g for g in match.groups() if g and str(g).isdigit()), None)
    if not number_raw:
        return None
    number = int(number_raw)
    if not (1 <= number <= MAX_NUMBERED_PART):
        return None
    series = _normalize_annexure_series(
        groups.get("series") or groups.get("bare_series")
    )
    return AnnexureMark(number=number, series=series)


def numbered_part_slot_id(name: str) -> str | None:
    """Map `Annexure P-12` / `Annexure E-2` / `Application 7` onto a dynamic slot id."""
    folded = _fold(name)
    if folded in {"annexures", "annexure"}:
        return "annexures"
    match = re.fullmatch(r"annexure ([a-z])-?(\d{1,3})", folded)
    if match:
        series = match.group(1)
        number = int(match.group(2))
        if 1 <= number <= MAX_NUMBERED_PART:
            return f"annexure_{series}{number}"
    if folded == "application":
        return "applications"
    match = re.fullmatch(r"application (\d{1,3})", folded)
    if match:
        number = int(match.group(1))
        if 1 <= number <= MAX_NUMBERED_PART:
            return f"application_{number}"
    return None


def _annexure_mark_in_window(text: str) -> AnnexureMark | None:
    match = _ANNEXURE_HEADING_RE.search(text or "")
    if not match:
        return None
    return _annexure_mark_from_match(match)


def _annexure_mark_from_title_or_stamp(text: str) -> AnnexureMark | None:
    """Series+number from a title line near the top or a short foot stamp line."""
    lines = [ln.strip() for ln in (text or "").splitlines() if ln.strip()]
    if not lines:
        return None
    if _looks_like_index_table(text) or _looks_like_sci_interlocutory(text):
        return None

    def _mark_from_title_line(
        line: str, *, allow_body_tail: bool, prev_line: str = ""
    ) -> AnnexureMark | None:
        if _INDEX_ROW_ANNEXURE_RE.match(line):
            return None
        match = _ANNEXURE_TITLE_LINE_RE.match(line)
        if not match:
            return None
        mark = _annexure_mark_from_match(match)
        if mark is None:
            return None
        # LOD narrative: "... marked as" / "annexed herewith" then ANNEXURE P-n [Pg].
        if _ANNEXURE_CITATION_PREV_RE.search(prev_line):
            return None
        # Narrative citations like "ANNEXURE-P/4 (Pg 72-95)." are not stamps.
        remainder = line[match.end() :].strip(" .;:-")
        if remainder:
            if _ANNEXURE_PAGE_CITE_RE.search(remainder):
                return None
            if not allow_body_tail and not remainder.isdigit():
                return None
        return mark

    for index, line in enumerate(lines[:24]):
        prev = lines[index - 1] if index else ""
        mark = _mark_from_title_line(line, allow_body_tail=True, prev_line=prev)
        if mark is not None:
            return mark
    for line in lines[-10:]:
        if len(line) > 40:
            continue
        mark = _mark_from_title_line(line, allow_body_tail=False)
        if mark is not None:
            return mark
    return None


def annexure_ref_in_heading(text: str) -> AnnexureMark | None:
    """Return printed Annexure series+number from a page title or stamp."""
    folded_head = _fold(_heading_window(text, lines=6))
    if (
        _looks_like_index_table(text)
        or _looks_like_sci_interlocutory(text)
        or "list of dates" in folded_head
        or folded_head.startswith("synopsis")
    ):
        return None
    # Numbered pleading paragraphs ("12. That the petitioner…") are not stamps.
    if re.search(r"(?m)^\s*\d{1,2}\.\s+that\s+the\b", text or "", re.I):
        return None
    return _annexure_mark_from_title_or_stamp(text)


def annexure_mark_in_heading(text: str) -> int | None:
    """Return printed annexure number from a page title or stamp (any series)."""
    mark = annexure_ref_in_heading(text)
    return mark.number if mark else None


def annexure_label_from_text(text: str) -> str | None:
    """Label like ``Annexure E-1`` when the page prints that stamp."""
    mark = annexure_ref_in_heading(text)
    return mark.label if mark else None


def page_starts_application(text: str) -> bool:
    if _is_sci_application_start(text):
        return True
    head = "\n".join((text or "").splitlines()[:20])[:2000]
    if re.search(
        r"(?m)^(?:A\s+P\s+P\s+L\s+I\s+C\s+A\s+T\s+I\s+O\s+N|APPLICATION)\b",
        head,
    ):
        return True
    if re.search(r"\bcrl\.?\s*m\.?p\.?\b", head, re.I):
        return True
    if re.search(r"application seeking exemption", head, re.I):
        return True
    return bool(
        _APPLICATION_CAUSE_RE.search(head) and re.search(r"(?m)^APPLICATION\b", head)
    )


def _heading_window(text: str, lines: int = 12) -> str:
    return "\n".join((text or "").splitlines()[:lines])[:1200]


def _is_sci_application_start(text: str) -> bool:
    head = _fold(_heading_window(text, lines=20))
    window = _fold((text or "")[:2000])
    if "in the supreme court of india" not in head and "in the supreme court of india" not in window:
        return False
    # Paper-book covers list pending I.A.s; that is not an application start.
    if re.search(
        r"for index\s+(?:kindly|please)\s+see\s+inside|"
        r"\{\s*cover\s+page\s*\}|"
        r"\bpaper\s+book\b",
        text[:2200] or "",
        re.I,
    ):
        return False
    return bool(
        re.search(r"(?m)^application\b", head)
        or re.search(r"\bi\.?\s*a\.?\b", head)
        or "interlocutory application" in head
        or "crl. mp" in head
        or "crl mp" in head
        or "application seeking" in window
    )


def _is_strong_document_start(text: str) -> bool:
    head = _heading_window(text)
    return bool(
        _is_sci_application_start(text)
        or _AFFIDAVIT_HEADING_RE.search(head)
        or _CERTIFICATE_HEADING_RE.search(head)
        or re.search(r"(?m)^VAKALAT\s*NAMA\b", head, re.I)
        or re.search(r"(?m)^MEMO OF PART", head, re.I)
    )


def _page_has_protected_part(names: Sequence[str] | None) -> bool:
    return any(name in _PROTECTED_HEADING_PARTS for name in parts_on_page(names))


def _stealable_family(names: Sequence[str] | None) -> str | None:
    for name in parts_on_page(names):
        family = family_split_name(name)
        if family in _RECLASSIFY_FAMILIES:
            return family
    return None


_UNCATEGORIZED_LABELS = frozenset(
    {"uncategorized", "undefined", "unknown", "other", "n/a", "na"}
)
_CARRY_BLOCKING_PARTS = frozenset(
    {
        "Cover Page",
        "Index",
        "Record of Proceedings",
        # Checklist / Listing / OR may span 2 pages — allow forward carry.
    }
)


def _is_real_split_label(name: str) -> bool:
    folded = _fold(name)
    return bool(folded) and folded not in _UNCATEGORIZED_LABELS


def _looks_like_index_table(text: str) -> bool:
    head = _fold(text[:900])
    return "index" in head and (
        "particulars" in head or "page no" in head or "part i" in head
    )


def _looks_like_sci_interlocutory(text: str) -> bool:
    """True for this-Court I.A. / application pages that only mention annexures."""
    if _is_sci_application_start(text):
        return True
    blob = _fold(text[:1600])
    cites_annexure = "annexure" in blob
    ia_prose = (
        "true translated copy" in blob
        or "prima facie case" in blob
        or "exempt" in blob
        and "translation" in blob
        or "balance of convenience" in blob
    )
    return cites_annexure and ia_prose


def _memo_of_parties_heading(text: str) -> bool:
    head = _fold(_heading_window(text, lines=8))
    return "memo of part" in head


def _vakalatnama_heading(text: str) -> bool:
    head = _fold(_heading_window(text, lines=12))
    compact = re.sub(r"[^a-z]", "", head)
    return "vakalatnama" in compact


def _can_override_with_annexure(names: Sequence[str] | None, text: str) -> bool:
    """True when a printed Annexure P-n stamp should win over the current label."""
    parts = parts_on_page(names)
    if not parts:
        return True
    if any(not _is_real_split_label(name) for name in parts):
        return True
    if _looks_like_index_table(text):
        return False
    if any(name in _CARRY_BLOCKING_PARTS for name in parts):
        return False
    if _stealable_family(parts):
        return True
    # Checklist exhibits often mislabel body pages as Affidavit.
    if any(name == "Affidavit" for name in parts):
        return True
    return all(family_split_name(name) == ANNEXURE_FAMILY for name in parts)


def _replace_stealable_with_annexure(names: list[str], label: str) -> list[str]:
    replaced: list[str] = []
    for name in names:
        family = family_split_name(name)
        if family in _RECLASSIFY_FAMILIES or name == "Affidavit":
            if label not in replaced:
                replaced.append(label)
            continue
        if name not in replaced:
            replaced.append(name)
    if label not in replaced:
        replaced.append(label)
    return replaced


def reclassify_pages_from_headings(
    page_parts: PagePartMap,
    page_text: Mapping[int, str],
) -> PagePartMap:
    """Force Annexure P-n when a Main Petition/Application page prints that banner.

    Extends forward only through the same wrong family, consecutive pages, until
    the next annexure heading or an SCI Application / Affidavit / Certificate
    start. A later Main Petition block with no banner is left alone. Affidavit
    mislabels on stamped annexure pages are also retagged.
    """
    updated = {page: list(names) for page, names in page_parts.items()}
    if not updated:
        return updated
    last_page = max(updated)
    page = min(updated)
    while page <= last_page:
        names = updated.get(page)
        text = page_text.get(page, "")
        mark = annexure_ref_in_heading(text)
        if not names:
            page += 1
            continue
        if _page_has_protected_part(names) and not (
            mark and _can_override_with_annexure(names, text)
        ):
            page += 1
            continue
        wrong_family = _stealable_family(names)
        affidavit_only = parts_on_page(names) == ["Affidavit"]
        if not mark or (not wrong_family and not affidavit_only):
            page += 1
            continue
        label = mark.label
        updated[page] = _replace_stealable_with_annexure(updated[page], label)
        cursor = page + 1
        while cursor in updated:
            nxt_names = updated[cursor]
            nxt_text = page_text.get(cursor, "")
            if annexure_ref_in_heading(nxt_text):
                break
            if _is_strong_document_start(nxt_text):
                break
            if _page_has_protected_part(nxt_names) and not _can_override_with_annexure(
                nxt_names, nxt_text
            ):
                break
            nxt_family = _stealable_family(nxt_names)
            nxt_affidavit = parts_on_page(nxt_names) == ["Affidavit"]
            if (
                wrong_family
                and nxt_family not in {wrong_family, None}
                and not nxt_affidavit
            ):
                break
            if not wrong_family and not nxt_affidavit and nxt_family:
                break
            if nxt_family or nxt_affidavit or _annexure_claimable(nxt_names):
                updated[cursor] = _replace_stealable_with_annexure(nxt_names, label)
            else:
                break
            cursor += 1
        page = cursor if cursor > page else page + 1
    return updated


def _replace_family_label(names: list[str], family: str, label: str) -> list[str]:
    replaced = [label if family_split_name(name) == family else name for name in names]
    if label not in replaced:
        replaced.append(label)
    return replaced


def _annexure_claimable(names: Sequence[str] | None) -> bool:
    """Unlabeled or annexure-only pages may take a P-n label. Never steal other parts."""
    parts = parts_on_page(names)
    if not parts:
        return True
    return all(family_split_name(name) == ANNEXURE_FAMILY for name in parts)


def _annexure_number_from_label(name: str) -> int | None:
    match = re.fullmatch(r"annexure [a-z]-?(\d{1,3})", _fold(name))
    if not match:
        return None
    number = int(match.group(1))
    return number if 1 <= number <= MAX_NUMBERED_PART else None


def _page_annexure_number(names: Sequence[str] | None) -> int | None:
    for name in parts_on_page(names):
        number = _annexure_number_from_label(name)
        if number is not None:
            return number
    return None


def _contiguous_annexure_pages(
    start: int,
    stop_before: int,
    page_parts: PagePartMap,
    page_text: Mapping[int, str] | None = None,
    *,
    keep_number: int | None = None,
) -> list[int]:
    """Already-labelled annexure pages from start. Stops at a gap or other document."""
    texts = page_text or {}
    pages: list[int] = []
    for page in range(start, stop_before):
        names = page_parts.get(page)
        if names is None:
            break
        if not _annexure_claimable(names):
            break
        existing = _page_annexure_number(names)
        if (
            keep_number is not None
            and existing is not None
            and existing > keep_number
            and len((texts.get(page) or "").strip()) < 40
            and page > start
        ):
            # Image-only later Llama P-n after this stamp — stop extending.
            break
        pages.append(page)
    return pages


def _contiguous_gap_pages(
    start: int,
    stop_before: int,
    page_parts: PagePartMap,
    page_text: Mapping[int, str] | None = None,
    *,
    keep_number: int | None = None,
) -> list[int]:
    """Claimable pages in a bounded heading gap, including unlabeled leaves."""
    texts = page_text or {}
    pages: list[int] = []
    for page in range(start, stop_before):
        names = page_parts.get(page)
        if not _annexure_claimable(names):
            break
        existing = _page_annexure_number(names)
        if (
            keep_number is not None
            and existing is not None
            and existing > keep_number
            and len((texts.get(page) or "").strip()) < 40
            and page > start
        ):
            break
        pages.append(page)
    return pages


def _assign_gap_to_missing_marks(
    pages: Sequence[int], missing: Sequence[AnnexureMark]
) -> dict[int, str]:
    """Split gap pages across skipped annexure marks in order.

    One skipped number gets the whole gap. Several skipped numbers share
    the pages as evenly as possible, earlier marks first.
    """
    labels: dict[int, str] = {}
    if not pages or not missing:
        return labels
    n_pages = len(pages)
    n_marks = len(missing)
    base, extra = divmod(n_pages, n_marks)
    index = 0
    for offset, mark in enumerate(missing):
        count = base + (1 if offset < extra else 0)
        for _ in range(count):
            labels[int(pages[index])] = mark.label
            index += 1
    return labels


def _annexure_printed_labels(
    page_parts: PagePartMap,
    page_text: Mapping[int, str],
) -> dict[int, str] | None:
    """Label annexure pages from the printed series+number heading on the page.

    Printed ``ANNEXURE-E-n`` stamps normalize to paper-book ``Annexure P-n``.
    Consecutive marks keep body pages with the earlier mark. A short skipped
    heading gap (P-2 then P-4 over a few sheets) invents the missing P-3. A
    long body between P-1 and P-4 keeps P-1. Never overwrite a differently
    numbered Llama annexure (P-7) with an earlier stamp run (P-6).
    """
    annexure_pages = sorted(
        page
        for page, names in page_parts.items()
        if any(
            family_split_name(name) == ANNEXURE_FAMILY for name in parts_on_page(names)
        )
    )
    if not annexure_pages:
        return None
    first = annexure_pages[0]
    last = annexure_pages[-1]
    starts: list[tuple[int, AnnexureMark]] = []
    for page in range(first, last + 1):
        if not _annexure_claimable(page_parts.get(page)):
            continue
        mark = annexure_ref_in_heading(page_text.get(page, ""))
        if mark:
            starts.append((page, mark))
    if not starts:
        return None

    labels: dict[int, str] = {}
    first_start, first_mark = starts[0]
    page = first_start - 1
    while page >= first and _annexure_claimable(page_parts.get(page)):
        existing = _page_annexure_number(page_parts.get(page))
        if existing is not None and existing > first_mark.number:
            break
        labels[page] = first_mark.label
        page -= 1

    for index, (start_page, mark) in enumerate(starts):
        if index + 1 < len(starts):
            next_page, next_mark = starts[index + 1]
            same_series = next_mark.series == mark.series
            # Between two printed stamps, the gap belongs to this run (or short
            # invented missing marks). Do not preserve stray Llama P-7 mid-gap.
            if same_series and next_mark.number > mark.number + 1:
                if _annexure_claimable(page_parts.get(start_page)):
                    labels[start_page] = mark.label
                gap = _contiguous_gap_pages(
                    start_page + 1,
                    next_page,
                    page_parts,
                    page_text,
                )
                missing = [
                    AnnexureMark(number=n, series=mark.series)
                    for n in range(mark.number + 1, next_mark.number)
                ]
                # Short heading skips invent missing P-n; long exhibit bodies
                # between distant stamps keep the earlier mark.
                if len(gap) <= max(3, len(missing) * 2):
                    labels.update(_assign_gap_to_missing_marks(gap, missing))
                else:
                    for page in gap:
                        labels[page] = mark.label
            else:
                for page in _contiguous_gap_pages(
                    start_page,
                    next_page,
                    page_parts,
                    page_text,
                ):
                    labels[page] = mark.label
            continue
        for page in _contiguous_annexure_pages(
            start_page,
            last + 1,
            page_parts,
            page_text,
            keep_number=mark.number,
        ):
            labels[page] = mark.label
    return labels


def _apply_segment_numbers(
    page_parts: PagePartMap,
    groups: list[list[int]],
    family: str,
    label_fmt: str,
) -> PagePartMap:
    updated = {page: list(names) for page, names in page_parts.items()}
    index = 0
    for group in groups:
        pages = [
            page
            for page in group
            if page in updated
            and any(family_split_name(name) == family for name in updated[page])
        ]
        if not pages:
            continue
        index += 1
        label = label_fmt.format(index)
        for page in pages:
            updated[page] = _replace_family_label(updated[page], family, label)
    return updated


def _sequential_family_labels(
    pages: list[int],
    page_text: Mapping[int, str],
    family: str,
) -> dict[int, str] | None:
    """Number application pages at each new cause title. Annexures use printed P-n."""
    if family == ANNEXURE_FAMILY:
        return None
    if len(pages) < 2:
        return None
    labels: dict[int, str] = {}
    index = 0
    starts = 0
    for offset, page in enumerate(pages):
        text = page_text.get(page, "")
        is_start = offset == 0 or page_starts_application(text)
        if offset == 0:
            index = 1
            starts = 1
        elif is_start:
            index += 1
            starts += 1
        if index:
            labels[page] = f"Application {index}"
    if starts < 2:
        return None
    return labels


def explode_repeating_split_parts(
    page_parts: PagePartMap,
    page_text: Mapping[int, str] | None = None,
) -> PagePartMap:
    """Split a merged Annexures/Application run into P-n / Application n.

    Printed ANNEXURE-Pn banners retag Main Petition/Application pages first.
    The printed mark on the page is the annexure number: a heading Annexure P-1
    is Annexure P-1 even if LlamaSplit ordered that segment as P-2. Consecutive
    marks keep body pages with the earlier annexure; a skipped mark fills the
    gap as the missing P-n. Applications stay sequential at each new cause
    title. Each numbered Annexure P-n keeps only its first contiguous page run.
    Other types keep only their longest contiguous page run.
    """
    updated = {page: list(names) for page, names in page_parts.items()}
    texts = page_text or {}
    if texts:
        updated = reclassify_pages_from_headings(updated, texts)
        annexure_labels = _annexure_printed_labels(updated, texts)
        if annexure_labels:
            for page, label in annexure_labels.items():
                updated[page] = _replace_family_label(
                    updated.get(page, []), ANNEXURE_FAMILY, label
                )
        for page, names in list(updated.items()):
            if _page_has_protected_part(names):
                continue
            mark = annexure_ref_in_heading(texts.get(page, ""))
            if not mark:
                continue
            label = mark.label
            if _stealable_family(names):
                updated[page] = _replace_stealable_with_annexure(names, label)
            elif _annexure_claimable(names):
                updated[page] = _replace_family_label(names, ANNEXURE_FAMILY, label)
        app_pages = sorted(
            page
            for page, names in updated.items()
            if any(
                family_split_name(name) == APPLICATION_FAMILY
                for name in parts_on_page(names)
            )
        )
        app_labels = (
            _sequential_family_labels(app_pages, texts, APPLICATION_FAMILY)
            if app_pages
            else None
        )
        if app_labels:
            for page, label in app_labels.items():
                updated[page] = _replace_family_label(
                    updated.get(page, []), APPLICATION_FAMILY, label
                )
    return collapse_repeated_split_pages(updated)


def page_parts_from_split(job: Any) -> PagePartMap:
    """Map 1-indexed page number → one or more split category names."""
    result = getattr(job, "result", None) or job
    segments = getattr(result, "segments", None)
    if segments is None and isinstance(result, dict):
        segments = result.get("segments")
    if not segments:
        return {}

    mapping: PagePartMap = {}
    for segment in segments:
        if isinstance(segment, dict):
            category = segment.get("category")
            pages = segment.get("pages") or []
        else:
            category = getattr(segment, "category", None)
            pages = getattr(segment, "pages", None) or []
        numbers: list[int] = []
        for page in pages:
            try:
                numbers.append(int(page))
            except (TypeError, ValueError):
                continue
        parts = [
            part for part in parts_on_page(category) if _is_real_split_label(part)
        ]
        for part in parts:
            for page in numbers:
                current = mapping.setdefault(page, [])
                if part not in current:
                    current.append(part)
    return mapping


def complete_split_page_coverage(
    page_parts: PagePartMap,
    page_text: Mapping[int, str],
    *,
    page_count: int,
) -> PagePartMap:
    """Hybrid repair after LlamaSplit (anchors, nesting, first-run outer parts).

    Prefer :func:`repair_compiled_split` when duplicate metadata is needed.
    """
    from .split_repair import repair_compiled_split

    repaired, _duplicates = repair_compiled_split(
        page_parts, page_text, page_count=page_count
    )
    return repaired


def parts_named_in_text(text: str) -> list[str]:
    """Split parts whose name or description nickname appears in `text`."""
    blob = _fold(text)
    found: list[str] = []
    for name, description in _split_categories():
        needles = _needles_for_part(name, description)
        matched = any(needle in blob for needle in needles)
        if not matched and name == MAIN_PETITION_PART:
            # "the Petition" in catalogue prose; do not match "petitioner".
            matched = bool(re.search(r"(?<![a-z])petition(?![a-z])", blob))
        if matched and name not in found:
            found.append(name)
    return found


def _strip_landmark_clauses(text: str) -> str:
    """Drop paper-book locators so 'before the Cover Page' is not a target."""
    cleaned = _PLACEMENT_PAREN.sub(" ", text or "")
    pieces: list[str] = []
    cursor = 0
    for match in _LANDMARK_PREP.finditer(cleaned):
        stop = _LANDMARK_STOP.search(cleaned, match.end())
        end = stop.start() if stop else len(cleaned)
        clause = cleaned[match.start() : end]
        pieces.append(cleaned[cursor : match.start()])
        if not parts_named_in_text(clause):
            pieces.append(clause)
        cursor = end
    pieces.append(cleaned[cursor:])
    return re.sub(r"[\s,]+", " ", "".join(pieces)).strip(" ,")


def _normalize_part_list(names: list[str] | None) -> list[str]:
    found: list[str] = []
    for raw in names or []:
        part = normalize_part_name(str(raw or "").strip())
        if part and part not in found:
            found.append(part)
    return found


def catalogue_inspect_parts(defect: Defect) -> list[str]:
    """Explicit inspect_parts from the catalogue, if authored."""
    return _normalize_part_list(getattr(defect, "inspect_parts", None))


def catalogue_context_parts(defect: Defect) -> list[str]:
    """Explicit context_parts from the catalogue, if authored."""
    return _normalize_part_list(getattr(defect, "context_parts", None))


def catalogue_exclude_parts(defect: Defect) -> list[str] | None:
    """Explicit exclude_parts, or None when the catalogue omitted the field."""
    raw = getattr(defect, "exclude_parts", None)
    if raw is None:
        return None
    return _normalize_part_list(raw)


def _parts_parsed_from_where_to_look(defect: Defect) -> list[str]:
    """Fallback: parse Split names from where_to_look, ignoring landmarks."""
    found: list[str] = []
    for step in defect.where_to_look:
        for name in parts_named_in_text(_strip_landmark_clauses(step)):
            if name not in found:
                found.append(name)
    return found


def parts_named_in_where_to_look(defect: Defect) -> list[str]:
    """Split parts this check should open for retrieval.

    Prefers catalogue inspect_parts + context_parts. Falls back to parsing
    where_to_look when those fields are empty (legacy rows).
    """
    inspect = catalogue_inspect_parts(defect)
    context = catalogue_context_parts(defect)
    if inspect or context:
        parts = list(inspect)
        for name in context:
            if name not in parts:
                parts.append(name)
        return parts
    return _parts_parsed_from_where_to_look(defect)


def preferred_parts_for_defect(defect: Defect) -> list[str]:
    """Retrieval targets from catalogue inspect/context parts (or parsed where_to_look)."""
    return parts_named_in_where_to_look(defect)


def pool_search_queries() -> list[str]:
    """Short caption queries covering typical SCI filing parts."""
    seen: set[str] = set()
    queries: list[str] = []
    for query in (*FILING_CAPTION_QUERIES, *split_part_names()):
        key = query.strip().lower()
        if not key or key in seen:
            continue
        seen.add(key)
        queries.append(query)
    return queries


def _clip_query(text: str, *, limit: int = PINECONE_QUERY_MAX_CHARS) -> str:
    cleaned = re.sub(r"\s+", " ", (text or "").strip())
    if len(cleaned) <= limit:
        return cleaned
    return cleaned[:limit].rsplit(" ", 1)[0].strip()


# OCR phrases that often stand in for stamps, seals, signatures, and paper size.
# Needles are matched as whole words against the catalogue row.
_PRESENCE_QUERY_HINTS: tuple[tuple[str, str], ...] = (
    ("stamp", "Advocates Welfare Fund stamp court fee stamp"),
    ("seal", "notarial seal company seal"),
    ("notary", "notary oath commissioner"),
    ("signature", "Sd/- digitally signed signature"),
    ("quarter margin", "quarter margin 4 cm 2 cm"),
    ("4 cm", "4 cm 2 cm margin"),
    ("a4", "A4 29.7 cm 21 cm"),
    ("foolscap", "demy foolscap A4"),
    ("times new roman", "Times New Roman font size 14"),
    ("line spacing", "one and a half line spacing"),
)


def _presence_queries_for_defect(defect: Defect) -> list[str]:
    """Extra Pinecone queries so stamps, seals and layout marks are retrieved."""
    blob = " ".join([defect.defect, defect.requirement, *defect.where_to_look]).lower()
    queries: list[str] = []
    for needle, query in _PRESENCE_QUERY_HINTS:
        if re.search(rf"\b{re.escape(needle)}\b", blob):
            queries.append(query)
    return queries


def pinecone_queries_for_defect(defect: Defect) -> list[str]:
    """Queries sent to Pinecone: target part names, short headings, and cues.

    Long where-to-look sentences are not sent — they mention landmarks
    ("before the Cover Page") and pull the wrong pages.
    """
    queries: list[str] = []
    queries.extend(parts_named_in_where_to_look(defect))
    queries.extend(_presence_queries_for_defect(defect))
    if defect.trigger_words:
        queries.extend(
            p.strip() for p in re.split(r"[;|]", defect.trigger_words) if p.strip()
        )
    for step in defect.where_to_look:
        heading = _heading_from_where_to_look(step)
        if heading:
            queries.append(heading)
        clipped = _clip_query(_bare_where_to_look(_strip_landmark_clauses(step)))
        if clipped and len(clipped) <= 80:
            queries.append(clipped)
    if not queries:
        queries.extend(preferred_parts_for_defect(defect))
    return _unique_terms(queries)


def match_terms_for_defect(defect: Defect) -> list[str]:
    """Phrases to score an in-memory chunk against this defect."""
    terms = list(pinecone_queries_for_defect(defect))
    terms.extend(preferred_parts_for_defect(defect))
    return _unique_terms(terms)


def _unique_terms(terms: list[str]) -> list[str]:
    seen: set[str] = set()
    unique: list[str] = []
    for term in terms:
        key = term.lower().strip()
        if not key or key in seen:
            continue
        seen.add(key)
        unique.append(term.strip())
    return unique


def required_parts_for_defect(defect: Defect) -> list[str]:
    """Parts that must be in excerpts before defect_found is allowed.

    Catalogue inspect_parts is the source of truth. When absent, fall back to
    parsed where_to_look with a few legacy category heuristics.
    """
    inspect = catalogue_inspect_parts(defect)
    if inspect:
        return inspect

    named = _parts_parsed_from_where_to_look(defect)
    if not named:
        return preferred_parts_for_defect(defect)

    if len(named) > 3 and defect.where_to_look:
        first = parts_named_in_text(_strip_landmark_clauses(defect.where_to_look[0]))
        if first:
            return first
    return named


def excluded_parts_for_defect(defect: Defect) -> list[str]:
    """Parts that must not be used as evidence / Index-only filler."""
    explicit = catalogue_exclude_parts(defect)
    if explicit is not None:
        return explicit
    if allows_index_evidence(defect):
        return []
    return ["Index"]


def allows_index_evidence(defect: Defect) -> bool:
    """Whether Index listing lines may be used as evidence for this check."""
    explicit = catalogue_exclude_parts(defect)
    if explicit is not None:
        return "Index" not in explicit
    inspect = catalogue_inspect_parts(defect)
    if inspect:
        return "Index" in inspect
    return "Index" in _parts_parsed_from_where_to_look(defect)


def chunks_cover_part(chunks: list[dict[str, Any]], part: str) -> bool:
    pages = [c for c in chunks if c.get("chunk_kind") != "summary"]
    return any(_part_match(c, [part]) for c in pages)


def missing_required_parts(defect: Defect, chunks: list[dict[str, Any]]) -> list[str]:
    return [
        part
        for part in required_parts_for_defect(defect)
        if not chunks_cover_part(chunks, part)
    ]


def _bare_where_to_look(step: str) -> str:
    return re.sub(
        r"^(check|read|look at|look for|examine|verify|confirm|review|go to|search)\s+"
        r"(the\s+|for\s+the\s+|for\s+|a\s+|for a\s+)?",
        "",
        step.strip(),
        flags=re.IGNORECASE,
    )


def _heading_from_where_to_look(step: str) -> str | None:
    """Pull a form title out of a 'Check the Listing Proforma…' sentence."""
    text = _bare_where_to_look(_strip_landmark_clauses(step))
    named = parts_named_in_text(text)
    if named:
        return named[0]
    # First clause before 'and' / comma, if it is short.
    clause = re.split(r"[,.]|\band\b", text, maxsplit=1)[0].strip()
    if 3 <= len(clause) <= 80:
        return clause
    return None


def max_chunks_for_defect(defect: Defect, *, ceiling: int) -> int:
    """How many page excerpts this defect is allowed, at most.

    One-form checks (Listing Proforma columns) stay at 3 pages. Checks that
    must open several documents get PAGES_PER_TARGET_PART each so a
    high-scoring Affidavit cannot crowd out Vakalatnama.
    """
    targets = parts_named_in_where_to_look(defect) or preferred_parts_for_defect(defect)
    budget = ceiling
    need = max(len(targets), 1) * PAGES_PER_TARGET_PART
    budget = max(budget, min(need, ceiling))
    # One document part stays small so a high-scoring neighbour cannot crowd it.
    if len(targets) <= 1:
        tight = 3 if len(defect.where_to_look) <= 2 else 6
        if defect.parent_check_id:
            tight = max(tight, 4)
        budget = min(budget, tight)
    return max(1, min(budget, ceiling))


def _chunk_score(chunk: dict[str, Any]) -> float:
    try:
        return float(chunk.get("score") or 0)
    except (TypeError, ValueError):
        return 0.0


def keep_nearby_scores(
    chunks: list[dict[str, Any]],
    *,
    max_n: int,
    max_log_gap: float = MAX_SCORE_LOG_GAP,
) -> list[dict[str, Any]]:
    """Drop neighbours that are far behind the best score.

    Pinecone `top_k` always fills the quota. Rank 4–8 can be a different
    document part with a much lower score; sending those invites the model
    to quote the wrong page. The cutoff is logarithmic so it does not depend
    on whether the best score is 0.9 or 0.3: keep if ln(best / score) ≤ gap.
    """
    if not chunks or max_n <= 0:
        return []
    best = max(_chunk_score(c) for c in chunks)
    kept: list[dict[str, Any]] = []
    for chunk in chunks:
        if len(kept) >= max_n:
            break
        score = _chunk_score(chunk)
        if best <= 0:
            kept.append(chunk)
            continue
        if score <= 0:
            continue
        log_gap = math.log(best) - math.log(score)
        if log_gap > max_log_gap:
            continue
        kept.append(chunk)
    return kept or chunks[:1]


def slice_record_for_defect(
    record: dict[str, Any] | None,
    defect: Defect,
) -> dict[str, Any] | None:
    if not record:
        return record
    keys = list(ALWAYS_RECORD_FIELDS)
    keys.extend(k for k in record if k not in keys)
    sliced = {key: record[key] for key in keys if key in record}
    return sliced or record


def _part_families(name: str) -> set[str]:
    folded = (name or "").strip().lower()
    families = {folded}
    if folded.startswith("annexure"):
        families.add("annexures")
        families.add("annexure")
    if folded.startswith("application"):
        families.add("application")
    return families


def expand_parts_for_retrieval(parts: list[str]) -> list[str]:
    """Include numbered Annexure P-n / Application n labels under catch-alls."""
    extra: list[str] = []
    names = split_part_names()
    for part in parts:
        if part not in extra:
            extra.append(part)
        folded = part.lower()
        if folded in {"annexures", "annexure"}:
            for name in names:
                if name.lower().startswith("annexure") and name not in extra:
                    extra.append(name)
        elif folded == "application":
            for name in names:
                if name.lower().startswith("application") and name not in extra:
                    extra.append(name)
    return extra


def _part_match(chunk: dict[str, Any], preferred: list[str]) -> bool:
    names = parts_on_page(chunk.get("document_part"))
    if not names or not preferred:
        return False
    preferred_l = [p.lower() for p in preferred]
    return any(
        name.lower() == needle
        or needle in name.lower()
        or needle in _part_families(name)
        for name in names
        for needle in preferred_l
    )


def _term_hits(chunk: dict[str, Any], terms: list[str]) -> int:
    text = (chunk.get("text") or "").lower()
    if not text:
        return 0
    return sum(1 for term in terms if term.lower() in text)


def select_chunks_for_defect(
    pool: list[dict[str, Any]],
    defect: Defect,
    *,
    max_chunks: int,
) -> list[dict[str, Any]]:
    """Pick the best excerpts for one defect from a shared filing pool.

    `max_chunks` is a global ceiling (SCRUTINY_MAX_CHUNKS). This then:
    1. Takes at least one page from each preferred document part in the pool
       so Affidavit cannot crowd out Vakalatnama on a date-comparison check.
    2. Caps to a smaller per-defect budget.
    3. Drops neighbours whose Pinecone score is logarithmically far from the
       best remaining hit among the leftover filler pages.
    """
    if not pool:
        return []

    look_parts = parts_named_in_where_to_look(defect)
    preferred = preferred_parts_for_defect(defect)
    terms = match_terms_for_defect(defect)
    summary = [c for c in pool if c.get("chunk_kind") == "summary"]
    pages = [c for c in pool if c.get("chunk_kind") != "summary"]
    excluded = {p.lower() for p in excluded_parts_for_defect(defect)}
    if excluded:
        pages = [
            c
            for c in pages
            if not (
                parts_on_page(c.get("document_part"))
                and {name.lower() for name in parts_on_page(c.get("document_part"))}
                <= excluded
            )
        ]

    def rank_key(chunk: dict[str, Any]) -> tuple[int, int, int, float]:
        look_hit = 1 if look_parts and _part_match(chunk, look_parts) else 0
        part_hit = 1 if _part_match(chunk, preferred) else 0
        hits = _term_hits(chunk, terms)
        score = _chunk_score(chunk)
        return (look_hit, part_hit, hits, score)

    ranked = sorted(pages, key=rank_key, reverse=True)
    focused = [
        c for c in ranked if _part_match(c, preferred) or _term_hits(c, terms) > 0
    ]
    candidates = focused if focused else ranked
    if not candidates:
        return summary[:1]

    page_ceiling = max(0, max_chunks - len(summary[:1]))
    page_budget = max_chunks_for_defect(defect, ceiling=page_ceiling)

    chosen: list[dict[str, Any]] = []
    used: set[str] = set()
    reserve_order = look_parts + [p for p in preferred if p not in look_parts]
    for part in reserve_order:
        if len(chosen) >= page_budget:
            break
        for chunk in ranked:
            record_id = str(chunk.get("record_id") or "")
            if record_id in used:
                continue
            if not _part_match(chunk, [part]):
                continue
            chosen.append(chunk)
            if record_id:
                used.add(record_id)
            break

    remaining = page_budget - len(chosen)
    if remaining > 0:
        leading = [c for c in candidates if str(c.get("record_id") or "") not in used]
        for chunk in keep_nearby_scores(leading, max_n=remaining):
            record_id = str(chunk.get("record_id") or "")
            if record_id in used:
                continue
            chosen.append(chunk)
            if record_id:
                used.add(record_id)

    chosen.sort(key=lambda c: (c.get("page") is None, c.get("page") or 0))
    return summary[:1] + chosen
