"""Assemble the stored compiled-petition record after LlamaExtract."""

from __future__ import annotations

import re
from collections.abc import Mapping
from datetime import datetime
from difflib import SequenceMatcher
from typing import Any
from zoneinfo import ZoneInfo

from .document_parts import (
    FILING_TYPE_LABELS,
    MAIN_PETITION_PART,
    document_spans_from_page_parts,
    parts_on_page,
)
from .scrutiny.rules import normalize_filing_type

SCHEMA_VERSION = "extraction-v1"
JOB_TYPE = "compiled_petition"
REVIEW_STATUSES = frozenset({"pending_review", "approved", "rejected"})
LEGAL_EXTRACT_FIELDS = (
    "court",
    "petition_type",
    "cause_title",
    "petitioners",
    "respondents",
    "advocates_on_record",
    "impugned_orders",
    "relief_sort",
    "petition_date",
    "inconsistencies",
)

_ORG_PREFIXES = (
    "government of",
    "ministry of",
    "department of",
    "messrs",
    "m/s.",
    "m/s",
    "union",
)
_ORG_SUFFIXES = (
    "private limited",
    "pvt. ltd.",
    "pvt ltd",
    "limited",
    "ltd",
    "llp",
    "llc",
    "inc.",
    "corp.",
    "corporation",
    "co.",
    "company",
    "foundation",
    "trust",
    "society",
    "association",
)
_CONFIDENCE_OBJECT_KEYS = ("cause_title", "inconsistencies")
_CONFIDENCE_LIST_KEYS = (
    "petitioners",
    "respondents",
    "advocates_on_record",
    "impugned_orders",
)
_NAME_SPLIT = re.compile(r"[^a-z0-9]+")
_QUOTED_TEXT = re.compile(r'"([^"]+)"')
_QUOTED_TEXT_SINGLE = re.compile(r"'([^']+)'")
_VS_SPLIT = re.compile(r"\bvs\.?\b", re.IGNORECASE)
_ROLE_TAIL = re.compile(
    r"(?:[\s.·…]+|\s+)*(?:petitioners?|respondents?)(?:\s*\(\s*s\s*\))?\s*$",
    re.IGNORECASE,
)
_ANR_ORS_TAIL = re.compile(
    r"(?:\s*[,&]?\s*[\[(]?\s*(?:and\s+)?(?:anrs?|ors|another|others)\.?\s*[\])]?)+\s*$",
    re.IGNORECASE,
)
_BRACKETED_ANR_ORS = re.compile(
    r"\[+\s*((?:and\s+)?(?:anrs?|ors)\.?)\s*\]+",
    re.IGNORECASE,
)
_LIST_SERIAL = re.compile(r"^\s*(?:\(\s*)?\d+\s*[.)]\s*")
_TITLE_TOKEN = re.compile(
    r"\b(?:smt|shri|sh|mr|mrs|ms|km|kumari|dr|sri)\b\.?",
    re.IGNORECASE,
)
_ANR_ORS_MARK = re.compile(
    r"\b(?:and\s+)?(?:anr|ors|another|others)\b",
    re.IGNORECASE,
)
_EXTRA_PARTY_LABEL = re.compile(
    r"\b(?:petitioner|respondent)\s*(?:no\.?\s*)?(?:[2-9]|[1-9]\d)\b",
    re.IGNORECASE,
)
_CAPTION_SOURCE = re.compile(
    r"cover page|cause title|vakalatnama|affidavit|aor's declaration|"
    r"aors declaration|aor's certificate|aors certificate",
    re.IGNORECASE,
)
_PRAYER_HEADING_LINE = re.compile(
    r"^\s*(?:\d+\s*[.)]\s*)?(?:</?[^>]+>|\*{1,3}|_{1,3}|`+|#+\s*)*"
    r"(?:main\s+)?prayer"
    r"(?:</?[^>]+>|\*{1,3}|_{1,3}|`+)*\s*:?\s*$",
    re.IGNORECASE,
)
_PRAYER_HEADING_PREFIX = re.compile(
    r"^\s*(?:\d+\s*[.)]\s*)?(?:</?[^>]+>|\*{1,3}|_{1,3}|`+|#+\s*)*"
    r"(?:main\s+)?prayer"
    r"(?:</?[^>]+>|\*{1,3}|_{1,3}|`+)*\s*:?\s*",
    re.IGNORECASE,
)
_PRAYER_MARKUP = re.compile(r"</?[^>]+>|\*{1,3}|_{2}")
_PRAYER_HEADING_ANY = re.compile(
    r"(?:^|\n)\s*(?:\d+\s*[.)]\s*)?(?:</?[^>]+>|\*{1,3}|_{1,3}|`+|#+\s*)*"
    r"(?:main\s+)?prayer"
    r"(?:</?[^>]+>|\*{1,3}|_{1,3}|`+)*\s*:?\s*(?:\n|$)",
    re.IGNORECASE,
)
_MONTH_NAME = (
    r"(?:January|February|March|April|May|June|July|August|September|October|"
    r"November|December|Sept|Jan|Feb|Mar|Apr|Jun|Jul|Aug|Sep|Oct|Nov|Dec)"
)
_DAY_MONTH_DATE = (
    rf"\d{{1,2}}(?:st|nd|rd|th)?(?:\s+day)?(?:\s+of)?\s+{_MONTH_NAME},?\s+\d{{4}}"
)
_MONTH_DAY_DATE = rf"{_MONTH_NAME}\s+\d{{1,2}}(?:st|nd|rd|th)?,?\s+\d{{4}}"
_NUMERIC_DATE = r"\d{1,2}\s*[./-]\s*\d{1,2}\s*[./-]\s*\d{2,4}"
_DATE_VALUE = rf"(?:{_DAY_MONTH_DATE}|{_MONTH_DAY_DATE}|{_NUMERIC_DATE})"
_AFTER_DATE_LABEL = (
    r"\s*[:.\-]?\s*"
    r"(?:(?:at|this)\s+[A-Za-z][A-Za-z. ]{0,40}?\s+on\s+)?"
    r"(?:new\s+delhi\s*,?\s*(?:on\s+)?)?"
)
_DATE_DRAFTED = re.compile(
    rf"\bdate\s+(?:of\s+)?draft(?:ed|ing)\b{_AFTER_DATE_LABEL}({_DATE_VALUE})",
    re.IGNORECASE,
)
_DATE_FILED = re.compile(
    rf"(?:"
    rf"(?:^|\n)\s*(?:date\s+(?:of\s+)?fil(?:ed|ing)(?:\s+on)?|filed\s+on)"
    rf"|\bdate\s+(?:of\s+)?fil(?:ed|ing)(?:\s+on)?"
    rf"){_AFTER_DATE_LABEL}({_DATE_VALUE})",
    re.IGNORECASE,
)
_DATE_PLAIN = re.compile(
    rf"(?:"
    rf"(?:^|\n)\s*date{_AFTER_DATE_LABEL}"
    rf"|dated\s*:{_AFTER_DATE_LABEL}"
    rf"|\bdate\s*:{_AFTER_DATE_LABEL}"
    rf"|\bdated\s+at\s+[A-Za-z][A-Za-z. ]{{0,40}}?\s+on\s+"
    rf")({_DATE_VALUE})",
    re.IGNORECASE,
)
_BARE_DATE = re.compile(rf"({_DATE_VALUE})", re.IGNORECASE)
_PETITION_CLOSING_PAGES = 3
_EMPTY_DATE = frozenset({"n/a", "na", "none", "null", "-", "—"})
_LISTING_PARTY_MARKER = re.compile(
    r"\b(petitioner|respondent)\s+no\.?\s*(\d+)\s*[.:)]*\s*",
    re.IGNORECASE,
)
_LISTING_FIELD_AFTER_PARTY = re.compile(
    r"\(\s*[a-z]\s*\)\s*(?:phone|mobile|e-?mail|main\s+category|sub\s+classification)\b",
    re.IGNORECASE,
)
_NUMBERED_PARTY_LINE = re.compile(
    r"(?m)^\s*\|?\s*(?:[*_`#]+\s*)?(\d{1,3})\s*[.)]\s*(.+?)\s*$"
)
_NUMBERED_PARTY_TABLE_CELL = re.compile(
    r"(?is)<t[dh]\b[^>]*>\s*(?:<[^>]+>\s*)*"
    r"(\d{1,3})\s*[.)]\s*(.*?)</t[dh]>"
)
_HEADING_PREFIX = r"(?:[*_`#]+\s*)*(?:<[^>]+>\s*)*"
_HEADING_SUFFIX = r"(?:\s*</[^>]+>)*(?:\s*[*_`#]+)*"
_VERSUS_LINE = re.compile(
    rf"(?im)^\s*\|?\s*{_HEADING_PREFIX}(?:versus|vs\.?)"
    rf"{_HEADING_SUFFIX}\s*\|?\s*$"
)
_BETWEEN_LINE = re.compile(
    rf"(?im)^\s*\|?\s*{_HEADING_PREFIX}between\s*:?"
    rf"{_HEADING_SUFFIX}\s*\|?\s*$"
)
_PARTY_SCHEDULE_END = re.compile(
    rf"(?im)^\s*\|?\s*{_HEADING_PREFIX}(?:"
    r"special\s+leave\s+petition|writ\s+petition|civil\s+appeal|criminal\s+appeal|"
    r"transfer\s+petition|review\s+petition|curative\s+petition|arbitration\s+petition|"
    r"petition\s+under|appeal\s+under|to\s*,?\s*$|most\s+respectfully\s+showeth"
    r")"
)
_PARTY_NAME_TAIL = re.compile(
    r"\s+(?:s\s*/\s*o|d\s*/\s*o|w\s*/\s*o|r\s*/\s*o|c\s*/\s*o|"
    r"aged?\b|age\b|resident\b|residing\b|address\b|through\b|"
    r"petitioner\s+no\b|respondent\s+no\b).*$",
    re.IGNORECASE,
)
_CORAM_JUDGE = re.compile(
    r"hon[’']?ble\s+(?:mr|ms|mrs|dr)\.?\s+justice\s+"
    r"[a-z][a-z .’'\-]{2,80}",
    re.IGNORECASE,
)


def _as_dict(value: Any) -> dict[str, Any]:
    if isinstance(value, dict):
        return value
    dump = getattr(value, "model_dump", None)
    if callable(dump):
        payload = dump()
        return payload if isinstance(payload, dict) else {}
    return {}


def _named(value: Any) -> str | None:
    if isinstance(value, str):
        text = value.strip()
        return text or None
    data = _as_dict(value)
    name = data.get("name")
    if isinstance(name, str) and name.strip():
        return name.strip()
    return None


def _pages_int(value: Any) -> list[int]:
    pages: list[int] = []
    if value is None:
        return pages
    if isinstance(value, (int, float)):
        pages.append(int(value))
        return pages
    if isinstance(value, str):
        for bit in value.replace(";", ",").split(","):
            bit = bit.strip()
            if not bit:
                continue
            try:
                pages.append(int(bit))
            except ValueError:
                continue
        return pages
    if isinstance(value, (list, tuple)):
        for item in value:
            pages.extend(_pages_int(item))
    return pages


def _walk_for_confidence(value: Any) -> list[float]:
    scores: list[float] = []
    if isinstance(value, (int, float)) and not isinstance(value, bool):
        number = float(value)
        if 0.0 <= number <= 1.0:
            scores.append(number)
        return scores
    if isinstance(value, dict):
        for key, nested in value.items():
            lowered = str(key).lower()
            if lowered in {"confidence", "score", "overall_confidence"}:
                scores.extend(_walk_for_confidence(nested))
            elif isinstance(nested, (dict, list)):
                scores.extend(_walk_for_confidence(nested))
        return scores
    if isinstance(value, list):
        for item in value:
            scores.extend(_walk_for_confidence(item))
    return scores


def overall_confidence_from_job(job: Any) -> float | None:
    dumped = _as_dict(job)
    for key in ("extract_metadata", "extraction_metadata", "metadata"):
        blob = dumped.get(key)
        scores = _walk_for_confidence(blob)
        if scores:
            return round(sum(scores) / len(scores), 3)
    extra = getattr(job, "extract_metadata", None) or getattr(
        job, "extraction_metadata", None
    )
    scores = _walk_for_confidence(_as_dict(extra) or extra)
    if scores:
        return round(sum(scores) / len(scores), 3)
    return None


def field_confidence_from_job(job: Any) -> dict[str, Any]:
    dumped = _as_dict(job)
    for key in ("extract_metadata", "extraction_metadata", "metadata"):
        blob = dumped.get(key)
        if not isinstance(blob, dict):
            continue
        for nested_key in ("confidence", "confidence_scores", "field_confidence"):
            nested = blob.get(nested_key)
            if isinstance(nested, dict) and nested:
                return nested
    return {}


def confidence_percent_string(value: Any) -> str | None:
    """Turn 0.91, 91, or '91%' into '91%'."""
    if value is None or isinstance(value, bool):
        return None
    if isinstance(value, str):
        text = value.strip()
        if not text:
            return None
        if text.endswith("%"):
            number_text = text[:-1].strip()
            try:
                number = float(number_text)
            except ValueError:
                return text
            return f"{int(round(number))}%"
        try:
            number = float(text)
        except ValueError:
            return None
    elif isinstance(value, (int, float)):
        number = float(value)
    elif isinstance(value, dict):
        for key in ("overall_confidence", "confidence", "score"):
            if key in value:
                return confidence_percent_string(value.get(key))
        scores = _walk_for_confidence(value)
        if not scores:
            return None
        number = sum(scores) / len(scores)
    else:
        return None
    if 0.0 <= number <= 1.0:
        return f"{int(round(number * 100))}%"
    if 0.0 <= number <= 100.0:
        return f"{int(round(number))}%"
    return None


def clean_relief_sort(text: str | None) -> str | None:
    """Keep the Main Prayer body; drop heading, clause number, and markdown."""
    if not isinstance(text, str):
        return None
    blob = text.replace("\r\n", "\n").strip()
    if not blob:
        return None
    lines = blob.split("\n")
    while lines and _PRAYER_HEADING_LINE.match(lines[0]):
        lines.pop(0)
    blob = "\n".join(lines).strip()
    blob = _PRAYER_HEADING_PREFIX.sub("", blob, count=1).strip()
    blob = _PRAYER_MARKUP.sub("", blob)
    blob = re.sub(r"\n{3,}", "\n\n", blob).strip()
    return blob or None


def _normalize_printed_date(value: str) -> str:
    cleaned = re.sub(r"\s+", " ", value).strip(" .,:;-")
    return re.sub(r"\s*([./-])\s*", r"\1", cleaned)


def _last_printed_date(pattern: re.Pattern[str], text: str) -> str | None:
    found = list(pattern.finditer(text))
    if not found:
        return None
    return _normalize_printed_date(found[-1].group(1))


def _pick_petition_date(text: str) -> str | None:
    """Date Drafted wins when Date Filed on is also printed."""
    drafted = _last_printed_date(_DATE_DRAFTED, text)
    if drafted:
        return drafted
    plain = _last_printed_date(_DATE_PLAIN, text)
    if plain:
        return plain
    return _last_printed_date(_DATE_FILED, text)


def clean_petition_date(text: str | None) -> str | None:
    """Keep the petition date. Prefer Date Drafted over Date Filed on."""
    if not isinstance(text, str):
        return None
    blob = text.strip()
    if not blob or blob.casefold() in _EMPTY_DATE:
        return None
    picked = _pick_petition_date(blob)
    if picked:
        return picked
    bare = _BARE_DATE.fullmatch(blob)
    if bare:
        return _normalize_printed_date(bare.group(1))
    return None


def petition_date_from_closing(text: str | None) -> str | None:
    """Read the date after Main Prayer on the main petition's last page."""
    if not isinstance(text, str) or not text.strip():
        return None
    matches = list(_PRAYER_HEADING_ANY.finditer(text))
    scope = text[matches[-1].end() :] if matches else text
    return _date_after_prayer_label(scope)


def _date_after_prayer_label(scope: str) -> str | None:
    """Date after a prayer. A lone page number between the label and the date is skipped."""
    picked = _pick_petition_date(scope)
    if picked:
        return picked
    label = re.search(r"(?:^|\n)\s*(?:dated|date)\s*:", scope, re.IGNORECASE)
    if label is None:
        return None
    window = scope[label.end() : label.end() + 200]
    found = _BARE_DATE.search(window)
    if found is None:
        return None
    return _normalize_printed_date(found.group(1))


def petition_date_from_opening(
    page_markdown: Mapping[int, str] | None,
    opening_pages: int = 30,
) -> str | None:
    """Petition date printed after Main Prayer inside the opening window."""
    if not page_markdown:
        return None
    pages = sorted(page_markdown)
    if not pages:
        return None
    limit = pages[-1] if pages[-1] <= opening_pages else opening_pages
    text = "\n".join(
        page_markdown.get(page) or ""
        for page in pages
        if page <= limit or page > pages[-1] - 5
    )
    mains = list(
        re.finditer(r"(?:^|\n)\s*main\s+prayer\s*:?\s*(?:\n|$)", text, re.IGNORECASE)
    )
    if not mains:
        return None
    return _date_after_prayer_label(text[mains[-1].end() :])


def main_petition_closing_text(
    page_markdown: Mapping[int, str] | None,
    page_parts: Mapping[int, Any] | None,
) -> str:
    """Last pages of the Main Petition, where the prayer and date are printed."""
    if not page_markdown:
        return ""
    pages = [
        page
        for page in sorted(page_markdown)
        if MAIN_PETITION_PART in parts_on_page((page_parts or {}).get(page))
    ]
    chosen = pages[-_PETITION_CLOSING_PAGES:]
    return "\n".join((page_markdown.get(page) or "") for page in chosen)


def _listing_proforma_party_names(
    page_markdown: Mapping[int, str] | None,
    page_parts: Mapping[int, Any] | None,
) -> dict[str, list[tuple[int, str, int]]]:
    """Read numbered party names printed in Listing Proforma fields."""
    found: dict[str, list[tuple[int, str, int]]] = {
        "petitioners": [],
        "respondents": [],
    }
    for page in sorted(page_markdown or {}):
        if "Listing Proforma" not in parts_on_page((page_parts or {}).get(page)):
            continue
        text = re.sub(r"\s+", " ", str((page_markdown or {}).get(page) or ""))
        matches = list(_LISTING_PARTY_MARKER.finditer(text))
        for index, match in enumerate(matches):
            end = matches[index + 1].start() if index + 1 < len(matches) else len(text)
            value = text[match.end() : end]
            value = _LISTING_FIELD_AFTER_PARTY.split(value, maxsplit=1)[0]
            value = re.sub(
                r"\(\s*[a-z]\s*\)\s*$", "", value, flags=re.IGNORECASE
            )
            value = re.sub(r"\s+", " ", value).strip(" ,.;:-")
            if not value or len(value) > 180 or value.casefold() in _EMPTY_DATE:
                continue
            key = "petitioners" if match.group(1).casefold() == "petitioner" else "respondents"
            found[key].append((int(match.group(2)), value, page))
    return found


def _clean_numbered_party_name(value: str) -> str | None:
    """Return the printed name at the start of an explicit cause-title row."""
    text = re.sub(r"</?[^>]+>|[*_`#]", " ", value)
    text = re.sub(r"\s*\|\s*", " ", text)
    text = _PARTY_NAME_TAIL.sub("", text)
    text = _ROLE_TAIL.sub("", text)
    text = re.sub(r"\s+", " ", text).strip(" ,.;:-")
    if not text or len(text) > 180:
        return None
    if text.casefold() in _EMPTY_DATE or not re.search(r"[A-Za-z]", text):
        return None
    return text


def _numbered_parties_in_scope(
    text: str,
    *,
    start: int,
    end: int,
) -> list[tuple[int, str, int]]:
    found: list[tuple[int, str, int]] = []
    matches = [
        *_NUMBERED_PARTY_LINE.finditer(text, start, end),
        *_NUMBERED_PARTY_TABLE_CELL.finditer(text, start, end),
    ]
    for match in sorted(matches, key=lambda item: item.start()):
        serial = int(match.group(1))
        name = _clean_numbered_party_name(match.group(2))
        if name and not any(
            existing_serial == serial and _names_match(existing_name, name)
            for existing_serial, existing_name, _offset in found
        ):
            found.append((serial, name, match.start()))
    return found


def _main_petition_party_names(
    page_markdown: Mapping[int, str] | None,
    page_parts: Mapping[int, Any] | None,
) -> dict[str, list[tuple[int, str, int]]]:
    """Read explicit numbered parties from the Main Petition cause title.

    This deliberately requires a VERSUS separator and inspects only the cause-title
    scope. It therefore cannot reinterpret numbered pleading paragraphs or parties
    appearing in an annexure/lower-court record as filing parties.
    """
    found: dict[str, list[tuple[int, str, int]]] = {
        "petitioners": [],
        "respondents": [],
    }
    chunks: list[str] = []
    spans: list[tuple[int, int, int]] = []
    cursor = 0
    for page in sorted(page_markdown or {}):
        if MAIN_PETITION_PART not in parts_on_page((page_parts or {}).get(page)):
            continue
        text = str((page_markdown or {}).get(page) or "").replace("\r\n", "\n")
        if chunks:
            chunks.append("\n")
            cursor += 1
        start = cursor
        chunks.append(text)
        cursor += len(text)
        spans.append((start, cursor, page))
    text = "".join(chunks)
    versus = _VERSUS_LINE.search(text)
    if versus is None:
        return found

    def page_at(offset: int) -> int:
        for page_start, page_end, page in spans:
            if page_start <= offset < page_end:
                return page
        return spans[-1][2]

    between = _BETWEEN_LINE.search(text, 0, versus.start())
    petitioner_start = between.end() if between is not None else 0
    schedule_end = _PARTY_SCHEDULE_END.search(text, versus.end())
    if schedule_end is not None:
        respondent_end = schedule_end.start()
    else:
        respondent_end = next(
            page_end
            for page_start, page_end, _page in spans
            if page_start <= versus.start() < page_end
        )
    found["petitioners"].extend(
        (serial, name, page_at(offset))
        for serial, name, offset in _numbered_parties_in_scope(
            text, start=petitioner_start, end=versus.start()
        )
    )
    found["respondents"].extend(
        (serial, name, page_at(offset))
        for serial, name, offset in _numbered_parties_in_scope(
            text, start=versus.end(), end=respondent_end
        )
    )
    return found


def _supplement_parties_from_main_petition(
    payload: dict[str, Any],
    *,
    page_markdown: Mapping[int, str] | None,
    page_parts: Mapping[int, Any] | None,
) -> None:
    """Restore explicit numbered filing parties omitted by extraction."""
    recovered = _main_petition_party_names(page_markdown, page_parts)
    for key in ("petitioners", "respondents"):
        parties = payload.get(key)
        if not isinstance(parties, list):
            parties = []
            payload[key] = parties
        # Match returned names before applying positional fallbacks. This handles a
        # model that returns only party 2 (or another later serial) without a serial.
        for party in parties:
            if not isinstance(party, dict) or party.get("serial") is not None:
                continue
            name = _party_name(party)
            matched = next(
                (
                    serial
                    for serial, recovered_name, _page in recovered[key]
                    if _names_match(name, recovered_name)
                ),
                None,
            )
            if matched is not None:
                party["serial"] = matched
        for serial, name, page in recovered[key]:
            existing = next(
                (
                    party
                    for party in parties
                    if isinstance(party, dict)
                    and _names_match(_party_name(party), name)
                ),
                None,
            )
            if existing is not None:
                if existing.get("serial") is None:
                    existing["serial"] = serial
                continue
            same_serial = next(
                (
                    party
                    for party in parties
                    if isinstance(party, dict)
                    and str(party.get("serial") or "").isdigit()
                    and int(party["serial"]) == serial
                ),
                None,
            )
            if same_serial is not None:
                continue
            # Models commonly omit serial on returned parties while preserving list
            # order. Treat an occupied position as that serial instead of duplicating
            # it; explicit later serials are still appended.
            if 1 <= serial <= len(parties):
                positional = parties[serial - 1]
                if isinstance(positional, dict) and positional.get("serial") is None:
                    positional["serial"] = serial
                    continue
            parties.append(
                {
                    "serial": serial,
                    "name": name,
                    "kind": normalize_party_kind(name, None),
                    "source_part": MAIN_PETITION_PART,
                    "source_pages": [page],
                    "raw_text": name,
                }
            )
        parties.sort(
            key=lambda party: (
                int(party.get("serial"))
                if isinstance(party, dict) and str(party.get("serial") or "").isdigit()
                else 10_000
            )
        )


def _supplement_parties_from_listing_proforma(
    payload: dict[str, Any],
    *,
    page_markdown: Mapping[int, str] | None,
    page_parts: Mapping[int, Any] | None,
) -> None:
    """Restore party names/counts that scanned Main Petition OCR omitted."""
    recovered = _listing_proforma_party_names(page_markdown, page_parts)
    for key in ("petitioners", "respondents"):
        parties = payload.get(key)
        if not isinstance(parties, list):
            parties = []
            payload[key] = parties
        for serial, name, page in recovered[key]:
            existing = next(
                (
                    party
                    for party in parties
                    if isinstance(party, dict)
                    and _names_match(_party_name(party), name)
                ),
                None,
            )
            if existing is not None:
                if existing.get("serial") is None:
                    existing["serial"] = serial
                continue
            parties.append(
                {
                    "serial": serial,
                    "name": name,
                    "kind": normalize_party_kind(name, None),
                    "source_part": "Listing Proforma",
                    "source_pages": [page],
                    "raw_text": name,
                }
            )
        parties.sort(
            key=lambda party: (
                int(party.get("serial"))
                if isinstance(party, dict) and str(party.get("serial") or "").isdigit()
                else 10_000
            )
        )


def _coram_judges(
    page_markdown: Mapping[int, str] | None,
    page_parts: Mapping[int, Any] | None,
) -> list[str]:
    judges: list[str] = []
    for page in sorted(page_markdown or {}):
        if "Impugned Order" not in parts_on_page((page_parts or {}).get(page)):
            continue
        text = str((page_markdown or {}).get(page) or "")
        if "coram" not in text.casefold():
            continue
        for match in _CORAM_JUDGE.finditer(text):
            judge = re.sub(r"\s+", " ", match.group(0)).strip(" |*_.,;:-")
            key = judge.casefold()
            if judge and all(existing.casefold() != key for existing in judges):
                judges.append(judge)
    return judges


def _correct_impugned_order_bench(
    payload: dict[str, Any],
    *,
    page_markdown: Mapping[int, str] | None,
    page_parts: Mapping[int, Any] | None,
) -> None:
    judges = _coram_judges(page_markdown, page_parts)
    if not judges:
        return
    orders = payload.get("impugned_orders")
    if not isinstance(orders, list) or not orders:
        return
    primary = next((order for order in orders if isinstance(order, dict)), None)
    if primary is not None:
        primary["bench"] = "; ".join(judges)


def strip_anr_ors_suffix(text: str | None) -> str:
    """Drop trailing And Anr / And Ors / [And ors.] already printed on the Cover Page."""
    cleaned = strip_party_role_label(text)
    while True:
        updated = _ANR_ORS_TAIL.sub("", cleaned).rstrip(" .,&")
        if updated == cleaned:
            break
        cleaned = updated
    return cleaned


def strip_formatted_title_brackets(text: str | None) -> str | None:
    """Keep 'and Anr.' / 'and Ors.' as plain text, never [And ors.]."""
    if not isinstance(text, str) or not text.strip():
        return None
    cleaned = _BRACKETED_ANR_ORS.sub(r" \1", text)
    cleaned = re.sub(r"\s+", " ", cleaned).strip(" ,")
    return cleaned or None


def format_side_title(main_name: str | None, count: int) -> str | None:
    """Build one side of formatted_title: Name / Name and Anr. / Name and Ors."""
    name = strip_anr_ors_suffix(main_name)
    if not name:
        return None
    if count <= 1:
        return name
    if count == 2:
        return f"{name} and Anr."
    return f"{name} and Ors."


def build_formatted_title(
    main_petitioner: str | None,
    petitioner_count: int,
    main_respondent: str | None,
    respondent_count: int,
) -> str | None:
    left = format_side_title(main_petitioner, petitioner_count)
    right = format_side_title(main_respondent, respondent_count)
    if left and right:
        return f"{left} VS {right}"
    return left or right


def _name_tokens(name: str) -> str:
    return _NAME_SPLIT.sub(" ", name.strip().lower()).strip()


def is_organization_name(name: str | None) -> bool:
    text = (name or "").strip()
    if not text:
        return False
    original_lower = text.lower()
    if original_lower == "the" or original_lower.startswith("the "):
        return True
    for prefix in _ORG_PREFIXES:
        if original_lower == prefix:
            return True
        if original_lower.startswith(prefix) and (
            len(original_lower) == len(prefix)
            or not original_lower[len(prefix)].isalnum()
        ):
            return True
    compact = re.sub(r"\s+", " ", original_lower).rstrip(" .")
    for suffix in _ORG_SUFFIXES:
        needle = suffix.rstrip(".")
        if compact == needle or compact.endswith(" " + needle):
            return True
    return False


def normalize_party_kind(name: str | None, kind: str | None) -> str | None:
    if is_organization_name(name):
        return "ORGANIZATION"
    if isinstance(kind, str):
        text = kind.strip().upper()
        if text in {"ORGANISATION", "ORG"}:
            return "ORGANIZATION"
        if text in {"INDIVIDUAL", "ORGANIZATION"}:
            return text
    if (name or "").strip():
        return "INDIVIDUAL"
    return None


def _party_name(party: Any) -> str | None:
    data = _as_dict(party) if not isinstance(party, dict) else party
    name = data.get("name")
    if isinstance(name, str) and name.strip():
        return name.strip()
    return None


def _edit_distance_at_most(left: str, right: str, limit: int = 2) -> bool:
    if abs(len(left) - len(right)) > limit:
        return False
    previous = list(range(len(right) + 1))
    for index, char in enumerate(left, 1):
        current = [index]
        for other_index, other in enumerate(right, 1):
            current.append(
                min(
                    previous[other_index] + 1,
                    current[other_index - 1] + 1,
                    previous[other_index - 1] + (char != other),
                )
            )
        if min(current) > limit:
            return False
        previous = current
    return previous[-1] <= limit


_HONORIFIC_TOKEN = re.compile(
    r"^(?:shri|sri|smt|sh|mr|mrs|ms|dr|km|kumari|master)\.?$",
    re.IGNORECASE,
)


def _given_name_tokens(name: str | None) -> list[str]:
    """Drop Shri / Mr / Smt so two brothers are not the same person."""
    tokens: list[str] = []
    for token in (name or "").split():
        bare = token.strip(".,")
        if not bare or _HONORIFIC_TOKEN.fullmatch(bare):
            continue
        tokens.append(bare)
    return tokens


def _same_party_person(left: Mapping[str, Any], right: Mapping[str, Any]) -> bool:
    """Rajiv Lakhanpal S/o Naranjan Dass and Rajeev Lakhanpal are one person."""
    if _names_match(_party_name(left), _party_name(right)):
        return True
    left_guard = re.sub(r"\s+", " ", str(left.get("guardian") or "")).strip()
    right_guard = re.sub(r"\s+", " ", str(right.get("guardian") or "")).strip()
    if not left_guard or not right_guard or not _names_match(left_guard, right_guard):
        return False
    left_tokens = _given_name_tokens(_party_name(left))
    right_tokens = _given_name_tokens(_party_name(right))
    if len(left_tokens) < 2 or len(right_tokens) < 2:
        return False
    if left_tokens[-1].casefold() != right_tokens[-1].casefold():
        return False
    return _edit_distance_at_most(left_tokens[0].casefold(), right_tokens[0].casefold())


def _clean_party_address(address: str | None) -> str | None:
    if not address:
        return None
    text = address.replace("&x26;", "&").replace("&amp;", "&")
    # Clean trailing petition/appeal boilerplate and the next column's role.
    text = re.sub(
        r"\s+(?:THE\s+HON['’]?BLE|HON['’]?BLE\s+THE|THE\s+CONSTITUTION\s+OF\s+INDIA|SPECIAL\s+LEAVE\s+PETITION|THE\s+HUMBLE\s+(?:APPEAL|PETITION)|THE\s+APPEAL\s+OF|THE\s+PETITION\s+OF|MOST\s+RESPECTFULLY\s+SHOWETH|WRIT\s+PETITION|CRIMINAL\s+APPEAL|DEFECTO|DEFACTO|DE\s+FACTO)\b.*",
        "",
        text,
        flags=re.IGNORECASE,
    )
    # Clean trailing role tags like ... P R, ... R P, .... Petitioner No. 1, ... Respondent No. 1
    text = re.sub(
        r"(?:\.\.\.|\b)(?:petitioners?|respondents?|appellants?|accused)(?:\s*(?:no|number)\.?)?\s*[.\s]*\d*.*$",
        "",
        text,
        flags=re.IGNORECASE,
    )
    text = re.sub(r"\s*(?:\.\.\.|\b)[PR]\s*[RP]?\s*$", "", text, flags=re.IGNORECASE)
    text = re.sub(r"\s*\\{2,}.*$", "", text)
    text = re.sub(r"\s*\bN\.A\.\s*\.{2,}\s*$", "", text, flags=re.IGNORECASE)
    text = re.sub(r"\s+", " ", text).strip(" ,.;")
    text = _strip_address_labels(text)
    return text or None


_ADDRESS_PLACE = re.compile(
    r"\b(?:police\s+station|p\.?\s*s\.?|r/o|resident(?:\s+of)?|residing(?:\s+at)?|"
    r"registered\s+address|gut\s+no|plot(?:\s+no)?|village|dist(?:rict)?\.?)\b",
    re.IGNORECASE,
)
_THROUGH_LABEL = re.compile(
    r"^(?:through|represented\s+by|rep\.?\s+by)$",
    re.IGNORECASE,
)
_THROUGH_BODY = re.compile(
    r"^(?:through|represented\s+by|rep\.?\s+by)\s+(.+)$",
    re.IGNORECASE,
)
_ADDRESS_CONTINUATION = re.compile(
    r"^(?:presently\s+at|second\s+address|also\s+at)\b",
    re.IGNORECASE,
)
_RELATION_MARK = re.compile(
    r"\b([SDW]/o\.?)\s+(.+)$",
    re.IGNORECASE,
)
_PLACE_MARK = re.compile(
    r"\b(?:r/o\.?|resident(?:\s+of)?|residing(?:\s+at)?|presently\s+at|second\s+address|"
    r"also\s+at|south|north|east|west|gut\s+no|plot|village|vpo)\b",
    re.IGNORECASE,
)
_OFFICE_PLACE_SEGMENT = re.compile(
    r"(?:"
    r"^(?:"
    r"old\s+secretariat\b"
    r"|world\s+trade\s+centre\b"
    r"|town\s+hall\b"
    r"|new\s+delhi\b"
    r"|delhi\b"
    r"|[\w.'’/-]+(?:\s+[\w.'’/-]+){0,5}\s+"
    r"(?:road|chowk|street|lane|marg|nagar|colony|bhawan|vihar)\b"
    r")"
    r"|police\s+station\b"
    r"|district\b"
    r"|tamil\s*nadu\b"
    r")",
    re.IGNORECASE,
)
_NAME_WRAP_TAIL = re.compile(
    r"^((?:of|and)\s+\S+(?:\s+\S+){0,2})(?:,\s*(.*))?$",
    re.IGNORECASE,
)
_PLACE_WORD_CONTINUATION = re.compile(
    r"^(?:chowk|road|street|lane|marg)\b",
    re.IGNORECASE,
)
_SCHEDULE_ROLE_TOKEN = (
    r"(?:not\s+a\s+party(?:\s+proforma)?|proforma(?:\s+respondents?)?|dead|"
    r"(?:appellants?|petitioners?|respondents?)\s*(?:no\.?|number)?\s*[\d/]*)"
)
_ROLE_NUMBER_LINE = re.compile(r"^(?:no\.?|number)\s*[\d/]+$", re.IGNORECASE)
_SCHEDULE_ROLE_LINE = re.compile(
    rf"^(?:{_SCHEDULE_ROLE_TOKEN})(?:\s+{_SCHEDULE_ROLE_TOKEN})*$",
    re.IGNORECASE,
)
_BARE_LIST_NUMBER = re.compile(r"^[(\[]?\s*\d+\s*[.)\]]$")


def _normalize_relation(token: str) -> str:
    return f"{token[0].upper()}/o"


def _split_office_place(name: str) -> tuple[str, str | None]:
    """An office respondent's place starts at Old secretariat, Town Hall, or a road."""
    parts = [part.strip(" ,.") for part in name.split(",") if part.strip(" ,.")]
    for index, part in enumerate(parts):
        if index == 0 or not _OFFICE_PLACE_SEGMENT.search(part):
            continue
        kept = ", ".join(parts[:index]).strip(" ,.")
        place = ", ".join(parts[index:]).strip(" ,.")
        if kept and place:
            return kept, place
    return name, None


def _split_relation_tail(rest: str) -> tuple[str, str | None]:
    """The S/o name stops at R/o, Presently at, or Second Address."""
    place = _PLACE_MARK.search(rest)
    if not place:
        guardian = rest.strip(" ,.")
        return guardian, None
    guardian = rest[: place.start()].strip(" ,.")
    return guardian, rest[place.start() :].strip(" ,.")


def _is_schedule_role_line(line: str) -> bool:
    text = line.strip()
    return (
        _SCHEDULE_ROLE_LINE.fullmatch(text) is not None
        or _ROLE_NUMBER_LINE.fullmatch(text) is not None
    )


def _join_open_parentheses(lines: list[str]) -> list[str]:
    """A name wrapped inside parentheses stays one party, not an address."""
    joined: list[str] = []
    buffer = ""
    for line in lines:
        if buffer:
            buffer = f"{buffer} {line}"
            if buffer.count("(") <= buffer.count(")"):
                joined.append(buffer)
                buffer = ""
            continue
        if line.count("(") > line.count(")"):
            buffer = line
            continue
        joined.append(line)
    if buffer:
        joined.append(buffer)
    return joined


def _join_through_tails(lines: list[str]) -> list[str]:
    """'through his' / 'through' at the end of a line belongs with 'LR's'."""
    joined: list[str] = []
    index = 0
    while index < len(lines):
        line = lines[index]
        nxt = lines[index + 1] if index + 1 < len(lines) else ""
        if (
            nxt
            and re.search(r"\bthrough(?:\s+(?:his|her|its))?$", line, re.IGNORECASE)
            and len(nxt.split()) <= 4
            and not re.match(r"^\d", nxt)
            and not _is_schedule_role_line(nxt)
        ):
            joined.append(f"{line} {nxt}")
            index += 2
            continue
        joined.append(line)
        index += 1
    return joined


def _join_bare_list_numbers(lines: list[str]) -> list[str]:
    """'1.' on its own line belongs to the name printed underneath."""
    joined: list[str] = []
    index = 0
    while index < len(lines):
        line = lines[index]
        if _BARE_LIST_NUMBER.fullmatch(line) and index + 1 < len(lines):
            joined.append(f"{line} {lines[index + 1]}")
            index += 2
            continue
        joined.append(line)
        index += 1
    return joined


def _split_through_and_place(text: str) -> tuple[str, str]:
    """'SHO, P.S- Motihari' is the representative and the police-station place."""
    cleaned = re.sub(
        r"\s+(?:petitioners?|respondents?|applicants?|accused)\b.*$",
        "",
        text,
        flags=re.IGNORECASE,
    ).strip(" ,.")
    place = _ADDRESS_PLACE.search(cleaned)
    if place and place.start() > 0:
        return cleaned[: place.start()].strip(" ,."), cleaned[place.start() :].strip(" ,.")
    return cleaned, ""


def _strip_address_labels(text: str) -> str:
    """Keep the place. R/o and Through name the place and the representative."""
    lead = re.match(
        r"^(?:through|represented\s+by|rep\.?\s+by)\s+(.+)$",
        text,
        re.IGNORECASE,
    )
    if lead:
        rest = lead.group(1).strip()
        place = _ADDRESS_PLACE.search(rest)
        text = rest[place.start():].strip() if place else ""
    text = re.sub(
        r"\s+\b(?:through|represented\s+by)\s+.+$",
        "",
        text,
        flags=re.IGNORECASE,
    )
    text = re.sub(
        r"^(?:r/o|resident(?:\s+of)?|residing(?:\s+at)?|registered\s+address(?:\s+at)?)\s*[-:]?\s*",
        "",
        text,
        flags=re.IGNORECASE,
    )
    return re.sub(r"\s+", " ", text).strip(" ,.;")


def _party_richness(party: Mapping[str, Any]) -> int:
    score = 0
    if re.match(r"^[SDW]/o$", str(party.get("relation") or ""), re.IGNORECASE):
        score += 2
    if re.search(r"\b(?:r/o|road|house|place|delhi)\b", str(party.get("address") or ""), re.IGNORECASE):
        score += 2
    pages = party.get("source_pages") or []
    if isinstance(pages, list) and pages and min(pages) <= 30:
        score += 1
    return score


def _collapse_same_parties(parties: list[dict[str, Any]]) -> list[dict[str, Any]]:
    kept: list[dict[str, Any]] = []
    for party in parties:
        match = next((item for item in kept if _same_party_person(item, party)), None)
        if match is None:
            kept.append(party)
            continue
        preferred = party if _party_richness(party) > _party_richness(match) else match
        other = match if preferred is party else party
        for field in (
            "address",
            "relation",
            "guardian",
            "occupation",
            "age",
            "acting_through",
            "mobile",
            "email",
        ):
            if not preferred.get(field) and other.get(field):
                preferred[field] = other[field]
        if preferred is party:
            kept[kept.index(match)] = party
    for index, party in enumerate(kept, 1):
        party["serial"] = index
    return kept


def _names_match(left: str | None, right: str | None) -> bool:
    a = _name_tokens(left or "")
    b = _name_tokens(right or "")
    if a and b and (a == b or a in b or b in a):
        return True
    return names_are_spelling_variants(left, right)


def strip_party_role_label(text: str | None) -> str:
    """Drop trailing Petitioner/Respondent caption marks, with or without dots."""
    cleaned = re.sub(r"\s+", " ", (text or "").replace("…", "...")).strip()
    while True:
        updated = _ROLE_TAIL.sub("", cleaned).rstrip(" .")
        if updated == cleaned:
            break
        cleaned = updated
    return cleaned


def _inconsistency_compare_texts(raw_text: str | None) -> list[str]:
    blob = (raw_text or "").strip()
    if not blob:
        return []
    quoted = _QUOTED_TEXT.findall(blob)
    if len(quoted) < 2:
        quoted = _QUOTED_TEXT_SINGLE.findall(blob)
    if len(quoted) >= 2:
        return quoted
    parts = [bit.strip() for bit in _VS_SPLIT.split(blob) if bit.strip()]
    if len(parts) < 2:
        return []
    sides: list[str] = []
    for part in parts:
        if ":" in part:
            part = part.split(":", 1)[1].strip()
        if part:
            sides.append(part)
    return sides


def is_party_role_label_mismatch(raw_text: str | None) -> bool:
    """True when quoted sides differ only by caption labels, Anr/Ors, or letter case."""
    return is_cosmetic_name_mismatch(raw_text)


def normalize_compared_name(text: str | None) -> str:
    """Name used to decide if two inconsistency quotes are the same person."""
    cleaned = _LIST_SERIAL.sub("", strip_anr_ors_suffix(text)).strip()
    return re.sub(r"\s+", " ", cleaned.casefold()).strip()


def _core_person_name(text: str | None) -> str:
    cleaned = _TITLE_TOKEN.sub(" ", normalize_compared_name(text))
    cleaned = re.sub(r"\([^)]*\)", " ", cleaned)
    return re.sub(r"\s+", " ", cleaned).strip()


def names_are_spelling_variants(left: str | None, right: str | None) -> bool:
    """True when two strings look like the same person with a spelling slip."""
    a = _core_person_name(left)
    b = _core_person_name(right)
    if not a or not b:
        return False
    if a == b or a in b or b in a:
        return True
    tokens_a = a.split()
    tokens_b = b.split()
    last_a = tokens_a[-1] if tokens_a else ""
    last_b = tokens_b[-1] if tokens_b else ""
    given_a = " ".join(tokens_a[:-1]) if len(tokens_a) > 1 else a
    given_b = " ".join(tokens_b[:-1]) if len(tokens_b) > 1 else b
    if last_a and last_b and last_a == last_b:
        return SequenceMatcher(None, given_a, given_b).ratio() >= 0.75
    return SequenceMatcher(None, a, b).ratio() >= 0.85


def is_extra_party_caption_mismatch(
    raw_text: str | None, label: str | None = None
) -> bool:
    """True when Cover Page 'And Anr/Ors' is compared to a later-listed party."""
    sides = _inconsistency_compare_texts(raw_text)
    blob = raw_text or ""
    has_shorthand = any(_ANR_ORS_MARK.search(side) for side in sides)
    caption_context = bool(_CAPTION_SOURCE.search(blob))
    extra_label = bool(_EXTRA_PARTY_LABEL.search(label or ""))
    cores = [name for name in (_core_person_name(side) for side in sides) if name]
    distinct = list(dict.fromkeys(cores))
    if len(distinct) >= 2:
        base = distinct[0]
        if all(names_are_spelling_variants(base, other) for other in distinct[1:]):
            return False
    if extra_label and (has_shorthand or caption_context):
        return True
    if len(sides) < 2:
        return False
    if not has_shorthand and not caption_context:
        return False
    return len(distinct) >= 2


def is_cosmetic_name_mismatch(raw_text: str | None) -> bool:
    """True when compared names match after ignoring case, role labels, and Anr/Ors."""
    sides = _inconsistency_compare_texts(raw_text)
    if len(sides) < 2:
        return False
    normalized = [normalize_compared_name(side) for side in sides]
    if not all(normalized):
        return False
    return len(set(normalized)) == 1


def _inconsistency_name_key(raw_text: str | None) -> frozenset[str] | None:
    names = {
        normalize_compared_name(side)
        for side in _inconsistency_compare_texts(raw_text)
    }
    names.discard("")
    if len(names) < 2:
        return None
    return frozenset(names)


def _clean_quoted_party_name(text: str | None) -> str:
    cleaned = strip_anr_ors_suffix(strip_party_role_label(text))
    cleaned = _LIST_SERIAL.sub("", cleaned).strip()
    return re.sub(r"\s+", " ", cleaned).strip(" ,;")


def format_party_spelling_raw_text(cover: str, listed: str) -> str:
    return f'Cause Title: "{cover}"; Main Petition: "{listed}"'


def _looks_like_person_name(text: str | None) -> bool:
    core = _core_person_name(text)
    if not core or re.search(r"\b(?:court|petition|affidavit|vakalatnama)\b", core):
        return False
    return 1 <= len(core.split()) <= 6


def _is_party_name_spelling_item(raw_text: str | None, label: str | None) -> bool:
    label_l = (label or "").lower()
    if "acting through" in label_l:
        return False
    sides = _inconsistency_compare_texts(raw_text)
    if len(sides) < 2:
        return False
    if not all(_looks_like_person_name(side) for side in (sides[0], sides[-1])):
        return False
    if names_are_spelling_variants(sides[0], sides[-1]):
        return True
    return bool(re.search(r"\b(?:petitioner|respondent|party)\b", label_l))


def _party_spelling_label(
    cover: str,
    listed: str,
    payload: dict[str, Any],
    current: str | None,
) -> str:
    cause = payload.get("cause_title")
    cause_data = cause if isinstance(cause, dict) else {}
    main_respondent = cause_data.get("main_respondent")
    main_petitioner = cause_data.get("main_petitioner")
    if isinstance(main_respondent, str) and (
        names_are_spelling_variants(cover, main_respondent)
        or names_are_spelling_variants(listed, main_respondent)
    ):
        return "Main respondent spelling"
    if isinstance(main_petitioner, str) and (
        names_are_spelling_variants(cover, main_petitioner)
        or names_are_spelling_variants(listed, main_petitioner)
    ):
        return "Main petitioner spelling"
    label_l = (current or "").lower()
    if "respondent" in label_l:
        return "Main respondent spelling"
    if "petitioner" in label_l:
        return "Main petitioner spelling"
    return current or "Name spelling"


def _cause_title_mismatch_label(
    cover: str,
    listed: str,
    payload: dict[str, Any],
    current: str | None,
) -> str:
    cause = payload.get("cause_title")
    cause_data = cause if isinstance(cause, dict) else {}
    main_respondent = cause_data.get("main_respondent")
    main_petitioner = cause_data.get("main_petitioner")
    if isinstance(main_respondent, str) and (
        _names_match(cover, main_respondent) or _names_match(listed, main_respondent)
    ):
        return "Cause title respondent mismatch"
    if isinstance(main_petitioner, str) and (
        _names_match(cover, main_petitioner) or _names_match(listed, main_petitioner)
    ):
        return "Cause title petitioner mismatch"
    label_l = (current or "").lower()
    if "respondent" in label_l:
        return "Cause title respondent mismatch"
    if "petitioner" in label_l:
        return "Cause title petitioner mismatch"
    return current or "Cause title party mismatch"


def _rewrite_party_spelling_item(item: dict[str, Any], payload: dict[str, Any]) -> None:
    raw = item.get("raw_text") or item.get("detail")
    raw_text = raw if isinstance(raw, str) else None
    label = item.get("label") if isinstance(item.get("label"), str) else None
    if not _is_party_name_spelling_item(raw_text, label):
        return
    sides = _inconsistency_compare_texts(raw_text)
    cover = _clean_quoted_party_name(sides[0])
    listed = _clean_quoted_party_name(sides[-1])
    if not cover or not listed:
        return
    item["raw_text"] = format_party_spelling_raw_text(cover, listed)
    label_l = (label or "").lower()
    if "listed as" in label_l or "missing" in label_l:
        return
    if names_are_spelling_variants(cover, listed):
        item["label"] = _party_spelling_label(cover, listed, payload, label)
        return
    item["label"] = _cause_title_mismatch_label(cover, listed, payload, label)


def _first_listed_party(parties: Any) -> dict[str, Any] | None:
    if not isinstance(parties, list):
        return None
    for party in parties:
        if isinstance(party, dict) and _party_name(party):
            return party
    return None


def _find_party_by_name(parties: Any, name: str) -> dict[str, Any] | None:
    if not isinstance(parties, list):
        return None
    for party in parties:
        if isinstance(party, dict) and _names_match(_party_name(party), name):
            return party
    return None


def _ensure_inconsistencies_blob(payload: dict[str, Any]) -> dict[str, Any]:
    blob = payload.get("inconsistencies")
    if not isinstance(blob, dict):
        blob = {"items": [], "source_part": [], "source_pages": []}
        payload["inconsistencies"] = blob
    if not isinstance(blob.get("items"), list):
        blob["items"] = []
    return blob


_CAUSE_TITLE_CHECKS = (
    {
        "parties": "petitioners",
        "opposite": "respondents",
        "main": "main_petitioner",
        "spelling": "Main petitioner spelling",
        "mismatch": "Cause title petitioner mismatch",
        "swap": "Cause title petitioner listed as respondent",
        "missing": "Cause title petitioner missing on Main Petition",
        "side": "petitioner",
        "other": "respondent",
    },
    {
        "parties": "respondents",
        "opposite": "petitioners",
        "main": "main_respondent",
        "spelling": "Main respondent spelling",
        "mismatch": "Cause title respondent mismatch",
        "swap": "Cause title respondent listed as petitioner",
        "missing": "Cause title respondent missing on Main Petition",
        "side": "respondent",
        "other": "petitioner",
    },
)


def _append_main_party_spelling(payload: dict[str, Any]) -> None:
    cause = payload.get("cause_title")
    if not isinstance(cause, dict):
        return
    pending: list[dict[str, str]] = []
    blob = payload.get("inconsistencies")
    items = blob.get("items") if isinstance(blob, dict) else []
    if not isinstance(items, list):
        items = []
    existing = {
        key
        for item in items
        for key in (
            _inconsistency_name_key(
                (item.get("raw_text") or item.get("detail"))
                if isinstance(item, dict)
                else None
            ),
        )
        if key is not None
    }
    existing_labels: set[tuple[str, str]] = set()
    for spec in _CAUSE_TITLE_CHECKS:
        cover = cause.get(spec["main"])
        if not isinstance(cover, str) or not cover.strip():
            continue
        cover_clean = _clean_quoted_party_name(cover)
        if not cover_clean:
            continue
        listed = _party_name(_first_listed_party(payload.get(spec["parties"])))
        side = spec["side"]
        if not listed:
            marker = (spec["missing"], cover_clean)
            if marker not in existing_labels:
                pending.append(
                    {
                        "label": spec["missing"],
                        "raw_text": (
                            f'Cause Title {side}: "{cover_clean}"; '
                            f"not found as {side} on Main Petition starting pages."
                        ),
                    }
                )
                existing_labels.add(marker)
            continue
        listed_clean = _clean_quoted_party_name(listed)
        if not listed_clean:
            continue
        cover_n = normalize_compared_name(cover_clean)
        listed_n = normalize_compared_name(listed_clean)
        if cover_n == listed_n:
            continue
        key = frozenset({cover_n, listed_n})
        if key in existing:
            continue
        compare = format_party_spelling_raw_text(
            cover_clean, listed_clean
        )
        if names_are_spelling_variants(cover_clean, listed_clean):
            pending.append({"label": spec["spelling"], "raw_text": compare})
            existing.add(key)
            continue
        swapped = _find_party_by_name(
            payload.get(spec["opposite"]), cover_clean
        )
        if swapped is not None:
            marker = (spec["swap"], cover_clean)
            other = spec["other"]
            if marker not in existing_labels:
                pending.append(
                    {
                        "label": spec["swap"],
                        "raw_text": (
                            f'Cause Title {side}: "{cover_clean}"; '
                            f"Main Petition starting pages list this "
                            f"person as {other}."
                        ),
                    }
                )
                existing_labels.add(marker)
            continue
        pending.append({"label": spec["mismatch"], "raw_text": compare})
        existing.add(key)
    if not pending:
        return
    blob = _ensure_inconsistencies_blob(payload)
    items = blob["items"]
    for item in pending:
        items.append(
            {
                "id": _next_inconsistency_id(items),
                **item,
            }
        )
    part_list = blob.get("source_part")
    if not isinstance(part_list, list):
        part_list = []
        blob["source_part"] = part_list
    for part in ("Cover Page", "Main Petition"):
        if part not in part_list:
            part_list.append(part)


def _drop_role_label_inconsistencies(payload: dict[str, Any]) -> None:
    blob = payload.get("inconsistencies")
    if not isinstance(blob, dict):
        return
    items = blob.get("items")
    if not isinstance(items, list):
        return
    kept: list[Any] = []
    seen_names: dict[frozenset[str], int] = {}
    for item in items:
        data = item if isinstance(item, dict) else _as_dict(item)
        raw = data.get("raw_text") or data.get("detail")
        raw_text = raw if isinstance(raw, str) else None
        if is_cosmetic_name_mismatch(raw_text):
            continue
        if is_extra_party_caption_mismatch(
            raw_text, data.get("label") if isinstance(data.get("label"), str) else None
        ):
            continue
        key = _inconsistency_name_key(raw_text)
        if key is not None and key in seen_names:
            existing = kept[seen_names[key]]
            existing_raw = ""
            if isinstance(existing, dict):
                existing_raw = str(existing.get("raw_text") or existing.get("detail") or "")
            if raw_text and (not existing_raw or len(raw_text) < len(existing_raw)):
                kept[seen_names[key]] = item
            continue
        if key is not None:
            seen_names[key] = len(kept)
        kept.append(item)
    for item in kept:
        if isinstance(item, dict):
            _rewrite_party_spelling_item(item, payload)
    for index, item in enumerate(kept, start=1):
        if isinstance(item, dict):
            item["id"] = str(index)
    blob["items"] = kept
    if not kept:
        blob["source_part"] = []
        blob["source_pages"] = []


def _next_inconsistency_id(items: list[Any]) -> str:
    used: set[int] = set()
    for item in items:
        data = _as_dict(item) if not isinstance(item, dict) else item
        raw = str(data.get("id") or "").strip()
        if raw.isdigit():
            used.add(int(raw))
    next_id = 1
    while next_id in used:
        next_id += 1
    return str(next_id)


_AOR_PHONE = re.compile(
    r"\b(?:ph|phone|tel|telephone|mob|mobile)\.?\s*[:.]?\s*"
    r"(\+?\d[\d\s()./-]{6,}\d)",
    re.IGNORECASE,
)


def _printed_phone(text: str) -> str | None:
    """The number after Ph., Tel., or Mobile. A PIN or chamber number is not a phone."""
    match = _AOR_PHONE.search(text or "")
    if not match:
        return None
    number = re.sub(r"\s+", " ", match.group(1)).strip(" .,-")
    digits = re.sub(r"\D", "", number)
    if not 8 <= len(digits) <= 15:
        return None
    return number


def _phone_after_chamber(address: str, pages: Mapping[int, str] | None) -> str | None:
    """The phone printed on the same chamber line when the address field omitted it."""
    place = re.sub(r"^\s*chamber\s*:\s*", "", address or "", flags=re.IGNORECASE)
    place = re.sub(r"\s+", " ", place).strip(" ,;")
    if len(place) < 12 or not pages:
        return None
    needle = place[:48].casefold()
    for text in pages.values():
        folded = re.sub(r"\s+", " ", text or "")
        at = folded.casefold().find(needle)
        if at < 0:
            continue
        phone = _printed_phone(folded[at : at + len(place) + 48])
        if phone:
            return phone
    return None


def _fill_aor_phones(
    payload: dict[str, Any],
    page_markdown: Mapping[int, str] | None = None,
) -> None:
    """Put Ph. / Tel. / Mobile from the chamber line into the advocate phone field."""
    for aor in payload.get("advocates_on_record") or []:
        if not isinstance(aor, dict):
            continue
        address = aor.get("office_address") if isinstance(aor.get("office_address"), str) else ""
        raw = aor.get("raw_text") if isinstance(aor.get("raw_text"), str) else ""
        phone = _printed_phone(address) or _printed_phone(raw)
        if phone and address:
            cleaned = _AOR_PHONE.sub(" ", address)
            cleaned = re.sub(r"\s+", " ", cleaned).strip(" ,;")
            aor["office_address"] = cleaned or None
        if str(aor.get("mobile") or "").strip():
            continue
        if not phone:
            phone = _phone_after_chamber(address, page_markdown)
        if phone:
            aor["mobile"] = phone


def _normalize_legacy_keys(payload: dict[str, Any]) -> None:
    if not payload.get("relief_sort"):
        relief = payload.get("relief")
        sought = None
        if isinstance(relief, dict):
            sought = relief.get("sought")
        elif isinstance(relief, str):
            sought = relief
        if isinstance(sought, str) and sought.strip():
            payload["relief_sort"] = sought.strip()
    if "relief_sort" in payload:
        payload["relief_sort"] = clean_relief_sort(
            payload.get("relief_sort") if isinstance(payload.get("relief_sort"), str) else None
        )
    if "petition_date" in payload:
        raw_date = payload.get("petition_date")
        payload["petition_date"] = clean_petition_date(
            raw_date if isinstance(raw_date, str) else None
        )
    for aor in payload.get("advocates_on_record") or []:
        if isinstance(aor, dict) and not aor.get("office_address") and aor.get("office"):
            aor["office_address"] = aor.get("office")
    _fill_aor_phones(payload)
    for order in payload.get("impugned_orders") or []:
        if not isinstance(order, dict):
            continue
        if not order.get("Forum") and not order.get("forum") and order.get("court_name"):
            order["Forum"] = order.get("court_name")
    blob = payload.get("inconsistencies")
    if not isinstance(blob, dict):
        return
    for item in blob.get("items") or []:
        if isinstance(item, dict) and not item.get("raw_text") and item.get("detail"):
            item["raw_text"] = item.get("detail")
            item.pop("detail", None)


def _normalize_parties(payload: dict[str, Any]) -> None:
    cause = payload.get("cause_title")
    cause_data = cause if isinstance(cause, dict) else _as_dict(cause)
    main_petitioner = (
        cause_data.get("main_petitioner") if isinstance(cause_data, dict) else None
    )
    main_respondent = (
        cause_data.get("main_respondent") if isinstance(cause_data, dict) else None
    )
    if not isinstance(main_petitioner, str):
        main_petitioner = None
    else:
        main_petitioner = strip_anr_ors_suffix(main_petitioner) or None
    if not isinstance(main_respondent, str):
        main_respondent = None
    else:
        main_respondent = strip_anr_ors_suffix(main_respondent) or None
    if isinstance(cause, dict):
        if main_petitioner:
            cause["main_petitioner"] = main_petitioner
        if main_respondent:
            cause["main_respondent"] = main_respondent

    for key, main_name in (
        ("petitioners", main_petitioner),
        ("respondents", main_respondent),
    ):
        parties = payload.get(key)
        if not isinstance(parties, list):
            continue
        for index, party in enumerate(parties):
            if not isinstance(party, dict):
                continue
            name = _party_name(party)
            party["kind"] = normalize_party_kind(name, party.get("kind"))
            if main_name:
                party["is_primary"] = _names_match(name, main_name)
            elif party.get("is_primary") is None:
                party["is_primary"] = index == 0
            addr = party.get("address")
            if isinstance(addr, str) and key == "respondents":
                if re.search(
                    r"\b(?:v\s*a\s*k\s*a\s*l\s*a\s*t\s*n\s*a\s*m\s*a|solemnly\s+affirm|aged\s+about\s+\d+\s+years\s+[SDW]/o)\b",
                    addr,
                    re.IGNORECASE,
                ):
                    party["address"] = None

    if isinstance(cause, dict):
        formatted = build_formatted_title(
            main_petitioner,
            len(payload.get("petitioners") or []),
            main_respondent,
            len(payload.get("respondents") or []),
        )
        if formatted:
            cause["formatted_title"] = formatted
        elif cause.get("formatted_title"):
            stripped = strip_formatted_title_brackets(str(cause.get("formatted_title")))
            if stripped:
                cause["formatted_title"] = stripped


def _drop_resolved_acting_through_notes(payload: dict[str, Any]) -> None:
    """Drop a 'no acting_through' note once that party has a representative."""
    blob = payload.get("inconsistencies")
    if not isinstance(blob, dict) or not isinstance(blob.get("items"), list):
        return
    filled = {
        _party_name(party).casefold()
        for key in ("petitioners", "respondents")
        for party in payload.get(key) or []
        if isinstance(party, dict)
        and _party_name(party)
        and str(party.get("acting_through") or "").strip()
    }
    if not filled:
        return
    kept = []
    for item in blob["items"]:
        raw = str(item.get("raw_text") or "") if isinstance(item, dict) else ""
        folded = raw.casefold()
        stale = (
            re.search(r"acting[-_ ]through", folded) is not None
            and re.search(r"\b(?:no|without|missing)\b", folded) is not None
            and any(name in folded for name in filled)
        )
        if not stale:
            kept.append(item)
    blob["items"] = kept


def _append_missing_acting_through(payload: dict[str, Any]) -> None:
    missing: list[str] = []
    pages: list[int] = []
    parts: list[str] = []
    for key in ("petitioners", "respondents"):
        for party in payload.get(key) or []:
            if not isinstance(party, dict):
                continue
            if party.get("kind") != "ORGANIZATION":
                continue
            acting = party.get("acting_through")
            if isinstance(acting, str) and acting.strip():
                continue
            name = _party_name(party) or "unnamed party"
            side = "petitioner" if key == "petitioners" else "respondent"
            missing.append(f"ORGANIZATION {side} '{name}' has no acting_through.")
            pages.extend(_pages_int(party.get("source_pages")))
            part = party.get("source_part")
            if isinstance(part, str) and part.strip():
                parts.append(part.strip())

    if not missing:
        return
    blob = payload.get("inconsistencies")
    if not isinstance(blob, dict):
        blob = {"items": [], "source_part": [], "source_pages": []}
        payload["inconsistencies"] = blob
    items = blob.get("items")
    if not isinstance(items, list):
        items = []
        blob["items"] = items
    existing = " ".join(
        str((_as_dict(item) if not isinstance(item, dict) else item).get("raw_text") or "")
        for item in items
    ).lower()
    for text in missing:
        if text.lower() in existing:
            continue
        items.append(
            {
                "id": _next_inconsistency_id(items),
                "label": "Missing acting through",
                "raw_text": text,
            }
        )
    part_list = blob.get("source_part")
    if not isinstance(part_list, list):
        part_list = []
        blob["source_part"] = part_list
    for part in parts:
        if part not in part_list:
            part_list.append(part)
    blob["source_pages"] = sorted(set(_pages_int(blob.get("source_pages")) + pages))


def _score_from_field_confidence(scores: Any) -> Any:
    if isinstance(scores, dict):
        nested = scores.get("confidence") or scores.get("score")
        if nested is not None:
            return nested
    return scores


def _stamp_object_confidence(node: dict[str, Any], scores: Any) -> None:
    existing = node.get("confidence")
    converted = confidence_percent_string(existing)
    if converted:
        node["confidence"] = converted
        return
    converted = confidence_percent_string(_score_from_field_confidence(scores))
    if converted:
        node["confidence"] = converted


def stamp_confidence(
    record: dict[str, Any],
    *,
    field_confidence: dict[str, Any] | None = None,
) -> None:
    scores = field_confidence if isinstance(field_confidence, dict) else {}
    for key in _CONFIDENCE_OBJECT_KEYS:
        node = record.get(key)
        if isinstance(node, dict):
            _stamp_object_confidence(node, scores.get(key))
    for key in _CONFIDENCE_LIST_KEYS:
        items = record.get(key)
        item_scores = scores.get(key)
        if not isinstance(items, list):
            continue
        for index, item in enumerate(items):
            if not isinstance(item, dict):
                continue
            per_item = None
            if isinstance(item_scores, list) and index < len(item_scores):
                per_item = item_scores[index]
            elif not isinstance(item_scores, list):
                per_item = item_scores
            _stamp_object_confidence(item, per_item)


def _petition_type_name(record: dict[str, Any], filing_type: str | None) -> str | None:
    named = _named(record.get("petition_type"))
    if named:
        return named
    key = normalize_filing_type(filing_type)
    if key in FILING_TYPE_LABELS:
        return FILING_TYPE_LABELS[key]
    raw = (filing_type or "").strip()
    return raw or None


def _drop_removed_fields(payload: dict[str, Any]) -> None:
    for key in ("classification", "applications", "relief", "filing_summary"):
        payload.pop(key, None)
    for key in ("petitioners", "respondents"):
        for party in payload.get(key) or []:
            if isinstance(party, dict):
                party.pop("age", None)
    for aor in payload.get("advocates_on_record") or []:
        if isinstance(aor, dict):
            aor.pop("office", None)
    for order in payload.get("impugned_orders") or []:
        if isinstance(order, dict):
            for extra in ("court_name", "petition_type_name", "judges", "lower_court_name"):
                order.pop(extra, None)
    blob = payload.get("inconsistencies")
    if isinstance(blob, dict):
        blob.pop("summary", None)


def _discard_reference_only_orders(payload: dict[str, Any]) -> None:
    """Do not promote a narrated earlier order to the order under challenge.

    A split filename is only a hint. Even a file named Impugned Order can
    contain chronology pages; inspect the actual quoted identification.
    Keep explicit challenge particulars and standalone order identifications.
    """
    orders = payload.get("impugned_orders")
    if not isinstance(orders, list):
        return
    kept = []
    for order in orders:
        if not isinstance(order, dict):
            kept.append(order)
            continue
        raw = re.sub(r"\s+", " ", str(order.get("raw_text") or "")).strip()
        challenged = re.search(
            r"\bimpugned\b|\bunder challenge\b|\bchalleng(?:e|es|ed|ing)\b|"
            r"\baggrieved\b|\barising (?:out of|from)\b|\bset aside\b",
            raw, re.IGNORECASE,
        )
        reference = re.search(
            r"\b(?:summari[sz]ed|quoted|reproduced|extracted)\s+(?:herein\s+)?"
            r"(?:below|above|as follows)|"
            r"\b(?:in compliance with|pursuant to|in pursuance of)\b.{0,100}\border\b",
            raw, re.IGNORECASE,
        )
        if reference and not challenged:
            continue
        kept.append(order)
    payload["impugned_orders"] = kept


def apply_extract_envelope(
    record: dict[str, Any] | None,
    *,
    page_parts: dict[int, Any] | None = None,
    filing_type: str | None = None,
    overall_confidence: float | str | None = None,
    field_confidence: dict[str, Any] | None = None,
    generated_at: str | None = None,
    page_markdown: Mapping[int, str] | None = None,
) -> dict[str, Any]:
    """Add envelope keys, stitch documents, and normalize legal fields."""
    payload = dict(record or {})
    spans = document_spans_from_page_parts(page_parts or {})
    petition_name = _petition_type_name(payload, filing_type)
    if petition_name:
        payload["petition_type"] = petition_name
    court_name = _named(payload.get("court"))
    if court_name:
        payload["court"] = court_name

    _normalize_legacy_keys(payload)
    _discard_reference_only_orders(payload)
    printed_date = petition_date_from_closing(
        main_petition_closing_text(page_markdown, page_parts)
    )
    if printed_date:
        payload["petition_date"] = printed_date
    _supplement_parties_from_main_petition(
        payload,
        page_markdown=page_markdown,
        page_parts=page_parts,
    )
    _supplement_parties_from_listing_proforma(
        payload,
        page_markdown=page_markdown,
        page_parts=page_parts,
    )
    _correct_impugned_order_bench(
        payload,
        page_markdown=page_markdown,
        page_parts=page_parts,
    )
    _normalize_parties(payload)
    _append_missing_acting_through(payload)
    _drop_role_label_inconsistencies(payload)
    _append_main_party_spelling(payload)
    stamp_confidence(payload, field_confidence=field_confidence)
    _drop_removed_fields(payload)

    payload["schema_version"] = SCHEMA_VERSION
    payload["job_type"] = JOB_TYPE
    payload["organization_id"] = None
    payload["workspace_id"] = None
    payload["user_id"] = None
    payload["primary_document_id"] = None
    payload["documents"] = spans
    payload["document_counts"] = {
        "processed": len(spans),
        "failed": 0,
    }
    payload["overall_confidence"] = confidence_percent_string(overall_confidence)
    if generated_at is None:
        generated_at = datetime.now(ZoneInfo("Asia/Kolkata")).isoformat(
            timespec="seconds"
        )
    payload["generated_at"] = generated_at
    return payload


_SPECIAL_CATEGORY_PATTERNS: tuple[tuple[re.Pattern[str], str], ...] = (
    (
        re.compile(r"civil appeal with high court certificate", re.IGNORECASE),
        "Civil Appeal with High Court Certificate",
    ),
    (
        re.compile(r"criminal appeal with high court certificate", re.IGNORECASE),
        "Criminal Appeal with High Court Certificate",
    ),
    (
        re.compile(r"bail applications?(?:\s*/\s*|\s+)bail matters", re.IGNORECASE),
        "Bail Applications/Bail Matters",
    ),
    (
        re.compile(r"bail applications?", re.IGNORECASE),
        "Bail Applications/Bail Matters",
    ),
    (
        re.compile(r"bail matters", re.IGNORECASE),
        "Bail Applications/Bail Matters",
    ),
    (
        re.compile(r"matters involving\s*/\s*pertaining to minors", re.IGNORECASE),
        "Matters involving/pertaining to minors",
    ),
    (
        re.compile(
            r"matters (?:involving|pertaining) to minors",
            re.IGNORECASE,
        ),
        "Matters involving/pertaining to minors",
    ),
    (
        re.compile(r"motor vehicles?\s+act(?:\s*,?\s*1988)?", re.IGNORECASE),
        "Motor Vehicle Act",
    ),
    (
        re.compile(r"consumer protection act,?\s*2019", re.IGNORECASE),
        "Consumer Protection Act, 2019",
    ),
    (
        re.compile(r"advocates act,?\s*1961", re.IGNORECASE),
        "Advocates Act, 1961",
    ),
    (
        re.compile(r"appeal\s*\(?\s*armed forces\s*\)?", re.IGNORECASE),
        "Appeal (Armed Forces)",
    ),
    (
        re.compile(r"demolition matters?", re.IGNORECASE),
        "Demolition Matters",
    ),
    (
        re.compile(r"eviction matters?", re.IGNORECASE),
        "Eviction Matters",
    ),
    (
        re.compile(r"tax matters?", re.IGNORECASE),
        "Tax Matters",
    ),
    (
        re.compile(r"indigent persons?", re.IGNORECASE),
        "Indigent person",
    ),
    (
        re.compile(r"jail petitions?", re.IGNORECASE),
        "Jail Petition",
    ),
    (
        re.compile(r"public interest litigation", re.IGNORECASE),
        "PIL",
    ),
    (re.compile(r"\bPIL\b", re.IGNORECASE), "PIL"),
    (
        re.compile(r"re-?\s*filing defects?", re.IGNORECASE),
        "Re-filing defect",
    ),
    (
        re.compile(r"transfer petitions?", re.IGNORECASE),
        "Transfer Petition",
    ),
)

_LISTING_CATEGORY_LINE = re.compile(
    r"state the tax effect"
    r"|special category\s*\([^)\n]*petitioner"
    r"|land acquisition matters\s*:"
    r"|senior citizen"
    r"|legal aid case"
    r"|in custody"
    r"|whether any bail application"
    r"|in a petition\s*\(\s*pil\s*\)",
    re.IGNORECASE,
)
_CHECKLIST_ANSWER_LINE = re.compile(
    r":\s*(?:n\.?\s*a\.?|na|yes|no)\s*$",
    re.IGNORECASE,
)
_SYNOPSIS_OR_PETITION_HEADING = re.compile(
    r"(?i)^\s*(synopsis|main petition)\b(.*)$"
)
_MAIN_PETITION_BODY = re.compile(
    r"most respectfully sh[eo]weth|the humble petition|petition under article|"
    r"in the matter of|arising (?:from|out of) the impugned",
    re.IGNORECASE,
)
_IDENTITY_LINE = re.compile(
    r"^(?:aadhaar|aadhar|uid|pan|mobile|mob\.?|phone|email|e-mail)\b",
    re.IGNORECASE,
)
_AGE_ABOUT_LINE = re.compile(
    r"^aged?\s+about\s+(\d+)\s+years?\b",
    re.IGNORECASE,
)
_PETITION_TYPE_CAPTION = re.compile(
    r"^\s*(?:\(?\s*)?(?:"
    r"special\s+leave\s+petition"
    r"|writ\s+petition"
    r"|review\s+petition"
    r"|transfer\s+petition"
    r"|jail\s+petition"
    r"|civil\s+appeal"
    r"|criminal\s+appeal"
    r"|petition\s+under\s+article"
    r"|in\s+the\s+matter\s+of"
    r")\b",
    re.IGNORECASE,
)


def _without_listing_category_lines(text: str | None) -> str:
    kept = [
        line
        for line in (text or "").splitlines()
        if line.strip()
        and not _LISTING_CATEGORY_LINE.search(line)
        and not _CHECKLIST_ANSWER_LINE.search(line)
    ]
    return "\n".join(kept)


def match_front_special_categories(
    text: str | None,
    *,
    limit: int | None = 3,
) -> list[str]:
    """Catalogue special-category names, in the order the text states them."""
    hits: list[tuple[int, int, str]] = []
    for pattern, name in _SPECIAL_CATEGORY_PATTERNS:
        for match in pattern.finditer(_without_listing_category_lines(text)):
            hits.append((match.start(), match.end(), name))
    hits.sort(key=lambda item: (item[0], -(item[1] - item[0])))
    chosen: list[str] = []
    covered_to = -1
    for start, end, name in hits:
        if start < covered_to:
            continue
        if name not in chosen:
            chosen.append(name)
        covered_to = end
        if limit is not None and len(chosen) >= limit:
            break
    return chosen


def _category_heading_line(line: str) -> bool:
    match = _SYNOPSIS_OR_PETITION_HEADING.match(line.strip())
    if not match:
        return False
    rest = match.group(2).strip(" \t.:|-")
    if not rest:
        return True
    if re.fullmatch(r"\d+", rest):
        return False
    return bool(re.search(r"[A-Za-z]{3,}", rest))


def _special_category_page_text(pages: Mapping[int, str] | None) -> str:
    """Synopsis and main-petition text, without listing-proforma answer rows."""
    if not pages:
        return ""
    chunks: list[str] = []
    for page in sorted(pages):
        text = pages.get(page) or ""
        lines = text.splitlines()
        source = any(_category_heading_line(line) for line in lines) or bool(
            _MAIN_PETITION_BODY.search(text)
        )
        if not source:
            continue
        kept = [
            line
            for line in lines
            if not _LISTING_CATEGORY_LINE.search(line)
            and not _CHECKLIST_ANSWER_LINE.search(line)
            and not _PETITION_TYPE_CAPTION.match(line.strip())
        ]
        if kept:
            chunks.append("\n".join(kept))
    return "\n".join(chunks)


def _impugned_category_text(payload: Mapping[str, Any] | None) -> str:
    """The challenged order can name the category, as in Bail Application No."""
    if not payload:
        return ""
    bits: list[str] = []
    for order in payload.get("impugned_orders") or []:
        if not isinstance(order, dict):
            continue
        for key in ("raw_text", "case_number", "Forum", "forum"):
            value = order.get(key)
            if isinstance(value, str) and value.strip():
                bits.append(value.strip())
    return "\n".join(bits)


def _printed_special_category(
    value: object,
    page_markdown: Mapping[int, str] | None = None,
    impugned_text: str | None = None,
) -> str | None:
    filed = "\n".join(
        part
        for part in (
            _special_category_page_text(page_markdown),
            impugned_text or "",
        )
        if part and part.strip()
    )
    if filed.strip():
        names = match_front_special_categories(filed)
        return ", ".join(names) if names else None
    names = match_front_special_categories(
        str(value) if value is not None else ""
    )
    if not names:
        return None
    return ", ".join(names)


_NOT_A_PARTY_NAME = re.compile(
    r"("
    r"\b(yes|n/?a)\b"
    r"|whether\b"
    r"|disposed of"
    r"|learned counsel"
    r"|submits that"
    r"|per contra"
    r"|heard the"
    r"|perused"
    r"|petition is barred"
    r"|condonation"
    r"|main category"
    r"|not to be listed"
    r"|criminal matters"
    r"|tax matters"
    r"|special category"
    r"|vehicle number"
    r"|pending applications"
    r"|order be communicated"
    r"|it is made clear"
    r"|application is accordingly"
    r"|\?"
    r")",
    re.IGNORECASE,
)


def _clean_front_party_name(name: str | None) -> str:
    """Drop LlamaParse markdown and a leading list number from a party name."""
    text = re.sub(r"\*+", "", name or "")
    text = _LIST_SERIAL.sub("", text)
    text = strip_party_role_label(text)
    text = strip_anr_ors_suffix(text) or text
    return re.sub(r"\s+", " ", text).strip(" ,.;")


def _looks_like_party_name(name: str | None) -> bool:
    text = _clean_front_party_name(name)
    if not text:
        return False
    if not re.search(r"[A-Za-z]{2,}", text):
        return False
    if re.match(r"^\d{4}\b", text) or re.search(r"\bpassed by\b", text, re.IGNORECASE):
        return False
    if re.fullmatch(r"(?:col|mr|mrs|ms|shri|smt|the)\.?", text, re.IGNORECASE):
        return False
    if re.match(r"^(?:occupation|age|years|aged\s+about)\b", text, re.IGNORECASE):
        return False
    if _IDENTITY_LINE.match(text):
        return False
    if re.match(r"^the\s+humble\s+petition\b", text, re.IGNORECASE):
        return False
    if re.match(r"^signature\b", text, re.IGNORECASE):
        return False
    if re.match(
        r"^(?:presently\s+at|second\s+address|also\s+at|not\s+a\s+party|dead)\b",
        text,
        re.IGNORECASE,
    ):
        return False
    if re.fullmatch(r"(?:and\s+)?(?:anrs?|ors)\.?", text, re.IGNORECASE):
        return False
    if re.search(r"\bwrit(?:\s*-\s*[a-z])?\b", text, re.IGNORECASE) and re.search(r"\d", text):
        return False
    if re.match(r"^[A-Za-z]?\s*/\s*\d+\s*/\s*\d{4}\b", text):
        return False
    if re.fullmatch(r"[\d\s\-_.,/()]+", text):
        return False
    if re.match(
        r"^(?:crl\.?\s*m\.?p\.?|s\.?c\.?|annexure|judgment|order|appeal|fir|deposition|memo\s+of\s+parties)\b",
        text,
        re.IGNORECASE,
    ):
        return False
    words = text.split()
    word_limit = 24 if "(" in text or re.search(r"\bthrough\b", text, re.IGNORECASE) else 12
    if len(words) > word_limit or len(text) > 180:
        return False
    if _NOT_A_PARTY_NAME.search(text):
        return False
    if re.match(
        r"^(the petition|there is|there was|learned|heard|per contra|be that|"
        r"as already|pending|copy of the|order be|application is|it is made|"
        r"not to be listed|\(?\s*a\s*\)|similar disposed|"
        r"nature\s+of|a\s+brief|brief\s+background|background|special\s+powers|in\s+view\s+of|"
        r"the\s+present\s+petition|office\s+report|proforma|listing|question\s+of\s+law|grounds|prayer|declaration|certificate|"
        r"most\s+respectfully|respectfully\s+showeth|questions?\s+of\s+law|has\s+reversed|brief\s+facts|under\s+section|affidavit|vakalatnama)",
        text,
        re.IGNORECASE,
    ):
        return False
    return True


def _party_record(
    name: str,
    *,
    serial: int,
    page: int | None = None,
    pages: Sequence[int] | None = None,
    relation: str | None = None,
    guardian: str | None = None,
    occupation: str | None = None,
    age: str | None = None,
    acting_through: str | None = None,
    address: str | None = None,
    source_part: str = "Opening pages",
) -> dict[str, Any]:
    cleaned = _clean_front_party_name(name)
    kind = (
        "ORGANIZATION"
        if re.search(
            r"\b(?:state|union of india|govt|government|ltd|limited|corporation|bank|department|police)\b",
            cleaned,
            re.IGNORECASE,
        )
        else None
    )
    source_pages = list(pages) if pages else ([page] if page else [])
    return {
        "serial": serial,
        "kind": normalize_party_kind(cleaned, kind),
        "name": cleaned,
        "relation": relation,
        "guardian": guardian,
        "occupation": occupation,
        "age": age,
        "acting_through": acting_through,
        "address": address,
        "source_part": source_part,
        "source_pages": source_pages,
    }


def _clean_markdown_to_lines(text: str) -> list[str]:
    """Convert HTML table and markdown text to clean logical lines."""
    s = re.sub(r"(?i)<br\s*/?>", "\n", text)
    s = re.sub(r"(?i)</(?:tr|td|th|p|div|li|h[1-6])>", "\n", s)
    s = re.sub(r"<[^>]+>", " ", s)
    s = re.sub(r"[*_`#]+", "", s)
    raw_lines = [re.sub(r"\s+", " ", line).strip() for line in s.splitlines()]
    return [line for line in raw_lines if line]


def _is_party_role_or_header(line: str) -> bool:
    l = re.sub(r"^[.\s,\-—_`*#~|]+", "", line).strip().lower()
    if not l:
        return True
    if re.match(r"^(?:in\s+the\s+)?(?:supreme|high)\s+court\s+of\s+india\b", l):
        return True
    if re.match(r"^(?:civil|criminal)\s+(?:appellate\s+)?jurisdiction\b", l):
        return True
    if re.match(r"^(?:under\s+(?:article|order|section|rule)|\(?against\s+the\s+impugned\b)", l):
        return True
    if re.match(r"^special\s+leave\s+petition\b", l):
        return True
    if re.match(
        r"^(?:in\s+the\s+)?(?:trial\s*court|high\s*court|in\s*this\s*court|this\s*hon[\x27\x60’]?ble\s*court|court)\b",
        l,
    ):
        return True
    if re.match(
        r"^(?:the\s+)?(?:position\s*of\s*(?:the\s*)?parties|memo\s*of\s*(?:the\s*)?parties)\b",
        l,
    ):
        return True
    if re.match(r"^in\s+the\s+matter\s+of(?:\s*[:-]+)?", l):
        return True
    if re.match(r"^(?:to\s*,?\s*)?(?:the\s+)?hon[\x27\x60’]?ble\s+(?:chief\s+justice|court)\b", l):
        return True
    if re.match(r"^and\s+his\s+companion\s+judges\b", l):
        return True
    if re.match(r"^(?:the\s+)?special\s+leave\s+petition\s+of\b", l):
        return True
    if re.match(r"^(?:the\s+)?petitioners?\s+above[- ]named\b", l):
        return True
    if re.match(r"^the\s+humble\s+petition\b", l):
        return True
    if re.match(r"^between\b", l):
        return True
    if re.match(r"^(?:defacto|defecto|de\s*facto)\b", l):
        return True
    if re.match(r"^to\s*,?\s*$", l):
        return True
    if re.match(
        r"^(?:accused|respondents?|petitioners?|appellants?|complainants?|applicants?|opposite\s*parties|opposite\s*party|def[ae]cto\s*complainant|contesting\s*respondents?|contesting)(?:\s*(?:no|number)\.?)?\s*[.\s]*\d*$",
        l,
    ):
        return True
    if re.match(
        r"^\d+\s*(?:st|nd|rd|th)?\s*(?:respondents?|petitioners?|appellants?|accused)$",
        l,
    ):
        return True
    if re.match(r"^(?:no|number)\.?\s*\d*$", l):
        return True
    if re.match(r"^[.\s,\-—_`*#~|]+$", line) or re.match(r"^\.?\d+$", line.strip()):
        return True
    return False


def _parse_party_section(
    lines: list[str],
    *,
    side: str = "petitioner",
    page: int | None = None,
    pages: Sequence[int] | None = None,
) -> list[dict[str, Any]]:
    parties: list[dict[str, Any]] = []
    stop_regex = re.compile(
        r"^(?:most\s+respectfully\s+sh[eo]weth|the\s+humble\s+petition\b|note\s*:|1\.\s+that\b|synopsis\b|grounds\b|prayer\b|prayers\b|questions\s+of\s+law\b|verification\b|affidavit\b|index\s+of\b)",
        re.IGNORECASE,
    )
    idx = 0
    while idx < len(lines):
        line = lines[idx]
        if stop_regex.search(line):
            break
        if parties and re.match(r"^to\s*,\s*(?!hon[\x27\x60’]?ble)", line, re.IGNORECASE):
            break
        if _is_party_role_or_header(line) or re.match(
            r"^(?:versus|vs\.?|v/s|versuu)\b", line, re.IGNORECASE
        ):
            idx += 1
            continue

        m_num = re.match(r"^[(\[]?\s*([0-9lI|]+)\s*[.)\]]\s*(.+)$", line)
        rel_check = re.match(r"^[SDW]/o\b", line, re.IGNORECASE)
        party_name = None
        raw_cand = None

        if m_num and not _is_party_role_or_header(m_num.group(2)):
            raw_cand = m_num.group(2).strip()
        elif rel_check and idx > 0:
            for b in range(idx - 1, max(-1, idx - 6), -1):
                c = lines[b]
                if not _is_party_role_or_header(c) and _looks_like_party_name(c):
                    raw_cand = c
                    idx = b
                    break
        elif not _is_party_role_or_header(line) and _looks_like_party_name(line):
            next_line = lines[idx + 1] if idx + 1 < len(lines) else ""
            if (
                re.match(r"^[SDW]/o\b", next_line, re.IGNORECASE)
                or re.match(
                    r"^(?:through|represented\s+by|r/o|registered\s+address|resident(?:\s+of)?|residing)\b",
                    next_line,
                    re.IGNORECASE,
                )
                or re.search(r"(?:\.\.\.|\b)(?:petitioners?|respondents?|appellants?|accused)", line, re.IGNORECASE)
            ):
                raw_cand = line

        if not raw_cand:
            idx += 1
            continue

        raw_cand = re.sub(r"[*_`#]+", "", raw_cand).strip()
        raw_cand = re.sub(
            r"(?:\.\.\.|\b)(?:petitioners?|respondents?|appellants?|accused)(?:\s*(?:no|number)\.?)?\s*[.\s]*\d*.*$",
            "",
            raw_cand,
            flags=re.IGNORECASE,
        ).strip()

        party_name = raw_cand
        relation = None
        guardian = None
        acting_through = None
        office_place = None
        age = None
        addr_parts: list[str] = []

        rel_m = _RELATION_MARK.search(raw_cand)
        if rel_m:
            party_name = raw_cand[: rel_m.start()].strip(" ,")
            relation = _normalize_relation(rel_m.group(1))
            guardian, leftover = _split_relation_tail(rel_m.group(2))
            if leftover:
                addr_parts.append(leftover)
        else:
            party_name, office_place = _split_office_place(raw_cand)
            if office_place:
                addr_parts.append(office_place)
        office_at = len(addr_parts) - 1 if office_place else None

        rep_m = re.search(
            r"(?:rep(?:\.|resented)?\s+by|through)\s+([A-Za-z\s.\x27\x60’]+?)(?:,|$)",
            party_name,
            re.IGNORECASE,
        )
        if rep_m:
            acting_through = rep_m.group(1).strip(" ,.'")
            if re.match(
                r"through\s+(?:his|her|its)\b",
                party_name[rep_m.start() :],
                re.IGNORECASE,
            ):
                party_name = party_name[: rep_m.start()].strip(" ,.")

        follow_at = idx + 1
        if (
            follow_at < len(lines)
            and re.search(r"(?:government|govt\.?)$", party_name, re.IGNORECASE)
        ):
            wrapped = _NAME_WRAP_TAIL.match(lines[follow_at])
            if wrapped:
                party_name = f"{party_name.rstrip(' ,')} {wrapped.group(1).strip()}"
                remainder = (wrapped.group(2) or "").strip(" ,")
                if remainder:
                    lines[follow_at] = remainder
                else:
                    follow_at += 1
                party_name, office_place = _split_office_place(party_name)
                if office_place:
                    addr_parts.append(office_place)
                    office_at = len(addr_parts) - 1
        if office_at is not None:
            follow = lines[follow_at] if follow_at < len(lines) else ""
            if not _PLACE_WORD_CONTINUATION.match(follow):
                addr_parts[office_at] = addr_parts[office_at].rstrip(" ,") + ","
        idx = follow_at - 1

        cursor = idx + 1
        while cursor < len(lines):
            c_line = lines[cursor]
            if stop_regex.search(c_line) or re.match(
                r"^(?:versus|vs\.?|v/s|versuu)\b", c_line, re.IGNORECASE
            ):
                break
            if re.match(r"^[(\[]?\s*[0-9lI|]+\s*[.)\]]\s*[A-Za-z]", c_line):
                break
            if _BARE_LIST_NUMBER.fullmatch(c_line):
                break
            if _is_schedule_role_line(c_line):
                cursor += 1
                continue
            if _ADDRESS_CONTINUATION.match(c_line):
                addr_parts.append(c_line)
                cursor += 1
                continue
            if _IDENTITY_LINE.match(c_line):
                cursor += 1
                continue
            age_about = _AGE_ABOUT_LINE.match(c_line)
            if age_about:
                age = age or f"{age_about.group(1)} years"
                cursor += 1
                continue
            if re.match(
                r"^(?:special\s+leave\s+petition|the\s+constitution|the\s+humble\s+petition)\b",
                c_line,
                re.IGNORECASE,
            ):
                break
            next_line = lines[cursor + 1] if cursor + 1 < len(lines) else ""
            if (
                next_line
                and re.search(r"[A-Za-z]{2,}", c_line)
                and not re.search(r"\d", c_line)
                and len(c_line.split()) <= 8
                and not re.match(
                    r"^(?:[SDW]/o|r/o|through|resident|residing|registered)\b",
                    c_line,
                    re.IGNORECASE,
                )
                and re.match(
                    r"^(?:[SDW]/o|r/o|registered\s+address|resident(?:\s+of)?|residing)\b",
                    next_line,
                    re.IGNORECASE,
                )
            ):
                break
            if _is_party_role_or_header(c_line):
                cursor += 1
                continue
            c_rel = _RELATION_MARK.match(c_line)
            if c_rel and not relation:
                relation = _normalize_relation(c_rel.group(1))
                guardian, leftover = _split_relation_tail(c_rel.group(2))
                if leftover:
                    addr_parts.append(leftover)
                cursor += 1
                continue
            through_body = None
            if _THROUGH_LABEL.fullmatch(c_line):
                cursor += 1
                body_bits: list[str] = []
                while cursor < len(lines):
                    nxt = lines[cursor]
                    if (
                        stop_regex.search(nxt)
                        or _is_party_role_or_header(nxt)
                        or re.match(r"^(?:versus|vs\.?|v/s|versuu)\b", nxt, re.IGNORECASE)
                        or re.match(r"^[(\[]?\s*[0-9lI|]+\s*[.)\]]\s*[A-Za-z]", nxt)
                    ):
                        break
                    body_bits.append(nxt)
                    cursor += 1
                through_body = " ".join(body_bits)
            else:
                c_rep = _THROUGH_BODY.match(c_line)
                if c_rep:
                    through_body = c_rep.group(1).strip()
                    cursor += 1
            if through_body is not None:
                if not acting_through:
                    representative, place = _split_through_and_place(through_body)
                    if representative:
                        acting_through = representative
                    if place:
                        addr_parts.append(place)
                continue
            addr_parts.append(c_line)
            cursor += 1

        address = _clean_party_address(
            " ".join([p for p in addr_parts if not _is_party_role_or_header(p)])
        ) or ""
        address = re.sub(
            r"^(?:registered\s+address\s+at)\s*", "", address, flags=re.IGNORECASE
        ).strip()
        address = re.sub(r"\s*\bTO\s*[,:]?\s*$", "", address, flags=re.IGNORECASE).strip()
        address = re.sub(
            r"(?:\s*(?:contesting\s*)?(?:respondents?|petitioners?|appellants?|accused)?\s*(?:no|number)\.?\s*\d*)+\s*$",
            "",
            address,
            flags=re.IGNORECASE,
        ).strip()
        address = re.sub(
            r"(?:\.\.\.|\b)(?:petitioners?|respondents?|appellants?|accused)(?:\s*(?:no|number)\.?)?\s*[.\s]*\d*.*$",
            "",
            address,
            flags=re.IGNORECASE,
        ).strip()
        address = re.sub(
            r"\s*(?:\.\.\.|\b)(?:filed\s*by|advocate\s*for|signature|date\s*:|place\s*:|hon[\x27\x60’]?ble\s+supreme\s+court|supreme\s+court\s+of\s+india|the\s+special\s+leave\s+petition|the\s+humble\s+petition|chief\s+justice|companion\s+justices).*",
            "",
            address,
            flags=re.IGNORECASE,
        ).strip()

        clean_name = re.sub(
            r"(?:accused|respondents?|appellants?|petitioners?|complainant).*",
            "",
            party_name,
            flags=re.IGNORECASE,
        ).strip()
        clean_name = re.sub(r"[.\s\d|,\-—_]+$", "", clean_name).strip()

        if _looks_like_party_name(clean_name or party_name):
            p_rec = _party_record(
                clean_name or party_name,
                serial=len(parties) + 1,
                page=page,
                pages=pages,
                relation=relation,
                guardian=guardian or None,
                age=age,
                address=address or None,
            )
            if acting_through:
                p_rec["acting_through"] = acting_through
            parties.append(p_rec)
        idx = cursor
    return parties


def _is_name_fragment(line: str) -> bool:
    """A wrapped name token such as 'Anoop' or '@', not an address line."""
    stripped = line.strip()
    if not stripped or _is_party_role_or_header(stripped):
        return False
    if re.fullmatch(r"[@&]|alias", stripped, re.IGNORECASE):
        return True
    if re.search(r"\d", stripped):
        return False
    if re.match(
        r"^(?:[SDW]/o|r/o|resident|residing|registered\s+address|address|through|versus|vs\.?)\b",
        stripped,
        re.IGNORECASE,
    ):
        return False
    if re.match(
        r"^(?:petitioners?|respondents?|applicants?|accused|complainants?)\b",
        stripped,
        re.IGNORECASE,
    ):
        return False
    if re.search(
        r"\b(?:road|hotel|floor|place|delhi|nagar|house|district|village|street|lane)\b",
        stripped,
        re.IGNORECASE,
    ):
        return False
    words = [word for word in re.split(r"\s+", stripped) if word and word not in {"@", "&"}]
    return 0 < len(words) <= 3


def _join_wrapped_party_lines(lines: list[str]) -> list[str]:
    """Join a name that the parser split onto one word per line."""
    joined: list[str] = []
    buffer: list[str] = []

    def flush() -> None:
        if buffer:
            joined.append(" ".join(buffer))
            buffer.clear()

    for line in lines:
        if _is_name_fragment(line):
            buffer.append(line)
            continue
        flush()
        joined.append(line)
    flush()
    return joined


def _parse_memo_of_parties(
    text: str,
    page: int | None = None,
    pages: Sequence[int] | None = None,
) -> dict[str, list[dict[str, Any]]]:
    clean_lines = _join_through_tails(
        _join_open_parentheses(
            _join_bare_list_numbers(
                _join_wrapped_party_lines(_clean_markdown_to_lines(text))
            )
        )
    )
    clean_lines = [
        line
        for line in clean_lines
        if not re.search(r"in the matter of.*versus", line, re.IGNORECASE)
        and not _is_schedule_role_line(line)
    ]
    source_pages = list(pages) if pages else ([page] if page else [])

    heading_at = None
    for idx, line in enumerate(clean_lines):
        if re.search(
            r"\b(?:memo\s*of|posit[io1lt]{2,6}n?\s*of)\s*(?:the\s*)?parties\b",
            line,
            re.IGNORECASE,
        ):
            heading_at = idx
    search_from = heading_at + 1 if heading_at is not None else 0
    divider_idx = None
    for idx, line in enumerate(clean_lines):
        if idx < search_from:
            continue
        if re.match(r"^(?:versus|vs\.?|v/s|versuu)\b", line, re.IGNORECASE):
            prev_is_role = idx > 0 and _is_party_role_or_header(clean_lines[idx - 1])
            next_is_role = idx + 1 < len(clean_lines) and _is_party_role_or_header(clean_lines[idx + 1])
            if prev_is_role and next_is_role and idx < 6:
                continue
            divider_idx = idx
            break

    if divider_idx is None:
        has_pet_so_far = False
        for idx, line in enumerate(clean_lines):
            if re.match(r"^[SDW]/o\b", line, re.IGNORECASE):
                has_pet_so_far = True
            elif has_pet_so_far and re.match(r"^[(\[]?\s*1\s*[.)\]]\s*[A-Za-z]", line):
                divider_idx = idx
                break

    if divider_idx is not None:
        pet_lines = clean_lines[search_from:divider_idx]
        resp_lines = clean_lines[divider_idx:]
        if resp_lines and re.match(r"^(?:versus|vs\.?|v/s|versuu)\b", resp_lines[0], re.IGNORECASE):
            resp_lines = resp_lines[1:]
    else:
        pet_lines = clean_lines
        resp_lines = []

    petitioners = _parse_party_section(pet_lines, side="petitioner", page=page, pages=source_pages)
    respondents = _parse_party_section(resp_lines, side="respondent", page=page, pages=source_pages)

    if not respondents and divider_idx is None:
        respondents = _parse_party_section(clean_lines, side="respondent", page=page, pages=source_pages)

    return {"petitioners": petitioners, "respondents": respondents}


_COLUMN_ROLE = re.compile(
    r"^(?:petitioners?|respondents?)(?:\s*(?:no|number)\.?)?\s*\d*$",
    re.IGNORECASE,
)
_COLUMN_STOP = re.compile(
    r"^(?:most\s+respectfully\s+sh[eo]weth|questions?\s+of\s+law|to\s*,|1\.\s+that\b)",
    re.IGNORECASE,
)
_SCHEDULE_NUMBERED_LINE = re.compile(r"^\d+\s*[.)]\s*\S")
_DETAIL_SPLIT = re.compile(
    r"(?i)(?=,?\s*(?:age\b|occupation\b|gut\s+no\b|[sdw]/o|[sdw]\.o\b|r/o\b|through\b|having\s+office\b))"
)


def _column_role_kind(line: str) -> str | None:
    if re.match(r"^petitioner", line, re.IGNORECASE):
        return "petitioner"
    if re.match(r"^respondent", line, re.IGNORECASE):
        return "respondent"
    return None


def _has_court_columns(text: str) -> bool:
    """A party schedule printed as Before High Court beside Before Supreme Court."""
    lowered = text.lower()
    return (
        re.search(r"before\s+(?:the\s+)?high\s+court", lowered) is not None
        and re.search(r"before\s+(?:the\s+)?supreme\s+court", lowered) is not None
    )


def _split_at_numbered_party(lines: list[str]) -> tuple[list[str], list[str]]:
    for index, line in enumerate(lines):
        if _SCHEDULE_NUMBERED_LINE.match(line):
            return lines[:index], lines[index:]
    return lines, []


def _is_paper_folio(line: str, previous: str) -> bool:
    """A bare paper-book page number, not the second half of a pin code or an age."""
    if not re.fullmatch(r"\d{1,3}", line):
        return False
    if previous and re.search(r"[–-]\s*\d+\s*$", previous):
        return False
    if previous and re.search(r"\bage[d]?$", previous, re.IGNORECASE):
        return False
    return int(line) < 200


def _printed_age(blob: str) -> str | None:
    """'age 52 years' and 'aged 56 years' become '52 years'."""
    match = re.search(r"\bage[d]?\s+(\d+)\s+years?\b", blob, re.IGNORECASE)
    if not match:
        return None
    return f"{match.group(1)} years"


def _trailing_occupation(blob: str) -> str | None:
    """A short descriptor with no place line, such as 'Indian Jew'."""
    text = blob
    text = re.sub(
        r"\b(?:[sdw]/o|[sdw]\.o)\s+.+?(?=,|$)",
        " ",
        text,
        flags=re.IGNORECASE,
    )
    text = re.sub(r"\bage[d]?\s+\d+\s+years?\b", " ", text, flags=re.IGNORECASE)
    text = re.sub(r"\boccupation\s+[A-Za-z]+\b", " ", text, flags=re.IGNORECASE)
    text = re.sub(
        r"\b(?:r/o|gut\s+no\.?|having\s+office\s+at|office\s+at|through)\b.*",
        " ",
        text,
        flags=re.IGNORECASE,
    )
    text = re.sub(r"[^A-Za-z\s]", " ", text)
    text = re.sub(r"\s+", " ", text).strip()
    if not text or not re.fullmatch(r"[A-Za-z]+(?:\s+[A-Za-z]+){0,3}", text):
        return None
    if re.search(
        r"\b(?:district|aurangabad|mumbai|mantralaya|road|plot|nagar|delhi|"
        r"village|petitioner|respondent|court|society|housing|farms|secretary|chairman)\b",
        text,
        re.IGNORECASE,
    ):
        return None
    return text


def _schedule_party_record(
    lines: list[str],
    *,
    serial: int,
    pages: Sequence[int],
) -> dict[str, Any] | None:
    """Name, relation, occupation, and address from one numbered schedule block."""
    name_bits: list[str] = []
    rest: list[str] = []
    started = False
    for index, raw in enumerate(lines):
        line = re.sub(r"^\d+\s*[.)]\s*", "", raw).strip() if index == 0 else raw.strip()
        if not line or _is_paper_folio(line, rest[-1] if rest else ""):
            continue
        pieces = _DETAIL_SPLIT.split(line, maxsplit=1)
        head = pieces[0].strip(" ,")
        tail = pieces[1].strip() if len(pieces) == 2 else ""
        if not started and tail:
            if head:
                name_bits.append(head)
            rest.append(tail)
            started = True
            continue
        if not started and (
            re.match(
                r"^(?:[sdw]/o|[sdw]\.o|r/o|through|age|occupation|gut\s+no|having\s+office)\b",
                line,
                re.IGNORECASE,
            )
            or (
                len(name_bits) >= 3
                and re.match(r"^(?:taluka|district|city|mantralaya|mumbai|aurangabad)\b", line, re.IGNORECASE)
            )
        ):
            started = True
        if started:
            rest.append(line.strip())
        elif head:
            name_bits.append(head)
    name = re.sub(r"\s+", " ", " ".join(name_bits)).strip(" ,.")
    name = re.sub(r"\bCo\.\s+op\b", "Co-op", name, flags=re.IGNORECASE)
    if not _looks_like_party_name(name):
        return None
    blob = re.sub(r"\s+", " ", " ".join(part for part in rest if part)).strip(" ,")
    relation = None
    guardian = None
    relation_match = re.search(
        r"\b([sdw]/o|[sdw]\.o)\s+([A-Za-z][A-Za-z .'\u2019]+?)(?=\s*(?:,|$|\bage\b|\boccupation\b|\br/o\b|\bthrough\b))",
        blob,
        re.IGNORECASE,
    )
    if relation_match:
        prefix = relation_match.group(1).lower().replace(".", "/")
        relation = prefix[0].upper() + prefix[1:]
        guardian_words: list[str] = []
        for word in relation_match.group(2).split():
            if re.match(r"^(?:indian|jew|years?|occupation|age)$", word, re.IGNORECASE):
                break
            guardian_words.append(word.strip(" ,."))
            if len(guardian_words) == 4:
                break
        guardian = " ".join(word for word in guardian_words if word)
    occupation_match = re.search(r"\boccupation\s+([A-Za-z]+)", blob, re.IGNORECASE)
    occupation = occupation_match.group(1).strip() if occupation_match else None
    if not occupation:
        occupation = _trailing_occupation(blob)
    age = _printed_age(blob)
    acting_through = None
    address = None
    place_match = re.search(
        r"\b((?:r/o|gut\s+no\.?|having\s+office\s+at|office\s+at)\s+.+?)(?:\s+through\b|$)",
        blob,
        re.IGNORECASE,
    )
    if place_match:
        address = re.sub(
            r"\bCo\.\s+op\.?\b",
            "Co-op",
            place_match.group(1),
            flags=re.IGNORECASE,
        ).strip(" ,.")
    through_match = re.search(
        r"\bthrough\s+((?:its|his|her)\s+)?([^,]+?)(?:,\s*(.*))?$",
        blob,
        re.IGNORECASE,
    )
    if through_match:
        acting_through = re.sub(
            r"\s+",
            " ",
            " ".join(
                part.strip()
                for part in (through_match.group(1), through_match.group(2))
                if part and part.strip()
            ),
        ).strip(" ,.")
        if not address and through_match.group(3):
            address = through_match.group(3).strip(" ,.")
    if not address:
        tail_match = re.search(
            r"((?:Aurangabad|Mumbai|Mantralaya|Plot\s+no\.?).+)$",
            blob,
            re.IGNORECASE,
        )
        if tail_match:
            address = tail_match.group(1).strip(" ,.")
    record = _party_record(
        name,
        serial=serial,
        pages=pages,
        relation=relation,
        guardian=guardian or None,
        occupation=occupation,
        age=age,
        acting_through=acting_through,
        address=_clean_party_address(address) if address else None,
    )
    return record


def _parse_court_column_schedule(
    text: str,
    pages: Sequence[int],
) -> dict[str, list[dict[str, Any]]] | None:
    """Keep the current-court column when a schedule prints both court columns."""
    if not _has_court_columns(text):
        return None
    lines = _clean_markdown_to_lines(text)
    page_numbers = {str(page) for page in pages}
    blocks: list[tuple[str, list[str]]] = []
    buffer: list[str] = []
    pairs = 0
    index = 0

    def attach(extra: list[str]) -> None:
        if extra and blocks:
            blocks[-1][1].extend(extra)

    while index < len(lines):
        line = lines[index]
        if _COLUMN_STOP.match(line):
            continuation, _body = _split_at_numbered_party(buffer)
            attach(continuation)
            buffer = []
            break
        previous = buffer[-1] if buffer else ""
        if (
            line in page_numbers
            or _is_paper_folio(line, previous)
            or re.match(r"^(?:versus|vs\.?|v/s)\b", line, re.IGNORECASE)
        ):
            index += 1
            continue
        if _COLUMN_ROLE.match(line):
            nxt = index + 1
            while nxt < len(lines) and lines[nxt] in page_numbers:
                nxt += 1
            if nxt < len(lines) and _COLUMN_ROLE.match(lines[nxt]):
                kind = _column_role_kind(lines[nxt])
                continuation, body = _split_at_numbered_party(buffer)
                attach(continuation)
                if body and kind:
                    blocks.append((kind, body))
                    pairs += 1
                buffer = []
                index = nxt + 1
                continue
        buffer.append(line)
        index += 1
    if pairs < 2:
        return None
    petitioners: list[dict[str, Any]] = []
    respondents: list[dict[str, Any]] = []
    for kind, body in blocks:
        target = petitioners if kind == "petitioner" else respondents
        record = _schedule_party_record(body, serial=len(target) + 1, pages=pages)
        if record:
            target.append(record)
    if not petitioners or not respondents:
        return None
    return {"petitioners": petitioners, "respondents": respondents, "_replace": True}


_PARTY_SCHEDULE_HEADING = re.compile(
    r"\b(?:memo\s*of|posit[io1lt]{2,6}n?\s*of)\s*(?:the\s*)?parties\b",
    re.IGNORECASE,
)


def _is_explicit_party_schedule(text: str) -> bool:
    """A position-of-parties or memo-of-parties page, not an annexure cover."""
    if not text or not _PARTY_SCHEDULE_HEADING.search(text):
        return False
    head = "\n".join(text.splitlines()[:12])
    return re.search(r"\bannexure\b", head, re.IGNORECASE) is None


_SCHEDULE_END = re.compile(
    r"most\s+respectfully\s+sh[eo]weth|the\s+humble\s+petition",
    re.IGNORECASE,
)


def _schedule_follow_pages(
    layer: Mapping[int, str],
    start: int,
    *,
    skip: set[int] | None = None,
) -> dict[int, str]:
    """The schedule page and the next pages, through the prayer."""
    pages: dict[int, str] = {}
    blocked = skip or set()
    for follow in range(start, start + 4):
        if follow in blocked:
            break
        follow_text = layer.get(follow)
        if follow_text is None:
            break
        pages[follow] = follow_text
        if follow != start and _SCHEDULE_END.search(follow_text):
            break
    return pages


def supplemental_front_party_pages(
    full_text: Mapping[int, str],
    included_pages: set[int],
) -> dict[int, str]:
    """Party pages that sit past the opening slice and before the closing pages."""
    pages: dict[int, str] = {}
    layer = {int(page): text or "" for page, text in full_text.items()}
    for number, text in sorted(layer.items()):
        if number in included_pages or number in pages:
            continue
        if not _is_explicit_party_schedule(text):
            continue
        pages.update(_schedule_follow_pages(layer, number, skip=included_pages))
    return pages


def merge_dropped_party_schedule(
    page_markdown: Mapping[int, str],
    text_layer: Mapping[int, str] | None,
) -> dict[int, str]:
    """Use the PDF text layer when parsing dropped a two-column party schedule.

    LlamaParse often returns only the page number for that schedule. The text
    layer still has every numbered party and the lines under the name.
    """
    merged = {int(page): text for page, text in page_markdown.items()}
    if not text_layer:
        return merged
    layer = {int(page): text or "" for page, text in text_layer.items()}
    for page in sorted(layer):
        page_text = layer[page]
        if not _has_court_columns(page_text):
            if _is_explicit_party_schedule(page_text):
                for follow, follow_text in _schedule_follow_pages(layer, page).items():
                    current = merged.get(follow) or ""
                    if follow == page and _is_explicit_party_schedule(current):
                        continue
                    if follow != page and current.strip():
                        continue
                    merged[follow] = follow_text
            continue
        for follow in range(page, page + 4):
            follow_text = layer.get(follow)
            if follow_text is None:
                break
            current = merged.get(follow) or ""
            # A long OCR dump can still omit the schedule. Prefer the text layer
            # until that schedule has already been read.
            if not _has_court_columns(current):
                merged[follow] = follow_text
            if re.search(r"most\s+respectfully\s+sh[eo]weth", follow_text, re.IGNORECASE):
                break
    return merged


def _court_column_parties(
    page_markdown: Mapping[int, str] | None,
) -> dict[str, list[dict[str, Any]]] | None:
    if not page_markdown:
        return None
    sorted_pages = sorted(page_markdown)
    for page in sorted_pages:
        text = page_markdown.get(page) or ""
        if not _has_court_columns(text):
            continue
        pages_to_check = [page]
        for next_page in range(page + 1, min(page + 4, max(sorted_pages) + 1)):
            pages_to_check.append(next_page)
            next_text = page_markdown.get(next_page) or ""
            if re.search(r"most\s+respectfully\s+sh[eo]weth", next_text, re.IGNORECASE):
                break
        combined = "\n".join(page_markdown.get(item, "") for item in pages_to_check)
        parsed = _parse_court_column_schedule(combined, pages_to_check)
        if parsed:
            return parsed
    return None


def _parties_from_memo(
    page_markdown: Mapping[int, str] | None,
) -> dict[str, list[dict[str, Any]]] | None:
    column = _court_column_parties(page_markdown)
    if column:
        return column
    if not page_markdown:
        return None
    sorted_pages = sorted(page_markdown)
    best_candidate: dict[str, list[dict[str, Any]]] | None = None
    best_score = -1

    for page in sorted_pages:
        text = page_markdown.get(page) or ""
        # Skip if page is clearly an index, synopsis, proforma, affidavit, or vakalatnama
        first_lines = "\n".join(text.splitlines()[:15]).lower()
        if re.search(
            r"\b(?:index\s+of\s+papers|index\s+of\s+filing|listing\s+proforma|proforma\s+for\s+first\s+listing|synopsis\s+and\s+list\s+of\s+dates|synopsis|list\s+of\s+dates|office\s+report\s+on\s+limitation|office\s+report|affidavit|vakalatnama)\b",
            first_lines,
        ) or re.search(r"\[index\s+of\s+filing\]", text, re.IGNORECASE):
            continue

        has_heading = False
        is_explicit_memo_heading = False
        collapsed = re.sub(r"<[^>]+>", " ", text)
        collapsed = re.sub(r"[*_`#]+", "", collapsed)
        collapsed = re.sub(r"\s+", " ", collapsed)
        if _PARTY_SCHEDULE_HEADING.search(collapsed):
            has_heading = True
            is_explicit_memo_heading = True
        for line in text.splitlines():
            line_clean = re.sub(r"<[^>]+>", " ", line)
            line_clean = re.sub(r"[*_`#]+", "", line_clean).strip()
            # Ignore lines that look like table-of-contents entries (e.g. "39. Memo of parties 234-224")
            if re.search(r"^\d+\s*[.)\-]", line_clean) or re.search(
                r"\d+\s*[-–—]\s*\d+\s*$", line_clean
            ):
                continue
            if re.search(
                r"\b(?:memo\s*of|posit[io1lt]{2,6}n?\s*of)\s*(?:the\s*)?parties\b",
                line_clean,
                re.IGNORECASE,
            ):
                has_heading = True
                is_explicit_memo_heading = True
                break

        if not has_heading:
            # Also recognize Main Petition opening with cause title and versus
            if (
                re.search(r"\b(?:in\s+the\s+matter\s+of|between)\b", text, re.IGNORECASE)
                and re.search(r"\b(?:versus|vs\.?|v/s|versuu)\b", text, re.IGNORECASE)
                and (
                    re.search(r"\b(?:supreme\s+court|high\s+court|jurisdiction|petition|appeal)\b", text, re.IGNORECASE)
                    or re.search(r"\b(?:petitioner|appellant|respondent)\b", text, re.IGNORECASE)
                )
                and not re.search(r"\b(?:for\s+index\s+(?:please|kindly)\s+see\s+inside|paper\s*book\b)", text, re.IGNORECASE)
            ):
                has_heading = True

        if not has_heading:
            continue

        pages_to_check = [page]
        for next_p in range(page + 1, min(page + 4, max(sorted_pages) + 1)):
            n_text = page_markdown.get(next_p) or ""
            pages_to_check.append(next_p)
            if re.search(
                r"(?:most\s+respectfully\s+sh[eo]weth|to\s*,\s*the\s+hon[\x27\x60’]?ble\s+chief\s+justice|1\.\s+that\s+the)",
                n_text,
                re.IGNORECASE,
            ):
                break

        combined_text = "\n".join(page_markdown.get(p, "") for p in pages_to_check)
        parsed = _parse_memo_of_parties(combined_text, page=page, pages=pages_to_check)
        pets = parsed.get("petitioners") or []
        resps = parsed.get("respondents") or []
        if not pets and not resps:
            continue

        # Score candidates: genuine memo pages have explicit heading, both parties, relations, addresses, versus
        score = 0
        if is_explicit_memo_heading:
            score += 25
        if pets and resps:
            score += 20
        elif pets or resps:
            score += 5

        for p in pets:
            if p.get("relation"):
                score += 5
            if p.get("address"):
                score += 5
        for r in resps:
            if r.get("relation") or r.get("acting_through"):
                score += 5
            if r.get("address"):
                score += 5
        if re.search(r"\b(?:versus|vs\.?|v/s|versuu)\b", combined_text, re.IGNORECASE):
            score += 10

        head = combined_text[:700].lower()
        if "high court" in head and "supreme court" not in head:
            score -= 20
        if "supreme court" in head and pets and resps:
            score += 15
        if any(party.get("address") for party in [*pets, *resps]):
            score += 10

        if pets and resps and (
            is_explicit_memo_heading
            or any(party.get("address") for party in [*pets, *resps])
        ):
            parsed = dict(parsed)
            parsed["_replace"] = True
        if score > best_score:
            best_score = score
            best_candidate = parsed

    return best_candidate


def _drop_non_party_names(parties: Any) -> list[dict[str, Any]]:
    if not isinstance(parties, list):
        return []
    kept: list[dict[str, Any]] = []
    for party in parties:
        if not isinstance(party, dict):
            continue
        name = _party_name(party)
        if not _looks_like_party_name(name):
            continue
        party = dict(party)
        party["name"] = _clean_front_party_name(name)
        if isinstance(party.get("address"), str):
            party["address"] = _clean_party_address(party["address"])
        kept.append(party)
    return kept


def _merge_memo_parties(
    existing: list[dict[str, Any]],
    memo_parties: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    """Merge memo parties into existing extract, preserving rich extract fields."""
    if not existing:
        return [dict(p) for p in memo_parties]
    if not memo_parties:
        return existing

    result: list[dict[str, Any]] = [dict(p) for p in existing]
    for memo_p in memo_parties:
        memo_name = _party_name(memo_p)
        matched = next(
            (p for p in result if _names_match(_party_name(p), memo_name)),
            None,
        )
        if matched is not None:
            if memo_p.get("name"):
                matched["name"] = memo_p["name"]
            for field in (
                "address",
                "relation",
                "guardian",
                "occupation",
                "age",
                "acting_through",
            ):
                if memo_p.get(field):
                    matched[field] = memo_p[field]
            if memo_p.get("source_part"):
                matched["source_part"] = memo_p["source_part"]
            if memo_p.get("source_pages"):
                matched["source_pages"] = memo_p["source_pages"]
        else:
            new_p = dict(memo_p)
            new_p["serial"] = len(result) + 1
            result.append(new_p)
    return result


def repair_front_parties(
    payload: dict[str, Any],
    page_markdown: Mapping[int, str] | None = None,
) -> None:
    """Keep real party names. Checklist lines and order paragraphs are not parties."""
    existing_petitioners = _drop_non_party_names(payload.get("petitioners"))
    existing_respondents = _drop_non_party_names(payload.get("respondents"))

    memo = _parties_from_memo(page_markdown)
    memo_petitioners = memo.get("petitioners") or [] if memo else []
    memo_respondents = memo.get("respondents") or [] if memo else []
    replace_with_schedule = bool(memo and memo.get("_replace"))

    if replace_with_schedule:
        payload["petitioners"] = _collapse_same_parties(memo_petitioners)
        payload["respondents"] = _collapse_same_parties(memo_respondents)
    else:
        payload["petitioners"] = _collapse_same_parties(
            _merge_memo_parties(existing_petitioners, memo_petitioners)
        )
        payload["respondents"] = _collapse_same_parties(
            _merge_memo_parties(existing_respondents, memo_respondents)
        )

    cause = payload.get("cause_title")
    if not isinstance(cause, dict):
        cause = {}
        payload["cause_title"] = cause
    petitioners = payload.get("petitioners") or []
    respondents = payload.get("respondents") or []
    if petitioners:
        cause["main_petitioner"] = _party_name(petitioners[0])
    if respondents:
        cause["main_respondent"] = _party_name(respondents[0])
    _normalize_parties(payload)
    _drop_resolved_acting_through_notes(payload)


def finalize_front_extract(
    payload: dict[str, Any],
    page_markdown: Mapping[int, str] | None = None,
) -> dict[str, Any]:
    """Keep the extract keys, put special_category beside petition_type, and clear later-page fields."""
    repair_front_parties(payload, page_markdown)
    _fill_aor_phones(payload, page_markdown)
    category = _printed_special_category(
        payload.get("special_category"),
        page_markdown,
        _impugned_category_text(payload),
    )
    ordered: dict[str, Any] = {}
    for key, value in payload.items():
        if key == "special_category":
            continue
        ordered[key] = value
        if key == "petition_type":
            ordered["special_category"] = category
    if "special_category" not in ordered:
        ordered["special_category"] = category
    relief = ordered.get("relief_sort")
    ordered["relief_sort"] = clean_relief_sort(relief) if isinstance(relief, str) else None
    dated = clean_petition_date(
        ordered.get("petition_date") if isinstance(ordered.get("petition_date"), str) else None
    )
    if not dated:
        dated = petition_date_from_opening(page_markdown)
    ordered["petition_date"] = dated
    ordered["documents"] = []
    counts = ordered.get("document_counts")
    if isinstance(counts, dict):
        ordered["document_counts"] = {"processed": 0, "failed": 0}
    return ordered


def stamp_review_status(extracted: dict[str, Any]) -> dict[str, Any]:
    """Keep LlamaExtract job status off the Agent Data review-status field."""
    payload = dict(extracted)
    current = str(payload.get("status") or "").strip().lower()
    if current in REVIEW_STATUSES:
        return payload
    metadata = payload.get("metadata")
    if not isinstance(metadata, dict):
        metadata = {}
    metadata["extract_status"] = payload.get("status")
    payload["metadata"] = metadata
    payload["status"] = "pending_review"
    return payload


def unwrap_extracted_record(extracted_data: Any) -> dict[str, Any]:
    """Agent Data stores ExtractedData; legal fields live on `.data`."""
    dumped = _as_dict(extracted_data)
    inner = dumped.get("data")
    if isinstance(inner, dict):
        return inner
    return dumped


def stamp_source_pages(record: dict[str, Any]) -> None:
    """Coerce source_pages on nested objects to integer lists."""

    def visit(node: Any) -> None:
        if isinstance(node, dict):
            if "source_pages" in node:
                node["source_pages"] = _pages_int(node.get("source_pages"))
            for value in node.values():
                visit(value)
        elif isinstance(node, list):
            for item in node:
                visit(item)

    visit(record)
