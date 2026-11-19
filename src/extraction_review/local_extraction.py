"""Conservative extraction from the persisted full-file ParseArtifact.

This module deliberately does not call an LLM.  A value is emitted only when a
configured, field-specific rule finds supporting text in the parsed Markdown.
Every emitted value carries its physical source page and confidence so the
result remains auditable.
"""

from __future__ import annotations

import html
import os
import re
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from typing import Any

from .extract_record import petition_date_from_opening, repair_front_parties

_SPACE = re.compile(r"[ \t]+")
_MARKUP = re.compile(r"(?:</?[^>]+>|!\[[^]]*]\([^)]*\)|\[([^]]+)]\([^)]*\))")
_DATE = r"(?:[0-3]?\d[./-][01]?\d[./-](?:19|20)?\d{2}|[0-3]?\d\s+(?:Jan(?:uary)?|Feb(?:ruary)?|Mar(?:ch)?|Apr(?:il)?|May|Jun(?:e)?|Jul(?:y)?|Aug(?:ust)?|Sep(?:tember)?|Oct(?:ober)?|Nov(?:ember)?|Dec(?:ember)?)[, ]+\d{4})"
_CASE_NUMBER = re.compile(
    r"\b(?:W\.?\s*P\.?|SLP|C\.?\s*A\.?|Crl\.?\s*A\.?|Civil\s+Appeal|"
    r"Criminal\s+Appeal|Review\s+Petition|R\.?\s*P\.?|LPA|FAO|RSA|OA|TA)"
    r"\s*(?:\([^)]{1,16}\))?\s*(?:No\.?\s*)?[A-Za-z0-9./() -]{1,55}?"
    r"(?:of|/)\s*\d{4}\b",
    re.IGNORECASE,
)
_COURT = re.compile(
    r"\b(?:IN\s+THE\s+)?((?:SUPREME\s+COURT\s+OF\s+INDIA)|"
    r"(?:HIGH\s+COURT\s+OF\s+[A-Z][A-Z .'-]{2,60})|"
    r"(?:HIGH\s+COURT\s+OF\s+JUDICATURE\s+AT\s+[A-Z][A-Z .'-]{2,60})|"
    r"(?:[A-Z][A-Z .'-]{2,50}\s+TRIBUNAL))\b",
    re.IGNORECASE,
)
_PETITION_TYPES: tuple[tuple[re.Pattern[str], str], ...] = (
    (
        re.compile(r"\bSPECIAL\s+LEAVE\s+PETITION\s*\(\s*CIVIL\s*\)", re.IGNORECASE),
        "Special Leave Petition (Civil)",
    ),
    (
        re.compile(
            r"\bSPECIAL\s+LEAVE\s+PETITION\s*\(\s*(?:CRL\.?|CRIMINAL)\s*\)",
            re.IGNORECASE,
        ),
        "Special Leave Petition (Criminal)",
    ),
    (
        re.compile(r"\bWRIT\s+PETITION\s*\(\s*CIVIL\s*\)", re.IGNORECASE),
        "Writ Petition (Civil)",
    ),
    (
        re.compile(r"\bWRIT\s+PETITION\s*\(\s*(?:CRL\.?|CRIMINAL)\s*\)", re.IGNORECASE),
        "Writ Petition (Criminal)",
    ),
    (
        re.compile(r"\bTRANSFER\s+PETITION\s*\(\s*CIVIL\s*\)", re.IGNORECASE),
        "Transfer Petition (Civil)",
    ),
    (
        re.compile(
            r"\bTRANSFER\s+PETITION\s*\(\s*(?:CRL\.?|CRIMINAL)\s*\)", re.IGNORECASE
        ),
        "Transfer Petition (Criminal)",
    ),
    (
        re.compile(r"\bREVIEW\s+PETITION\s*\(\s*CIVIL\s*\)", re.IGNORECASE),
        "Review Petition (Civil)",
    ),
    (
        re.compile(
            r"\bREVIEW\s+PETITION\s*\(\s*(?:CRL\.?|CRIMINAL)\s*\)", re.IGNORECASE
        ),
        "Review Petition (Criminal)",
    ),
    (
        re.compile(r"\bCURATIVE\s+PETITION\s*\(\s*CIVIL\s*\)", re.IGNORECASE),
        "Curative Petition (Civil)",
    ),
    (
        re.compile(
            r"\bCURATIVE\s+PETITION\s*\(\s*(?:CRL\.?|CRIMINAL)\s*\)", re.IGNORECASE
        ),
        "Curative Petition (Criminal)",
    ),
    (re.compile(r"\bCRIMINAL\s+APPEAL\b", re.IGNORECASE), "Criminal Appeal"),
    (re.compile(r"\bCIVIL\s+APPEAL\b", re.IGNORECASE), "Civil Appeal"),
    (
        re.compile(r"\bCONTEMPT\s+PETITION\s*\(\s*CIVIL\s*\)", re.IGNORECASE),
        "Contempt Petition (Civil)",
    ),
    (
        re.compile(
            r"\bCONTEMPT\s+PETITION\s*\(\s*(?:CRL\.?|CRIMINAL)\s*\)", re.IGNORECASE
        ),
        "Contempt Petition (Criminal)",
    ),
    (re.compile(r"\bARBITRATION\s+PETITION\b", re.IGNORECASE), "Arbitration Petition"),
    (
        re.compile(r"\b(?:MISCELLANEOUS|INTERLOCUTORY)\s+APPLICATION\b", re.IGNORECASE),
        "Miscellaneous Application",
    ),
)


def extraction_provider() -> str:
    value = os.getenv("EXTRACTION_PROVIDER", "local").strip().lower()
    if value not in {"local", "llamacloud"}:
        raise ValueError("EXTRACTION_PROVIDER must be 'local' or 'llamacloud'")
    return value


def _plain(markdown: str) -> str:
    text = html.unescape(markdown or "").replace("\r\n", "\n")
    text = _MARKUP.sub(lambda match: match.group(1) or "", text)
    lines = []
    for raw in text.splitlines():
        line = re.sub(r"^\s{0,3}(?:#{1,6}|[-*+]\s+|>\s*)", "", raw)
        line = re.sub(r"[*_`|]", " ", line)
        lines.append(_SPACE.sub(" ", line).strip())
    return "\n".join(lines)


def _parts(value: Any) -> set[str]:
    if isinstance(value, str):
        values: Sequence[Any] = [value]
    elif isinstance(value, Sequence):
        values = value
    else:
        values = []
    return {re.sub(r"[^a-z0-9]+", " ", str(item).lower()).strip() for item in values}


@dataclass
class LocalExtractionResult:
    data: dict[str, Any]
    confidence: dict[str, float] = field(default_factory=dict)
    provenance: dict[str, list[dict[str, Any]]] = field(default_factory=dict)

    @property
    def overall_confidence(self) -> float | None:
        values = list(self.confidence.values())
        return sum(values) / len(values) if values else None


class LocalParseExtractor:
    """Execute configured rules against all physical ParseArtifact pages."""

    def __init__(self, settings: Mapping[str, Any] | None = None):
        self.settings = dict(settings or {})
        configured = self.settings.get("enabled_fields")
        self.enabled = set(
            configured
            or (
                "court",
                "petition_type",
                "cause_title",
                "petitioners",
                "respondents",
                "advocates_on_record",
                "impugned_orders",
                "relief_sort",
                "petition_date",
                "special_category",
            )
        )
        self.max_field_chars = int(self.settings.get("max_field_chars") or 6000)

    def extract(
        self,
        *,
        page_markdown: Mapping[int, str],
        page_parts: Mapping[int, Any] | None = None,
        filing_type: str | None = None,
        front_extract: bool = False,
    ) -> LocalExtractionResult:
        pages = {
            int(page): _plain(text)
            for page, text in page_markdown.items()
            if str(text).strip()
        }
        ownership = {
            int(page): _parts(value) for page, value in (page_parts or {}).items()
        }
        result = LocalExtractionResult(data={})

        opening = self._ranked_pages(
            pages,
            ownership,
            {"cover page", "main petition", "petition", "synopsis"},
            30,
        )
        petition = self._ranked_pages(
            pages, ownership, {"main petition", "petition"}, None
        )
        challenged_order = self._ranked_pages(
            pages,
            ownership,
            {"impugned order", "impugned judgment", "judgment", "order"},
            None,
        )

        if "court" in self.enabled:
            found = self._first_match(opening, _COURT)
            if found:
                page, match = found
                self._record(
                    result,
                    "court",
                    self._court_name(match.group(1)),
                    page,
                    match.group(0),
                    0.96,
                )

        if "petition_type" in self.enabled:
            for pattern, label in _PETITION_TYPES:
                found = self._first_match(opening, pattern)
                if found:
                    page, match = found
                    self._record(
                        result, "petition_type", label, page, match.group(0), 0.97
                    )
                    break
            if "petition_type" not in result.data and filing_type:
                # Filing type is pipeline metadata, not extracted evidence. Keep it
                # only as a low-confidence fallback and identify that provenance.
                result.data["petition_type"] = filing_type
                result.confidence["petition_type"] = 0.70
                result.provenance["petition_type"] = [
                    {"source": "filing_type", "physical_page": None}
                ]

        if "relief_sort" in self.enabled:
            relief = self._relief(petition or opening)
            if relief:
                page, value, evidence = relief
                self._record(result, "relief_sort", value, page, evidence, 0.91)

        if "advocates_on_record" in self.enabled:
            aors = self._aors(petition or opening)
            if aors:
                result.data["advocates_on_record"] = aors
                result.confidence["advocates_on_record"] = 0.90
                result.provenance["advocates_on_record"] = [
                    {
                        "physical_page": item["source_pages"][0],
                        "evidence": item["raw_text"],
                    }
                    for item in aors
                ]

        if "impugned_orders" in self.enabled:
            orders = self._impugned_orders([*opening, *challenged_order])
            if orders:
                result.data["impugned_orders"] = orders
                result.confidence["impugned_orders"] = 0.91
                result.provenance["impugned_orders"] = [
                    {
                        "physical_page": item["source_pages"][0],
                        "evidence": item["raw_text"],
                    }
                    for item in orders
                ]

        # Existing deterministic party/date repairers are the canonical
        # implementation and already understand the project's party schedules.
        if front_extract and self.enabled & {
            "cause_title",
            "petitioners",
            "respondents",
        }:
            repair_front_parties(result.data, page_markdown)
            for key in ("cause_title", "petitioners", "respondents"):
                if key not in self.enabled:
                    result.data.pop(key, None)
        if "petition_date" in self.enabled:
            petition_date = petition_date_from_opening(page_markdown)
            if petition_date:
                result.data["petition_date"] = petition_date
        self._capture_repaired(result, "petitioners")
        self._capture_repaired(result, "respondents")
        self._capture_repaired(result, "cause_title")
        self._capture_repaired(result, "petition_date")
        return result

    @staticmethod
    def _ranked_pages(
        pages: Mapping[int, str],
        ownership: Mapping[int, set[str]],
        wanted: set[str],
        limit: int | None,
    ) -> list[tuple[int, str]]:
        numbered = sorted(pages.items())
        owned = [(p, text) for p, text in numbered if ownership.get(p, set()) & wanted]
        fallback = [(p, text) for p, text in numbered if not limit or p <= limit]
        seen: set[int] = set()
        return [
            (p, text)
            for p, text in [*owned, *fallback]
            if not (p in seen or seen.add(p))
        ]

    @staticmethod
    def _first_match(pages: Sequence[tuple[int, str]], pattern: re.Pattern[str]):
        for page, text in pages:
            match = pattern.search(text)
            if match:
                return page, match
        return None

    @staticmethod
    def _court_name(value: str) -> str:
        text = _SPACE.sub(" ", value).strip(" .,-")
        text = re.split(
            r"\s+in\s+(?=(?:W\.?\s*P\.?|C\.?\s*A\.?|CRL\.?|SLP|LPA|FAO|RSA)\b)",
            text,
            maxsplit=1,
            flags=re.IGNORECASE,
        )[0]
        if text.upper() == "SUPREME COURT OF INDIA":
            return "Supreme Court of India"
        return text.title().replace(" Of ", " of ").replace(" At ", " at ")

    def _record(
        self,
        result: LocalExtractionResult,
        field_name: str,
        value: Any,
        page: int,
        evidence: str,
        confidence: float,
    ) -> None:
        result.data[field_name] = value
        result.confidence[field_name] = confidence
        result.provenance[field_name] = [
            {
                "physical_page": page,
                "evidence": _SPACE.sub(" ", evidence).strip()[: self.max_field_chars],
            }
        ]

    def _relief(self, pages: Sequence[tuple[int, str]]) -> tuple[int, str, str] | None:
        heading = re.compile(
            r"(?:^|\n)\s*(?:MAIN\s+)?PRAYER\s*:?[ \t]*(?:\n|$)", re.IGNORECASE
        )
        stop = re.compile(
            r"(?:\n\s*(?:DRAWN\s*(?:&|AND)\s*FILED\s+BY|FILED\s+BY|PLACE\s*:|DATE(?:D|\s+DRAFTED|\s+FILED)?\s*:|VERIFICATION|AFFIDAVIT)\b)",
            re.IGNORECASE,
        )
        for page, text in reversed(pages):
            matches = list(heading.finditer(text))
            if not matches:
                continue
            body = text[matches[-1].end() :]
            end = stop.search(body)
            body = body[: end.start()] if end else body
            body = re.sub(r"\n{3,}", "\n\n", body).strip(" \n:;-")
            if 20 <= len(body) <= self.max_field_chars:
                return (
                    page,
                    body,
                    text[matches[-1].start() : matches[-1].end()] + body[:500],
                )
        return None

    def _aors(self, pages: Sequence[tuple[int, str]]) -> list[dict[str, Any]]:
        anchor = re.compile(
            r"ADVOCATE(?:S)?[- ]ON[- ]RECORD|\bAOR\b|DRAWN\s*(?:&|AND)\s*FILED\s+BY",
            re.IGNORECASE,
        )
        email_re = re.compile(r"[\w.+-]+@[\w.-]+\.[A-Za-z]{2,}")
        phone_re = re.compile(
            r"(?:Ph(?:one)?|Tel(?:ephone)?|Mobile|Mob)\s*[:.-]?\s*(\+?\d[\d ()-]{7,18}\d)",
            re.IGNORECASE,
        )
        code_re = re.compile(
            r"(?:AOR\s*(?:Code|No\.?|Reg(?:istration)?\.?\s*No\.?)|CC\s*No\.?)\s*[:.-]?\s*([A-Za-z0-9/-]+)",
            re.IGNORECASE,
        )
        found: list[dict[str, Any]] = []
        for page, text in pages:
            match = anchor.search(text)
            if not match:
                continue
            lines = [line.strip() for line in text.splitlines() if line.strip()]
            line_index = next(
                (i for i, line in enumerate(lines) if anchor.search(line)), 0
            )
            window = lines[max(0, line_index - 3) : min(len(lines), line_index + 10)]
            raw = "\n".join(window)
            name = None
            for candidate in reversed(window[: min(5, len(window))]):
                cleaned = re.sub(anchor, "", candidate).strip(" :-,()")
                if (
                    cleaned
                    and 2 <= len(cleaned.split()) <= 8
                    and not re.search(r"@|\d{4,}|CHAMBER|FILED", cleaned, re.IGNORECASE)
                ):
                    name = cleaned
                    break
            email = email_re.search(raw)
            phone = phone_re.search(raw)
            code = code_re.search(raw)
            chamber = next(
                (
                    line
                    for line in window
                    if re.search(
                        r"CHAMBER|OFFICE|BLOCK|SUPREME\s+COURT\s+COMPOUND",
                        line,
                        re.IGNORECASE,
                    )
                ),
                None,
            )
            if not any((name, email, phone, code, chamber)):
                continue
            item = {
                "name": name,
                "registration_number": code.group(1) if code else None,
                "email": email.group(0) if email else None,
                "mobile": phone.group(1).strip() if phone else None,
                "firm": None,
                "office_address": chamber,
                "source_part": "Opening pages or Closing pages",
                "raw_text": raw[: self.max_field_chars],
                "source_pages": [page],
                "confidence": "90%",
            }
            key = (item["name"], item["email"], item["registration_number"])
            if not any(
                (x["name"], x["email"], x["registration_number"]) == key for x in found
            ):
                found.append(item)
        return found[:5]

    def _impugned_orders(
        self, pages: Sequence[tuple[int, str]]
    ) -> list[dict[str, Any]]:
        trigger = re.compile(
            r"(?:impugned|challenged|under\s+challenge|against\s+the\s+(?:final\s+)?(?:judgment|order))",
            re.IGNORECASE,
        )
        for page, text in pages:
            match = trigger.search(text)
            if not match:
                continue
            raw = text[max(0, match.start() - 180) : match.end() + 700]
            date = re.search(_DATE, raw, re.IGNORECASE)
            case = _CASE_NUMBER.search(raw)
            courts = list(_COURT.finditer(raw))
            court = courts[-1] if courts else None
            if not any((date, case, court)):
                continue
            return [
                {
                    "is_primary": True,
                    "case_number": _SPACE.sub(" ", case.group(0)).strip()
                    if case
                    else None,
                    "order_date": date.group(0).strip() if date else None,
                    "Forum": self._court_name(court.group(1)) if court else None,
                    "bench": None,
                    "certified_copy_applied_on": None,
                    "certified_copy_obtained_on": None,
                    "raw_text": _SPACE.sub(" ", raw).strip()[: self.max_field_chars],
                    "source_part": "Opening pages",
                    "source_pages": [page],
                    "confidence": "91%",
                }
            ]
        return []

    @staticmethod
    def _capture_repaired(result: LocalExtractionResult, field_name: str) -> None:
        value = result.data.get(field_name)
        if value in (None, [], ""):
            return
        result.confidence.setdefault(field_name, 0.88)
        result.provenance.setdefault(
            field_name, [{"source": "deterministic_parse_rule", "physical_page": None}]
        )
