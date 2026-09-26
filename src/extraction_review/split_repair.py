"""Hybrid compiled-PDF split repair: anchors, court hierarchy, duplicates.

LlamaSplit is treated as a hint. Final page labels come from printed headings,
annexure stamps, and nesting rules so High Court exhibits never overwrite the
Supreme Court Main Petition / Vakalatnama slots.
"""

from __future__ import annotations

import re
from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any

from .document_parts import (
    ANNEXURE_FAMILY,
    MAIN_PETITION_PART,
    PagePartMap,
    _AFFIDAVIT_HEADING_RE,
    _CARRY_BLOCKING_PARTS,
    _contiguous_groups,
    _heading_window,
    _is_real_split_label,
    _is_sci_application_start,
    _looks_like_index_table,
    _memo_of_parties_heading,
    _vakalatnama_heading,
    annexure_label_from_text,
    annexure_mark_in_heading,
    annexure_ref_in_heading,
    collapse_repeated_split_pages,
    explode_repeating_split_parts,
    family_split_name,
    page_starts_application,
    parts_on_page,
)
from .split_audit import (
    IndexPrintedRow,
    aligned_index_printed_rows,
    collect_expected_annexures,
    collect_index_annexure_entries,
)

_SCI_CAPTION_RE = re.compile(r"in the supreme court of india", re.IGNORECASE)
# OCR often inserts punctuation inside words: S_UPRE1:IE, LISTIN.G, LEA VE.
_SCI_CAPTION_OCR_RE = re.compile(
    r"in\s+the\s+s\W*u\W*p\W*r\W*[eé0-9l:]+\W*m\W*e?\W*"
    r"c\W*[o0]\W*u\W*r\W*t\W*[o0]\W*f\W*i\W*n\W*d\W*i\W*a",
    re.IGNORECASE,
)
_HC_CAPTION_RE = re.compile(
    r"in the (?:hon'?ble\s+)?high court|"
    r"high court of judicature|"
    r"bench at\s+\w+",
    re.IGNORECASE,
)
_TRIBUNAL_CAPTION_RE = re.compile(
    r"before the (?:hon'?ble\s+)?(?:minister|tribunal|authority|registrar)|"
    r"revisional authority|"
    r"divisional joint registrar",
    re.IGNORECASE,
)
_OFFICE_REPORT_RE = re.compile(r"office report on limitation|o/?r on limitation", re.I)
_LISTING_RE = re.compile(
    r"proforma\s+for\s+first\s+listin\.?g?|"
    r"proforma\s+of\s+first\s+listing|"
    r"listing\s+proforma|listed\s+proforma|first\s+proforma|"
    r"performa\s+for\s+first\s+listing|"
    r"(?m:^\s*(?:proforma|performa)\s*$)",
    re.I,
)
# Printed folio letters (B, M, etc.) often occupy the line immediately before
# the section heading. Accept those prefixes so the heading remains an anchor.
_FOLIO_PREFIX = r"(?:(?:[A-Z]{1,2}|[IVX]{1,4})\s*\n\s*){0,2}"
_SYNOPSIS_RE = re.compile(rf"(?m)^\s*{_FOLIO_PREFIX}synopsis\b", re.I)
# Heading only. Affidavit verification and checklists mention "list of dates"
# in a sentence; that must not open the List of Dates slot.
_LOD_RE = re.compile(
    rf"(?m)^\s*{_FOLIO_PREFIX}(?:list of dates(?:\s*(?:and|&)\s*events)?|list of events)\b",
    re.I,
)
_APPENDIX_RE = re.compile(r"(?m)^\s*appendix\b", re.I)
_RECORD_RE = re.compile(
    # OCR: RECORD OF PROCE.J:o:DINGS / PROCEED1NGS
    r"record\s+of\s+proce[\w.:;]*",
    re.I,
)
_RECORD_NOTICE_RE = re.compile(
    r"whereas\s+the\s+petition|"
    r"listed\s+for\s+hearing\s+before\s+this\s+court|"
    r"court\s+was\s+pleased\s+to\s+pass|"
    r"delivery[_\s]*mode",
    re.I,
)
# Paper-book "Index of Record of Proceedings" blank form (dates/pages table).
_ROP_INDEX_FORM_RE = re.compile(
    r"date of record of proceedings|"
    r"dates?\s+of\s+(?:the\s+)?proceedings?\s+pages?|"
    r"sl\s*no\.?\s*.{0,40}date of record",
    re.I,
)
# Registry-issued SCI order sheet (often annexed as P-n after the petition).
# Do NOT match bare "Court No. 1, Shimla" inside Index rows — require ITEM NO /
# spaced SCI caption / petition(s) for SLP, optionally with Court No.
_SCI_COURT_ROP_RE = re.compile(
    # OCR: .TEl<1 NO_ 19 / ITEM N0.37 / ITEM NO_ 19
    r"(?:item|tel)\W*n[o0].{0,3}\d|"
    r"petition\(s\)\s+for\s+special\s+leave|"
    r"s\s+u\s+p\s+r\s+e\s+m\W*\s*c\s+o\s+u\s+r\s+t\s+o\s+f\s+i\s+n\s+d\s+i\s+a|"
    r"chamber\s+matter\s+section",
    re.I,
)
_FILING_MEMO_RE = re.compile(
    r"(?m)^\s*(?:filing memo|index of filing|filing index|index of documents)\b",
    re.I,
)
_AOR_CERT_RE = re.compile(
    r"confined\s+only\s+to\s+the\s+pleadings|"
    r"(?:^|\n)\s*(?:c\s+e\s+r\s+t\s+i\s+f\s+i\s+c\s+a\s+t\s+e|"
    # OCR often garbles Certificate → Q;RTIFICATE / C.RTIFICATE / CERTIFICA TE.
    r"[a-z]?\W{0,3}rtifica\s*te|certifica\s*te|certificate)\b",
    re.IGNORECASE,
)
# Calendar dates used to match Index annexure particulars to unstamped islands.
_DATE_NUMERIC_RE = re.compile(
    r"\b(?P<d>\d{1,2})[\s./\-]+(?P<m>\d{1,2})[\s./\-]+(?P<y>\d{2,4})\b"
)
_DATE_SPOKEN_RE = re.compile(
    r"\b(?P<d>\d{1,2})(?:st|nd|rd|th)?\s+"
    r"(?P<month>january|february|march|april|may|june|july|august|september|"
    r"october|november|december)\s*,?\s*(?P<y>\d{4})\b",
    re.IGNORECASE,
)
_MONTH_NUM = {
    "january": 1,
    "february": 2,
    "march": 3,
    "april": 4,
    "may": 5,
    "june": 6,
    "july": 7,
    "august": 8,
    "september": 9,
    "october": 10,
    "november": 11,
    "december": 12,
}
_SCI_CHECKLIST_RE = re.compile(
    r"(?m)advocate'?s?\s+check[\s\-]*list|check[\s\-]*list.*supreme court|"
    r"^\s*check[\s\-]*list\b",
    re.I,
)
_STATE_CHECKLIST_RE = re.compile(
    r"check[\s\-]*list.*(?:housing societ|co-?operative|registration of)|"
    r"commissioner for co-?operation",
    re.I,
)
# Listing Proforma page 2 (sections 8–11) is often misread as Advocate's Checklist.
_LISTING_CONTINUATION_RE = re.compile(
    r"land acquisition matters|tax matters|special category|"
    r"vehicle number\b|period of sentence undergone|"
    r"similar disposed of|not to be listed before",
    re.I,
)
_LISTING_FIELD_CUES = (
    r"\bcentral act\b|\bstate act\b|\bcentral rule\b|\bstate rule\b",
    r"impugned order dated|names of judges|tribunal\s*/\s*authority",
    r"nature of matter",
    r"name\s*\(?s\)? of petitioner|name\s*\(?s\)? of respondent",
    r"main category classification|sub classification",
    r"not to be listed before",
    r"similar disposed|similar pending",
    r"criminal matter|fir no\.?|police station|sentence awarded|sentence undergone",
    r"land acquisition matters|section 4 notification|tax matters|state the tax effect",
    r"special category|vehicle number|litigation on the same point of law",
)
_COVER_FOOTER_RE = re.compile(
    r"for index\s+(?:kindly|please)\s+see\s+inside|"
    r"\{\s*cover\s+page\s*\}|"
    r"cover\s+page\s+of\s+paper|"
    r"\bpaper\s+book\b",
    re.I,
)
# Form-28 party schedule (not the one-name-per-side paper-book cover).
_PARTY_SCHEDULE_RE = re.compile(
    r"before\s+high\s+court|before\s+supreme\s+court|"
    r"before\s+this\s+court|"
    r"respondent\s+no\.|petitioner\s+no\.|"
    r"positi[o0]n\s+of\s+pa|"
    r"(?:^|\n)\s*\d{1,2}\.\s*.{0,120}(?:s/?o|d/?o|w/?o|age\s+\d+)",
    re.I,
)
# Form-28 body cues. Do NOT use "prayer for interim relief" alone — covers print
# "(WITH PRAYER FOR INTERIM RELIEF)" in the caption.
_FORM28_BODY_RE = re.compile(
    r"form\s*28|"
    r"qu\W{0,4}stions?\s+of\s+law|questions\s+of\s+law|"
    r"most respectfully showeth|position of parties|positi[o0]n\s+of\s+pa|"
    r"humble petition of the|declaration in terms of rule",
    re.I,
)
# Document-start only — not Synopsis/LOD phrases like "true copy of the order … ANNEXURE".
_IMPUGNED_ORDER_RE = re.compile(
    r"(?m)^\s*(?:certified\s+copy\s+of\s+(?:the\s+)?)?impugned\s+"
    r"(?:final\s+)?(?:order|judgment|judgement)\b",
    re.I,
)
_INDEX_CONTINUATION_RE = re.compile(
    r"(?mi)^\s*\d{1,2}\.\s+.*"
    r"(?:special leave|petition|annexure|appendix|affidavit|"
    r"vakalat|application|memo of|office report|listing|synopsis)",
)
_APPEARANCE_RE = re.compile(
    r"(?mi)^\s*memo(?:randum)?\s+of\s+app.{0,4}rance\b", re.I
)
_NEAR_BLANK_RE = re.compile(r"^[\s\d\.]*$")

# Outer SCI paper-book parts: keep the first contiguous run, not the longest.
# (A later High Court writ mislabeled Main Petition must not win.)
# Index is omitted because multiple volume indexes are valid; mirrored in
# document_parts.collapse_repeated_split_pages.
_FIRST_RUN_PARTS = frozenset(
    {
        MAIN_PETITION_PART,
        "Cover Page",
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
    }
)

_NESTED_STEAL_PARTS = frozenset(
    {
        MAIN_PETITION_PART,
        "Affidavit",
        "Vakalatnama",
        "Memo of Appearance",
        "Memo of Parties",
        "Filing Memo",
        "AOR's Certificate",
        "Advocate's Checklist",
        "Impugned Order",
        "Cover Page",
        # These labels commonly recur inside reproduced lower-court records.
        "Index",
        "Synopsis",
        "List of Dates & Events",
    }
)

# These slots must not expand backward into unlabeled gaps (would swallow Main /
# annexure body into Impugned / Vakalatnama / AOR).
_NO_BACKWARD_CARRY_PARTS = frozenset(
    {
        "Impugned Order",
        "AOR's Certificate",
        "Affidavit",
        "Vakalatnama",
        "Memo of Appearance",
        "Filing Memo",
        "Memo of Parties",
        "Appendix",
        "Listing Proforma",
        "Synopsis",
        "List of Dates & Events",
        "Main Petition",
        "Office Report on Limitation",
    }
)


@dataclass(frozen=True)
class DuplicateSplitHit:
    part: str
    kept_span: tuple[int, int]
    duplicate_spans: tuple[tuple[int, int], ...]

    def as_dict(self) -> dict[str, Any]:
        return {
            "part": self.part,
            "kept_span": {"start_page": self.kept_span[0], "end_page": self.kept_span[1]},
            "duplicate_spans": [
                {"start_page": start, "end_page": end}
                for start, end in self.duplicate_spans
            ],
        }


def _fold(text: str) -> str:
    return re.sub(r"\s+", " ", (text or "").casefold()).strip()


def _is_sci_caption(text: str) -> bool:
    # OCR often inserts extra spaces/punctuation: "IN   THE    SUPREME", "S_UPRE1:IE".
    head = _fold(_heading_window(text, lines=12))
    if _SCI_CAPTION_RE.search(head):
        return True
    if _SCI_CAPTION_OCR_RE.search(head) or _SCI_CAPTION_OCR_RE.search(text[:900]):
        return True
    letters = re.sub(r"[^a-z0-9]+", "", head)
    if "inthesupremecourtofindia" in letters:
        return True
    # Digits/punctuation inside "supreme" (supre1ie, s_upreme, …).
    return bool(re.search(r"inthesupr[a-z0-9]{0,10}courtofindia", letters))


def _is_lower_court_caption(text: str) -> bool:
    head = _fold(_heading_window(text, lines=14))
    if _is_sci_caption(text):
        return False
    return bool(_HC_CAPTION_RE.search(head) or _TRIBUNAL_CAPTION_RE.search(head))


def _looks_like_rop_index_form(text: str) -> bool:
    """Blank Index of Record of Proceedings table in the paper-book front matter."""
    head = _heading_window(text, lines=12)
    if not _RECORD_RE.search(head):
        return False
    return bool(_ROP_INDEX_FORM_RE.search(text[:1800]))


def _looks_like_sci_court_rop_extract(text: str) -> bool:
    """Registry SCI order sheet headed RECORD OF PROCEEDINGS (often an Annexure)."""
    head = text[:2200]
    if not (_RECORD_RE.search(head) or _SCI_COURT_ROP_RE.search(head)):
        return False
    if _looks_like_rop_index_form(text):
        return False
    return bool(_SCI_COURT_ROP_RE.search(head))


def _looks_like_court_notice_or_rop(text: str) -> bool:
    """Registry notice / RoP extract — not Cover, Filing Memo, or Main Petition."""
    head = text[:2200]
    folded = _fold(head)
    if _looks_like_rop_index_form(text):
        return True
    if _looks_like_sci_court_rop_extract(text):
        return True
    if _RECORD_RE.search(_heading_window(text, lines=10)):
        return True
    if _RECORD_NOTICE_RE.search(head):
        # SLP captions alone are not RoP; require notice/hearing language.
        return bool(
            "whereas" in folded
            or "listed for hearing" in folded
            or "delivery" in folded
            or "issue notice" in folded
            or "pid:" in folded
        )
    return False



def _is_near_blank_page(text: str) -> bool:
    stripped = (text or "").strip()
    if len(stripped) <= 40 and _NEAR_BLANK_RE.match(stripped or ""):
        return True
    return len(stripped) < 8


def _looks_like_listing_proforma(text: str) -> bool:
    """Recognize the first-listing form by its heading or repeated field layout."""
    if _LISTING_RE.search(_heading_window(text, lines=10)) or _LISTING_RE.search(
        text[:900]
    ):
        return True
    folded = _fold(text[:4000])
    return sum(
        bool(re.search(pattern, folded, re.I)) for pattern in _LISTING_FIELD_CUES
    ) >= 3


def _looks_like_sci_checklist(text: str) -> bool:
    head = text[:1200]
    if _STATE_CHECKLIST_RE.search(head):
        return False
    # Listing Proforma also says "tick/check the correct box" — not a checklist.
    if _looks_like_listing_proforma(head) or _LISTING_CONTINUATION_RE.search(
        text[:2000]
    ):
        return False
    if _OFFICE_REPORT_RE.search(text[:2000]):
        return False
    if "proforma for first" in _fold(head) or "section -" in _fold(head[:400]):
        if "nature of matter" in _fold(head):
            return False
    # OCR can emit control characters in place of spaces (for example
    # ``ADVOCATE'S\x03CHECK\x03LIST\x03TO...``). Match a compact form too,
    # otherwise the checklist page can be mistaken for an Index continuation.
    compact = re.sub(r"[^a-z0-9]+", "", head.casefold())
    if _SCI_CHECKLIST_RE.search(head) or (
        "advocateschecklist" in compact
        and "advocateonrecord" in compact
    ):
        return True
    # OCR of ticked Advocate's Checklist is often only YES / N.A. answers.
    # Require a real checklist heading cue — Listing page-2 also has N.A. + AOR code.
    answers = re.findall(r"\b(?:yes|n/?a|n\.a\.?)\b", text or "", re.I)
    yes_count = sum(1 for token in answers if token.lower() == "yes")
    folded = _fold(text[:1200])
    checklist_cue = (
        "check list" in folded
        or "checklist" in folded
        or "advocate-on-record" in folded
        or "whether the petition" in folded
    )
    if not checklist_cue:
        return False
    if yes_count >= 4 and len(answers) >= 8:
        return True
    return yes_count >= 2 and len(answers) >= 3 and (
        "aor" in folded or "aor code" in folded or "advocate for petitioner" in folded
    )


def _looks_like_impugned_order_start(text: str) -> bool:
    """True only for a standalone Impugned Order document start.

    Annexed HC / tribunal judgments without an ``ANNEXURE P-n`` stamp must
    not be treated as the paper-book Impugned Order slot — those stay
    unlabeled (or stamped annexures) unless the page itself is titled
    Impugned Order / certified copy of the impugned order.
    """
    if annexure_mark_in_heading(text):
        return False
    folded = _fold(text[:1800])
    # Petition body / SLP caption pages often mention the impugned judgment.
    if any(
        cue in folded
        for cue in (
            "showeth",
            "humble petition",
            "position of parties",
            "positi",
            "questions of law",
            "qustions of law",
            "prayer for interim relief",
            "special leave petition",
            "special lea ve petition",
            "under article 136",
        )
    ):
        return False
    if "list of dates" in folded or folded.startswith("synopsis"):
        return False
    if "matter in issue" in folded or "res judicata" in folded:
        return False
    if _is_sci_caption(text) and (
        "special leave" in folded or "petition for special leave" in folded
    ):
        return False
    head = _heading_window(text, lines=12)
    if _IMPUGNED_ORDER_RE.search(head):
        if "annexure" in _fold(head) and "true copy" in _fold(head):
            return False
        if _is_lower_court_caption(text):
            return True
        if re.search(
            r"(?mi)certified\s+copy\s+of\s+(?:the\s+)?impugned", head
        ):
            return True
        # Bare title line at the top of the page.
        if re.match(
            r"(?is)^\s*(?:certified\s+copy\s+of\s+(?:the\s+)?)?impugned\s+"
            r"(?:final\s+)?(?:order|judgment|judgement)\b",
            head.strip()[:120],
        ):
            return True
        return False
    # Do not treat bare High Court / tribunal judgments as Impugned Order.
    # Those copies are usually Annexure P-n (stamp) or sit unlabeled until
    # the LOD→Main near-blank gap filler places the Impugned section.
    return False


def _looks_like_index_continuation(text: str) -> bool:
    """Index pages after the heading often omit the word INDEX."""
    if _looks_like_index_table(text):
        return True
    # List of Dates pages also use serial-numbered rows and cite Annexures.
    # A numbered row whose first field is a date/year is chronology, not Index.
    if _looks_like_lod_continuation(text):
        return False
    # Numbered paragraphs on OR / petition / HC bail pleadings are not Index.
    if _OFFICE_REPORT_RE.search(text[:2000]):
        return False
    if _LISTING_RE.search(_heading_window(text, lines=10)):
        return False
    if _SYNOPSIS_RE.search(_heading_window(text, lines=8)):
        return False
    if _LOD_RE.search(_heading_window(text, lines=8)):
        return False
    if _is_lower_court_caption(text):
        return False
    if _is_sci_caption(text) and _FORM28_BODY_RE.search(text[:3000]):
        return False
    folded = _fold(text[:1000])
    if re.search(r"(?m)^\s*\d{1,2}\.\s+that\s+the\b", text or "", re.I):
        return False
    if "anticipatory bail" in folded and "prayer" in folded:
        return False
    head = text[:2000]
    hits = _INDEX_CONTINUATION_RE.findall(head)
    if len(hits) >= 3:
        return True
    # Trailing Index rows often list I.A.s / Filing Memo / Vakalatnama.
    if (
        len(
            re.findall(
                r"(?mi)^\s*\d{1,2}\.\s+.*\b(?:i\.?\s*a\.?|filing memo|vakalat|"
                r"memo of parties|application for)\b",
                head,
            )
        )
        >= 2
    ):
        return True
    if re.search(r"(?mi)^\s*\d{1,2}\.\s+i\.?\s*a\.?\b", head) and (
        "application" in folded or "vakalat" in folded or "filing memo" in folded
    ):
        return True
    range_lines = re.findall(r"(?m)^\s*\d{1,3}\s*[-–—]\s*\d{1,3}\s*$", head)
    if len(range_lines) >= 4 and "annexure-p" not in _fold(head[:200]):
        return True
    return False


def _looks_like_lod_continuation(text: str) -> bool:
    """Recognize dated serial entries on continuation pages of a chronology."""
    return bool(
        re.search(
            r"(?mi)^\s*\d{1,3}[.)]?\s+(?:(?:\d{1,2}[./-]){2}\d{2,4}|\d{4})\b",
            text or "",
        )
    )


def _looks_like_checklist_closing_page(text: str) -> bool:
    """A checklist's final page may contain only its date and AOR signature."""
    folded = _fold(text[:1800])
    has_aor_block = bool(
        re.search(r"\b(?:aor|advocate(?:\s+for|\s*-on-record)?)\b", folded)
        and re.search(r"\b(?:code\s*no\.?|aor\s*code|cc\s*no\.?|signature|date)\b", folded)
    )
    return has_aor_block and not _looks_like_listing_proforma(text)


def _looks_like_filing_memo_continuation(text: str) -> bool:
    """A filing memo may end on a numbered list row on the next page."""
    head = text[:1800]
    return bool(
        re.search(r"(?mi)^\s*\d{1,3}[.)]?\s+", head)
        and re.search(r"\b(?:1\s*\+\s*1|copies|sets)\b", head, re.I)
        and re.search(
            r"\b(?:petition|annexure|application|vakalatnama|memo)\b", head, re.I
        )
    )


def _looks_like_cover_page(text: str) -> bool:
    page_head = text[:2200]
    folded = _fold(page_head)
    if _looks_like_court_notice_or_rop(text):
        return False
    # Multi-party Form-28 schedule is never the paper-book cover.
    if _PARTY_SCHEDULE_RE.search(page_head):
        return False
    has_cover_footer = bool(_COVER_FOOTER_RE.search(page_head))
    # Strong paper-book cover stamps win even when caption OCR is noisy.
    if has_cover_footer and (
        _is_sci_caption(text) or "supreme court of india" in folded
    ):
        # Still refuse real Form-28 / OR / Listing bodies.
        if "questions of law" in folded or "form 28" in folded:
            return False
        if "most respectfully showeth" in folded or "humble petition" in folded:
            return False
        if "position of parties" in folded or "positi" in folded:
            return False
        if _OFFICE_REPORT_RE.search(text[:2000]):
            return False
        if _LISTING_RE.search(text[:900]):
            return False
        return True

    if not _is_sci_caption(text):
        return False
    # Caption-only covers: reject Form-28 body, not the common cover line
    # "(WITH PRAYER FOR INTERIM RELIEF)".
    if "questions of law" in folded or "form 28" in folded:
        return False
    if "position of parties" in folded or "positi" in folded:
        return False
    if "most respectfully showeth" in folded or "humble petition" in folded:
        return False
    if _OFFICE_REPORT_RE.search(text[:2000]):
        return False
    if _LISTING_RE.search(text[:900]):
        return False
    if _is_sci_application_start(text) and "with" not in folded[:400]:
        return False
    head = _fold(_heading_window(text, lines=16))
    appellate = (
        "civil appellate jurisdiction" in head
        or "civl appellate jurisdiction" in head
        or "civil appeallate jurisdiction" in head
        or "criminal appellate jurisdiction" in head
        or "extra-ordinary appellate" in head
        or "extraordinary appellate" in head
    )
    # Without PAPER BOOK / index footer, appellate + SLP caption alone is too
    # weak — that pattern also opens the Main Petition party schedule.
    return bool(
        appellate
        and has_cover_footer
        and "questions of law" not in folded
    )


def _looks_like_sci_main_petition(text: str) -> bool:
    if _looks_like_cover_page(text):
        return False
    if _looks_like_court_notice_or_rop(text):
        return False
    window = text[:3000]
    folded = _fold(window)
    # The SLP's opening party schedule includes the impugned High Court case
    # number (often "CM Application No."), which can trigger the generic
    # Application detector. Its own heading and SLP caption are decisive.
    if (
        _is_sci_caption(text)
        and "position of parties" in folded
        and re.search(r"\b(?:s\.?\s*l\.?\s*p\.?|special leave petition)\b", folded)
    ):
        return True
    if page_starts_application(text):
        return False
    if _OFFICE_REPORT_RE.search(text[:2000]):
        return False
    if _LISTING_RE.search(text[:900]):
        return False
    # Synopsis / List of Dates narrative repeats "Special Leave Petition" — never
    # treat those headings as Form-28, or demote/collapse keep the wrong island.
    head8 = _heading_window(text, lines=8)
    if _SYNOPSIS_RE.search(head8) or _LOD_RE.search(head8):
        return False
    if _FILING_MEMO_RE.search(_heading_window(text, lines=6)):
        return False
    # The first Form-28 page commonly says "Humble Petition" and describes
    # the High Court order being challenged. That mention must not make this
    # Supreme Court petition look like a High Court document.
    sci_form28_opening = bool(
        "companion justices" in folded
        and ("humble petition" in folded or "most respectfully showeth" in folded)
        and ("special leave" in folded or "article 136" in folded)
    )
    if _is_lower_court_caption(text) and not sci_form28_opening:
        return False
    if _AFFIDAVIT_HEADING_RE.search(_heading_window(text, lines=20)):
        return False
    if _AOR_CERT_RE.search(text[:2500]):
        return False
    # Form-28 party schedule under SCI caption (names live here, not on Cover).
    if _is_sci_caption(text) and _PARTY_SCHEDULE_RE.search(window):
        return True
    if not _is_sci_caption(text):
        # Body pages often omit a fresh SCI caption after the party schedule.
        # Synopsis often has "questions of law" + "special leave" + "Petitioner
        # No." — that must NOT count as Form-28. Require hard Form-28 cues.
        # HC writs also say "showeth" — require an SCI signal with showeth.
        strong_body = (
            "most respectfully showeth" in folded
            or "humble petition of the" in folded
            or "position of parties" in folded
            or "declaration in terms of rule" in folded
            or "form 28" in folded
        )
        sci_signal = (
            "special leave" in folded
            or "supreme court of india" in folded
            or "companion justices" in folded
            or "slp (" in folded
            or "slp(civil)" in folded
            or "slp(criminal)" in folded
        )
        if _FORM28_BODY_RE.search(window) and strong_body:
            if sci_signal:
                return True
            if "declaration in terms of rule" in folded or "form 28" in folded:
                return True
        return False
    if _FORM28_BODY_RE.search(window):
        return True
    return (
        (
            "special leave petition" in folded
            or "special lea ve petition" in folded
            or "slp (criminal)" in folded
            or "slp (civil)" in folded
        )
        and (
            "humble petition" in folded
            or "showeth" in folded
            or "position of parties" in folded
            or "positi" in folded
            or _PARTY_SCHEDULE_RE.search(window)
        )
    )


def _outer_anchor_label(text: str) -> str | None:
    """Return a strong outer-document label from page text, or None."""
    if not (text or "").strip():
        return None
    # Advocate's Check List tables say "index" and "particulars" in the rows.
    # That is not the paper-book Index — classify the checklist first.
    if _looks_like_sci_checklist(text):
        return "Advocate's Checklist"
    # A paper-book Index can list Filing Memo, Annexures, and other sections.
    # Its table heading takes precedence over those row entries.
    if _looks_like_index_table(text):
        return "Index"
    # A Filing Memo is an outer section only when it belongs to the current
    # Supreme Court filing. A lower-court filing memo reproduced in an exhibit
    # must stay with that record.
    if _FILING_MEMO_RE.search(text[:1800]) and not (
        _looks_like_court_notice_or_rop(text)
    ) and _is_sci_caption(text) and not _is_lower_court_caption(text):
        return "Filing Memo"
    # A lower-court report reproduced in an exhibit is not the Supreme Court
    # paper-book's Office Report on Limitation.
    if _OFFICE_REPORT_RE.search(text[:2000]):
        if _is_lower_court_caption(text):
            return None
        return "Office Report on Limitation"
    if _looks_like_listing_proforma(text) or _LISTING_CONTINUATION_RE.search(
        text[:2000]
    ):
        return "Listing Proforma"
    if _looks_like_court_notice_or_rop(text):
        return "Record of Proceedings"
    if _RECORD_RE.search(_heading_window(text, lines=10)):
        return "Record of Proceedings"
    if _SYNOPSIS_RE.search(
        _heading_window(text, lines=8)
    ) and not _is_lower_court_caption(text):
        return "Synopsis"
    if _LOD_RE.search(
        _heading_window(text, lines=8)
    ) and not _is_lower_court_caption(text):
        return "List of Dates & Events"
    if _APPENDIX_RE.search(_heading_window(text, lines=6)) and not annexure_ref_in_heading(
        text
    ):
        return "Appendix"
    # Affidavit before AOR: curative affidavits often say "confined only to the
    # pleadings" and would otherwise steal the AOR Certificate slot.
    if _AFFIDAVIT_HEADING_RE.search(_heading_window(text, lines=20)) and (
        _is_sci_caption(text)
        or "deponent" in _fold(text[:1500])
        or "solemnly affirm" in _fold(text[:1500])
    ):
        return "Affidavit"
    if _AOR_CERT_RE.search(text[:2500]) and _is_sci_caption(text):
        # Require a Certificate title when the page is affidavit-like prose.
        cert_title = re.search(
            r"(?m)^\s*(?:c\s+e\s+r\s+t\s+i\s+f\s+i\s+c\s+a\s+t\s+e|"
            r"[a-z]?\W{0,3}rtifica\s*te|certifica\s*te|certificate)\b",
            _heading_window(text, lines=24),
            re.I,
        )
        if cert_title or "confined only to the pleadings" in _fold(text[:2500]):
            if cert_title or not _AFFIDAVIT_HEADING_RE.search(
                _heading_window(text, lines=20)
            ):
                return "AOR's Certificate"
    if _looks_like_cover_page(text):
        return "Cover Page"
    if (
        page_starts_application(text)
        and not _looks_like_cover_page(text)
        and not _looks_like_sci_main_petition(text)
    ):
        return "Application 1"
    if (
        _vakalatnama_heading(text)
        and _is_sci_caption(text)
        and not _is_lower_court_caption(text)
    ):
        return "Vakalatnama"
    if (
        _APPEARANCE_RE.search(_heading_window(text, lines=10))
        and _is_sci_caption(text)
        and not _is_lower_court_caption(text)
    ):
        return "Memo of Appearance"
    if (
        _memo_of_parties_heading(text)
        and _is_sci_caption(text)
        and not _is_lower_court_caption(text)
    ):
        return "Memo of Parties"
    if _looks_like_impugned_order_start(text):
        return "Impugned Order"
    if _looks_like_sci_main_petition(text):
        return MAIN_PETITION_PART
    if _looks_like_index_continuation(text):
        folded_body = _fold(text[:1200])
        # I.A. / petition paragraphs are not Index rows.
        if (
            "present application" in folded_body
            or "grave prejudice" in folded_body
            or "most respectfully showeth" in folded_body
            or re.search(
                r"(?mi)^\s*\d{1,2}\.\s+(?:that\s+(?:as\s+on|it\s+is)|it\s+is\s+stated)\b",
                text[:1200] or "",
            )
        ):
            pass
        else:
            return "Index"
    return None


def _annexure_run_bounds(
    page_text: Mapping[int, str], page_count: int
) -> list[tuple[int, int, str]]:
    """(start_page, end_page, label) for each printed annexure run.

    The last run must not swallow the rest of the PDF — later Annexure P-7+
    pages are often image-only with no E/P stamp, and Llama already numbered
    them. Cap trailing blank sheets so those later P-n labels survive.
    """
    starts: list[tuple[int, str]] = []
    for page in range(1, page_count + 1):
        label = annexure_label_from_text(page_text.get(page, ""))
        if label:
            starts.append((page, label))
    if not starts:
        return []

    runs: list[tuple[int, int, str]] = []
    for index, (start, label) in enumerate(starts):
        if index + 1 < len(starts):
            end = starts[index + 1][0] - 1
            # Stamps alone do not define the end — an I.A. / Vakalatnama /
            # Filing Memo between P-n and P-(n+1) must close the earlier run.
            for page in range(start + 1, end + 1):
                text = page_text.get(page, "")
                # Applications reproduced in a High Court record (e.g.
                # Order XXXIX / Section 151 CPC applications) are part of the
                # enclosing annexure. Only a current Supreme Court application
                # can terminate an outer annexure run.
                if page_starts_application(text) and _is_sci_caption(text):
                    end = page - 1
                    break
                if _outer_anchor_label(text) in {
                    "Vakalatnama",
                    "Memo of Appearance",
                    "Memo of Parties",
                    "Filing Memo",
                }:
                    end = page - 1
                    break
                if (
                    _FILING_MEMO_RE.search(_heading_window(text, lines=8))
                    and _is_sci_caption(text)
                    and not _is_lower_court_caption(text)
                ):
                    end = page - 1
                    break
        else:
            end = start
            blank_streak = 0
            for page in range(start + 1, page_count + 1):
                text = page_text.get(page, "")
                if page_starts_application(text) and _is_sci_caption(text):
                    break
                if _outer_anchor_label(text) in {
                    "Vakalatnama",
                    "Memo of Appearance",
                    "Memo of Parties",
                    "Filing Memo",
                }:
                    break
                if (
                    _FILING_MEMO_RE.search(_heading_window(text, lines=8))
                    and _is_sci_caption(text)
                    and not _is_lower_court_caption(text)
                ):
                    break
                if annexure_label_from_text(text):
                    break
                stripped = (text or "").strip()
                if len(stripped) < 40:
                    blank_streak += 1
                    # Keep at most one OCR-blank sheet inside an exhibit.
                    if blank_streak > 1:
                        break
                    end = page
                    continue
                blank_streak = 0
                end = page
        if end >= start:
            runs.append((start, end, label))
    return runs


def _annexure_number_from_label(name: str) -> int | None:
    match = re.fullmatch(r"annexure [a-z]-?(\d+)", _fold(name))
    if not match:
        return None
    number = int(match.group(1))
    return number if 1 <= number <= 999 else None


def _force_annexure_nesting(
    page_parts: PagePartMap,
    page_text: Mapping[int, str],
    page_count: int,
) -> PagePartMap:
    """Pages inside Annexure X-n keep that label even if they look like Main Petition.

    Image-only Llama Annexure P-7+ pages after the last printed stamp are kept.
    Blank sheets between earlier stamps (P-3 body) are not treated as P-7.
    """
    updated = {page: list(names) for page, names in page_parts.items()}
    runs = _annexure_run_bounds(page_text, page_count)
    last_stamp_page = max((start for start, _end, _label in runs), default=0)
    for start, end, label in runs:
        run_number = _annexure_number_from_label(label)
        for page in range(start, end + 1):
            text = page_text.get(page, "")
            # Never nest over a real SCI Form-28 / AOR / Affidavit start.
            if (
                _looks_like_sci_main_petition(text)
                or (
                    _AOR_CERT_RE.search(text[:2500])
                    and _is_sci_caption(text)
                )
                or (
                    _AFFIDAVIT_HEADING_RE.search(_heading_window(text, lines=20))
                    and _is_sci_caption(text)
                )
            ):
                continue
            names = parts_on_page(updated.get(page))
            # Keep SCI petition / certificate / affidavit body pages unless this
            # sheet itself prints an Annexure stamp.
            if names and not annexure_ref_in_heading(text):
                if any(
                    name
                    in {
                        MAIN_PETITION_PART,
                        "AOR's Certificate",
                        "Affidavit",
                        "Appendix",
                    }
                    for name in names
                ):
                    continue
            existing_annex = next(
                (
                    name
                    for name in names
                    if family_split_name(name) == ANNEXURE_FAMILY
                ),
                None,
            )
            if existing_annex:
                existing_number = _annexure_number_from_label(existing_annex)
                if (
                    existing_number is not None
                    and run_number is not None
                    and existing_number > run_number
                    and len((text or "").strip()) < 40
                    and page > last_stamp_page
                ):
                    continue
            # The outer paper-book Index is before any annexure run. An Index
            # encountered within this run is a reproduced lower-court index.
            if (
                names
                and all(name in _CARRY_BLOCKING_PARTS for name in names)
                and "Index" not in names
            ):
                if not annexure_ref_in_heading(text):
                    continue
            if not names or any(
                name in _NESTED_STEAL_PARTS or name == "Appendix" for name in names
            ):
                updated[page] = [label]
                continue
            if any(family_split_name(name) == ANNEXURE_FAMILY for name in names):
                updated[page] = [label]
                continue
            if _is_lower_court_caption(text) or annexure_ref_in_heading(text):
                updated[page] = [label]
    return updated


def _apply_outer_anchors(
    page_parts: PagePartMap,
    page_text: Mapping[int, str],
    page_count: int,
) -> PagePartMap:
    """Stamp strong outer headings onto pages outside annexure runs."""
    updated = {page: list(names) for page, names in page_parts.items()}
    annexure_pages: set[int] = set()
    for start, end, _label in _annexure_run_bounds(page_text, page_count):
        annexure_pages.update(range(start, end + 1))

    for page in range(1, page_count + 1):
        if page in annexure_pages:
            continue
        text = page_text.get(page, "")
        if _OFFICE_REPORT_RE.search(text[:2000]) and _is_lower_court_caption(text):
            # Remove a mistaken outer Office Report label. Gap filling can then
            # inherit the surrounding Annexure label when the exhibit is unstamped.
            names = parts_on_page(updated.get(page))
            remaining = [
                name for name in names if name != "Office Report on Limitation"
            ]
            if remaining:
                updated[page] = remaining
            elif page in updated:
                updated.pop(page)
            continue
        label = _outer_anchor_label(text)
        if not label:
            continue
        # Annexed SCI order sheets after Form-28 are not the paper-book ROP slot.
        if (
            label == "Record of Proceedings"
            and _looks_like_sci_court_rop_extract(text)
            and any(
                MAIN_PETITION_PART in parts_on_page(updated.get(prior))
                for prior in range(1, page)
            )
        ):
            continue
        # High Court memo of parties after annexures: keep as Memo of Parties
        # but do not treat HC caption pages as Main Petition (handled by nesting).
        if label == MAIN_PETITION_PART and _is_lower_court_caption(text):
            continue
        if label == "Advocate's Checklist" and _STATE_CHECKLIST_RE.search(text[:1200]):
            continue
        names = parts_on_page(updated.get(page))
        if not names or any(
            name in _NESTED_STEAL_PARTS
            or family_split_name(name) == ANNEXURE_FAMILY
            or not _is_real_split_label(name)
            for name in names
        ):
            # Never overwrite a Cover Page with Main Petition from a weak cue;
            # Cover is only replaced when the page itself is a Form-28 start.
            if (
                label == MAIN_PETITION_PART
                and "Cover Page" in names
                and not _looks_like_sci_main_petition(text)
            ):
                continue
            updated[page] = [label]
            continue
        # Upgrade weak/wrong labels when a strong outer heading is present.
        if label in {
            "Office Report on Limitation",
            "Listing Proforma",
            "Index",
            "Advocate's Checklist",
            "Appendix",
            "Record of Proceedings",
            "Synopsis",
            "List of Dates & Events",
            MAIN_PETITION_PART,
            "Cover Page",
        }:
            if label == MAIN_PETITION_PART and "Cover Page" in names:
                if not _looks_like_sci_main_petition(text):
                    continue
            updated[page] = [label]

    # Carry a Synopsis or List of Dates label across its lettered folios. The
    # explicit List of Dates heading switches the active section; this keeps a
    # B-L narrative Synopsis separate from an M-V date/event table even when
    # LlamaSplit assigns both runs the same label.
    active_front_matter: str | None = None
    active_checklist = False
    active_filing_memo = False
    for page in range(1, page_count + 1):
        if page in annexure_pages:
            active_front_matter = None
            active_checklist = False
            continue
        text = page_text.get(page, "")
        anchor = _outer_anchor_label(text)
        # A chronology continuation can look like numbered Index rows because
        # its events cite annexures. Once LIST OF DATES starts, dated serial
        # entries stay with that section until a new document heading appears.
        if (
            active_front_matter == "List of Dates & Events"
            and anchor == "Index"
            and _looks_like_lod_continuation(text)
        ):
            anchor = "List of Dates & Events"
        if (
            active_filing_memo
            and anchor in {None, "Index"}
            and _looks_like_filing_memo_continuation(text)
        ):
            anchor = "Filing Memo"
        if anchor in {"Synopsis", "List of Dates & Events"}:
            active_front_matter = anchor
            active_checklist = False
            active_filing_memo = False
            updated[page] = [anchor]
            continue
        if anchor == "Advocate's Checklist":
            active_front_matter = None
            active_checklist = True
            active_filing_memo = False
            updated[page] = [anchor]
            continue
        if anchor == "Filing Memo":
            active_front_matter = None
            active_checklist = False
            active_filing_memo = True
            updated[page] = [anchor]
            continue
        if anchor is not None:
            active_front_matter = None
            active_checklist = False
            active_filing_memo = False
            continue
        if active_checklist and _looks_like_checklist_closing_page(text):
            updated[page] = ["Advocate's Checklist"]
            continue
        if _is_lower_court_caption(text) or _looks_like_impugned_order_start(text):
            active_front_matter = None
            active_checklist = False
            active_filing_memo = False
            continue
        folio = re.match(r"(?is)^\s*([b-v])\b", text[:500])
        dated_row = bool(
            re.match(
                r"(?m)^\s*\d{1,2}[\s./\-]+\d{1,2}[\s./\-]+\d{2,4}\b",
                text[:500],
            )
        )
        is_continuation = bool(folio) or (
            active_front_matter == "List of Dates & Events"
            and (dated_row or _looks_like_lod_continuation(text))
        )
        if active_front_matter and is_continuation:
            updated[page] = [active_front_matter]
    # The standalone challenged judgment often has no printed heading saying
    # "Impugned Order". In the Supreme Court paper-book sequence it follows
    # the SLP's List of Dates and precedes Form 28. Recognize the first
    # identifiable lower-court judgment in that outer interval, while never
    # considering pages already bounded inside an Annexure.
    first_petition = next(
        (
            page
            for page in range(1, page_count + 1)
            if page not in annexure_pages
            and _looks_like_sci_main_petition(page_text.get(page, ""))
        ),
        page_count + 1,
    )
    lod_pages = [
        page
        for page in range(1, first_petition)
        if page not in annexure_pages
        and _outer_anchor_label(page_text.get(page, "")) == "List of Dates & Events"
    ]
    if lod_pages:
        chronology_end = max(lod_pages)
        for page in range(chronology_end + 1, first_petition):
            if page in annexure_pages:
                continue
            text = page_text.get(page, "")
            head = _heading_window(text, lines=20)
            folded = _fold(text[:2500])
            is_judgment_title = bool(
                _is_lower_court_caption(text)
                and re.search(r"\b(judg(?:e)?ment|order|decree)\b", folded)
                and (
                    "date of decision" in folded
                    or "reserved on" in folded
                    or "pronounced on" in folded
                    or re.search(r"\b(?:fa[o]?|lpa|w\.p\.|writ petition|civil appeal)\b", folded)
                )
            )
            if not is_judgment_title:
                continue
            updated[page] = ["Impugned Order"]
            total_pages = None
            footer = re.search(r"\bpage\s*1\s+of\s*(\d{1,3})\b", folded)
            if footer:
                total_pages = int(footer.group(1))
            if total_pages:
                last_page = min(first_petition - 1, page + total_pages - 1)
                for order_page in range(page + 1, last_page + 1):
                    if order_page in annexure_pages:
                        break
                    updated[order_page] = ["Impugned Order"]
            break
    return updated


def _keep_application_party_lists_nested(
    page_parts: PagePartMap,
    page_text: Mapping[int, str],
    page_count: int,
) -> PagePartMap:
    """Keep party lists and filing slips inside their enclosing application."""
    updated = {page: list(names) for page, names in page_parts.items()}
    annexure_pages: set[int] = set()
    for start, end, _label in _annexure_run_bounds(page_text, page_count):
        annexure_pages.update(range(start, end + 1))

    active_application: str | None = None
    for page in range(1, page_count + 1):
        if page in annexure_pages:
            active_application = None
            continue
        text = page_text.get(page, "")
        names = parts_on_page(updated.get(page))
        app_name = next(
            (
                name
                for name in names
                if family_split_name(name) == "Application"
                or re.fullmatch(r"(?i)application\s+\d+", name)
            ),
            None,
        )
        if page_starts_application(text) and not _is_lower_court_caption(text):
            active_application = app_name or "Application 1"
            continue
        if app_name:
            active_application = app_name
            continue
        if not active_application:
            continue

        anchor = _outer_anchor_label(text)
        if anchor == "Memo of Parties" and _memo_of_parties_heading(text):
            updated[page] = [active_application]
            continue
        if anchor == "Vakalatnama" and _vakalatnama_heading(text):
            updated[page] = [active_application]
            continue
        if anchor == "Memo of Appearance" and _APPEARANCE_RE.search(
            _heading_window(text, lines=10)
        ):
            updated[page] = [active_application]
            continue
        if anchor == "Filing Memo" and _FILING_MEMO_RE.search(text[:1800]):
            updated[page] = [active_application]
            continue
        # A new outer paper-book section ends the IA run. Annexures were
        # handled above so attached exhibits remain independently classified.
        if anchor and anchor != "Application 1":
            active_application = None
    return updated


def _demote_unverified_representation_parts(
    page_parts: PagePartMap,
    page_text: Mapping[int, str],
) -> PagePartMap:
    """A representation heading without this Court's caption is not an outer slot."""
    updated = {page: list(names) for page, names in page_parts.items()}
    representation_parts = {
        "Vakalatnama",
        "Memo of Appearance",
        "Memo of Parties",
    }
    for page, names in list(updated.items()):
        text = page_text.get(page, "")
        trusted = _is_sci_caption(text) and not _is_lower_court_caption(text)
        if trusted:
            continue
        remaining = [name for name in names if name not in representation_parts]
        if remaining:
            updated[page] = remaining
        else:
            updated.pop(page, None)
    return updated


def _first_outer_form28_page(
    page_text: Mapping[int, str],
    page_count: int,
) -> int | None:
    """Earliest Form-28 lookalike in the outer paper-book, not annexed copies.

    Annexed SLP / review order sheets later in the book often reprint a full
    SCI caption and would otherwise become ``form28_start``, causing demote to
    clear the real Main Petition body into Unidentified.
    """
    candidates = [
        page
        for page in range(1, page_count + 1)
        if _looks_like_sci_main_petition(page_text.get(page, ""))
    ]
    if not candidates:
        return None
    front = next(
        (
            page
            for page in range(1, page_count + 1)
            if _outer_anchor_label(page_text.get(page, ""))
            in {"Synopsis", "List of Dates & Events"}
        ),
        None,
    )
    annex_start = next(
        (
            page
            for page in range(1, page_count + 1)
            if annexure_mark_in_heading(page_text.get(page, ""))
        ),
        None,
    )
    scoped = candidates
    if front is not None:
        after_front = [page for page in scoped if page > front]
        if after_front:
            scoped = after_front
    if annex_start is not None:
        before_annex = [page for page in scoped if page < annex_start]
        if before_annex:
            scoped = before_annex
        else:
            # Only annexed SLP/order reprints look like Form-28 — ignore them so
            # demote does not wipe the real petition body before the annexures.
            return None
    return scoped[0] if scoped else None


def _demote_front_matter_mislabeled_as_main(
    page_parts: PagePartMap,
    page_text: Mapping[int, str],
    page_count: int,
) -> PagePartMap:
    """Clear Llama Main Petition tags on Synopsis/LOD pages before Form-28.

    Synopsis narrative often repeats \"Special Leave Petition\". LlamaSplit then
    tags those continuation pages as Main Petition. The false early island wins
    first-run collapse and erases the real Form-28 body later in the book
    (those pages then land in Unidentified / undefined).
    """
    updated = {page: list(names) for page, names in page_parts.items()}
    # Always peel strong Synopsis / LOD headings off Main, even when Form-28
    # OCR cues are missing — otherwise demote is a no-op and first-run collapse
    # keeps the synopsis island.
    for page in range(1, page_count + 1):
        names = parts_on_page(updated.get(page))
        if MAIN_PETITION_PART not in names:
            continue
        text = page_text.get(page, "")
        anchor = _outer_anchor_label(text)
        if anchor in {"Synopsis", "List of Dates & Events"}:
            updated[page] = [anchor]

    form28_start = _first_outer_form28_page(page_text, page_count)
    if form28_start is None:
        # No Form-28 cue: clear Main only on the first contiguous island that
        # overlaps Synopsis/LOD. Do not walk through Impugned blanks into a
        # later Main island (that drops the real petition into Unidentified).
        front_start = next(
            (
                page
                for page in range(1, page_count + 1)
                if _outer_anchor_label(page_text.get(page, ""))
                in {"Synopsis", "List of Dates & Events"}
            ),
            None,
        )
        if front_start is None:
            return updated
        main_pages = sorted(
            page
            for page in range(front_start, page_count + 1)
            if MAIN_PETITION_PART in parts_on_page(updated.get(page))
        )
        groups = _contiguous_groups(main_pages)
        if not groups:
            return updated
        start, end = groups[0]
        for page in range(start, end + 1):
            text = page_text.get(page, "")
            if _looks_like_sci_main_petition(text):
                break
            folded = _fold(text[:2500])
            if any(
                cue in folded
                for cue in (
                    "most respectfully showeth",
                    "position of parties",
                    "questions of law",
                    "declaration in terms",
                    "main prayer",
                    "interim prayer",
                    "humble petition",
                )
            ) and "synopsis" not in folded[:200]:
                break
            names = parts_on_page(updated.get(page))
            if MAIN_PETITION_PART not in names:
                continue
            anchor = _outer_anchor_label(text)
            if anchor and anchor != MAIN_PETITION_PART:
                updated[page] = [anchor]
            else:
                updated.pop(page, None)
        return updated

    for page in range(1, form28_start):
        names = parts_on_page(updated.get(page))
        if MAIN_PETITION_PART not in names:
            continue
        text = page_text.get(page, "")
        if _looks_like_sci_main_petition(text):
            continue
        anchor = _outer_anchor_label(text)
        if anchor and anchor != MAIN_PETITION_PART:
            updated[page] = [anchor]
            continue
        # Leave unlabeled so _fill_gaps can carry Synopsis / LOD / Impugned Order.
        updated.pop(page, None)
    return updated


def _prefer_form28_main_island(
    page_parts: PagePartMap,
    page_text: Mapping[int, str],
) -> PagePartMap:
    """Keep the Main Petition island that contains Form-28; drop Synopsis islands.

    First-run collapse keeps the earliest Main island. When Llama tags Synopsis
    as Main before the real Form-28, that early island wins and the petition
    pages become unlabeled → Unidentified. Prefer any island with a Form-28
    lookalike; otherwise prefer a later island that is not Synopsis/LOD-led.
    """
    updated = {page: list(names) for page, names in page_parts.items()}
    main_pages = sorted(
        page
        for page, names in updated.items()
        if MAIN_PETITION_PART in parts_on_page(names)
    )
    groups = _contiguous_groups(main_pages)
    if len(groups) < 2:
        return updated

    annex_start = next(
        (
            page
            for page in sorted(page_text)
            if annexure_mark_in_heading(page_text.get(page, ""))
        ),
        None,
    )

    def _score(group: tuple[int, int]) -> tuple[int, int, int, int, int]:
        start, end = group
        form28 = 0
        syn = 0
        for page in range(start, end + 1):
            text = page_text.get(page, "")
            if _looks_like_sci_main_petition(text):
                form28 += 1
            anchor = _outer_anchor_label(text)
            if anchor in {"Synopsis", "List of Dates & Events"}:
                syn += 1
        after_annex = bool(annex_start is not None and start >= annex_start)
        if form28 and not after_annex:
            # Earliest outer Form-28 island (later HC / annexed SLP copies lose).
            return (3, form28, -syn, -start, end - start)
        if form28 and after_annex:
            return (1, form28, -syn, -start, end - start)
        if syn:
            return (0, -syn, 0, start, end - start)
        # No cue: prefer later islands before annexures (front-matter false Main is early).
        if after_annex:
            return (0, 0, 0, start, end - start)
        return (2, 0, 0, start, end - start)

    keep_start, keep_end = max(groups, key=_score)
    for start, end in groups:
        if start == keep_start and end == keep_end:
            continue
        for page in range(start, end + 1):
            names = parts_on_page(updated.get(page))
            if MAIN_PETITION_PART not in names:
                continue
            text = page_text.get(page, "")
            anchor = _outer_anchor_label(text)
            if anchor and anchor != MAIN_PETITION_PART:
                updated[page] = [anchor]
                continue
            remaining = [name for name in names if name != MAIN_PETITION_PART]
            if remaining:
                updated[page] = remaining
            else:
                updated.pop(page, None)
    return updated


def _demote_annexed_sci_rop(
    page_parts: PagePartMap,
    page_text: Mapping[int, str],
    page_count: int,
) -> PagePartMap:
    """Drop outer ROP labels on annexed SCI order sheets after Main Petition.

    Paper-book Index of RoP is an early blank form. Later ``RECORD OF
    PROCEEDINGS`` / ITEM NO sheets are usually Annexure P-n copies of prior
    SLP/Review orders. Longest-run collapse otherwise puts those exhibits into
    the Record of Proceedings slot.
    """
    updated = {page: list(names) for page, names in page_parts.items()}
    main_pages = sorted(
        page
        for page, names in updated.items()
        if MAIN_PETITION_PART in parts_on_page(names)
    )
    if not main_pages:
        return updated
    cutoff = max(main_pages)
    for page in range(cutoff + 1, page_count + 1):
        text = page_text.get(page, "")
        names = parts_on_page(updated.get(page))
        is_rop_label = "Record of Proceedings" in names
        is_court_extract = _looks_like_sci_court_rop_extract(text)
        if not is_rop_label and not is_court_extract:
            continue
        if _looks_like_rop_index_form(text):
            continue
        if is_rop_label:
            remaining = [name for name in names if name != "Record of Proceedings"]
            if remaining:
                updated[page] = remaining
            else:
                updated.pop(page, None)
            continue
        # Fresh SCI court RoP heading after the petition: do not stamp outer ROP.
        if is_court_extract and not names:
            continue
    return updated


def _fill_gaps(
    page_parts: PagePartMap,
    page_text: Mapping[int, str],
    page_count: int,
) -> PagePartMap:
    updated = {page: list(names) for page, names in page_parts.items() if parts_on_page(names)}
    last_label: list[str] | None = None
    for page in range(1, page_count + 1):
        names = parts_on_page(updated.get(page))
        if names:
            last_label = list(names)
            continue
        text = page_text.get(page, "")
        if not last_label:
            continue
        if any(name in _CARRY_BLOCKING_PARTS for name in last_label):
            continue
        # Never invent labels across pages with no OCR text.
        if not (text or "").strip():
            continue
        # Do not extend Synopsis/LOD across OCR-blank sheets (Impugned Order scans).
        if _is_near_blank_page(text) and any(
            name in {"Synopsis", "List of Dates & Events"} for name in last_label
        ):
            continue
        if _looks_like_index_table(text) or _looks_like_index_continuation(text):
            continue
        if page_starts_application(text) or _vakalatnama_heading(text):
            continue
        if _outer_anchor_label(text):
            continue
        # Short slots must not absorb petition / HC / annexure body pages.
        if any(name in {"Affidavit", "Filing Memo"} for name in last_label):
            folded = _fold(text[:1800])
            if (
                _is_lower_court_caption(text)
                or _looks_like_sci_court_rop_extract(text)
                or "curative petition" in folded
                or "review petition" in folded
                or "because this" in folded
                or "most respectfully showeth" in folded
                or annexure_mark_in_heading(text)
            ):
                last_label = None
                continue
        # AOR Certificate body often narrates the curative/review petition — only
        # stop at a real following document start (HC / RoP / stamp / Form-28).
        if "AOR's Certificate" in last_label:
            if (
                _is_lower_court_caption(text)
                or _looks_like_sci_court_rop_extract(text)
                or annexure_mark_in_heading(text)
                or _looks_like_sci_main_petition(text)
                or _is_unstamped_annexure_island_start(text)
            ):
                # Do not resume Certificate after leaping an unlabeled island page.
                last_label = None
                continue
        if any(
            name in {"Vakalatnama", "Memo of Appearance"} for name in last_label
        ):
            folded = _fold(text[:1800])
            if (
                "because this" in folded
                or "most respectfully showeth" in folded
                or _is_lower_court_caption(text)
                or _looks_like_sci_main_petition(text)
            ):
                continue
        if "Impugned Order" in last_label:
            if _looks_like_sci_court_rop_extract(text) or (
                _is_sci_caption(text)
                and any(
                    cue in _fold(text[:1200])
                    for cue in ("review petition", "curative petition", "special leave")
                )
            ):
                continue
        # Do not bridge two islands of the same outer part (e.g. two Vakalatnamas).
        next_labeled: list[str] | None = None
        for ahead in range(page + 1, page_count + 1):
            ahead_names = parts_on_page(updated.get(ahead))
            if ahead_names:
                next_labeled = ahead_names
                break
        if next_labeled and (set(last_label) & set(next_labeled) & _FIRST_RUN_PARTS):
            continue
        updated[page] = list(last_label)

    for page in range(page_count, 0, -1):
        if parts_on_page(updated.get(page)):
            continue
        nxt = parts_on_page(updated.get(page + 1))
        if not nxt:
            continue
        # Front-matter slots must not expand backward into unlabeled gaps.
        if any(name in _CARRY_BLOCKING_PARTS for name in nxt):
            continue
        if any(name in _NO_BACKWARD_CARRY_PARTS for name in nxt):
            continue
        text = page_text.get(page, "")
        # Near-blank sheets between LOD and petition are Impugned Order scans.
        if _is_near_blank_page(text):
            continue
        if _looks_like_index_table(text) or _outer_anchor_label(text):
            continue
        prev_labeled: list[str] | None = None
        for back in range(page - 1, 0, -1):
            back_names = parts_on_page(updated.get(back))
            if back_names:
                prev_labeled = back_names
                break
        if prev_labeled and (set(nxt) & set(prev_labeled) & _FIRST_RUN_PARTS):
            continue
        updated[page] = list(nxt)
    return updated


def _looks_like_lod_or_synopsis_continuation(text: str) -> bool:
    """True for LOD date columns / Synopsis letter pages (B–V) without a heading."""
    head = (text or "")[:500]
    if re.match(r"(?is)^\s*[b-v]\b", head):
        return True
    if re.match(
        r"(?m)^\s*\d{1,2}[\s./\-]+\d{1,2}[\s./\-]+\d{2,4}\b",
        head,
    ):
        return True
    if _LOD_RE.search(head) or _SYNOPSIS_RE.search(head):
        return True
    return False


_AFFIDAVIT_BODY_RE = re.compile(
    r"deponent|verification|solemnly affirm|solemnly declare",
    re.I,
)


def _restore_split_preface(
    page_parts: PagePartMap,
    page_text: Mapping[int, str],
    page_count: int,
) -> PagePartMap:
    """Keep Synopsis, List of Dates, and Affidavit off the Impugned Order slot.

    LlamaSplit tags that preface as Impugned Order on arbitration and other
    filings that have no judgment under challenge. A near-blank scan between
    List of Dates and the petition can still be the impugned order. Affidavit
    verification that only mentions the List of Dates stays an affidavit.
    """
    updated = {page: list(names) for page, names in page_parts.items()}
    petition_at = page_count + 1
    for page in range(1, page_count + 1):
        if _looks_like_sci_main_petition(page_text.get(page, "")):
            petition_at = page
            break

    preface: str | None = None
    for page in range(1, petition_at):
        text = page_text.get(page, "")
        anchor = _outer_anchor_label(text)
        if anchor in {"Synopsis", "List of Dates & Events"}:
            preface = anchor
            updated[page] = [anchor]
            continue
        if _looks_like_impugned_order_start(text) or _is_lower_court_caption(text):
            preface = None
            continue
        if anchor in {
            MAIN_PETITION_PART,
            "Affidavit",
            "Cover Page",
            "Index",
            "Advocate's Checklist",
            "Office Report on Limitation",
            "Listing Proforma",
            "Record of Proceedings",
            "AOR's Certificate",
            "Filing Memo",
            "Vakalatnama",
            "Memo of Parties",
            "Appendix",
        }:
            preface = None
            continue
        if annexure_ref_in_heading(text) or page_starts_application(text):
            preface = None
            continue
        if not preface:
            continue
        if _is_near_blank_page(text) or _looks_like_garbled_scan_ocr(text):
            continue
        names = parts_on_page(updated.get(page))
        if "Impugned Order" in names:
            updated[page] = [preface]

    for page in range(2, page_count + 1):
        text = page_text.get(page, "")
        if annexure_ref_in_heading(text) or _looks_like_sci_main_petition(text):
            continue
        prev = parts_on_page(updated.get(page - 1))
        if "Affidavit" not in prev:
            continue
        names = parts_on_page(updated.get(page))
        heading_lod = bool(_LOD_RE.search(_heading_window(text, lines=8)))
        false_lod = "List of Dates & Events" in names and not heading_lod
        affidavit_body = bool(_AFFIDAVIT_BODY_RE.search(text[:2500] or ""))
        anchor = _outer_anchor_label(text)
        if anchor not in {None, "List of Dates & Events"}:
            continue
        if false_lod or (not names and affidavit_body) or (
            names == ["List of Dates & Events"] and affidavit_body
        ):
            updated[page] = ["Affidavit"]
    return updated


def _looks_like_garbled_scan_ocr(text: str) -> bool:
    """Image Impugned scans often decode as control-heavy gibberish."""
    raw = text or ""
    stripped = raw.strip()
    if not stripped:
        return True
    if len(stripped) < 40:
        return True
    weird = sum(1 for char in raw if ord(char) < 32 and char not in "\n\r\t")
    if weird >= 15:
        return True
    printable = sum(1 for char in raw if char.isprintable() or char in "\n\r\t")
    return len(raw) > 200 and (printable / len(raw)) < 0.88


def _label_near_blank_impugned_gap(
    page_parts: PagePartMap,
    page_text: Mapping[int, str],
    page_count: int,
) -> PagePartMap:
    """Impugned Order pages between LOD/Synopsis and Main Petition.

    Covers OCR-blank sheets and control-character gibberish from image scans.
    Must not carve real Synopsis letter pages or LOD date columns.
    """
    updated = {page: list(names) for page, names in page_parts.items() if parts_on_page(names)}
    petition_pages = sorted(
        page
        for page, names in updated.items()
        if MAIN_PETITION_PART in parts_on_page(names)
    )
    if not petition_pages:
        return updated
    petition_start = petition_pages[0]
    # Impugned sits after Synopsis/LOD. Without that preface, do not invent a
    # gap (avoids stealing a real Impugned island that appears after Main).
    if not any(
        any(
            name in {"Synopsis", "List of Dates & Events"}
            for name in parts_on_page(names)
        )
        for page, names in updated.items()
        if page < petition_start
    ):
        return updated

    def _is_impugned_gap_page(text: str) -> bool:
        if _looks_like_sci_main_petition(text):
            return False
        if _looks_like_lod_or_synopsis_continuation(text):
            return False
        if annexure_label_from_text(text) or page_starts_application(text):
            return False
        anchor = _outer_anchor_label(text)
        if anchor in {
            MAIN_PETITION_PART,
            "AOR's Certificate",
            "Appendix",
            "Cover Page",
            "Affidavit",
            "Index",
            "Listing Proforma",
            "Office Report on Limitation",
            "Synopsis",
            "List of Dates & Events",
        }:
            return False
        if _is_near_blank_page(text) or _looks_like_garbled_scan_ocr(text):
            return True
        return False

    gap_end = petition_start - 1
    gap_start = gap_end
    while gap_start >= 1:
        text = page_text.get(gap_start, "")
        names = parts_on_page(updated.get(gap_start))
        if names and not any(
            name in {"List of Dates & Events", "Synopsis", "Impugned Order"}
            for name in names
        ):
            break
        if names and any(
            name in {"List of Dates & Events", "Synopsis"} for name in names
        ):
            if not _is_impugned_gap_page(text):
                break
        elif names and "Impugned Order" in names:
            pass
        elif names:
            break
        elif not _is_impugned_gap_page(text):
            break
        gap_start -= 1
    gap_start += 1
    if gap_start > gap_end:
        return updated
    for page in range(gap_start, petition_start):
        if _is_impugned_gap_page(page_text.get(page, "")):
            updated[page] = ["Impugned Order"]
    return updated


def _demote_cover_mislabeled_as_main(
    page_parts: PagePartMap, page_text: Mapping[int, str]
) -> PagePartMap:
    """Llama often tags the paper-book cover as Main Petition."""
    updated = {page: list(names) for page, names in page_parts.items()}
    for page, names in list(updated.items()):
        if MAIN_PETITION_PART not in parts_on_page(names):
            continue
        text = page_text.get(page, "")
        if _looks_like_cover_page(text):
            updated[page] = ["Cover Page"]
    return updated


def _extend_main_petition_body(
    page_parts: PagePartMap,
    page_text: Mapping[int, str],
    page_count: int,
) -> PagePartMap:
    """Grow Main Petition from its Form-28 start through body pages."""
    updated = {page: list(names) for page, names in page_parts.items() if parts_on_page(names)}
    start = _first_outer_form28_page(page_text, page_count)
    if start is None:
        starts = [
            page
            for page, names in updated.items()
            if MAIN_PETITION_PART in parts_on_page(names)
            and not _looks_like_cover_page(page_text.get(page, ""))
            and _outer_anchor_label(page_text.get(page, ""))
            not in {"Synopsis", "List of Dates & Events"}
        ]
        if not starts:
            return updated
        start = min(starts)
    updated[start] = [MAIN_PETITION_PART]
    stop_labels = {
        "AOR's Certificate",
        "Affidavit",
        "Appendix",
        "Filing Memo",
        "Vakalatnama",
        "Memo of Appearance",
        "Memo of Parties",
        "Application 1",
        "Cover Page",
        "Index",
        "Office Report on Limitation",
        "Listing Proforma",
        "Record of Proceedings",
        "Advocate's Checklist",
        "Impugned Order",
        "Synopsis",
        "List of Dates & Events",
    }
    for page in range(start + 1, page_count + 1):
        text = page_text.get(page, "")
        if annexure_mark_in_heading(text):
            break
        if page_starts_application(text):
            break
        if _vakalatnama_heading(text) and _is_sci_caption(text):
            break
        label = _outer_anchor_label(text)
        if label and label != MAIN_PETITION_PART and label in stop_labels:
            break
        if label and family_split_name(label) == ANNEXURE_FAMILY:
            break
        names = parts_on_page(updated.get(page))
        if names and any(
            name in stop_labels or family_split_name(name) == ANNEXURE_FAMILY
            for name in names
        ):
            # Allow overwriting weak wrong labels inside the petition body.
            if any(name == MAIN_PETITION_PART for name in names):
                continue
            if any(name in _NESTED_STEAL_PARTS for name in names):
                updated[page] = [MAIN_PETITION_PART]
                continue
            break
        if not (text or "").strip():
            # Scanned books often have OCR-blank sheets inside Main Petition.
            ahead_main = False
            for ahead in range(page + 1, min(page + 8, page_count + 1)):
                ahead_text = page_text.get(ahead, "")
                ahead_label = _outer_anchor_label(ahead_text)
                if ahead_label and ahead_label in stop_labels:
                    break
                if annexure_mark_in_heading(ahead_text):
                    break
                if _looks_like_sci_main_petition(ahead_text):
                    ahead_main = True
                    break
                ahead_fold = _fold(ahead_text[:1200])
                if ahead_fold and any(
                    cue in ahead_fold
                    for cue in (
                        "prayer",
                        "grounds",
                        "showeth",
                        "questions of law",
                        "stions of law",
                        "declaration in terms",
                        "because the",
                        "because,",
                    )
                ):
                    ahead_main = True
                    break
            if ahead_main:
                updated[page] = [MAIN_PETITION_PART]
                continue
            break
        # After a Form-28 start, body pages (GROUNDS A/B/C, prayers) often omit
        # another caption. Keep extending until a hard stop above — do not
        # require "grounds"/"prayer" on every sheet.
        updated[page] = [MAIN_PETITION_PART]
    return updated


def _demote_false_advocate_checklist(
    page_parts: PagePartMap, page_text: Mapping[int, str]
) -> PagePartMap:
    updated = {page: list(names) for page, names in page_parts.items()}
    for page, names in list(updated.items()):
        labels = parts_on_page(names)
        text = page_text.get(page, "")
        if "Advocate's Checklist" in labels:
            if (
                _STATE_CHECKLIST_RE.search(text[:1500])
                or annexure_mark_in_heading(text)
                or _LISTING_RE.search(text[:900])
                or _LISTING_CONTINUATION_RE.search(text[:2000])
                or _looks_like_court_notice_or_rop(text)
            ):
                mark = annexure_ref_in_heading(text)
                if mark:
                    updated[page] = [mark.label]
                elif _LISTING_RE.search(text[:900]) or _LISTING_CONTINUATION_RE.search(
                    text[:2000]
                ):
                    updated[page] = ["Listing Proforma"]
                elif _looks_like_court_notice_or_rop(text):
                    updated[page] = ["Record of Proceedings"]
                else:
                    updated[page] = ["Annexure P-1"]
            elif not _looks_like_sci_checklist(text):
                # Blank / OCR-empty sheets are not the SCI Advocate's Check List.
                # Listing Proforma often continues on the next (image-only) page.
                if _is_near_blank_page(text) or not (text or "").strip():
                    prev = parts_on_page(updated.get(page - 1))
                    nxt = parts_on_page(updated.get(page + 1))
                    if "Listing Proforma" in prev or "Listing Proforma" in nxt:
                        updated[page] = ["Listing Proforma"]
                    else:
                        updated.pop(page, None)
                    continue
                # Llama invented Checklist on a page that is not one.
                anchor = _outer_anchor_label(text)
                if anchor and anchor != "Advocate's Checklist":
                    updated[page] = [anchor]
        if "Filing Memo" in labels and (
            _looks_like_court_notice_or_rop(text)
            or _LISTING_RE.search(text[:900])
            or _looks_like_sci_main_petition(text)
        ):
            anchor = _outer_anchor_label(text)
            if anchor:
                updated[page] = [anchor]
        if "Impugned Order" in labels and not _looks_like_impugned_order_start(text):
            anchor = _outer_anchor_label(text)
            if anchor and anchor != "Impugned Order":
                updated[page] = [anchor]
            elif _looks_like_sci_main_petition(text):
                updated[page] = [MAIN_PETITION_PART]
    return updated


def _preserve_later_llama_annexures(
    page_parts: PagePartMap,
    page_text: Mapping[int, str],
    page_count: int,
) -> PagePartMap:
    """Keep Llama Annexure P-n that sit after the last printed stamp.

    Printed ANNEXURE-E-1…E-6 become P-1…P-6. Image-only later exhibits are
    often only numbered by Llama (P-7, P-8, …). Extend each such island across
    adjacent near-blank sheets so multi-page P-7 is not reduced to one page.
    """
    updated = {page: list(names) for page, names in page_parts.items()}
    last_stamp = 0
    last_stamp_number = 0
    for page in range(1, page_count + 1):
        mark = annexure_ref_in_heading(page_text.get(page, ""))
        if mark:
            last_stamp = page
            last_stamp_number = max(last_stamp_number, mark.number)

    seeds: list[tuple[int, str, int]] = []
    for page, names in updated.items():
        if page <= last_stamp:
            continue
        for name in parts_on_page(names):
            number = _annexure_number_from_label(name)
            if number is not None and number > last_stamp_number:
                seeds.append((page, name, number))
                break

    for start, label, _number in seeds:
        updated[start] = [label]
        for page in range(start - 1, last_stamp, -1):
            text = page_text.get(page, "")
            names = parts_on_page(updated.get(page))
            if names and family_split_name(names[0]) == ANNEXURE_FAMILY:
                existing = _annexure_number_from_label(names[0])
                if existing is not None and existing <= last_stamp_number:
                    break
            if names and family_split_name(names[0]) != ANNEXURE_FAMILY:
                break
            if names and names[0] == label:
                continue
            if not names and (
                _is_near_blank_page(text) or not (text or "").strip()
            ):
                # Do not walk forever into the gap before this annexure.
                break
            break
        for page in range(start + 1, page_count + 1):
            text = page_text.get(page, "")
            names = parts_on_page(updated.get(page))
            if names:
                if names[0] == label:
                    continue
                other = _annexure_number_from_label(names[0])
                if other is not None:
                    break
                if family_split_name(names[0]) != ANNEXURE_FAMILY:
                    break
                break
            if _is_near_blank_page(text) or not (text or "").strip():
                updated[page] = [label]
                continue
            # Text on the next sheet stops this annexure body.
            break
    return updated


def collapse_repeated_split_pages_first_outer(page_parts: PagePartMap) -> PagePartMap:
    """Delegate to document_parts collapse (first-run outer parts)."""
    return collapse_repeated_split_pages(page_parts)


def find_duplicate_split_parts(page_parts: PagePartMap) -> list[DuplicateSplitHit]:
    """Outer parts with more than one contiguous island before collapse."""
    pages_by_part: dict[str, list[int]] = {}
    for page, names in page_parts.items():
        for part in parts_on_page(names):
            pages_by_part.setdefault(part, []).append(page)

    hits: list[DuplicateSplitHit] = []
    for part, pages in sorted(pages_by_part.items()):
        groups = _contiguous_groups(sorted(set(pages)))
        if len(groups) < 2:
            continue
        if part not in _FIRST_RUN_PARTS and not re.fullmatch(
            r"(?i)annexure p-?\d+|application \d+", part
        ):
            continue
        kept = groups[0]
        dupes = tuple(groups[1:])
        hits.append(DuplicateSplitHit(part=part, kept_span=kept, duplicate_spans=dupes))
    return hits


def _normalize_year(year: int) -> int:
    if year < 100:
        return 2000 + year if year < 70 else 1900 + year
    return year


def _extract_date_keys(text: str) -> set[tuple[int, int, int]]:
    """Normalize calendar dates in text to (year, month, day) keys."""
    found: set[tuple[int, int, int]] = set()
    for match in _DATE_NUMERIC_RE.finditer(text or ""):
        day = int(match.group("d"))
        month = int(match.group("m"))
        year = _normalize_year(int(match.group("y")))
        if 1 <= month <= 12 and 1 <= day <= 31 and 1900 <= year <= 2100:
            found.add((year, month, day))
            # OCR often drops a leading digit (16/03 → 6/03).
            if day < 10:
                found.add((year, month, day + 10))
    for match in _DATE_SPOKEN_RE.finditer(text or ""):
        month = _MONTH_NUM.get(match.group("month").lower())
        if not month:
            continue
        day = int(match.group("d"))
        year = int(match.group("y"))
        if 1 <= day <= 31:
            found.add((year, month, day))
    # OCR ordinals: "20 th", "201h", "20lh" February.
    for match in _DATE_SPOKEN_OCR_RE.finditer(text or ""):
        month = _MONTH_NUM.get(match.group("month").lower())
        if not month:
            continue
        day = int(match.group("d"))
        year = int(match.group("y"))
        if 1 <= day <= 31:
            found.add((year, month, day))
    return found


# Adjournment / listing phrasing — dates here are not the order's pronouncement date.
_ADJOURNMENT_DATE_RE = re.compile(
    r"(?:stand\s+over\s+to|(?:stand\s+)?over\s+to|adjourned\s+to|"
    r"listed\s+(?:on|for)|"
    r"next\s+date(?:\s+of\s+hearing)?\s*(?:is|:)?|"
    r"come\s+up\s+(?:on|for)|posted\s+(?:to|on))\s*"
    r"(?P<body>"
    r"\d{1,2}[\s./\-]+\d{1,2}[\s./\-]+\d{2,4}|"
    r"\d{1,2}\s*(?:st|nd|rd|th|h)?\s*"
    r"(?:january|february|march|april|may|june|july|august|september|"
    r"october|november|december)\s*,?\s*\d{4}"
    r")",
    re.IGNORECASE,
)
# Pronounced / caption date on HC / tribunal orders (DATE: 20th FEBRUARY, 2015).
# OCR often yields "201h", "20 th", "20lh" for "20th".
_PRONOUNCED_DATE_RE = re.compile(
    r"(?m)^\s*date\s*[:.\-–—]?\s*"
    r"(?P<body>"
    r"\d{1,2}[\s./\-]+\d{1,2}[\s./\-]+\d{2,4}|"
    r"\d{1,2}\s*(?:st|nd|rd|th|h|lh|1h)?\s*"
    r"(?:january|february|march|april|may|june|july|august|september|"
    r"october|november|december)\s*,?\s*\d{4}"
    r")",
    re.IGNORECASE,
)
_DATE_SPOKEN_OCR_RE = re.compile(
    r"\b(?P<d>\d{1,2})\s*(?:st|nd|rd|th|h|lh|1h)?\s+"
    r"(?P<month>january|february|march|april|may|june|july|august|september|"
    r"october|november|december)\s*,?\s*(?P<y>\d{4})\b",
    re.IGNORECASE,
)


def _strip_adjournment_date_phrases(text: str) -> str:
    """Remove stand-over / listed-on phrases so Index dates do not false-match."""
    if not text:
        return ""
    return _ADJOURNMENT_DATE_RE.sub(" ", text)


def _extract_island_match_date_keys(text: str) -> set[tuple[int, int, int]]:
    """Dates useful for Index matching — exclude adjournment / next-listing dates."""
    return _extract_date_keys(_strip_adjournment_date_phrases(text or ""))


def _extract_pronounced_date_keys(text: str) -> set[tuple[int, int, int]]:
    """Order date from a DATE: / Date caption line (strongest Index match signal)."""
    found: set[tuple[int, int, int]] = set()
    for match in _PRONOUNCED_DATE_RE.finditer(text or ""):
        found |= _extract_date_keys(match.group("body") or "")
    return found


def _post_petition_zone_start(page_parts: PagePartMap, page_count: int) -> int | None:
    """First page after Affidavit / AOR Certificate / Appendix (annexure zone)."""
    markers = {
        "Affidavit",
        "AOR's Certificate",
        "Appendix",
        MAIN_PETITION_PART,
    }
    last = 0
    for page, names in page_parts.items():
        labels = parts_on_page(names)
        if any(name in markers for name in labels):
            last = max(last, int(page))
    if last <= 0 or last >= page_count:
        return None
    return last + 1


def _is_unstamped_annexure_island_start(text: str) -> bool:
    """True when an unstamped exhibit document likely begins on this page."""
    if annexure_label_from_text(text):
        return False
    if page_starts_application(text):
        return False
    if _vakalatnama_heading(text) and _is_sci_caption(text):
        return False
    if _FILING_MEMO_RE.search(_heading_window(text, lines=8)):
        return False
    if _looks_like_sci_court_rop_extract(text):
        return True
    if _is_lower_court_caption(text):
        return True
    # Annexed SCI review / order sheets (caption + ORDER) after the petition.
    if _is_sci_caption(text) and not _looks_like_sci_main_petition(text):
        head = _heading_window(text, lines=24)
        if _AOR_CERT_RE.search(text[:2500]):
            return False
        if _AFFIDAVIT_HEADING_RE.search(head):
            return False
        if re.search(r"(?mi)^\s*0?\s*r\s*d\s*e\s*r\b|^\s*order\b", head):
            return True
        if "review petition" in _fold(head) and "order" in _fold(text[:1200]):
            return True
    return False


def _score_island_for_index_annexure(
    island_text: str, particulars: str, label: str
) -> int:
    """Score how well an island matches an Index annexure row."""
    score = 0
    folded_island = _fold(island_text)
    folded_part = _fold(particulars or "")
    # Narrative Certificate / petition pages cite annexure dates — do not match.
    if _AOR_CERT_RE.search(island_text[:2500]) and _is_sci_caption(island_text):
        return -1
    if _looks_like_sci_main_petition(island_text):
        return -1

    is_hc = _is_lower_court_caption(island_text)
    is_rop = _looks_like_sci_court_rop_extract(island_text)
    is_review = bool(
        "review petition" in folded_island
        or re.search(r"\br\.?\s*p\.?\s*\(?\s*c", folded_island)
    )
    is_letter = bool(
        re.search(r"(?mi)^\s*from\s*,", (island_text or "")[:500])
        or "sub-divisional" in folded_island
        or "block development" in folded_island
        or "first information report" in folded_island
        or re.search(r"\bfir\s*no", folded_island)
    )
    structural = is_hc or is_rop or is_letter or (
        _is_sci_caption(island_text)
        and re.search(r"(?mi)^\s*0?\s*r\s*d\s*e\s*r\b|^\s*order\b", island_text[:900])
    )
    if not structural:
        return 0

    part_dates = _extract_date_keys(particulars or "")
    pronounced = _extract_pronounced_date_keys(island_text)
    island_dates = _extract_island_match_date_keys(island_text)
    # Prefer the order's DATE:/caption line over incidental body dates
    # (e.g. interim order "stand over to 20th February" must not win P-3).
    if part_dates and pronounced and part_dates & pronounced:
        score += 18
    elif part_dates and island_dates and part_dates & island_dates:
        score += 10
    if "writ petition" in folded_part or "w.p" in folded_part:
        if is_hc and ("writ petition" in folded_island or "w.p" in folded_island):
            score += 6
        if (
            "copy of the writ petition" in folded_part
            and is_hc
            and re.search(r"humble\s+petition|most\s+respectfully\s+showeth", folded_island)
        ):
            score += 12
    if any(word in folded_part for word in ("judgment", "judgement", "final order")):
        if is_hc and (
            "coram" in folded_island or "p.c" in folded_island or "pc" in folded_island
        ):
            score += 5
        # Interim / adjournment sheets are not the "judgment and final order".
        if is_hc and re.search(
            r"stand\s+over|adjourned\s+to|listed\s+on\s+the\s+next",
            folded_island,
        ):
            score -= 8
        if is_hc and re.search(
            r"petition\s+(?:is\s+)?dismissed|dismissed\.|final\s+order|"
            r"accordingly,?\s+the\s+petition\s+dismissed",
            folded_island,
        ):
            score += 4
    if "special leave" in folded_part or "s.l.p" in folded_part or "slp" in folded_part:
        if is_rop:
            score += 6
    if "review" in folded_part:
        if is_review or (is_rop and re.search(r"\b1512\b|\breview\b", folded_island)):
            score += 6
        # Prefer the Review Petition ORDER sheet over its RoP twin.
        if (
            is_review
            and _is_sci_caption(island_text)
            and re.search(r"(?mi)^\s*0?\s*r\s*d\s*e\s*r\b|^\s*order\b", island_text[:1200])
            and not is_rop
        ):
            score += 12
        # Shared review petition number from Index particulars.
        for num in re.findall(r"\breview\s+petition[^0-9]{0,40}(\d{2,6})", folded_part):
            if re.search(rf"\b{re.escape(num)}\b", folded_island):
                score += 8
                break
    # Letter / communication / FIR cues from Index particulars.
    for token in (
        "sub-divisional",
        "sub divisional",
        "communication",
        "recovery notice",
        "fir",
        "first information",
        "anticipatory bail",
        "rent controller",
        "written statement",
        "plaint",
    ):
        if token in folded_part and token in folded_island:
            score += 5
    # Several index entries describe short records by their document type and
    # party name rather than a unique date (for example, a witness statement).
    # These title cues distinguish adjacent exhibits that share the same HC
    # caption and case number.
    title_cues = (
        ("compromise deed", r"\bcompromise\b.{0,80}\bdeed\b"),
        ("statement", r"\bstatement\s+of\s+ram\s+bachan\s+singh\b"),
        ("counter-affidavit", r"\bcounter\W+affidavit\b"),
        ("rejoinder", r"\brejoinder\b"),
        ("medical records", r"\bmedical\s+records?\b"),
        ("will", r"\bwill\s+of\b"),
    )
    for cue, pattern in title_cues:
        if cue in folded_part and re.search(pattern, island_text[:1800], re.I):
            score += 9
    # Shared case / document numbers (FIR 177, CRM-M 16067, etc.).
    for num in re.findall(r"\b\d{2,6}\b", particulars or ""):
        if len(num) >= 3 and re.search(rf"\b{re.escape(num)}\b", island_text or ""):
            score += 3
            break
    number = _annexure_number_from_label(label)
    if number is not None and re.search(
        rf"\b(?:annexure\s*)?[pe]\s*[-–—./\s]*{number}\b", folded_island
    ):
        score += 8
    return score


def _annexure_island_may_continue(
    start_text: str, next_text: str
) -> bool:
    """True when next island is a same-matter RoP twin of a Review/SLP order."""
    if not _looks_like_sci_court_rop_extract(next_text):
        return False
    start_fold = _fold(start_text[:1800])
    is_order_sheet = bool(
        ("review petition" in start_fold or "special leave" in start_fold)
        and re.search(r"(?mi)^\s*0?\s*r\s*d\s*e\s*r\b|^\s*order\b", start_text[:1200])
    ) or (
        _is_sci_caption(start_text)
        and re.search(r"(?mi)^\s*0?\s*r\s*d\s*e\s*r\b|^\s*order\b", start_text[:1200])
        and not _looks_like_sci_court_rop_extract(start_text)
    )
    if not is_order_sheet:
        return False
    start_nums = set(re.findall(r"\b\d{3,6}\b", start_text[:1500]))
    next_nums = set(re.findall(r"\b\d{3,6}\b", next_text[:1500]))
    return bool(start_nums & next_nums)


def _place_index_expected_annexures(
    page_parts: PagePartMap,
    page_text: Mapping[int, str],
    page_count: int,
) -> PagePartMap:
    """Label unstamped exhibit islands from Index inventory (P-n order / dates).

    Papers without ``ANNEXURE P-n`` stamps still list exhibits in the Index.
    Detect HC captions / SCI RoP / review-order starts after Affidavit and map
    them onto expected Index annexures so they are not left blank or swallowed
    by Record of Proceedings / Impugned Order.
    """
    entries = collect_index_annexure_entries(page_parts, page_text)
    if not entries:
        return page_parts

    zone_start = _post_petition_zone_start(page_parts, page_count)
    if zone_start is None:
        return page_parts

    updated = {page: list(names) for page, names in page_parts.items()}

    # Pages already claimed by a printed stamp stay authoritative.
    stamped_pages: set[int] = set()
    for start, end, _label in _annexure_run_bounds(page_text, page_count):
        stamped_pages.update(range(start, end + 1))

    attached = {
        name
        for names in updated.values()
        for name in parts_on_page(names)
        if family_split_name(name) == ANNEXURE_FAMILY
    }
    pending = [(label, particulars) for label, particulars in entries if label not in attached]
    if not pending:
        return updated

    # Drop post-petition RoP / Impugned labels so islands can be claimed.
    for page in range(zone_start, page_count + 1):
        if page in stamped_pages:
            continue
        names = parts_on_page(updated.get(page))
        if not names:
            continue
        if any(family_split_name(name) == ANNEXURE_FAMILY for name in names):
            continue
        text = page_text.get(page, "")
        drop = False
        if "Record of Proceedings" in names and (
            _looks_like_sci_court_rop_extract(text) or not _looks_like_rop_index_form(text)
        ):
            drop = True
        if "Impugned Order" in names and not _looks_like_impugned_order_start(text):
            drop = True
        if drop:
            remaining = [
                name
                for name in names
                if name not in {"Record of Proceedings", "Impugned Order"}
            ]
            if remaining:
                updated[page] = remaining
            else:
                updated.pop(page, None)

    candidates: list[int] = []
    for page in range(zone_start, page_count + 1):
        if page in stamped_pages:
            continue
        text = page_text.get(page, "")
        names = parts_on_page(updated.get(page))
        if names and any(family_split_name(name) == ANNEXURE_FAMILY for name in names):
            continue
        if names and any(
            name
            in {
                MAIN_PETITION_PART,
                "Affidavit",
                "AOR's Certificate",
                "Appendix",
                "Vakalatnama",
                "Memo of Appearance",
                "Filing Memo",
                "Cover Page",
                "Index",
            }
            for name in names
        ):
            continue
        if _is_unstamped_annexure_island_start(text):
            candidates.append(page)

    if not candidates:
        return updated

    # Score each pending Index annexure against each candidate island.
    assignments: dict[int, str] = {}
    used_pages: set[int] = set()
    used_labels: set[str] = set()

    scored: list[tuple[int, int, str, int]] = []
    for label, particulars in pending:
        for page in candidates:
            window = "\n".join(
                page_text.get(offset, "")
                for offset in range(page, min(page + 2, page_count + 1))
            )
            score = _score_island_for_index_annexure(window, particulars, label)
            scored.append((score, page, label, _annexure_number_from_label(label) or 999))

    # Prefer strong date/keyword matches first.
    for score, page, label, _number in sorted(
        scored, key=lambda item: (-item[0], item[1], item[3])
    ):
        if score < 5:
            continue
        if page in used_pages or label in used_labels:
            continue
        assignments[page] = label
        used_pages.add(page)
        used_labels.add(label)

    # Sequential fallback for remaining Index annexures ↔ unused islands.
    remaining_labels = [label for label, _ in pending if label not in used_labels]
    remaining_pages = [page for page in candidates if page not in used_pages]
    for label, page in zip(remaining_labels, remaining_pages, strict=False):
        assignments[page] = label
        used_pages.add(page)
        used_labels.add(label)

    if not assignments:
        return updated

    ordered_starts = sorted(assignments)
    candidate_set = set(candidates)
    for index, start in enumerate(ordered_starts):
        label = assignments[start]
        end = (
            ordered_starts[index + 1] - 1
            if index + 1 < len(ordered_starts)
            else page_count
        )
        for page in range(start, end + 1):
            if page in stamped_pages:
                break
            text = page_text.get(page, "")
            names = parts_on_page(updated.get(page))
            # A later unstamped island (new HC caption / RoP) is its own exhibit —
            # do not let an earlier Index assignment swallow it (interim→final).
            # Exception: RoP twin of a Review/SLP order sheet for the same matter.
            if page > start and page in candidate_set and page not in assignments:
                if not _annexure_island_may_continue(
                    page_text.get(start, ""), text
                ):
                    break
            if page_starts_application(text):
                break
            if _vakalatnama_heading(text) and _is_sci_caption(text):
                break
            if _FILING_MEMO_RE.search(_heading_window(text, lines=8)):
                break
            if names and any(
                name
                in {
                    MAIN_PETITION_PART,
                    "Affidavit",
                    "AOR's Certificate",
                    "Appendix",
                    "Vakalatnama",
                    "Memo of Appearance",
                    "Filing Memo",
                }
                for name in names
            ):
                break
            if names and any(
                family_split_name(name) == ANNEXURE_FAMILY and name != label
                for name in names
            ):
                break
            updated[page] = [label]

    return updated


def _demote_unmentioned_annexures(
    page_parts: PagePartMap,
    page_text: Mapping[int, str],
) -> PagePartMap:
    """Drop annexure labels not cited in Index / LOD / Main → Unidentified.

    When those source docs name no annexures at all, leave stamped labels alone
    (cannot inventory-gate without a mention list).

    Index OCR often loses later rows (P-8+) while the paper-book still prints
    those stamps — keep contiguous stamped numbers after the last Index cite.
    """
    expected = collect_expected_annexures(page_parts, page_text)
    if not expected:
        return page_parts

    expected = set(expected)
    max_by_series: dict[str, int] = {}
    for label in expected:
        match = re.fullmatch(r"(?i)annexure\s+([a-z])-?(\d+)", label.strip())
        if not match:
            continue
        series = match.group(1).upper()
        number = int(match.group(2))
        max_by_series[series] = max(max_by_series.get(series, 0), number)

    stamped_marks: list[Any] = []
    for text in page_text.values():
        mark = annexure_ref_in_heading(text or "")
        if mark is not None:
            stamped_marks.append(mark)

    # Index OCR often drops later rows while stamps remain — keep stamped
    # labels near the Index inventory for the same series (P or R). Far stray
    # P stamps (e.g. P-9 when Index only cites P-1) still demote.
    if max_by_series:
        for mark in stamped_marks:
            series_max = max_by_series.get(mark.series)
            if series_max is None:
                # Respondent stamps often omitted from Index OCR — keep them.
                if mark.series == "R":
                    expected.add(mark.label)
                continue
            if mark.number <= series_max + 5:
                expected.add(mark.label)

    updated: PagePartMap = {}
    for page, names in page_parts.items():
        kept: list[str] = []
        for name in parts_on_page(names):
            if family_split_name(name) == ANNEXURE_FAMILY and name not in expected:
                continue
            kept.append(name)
        if kept:
            updated[int(page)] = kept
    return updated


def _realign_stamped_annexures_to_index(
    page_parts: PagePartMap,
    page_text: Mapping[int, str],
    page_count: int,
) -> PagePartMap:
    """Correct wrong ANNEXURE stamps using Index particulars (e.g. P-3 stamped P-4)."""
    entries = collect_index_annexure_entries(page_parts, page_text)
    if len(entries) < 2:
        return page_parts

    updated = {page: list(names) for page, names in page_parts.items()}
    runs = _annexure_run_bounds(page_text, page_count)
    if not runs:
        return updated

    # Score each stamped island start against every Index row.
    assignments: dict[int, str] = {}
    used_labels: set[str] = set()
    used_starts: set[int] = set()
    stamp_counts: dict[str, int] = {}
    for _start, _end, stamp_label in runs:
        stamp_counts[stamp_label] = stamp_counts.get(stamp_label, 0) + 1

    scored: list[tuple[int, int, str, str]] = []
    for start, _end, stamp_label in runs:
        window = "\n".join(
            page_text.get(offset, "")
            for offset in range(start, min(start + 2, page_count + 1))
        )
        for label, particulars in entries:
            score = _score_island_for_index_annexure(window, particulars, label)
            scored.append((score, start, label, stamp_label))

    for score, start, label, stamp_label in sorted(
        scored,
        key=lambda item: (
            -item[0],
            item[1],
            # Prefer Index corrections over keeping a duplicated wrong stamp.
            0 if item[2] != item[3] else 1,
            item[2],
        ),
    ):
        if score < 8:
            continue
        if start in used_starts or label in used_labels:
            continue
        if label != stamp_label:
            # Only remap when the same stamp number was printed on two islands
            # (Leelawati: P-3 content stamped P-4 twice). Do not remap unique
            # later stamps (P-4/P-5) just because Index OCR omitted those rows.
            if stamp_counts.get(stamp_label, 0) < 2:
                continue
        assignments[start] = label
        used_starts.add(start)
        used_labels.add(label)

    if not assignments:
        return updated

    # Paint each reassigned island through the next assigned/stamp boundary.
    ordered = sorted(assignments)
    run_by_start = {start: (start, end, label) for start, end, label in runs}
    for index, start in enumerate(ordered):
        label = assignments[start]
        stamp_end = run_by_start.get(start, (start, start, label))[1]
        next_bound = (
            ordered[index + 1]
            if index + 1 < len(ordered)
            else stamp_end + 1
        )
        end = min(stamp_end, next_bound - 1)
        # If Index says this stamp is really an earlier annexure, stop before
        # the next stamped island that kept/got a different label.
        for page in range(start, end + 1):
            text = page_text.get(page, "")
            if page > start and annexure_label_from_text(text):
                other = annexure_label_from_text(text)
                if other and other != label and page not in assignments:
                    break
            if page_starts_application(text):
                break
            updated[page] = [label]
    return updated


def _realign_annexure_boundaries_to_index_content(
    page_parts: PagePartMap,
    page_text: Mapping[int, str],
    page_count: int,
) -> PagePartMap:
    """Split merged exhibit runs where a new Index-described record starts.

    A reproduced High Court record can carry its own ``Annexure No. 1`` or
    ``Annexure No. 2`` stamp. Those are local exhibit numbers and may not match
    the Supreme Court paper-book's P-n sequence. Use the outer Index
    particulars (date, record type and case details) to anchor each new
    document inside an already identified Annexure run.
    """
    entries = collect_index_annexure_entries(page_parts, page_text)
    if len(entries) < 2:
        return page_parts

    annexure_pages = {
        page
        for page, names in page_parts.items()
        if any(family_split_name(name) == ANNEXURE_FAMILY for name in parts_on_page(names))
    }
    if not annexure_pages:
        return page_parts

    candidates: list[int] = []
    for page in sorted(annexure_pages):
        text = page_text.get(page, "")
        # A new record normally opens with its own lower-court caption. Also
        # allow an explicit exhibit heading when OCR omitted the caption.
        if _is_lower_court_caption(text) or annexure_ref_in_heading(text):
            candidates.append(page)
    if len(candidates) < 2:
        return page_parts

    scored: list[tuple[int, int, str]] = []
    for index, start in enumerate(candidates):
        next_candidate = (
            candidates[index + 1]
            if index + 1 < len(candidates)
            else page_count + 1
        )
        # Include the full short record where possible, but don't let a later
        # exhibit's date/title contaminate the match for this one.
        window_end = min(next_candidate, start + 25, page_count + 1)
        window = "\n".join(
            page_text.get(offset, "")
            for offset in range(start, window_end)
        )
        for label, particulars in entries:
            score = _score_island_for_index_annexure(window, particulars, label)
            scored.append((score, start, label))

    assignments: dict[int, str] = {}
    used_starts: set[int] = set()
    used_labels: set[str] = set()
    for score, start, label in sorted(scored, key=lambda row: (-row[0], row[1], row[2])):
        if score < 9 or start in used_starts or label in used_labels:
            continue
        assignments[start] = label
        used_starts.add(start)
        used_labels.add(label)
    if len(assignments) < 2:
        return page_parts

    updated = {page: list(names) for page, names in page_parts.items()}
    starts = sorted(assignments)
    for index, start in enumerate(starts):
        stop = starts[index + 1] if index + 1 < len(starts) else page_count + 1
        for page in range(start, stop):
            names = parts_on_page(updated.get(page))
            if not any(family_split_name(name) == ANNEXURE_FAMILY for name in names):
                continue
            text = page_text.get(page, "")
            if page_starts_application(text):
                break
            updated[page] = [assignments[start]]
    return updated


def _restore_indexed_back_matter_parts(
    page_parts: PagePartMap,
    page_text: Mapping[int, str],
    page_count: int,
) -> PagePartMap:
    """Recover scan-only Vakalatnama and separately indexed High Court memo.

    Some compiled paper books place these standalone back-matter items after
    the Filing Memo. A Vakalatnama page may yield only its printed folio in text
    extraction; an outer-Index-listed High Court Memo of Parties is likewise
    distinct from a memo reproduced inside an Annexure.
    """
    updated = {page: list(names) for page, names in page_parts.items()}
    index_text = "\n".join(
        page_text.get(page, "")
        for page, names in page_parts.items()
        if "Index" in parts_on_page(names)
    )
    indexed_hc_memo = bool(
        re.search(r"memo\s+of\s+parties\s+in\s+(?:the\s+)?high\s+court", index_text, re.I)
    )
    filing_pages = [
        page
        for page in range(1, page_count + 1)
        if "Filing Memo" in parts_on_page(updated.get(page))
        or _outer_anchor_label(page_text.get(page, "")) == "Filing Memo"
        or _looks_like_filing_memo_continuation(page_text.get(page, ""))
    ]
    if not filing_pages:
        return updated
    filing_end = max(filing_pages)

    def prior_filing_context(page: int) -> bool:
        start = max(1, page - 4)
        return any(
            "Filing Memo" in parts_on_page(updated.get(prior))
            or _outer_anchor_label(page_text.get(prior, "")) == "Filing Memo"
            or _looks_like_filing_memo_continuation(page_text.get(prior, ""))
            for prior in range(start, page)
        )

    # Index row plus sequential folio is strong evidence for a scanned
    # Vakalatnama sheet immediately after the filing list.
    for page in range(filing_end + 1, min(page_count, filing_end + 5) + 1):
        text = (page_text.get(page, "") or "").strip()
        if not re.fullmatch(r"\d{1,4}", text):
            continue
        previous = page_text.get(page - 1, "")
        if (
            re.search(r"vakalatnama", previous, re.I)
            and re.search(r"power\s+of\s+attorney|1\s*\+\s*1", previous, re.I)
            and prior_filing_context(page)
        ):
            updated[page] = ["Vakalatnama"]

    if not indexed_hc_memo:
        return updated

    # The memo may have no heading of its own: its first page is a High Court
    # cause title and party list, followed by one or more continuation pages.
    start = next(
        (
            page
            for page in range(filing_end + 1, page_count + 1)
            if _is_lower_court_caption(page_text.get(page, ""))
            and re.search(r"\b(?:petitioner|opposite\s+parties)\b", page_text.get(page, ""), re.I)
            and not annexure_ref_in_heading(page_text.get(page, ""))
        ),
        None,
    )
    if start is None:
        return updated
    for page in range(start, page_count + 1):
        text = page_text.get(page, "")
        if page > start and (
            _is_sci_caption(text)
            or annexure_ref_in_heading(text)
            or _outer_anchor_label(text) in {"Filing Memo", "Vakalatnama", "Memo of Appearance"}
        ):
            break
        if page == start or re.search(r"\b(?:opposite\s+parties|respondents?)\b", text, re.I):
            updated[page] = ["Memo of Parties"]
        else:
            break
    return updated


def _demote_false_affidavit_between_annexures(
    page_parts: PagePartMap,
    page_text: Mapping[int, str],
    page_count: int,
) -> PagePartMap:
    """Annexed court affidavits between P-n stamps are not the SCI Affidavit slot."""
    updated = {page: list(names) for page, names in page_parts.items()}
    annex_pages = {
        page
        for page, names in updated.items()
        if any(family_split_name(name) == ANNEXURE_FAMILY for name in parts_on_page(names))
    }
    if not annex_pages:
        return updated
    first_ann = min(annex_pages)
    for page in range(first_ann, page_count + 1):
        names = parts_on_page(updated.get(page))
        if "Affidavit" not in names:
            continue
        text = page_text.get(page, "")
        # Keep a real SCI petition affidavit (rare after annexures start).
        if _is_sci_caption(text) and _AFFIDAVIT_HEADING_RE.search(
            _heading_window(text, lines=20)
        ):
            if not _is_lower_court_caption(text) and "in the court of" not in _fold(
                text[:800]
            ):
                continue
        prior_ann = max((p for p in annex_pages if p < page), default=0)
        next_ann = min((p for p in annex_pages if p > page), default=0)
        if prior_ann and (next_ann or annexure_label_from_text(text)):
            # Absorb into the preceding annexure island.
            prior_names = parts_on_page(updated.get(prior_ann))
            prior_label = next(
                (
                    name
                    for name in prior_names
                    if family_split_name(name) == ANNEXURE_FAMILY
                ),
                None,
            )
            if prior_label:
                updated[page] = [prior_label]
            else:
                updated.pop(page, None)
    return updated


def repair_compiled_split(
    page_parts: PagePartMap,
    page_text: Mapping[int, str],
    *,
    page_count: int,
) -> tuple[PagePartMap, list[DuplicateSplitHit]]:
    """Full hybrid repair used after LlamaSplit for compiled petitions."""
    if page_count < 1:
        cleaned = {
            int(page): list(names)
            for page, names in page_parts.items()
            if parts_on_page(names)
        }
        return cleaned, []

    updated: PagePartMap = {}
    for page, names in page_parts.items():
        try:
            number = int(page)
        except (TypeError, ValueError):
            continue
        kept = [name for name in parts_on_page(names) if _is_real_split_label(name)]
        if kept:
            updated[number] = kept

    updated = _demote_false_advocate_checklist(updated, page_text)
    updated = _demote_cover_mislabeled_as_main(updated, page_text)
    updated = _apply_outer_anchors(updated, page_text, page_count)
    updated = _restore_split_preface(updated, page_text, page_count)
    updated = _demote_front_matter_mislabeled_as_main(updated, page_text, page_count)
    updated = _demote_cover_mislabeled_as_main(updated, page_text)
    updated = _extend_main_petition_body(updated, page_text, page_count)
    updated = _demote_annexed_sci_rop(updated, page_text, page_count)
    # Capture Llama wrong-label duplicates before nesting absorbs HC exhibits.
    duplicates = find_duplicate_split_parts(updated)
    updated = _force_annexure_nesting(updated, page_text, page_count)
    updated = _fill_gaps(updated, page_text, page_count)
    updated = _label_near_blank_impugned_gap(updated, page_text, page_count)
    updated = _fill_gaps(updated, page_text, page_count)
    updated = _force_annexure_nesting(updated, page_text, page_count)
    updated = _demote_false_advocate_checklist(updated, page_text)
    updated = _demote_front_matter_mislabeled_as_main(updated, page_text, page_count)
    updated = _demote_cover_mislabeled_as_main(updated, page_text)
    updated = _extend_main_petition_body(updated, page_text, page_count)
    updated = _demote_annexed_sci_rop(updated, page_text, page_count)

    exploded = explode_repeating_split_parts(updated, page_text)
    exploded = _force_annexure_nesting(exploded, page_text, page_count)
    exploded = _demote_cover_mislabeled_as_main(exploded, page_text)
    exploded = _demote_annexed_sci_rop(exploded, page_text, page_count)
    exploded = _prefer_form28_main_island(exploded, page_text)
    more = find_duplicate_split_parts(exploded)
    seen = {(hit.part, hit.kept_span) for hit in duplicates}
    for hit in more:
        key = (hit.part, hit.kept_span)
        if key not in seen:
            duplicates.append(hit)
            seen.add(key)
    repaired = collapse_repeated_split_pages(exploded)
    repaired = _force_annexure_nesting(repaired, page_text, page_count)
    repaired = _demote_cover_mislabeled_as_main(repaired, page_text)
    repaired = _extend_main_petition_body(repaired, page_text, page_count)
    repaired = _prefer_form28_main_island(repaired, page_text)
    repaired = collapse_repeated_split_pages(repaired)
    # Application nesting can leave I.A. body pages blank — refill Applications
    # only (do not re-expand Vakalatnama / other slots past collapsed dupes).
    repaired = _fill_application_gaps(repaired, page_text, page_count)
    repaired = collapse_repeated_split_pages(repaired)
    # Index lists exhibits even when sheets lack ANNEXURE stamps — place those
    # islands before inventory demote so P-n land as annexures, not RoP/blank.
    repaired = _place_index_expected_annexures(repaired, page_text, page_count)
    # Wrong stamps (P-3 content printed as P-4) → Index particulars win.
    repaired = _realign_stamped_annexures_to_index(repaired, page_text, page_count)
    repaired = _demote_false_affidavit_between_annexures(
        repaired, page_text, page_count
    )
    # After all stamp/nest work: annexures not cited in Index/LOD/Main become
    # unlabeled leftovers → Undefined / Unidentified at slice time.
    repaired = _demote_unmentioned_annexures(repaired, page_text)
    repaired = _apply_index_printed_pages(repaired, page_text, page_count)
    repaired = _keep_application_party_lists_nested(
        repaired, page_text, page_count
    )
    # Keep reproduced High Court documents inside their printed enclosing
    # exhibit even if a later Index-folio reconciliation assigned a top-level
    # heading such as List of Dates & Events.
    repaired = _force_annexure_nesting(repaired, page_text, page_count)
    repaired = _demote_unverified_representation_parts(repaired, page_text)
    # The outer Index describes the Supreme Court paper-book's P-n sequence;
    # local High Court exhibit numbers can differ and must not merge adjacent
    # paper-book annexures into one output.
    repaired = _realign_annexure_boundaries_to_index_content(
        repaired, page_text, page_count
    )
    repaired = _restore_indexed_back_matter_parts(repaired, page_text, page_count)
    return repaired, duplicates


_FOLIO_NUM_RE = re.compile(r"^(?P<n>\d{1,4})(?P<suffix>[A-Za-z])?$")
_FOLIO_LETTER_RE = re.compile(r"^[A-Za-z]$")
_COURT_FEE_PAGE_RE = re.compile(r"cash\s*&?\s*accounts|bank draft|payment receipt", re.I)


def _printed_folio(text: str) -> tuple[str, int, str] | None:
    """Corner folio such as ``36``, ``63B``, or ``B``.

    Only the bottom line counts. A page number cited in the paragraph above
    it (``36`` then ``37 to 48``) is not the folio of this sheet.
    """
    lines = [line.strip() for line in (text or "").splitlines() if line.strip()]
    skipped_word = False
    for line in reversed(lines):
        if re.fullmatch(r"[\W_]+", line):
            continue
        numbered = _FOLIO_NUM_RE.fullmatch(line)
        if numbered:
            number = int(numbered.group("n"))
            if 1900 <= number <= 2099:
                return None
            return ("number", number, (numbered.group("suffix") or "").upper())
        if _FOLIO_LETTER_RE.fullmatch(line):
            return ("letter", ord(line.upper()), "")
        # "61" then "December": the folio sits just above a one-word signature.
        if not skipped_word and re.fullmatch(r"[A-Za-z]{3,}", line):
            skipped_word = True
            continue
        return None
    return None


def _folio_in_printed_row(
    folio: tuple[str, int, str], row: IndexPrintedRow
) -> bool:
    kind, number, suffix = folio
    if kind != row.kind or number < row.start or number > row.end:
        return False
    if number < row.end:
        return True
    if not row.end_suffix:
        return suffix == ""
    if not suffix:
        return True
    return suffix <= row.end_suffix


def _index_heading_part(text: str) -> str | None:
    stamped = annexure_label_from_text(text)
    if stamped:
        return stamped
    return _outer_anchor_label(text)


def _index_row_confirmed(text: str, part: str | None) -> bool:
    if not part:
        return False
    if _index_heading_part(text) == part:
        return True
    if part == "Court Fees" and _COURT_FEE_PAGE_RE.search(text or ""):
        return True
    return False


def _apply_index_printed_pages(
    page_parts: PagePartMap,
    page_text: Mapping[int, str],
    page_count: int,
) -> PagePartMap:
    """Open the pages the Index lists, and use them when the heading agrees.

    The Index page column is the folio printed on the sheet (36, 61–63B), not
    the PDF page index. A span is used only after one of those sheets shows
    the same document. A different printed heading on a sheet is left as-is.
    """
    rows = aligned_index_printed_rows(page_parts, page_text)
    if not rows:
        return page_parts
    folios = {
        page: _printed_folio(page_text.get(page, ""))
        for page in range(1, page_count + 1)
    }
    confirmed: list[IndexPrintedRow] = []
    for row in rows:
        if not row.mapped_part:
            continue
        if any(
            folio
            and _folio_in_printed_row(folio, row)
            and _index_row_confirmed(page_text.get(page, ""), row.mapped_part)
            for page, folio in folios.items()
        ):
            confirmed.append(row)
    if not confirmed:
        return page_parts

    def _in_part(part: str, folio: tuple[str, int, str]) -> bool:
        return any(
            row.mapped_part == part and _folio_in_printed_row(folio, row)
            for row in confirmed
        )

    anchored: set[str] = set()
    for page, names in page_parts.items():
        folio = folios.get(page)
        current = parts_on_page(names)
        if not folio or not current:
            continue
        if _in_part(current[0], folio):
            anchored.add(current[0])

    updated = {page: list(names) for page, names in page_parts.items()}
    for page in range(1, page_count + 1):
        folio = folios.get(page)
        if not folio:
            continue
        text = page_text.get(page, "")
        heading = _index_heading_part(text)
        if heading == "Index":
            continue
        owner = next(
            (
                row.mapped_part
                for row in confirmed
                if row.mapped_part and _folio_in_printed_row(folio, row)
            ),
            None,
        )
        if heading and owner and heading != owner:
            continue
        names = parts_on_page(updated.get(page))
        current = names[0] if names else None
        leaked = bool(
            current
            and current in anchored
            and not _in_part(current, folio)
        )
        weak = current in {"PoA/BR"} or (
            current == "Impugned Order" and not _looks_like_impugned_order_start(text)
        )
        if owner and (not current or weak or leaked or current == owner):
            updated[page] = [owner]
            continue
        if leaked and not owner:
            updated.pop(page, None)
    return updated


def _fill_application_gaps(
    page_parts: PagePartMap,
    page_text: Mapping[int, str],
    page_count: int,
) -> PagePartMap:
    """Extend Application n labels across body pages until the next document."""
    updated = {page: list(names) for page, names in page_parts.items() if parts_on_page(names)}
    last_app: str | None = None
    for page in range(1, page_count + 1):
        names = parts_on_page(updated.get(page))
        if names:
            app = next(
                (
                    name
                    for name in names
                    if family_split_name(name) == "Application"
                    or re.fullmatch(r"(?i)application\s+\d+", name)
                ),
                None,
            )
            last_app = app
            continue
        if not last_app:
            continue
        text = page_text.get(page, "")
        if not (text or "").strip():
            continue
        if page_starts_application(text):
            last_app = None
            continue
        if annexure_label_from_text(text):
            last_app = None
            continue
        if _vakalatnama_heading(text) and _is_sci_caption(text):
            last_app = None
            continue
        if _FILING_MEMO_RE.search(_heading_window(text, lines=8)):
            last_app = None
            continue
        anchor = _outer_anchor_label(text)
        if anchor and anchor not in {last_app, "Application 1"}:
            if family_split_name(anchor) != "Application":
                last_app = None
                continue
        updated[page] = [last_app]
    return updated
