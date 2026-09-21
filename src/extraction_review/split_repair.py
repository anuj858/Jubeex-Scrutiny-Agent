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
    r"listing\s+proforma|listed\s+proforma|"
    r"proforma\s+for\s+first\s+listing",
    re.I,
)
_SYNOPSIS_RE = re.compile(r"(?m)^\s*synopsis\b", re.I)
_LOD_RE = re.compile(r"list of dates", re.I)
_APPENDIX_RE = re.compile(r"(?m)^\s*appendix\b", re.I)
_RECORD_RE = re.compile(r"record of proceedings?", re.I)
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
_SCI_COURT_ROP_RE = re.compile(
    r"item\s*n[o0]\.?\s*\d|"
    r"court\s*n[o0]\.?\s*\d|"
    r"petition\(s\)\s+for\s+special\s+leave|"
    r"s\s+u\s+p\s+r\s+e\s+m\s+e\s+c\s+o\s+u\s+r\s+t\s+o\s+f\s+i\s+n\s+d\s+i\s+a|"
    r"chamber\s+matter\s+section",
    re.I,
)
_FILING_MEMO_RE = re.compile(r"filing memo|index of filing|filing index", re.I)
_AOR_CERT_RE = re.compile(
    r"confined\s+only\s+to\s+the\s+pleadings|"
    r"(?:^|\n)\s*(?:c\s+e\s+r\s+t\s+i\s+f\s+i\s+c\s+a\s+t\s+e|"
    r"certifica\s*te|certificate)\b",
    re.IGNORECASE,
)
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
_APPEARANCE_RE = re.compile(r"(?m)^\s*memo of appearance\b", re.I)
_NEAR_BLANK_RE = re.compile(r"^[\s\d\.]*$")

# Outer SCI paper-book parts: keep the first contiguous run, not the longest.
# (A later High Court writ mislabeled Main Petition must not win.)
# Mirrored in document_parts.collapse_repeated_split_pages.
_FIRST_RUN_PARTS = frozenset(
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
    }
)

_NESTED_STEAL_PARTS = frozenset(
    {
        MAIN_PETITION_PART,
        "Affidavit",
        "Vakalatnama",
        "Memo of Appearance",
        "Memo of Parties",
        "AOR's Certificate",
        "Advocate's Checklist",
        "Impugned Order",
        "Cover Page",
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


def _looks_like_sci_checklist(text: str) -> bool:
    head = text[:1200]
    if _STATE_CHECKLIST_RE.search(head):
        return False
    # Listing Proforma also says "tick/check the correct box" — not a checklist.
    if _LISTING_RE.search(head) or _LISTING_RE.search(_fold(head)):
        return False
    if _LISTING_CONTINUATION_RE.search(text[:2000]):
        return False
    if _OFFICE_REPORT_RE.search(text[:2000]):
        return False
    if "proforma for first" in _fold(head) or "section -" in _fold(head[:400]):
        if "nature of matter" in _fold(head):
            return False
    if _SCI_CHECKLIST_RE.search(head):
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
    """True only for a standalone Impugned Order document start."""
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
    # Certified HC judgment placed as the Impugned Order section.
    if _is_lower_court_caption(text) and (
        "date of decision" in folded
        or "coram:" in folded
        or re.search(r"\bcrm-?m-?\d", folded)
        or "having heard learned counsel" in folded
    ):
        return True
    return False


def _looks_like_index_continuation(text: str) -> bool:
    """Index pages after the heading often omit the word INDEX."""
    if _looks_like_index_table(text):
        return True
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
    range_lines = re.findall(r"(?m)^\s*\d{1,3}\s*[-–—]\s*\d{1,3}\s*$", head)
    if len(range_lines) >= 4 and "annexure-p" not in _fold(head[:200]):
        return True
    return False


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
    if _is_lower_court_caption(text):
        return False
    if _AFFIDAVIT_HEADING_RE.search(_heading_window(text, lines=20)):
        return False
    if _AOR_CERT_RE.search(text[:2500]):
        return False
    window = text[:3000]
    folded = _fold(window)
    # Form-28 party schedule under SCI caption (names live here, not on Cover).
    if _is_sci_caption(text) and _PARTY_SCHEDULE_RE.search(window):
        return True
    if not _is_sci_caption(text):
        # Body pages often omit a fresh SCI caption after the party schedule.
        if _FORM28_BODY_RE.search(window) and (
            "special leave" in folded
            or "supreme court of india" in folded
            or "companion justices" in folded
            or "declaration in terms of rule" in folded
            or "questions of law" in folded
            or "stions of law" in folded
        ):
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
    if _looks_like_index_table(text):
        return "Index"
    # OR heading often sits below the cause title — search more than 12 lines.
    if _OFFICE_REPORT_RE.search(text[:2000]):
        return "Office Report on Limitation"
    if _LISTING_RE.search(_heading_window(text, lines=10)) or _LISTING_RE.search(
        text[:900]
    ):
        return "Listing Proforma"
    if _looks_like_court_notice_or_rop(text):
        return "Record of Proceedings"
    if _RECORD_RE.search(_heading_window(text, lines=10)):
        return "Record of Proceedings"
    if _SYNOPSIS_RE.search(_heading_window(text, lines=8)):
        return "Synopsis"
    if _LOD_RE.search(_heading_window(text, lines=8)):
        return "List of Dates & Events"
    if _APPENDIX_RE.search(_heading_window(text, lines=6)) and not annexure_ref_in_heading(
        text
    ):
        return "Appendix"
    # Filing Memo often shares a sheet with a trailing SCI caption — trust the
    # top-of-page heading, not a later caption on the same page.
    if _FILING_MEMO_RE.search(_heading_window(text, lines=6)) and not (
        _looks_like_court_notice_or_rop(text)
    ):
        return "Filing Memo"
    if _looks_like_sci_checklist(text):
        return "Advocate's Checklist"
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
            r"certifica\s*te|certificate)\b",
            _heading_window(text, lines=16),
            re.I,
        )
        if cert_title or "confined only to the pleadings" in _fold(text[:2500]):
            if cert_title or not _AFFIDAVIT_HEADING_RE.search(
                _heading_window(text, lines=20)
            ):
                return "AOR's Certificate"
    if _looks_like_cover_page(text):
        return "Cover Page"
    if page_starts_application(text) and not _looks_like_cover_page(text):
        return "Application 1"
    if _vakalatnama_heading(text) and (
        _is_sci_caption(text) or not _is_lower_court_caption(text)
    ):
        return "Vakalatnama"
    if _APPEARANCE_RE.search(_heading_window(text, lines=10)):
        return "Memo of Appearance"
    if _memo_of_parties_heading(text):
        return "Memo of Parties"
    if _looks_like_impugned_order_start(text):
        return "Impugned Order"
    if _looks_like_sci_main_petition(text):
        return MAIN_PETITION_PART
    if _looks_like_index_continuation(text):
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
        else:
            end = start
            blank_streak = 0
            for page in range(start + 1, page_count + 1):
                text = page_text.get(page, "")
                if page_starts_application(text):
                    break
                if _vakalatnama_heading(text) and _is_sci_caption(text):
                    break
                if _FILING_MEMO_RE.search(_heading_window(text, lines=8)):
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
    match = re.fullmatch(r"annexure [a-z]-?(\d{1,3})", _fold(name))
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
            # Never steal Index / OR / Listing that somehow overlaps (shouldn't).
            if names and all(name in _CARRY_BLOCKING_PARTS for name in names):
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
        if any(
            name in {"AOR's Certificate", "Affidavit", "Filing Memo"}
            for name in last_label
        ):
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


def _label_near_blank_impugned_gap(
    page_parts: PagePartMap,
    page_text: Mapping[int, str],
    page_count: int,
) -> PagePartMap:
    """OCR-blank Impugned Order pages sit between LOD and Main Petition."""
    updated = {page: list(names) for page, names in page_parts.items() if parts_on_page(names)}
    petition_pages = sorted(
        page
        for page, names in updated.items()
        if MAIN_PETITION_PART in parts_on_page(names)
    )
    if not petition_pages:
        return updated
    petition_start = petition_pages[0]
    # Last substantive page before the petition that is LOD/Synopsis
    # (not Listing/OR — those sit much earlier in the paper-book).
    pre_pages = [
        page
        for page, names in updated.items()
        if page < petition_start
        and any(
            name in {"List of Dates & Events", "Synopsis"}
            for name in parts_on_page(names)
        )
        and not _is_near_blank_page(page_text.get(page, ""))
    ]
    if not pre_pages:
        return updated
    gap_start = max(pre_pages) + 1
    if gap_start >= petition_start:
        return updated
    for page in range(gap_start, petition_start):
        text = page_text.get(page, "")
        if _outer_anchor_label(text) in {
            MAIN_PETITION_PART,
            "AOR's Certificate",
            "Appendix",
            "Cover Page",
        }:
            continue
        names = parts_on_page(updated.get(page))
        if names and not any(
            name in {"List of Dates & Events", "Synopsis", "Impugned Order"}
            for name in names
        ):
            continue
        if _is_near_blank_page(text) or len((text or "").strip()) < 120:
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
            r"(?i)annexure p-?\d{1,3}|application \d{1,3}", part
        ):
            continue
        kept = groups[0]
        dupes = tuple(groups[1:])
        hits.append(DuplicateSplitHit(part=part, kept_span=kept, duplicate_spans=dupes))
    return hits


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
    return repaired, duplicates
