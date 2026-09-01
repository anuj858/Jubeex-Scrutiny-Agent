"""Document-part labels, record slicing, and local chunk selection.

Split category names come from `configs/config.json` `split.categories`.
Each concurrent defect queries Pinecone with these captions, then this module
picks the excerpts that defect is allowed to see.
"""

from __future__ import annotations

import math
import re
from typing import Any

from .scrutiny.rules import Defect, normalize_filing_type

# Captions and headings as they appear on SCI forms — used as search text
# (not the long legal objection sentences).
FILING_CAPTION_QUERIES: tuple[str, ...] = (
    "IN THE SUPREME COURT OF INDIA CIVIL APPELLATE JURISDICTION",
    "SPECIAL LEAVE PETITION UNDER ARTICLE 136 Form 28",
    "QUESTIONS OF LAW GROUNDS MAIN PRAYER INTERIM RELIEF",
    "Listing Proforma Proforma for First Listing",
    "Advocate's Check List Advocate-on-Record certificate",
    "DECLARATION IN TERMS OF RULE 3(2) Affidavit",
    "Cover Page Index Office Report on Limitation",
    "Vakalatnama AOR Declaration Memo of Parties",
)

# Names must match configs/config.json split.categories.
SPLIT_PART_NAMES: tuple[str, ...] = (
    "Advocate's Checklist",
    "Caveat Report",
    "Cover Page",
    "Record of Proceedings",
    "AOR's Declaration",
    "Index",
    "Office Report on Limitation",
    "Listing Proforma",
    "Synopsis",
    "List of Dates & Events",
    "Petition",
    "Affidavit",
    "Annexures",
    "Appendix",
    "Memo of Parties",
    "Memo of Appearance",
    "Impugned Order",
    "Vakalatnama + PoA/BR",
    "Court Fees",
)

CATEGORY_TO_PARTS: dict[str, tuple[str, ...]] = {
    "filing_formalities": ("Petition", "Affidavit"),
    "advocate_checklist": ("Advocate's Checklist", "Vakalatnama + PoA/BR"),
    "listing_proforma": ("Listing Proforma",),
    "petition_presentation": ("Petition",),
    "applications": ("Petition", "Annexures", "Index"),
    "annexures": ("Annexures", "Index", "List of Dates & Events"),
    "parties": ("Memo of Parties", "Cover Page", "Petition"),
    "dates_execution": ("Petition", "Affidavit", "Vakalatnama + PoA/BR"),
    "index_paper_book": ("Index",),
    "limitation": ("Office Report on Limitation", "Petition"),
    "affidavit": ("Affidavit", "Petition"),
    "translations": ("Annexures", "Vakalatnama + PoA/BR"),
    "vakalatnama": ("Vakalatnama + PoA/BR",),
    "memo_of_appearance": ("Memo of Appearance",),
    "list_of_dates": ("List of Dates & Events", "Synopsis"),
}

# Structured CoreFilingRecord keys sent to the model for that category.
# Always include court / petition_type; never dump unused party lists.
CATEGORY_RECORD_FIELDS: dict[str, tuple[str, ...]] = {
    "filing_formalities": (
        "court",
        "petition_type",
        "special_category",
        "cause_title",
        "impugned_order",
        "filing_summary",
    ),
    "advocate_checklist": (
        "court",
        "petition_type",
        "advocate_on_record",
    ),
    "listing_proforma": (
        "court",
        "petition_type",
        "special_category",
        "cause_title",
        "matter_classification",
    ),
    "petition_presentation": (
        "court",
        "petition_type",
        "filing_summary",
    ),
    "applications": ("court", "petition_type", "filing_summary"),
    "annexures": ("court", "petition_type", "filing_summary"),
    "parties": (
        "court",
        "petition_type",
        "cause_title",
        "impugned_order",
    ),
    "dates_execution": ("court", "petition_type"),
    "index_paper_book": ("court", "petition_type", "filing_summary"),
    "limitation": ("court", "petition_type", "impugned_order"),
    "affidavit": ("court", "petition_type"),
    "translations": ("court", "petition_type", "filing_summary"),
    "vakalatnama": ("court", "petition_type", "advocate_on_record"),
    "memo_of_appearance": ("court", "petition_type", "advocate_on_record"),
    "list_of_dates": ("court", "petition_type", "filing_summary"),
}

ALWAYS_RECORD_FIELDS: tuple[str, ...] = ("court", "petition_type", "special_category")

# Ceiling on page excerpts sent to the LLM (summary is extra). Narrow checks
# do not need the global SCRUTINY_MAX_CHUNKS dump.
CATEGORY_MAX_CHUNKS: dict[str, int] = {
    "listing_proforma": 3,
    "advocate_checklist": 3,
    "filing_formalities": 10,
    "petition_presentation": 6,
    "applications": 4,
    "annexures": 6,
    "parties": 4,
    "dates_execution": 3,
    "index_paper_book": 4,
    "limitation": 3,
    "affidavit": 4,
    "translations": 4,
    "vakalatnama": 3,
    "memo_of_appearance": 3,
    "list_of_dates": 3,
}

# Keep a page chunk only if ln(best_score / this_score) is below this.
# 0.36 ≈ keep scores at least ~70% of the best hit; farther neighbours are dropped.
MAX_SCORE_LOG_GAP = 0.36

FILING_TYPE_LABELS: dict[str, str] = {
    "slp_civil": "Special Leave Petition (Civil)",
    "slp_criminal": "Special Leave Petition (Criminal)",
    "arbitration_petition": "Arbitration Petition",
    "writ_petition_civil": "Writ Petition (Civil)",
    "writ_petition_criminal": "Writ Petition (Criminal)",
}


def normalize_part_name(name: str | None) -> str:
    if not name:
        return ""
    text = name.replace("\u2019", "'").replace("\u2018", "'")
    text = re.sub(r"\s+", " ", text).strip()
    aliases = {
        "advocate's check list": "Advocate's Checklist",
        "advocates checklist": "Advocate's Checklist",
        "aor declaration": "AOR's Declaration",
        "vakalatnama": "Vakalatnama + PoA/BR",
        "form 28": "Petition",
        "main petition": "Petition",
        "slp": "Petition",
    }
    return aliases.get(text.lower(), text)


def filing_type_label(filing_type: str | None) -> str:
    key = normalize_filing_type(filing_type)
    if key in FILING_TYPE_LABELS:
        return FILING_TYPE_LABELS[key]
    raw = (filing_type or "").strip()
    return raw or "this filing"


def page_parts_from_split(job: Any) -> dict[int, str]:
    """Map 1-indexed page number → split category name."""
    result = getattr(job, "result", None) or job
    segments = getattr(result, "segments", None)
    if segments is None and isinstance(result, dict):
        segments = result.get("segments")
    if not segments:
        return {}

    mapping: dict[int, str] = {}
    for segment in segments:
        if isinstance(segment, dict):
            category = segment.get("category")
            pages = segment.get("pages") or []
        else:
            category = getattr(segment, "category", None)
            pages = getattr(segment, "pages", None) or []
        part = normalize_part_name(str(category or ""))
        if not part:
            continue
        for page in pages:
            try:
                mapping[int(page)] = part
            except (TypeError, ValueError):
                continue
    return mapping


def preferred_parts_for_defect(defect: Defect) -> list[str]:
    parts = list(CATEGORY_TO_PARTS.get(defect.category_id or "", ()))
    blob = " ".join(defect.where_to_look).lower()
    blob = blob.replace("\u2019", "'")
    for name in SPLIT_PART_NAMES:
        needle = name.lower()
        short = needle.split("+", 1)[0].strip()
        if needle in blob or short in blob:
            if name not in parts:
                parts.append(name)
    return parts


def pool_search_queries() -> list[str]:
    """Short caption queries covering typical SCI filing parts."""
    seen: set[str] = set()
    queries: list[str] = []
    for query in (*FILING_CAPTION_QUERIES, *SPLIT_PART_NAMES):
        key = query.strip().lower()
        if not key or key in seen:
            continue
        seen.add(key)
        queries.append(query)
    return queries


def match_terms_for_defect(defect: Defect) -> list[str]:
    """Phrases to score an in-memory chunk against this defect."""
    terms: list[str] = []
    if defect.trigger_words:
        terms.extend(
            p.strip()
            for p in re.split(r"[;|]", defect.trigger_words)
            if p.strip()
        )
    terms.extend(preferred_parts_for_defect(defect))
    for step in defect.where_to_look:
        heading = _heading_from_where_to_look(step)
        if heading:
            terms.append(heading)
    # Dedup while preserving order.
    seen: set[str] = set()
    unique: list[str] = []
    for term in terms:
        key = term.lower()
        if key in seen:
            continue
        seen.add(key)
        unique.append(term)
    return unique


def _heading_from_where_to_look(step: str) -> str | None:
    """Pull a form title out of a 'Check the Listing Proforma…' sentence."""
    text = re.sub(
        r"^(check|read|look at|examine|verify|confirm|review)\s+(the\s+|for\s+the\s+)?",
        "",
        step.strip(),
        flags=re.IGNORECASE,
    )
    for name in SPLIT_PART_NAMES:
        if name.lower() in text.lower():
            return name
    # First clause before 'and' / comma, if it is short.
    clause = re.split(r"[,.]|\band\b", text, maxsplit=1)[0].strip()
    if 3 <= len(clause) <= 80:
        return clause
    return None


def max_chunks_for_defect(defect: Defect, *, ceiling: int) -> int:
    """How many page excerpts this defect is allowed, at most.

    Listing / AOR-code checks are one form; Form 28 needs more pages.
    A child defect is a single point, so it stays tighter than its parent.
    """
    budget = CATEGORY_MAX_CHUNKS.get(defect.category_id or "", ceiling)
    preferred = preferred_parts_for_defect(defect)
    if len(defect.where_to_look) <= 2 and len(preferred) <= 1:
        budget = min(budget, 3)
    if defect.parent_check_id:
        budget = min(budget, 4)
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
    fields = CATEGORY_RECORD_FIELDS.get(defect.category_id or "")
    if not fields:
        keys = list(ALWAYS_RECORD_FIELDS)
        keys.extend(k for k in record if k not in keys)
        fields = tuple(keys)
    sliced = {key: record[key] for key in fields if key in record}
    return sliced or record


def _part_match(chunk: dict[str, Any], preferred: list[str]) -> bool:
    part = normalize_part_name(str(chunk.get("document_part") or ""))
    if not part or not preferred:
        return False
    part_l = part.lower()
    return any(part_l == p.lower() or p.lower() in part_l for p in preferred)


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
    1. Prefers the labelled document part / trigger hits for this defect.
    2. Caps to a smaller per-defect budget (one-form checks do not need 8 pages).
    3. Drops neighbours whose Pinecone score is logarithmically far from the
       best remaining hit, so a filled `top_k` does not ship unrelated pages.
    """
    if not pool:
        return []

    preferred = preferred_parts_for_defect(defect)
    terms = match_terms_for_defect(defect)
    summary = [c for c in pool if c.get("chunk_kind") == "summary"]
    pages = [c for c in pool if c.get("chunk_kind") != "summary"]

    def rank_key(chunk: dict[str, Any]) -> tuple[int, int, float]:
        part_hit = 1 if _part_match(chunk, preferred) else 0
        hits = _term_hits(chunk, terms)
        score = _chunk_score(chunk)
        return (part_hit, hits, score)

    ranked = sorted(pages, key=rank_key, reverse=True)
    focused = [
        c
        for c in ranked
        if _part_match(c, preferred) or _term_hits(c, terms) > 0
    ]
    candidates = focused if focused else ranked
    if not candidates:
        return summary[:1]

    # Score cutoff only among the leading relevance tier (preferred part, or
    # term hits). Otherwise a high-scoring wrong part would drop the right page.
    top_part = rank_key(candidates[0])[0]
    leading = [c for c in candidates if rank_key(c)[0] == top_part] or candidates

    page_ceiling = max(0, max_chunks - len(summary[:1]))
    page_budget = max_chunks_for_defect(defect, ceiling=page_ceiling)
    chosen_pages = keep_nearby_scores(leading, max_n=page_budget)
    chosen_pages.sort(key=lambda c: (c.get("page") is None, c.get("page") or 0))
    return summary[:1] + chosen_pages
