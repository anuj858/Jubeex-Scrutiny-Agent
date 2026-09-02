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
    "DECLARATION IN TERMS OF RULE 3(2) Affidavit",
    "Cover Page Index Office Report on Limitation",
    "Vakalatnama AOR Declaration Memo of Parties",
)

PINECONE_QUERY_MAX_CHARS = 110

# One PDF page can carry more than one Split label (e.g. Affidavit + Vakalatnama).
PagePartMap = dict[int, list[str]]

CATEGORY_TO_PARTS: dict[str, tuple[str, ...]] = {
    "filing_formalities": ("Petition", "Affidavit"),
    "advocate_checklist": ("Advocate's Checklist", "Vakalatnama + PoA/BR", "Vakalatnama"),
    "listing_proforma": ("Listing Proforma",),
    "petition_presentation": ("Petition",),
    "applications": ("Petition", "Annexures", "Index"),
    "annexures": ("Annexures", "Index", "List of Dates & Events"),
    "parties": ("Memo of Parties", "Cover Page", "Petition"),
    "dates_execution": (
        "Petition",
        "Affidavit",
        "Vakalatnama + PoA/BR",
        "Vakalatnama",
        "PoA/BR",
    ),
    "index_paper_book": ("Index",),
    "limitation": ("Office Report on Limitation", "Petition"),
    "affidavit": ("Affidavit", "Petition"),
    "translations": ("Annexures", "Vakalatnama + PoA/BR", "Vakalatnama", "PoA/BR"),
    "vakalatnama": ("Vakalatnama + PoA/BR", "Vakalatnama", "PoA/BR"),
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
    "dates_execution": ("court", "petition_type", "filing_summary"),
    "index_paper_book": ("court", "petition_type", "filing_summary"),
    "limitation": ("court", "petition_type", "impugned_order", "filing_summary"),
    "affidavit": ("court", "petition_type"),
    "translations": ("court", "petition_type", "filing_summary"),
    "vakalatnama": ("court", "petition_type", "advocate_on_record", "filing_summary"),
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
    "dates_execution": 6,
    "index_paper_book": 4,
    "limitation": 4,
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
    return re.sub(
        r"\s+", " ", (text or "").replace("\u2019", "'").replace("\u2018", "'")
    ).strip().lower()


def _needles_for_part(name: str, description: str = "") -> tuple[str, ...]:
    """Search phrases for one Split part: official name plus description nicknames."""
    needles: list[str] = []

    def add(raw: str) -> None:
        folded = _fold(raw)
        if len(folded) >= 3 and folded not in needles:
            needles.append(folded)

    add(name)
    left = name.split("+", 1)[0].strip()
    if left != name:
        add(left)
    for nick in re.findall(r"\(([^)]{2,48})\)", description):
        add(nick)
    return tuple(needles)


def normalize_part_name(name: str | None) -> str:
    if not name:
        return ""
    text = re.sub(r"\s+", " ", name.replace("\u2019", "'").replace("\u2018", "'")).strip()
    folded = _fold(text)
    for canonical, _description in _split_categories():
        if folded == _fold(canonical):
            return canonical
    for canonical, description in _split_categories():
        for needle in _needles_for_part(canonical, description):
            if folded == needle:
                return canonical
    return text


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
        if part and part not in found:
            found.append(part)
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


def _format_page_span(pages: list[int]) -> str:
    groups = _contiguous_groups(pages)
    if not groups:
        return ""
    bits = [str(a) if a == b else f"{a}–{b}" for a, b in groups]
    prefix = "p. " if sum(b - a + 1 for a, b in groups) == 1 else "pp. "
    return prefix + ", ".join(bits)


# Later pages labelled the same part after this many skipped pages are a
# mis-tag (second Index, second Cover), not a continuation.
_FIRST_PART_GAP = 20


def collapse_repeated_split_pages(page_parts: PagePartMap) -> PagePartMap:
    """Keep the first occurrence of each Split part, including nearby pages.

    Index at 5–7 is kept; Index at 55–57 is dropped (that page is something
    else). A mixed page can still keep its other label. Petition at 17 then
    25–35 is kept (other documents sit in between).
    """
    pages_by_part: dict[str, list[int]] = {}
    for page, names in page_parts.items():
        for part in parts_on_page(names):
            pages_by_part.setdefault(part, []).append(page)

    keep: set[tuple[int, str]] = set()
    for part, pages in pages_by_part.items():
        last_kept: int | None = None
        for page in sorted(set(pages)):
            if last_kept is None or page - last_kept <= _FIRST_PART_GAP:
                keep.add((page, part))
                last_kept = page

    collapsed: PagePartMap = {}
    for page, names in page_parts.items():
        kept = [part for part in parts_on_page(names) if (page, part) in keep]
        if kept:
            collapsed[page] = kept
    return collapsed


def documents_from_page_parts(page_parts: PagePartMap | dict[int, str]) -> dict[str, Any]:
    """Build filing_summary.documents from Split labels, with page spans."""
    order: list[str] = []
    pages_by_part: dict[str, list[int]] = {}
    for page in sorted(page_parts):
        for name in parts_on_page(page_parts.get(page)):
            if name not in pages_by_part:
                pages_by_part[name] = []
                order.append(name)
            pages_by_part[name].append(page)
    items = [
        f"{name} ({_format_page_span(pages_by_part[name])})" for name in order
    ]
    return {"count": len(items), "items": items}


def overlay_split_documents(
    payload: dict[str, Any], page_parts: PagePartMap | dict[int, str]
) -> None:
    """Replace Index slang (V/A) with Split part names and page sources."""
    docs = documents_from_page_parts(page_parts)
    if not docs["items"]:
        return
    record = payload
    while (
        isinstance(record, dict)
        and "filing_summary" not in record
        and "petition_type" not in record
        and isinstance(record.get("data"), dict)
    ):
        record = record["data"]
    if not isinstance(record, dict):
        return
    summary = record.get("filing_summary")
    if not isinstance(summary, dict):
        summary = {}
        record["filing_summary"] = summary
    summary["documents"] = docs


def filing_type_label(filing_type: str | None) -> str:
    key = normalize_filing_type(filing_type)
    if key in FILING_TYPE_LABELS:
        return FILING_TYPE_LABELS[key]
    raw = (filing_type or "").strip()
    return raw or "this filing"


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
        part = normalize_part_name(str(category or ""))
        if not part:
            continue
        for page in pages:
            try:
                number = int(page)
            except (TypeError, ValueError):
                continue
            current = mapping.setdefault(number, [])
            if part not in current:
                current.append(part)
    return collapse_repeated_split_pages(mapping)


def parts_named_in_text(text: str) -> list[str]:
    """Split parts whose name or description nickname appears in `text`."""
    blob = _fold(text)
    found: list[str] = []
    for name, description in _split_categories():
        if any(needle in blob for needle in _needles_for_part(name, description)):
            if name not in found:
                found.append(name)
    return found


def parts_named_in_where_to_look(defect: Defect) -> list[str]:
    """Split parts the catalogue told this check to open."""
    return parts_named_in_text(" ".join(defect.where_to_look))


def preferred_parts_for_defect(defect: Defect) -> list[str]:
    """Where-to-look parts first, then the rest of the category."""
    named = parts_named_in_where_to_look(defect)
    category = list(CATEGORY_TO_PARTS.get(defect.category_id or "", ()))
    parts = list(named)
    for name in category:
        if name not in parts:
            parts.append(name)
    return parts


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


def pinecone_queries_for_defect(defect: Defect) -> list[str]:
    """Queries sent to Pinecone: where-to-look parts, headings, and cues."""
    queries: list[str] = []
    queries.extend(parts_named_in_where_to_look(defect))
    if defect.trigger_words:
        queries.extend(
            p.strip()
            for p in re.split(r"[;|]", defect.trigger_words)
            if p.strip()
        )
    for step in defect.where_to_look:
        heading = _heading_from_where_to_look(step)
        if heading:
            queries.append(heading)
        clipped = _clip_query(_bare_where_to_look(step))
        if clipped:
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
    """Parts named in where-to-look; otherwise every preferred part."""
    return parts_named_in_where_to_look(defect) or preferred_parts_for_defect(
        defect
    )


def chunks_cover_part(chunks: list[dict[str, Any]], part: str) -> bool:
    pages = [c for c in chunks if c.get("chunk_kind") != "summary"]
    return any(_part_match(c, [part]) for c in pages)


def missing_required_parts(
    defect: Defect, chunks: list[dict[str, Any]]
) -> list[str]:
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
    text = _bare_where_to_look(step)
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

    Listing / AOR-code checks are one form; Form 28 needs more pages.
    A child defect stays tighter than its parent unless several document
    parts must be compared (Vakalatnama vs petition drafting date).
    """
    named = parts_named_in_where_to_look(defect)
    preferred = preferred_parts_for_defect(defect)
    budget = CATEGORY_MAX_CHUNKS.get(defect.category_id or "", ceiling)
    if len(defect.where_to_look) <= 2 and len(preferred) <= 1:
        budget = min(budget, 3)
    if defect.parent_check_id:
        budget = min(budget, max(4, len(preferred)))
    need = len(named) or len(preferred)
    budget = max(budget, min(need, ceiling))
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
    names = parts_on_page(chunk.get("document_part"))
    if not names or not preferred:
        return False
    preferred_l = [p.lower() for p in preferred]
    return any(
        name.lower() == needle or needle in name.lower()
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

    def rank_key(chunk: dict[str, Any]) -> tuple[int, int, int, float]:
        look_hit = 1 if look_parts and _part_match(chunk, look_parts) else 0
        part_hit = 1 if _part_match(chunk, preferred) else 0
        hits = _term_hits(chunk, terms)
        score = _chunk_score(chunk)
        return (look_hit, part_hit, hits, score)

    ranked = sorted(pages, key=rank_key, reverse=True)
    focused = [
        c
        for c in ranked
        if _part_match(c, preferred) or _term_hits(c, terms) > 0
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
