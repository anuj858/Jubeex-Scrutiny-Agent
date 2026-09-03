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
    "advocate_checklist": ("Advocate's Checklist", "Vakalatnama"),
    "listing_proforma": ("Listing Proforma",),
    "petition_presentation": (
        "Petition",
        "AOR's Declaration",
        "Listing Proforma",
        "Advocate's Checklist",
    ),
    "applications": ("Petition", "Annexures", "Index"),
    "annexures": ("Annexures", "Index", "List of Dates & Events"),
    "parties": ("Memo of Parties", "Cover Page", "Petition"),
    "dates_execution": (
        "Petition",
        "Affidavit",
        "Vakalatnama",
        "PoA/BR",
    ),
    "index_paper_book": ("Index",),
    "limitation": ("Office Report on Limitation", "Petition"),
    # Affidavit is the inspect target; Petition is only retrieval context.
    "affidavit": ("Affidavit", "Petition"),
    "translations": ("Annexures", "Vakalatnama", "PoA/BR"),
    "vakalatnama": ("Vakalatnama", "PoA/BR"),
    "memo_of_appearance": ("Memo of Appearance",),
    "list_of_dates": ("List of Dates & Events", "Synopsis"),
}

# Extra catalogue phrases → Split labels (beyond the config name/description).
_PART_ALIASES: dict[str, tuple[str, ...]] = {
    # Prefer "check the declaration" so "below the declaration" on the
    # checklist form does not pull AOR's Declaration into other checks.
    "AOR's Declaration": (
        "check the declaration",
        "declaration in terms of rule",
        "advocate-on-record declaration",
        "advocate on record declaration",
    ),
    "Impugned Order": (
        "impugned judgment",
        "impugned order",
        "judgment under challenge",
    ),
    "PoA/BR": (
        "power of attorney",
        "board resolution",
    ),
}

# Paper-book Index lists document names; quoting those lines is not proof
# that the named document was filed or signed.
_NON_CONTENT_PARTS = frozenset({"Index"})

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
# do not need the global SCRUTINY_MAX_CHUNKS dump. Multi-part checks then
# raise this to PAGES_PER_TARGET_PART each.
CATEGORY_MAX_CHUNKS: dict[str, int] = {
    "listing_proforma": 3,
    "advocate_checklist": 3,
    "filing_formalities": 10,
    "petition_presentation": 12,
    "applications": 4,
    "annexures": 6,
    "parties": 6,
    "dates_execution": 6,
    "index_paper_book": 4,
    "limitation": 4,
    "affidavit": 6,
    "translations": 4,
    "vakalatnama": 3,
    "memo_of_appearance": 3,
    "list_of_dates": 3,
}

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
_PLACEMENT_PAREN = re.compile(r"\((?:placed|found|located|situated)\b[^)]{0,240}\)", re.I)

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
        for part in parts_on_page(category):
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
    """Retrieval targets: catalogue/parsed parts first, then category defaults."""
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
    blob = " ".join(
        [defect.defect, defect.requirement, *defect.where_to_look]
    ).lower()
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
            p.strip()
            for p in re.split(r"[;|]", defect.trigger_words)
            if p.strip()
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

    category = defect.category_id or ""
    if category == "affidavit":
        primary = [p for p in named if p == "Affidavit"]
        return primary or named[:1]
    if category == "parties":
        primary = [p for p in named if p in {"Petition", "Memo of Parties"}]
        return primary or named[:1]
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
    if (defect.category_id or "") == "index_paper_book":
        return True
    return "Index" in _parts_parsed_from_where_to_look(defect)


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
    targets = parts_named_in_where_to_look(defect) or preferred_parts_for_defect(
        defect
    )
    budget = CATEGORY_MAX_CHUNKS.get(defect.category_id or "", ceiling)
    need = max(len(targets), 1) * PAGES_PER_TARGET_PART
    budget = max(budget, min(need, ceiling))
    # One document part stays small even when the category budget is large
    # (signature audits raise petition_presentation to 12 pages).
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
    excluded = {p.lower() for p in excluded_parts_for_defect(defect)}
    if excluded:
        pages = [
            c
            for c in pages
            if not (
                parts_on_page(c.get("document_part"))
                and {
                    name.lower() for name in parts_on_page(c.get("document_part"))
                }
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
