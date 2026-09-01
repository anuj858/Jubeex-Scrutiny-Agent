"""Prompt construction for defect scrutiny.

Catalogue rows stay in spreadsheet/API wording. This module rewrites them into
an LLM brief so the model is not sent product copy, column labels, or
word-for-word registry phrasing as if it were the instruction.
"""

from __future__ import annotations

import json
import re
from typing import Any

from .rules import Catalogue, Defect, DefectCategory, normalize_filing_type
from ..document_parts import filing_type_label, pinecone_queries_for_defect

MAX_EVIDENCE_CHARS = 60_000

_PRODUCT_PREFIXES = (
    "option for jubeex to ",
    "option for the user to ",
    "prompt the user to ",
    "prompt the user for ",
)

_SEARCH_PREFIX = re.compile(
    r"^(check|read|look at|examine|verify|confirm)\s+(the\s+)?",
    re.IGNORECASE,
)


def _system_prompt(catalogue: Catalogue, filing_type: str | None = None) -> str:
    label = filing_type_label(filing_type)
    filing_block = (
        f"\nThis filing is classified as {label}. Apply only the standards that "
        "belong to this petition type. Checks that apply to every petition type "
        "still apply. Do not treat this as a different kind of petition.\n"
    )

    return f"""You are a pre-filing scrutiny assistant for the {catalogue.jurisdiction}. \
A Scrutiny Assistant at the filing counter would use the same standard. You \
are not the Registry and you do not decide legal validity.
{filing_block}
Decide exactly one defect. Return one status:

- defect_found: the excerpts for the document part you were told to search \
show the required material is missing or incomplete. If you searched those \
parts and the required material is not there, that is a defect (not found)
- compliant: the excerpts show the required material is present and complete
- not_applicable: this defect does not apply to this filing
- not_determined: the check cannot be decided from extracted text because it \
depends on stamps, signatures, seals, wet-ink, or page layout. Do not use \
this for missing text
- needs_review: the excerpts conflict, the wording is legally ambiguous, an \
Advocate-on-Record must decide, confidence is low, OR the excerpts do not \
include the document part you were told to search (for example Vakalatnama \
or Office Report on Limitation)

Rules:

1. Quote the filing. If you cannot quote the required material from the \
parts you were given, it has not been found in those excerpts.
2. If the excerpts do not include a document part named in Where to search, \
return needs_review — not defect_found. Missing excerpts are a retrieval \
gap, not proof the filing lacks that document.
3. If those parts ARE in the excerpts and the required content is still \
missing, return defect_found.
4. You are reading extracted text, not a scanned page. Do not infer stamps, \
signatures, seals, or layout. Those checks alone are not_determined.
5. Do not add requirements that are not in this task. Do not score sibling or \
parent defects.
6. confidence is 0.0 to 1.0. Partial evidence means lower confidence. If you \
would mark defect_found or compliant but confidence is below 0.6, return \
needs_review instead.
7. suggested_fix is allowed only when status is defect_found. Describe what \
the filing itself must contain or attach. Follow the cure aims. Never mention \
Jubeex, auto-generation, uploads, or user-interface options. Otherwise set \
suggested_fix and fix_rationale to null.

Respond with JSON matching the required schema. No prose outside the JSON."""


def _format_category(defect: Defect, category: DefectCategory | None) -> str:
    filing = _filing_phrase(defect.main_category)
    if category:
        return (
            f"\nThis task is in the area “{category.label}”, for {filing}.\n"
            f"{category.prompt.strip()}\n"
        )
    return f"\nThis task is for {filing}.\n"


def _filing_phrase(main_category: str) -> str:
    if normalize_filing_type(main_category) == "global":
        return "this filing (the check applies to every petition type)"
    return f"this {main_category} filing"


def _prune(value: Any) -> Any:
    """Drop nulls and empty containers so the record reads cleanly in a prompt."""
    if isinstance(value, dict):
        cleaned = {k: _prune(v) for k, v in value.items()}
        return {k: v for k, v in cleaned.items() if v not in (None, {}, [], "")}
    if isinstance(value, list):
        pruned = [_prune(v) for v in value]
        return [v for v in pruned if v not in (None, {}, [], "")]
    return value


def _format_record(record: dict[str, Any] | None) -> str:
    if not record:
        return "No structured filing record is available for this document."
    pruned = _prune(record)
    if not pruned:
        return "The structured filing record is empty."
    return json.dumps(pruned, indent=2, ensure_ascii=False, default=str)


def _format_evidence(chunks: list[dict[str, Any]]) -> str:
    if not chunks:
        return "No document excerpts could be retrieved."

    blocks: list[str] = []
    budget = MAX_EVIDENCE_CHARS
    truncated = 0

    for chunk in chunks:
        kind = chunk.get("chunk_kind")
        page = chunk.get("page")
        if kind == "summary":
            header = "[Filing summary — derived from the extracted record]"
        elif page is not None:
            part = chunk.get("document_part")
            page_end = chunk.get("page_end")
            if page_end is not None and page_end != page:
                loc = f"Pages {page}–{page_end}"
            else:
                loc = f"Page {page}"
            header = f"[{loc} — {part}]" if part else f"[{loc}]"
        else:
            header = "[Document excerpt]"

        text = chunk.get("text") or ""
        block = f"{header}\n{text}"
        if len(block) > budget:
            truncated += 1
            continue
        blocks.append(block)
        budget -= len(block)

    if truncated:
        blocks.append(
            f"[{truncated} further excerpt(s) omitted because the evidence budget "
            f"was reached. Treat coverage as incomplete.]"
        )
    return "\n\n".join(blocks)


def _strip_annotation(text: str) -> str:
    return re.sub(r"\[PERFECTLY OKAY\]", "", text, flags=re.IGNORECASE).strip()


def _polish_registry_text(text: str) -> str:
    """Normalise spreadsheet phrasing without changing the legal test."""
    text = _strip_annotation(text)
    text = re.sub(r"\bmust in be in\b", "must be in", text, flags=re.IGNORECASE)
    text = re.sub(r"\bColumns Nos?\.?\s*", "columns ", text, flags=re.IGNORECASE)
    text = re.sub(
        r"Listing Proforma/Proforma for First Listing",
        "Listing Proforma (Proforma for First Listing)",
        text,
        flags=re.IGNORECASE,
    )
    text = re.sub(r"\bPetition/Appeal\b", "petition or appeal", text)
    text = re.sub(r"\bAoR\b", "Advocate-on-Record", text)
    text = re.sub(r"\bAOR('s)?\b", r"Advocate-on-Record\1", text)
    text = re.sub(r"\s+", " ", text).strip().rstrip(" .")
    return text


def _sentence_case(text: str) -> str:
    if not text:
        return text
    return text[0].upper() + text[1:]


def _lc_first(text: str) -> str:
    if not text:
        return text
    keep = (
        "Advocate-on-Record",
        "Form ",
        "Order ",
        "Article ",
        "Supreme ",
        "Listing ",
        "Special ",
        "Civil SLP",
    )
    if text.startswith(keep):
        return text
    if text[0].isupper() and (len(text) == 1 or not text[1].isupper()):
        return text[0].lower() + text[1:]
    return text


def _content_tokens(text: str) -> set[str]:
    stop = {
        "the", "a", "an", "of", "and", "or", "to", "in", "for", "must", "shall",
        "does", "not", "duly", "be", "is", "has", "have", "with", "along",
        "every", "this", "that", "as", "if", "so", "whether",
    }
    words = re.sub(r"[^a-z0-9]+", " ", text.lower()).split()
    return {w for w in words if w not in stop and len(w) > 1}


def _same_substance(left: str, right: str) -> bool:
    a, b = _content_tokens(left), _content_tokens(right)
    if not a or not b:
        return False
    return len(a & b) / min(len(a), len(b)) >= 0.7


def _as_compliance_bar(text: str) -> str:
    """Embed a requirement cell after 'compliant only if' without double modals."""
    t = re.sub(
        r"^(Confirm|Check|Verify|Ensure) that\s+",
        "",
        text,
        flags=re.IGNORECASE,
    )
    t = re.sub(
        r"^No [^.]*?unless (?:it|the petition) contains\s+",
        "the petition contains ",
        t,
        flags=re.IGNORECASE,
    )
    t = re.sub(
        r"\bAdvocate-on-Record must file\b",
        "the Advocate-on-Record has filed",
        t,
        flags=re.IGNORECASE,
    )
    t = re.sub(
        r"\. The checklist must be in accordance with",
        " in accordance with",
        t,
        flags=re.IGNORECASE,
    )
    t = re.sub(r"\bmust be duly filled in\b", "are duly filled in", t, flags=re.I)
    t = re.sub(r"\bof Listing Proforma\b", "of the Listing Proforma", t)
    t = re.sub(r"\s+", " ", t).strip()
    return _lc_first(t)


def _format_standard(defect: Defect) -> str:
    """Turn Defect + Requirement cells into one decision brief."""
    requirement = _polish_registry_text(defect.requirement)
    failure = _polish_registry_text(defect.defect)
    bar = _as_compliance_bar(requirement)

    lines = [
        "Use the following as the legal test for this one check. Restate the "
        "outcome in your own words; do not copy this paragraph back as the finding.",
        "",
    ]
    if _same_substance(requirement, failure):
        lines.append(
            f"The filing is compliant only if {bar}. "
            "Find a defect if that material is missing, blank, or incomplete. "
            "Do not add any further requirement."
        )
    else:
        lines.extend(
            [
                f"The filing is compliant only if {bar}.",
                f"A defect is made out only when the evidence shows that "
                f"{_lc_first(failure)}.",
                "Do not add any further requirement.",
            ]
        )
    return "\n".join(lines)


def _search_step(step: str) -> str:
    text = _polish_registry_text(step)
    text = _SEARCH_PREFIX.sub("", text).strip()
    text = re.sub(
        r"\s*Flag if these details are stated/filed\.?",
        "",
        text,
        flags=re.IGNORECASE,
    )
    text = re.sub(r"\s+and check\s+", ". Also confirm ", text, flags=re.IGNORECASE)
    text = text.strip().rstrip(".")
    if text.lower().startswith("whether "):
        text = "Confirm " + text
    if text:
        text = _sentence_case(text) + "."
    return text


def _strip_product_language(text: str) -> str:
    lowered = text.lower()
    for prefix in _PRODUCT_PREFIXES:
        if lowered.startswith(prefix):
            text = text[len(prefix) :].strip()
            lowered = text.lower()
            break
    text = re.sub(
        r",?\s*(and\s+)?option for jubeex to .+$",
        "",
        text,
        flags=re.IGNORECASE,
    )
    text = re.sub(
        r",?\s*(and\s+)?option for the user to .+$",
        "",
        text,
        flags=re.IGNORECASE,
    )
    text = re.sub(
        r",?\s*(and\s+)?prompt the user to .+$",
        "",
        text,
        flags=re.IGNORECASE,
    )
    text = re.sub(r"\s+with jubeex\.?$", "", text, flags=re.IGNORECASE)
    return text.strip(" ,")


def _split_cure_topic(text: str) -> tuple[str | None, str]:
    for sep in (" — ", " – ", " - "):
        if sep in text:
            head, tail = text.split(sep, 1)
            return head.strip(), tail.strip()
    match = re.search(
        r":\s+(?=Prompt the User|Option for (?:Jubeex|the User))",
        text,
        flags=re.IGNORECASE,
    )
    if match:
        return text[: match.start()].strip(), text[match.end() :].strip()
    return None, text


def _cure_aim(step: str) -> str:
    """Turn product/UI cure copy into what the filing must contain."""
    raw = _polish_registry_text(step)
    topic, text = _split_cure_topic(raw)
    text = _strip_product_language(text)
    text = re.sub(r"^auto-generate\s+", "Include ", text, flags=re.IGNORECASE)
    text = re.sub(r"^draft\s+", "Include ", text, flags=re.IGNORECASE)
    text = re.sub(r"^upload\s+", "File ", text, flags=re.IGNORECASE)
    text = re.sub(r"^re-upload\s+", "Re-file ", text, flags=re.IGNORECASE)
    text = re.sub(r"^supply missing\s+", "Include the missing ", text, flags=re.IGNORECASE)
    text = re.sub(r"^provide details of\s+", "State ", text, flags=re.IGNORECASE)
    text = re.sub(
        r"^review and edit the statement\.?$",
        "the required statement, complete and accurate",
        text,
        flags=re.IGNORECASE,
    )
    text = re.sub(
        r"^enter the information required for\s+",
        "Complete ",
        text,
        flags=re.IGNORECASE,
    )
    text = text.strip().rstrip(".")
    if text and not re.match(
        r"^(Include|File|Re-file|Complete|State|Supply|Add|Insert|Attach)\b",
        text,
        flags=re.IGNORECASE,
    ):
        text = "Include " + text
    if topic and text and not text.lower().startswith(topic.lower()):
        text = f"{topic}: {text}"
    elif not text and topic:
        text = topic
    if text:
        text = _sentence_case(text) + "."
    return text


def _trigger_cues(trigger_words: str | None) -> str | None:
    if not trigger_words or not trigger_words.strip():
        return None
    parts = [p.strip() for p in re.split(r"[;|]", trigger_words) if p.strip()]
    if not parts:
        parts = [trigger_words.strip()]
    quoted = ", ".join(f"“{p}”" for p in parts)
    return (
        "These phrases often mark the required material when it is present: "
        f"{quoted}. Use them to recognise the content. Their absence from the "
        "excerpts is not, by itself, a defect."
    )


def _format_authority(defect: Defect, catalogue: Catalogue | None) -> str:
    lines: list[str] = []
    if catalogue:
        cited = catalogue.sources_cited_by(defect)
        for source in cited:
            line = f"Official source: {source.title} ({source.source_id})"
            if source.url:
                line += f" — {source.url}"
            lines.append(line)
    if defect.location_source:
        lines.append(f"Place in that source: {defect.location_source}.")
    return "\n".join(lines)


def _format_parent(defect: Defect) -> str | None:
    if not defect.parent_check_id:
        return None
    lines = [
        f"This is a narrower check under {defect.parent_check_id}. "
        "Decide only the standard in this task. Do not re-score the parent."
    ]
    if defect.overlap_note:
        note = _sentence_case(_polish_registry_text(defect.overlap_note))
        if not note.lower().startswith("narrower"):
            note = "Keep this distinction: " + note
        lines.append(note + ".")
    return " ".join(lines)


def build_defect_prompt(
    defect: Defect,
    *,
    record: dict[str, Any] | None,
    chunks: list[dict[str, Any]],
    file_name: str | None = None,
    catalogue: Catalogue | None = None,
) -> str:
    """Rewrite one catalogue row into the user message for the model."""
    filing = _filing_phrase(defect.main_category)
    category = catalogue.category_for(defect) if catalogue else None
    category_block = _format_category(defect, category)
    search = "\n".join(
        f"{i}. {_search_step(step)}"
        for i, step in enumerate(defect.where_to_look, start=1)
    )
    cues = _trigger_cues(defect.trigger_words)
    parent = _format_parent(defect)
    excerpts_label = (
        f"## Document excerpts from {file_name}"
        if file_name
        else "## Document excerpts"
    )

    cure_lines: list[str] = []
    seen_topics: set[str] = set()
    for step in defect.how_to_cure:
        aim = _cure_aim(step)
        if not aim:
            continue
        topic = aim.split(":", 1)[0].strip().lower()
        if topic in seen_topics:
            continue
        seen_topics.add(topic)
        cure_lines.append(f"- {aim}")
    sections = [
        category_block.strip(),
        "",
        f"# Task {defect.check_id}",
        (
            f"For {filing}, decide one registry objection. Ignore every other "
            "defect, even if the excerpts mention it."
        ),
        "",
        "## Standard",
        _format_standard(defect),
        "",
        "## Where to search",
        "Use the structured record and the excerpts. Work through this plan:",
        search,
        (
            "A heading without the required content is not compliance. If the "
            "excerpts do not include a part named above (Vakalatnama, Office "
            "Report on Limitation, petition, …), return needs_review — not "
            "defect_found. If those parts are in the excerpts and the required "
            "content is still missing, return defect_found. Use not_determined "
            "only for stamps, signatures, seals, or layout. If you suspect a "
            "defect but confidence is below 0.6, return needs_review."
        ),
    ]
    if cues:
        sections.extend(["", "## Recognition cues", cues])
    sections.extend(
        [
            "",
            "## Authority",
            _format_authority(defect, catalogue),
        ]
    )
    if parent:
        sections.extend(["", "## Scope", parent])
    sections.extend(
        [
            "",
            "## If you find a defect",
            (
                "Write suggested_fix as what must appear in or with the filing. "
                "Use these aims as the substance of the cure — describe the "
                "document contents, not a product workflow:"
            ),
            "\n".join(cure_lines) if cure_lines else (
                "State the missing or incomplete material in filing terms."
            ),
            "",
            "## Structured filing record",
            _format_record(record),
            "",
            excerpts_label,
            _format_evidence(chunks),
            "",
            "## Output",
            (
                f'Return JSON with check_id "{defect.check_id}", status, '
                "confidence, summary, reasoning that quotes the evidence, "
                "evidence, suggested_fix, and fix_rationale."
            ),
        ]
    )
    return "\n".join(sections)


def build_system_prompt(
    catalogue: Catalogue,
    filing_type: str | None = None,
) -> str:
    return _system_prompt(catalogue, filing_type)


def build_evidence_queries(defect: Defect) -> list[str]:
    """Pinecone queries for one defect, taken from where-to-look."""
    return pinecone_queries_for_defect(defect)
