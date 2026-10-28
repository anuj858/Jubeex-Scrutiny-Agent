"""Prompt construction for defect scrutiny.

Catalogue rows stay in spreadsheet/API wording. This module rewrites them into
an LLM brief so the model is not sent product copy, column labels, or
word-for-word registry phrasing as if it were the instruction.
"""

from __future__ import annotations

import ast
import json
import re
from typing import Any

from ..document_parts import (
    catalogue_inspect_parts,
    family_split_name,
    filing_type_label,
    format_document_parts,
    parts_on_page,
    pinecone_queries_for_defect,
)
from ..layout_index import page_entry
from .rules import (
    Catalogue,
    Defect,
    normalize_filing_type,
    split_main_categories,
)

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
        "belong to this petition type. Shared family checks (for example every "
        "SLP check on an SLP Civil filing) and checks that apply to every "
        "petition type still apply. Do not treat this as a different kind of petition.\n"
    )

    return f"""You are a pre-filing scrutiny assistant for the {catalogue.jurisdiction}. \
A Scrutiny Assistant at the filing counter would use the same standard. You \
are not the Registry and you do not decide legal validity.
{filing_block}
Decide exactly one defect. Return one status:

- defect_found: grounded filing evidence positively shows a missing, blank, \
incorrect, inconsistent, or incomplete value. OCR silence or a failed semantic \
search is not proof of absence. Physical layout properties such \
as paper size, margins, duplex printing, font size, and spacing require \
actual layout or visual evidence; do not infer them from OCR silence
- compliant: the excerpts show the required material is present and complete
- not_applicable: this defect does not apply to this filing
- not_determined: do not use this because a stamp, seal, signature, paper \
size, margin, or layout is hard to see. Use it only when the excerpts are \
blank or unreadable so no search was possible
- needs_review: the excerpts conflict, the wording is legally ambiguous, an \
Advocate-on-Record must decide, confidence is low, OR none of the document \
parts this task is inspecting appear in the excerpts

Rules:

1. Quote the filing. A defect requires a grounded observation such as a blank \
field, an incorrect value, an inconsistent date, or text that omits a required \
element inside an otherwise present clause. Merely failing to retrieve the \
required words does not prove that they are absent from the filing. For \
paper size, margins, duplex printing, font size, spacing, or other visual \
layout, return needs_review unless the excerpts include affirmative visual or \
layout evidence of compliance or noncompliance.
2. If the excerpts do not include the document part this task is inspecting \
(listed under Document types to inspect), return needs_review — not defect_found. \
Missing excerpts are a retrieval gap, not proof the filing lacks that \
document. Names used only as landmarks (for example “before the Cover \
Page”) are not extra parts you must have.
3. If those parts are in the excerpts but the evidence does not positively \
show the failure, return needs_review. Return defect_found only when the cited \
filing observation and your reasoning identify the exact failed element.
4. Search the extracted text for required marks. OCR often records stamps \
and seals as words such as STAMP, SEAL, NOTARY, WELFARE FUND, COURT FEE, \
Sd/-, or digitally signed. Do not treat absence of the word A4, a margin \
measurement, or a font name in OCR as proof of a physical-layout defect.
5. Do not add requirements that are not in this task.
6. confidence is 0.0 to 1.0. Partial evidence means lower confidence. If you \
would mark defect_found or compliant but confidence is below 0.8, return \
needs_review instead.
7. evidence is a list of one or more independent items, each with \
{{"chunk_id", "quote"}}. Use multiple items when the decision combines \
different clauses, pages, document parts, or contradictory observations. \
chunk_id is the excerpt label copied exactly, such as c1 from \
“[c1 | Page 12 — Main Petition]”. Do not invent an id. Each quote must be a \
verbatim passage that supports a distinct part of the decision. Return at most \
six evidence items and do not repeat overlapping text. \
If the quote is not in an excerpt, set chunk_id and page to null. Never \
copy a page number from Authority or location_source — those are \
official-rulebook locators, not pages of this filing.
8. Do not quote Index / paper-book listing lines (for example “SLP with \
Affidavit 1+3”) as proof that a document was filed, sworn, or signed. \
Quote the document part itself (Affidavit, Main Petition, Declaration, …). \
If that part is not in the excerpts, return needs_review.
9. A reproduced document from a lower-court record or an Annexure cannot \
satisfy a requirement for the current Supreme Court filing document. Use \
such material only as context. For example, an affidavit inside an Annexure \
is not the affidavit accompanying the petition, and a lower-court memo of \
parties is not the Supreme Court Memo of Parties.
10. Decide applicability before searching for a defect. For standards that \
apply only “if”, “where”, or “in case” a condition exists (for example a \
particular application, power-of-attorney arrangement, multiple signatories, \
or a special case category), return not_applicable when the filing evidence \
shows that condition does not exist. Do not return needs_review merely because \
an inapplicable conditional document is absent. A form field marked N.A. is \
not a defect when the field does not apply to this filing.
11. When the standard says each, every, or all Annexures, one compliant \
Annexure is not proof that the entire set complies. Inspect the distinct \
numbered Annexure excerpts supplied. If any inspected Annexure lacks the \
required endorsement or conflicts with the Index/List of Dates, report the \
defect and cite that Annexure. If the excerpts are only a sample and none \
shows a defect, return needs_review rather than compliant.
12. Distinguish a missing paragraph from defective content inside a present \
paragraph. If the target heading or paragraph is present, acknowledge it and \
compare its actual words with every element of the Standard. Do not say the \
paragraph or declaration is absent merely because its wording is incomplete.
13. Make observed and conclusion match the quoted evidence precisely. Name the \
exact missing or satisfied element (for example endorsement wording versus the \
Advocate-on-Record's accepting signature). Do not claim a visible phrase, \
signature, date, paragraph, or document is missing when the quoted excerpt \
shows it; explain what remains deficient instead.
14. Treat visible stamps, handwritten dates, signatures, seals, and equivalent \
wording as evidence. If OCR and stored visual observations conflict, return \
needs_review rather than defect_found.
15. Do not require an exact phrase when the requirement can be satisfied by \
equivalent wording, unless the Standard explicitly says the exact wording is mandatory.

How to decide:

- When a Note for AI is present, its decision paragraph controls the status for this row.
- Fill elements with one item for each element that the Note or Requirement names, in the same order. Write only the element name; omit source-list labels such as "a.", "b.", "(a)", or "(b)".
- The excerpts are not the whole filing. If a document or element is not in the excerpts, set gap. Do not say it is absent from the filing.

How to write observed and conclusion:

- Write to an Advocate-on-Record, in plain sentences.
- Name the document and the paragraph or clause, for example "paragraph 1A of the Main Petition".
- Do not mention excerpt ids such as c1 or c2, page numbers, chunks, search, retrieval, or OCR. Page numbers are added later.
- Cite a rule only when that citation is written in the Requirement.
- For defect_found, say what the filing shows and which element is missing or incomplete.
- For needs_review because the relevant pages were not shown, name the document and the element that was not available. Do not invent the contents of pages you were not shown.
- Example, the clause was not shown: "The pages of the Main Petition that were shown omit paragraph 1A. Whether a Letters Patent Appeal or Writ Appeal lies cannot be determined."
- Example, the document is present but incomplete: "The Appendix reproduces some provisions but omits a provision referred to in the impugned order."

Respond with JSON matching the required schema. No prose outside the JSON."""


def _format_category(defect: Defect) -> str:
    filing = _filing_phrase(defect.main_category)
    return f"\nThis task is for {filing}.\n"


def _filing_phrase(main_category: str) -> str:
    parts = split_main_categories(main_category)
    if len(parts) > 1:
        return "this filing (the check applies to " + ", ".join(parts) + ")"
    label = parts[0] if parts else main_category
    key = normalize_filing_type(label)
    if key == "global":
        return "this filing (the check applies to every petition type)"
    if key in {
        "slp",
        "transfer_petition",
        "writ_petition",
        "review_petition",
        "contempt_petition",
        "curative_petition",
    }:
        return f"this {label} filing (shared across civil and criminal)"
    return f"this {label} filing"


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


def _document_page(
    chunk: dict[str, Any], layout: dict[int, dict[str, Any]] | None
) -> int | None:
    global_page = _safe_int(chunk.get("page"))
    if global_page is None:
        return None
    if layout:
        meta = page_entry(layout, global_page)
        local = _safe_int((meta or {}).get("local_page")) if meta else None
        if local is not None:
            return local
    return global_page


def _format_evidence(
    chunks: list[dict[str, Any]],
    layout: dict[int, dict[str, Any]] | None = None,
    *,
    preserve_excerpt_ids: bool = False,
    max_chars: int = MAX_EVIDENCE_CHARS,
) -> str:
    if not chunks:
        return "No document excerpts could be retrieved."

    if not preserve_excerpt_ids:
        for chunk in chunks:
            chunk.pop("excerpt_id", None)

    blocks: list[str] = []
    budget = max_chars
    truncated = 0
    excerpt_n = 0

    for chunk in chunks:
        kind = chunk.get("chunk_kind")
        page = chunk.get("page")
        label = ""
        if kind != "summary" and page is not None:
            if preserve_excerpt_ids:
                label = str(chunk.get("excerpt_id") or "").strip()
            if not label:
                excerpt_n += 1
                label = f"c{excerpt_n}"
        if kind == "summary":
            header = "[Filing summary — derived from the extracted record]"
        elif page is not None:
            part = format_document_parts(chunk.get("document_part"))
            loc = f"Page {page}"
            place = f"{loc} — {part}" if part else loc
            provenance: list[str] = []
            original_page = chunk.get("original_page")
            if original_page not in (None, "", page, str(page)):
                provenance.append(f"original compiled page {original_page}")
            printed = chunk.get("printed_page_number")
            if printed not in (None, "", page, str(page)):
                provenance.append(f"printed page {printed}")
            headings = chunk.get("heading_path") or []
            if isinstance(headings, list) and headings:
                provenance.append("heading: " + " > ".join(str(v) for v in headings))
            row_start = chunk.get("table_row_start")
            row_end = chunk.get("table_row_end")
            if row_start is not None:
                rows = (
                    str(row_start)
                    if row_end in (None, row_start)
                    else f"{row_start}-{row_end}"
                )
                provenance.append(f"table rows {rows}")
            if provenance:
                place += " | " + " | ".join(provenance)
            header = f"[{label} | {place}]" if label else f"[{place}]"
        else:
            header = "[Document excerpt]"

        text = chunk.get("text") or ""
        block = f"{header}\n{text}"
        if len(block) > budget:
            if label and not preserve_excerpt_ids:
                excerpt_n -= 1
            truncated += 1
            continue
        if label:
            chunk["excerpt_id"] = label
        blocks.append(block)
        budget -= len(block)

    if truncated:
        blocks.append(
            f"[{truncated} further excerpt(s) omitted because the evidence budget "
            f"was reached. Treat coverage as incomplete.]"
        )
    return "\n\n".join(blocks)


def _batch_chunk_key(chunk: dict[str, Any]) -> tuple[Any, ...]:
    """Stable identity used to print a batch excerpt only once."""
    record_id = str(chunk.get("record_id") or "").strip()
    if record_id:
        return ("record", record_id)
    return (
        "content",
        chunk.get("chunk_kind"),
        chunk.get("page"),
        chunk.get("page_end"),
        format_document_parts(chunk.get("document_part")),
        str(chunk.get("text") or ""),
    )


def _prepare_batch_evidence(
    tasks: list[tuple[Defect, dict[str, Any] | None, list[dict[str, Any]]]],
    layout: dict[int, dict[str, Any]] | None,
) -> tuple[str, dict[str, list[str]]]:
    """Deduplicate excerpts while preserving each task's citation boundary."""
    pooled: list[dict[str, Any]] = []
    by_key: dict[tuple[Any, ...], dict[str, Any]] = {}
    allowed: dict[str, list[str]] = {}
    next_id = 1
    for defect, _record, chunks in tasks:
        task_ids: list[str] = []
        for chunk in chunks:
            key = _batch_chunk_key(chunk)
            shared = by_key.get(key)
            if shared is None:
                shared = chunk
                if chunk.get("chunk_kind") != "summary" and chunk.get("page") is not None:
                    shared["excerpt_id"] = f"c{next_id}"
                    next_id += 1
                by_key[key] = shared
                pooled.append(shared)
            excerpt_id = str(shared.get("excerpt_id") or "").strip()
            if excerpt_id:
                chunk["excerpt_id"] = excerpt_id
                if excerpt_id not in task_ids:
                    task_ids.append(excerpt_id)
        allowed[defect.check_id] = task_ids
    return (
        _format_evidence(
            pooled,
            layout,
            preserve_excerpt_ids=True,
            max_chars=MAX_EVIDENCE_CHARS * max(1, len(tasks)),
        ),
        allowed,
    )


def _visual_part_matches(document_type: str, targets: list[str]) -> bool:
    found = family_split_name(document_type)
    for target in targets:
        if document_type.casefold() == target.casefold():
            return True
        if found == family_split_name(target):
            return True
    return False


def _format_visual_observations(
    defect: Defect, visual_index: dict[str, Any] | None
) -> str:
    """Format positive stored visual detections for the inspected parts."""
    if not isinstance(visual_index, dict):
        return "No stored visual observations are available."
    targets = catalogue_inspect_parts(defect)
    lines: list[str] = []
    for mark in visual_index.get("marks") or []:
        if not isinstance(mark, dict):
            continue
        document_type = str(mark.get("document_type") or "").strip()
        if targets and not _visual_part_matches(document_type, targets):
            continue
        marking_type = str(mark.get("marking_type") or "").strip()
        if not document_type or not marking_type:
            continue
        page = mark.get("local_page") or mark.get("page")
        details = [marking_type.replace("_", " ")]
        role = str(mark.get("signature_role") or "").strip()
        if role and role not in {"not_applicable", "unknown"}:
            details.append(f"role: {role}")
        label = str(mark.get("associated_label") or "").strip()
        if label:
            details.append(f"near label: {label}")
        visible = str(mark.get("visible_text") or "").strip()
        if visible:
            details.append(f"visible text: {visible}")
        lines.append(f"- {document_type}, page {page}: " + "; ".join(details))
        if len(lines) >= 20:
            break
    if not lines:
        return (
            "No positive visual mark relevant to these document parts was stored. "
            "This absence is not proof that a signature, stamp, seal, or handwritten "
            "value is missing."
        )
    return (
        "These are positive detections only; absence from this list is not proof of "
        "absence. Do not infer authenticity or signer identity.\n" + "\n".join(lines)
    )


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
        "the",
        "a",
        "an",
        "of",
        "and",
        "or",
        "to",
        "in",
        "for",
        "must",
        "shall",
        "does",
        "not",
        "duly",
        "be",
        "is",
        "has",
        "have",
        "with",
        "along",
        "every",
        "this",
        "that",
        "as",
        "if",
        "so",
        "whether",
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
    t = re.sub(
        r"\bmust be duly filled in\b",
        "are duly filled in",
        t,
        flags=re.IGNORECASE,
    )
    t = re.sub(r"\bof Listing Proforma\b", "of the Listing Proforma", t)
    t = re.sub(r"\s+", " ", t).strip()
    return _lc_first(t)


def _format_standard(defect: Defect) -> str:
    """Turn Defect + Requirement cells into one decision brief."""
    requirement = _polish_registry_text(defect.requirement)
    failure = _polish_registry_text(defect.defect)
    bar = _as_compliance_bar(requirement)

    lines = [
        (
            "Use the following as the legal test for this one check. Restate the "
            "outcome in your own words; do not copy this paragraph back as the finding."
        ),
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
                (
                    "A defect is made out only when the evidence shows that "
                    f"{_lc_first(failure)}."
                ),
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
    """Remove Jubeex/UI wrappers; keep the filing substance from the catalogue."""
    text = text.strip(" ,")
    # Numbered catalogue rows ("1. Prompt the User to …") must lose the number
    # before prefix matching, otherwise mid-string wipes leave only "1.".
    text = re.sub(r"^\d+[\.)]\s*", "", text).strip()
    changed = True
    while changed and text:
        changed = False
        lowered = text.lower()
        for prefix in _PRODUCT_PREFIXES:
            if lowered.startswith(prefix):
                text = text[len(prefix) :].strip()
                changed = True
                break
    # Drop trailing product-only tails; do not erase the clause before them.
    text = re.sub(
        r",?\s*and\s+auto-fill(?:\s+the\s+same)?(?:\s+in\b.*)?$",
        "",
        text,
        flags=re.IGNORECASE,
    )
    text = re.sub(r"\s+with jubeex\.?$", "", text, flags=re.IGNORECASE)
    text = re.sub(
        r",?\s*and\s+prompt the user to .+$",
        "",
        text,
        flags=re.IGNORECASE,
    )
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
    text = re.sub(
        r"^supply missing\s+", "Include the missing ", text, flags=re.IGNORECASE
    )
    text = re.sub(r"^provide details of\s+", "State ", text, flags=re.IGNORECASE)
    text = re.sub(r"^provide\s+", "Provide ", text, flags=re.IGNORECASE)
    text = re.sub(r"^edit\s+", "Edit ", text, flags=re.IGNORECASE)
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
    # Bare numbering leftovers from a bad strip — drop them.
    if re.fullmatch(r"\d+", text or ""):
        text = ""
    if text and not re.match(
        r"^(Include|File|Re-file|Complete|State|Supply|Add|Insert|Attach|"
        r"Provide|Edit|Correct)\b",
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


def display_cure_steps(steps: list[str] | None) -> list[str]:
    """Catalogue how_to_cure for the finding UI — keep registry wording.

    Product chrome ("with Jubeex") is dropped; numbered steps and substance
    from sci_registry_defects stay intact.
    """
    cleaned: list[str] = []
    for step in steps or []:
        text = _polish_registry_text(str(step))
        text = re.sub(r"\s+with jubeex\.?$", "", text, flags=re.IGNORECASE).strip()
        if not text:
            continue
        if not text.endswith("."):
            text = text + "."
        if text not in cleaned:
            cleaned.append(text)
    return cleaned


def _format_authority(defect: Defect, catalogue: Catalogue | None) -> str:
    lines: list[str] = [
        (
            "These locators point at the official circular, form or handbook. "
            "Their page numbers are NOT pages of this filing. Do not copy them "
            "into evidence.page."
        )
    ]
    if catalogue:
        cited = catalogue.sources_cited_by(defect)
        for source in cited:
            line = f"Official source: {source.title} ({source.source_id})"
            if source.url:
                line += f" — {source.url}"
            lines.append(line)
    if defect.location_source:
        lines.append(f"Rulebook locator: {defect.location_source}.")
    return "\n".join(lines)


def _short_objection(text: str) -> str:
    """One readable heading from a catalogue Defect cell."""
    t = _polish_registry_text(text)
    t = re.sub(r"^(The|A|An)\s+", "", t, flags=re.IGNORECASE)
    first = re.split(r"(?<=\.)\s+", t, maxsplit=1)[0].rstrip(".")
    if len(first) > 140:
        first = first[:137].rsplit(" ", 1)[0].rstrip(",;") + "…"
    return _sentence_case(first) if first else _polish_registry_text(text)


def finding_title(defect: Defect, catalogue: Catalogue | None = None) -> str:
    """Short UI/API title from the objection. Never a serial number."""
    objection = _short_objection(defect.defect)
    objection = re.sub(
        r"^(?:D-\d{1,4}[A-Z]?|\d{1,4})[.):\s-]+",
        "",
        objection,
        count=1,
        flags=re.IGNORECASE,
    ).strip()
    label = (defect.special_category or "").strip() or None
    if label:
        folded_label = label.lower()
        if not objection.lower().startswith(folded_label):
            return f"{label}: {objection}"
        return objection
    return objection


def readable_location_source(defect: Defect, catalogue: Catalogue | None = None) -> str:
    """Official-source citation. Never looks like a page of this filing."""
    raw = (defect.location_source or "").strip()
    if not raw or "did not give" in raw.lower():
        return "Official source page missing — not recorded in the catalogue."

    titles: list[str] = []
    if catalogue:
        titles = [source.title for source in catalogue.sources_cited_by(defect)]
    page_match = re.search(
        r"pages?\s+(\d+(?:\s*[-–]\s*\d+)?)", raw, flags=re.IGNORECASE
    )
    page = page_match.group(1).replace(" ", "") if page_match else None
    if titles:
        head = titles[0] if len(titles) == 1 else "; ".join(titles)
        if page:
            return f"Official source (not a page of this filing): {head}, page {page}"
        return (
            f"Official source page missing — {head} "
            f"(rulebook page not recorded; this is not a filing page)."
        )
    cleaned = re.sub(r"\s+", " ", raw)
    if page:
        return f"Official source (not a page of this filing): {cleaned}"
    return (
        f"Official source page missing — {cleaned} "
        f"(rulebook page not recorded; this is not a filing page)."
    )


def part_label(value: Any) -> str:
    """Document-type name. A stored list is never printed as ['Main Petition']."""
    if isinstance(value, str):
        text = value.strip()
        if text.startswith("[") and text.endswith("]"):
            try:
                parsed = ast.literal_eval(text)
            except (ValueError, SyntaxError):
                parsed = None
            if isinstance(parsed, (list, tuple)):
                value = list(parsed)
            else:
                value = text
        else:
            value = text
    return format_document_parts(value)


def _page_phrase(pages: list[int]) -> str:
    ordered = sorted({int(page) for page in pages if page is not None})
    if not ordered:
        return ""
    if len(ordered) == 1:
        return f"page {ordered[0]}"
    contiguous = ordered == list(range(ordered[0], ordered[-1] + 1))
    if contiguous and len(ordered) > 1:
        return f"pages {ordered[0]}–{ordered[-1]}"
    if len(ordered) == 2:
        return f"pages {ordered[0]} and {ordered[1]}"
    body = ", ".join(str(page) for page in ordered[:-1])
    return f"pages {body} and {ordered[-1]}"


def _format_part_pages(pages_by_part: dict[str, list[int]]) -> str:
    pieces: list[str] = []
    for label, pages in pages_by_part.items():
        phrase = _page_phrase(pages)
        name = label or "Filing"
        pieces.append(f"{name} — {phrase}." if phrase else f"{name}.")
    return " ".join(pieces)


def filing_location(
    *,
    evidence_pages: list[int] | None = None,
    reviewed_pages: list[int] | None = None,
    document_parts: list[str] | None = None,
    pages_by_part: dict[str, list[int]] | None = None,
) -> str:
    """Each document's own pages, grouped by document type."""
    if pages_by_part:
        text = _format_part_pages(pages_by_part)
        if text:
            return text
    labels = [part_label(part) for part in (document_parts or [])]
    labels = [label for label in labels if label]
    label = ", ".join(dict.fromkeys(labels))
    cited = sorted({int(page) for page in (evidence_pages or []) if page is not None})
    if cited:
        phrase = _page_phrase(cited)
        if label:
            return f"{label} — {phrase}."
        return f"{phrase[:1].upper()}{phrase[1:]}."
    reviewed = sorted(
        {int(page) for page in (reviewed_pages or []) if page is not None}
    )
    if reviewed:
        phrase = _page_phrase(reviewed[:8])
        if len(reviewed) > 8:
            phrase += ", …"
        if label:
            return f"{label} — {phrase}."
        return f"{phrase[:1].upper()}{phrase[1:]}."
    return "No page was identified in the pages reviewed."


def finding_location(
    defect: Defect,
    catalogue: Catalogue | None = None,
    *,
    evidence_pages: list[int] | None = None,
    reviewed_pages: list[int] | None = None,
    document_parts: list[str] | None = None,
) -> str:
    """Filing page first, then the official source. States when a page is missing."""
    return (
        f"{filing_location(evidence_pages=evidence_pages, reviewed_pages=reviewed_pages, document_parts=document_parts)} "
        f"{readable_location_source(defect, catalogue)}"
    )


_SOURCE_ID = re.compile(
    r"\bSCI_(?:CHECKLIST_2025|COURT_PROCESSES_HANDBOOK|RULES_2013|RULES_CURRENT|"
    r"FORM_28|DEFAULT_LIST|FORMS_PAGE)\b"
)
_RULEBOOK_PAGE = re.compile(
    r"(?:page|pages|pg\.?)\s+\d+(?:\s*[-–]\s*\d+)?\s+of the PDF",
    re.IGNORECASE,
)
_WEAK_REASONING = re.compile(
    r"\b(cannot see|can't see|cannot determine|undetermined|visual(?:ly)?|"
    r"not_determined|extracted text only|hard to see|cannot be decided from)\b",
    re.IGNORECASE,
)


def _filing_page_phrase(
    pages: list[int] | None,
    evidence_pages: list[int] | None,
    document_parts: list[str] | None = None,
) -> str:
    part = ", ".join(document_parts or [])
    cited = [p for p in (evidence_pages or []) if p is not None]
    if cited:
        unique = sorted(set(cited))
        prefix = f"{part}, " if part else ""
        if len(unique) == 1:
            return f" See {prefix}page {unique[0]}."
        return f" See {prefix}pages {', '.join(str(p) for p in unique)}."
    reviewed = [p for p in (pages or []) if p is not None]
    if reviewed:
        unique = sorted(set(reviewed))
        shown = unique[:8]
        extra = "" if len(unique) <= 8 else ", …"
        label = "pages" if len(shown) > 1 else "page"
        return (
            f" Page missing on the citation. "
            f"Excerpts were reviewed on {label} "
            f"{', '.join(str(p) for p in shown)}{extra}."
        )
    return " Page missing — no page was identified in the retrieved excerpts."


def fallback_reasoning(
    defect: Defect,
    status: str,
    *,
    pages: list[int] | None = None,
    evidence_pages: list[int] | None = None,
    evidence_quote: str | None = None,
    document_parts: list[str] | None = None,
) -> str:
    """Plain-language reason tied to this catalogue row when the model copy is unusable."""
    requirement = _polish_registry_text(defect.requirement)
    page_bit = _filing_page_phrase(pages, evidence_pages, document_parts)
    part = ", ".join(document_parts or []) or "the inspected document part"
    quote = " ".join((evidence_quote or "").split())
    if len(quote) > 240:
        quote = quote[:237].rstrip() + "…"
    observed = (
        f'Observed in {part}: "{quote}"'
        if quote
        else f"Observed in {part}: no conclusive filing-page value was grounded"
    )
    if status == "defect_found":
        return (
            f"{observed}. Required: {requirement}. "
            f"Conclusion: the observed filing content does not meet that requirement."
            f"{page_bit}"
        )
    if status == "compliant":
        return (
            f"{observed}. Required: {requirement}. "
            f"Conclusion: the observed filing content meets that requirement.{page_bit}"
        )
    if status == "needs_review":
        return (
            f"{observed}. Required: {requirement}. Conclusion: the available "
            f"evidence does not safely prove compliance or a defect, so human review "
            f"is required.{page_bit}"
        )
    if status == "not_applicable":
        return (
            f"This check does not apply to this filing. "
            f"The catalogue test is: {requirement}"
        )
    return f"This check could not be completed. The test is: {requirement}{page_bit}"


def _strip_rulebook_noise(text: str) -> str:
    cleaned = _SOURCE_ID.sub("the official circular", text or "")
    cleaned = _RULEBOOK_PAGE.sub("", cleaned)
    cleaned = re.sub(r"\brulebook locator:?\s*", "", cleaned, flags=re.IGNORECASE)
    cleaned = re.sub(r"\blocation_source:?\s*", "", cleaned, flags=re.IGNORECASE)
    cleaned = re.sub(r"\s+", " ", cleaned).strip(" ,;.")
    return cleaned


def validated_reasoning(
    defect: Defect,
    text: str,
    status: str,
    *,
    pages: list[int] | None = None,
    evidence_pages: list[int] | None = None,
    evidence_quote: str | None = None,
    document_parts: list[str] | None = None,
) -> str:
    """Clean legacy free-form reasoning while retaining source-page context."""
    cleaned = _strip_rulebook_noise(text)
    cleaned = re.sub(
        r"^The\s+.{1,80}?\s+was\s+(?:inspected|examined|reviewed)\s+at\s+"
        r"pages?\s+\d+(?:\s*[-–]\s*\d+)?\s*(?:\(chunk\s+[\w-]+\))?\.\s*",
        "",
        cleaned,
        flags=re.IGNORECASE,
    )
    cleaned = re.sub(r"\(?\bchunk\s+[a-z0-9_-]+\)?", "", cleaned, flags=re.IGNORECASE)
    cleaned = re.sub(r"\s+", " ", cleaned).strip(" ,;.")
    if _copy_is_weak(cleaned, defect):
        return fallback_reasoning(
            defect,
            status,
            pages=pages,
            evidence_pages=evidence_pages,
            evidence_quote=evidence_quote,
            document_parts=document_parts,
        )
    if status == "defect_found":
        if not re.match(r"^Observed\s*:", cleaned, flags=re.IGNORECASE):
            cleaned = f"Observed: {cleaned}."
        if not re.search(r"\bRequired\s*:", cleaned, flags=re.IGNORECASE):
            cleaned += f" Required: {_polish_registry_text(defect.requirement)}."
        if not re.search(r"\bConclusion\s*:", cleaned, flags=re.IGNORECASE):
            cleaned += (
                " Conclusion: the cited filing content does not meet this requirement."
            )
    if not re.search(r"\bpages?\s+\d+", cleaned, flags=re.IGNORECASE):
        cleaned = (
            cleaned.rstrip(".")
            + "."
            + _filing_page_phrase(pages, evidence_pages, document_parts)
        )
    return cleaned


def _copy_is_weak(text: str, defect: Defect) -> bool:
    if not text or len(text) < 40:
        return True
    if _WEAK_REASONING.search(text):
        return True
    if defect.location_source and defect.location_source.lower() in text.lower():
        return True
    topic = _content_tokens(defect.defect) | _content_tokens(defect.requirement)
    words = _content_tokens(text)
    return bool(
        topic and words and len(topic & words) / min(len(topic), len(words)) < 0.12
    )


def validated_summary(
    defect: Defect,
    text: str,
    status: str,
    *,
    pages: list[int] | None = None,
    evidence_pages: list[int] | None = None,
) -> str:
    cleaned = _strip_rulebook_noise(text)
    if _copy_is_weak(cleaned, defect):
        requirement = _lc_first(_polish_registry_text(defect.requirement))
        if status == "defect_found":
            return f"Defect: {requirement}."
        if status == "compliant":
            return f"Compliant: {requirement}."
        if status == "needs_review":
            return f"Needs review against this check: {requirement}."
        if status == "not_applicable":
            return "This check does not apply to this filing."
        return f"This check could not be completed: {requirement}."
    return cleaned


_RULE_CITE = re.compile(
    r"\b(?:Order\s+[IVXLC]+(?:\s*,?\s*Rule\s+[\d()A-Za-z]+)?|Article\s+\d+[A-Z]?|"
    r"Form\s+No\.?\s+\d+)\b",
    re.IGNORECASE,
)
_CHUNK_ID = re.compile(r"\bc\d+\b", re.IGNORECASE)
_PIPELINE_WORD = re.compile(
    r"\b(?:excerpts?|chunks?|pinecone|retrieval)\b", re.IGNORECASE
)


def _element_bits(response: Any) -> list[Any]:
    items = getattr(response, "elements", None) or []
    bits: list[Any] = []
    for item in items:
        if isinstance(item, dict):
            bits.append(item)
            continue
        bits.append(item)
    return bits


def _element_name(item: Any) -> str:
    if isinstance(item, dict):
        return str(item.get("name") or "").strip()
    return str(getattr(item, "name", "") or "").strip()


def _element_state(item: Any) -> str:
    if isinstance(item, dict):
        return str(item.get("state") or "").strip()
    return str(getattr(item, "state", "") or "").strip()


def _element_detail(item: Any) -> str:
    if isinstance(item, dict):
        return str(item.get("detail") or "").strip()
    return str(getattr(item, "detail", "") or "").strip()


def _safe_int(value: Any) -> int | None:
    try:
        if value is None or value == "":
            return None
        return int(value)
    except (TypeError, ValueError):
        return None


def _layout_pages_for(
    layout: dict[int, dict[str, Any]] | None, part: str
) -> tuple[list[int], list[int]]:
    """Global pages and the document's own pages for one part."""
    global_pages: list[int] = []
    local_pages: list[int] = []
    if not layout or not part:
        return global_pages, local_pages
    wanted = part.casefold()
    for key, meta in layout.items():
        if not isinstance(meta, dict):
            continue
        names = parts_on_page(meta.get("document_part"))
        if not any(name.casefold() == wanted for name in names):
            continue
        number = _safe_int(key)
        if number is None:
            continue
        global_pages.append(number)
        local = _safe_int(meta.get("local_page"))
        local_pages.append(local if local is not None else number)
    return global_pages, local_pages


def _sent_pages(chunks: list[dict[str, Any]] | None) -> set[int]:
    sent: set[int] = set()
    for chunk in chunks or []:
        if chunk.get("chunk_kind") == "summary":
            continue
        number = _safe_int(chunk.get("page"))
        if number is not None:
            sent.add(number)
    return sent


def _fully_reviewed(
    layout: dict[int, dict[str, Any]] | None,
    part: str,
    chunks: list[dict[str, Any]] | None,
) -> bool:
    global_pages, _local = _layout_pages_for(layout, part)
    if not global_pages:
        return False
    return set(global_pages) <= _sent_pages(chunks)


def _sent_local_by_part(
    chunks: list[dict[str, Any]] | None,
    layout: dict[int, dict[str, Any]] | None,
) -> dict[str, list[int]]:
    groups: dict[str, list[int]] = {}
    for chunk in chunks or []:
        if chunk.get("chunk_kind") == "summary":
            continue
        global_page = _safe_int(chunk.get("page"))
        if global_page is None:
            continue
        meta = page_entry(layout, global_page) if layout else None
        names = parts_on_page((meta or {}).get("document_part")) or parts_on_page(
            chunk.get("document_part")
        )
        label = " / ".join(names)
        if not label:
            continue
        local = _safe_int((meta or {}).get("local_page")) if meta else None
        bucket = groups.setdefault(label, [])
        if local is not None and local not in bucket:
            bucket.append(local)
    return groups


def _evidence_by_part(evidence: list[Any] | None) -> dict[str, list[int]]:
    groups: dict[str, list[int]] = {}
    for ref in evidence or []:
        page = _safe_int(getattr(ref, "page", None))
        if page is None:
            continue
        label = part_label(getattr(ref, "document_part", None)) or "the document"
        bucket = groups.setdefault(label, [])
        if page not in bucket:
            bucket.append(page)
    return groups


def _inspected(label: str, pages: list[int], *, full: bool = False) -> str:
    name = label or "document"
    subject = _with_definite_article(name)
    phrase = _page_phrase(pages)
    if full and phrase:
        return f"{subject} was inspected in full ({phrase})"
    if phrase:
        return f"{subject} was inspected at {phrase}"
    return f"{subject} was inspected"


def _primary_part(defect: Any, groups: dict[str, list[int]]) -> tuple[str, list[int]]:
    inspect = [
        part_label(part) for part in (getattr(defect, "inspect_parts", None) or [])
    ]
    inspect = [part for part in inspect if part]
    for part in inspect:
        for label, pages in groups.items():
            if part.casefold() == label.casefold():
                return label, pages
    if groups:
        label, pages = next(iter(groups.items()))
        return label, pages
    return (inspect[0] if inspect else "document"), []


def _compliance_with(defect: Any) -> str:
    rule = str(getattr(defect, "applicable_rule", "") or "").strip()
    if rule and rule.casefold() not in {"n.a", "na", "n/a", "n.a."}:
        return rule
    requirement = str(getattr(defect, "requirement", "") or "")
    match = re.search(
        r"Order\s+[IVXLC]+(?:\s*,?\s*Rule\s+[\d()A-Za-z]+)?"
        r"(?:\s+of\s+the\s+Supreme\s+Court\s+Rules,\s*\d{4})?",
        requirement,
        flags=re.IGNORECASE,
    )
    if match:
        return match.group(0).strip()
    return "this requirement"


def _allowed_citations(defect: Any) -> str:
    return " ".join(
        [
            str(getattr(defect, "requirement", "") or ""),
            str(getattr(defect, "applicable_rule", "") or ""),
            str(getattr(defect, "defect", "") or ""),
        ]
    ).casefold()


def _drop_uncited_rules(text: str, defect: Any) -> str:
    allowed = _allowed_citations(defect)

    def keep(match: re.Match[str]) -> str:
        citation = match.group(0)
        if citation.casefold() in allowed:
            return citation
        return ""

    return _RULE_CITE.sub(keep, text)


def _clean_model_text(text: str, defect: Any) -> str:
    cleaned = _strip_rulebook_noise(text or "")
    cleaned = _CHUNK_ID.sub("", cleaned)
    cleaned = _PIPELINE_WORD.sub("", cleaned)
    cleaned = _drop_uncited_rules(cleaned, defect)
    cleaned = re.sub(r"\s{2,}", " ", cleaned)
    cleaned = re.sub(r"\s+([,.;])", r"\1", cleaned)
    return cleaned.strip(" ,;.")


def _lacking_phrase(response: Any) -> str:
    lacking = [
        item
        for item in _element_bits(response)
        if _element_state(item) in {"missing", "incomplete"}
    ]
    if not lacking:
        return ""
    pieces: list[str] = []
    for item in lacking:
        name = _unlabel_element_name(_element_name(item))
        detail = _element_detail(item)
        if name and detail:
            pieces.append(f"{name} ({detail})")
        elif name or detail:
            pieces.append(name or detail)
    if not pieces:
        return ""
    if len(pieces) == 1:
        return pieces[0]
    return _english_list(pieces)


_ELEMENT_LABEL = re.compile(r"^\s*(?:\(([A-Za-z]|\d+)\)|([A-Za-z]|\d+)[.)])\s*")


def _unlabel_element_name(raw_name: str) -> str:
    """Remove catalogue/prompt labels that have no meaning outside their source list."""
    match = _ELEMENT_LABEL.match(raw_name)
    if match:
        return raw_name[match.end() :].strip().rstrip(" .")
    return raw_name.strip().rstrip(" .")


def _element_list_lines(response: Any) -> list[str]:
    """Render only missing elements as a fresh, self-contained numbered list."""
    lacking = [
        item
        for item in _element_bits(response)
        if _element_state(item) in {"missing", "incomplete"}
    ]
    lines: list[str] = []
    for index, item in enumerate(lacking):
        raw_name = _element_name(item)
        detail = _element_detail(item).strip().rstrip(" .")
        name = _unlabel_element_name(raw_name)
        description = name or detail or "Required element"
        if name and detail and detail.casefold() != name.casefold():
            description = f"{name} — {detail}"
        lines.append(_finish(f"{index + 1}) {description}"))
    return lines


def _finish_reasoning_blocks(blocks: list[str]) -> str:
    return "\n\n".join(_finish(block) for block in blocks if block.strip())


def _english_list(items: list[str]) -> str:
    if len(items) == 1:
        return items[0]
    if len(items) == 2:
        return f"{items[0]} and {items[1]}"
    return ", ".join(items[:-1]) + f" and {items[-1]}"


def _finish(text: str) -> str:
    cleaned = re.sub(r"\s{2,}", " ", text).strip()
    cleaned = re.sub(r"\s+([,.;])", r"\1", cleaned)
    cleaned = re.sub(r"\.{2,}", ".", cleaned)
    if cleaned and cleaned[-1] not in ".!?:":
        cleaned += "."
    return cleaned


def _with_definite_article(value: str) -> str:
    """Prefix a document name with 'The' only when it has no article already."""
    cleaned = value.strip()
    if cleaned.casefold().startswith("the "):
        return f"The {cleaned[4:].strip()}"
    return f"The {cleaned}"


def _as_object_phrase(value: str) -> str:
    """Return a phrase suitable after a verb, without an intrusive capital 'The'."""
    cleaned = value.strip()
    if cleaned.casefold().startswith("the "):
        cleaned = cleaned[4:].strip()
    return f"the {cleaned}" if cleaned else "the required portion"


def compose_reasoning(
    defect: Any,
    response: Any,
    *,
    chunks: list[dict[str, Any]] | None = None,
    layout: dict[int, dict[str, Any]] | None = None,
    evidence: list[Any] | None = None,
) -> str:
    """Lawyer-style paragraph. Page numbers are the document's own pages."""
    observed = _clean_model_text(str(getattr(response, "observed", "") or ""), defect)
    conclusion = _clean_model_text(
        str(getattr(response, "conclusion", "") or ""), defect
    )
    gap = getattr(response, "gap", None)
    gap_detail = _clean_model_text(
        str(getattr(response, "gap_detail", "") or ""), defect
    )
    status = str(getattr(response, "status", "") or "")
    sent = _sent_local_by_part(chunks, layout)
    cited = _evidence_by_part(evidence)
    part, pages = _primary_part(defect, cited or sent)
    full = _fully_reviewed(layout, part, chunks)

    if gap == "part_not_in_excerpts":
        names = [name.strip() for name in gap_detail.split(",") if name.strip()] or [
            part
        ]
        sentences = []
        for name in names:
            _global, local = _layout_pages_for(layout, name)
            phrase = _page_phrase(local)
            if phrase:
                sentences.append(
                    f"{_with_definite_article(name)} ({phrase}) was not among the pages reviewed; "
                    "compliance with this requirement could not be verified"
                )
            else:
                sentences.append(
                    f"{_with_definite_article(name)} was not among the pages reviewed; "
                    "compliance with this requirement could not be verified"
                )
        return _finish(" ".join(sentences))

    if gap == "element_page_not_in_excerpts":
        element = gap_detail or _lacking_phrase(response) or "the required portion"
        place = _inspected(part, pages, full=full and not cited)
        return _finish(
            f"{place}. The pages reviewed do not include {_as_object_phrase(element)}. "
            f"As the relevant portion of the {part} is not available, "
            f"compliance with {_compliance_with(defect)} cannot be determined"
        )

    if gap == "low_confidence":
        tail = (
            "The evidence is not conclusive enough to record a defect; "
            "an Advocate-on-Record should verify"
        )
        return _finish(f"{observed}. {tail}" if observed else tail)

    if gap == "conflict":
        place = _inspected(part, pages, full=False)
        mark = f" ({gap_detail})" if gap_detail else ""
        body = observed or "The text and a visible mark do not agree"
        return _finish(
            f"{place}. {body}. A visible mark conflicts with that reading{mark}. "
            "A reviewer must reconcile the visible mark with the requirement"
        )

    if gap == "unreadable":
        place = _inspected(part, pages, full=full)
        body = observed or "The relevant portion cannot be read"
        return _finish(
            f"{place}. {body}. Compliance with {_compliance_with(defect)} "
            "cannot be determined"
        )

    if gap == "ambiguous":
        place = _inspected(part, pages, full=full)
        body = observed or "The wording does not decide this requirement"
        return _finish(
            f"{place}. {body}. Compliance with {_compliance_with(defect)} "
            "cannot be determined"
        )

    if status == "defect_found":
        place = _inspected(part, pages, full=False)
        element_lines = _element_list_lines(response)
        if element_lines:
            middle = "The following requirement element is not met:"
            if len(element_lines) > 1:
                middle = "The following requirement elements are not met:"
        elif observed:
            middle = observed
        else:
            middle = "The required element is not met"
        closing = conclusion or "the requirement has not been complied with"
        return _finish_reasoning_blocks(
            [place, middle, "\n".join(element_lines), f"Accordingly, {closing}"]
        )

    if status == "compliant":
        place = _inspected(part, pages, full=full)
        body = observed or "The required element is present"
        return _finish(f"{place}. {body}. Accordingly, the requirement is met")

    if status == "not_applicable":
        body = observed or "The condition for this requirement is not present"
        return _finish(
            f"{body}. Accordingly, this requirement does not apply to this filing"
        )

    if observed:
        return _finish(observed)
    return "This check could not be completed."


def location_for_finding(
    *,
    evidence: list[Any] | None = None,
    chunks: list[dict[str, Any]] | None = None,
    layout: dict[int, dict[str, Any]] | None = None,
    reviewed_pages: list[int] | None = None,
) -> str:
    """Location line grouped by document type, using each document's own pages."""
    cited = _evidence_by_part(evidence)
    if cited:
        return filing_location(pages_by_part=cited)
    wanted = {int(page) for page in (reviewed_pages or []) if page is not None}
    reviewed_chunks = (
        [chunk for chunk in (chunks or []) if _safe_int(chunk.get("page")) in wanted]
        if wanted
        else list(chunks or [])
    )
    grouped = _sent_local_by_part(reviewed_chunks, layout)
    if grouped:
        return filing_location(pages_by_part=grouped)
    return filing_location(reviewed_pages=list(wanted))


def _check_specific_guidance(defect: Defect) -> str | None:
    check_id = re.sub(r"[^A-Z0-9]", "", defect.check_id.upper())
    if check_id == "D81":
        return (
            "This is a visual-quality check. OCR text, OCR errors, or retrieval "
            "results cannot prove that a scan is dim, blurred, clipped, or illegible. "
            "Return defect_found only when supplied visual-quality evidence identifies "
            "the exact affected document/page and the filing also lacks the required "
            "certified typed or written companion copy. Otherwise return needs_review."
        )
    if check_id == "D86":
        return (
            "Identify the exact Index row and the exact missing or inconsistent "
            "pagination field. Do not merely say that the Index was inspected or that "
            "some Index pages were unavailable."
        )
    if check_id == "D93":
        return (
            "This check tests only whether a Memo of Parties is present. Party "
            "particulars under the Main Petition heading 'IN THE MATTER OF' count as "
            "the Memo of Parties. Do not report missing or incomplete addresses under "
            "this check; address completeness is a separate defect."
        )
    return None


def _format_parent(defect: Defect) -> str | None:
    if not defect.parent_check_id:
        return None
    lines = [
        (
            f"This is a narrower check under {defect.parent_check_id}. "
            "Decide only the standard in this task. Do not re-score the parent."
        )
    ]
    if defect.overlap_note:
        note = _sentence_case(_polish_registry_text(defect.overlap_note))
        if not note.lower().startswith("narrower"):
            note = "Keep this distinction: " + note
        lines.append(note + ".")
    return " ".join(lines)


def _document_types_for_prompt(defect: Defect) -> str:
    groups = list(defect.inspect_part_groups or [])
    lines: list[str] = []
    for index, name in enumerate(defect.inspect_parts):
        group = groups[index] if index < len(groups) else ""
        lines.append(f"- {name} ({group})" if group else f"- {name}")
    return "\n".join(lines)


def build_defect_prompt(
    defect: Defect,
    *,
    record: dict[str, Any] | None,
    chunks: list[dict[str, Any]],
    catalogue: Catalogue | None = None,
    visual_index: dict[str, Any] | None = None,
    layout: dict[int, dict[str, Any]] | None = None,
) -> str:
    """Rewrite one catalogue row into the user message for the model."""
    search = "\n".join(
        f"{i}. {_search_step(step)}"
        for i, step in enumerate(defect.where_to_look, start=1)
    )

    document_types = _document_types_for_prompt(defect)
    sections = [
        f"# Task {defect.check_id}",
        "Decide this one defect. Ignore every other defect, even if the excerpts mention it.",
        "",
        "## Standard",
        _format_standard(defect),
        "",
        "## Where to look",
        search or "Use the document types below.",
        "",
        "## Document types to inspect",
        document_types or "The parts named in where to look.",
        "",
        (
            "If the excerpts do not include the document type this task is "
            "inspecting, return needs_review — not defect_found. If those "
            "parts are present but the evidence does not positively show the "
            "exact failure, return needs_review. Return defect_found only for "
            "a grounded blank, incorrect value, inconsistency, or incomplete "
            "clause. If you suspect a defect but "
            "confidence is below 0.8, return needs_review."
        ),
    ]
    ai_note = getattr(defect, "ai_note", None)
    if ai_note:
        sections.extend(["", "## Note for AI", ai_note.strip()])
    check_guidance = _check_specific_guidance(defect)
    if check_guidance:
        sections.extend(["", "## Decision guardrail", check_guidance])
    sections.extend(
        [
            "",
            "## Structured filing record",
            _format_record(record),
            "",
            "## Document excerpts",
            _format_evidence(chunks, layout),
            "",
            "## Stored visual observations",
            _format_visual_observations(defect, visual_index),
            "",
            "## Output",
            (
                f'Return JSON with check_id "{defect.check_id}", status, '
                "confidence, summary, observed, elements, gap, gap_detail, "
                "conclusion, and evidence. "
                "summary is one plain sentence naming the exact failed or satisfied "
                "element in this filing and this defect only — do not copy the Standard "
                "paragraph. observed is one or two sentences on what this filing "
                "shows. elements has one item per element in the Note for AI or "
                "the Requirement. Start each element name with a plain lettered "
                "label such as 'a. Court/Tribunal' and never '(a)'. "
                "gap is null unless status is needs_review. "
                "conclusion is one sentence and may cite a rule only when the "
                "Requirement writes that citation. Do not mention excerpt ids, "
                "page numbers, location_source, SCI_CHECKLIST, handbook PDF pages, "
                "or check_id in observed or conclusion. "
                "evidence is a list of one or more non-overlapping items (maximum six), "
                "{chunk_id, quote}. chunk_id MUST be copied exactly from "
                "the excerpt label (“[c1 | Main Petition, page 1]” → c1) and is "
                "not shown to the user. "
                "Quote each decisive passage in the document parts this task inspects. "
                "Use multiple quotes when multiple elements or documents must be compared. "
                "Do not quote an Index listing "
                "line that merely names that document. If the quote is "
                "not from an excerpt, set chunk_id and page to null."
            ),
        ]
    )
    return "\n".join(sections)


def build_system_prompt(
    catalogue: Catalogue,
    filing_type: str | None = None,
) -> str:
    return _system_prompt(catalogue, filing_type)


def build_defect_batch_prompt(
    tasks: list[tuple[Defect, dict[str, Any] | None, list[dict[str, Any]]]],
    *,
    catalogue: Catalogue,
    filing_type: str | None,
    visual_index: dict[str, Any] | None = None,
    layout: dict[int, dict[str, Any]] | None = None,
) -> tuple[str, str]:
    """Build a compact request with shared evidence and isolated decisions."""

    check_ids = [defect.check_id for defect, _, _ in tasks]
    system = _system_prompt(catalogue, filing_type).replace(
        "Decide exactly one defect. Return one status:",
        (
            "Decide every supplied defect independently. Never use one task's "
            "document excerpts as evidence for another task. Return one status "
            "for each task:"
        ),
    )
    shared_evidence, allowed_ids = _prepare_batch_evidence(tasks, layout)
    sections = [
        "# Batched scrutiny request",
        (
            "Evaluate every task below independently. Return exactly one result for "
            f"each of these check_ids, in this order: {', '.join(check_ids)}."
        ),
        (
            'Return one JSON object shaped as {"findings": [<result for the first '
            "task>, ...]}. Do not omit, duplicate, merge, or reorder checks. The "
            "result fields are check_id, status, confidence, summary, observed, "
            "elements, gap, gap_detail, conclusion, and evidence."
        ),
    ]
    for index, (defect, record, _chunks) in enumerate(tasks, start=1):
        search = "\n".join(
            f"{i}. {_search_step(step)}"
            for i, step in enumerate(defect.where_to_look, start=1)
        )
        task_sections = [
            "",
            f"# Independent task {index} of {len(tasks)} — {defect.check_id}",
            "## Standard",
            _format_standard(defect),
            "## Where to look",
            search or "Use the document types below.",
            "## Document types to inspect",
            _document_types_for_prompt(defect) or "The parts named above.",
        ]
        if defect.ai_note:
            task_sections.extend(["## Note for AI", defect.ai_note.strip()])
        guidance = _check_specific_guidance(defect)
        if guidance:
            task_sections.extend(["## Decision guardrail", guidance])
        ids = allowed_ids.get(defect.check_id) or []
        task_sections.extend(
            [
                "## Structured filing record",
                _format_record(record),
                "## Allowed shared excerpts",
                (
                    ", ".join(ids)
                    if ids
                    else "None. Return needs_review if filing evidence is required."
                ),
                "## Stored visual observations",
                _format_visual_observations(defect, visual_index),
            ]
        )
        sections.extend(
            task_sections
        )
    sections.extend(
        [
            "",
            "# Shared document excerpts",
            (
                "Each task may cite only the excerpt ids listed in its Allowed shared "
                "excerpts section. The same excerpt is printed once even when several "
                "tasks may use it."
            ),
            shared_evidence,
            "",
            "# Output fields",
            (
                "For each result, summary is one plain sentence; observed states only "
                "what the filing shows; elements contains the Requirement/Note elements; "
                "gap is null unless status is needs_review; conclusion is one sentence; "
                "and evidence contains at most six non-overlapping {chunk_id, quote} "
                "items. Copy chunk_id exactly from an allowed shared excerpt and quote "
                "verbatim. Do not mention excerpt ids, retrieval, page numbers, check_id, "
                "or rulebook locations in observed or conclusion."
            ),
            "",
            "# Final batch output",
            (
                'Return only {"findings": [...]}. The findings array must contain '
                f"exactly {len(tasks)} items in this order: {', '.join(check_ids)}."
            ),
        ]
    )
    return system, "\n".join(sections)


def build_evidence_queries(defect: Defect) -> list[str]:
    """Pinecone queries for one defect, taken from where-to-look."""
    return pinecone_queries_for_defect(defect)
