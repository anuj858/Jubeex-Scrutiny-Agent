#!/usr/bin/env python3
"""Rewrite scrutiny AI notes into one concise, operational contract.

The catalogue remains the legal source of truth. This transformer only uses
existing requirement, scope, evidence-location, dependency, presence-owner and
AI-note text; it does not introduce new legal requirements.
"""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path
from typing import Any


SECTION_RE = re.compile(
    r"(?im)^\s*(APPLICABILITY|SCOPE|DOCUMENTS? AND PORTIONS? TO INSPECT|"
    r"INSPECTION|WHAT THE REQUIREMENT MEANS|VERIFICATION|TEST|"
    r"WHAT DOES NOT SATISFY THE REQUIREMENT|GUARDRAILS?|"
    r"DEFECT FOUND|DEFECT|COMPLIANT|NOT APPLICABLE|N/?A|"
    r"NEEDS? REVIEW|EVIDENCE TO RECORD|RECORD|EVIDENCE)\s*[:.—-]\s*"
)

BOILERPLATE_PATTERNS = tuple(
    re.compile(pattern, re.IGNORECASE)
    for pattern in (
        r"identify each document by its actual heading",
        r"trigger words (?:alone )?(?:are|do) not",
        r"whenever needs? review is returned",
        r"do not return needs? review because unrelated",
        r"the complete pdf is unavailable",
        r"a keyword search failed",
        r"an unsuccessful (?:keyword|semantic )?search",
        r"inspect the supplied (?:text|material|documents|pages)",
        r"do not invent (?:dates|facts|marks|document availability)",
        r"do not infer facts",
        r"a complete relevant page(?:/section| or section) suffices",
        r"one complete local failure is sufficient",
        r"headings? and trigger words are locators",
        r"use trigger words only as locators",
        r"a split label is a locator",
        r"the retrieval list does not itself require",
        r"pending legal-source metadata",
        r"legal-source gate",
        r"pending legal verification",
        r"marked unverified",
        r"authoritative source",
        r"if the available evidence already establishes",
    )
)


def _clean_space(value: Any) -> str:
    return re.sub(r"\s+", " ", str(value or "")).strip()


def _sentences(text: str) -> list[str]:
    cleaned = _clean_space(text)
    if not cleaned:
        return []
    return [
        item.strip()
        for item in re.split(r"(?<=[.!?])\s+(?=[A-Z(\[])" , cleaned)
        if item.strip()
    ]


def _without_legal_gate(text: str) -> str:
    value = _clean_space(text)
    replacements = (
        (
            r"(?i)once the legal-source gate has been satisfied,?\s*",
            "",
        ),
        (
            r"(?i)when applicability is established,\s*the legal-source gate "
            r"is satisfied,\s*and\s*",
            "when applicability is established and ",
        ),
        (
            r"(?i),?\s*and the legal-source gate is satisfied",
            "",
        ),
        (
            r"(?i)apply defect found only after the legal-source gate has been satisfied\.?",
            "",
        ),
        (
            r"(?i)until (?:the )?gate is satisfied,?\s*return needs? review instead\.?",
            "",
        ),
        (
            r"(?i)this obligation is marked unverified\.?",
            "",
        ),
        (
            r"(?i)until a current authoritative source .*?(?=(?:Defect Found|"
            r"Compliant|Not Applicable|Needs? Review|$))",
            "",
        ),
        (
            r"(?i)after legal verification is completed,?\s*",
            "",
        ),
    )
    for pattern, replacement in replacements:
        value = re.sub(pattern, replacement, value)
    return _clean_space(value)


def _sections(note: str) -> dict[str, str]:
    matches = list(SECTION_RE.finditer(note or ""))
    found: dict[str, list[str]] = {}
    for index, match in enumerate(matches):
        name = match.group(1).upper()
        body = note[match.end() : matches[index + 1].start() if index + 1 < len(matches) else None]
        if name in {"APPLICABILITY", "SCOPE"}:
            key = "scope"
        elif name in {"DOCUMENTS AND PORTIONS TO INSPECT", "DOCUMENT AND PORTIONS TO INSPECT", "INSPECTION"}:
            key = "evidence"
        elif name in {"WHAT THE REQUIREMENT MEANS", "VERIFICATION", "TEST"}:
            key = "test"
        elif name in {"WHAT DOES NOT SATISFY THE REQUIREMENT", "GUARDRAIL", "GUARDRAILS"}:
            key = "guardrails"
        elif name in {"DEFECT", "DEFECT FOUND"}:
            key = "defect"
        elif name == "COMPLIANT":
            key = "compliant"
        elif name in {"NOT APPLICABLE", "N/A", "NA"}:
            key = "not_applicable"
        elif name in {"NEED REVIEW", "NEEDS REVIEW"}:
            key = "review"
        else:
            key = "record"
        found.setdefault(key, []).append(body.strip())
    return {key: " ".join(values) for key, values in found.items()}


def _compact(text: str, *, limit: int, keep: int = 4) -> str:
    text = _without_legal_gate(text)
    selected: list[str] = []
    eligible: list[str] = []
    for sentence in _sentences(text):
        if any(pattern.search(sentence) for pattern in BOILERPLATE_PATTERNS):
            continue
        eligible.append(sentence)
        candidate = " ".join([*selected, sentence]).strip()
        if len(candidate) > limit:
            break
        selected.append(sentence)
        if len(selected) >= keep:
            break
    if selected:
        return " ".join(selected)
    if not eligible:
        return ""
    first = eligible[0]
    extended_limit = min(max(limit * 2, limit), 900)
    if len(first) <= extended_limit:
        return first
    clipped = first[:extended_limit].rsplit(" ", 1)[0].rstrip(" ,;:")
    return clipped + "."


def _outcome_from_unstructured(note: str, status: str) -> str:
    boundary = (
        r"(?=return\s+(?:defect|compliant|needs?_?review|not_?applicable)|"
        r"(?:Defect|Compliant|Needs? Review|Not Applicable|N/?A|Evidence)\s*[—:-]|$)"
    )
    patterns = {
        "defect": rf"(?is)return\s+defect(?:_found| found)\s+(.*?){boundary}",
        "compliant": rf"(?is)return\s+compliant\s+(.*?){boundary}",
        "review": rf"(?is)return\s+needs?_?review\s+(.*?){boundary}",
        "not_applicable": rf"(?is)return\s+not_?applicable\s+(.*?){boundary}",
    }
    match = re.search(patterns[status], note or "")
    if not match:
        return ""
    return f"Return {status.replace('_', ' ')} {match.group(1).strip()}"


def _scope(row: dict[str, Any], parsed: dict[str, str]) -> str:
    categories = _clean_space(row.get("main_category")) or "the configured filing types"
    special = _clean_space(row.get("special_category"))
    base = f"Apply to: {categories}."
    if special:
        base += f" Special category: {special}."
    existing = _compact(parsed.get("scope", ""), limit=260, keep=2)
    conditional = re.search(
        r"\b(if|where|when|in case of|only when|provided that)\b",
        _clean_space(row.get("requirement")),
        re.IGNORECASE,
    )
    if conditional:
        base += " Establish the stated factual trigger from filing evidence before applying the check."
    if existing and special and special.casefold() not in existing.casefold():
        return base
    return base


def _test(row: dict[str, Any], parsed: dict[str, str]) -> str:
    existing = parsed.get("test", "")
    candidates = [
        sentence
        for sentence in _sentences(existing)
        if re.search(r"\b(test|element|compliance requires|requires? (?:both|all|either)|verify)\b", sentence, re.IGNORECASE)
    ]
    if candidates:
        detail = _compact(" ".join(candidates), limit=560, keep=3)
        return "1. Apply every element stated in the Requirement.\n2. " + detail
    return "1. Apply every element stated in the Requirement exactly as written; do not add an unstated requirement."


def _evidence(row: dict[str, Any]) -> str:
    inspect = [_clean_space(value) for value in row.get("inspect_parts") or [] if _clean_space(value)]
    dependency = [_clean_space(value) for value in row.get("dependency_parts") or [] if _clean_space(value)]
    lines = ["Primary documents: " + ", ".join(inspect) + "."] if inspect else []
    if row.get("manifest_absence_proves_defect"):
        lines.append("This is the presence-owner check: a complete manifest proving the required document absent establishes the defect.")
    elif dependency:
        lines.append("Primary dependency: " + ", ".join(dependency) + ".")
    locations = [_clean_space(value) for value in row.get("where_to_look") or [] if _clean_space(value)]
    if locations:
        lines.append("Focus: " + _compact(" ".join(locations[:2]), limit=340, keep=2))
    return "\n".join(lines)


def _decision_text(row: dict[str, Any], parsed: dict[str, str], status: str) -> str:
    existing = parsed.get(status, "") or _outcome_from_unstructured(row.get("ai_note", ""), status)
    limits = {"defect": 560, "compliant": 320, "not_applicable": 300, "review": 340}
    text = _compact(existing, limit=limits[status], keep=5)
    other_outcomes = {
        "defect": re.compile(r"\breturn\s+(?:compliant|needs?_?review|not_?applicable)\b", re.IGNORECASE),
        "compliant": re.compile(r"\b(?:needs? review|n/?a|not applicable)\s*[—:-]|\breturn\s+(?:defect|needs?_?review|not_?applicable)\b", re.IGNORECASE),
        "not_applicable": re.compile(r"\breturn\s+(?:defect|compliant|needs?_?review)\b", re.IGNORECASE),
        "review": re.compile(r"\breturn\s+(?:defect|compliant|not_?applicable)\b", re.IGNORECASE),
    }
    clean_sentences: list[str] = []
    for sentence in _sentences(text):
        if other_outcomes[status].search(sentence):
            match = other_outcomes[status].search(sentence)
            if match and match.start() > 0:
                clean_sentences.append(sentence[: match.start()].rstrip(" ,;:.-") + ".")
            break
        clean_sentences.append(sentence)
    text = " ".join(clean_sentences).strip()
    if status == "defect" and row.get("manifest_absence_proves_defect"):
        owner = "A complete manifest proving the required document absent is Defect."
        if text and "manifest" not in text.casefold():
            text = owner + " " + text
        elif not text:
            text = owner
    if status == "not_applicable":
        if row.get("manifest_absence_proves_defect"):
            text = "N/A only when the filing type or activating factual condition is affirmatively outside this check; document absence belongs to this Defect check."
    defaults = {
        "defect": "Defect when the supplied filing evidence positively establishes failure of a required element.",
        "compliant": "Compliant only when complete relevant evidence establishes every applicable element.",
        "not_applicable": "N/A only when affirmative filing evidence establishes that the rule or its activating condition does not apply.",
        "review": "Need Review only when a specific decisive fact remains genuinely missing, illegible or materially contradictory after inspecting the available text and images.",
    }
    return text or defaults[status]


def _guardrails(row: dict[str, Any], parsed: dict[str, str]) -> str:
    text = _compact(parsed.get("guardrails", ""), limit=280, keep=2)
    if text:
        return text
    note_sentences = [
        sentence
        for sentence in _sentences(row.get("ai_note", ""))
        if sentence.casefold().startswith("do not ")
    ]
    text = _compact(" ".join(note_sentences), limit=280, keep=2)
    return text or "Do not treat an Index entry, failed search, OCR silence or an unrelated missing page as proof of this defect."


def _record(row: dict[str, Any], parsed: dict[str, str]) -> str:
    parts = ", ".join(row.get("inspect_parts") or []) or "the relevant document"
    return f"Record the physical filing page, decisive quotation or visual observation from {parts}, and every compared value used in the decision."


def rewrite(row: dict[str, Any]) -> str:
    parsed = _sections(row.get("ai_note", ""))
    sections = (
        ("SCOPE", _scope(row, parsed)),
        ("TEST", _test(row, parsed)),
        ("EVIDENCE", _evidence(row)),
        ("DECISION", "\n".join(
            [
                "DEFECT: " + _decision_text(row, parsed, "defect"),
                "COMPLIANT: " + _decision_text(row, parsed, "compliant"),
                "N/A: " + _decision_text(row, parsed, "not_applicable"),
                "NEED REVIEW: " + _decision_text(row, parsed, "review"),
            ]
        )),
        ("GUARDRAILS", _guardrails(row, parsed)),
        ("RECORD", _record(row, parsed)),
    )
    note = "\n\n".join(f"{heading}\n{body.strip()}" for heading, body in sections if body.strip())
    if len(note) > 4_000:
        raise ValueError(f"{row.get('check_id')} generated {len(note)} characters")
    if re.search(r"legal[- ]source gate|legal_verification_status|rule verification pending", note, re.IGNORECASE):
        raise ValueError(f"{row.get('check_id')} retained a legal-source gate")
    return note


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("path", type=Path)
    parser.add_argument("--write", action="store_true")
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()

    payload = json.loads(args.path.read_text(encoding="utf-8"))
    rows = payload.get("defects") or []
    for row in rows:
        row["ai_note"] = rewrite(row)

    rendered = json.dumps(payload, ensure_ascii=False, indent=2) + "\n"
    target = args.path if args.write else args.output
    if target:
        target.write_text(rendered, encoding="utf-8")
    lengths = [len(row["ai_note"]) for row in rows]
    print(
        json.dumps(
            {
                "notes": len(rows),
                "minimum": min(lengths),
                "maximum": max(lengths),
                "over_4000": sum(length > 4_000 for length in lengths),
                "output": str(target) if target else None,
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
