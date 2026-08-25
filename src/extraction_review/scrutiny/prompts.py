"""Prompt construction for defect scrutiny.

The catalogue already carries model-facing `instruction` text and a
`decision_policy` per defect, so the prompts here mostly assemble that authored
content around the retrieved evidence rather than inventing new guidance.
"""

from __future__ import annotations

import json
from typing import Any

from .rules import Catalogue, Defect, Subcheck

MAX_EVIDENCE_CHARS = 60_000


def _system_prompt(catalogue: Catalogue) -> str:
    policy = catalogue.global_decision_policy
    return f"""You are a filing scrutiny assistant for the {catalogue.jurisdiction}. \
You check a Special Leave Petition (Civil) against the registry defect catalogue \
"{catalogue.catalogue_id}" v{catalogue.catalogue_version} and report what a \
Scrutiny Assistant at the filing counter would flag.

You assess exactly one defect per request, subcheck by subcheck. For every \
subcheck you must return one of these five statuses:

- defect_found: {policy.defect_found}
- compliant: {policy.compliant}
- not_applicable: {policy.not_applicable}
- not_determined: {policy.not_determined}
- needs_review: {policy.needs_review}

Rules you must not break:

1. Ground every statement in the supplied evidence. Quote verbatim from it. If \
you cannot quote it, you have not found it.
2. Absence of evidence is not evidence of absence. If the required section is \
simply not present in the supplied excerpts, return not_determined, not \
defect_found. Only return defect_found when the evidence positively shows the \
requirement is unmet.
3. Never infer from formatting, signatures, stamps, seals, page rendering or \
typography. You are reading extracted text, not the rendered page. Anything \
that depends on how the page looks is not_determined.
4. When the law is ambiguous, the evidence conflicts, or an Advocate-on-Record \
must make the call, return needs_review rather than deciding.
5. Do not apply requirements that the subcheck's own conditions exclude. If a \
condition cannot be established from the evidence, return not_determined.
6. Return a result for every subcheck listed in the request, using the exact \
subcheck_id given. Do not invent subchecks.
7. confidence is your certainty in the status you assigned, from 0.0 to 1.0. \
Be honest: partial evidence means lower confidence.
8. Set suggested_fix only when status is defect_found. It must say concretely \
what should appear in the petition instead of what is there now, in the \
register of a court filing. Otherwise set suggested_fix and fix_rationale to \
null.

You are assisting a pre-filing review. You do not decide legal validity and you \
do not speak for the Registry.

Respond with JSON matching the required schema. No prose outside the JSON."""


def _format_subcheck(subcheck: Subcheck) -> str:
    lines = [
        f"### {subcheck.subcheck_id} — {subcheck.title}",
        f"Criterion: {subcheck.criterion}",
        f"If this criterion is not met, the status is: {subcheck.failure_result}",
        f"Evidence this subcheck relies on: {', '.join(subcheck.required_evidence)}",
    ]
    if subcheck.applicability.conditions:
        conditions = "; ".join(
            f"{c.fact} {c.operator} {json.dumps(c.value)}"
            for c in subcheck.applicability.conditions
        )
        lines.append(
            f"Only applies when: {conditions}. If you cannot establish this from "
            f"the evidence, return {subcheck.applicability.unknown_condition_result}."
        )
    if subcheck.manual_review_when:
        lines.append(
            "Return needs_review if: " + "; ".join(subcheck.manual_review_when) + "."
        )
    return "\n".join(lines)


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
            header = f"[Page {page}]"
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


def build_defect_prompt(
    defect: Defect,
    *,
    record: dict[str, Any] | None,
    chunks: list[dict[str, Any]],
    file_name: str | None = None,
    skipped: list[Subcheck] | None = None,
) -> str:
    """Assemble the user message for one defect."""
    evaluated = [s for s in defect.subchecks if s not in (skipped or [])]
    subcheck_ids = ", ".join(s.subcheck_id for s in evaluated)
    policy = defect.decision_policy

    sections = [
        f"# Check {defect.check_id}: {defect.title}",
        f"Severity: {defect.severity}",
        f"Objective: {defect.objective}",
        "",
        "## How to run this check",
        defect.instruction,
        "",
        "## Decision policy for this check",
        f"- compliant: {policy.compliant}",
        f"- defect_found: {policy.defect_found}",
        f"- not_applicable: {policy.not_applicable}",
        f"- not_determined: {policy.not_determined}",
        f"- needs_review: {policy.needs_review}",
        "",
        "## Subchecks to evaluate",
        "\n\n".join(_format_subcheck(s) for s in evaluated),
        "",
        "## Structured filing record (already extracted from this document)",
        _format_record(record),
        "",
        f"## Document excerpts{f' from {file_name}' if file_name else ''}",
        _format_evidence(chunks),
        "",
        "## Your task",
        (
            f'Return a JSON object with check_id set to "{defect.check_id}", a short '
            "summary of this check across the document, and one entry in "
            f"subcheck_results for each of: {subcheck_ids}."
        ),
    ]

    if skipped:
        sections.append(
            "Note: "
            + ", ".join(s.subcheck_id for s in skipped)
            + " have been excluded because they need visual evidence that is not "
            "available. Do not return results for them."
        )

    return "\n".join(sections)


def build_system_prompt(catalogue: Catalogue) -> str:
    return _system_prompt(catalogue)


def build_evidence_queries(defect: Defect, subchecks: list[Subcheck]) -> list[str]:
    """One retrieval query per subcheck, plus a defect-level query.

    Per-subcheck queries matter for recall: a single query for the whole defect
    tends to surface only the most prominent section and miss the others.
    """
    queries = [f"{defect.title}. {defect.objective}"]
    for subcheck in subchecks:
        evidence = " ".join(e.replace("_", " ") for e in subcheck.required_evidence)
        queries.append(f"{subcheck.title}. {subcheck.criterion} {evidence}")
    return queries
