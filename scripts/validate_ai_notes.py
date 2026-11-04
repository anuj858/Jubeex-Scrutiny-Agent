#!/usr/bin/env python3
"""Validate the operational integrity of every scrutiny catalogue AI Note."""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path
from typing import Any


HEADINGS = ("SCOPE", "TEST", "EVIDENCE", "DECISION", "GUARDRAILS", "RECORD")
OUTCOMES = ("DEFECT:", "COMPLIANT:", "N/A:", "NEED REVIEW:")
FORBIDDEN = (
    "Defect when the supplied filing evidence positively establishes failure of a required element.",
    "Compliant only when complete relevant evidence establishes every applicable element.",
    "when the shown excerpts establish.",
    "when the excerpts establish the element.",
    "any of the three documents",
    "Legal-source gate",
    "rule verification pending",
)


def _section(note: str, heading: str, next_heading: str | None) -> str:
    start = note.index(f"{heading}\n") + len(heading) + 1
    end = note.index(f"\n\n{next_heading}\n", start) if next_heading else len(note)
    return note[start:end].strip()


def _validate(row: dict[str, Any]) -> list[str]:
    code = str(row.get("check_id") or "<missing>")
    note = str(row.get("ai_note") or "")
    errors: list[str] = []
    if not 0 < len(note) <= 4_000:
        errors.append(f"{code}: AI Note length is {len(note)}, expected 1-4000")
        return errors

    positions: list[int] = []
    for heading in HEADINGS:
        marker = f"{heading}\n"
        if note.count(marker) != 1:
            errors.append(f"{code}: expected exactly one {heading} section")
            continue
        positions.append(note.index(marker))
    if len(positions) == len(HEADINGS) and positions != sorted(positions):
        errors.append(f"{code}: sections are not in canonical order")

    for outcome in OUTCOMES:
        if note.count(outcome) != 1:
            errors.append(f"{code}: expected exactly one {outcome} decision")
    for fragment in FORBIDDEN:
        if fragment.casefold() in note.casefold():
            errors.append(f"{code}: contains non-operational fragment {fragment!r}")
    if re.search(r",\s*\.($|\s)", note):
        errors.append(f"{code}: contains a malformed ',.' sentence")

    inspect = [str(value).strip() for value in row.get("inspect_parts") or []]
    dependencies = [
        str(value).strip() for value in row.get("dependency_parts") or []
    ]
    missing_dependencies = sorted(set(dependencies) - set(inspect))
    if missing_dependencies:
        errors.append(
            f"{code}: dependencies are not inspect targets: {missing_dependencies}"
        )
    for label in inspect:
        if label.casefold() not in note.casefold():
            errors.append(f"{code}: inspect target {label!r} is absent from AI Note")
    if row.get("manifest_absence_proves_defect"):
        if dependencies:
            errors.append(f"{code}: presence-owner check must not have dependencies")
        defect_text = note.split("COMPLIANT:", 1)[0].casefold()
        if "manifest" not in defect_text or not re.search(
            r"\b(absent|absence|no|required document)\b", defect_text
        ):
            errors.append(
                f"{code}: presence-owner Defect rule does not cover manifest absence"
            )

    if all(f"{heading}\n" in note for heading in HEADINGS):
        for index, heading in enumerate(HEADINGS):
            following = HEADINGS[index + 1] if index + 1 < len(HEADINGS) else None
            if len(_section(note, heading, following)) < 20:
                errors.append(f"{code}: {heading} section is not operationally complete")

    requirement = str(row.get("requirement") or "").casefold()
    decision = note.split("DECISION\n", 1)[-1].split("\n\nGUARDRAILS", 1)[0]
    if (
        "main petition" in requirement
        and "impugned" in requirement
        and re.search(r"\b(correspond|consistent|same|match)\b", requirement)
        and not re.search(
            r"\b(differ(?:ent|s|ed)?|mismatch|conflict|correspond|match)\b",
            decision,
            re.I,
        )
    ):
        errors.append(f"{code}: comparison requirement lacks a mismatch decision rule")
    return errors


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("path", type=Path)
    args = parser.parse_args()
    payload = json.loads(args.path.read_text(encoding="utf-8"))
    rows = payload.get("defects") or []
    errors = [error for row in rows for error in _validate(row)]
    report = {
        "defects": len(rows),
        "validated": len(rows) - len({error.split(":", 1)[0] for error in errors}),
        "errors": len(errors),
        "maximum_ai_note_length": max(len(row.get("ai_note") or "") for row in rows),
    }
    print(json.dumps(report, indent=2))
    if errors:
        raise SystemExit("\n".join(errors))


if __name__ == "__main__":
    main()
