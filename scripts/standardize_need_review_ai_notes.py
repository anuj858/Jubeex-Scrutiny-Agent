#!/usr/bin/env python3
"""Standardize only the NEED REVIEW section of catalogue AI Notes."""

from __future__ import annotations

import argparse
import copy
import json
import re
from pathlib import Path
from typing import Any

from rewrite_ai_notes_canonical import D68_NEED_REVIEW_POLICY, NEED_REVIEW_POLICY

NEED_REVIEW_RE = re.compile(
    r"(?s)(\nNEED REVIEW:\s*).*?(\n\nGUARDRAILS\n)"
)


def standardize(row: dict[str, Any]) -> None:
    note = str(row.get("ai_note") or "")
    policy = (
        D68_NEED_REVIEW_POLICY
        if row.get("check_id") == "D-68"
        else NEED_REVIEW_POLICY
    )
    updated, count = NEED_REVIEW_RE.subn(
        lambda match: f"{match.group(1)}{policy}{match.group(2)}",
        note,
        count=1,
    )
    if count != 1:
        raise ValueError(f"{row.get('check_id')}: expected one NEED REVIEW section")
    if row.get("check_id") == "D-68":
        marker = "\n\nRECORD\n"
        guardrail = (
            " Do not call text garbled or illegible from OCR output alone when "
            "the rendered page is visually readable."
        )
        if guardrail.strip() not in updated:
            if marker not in updated:
                raise ValueError("D-68: RECORD section not found")
            updated = updated.replace(marker, guardrail + marker, 1)
    if len(updated) > 4_000:
        raise ValueError(f"{row.get('check_id')}: AI Note exceeds 4,000 characters")
    row["ai_note"] = updated


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("path", type=Path)
    parser.add_argument("--write", action="store_true")
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()

    payload = json.loads(args.path.read_text(encoding="utf-8"))
    original = copy.deepcopy(payload)
    rows = payload.get("defects") or []
    for row in rows:
        standardize(row)

    for before, after in zip(original.get("defects") or [], rows, strict=True):
        if {key: value for key, value in before.items() if key != "ai_note"} != {
            key: value for key, value in after.items() if key != "ai_note"
        }:
            raise ValueError(f"{before.get('check_id')}: non-AI-Note data changed")

    target = args.path if args.write else args.output
    if target is None:
        raise ValueError("Specify --write or --output")
    target.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    print(
        json.dumps(
            {
                "updated_notes": len(rows),
                "maximum_ai_note_length": max(len(row["ai_note"]) for row in rows),
                "output": str(target),
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
