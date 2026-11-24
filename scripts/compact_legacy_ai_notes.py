#!/usr/bin/env python3
"""Compact the remaining overlength catalogue AI Notes without losing rules.

The general canonical rewriter is intentionally not used here: these checks
contain legal calculation and signature-placement rules that must survive a
rewrite.  This script is deliberately limited to the known legacy rows and
fails if any non-AI-Note field changes.
"""

from __future__ import annotations

import argparse
import copy
import json
import re
from pathlib import Path
from typing import Any

from rewrite_ai_notes_canonical import NEED_REVIEW_POLICY

LIMITATION_CODES = {
    "D-334", "D-335", "D-336", "D-337", "D-338", "D-339", "D-340",
    "D-341", "D-345", "D-346", "D-347", "D-348", "D-349", "D-350",
    "D-351", "D-352", "D-353", "D-354", "D-355", "D-356", "D-357",
    "D-358",
}


def _section(note: str, heading: str) -> str:
    match = re.search(
        rf"(?ms)^{re.escape(heading)}\n(.*?)(?=\n\n[A-Z][A-Z ]+\n|\Z)", note
    )
    return match.group(1).strip() if match else ""


def _documents(row: dict[str, Any]) -> str:
    return ", ".join(str(value) for value in row.get("inspect_parts") or [])


def _limitation_note(row: dict[str, Any]) -> str:
    scope = _section(str(row["ai_note"]), "SCOPE")
    requirement = str(row["requirement"]).strip()
    act_guard = (
        " Confirm the governing Act or provision from the Listing Proforma, "
        "Synopsis/List of Dates and Main Petition; do not infer it from the "
        "case-type label alone."
        if "identify the applicable Act" in scope
        else ""
    )
    special_guard = ""
    if row["check_id"] == "D-357":
        special_guard = (
            " For Section 62 IBC, distinguish 45-day expiry from the 45+15-day "
            "outer threshold exactly as the Requirement states."
        )
    return f"""SCOPE
{scope}

TEST
1. Apply this rule exactly: {requirement}
2. Read the prescribed starting date from the controlling source named in the Requirement. Fetch the current Scrutiny Date afresh for this run. Count from the next day through the Scrutiny Date and calculate expiry and delay under the stated period.
3. If limitation expired, use the complete manifest to determine whether an Application for Condonation of Delay is filed. Read the Office Report and COD delay figures only for the independent comparison checks D-361 to D-363.

EVIDENCE
Primary documents: {_documents(row)}.

DECISION
DEFECT: Return Defect when the independent calculation exceeds the applicable period and the complete manifest proves that no COD Application is filed. State the source date, Scrutiny Date, elapsed period and delay.
COMPLIANT: Return Compliant when the independent calculation is within the applicable period, or the required COD Application is present.
N/A: Return N/A when the filing or governing provision is affirmatively outside this row's scope.
NEED REVIEW: {NEED_REVIEW_POLICY}

GUARDRAILS
Use the Scrutiny Date as the end point, never the filing date, drafting date or a date cached from an earlier run. Recalculate on every run and update the existing result rather than duplicating it. Use only the starting-date source prescribed by the Requirement; do not substitute an Index/Synopsis date, an order from another matter or an unsupported user answer. Do not deduct certified-copy time unless the Requirement expressly directs it. A mismatch with the Registry or COD figure belongs only to D-361, D-362 or D-363.{act_guard}{special_guard}

RECORD
Record the controlling source and physical page, quoted starting date, Scrutiny Date and fetch time, period applied, elapsed time, calculated delay, COD presence, and whether the result changed from the previous run."""


def _d73_note() -> str:
    return f"""SCOPE
Apply to every filing requiring a Vakalatnama. This is the presence-owner check: a complete manifest proving the Vakalatnama absent establishes Defect.

TEST
Inspect the complete rendered Vakalatnama. (1) Match every petitioner in the cause title/party array to the executant named after “I/We” and before “do hereby appoint and retain”; the later name is the appointed advocate. (2) Confirm a signature stroke for every petitioner in the petitioner/appellant/executant signature block. (3) Confirm an acceptance endorsement and a separate AOR signature attributable through the adjacent name/code block.

EVIDENCE
Primary document: Vakalatnama. Use the rendered page, opening formula, executant and petitioner signature blocks, acceptance endorsement, AOR signature and name/code block. OCR is only a locator.

DECISION
DEFECT: Return Defect if the complete manifest proves the Vakalatnama absent, any petitioner is not identified as an executant, any identified petitioner lacks a visible signature stroke, the AOR acceptance endorsement is absent, or the endorsement lacks an attributable AOR signature. Identify the petitioner or missing element and page.
COMPLIANT: Return Compliant only when every petitioner is identified and has a visible signature stroke, and an acceptance endorsement with an attributable AOR signature is visible.
N/A: Return N/A only when affirmative filing evidence establishes that this requirement does not apply; absence of the Vakalatnama is Defect, not N/A.
NEED REVIEW: {NEED_REVIEW_POLICY}

GUARDRAILS
Do not use a Memo of Appearance or a Vakalatnama from a lower court. Do not confuse petitioner and AOR signatures. A handwritten executant name is a completed field but does not replace the petitioner's signature. A cursive or unspellable visible stroke is still a signature. A blank intervenor/respondent line is irrelevant. Acceptance wording need not reproduce “Satisfied, Certified, Identified and Accepted” exactly; equivalent acceptance wording is sufficient. A signature below or beside the endorsement with the advocate's name/code is not missing. Never call a visible mark absent because OCR is silent.

RECORD
Record the physical page, each petitioner/executant match, location of each petitioner signature, exact acceptance wording, and location and attribution of the AOR signature."""


def _d227_note() -> str:
    return f"""SCOPE
Apply to SLP (Civil) and SLP (Criminal) filed through an AOR. It does not apply to a petitioner-in-person filing.

TEST
Read the complete AOR's Certificate and test four elements: (1) the SLP is confined to pleadings and relied-on documents before the challenged Court/Tribunal; (2) no additional facts, documents or grounds are relied on, except material expressly covered by a separate permission application; (3) the annexures are necessary to answer the questions of law; and (4) the certificate is based on the petitioner's instructions. If an exception is stated, compare the Main Petition, additional material and corresponding permission application.

EVIDENCE
Primary documents: AOR's Certificate, Main Petition and any application seeking permission to introduce additional facts, grounds or documents. Primary dependency: AOR's Certificate.

DECISION
DEFECT: Return Defect when the complete certificate omits any of the four elements; or when it excepts additional material but the complete manifest proves the corresponding permission application absent, the application does not cover that material, or the filing otherwise contradicts the certification. Quote the missing or inconsistent clause.
COMPLIANT: Return Compliant when all four elements are certified and any exception is supported by a present permission application covering the additional material.
N/A: Return N/A for a petitioner-in-person filing.
NEED REVIEW: {NEED_REVIEW_POLICY}

GUARDRAILS
Do not substitute an Affidavit or Main Petition averment for the certificate. Do not treat an exception as an automatic defect when a matching permission application is present, and do not treat the application's mere existence as proof that it covers the material. Independently verify the annexure-necessity and petitioner-instructions clauses. A missing certificate is handled by its presence-owner check; this row evaluates certificate contents.

RECORD
Record the certificate page and decisive quotations for all four elements; where an exception exists, record the additional material, permission-application pages and the comparison result."""


def _d361_note(row: dict[str, Any]) -> str:
    return f"""SCOPE
Apply when an applicable limitation rule under D-334 to D-360 and a COD Application are present; Original Suit D-333 is outside scope.

TEST
Independently calculate delay from the starting date and statutory period prescribed by the applicable limitation row to the current Scrutiny Date. Read the delay in days stated in the COD Application and compare the two numbers.

EVIDENCE
Primary documents: {_documents(row)}. Primary dependency: Application for Condonation of Delay.

DECISION
DEFECT: Return Defect only when both figures are legible and the calculated delay differs from the COD figure; state both.
COMPLIANT: Return Compliant when both figures match.
N/A: Return N/A when no COD Application is present or no limitation calculation applies.
NEED REVIEW: {NEED_REVIEW_POLICY}

GUARDRAILS
Fetch the Scrutiny Date and recalculate on every run. Use the applicable row's exact starting-date source and period. Compare days only; wording differences are irrelevant. Do not use the Office Report figure as the AI calculation. Do not assume a mismatch from a missing or illegible number.

RECORD
Record the applicable limitation code, source page/date, Scrutiny Date and fetch time, statutory period, calculated delay, COD page and quoted delay, and whether the result changed."""


def _d362_note(row: dict[str, Any]) -> str:
    return f"""SCOPE
Apply when an applicable limitation rule under D-334 to D-360 and an Office Report on Limitation are present; Original Suit D-333 is outside scope.

TEST
Independently calculate delay to the current Scrutiny Date using the applicable row's statutory period and starting date: the Impugned Order date, or for arbitration one month after the valid notice date, or for contempt the alleged-contempt date. Compare that delay with the Registry delay in the Office Report.

EVIDENCE
Primary documents: {_documents(row)}. Primary dependency: Office Report on Limitation.

DECISION
DEFECT: Return Defect only when both figures are legible and the calculated delay differs from the Office Report delay; state both figures and the starting-date source.
COMPLIANT: Return Compliant when both figures match.
N/A: Return N/A when no Office Report is present or no limitation calculation applies.
NEED REVIEW: {NEED_REVIEW_POLICY}

GUARDRAILS
Fetch the Scrutiny Date and recalculate on every run. Use the applicable row's exact starting date and period; do not substitute the COD figure. Compare days only. For Section 62 IBC use the basis stated in the applicable row and identify a genuinely unclear 45/60-day basis. Do not assume a mismatch from a missing or illegible number.

RECORD
Record the applicable limitation code, starting-date source/page, Scrutiny Date and fetch time, period, calculated delay, Office Report page and quoted delay, and whether the result changed."""


def compact(row: dict[str, Any]) -> None:
    code = row.get("check_id")
    if code == "D-64":
        row["ai_note"] = str(row["ai_note"]).replace(
            "Return Defect Found where the supporting affidavit is absent",
            "Return Defect Found when the complete manifest proves the required "
            "supporting Affidavit absent",
            1,
        )
    elif code == "D-73":
        row["ai_note"] = _d73_note()
    elif code == "D-227":
        row["ai_note"] = _d227_note()
    elif code in LIMITATION_CODES:
        row["ai_note"] = _limitation_note(row)
    elif code == "D-361":
        row["ai_note"] = _d361_note(row)
        row["dependency_parts"] = ["Application for Condonation of Delay"]
    elif code == "D-362":
        row["inspect_parts"] = [
            "Listing Proforma",
            "Synopsis",
            "List of Dates and Events",
            "Main Petition",
            "Annexures",
            "Impugned Order",
            "Office Report on Limitation",
        ]
        row["ai_note"] = _d362_note(row)
        row["dependency_parts"] = ["Office Report on Limitation"]
    elif code == "D-363":
        row["inspect_parts"] = [
            "Office Report on Limitation",
            "Application for Condonation of Delay",
        ]
        row["dependency_parts"] = [
            "Office Report on Limitation",
            "Application for Condonation of Delay",
        ]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("path", type=Path)
    parser.add_argument("--write", action="store_true")
    args = parser.parse_args()
    payload = json.loads(args.path.read_text(encoding="utf-8"))
    original = copy.deepcopy(payload)
    for row in payload.get("defects") or []:
        compact(row)
    allowed_non_note = {"D-361", "D-362", "D-363"}
    for before, after in zip(original["defects"], payload["defects"], strict=True):
        before_other = {k: v for k, v in before.items() if k != "ai_note"}
        after_other = {k: v for k, v in after.items() if k != "ai_note"}
        if before_other != after_other and before["check_id"] not in allowed_non_note:
            raise ValueError(f"{before['check_id']}: unexpected non-note change")
        if len(str(after.get("ai_note") or "")) > 4_000:
            raise ValueError(f"{after['check_id']}: AI Note still exceeds 4,000")
    if not args.write:
        raise ValueError("Pass --write to update the catalogue")
    args.path.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )


if __name__ == "__main__":
    main()
