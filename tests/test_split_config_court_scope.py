"""Prompt ownership contract; real classification is covered by split replay tests."""

from __future__ import annotations

import json
from pathlib import Path

import pytest


@pytest.fixture
def split_rules() -> tuple[dict[str, str], str]:
    path = Path(__file__).resolve().parents[1] / "configs" / "config.json"
    split = json.loads(path.read_text(encoding="utf-8"))["split"]
    return (
        {item["name"]: item["description"] for item in split["categories"]},
        split["splitting_strategy"]["custom_instructions"],
    )


@pytest.mark.parametrize(
    "category",
    [
        "Main Petition",
        "Application",
        "Affidavit",
        "Record of Proceedings",
        "Memo of Appearance",
        "Vakalatnama",
    ],
)
def test_outer_procedural_rules_exclude_historical_records(
    split_rules: tuple[dict[str, str], str], category: str
) -> None:
    descriptions, instructions = split_rules
    description = descriptions[category]
    negatives = description.split("DO NOT SPLIT: ", 1)[1].split("\nREPEAT:", 1)[0]
    assert "lower-court" in negatives.casefold()
    assert "historical supreme court" in negatives.casefold()
    assert "Annexure" in negatives
    assert "stay enclosed" in negatives or "stay with that Annexure" in negatives
    assert "matching parties or titles do not promote" in instructions
    assert "purpose, case identity and enclosing context" in instructions


def test_captionless_current_documents_and_blank_rop_register_remain_valid(
    split_rules: tuple[dict[str, str], str],
) -> None:
    descriptions, instructions = split_rules
    rop = descriptions["Record of Proceedings"]
    assert "blank register with serial/date/pages columns" in rop
    assert "position and surrounding current-filing documents" in rop
    assert "need not repeat a Supreme Court caption" in rop
    assert "connected-case captions" in rop
    assert "'having come up for hearing'" in rop
    assert "'delivered the following'" in rop
    assert "Captionless continuations inherit their established parent" in instructions


def test_supreme_court_application_affidavit_retains_application_ownership(
    split_rules: tuple[dict[str, str], str],
) -> None:
    descriptions, instructions = split_rules
    assert "current primary pleading" in descriptions["Affidavit"]
    assert (
        "stays with that Application, even if separately headed or Supreme Court-captioned"
        in descriptions["Affidavit"]
    )
    assert (
        "even when the affidavit has a separate Supreme Court caption"
        in (descriptions["Application"])
    )
    assert "application-supporting affidavit stays with its Application" in instructions


def test_scope_is_not_a_blanket_supreme_court_caption_filter(
    split_rules: tuple[dict[str, str], str],
) -> None:
    descriptions, instructions = split_rules
    assert (
        "Do not impose a Supreme Court caption on these or on Annexures" in instructions
    )
    assert "do not require a Supreme Court caption" in descriptions["Impugned Order"]
    assert (
        "a lower-court caption alone does not exclude it"
        in descriptions["Memo of Parties"]
    )
    assert (
        "a Supreme Court caption is not required" in descriptions["Custody Certificate"]
    )
    assert "a Supreme Court caption is not required" in descriptions["PoA/BR"]
    assert (
        "including cover, translations, certification and reproduced records"
        in (descriptions["Annexures"])
    )
