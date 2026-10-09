"""Split prompt/request contract tests, not a substitute for live PDF accuracy QA."""

from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import AsyncMock

import pytest

from extraction_review.config import Config
from extraction_review.document_parts import _needles_for_part
from extraction_review.process_file import (
    _SPLIT_CUSTOM_INSTRUCTIONS_MAX,
    _split_api_configuration,
    _split_page_parts,
)


def _config_payload() -> dict:
    path = Path(__file__).resolve().parents[1] / "configs" / "config.json"
    return json.loads(path.read_text(encoding="utf-8"))


def test_split_guidance_is_sent_in_full_with_existing_category_contract() -> None:
    payload = _config_payload()
    split = Config.model_validate(payload).split
    assert split is not None
    source = payload["split"]
    instructions = source["splitting_strategy"]["custom_instructions"]
    assert 0 < len(instructions) <= _SPLIT_CUSTOM_INSTRUCTIONS_MAX
    sent = _split_api_configuration(split)
    assert sent["splitting_strategy"] == source["splitting_strategy"]
    assert sent["splitting_strategy"]["min_pages_per_split"] == 1
    assert sent["categories"] == source["categories"]
    assert source["configuration_id"] is None  # The inline rules must be used.
    assert set(sent) == {"categories", "splitting_strategy"}
    assert [item["name"] for item in sent["categories"]] == [
        "Advocate's Checklist",
        "Cover Page",
        "Record of Proceedings",
        "AOR's Declaration",
        "AOR's Certificate",
        "Index",
        "Office Report on Limitation",
        "Listing Proforma",
        "Synopsis",
        "List of Dates & Events",
        "Main Petition",
        "Affidavit",
        "Annexures",
        "Application",
        "Appendix",
        "Memo of Parties",
        "Memo of Appearance",
        "Impugned Order",
        "Vakalatnama",
        "PoA/BR",
        "Filing Memo",
        "Court Fees",
        "Custody Certificate",
    ]
    for category in sent["categories"]:
        assert set(category) == {"name", "description"}
        assert 0 < len(category["name"]) <= 200
        assert 0 < len(category["description"]) <= 2000
        for section in (
            "IDENTITY",
            "START",
            "CONTINUE",
            "END",
            "DO NOT SPLIT",
            "REPEAT",
        ):
            assert f"{section}: " in category["description"]


@pytest.mark.parametrize(
    ("category", "required_cues"),
    [
        ("Index", ("function", "unheaded continuation pages", "FILING INDEX")),
        ("Listing Proforma", ("PRO FORMA", "LISTED PROFORMA", "closing date")),
        (
            "Main Petition",
            ("primary substantive pleading", "Writ Petition", "Form 28"),
        ),
        (
            "Impugned Order",
            (
                "do not require a Supreme Court caption",
                "complete decision",
                "Annexures",
            ),
        ),
        (
            "Memo of Parties",
            ("BEFORE THE HIGH COURT", "separately presented party list"),
        ),
        (
            "Annexures",
            (
                "one segment",
                "without renumbering",
                "imposing a maximum",
                "own issuer/date",
                "case-specific APPENDIX OF WP(C) exhibit lists",
            ),
        ),
        (
            "Application",
            (
                "affidavit expressly supporting",
                "adjacent applications",
                "Kindly see Pages",
                "body or prayer",
            ),
        ),
        (
            "Appendix",
            ("APPENDIX OF WP(C)", "PETITIONER/RESPONDENT EXHIBITS", "stays enclosed"),
        ),
        (
            "Affidavit",
            ("primary pleading's affidavit", "stays with that Application"),
        ),
        (
            "Vakalatnama",
            ("same physical page", "separately starting Memo of Appearance"),
        ),
    ],
)
def test_split_descriptions_preserve_reported_boundary_guards(
    category: str, required_cues: tuple[str, ...]
) -> None:
    descriptions = {
        item["name"]: item["description"]
        for item in _config_payload()["split"]["categories"]
    }
    for cue in required_cues:
        assert cue in descriptions[category]


def test_shared_guidance_does_not_restore_conflicting_old_requirements() -> None:
    instructions = _config_payload()["split"]["splitting_strategy"][
        "custom_instructions"
    ]
    assert (
        "Follow the category descriptions for document identity and boundaries"
        in instructions
    )
    assert "combined SYNOPSIS AND LIST OF DATES heading" in instructions
    assert "without a fixed heading-position requirement" in instructions
    assert (
        "Do not merge separate Memo of Appearance and Vakalatnama segments"
        in instructions
    )
    assert (
        "A standalone Impugned Order can have the caption of the court/authority under challenge"
        in instructions
    )
    assert "do not invent category names" in instructions
    assert "reason/metadata fields" in instructions
    assert "do not infer a missing annexure from a numbering gap" in instructions
    assert "Do not extrapolate one offset across volumes" in instructions
    assert (
        "never give one document a whole gap shared by multiple unresolved Index rows"
        in instructions
    )
    assert (
        "case-specific exhibit-list appendices remain in that same judgment"
        in instructions
    )
    assert "Main Petition is Form 28 in this Court" not in instructions
    assert (
        "only the standalone memo matching this Supreme Court petition"
        not in instructions
    )
    assert "marks Main Petition or Impugned Order" not in instructions


def test_description_examples_do_not_create_aliases_for_other_categories() -> None:
    # Parenthesized description phrases also feed local category lookup.
    # Contrasting '(Vakalatnama)' with PoA must not alias one to the other.
    categories = _config_payload()["split"]["categories"]
    for category in categories:
        own_needles = set(_needles_for_part(category["name"]))
        added_needles = (
            set(_needles_for_part(category["name"], category["description"]))
            - own_needles
        )
        for other in categories:
            if other["name"] != category["name"]:
                assert not added_needles.intersection(_needles_for_part(other["name"]))


@pytest.mark.asyncio
async def test_split_request_uses_entire_current_inline_configuration() -> None:
    payload = _config_payload()
    result_payload = {
        "id": "spl-config-test",
        "status": "COMPLETED",
        "result": {"segments": [{"category": "Index", "pages": [2, 3]}]},
    }
    completed = SimpleNamespace(
        **result_payload,
        model_dump=lambda **_kwargs: result_payload,
    )
    client = SimpleNamespace(
        split=SimpleNamespace(
            create=AsyncMock(return_value=SimpleNamespace(id="spl-config-test")),
            get=AsyncMock(return_value=completed),
        )
    )
    mapping, job_id, exchange = await _split_page_parts(
        client,
        file_id="file-config-test",
        split_config=Config.model_validate(payload).split,
        filename="compiled.pdf",
    )
    client.split.create.assert_awaited_once()
    sent = client.split.create.await_args.kwargs
    assert sent["file_input"] == "file-config-test"
    assert "configuration_id" not in sent
    assert sent["configuration"]["categories"] == payload["split"]["categories"]
    assert (
        sent["configuration"]["splitting_strategy"]
        == payload["split"]["splitting_strategy"]
    )
    assert exchange == {"sent": sent, "returned": result_payload}
    assert mapping == {2: ["Index"], 3: ["Index"]}
    assert job_id == "spl-config-test"
