"""Inline split rules must reach the API intact or fail before submission."""

from __future__ import annotations

from types import SimpleNamespace
from unittest.mock import AsyncMock

import pytest

from extraction_review.config import SplitConfig
from extraction_review.process_file import (
    _split_api_configuration,
    _split_page_parts,
    _split_strategy_payload,
)


def _split_config(**overrides: object) -> SplitConfig:
    return SplitConfig.model_validate(
        {
            "product_type": "split_v1",
            "categories": [{"name": "Application", "description": "One application."}],
            "splitting_strategy": {
                "custom_instructions": "Keep each application separate."
            },
            **overrides,
        }
    )


def test_inline_rules_at_api_limits_are_preserved_in_full() -> None:
    category = {"name": "名" * 200, "description": "d" * 1996 + "END!"}
    strategy = {
        "allow_uncategorized": "include",
        "custom_instructions": "i" * 4996 + "END!",
        "min_pages_per_split": 1,
    }
    config = _split_config(categories=[category], splitting_strategy=strategy)

    sent = _split_api_configuration(config)

    assert sent == {"categories": [category], "splitting_strategy": strategy}
    assert config.categories[0].description == category["description"]
    assert config.splitting_strategy.model_dump(exclude_none=True) == strategy


@pytest.mark.parametrize("as_model", [False, True], ids=["dictionary", "sdk_model"])
def test_oversize_instructions_fail_without_mutating_source(as_model: bool) -> None:
    strategy = {"custom_instructions": "x" * 5001}
    source = (
        _split_config(splitting_strategy=strategy).splitting_strategy
        if as_model
        else strategy
    )

    with pytest.raises(
        ValueError,
        match=r"custom_instructions has 5001 characters; the Split API limit is 5000",
    ):
        _split_strategy_payload(source)

    remaining = source.model_dump(exclude_none=True) if as_model else source
    assert remaining == strategy
    assert len(remaining["custom_instructions"]) == 5001


@pytest.mark.parametrize(
    ("overrides", "message"),
    [
        (
            {"splitting_strategy": {"custom_instructions": "x" * 5001}},
            r"custom_instructions has 5001 characters; the Split API limit is 5000",
        ),
        (
            {"categories": [{"name": "x" * 201}]},
            r"categories\[0\].name has 201 characters; the Split API limit is 200",
        ),
        (
            {"categories": [{"name": "Application", "description": "x" * 2001}]},
            (
                r"categories\[0\].description for 'Application' has 2001 characters; "
                r"the Split API limit is 2000"
            ),
        ),
        (
            {"categories": [{"name": "   "}]},
            r"categories\[0\].name must be a non-empty string",
        ),
    ],
    ids=["instructions", "category_name", "category_description", "empty_name"],
)
@pytest.mark.asyncio
async def test_invalid_inline_rules_fail_before_creating_a_split_job(
    overrides: dict, message: str
) -> None:
    client = SimpleNamespace(split=SimpleNamespace(create=AsyncMock(), get=AsyncMock()))

    with pytest.raises(ValueError, match=message):
        await _split_page_parts(
            client,
            file_id="file-test",
            split_config=_split_config(**overrides),
        )

    client.split.create.assert_not_awaited()
    client.split.get.assert_not_awaited()


def test_optional_descriptions_and_strategy_are_still_optional() -> None:
    config = _split_config(
        categories=[{"name": "Application"}, {"name": "Annexures", "description": ""}],
        splitting_strategy=None,
    )

    assert _split_api_configuration(config) == {
        "categories": [{"name": "Application"}, {"name": "Annexures"}]
    }


@pytest.mark.asyncio
async def test_saved_configuration_does_not_validate_unused_inline_rules() -> None:
    config = _split_config(
        configuration_id="saved-split-config",
        splitting_strategy={"custom_instructions": "x" * 5001},
    )
    returned = {
        "id": "split-test",
        "status": "COMPLETED",
        "result": {"segments": [{"category": "Application", "pages": [1]}]},
    }
    completed = SimpleNamespace(**returned, model_dump=lambda **_kwargs: returned)
    client = SimpleNamespace(
        split=SimpleNamespace(
            create=AsyncMock(return_value=SimpleNamespace(id="split-test")),
            get=AsyncMock(return_value=completed),
        )
    )

    mapping, job_id, exchange = await _split_page_parts(
        client, file_id="file-test", split_config=config
    )

    sent = client.split.create.await_args.kwargs
    assert sent["configuration_id"] == "saved-split-config"
    assert "configuration" not in sent
    assert exchange == {"sent": sent, "returned": returned}
    assert mapping == {1: ["Application"]}
    assert job_id == "split-test"
