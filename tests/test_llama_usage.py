from __future__ import annotations

from types import SimpleNamespace
from unittest.mock import AsyncMock

import pytest

from extraction_review.llama_usage import (
    collect_optional_usage,
    collect_parse_usage,
    summarize_llamacloud_usage,
    usage_fields,
    usage_job,
    usage_status_message,
)


def test_usage_fields_reads_parse_job_wrapper() -> None:
    fields = usage_fields(
        {"job": {"id": "pjb-1", "tier": "fast", "usage": {"credits": 8.0}}}
    )
    assert fields["credits"] == 8.0
    assert fields["extract_credits"] is None


def test_usage_fields_reads_extract_breakdown() -> None:
    fields = usage_fields(
        {
            "id": "ext-1",
            "usage": {
                "credits": 75.0,
                "extract_credits": 45.0,
                "parse_credits": 30.0,
            },
        }
    )
    assert fields == {
        "credits": 75.0,
        "extract_credits": 45.0,
        "parse_credits": 30.0,
    }


def test_usage_fields_treats_missing_as_null() -> None:
    assert usage_fields({"id": "spl-1", "status": "COMPLETED"})["credits"] is None


def test_summarize_llamacloud_usage_sums_billed_jobs() -> None:
    summary = summarize_llamacloud_usage(
        [
            usage_job(
                product="parse",
                job_id="pjb-1",
                payload={"job": {"usage": {"credits": 12}}},
                slot_id="petition",
            ),
            usage_job(
                product="parse",
                job_id="pjb-2",
                payload={"job": {"usage": {"credits": 4}}},
                slot_id="annexure_p1",
            ),
            usage_job(
                product="extract",
                job_id="ext-1",
                payload={
                    "usage": {
                        "credits": 20,
                        "extract_credits": 20,
                        "parse_credits": 0,
                    }
                },
            ),
            usage_job(product="split", job_id="spl-1"),
        ]
    )
    assert summary["credits"] == 36.0
    assert summary["by_product"] == {"extract": 20.0, "parse": 16.0}
    assert summary["pending_jobs"] == 1
    assert summary["estimated_usd"] == 0.045
    assert "LlamaCloud 36 credits" in usage_status_message(summary)


@pytest.mark.asyncio
async def test_collect_parse_usage_retries_null_credits(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr("extraction_review.llama_usage.USAGE_RETRY_S", 0)
    client = SimpleNamespace(
        parsing=SimpleNamespace(
            get=AsyncMock(
                return_value={"job": {"id": "pjb-1", "usage": {"credits": 9}}}
            )
        )
    )
    row = await collect_parse_usage(
        client,
        "pjb-1",
        payload={"job": {"id": "pjb-1", "usage": {"credits": None}}},
        slot_id="cover_page",
        pages=1,
    )
    assert row["credits"] == 9
    assert row["slot_id"] == "cover_page"
    client.parsing.get.assert_awaited_once()


@pytest.mark.asyncio
async def test_collect_optional_usage_does_not_sleep_when_api_has_no_usage() -> None:
    client = SimpleNamespace(
        classify=SimpleNamespace(
            get=AsyncMock(return_value={"id": "cls-1", "status": "COMPLETED"})
        )
    )
    row = await collect_optional_usage(
        client,
        product="classify",
        job_id="cls-1",
        payload={"id": "cls-1", "status": "COMPLETED"},
    )
    assert row["product"] == "classify"
    assert row["job_id"] == "cls-1"
    assert row["credits"] is None
    client.classify.get.assert_awaited_once()
