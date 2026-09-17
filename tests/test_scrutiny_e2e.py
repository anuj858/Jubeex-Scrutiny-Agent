"""End-to-end scrutiny-check: workflow, streaming, persist, fail-fast."""

from __future__ import annotations

from types import SimpleNamespace

import pytest

from extraction_review.scrutiny.rules import get_catalogue
from extraction_review.scrutiny_workflow import (
    assert_filing_ready_for_scrutiny,
    collect_defect_findings,
)


def test_scrutiny_allows_pending_review_and_blocks_rejected() -> None:
    assert_filing_ready_for_scrutiny("pending_review", "Cover Page.pdf")
    assert_filing_ready_for_scrutiny("approved", "Cover Page.pdf")
    assert_filing_ready_for_scrutiny("error", "Cover_Page.pdf")
    assert_filing_ready_for_scrutiny("success", "Cover_Page.pdf")
    assert_filing_ready_for_scrutiny(None, "Cover Page.pdf")
    with pytest.raises(ValueError, match="rejected"):
        assert_filing_ready_for_scrutiny("rejected", "Cover Page.pdf")


@pytest.mark.asyncio
async def test_all_defects_are_catalogue_sized() -> None:
    get_catalogue.cache_clear()
    catalogue = get_catalogue()
    assert catalogue.catalogue_version == "2.8.0"
    assert len(catalogue.defects) == 327
    assert catalogue.defect("D-1").check_id == "D-1"
    assert catalogue.defect("D-323").check_id == "D-323"


@pytest.mark.asyncio
async def test_collect_queue_does_not_start_after_abort() -> None:
    started: list[str] = []
    defects = [SimpleNamespace(check_id=f"D{i:03d}") for i in range(1, 13)]

    async def runner(defect: SimpleNamespace) -> SimpleNamespace:
        started.append(defect.check_id)
        if defect.check_id == "D001":
            return SimpleNamespace(check_id="D001", error="stop")
        return SimpleNamespace(check_id=defect.check_id, error=None)

    findings, stopped = await collect_defect_findings(
        defects,  # type: ignore[arg-type]
        runner,
        concurrency=6,
    )
    assert stopped is True
    assert findings[0].check_id == "D001"
    assert len(started) <= 6
    assert "D012" not in started
