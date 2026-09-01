import asyncio
from types import SimpleNamespace

import pytest

from extraction_review.scrutiny_workflow import collect_defect_findings


@pytest.mark.asyncio
async def test_collect_publishes_findings_as_they_finish() -> None:
    defects = [SimpleNamespace(check_id=f"D{i:03d}") for i in range(1, 5)]
    sizes: list[int] = []

    async def runner(defect: SimpleNamespace) -> SimpleNamespace:
        return SimpleNamespace(check_id=defect.check_id, error=None)

    async def on_update(findings: list[SimpleNamespace], stopped: bool) -> None:
        sizes.append(len(findings))
        assert stopped is False

    findings, stopped = await collect_defect_findings(
        defects,  # type: ignore[arg-type]
        runner,
        concurrency=2,
        on_update=on_update,
    )
    assert stopped is False
    assert {f.check_id for f in findings} == {d.check_id for d in defects}
    assert sizes == [1, 2, 3, 4]


@pytest.mark.asyncio
async def test_collect_stops_on_error_and_skips_queued_calls() -> None:
    started: list[str] = []
    defects = [SimpleNamespace(check_id=f"D{i:03d}") for i in range(1, 9)]

    async def runner(defect: SimpleNamespace) -> SimpleNamespace:
        started.append(defect.check_id)
        await asyncio.sleep(0.02)
        if defect.check_id == "D003":
            return SimpleNamespace(check_id="D003", error="boom")
        return SimpleNamespace(check_id=defect.check_id, error=None)

    findings, stopped = await collect_defect_findings(
        defects,  # type: ignore[arg-type]
        runner,
        concurrency=2,
    )
    assert stopped is True
    ids = {f.check_id for f in findings}
    assert "D003" in ids
    assert any(getattr(f, "error", None) for f in findings)
    assert len(started) < len(defects)
    assert "D008" not in started
