from types import SimpleNamespace

import pytest

from extraction_review.scrutiny_workflow import require_successful_checks


def test_failed_defect_check_rejects_completed_scrutiny() -> None:
    findings = [SimpleNamespace(error=None), SimpleNamespace(error="Vertex 429")]
    with pytest.raises(RuntimeError, match="1 of 2 checks failed"):
        require_successful_checks(findings, planned=2)


def test_inconclusive_but_evaluated_check_can_complete() -> None:
    require_successful_checks([SimpleNamespace(error=None)], planned=1)
