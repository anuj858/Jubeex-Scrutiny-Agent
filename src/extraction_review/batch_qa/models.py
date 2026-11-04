"""Serializable result models for deployed batch QA runs."""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any


@dataclass
class StageResult:
    name: str
    status: str = "not_started"
    duration_seconds: float | None = None
    job_id: str | None = None
    error: str | None = None
    observations: list[dict[str, Any]] = field(default_factory=list)


@dataclass
class CaseResult:
    filename: str
    source_path: str
    file_hash: str
    page_count: int
    filing_type: str
    special_category: str | None
    organization_id: str
    workspace_id: str
    status: str = "running"
    verification_status: str = "not_reviewed"
    first_failure_stage: str | None = None
    error: str | None = None
    total_seconds: float | None = None
    agent_data_id: str | None = None
    staged_s3_key: str | None = None
    stages: dict[str, StageResult] = field(default_factory=dict)
    raw_jobs: dict[str, dict[str, Any]] = field(default_factory=dict)
    split_parts: list[dict[str, Any]] = field(default_factory=list)
    extraction_data: dict[str, Any] = field(default_factory=dict)
    scrutiny_report: dict[str, Any] = field(default_factory=dict)
    split_evaluation: dict[str, Any] = field(default_factory=dict)
    extraction_evaluation: dict[str, Any] = field(default_factory=dict)
    scrutiny_evaluation: dict[str, Any] = field(default_factory=dict)

    def fail(self, stage: str, error: BaseException | str) -> None:
        message = str(error)
        self.status = "failed"
        self.first_failure_stage = self.first_failure_stage or stage
        self.error = message
        result = self.stages.setdefault(stage, StageResult(name=stage))
        result.status = "failed"
        result.error = message

    def as_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass
class BatchResult:
    run_id: str
    started_at: str
    completed_at: str | None = None
    api_url: str = ""
    input_dir: str = ""
    cases: list[CaseResult] = field(default_factory=list)

    def as_dict(self) -> dict[str, Any]:
        return asdict(self)

    @property
    def completed_count(self) -> int:
        return sum(case.status == "completed" for case in self.cases)

    @property
    def failed_count(self) -> int:
        return sum(case.status == "failed" for case in self.cases)

    @property
    def report_stem(self) -> str:
        return f"jubeex-batch-qa-{self.run_id}"

    def output_path(self, report_dir: Path, suffix: str) -> Path:
        return report_dir / f"{self.report_stem}.{suffix.lstrip('.')}"
