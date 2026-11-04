from __future__ import annotations

import json
import zipfile
from pathlib import Path
from typing import Any

from pypdf import PdfWriter

from extraction_review.batch_qa import client as batch_client
from extraction_review.batch_qa.client import PolledJob, QaApiError, S3InputStager
from extraction_review.batch_qa.evaluate import (
    evaluate_extraction,
    evaluate_scrutiny,
    evaluate_split,
)
from extraction_review.batch_qa.models import BatchResult
from extraction_review.batch_qa.reporting import write_all_reports
from extraction_review.batch_qa.runner import BatchQaRunner, BatchRunConfig


def _pdf(path: Path) -> None:
    writer = PdfWriter()
    writer.add_blank_page(width=612, height=792)
    with path.open("wb") as handle:
        writer.write(handle)


class FakeStager:
    def __init__(self) -> None:
        self.uploaded: list[str] = []

    def upload(self, path: Path, *, case_id: str) -> tuple[str, str]:
        self.uploaded.append(path.name)
        return f"qa/{case_id}/{path.name}", f"https://files.test/{path.name}"

    def delete(self, key: str) -> None:
        return None


class FakeClient:
    def __init__(
        self, *, fail_file: str | None = None, include_split_file_ids: bool = True
    ) -> None:
        self.fail_file = fail_file
        self.include_split_file_ids = include_split_file_ids
        self.started: list[tuple[str, str]] = []
        self.payloads: dict[str, dict[str, Any]] = {}
        self.extract_documents: list[dict[str, Any]] = []

    def close(self) -> None:
        return None

    def health(self) -> dict[str, str]:
        return {"status": "ok"}

    def start_split(self, payload: dict[str, Any]) -> dict[str, str]:
        return self._accept_split(payload)

    def start_compiled(self, payload: dict[str, Any]) -> dict[str, str]:
        assert payload["job_type"] == "upload_compiled"
        return self._accept_split(payload)

    def _accept_split(self, payload: dict[str, Any]) -> dict[str, str]:
        name = payload["documents"][0]["name"]
        job_id = f"split-{name}"
        self.started.append(("split", name))
        self.payloads[job_id] = {
            "status": "completed",
            "result": {
                "filing_type": "SLP_CIVIL",
                "parts": [
                    {
                        "slot_id": "petition",
                        "label": "Main Petition",
                        "filename": "Main Petition.pdf",
                        **(
                            {"file_id": f"file-{name}"}
                            if self.include_split_file_ids
                            else {}
                        ),
                        "page_span": [{"start": 1, "end": 1}],
                    }
                ],
            },
        }
        return {"job_id": job_id}

    def start_extract(self, payload: dict[str, Any]) -> dict[str, str]:
        document = payload["documents"][0]
        self.extract_documents = payload["documents"]
        file_id = str(document.get("file_id") or "")
        name = file_id.removeprefix("file-") or str(document["name"])
        if name == self.fail_file:
            raise QaApiError("planned extraction failure")
        job_id = f"extract-{name}"
        self.started.append(("extract", name))
        self.payloads[job_id] = {
            "status": "completed",
            "agent_data_id": f"agent-{name}",
            "result": {
                "agent_data_id": f"agent-{name}",
                "parsed_slots": ["petition"],
            },
        }
        return {"job_id": job_id}

    def start_index(
        self, agent_data_id: str, payload: dict[str, Any]
    ) -> dict[str, str]:
        name = agent_data_id.removeprefix("agent-")
        job_id = f"index-{name}"
        self.started.append(("index", name))
        self.payloads[job_id] = {
            "status": "completed",
            "result": {
                "report": {
                    "findings": [
                        {
                            "check_id": "D-1",
                            "status": "defect_found",
                            "defect": "Test defect",
                        }
                    ]
                }
            },
        }
        return {"job_id": job_id}

    def poll_job(self, job_id: str, **_kwargs: Any) -> PolledJob:
        return PolledJob(self.payloads[job_id], 1.25, [])

    def fetch_json_url(self, _url: str) -> dict[str, Any]:
        return {}


def test_expected_evaluators_are_deterministic() -> None:
    parts = [{"slot_id": "petition", "page_span": [{"start": 1, "end": 3}]}]
    assert evaluate_split(parts, {"petition": [[1, 3]]})["status"] == "passed"
    assert (
        evaluate_extraction({"data": {"name": "A"}}, {"data.name": "A"})["status"]
        == "passed"
    )
    report = {"findings": [{"check_id": "D-1", "status": "defect_found"}]}
    assert evaluate_scrutiny(report, {"D-1": "defect_found"})["status"] == "passed"
    assert (
        evaluate_scrutiny(
            report,
            {"strict": False, "checks": {"D-1": "defect_found"}},
        )["status"]
        == "passed"
    )


def test_runner_processes_files_sequentially_and_continues_after_failure(
    tmp_path: Path,
) -> None:
    _pdf(tmp_path / "a.pdf")
    _pdf(tmp_path / "b.pdf")
    client = FakeClient(fail_file="a.pdf")
    runner = BatchQaRunner(
        BatchRunConfig(
            input_dir=tmp_path,
            api_url="https://api.test",
            bucket="bucket",
            filing_type="SLP_CIVIL",
        ),
        client=client,  # type: ignore[arg-type]
        stager=FakeStager(),  # type: ignore[arg-type]
    )
    batch = runner.run()
    assert [case.filename for case in batch.cases] == ["a.pdf", "b.pdf"]
    assert batch.cases[0].status == "failed"
    assert batch.cases[0].first_failure_stage == "extraction"
    assert batch.cases[1].status == "completed"
    assert batch.cases[1].verification_status == "not_reviewed"
    assert client.started == [
        ("split", "a.pdf"),
        ("split", "b.pdf"),
        ("extract", "b.pdf"),
        ("index", "b.pdf"),
    ]


def test_runner_stages_qa_slices_when_worker_omits_file_ids(tmp_path: Path) -> None:
    _pdf(tmp_path / "case.pdf")
    client = FakeClient(include_split_file_ids=False)
    stager = FakeStager()
    runner = BatchQaRunner(
        BatchRunConfig(
            input_dir=tmp_path,
            api_url="https://api.test",
            bucket="bucket",
            filing_type="SLP_CIVIL",
        ),
        client=client,  # type: ignore[arg-type]
        stager=stager,  # type: ignore[arg-type]
    )

    batch = runner.run()

    assert batch.cases[0].status == "completed"
    assert stager.uploaded == ["case.pdf", "001-Main-Petition.pdf"]
    assert client.extract_documents[0]["download_url"] == (
        "https://files.test/001-Main-Petition.pdf"
    )
    assert "file_id" not in client.extract_documents[0]


def test_runner_records_input_removed_after_discovery_and_continues(
    tmp_path: Path,
) -> None:
    removed = tmp_path / "a.pdf"
    remaining = tmp_path / "b.pdf"
    _pdf(removed)
    _pdf(remaining)
    client = FakeClient()
    runner = BatchQaRunner(
        BatchRunConfig(
            input_dir=tmp_path,
            api_url="https://api.test",
            bucket="bucket",
            filing_type="SLP_CIVIL",
        ),
        client=client,  # type: ignore[arg-type]
        stager=FakeStager(),  # type: ignore[arg-type]
    )
    discovered = runner.discover()
    removed.unlink()
    runner.discover = lambda: discovered  # type: ignore[method-assign]

    batch = runner.run()

    assert batch.cases[0].status == "failed"
    assert batch.cases[0].first_failure_stage == "input"
    assert batch.cases[1].status == "completed"


def test_reports_include_excel_sheets_and_all_defects(tmp_path: Path) -> None:
    _pdf(tmp_path / "case.pdf")
    runner = BatchQaRunner(
        BatchRunConfig(
            input_dir=tmp_path,
            api_url="https://api.test",
            bucket="bucket",
            filing_type="SLP_CIVIL",
        ),
        client=FakeClient(),  # type: ignore[arg-type]
        stager=FakeStager(),  # type: ignore[arg-type]
    )
    batch = runner.run()
    output = tmp_path / "reports"
    paths = write_all_reports(batch, output)
    assert all(path.exists() for path in paths.values())
    payload = json.loads(paths["json"].read_text(encoding="utf-8"))
    assert payload["cases"][0]["scrutiny_report"]["findings"][0]["check_id"] == "D-1"
    with zipfile.ZipFile(paths["xlsx"]) as archive:
        workbook_xml = archive.read("xl/workbook.xml").decode("utf-8")
    for sheet in (
        "Summary",
        "Split results",
        "All defects",
        "Extraction data",
        "Timings",
        "Expected comparison",
    ):
        assert sheet in workbook_xml


def test_dry_run_does_not_need_s3_or_api(tmp_path: Path) -> None:
    _pdf(tmp_path / "case.pdf")
    config = BatchRunConfig(
        input_dir=tmp_path,
        api_url="https://api.test",
        bucket="dry-run",
        filing_type="SLP_CIVIL",
        dry_run=True,
    )
    runner = BatchQaRunner(config, client=FakeClient())  # type: ignore[arg-type]
    batch: BatchResult = runner.run()
    assert batch.cases[0].status == "dry_run"


def test_runner_can_classify_when_no_filing_type_is_supplied(tmp_path: Path) -> None:
    _pdf(tmp_path / "case.pdf")
    runner = BatchQaRunner(
        BatchRunConfig(
            input_dir=tmp_path,
            api_url="https://api.test",
            bucket="bucket",
        ),
        client=FakeClient(),  # type: ignore[arg-type]
        stager=FakeStager(),  # type: ignore[arg-type]
    )
    batch = runner.run()
    assert batch.cases[0].status == "completed"
    assert batch.cases[0].filing_type == "SLP_CIVIL"


def test_s3_stager_uses_regional_sigv4_endpoint(monkeypatch: Any) -> None:
    captured: dict[str, Any] = {}

    def fake_client(service: str, **kwargs: Any) -> object:
        captured["service"] = service
        captured.update(kwargs)
        return object()

    monkeypatch.setattr(batch_client.boto3, "client", fake_client)

    S3InputStager(
        "example-bucket",
        region="ap-south-1",
        prefix="qa-runs/test",
    )

    assert captured["service"] == "s3"
    assert captured["region_name"] == "ap-south-1"
    assert captured["endpoint_url"] == "https://s3.ap-south-1.amazonaws.com"
    assert captured["config"].signature_version == "s3v4"
