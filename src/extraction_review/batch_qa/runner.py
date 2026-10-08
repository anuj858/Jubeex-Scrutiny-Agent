"""Sequential end-to-end runner against a deployed JubeeX scrutiny API."""

from __future__ import annotations

import hashlib
import json
import logging
import re
import tempfile
import time
import uuid
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from pypdf import PdfReader, PdfWriter

from .client import DeployedApiClient, QaApiError, S3InputStager
from .evaluate import evaluate_extraction, evaluate_scrutiny, evaluate_split
from .models import BatchResult, CaseResult, StageResult

logger = logging.getLogger(__name__)


@dataclass
class BatchRunConfig:
    input_dir: Path
    api_url: str
    bucket: str
    region: str = "ap-south-1"
    api_key: str | None = None
    filing_type: str | None = None
    special_category: str | None = None
    poll_interval: float = 3
    job_timeout: float = 7200
    request_timeout: float = 60
    cleanup_staged_inputs: bool = False
    dry_run: bool = False


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _page_count(path: Path) -> int:
    return len(PdfReader(str(path)).pages)


def _job_result(payload: dict[str, Any]) -> dict[str, Any]:
    result = payload.get("result")
    return result if isinstance(result, dict) else {}


def _artifact_url(payload: dict[str, Any], step: str) -> str | None:
    direct = payload.get(step)
    if isinstance(direct, dict) and direct.get("url"):
        return str(direct["url"])
    artifacts = payload.get("artifacts")
    if isinstance(artifacts, dict):
        item = artifacts.get(step)
        if isinstance(item, dict) and item.get("url"):
            return str(item["url"])
    return None


def _documents_from_parts(parts: list[dict[str, Any]]) -> list[dict[str, Any]]:
    documents = []
    for part in parts:
        file_id = str(part.get("file_id") or "").strip()
        if not file_id:
            continue
        filename = str(part.get("filename") or part.get("name") or "part.pdf")
        documents.append(
            {
                "name": filename,
                "filename": filename,
                "slot_id": str(part.get("slot_id") or "undefined"),
                "file_id": file_id,
                "document_id": str(part.get("document_id") or uuid.uuid4()),
            }
        )
    return documents


def _part_page_numbers(part: dict[str, Any], *, page_count: int) -> list[int]:
    """Return validated, one-based source pages for one split result part."""
    spans = part.get("page_span")
    if not isinstance(spans, list):
        return []
    pages: list[int] = []
    seen: set[int] = set()
    for span in spans:
        if not isinstance(span, dict):
            continue
        try:
            start = int(span.get("start"))
            end = int(span.get("end"))
        except (TypeError, ValueError):
            continue
        if start < 1 or end < start or end > page_count:
            raise QaApiError(
                f"Invalid split page span {start}-{end}; source has {page_count} pages"
            )
        for page_number in range(start, end + 1):
            if page_number not in seen:
                pages.append(page_number)
                seen.add(page_number)
    return pages


def _safe_part_filename(part: dict[str, Any], index: int) -> str:
    raw = str(
        part.get("filename")
        or part.get("name")
        or part.get("label")
        or part.get("slot_id")
        or f"part-{index}"
    ).strip()
    stem = Path(raw).stem or f"part-{index}"
    safe_stem = re.sub(r"[^A-Za-z0-9._-]+", "-", stem).strip("-.")
    return f"{index:03d}-{safe_stem or f'part-{index}'}.pdf"


class BatchQaRunner:
    def __init__(
        self,
        config: BatchRunConfig,
        *,
        manifest: dict[str, dict[str, Any]] | None = None,
        client: DeployedApiClient | None = None,
        stager: S3InputStager | None = None,
    ) -> None:
        self.config = config
        self.manifest = manifest or {}
        self.client = client or DeployedApiClient(
            config.api_url,
            api_key=config.api_key,
            request_timeout=config.request_timeout,
        )
        run_prefix = _new_run_id()
        self.stager = stager
        if self.stager is None and not config.dry_run:
            self.stager = S3InputStager(
                config.bucket,
                region=config.region,
                prefix=f"qa-runs/{run_prefix}/inputs",
            )

    def close(self) -> None:
        self.client.close()

    def discover(self) -> list[Path]:
        if not self.config.input_dir.is_dir():
            raise ValueError(f"Input directory does not exist: {self.config.input_dir}")
        return sorted(
            path
            for path in self.config.input_dir.iterdir()
            if path.is_file() and path.suffix.lower() == ".pdf"
        )

    def run(self) -> BatchResult:
        run_id = _new_run_id()
        batch = BatchResult(
            run_id=run_id,
            started_at=datetime.now(UTC).isoformat(),
            api_url=self.config.api_url,
            input_dir=str(self.config.input_dir.resolve()),
        )
        files = self.discover()
        if not files:
            raise ValueError(f"No PDF files found in {self.config.input_dir}")
        if not self.config.dry_run:
            self.client.health()
        for position, path in enumerate(files, start=1):
            logger.info("[%s/%s] Processing %s", position, len(files), path.name)
            batch.cases.append(self.run_case(path))
        batch.completed_at = datetime.now(UTC).isoformat()
        return batch

    def run_case(self, path: Path) -> CaseResult:
        expected = self.manifest.get(path.name, {})
        filing_type = str(expected.get("filing_type") or self.config.filing_type or "")
        special_category = expected.get(
            "special_category", self.config.special_category
        )
        organization_id = str(uuid.uuid4())
        workspace_id = str(uuid.uuid4())
        try:
            file_hash = _sha256(path)
            page_count = _page_count(path)
        except Exception as exc:
            case = CaseResult(
                filename=path.name,
                source_path=str(path.resolve()),
                file_hash="",
                page_count=0,
                filing_type=filing_type or "AUTO",
                special_category=str(special_category) if special_category else None,
                organization_id=organization_id,
                workspace_id=workspace_id,
            )
            case.fail("input", exc)
            case.total_seconds = 0.0
            logger.exception("Case %s failed while reading input", path.name)
            return case
        case = CaseResult(
            filename=path.name,
            source_path=str(path.resolve()),
            file_hash=file_hash,
            page_count=page_count,
            filing_type=filing_type or "AUTO",
            special_category=str(special_category) if special_category else None,
            organization_id=organization_id,
            workspace_id=workspace_id,
        )
        if self.config.dry_run:
            case.status = "dry_run"
            return case
        started = time.monotonic()
        staged_keys: list[str] = []
        active_stage = "upload"
        try:
            staged_key, download_url = self._stage_input(case, path)
            staged_keys.append(staged_key)
            split_payload = self._base_payload(
                case, include_filing_type=bool(filing_type)
            )
            split_payload["documents"] = [
                {
                    "name": path.name,
                    "filename": path.name,
                    "document_id": str(uuid.uuid4()),
                    "download_url": download_url,
                    "slot_id": "compiled_petition",
                }
            ]
            active_stage = "split"
            if filing_type:
                accepted = self.client.start_split(split_payload)
            else:
                split_payload["job_type"] = "upload_compiled"
                accepted = self.client.start_compiled(split_payload)
            split_job = self._run_job(case, "split", accepted)
            case.raw_jobs["split"] = split_job
            split_result = _job_result(split_job)
            detected_type = str(split_result.get("filing_type") or "").strip()
            if detected_type:
                case.filing_type = detected_type
            if case.filing_type == "AUTO":
                raise QaApiError("Compiled split completed without a filing_type")
            case.split_parts = [
                item for item in split_result.get("parts", []) if isinstance(item, dict)
            ]
            case.split_evaluation = evaluate_split(
                case.split_parts, expected.get("split")
            )

            documents, locally_staged_keys = self._documents_for_extraction(case, path)
            staged_keys.extend(locally_staged_keys)
            extract_payload = self._base_payload(case)
            extract_payload["documents"] = documents
            active_stage = "extraction"
            extract_job = self._run_job(
                case, "extraction", self.client.start_extract(extract_payload)
            )
            case.raw_jobs["extraction"] = extract_job
            extract_result = _job_result(extract_job)
            case.agent_data_id = (
                str(
                    extract_job.get("agent_data_id")
                    or extract_result.get("agent_data_id")
                    or ""
                )
                or None
            )
            extract_url = _artifact_url(extract_result, "extract")
            if extract_url:
                case.extraction_data = self.client.fetch_json_url(extract_url)
            case.extraction_evaluation = evaluate_extraction(
                case.extraction_data, expected.get("extraction")
            )
            if not case.agent_data_id:
                raise QaApiError("Extraction completed without agent_data_id")

            index_payload = self._base_payload(case)
            index_payload.update(
                {
                    "documents": documents,
                    "parsed_slots": extract_result.get("parsed_slots") or [],
                    "edited": False,
                }
            )
            active_stage = "assemble_scrutiny"
            scrutiny_job = self._run_job(
                case,
                "assemble_scrutiny",
                self.client.start_index(case.agent_data_id, index_payload),
            )
            case.raw_jobs["assemble_scrutiny"] = scrutiny_job
            scrutiny_result = _job_result(scrutiny_job)
            report = scrutiny_result.get("report")
            if isinstance(report, dict):
                case.scrutiny_report = report
            else:
                defects_url = _artifact_url(scrutiny_result, "defects")
                if defects_url:
                    case.scrutiny_report = self.client.fetch_json_url(defects_url)
            case.scrutiny_evaluation = evaluate_scrutiny(
                case.scrutiny_report, expected.get("scrutiny")
            )
            statuses = {
                case.split_evaluation.get("status"),
                case.extraction_evaluation.get("status"),
                case.scrutiny_evaluation.get("status"),
            }
            if "failed" in statuses:
                case.verification_status = "failed"
            elif statuses == {"passed"}:
                case.verification_status = "passed"
            else:
                case.verification_status = "not_reviewed"
            case.status = "completed"
        except Exception as exc:
            case.fail(active_stage, exc)
            logger.exception("Case %s failed during %s", path.name, active_stage)
        finally:
            case.total_seconds = round(time.monotonic() - started, 3)
            if staged_keys and self.config.cleanup_staged_inputs:
                assert self.stager is not None
                for key in staged_keys:
                    try:
                        self.stager.delete(key)
                    except Exception:
                        logger.exception("Could not delete staged input %s", key)
        return case

    def _documents_for_extraction(
        self, case: CaseResult, source_path: Path
    ) -> tuple[list[dict[str, Any]], list[str]]:
        """Use worker file IDs, or stage QA-only slices when uploads were skipped."""
        if self.stager is None:
            raise RuntimeError("S3 input stager is not configured")
        reader: PdfReader | None = None
        documents: list[dict[str, Any]] = []
        staged_keys: list[str] = []
        with tempfile.TemporaryDirectory(prefix="jubeex-batch-qa-") as temp_dir:
            temp_root = Path(temp_dir)
            for index, part in enumerate(case.split_parts, start=1):
                file_id = str(part.get("file_id") or "").strip()
                filename = str(
                    part.get("filename") or part.get("name") or f"part-{index}.pdf"
                )
                document: dict[str, Any] = {
                    "name": filename,
                    "filename": filename,
                    "slot_id": str(part.get("slot_id") or "undefined"),
                    "document_id": str(part.get("document_id") or uuid.uuid4()),
                }
                if file_id:
                    document["file_id"] = file_id
                    documents.append(document)
                    continue

                if reader is None:
                    reader = PdfReader(str(source_path))
                pages = _part_page_numbers(part, page_count=len(reader.pages))
                if not pages:
                    slot_id = document["slot_id"]
                    raise QaApiError(
                        f"Split part {slot_id!r} has neither file_id nor usable page_span"
                    )
                slice_path = temp_root / _safe_part_filename(part, index)
                writer = PdfWriter()
                for page_number in pages:
                    writer.add_page(reader.pages[page_number - 1])
                with slice_path.open("wb") as handle:
                    writer.write(handle)
                key, url = self.stager.upload(slice_path, case_id=case.workspace_id)
                staged_keys.append(key)
                document["download_url"] = url
                documents.append(document)
                logger.info(
                    "%s [split] staged QA slice %s (%s page(s))",
                    case.filename,
                    document["slot_id"],
                    len(pages),
                )
        if not documents:
            raise QaApiError("Split completed without any usable document parts")
        return documents, staged_keys

    def _base_payload(
        self, case: CaseResult, *, include_filing_type: bool = True
    ) -> dict[str, Any]:
        payload: dict[str, Any] = {
            "court": "sci",
            "organization_id": case.organization_id,
            "workspace_id": case.workspace_id,
            "user_id": "batch-qa",
        }
        if include_filing_type:
            payload["filing_type"] = case.filing_type
        if case.special_category:
            payload["special_category"] = case.special_category
        return payload

    def _stage_input(self, case: CaseResult, path: Path) -> tuple[str, str]:
        stage = StageResult(name="upload", status="running")
        case.stages["upload"] = stage
        started = time.monotonic()
        try:
            if self.stager is None:
                raise RuntimeError("S3 input stager is not configured")
            key, url = self.stager.upload(path, case_id=case.workspace_id)
            case.staged_s3_key = key
            stage.status = "completed"
            return key, url
        except Exception as exc:
            stage.status = "failed"
            stage.error = str(exc)
            raise
        finally:
            stage.duration_seconds = round(time.monotonic() - started, 3)

    def _run_job(
        self, case: CaseResult, stage_name: str, accepted: dict[str, Any]
    ) -> dict[str, Any]:
        job_id = str(accepted.get("job_id") or "")
        if not job_id:
            raise QaApiError(f"{stage_name} did not return job_id")
        stage = StageResult(name=stage_name, status="running", job_id=job_id)
        case.stages[stage_name] = stage

        def progress(observation: dict[str, Any]) -> None:
            logger.info(
                "%s [%s] %s%% %s: %s",
                case.filename,
                stage_name,
                observation.get("progress"),
                observation.get("stage"),
                observation.get("message"),
            )

        try:
            polled = self.client.poll_job(
                job_id,
                poll_interval=self.config.poll_interval,
                timeout_seconds=self.config.job_timeout,
                on_progress=progress,
            )
            stage.status = "completed"
            stage.duration_seconds = polled.duration_seconds
            stage.observations = polled.observations
            return polled.payload
        except Exception as exc:
            stage.status = "failed"
            stage.error = str(exc)
            raise


def write_raw_case(case: CaseResult, case_dir: Path) -> None:
    case_dir.mkdir(parents=True, exist_ok=True)
    (case_dir / "result.json").write_text(
        json.dumps(case.as_dict(), ensure_ascii=False, indent=2), encoding="utf-8"
    )


def _new_run_id() -> str:
    timestamp = datetime.now(UTC).strftime("%Y%m%dT%H%M%SZ")
    return f"{timestamp}-{uuid.uuid4().hex[:8]}"
