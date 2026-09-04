"""Production HTTP API for the JubeeX backend (not the Llama UI).

Run:

    uv run uvicorn extraction_review.api:app --host 0.0.0.0 --port 8000

If ``JUBEEX_API_KEY`` is set, protected routes require
``Authorization: Bearer <key>`` or ``X-API-Key: <key>``.
"""

from __future__ import annotations

import logging
import os
import uuid
from datetime import UTC, datetime
from typing import Any, Literal

from dotenv import load_dotenv
from fastapi import BackgroundTasks, Depends, FastAPI, Header, HTTPException, status
from fastapi.middleware.cors import CORSMiddleware
from pydantic import (
    AliasChoices,
    BaseModel,
    ConfigDict,
    Field,
    ValidationError,
    field_validator,
    model_validator,
)

from .clients import get_llama_cloud_client
from .config import EXTRACTED_DATA_COLLECTION as FILING_COLLECTION
from .config import JUBEEX_FILING_TYPES
from .process_file import (
    FileEvent,
    blank_or_placeholder,
    intake_mode,
    normalize_job_type,
)
from .process_file import workflow as process_file_workflow
from .scrutiny_workflow import ScrutinyEvent
from .scrutiny_workflow import workflow as scrutiny_workflow
from .split_upload import type_catalog, ui_catalog

load_dotenv()

logger = logging.getLogger(__name__)

JobKind = Literal["process_file", "scrutiny"]
JobStatus = Literal["running", "completed", "failed"]


class DocumentIn(BaseModel):
    model_config = ConfigDict(extra="ignore")

    name: str | None = Field(default=None, examples=["01_Petition.pdf"])
    document_id: str | None = Field(
        default=None, examples=["11aa22bb-33cc-44dd-85ee-66ff77889900"]
    )
    download_url: str | None = Field(
        default=None,
        examples=["https://storage.example/filings/01_Petition.pdf"],
    )
    slot_id: str | None = None
    file_id: str | None = None
    filename: str | None = None
    file_url: str | None = None

    @field_validator(
        "name",
        "document_id",
        "download_url",
        "slot_id",
        "file_id",
        "filename",
        "file_url",
        mode="before",
    )
    @classmethod
    def _drop_swagger_placeholders(cls, value: object) -> str | None:
        return blank_or_placeholder(value)


class CreateFilingRequest(BaseModel):
    """Body for POST /v1/filings — same fields as process-file start_event."""

    model_config = ConfigDict(
        json_schema_extra={
            "examples": [
                {
                    "job_type": "upload_separate",
                    "filing_type": "SLP_CIVIL",
                    "organization_id": "3f2a9c1e-8b44-4d21-9a70-1c8d4e6b2f11",
                    "workspace_id": "b20c7d91-4e55-48aa-a013-9d6e2f88c104",
                    "user_id": "7b12e4aa-0d55-4c91-b3e8-2a6f19c8d447",
                    "documents": [
                        {
                            "name": "01_Petition.pdf",
                            "document_id": "11aa22bb-33cc-44dd-85ee-66ff77889900",
                            "download_url": "https://storage.example/01_Petition.pdf",
                        }
                    ],
                }
            ]
        }
    )

    job_type: str = Field(examples=["upload_separate", "upload_compiled"])
    filing_type: str | None = Field(
        default=None,
        examples=["SLP_CIVIL", "SLP_CRIMINAL"],
    )
    organization_id: str | None = None
    workspace_id: str | None = None
    user_id: str | None = None
    documents: list[DocumentIn] = Field(default_factory=list)

    @field_validator(
        "job_type",
        "filing_type",
        "organization_id",
        "workspace_id",
        "user_id",
        mode="before",
    )
    @classmethod
    def _drop_swagger_placeholders(cls, value: object) -> str | None:
        return blank_or_placeholder(value)

    @model_validator(mode="after")
    def _require_real_job_and_filing_type(self) -> CreateFilingRequest:
        job = (self.job_type or "").strip().lower()
        if not job:
            raise ValueError(
                "job_type must be upload_compiled or upload_separate, "
                "not the Swagger placeholder 'string'"
            )
        self.job_type = job
        catalog_types = set(ui_catalog())
        mode = normalize_job_type(job)
        if mode == "split" or (
            mode is None and len(self.documents) > 1
        ):
            if not self.filing_type:
                raise ValueError(
                    "filing_type is required for upload_separate. "
                    "Use a real type such as SLP_CIVIL, not 'string' or a test value."
                )
            if catalog_types and self.filing_type not in catalog_types:
                allowed = ", ".join(sorted(catalog_types))
                raise ValueError(
                    f"Unknown filing_type {self.filing_type!r}. Use one of: {allowed}"
                )
        elif self.filing_type and catalog_types and self.filing_type not in catalog_types:
            allowed = ", ".join(sorted(catalog_types))
            raise ValueError(
                f"Unknown filing_type {self.filing_type!r}. Use one of: {allowed}"
            )
        return self


class CreateScrutinyRequest(BaseModel):
    """Optional locators. The path already has agent_data_id; both fields may be sent."""

    model_config = ConfigDict(extra="ignore")

    file_hash: str | None = Field(
        default=None,
        examples=["8adf76ba0ba44d4561ab5a4ad88e3d6e97e56a32010ac8c08f39b5f8c1d01340"],
    )
    file_url: str | None = Field(
        default=None,
        validation_alias=AliasChoices("file_url", "download_url"),
        examples=["https://storage.example/filings/Defect_SLP_Civil.pdf"],
    )

    @field_validator("file_hash", "file_url", mode="before")
    @classmethod
    def _drop_swagger_placeholders(cls, value: object) -> str | None:
        return blank_or_placeholder(value)


class JobAccepted(BaseModel):
    job_id: str
    status: Literal["accepted"] = "accepted"
    poll_url: str


class JobRecordOut(BaseModel):
    job_id: str
    kind: JobKind
    status: JobStatus
    organization_id: str | None = None
    workspace_id: str | None = None
    user_id: str | None = None
    agent_data_id: str | None = None
    result: dict[str, Any] | None = None
    error: str | None = None
    created_at: str
    completed_at: str | None = None


class JobState:
    def __init__(
        self,
        *,
        job_id: str,
        kind: JobKind,
        organization_id: str | None = None,
        workspace_id: str | None = None,
        user_id: str | None = None,
    ) -> None:
        self.job_id = job_id
        self.kind = kind
        self.status: JobStatus = "running"
        self.organization_id = organization_id
        self.workspace_id = workspace_id
        self.user_id = user_id
        self.agent_data_id: str | None = None
        self.result: dict[str, Any] | None = None
        self.error: str | None = None
        self.created_at = datetime.now(UTC).isoformat()
        self.completed_at: str | None = None

    def as_out(self) -> JobRecordOut:
        return JobRecordOut(
            job_id=self.job_id,
            kind=self.kind,
            status=self.status,
            organization_id=self.organization_id,
            workspace_id=self.workspace_id,
            user_id=self.user_id,
            agent_data_id=self.agent_data_id,
            result=self.result,
            error=self.error,
            created_at=self.created_at,
            completed_at=self.completed_at,
        )


JOBS: dict[str, JobState] = {}


def _serialize_result(result: Any) -> dict[str, Any]:
    if hasattr(result, "model_dump"):
        payload = result.model_dump(mode="json")
        payload["type"] = type(result).__name__
        return payload
    if isinstance(result, dict):
        return result
    return {"result": result}


def _agent_data_id_from(payload: dict[str, Any]) -> str | None:
    report = payload.get("report")
    if isinstance(report, dict) and report.get("agent_data_id"):
        return str(report["agent_data_id"])
    for key in ("agent_data_id", "result"):
        value = payload.get(key)
        if isinstance(value, str) and value.startswith("agd-"):
            return value
        if isinstance(value, str) and key == "agent_data_id" and value:
            return value
    return None


async def _run_workflow(job: JobState, handler: Any) -> None:
    try:
        result = await handler
        payload = _serialize_result(result)
        job.result = payload
        job.agent_data_id = _agent_data_id_from(payload)
        job.status = "completed"
    except Exception as exc:
        logger.exception("Job %s failed", job.job_id)
        job.status = "failed"
        job.error = str(exc)
    finally:
        job.completed_at = datetime.now(UTC).isoformat()


async def _start_process_file(job: JobState, event: FileEvent) -> None:
    handler = process_file_workflow.run(start_event=event)
    await _run_workflow(job, handler)


async def _start_scrutiny(job: JobState, event: ScrutinyEvent) -> None:
    handler = scrutiny_workflow.run(start_event=event)
    await _run_workflow(job, handler)


def require_api_key(
    authorization: str | None = Header(default=None),
    x_api_key: str | None = Header(default=None, alias="X-API-Key"),
) -> None:
    expected = (os.getenv("JUBEEX_API_KEY") or "").strip()
    if not expected:
        return
    token = (x_api_key or "").strip()
    if authorization and authorization.lower().startswith("bearer "):
        token = authorization[7:].strip()
    if token != expected:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid or missing API key",
        )


def _cors_origins() -> list[str]:
    raw = (os.getenv("JUBEEX_CORS_ORIGINS") or "*").strip()
    if raw == "*":
        return ["*"]
    return [item.strip() for item in raw.split(",") if item.strip()]


app = FastAPI(
    title="JubeeX Scrutiny API",
    version="1.0.0",
    description=(
        "Production endpoints for the JubeeX backend. "
        "Start a filing, poll the job, fetch Agent Data, then run scrutiny."
    ),
)
app.add_middleware(
    CORSMiddleware,
    allow_origins=_cors_origins(),
    allow_credentials=False,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get("/v1/health", tags=["health"])
async def health() -> dict[str, str]:
    return {"status": "ok"}


@app.get("/v1/catalog", tags=["catalog"], dependencies=[Depends(require_api_key)])
async def catalog() -> dict[str, Any]:
    return {
        "filing_types": list(JUBEEX_FILING_TYPES),
        "split_upload_types": ui_catalog(),
        "collection": FILING_COLLECTION,
        "job_types": ["upload_compiled", "upload_separate"],
    }


@app.post(
    "/v1/filings",
    response_model=JobAccepted,
    status_code=status.HTTP_202_ACCEPTED,
    tags=["filings"],
    dependencies=[Depends(require_api_key)],
)
async def create_filing(
    body: CreateFilingRequest,
    background_tasks: BackgroundTasks,
) -> JobAccepted:
    payload = body.model_dump(exclude_none=True)
    try:
        event = FileEvent(**payload)
    except (ValidationError, ValueError) as exc:
        if isinstance(exc, ValidationError):
            detail = [
                {"loc": list(err.get("loc") or ()), "msg": err.get("msg")}
                for err in exc.errors()
            ]
        else:
            detail = str(exc)
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            detail=detail,
        ) from exc

    if intake_mode(event) == "split" and event.filing_type:
        catalog = type_catalog(event.filing_type)
        seen = {(item.slot_id or "").strip() for item in event.documents}
        missing = [
            slot.label
            for slot in catalog.slots
            if slot.required and slot.id not in seen
        ]
        if missing:
            hint = ""
            if len(event.documents) == 1:
                hint = (
                    " You sent one file. If it is a compiled petition PDF, "
                    "use job_type upload_compiled, not upload_separate."
                )
            raise HTTPException(
                status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
                detail="Missing required documents: "
                + ", ".join(missing)
                + "."
                + hint,
            )

    job_id = str(uuid.uuid4())
    job = JobState(
        job_id=job_id,
        kind="process_file",
        organization_id=event.organization_id,
        workspace_id=event.workspace_id,
        user_id=event.user_id,
    )
    JOBS[job_id] = job
    background_tasks.add_task(_start_process_file, job, event)
    return JobAccepted(job_id=job_id, poll_url=f"/v1/jobs/{job_id}")


@app.get(
    "/v1/jobs/{job_id}",
    response_model=JobRecordOut,
    tags=["jobs"],
    dependencies=[Depends(require_api_key)],
)
async def get_job(job_id: str) -> JobRecordOut:
    job = JOBS.get(job_id)
    if job is None:
        raise HTTPException(status_code=404, detail="Unknown job_id")
    return job.as_out()


@app.get(
    "/v1/filings/{agent_data_id}",
    tags=["filings"],
    dependencies=[Depends(require_api_key)],
)
async def get_filing(agent_data_id: str) -> dict[str, Any]:
    client = get_llama_cloud_client()
    try:
        item = await client.beta.agent_data.get(agent_data_id)
    except Exception as exc:
        logger.exception("Failed to load Agent Data %s", agent_data_id)
        raise HTTPException(
            status_code=404,
            detail=f"Filing not found: {agent_data_id}",
        ) from exc
    data = getattr(item, "data", None)
    if hasattr(data, "model_dump"):
        data = data.model_dump(mode="json")
    return {
        "id": getattr(item, "id", agent_data_id),
        "collection": FILING_COLLECTION,
        "data": data,
    }


@app.post(
    "/v1/filings/{agent_data_id}/scrutiny",
    response_model=JobAccepted,
    status_code=status.HTTP_202_ACCEPTED,
    tags=["scrutiny"],
    dependencies=[Depends(require_api_key)],
)
async def create_scrutiny(
    agent_data_id: str,
    background_tasks: BackgroundTasks,
    body: CreateScrutinyRequest | None = None,
) -> JobAccepted:
    event = ScrutinyEvent(
        agent_data_id=agent_data_id,
        file_hash=(body.file_hash if body else None),
        file_url=(body.file_url if body else None),
    )
    job_id = str(uuid.uuid4())
    job = JobState(job_id=job_id, kind="scrutiny")
    JOBS[job_id] = job
    background_tasks.add_task(_start_scrutiny, job, event)
    return JobAccepted(job_id=job_id, poll_url=f"/v1/jobs/{job_id}")


def main() -> None:
    import uvicorn

    uvicorn.run(
        "extraction_review.api:app",
        host=os.getenv("JUBEEX_API_HOST", "0.0.0.0"),
        port=int(os.getenv("JUBEEX_API_PORT", "8000")),
        reload=False,
    )


__all__ = ["JOBS", "app", "main"]
