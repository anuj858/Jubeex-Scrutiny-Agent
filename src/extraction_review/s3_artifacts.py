"""Upload Agent step JSON to the same S3 layout as filing PDFs."""

from __future__ import annotations

import json
import logging
import os
import uuid
from contextvars import ContextVar
from datetime import datetime
from pathlib import Path
from typing import Any

from dotenv import load_dotenv

from .queue import _client_kwargs

# llamactl workflows do not import api.py, so they never saw `.env` unless
# the var was also listed in pyproject / deployment secrets.
load_dotenv(Path(__file__).resolve().parents[2] / ".env")

logger = logging.getLogger(__name__)

STEP_SPLIT = "split"
STEP_EXTRACT = "extract"
STEP_LAYOUT = "layout"
STEP_DEFECTS = "defects"
STEP_FOLDERS = {
    STEP_SPLIT: "splitfiles",
    STEP_EXTRACT: "extractedfiles",
    STEP_LAYOUT: "layoutfiles",
    STEP_DEFECTS: "defectfiles",
}
STEP_LABELS = {
    STEP_SPLIT: "agent-split",
    STEP_EXTRACT: "agent-extract",
    STEP_LAYOUT: "agent-layout",
    STEP_DEFECTS: "agent-defects",
}

_job_id: ContextVar[str | None] = ContextVar("artifact_job_id", default=None)
_organization_id: ContextVar[str | None] = ContextVar(
    "artifact_organization_id", default=None
)
_workspace_id: ContextVar[str | None] = ContextVar(
    "artifact_workspace_id", default=None
)
_ARTIFACTS_BY_JOB: dict[str, dict[str, dict[str, str]]] = {}


def _clean_id(value: str | None) -> str | None:
    cleaned = (value or "").strip()
    return cleaned or None


def set_job_context(
    job_id: str | None,
    organization_id: str | None = None,
    workspace_id: str | None = None,
) -> None:
    job = _clean_id(job_id)
    _job_id.set(job)
    _organization_id.set(_clean_id(organization_id))
    _workspace_id.set(_clean_id(workspace_id))
    if job is not None:
        _ARTIFACTS_BY_JOB[job] = {}


def recorded_artifacts(job_id: str | None = None) -> dict[str, dict[str, str]]:
    key = _clean_id(job_id) or _job_id.get()
    if key and key in _ARTIFACTS_BY_JOB:
        return dict(_ARTIFACTS_BY_JOB[key])
    return {}


def artifact_bucket() -> str:
    return (
        (os.getenv("AWS_S3_BUCKET") or "").strip()
        or (os.getenv("JUBEEX_ARTIFACT_BUCKET") or "").strip()
    )


def _safe_segment(value: str, fallback: str = "unknown") -> str:
    cleaned = "".join(
        ch if ch.isalnum() or ch in {"-", "_"} else "-" for ch in (value or "").strip()
    ).strip("-")
    return cleaned[:80] or fallback


def artifact_folder(step: str) -> str:
    return STEP_FOLDERS.get(step) or f"{_safe_segment(step, 'file')}files"


def artifact_filename(
    step: str,
    *,
    object_id: str | None = None,
    job_id: str | None = None,
) -> str:
    label = STEP_LABELS.get(step) or f"agent-{_safe_segment(step, 'file')}"
    stem = _safe_segment(object_id or "", "")
    job = _safe_segment(job_id or "", "")
    if stem and job:
        return f"{stem}-v001-{label}-{job}.json"
    if stem:
        return f"{stem}-v001-{label}.json"
    if job:
        return f"{job}-v001-{label}.json"
    return f"{label}.json"


def artifact_key(
    step: str,
    *,
    organization_id: str | None = None,
    workspace_id: str | None = None,
    job_id: str | None = None,
    object_id: str | None = None,
) -> str:
    org = _safe_segment(organization_id or _organization_id.get() or "", "org")
    workspace = _safe_segment(workspace_id or _workspace_id.get() or "", "workspace")
    return (
        f"org/{org}/filing-workspace/{workspace}/"
        f"{artifact_folder(step)}/"
        f"{artifact_filename(step, object_id=object_id, job_id=job_id)}"
    )


def _json_bytes(payload: Any) -> bytes:
    def default(obj: Any) -> Any:
        if hasattr(obj, "model_dump"):
            return obj.model_dump(mode="json")
        if isinstance(obj, datetime):
            return obj.isoformat()
        if isinstance(obj, uuid.UUID):
            return str(obj)
        return str(obj)

    return json.dumps(
        payload,
        default=default,
        ensure_ascii=False,
        indent=2,
    ).encode("utf-8")


def _s3_client():
    import boto3
    from botocore.config import Config

    kwargs: dict[str, Any] = dict(_client_kwargs())
    region = str(kwargs.get("region_name") or "ap-south-1")
    explicit = (os.getenv("AWS_S3_ENDPOINT_URL") or "").strip()
    kwargs["config"] = Config(
        signature_version="s3v4",
        s3={"addressing_style": "virtual"},
    )
    kwargs["endpoint_url"] = explicit or f"https://s3.{region}.amazonaws.com"
    return boto3.client("s3", **kwargs)


def _url_expires() -> int:
    raw = (os.getenv("JUBEEX_ARTIFACT_URL_EXPIRES") or "").strip()
    try:
        return max(300, int(raw)) if raw else 43200
    except ValueError:
        return 43200


def _id_from_payload(payload: Any, key: str) -> str | None:
    if not isinstance(payload, dict):
        return None
    value = payload.get(key)
    if value:
        return _clean_id(str(value))
    metadata = payload.get("metadata")
    if isinstance(metadata, dict) and metadata.get(key):
        return _clean_id(str(metadata[key]))
    return None


def _first_document_id(payload: Any) -> str | None:
    if not isinstance(payload, dict):
        return None
    buckets = [payload.get("documents"), payload.get("parts")]
    metadata = payload.get("metadata")
    if isinstance(metadata, dict):
        buckets.append(metadata.get("documents"))
    for bucket in buckets:
        if not isinstance(bucket, list):
            continue
        for item in bucket:
            if not isinstance(item, dict):
                continue
            found = _clean_id(str(item.get("document_id") or item.get("id") or ""))
            if found:
                return found
    return None


def upload_step_json(
    step: str,
    payload: Any,
    *,
    organization_id: str | None = None,
    workspace_id: str | None = None,
    job_id: str | None = None,
) -> dict[str, str] | None:
    bucket = artifact_bucket()
    org = (
        _clean_id(organization_id)
        or _organization_id.get()
        or _id_from_payload(payload, "organization_id")
        or _id_from_payload(payload, "org_id")
    )
    workspace = (
        _clean_id(workspace_id)
        or _workspace_id.get()
        or _id_from_payload(payload, "workspace_id")
    )
    job = _clean_id(job_id) or _job_id.get()
    object_id = (
        _id_from_payload(payload, "primary_document_id")
        or _id_from_payload(payload, "document_id")
        or _id_from_payload(payload, "file_id")
        or _first_document_id(payload)
        or job
        or workspace
    )
    if not bucket:
        logger.warning(
            "Skipping %s artifact upload: AWS_S3_BUCKET/JUBEEX_ARTIFACT_BUCKET is not set",
            step,
        )
        return None
    if not org or not workspace:
        logger.warning(
            "Skipping %s artifact upload: organization_id/workspace_id missing",
            step,
        )
        return None
    filename = artifact_filename(step, object_id=object_id, job_id=job)
    key = artifact_key(
        step,
        organization_id=org,
        workspace_id=workspace,
        job_id=job,
        object_id=object_id,
    )
    body = _json_bytes(payload)
    try:
        client = _s3_client()
        client.put_object(
            Bucket=bucket,
            Key=key,
            Body=body,
            ContentType="application/json",
        )
        url = client.generate_presigned_url(
            "get_object",
            Params={
                "Bucket": bucket,
                "Key": key,
                "ResponseContentDisposition": (
                    f'attachment; filename="{filename}"'
                ),
            },
            ExpiresIn=_url_expires(),
        )
    except Exception:
        logger.exception("Failed to upload %s artifact to s3://%s/%s", step, bucket, key)
        return None
    record = {
        "step": step,
        "url": url,
        "key": key,
        "bucket": bucket,
    }
    registry_key = job or f"{org}:{workspace}"
    current = dict(_ARTIFACTS_BY_JOB.get(registry_key, {}))
    current[step] = record
    _ARTIFACTS_BY_JOB[registry_key] = current
    logger.info("Uploaded %s artifact s3://%s/%s", step, bucket, key)
    return record
