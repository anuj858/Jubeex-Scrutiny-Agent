"""Upload Agent step JSON to the same S3 layout as filing PDFs."""

from __future__ import annotations

import json
import logging
import os
import uuid
from contextvars import ContextVar
from datetime import datetime
from typing import Any

from .queue import _client_kwargs

logger = logging.getLogger(__name__)

STEP_SPLIT = "split"
STEP_EXTRACT = "extract"
STEP_LAYOUT = "layout"
STEP_DEFECTS = "defects"

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
    job = _safe_segment(job_id or _job_id.get() or str(uuid.uuid4()), "job")
    token = _safe_segment(object_id or str(uuid.uuid4()), "file")
    folder = f"org/{org}/filing-workspace/{workspace}/"
    if step == STEP_LAYOUT:
        folder = f"{folder}coordinate/"
    return f"{folder}{job}-v001-agent-{_safe_segment(step)}-{token}.json"


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
    if not bucket:
        logger.warning("Skipping %s artifact upload: AWS_S3_BUCKET is not set", step)
        return None
    if not org or not workspace:
        logger.warning(
            "Skipping %s artifact upload: organization_id/workspace_id missing",
            step,
        )
        return None
    key = artifact_key(
        step,
        organization_id=org,
        workspace_id=workspace,
        job_id=job,
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
                    f'attachment; filename="{step}.json"'
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
