"""Shared job progress snapshots for GET /v1/jobs across API + SQS worker.

In-memory JOBS only works in one process. When SQS is on, the API and worker
are separate; progress is written to S3 so polling still sees live %.
"""

from __future__ import annotations

import json
import logging
import os
import re
import time
from typing import Any

from .queue import _client_kwargs

logger = logging.getLogger(__name__)

# Monotonic stage → percent. Keywords matched against Status.message (casefold).
_PROCESS_RULES: list[tuple[int, str, tuple[str, ...]]] = [
    (8, "starting", ("starting", "accepted", "queued")),
    (15, "loading", ("download", "loading", "fetch", "uploaded", "reading file")),
    (28, "classify", ("classifying", "classified")),
    (42, "split", ("splitting", "split into", "cutting", "layout index", "page range")),
    (55, "parse", ("parsing", "parsed", "ocr")),
    (70, "extract", ("extracting", "extracted", "llamaextract", "matter")),
    (82, "index", ("indexing", "pinecone", "vector", "upsert")),
    (90, "assemble", ("agent data", "writing", "uploading artifact", "assembling")),
    (96, "finalize", ("usage summary", "job complete", "workflow finished")),
]

_SCRUTINY_RULES: list[tuple[int, str, tuple[str, ...]]] = [
    (12, "loading", ("loading", "fetch", "agent data", "starting", "workflow started")),
    (28, "retrieve", ("retriev", "pinecone", "evidence", "excerpt", "search")),
    # Per-defect Status lines; fine-grained % comes from ScrutinyPartial.
    (40, "checking", ("checking", "defect", "scrutin", "checked ")),
    (78, "reasoning", ("reasoning", "llm", "model", "openrouter")),
    (92, "report", ("report", "writing", "artifact", "usage", "complete")),
]

_last_write_at: dict[str, float] = {}
_WRITE_MIN_INTERVAL_S = 2.0


def progress_for_status(message: str, *, kind: str = "process_file") -> tuple[int, str]:
    """Map a workflow Status message to (percent, stage). Never decreases."""
    text = (message or "").casefold()
    rules = _SCRUTINY_RULES if kind == "scrutiny" else _PROCESS_RULES
    best_pct = 5
    best_stage = "running"
    for pct, stage, keywords in rules:
        if any(k in text for k in keywords):
            if pct >= best_pct:
                best_pct = pct
                best_stage = stage
    return best_pct, best_stage


def job_status_key(job_id: str) -> str:
    safe = re.sub(r"[^a-zA-Z0-9_-]", "-", (job_id or "").strip())[:80] or "unknown"
    return f"agent-jobs/{safe}/status.json"


def _bucket() -> str:
    return (
        (os.getenv("AWS_S3_BUCKET") or "").strip()
        or (os.getenv("JUBEEX_ARTIFACT_BUCKET") or "").strip()
    )


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


def save_job_status(payload: dict[str, Any], *, force: bool = False) -> None:
    """Persist job poll payload. Throttled unless force (accept / terminal)."""
    job_id = str(payload.get("job_id") or "").strip()
    if not job_id:
        return
    bucket = _bucket()
    if not bucket:
        return
    now = time.monotonic()
    last = _last_write_at.get(job_id, 0.0)
    if not force and (now - last) < _WRITE_MIN_INTERVAL_S:
        return
    try:
        client = _s3_client()
        client.put_object(
            Bucket=bucket,
            Key=job_status_key(job_id),
            Body=json.dumps(payload, ensure_ascii=False, default=str).encode("utf-8"),
            ContentType="application/json",
        )
        _last_write_at[job_id] = now
    except Exception:
        logger.warning("Could not persist job status for %s", job_id, exc_info=True)


def load_job_status(job_id: str) -> dict[str, Any] | None:
    bucket = _bucket()
    cleaned = (job_id or "").strip()
    if not bucket or not cleaned:
        return None
    try:
        client = _s3_client()
        response = client.get_object(Bucket=bucket, Key=job_status_key(cleaned))
        body = response["Body"].read()
        data = json.loads(body)
        return data if isinstance(data, dict) else None
    except Exception:
        return None
