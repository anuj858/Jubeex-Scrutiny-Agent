"""HTTP callback to the Jubeex Backend when an Agent job finishes."""

from __future__ import annotations

import hashlib
import hmac
import json
import logging
import os
import time
from typing import Any

import httpx

logger = logging.getLogger(__name__)


def callback_secret() -> str:
    return (os.getenv("JUBEEX_CALLBACK_SECRET") or "").strip()


def sign_callback(body: bytes, *, secret: str, timestamp: str) -> str:
    return hmac.new(
        secret.encode(),
        timestamp.encode() + b"." + body,
        hashlib.sha256,
    ).hexdigest()


async def notify_job_finished(
    *,
    callback_url: str | None,
    job_id: str,
    kind: str,
    status: str,
    agent_data_id: str | None,
    organization_id: str | None,
    error: str | None,
    result: dict[str, Any] | None,
    event_id: str,
) -> None:
    url = (callback_url or os.getenv("JUBEEX_CALLBACK_URL") or "").strip()
    if not url:
        return
    completed = status == "completed"
    event = (
        "INGESTION_COMPLETED"
        if kind == "process_file" and completed
        else "INGESTION_FAILED"
        if kind == "process_file"
        else "SCRUTINY_COMPLETED"
        if completed
        else "SCRUTINY_FAILED"
    )
    payload = {
        "event_id": event_id,
        "event": event,
        "agent_job_id": job_id,
        "status": "COMPLETED" if completed else "FAILED",
        "agent_data_id": agent_data_id,
        "organization_id": organization_id,
        "error": error,
        "result": result or {},
    }
    body = json.dumps(payload, separators=(",", ":"), ensure_ascii=True).encode()
    timestamp = str(int(time.time()))
    headers = {
        "Content-Type": "application/json",
        "X-Agent-Timestamp": timestamp,
        "X-Agent-Event-Id": event_id,
    }
    secret = callback_secret()
    if secret:
        headers["X-Agent-Signature"] = sign_callback(body, secret=secret, timestamp=timestamp)
    last_error: Exception | None = None
    for attempt in range(1, 4):
        try:
            async with httpx.AsyncClient(timeout=30.0) as client:
                response = await client.post(url, content=body, headers=headers)
                response.raise_for_status()
            return
        except Exception as exc:
            last_error = exc
            logger.warning(
                "Callback attempt %s/3 failed for job %s at %s",
                attempt,
                job_id,
                url,
            )
    logger.exception(
        "Failed to notify Jubeex Backend at %s for job %s",
        url,
        job_id,
        exc_info=last_error,
    )
