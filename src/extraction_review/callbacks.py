"""HTTP callback to the Jubeex Backend when an Agent job finishes."""

from __future__ import annotations

import hashlib
import hmac
import json
import logging
import os
import time
from typing import Any
from urllib.parse import urlparse, urlunparse

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


def _is_loopback_url(url: str) -> bool:
    lowered = url.strip().lower()
    return any(
        host in lowered
        for host in ("://localhost", "://127.0.0.1", "://0.0.0.0", "://[::1]")
    )


def resolve_callback_url(callback_url: str | None) -> str:
    """Prefer a public URL. Request localhost must not override ECS env."""
    requested = (callback_url or "").strip()
    configured = (os.getenv("JUBEEX_CALLBACK_URL") or "").strip()
    if requested and not _is_loopback_url(requested):
        return requested
    if configured and not _is_loopback_url(configured):
        if requested and _is_loopback_url(requested):
            logger.warning(
                "Ignoring loopback callback_url=%s; using JUBEEX_CALLBACK_URL=%s",
                requested,
                configured,
            )
        return configured
    return requested or configured


def callback_url_candidates(url: str) -> list[str]:
    """Try the configured URL, then :8000 when the host has no port.

    A common misconfig points at the frontend on :80 while the Jubeex API
    listens on :8000 — that returns 404 for /api/v1/webhooks/ai-agent.
    """
    cleaned = (url or "").strip()
    if not cleaned:
        return []
    candidates = [cleaned]
    parsed = urlparse(cleaned)
    if parsed.scheme in {"http", "https"} and parsed.hostname and parsed.port is None:
        with_port = urlunparse(parsed._replace(netloc=f"{parsed.hostname}:8000"))
        if with_port not in candidates:
            candidates.append(with_port)
    return candidates


async def notify_job_finished(
    *,
    callback_url: str | None,
    job_id: str,
    kind: str,
    status: str,
    agent_data_id: str | None,
    organization_id: str | None,
    workspace_id: str | None = None,
    error: str | None,
    result: dict[str, Any] | None,
    artifacts: dict[str, Any] | None = None,
    event_id: str,
) -> None:
    url = resolve_callback_url(callback_url)
    if not url:
        return
    if _is_loopback_url(url):
        logger.error(
            "Skipping callback for job %s — URL %s is not reachable from this "
            "worker. Set JUBEEX_CALLBACK_URL / callback_url to a public backend "
            "webhook (or rely on backend job polling).",
            job_id,
            url,
        )
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
    artifact_map = artifacts or {}
    payload = {
        "event_id": event_id,
        "event": event,
        "agent_job_id": job_id,
        "task_id": job_id,
        "status": "COMPLETED" if completed else "FAILED",
        "agent_data_id": agent_data_id,
        "organization_id": organization_id,
        "workspace_id": workspace_id,
        "error": error,
        "artifacts": artifact_map,
        "result": {**(result or {}), "artifacts": artifact_map},
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
        headers["X-Agent-Signature"] = sign_callback(
            body, secret=secret, timestamp=timestamp
        )
    last_error: Exception | None = None
    for candidate in callback_url_candidates(url):
        for attempt in range(1, 4):
            try:
                async with httpx.AsyncClient(timeout=30.0) as client:
                    response = await client.post(
                        candidate, content=body, headers=headers
                    )
                    if response.status_code == 404:
                        logger.warning(
                            "Callback 404 for job %s at %s — trying next candidate "
                            "if any (API is often on :8000, not the frontend host)",
                            job_id,
                            candidate,
                        )
                        last_error = httpx.HTTPStatusError(
                            f"404 for {candidate}",
                            request=response.request,
                            response=response,
                        )
                        break  # next candidate
                    response.raise_for_status()
                if candidate != url:
                    logger.info(
                        "Callback for job %s succeeded at fallback URL %s "
                        "(update JUBEEX_CALLBACK_URL / AI_AGENT_CALLBACK_URL)",
                        job_id,
                        candidate,
                    )
                return
            except httpx.HTTPStatusError as exc:
                last_error = exc
                if exc.response is not None and exc.response.status_code == 404:
                    break
                logger.warning(
                    "Callback attempt %s/3 failed for job %s at %s",
                    attempt,
                    job_id,
                    candidate,
                )
            except Exception as exc:
                last_error = exc
                logger.warning(
                    "Callback attempt %s/3 failed for job %s at %s",
                    attempt,
                    job_id,
                    candidate,
                )
    logger.exception(
        "Failed to notify Jubeex Backend at %s for job %s",
        url,
        job_id,
        exc_info=last_error,
    )
