"""SQS enqueue/dequeue for Agent ingestion and scrutiny jobs."""

from __future__ import annotations

import json
import logging
import os
from typing import Any

logger = logging.getLogger(__name__)

INGESTION_QUEUE_NAME = "jubeex-ingestion-jobs"
SCRUTINY_QUEUE_NAME = "jubeex-scrutiny-jobs"
DEFAULT_VISIBILITY_TIMEOUT = 3600


def sqs_enabled() -> bool:
    return bool(
        (os.getenv("JUBEEX_SQS_INGESTION_QUEUE_URL") or "").strip()
        or (os.getenv("JUBEEX_SQS_SCRUTINY_QUEUE_URL") or "").strip()
        or (os.getenv("JUBEEX_SQS_ENABLED") or "").strip().lower() in {"1", "true", "yes"}
    )


def queue_kind(kind: str) -> str:
    return "scrutiny" if kind == "scrutiny" else "process_file"


def queue_name(kind: str) -> str:
    if queue_kind(kind) == "scrutiny":
        return (os.getenv("JUBEEX_SQS_SCRUTINY_QUEUE_NAME") or SCRUTINY_QUEUE_NAME).strip()
    return (os.getenv("JUBEEX_SQS_INGESTION_QUEUE_NAME") or INGESTION_QUEUE_NAME).strip()


def _visibility_timeout() -> int:
    raw = (os.getenv("JUBEEX_SQS_VISIBILITY_TIMEOUT") or "").strip()
    try:
        return max(60, int(raw)) if raw else DEFAULT_VISIBILITY_TIMEOUT
    except ValueError:
        return DEFAULT_VISIBILITY_TIMEOUT


def _client_kwargs() -> dict[str, str]:
    kwargs = {
        "region_name": os.getenv("AWS_REGION") or os.getenv("AWS_DEFAULT_REGION") or "ap-south-1",
    }
    access_key = (os.getenv("AWS_ACCESS_KEY_ID") or "").strip()
    secret_key = (os.getenv("AWS_SECRET_ACCESS_KEY") or "").strip()
    if access_key and secret_key:
        kwargs["aws_access_key_id"] = access_key
        kwargs["aws_secret_access_key"] = secret_key
        session_token = (os.getenv("AWS_SESSION_TOKEN") or "").strip()
        if session_token:
            kwargs["aws_session_token"] = session_token
    return kwargs


def _client():
    import boto3

    return boto3.client("sqs", **_client_kwargs())


def _queue_url(kind: str) -> str:
    if queue_kind(kind) == "scrutiny":
        explicit = (os.getenv("JUBEEX_SQS_SCRUTINY_QUEUE_URL") or "").strip()
    else:
        explicit = (os.getenv("JUBEEX_SQS_INGESTION_QUEUE_URL") or "").strip()
    if explicit:
        return explicit
    client = _client()
    name = queue_name(kind)
    try:
        return str(client.get_queue_url(QueueName=name)["QueueUrl"])
    except Exception:
        created = client.create_queue(
            QueueName=name,
            Attributes={
                "VisibilityTimeout": str(_visibility_timeout()),
                "MessageRetentionPeriod": "1209600",
                "ReceiveMessageWaitTimeSeconds": "20",
            },
        )
        return str(created["QueueUrl"])


def enqueue_job(message: dict[str, Any]) -> str:
    kind = str(message.get("kind") or "process_file")
    url = _queue_url(kind)
    body = json.dumps(message, separators=(",", ":"), ensure_ascii=True)
    sent = _client().send_message(QueueUrl=url, MessageBody=body)
    logger.info("Queued %s job %s on %s", kind, message.get("job_id"), url)
    return str(sent.get("MessageId") or "")


def receive_jobs(kind: str, *, wait_seconds: int = 20, max_messages: int = 1) -> list[dict[str, Any]]:
    url = _queue_url(kind)
    response = _client().receive_message(
        QueueUrl=url,
        MaxNumberOfMessages=max(1, min(max_messages, 10)),
        WaitTimeSeconds=max(0, min(wait_seconds, 20)),
        VisibilityTimeout=_visibility_timeout(),
    )
    items: list[dict[str, Any]] = []
    for raw in response.get("Messages") or []:
        try:
            payload = json.loads(raw.get("Body") or "{}")
        except json.JSONDecodeError:
            logger.warning("Skipping non-JSON SQS message %s", raw.get("MessageId"))
            continue
        if not isinstance(payload, dict):
            continue
        payload["_receipt_handle"] = raw.get("ReceiptHandle")
        payload["_queue_url"] = url
        items.append(payload)
    return items


def delete_job(message: dict[str, Any]) -> None:
    handle = message.get("_receipt_handle")
    url = message.get("_queue_url")
    if not handle or not url:
        return
    _client().delete_message(QueueUrl=str(url), ReceiptHandle=str(handle))
