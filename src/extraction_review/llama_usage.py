"""Collect LlamaCloud credit usage from finished jobs.

Parse and extract expose billed credits via ``expand=usage``. Classify and
split may not; those rows are still recorded with ``credits=null`` so the job
ids stay on the filing. Credits can be null for a few seconds after a job
completes — retry once, never treat null as zero.
"""

from __future__ import annotations

import asyncio
import logging
from collections.abc import Awaitable, Callable
from typing import Any

from llama_cloud import AsyncLlamaCloud

from .clients import project_id

logger = logging.getLogger(__name__)

USAGE_RETRY_S = 2.0
CREDIT_USD_RATE = 0.00125  # $1.25 per 1,000 credits on paid overage


def _as_mapping(value: Any) -> dict[str, Any] | None:
    if value is None:
        return None
    if isinstance(value, dict):
        return value
    dump = getattr(value, "model_dump", None)
    if callable(dump):
        payload = dump(mode="json")
        return payload if isinstance(payload, dict) else None
    return None


def _float_or_none(value: Any) -> float | None:
    if value is None:
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def usage_fields(payload: Any) -> dict[str, float | None]:
    """Read billed credits from a parse/extract/classify/split job body."""
    data = _as_mapping(payload) or {}
    usage = data.get("usage")
    if usage is None and isinstance(data.get("job"), dict):
        usage = data["job"].get("usage")
    usage_map = usage if isinstance(usage, dict) else {}
    credits = _float_or_none(usage_map.get("credits"))
    if credits is None:
        credits = _float_or_none(data.get("credits"))
        job = data.get("job")
        if credits is None and isinstance(job, dict):
            credits = _float_or_none(job.get("credits"))
    return {
        "credits": credits,
        "extract_credits": _float_or_none(usage_map.get("extract_credits")),
        "parse_credits": _float_or_none(usage_map.get("parse_credits")),
    }


def usage_job(
    *,
    product: str,
    job_id: str | None,
    payload: Any = None,
    slot_id: str | None = None,
    tier: str | None = None,
    pages: int | None = None,
) -> dict[str, Any]:
    fields = usage_fields(payload)
    row: dict[str, Any] = {
        "product": product,
        "job_id": job_id,
        "credits": fields["credits"],
    }
    if slot_id:
        row["slot_id"] = slot_id
    if tier:
        row["tier"] = str(tier)
    if pages is not None:
        row["pages"] = pages
    if fields["extract_credits"] is not None:
        row["extract_credits"] = fields["extract_credits"]
    if fields["parse_credits"] is not None:
        row["parse_credits"] = fields["parse_credits"]
    return row


def summarize_llamacloud_usage(jobs: list[dict[str, Any]]) -> dict[str, Any]:
    billed = [row for row in jobs if row.get("credits") is not None]
    total = sum(float(row["credits"]) for row in billed) if billed else None
    by_product: dict[str, float] = {}
    for row in billed:
        product = str(row.get("product") or "other")
        by_product[product] = by_product.get(product, 0.0) + float(row["credits"])
    estimated = None if total is None else round(total * CREDIT_USD_RATE, 6)
    return {
        "credits": total,
        "estimated_usd": estimated,
        "credit_usd_rate": CREDIT_USD_RATE,
        "pending_jobs": sum(1 for row in jobs if row.get("credits") is None),
        "by_product": {key: by_product[key] for key in sorted(by_product)},
        "jobs": jobs,
        "note": (
            "LlamaCloud credits from expand=usage on each finished job. "
            "Null credits mean billing had not recorded that job yet. "
            "estimated_usd uses $1.25 per 1,000 credits."
        ),
    }


def has_usage_object(payload: Any) -> bool:
    data = _as_mapping(payload) or {}
    if isinstance(data.get("usage"), dict):
        return True
    job = data.get("job")
    return isinstance(job, dict) and isinstance(job.get("usage"), dict)


def parse_job_tier(payload: Any) -> str | None:
    data = _as_mapping(payload) or {}
    job = data.get("job")
    if isinstance(job, dict) and job.get("tier"):
        return str(job["tier"])
    return None


async def _retry_if_pending(
    row: dict[str, Any],
    refetch: Callable[[], Awaitable[dict[str, Any]]],
) -> dict[str, Any]:
    if row.get("credits") is not None:
        return row
    await asyncio.sleep(USAGE_RETRY_S)
    try:
        again = await refetch()
    except Exception:
        logger.warning(
            "[Usage] Retry failed for %s %s",
            row.get("product"),
            row.get("job_id"),
        )
        return row
    return again if again.get("credits") is not None else row


async def collect_parse_usage(
    client: AsyncLlamaCloud,
    job_id: str,
    *,
    payload: Any = None,
    slot_id: str | None = None,
    pages: int | None = None,
) -> dict[str, Any]:
    def _row(body: Any) -> dict[str, Any]:
        return usage_job(
            product="parse",
            job_id=job_id,
            payload=body,
            slot_id=slot_id,
            tier=parse_job_tier(body),
            pages=pages,
        )

    row = _row(payload)
    return await _retry_if_pending(
        row,
        lambda: _fetch_parse_usage(client, job_id, slot_id=slot_id, pages=pages),
    )


async def _fetch_parse_usage(
    client: AsyncLlamaCloud,
    job_id: str,
    *,
    slot_id: str | None,
    pages: int | None,
) -> dict[str, Any]:
    body = await client.parsing.get(
        job_id,
        expand=["usage"],
        project_id=project_id,
    )
    return usage_job(
        product="parse",
        job_id=job_id,
        payload=body,
        slot_id=slot_id,
        tier=parse_job_tier(body),
        pages=pages,
    )


async def collect_extract_usage(
    client: AsyncLlamaCloud,
    job_id: str,
    *,
    payload: Any = None,
) -> dict[str, Any]:
    row = usage_job(product="extract", job_id=job_id, payload=payload)
    return await _retry_if_pending(
        row,
        lambda: _fetch_extract_usage(client, job_id),
    )


async def _fetch_extract_usage(
    client: AsyncLlamaCloud, job_id: str
) -> dict[str, Any]:
    body = await client.extract.get(
        job_id,
        expand=["usage"],
        project_id=project_id,
    )
    return usage_job(product="extract", job_id=job_id, payload=body)


async def collect_optional_usage(
    client: AsyncLlamaCloud,
    *,
    product: str,
    job_id: str,
    payload: Any = None,
) -> dict[str, Any]:
    """Classify/split: record the job even when expand=usage is not supported."""
    row = usage_job(product=product, job_id=job_id, payload=payload)
    if row.get("credits") is not None:
        return row

    getter = getattr(client, product, None)
    if getter is None:
        return row

    async def _refetch() -> dict[str, Any]:
        body = await getter.get(
            job_id,
            project_id=project_id,
            extra_query={"expand": "usage"},
        )
        return usage_job(product=product, job_id=job_id, payload=body)

    if has_usage_object(payload):
        return await _retry_if_pending(row, _refetch)
    try:
        return await _refetch()
    except Exception:
        logger.info(
            "[Usage] %s %s has no billed credits on the job payload",
            product,
            job_id,
        )
        return row


def usage_status_message(summary: dict[str, Any]) -> str:
    credits = summary.get("credits")
    if credits is None:
        pending = summary.get("pending_jobs") or 0
        return f"LlamaCloud usage pending ({pending} job(s) not billed yet)"
    usd = summary.get("estimated_usd")
    usd_bit = f" (~${usd:.4f})" if usd is not None else ""
    parts = summary.get("by_product") or {}
    detail = ", ".join(f"{name} {value:g}" for name, value in parts.items())
    extra = f" [{detail}]" if detail else ""
    return f"LlamaCloud {credits:g} credits{usd_bit}{extra}"
