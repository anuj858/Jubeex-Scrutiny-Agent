"""Persist and reload visual_index_v1 JSON beside the filing."""

from __future__ import annotations

import logging
from collections.abc import Mapping
from typing import Any

import httpx
from pydantic import ValidationError

from ..process_file import FILE_DOWNLOAD_TIMEOUT_S
from ..s3_artifacts import (
    STEP_VISUAL,
    download_json_object,
    upload_step_json,
)
from ..scrutiny.schema import LlmUsage
from .schema import (
    PROMPT_VERSION,
    VISUAL_SCHEMA,
    VisualMark,
    coerce_visual_status,
    empty_visual_index,
)

logger = logging.getLogger(__name__)

VISUAL_ARTIFACT_URL_KEY = "visual_artifact_url"
VISUAL_ARTIFACT_KEY_KEY = "visual_artifact_key"
VISUAL_SUMMARY_KEY = "visual_summary"


def _coerce_failures(raw: Any) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for item in raw or []:
        if not isinstance(item, Mapping):
            continue
        try:
            page = int(item.get("page"))
        except (TypeError, ValueError):
            continue
        types = [
            str(name).strip()
            for name in (item.get("document_types") or [])
            if str(name).strip()
        ]
        reason = str(item.get("reason") or "").strip() or "vision_failed"
        detail = item.get("detail")
        slot_id = str(item.get("slot_id") or "").strip() or None
        rows.append(
            {
                "page": page,
                "slot_id": slot_id,
                "document_types": types,
                "reason": reason,
                "detail": (str(detail).strip()[:300] if detail else None) or None,
            }
        )
    return rows


def _coerce_usage(raw: Any) -> dict[str, Any] | None:
    if raw is None:
        return None
    try:
        if isinstance(raw, LlmUsage):
            usage = raw
        elif isinstance(raw, Mapping):
            usage = LlmUsage.model_validate(dict(raw))
        else:
            return None
    except ValidationError:
        return None
    if usage.calls <= 0 and usage.cost_usd is None and usage.total_tokens <= 0:
        return None
    return {
        "cost_usd": usage.cost_usd,
        "prompt_tokens": usage.prompt_tokens,
        "completion_tokens": usage.completion_tokens,
        "total_tokens": usage.total_tokens,
        "calls": usage.calls,
        "model": usage.model,
    }


def dump_visual_index(payload: Mapping[str, Any] | None) -> dict[str, Any]:
    blob = dict(payload or empty_visual_index())
    blob.setdefault("schema", VISUAL_SCHEMA)
    blob.setdefault("prompt_version", PROMPT_VERSION)
    blob.setdefault("error", None)
    blob.setdefault("pages", [])
    blob.setdefault("targets", [])
    blob.setdefault("failures", [])
    blob.setdefault("marks", [])
    blob["status"] = coerce_visual_status(str(blob.get("status") or "ok"))
    return blob


def coerce_visual_index(raw: Mapping[str, Any] | None) -> dict[str, Any]:
    blob = dump_visual_index(raw)
    marks: list[dict[str, Any]] = []
    for item in blob.get("marks") or []:
        if not isinstance(item, Mapping):
            continue
        try:
            marks.append(VisualMark.model_validate(dict(item)).model_dump())
        except ValidationError:
            continue
    pages: list[int] = []
    for value in blob.get("pages") or []:
        try:
            pages.append(int(value))
        except (TypeError, ValueError):
            continue
    targets: list[dict[str, Any]] = []
    for item in blob.get("targets") or []:
        if not isinstance(item, Mapping):
            continue
        try:
            page = int(item.get("page"))
        except (TypeError, ValueError):
            continue
        types = [
            str(name).strip()
            for name in (item.get("document_types") or [])
            if str(name).strip()
        ]
        targets.append({"page": page, "document_types": types})
    blob["marks"] = marks
    blob["pages"] = pages
    blob["targets"] = targets
    blob["failures"] = _coerce_failures(blob.get("failures"))
    usage = _coerce_usage(blob.get("usage"))
    if usage:
        blob["usage"] = usage
    else:
        blob.pop("usage", None)
    return blob


def _compact_mark(mark: Mapping[str, Any]) -> dict[str, Any]:
    return {
        "page": mark.get("page"),
        "document_type": mark.get("document_type"),
        "marking_type": mark.get("marking_type"),
        "signature_role": mark.get("signature_role"),
        "bbox": mark.get("bbox"),
        "confidence": mark.get("confidence"),
    }


def visual_summary(payload: Mapping[str, Any] | None) -> dict[str, Any]:
    """Compact Agent Data metadata so the filing shows marks without fetching S3."""
    coerced = coerce_visual_index(payload)
    marks = [
        _compact_mark(mark)
        for mark in (coerced.get("marks") or [])
        if isinstance(mark, Mapping)
    ]
    summary = {
        "status": coerced.get("status"),
        "error": coerced.get("error"),
        "mark_count": len(marks),
        "marks": marks,
        "target_count": len(coerced.get("targets") or []),
        "failures": list(coerced.get("failures") or []),
        "targets": list(coerced.get("targets") or []),
    }
    if coerced.get("usage"):
        summary["usage"] = coerced["usage"]
    return summary


def upload_visual_index(
    payload: Mapping[str, Any] | None,
    *,
    organization_id: str | None = None,
    workspace_id: str | None = None,
    job_id: str | None = None,
) -> dict[str, str] | None:
    dumped = dump_visual_index(payload)
    if organization_id:
        dumped["organization_id"] = organization_id
    if workspace_id:
        dumped["workspace_id"] = workspace_id
    return upload_step_json(
        STEP_VISUAL,
        dumped,
        organization_id=organization_id,
        workspace_id=workspace_id,
        job_id=job_id,
    )


async def load_visual_index(
    url: str | None,
    *,
    key: str | None = None,
) -> dict[str, Any]:
    """Load stored visual marks. Empty on any failure — never re-run vision."""
    payload: Any = None
    cleaned_url = (url or "").strip()
    cleaned_key = (key or "").strip()
    if cleaned_url:
        timeout = httpx.Timeout(FILE_DOWNLOAD_TIMEOUT_S)
        try:
            async with httpx.AsyncClient(
                timeout=timeout, follow_redirects=True
            ) as client:
                response = await client.get(cleaned_url)
                response.raise_for_status()
                payload = response.json()
        except (httpx.HTTPError, ValueError, OSError, TypeError):
            logger.warning(
                "Failed to load visual index from URL; trying S3 key",
                exc_info=True,
            )
            payload = None
    if payload is None and cleaned_key:
        payload = download_json_object(cleaned_key)
    if not isinstance(payload, Mapping):
        return empty_visual_index(status="skipped", error="missing")
    return coerce_visual_index(payload)
