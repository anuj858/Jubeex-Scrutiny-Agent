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


def dump_visual_index(payload: Mapping[str, Any] | None) -> dict[str, Any]:
    blob = dict(payload or empty_visual_index())
    blob.setdefault("schema", VISUAL_SCHEMA)
    blob.setdefault("prompt_version", PROMPT_VERSION)
    blob.setdefault("error", None)
    blob.setdefault("pages", [])
    blob.setdefault("targets", [])
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
    return blob


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
