"""Assemble the stored compiled-petition record after LlamaExtract."""

from __future__ import annotations

from datetime import datetime
from typing import Any
from zoneinfo import ZoneInfo

from .document_parts import (
    FILING_TYPE_LABELS,
    document_spans_from_page_parts,
    documents_from_page_parts,
)
from .scrutiny.rules import normalize_filing_type

SCHEMA_VERSION = "extraction-v1"
JOB_TYPE = "compiled_petition"
LEGAL_EXTRACT_FIELDS = (
    "court",
    "petition_type",
    "cause_title",
    "petitioners",
    "respondents",
    "advocates_on_record",
    "classification",
    "impugned_orders",
    "relief",
    "applications",
    "inconsistencies",
)


def _as_dict(value: Any) -> dict[str, Any]:
    if isinstance(value, dict):
        return value
    dump = getattr(value, "model_dump", None)
    if callable(dump):
        payload = dump()
        return payload if isinstance(payload, dict) else {}
    return {}


def _named(value: Any) -> str | None:
    if isinstance(value, str):
        text = value.strip()
        return text or None
    data = _as_dict(value)
    name = data.get("name")
    if isinstance(name, str) and name.strip():
        return name.strip()
    return None


def _pages_int(value: Any) -> list[int]:
    pages: list[int] = []
    if value is None:
        return pages
    if isinstance(value, (int, float)):
        pages.append(int(value))
        return pages
    if isinstance(value, str):
        for bit in value.replace(";", ",").split(","):
            bit = bit.strip()
            if not bit:
                continue
            try:
                pages.append(int(bit))
            except ValueError:
                continue
        return pages
    if isinstance(value, (list, tuple)):
        for item in value:
            pages.extend(_pages_int(item))
    return pages


def _walk_for_confidence(value: Any) -> list[float]:
    scores: list[float] = []
    if isinstance(value, (int, float)) and not isinstance(value, bool):
        number = float(value)
        if 0.0 <= number <= 1.0:
            scores.append(number)
        return scores
    if isinstance(value, dict):
        for key, nested in value.items():
            lowered = str(key).lower()
            if lowered in {"confidence", "score", "overall_confidence"}:
                scores.extend(_walk_for_confidence(nested))
            elif isinstance(nested, (dict, list)):
                scores.extend(_walk_for_confidence(nested))
        return scores
    if isinstance(value, list):
        for item in value:
            scores.extend(_walk_for_confidence(item))
    return scores


def overall_confidence_from_job(job: Any) -> float | None:
    dumped = _as_dict(job)
    for key in ("extract_metadata", "extraction_metadata", "metadata"):
        blob = dumped.get(key)
        scores = _walk_for_confidence(blob)
        if scores:
            return round(sum(scores) / len(scores), 3)
    extra = getattr(job, "extract_metadata", None) or getattr(
        job, "extraction_metadata", None
    )
    scores = _walk_for_confidence(_as_dict(extra) or extra)
    if scores:
        return round(sum(scores) / len(scores), 3)
    return None


def field_confidence_from_job(job: Any) -> dict[str, Any]:
    dumped = _as_dict(job)
    for key in ("extract_metadata", "extraction_metadata", "metadata"):
        blob = dumped.get(key)
        if not isinstance(blob, dict):
            continue
        for nested_key in ("confidence", "confidence_scores", "field_confidence"):
            nested = blob.get(nested_key)
            if isinstance(nested, dict) and nested:
                return nested
    return {}


def _petition_type_name(record: dict[str, Any], filing_type: str | None) -> str | None:
    named = _named(record.get("petition_type"))
    if named:
        return named
    key = normalize_filing_type(filing_type)
    if key in FILING_TYPE_LABELS:
        return FILING_TYPE_LABELS[key]
    raw = (filing_type or "").strip()
    return raw or None


def apply_extract_envelope(
    record: dict[str, Any] | None,
    *,
    page_parts: dict[int, Any] | None = None,
    filing_type: str | None = None,
    overall_confidence: float | None = None,
    generated_at: str | None = None,
) -> dict[str, Any]:
    """Add envelope keys, stitch documents, and derived filing_summary fields."""
    payload = dict(record or {})
    spans = document_spans_from_page_parts(page_parts or {})
    listing = documents_from_page_parts(page_parts or {})
    cause = _as_dict(payload.get("cause_title"))
    petition_name = _petition_type_name(payload, filing_type)
    if petition_name:
        payload["petition_type"] = {"name": petition_name}
    court_name = _named(payload.get("court"))
    if court_name:
        payload["court"] = {"name": court_name}

    summary = _as_dict(payload.get("filing_summary"))
    summary["matter_title"] = (
        cause.get("title") or cause.get("formatted_title") or summary.get("matter_title")
    )
    summary["matter_type"] = petition_name or summary.get("matter_type")
    if listing.get("items"):
        summary["documents"] = listing
    payload["filing_summary"] = summary or None

    payload["schema_version"] = SCHEMA_VERSION
    payload["job_type"] = JOB_TYPE
    payload["organization_id"] = None
    payload["workspace_id"] = None
    payload["user_id"] = None
    payload["primary_document_id"] = None
    payload["documents"] = spans
    payload["document_counts"] = {
        "processed": len(spans),
        "failed": 0,
    }
    payload["overall_confidence"] = overall_confidence
    if generated_at is None:
        generated_at = datetime.now(ZoneInfo("Asia/Kolkata")).isoformat(
            timespec="seconds"
        )
    payload["generated_at"] = generated_at
    return payload


def unwrap_extracted_record(extracted_data: Any) -> dict[str, Any]:
    """Agent Data stores ExtractedData; legal fields live on `.data`."""
    dumped = _as_dict(extracted_data)
    inner = dumped.get("data")
    if isinstance(inner, dict):
        return inner
    return dumped


def stamp_source_pages(record: dict[str, Any]) -> None:
    """Coerce source_pages on nested objects to integer lists."""

    def visit(node: Any) -> None:
        if isinstance(node, dict):
            if "source_pages" in node:
                node["source_pages"] = _pages_int(node.get("source_pages"))
            for value in node.values():
                visit(value)
        elif isinstance(node, list):
            for item in node:
                visit(item)

    visit(record)
