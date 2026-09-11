"""Elapsed-time helpers for bundle upload and slot Submit."""

from __future__ import annotations

import time
from pathlib import PurePosixPath
from typing import Any


def start_timer() -> float:
    return time.perf_counter()


def elapsed_seconds(started: float | None) -> float | None:
    if started is None:
        return None
    return round(max(0.0, time.perf_counter() - started), 1)


def uploaded_filename(name: str | None) -> str | None:
    raw = (name or "").strip().replace("\\", "/")
    if not raw:
        return None
    return PurePosixPath(raw).name or None


def format_duration(seconds: float | None) -> str:
    if seconds is None:
        return "n/a"
    total = max(0, int(round(float(seconds))))
    hours, rem = divmod(total, 3600)
    minutes, secs = divmod(rem, 60)
    if hours:
        return f"{hours}h {minutes}m {secs}s"
    if minutes:
        return f"{minutes}m {secs}s"
    if seconds < 10:
        return f"{seconds:.1f}s"
    return f"{total}s"


def timing_payload(
    *,
    file_name: str | None,
    classify_split_seconds: float | None = None,
    parse_extract_seconds: float | None = None,
) -> dict[str, Any]:
    name = uploaded_filename(file_name)
    total = None
    parts = [
        value
        for value in (classify_split_seconds, parse_extract_seconds)
        if value is not None
    ]
    if parts:
        total = round(sum(parts), 1)
    return {
        "file_name": name,
        "classify_split_seconds": classify_split_seconds,
        "parse_extract_seconds": parse_extract_seconds,
        "total_seconds": total,
        "classify_split": format_duration(classify_split_seconds)
        if classify_split_seconds is not None
        else None,
        "parse_extract": format_duration(parse_extract_seconds)
        if parse_extract_seconds is not None
        else None,
        "total": format_duration(total) if total is not None else None,
    }


def attach_timing(payload: dict[str, Any], timing: dict[str, Any]) -> dict[str, Any]:
    """Put duration on the JSON artifact and under metadata so both are visible."""
    stamped = dict(payload)
    stamped["timing"] = timing
    meta = stamped.get("metadata")
    if not isinstance(meta, dict):
        meta = {}
        stamped["metadata"] = meta
    else:
        meta = dict(meta)
        stamped["metadata"] = meta
    meta["timing"] = timing
    return stamped


def timing_status_message(
    *,
    file_name: str | None,
    classify_split_seconds: float | None = None,
    parse_extract_seconds: float | None = None,
) -> str:
    name = uploaded_filename(file_name) or "filing"
    bits: list[str] = []
    if classify_split_seconds is not None:
        bits.append(f"bundle upload {format_duration(classify_split_seconds)}")
    if parse_extract_seconds is not None:
        bits.append(f"slot submit {format_duration(parse_extract_seconds)}")
    payload = timing_payload(
        file_name=name,
        classify_split_seconds=classify_split_seconds,
        parse_extract_seconds=parse_extract_seconds,
    )
    total = payload["total_seconds"]
    if total is not None and len(bits) > 1:
        bits.append(f"total {format_duration(total)}")
    if not bits:
        return f"{name}: timing unavailable"
    return f"{name}: " + " · ".join(bits)
