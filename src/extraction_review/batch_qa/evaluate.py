"""Ground-truth loading and deterministic result comparison."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any


def load_case_manifest(path: Path | None) -> dict[str, dict[str, Any]]:
    if path is None:
        return {}
    payload = json.loads(path.read_text(encoding="utf-8"))
    files = payload.get("files", payload) if isinstance(payload, dict) else None
    if not isinstance(files, dict):
        raise TypeError("Expected manifest must be an object or contain a files object")
    return {
        str(name): value for name, value in files.items() if isinstance(value, dict)
    }


def normalized_ranges(value: Any) -> list[tuple[int, int]]:
    if isinstance(value, dict) and "pages" in value:
        value = value["pages"]
    if not isinstance(value, list):
        return []
    ranges: list[tuple[int, int]] = []
    for item in value:
        if isinstance(item, int):
            ranges.append((item, item))
        elif isinstance(item, dict):
            start = item.get("start")
            end = item.get("end", start)
            if isinstance(start, int) and isinstance(end, int):
                ranges.append((start, end))
        elif (
            isinstance(item, list)
            and len(item) == 2
            and all(isinstance(number, int) for number in item)
        ):
            ranges.append((item[0], item[1]))
    return sorted(ranges)


def split_map(parts: list[dict[str, Any]]) -> dict[str, list[tuple[int, int]]]:
    result: dict[str, list[tuple[int, int]]] = {}
    for part in parts:
        slot = str(part.get("slot_id") or part.get("label") or "").strip()
        if slot:
            result.setdefault(slot, []).extend(normalized_ranges(part.get("page_span")))
    return {slot: sorted(ranges) for slot, ranges in result.items()}


def evaluate_split(parts: list[dict[str, Any]], expected: Any) -> dict[str, Any]:
    if not isinstance(expected, dict):
        return {"status": "not_reviewed", "differences": []}
    actual_map = split_map(parts)
    expected_map = {
        str(slot): normalized_ranges(value) for slot, value in expected.items()
    }
    differences: list[dict[str, Any]] = []
    for slot in sorted(set(actual_map) | set(expected_map)):
        actual = actual_map.get(slot, [])
        wanted = expected_map.get(slot, [])
        if actual != wanted:
            differences.append({"slot_id": slot, "expected": wanted, "actual": actual})
    return {
        "status": "passed" if not differences else "failed",
        "differences": differences,
    }


def value_at_path(payload: Any, path: str) -> Any:
    current = payload
    for component in path.split("."):
        if not isinstance(current, dict) or component not in current:
            return None
        current = current[component]
    return current


def evaluate_extraction(payload: dict[str, Any], expected: Any) -> dict[str, Any]:
    if not isinstance(expected, dict):
        return {"status": "not_reviewed", "differences": []}
    differences = []
    for path, wanted in expected.items():
        actual = value_at_path(payload, str(path))
        if actual != wanted:
            differences.append({"path": path, "expected": wanted, "actual": actual})
    return {
        "status": "passed" if not differences else "failed",
        "differences": differences,
    }


def finding_code(finding: dict[str, Any]) -> str:
    return str(
        finding.get("check_id")
        or finding.get("defect_code")
        or finding.get("code")
        or finding.get("id")
        or ""
    ).strip()


def finding_status(finding: dict[str, Any]) -> str:
    return (
        str(
            finding.get("status")
            or finding.get("outcome")
            or finding.get("result")
            or "unknown"
        )
        .strip()
        .lower()
    )


def report_findings(report: dict[str, Any]) -> list[dict[str, Any]]:
    findings = report.get("findings") or report.get("defects") or []
    return [item for item in findings if isinstance(item, dict)]


def evaluate_scrutiny(report: dict[str, Any], expected: Any) -> dict[str, Any]:
    if not isinstance(expected, dict):
        return {"status": "not_reviewed", "differences": []}
    checks = expected.get("checks")
    expected_checks = checks if isinstance(checks, dict) else expected
    expected_checks = {
        str(code): status
        for code, status in expected_checks.items()
        if not str(code).startswith("__") and code != "strict"
    }
    strict = bool(expected.get("strict") or expected.get("__strict__"))
    actual = {
        finding_code(item): finding_status(item)
        for item in report_findings(report)
        if finding_code(item)
    }
    differences = []
    compared_codes = set(expected_checks)
    if strict:
        compared_codes.update(actual)
    for code in sorted(compared_codes):
        wanted = str(expected_checks.get(code, "missing")).strip().lower()
        observed = actual.get(code, "missing")
        if wanted != observed:
            differences.append(
                {"defect_code": code, "expected": wanted, "actual": observed}
            )
    return {
        "status": "passed" if not differences else "failed",
        "differences": differences,
    }
