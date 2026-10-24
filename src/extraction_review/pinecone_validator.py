"""Read-only validation of Pinecone records belonging to one workspace."""

from __future__ import annotations

import argparse
import importlib
import json
import sys
from collections import Counter, defaultdict
from collections.abc import Iterable
from typing import Any
from uuid import UUID

from .config import load_repo_dotenv

# vector_store reads its settings at import time, so load the repo .env first.
load_repo_dotenv()

store = importlib.import_module(".vector_store", __package__)


REQUIRED_COMMON_FIELDS = (
    "workspace_id",
    "organization_id",
    "file_hash",
    "index_generation",
    "chunk_kind",
    "vector_schema_version",
)
REQUIRED_PAGE_FIELDS = (
    "page_start",
    "page_end",
    "primary_page",
    "source_pages",
    "chunk_index",
    "ocr_status",
)


def _as_dict(value: Any) -> dict[str, Any]:
    if isinstance(value, dict):
        return value
    if hasattr(value, "to_dict"):
        result = value.to_dict()
        return result if isinstance(result, dict) else {}
    if hasattr(value, "model_dump"):
        result = value.model_dump(mode="json")
        return result if isinstance(result, dict) else {}
    return {}


def _listed_ids(page: Any) -> list[str]:
    data = _as_dict(page)
    vectors = data.get("vectors") or data.get("records") or []
    ids: list[str] = []
    for item in vectors:
        if isinstance(item, str):
            ids.append(item)
            continue
        row = _as_dict(item)
        record_id = row.get("id") or row.get("_id")
        if record_id:
            ids.append(str(record_id))
    return ids


def _next_token(page: Any) -> str | None:
    data = _as_dict(page)
    pagination = _as_dict(data.get("pagination"))
    token = pagination.get("next") or pagination.get("next_token")
    return str(token) if token else None


def _iter_namespace_ids(index: Any, namespace: str, *, page_size: int) -> Iterable[str]:
    """Yield every record id, supporting Pinecone SDK v7 response shapes."""
    token: str | None = None
    while True:
        kwargs: dict[str, Any] = {"namespace": namespace, "limit": page_size}
        if token:
            kwargs["pagination_token"] = token
        page = index.list_paginated(**kwargs)
        yield from _listed_ids(page)
        token = _next_token(page)
        if not token:
            return


def _record_fields(record: Any) -> dict[str, Any]:
    return store._record_fields(record)


def _issue(
    issues: list[dict[str, Any]],
    severity: str,
    code: str,
    message: str,
    *,
    record_id: str | None = None,
) -> None:
    row: dict[str, Any] = {
        "severity": severity,
        "code": code,
        "message": message,
    }
    if record_id:
        row["record_id"] = record_id
    issues.append(row)


def _validate_uuid(workspace_id: str) -> str:
    value = (workspace_id or "").strip()
    if not value:
        raise ValueError("workspace_id is required")
    try:
        UUID(value)
    except ValueError as exc:
        raise ValueError("workspace_id must be a UUID") from exc
    return value


def _validate_record(
    record_id: str,
    fields: dict[str, Any],
    workspace_id: str,
    text_field: str,
    issues: list[dict[str, Any]],
) -> None:
    for field in REQUIRED_COMMON_FIELDS:
        if fields.get(field) in (None, "", []):
            _issue(
                issues,
                "error",
                "missing_required_field",
                f"Required field {field!r} is missing",
                record_id=record_id,
            )

    if fields.get("workspace_id") != workspace_id:
        _issue(
            issues,
            "error",
            "workspace_mismatch",
            "Record workspace_id does not match the requested workspace",
            record_id=record_id,
        )

    text = store._text_from_fields(fields, text_field)
    if not text:
        _issue(
            issues,
            "error",
            "missing_text",
            f"No non-empty text found in {text_field!r}",
            record_id=record_id,
        )

    kind = fields.get("chunk_kind")
    if kind not in {"summary", "page", "page_manifest"}:
        _issue(
            issues,
            "error",
            "invalid_chunk_kind",
            f"Unsupported chunk_kind {kind!r}",
            record_id=record_id,
        )
    if kind in {"page", "page_manifest"}:
        for field in REQUIRED_PAGE_FIELDS:
            if fields.get(field) in (None, "", []):
                _issue(
                    issues,
                    "error",
                    "missing_page_field",
                    f"Page record is missing {field!r}",
                    record_id=record_id,
                )

    schema_version = fields.get("vector_schema_version")
    if schema_version is not None:
        try:
            schema_version = int(schema_version)
        except (TypeError, ValueError):
            schema_version = None
        if schema_version != store.VECTOR_SCHEMA_VERSION:
            _issue(
                issues,
                "error",
                "schema_version_mismatch",
                f"Expected vector schema {store.VECTOR_SCHEMA_VERSION}, got {fields.get('vector_schema_version')!r}",
                record_id=record_id,
            )

    identity = {
        key: str(fields.get(key) or "").strip()
        for key in ("file_hash", "organization_id", "index_generation")
    }
    if all(identity.values()):
        expected_base = store.scoped_record_base_id(
            file_hash=identity["file_hash"],
            organization_id=identity["organization_id"],
            workspace_id=workspace_id,
            index_generation=identity["index_generation"],
        )
        if not record_id.startswith(f"{expected_base}:"):
            _issue(
                issues,
                "error",
                "record_id_scope_mismatch",
                "Record ID is not derived from its tenant and generation metadata",
                record_id=record_id,
            )


def validate_workspace(
    workspace_id: str,
    *,
    page_size: int = 100,
    fetch_batch_size: int = 100,
    max_scan: int = 250_000,
) -> dict[str, Any]:
    """Validate all records for ``workspace_id`` without mutating Pinecone."""
    workspace_id = _validate_uuid(workspace_id)
    if page_size < 1 or page_size > 100:
        raise ValueError("page_size must be between 1 and 100")
    if fetch_batch_size < 1 or fetch_batch_size > 100:
        raise ValueError("fetch_batch_size must be between 1 and 100")
    if max_scan < 1:
        raise ValueError("max_scan must be positive")

    client = store.get_pinecone_client()
    if not client.has_index(store.pinecone_index_name):
        raise RuntimeError(
            f"Pinecone index {store.pinecone_index_name!r} does not exist"
        )
    description = client.describe_index(store.pinecone_index_name)
    text_field = store.validate_index_contract(description)
    index = client.Index(store.pinecone_index_name)

    issues: list[dict[str, Any]] = []
    records: list[tuple[str, dict[str, Any]]] = []
    scanned = 0
    batch: list[str] = []

    def fetch_current_batch() -> None:
        if not batch:
            return
        result = index.fetch(ids=list(batch), namespace=store.pinecone_namespace)
        found = store._vectors_from_fetch(result)
        for record_id in batch:
            record = found.get(record_id)
            if record is None:
                _issue(
                    issues,
                    "error",
                    "listed_record_not_fetched",
                    "Record was listed but could not be fetched",
                    record_id=record_id,
                )
                continue
            fields = _record_fields(record)
            if fields.get("workspace_id") == workspace_id:
                records.append((record_id, fields))
        batch.clear()

    for record_id in _iter_namespace_ids(
        index, store.pinecone_namespace, page_size=page_size
    ):
        scanned += 1
        if scanned > max_scan:
            raise RuntimeError(
                f"Namespace scan exceeded --max-scan={max_scan}; rerun with a higher limit"
            )
        batch.append(record_id)
        if len(batch) >= fetch_batch_size:
            fetch_current_batch()
    fetch_current_batch()

    for record_id, fields in records:
        _validate_record(record_id, fields, workspace_id, text_field, issues)

    by_generation: dict[str, list[tuple[str, dict[str, Any]]]] = defaultdict(list)
    for record in records:
        generation = str(record[1].get("index_generation") or "<missing>")
        by_generation[generation].append(record)

    for generation, generation_records in by_generation.items():
        kinds = Counter(
            str(fields.get("chunk_kind") or "<missing>")
            for _, fields in generation_records
        )
        if kinds["summary"] == 0:
            _issue(
                issues,
                "warning",
                "missing_summary",
                f"Generation {generation} has no summary record",
            )
        elif kinds["summary"] > 1:
            _issue(
                issues,
                "error",
                "duplicate_summary",
                f"Generation {generation} has {kinds['summary']} summary records",
            )

        logical_keys: Counter[tuple[Any, ...]] = Counter()
        for _, fields in generation_records:
            if fields.get("chunk_kind") in {"page", "page_manifest"}:
                logical_keys[
                    (
                        fields.get("chunk_kind"),
                        fields.get("primary_page", fields.get("page_start")),
                        fields.get("chunk_index"),
                    )
                ] += 1
        for key, count in logical_keys.items():
            if count > 1:
                _issue(
                    issues,
                    "error",
                    "duplicate_logical_chunk",
                    f"Generation {generation} contains {count} records for chunk {key}",
                )

    if not records:
        _issue(
            issues,
            "error",
            "workspace_not_found",
            "No records were found for this workspace in the configured namespace",
        )
    if len(by_generation) > 1:
        _issue(
            issues,
            "warning",
            "multiple_generations",
            f"Workspace contains {len(by_generation)} vector generations",
        )

    errors = sum(issue["severity"] == "error" for issue in issues)
    warnings = sum(issue["severity"] == "warning" for issue in issues)
    kind_counts = Counter(
        str(fields.get("chunk_kind") or "<missing>") for _, fields in records
    )
    generation_summary = {
        generation: {
            "records": len(generation_records),
            "chunk_kinds": dict(
                sorted(
                    Counter(
                        str(fields.get("chunk_kind") or "<missing>")
                        for _, fields in generation_records
                    ).items()
                )
            ),
            "file_hashes": sorted(
                {
                    str(fields.get("file_hash"))
                    for _, fields in generation_records
                    if fields.get("file_hash")
                }
            ),
        }
        for generation, generation_records in sorted(by_generation.items())
    }
    return {
        "status": "pass" if errors == 0 else "fail",
        "workspace_id": workspace_id,
        "index": store.pinecone_index_name,
        "namespace": store.pinecone_namespace,
        "text_field": text_field,
        "records_scanned": scanned,
        "workspace_records": len(records),
        "chunk_kinds": dict(sorted(kind_counts.items())),
        "generations": generation_summary,
        "errors": errors,
        "warnings": warnings,
        "issues": issues,
    }


def _print_human(report: dict[str, Any]) -> None:
    print(f"Pinecone workspace validation: {report['status'].upper()}")
    print(f"Workspace: {report['workspace_id']}")
    print(f"Index/namespace: {report['index']}/{report['namespace']}")
    print(
        f"Scanned: {report['records_scanned']} | "
        f"Workspace records: {report['workspace_records']} | "
        f"Errors: {report['errors']} | Warnings: {report['warnings']}"
    )
    print(f"Chunk kinds: {json.dumps(report['chunk_kinds'], sort_keys=True)}")
    print(f"Generations: {len(report['generations'])}")
    for issue in report["issues"]:
        record = f" [{issue['record_id']}]" if issue.get("record_id") else ""
        print(
            f"- {issue['severity'].upper()} {issue['code']}{record}: {issue['message']}"
        )


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Read-only validation of Pinecone data for one workspace"
    )
    parser.add_argument("workspace_id", help="Workspace UUID to validate")
    parser.add_argument("--json", action="store_true", help="Print JSON report")
    parser.add_argument(
        "--max-scan",
        type=int,
        default=250_000,
        help="Safety limit for records scanned in the namespace (default: 250000)",
    )
    args = parser.parse_args(argv)
    try:
        report = validate_workspace(args.workspace_id, max_scan=args.max_scan)
    except Exception as exc:  # noqa: BLE001 - CLI converts SDK failures to exit code 2.
        if args.json:
            print(json.dumps({"status": "error", "error": str(exc)}, indent=2))
        else:
            print(f"Pinecone validation could not run: {exc}", file=sys.stderr)
        return 2
    if args.json:
        print(json.dumps(report, indent=2, sort_keys=True))
    else:
        _print_human(report)
    return 0 if report["status"] == "pass" else 1


if __name__ == "__main__":
    raise SystemExit(main())
