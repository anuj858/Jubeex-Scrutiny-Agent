"""Tests for the read-only Pinecone workspace validator."""

from __future__ import annotations

import extraction_review.pinecone_validator as validator

WORKSPACE_ID = "e4da3da2-1ba9-4d52-a6a2-fcd151deda3a"


class FakeIndex:
    def __init__(self, records):
        self.records = records
        self.fetch_calls = []

    def list_paginated(self, **kwargs):
        token = kwargs.get("pagination_token")
        ids = list(self.records)
        if token is None:
            return {
                "vectors": [{"id": record_id} for record_id in ids[:2]],
                "pagination": {"next": "page-2"} if len(ids) > 2 else {},
            }
        return {"vectors": [{"id": record_id} for record_id in ids[2:]]}

    def fetch(self, *, ids, namespace):
        self.fetch_calls.append((ids, namespace))
        return {"vectors": {record_id: self.records[record_id] for record_id in ids}}


class FakeClient:
    def __init__(self, index):
        self.index = index

    def has_index(self, name):
        return True

    def describe_index(self, name):
        return {
            "embed": {
                "model": "llama-text-embed-v2",
                "field_map": {"text": "normalized_text"},
            }
        }

    def Index(self, name):
        return self.index


def _record(record_id, *, workspace_id=WORKSPACE_ID, kind="summary", **overrides):
    fields = {
        "workspace_id": workspace_id,
        "organization_id": "org-1",
        "file_hash": "a" * 64,
        "index_generation": "generation-1",
        "chunk_kind": kind,
        "vector_schema_version": 3,
        "normalized_text": "valid searchable text",
    }
    if kind in {"page", "page_manifest"}:
        fields.update(
            {
                "page_start": 1,
                "page_end": 1,
                "primary_page": 1,
                "source_pages": ["1"],
                "chunk_index": 0,
                "ocr_status": "text",
            }
        )
    fields.update(overrides)
    return {"id": record_id, "metadata": fields}


def _base_id():
    return validator.store.scoped_record_base_id(
        file_hash="a" * 64,
        organization_id="org-1",
        workspace_id=WORKSPACE_ID,
        index_generation="generation-1",
    )


def test_validates_all_workspace_records_and_ignores_other_tenants(monkeypatch):
    base = _base_id()
    records = {
        f"{base}:summary": _record(f"{base}:summary"),
        f"{base}:page:1:0": _record(f"{base}:page:1:0", kind="page"),
        "other:summary": _record("other:summary", workspace_id="other-workspace"),
    }
    index = FakeIndex(records)
    monkeypatch.setattr(
        validator.store, "get_pinecone_client", lambda: FakeClient(index)
    )

    report = validator.validate_workspace(WORKSPACE_ID)

    assert report["status"] == "pass"
    assert report["records_scanned"] == 3
    assert report["workspace_records"] == 2
    assert report["chunk_kinds"] == {"page": 1, "summary": 1}
    assert report["issues"] == []


def test_reports_schema_text_id_and_duplicate_chunk_errors(monkeypatch):
    base = _base_id()
    records = {
        f"{base}:summary": _record(f"{base}:summary"),
        "bad:page:1:0": _record(
            "bad:page:1:0",
            kind="page",
            normalized_text="",
            vector_schema_version=2,
        ),
        "bad:page:1:duplicate": _record(
            "bad:page:1:duplicate",
            kind="page",
        ),
    }
    index = FakeIndex(records)
    monkeypatch.setattr(
        validator.store, "get_pinecone_client", lambda: FakeClient(index)
    )

    report = validator.validate_workspace(WORKSPACE_ID)

    codes = {issue["code"] for issue in report["issues"]}
    assert report["status"] == "fail"
    assert "missing_text" in codes
    assert "schema_version_mismatch" in codes
    assert "record_id_scope_mismatch" in codes
    assert "duplicate_logical_chunk" in codes


def test_missing_workspace_is_failure(monkeypatch):
    index = FakeIndex({})
    monkeypatch.setattr(
        validator.store, "get_pinecone_client", lambda: FakeClient(index)
    )

    report = validator.validate_workspace(WORKSPACE_ID)

    assert report["status"] == "fail"
    assert report["issues"][0]["code"] == "workspace_not_found"


def test_main_rejects_non_uuid_workspace_id(capsys):
    assert validator.main(["not-a-uuid"]) == 2
    assert "must be a UUID" in capsys.readouterr().err
