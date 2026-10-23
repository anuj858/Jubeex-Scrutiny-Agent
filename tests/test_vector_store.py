"""Vector record provenance, isolation, and index-contract regression tests."""

from __future__ import annotations

import pytest

import extraction_review.vector_store as store
from extraction_review.scrutiny.rules import Defect
from extraction_review.scrutiny.schema import (
    DefectResponse,
    EvidenceRef,
    apply_evidence_pages,
)


def _defect() -> Defect:
    return Defect(
        check_id="D-TEST",
        main_category="General/Global",
        defect="The Vakalatnama date is missing.",
        requirement="The Vakalatnama must contain its execution date.",
        where_to_look=["Check the Vakalatnama"],
        inspect_parts=["Vakalatnama"],
        how_to_cure=["Add the date"],
        location_source="Official source not recorded",
    )


def test_scoped_record_ids_change_across_tenants_and_generations() -> None:
    common = {"file_hash": "a" * 64, "workspace_id": "workspace-1"}
    first = store.scoped_record_base_id(
        **common, organization_id="org-1", index_generation="generation-1"
    )
    assert first != store.scoped_record_base_id(
        **common, organization_id="org-2", index_generation="generation-1"
    )
    assert first != store.scoped_record_base_id(
        **common, organization_id="org-1", index_generation="generation-2"
    )
    with pytest.raises(ValueError, match="required"):
        store.scoped_record_base_id(
            file_hash="a" * 64,
            organization_id="",
            workspace_id="workspace-1",
            index_generation="generation-1",
        )


def test_existing_index_contract_must_match_model_and_text_field(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(store, "pinecone_embed_model", "llama-text-embed-v2")
    monkeypatch.setattr(store, "pinecone_text_field", "normalized_text")
    valid = {
        "embed": {
            "model": "llama-text-embed-v2",
            "field_map": {"text": "normalized_text"},
        }
    }
    assert store.validate_index_contract(valid) == "normalized_text"
    invalid = {
        "embed": {
            "model": "another-model",
            "field_map": {"text": "normalized_text"},
        }
    }
    with pytest.raises(RuntimeError, match="another-model"):
        store.validate_index_contract(invalid)


def test_cold_ensure_index_can_initialize_client_without_lock_deadlock(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    marker = object()

    class Client:
        def has_index(self, name: str) -> bool:
            assert name == store.pinecone_index_name
            return True

        def describe_index(self, name: str) -> dict[str, object]:
            assert name == store.pinecone_index_name
            return {
                "embed": {
                    "model": "llama-text-embed-v2",
                    "field_map": {"text": "normalized_text"},
                }
            }

        def Index(self, name: str) -> object:
            assert name == store.pinecone_index_name
            return marker

    monkeypatch.setattr(store, "pinecone_api_key", "test-key")
    monkeypatch.setattr(store, "pinecone_embed_model", "llama-text-embed-v2")
    monkeypatch.setattr(store, "pinecone_text_field", "normalized_text")
    monkeypatch.setattr(store, "Pinecone", lambda **_kwargs: Client())
    monkeypatch.setattr(store, "_cached_client", None)
    monkeypatch.setattr(store, "_cached_index", None)
    monkeypatch.setattr(store, "_cached_text_field", None)

    assert store.ensure_index() is marker


def test_chunking_prefers_paragraph_boundary_and_validates_overlap() -> None:
    text = ("First paragraph sentence. " * 8) + "\n\n" + (
        "Second paragraph sentence. " * 8
    )
    windows = store.split_text_windows(text, max_chars=260, overlap=30)
    assert len(windows) == 2
    assert windows[0].endswith("sentence.")
    assert "Second paragraph" in windows[1]
    with pytest.raises(ValueError, match="smaller"):
        store.split_text_windows(text, max_chars=100, overlap=100)


def test_empty_ocr_page_is_manifest_not_model_evidence() -> None:
    records = store.build_page_records(
        base_id="filing",
        page_markdown={1: "", 2: "Signed Vakalatnama"},
        page_parts={1: "Vakalatnama", 2: "Vakalatnama"},
    )
    empty = records[0]
    assert empty["metadata"]["chunk_kind"] == "page_manifest"
    assert empty["metadata"]["ocr_status"] == "empty"
    assert empty["metadata"]["requires_visual_review"] is True
    response = DefectResponse(
        check_id="D-TEST",
        status="defect_found",
        confidence=0.99,
        summary="The page is blank.",
        reasoning="The page has no text.",
        evidence=[EvidenceRef(page=1, quote=store.EMPTY_PAGE_MARKER)],
    )
    chunk = {
        "record_id": empty["record_id"],
        "text": empty["chunk_text"],
        "page": 1,
        "page_end": 1,
        "chunk_kind": "page_manifest",
        "document_part": "Vakalatnama",
    }
    assert apply_evidence_pages(response, [chunk], _defect()).evidence[0].page is None


def test_overlap_quote_resolves_to_its_actual_source_page() -> None:
    records = store.build_page_records(
        base_id="filing",
        page_markdown={
            10: "x" * 220 + " First-page execution clause",
            11: "Second-page signature block " + "y" * 220,
        },
        page_parts={10: "Vakalatnama", 11: "Vakalatnama"},
    )
    page_11 = next(row for row in records if row["metadata"]["page_start"] == 11)
    page_texts = store._parse_page_segments(page_11["chunk_text"])
    quote = "First-page execution clause"
    assert quote in page_texts[10]
    response = DefectResponse(
        check_id="D-TEST",
        status="defect_found",
        confidence=0.99,
        summary="The execution clause is incorrect.",
        reasoning="The quoted clause is incorrect.",
        evidence=[EvidenceRef(page=11, quote=quote)],
    )
    chunk = {
        "record_id": page_11["record_id"],
        "text": page_11["chunk_text"],
        "page": 11,
        "page_end": 11,
        "page_texts": page_texts,
        "chunk_kind": "page",
        "document_part": "Vakalatnama",
    }
    grounded = apply_evidence_pages(response, [chunk], _defect())
    assert grounded.evidence[0].page == 10


def test_search_filters_active_tenant_generation(monkeypatch: pytest.MonkeyPatch) -> None:
    captured: dict[str, object] = {}

    class Index:
        def search(self, **kwargs):
            captured.update(kwargs)
            return {"result": {"hits": []}}

    monkeypatch.setattr(store, "get_pinecone_client", lambda: object())
    monkeypatch.setattr(store, "ensure_index", lambda _client=None: Index())
    monkeypatch.setattr(store, "resolve_text_field", lambda _client=None: "normalized_text")
    store.search_filing_chunks(
        "vakalatnama",
        file_hash="a" * 64,
        organization_id="org-1",
        workspace_id="workspace-1",
        index_generation="generation-1",
    )
    query = captured["query"]
    assert isinstance(query, dict)
    assert query["filter"] == {
        "file_hash": {"$eq": "a" * 64},
        "organization_id": {"$eq": "org-1"},
        "workspace_id": {"$eq": "workspace-1"},
        "index_generation": {"$eq": "generation-1"},
    }

