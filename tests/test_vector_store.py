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
    text = (
        ("First paragraph sentence. " * 8)
        + "\n\n"
        + ("Second paragraph sentence. " * 8)
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


def _structured_text_page(page: int, text: str, *, kind: str = "text") -> dict:
    return {
        "page_number": page,
        "success": True,
        "metadata": {},
        "items": [{"type": kind, "item_index": 0, "md": text}],
    }


def test_low_information_pages_are_cleaned_before_vector_storage() -> None:
    body = (
        "The certified copy records the judgment and operative directions "
        "challenged in this special leave petition."
    )
    records = store.build_page_records(
        base_id="filing",
        page_markdown={
            1: "56",
            2: "IN THE SUPREME COURT OF INDIA",
            3: "TRUE COPY",
            4: body,
        },
        page_parts={page: "Impugned Order" for page in range(1, 5)},
        page_structures={
            1: _structured_text_page(1, "56"),
            2: _structured_text_page(
                2, "IN THE SUPREME COURT OF INDIA", kind="heading"
            ),
            3: _structured_text_page(3, "TRUE COPY", kind="heading"),
            4: _structured_text_page(4, body),
        },
    )

    assert len(records) == 1
    record = records[0]
    assert record["metadata"]["primary_page"] == 4
    assert record["metadata"]["source_pages"] == ["3", "4"]
    assert record["metadata"]["folio_markers"] == ["1:56"]
    assert record["metadata"]["merged_short_pages"] == ["3"]
    assert "IN THE SUPREME COURT OF INDIA" not in record["chunk_text"]
    segments = store._parse_page_segments(record["chunk_text"])
    assert segments[3] == "TRUE COPY"
    assert segments[4] == body


def test_material_short_text_is_retained_when_cross_part_merge_is_unsafe() -> None:
    records = store.build_page_records(
        base_id="filing",
        page_markdown={1: "TRUE COPY", 2: "A substantial petition paragraph " * 4},
        page_parts={1: "Impugned Order", 2: "Main Petition"},
        page_structures={
            1: _structured_text_page(1, "TRUE COPY", kind="heading"),
            2: _structured_text_page(2, "A substantial petition paragraph " * 4),
        },
    )

    assert len(records) == 2
    assert records[0]["chunk_text"] == "TRUE COPY"


def test_fallback_folio_is_not_embedded_with_neighbour_overlap() -> None:
    body = "Substantive affidavit evidence and verification text. " * 3
    records = store.build_page_records(
        base_id="filing",
        page_markdown={1: "B", 2: body},
        page_parts={1: "Affidavit", 2: "Affidavit"},
    )

    assert len(records) == 1
    assert records[0]["metadata"]["primary_page"] == 2
    assert records[0]["metadata"]["folio_markers"] == ["1:B"]
    assert "--- page 1" not in records[0]["chunk_text"]


def test_upsert_has_final_nonsemantic_page_guard(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    captured: list[dict] = []

    class Index:
        def upsert_records(self, *, namespace: str, records: list[dict]) -> None:
            captured.extend(records)

    monkeypatch.setattr(store, "get_pinecone_client", lambda: object())
    monkeypatch.setattr(store, "ensure_index", lambda _client=None: Index())
    monkeypatch.setattr(
        store, "resolve_text_field", lambda _client=None: "normalized_text"
    )
    items = [
        {
            "record_id": "folio",
            "chunk_text": "79",
            "metadata": {"chunk_kind": "page"},
        },
        {
            "record_id": "legal",
            "chunk_text": "TRUE COPY",
            "metadata": {"chunk_kind": "page"},
        },
    ]

    assert store.upsert_records(items) == 1
    assert [record["_id"] for record in captured] == ["legal"]


def test_search_filters_active_tenant_generation(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    captured: dict[str, object] = {}

    class Index:
        def search(self, **kwargs):
            captured.update(kwargs)
            return {"result": {"hits": []}}

    monkeypatch.setattr(store, "get_pinecone_client", lambda: object())
    monkeypatch.setattr(store, "ensure_index", lambda _client=None: Index())
    monkeypatch.setattr(
        store, "resolve_text_field", lambda _client=None: "normalized_text"
    )
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
        "vector_schema_version": {"$eq": 4},
    }


def test_search_applies_canonical_part_and_subtype_before_top_k(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    calls: list[dict] = []

    class Index:
        def search(self, **kwargs):
            calls.append(kwargs)
            return {
                "result": {
                    "hits": [
                        {
                            "_id": "hit",
                            "_score": 0.8,
                            "fields": {
                                "normalized_text": "APPLICATION FOR EXEMPTION FROM FILING CERTIFIED COPY",
                                "chunk_kind": "page",
                                "primary_page": 4,
                                "document_part": "Application 2",
                                "document_part_id": "application",
                                "document_subtype_id": "application_exemption_certified_copy",
                                "vector_schema_version": 4,
                            },
                        }
                    ]
                }
            }

    monkeypatch.setattr(store, "get_pinecone_client", lambda: object())
    monkeypatch.setattr(store, "ensure_index", lambda _client=None: Index())
    monkeypatch.setattr(
        store, "resolve_text_field", lambda _client=None: "normalized_text"
    )

    chunks = store.search_filing_chunks(
        "certified copy exemption",
        file_hash="a" * 64,
        document_part="Application",
        document_subtype_id="application_exemption_certified_copy",
    )

    assert len(calls) == 1
    assert calls[0]["query"]["filter"]["document_part_id"] == {"$eq": "application"}
    assert calls[0]["query"]["filter"]["document_subtype_id"] == {
        "$eq": "application_exemption_certified_copy"
    }
    assert chunks[0]["document_subtype_id"] == "application_exemption_certified_copy"


def test_page_records_include_canonical_part_and_propagated_application_subtype() -> (
    None
):
    records = store.build_page_records(
        base_id="filing",
        page_markdown={
            1: "APPLICATION FOR EXEMPTION FROM FILING CERTIFIED COPY " + "ground " * 12,
            2: "The applicant respectfully requests the relief stated above. " * 2,
        },
        page_parts={1: "Application 2", 2: "Application 2"},
    )

    assert records
    for record in records:
        metadata = record["metadata"]
        assert metadata["document_part_id"] == "application"
        assert metadata["document_subtype_id"] == (
            "application_exemption_certified_copy"
        )
        assert metadata["vector_schema_version"] == store.VECTOR_SCHEMA_VERSION


def test_current_schema_legacy_record_uses_conservative_subtype_fallback(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    calls: list[dict] = []

    class Index:
        def search(self, **kwargs):
            calls.append(kwargs)
            if "document_part_id" in kwargs["query"]["filter"]:
                return {"result": {"hits": []}}
            return {
                "result": {
                    "hits": [
                        {
                            "_id": "legacy-schema-4",
                            "_score": 0.7,
                            "fields": {
                                "normalized_text": (
                                    "APPLICATION FOR EXEMPTION FROM FILING "
                                    "CERTIFIED COPY"
                                ),
                                "chunk_kind": "page",
                                "primary_page": 8,
                                "document_part": "Application 3",
                                "vector_schema_version": 4,
                            },
                        }
                    ]
                }
            }

    monkeypatch.setattr(store, "get_pinecone_client", lambda: object())
    monkeypatch.setattr(store, "ensure_index", lambda _client=None: Index())
    monkeypatch.setattr(
        store, "resolve_text_field", lambda _client=None: "normalized_text"
    )

    chunks = store.search_filing_chunks(
        "certified copy exemption",
        file_hash="a" * 64,
        document_part="Application",
        document_subtype_id="application_exemption_certified_copy",
    )

    assert len(calls) == 2
    assert chunks[0]["document_subtype_inferred"] is True
