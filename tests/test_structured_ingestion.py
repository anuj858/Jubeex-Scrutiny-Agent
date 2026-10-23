"""Expanded LlamaParse JSON through semantic chunks and Pinecone records."""

from __future__ import annotations

from typing import Any

from extraction_review import vector_store
from extraction_review.parsed_document import (
    build_semantic_chunks,
    extract_structured_pages,
)
from extraction_review.scrutiny.prompts import _format_evidence


def _expanded_parse_payload() -> dict[str, Any]:
    rows = [
        [
            "S.No.",
            "PARTICULARS",
            "Page No. of part to which it belongs",
            "Page No. of part to which it belongs",
            "Remarks",
        ],
        ["", "", "Part – I<br/>(Contents<br/>of Paper<br/>Book)", "Part – II", ""],
        ["(i)", "(ii)", "(iii)", "(iv)", "(v)"],
        ["1.", "Office Report on Limitation", "A", "A", ""],
        ["2.", "Listing Proforma", "A1-A2", "A1-A2", ""],
    ]
    return {
        "markdown": {
            "pages": [
                {
                    "page_number": 1,
                    "markdown": "# I N D E X\n\n<table><tr><td>duplicate</td></tr></table>",
                }
            ]
        },
        "text": {"pages": [{"page_number": 1, "text": "I N D E X raw text"}]},
        "items": {
            "pages": [
                {
                    "page_number": 1,
                    "success": True,
                    "items": [
                        {"type": "header", "md": "16"},
                        {"type": "heading", "level": 1, "md": "# I N D E X"},
                        {
                            "type": "table",
                            "md": "<table>duplicate rendering</table>",
                            "html": "<table>duplicate rendering</table>",
                            "csv": "duplicate,rendering",
                            "rows": rows,
                            "parse_concerns": [{"type": "possible_merged_columns"}],
                        },
                    ],
                }
            ]
        },
        "metadata": {
            "pages": [
                {
                    "page_number": 1,
                    "confidence": 0.94,
                    "printed_page_number": "16",
                    "original_orientation_angle": 0,
                }
            ]
        },
    }


def test_expanded_items_are_the_chunk_source_and_tables_use_rows() -> None:
    pages = extract_structured_pages(_expanded_parse_payload())
    assert list(pages) == [1]
    table = pages[1]["items"][1]
    assert "html" not in table
    assert "csv" not in table

    chunks = build_semantic_chunks(pages[1])
    combined = "\n".join(str(chunk["text"]) for chunk in chunks)
    assert "I N D E X" in combined
    assert "S.No.: 1." in combined
    assert "PARTICULARS: Office Report on Limitation" in combined
    assert "Part – I" in combined
    assert "<table>" not in combined
    assert "duplicate rendering" not in combined
    assert "I N D E X raw text" not in combined
    assert all(chunk["heading_path"] == ["I N D E X"] for chunk in chunks)


def test_structured_chunk_becomes_exact_integrated_embedding_field(
    monkeypatch,
) -> None:
    page = extract_structured_pages(_expanded_parse_payload())[1]
    records = vector_store.build_page_records(
        base_id="tenant-generation",
        page_markdown={1: "markdown fallback must not be embedded"},
        page_parts={1: ["Index"]},
        page_structures={1: page},
        metadata={
            "file_hash": "f" * 64,
            "organization_id": "org-1",
            "workspace_id": "workspace-1",
            "index_generation": "generation-1",
        },
    )
    assert records
    assert all(record["metadata"]["parser_source"] == "items" for record in records)
    assert records[0]["metadata"]["page_start"] == 1
    assert records[0]["metadata"]["printed_page_number"] == "16"
    assert records[0]["metadata"]["document_part"] == "Index"
    assert records[0]["metadata"]["heading_path"] == ["I N D E X"]

    captured: list[dict[str, Any]] = []

    class Index:
        def upsert_records(
            self, *, namespace: str, records: list[dict[str, Any]]
        ) -> None:
            assert namespace == "test-namespace"
            captured.extend(records)

    monkeypatch.setattr(vector_store, "get_pinecone_client", lambda: object())
    monkeypatch.setattr(vector_store, "ensure_index", lambda _client=None: Index())
    monkeypatch.setattr(
        vector_store, "resolve_text_field", lambda _client=None: "normalized_text"
    )
    monkeypatch.setattr(vector_store, "pinecone_namespace", "test-namespace")
    assert vector_store.upsert_records(records) == len(records)
    assert captured[0]["normalized_text"] == records[0]["chunk_text"]
    assert captured[0]["_id"] == records[0]["record_id"]
    assert captured[0]["workspace_id"] == "workspace-1"
    assert "chunk_text" not in captured[0]


def test_gemini_evidence_label_includes_parser_provenance() -> None:
    evidence = _format_evidence(
        [
            {
                "record_id": "chunk-1",
                "chunk_kind": "page",
                "page": 2,
                "document_part": "Index",
                "printed_page_number": "A-1",
                "heading_path": ["I N D E X"],
                "table_row_start": 3,
                "table_row_end": 11,
                "text": "S.No.: 1.\nPARTICULARS: Office Report on Limitation",
            }
        ]
    )
    assert "Page 2 — Index" in evidence
    assert "printed page A-1" in evidence
    assert "heading: I N D E X" in evidence
    assert "table rows 3-11" in evidence
