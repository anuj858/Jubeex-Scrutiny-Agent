"""Defect-specific retrieval recall without weakening tenant isolation."""

from __future__ import annotations

import pytest

import extraction_review.scrutiny_workflow as workflow
from extraction_review.document_parts import select_chunks_for_defect
from extraction_review.scrutiny.rules import Defect


def _defect() -> Defect:
    return Defect(
        check_id="D-PARTIES",
        main_category="General/Global",
        defect="Respondent particulars are incomplete.",
        requirement="Every respondent must have complete particulars.",
        where_to_look=["Check the Memo of Parties for all respondent particulars."],
        inspect_parts=["Main Petition", "Memo of Parties"],
        how_to_cure=["Complete the party particulars."],
        location_source="Registry catalogue",
    )


@pytest.mark.asyncio
async def test_missing_required_part_gets_one_bounded_broader_search(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    calls: list[dict[str, object]] = []
    main = {
        "record_id": "main-16",
        "chunk_kind": "page",
        "page": 16,
        "document_part": "Main Petition",
        "text": "3. State of Maharashtra — Respondent No.",
        "score": 0.9,
    }
    memo = {
        "record_id": "memo-104",
        "chunk_kind": "page",
        "page": 104,
        "document_part": "Memo of Parties",
        "text": "Respondent 5 name and postal address",
        "score": 0.8,
    }

    def gather(_queries: list[str], **kwargs):
        calls.append(kwargs)
        return [main] if len(calls) == 1 else [memo]

    monkeypatch.setattr(workflow, "gather_filing_evidence", gather)
    monkeypatch.setattr(
        workflow, "rerank_chunks_for_defect", lambda chunks, _defect: chunks
    )
    monkeypatch.setattr(workflow, "scrutiny_top_k", lambda: 8)

    chunks = await workflow._chunks_for_defect(
        _defect(),
        file_hash="f" * 64,
        max_chunks=12,
        use_pinecone=True,
        organization_id="org-1",
        workspace_id="workspace-1",
        index_generation="generation-1",
    )

    assert len(calls) == 2
    assert calls[1]["top_k"] == 16
    assert calls[1]["document_parts"] == ["Memo of Parties"]
    assert calls[1]["organization_id"] == "org-1"
    assert calls[1]["workspace_id"] == "workspace-1"
    assert calls[1]["index_generation"] == "generation-1"
    assert {chunk["record_id"] for chunk in chunks} == {"main-16", "memo-104"}


def test_table_continuation_reserves_preceding_page_context() -> None:
    defect = _defect().model_copy(
        update={"inspect_parts": ["Main Petition"], "context_parts": []}
    )
    previous = {
        "record_id": "main-16",
        "chunk_kind": "page",
        "content_kind": "table_rows",
        "page": 16,
        "document_part": "Main Petition",
        "text": "Column 1: 7. Raibham, age 68 years,",
        "score": 0.4,
    }
    continuation = {
        "record_id": "main-17",
        "chunk_kind": "page",
        "content_kind": "table_rows",
        "page": 17,
        "document_part": "Main Petition",
        "text": "Column 1: occupation Retired, r/o Plot no. 25",
        "rerank_score": 0.99,
        "score": 0.9,
    }

    chosen = select_chunks_for_defect(
        [previous, continuation], defect, max_chunks=2
    )

    assert [chunk["record_id"] for chunk in chosen] == ["main-16", "main-17"]


def test_generic_table_columns_reserve_inherited_header_page() -> None:
    defect = _defect().model_copy(
        update={"inspect_parts": ["Main Petition"], "context_parts": []}
    )
    header = {
        "record_id": "main-15",
        "chunk_kind": "page",
        "content_kind": "structured_text",
        "item_types": ["heading", "table"],
        "page": 15,
        "document_part": "Main Petition",
        "text": "BEFORE HIGH COURT\nBEFORE SUPREME COURT",
        "score": 0.2,
    }
    rows = {
        "record_id": "main-16",
        "chunk_kind": "page",
        "content_kind": "table_rows",
        "item_types": ["table"],
        "page": 16,
        "document_part": "Main Petition",
        "text": "Column 1: 3. State of Maharashtra\nColumn 2: Respondent No.",
        "rerank_score": 0.99,
        "score": 0.9,
    }

    chosen = select_chunks_for_defect([header, rows], defect, max_chunks=2)

    assert [chunk["record_id"] for chunk in chosen] == ["main-15", "main-16"]
