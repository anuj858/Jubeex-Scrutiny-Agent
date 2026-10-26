"""Defect-specific reranking without weakening document-part coverage."""

from __future__ import annotations

from types import SimpleNamespace

from extraction_review import reranking
from extraction_review.document_parts import select_chunks_for_defect
from extraction_review.scrutiny.rules import Defect


def _defect() -> Defect:
    return Defect(
        check_id="D-DATE",
        main_category="General/Global",
        defect="The Vakalatnama execution date is missing.",
        requirement="The Vakalatnama must state its execution date.",
        trigger_words="date; executed",
        where_to_look=["Inspect the Vakalatnama signature and execution block."],
        inspect_parts=["Vakalatnama"],
        how_to_cure=["Add the execution date."],
        location_source="Registry catalogue",
        ai_note="Equivalent date labels are acceptable.",
    )


def test_reranker_annotates_candidates_and_selector_uses_score(monkeypatch) -> None:
    monkeypatch.setenv("SCRUTINY_RERANK_ENABLED", "true")
    class Inference:
        def rerank(self, **kwargs):
            assert kwargs["model"] == "bge-reranker-v2-m3"
            assert "execution date" in kwargs["query"]
            assert kwargs["return_documents"] is False
            assert kwargs["parameters"] == {"truncate": "END"}
            return SimpleNamespace(
                data=[
                    SimpleNamespace(index=1, score=0.98),
                    SimpleNamespace(index=0, score=0.12),
                ]
            )

    monkeypatch.setattr(
        reranking,
        "get_pinecone_client",
        lambda: SimpleNamespace(inference=Inference()),
    )
    chunks = [
        {
            "record_id": "less-relevant",
            "chunk_kind": "page",
            "page": 4,
            "document_part": "Vakalatnama",
            "text": "Vakalatnama general authorization text",
            "score": 0.99,
        },
        {
            "record_id": "date-block",
            "chunk_kind": "page",
            "page": 5,
            "document_part": "Vakalatnama",
            "text": "Executed on 12 January 2026 and signed by the petitioner",
            "score": 0.6,
        },
    ]
    annotated = reranking.rerank_chunks_for_defect(chunks, _defect())
    assert annotated[1]["rerank_score"] == 0.98
    assert annotated[1]["rerank_rank"] == 1
    selected = select_chunks_for_defect(annotated, _defect(), max_chunks=1)
    assert selected[0]["record_id"] == "date-block"


def test_reranking_fails_open(monkeypatch) -> None:
    monkeypatch.setenv("SCRUTINY_RERANK_ENABLED", "true")
    class Inference:
        def rerank(self, **_kwargs):
            raise RuntimeError("temporary inference failure")

    monkeypatch.setattr(
        reranking,
        "get_pinecone_client",
        lambda: SimpleNamespace(inference=Inference()),
    )
    chunks = [
        {"record_id": "a", "chunk_kind": "page", "text": "one"},
        {"record_id": "b", "chunk_kind": "page", "text": "two"},
    ]
    assert reranking.rerank_chunks_for_defect(chunks, _defect()) is chunks
