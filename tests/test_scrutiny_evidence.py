"""Shared Pinecone pool, record slicing, and petition-type system prompts."""

from types import SimpleNamespace

import pytest

from extraction_review.document_parts import (
    match_terms_for_defect,
    page_parts_from_split,
    pool_search_queries,
    select_chunks_for_defect,
    slice_record_for_defect,
)
from extraction_review.process_file import _split_page_parts
from extraction_review.scrutiny.prompts import (
    build_defect_prompt,
    build_system_prompt,
)
from extraction_review.scrutiny.rules import get_catalogue
from extraction_review.vector_store import build_page_records


def test_pool_queries_are_captions_not_legal_sentences() -> None:
    queries = pool_search_queries()
    assert "Listing Proforma" in queries
    assert any("QUESTIONS OF LAW" in q for q in queries)
    for query in queries:
        assert "does not follow" not in query.lower()
        assert len(query) < 120


def test_page_parts_from_split_maps_pages() -> None:
    job = SimpleNamespace(
        result=SimpleNamespace(
            segments=[
                SimpleNamespace(category="Listing Proforma", pages=[3, 4]),
                SimpleNamespace(category="Petition", pages=[10]),
            ]
        )
    )
    assert page_parts_from_split(job) == {3: "Listing Proforma", 4: "Listing Proforma", 10: "Petition"}


@pytest.mark.asyncio
async def test_split_hard_fails_without_config() -> None:
    with pytest.raises(RuntimeError, match="Split config is missing"):
        await _split_page_parts(
            SimpleNamespace(split=object()),
            file_id="file-1",
            parse_job_id=None,
            split_config=None,
            filename="petition.pdf",
        )


def test_build_page_records_stamps_document_part() -> None:
    records = build_page_records(
        base_id="abc",
        page_markdown={1: "cover text", 3: "listing proforma columns 6 and 7"},
        metadata={"file_hash": "abc"},
        page_parts={3: "Listing Proforma"},
    )
    by_page = {r["metadata"]["page_start"]: r["metadata"] for r in records}
    assert "document_part" not in by_page[1]
    assert by_page[3]["document_part"] == "Listing Proforma"


def test_slice_record_drops_unrelated_blocks() -> None:
    catalogue = get_catalogue()
    record = {
        "court": "Supreme Court of India",
        "petition_type": "SLP_CIVIL",
        "advocate_on_record": {"name": "A", "registration_number": "1234"},
        "matter_classification": {"main_category": "Service"},
        "petitioners": [{"full_name": "X"}],
        "impugned_order": {"case_number": "1"},
    }
    listing = slice_record_for_defect(record, catalogue.defect("D004"))
    aor = slice_record_for_defect(record, catalogue.defect("D005"))
    assert listing is not None and "matter_classification" in listing
    assert "advocate_on_record" not in listing
    assert "petitioners" not in listing
    assert aor is not None and "advocate_on_record" in aor
    assert "matter_classification" not in aor
    assert "petitioners" not in aor


def test_select_chunks_prefers_labelled_part() -> None:
    catalogue = get_catalogue()
    pool = [
        {
            "record_id": "s",
            "chunk_kind": "summary",
            "text": "SLP Civil summary",
            "score": 0.1,
            "page": None,
        },
        {
            "record_id": "p10",
            "chunk_kind": "page",
            "page": 10,
            "document_part": "Petition",
            "text": "GROUNDS FOR INTERIM RELIEF",
            "score": 0.99,
        },
        {
            "record_id": "p3",
            "chunk_kind": "page",
            "page": 3,
            "document_part": "Listing Proforma",
            "text": "Proforma for First Listing columns 6 and 7",
            "score": 0.2,
        },
    ]
    chunks = select_chunks_for_defect(
        pool, catalogue.defect("D004"), max_chunks=3
    )
    ids = [c["record_id"] for c in chunks]
    assert ids[0] == "s"
    assert "p3" in ids
    assert "p10" not in ids


def test_match_terms_use_trigger_captions() -> None:
    catalogue = get_catalogue()
    terms = match_terms_for_defect(catalogue.defect("D001"))
    assert any("QUESTIONS OF LAW" in t for t in terms)
    assert all("does not follow the official Form 28" not in t for t in terms)


def test_system_prompt_stable_within_petition_type() -> None:
    catalogue = get_catalogue()
    civil = build_system_prompt(catalogue, "SLP_CIVIL")
    civil_again = build_system_prompt(catalogue, "SLP_CIVIL")
    criminal = build_system_prompt(catalogue, "SLP_CRIMINAL")
    assert civil == civil_again
    assert "Special Leave Petition (Civil)" in civil
    assert "Special Leave Petition (Criminal)" in criminal
    assert civil != criminal
    assert "Filing Formalities" not in civil
    assert "Listing Proforma" not in civil


def test_user_prompt_carries_category_and_sliced_record() -> None:
    catalogue = get_catalogue()
    defect = catalogue.defect("D005")
    record = slice_record_for_defect(
        {
            "court": "SCI",
            "petition_type": "SLP_CIVIL",
            "advocate_on_record": {"registration_number": "1234"},
            "petitioners": [{"full_name": "Should not appear"}],
        },
        defect,
    )
    prompt = build_defect_prompt(
        defect,
        record=record,
        chunks=[
            {
                "chunk_kind": "page",
                "page": 1,
                "document_part": "Advocate's Checklist",
                "text": "Name & Code of AOR 1234",
            }
        ],
        file_name="petition.pdf",
        catalogue=catalogue,
    )
    assert "Advocate" in prompt
    assert "Page 1 — Advocate's Checklist" in prompt
    assert "registration_number" in prompt
    assert "Should not appear" not in prompt
    system = build_system_prompt(catalogue, "SLP_CIVIL")
    assert "Filing Formalities" not in system
    assert "Advocate's Check List" not in system
