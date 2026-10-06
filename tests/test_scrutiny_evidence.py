"""Shared Pinecone pool, record slicing, and petition-type system prompts."""

from types import SimpleNamespace
from unittest.mock import AsyncMock

import pytest

from extraction_review.document_parts import (
    documents_from_page_parts,
    explode_repeating_split_parts,
    keep_nearby_scores,
    max_chunks_for_defect,
    overlay_split_documents,
    page_parts_from_split,
    parts_named_in_text,
    pool_search_queries,
    select_chunks_for_defect,
)
from extraction_review.process_file import _split_page_parts
from extraction_review.scrutiny.prompts import filing_location
from extraction_review.scrutiny.schema import (
    BoundingBox,
    DefectResponse,
    EvidenceRef,
    FindingEvidence,
    _one_published_evidence,
    apply_evidence_pages,
    apply_status_policy,
)
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
                SimpleNamespace(category="Main Petition", pages=[10]),
            ]
        )
    )
    assert page_parts_from_split(job) == {
        3: ["Listing Proforma"],
        4: ["Listing Proforma"],
        10: ["Main Petition"],
    }


def test_split_keeps_each_document_once_when_paper_book_is_duplicated() -> None:
    job = SimpleNamespace(
        result=SimpleNamespace(
            segments=[
                SimpleNamespace(category="Advocate's Checklist", pages=[1, 2, 51, 52]),
                SimpleNamespace(category="Cover Page", pages=[3, 53]),
                SimpleNamespace(category="Index", pages=[5, 6, 7, 55, 56, 57]),
                SimpleNamespace(category="Office Report on Limitation", pages=[8, 58]),
                SimpleNamespace(category="Listing Proforma", pages=[9, 10, 59, 60]),
                SimpleNamespace(category="Main Petition", pages=[17, 25, 26, 35]),
                SimpleNamespace(
                    category="Record of Proceedings", pages=[4, 20, 21, 23, 54]
                ),
            ]
        )
    )
    mapping = explode_repeating_split_parts(page_parts_from_split(job))
    docs = documents_from_page_parts(mapping)
    assert "Index (pp. 5–7, 55–57)" in docs["items"]
    assert mapping[1] == ["Advocate's Checklist"]
    assert 51 not in mapping
    # First contiguous outer run wins (not the longest): page 17 is kept;
    # later Main Petition islands at 25–26 and 35 are dropped.
    assert mapping[17] == ["Main Petition"]
    assert 25 not in mapping
    assert 26 not in mapping
    assert 35 not in mapping
    assert 4 not in mapping
    assert 54 not in mapping
    assert mapping[20] == ["Record of Proceedings"]
    assert mapping[21] == ["Record of Proceedings"]
    assert 23 not in mapping  # singleton RoP island dropped (longest-run wins)


def test_distinct_index_runs_are_preserved() -> None:
    job = SimpleNamespace(
        result=SimpleNamespace(
            segments=[
                SimpleNamespace(category="Index", pages=[5, 6, 7, 55, 56, 57]),
                SimpleNamespace(category="Main Petition", pages=[10]),
            ]
        )
    )
    mapping = explode_repeating_split_parts(page_parts_from_split(job))
    assert mapping == {
        5: ["Index"],
        6: ["Index"],
        7: ["Index"],
        10: ["Main Petition"],
        55: ["Index"],
        56: ["Index"],
        57: ["Index"],
    }


def test_one_page_can_carry_two_document_parts() -> None:
    job = SimpleNamespace(
        result=SimpleNamespace(
            segments=[
                SimpleNamespace(category="Affidavit", pages=[50]),
                SimpleNamespace(category="Vakalatnama", pages=[50]),
            ]
        )
    )
    mapping = page_parts_from_split(job)
    assert mapping[50] == ["Affidavit", "Vakalatnama"]
    docs = documents_from_page_parts(mapping)
    assert "Affidavit (p. 50)" in docs["items"]
    assert "Memo of Appearance + Vakalatnama (p. 50)" in docs["items"]
    records = build_page_records(
        base_id="abc",
        page_markdown={50: "affidavit then vakalatnama on the same leaf"},
        page_parts=mapping,
    )
    assert records[0]["metadata"]["document_part"] == [
        "Affidavit",
        "Vakalatnama",
    ]


def test_legacy_combined_vakalatnama_expands_to_separate_parts() -> None:
    job = SimpleNamespace(
        result=SimpleNamespace(
            segments=[
                SimpleNamespace(category="Vakalatnama + PoA/BR", pages=[50]),
            ]
        )
    )
    mapping = page_parts_from_split(job)
    assert mapping[50] == ["Vakalatnama", "PoA/BR"]
    docs = documents_from_page_parts(mapping)
    assert docs["items"] == [
        "Memo of Appearance + Vakalatnama (p. 50)",
        "PoA/BR (p. 50)",
    ]


@pytest.mark.asyncio
async def test_split_hard_fails_without_config() -> None:
    with pytest.raises(RuntimeError, match="Split config is missing"):
        await _split_page_parts(
            SimpleNamespace(split=object()),
            file_id="file-1",
            split_config=None,
            filename="petition.pdf",
        )


@pytest.mark.asyncio
async def test_split_sends_file_uuid_not_parse_job_id() -> None:
    created = SimpleNamespace(id="split-job-1")
    completed = SimpleNamespace(
        status="COMPLETED",
        result=SimpleNamespace(
            segments=[SimpleNamespace(category="Main Petition", pages=[1])]
        ),
    )
    split_api = SimpleNamespace(
        create=AsyncMock(return_value=created),
        get=AsyncMock(return_value=completed),
    )
    split_config = SimpleNamespace(
        configuration_id=None,
        categories=[SimpleNamespace(name="Main Petition")],
        model_dump=lambda **_kwargs: {"categories": [{"name": "Main Petition"}]},
    )
    mapping, split_job_id, exchange = await _split_page_parts(
        SimpleNamespace(split=split_api),
        file_id="aaaaaaaa-bbbb-cccc-dddd-eeeeeeeeeeee",
        split_config=split_config,
        filename="petition.pdf",
    )
    split_api.create.assert_awaited_once()
    kwargs = split_api.create.await_args.kwargs
    assert kwargs["file_input"] == "aaaaaaaa-bbbb-cccc-dddd-eeeeeeeeeeee"
    assert mapping == {1: ["Main Petition"]}
    assert split_job_id == "split-job-1"


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
    assert "--- from page" not in records[0]["chunk_text"]
    assert "--- from page" not in records[1]["chunk_text"]


def test_page_records_borrow_same_part_neighbours_only() -> None:
    listing_3 = "L3 " + ("a" * 40)
    listing_4 = "L4 " + ("b" * 40)
    petition = "Petition body " + ("c" * 40)
    records = build_page_records(
        base_id="abc",
        page_markdown={
            2: "Cover page heading",
            3: listing_3,
            4: listing_4,
            5: petition,
        },
        page_parts={
            2: "Cover Page",
            3: "Listing Proforma",
            4: "Listing Proforma",
            5: "Main Petition",
        },
    )
    by_page = {r["metadata"]["page_start"]: r for r in records}

    cover = by_page[2]["chunk_text"]
    assert cover == "Cover page heading"
    assert "--- from page" not in cover
    assert by_page[2]["metadata"]["page_end"] == 2

    page3 = by_page[3]["chunk_text"]
    assert page3.startswith(listing_3) or listing_3 in page3
    assert "--- from page 4 ---" in page3
    assert listing_4[:20] in page3
    assert "Cover page" not in page3
    assert "Petition body" not in page3
    assert by_page[3]["metadata"]["page_start"] == 3
    assert by_page[3]["metadata"]["page_end"] == 4
    assert by_page[3]["metadata"]["pages"] == "3-4"

    page4 = by_page[4]["chunk_text"]
    assert "--- from page 3 ---" in page4
    assert listing_3[-20:] in page4
    assert "--- from page 5 ---" not in page4
    assert "Petition body" not in page4
    assert by_page[4]["metadata"]["page_end"] == 4

    page5 = by_page[5]["chunk_text"]
    assert page5 == petition
    assert "--- from page" not in page5


def test_unlabelled_pages_do_not_borrow_neighbours() -> None:
    records = build_page_records(
        base_id="abc",
        page_markdown={1: "first page text", 2: "second page text"},
        page_parts={},
    )
    texts = [r["chunk_text"] for r in records]
    assert texts == ["first page text", "second page text"]


def test_documents_from_split_include_page_spans() -> None:
    docs = documents_from_page_parts(
        {
            3: "Office Report on Limitation",
            4: "Office Report on Limitation",
            50: "Vakalatnama",
        }
    )
    assert docs["count"] == 2
    assert docs["items"][0] == "Office Report on Limitation (pp. 3–4)"
    assert docs["items"][1] == "Memo of Appearance + Vakalatnama (p. 50)"


def test_overlay_split_documents_replaces_index_slang() -> None:
    payload = {
        "data": {
            "petition_type": "SLP_CIVIL",
            "filing_summary": {
                "documents": {"count": 1, "items": ["V/A"]},
            },
        }
    }
    overlay_split_documents(payload, {50: "Vakalatnama"})
    assert "filing_summary" not in payload["data"]
    assert payload["data"]["documents"] == [
        {"name": "Memo of Appearance + Vakalatnama", "start_page": 50, "end_page": 50},
    ]


def test_keep_nearby_scores_drops_far_neighbours() -> None:
    chunks = [
        {"record_id": "a", "score": 0.90},
        {"record_id": "b", "score": 0.88},
        {"record_id": "c", "score": 0.85},
        {"record_id": "d", "score": 0.40},
        {"record_id": "e", "score": 0.12},
    ]
    kept = keep_nearby_scores(chunks, max_n=8)
    assert [c["record_id"] for c in kept] == ["a", "b", "c"]


def test_each_annexure_check_uses_full_budget_and_distinct_annexures() -> None:
    defect = SimpleNamespace(
        defect="Annexures are not true copies",
        requirement="Each Annexure must be a true copy",
        where_to_look=["Check each Annexure for a true copy endorsement."],
        inspect_parts=["Annexures"],
        context_parts=[],
        exclude_parts=None,
        trigger_words="ANNEXURE P; TRUE COPY",
        parent_check_id=None,
    )
    pool = [
        {
            "record_id": f"p{number}",
            "score": 0.99 - number / 100,
            "chunk_kind": "page",
            "page": number,
            "document_part": f"Annexure P-{number}",
            "text": "TRUE COPY" if number != 2 else "ANNEXURE P-2 order",
        }
        for number in range(1, 7)
    ]

    assert max_chunks_for_defect(defect, ceiling=12) == 12
    chosen = select_chunks_for_defect(pool, defect, max_chunks=12)
    assert {chunk["document_part"] for chunk in chosen} == {
        f"Annexure P-{number}" for number in range(1, 7)
    }


def test_single_annexure_check_keeps_narrow_budget() -> None:
    defect = SimpleNamespace(
        defect="Annexure P-1 heading is missing",
        requirement="Annexure P-1 must have a heading",
        where_to_look=["Check Annexures for the heading."],
        inspect_parts=["Annexures"],
        context_parts=[],
        exclude_parts=None,
        trigger_words=None,
        parent_check_id=None,
    )

    assert max_chunks_for_defect(defect, ceiling=12) == 3


def test_common_index_pagination_check_reads_all_index_pages() -> None:
    defect = SimpleNamespace(
        defect="The common index lacks running pagination",
        requirement="The paper book shall have one common index with pagination",
        where_to_look=["Check the Index and its page numbers."],
        inspect_parts=["Index"],
        context_parts=["Application"],
        exclude_parts=None,
        trigger_words=None,
        parent_check_id=None,
    )
    pool = [
        {
            "record_id": f"index-{page}",
            "score": score,
            "chunk_kind": "page",
            "page": page,
            "document_part": "Index",
            "text": text,
        }
        for page, score, text in (
            (3, 0.99, "INDEX PART I PAGE NO"),
            (4, 0.20, "ANNEXURE P-1 32-81"),
            (5, 0.10, "FILING MEMO 304 VAKALATNAMA 305-306"),
        )
    ]

    assert max_chunks_for_defect(defect, ceiling=12) == 12
    chosen = select_chunks_for_defect(pool, defect, max_chunks=12)
    assert [chunk["page"] for chunk in chosen] == [3, 4, 5]


def test_split_nicknames_come_from_config_not_a_python_map() -> None:
    assert "Vakalatnama" in parts_named_in_text("Go to the V/A")
    assert "Office Report on Limitation" in parts_named_in_text(
        "Go to the O/R on Limitation"
    )
    assert "Listing Proforma" in parts_named_in_text(
        "Check the Proforma for First Listing"
    )


def test_low_confidence_defect_becomes_needs_review() -> None:
    weak = DefectResponse(
        check_id="D003",
        status="defect_found",
        confidence=0.4,
        summary="Looks missing but the quote is thin.",
        reasoning="Partial match only.",
        evidence=[],
    )
    gated = apply_status_policy(weak)
    assert gated.status == "needs_review"


def test_confident_defect_stays_defect_found() -> None:
    strong = DefectResponse(
        check_id="D003",
        status="defect_found",
        confidence=0.9,
        summary="The required declaration is not in the filing.",
        reasoning="Searched the affidavit excerpts; it is not there.",
        evidence=[],
    )
    assert apply_status_policy(strong).status == "defect_found"


def test_evidence_pages_snap_to_excerpt_not_rulebook() -> None:
    chunks = [
        {
            "record_id": "cl1",
            "chunk_kind": "page",
            "page": 2,
            "page_end": 2,
            "document_part": "Advocate's Checklist",
            "text": "ADVOCATE'S CHECK LIST\n1. Whether the petition is in Form 28  YES",
        }
    ]
    response = DefectResponse(
        check_id="D003",
        status="defect_found",
        confidence=0.9,
        summary="Checklist format is incomplete.",
        reasoning="Quoted the checklist excerpt.",
        evidence=[
            EvidenceRef(
                page=5,
                quote="Whether the petition is in Form 28  YES",
            )
        ],
    )
    grounded = apply_evidence_pages(response, chunks)
    assert grounded.evidence[0].page == 2


def test_invented_evidence_page_becomes_null() -> None:
    chunks = [
        {
            "record_id": "cl1",
            "chunk_kind": "page",
            "page": 2,
            "document_part": "Advocate's Checklist",
            "text": "ADVOCATE'S CHECK LIST",
        }
    ]
    response = DefectResponse(
        check_id="D003",
        status="needs_review",
        confidence=0.7,
        summary="Unclear.",
        reasoning="The quote is not in the excerpts.",
        evidence=[EvidenceRef(page=19, quote="something that was never retrieved")],
    )
    grounded = apply_evidence_pages(response, chunks)
    assert grounded.evidence[0].page is None
    assert grounded.evidence[0].quote == "something that was never retrieved"


def test_evidence_keeps_retrieved_page_when_quote_ocr_differs() -> None:
    chunks = [
        {
            "record_id": "aff1",
            "chunk_kind": "page",
            "page": 26,
            "document_part": "Affidavit",
            "text": "AFFIDAVIT truncated pinecone excerpt",
        }
    ]
    response = DefectResponse(
        check_id="D077",
        status="defect_found",
        confidence=0.9,
        summary="Affidavit date cannot be compared.",
        reasoning="The affidavit verification is on filing page 26.",
        evidence=[
            EvidenceRef(
                page=26,
                quote="Verified at Una on 17/4 day of April, 2026",
            )
        ],
    )
    grounded = apply_evidence_pages(response, chunks)
    assert grounded.evidence[0].page == 26


def test_filing_location_states_page_or_page_missing() -> None:
    assert (
        filing_location(
            evidence_pages=[2],
            reviewed_pages=[2],
            document_parts=["Advocate's Checklist"],
        )
        == "Advocate's Checklist, page 2."
    )
    assert (
        filing_location(evidence_pages=[], reviewed_pages=[3, 4], document_parts=[])
        == "Page missing — no page number on the citation. Excerpts were reviewed on pages 3, 4."
    )
    assert (
        filing_location(evidence_pages=[], reviewed_pages=[], document_parts=[])
        == "Page missing — no page was identified in the retrieved excerpts."
    )


def test_published_finding_keeps_one_evidence_item() -> None:
    matched = FindingEvidence(
        page=1,
        quote="the line with a box",
        bounding_boxes=[BoundingBox(page=1, x=0.1, y=0.2, w=0.3, h=0.04)],
        boxes_status="matched",
        document_part="Annexure P-4",
    )
    page_only = FindingEvidence(
        page=1,
        quote="a sentence the page never boxed",
        boxes_status="page_only",
        document_part="Annexure P-4",
    )
    kept = _one_published_evidence([page_only, matched])
    assert len(kept) == 1
    assert kept[0].quote == "the line with a box"
    assert kept[0].boxes_status == "matched"
