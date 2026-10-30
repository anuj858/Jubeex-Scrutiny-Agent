"""Shared Pinecone pool, record slicing, and petition-type system prompts."""

import re
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
from extraction_review.scrutiny.prompts import (
    compose_reasoning,
    filing_location,
    validated_reasoning,
)
from extraction_review.scrutiny.schema import (
    BoundingBox,
    DefectResponse,
    ElementFinding,
    EvidenceRef,
    FindingEvidence,
    _display_evidence_quote,
    _one_published_evidence,
    apply_evidence_capability_policy,
    apply_deterministic_postclassification_policy,
    apply_conditional_manifest_policy,
    apply_evidence_pages,
    apply_grounded_defect_policy,
    apply_internal_contradiction_policy,
    apply_legal_verification_policy,
    apply_margin_layout_policy,
    apply_manifest_absence_policy,
    apply_presence_only_policy,
    apply_review_reason_consistency_policy,
    apply_scope_coverage_policy,
    apply_status_policy,
    apply_supported_evidence_policy,
    apply_visual_conflict_policy,
    apply_visual_field_policy,
    manifest_absence_proven,
    missing_dependency_finding,
    missing_document_finding,
    missing_parts_in_layout,
    review_reason_for_response,
)
from extraction_review.scrutiny_workflow import (
    _absence_owner_check_ids,
    _add_layout_fallback_chunks,
    _complete_layout_chunks_for_defect,
    _complete_evidence_batches,
    _aggregate_complete_batch_responses,
    _retrieval_evidence_complete,
    _visual_index_for_defect,
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
    assert "--- page 4 (overlap) ---" in page3
    assert listing_4[:20] in page3
    assert "Cover page" not in page3
    assert "Petition body" not in page3
    assert by_page[3]["metadata"]["page_start"] == 3
    assert by_page[3]["metadata"]["page_end"] == 4
    assert by_page[3]["metadata"]["pages"] == "3-4"

    page4 = by_page[4]["chunk_text"]
    assert "--- page 3 (overlap) ---" in page4
    assert listing_3[-20:] in page4
    assert "--- page 5 (overlap) ---" not in page4
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


def test_vernacular_annexure_check_uses_collection_budget() -> None:
    defect = SimpleNamespace(
        defect="Translations of vernacular documents are not filed",
        requirement="If there are any vernacular documents, translations must be filed",
        where_to_look=["Check the Annexures for any vernacular document."],
        inspect_parts=["Annexures"],
        context_parts=["Index"],
        exclude_parts=None,
        trigger_words=None,
        parent_check_id=None,
    )

    assert max_chunks_for_defect(defect, ceiling=12) == 12


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


def test_chronology_check_reads_every_available_list_of_dates_page() -> None:
    defect = SimpleNamespace(
        defect="List of dates is not in chronological order",
        requirement="The List of Dates & Events must be in chronological order",
        where_to_look=[
            "List of Dates & Events: inspect every date and event in chronological order."
        ],
        inspect_parts=["List of Dates & Events"],
        context_parts=[],
        exclude_parts=None,
        trigger_words="DATES; EVENTS",
        parent_check_id=None,
    )
    pool = [
        {
            "record_id": f"dates-{page}",
            "score": score,
            "chunk_kind": "page",
            "page": page,
            "document_part": "List of Dates & Events",
            "text": text,
        }
        for page, score, text in (
            (4, 0.99, "DATES EVENTS 01.01.2020 First event"),
            (5, 0.08, "02.01.2020 Second event"),
            (6, 0.07, "03.01.2020 Third event"),
            (7, 0.06, "04.01.2020 Hence the present petition"),
        )
    ]

    assert max_chunks_for_defect(defect, ceiling=12) == 12
    chosen = select_chunks_for_defect(pool, defect, max_chunks=12)

    assert [chunk["page"] for chunk in chosen] == [4, 5, 6, 7]


def test_layout_fallback_supplies_scanned_vakalatnama_when_retrieval_misses_it() -> (
    None
):
    defect = SimpleNamespace(
        inspect_parts=["Vakalatnama"],
        context_parts=[],
        exclude_parts=None,
        where_to_look=["Check the Vakalatnama execution date and signatures."],
    )
    layout = {
        88: {
            "document_part": "Memo of Appearance, Vakalatnama",
            "local_page": 1,
            "words": [
                {"t": "VAKALATNAMA", "line": 1},
                {"t": "Dated", "line": 2},
                {"t": "9", "line": 2},
                {"t": "June", "line": 2},
                {"t": "2024", "line": 2},
            ],
        }
    }

    chunks = _add_layout_fallback_chunks(defect, [], layout=layout, max_chunks=3)

    assert len(chunks) == 1
    assert chunks[0]["page"] == 88
    assert chunks[0]["document_part"] == ["Memo of Appearance", "Vakalatnama"]
    assert "Dated 9 June 2024" in chunks[0]["text"]


def test_complete_retry_uses_all_required_and_context_pages() -> None:
    defect = SimpleNamespace(
        inspect_parts=["Main Petition"],
        context_parts=["Memo of Parties"],
        exclude_parts=None,
        where_to_look=[],
    )
    layout = {
        1: {"document_part": "Main Petition", "markdown": "petition one"},
        2: {"document_part": "Main Petition", "markdown": "petition two"},
        3: {"document_part": "Memo of Parties", "markdown": "party list"},
        4: {"document_part": "Annexure P-1", "markdown": "not required"},
    }

    chunks = _complete_layout_chunks_for_defect(defect, [], layout)

    assert [chunk["page"] for chunk in chunks] == [1, 2, 3]
    assert all(chunk["complete_part_retry"] for chunk in chunks)


def test_complete_evidence_batches_never_drop_oversized_page_text() -> None:
    text = "x" * 130_000
    batches = _complete_evidence_batches(
        [{"page": 7, "document_part": ["Annexures"], "text": text}]
    )
    reconstructed = "".join(
        str(chunk.get("text") or "")
        for batch in batches
        for chunk in batch
        if chunk.get("chunk_kind") != "summary"
    )
    assert reconstructed == text
    assert len(batches) >= 3


def test_complete_batch_aggregation_publishes_any_grounded_local_defect() -> None:
    compliant = DefectResponse(
        check_id="D-68",
        status="compliant",
        confidence=0.9,
        summary="This batch complies.",
        observed="This batch complies.",
        conclusion="",
        evidence=[EvidenceRef(page=1, quote="Translated")],
    )
    defect = DefectResponse(
        check_id="D-68",
        status="defect_found",
        confidence=0.95,
        summary="A vernacular passage has no translation.",
        observed="A vernacular passage has no translation.",
        conclusion="The translation is missing.",
        evidence=[EvidenceRef(page=9, quote="मराठी")],
    )
    assert _aggregate_complete_batch_responses([compliant, defect]) is defect


def test_layout_fallback_keeps_first_and_last_pages_of_missing_part() -> None:
    defect = SimpleNamespace(
        inspect_parts=["Main Petition"],
        context_parts=[],
        exclude_parts=None,
        where_to_look=["Check the opening cause title and drafting date at the end."],
    )
    layout = {
        page: {
            "document_part": "Main Petition",
            "words": [{"t": text, "line": 1}],
        }
        for page, text in ((12, "CAUSE TITLE"), (13, "BODY"), (28, "DATED PRAYER"))
    }

    chunks = _add_layout_fallback_chunks(defect, [], layout=layout, max_chunks=3)

    assert [chunk["page"] for chunk in chunks] == [12, 28]


def test_layout_fallback_reserves_explicit_petition_paragraph_page() -> None:
    defect = SimpleNamespace(
        defect="Paragraph 3 declaration is missing",
        requirement=(
            "Paragraph 3 must contain a declaration in terms of Rule 3(2) "
            "about an earlier petition"
        ),
        inspect_parts=["Main Petition"],
        context_parts=[],
        exclude_parts=None,
        where_to_look=[
            "Check Paragraph 3, 'DECLARATION IN TERMS OF RULE 3(2)' of the SLP."
        ],
    )
    layout = {
        page: {
            "document_part": "Main Petition",
            "words": [{"t": text, "line": 1}],
        }
        for page, text in (
            (74, "SPECIAL LEAVE PETITION"),
            (80, "3. DECLARATION IN TERMS OF RULE 3(2) no earlier petition filed"),
            (82, "DRAWN ON 23.10.2024"),
        )
    }

    chunks = _add_layout_fallback_chunks(defect, [], layout=layout, max_chunks=3)

    assert [chunk["page"] for chunk in chunks] == [74, 80, 82]
    assert "no earlier petition filed" in chunks[1]["text"]


def test_layout_fallback_reads_paragraph_one_after_showeth_page() -> None:
    defect = SimpleNamespace(
        defect="Paragraph 1 omits impugned-order particulars",
        requirement="Paragraph 1 must identify the order under challenge",
        inspect_parts=["Main Petition"],
        context_parts=[],
        exclude_parts=None,
        where_to_look=[
            'Paragraph 1 below "MOST RESPECTFULLY SHOWETH" in the Main Petition.'
        ],
    )
    layout = {
        page: {
            "document_part": "Main Petition",
            "words": [
                {"t": token, "line": line}
                for line, token in enumerate(text.split("\n"), start=1)
            ],
        }
        for page, text in (
            (10, "PETITIONERS\n1. A party named in the cause title"),
            (11, "MOST RESPECTFULLY SHOWETH"),
            (
                12,
                (
                    "1. This petition challenges the order dated 01.01.2024 "
                    "passed by the High Court in W.P. No. 10 of 2024 dismissing the case"
                ),
            ),
            (15, "MAIN PRAYER"),
        )
    }

    chunks = _add_layout_fallback_chunks(defect, [], layout=layout, max_chunks=3)

    assert 12 in {chunk["page"] for chunk in chunks}
    paragraph = next(chunk for chunk in chunks if chunk["page"] == 12)
    assert "challenges the order" in paragraph["text"]


def test_layout_fallback_reads_all_list_of_dates_pages() -> None:
    defect = SimpleNamespace(
        defect="List of dates is not in chronological order",
        requirement="The List of Dates & Events must be in chronological order",
        inspect_parts=["List of Dates & Events"],
        context_parts=[],
        exclude_parts=None,
        where_to_look=[
            "List of Dates & Events: inspect every date and event in chronological order."
        ],
    )
    layout = {
        page: {
            "document_part": "List of Dates & Events",
            "words": [{"t": f"DATE PAGE {page}", "line": 1}],
        }
        for page in range(4, 8)
    }

    chunks = _add_layout_fallback_chunks(defect, [], layout=layout, max_chunks=12)

    assert [chunk["page"] for chunk in chunks] == [4, 5, 6, 7]


def test_exhaustive_check_is_complete_only_after_every_target_page_is_reviewed() -> (
    None
):
    defect = SimpleNamespace(
        defect="List of dates is not in chronological order",
        requirement="Every date must be reviewed in chronological order",
        inspect_parts=["List of Dates & Events"],
        context_parts=[],
        exclude_parts=None,
        where_to_look=["Inspect every date in the List of Dates & Events."],
        trigger_words="DATES; EVENTS",
    )
    layout = {
        page: {
            "document_part": "List of Dates & Events",
            "words": [{"t": f"DATE {page}", "line": 1}],
        }
        for page in (4, 5, 6)
    }
    partial = [
        {
            "chunk_kind": "page",
            "page": page,
            "document_part": "List of Dates & Events",
            "text": f"DATE {page}",
        }
        for page in (4, 6)
    ]
    complete = [
        *partial,
        {
            "chunk_kind": "page",
            "page": 5,
            "document_part": "List of Dates & Events",
            "text": "DATE 5",
        },
    ]

    assert not _retrieval_evidence_complete(defect, partial, layout)
    assert _retrieval_evidence_complete(defect, complete, layout)


def test_exact_signature_anchor_adds_same_part_neighbors_without_crossing_boundary() -> (
    None
):
    defect = SimpleNamespace(
        defect="The execution date is missing",
        requirement="The Vakalatnama must contain the execution date and signature",
        inspect_parts=["Vakalatnama"],
        context_parts=[],
        exclude_parts=None,
        where_to_look=["Check the signing block at the end of the Vakalatnama."],
        trigger_words="Dated this; Sd/-",
    )
    layout = {
        10: {
            "document_part": "Vakalatnama",
            "words": [{"t": "AUTHORIZATION", "line": 1}],
        },
        11: {
            "document_part": "Vakalatnama",
            "words": [{"t": "Dated this 9 June 2024", "line": 1}],
        },
        12: {
            "document_part": "Vakalatnama",
            "words": [{"t": "Sd/- PETITIONER", "line": 1}],
        },
        13: {
            "document_part": "Filing Memo",
            "words": [{"t": "Dated this 10 June 2024", "line": 1}],
        },
    }

    chunks = _add_layout_fallback_chunks(defect, [], layout=layout, max_chunks=3)

    assert [chunk["page"] for chunk in chunks] == [10, 11, 12]
    assert all(chunk["document_part"] == ["Vakalatnama"] for chunk in chunks)


def test_layout_fallback_reserves_main_prayer_page() -> None:
    defect = SimpleNamespace(
        defect="Main prayer is missing",
        requirement="Paragraph 7 must clearly set out the main prayer",
        inspect_parts=["Main Petition"],
        context_parts=[],
        exclude_parts=None,
        where_to_look=["Check Paragraph 7, Main Prayer of the SLP."],
    )
    layout = {
        page: {
            "document_part": "Main Petition",
            "words": [{"t": text, "line": 1}],
        }
        for page, text in (
            (74, "SPECIAL LEAVE PETITION"),
            (81, "7. MAIN PRAYER grant special leave to appeal"),
            (82, "DRAWN ON 23.10.2024"),
        )
    }

    chunks = _add_layout_fallback_chunks(defect, [], layout=layout, max_chunks=3)

    assert [chunk["page"] for chunk in chunks] == [74, 81, 82]


def test_layout_fallback_finds_misnumbered_declaration_heading() -> None:
    defect = SimpleNamespace(
        defect="Paragraph 4 true-copy declaration is missing",
        requirement="Paragraph 4 must contain a declaration in terms of Rule 5",
        inspect_parts=["Main Petition"],
        context_parts=[],
        exclude_parts=None,
        where_to_look=["Check Paragraph 4, Declaration in terms of Rule 5."],
    )
    layout = {
        page: {
            "document_part": "Main Petition",
            "words": [{"t": text, "line": 1}],
        }
        for page, text in (
            (74, "4. Respondent No. 4"),
            (80, "4. DECLARATION IN TERMS OF RULE 105 annexures chronological"),
            (82, "DRAWN ON 23.10.2024"),
        )
    }

    chunks = _add_layout_fallback_chunks(defect, [], layout=layout, max_chunks=3)

    assert 80 in {chunk["page"] for chunk in chunks}


def test_layout_fallback_covers_each_annexure_and_its_last_page() -> None:
    defect = SimpleNamespace(
        defect="Annexures are not true copies",
        requirement="Each Annexure must be a true copy",
        inspect_parts=["Annexures"],
        context_parts=[],
        exclude_parts=None,
        where_to_look=["Check each Annexure for a true copy certification."],
    )
    layout = {
        page: {
            "document_part": label,
            "words": [{"t": text, "line": 1}],
        }
        for page, label, text in (
            (10, "Annexure P-1", "P1 OPEN"),
            (11, "Annexure P-1", "P1 END"),
            (20, "Annexure P-2", "P2 OPEN"),
            (21, "Annexure P-2", "P2 TRUE COPY"),
            (30, "Annexure P-3", "P3 OPEN"),
            (31, "Annexure P-3", "P3 TRUE COPY"),
        )
    }

    chunks = _add_layout_fallback_chunks(defect, [], layout=layout, max_chunks=6)

    assert [chunk["page"] for chunk in chunks] == [10, 11, 20, 21, 30, 31]
    # Certification checks must see the end of every Annexure, not only the
    # first caption page selected by semantic search.
    assert {11, 21, 31}.issubset({chunk["page"] for chunk in chunks})


def test_layout_boundaries_replace_incomplete_semantic_excerpt_set() -> None:
    defect = SimpleNamespace(
        defect="The drafting date is missing",
        requirement="The drafting date must appear at the end of the petition",
        inspect_parts=["Main Petition"],
        context_parts=[],
        exclude_parts=None,
        where_to_look=["Check the date of drafting below the prayer."],
    )
    retrieved = [
        {
            "record_id": "semantic-middle",
            "chunk_kind": "page",
            "page": 18,
            "document_part": "Main Petition",
            "text": "A middle ground paragraph",
        }
    ]
    layout = {
        page: {
            "document_part": "Main Petition",
            "words": [{"t": text, "line": 1}],
        }
        for page, text in ((12, "CAUSE TITLE"), (18, "BODY"), (28, "DATED PRAYER"))
    }

    chunks = _add_layout_fallback_chunks(defect, retrieved, layout=layout, max_chunks=3)

    assert [chunk["page"] for chunk in chunks] == [12, 18, 28]
    assert any(chunk["text"] == "DATED PRAYER" for chunk in chunks)


def test_main_petition_check_uses_petition_affidavit_not_application_affidavit() -> (
    None
):
    defect = SimpleNamespace(
        defect="The petition affidavit omits required particulars",
        requirement="The affidavit supporting the Main Petition must be complete",
        inspect_parts=["Main Petition", "Affidavit"],
        context_parts=[],
        exclude_parts=None,
        where_to_look=["Check the Affidavit filed after the Main Petition."],
    )
    layout = {
        12: {"document_part": "Main Petition", "words": [{"t": "OPEN", "line": 1}]},
        28: {"document_part": "Main Petition", "words": [{"t": "PRAYER", "line": 1}]},
        29: {"document_part": "Affidavit", "words": [{"t": "PETITION", "line": 1}]},
        30: {"document_part": "Affidavit", "words": [{"t": "VERIFY", "line": 1}]},
        82: {"document_part": "Application 1", "words": [{"t": "I.A.", "line": 1}]},
        85: {"document_part": "Affidavit", "words": [{"t": "APPLICATION", "line": 1}]},
        86: {"document_part": "Affidavit", "words": [{"t": "VERIFY", "line": 1}]},
    }

    chunks = _add_layout_fallback_chunks(defect, [], layout=layout, max_chunks=6)
    affidavit_pages = [
        chunk["page"] for chunk in chunks if "Affidavit" in chunk["document_part"]
    ]

    assert affidavit_pages == [29, 30]


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
    assert gated.gap == "low_confidence"
    assert review_reason_for_response(gated) == (
        "The evidence confidence is 40%, below the 60% threshold required to "
        "classify this as a confirmed defect or compliance result."
    )


def test_high_score_review_does_not_claim_score_is_below_threshold() -> None:
    response = DefectResponse(
        check_id="D077",
        status="needs_review",
        confidence=0.9,
        summary="The evidence requires manual review.",
        observed="The relevant affidavit page was not available.",
        gap="low_confidence",
        evidence=[],
    )

    reason = review_reason_for_response(response)

    assert reason is not None
    assert "below the 60% threshold" not in reason
    assert "confidence in sending this finding for manual review" in reason


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


def test_ungrounded_defect_is_not_published() -> None:
    response = DefectResponse(
        check_id="D003",
        status="defect_found",
        confidence=0.95,
        summary="The required endorsement is missing.",
        reasoning="Search did not find it.",
        evidence=[],
    )
    gated = apply_grounded_defect_policy(response)
    assert gated.status == "needs_review"
    assert gated.gap == "element_page_not_in_excerpts"
    assert "No filing-page quotation was grounded" in gated.gap_detail


def test_grounded_blank_field_can_remain_a_defect() -> None:
    response = DefectResponse(
        check_id="D003",
        status="defect_found",
        confidence=0.95,
        summary="The filing date field is blank.",
        reasoning="The filing date label is followed by an empty line.",
        evidence=[EvidenceRef(page=12, quote="Filed on: __________")],
    )
    assert apply_grounded_defect_policy(response).status == "defect_found"


def test_positive_visual_mark_routes_ocr_absence_to_review() -> None:
    defect = SimpleNamespace(
        defect="The affidavit contains a blank date",
        requirement="The affidavit date must be filled",
        inspect_parts=["Affidavit"],
    )
    response = DefectResponse(
        check_id="D-119",
        status="defect_found",
        confidence=0.95,
        summary="The date is blank.",
        reasoning="OCR shows a blank verification date.",
        evidence=[EvidenceRef(page=20, quote="Verified on this ___ day")],
    )
    gated = apply_visual_conflict_policy(
        defect,
        response,
        {
            "marks": [
                {
                    "document_type": "Affidavit",
                    "marking_type": "handwritten_field_value",
                    "associated_label": "verification date",
                    "visible_text": "05 JUN 2026",
                }
            ]
        },
    )
    assert gated.status == "needs_review"
    assert gated.gap == "conflict"
    assert "05 JUN 2026" in gated.gap_detail


def test_d81_legibility_is_not_published_without_page_quality_evidence() -> None:
    defect = SimpleNamespace(check_id="D-81")
    response = DefectResponse(
        check_id="D-81",
        status="defect_found",
        confidence=0.95,
        summary="Annexure P-5 is dim.",
        reasoning="The OCR text appears incomplete.",
        evidence=[EvidenceRef(page=84, quote="Annexure P-5")],
    )

    gated = apply_evidence_capability_policy(defect, response, {"marks": []})

    assert gated.status == "needs_review"
    assert gated.confidence < 0.6
    assert gated.evidence == []
    assert "no page-quality assessment" in gated.reasoning


def test_d93_present_party_section_cannot_be_reported_as_not_filed() -> None:
    defect = SimpleNamespace(check_id="D-93")
    response = DefectResponse(
        check_id="D-93",
        status="defect_found",
        confidence=0.95,
        summary="Memo of parties not filed.",
        reasoning="The petitioner address is incomplete.",
        evidence=[EvidenceRef(page=3, quote="Col. Pawan Kumar ... Petitioner")],
    )
    chunks = [
        {
            "chunk_kind": "page",
            "page": 3,
            "document_part": "Main Petition",
            "text": "IN THE MATTER OF Col. Pawan Kumar ... Petitioner Versus State ... Respondent",
        }
    ]

    decided = apply_presence_only_policy(defect, response, chunks, None)

    assert decided.status == "compliant"
    assert decided.confidence == 0.95
    assert "assessed separately under D-63" in decided.reasoning


def test_defect_reasoning_removes_internal_chunk_and_uses_document_page() -> None:
    defect = SimpleNamespace(
        defect="Paper Book does not have one common index or proper pagination",
        requirement="Paper Book must have one common index with proper pagination",
        location_source="",
    )

    reasoning = validated_reasoning(
        defect,
        (
            "The Index was inspected at pages 92–94 (chunk c1). "
            "Under item 7, the pagination field for Note Sheet is blank"
        ),
        "defect_found",
        evidence_pages=[1],
        document_parts=["Index"],
    )

    assert "chunk c1" not in reasoning
    assert "92" not in reasoning
    assert reasoning.startswith("Observed:")
    assert "Required:" in reasoning
    assert "Conclusion:" in reasoning
    assert "Index, page 1" in reasoning


def test_published_quote_removes_overlap_and_html_noise() -> None:
    quote = (
        "Note Sheet | | --- from page 93 --- <table><tr><td>APPENDIX</td></tr></table>"
    )
    assert _display_evidence_quote(quote) == "Note Sheet"


def test_fallback_reasoning_explains_observed_required_and_conclusion() -> None:
    defect = SimpleNamespace(
        requirement="The AOR code must be stated in the AOR Certificate",
        applicable_rule="",
        defect="AOR code missing",
        inspect_parts=["AOR's Certificate"],
    )
    response = SimpleNamespace(
        status="defect_found",
        observed="The certificate names the advocate and leaves the code blank.",
        conclusion="the Advocate-on-Record code has not been stated",
        gap=None,
        gap_detail="",
        elements=[
            SimpleNamespace(name="AOR code", state="missing", detail=""),
        ],
    )
    evidence = [
        SimpleNamespace(page=18, document_part="AOR's Certificate", quote="Filed by"),
    ]
    reasoning = compose_reasoning(defect, response, evidence=evidence)
    assert "AOR's Certificate" in reasoning
    assert "page 18" in reasoning
    assert "AOR code" in reasoning
    assert "c1" not in reasoning
    assert "excerpt" not in reasoning.casefold()


def test_complete_manifest_can_prove_explicit_missing_document_check() -> None:
    defect = SimpleNamespace(
        check_id="D-61",
        manifest_absence_proves_defect=True,
        inspect_parts=["Advocate's Checklist"],
        context_parts=[],
        where_to_look=[],
        requirement="An Advocate's Checklist must be filed.",
    )
    layout = {
        1: {"document_part": "Cover Page"},
        2: {"document_part": "Index"},
        3: {"document_part": "Main Petition"},
    }
    assert manifest_absence_proven(defect, layout)
    response = DefectResponse(
        check_id="D-61",
        status="needs_review",
        confidence=0.4,
        summary="Checklist was not retrieved.",
        reasoning="Retrieval was inconclusive.",
        evidence=[],
    )
    decided = apply_manifest_absence_policy(defect, response, layout)
    assert decided.status == "defect_found"
    assert decided.confidence == 1.0
    assert decided.summary == "The Advocate's Checklist is missing from the case file."


def test_partial_manifest_does_not_prove_document_absence() -> None:
    defect = SimpleNamespace(
        manifest_absence_proves_defect=True,
        inspect_parts=["Advocate's Checklist"],
        context_parts=[],
        where_to_look=[],
    )
    assert not manifest_absence_proven(
        defect,
        {1: {"document_part": "Cover Page"}, 2: {"document_part": None}},
    )


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
        == "Advocate's Checklist — page 2."
    )
    assert (
        filing_location(evidence_pages=[], reviewed_pages=[3, 4], document_parts=[])
        == "Pages 3–4."
    )
    assert (
        filing_location(evidence_pages=[], reviewed_pages=[], document_parts=[])
        == "No page was identified in the pages reviewed."
    )
    assert (
        filing_location(document_parts=["['Main Petition']"], evidence_pages=[1])
        == "Main Petition — page 1."
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


def _scrutiny_row(**overrides: object) -> SimpleNamespace:
    fields: dict[str, object] = {
        "check_id": "D-179",
        "defect": "paragraph 1A is missing",
        "requirement": (
            "Paragraph 1A must state whether a Letters Patent Appeal lies "
            "(Order XXI Rule 3(5), Supreme Court Rules, 2013)."
        ),
        "applicable_rule": "Order XXI Rule 3(5)",
        "inspect_parts": ["Main Petition"],
        "context_parts": [],
        "where_to_look": [],
        "how_to_cure": [],
        "remediation_config": {},
        "main_category": "SLP (Civil)",
        "special_category": None,
        "location_source": "",
        "location_text": "",
        "defect_version": 1,
        "manifest_absence_proves_defect": False,
    }
    fields.update(overrides)
    return SimpleNamespace(**fields)


def test_needs_review_names_the_unshown_paragraph_without_excerpt_ids() -> None:
    defect = _scrutiny_row()
    response = SimpleNamespace(
        status="needs_review",
        observed="The shown pages contain the heading and the prayers.",
        conclusion="compliance cannot be determined",
        gap="element_page_not_in_excerpts",
        gap_detail="paragraph 1A",
        elements=[],
    )
    chunks = [
        {
            "chunk_kind": "page",
            "page": 12,
            "document_part": "Main Petition",
            "text": "heading",
        },
        {
            "chunk_kind": "page",
            "page": 19,
            "document_part": "Main Petition",
            "text": "prayer",
        },
    ]
    layout = {
        12: {"document_part": "Main Petition", "local_page": 1},
        19: {"document_part": "Main Petition", "local_page": 8},
        20: {"document_part": "Main Petition", "local_page": 9},
    }
    reasoning = compose_reasoning(defect, response, chunks=chunks, layout=layout)
    assert "Main Petition" in reasoning
    assert "pages 1 and 8" in reasoning
    assert "paragraph 1A" in reasoning
    assert "Order XXI Rule 3(5)" in reasoning
    assert "cannot be determined" in reasoning
    assert not re.search(r"\bc\d+\b", reasoning)


def test_missing_part_reasoning_does_not_repeat_the_article() -> None:
    defect = _scrutiny_row(inspect_parts=["Affidavit"])
    response = SimpleNamespace(
        status="needs_review",
        observed="",
        conclusion="",
        gap="part_not_in_excerpts",
        gap_detail="The Affidavit accompanying the Special Leave Petition",
        elements=[],
    )

    reasoning = compose_reasoning(defect, response)

    assert reasoning.startswith(
        "The Affidavit accompanying the Special Leave Petition was not among"
    )
    assert "The The" not in reasoning


def test_inspected_reasoning_normalizes_leading_articles() -> None:
    defect = _scrutiny_row(inspect_parts=["The Affidavit"])
    response = SimpleNamespace(
        status="needs_review",
        observed="",
        conclusion="",
        gap="element_page_not_in_excerpts",
        gap_detail="The signing block",
        elements=[],
    )

    reasoning = compose_reasoning(defect, response)

    assert reasoning.startswith("The Affidavit was inspected")
    assert "include the signing block" in reasoning
    assert "The The" not in reasoning


def test_defect_found_lists_the_missing_provisions() -> None:
    defect = _scrutiny_row(
        check_id="D-226",
        defect="Appendix incomplete",
        requirement='The English version of the relevant provisions must be filed as an "Appendix".',
        applicable_rule="",
        inspect_parts=["Appendix", "Impugned Order"],
    )
    response = SimpleNamespace(
        status="defect_found",
        observed="The Appendix is headed APPENDIX.",
        conclusion="the Appendix is incomplete",
        gap=None,
        gap_detail="",
        elements=[
            SimpleNamespace(
                name="English text of Section 2",
                state="missing",
                detail="the Act referred to in the impugned order",
            )
        ],
    )
    evidence = [SimpleNamespace(page=2, document_part="Appendix", quote="APPENDIX")]
    reasoning = compose_reasoning(defect, response, evidence=evidence)
    assert "Appendix" in reasoning
    assert "page 2" in reasoning
    assert "Section 2" in reasoning
    assert "incomplete" in reasoning
    assert "Order XXI" not in reasoning


def test_defect_reasoning_renumbers_internal_element_labels_for_display() -> None:
    defect = _scrutiny_row(
        defect="Party addresses are incomplete",
        requirement="Every party must have a complete residential or office address.",
        inspect_parts=["Main Petition"],
    )
    response = SimpleNamespace(
        status="defect_found",
        observed="The parties are listed in the Main Petition.",
        conclusion="the party addresses are incomplete",
        gap=None,
        gap_detail="",
        elements=[
            SimpleNamespace(
                name="(b) complete residential or office address for each party",
                state="incomplete",
                detail="The PIN code is missing from the listed addresses.",
            ),
            SimpleNamespace(
                name="(c) email address",
                state="missing",
                detail="No email address is stated.",
            ),
        ],
    )
    evidence = [
        SimpleNamespace(page=1, document_part="Main Petition", quote="Petitioner")
    ]

    reasoning = compose_reasoning(defect, response, evidence=evidence)

    assert "requirement as to" not in reasoning
    assert "(b)" not in reasoning
    assert "(c)" not in reasoning
    assert "not met:." not in reasoning
    assert (
        "not met:\n\n1) complete residential or office address for each party"
        in reasoning
    )
    assert "\n2) email address" in reasoning


def test_single_unmet_element_starts_at_one_not_its_internal_label() -> None:
    defect = _scrutiny_row(
        defect="Party address is incomplete",
        requirement="Every party must have a complete address.",
        inspect_parts=["Main Petition"],
    )
    response = SimpleNamespace(
        status="defect_found",
        observed="",
        conclusion="the party address is incomplete",
        gap=None,
        gap_detail="",
        elements=[
            SimpleNamespace(
                name="b. complete residential or office address for each party",
                state="incomplete",
                detail="The PIN code is missing.",
            )
        ],
    )

    reasoning = compose_reasoning(defect, response)

    assert "not met:." not in reasoning
    assert "not met:\n\n1) complete residential or office address" in reasoning
    assert "\nb." not in reasoning


def test_full_range_is_used_only_when_every_page_was_sent() -> None:
    defect = _scrutiny_row()
    response = SimpleNamespace(
        status="compliant",
        observed="Paragraph 1A states that no Letters Patent Appeal lies.",
        conclusion="",
        gap=None,
        gap_detail="",
        elements=[],
    )
    layout = {
        1: {"document_part": "Main Petition", "local_page": 1},
        2: {"document_part": "Main Petition", "local_page": 2},
    }
    partial = compose_reasoning(
        defect,
        response,
        chunks=[{"chunk_kind": "page", "page": 1, "document_part": "Main Petition"}],
        layout=layout,
    )
    complete = compose_reasoning(
        defect,
        response,
        chunks=[
            {"chunk_kind": "page", "page": 1, "document_part": "Main Petition"},
            {"chunk_kind": "page", "page": 2, "document_part": "Main Petition"},
        ],
        layout=layout,
    )
    assert "in full" not in partial
    assert "inspected in full (pages 1–2)" in complete


def test_missing_appendix_skips_the_model_wording() -> None:
    defect = _scrutiny_row(
        check_id="D-226",
        inspect_parts=["Appendix"],
        manifest_absence_proves_defect=False,
    )
    layout = {
        1: {"document_part": "Main Petition", "local_page": 1},
        2: {"document_part": "Impugned Order", "local_page": 1},
    }
    assert missing_parts_in_layout(defect, layout) == ["Appendix"]
    finding = missing_document_finding(defect, ["Appendix"])
    assert finding.status == "defect_found"
    assert finding.confidence == 1.0
    assert finding.reasoning == "The Appendix is missing from the case file."
    assert finding.evidence == []


def test_numbered_annexures_and_application_count_as_present() -> None:
    annexures = _scrutiny_row(check_id="D-68", inspect_parts=["Annexures"])
    labelled = {
        index: {"document_part": f"Annexure P-{index}"} for index in range(1, 7)
    }
    assert missing_parts_in_layout(annexures, labelled) == []

    application = _scrutiny_row(
        check_id="D-112",
        inspect_parts=["Main Petition", "Application"],
    )
    present = {
        1: {"document_part": "Main Petition"},
        2: {"document_part": "Application 1"},
    }
    assert missing_parts_in_layout(application, present) == []


def test_missing_annexures_are_a_defect() -> None:
    defect = _scrutiny_row(check_id="D-68", inspect_parts=["Annexures"])
    layout = {1: {"document_part": "Main Petition"}}
    assert missing_parts_in_layout(defect, layout) == ["Annexures"]
    finding = missing_document_finding(defect, ["Annexures"])
    assert finding.status == "defect_found"
    assert finding.confidence == 1.0
    assert finding.reasoning == "The Annexures are missing from the case file."


def test_missing_vakalatnama_with_the_switch_on_is_only_that_sentence() -> None:
    defect = _scrutiny_row(
        check_id="D-61",
        inspect_parts=["Vakalatnama"],
        manifest_absence_proves_defect=True,
        requirement="A Vakalatnama must be filed.",
    )
    layout = {1: {"document_part": "Main Petition", "local_page": 1}}
    assert missing_parts_in_layout(defect, layout) == ["Vakalatnama"]
    finding = missing_document_finding(defect, ["Vakalatnama"])
    assert finding.status == "defect_found"
    assert finding.confidence == 1.0
    assert finding.reasoning == "The Vakalatnama is missing from the case file."
    assert finding.location == "Not found in the case file."
    assert finding.evidence == []


def test_missing_document_has_one_presence_owner_and_dependents_are_na() -> None:
    presence = _scrutiny_row(
        check_id="D-74",
        defect="Memo of Appearance is not filed",
        requirement="A Memo of Appearance must be filed.",
        inspect_parts=["Memo of Appearance"],
        manifest_absence_proves_defect=True,
    )
    signature = _scrutiny_row(
        check_id="D-114",
        defect="Memo of Appearance is not signed by the AOR",
        requirement="The Memo of Appearance must be signed.",
        inspect_parts=["Memo of Appearance"],
    )
    layout = {1: {"document_part": "Main Petition"}}

    owners = _absence_owner_check_ids([signature, presence], layout)

    assert owners == {"D-74": ["Memo of Appearance"]}
    skipped = missing_dependency_finding(signature, ["Memo of Appearance"])
    assert skipped.status == "not_applicable"
    assert "dependent check was skipped" in skipped.summary


def test_filing_specific_missing_document_check_owns_global_fallback() -> None:
    global_affidavit = _scrutiny_row(
        check_id="D-64",
        inspect_parts=["Affidavit"],
        manifest_absence_proves_defect=True,
        agent_filing_types=["slp_civil", "slp_criminal", "civil_appeal"],
    )
    slp_civil_affidavit = _scrutiny_row(
        check_id="D-181",
        inspect_parts=["Affidavit"],
        manifest_absence_proves_defect=True,
        agent_filing_types=["slp_civil"],
    )
    layout = {1: {"document_part": "Main Petition"}}

    owners = _absence_owner_check_ids(
        [global_affidavit, slp_civil_affidavit], layout
    )

    assert owners == {"D-181": ["Affidavit"]}


def test_missing_document_without_explicit_presence_owner_is_only_skipped() -> None:
    conditional_presence = _scrutiny_row(
        check_id="D-65",
        defect="A respondent Vakalatnama is not accompanied by a caveat",
        requirement="A caveat must accompany the respondent Vakalatnama.",
        inspect_parts=["Vakalatnama"],
    )
    execution = _scrutiny_row(
        check_id="D-75",
        defect="The Vakalatnama lacks the AOR execution endorsement",
        requirement="The AOR must endorse execution.",
        inspect_parts=["Vakalatnama"],
    )
    layout = {1: {"document_part": "Main Petition"}}

    assert _absence_owner_check_ids(
        [conditional_presence, execution], layout
    ) == {}
    for defect in (conditional_presence, execution):
        finding = missing_dependency_finding(defect, ["Vakalatnama"])
        assert finding.status == "not_applicable"


def test_explicit_multi_part_presence_owner_only_owns_declared_part() -> None:
    impugned_order = _scrutiny_row(
        check_id="D-325",
        defect="The impugned order is not filed",
        requirement="The impugned order must be filed with the petition.",
        inspect_parts=["Impugned Order"],
        context_parts=["Main Petition"],
        manifest_absence_proves_defect=True,
    )

    only_order_missing = {1: {"document_part": "Main Petition"}}
    assert _absence_owner_check_ids([impugned_order], only_order_missing) == {
        "D-325": ["Impugned Order"]
    }

    only_petition_missing = {1: {"document_part": "Impugned Order"}}
    assert _absence_owner_check_ids([impugned_order], only_petition_missing) == {}


def test_layout_fallback_uses_llamaparse_markdown_without_word_boxes() -> None:
    defect = _scrutiny_row(inspect_parts=["Vakalatnama"])
    layout = {
        105: {
            "document_part": "Vakalatnama",
            "markdown": "VAKALATNAMA\nAccepted by the Advocate-on-Record",
        }
    }

    chunks = _add_layout_fallback_chunks(defect, [], layout=layout, max_chunks=2)

    assert len(chunks) == 1
    assert chunks[0]["page"] == 105
    assert chunks[0]["layout_fallback"] is True
    assert "Accepted by" in chunks[0]["text"]


def test_visual_index_is_filtered_to_the_required_document_part() -> None:
    defect = _scrutiny_row(inspect_parts=["Vakalatnama"])
    visual = {
        "status": "ok",
        "pages": [20, 105],
        "targets": [
            {"page": 20, "document_types": ["Main Petition"]},
            {"page": 105, "document_types": ["Vakalatnama"]},
        ],
        "marks": [
            {"page": 20, "marking_type": "signature_like_mark"},
            {"page": 105, "marking_type": "signature_like_mark"},
        ],
        "failures": [{"page": 20, "error": "unrelated page timeout"}],
    }

    filtered = _visual_index_for_defect(defect, visual)

    assert filtered is not None
    assert filtered["pages"] == [105]
    assert [item["page"] for item in filtered["targets"]] == [105]
    assert [item["page"] for item in filtered["marks"]] == [105]
    assert filtered["failures"] == []


def test_one_of_two_documents_missing_reports_only_that_document() -> None:
    defect = _scrutiny_row(
        check_id="D-57",
        inspect_parts=["Vakalatnama", "Main Petition"],
        manifest_absence_proves_defect=False,
    )
    layout = {
        1: {"document_part": "Main Petition", "local_page": 1},
        2: {"document_part": "Index", "local_page": 1},
    }
    missing = missing_parts_in_layout(defect, layout)
    assert missing == ["Vakalatnama"]
    finding = missing_document_finding(defect, missing or [])
    assert finding.status == "defect_found"
    assert finding.confidence == 1.0
    assert finding.reasoning == "The Vakalatnama is missing from the case file."
    assert "Main Petition" not in finding.reasoning


def test_unlabelled_page_does_not_prove_a_document_is_missing() -> None:
    defect = _scrutiny_row(inspect_parts=["Vakalatnama"])
    layout = {
        1: {"document_part": "Main Petition"},
        2: {"document_part": None},
    }
    assert missing_parts_in_layout(defect, layout) is None


def test_address_rows_are_not_quoted_as_a_missing_address() -> None:
    defect = _scrutiny_row(
        check_id="D-63",
        defect=(
            "The Petition, etc., does not contain the complete residential/"
            " office addresses of the parties."
        ),
        requirement=(
            "The Main Petition must give the complete residential or office "
            "address of every party in its array of parties."
        ),
        ai_note="",
    )
    kept = (
        "Column 1: 1.Col. Pawan Kumar Gopinath Sharma, s/o Gopinath Sharma, "
        "Indian Jew\nColumn 2: Respondent No.\nColumn 3: Petitioner"
    )
    response = DefectResponse(
        check_id="D-63",
        status="defect_found",
        confidence=0.95,
        summary="Respondents No. 1 to 9 omit the PIN code.",
        observed="Respondents No. 1 to 9 omit the PIN code.",
        conclusion=(
            "The Main Petition lacks complete address elements such as "
            "locality, state, and PIN codes for the Respondents."
        ),
        elements=[
            ElementFinding(
                name="for each, a complete residential or office address",
                state="missing",
                detail=(
                    "The Petitioner's entry omits the entire address; "
                    "Respondent No. 10's entry omits house, street, locality, "
                    "and state; and Respondents No. 1 to 9 omit the PIN code."
                ),
            )
        ],
        evidence=[
            EvidenceRef(quote=kept, page=1),
            EvidenceRef(
                quote=(
                    "Column 1: 1. K.D.R. Farms Co. op Housing Society Ltd., "
                    "Gut No. 79, Sharnaur, Aurangabad, District Aurangabad, "
                    "through its Chairman\nColumn 2: Petitioner No.\n"
                    "Column 3: Respondent No.\n\n"
                    "Column 1: 2. Rajendra Madhukarrao Kulkarni, age 52 years, "
                    "occupation Business, r/o K.D.R. Farms Co. op. Housing "
                    "Society Ltd. Gut No. 79, Sharnuar, Aurangabad"
                ),
                page=1,
            ),
            EvidenceRef(
                quote=(
                    "Column 1: 6. The Taluka Registrar, Cooperative Societies, "
                    "Aurangabad, Taluka and District Aurangabad\n"
                    "Column 2: Dy.\nColumn 3: Respondent No.\n"
                    "Column 4: Respondent No."
                ),
                page=2,
            ),
            EvidenceRef(quote="Column 1: Aurangabad – 431\n210", page=4),
        ],
    )

    original = list(response.evidence)
    decided = apply_supported_evidence_policy(defect, response)

    assert [item.quote for item in decided.evidence] == [kept]
    assert "Pawan Kumar Gopinath Sharma" in decided.evidence[0].quote
    assert "PIN" not in decided.conclusion
    assert "PIN" not in decided.elements[0].detail
    assert decided.status == "defect_found"

    address_only = DefectResponse(
        check_id="D-63",
        status="defect_found",
        confidence=0.95,
        summary="Respondents No. 1 to 9 omit the PIN code.",
        observed="Respondents No. 1 to 9 omit the PIN code.",
        conclusion="The respondents omit the PIN code.",
        elements=[
            ElementFinding(
                name="for each, a complete residential or office address",
                state="missing",
                detail="Respondents No. 1 to 9 omit the PIN code.",
            )
        ],
        evidence=[
            EvidenceRef(quote=item.quote, page=item.page)
            for item in original
            if "Pawan Kumar" not in (item.quote or "")
        ],
    )
    dropped = apply_supported_evidence_policy(defect, address_only)
    reviewed = apply_grounded_defect_policy(dropped)
    assert dropped.evidence == []
    assert reviewed.status == "needs_review"


def test_reasoning_drops_a_citation_the_requirement_does_not_contain() -> None:
    defect = _scrutiny_row(
        requirement="Paragraph 1A must state whether a Letters Patent Appeal lies.",
        applicable_rule="",
    )
    response = SimpleNamespace(
        status="needs_review",
        observed="Shown pages omit paragraph 1A, contrary to Order XXI Rule 9.",
        conclusion="",
        gap="element_page_not_in_excerpts",
        gap_detail="paragraph 1A",
        elements=[],
    )
    reasoning = compose_reasoning(defect, response)
    assert "Order XXI Rule 9" not in reasoning
    assert "this requirement" in reasoning


def test_legal_verification_metadata_does_not_replace_filing_outcome() -> None:
    defect = _scrutiny_row(legal_verification_status="needs_verification")
    response = DefectResponse(
        check_id=defect.check_id,
        status="defect_found",
        confidence=0.95,
        summary="The field is blank.",
        observed="The field is blank.",
        conclusion="The requirement is not met.",
        evidence=[EvidenceRef(page=1, quote="Date:")],
    )
    gated = apply_legal_verification_policy(defect, response)
    assert gated.status == "defect_found"
    assert gated.gap is None


def test_high_confidence_review_cannot_use_low_confidence_reason() -> None:
    response = DefectResponse(
        check_id="D-89",
        status="needs_review",
        confidence=0.91,
        summary="The margin evidence is inconclusive.",
        observed="The margin evidence is inconclusive.",
        conclusion="Review is required.",
        gap="low_confidence",
        evidence=[EvidenceRef(page=1, quote="SYNOPSIS")],
    )
    checked = apply_review_reason_consistency_policy(response)
    assert checked.status == "needs_review"
    assert checked.gap == "ambiguous"
    assert checked.confidence == 0.91


def test_d89_uses_page_geometry_instead_of_low_confidence_review() -> None:
    defect = _scrutiny_row(
        check_id="D-89",
        defect="Margin of 4 cm on left and right and 2 cm top and bottom not used",
        requirement="Use 4 cm horizontal and 2 cm vertical margins.",
        inspect_parts=["Main Petition"],
    )
    response = DefectResponse(
        check_id="D-89",
        status="needs_review",
        confidence=0.91,
        summary="Margins require review.",
        observed="Margins require review.",
        conclusion="",
        gap="ambiguous",
        evidence=[EvidenceRef(page=1, quote="MOST RESPECTFULLY SHOWETH")],
    )
    layout = {
        1: {
            "width": 595.0,
            "height": 842.0,
            "document_part": ["Main Petition"],
            "words": [
                {"x": 0.10, "y": 0.08, "w": 0.30, "h": 0.03, "text": "MOST"},
                {"x": 0.55, "y": 0.80, "w": 0.30, "h": 0.03, "text": "PRAYER"},
            ],
        }
    }
    decided = apply_margin_layout_policy(defect, response, layout)
    assert decided.status == "defect_found"
    assert decided.gap is None
    assert decided.confidence >= 0.95


def test_incomplete_scope_cannot_publish_compliance() -> None:
    response = DefectResponse(
        check_id="D-1",
        status="compliant",
        confidence=0.95,
        summary="The sampled page complies.",
        observed="The sampled page complies.",
        conclusion="The requirement is satisfied.",
        evidence=[EvidenceRef(page=1, quote="Present")],
    )
    gated = apply_scope_coverage_policy(
        response,
        scope_complete=False,
        prompt_truncated=True,
    )
    assert gated.status == "needs_review"
    assert gated.gap == "element_page_not_in_excerpts"
    assert "evidence limit" in gated.gap_detail


def test_complete_visual_blank_date_is_a_defect() -> None:
    defect = _scrutiny_row(
        defect="Memo does not mention the date of execution",
        requirement="The Memo must mention the date on which it is signed.",
        inspect_parts=["Memo of Appearance"],
    )
    response = DefectResponse(
        check_id=defect.check_id,
        status="needs_review",
        confidence=0.59,
        summary="The date may be handwritten.",
        observed="The field shows Dated:",
        conclusion="The date cannot be determined.",
        gap="unreadable",
        evidence=[EvidenceRef(page=1, quote="Dated:")],
    )
    decided = apply_visual_field_policy(
        defect,
        response,
        {
            "field_assessments": [
                {
                    "page": 1,
                    "document_type": "Memo of Appearance",
                    "field": "execution_date",
                    "label_found": True,
                    "value_present": False,
                    "readable": True,
                    "confidence": 0.94,
                }
            ]
        },
    )
    assert decided.status == "defect_found"
    assert decided.gap is None


def test_d57_owns_blank_vakalatnama_execution_date() -> None:
    defect = _scrutiny_row(
        check_id="D-57",
        defect="Vakalatnama is not of the same or prior date to drafting",
        requirement=(
            "The Vakalatnama execution date must be the same as or earlier "
            "than the Main Petition drafting date."
        ),
        inspect_parts=["Vakalatnama", "Main Petition"],
    )
    response = DefectResponse(
        check_id="D-57",
        status="not_applicable",
        confidence=0.8,
        summary="The comparison dates are incomplete.",
        observed="The Vakalatnama date field is blank.",
        conclusion="The dates cannot be compared.",
        evidence=[],
    )

    decided = apply_visual_field_policy(
        defect,
        response,
        {
            "field_assessments": [
                {
                    "page": 105,
                    "document_type": "Vakalatnama",
                    "field": "execution_date",
                    "label_found": True,
                    "value_present": False,
                    "readable": True,
                    "confidence": 0.96,
                },
                {
                    "page": 80,
                    "document_type": "Main Petition",
                    "field": "execution_date",
                    "label_found": True,
                    "value_present": False,
                    "readable": True,
                    "confidence": 0.96,
                },
            ]
        },
    )

    assert decided.status == "defect_found"
    assert decided.confidence >= 0.9
    assert decided.gap is None


def test_blank_date_is_not_conflicted_by_present_advocate_signature() -> None:
    defect = _scrutiny_row(
        defect="Memo of Appearance does not mention the date of execution",
        requirement="The Memo must mention the date on which it is signed.",
        inspect_parts=["Memo of Appearance"],
    )
    response = DefectResponse(
        check_id=defect.check_id,
        status="defect_found",
        confidence=0.94,
        summary="The execution-date field is blank.",
        observed="The Advocate signature is present and the Dated field is blank.",
        conclusion="The required execution date is missing.",
        evidence=[
            EvidenceRef(
                page=1,
                quote="(AJIT SHARMA) Advocate for the Petitioner\n\nDated:",
            )
        ],
    )
    visual = {
        "status": "ok",
        "pages": [105],
        "targets": [
            {"page": 105, "document_types": ["Memo of Appearance"]}
        ],
        "failures": [],
        "marks": [
            {
                "page": 105,
                "document_type": "Memo of Appearance",
                "marking_type": "advocate_on_record_signature",
                "signature_role": "advocate",
                "associated_label": "Yours faithfully",
            }
        ],
    }
    conflict_checked = apply_visual_conflict_policy(defect, response, visual)
    assert conflict_checked.status == "defect_found"
    decided = apply_visual_field_policy(defect, conflict_checked, visual)
    assert decided.status == "defect_found"
    assert decided.gap is None


def test_complete_d75_endorsement_scope_becomes_defect_not_conflict() -> None:
    defect = _scrutiny_row(
        check_id="D-75",
        defect="AOR has not certified or endorsed due execution",
        requirement=(
            "The AOR must certify execution in his presence or endorse that he "
            "has satisfied himself about due execution."
        ),
        inspect_parts=["Vakalatnama"],
    )
    response = DefectResponse(
        check_id="D-75",
        status="needs_review",
        confidence=0.59,
        summary="A signature-like mark creates a conflict.",
        observed="No endorsement was found.",
        conclusion="Review is required.",
        gap="conflict",
        evidence=[EvidenceRef(page=1, quote="Dated this the 5TH (AJIT SHARMA)")],
    )
    visual = {
        "status": "ok",
        "pages": [1],
        "targets": [{"page": 1, "document_types": ["Vakalatnama"]}],
        "failures": [],
        "marks": [
            {
                "page": 1,
                "document_type": "Vakalatnama",
                "marking_type": "advocate_on_record_signature",
                "signature_role": "advocate",
            }
        ],
    }
    decided = apply_deterministic_postclassification_policy(
        defect,
        response,
        [{"page": 1, "document_part": ["Vakalatnama"], "text": "Dated this the 5TH"}],
        visual,
        scope_complete=True,
    )
    assert decided.status == "defect_found"
    assert decided.gap is None


def test_complete_manifest_without_conditional_trigger_becomes_not_applicable() -> None:
    defect = _scrutiny_row(
        check_id="D-96",
        defect="Substitution application not filed",
        requirement=(
            "Where a party in the court below has died, an Application for "
            "Substitution must be filed."
        ),
        trigger_words="since deceased; through LRs; legal representatives; bringing on record",
        inspect_parts=["Main Petition", "Application"],
    )
    response = DefectResponse(
        check_id="D-96",
        status="needs_review",
        confidence=0.79,
        summary="The death trigger is unresolved.",
        observed="The death trigger is unresolved.",
        conclusion="",
        gap="ambiguous",
        evidence=[EvidenceRef(page=1, quote="APPLICATION FOR INTERIM RELIEF")],
    )
    layout = {
        1: {
            "document_part": ["Main Petition"],
            "text": "The petitioner challenges the final order.",
            "words": [],
        },
        2: {
            "document_part": ["Application"],
            "text": "APPLICATION FOR INTERIM RELIEF",
            "words": [],
        },
    }
    decided = apply_conditional_manifest_policy(defect, response, layout)
    assert decided.status == "not_applicable"
    assert decided.confidence == 1.0


def test_internal_party_side_contradiction_is_defect() -> None:
    defect = _scrutiny_row(
        defect="Memo does not correctly specify the represented side",
        requirement="The Memo must specify whether counsel appears for the Petitioner or Respondent.",
    )
    response = DefectResponse(
        check_id=defect.check_id,
        status="needs_review",
        confidence=0.8,
        summary="The descriptions conflict.",
        observed="The descriptions conflict.",
        conclusion="Review is required.",
        gap="conflict",
        evidence=[
            EvidenceRef(page=1, quote="on behalf of the Respondent"),
            EvidenceRef(page=1, quote="Advocate for the Petitioner"),
        ],
    )
    decided = apply_internal_contradiction_policy(defect, response)
    assert decided.status == "defect_found"
    assert decided.gap is None
