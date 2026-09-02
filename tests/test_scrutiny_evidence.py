"""Shared Pinecone pool, record slicing, and petition-type system prompts."""

from types import SimpleNamespace
from unittest.mock import AsyncMock

import pytest

from extraction_review.document_parts import (
    documents_from_page_parts,
    keep_nearby_scores,
    match_terms_for_defect,
    max_chunks_for_defect,
    overlay_split_documents,
    page_parts_from_split,
    parts_named_in_text,
    parts_named_in_where_to_look,
    pinecone_queries_for_defect,
    pool_search_queries,
    select_chunks_for_defect,
    slice_record_for_defect,
)
from extraction_review.scrutiny.schema import (
    Coverage,
    DefectResponse,
    EvidenceRef,
    apply_evidence_pages,
    apply_retrieval_policy,
    apply_status_policy,
    apply_undetermined_policy,
    build_finding,
)
from extraction_review.process_file import _split_page_parts
from extraction_review.scrutiny.prompts import (
    build_defect_prompt,
    build_system_prompt,
    finding_title,
    filing_location,
    readable_location_source,
    validated_reasoning,
)
from extraction_review.scrutiny.rules import defects_for_filing_type, get_catalogue
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
            split_config=None,
            filename="petition.pdf",
        )


@pytest.mark.asyncio
async def test_split_sends_file_uuid_not_parse_job_id() -> None:
    created = SimpleNamespace(id="split-job-1")
    completed = SimpleNamespace(
        status="COMPLETED",
        result=SimpleNamespace(
            segments=[SimpleNamespace(category="Petition", pages=[1])]
        ),
    )
    split_api = SimpleNamespace(
        create=AsyncMock(return_value=created),
        get=AsyncMock(return_value=completed),
    )
    split_config = SimpleNamespace(
        configuration_id=None,
        categories=[SimpleNamespace(name="Petition")],
        model_dump=lambda **_kwargs: {"categories": [{"name": "Petition"}]},
    )
    mapping = await _split_page_parts(
        SimpleNamespace(split=split_api),
        file_id="aaaaaaaa-bbbb-cccc-dddd-eeeeeeeeeeee",
        split_config=split_config,
        filename="petition.pdf",
    )
    split_api.create.assert_awaited_once()
    kwargs = split_api.create.await_args.kwargs
    assert kwargs["file_input"] == "aaaaaaaa-bbbb-cccc-dddd-eeeeeeeeeeee"
    assert mapping == {1: "Petition"}


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
            5: "Petition",
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


def test_select_chunks_keeps_vakalatnama_for_date_check() -> None:
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
            "record_id": "aff1",
            "chunk_kind": "page",
            "page": 39,
            "document_part": "Affidavit",
            "text": "I have gone through the accompanying petition",
            "score": 0.99,
        },
        {
            "record_id": "aff2",
            "chunk_kind": "page",
            "page": 40,
            "document_part": "Affidavit",
            "text": "Verified at Nainital",
            "score": 0.98,
        },
        {
            "record_id": "aff3",
            "chunk_kind": "page",
            "page": 41,
            "document_part": "Affidavit",
            "text": "Deponent",
            "score": 0.97,
        },
        {
            "record_id": "pet",
            "chunk_kind": "page",
            "page": 16,
            "document_part": "Petition",
            "text": "Place: New Delhi Dated 10.04.2026 below the prayer",
            "score": 0.4,
        },
        {
            "record_id": "vak",
            "chunk_kind": "page",
            "page": 50,
            "document_part": "Vakalatnama + PoA/BR",
            "text": "VAKALATNAMA Dated this on 10th day of April 2026",
            "score": 0.2,
        },
    ]
    chunks = select_chunks_for_defect(
        pool, catalogue.defect("D013"), max_chunks=12
    )
    ids = {c["record_id"] for c in chunks}
    assert "vak" in ids
    assert "pet" in ids


def test_documents_from_split_include_page_spans() -> None:
    docs = documents_from_page_parts(
        {
            3: "Office Report on Limitation",
            4: "Office Report on Limitation",
            50: "Vakalatnama + PoA/BR",
        }
    )
    assert docs["count"] == 2
    assert docs["items"][0] == "Office Report on Limitation (pp. 3–4)"
    assert docs["items"][1] == "Vakalatnama + PoA/BR (p. 50)"


def test_overlay_split_documents_replaces_index_slang() -> None:
    payload = {
        "data": {
            "petition_type": "SLP_CIVIL",
            "filing_summary": {
                "documents": {"count": 1, "items": ["V/A"]},
            },
        }
    }
    overlay_split_documents(payload, {50: "Vakalatnama + PoA/BR"})
    items = payload["data"]["filing_summary"]["documents"]["items"]
    assert items == ["Vakalatnama + PoA/BR (p. 50)"]
    assert "V/A" not in items


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


def test_max_chunks_is_tighter_for_single_point_defects() -> None:
    catalogue = get_catalogue()
    assert max_chunks_for_defect(catalogue.defect("D004"), ceiling=12) == 3
    assert max_chunks_for_defect(catalogue.defect("D005"), ceiling=12) == 6
    assert max_chunks_for_defect(catalogue.defect("D006"), ceiling=12) == 6


def test_child_defects_run_immediately_after_parent() -> None:
    ids = [d.check_id for d in defects_for_filing_type("SLP_CIVIL")]
    assert "D001" not in ids
    assert "D002" not in ids
    assert ids.index("D003") + 1 == ids.index("D005")


def test_match_terms_use_form_captions_not_objection_text() -> None:
    catalogue = get_catalogue()
    terms = match_terms_for_defect(catalogue.defect("D004"))
    assert any("Listing Proforma" in t for t in terms)
    assert all("not duly filled" not in t.lower() for t in terms)


def test_pinecone_queries_follow_where_to_look() -> None:
    catalogue = get_catalogue()
    d013 = pinecone_queries_for_defect(catalogue.defect("D013"))
    assert any("Vakalatnama" in q for q in d013)
    assert any("Petition" in q or "petition" in q.lower() for q in d013)
    d047 = pinecone_queries_for_defect(catalogue.defect("D047"))
    assert any("Memo of Appearance" in q for q in d047)
    d017 = pinecone_queries_for_defect(catalogue.defect("D017"))
    assert any("Office Report on Limitation" in q for q in d017)
    named = parts_named_in_where_to_look(catalogue.defect("D013"))
    assert "Vakalatnama + PoA/BR" in named
    assert named[0] in {"Petition", "Vakalatnama + PoA/BR"}
    assert "Memo of Parties" not in named


def test_landmarks_are_not_required_parts() -> None:
    catalogue = get_catalogue()
    d003 = parts_named_in_where_to_look(catalogue.defect("D003"))
    assert d003 == ["Advocate's Checklist"]
    d005 = parts_named_in_where_to_look(catalogue.defect("D005"))
    assert "Advocate's Checklist" in d005
    assert "Vakalatnama + PoA/BR" in d005
    queries = pinecone_queries_for_defect(catalogue.defect("D003"))
    assert any("Checklist" in q or "Check List" in q for q in queries)
    assert all("Cover Page" not in q for q in queries)
    assert all("Office Report on Limitation" not in q for q in queries)


def test_split_nicknames_come_from_config_not_a_python_map() -> None:
    assert "Vakalatnama + PoA/BR" in parts_named_in_text("Go to the V/A")
    assert "Office Report on Limitation" in parts_named_in_text(
        "Go to the O/R on Limitation"
    )
    assert "Listing Proforma" in parts_named_in_text(
        "Check the Proforma for First Listing"
    )


def test_select_chunks_drops_far_pages_and_respects_defect_budget() -> None:
    catalogue = get_catalogue()
    pool: list[dict] = [
        {
            "record_id": "s",
            "chunk_kind": "summary",
            "text": "SLP Civil summary",
            "score": 0.1,
            "page": None,
        }
    ]
    # Two near hits, then a cliff — budget is 3, but far pages must not fill it.
    scores = (0.91, 0.88, 0.35, 0.30, 0.22, 0.18, 0.12, 0.08)
    for i, score in enumerate(scores, start=1):
        pool.append(
            {
                "record_id": f"p{i}",
                "chunk_kind": "page",
                "page": i,
                "document_part": "Listing Proforma",
                "text": "Proforma for First Listing columns 6 and 7",
                "score": score,
            }
        )
    chunks = select_chunks_for_defect(
        pool, catalogue.defect("D004"), max_chunks=12
    )
    ids = [c["record_id"] for c in chunks]
    assert ids[0] == "s"
    assert ids[1:] == ["p1", "p2"]
    assert "p3" not in ids


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
    assert "retrieval gap" in system.lower()
    assert "stamps, seals, signatures" in system.lower() or "stamp" in system.lower()
    assert "those checks alone are not_determined" not in system.lower()
    assert "evidence.page" in system.lower() or "excerpt header" in system.lower()
    prompt_l = prompt.lower()
    assert "needs_review" in prompt_l
    assert "return defect_found" in prompt_l
    assert "rulebook locator" in prompt_l
    assert "not pages of this filing" in prompt_l
    assert "place in that source" not in prompt_l
    assert "[page 1 — advocate's checklist]" in prompt_l
    assert "excerpt header" in prompt_l


def test_low_confidence_defect_becomes_needs_review() -> None:
    weak = DefectResponse(
        check_id="D003",
        status="defect_found",
        confidence=0.4,
        summary="Looks missing but the quote is thin.",
        reasoning="Partial match only.",
        evidence=[],
        suggested_fix="Add the missing heading.",
        fix_rationale="Required by the rule.",
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
        suggested_fix="File the declaration.",
        fix_rationale="The rule requires it.",
    )
    assert apply_status_policy(strong).status == "defect_found"


def test_missing_vakalatnama_excerpts_are_needs_review_not_defect() -> None:
    catalogue = get_catalogue()
    response = DefectResponse(
        check_id="D013",
        status="defect_found",
        confidence=1.0,
        summary="The Vakalatnama is missing from the extracted filing.",
        reasoning="Neither the Vakalatnama nor the drafting date is present.",
        evidence=[],
        suggested_fix="File a Vakalatnama.",
        fix_rationale="Required by the rule.",
    )
    affidavit_only = [
        {
            "record_id": "aff1",
            "chunk_kind": "page",
            "page": 39,
            "document_part": "Affidavit",
            "text": "I have gone through the petition",
        }
    ]
    gated = apply_retrieval_policy(catalogue.defect("D013"), response, affidavit_only)
    assert gated.status == "needs_review"


def test_checklist_defect_does_not_need_cover_page_excerpts() -> None:
    catalogue = get_catalogue()
    response = DefectResponse(
        check_id="D003",
        status="defect_found",
        confidence=0.95,
        summary="The Advocate's Check List is not in the prescribed format.",
        reasoning="The checklist excerpt has no YES/NO/N.A. rows.",
        evidence=[
            EvidenceRef(page=2, quote="ADVOCATE'S CHECK LIST without YES/NO columns")
        ],
        suggested_fix="File the checklist in the prescribed format.",
        fix_rationale="Required by the rule.",
    )
    checklist_only = [
        {
            "record_id": "cl1",
            "chunk_kind": "page",
            "page": 2,
            "document_part": "Advocate's Checklist",
            "text": "ADVOCATE'S CHECK LIST without YES/NO columns",
        }
    ]
    gated = apply_retrieval_policy(catalogue.defect("D003"), response, checklist_only)
    assert gated.status == "defect_found"


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
        suggested_fix="File the prescribed checklist.",
        fix_rationale="Required.",
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
        suggested_fix=None,
        fix_rationale=None,
    )
    grounded = apply_evidence_pages(response, chunks)
    assert grounded.evidence[0].page is None
    assert grounded.evidence[0].quote == "something that was never retrieved"


def test_missing_stamp_is_defect_not_undetermined() -> None:
    catalogue = get_catalogue()
    vakalatnama = [
        {
            "record_id": "vak1",
            "chunk_kind": "page",
            "page": 50,
            "document_part": "Vakalatnama + PoA/BR",
            "text": "VAKALATNAMA I appoint the advocate",
        }
    ]
    timid = DefectResponse(
        check_id="D040",
        status="not_determined",
        confidence=0.4,
        summary="Cannot see whether a welfare stamp is affixed.",
        reasoning="Stamps are visual.",
        evidence=[],
        suggested_fix=None,
        fix_rationale=None,
    )
    gated = apply_undetermined_policy(catalogue.defect("D040"), timid, vakalatnama)
    assert gated.status == "defect_found"


def test_undetermined_without_the_part_stays_needs_review() -> None:
    catalogue = get_catalogue()
    petition_only = [
        {
            "record_id": "p1",
            "chunk_kind": "page",
            "page": 10,
            "document_part": "Petition",
            "text": "SPECIAL LEAVE PETITION",
        }
    ]
    timid = DefectResponse(
        check_id="D040",
        status="not_determined",
        confidence=0.5,
        summary="No Vakalatnama excerpts.",
        reasoning="Cannot see a stamp.",
        evidence=[],
        suggested_fix=None,
        fix_rationale=None,
    )
    gated = apply_undetermined_policy(catalogue.defect("D040"), timid, petition_only)
    assert gated.status == "needs_review"


def test_visual_defects_query_stamp_and_margin_cues() -> None:
    catalogue = get_catalogue()
    stamp_queries = pinecone_queries_for_defect(catalogue.defect("D040"))
    assert any("stamp" in q.lower() for q in stamp_queries)
    margin_queries = pinecone_queries_for_defect(catalogue.defect("D006"))
    assert any("margin" in q.lower() or "a4" in q.lower() for q in margin_queries)


def test_visual_prompt_treats_missing_marks_as_defects() -> None:
    catalogue = get_catalogue()
    system = build_system_prompt(catalogue, "SLP_CIVIL")
    assert "not not_determined" in system.lower() or "defect_found, not not_determined" in system.lower()
    prompt = build_defect_prompt(
        catalogue.defect("D006"),
        record={"petition_type": "SLP_CIVIL"},
        chunks=[
            {
                "chunk_kind": "page",
                "page": 12,
                "document_part": "Petition",
                "text": "1. The petitioner states",
            }
        ],
        catalogue=catalogue,
    )
    prompt_l = prompt.lower()
    assert "stamp, seal, signature, paper size, margin" in prompt_l
    assert "use not_determined only for stamps" not in prompt_l


def test_finding_title_is_short_and_named_for_the_defect() -> None:
    catalogue = get_catalogue()
    d003 = finding_title(catalogue.defect("D003"), catalogue)
    assert d003.startswith("Advocate's Check List:")
    assert "checklist" in d003.lower() or "check list" in d003.lower()
    assert len(d003) < len(catalogue.defect("D003").defect) + 40
    d004 = finding_title(catalogue.defect("D004"), catalogue)
    assert d004.startswith("Listing Proforma:")
    assert "6" in d004 and "7" in d004


def test_location_source_is_official_not_filing_page() -> None:
    catalogue = get_catalogue()
    loc = readable_location_source(catalogue.defect("D003"), catalogue)
    assert loc.startswith("Official source (not a page of this filing):")
    assert "SCI_CHECKLIST_2025" not in loc
    assert "Check List" in loc or "checklist" in loc.lower()
    assert "page 5" in loc.lower()


def test_filing_location_states_page_or_page_missing() -> None:
    assert (
        filing_location(
            evidence_pages=[2],
            reviewed_pages=[2],
            document_parts=["Advocate's Checklist"],
        )
        == "Filing page 2 — Advocate's Checklist."
    )
    assert (
        filing_location(evidence_pages=[], reviewed_pages=[3, 4], document_parts=[])
        == "Filing page missing — no page number on the citation. Excerpts were reviewed on pages 3, 4."
    )
    assert (
        filing_location(evidence_pages=[], reviewed_pages=[], document_parts=[])
        == "Filing page missing — no page was identified in the retrieved excerpts."
    )


def test_weak_reasoning_is_rewritten_from_the_defect() -> None:
    catalogue = get_catalogue()
    defect = catalogue.defect("D040")
    rewritten = validated_reasoning(
        defect,
        "Stamps are visual. See SCI_CHECKLIST_2025 Page 5 of the PDF.",
        "defect_found",
        pages=[50],
        evidence_pages=[50],
    )
    assert "SCI_CHECKLIST" not in rewritten
    assert "welfare stamp" in rewritten.lower() or "vakalatnama" in rewritten.lower()
    assert "filing page 50" in rewritten.lower()


def test_build_finding_validates_title_reasoning_and_source() -> None:
    catalogue = get_catalogue()
    defect = catalogue.defect("D003")
    finding = build_finding(
        defect,
        DefectResponse(
            check_id="D003",
            status="defect_found",
            confidence=0.9,
            summary="Cannot see the checklist.",
            reasoning="Visual / not_determined. Page 5 of the PDF SCI_CHECKLIST_2025.",
            evidence=[EvidenceRef(page=2, quote="ADVOCATE'S CHECK LIST")],
            suggested_fix="File the prescribed checklist.",
            fix_rationale="Required.",
        ),
        evidence_ids=["cl1"],
        coverage=Coverage(chunks_reviewed=1, pages_reviewed=[2]),
        chunks=[
            {
                "record_id": "cl1",
                "chunk_kind": "page",
                "page": 2,
                "document_part": "Advocate's Checklist",
                "text": "ADVOCATE'S CHECK LIST",
            }
        ],
    )
    assert finding.title.startswith("Advocate's Check List:")
    assert finding.location == "Filing page 2 — Advocate's Checklist."
    assert finding.location_source and finding.location_source.startswith(
        "Official source (not a page of this filing):"
    )
    assert "SCI_CHECKLIST_2025" not in (finding.reasoning or "")
    assert "checklist" in finding.reasoning.lower() or "check list" in finding.reasoning.lower()
    assert "filing page 2" in finding.reasoning.lower()


def test_build_finding_says_page_missing_without_citation() -> None:
    catalogue = get_catalogue()
    finding = build_finding(
        catalogue.defect("D003"),
        DefectResponse(
            check_id="D003",
            status="needs_review",
            confidence=0.7,
            summary="The checklist excerpts are incomplete.",
            reasoning="The Advocate's Check List excerpts do not identify a page.",
            evidence=[EvidenceRef(page=None, quote="ADVOCATE'S CHECK LIST")],
            suggested_fix=None,
            fix_rationale=None,
        ),
        evidence_ids=[],
        coverage=Coverage(chunks_reviewed=0, pages_reviewed=[]),
    )
    assert finding.location == (
        "Filing page missing — no page was identified in the retrieved excerpts."
    )


def test_build_finding_uses_chunk_pages_when_citation_has_none() -> None:
    catalogue = get_catalogue()
    finding = build_finding(
        catalogue.defect("D003"),
        DefectResponse(
            check_id="D003",
            status="needs_review",
            confidence=0.7,
            summary="The checklist excerpts are incomplete.",
            reasoning="The Advocate's Check List excerpts do not identify a page.",
            evidence=[EvidenceRef(page=None, quote="ADVOCATE'S CHECK LIST")],
            suggested_fix=None,
            fix_rationale=None,
        ),
        evidence_ids=["cl1"],
        coverage=Coverage(chunks_reviewed=1, pages_reviewed=[]),
        chunks=[
            {
                "record_id": "cl1",
                "chunk_kind": "page",
                "page": 4,
                "document_part": "Advocate's Checklist",
                "text": "ADVOCATE'S CHECK LIST",
            }
        ],
    )
    assert finding.location == (
        "Filing page missing — no page number on the citation. "
        "Excerpts were reviewed on page 4 — Advocate's Checklist."
    )
