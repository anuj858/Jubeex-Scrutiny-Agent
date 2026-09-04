"""Already-split upload catalog, validation, extract pack, and document_part stamps."""

from __future__ import annotations

import inspect
from io import BytesIO

import pytest
from pypdf import PdfReader, PdfWriter
from workflows.events import StartEvent

from extraction_review.bundle_slicer import (
    extract_pdf_pages,
    leftover_pages,
    map_slot_pages,
    slice_bundle_pdf,
)
from extraction_review.document_parts import format_page_span, overlay_split_documents
from extraction_review.metadata_workflow import workflow as metadata_workflow
from extraction_review.process_file import ProcessFileWorkflow
from extraction_review.process_split_files import (
    ProcessSplitFilesWorkflow,
    SplitFilesState,
    SplitPartEvent,
    extract_input_file_id,
)
from extraction_review.split_upload import (
    SplitPartInput,
    SplitUploadError,
    build_extract_pack_markdown,
    bundle_file_hash,
    coerce_page_markdown,
    coerce_page_parts,
    extract_source_parts,
    inject_where_to_look,
    stitch_parsed_parts,
    type_catalog,
    ui_catalog,
    validate_parts,
)
from extraction_review.vector_store import build_page_records


def _required_parts(
    filing_type: str,
    *,
    extra: list[dict] | None = None,
    omit: set[str] | None = None,
) -> list[dict]:
    catalog = type_catalog(filing_type)
    skip = omit or set()
    parts = [
        {
            "slot_id": slot.id,
            "file_id": f"file-{slot.id}",
            "file_hash": f"hash-{slot.id}",
            "filename": f"{slot.id}.pdf",
        }
        for slot in catalog.slots
        if slot.required and slot.id not in skip
    ]
    if extra:
        parts.extend(extra)
    return parts


def test_ui_catalog_is_driven_by_config_types() -> None:
    catalog = ui_catalog()
    assert set(catalog) == {"SLP_CIVIL", "SLP_CRIMINAL"}
    assert catalog["SLP_CIVIL"]["label"] == "SLP (Civil)"
    civil_ids = [slot["id"] for slot in catalog["SLP_CIVIL"]["slots"]]
    criminal_ids = [slot["id"] for slot in catalog["SLP_CRIMINAL"]["slots"]]
    assert "court_fees" in civil_ids
    assert "court_fees" not in criminal_ids
    assert civil_ids[-1] == "undefined"
    assert criminal_ids[-1] == "undefined"
    civil_required = {
        slot["id"]: slot["required"] for slot in catalog["SLP_CIVIL"]["slots"]
    }
    criminal_required = {
        slot["id"]: slot["required"] for slot in catalog["SLP_CRIMINAL"]["slots"]
    }
    assert civil_required["memo_of_parties"] is False
    assert civil_required["court_fees"] is False
    assert civil_required["undefined"] is False
    assert civil_required["petition"] is True
    assert criminal_required["memo_of_parties"] is False
    assert criminal_required["vakalatnama"] is True
    assert criminal_required["poa_br"] is False
    assert criminal_required["undefined"] is False
    assert "court_fees" not in criminal_required
    assert civil_required["vakalatnama"] is True
    assert civil_required["poa_br"] is False
    assert "poa_br" in civil_ids
    assert "poa_br" in criminal_ids


def test_slp_civil_accepts_required_slots_without_optional_annexures() -> None:
    catalog, parts = validate_parts("SLP_CIVIL", _required_parts("SLP_CIVIL"))
    assert catalog.filing_type == "SLP_CIVIL"
    present = {item.slot_id for item in parts}
    assert "annexures" not in present
    assert "memo_of_parties" not in present
    assert "court_fees" not in present
    assert "poa_br" not in present
    assert "undefined" not in present
    assert "vakalatnama" in present


def test_slp_civil_allows_optional_memo_of_parties_and_court_fees() -> None:
    extra = [
        {
            "slot_id": "memo_of_parties",
            "file_id": "file-memo-of-parties",
        },
        {
            "slot_id": "court_fees",
            "file_id": "file-court-fees",
        },
        {
            "slot_id": "poa_br",
            "file_id": "file-poa-br",
        },
    ]
    _, parts = validate_parts(
        "SLP_CIVIL",
        _required_parts("SLP_CIVIL", extra=extra),
    )
    present = {item.slot_id for item in parts}
    assert "memo_of_parties" in present
    assert "court_fees" in present
    assert "poa_br" in present


def test_slp_criminal_omits_court_fees_and_rejects_it() -> None:
    _, parts = validate_parts("SLP_CRIMINAL", _required_parts("SLP_CRIMINAL"))
    present = {item.slot_id for item in parts}
    assert "court_fees" not in present
    assert "memo_of_parties" not in present
    assert "poa_br" not in present
    assert "vakalatnama" in present
    with pytest.raises(SplitUploadError, match="Unknown slot"):
        validate_parts(
            "SLP_CRIMINAL",
            _required_parts(
                "SLP_CRIMINAL",
                extra=[
                    {
                        "slot_id": "court_fees",
                        "file_id": "file-court-fees",
                    }
                ],
            ),
        )


def test_missing_required_petition_fails() -> None:
    with pytest.raises(SplitUploadError, match="Petition"):
        validate_parts("SLP_CIVIL", _required_parts("SLP_CIVIL", omit={"petition"}))


def test_compiled_slices_skip_missing_required_slots() -> None:
    catalog, parts = validate_parts(
        "SLP_CIVIL",
        [
            {"slot_id": "cover_page", "file_id": "file-cover"},
            {"slot_id": "petition", "file_id": "file-petition"},
        ],
        require_all_slots=False,
    )
    assert catalog.filing_type == "SLP_CIVIL"
    assert {item.slot_id for item in parts} == {"cover_page", "petition"}


def test_unknown_filing_type_fails() -> None:
    with pytest.raises(SplitUploadError, match="Unknown filing type"):
        validate_parts("WRIT_PETITION_CIVIL", [])


def test_duplicate_slot_fails() -> None:
    parts = _required_parts("SLP_CRIMINAL")
    parts.append({"slot_id": "petition", "file_id": "file-petition-2"})
    with pytest.raises(SplitUploadError, match="Duplicate slot"):
        validate_parts("SLP_CRIMINAL", parts)


def test_vakalatnama_and_poa_br_are_separate_slots() -> None:
    extra = [{"slot_id": "poa_br", "file_id": "file-poa-br"}]
    catalog, parts = validate_parts(
        "SLP_CRIMINAL",
        _required_parts("SLP_CRIMINAL", extra=extra),
    )
    pages_by_slot = {item.slot_id: {1: f"text for {item.slot_id}"} for item in parts}
    _markdown, page_parts = stitch_parsed_parts(catalog, parts, pages_by_slot)
    vakalatnama_pages = [
        page for page, names in page_parts.items() if names == ["Vakalatnama"]
    ]
    poa_pages = [page for page, names in page_parts.items() if names == ["PoA/BR"]]
    assert len(vakalatnama_pages) == 1
    assert len(poa_pages) == 1
    assert vakalatnama_pages[0] != poa_pages[0]


def test_synopsis_slot_stamps_both_document_parts() -> None:
    catalog, parts = validate_parts("SLP_CIVIL", _required_parts("SLP_CIVIL"))
    pages_by_slot = {item.slot_id: {1: f"text for {item.slot_id}"} for item in parts}
    pages_by_slot["synopsis_lod"] = {1: "synopsis page", 2: "list of dates page"}
    page_markdown, page_parts = stitch_parsed_parts(catalog, parts, pages_by_slot)
    synopsis_pages = [page for page, names in page_parts.items() if "Synopsis" in names]
    assert len(synopsis_pages) == 2
    for page in synopsis_pages:
        assert page_parts[page] == ["Synopsis", "List of Dates & Events"]
        assert page_markdown[page]

    records = build_page_records(
        base_id="bundle",
        page_markdown={page: page_markdown[page] for page in synopsis_pages},
        page_parts=page_parts,
    )
    assert records
    assert records[0]["metadata"]["document_part"] == [
        "Synopsis",
        "List of Dates & Events",
    ]


def test_extract_pack_keeps_source_parts_and_drops_noise() -> None:
    catalog = type_catalog("SLP_CIVIL")
    page_markdown = {
        1: "cover caption",
        2: "index listing",
        3: "listing columns",
        4: "petition grounds",
        5: "annexure p-1",
        6: "appendix text",
    }
    page_parts = {
        1: ["Cover Page"],
        2: ["Index"],
        3: ["Listing Proforma"],
        4: ["Petition"],
        5: ["Annexures"],
        6: ["Appendix"],
    }
    pack = build_extract_pack_markdown(
        page_markdown, page_parts, extract_source_parts(catalog)
    )
    assert "[Cover Page]" in pack
    assert "[Listing Proforma]" in pack
    assert "[Petition]" not in pack
    assert "[Index]" not in pack
    assert "petition grounds" not in pack
    assert "index listing" not in pack
    assert "annexure p-1" not in pack
    assert "appendix text" not in pack
    assert "Annexures" not in pack
    assert "Appendix" not in pack


def test_extract_source_parts_omit_index_and_petition() -> None:
    parts = extract_source_parts(type_catalog("SLP_CIVIL"))
    assert parts == {
        "Cover Page",
        "Listing Proforma",
        "Memo of Parties",
        "Impugned Order",
        "Vakalatnama",
        "AOR's Declaration",
    }
    assert "Undefined" not in parts


def test_inject_where_to_look_appends_field_guidance() -> None:
    schema = {
        "properties": {
            "cause_title": {"description": "Cause title."},
            "court": {"description": "The court."},
        }
    }
    updated = inject_where_to_look(
        schema,
        {"cause_title": ["Memo of Parties", "Cover Page"]},
    )
    assert (
        "Look only in Memo of Parties, Cover Page"
        in updated["properties"]["cause_title"]["description"]
    )
    assert "Look only in" not in updated["properties"]["court"]["description"]
    assert schema["properties"]["cause_title"]["description"] == "Cause title."


def test_overlay_uses_stitched_document_parts() -> None:
    catalog, parts = validate_parts(
        "SLP_CIVIL",
        _required_parts("SLP_CIVIL"),
    )
    pages_by_slot = {item.slot_id: {1: item.slot_id} for item in parts}
    _markdown, page_parts = stitch_parsed_parts(catalog, parts, pages_by_slot)
    payload = {"filing_summary": {}}
    overlay_split_documents(payload, page_parts)
    items = payload["filing_summary"]["documents"]["items"]
    assert any(item.startswith("Petition") for item in items)
    assert any(item.startswith("Synopsis") for item in items)
    assert any("List of Dates & Events" in item for item in items)


def test_bundle_hash_is_stable() -> None:
    parts = [
        SplitPartInput(slot_id="b", file_id="2", file_hash="h2"),
        SplitPartInput(slot_id="a", file_id="1", file_hash="h1"),
    ]
    assert bundle_file_hash(parts) == bundle_file_hash(list(reversed(parts)))


def test_empty_parse_still_stamps_document_part() -> None:
    catalog, parts = validate_parts(
        "SLP_CRIMINAL",
        [
            {
                "slot_id": "petition",
                "file_id": "file-petition",
                "filename": "petition.pdf",
            },
            *[
                item
                for item in _required_parts("SLP_CRIMINAL")
                if item["slot_id"] != "petition"
            ],
        ],
    )
    pages_by_slot = {
        item.slot_id: {} if item.slot_id == "petition" else {1: "ok"} for item in parts
    }
    page_markdown, page_parts = stitch_parsed_parts(catalog, parts, pages_by_slot)
    petition_pages = [
        page for page, names in page_parts.items() if names == ["Petition"]
    ]
    assert len(petition_pages) == 1
    assert "No parse text" in page_markdown[petition_pages[0]]
    pack = build_extract_pack_markdown(
        page_markdown, page_parts, extract_source_parts(catalog)
    )
    assert "No parse text" not in pack


def test_page_maps_survive_string_keys() -> None:
    markdown = coerce_page_markdown({"1": "cover", "2": "petition body"})
    parts = coerce_page_parts({"1": "Cover Page", "2": ["Petition"]})
    assert markdown == {1: "cover", 2: "petition body"}
    assert parts == {1: ["Cover Page"], 2: ["Petition"]}
    records = build_page_records(
        base_id="bundle",
        page_markdown=markdown,
        page_parts=parts,
    )
    by_page = {r["metadata"]["page_start"]: r["metadata"] for r in records}
    assert by_page[1]["document_part"] == "Cover Page"
    assert by_page[2]["document_part"] == "Petition"


@pytest.mark.asyncio
async def test_metadata_exposes_split_upload_types() -> None:
    result = await metadata_workflow.run(start_event=StartEvent())
    assert set(result.split_upload_types.keys()) == {"SLP_CIVIL", "SLP_CRIMINAL"}
    criminal_ids = [
        slot["id"] for slot in result.split_upload_types["SLP_CRIMINAL"]["slots"]
    ]
    assert "court_fees" not in criminal_ids


def test_process_split_files_does_not_call_llama_split() -> None:
    source = inspect.getsource(ProcessSplitFilesWorkflow)
    assert "split.create" not in source
    assert "client.split" not in source
    assert "_split_page_parts" not in source
    module = inspect.getsource(
        __import__("extraction_review.process_split_files", fromlist=["workflow"])
    )
    assert "llama_cloud_client.split" not in module
    assert ".split.create" not in module


def test_process_split_files_does_not_classify() -> None:
    source = inspect.getsource(ProcessSplitFilesWorkflow)
    assert "classify.create" not in source
    assert "_wait_for_classify" not in source
    module = inspect.getsource(
        __import__("extraction_review.process_split_files", fromlist=["workflow"])
    )
    assert "classify.create" not in module
    assert "classify_file" not in source


def test_extract_input_falls_back_to_compiled_file() -> None:
    state = SplitFilesState(
        fallback_file_id="dfl-compiled-1",
        parts=[
            SplitPartEvent(slot_id="undefined", file_id="dfl-undef-1"),
        ],
    )
    assert extract_input_file_id(state) == "dfl-compiled-1"
    state.petition_file_id = "dfl-petition-1"
    assert extract_input_file_id(state) == "dfl-petition-1"


def test_process_file_prepare_does_not_extract() -> None:
    source = inspect.getsource(ProcessFileWorkflow)
    module = inspect.getsource(
        __import__("extraction_review.process_file", fromlist=["workflow"])
    )
    assert "extract.create" not in source
    assert "extract.create" not in module
    assert "InputRequired" not in source
    assert "InputRequired" not in module
    assert "agent_data.create" not in source
    assert "classify.create" in source
    assert "_split_page_parts" in source
    assert "slice_bundle_pdf" in source
    assert "_extract_sliced_parts" in source


def _blank_pdf(page_count: int) -> bytes:
    writer = PdfWriter()
    for _ in range(page_count):
        writer.add_blank_page(width=72, height=72)
    buffer = BytesIO()
    writer.write(buffer)
    return buffer.getvalue()


def test_combined_vakalatnama_maps_to_vakalatnama_and_poa_slots() -> None:
    catalog = type_catalog("SLP_CIVIL")
    pages = map_slot_pages(
        catalog,
        {
            50: ["Vakalatnama + PoA/BR"],
            51: ["PoA/BR"],
        },
    )
    assert pages["vakalatnama"] == [50]
    assert pages["poa_br"] == [50, 51]


def test_vakalatnama_and_poa_are_separate_slots() -> None:
    catalog = type_catalog("SLP_CIVIL")
    slots = {slot.id: slot for slot in catalog.slots}
    assert slots["vakalatnama"].parts == ("Vakalatnama",)
    assert slots["poa_br"].parts == ("PoA/BR",)
    assert all("+" not in part for slot in catalog.slots for part in slot.parts)
    criminal = type_catalog("SLP_CRIMINAL")
    assert {slot.id for slot in criminal.slots} >= {"vakalatnama", "poa_br"}
    pages = map_slot_pages(
        catalog,
        {50: ["Vakalatnama"], 51: ["PoA/BR"]},
    )
    assert pages["vakalatnama"] == [50]
    assert pages["poa_br"] == [51]


def test_mixed_label_page_is_copied_into_both_slots() -> None:
    catalog = type_catalog("SLP_CIVIL")
    pages = map_slot_pages(
        catalog,
        {20: ["Affidavit", "Vakalatnama"]},
    )
    assert pages["affidavit"] == [20]
    assert pages["vakalatnama"] == [20]
    slices = {
        item.slot_id: item
        for item in slice_bundle_pdf(
            _blank_pdf(20), catalog, {20: ["Affidavit", "Vakalatnama"]}
        )
    }
    assert len(PdfReader(BytesIO(slices["affidavit"].pdf_bytes)).pages) == 1
    assert len(PdfReader(BytesIO(slices["vakalatnama"].pdf_bytes)).pages) == 1


def test_unmatched_labels_are_not_mapped_to_known_slots() -> None:
    catalog = type_catalog("SLP_CIVIL")
    pages = map_slot_pages(
        catalog,
        {
            1: ["Cover Page"],
            99: ["Caveat"],
            100: ["Uncategorized"],
        },
    )
    assert pages == {"cover_page": [1]}


def test_leftover_pages_go_to_undefined_slot() -> None:
    catalog = type_catalog("SLP_CIVIL")
    slices = {
        item.slot_id: item
        for item in slice_bundle_pdf(
            _blank_pdf(5),
            catalog,
            {
                1: ["Cover Page"],
                4: ["Caveat"],
                5: ["Uncategorized"],
            },
        )
    }
    assert slices["cover_page"].pages == (1,)
    assert slices["undefined"].pages == (2, 3, 4, 5)
    assert slices["undefined"].filename == "Undefined.pdf"
    assert slices["undefined"].page_span == "pp. 2–5"
    assert len(PdfReader(BytesIO(slices["undefined"].pdf_bytes)).pages) == 4
    assert leftover_pages(5, {"cover_page": [1]}) == [2, 3, 4, 5]


def test_no_undefined_slice_when_every_page_has_a_slot() -> None:
    catalog = type_catalog("SLP_CIVIL")
    slices = slice_bundle_pdf(
        _blank_pdf(2),
        catalog,
        {1: ["Cover Page"], 2: ["Petition"]},
    )
    by_id = {item.slot_id: item for item in slices}
    assert "undefined" not in by_id
    assert leftover_pages(2, {"cover_page": [1], "petition": [2]}) == []


def test_synopsis_slot_unions_synopsis_and_list_of_dates() -> None:
    catalog = type_catalog("SLP_CIVIL")
    pages = map_slot_pages(
        catalog,
        {
            10: ["Synopsis"],
            11: ["List of Dates & Events"],
            12: ["Synopsis", "List of Dates & Events"],
        },
    )
    assert pages["synopsis_lod"] == [10, 11, 12]
    assert format_page_span(pages["synopsis_lod"]) == "pp. 10–12"


def test_slice_uses_one_indexed_split_pages() -> None:
    pdf_bytes = _blank_pdf(3)
    sliced = extract_pdf_pages(pdf_bytes, [1, 3])
    reader = PdfReader(BytesIO(sliced))
    assert len(reader.pages) == 2
    skipped = extract_pdf_pages(pdf_bytes, [9])
    assert skipped == b""


def test_slice_bundle_pdf_uploads_shape_passes_validate_parts() -> None:
    catalog = type_catalog("SLP_CIVIL")
    pdf_bytes = _blank_pdf(6)
    slices = slice_bundle_pdf(
        pdf_bytes,
        catalog,
        {
            1: ["Advocate's Checklist"],
            2: ["Cover Page"],
            3: ["Record of Proceedings"],
            4: ["AOR's Declaration"],
            5: ["Index"],
            6: ["Office Report on Limitation"],
        },
    )
    by_id = {item.slot_id: item for item in slices}
    assert by_id["cover_page"].filename == "Cover Page.pdf"
    assert by_id["cover_page"].page_span == "p. 2"
    assert by_id["cover_page"].file_hash
    required = _required_parts("SLP_CIVIL")
    payload = []
    for item in required:
        slot_id = item["slot_id"]
        if slot_id in by_id:
            payload.append(
                {
                    "slot_id": slot_id,
                    "file_id": f"file-{slot_id}",
                    "file_hash": by_id[slot_id].file_hash,
                    "filename": by_id[slot_id].filename,
                }
            )
        else:
            payload.append(item)
    _, parts = validate_parts("SLP_CIVIL", payload)
    assert {item.slot_id for item in parts} >= {
        "cover_page",
        "petition",
        "vakalatnama",
    }
