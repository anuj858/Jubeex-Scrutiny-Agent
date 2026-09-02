"""Already-split upload catalog, validation, extract pack, and document_part stamps."""

from __future__ import annotations

import inspect

import pytest
from workflows.events import StartEvent

from extraction_review.document_parts import overlay_split_documents
from extraction_review.metadata_workflow import workflow as metadata_workflow
from extraction_review.process_split_files import ProcessSplitFilesWorkflow
from extraction_review.split_upload import (
    SplitPartInput,
    SplitUploadError,
    build_extract_pack_markdown,
    bundle_file_hash,
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
    assert civil_ids[-1] == "court_fees"


def test_slp_civil_accepts_required_slots_without_optional_annexures() -> None:
    catalog, parts = validate_parts("SLP_CIVIL", _required_parts("SLP_CIVIL"))
    assert catalog.filing_type == "SLP_CIVIL"
    assert all(item.slot_id != "annexures" for item in parts)
    assert any(item.slot_id == "court_fees" for item in parts)


def test_slp_criminal_omits_court_fees_and_rejects_it() -> None:
    _, parts = validate_parts("SLP_CRIMINAL", _required_parts("SLP_CRIMINAL"))
    assert all(item.slot_id != "court_fees" for item in parts)
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


def test_unknown_filing_type_fails() -> None:
    with pytest.raises(SplitUploadError, match="Unknown filing type"):
        validate_parts("WRIT_PETITION_CIVIL", [])


def test_duplicate_slot_fails() -> None:
    parts = _required_parts("SLP_CRIMINAL")
    parts.append({"slot_id": "petition", "file_id": "file-petition-2"})
    with pytest.raises(SplitUploadError, match="Duplicate slot"):
        validate_parts("SLP_CRIMINAL", parts)


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


def test_extract_pack_excludes_annexures_and_appendix() -> None:
    catalog = type_catalog("SLP_CIVIL")
    page_markdown = {
        1: "listing columns",
        2: "petition grounds",
        3: "annexure p-1",
        4: "appendix text",
    }
    page_parts = {
        1: ["Listing Proforma"],
        2: ["Petition"],
        3: ["Annexures"],
        4: ["Appendix"],
    }
    pack = build_extract_pack_markdown(
        page_markdown, page_parts, extract_source_parts(catalog)
    )
    assert "[Listing Proforma]" in pack
    assert "[Petition]" in pack
    assert "annexure p-1" not in pack
    assert "appendix text" not in pack
    assert "Annexures" not in pack
    assert "Appendix" not in pack


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
