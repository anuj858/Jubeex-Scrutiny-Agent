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
from extraction_review.extract_record import (
    apply_extract_envelope,
    build_formatted_title,
    clean_relief_sort,
    confidence_percent_string,
    format_side_title,
    is_extra_party_caption_mismatch,
    is_organization_name,
    is_party_role_label_mismatch,
    names_are_spelling_variants,
    stamp_source_pages,
    strip_party_role_label,
)
from extraction_review.metadata_workflow import workflow as metadata_workflow
from extraction_review.process_file import ProcessFileWorkflow
from extraction_review.process_split_files import ProcessSplitFilesWorkflow
from extraction_review.split_upload import (
    FieldSources,
    SplitPartInput,
    SplitUploadError,
    build_extract_pack_markdown,
    build_extract_system_prompt,
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
    with pytest.raises(SplitUploadError, match="Main Petition"):
        validate_parts("SLP_CIVIL", _required_parts("SLP_CIVIL", omit={"petition"}))


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
        4: "petition first page parties",
        5: "annexure p-1",
        6: "appendix text",
        7: "petition grounds later page",
        8: "memo of parties ram lal address",
        9: "vakalatnama petitioner names for aor",
        10: "petition prayer page one",
        11: "petition prayer page two",
        12: "petition prayer page three",
        13: "affidavit deponent",
        14: "office report limitation",
    }
    page_parts = {
        1: ["Cover Page"],
        2: ["Index"],
        3: ["Listing Proforma"],
        4: ["Main Petition"],
        5: ["Annexures"],
        6: ["Appendix"],
        7: ["Main Petition"],
        8: ["Memo of Parties"],
        9: ["Vakalatnama"],
        10: ["Main Petition"],
        11: ["Main Petition"],
        12: ["Main Petition"],
        13: ["Affidavit"],
        14: ["Office Report on Limitation"],
    }
    pack = build_extract_pack_markdown(
        page_markdown,
        page_parts,
        extract_source_parts(catalog),
        catalog=catalog,
    )
    assert "[Cover Page]" in pack
    assert "[Main Petition]" in pack
    assert "[Memo of Parties]" in pack
    assert "[Affidavit]" in pack
    assert "[Office Report on Limitation]" in pack
    assert "affidavit deponent" in pack
    assert "office report limitation" in pack
    assert "petition first page parties" in pack
    assert "petition prayer page one" in pack
    assert "petition prayer page three" in pack
    assert "memo of parties ram lal address" in pack
    assert "Fill advocates_on_record" in pack
    assert "Do not copy petitioner or respondent names" in pack
    assert "Never copy party names or addresses from Vakalatnama or Cover Page" in pack
    assert "and Anr." in pack
    assert "petition grounds later page" not in pack
    assert "index listing" not in pack
    assert "[Index]" not in pack
    assert "[Listing Proforma]" not in pack
    assert "annexure p-1" not in pack
    assert "appendix text" not in pack
    assert "Annexures" not in pack
    assert "Appendix" not in pack


def test_party_fields_prefer_memo_of_parties_then_petition() -> None:
    catalog = type_catalog("SLP_CIVIL")
    court_verify = (
        "Main Petition",
        "Vakalatnama",
        "Office Report on Limitation",
        "Affidavit",
        "Memo of Parties",
    )
    assert catalog.extract_field_sources["petitioners"] == FieldSources(
        fill=("Memo of Parties", "Main Petition"),
        verify=("Main Petition", "Cover Page"),
    )
    assert catalog.extract_field_sources["respondents"] == FieldSources(
        fill=("Memo of Parties", "Main Petition"),
        verify=("Main Petition", "Cover Page"),
    )
    assert catalog.extract_field_sources["court"] == FieldSources(
        fill=("Cover Page",),
        verify=court_verify,
    )
    assert catalog.extract_field_sources["petition_type"] == FieldSources(
        fill=("Cover Page",),
        verify=court_verify,
    )
    assert catalog.extract_field_sources["relief_sort"] == FieldSources(
        fill=("Main Petition",),
    )
    assert "applications" not in catalog.extract_field_sources
    assert "classification" not in catalog.extract_field_sources


def test_extract_source_parts_include_petition_and_index() -> None:
    parts = extract_source_parts(type_catalog("SLP_CIVIL"))
    assert parts == {
        "Cover Page",
        "Memo of Parties",
        "Main Petition",
        "Impugned Order",
        "Vakalatnama",
        "AOR's Declaration",
        "Affidavit",
        "Office Report on Limitation",
    }
    assert "Undefined" not in parts
    assert "Index" not in parts
    assert "Listing Proforma" not in parts


def test_inject_where_to_look_appends_field_guidance() -> None:
    schema = {
        "properties": {
            "cause_title": {"description": "Cause title."},
            "court": {"description": "The court."},
            "petitioners": {"description": "Array of petitioners."},
        }
    }
    updated = inject_where_to_look(
        schema,
        {
            "cause_title": ["Memo of Parties", "Cover Page"],
            "petitioners": ["Memo of Parties", "Main Petition"],
        },
    )
    assert (
        "Fill only from Memo of Parties, Cover Page"
        in updated["properties"]["cause_title"]["description"]
    )
    petitioners = updated["properties"]["petitioners"]["description"]
    assert "Prefer Memo of Parties" in petitioners
    assert "first page of the Main Petition" in petitioners
    assert "fill it from the other" in petitioners
    assert "Never copy party names or addresses from Vakalatnama" in petitioners
    assert "Look only in" not in updated["properties"]["court"]["description"]
    assert schema["properties"]["cause_title"]["description"] == "Cause title."


def test_extract_system_prompt_forbids_vakalatnama_for_parties() -> None:
    prompt = build_extract_system_prompt(type_catalog("SLP_CIVIL"))
    assert "petitioners: fill Memo of Parties, Main Petition; verify Main Petition, Cover Page" in prompt
    assert "Copy printed text only" in prompt
    assert "inconsistencies: one item per spelling" in prompt


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
    assert any(item.startswith("Main Petition") for item in items)
    assert any(item.startswith("Synopsis") for item in items)
    assert any("List of Dates & Events" in item for item in items)
    names = [span["name"] for span in payload["documents"]]
    assert "Main Petition" in names
    assert payload["document_counts"]["processed"] == len(payload["documents"])


def test_extract_envelope_sets_null_ids_and_stitch_documents() -> None:
    record = {
        "court": "Supreme Court of India",
        "petition_type": None,
        "cause_title": {"title": "A v. B"},
        "petitioners": [{"name": "A", "source_pages": "6, 7"}],
    }
    stamp_source_pages(record)
    wrapped = apply_extract_envelope(
        record,
        page_parts={1: ["Cover Page"], 2: ["Main Petition"]},
        filing_type="SLP_CIVIL",
        overall_confidence=0.91,
        generated_at="2026-09-04T12:00:00+05:30",
    )
    assert wrapped["schema_version"] == "extraction-v1"
    assert wrapped["job_type"] == "compiled_petition"
    assert wrapped["organization_id"] is None
    assert wrapped["workspace_id"] is None
    assert wrapped["user_id"] is None
    assert wrapped["primary_document_id"] is None
    assert wrapped["court"] == "Supreme Court of India"
    assert wrapped["petition_type"] == "Special Leave Petition (Civil)"
    assert wrapped["overall_confidence"] == "91%"
    assert wrapped["generated_at"] == "2026-09-04T12:00:00+05:30"
    assert wrapped["documents"] == [
        {"name": "Cover Page", "start_page": 1, "end_page": 1},
        {"name": "Main Petition", "start_page": 2, "end_page": 2},
    ]
    assert wrapped["petitioners"][0]["source_pages"] == [6, 7]
    assert "classification" not in wrapped
    assert "applications" not in wrapped


def test_formatted_title_anr_and_ors() -> None:
    assert build_formatted_title("Meera Krishnan", 1, "Union of India", 1) == (
        "Meera Krishnan VS Union of India"
    )
    assert build_formatted_title("Meera Krishnan", 2, "Union of India", 2) == (
        "Meera Krishnan and Anr. VS Union of India and Anr."
    )
    assert build_formatted_title("Meera Krishnan", 2, "Union of India", 4) == (
        "Meera Krishnan and Anr. VS Union of India and Ors."
    )
    assert build_formatted_title(
        "Kailash Negi Alias Anmol",
        1,
        "Smt. Shalija Shah And Anr",
        2,
    ) == "Kailash Negi Alias Anmol VS Smt. Shalija Shah and Anr."
    assert format_side_title("Smt. Shalija Shah And Anr ...Respondent(s)", 2) == (
        "Smt. Shalija Shah and Anr."
    )


def test_relief_sort_drops_main_prayer_heading() -> None:
    raw = (
        "7. <u>**MAIN PRAYER**</u>:\n"
        "\n"
        "In the circumstances stated above, the Petitioners pray that this "
        "Hon’ble Court may be pleased to:\n"
        "\n"
        "(i) Pass an order granting Special Leave to Appeal against judgement "
        "dated 09.02.2026 passed by the Hon’ble High Court of Uttarakhand at "
        "Nainital in MCRC No. 08/2026 in CLR No. 67/2022;"
    )
    cleaned = clean_relief_sort(raw)
    assert cleaned is not None
    assert "MAIN PRAYER" not in cleaned
    assert "<u>" not in cleaned
    assert cleaned.startswith("In the circumstances stated above")
    wrapped = apply_extract_envelope({"relief_sort": raw})
    assert wrapped["relief_sort"] == cleaned


def test_envelope_strips_cover_anr_before_formatting() -> None:
    wrapped = apply_extract_envelope(
        {
            "cause_title": {
                "main_petitioner": "Kailash Negi Alias Anmol",
                "main_respondent": "Smt. Shalija Shah And Anr",
            },
            "petitioners": [{"name": "Kailash Negi Alias Anmol"}],
            "respondents": [
                {"name": "Smt. Shalija Shah"},
                {"name": "Another"},
            ],
        }
    )
    cause = wrapped["cause_title"]
    assert cause["main_respondent"] == "Smt. Shalija Shah"
    assert cause["formatted_title"] == (
        "Kailash Negi Alias Anmol VS Smt. Shalija Shah and Anr."
    )


def test_organization_name_prefixes_and_suffixes() -> None:
    assert is_organization_name("Union of India")
    assert is_organization_name("The State of Maharashtra")
    assert is_organization_name("M/s Acme Traders")
    assert is_organization_name("Acme Pvt. Ltd.")
    assert is_organization_name("Helping Hands Foundation")
    assert not is_organization_name("Meera Krishnan")


def test_envelope_formats_title_kind_acting_through_and_confidence() -> None:
    record = {
        "court": {"name": "Supreme Court of India"},
        "cause_title": {
            "title": "Meera Krishnan v. Union of India",
            "main_petitioner": "Meera Krishnan",
            "main_respondent": "Union of India",
            "confidence": 0.95,
        },
        "petitioners": [
            {"name": "Meera Krishnan", "kind": "individual"},
            {"name": "Ramesh"},
        ],
        "respondents": [
            {
                "name": "Union of India",
                "kind": "individual",
                "source_part": "Memo of Parties",
                "source_pages": [8],
            },
            {"name": "Ajay"},
            {"name": "State of Karnataka"},
        ],
        "inconsistencies": {
            "items": [{"id": "1", "label": "Court spelling", "detail": "Cover vs petition"}],
        },
    }
    wrapped = apply_extract_envelope(
        record,
        overall_confidence=0.65,
        field_confidence={"petitioners": [0.9, 0.8]},
    )
    assert wrapped["court"] == "Supreme Court of India"
    assert wrapped["cause_title"]["formatted_title"] == (
        "Meera Krishnan and Anr. VS Union of India and Ors."
    )
    assert wrapped["cause_title"]["confidence"] == "95%"
    assert wrapped["petitioners"][0]["kind"] == "INDIVIDUAL"
    assert wrapped["petitioners"][0]["is_primary"] is True
    assert wrapped["petitioners"][0]["confidence"] == "90%"
    assert wrapped["respondents"][0]["kind"] == "ORGANIZATION"
    assert wrapped["respondents"][0]["is_primary"] is True
    missing = wrapped["inconsistencies"]["items"]
    assert any(item.get("label") == "Missing acting through" for item in missing)
    assert missing[0]["raw_text"] == "Cover vs petition"
    assert wrapped["overall_confidence"] == "65%"
    assert confidence_percent_string(0.91) == "91%"
    assert confidence_percent_string("95%") == "95%"


def test_cover_page_petitioner_respondent_labels_are_not_inconsistencies() -> None:
    dotted = (
        'Cover Page: "Smt. Shalija Shah And Anr ...Respondent(s)" vs '
        'Main Petition/Affidavit/Vakalatnama: "Smt. Shalija Shah And Anr"'
    )
    plain = (
        'Cover Page: "Smt. Shalija Shah And Anr Petitioner" vs '
        'Main Petition: "Smt. Shalija Shah And Anr"'
    )
    assert strip_party_role_label("Smt. Shalija Shah And Anr ...Respondent(s)") == (
        "Smt. Shalija Shah And Anr"
    )
    assert strip_party_role_label("Smt. Shalija Shah And Anr Petitioner") == (
        "Smt. Shalija Shah And Anr"
    )
    assert is_party_role_label_mismatch(dotted)
    assert is_party_role_label_mismatch(plain)
    assert not is_party_role_label_mismatch(
        'Cover Page: "Smt. Shalija Shah" vs Main Petition: "Smt. Shailja Shah"'
    )
    caps = (
        'Petitioner spelling mismatch in impugned order: Main Petition/Cover Page: '
        '"Kailash Negi Alias Anmol"; Impugned Order: "KAILASH NEGI ALIAS ANMOL"'
    )
    assert is_party_role_label_mismatch(caps)
    wrapped = apply_extract_envelope(
        {
            "inconsistencies": {
                "items": [
                    {
                        "id": "1",
                        "label": "cause title respondent capitalization",
                        "raw_text": dotted,
                    },
                    {
                        "id": "2",
                        "label": "Petitioner spelling mismatch in impugned order",
                        "raw_text": caps,
                    },
                    {
                        "id": "3",
                        "label": "Name spelling",
                        "raw_text": (
                            'Cover Page: "Smt. Shalija Shah" vs '
                            'Main Petition: "Smt. Shailja Shah"'
                        ),
                    },
                ],
                "source_part": ["Cover Page", "Main Petition"],
                "source_pages": [1, 27],
            }
        }
    )
    items = wrapped["inconsistencies"]["items"]
    assert len(items) == 1
    assert items[0]["id"] == "1"
    assert "Shailja" in items[0]["raw_text"]


def test_duplicate_respondent_spelling_inconsistencies_are_merged() -> None:
    wrapped = apply_extract_envelope(
        {
            "inconsistencies": {
                "items": [
                    {
                        "id": "1",
                        "label": "Main respondent spelling mismatch",
                        "raw_text": (
                            'Cover Page: "Smt. Shalija Shah And Anr ...Respondent(s)"; '
                            'Main Petition: "Smt. Shailja Shah" / "1. Smt. Shailja Shah"'
                        ),
                    },
                    {
                        "id": "2",
                        "label": "Respondent name spelling mismatch",
                        "raw_text": (
                            'Cover Page: "Smt. Shalija Shah"; '
                            'Main Petition: "Smt. Shailja Shah"'
                        ),
                    },
                ]
            }
        }
    )
    items = wrapped["inconsistencies"]["items"]
    assert len(items) == 1
    assert items[0]["id"] == "1"
    assert items[0]["raw_text"] == (
        'Cover Page: "Smt. Shalija Shah"; Main Petition: "Smt. Shailja Shah"'
    )


def test_extra_respondent_is_not_cover_page_anr_spelling_error() -> None:
    respondent_1 = (
        'Cover Page / AOR\'s Declaration / Affidavit / Vakalatnama: '
        '"Smt. Shalija Shah And Anr"; Main Petition: "Smt. Shailja Shah"'
    )
    respondent_2 = (
        'Cover Page / AOR\'s Declaration / Affidavit / Vakalatnama: '
        '"Smt. Shalija Shah And Anr"; Main Petition: "Smt. Bandana Shah"'
    )
    assert names_are_spelling_variants("Smt. Shalija Shah", "Smt. Shailja Shah")
    assert not names_are_spelling_variants("Smt. Shalija Shah", "Smt. Bandana Shah")
    assert not is_extra_party_caption_mismatch(respondent_1, "Respondent 1 spelling")
    assert is_extra_party_caption_mismatch(respondent_2, "Respondent 2 spelling")
    wrapped = apply_extract_envelope(
        {
            "inconsistencies": {
                "items": [
                    {
                        "id": "1",
                        "label": "Respondent 1 spelling",
                        "raw_text": respondent_1,
                    },
                    {
                        "id": "2",
                        "label": "Respondent 2 spelling",
                        "raw_text": respondent_2,
                    },
                ]
            }
        }
    )
    items = wrapped["inconsistencies"]["items"]
    assert len(items) == 1
    assert items[0]["label"] == "Respondent 1 spelling"
    assert "Shailja" in items[0]["raw_text"]
    assert "Bandana" not in items[0]["raw_text"]


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
        page for page, names in page_parts.items() if names == ["Main Petition"]
    ]
    assert len(petition_pages) == 1
    assert "No parse text" in page_markdown[petition_pages[0]]
    pack = build_extract_pack_markdown(
        page_markdown, page_parts, extract_source_parts(catalog)
    )
    assert "No parse text" not in pack


def test_page_maps_survive_string_keys() -> None:
    markdown = coerce_page_markdown({"1": "cover", "2": "petition body"})
    parts = coerce_page_parts({"1": "Cover Page", "2": ["Main Petition"]})
    assert markdown == {1: "cover", 2: "petition body"}
    assert parts == {1: ["Cover Page"], 2: ["Main Petition"]}
    assert coerce_page_parts({"3": "Petition"}) == {3: ["Main Petition"]}
    records = build_page_records(
        base_id="bundle",
        page_markdown=markdown,
        page_parts=parts,
    )
    by_page = {r["metadata"]["page_start"]: r["metadata"] for r in records}
    assert by_page[1]["document_part"] == "Cover Page"
    assert by_page[2]["document_part"] == "Main Petition"


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
        {1: ["Cover Page"], 2: ["Main Petition"]},
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
