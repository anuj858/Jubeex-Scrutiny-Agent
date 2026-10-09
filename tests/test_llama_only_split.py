"""Production split routing and lossless slot mapping, with no live AI calls."""

from pathlib import Path
from types import SimpleNamespace
from unittest.mock import AsyncMock, Mock

import pymupdf
import pytest

from extraction_review import process_file
from extraction_review.bundle_slicer import slice_bundle_pdf
from extraction_review.config import Config
from extraction_review.document_parts import page_parts_from_split
from extraction_review.split_upload import ordered_parts, type_catalog, validate_parts


def _pdf_bytes() -> bytes:
    # Deliberately conflicting headings: local content must not override Split.
    with pymupdf.open() as pdf:
        for text in (
            "INDEX\nANNEXURE P-9",
            "IN THE SUPREME COURT OF INDIA\nAPPLICATION",
            "LISTED PERFORMA",
            "AFFIDAVIT",
            "ANNEXURE P-12",
            "SYNOPSIS",
        ):
            page = pdf.new_page()
            page.insert_text((72, 72), text)
        return pdf.tobytes()


def _config() -> Config:
    return Config.model_validate_json(
        (Path(__file__).resolve().parents[1] / "configs/config.json").read_text()
    )


def _client(segments: list[dict]) -> SimpleNamespace:
    payload = {"id": "spl-raw", "status": "COMPLETED", "result": {"segments": segments}}
    return SimpleNamespace(
        split=SimpleNamespace(
            create=AsyncMock(return_value=SimpleNamespace(id="spl-raw")),
            get=AsyncMock(
                return_value=SimpleNamespace(
                    **payload, model_dump=lambda **_kwargs: payload
                )
            ),
        ),
        files=SimpleNamespace(
            create=AsyncMock(return_value=SimpleNamespace(id="dfl-part"))
        ),
    )


def _fail_local_rules(*_args, **_kwargs):
    raise AssertionError("Local OCR/document classification/repair must not run")


@pytest.mark.asyncio
@pytest.mark.parametrize("upload_slots", [False, True])
@pytest.mark.parametrize("rules_flag", [None, "false", "0", "invalid"])
async def test_compiled_split_uses_llama_only_and_preserves_existing_contract(
    monkeypatch, caplog, upload_slots, rules_flag
) -> None:
    if rules_flag is None:
        monkeypatch.delenv("SPLIT_PYTHON_RULES_ENABLED", raising=False)
    else:
        monkeypatch.setenv("SPLIT_PYTHON_RULES_ENABLED", rules_flag)
    for target in (
        "structure_split.extract_page_units",
        "structure_split.structure_aware_split",
        "split_repair.repair_compiled_split",
        "split_audit.audit_compiled_split",
        "split_ocr.ocr_sparse_pages",
        "split_pdf_layout.extract_split_layout",
        "document_parts.reclassify_pages_from_headings",
        "document_parts.explode_repeating_split_parts",
        "document_parts.collapse_repeated_split_pages",
    ):
        monkeypatch.setattr(f"extraction_review.{target}", _fail_local_rules)
    monkeypatch.setenv("SKIP_SLOT_PDF_UPLOAD", "false" if upload_slots else "true")
    monkeypatch.setattr(
        process_file, "_load_bundle_pdf", AsyncMock(return_value=_pdf_bytes())
    )
    monkeypatch.setattr(
        process_file, "collect_optional_usage", AsyncMock(return_value={})
    )
    artifact = Mock()
    monkeypatch.setattr(process_file, "upload_step_json", artifact)
    segments = [
        {"category": "Main Petition", "pages": [1, 2]},
        {"category": "Application", "pages": [3]},
        {"category": "Application", "pages": [4]},
        {"category": "uncategorized", "pages": [5]},
        # Page 6 omitted remotely: preserve it as Unidentified, not Synopsis.
    ]
    client = _client(segments)
    state = process_file.PrepareState(file_id="dfl-source", filename="source.pdf")
    events = []
    ctx = SimpleNamespace(
        store=SimpleNamespace(get_state=AsyncMock(return_value=state)),
        write_event_to_stream=events.append,
    )
    with caplog.at_level("INFO"):
        result = await process_file.ProcessFileWorkflow().prepare_bundle(
            process_file.FileClassifiedEvent(filing_type="SLP_CIVIL"),
            ctx,
            client,
            _config().split,
        )
    assert result.slot_pages == {
        "petition": [{"start": 1, "end": 2}],
        "application_1": [{"start": 3, "end": 3}],
        "application_2": [{"start": 4, "end": 4}],
        "undefined": [{"start": 5, "end": 6}],
    }
    assert result.llama_split["returned"]["result"]["segments"] == segments
    assert all(bool(part.file_id) == upload_slots for part in result.parts)
    assert client.files.create.await_count == (4 if upload_slots else 0)
    assert "[SplitTiming] mode=llama_only" in caplog.text
    assert not any("Structure-aware" in event.message for event in events)
    assert artifact.call_args.args[1]["duplicate_parts"] == []
    assert artifact.call_args.args[1]["split_audit"]["flags"] == []
    # Same response fields, including compatibility fields; no added API format.
    assert "llama_split" in result.model_dump()
    assert result.agent_data_id is None


@pytest.mark.parametrize(
    ("value", "enabled"),
    [
        (None, False),
        ("", False),
        ("false", False),
        ("0", False),
        ("no", False),
        ("unexpected", False),
        ("true", True),
        (" TRUE ", True),
        ("1", True),
        ("yes", True),
    ],
)
def test_split_python_rules_flag(monkeypatch, value, enabled):
    monkeypatch.delenv("SPLIT_RECONCILIATION_MODE", raising=False)
    if value is None:
        monkeypatch.delenv("SPLIT_PYTHON_RULES_ENABLED", raising=False)
    else:
        monkeypatch.setenv("SPLIT_PYTHON_RULES_ENABLED", value)
    assert process_file.split_python_rules_enabled() is enabled


@pytest.mark.parametrize(
    ("configured", "legacy", "expected"),
    [
        ("llama_only", "true", "llama_only"),
        ("targeted", "false", "targeted"),
        ("legacy_full", "false", "legacy_full"),
        ("invalid", "true", "targeted"),
        (None, "false", "llama_only"),
    ],
)
def test_split_reconciliation_mode(monkeypatch, configured, legacy, expected):
    if configured is None:
        monkeypatch.delenv("SPLIT_RECONCILIATION_MODE", raising=False)
    else:
        monkeypatch.setenv("SPLIT_RECONCILIATION_MODE", configured)
    monkeypatch.setenv("SPLIT_PYTHON_RULES_ENABLED", legacy)

    assert process_file.split_reconciliation_mode() == expected


@pytest.mark.asyncio
async def test_enabled_mode_runs_local_pipeline_and_slices_repaired_assignments(
    monkeypatch, caplog
):
    from extraction_review import split_audit, structure_split

    monkeypatch.setenv("SPLIT_PYTHON_RULES_ENABLED", "true")
    monkeypatch.setenv("SKIP_SLOT_PDF_UPLOAD", "true")
    pdf_bytes = _pdf_bytes()
    units = [
        SimpleNamespace(pdf_page=page, text=f"local page {page}")
        for page in range(1, 7)
    ]
    # Remote Annexures pages must not overwrite the local repair below.
    segments = [{"category": "Annexures", "pages": [1, 2, 3, 4, 5, 6]}]
    repaired = {
        1: ["Index"],
        2: ["Main Petition"],
        3: ["Main Petition"],
        4: ["Annexure P-9"],
        5: ["Annexure P-9"],
    }
    duplicate = {"part": "Annexure P-9", "pages": [6]}
    structured = SimpleNamespace(
        page_parts=repaired,
        page_units=units,
        duplicates=[SimpleNamespace(as_dict=lambda: duplicate)],
        ocr_needed_pages=[4],
        logical_documents=[1, 2, 3],
        report=lambda: {"auto_boundaries": 3, "verify_boundaries": 0},
    )

    def read_local_pages(*_args, **_kwargs):
        # An in-flight job must not switch mode between OCR and repair/audit.
        monkeypatch.setenv("SPLIT_PYTHON_RULES_ENABLED", "false")
        return units

    read = Mock(side_effect=read_local_pages)
    repair = Mock(return_value=structured)
    audit = Mock(
        return_value={
            "index_rows": [],
            "document_spans": [],
            "flags": [{"code": "test_flag"}],
            "flag_counts": {"error": 0, "warning": 1, "total": 1},
            "unidentified_reasons": [
                {"page_span": {"start": 6, "end": 6}, "reason": "Needs boundary review"}
            ],
        }
    )
    monkeypatch.setattr(structure_split, "extract_page_units", read)
    monkeypatch.setattr(structure_split, "structure_aware_split", repair)
    monkeypatch.setattr(split_audit, "audit_compiled_split", audit)
    monkeypatch.setattr(
        process_file, "_load_bundle_pdf", AsyncMock(return_value=pdf_bytes)
    )
    monkeypatch.setattr(
        process_file, "collect_optional_usage", AsyncMock(return_value={})
    )
    artifact = Mock()
    monkeypatch.setattr(process_file, "upload_step_json", artifact)
    ctx = SimpleNamespace(
        store=SimpleNamespace(
            get_state=AsyncMock(
                return_value=process_file.PrepareState(
                    file_id="dfl-source", filename="source.pdf"
                )
            )
        ),
        write_event_to_stream=Mock(),
    )
    with caplog.at_level("INFO"):
        result = await process_file.ProcessFileWorkflow().prepare_bundle(
            process_file.FileClassifiedEvent(filing_type="SLP_CIVIL"),
            ctx,
            _client(segments),
            _config().split,
        )
    read.assert_called_once_with(pdf_bytes, source_pdf="source.pdf")
    repair.assert_called_once()
    assert repair.call_args.kwargs["page_units"] is units
    assert repair.call_args.kwargs["run_hybrid_repair"] is True
    assert repair.call_args.kwargs["reconciliation_mode"] == "targeted"
    audit.assert_called_once_with(
        repaired, {unit.pdf_page: unit.text for unit in units}, page_count=6
    )
    assert result.slot_pages == {
        "index": [{"start": 1, "end": 1}],
        "petition": [{"start": 2, "end": 3}],
        "annexure_p9": [{"start": 4, "end": 5}],
        "undefined": [{"start": 6, "end": 6}],
    }
    assert result.parts[-1].reason == "Needs boundary review"
    assert result.llama_split["returned"]["result"]["segments"] == segments
    saved = artifact.call_args.args[1]
    assert saved["duplicate_parts"] == [duplicate]
    assert saved["split_audit"]["structure"] == structured.report()
    assert saved["split_audit"]["flags"] == [{"code": "test_flag"}]
    assert "mode=targeted" in caplog.text
    assert "repair_and_audit=" in caplog.text


@pytest.mark.asyncio
@pytest.mark.parametrize("mode", ["targeted", "legacy_full"])
async def test_reconciled_workflow_preserves_adjacent_remote_documents(
    monkeypatch, mode
):
    from extraction_review import split_audit, structure_split

    monkeypatch.setenv("SPLIT_RECONCILIATION_MODE", mode)
    monkeypatch.setenv("SKIP_SLOT_PDF_UPLOAD", "true")
    segments = [
        {"category": "Annexures", "pages": [1, 2]},
        {"category": "Annexures", "pages": [3, 4]},
        {"category": "Application", "pages": [5]},
        {"category": "Application", "pages": [6]},
    ]
    # Local repair changes a page but leaves generic document identities.
    repaired = page_parts_from_split({"segments": segments})
    repaired[1] = ["Index"]
    units = [SimpleNamespace(pdf_page=page, text="") for page in range(1, 7)]
    structured = SimpleNamespace(
        page_parts=repaired,
        page_units=units,
        duplicates=[],
        ocr_needed_pages=[],
        logical_documents=[],
        report=dict,
    )
    monkeypatch.setattr(structure_split, "extract_page_units", Mock(return_value=units))
    monkeypatch.setattr(
        structure_split, "structure_aware_split", Mock(return_value=structured)
    )
    monkeypatch.setattr(split_audit, "audit_compiled_split", Mock(return_value={}))
    monkeypatch.setattr(
        process_file, "_load_bundle_pdf", AsyncMock(return_value=_pdf_bytes())
    )
    monkeypatch.setattr(
        process_file, "collect_optional_usage", AsyncMock(return_value={})
    )
    monkeypatch.setattr(process_file, "upload_step_json", Mock())
    ctx = SimpleNamespace(
        store=SimpleNamespace(
            get_state=AsyncMock(
                return_value=process_file.PrepareState(
                    file_id="dfl-source", filename="source.pdf"
                )
            )
        ),
        write_event_to_stream=Mock(),
    )
    result = await process_file.ProcessFileWorkflow().prepare_bundle(
        process_file.FileClassifiedEvent(filing_type="SLP_CIVIL"),
        ctx,
        _client(segments),
        _config().split,
    )
    assert result.slot_pages == {
        "index": [{"start": 1, "end": 1}],
        "annexure_1": [{"start": 2, "end": 2}],
        "annexure_2": [{"start": 3, "end": 4}],
        "application_1": [{"start": 5, "end": 5}],
        "application_2": [{"start": 6, "end": 6}],
    }


@pytest.mark.asyncio
async def test_enabled_flag_reaches_ocr_and_real_repair_and_audit(monkeypatch):
    from extraction_review import split_audit, structure_split

    monkeypatch.setenv("SPLIT_PYTHON_RULES_ENABLED", "true")
    monkeypatch.setenv("SKIP_SLOT_PDF_UPLOAD", "true")
    monkeypatch.setattr(
        process_file, "_load_bundle_pdf", AsyncMock(return_value=_pdf_bytes())
    )
    monkeypatch.setattr(
        process_file, "collect_optional_usage", AsyncMock(return_value={})
    )
    artifact = Mock()
    monkeypatch.setattr(process_file, "upload_step_json", artifact)
    # Only the external OCR subprocess is stubbed. Exercise the real local
    # page extraction, structure, repair and audit chain behind the flag.
    ocr = Mock(return_value={})
    repair = Mock(wraps=structure_split.repair_compiled_split)
    audit = Mock(wraps=split_audit.audit_compiled_split)
    monkeypatch.setattr(structure_split, "ocr_sparse_pages", ocr)
    monkeypatch.setattr(structure_split, "repair_compiled_split", repair)
    monkeypatch.setattr(split_audit, "audit_compiled_split", audit)
    ctx = SimpleNamespace(
        store=SimpleNamespace(
            get_state=AsyncMock(
                return_value=process_file.PrepareState(
                    file_id="dfl-source", filename="source.pdf"
                )
            )
        ),
        write_event_to_stream=Mock(),
    )
    result = await process_file.ProcessFileWorkflow().prepare_bundle(
        process_file.FileClassifiedEvent(filing_type="SLP_CIVIL"),
        ctx,
        _client([{"category": "Main Petition", "pages": [1, 2, 3, 4, 5, 6]}]),
        _config().split,
    )
    ocr.assert_called_once()
    assert ocr.call_args.args[1]  # Sparse pages were selected for OCR.
    repair.assert_called_once()
    audit.assert_called_once()
    assert result.parts
    assert "structure" in artifact.call_args.args[1]["split_audit"]


def test_generic_segments_remain_separate_without_inventing_printed_annexure_numbers():
    segments = [
        {"category": "Annexures", "pages": [1, 2]},
        {"category": "Annexures", "pages": [3, 4]},
        {"category": "Application", "pages": [5, 6]},
    ]
    mapping = page_parts_from_split(SimpleNamespace(result={"segments": segments}))
    slices = slice_bundle_pdf(
        _pdf_bytes(), type_catalog("SLP_CIVIL"), mapping, split_segments=segments
    )
    by_id = {item.slot_id: item for item in slices}
    assert by_id["annexure_1"].pages == (1, 2)
    assert by_id["annexure_1"].label == "Annexure 1"
    assert by_id["annexure_2"].pages == (3, 4)
    assert by_id["application_1"].pages == (5, 6)
    assert all(not item.slot_id.startswith("annexure_p") for item in slices)
    catalog, parts = validate_parts(
        "SLP_CIVIL",
        [
            {"slot_id": item.slot_id, "file_id": f"file-{item.slot_id}"}
            for item in slices
        ],
    )
    assert parts[0].document_parts == ("Annexures",)
    assert [part.slot_id for part in ordered_parts(catalog, list(reversed(parts)))] == [
        "annexure_1",
        "annexure_2",
        "application_1",
    ]
    for item in slices:
        with pymupdf.open(stream=item.pdf_bytes, filetype="pdf") as pdf:
            assert pdf.page_count == len(item.pages)


def test_explicit_annexure_names_and_noncontiguous_page_lists_are_not_repaired():
    segments = [
        {"category": "Annexure A-14", "pages": [1, 3, 5]},
        {"category": "Annexures", "pages": [2, 4]},
    ]
    mapping = page_parts_from_split(SimpleNamespace(result={"segments": segments}))
    slices = slice_bundle_pdf(
        _pdf_bytes(), type_catalog("SLP_CIVIL"), mapping, split_segments=segments
    )
    by_id = {item.slot_id: item for item in slices}
    assert by_id["annexure_a14"].pages == (1, 3, 5)
    assert by_id["annexure_1"].pages == (2, 4)
    assert by_id["undefined"].pages == (6,)


@pytest.mark.asyncio
async def test_all_uncategorized_pages_are_retained():
    segments = [{"category": None, "pages": [1, 2, 3, 4, 5, 6]}]
    mapping, _, exchange = await process_file._split_page_parts(
        _client(segments), file_id="dfl-source", split_config=_config().split
    )
    assert mapping == {}
    slices = slice_bundle_pdf(
        _pdf_bytes(),
        type_catalog("SLP_CIVIL"),
        mapping,
        split_segments=exchange["returned"]["result"]["segments"],
    )
    assert len(slices) == 1
    assert slices[0].slot_id == "undefined"
    assert slices[0].pages == (1, 2, 3, 4, 5, 6)


@pytest.mark.asyncio
async def test_empty_remote_result_still_fails():
    with pytest.raises(RuntimeError, match="labelled no pages"):
        await process_file._split_page_parts(
            _client([]), file_id="dfl-source", split_config=_config().split
        )


def test_invalid_page_assignment_fails_instead_of_silently_dropping_pages():
    with pytest.raises(ValueError, match="out-of-range PDF pages"):
        slice_bundle_pdf(
            _pdf_bytes(), type_catalog("SLP_CIVIL"), {7: ["Main Petition"]}
        )


@pytest.mark.parametrize(
    ("labels", "matches"),
    [(["Annexures"], True), (["Annexure A-15"], True), (["Main Petition"], False)],
)
def test_generic_annexure_verification_does_not_require_a_printed_number(
    labels, matches
):
    assert (
        process_file.split_labels_match_expected_slot(
            expected_slot="annexure_2",
            catalog=type_catalog("SLP_CIVIL"),
            page_parts={1: labels},
        )
        is matches
    )
