"""Named-file verify: classify the first page against the split categories."""

import asyncio
import json
from pathlib import Path
from types import SimpleNamespace

import pymupdf
import pytest

from extraction_review.config import SplitConfig, load_config_payload
from extraction_review.process_file import (
    FilingPartIn,
    _classify_document_label,
    _label_matches_expected_slot,
    _leading_pdf_pages,
    _verify_one_document,
    parse_edited_flag,
    split_category_for_classify_type,
    split_labels_match_expected_slot,
    verify_classify_configuration,
    verify_classify_rules,
    verify_max_pages,
)
from extraction_review.split_upload import type_catalog


def test_cover_page_name_matches_cover_split_labels() -> None:
    catalog = type_catalog("SLP_CIVIL")
    assert (
        split_labels_match_expected_slot(
            expected_slot="cover_page",
            catalog=catalog,
            page_parts={1: ["Cover Page"]},
        )
        is True
    )


def test_vakalatnama_named_as_cover_page_is_false() -> None:
    catalog = type_catalog("SLP_CIVIL")
    assert (
        split_labels_match_expected_slot(
            expected_slot="cover_page",
            catalog=catalog,
            page_parts={1: ["Vakalatnama"], 2: ["Memo of Appearance"]},
        )
        is False
    )


def test_unrecognized_pages_do_not_beat_matching_cover() -> None:
    catalog = type_catalog("SLP_CIVIL")
    assert (
        split_labels_match_expected_slot(
            expected_slot="cover_page",
            catalog=catalog,
            page_parts={1: ["Cover Page"], 2: [], 3: None},
        )
        is True
    )


def test_undefined_expected_slot_never_matches() -> None:
    catalog = type_catalog("SLP_CIVIL")
    assert (
        split_labels_match_expected_slot(
            expected_slot="undefined",
            catalog=catalog,
            page_parts={1: ["Cover Page"]},
        )
        is False
    )


def test_overall_match_is_false_if_any_file_fails() -> None:
    matches = [True, False, True]
    overall = all(matches)
    assert overall is False
    assert all([True, True, True]) is True


def _split_config() -> SplitConfig:
    return SplitConfig.model_validate(load_config_payload()["split"])


def _pdf_with_pages(count: int) -> bytes:
    document = pymupdf.open()
    try:
        for index in range(count):
            page = document.new_page()
            page.insert_text((72, 72), f"page {index + 1}")
        return document.tobytes()
    finally:
        document.close()


def _page_count(pdf_bytes: bytes) -> int:
    document = pymupdf.open(stream=pdf_bytes, filetype="pdf")
    try:
        return int(document.page_count)
    finally:
        document.close()


def test_verify_max_pages_defaults_to_two(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("VERIFY_MAX_PAGES", raising=False)
    assert verify_max_pages() == 2
    monkeypatch.setenv("VERIFY_MAX_PAGES", "4")
    assert verify_max_pages() == 4
    monkeypatch.setenv("VERIFY_MAX_PAGES", "0")
    assert verify_max_pages() == 1
    monkeypatch.setenv("VERIFY_MAX_PAGES", "nope")
    assert verify_max_pages() == 2


def test_verify_rules_are_the_split_categories() -> None:
    config = _split_config()
    rules = verify_classify_rules(config)
    by_type = {rule["type"]: rule["description"] for rule in rules}
    assert "Cover Page" in by_type
    assert "SLP_CIVIL" not in by_type
    assert "Advocate's Checklist" not in by_type
    assert by_type["Cover Page"].startswith("Outer title sheet")
    assert "Shared split rules:" not in by_type["Cover Page"]
    assert "REPRESENTATION DOCUMENTS" not in by_type["Cover Page"]
    for rule in rules:
        assert 10 <= len(rule["description"]) <= 400
        assert rule["type"].replace(" ", "").replace("-", "").replace("_", "").isalnum()
    payload = json.loads(Path("configs/config.json").read_text(encoding="utf-8"))
    long_cover = next(
        item["description"]
        for item in payload["split"]["categories"]
        if item["name"] == "Cover Page"
    )
    assert long_cover not in by_type["Cover Page"]
    names = {item["name"] for item in payload["split"]["categories"]}
    assert {rule["category"] for rule in rules} == names
    assert (
        split_category_for_classify_type("Advocates Checklist", config)
        == "Advocate's Checklist"
    )
    assert (
        split_category_for_classify_type("List of Dates and Events", config)
        == "List of Dates & Events"
    )
    assert split_category_for_classify_type("PoA-BR", config) == "PoA/BR"


def test_verify_classify_configuration_limits_pages() -> None:
    config = _split_config()
    first = verify_classify_configuration(config, pages=1)
    assert "mode" not in first
    assert first["parsing_configuration"] == {"max_pages": 1, "target_pages": "1"}
    assert any(rule["type"] == "Impugned Order" for rule in first["rules"])
    assert all("category" not in rule for rule in first["rules"])
    fallback = verify_classify_configuration(config, pages=2)
    assert fallback["parsing_configuration"] == {
        "max_pages": 2,
        "target_pages": "1-2",
    }


def test_leading_pdf_pages_stop_at_the_limit() -> None:
    original = _pdf_with_pages(3)
    short, kept = _leading_pdf_pages(original, 2)
    assert kept == 2
    assert _page_count(short) == 2
    untouched, kept_all = _leading_pdf_pages(original, 5)
    assert kept_all == 3
    assert untouched == original


class _FakeClassify:
    def __init__(self, labels: list[str | None]) -> None:
        self.labels = labels
        self.calls: list[dict] = []

    async def create(self, **kwargs):
        self.calls.append(kwargs)
        return SimpleNamespace(id=f"job-{len(self.calls)}")

    async def get(self, job_id: str, project_id: str | None = None):
        del project_id
        index = int(job_id.rsplit("-", 1)[-1]) - 1
        return SimpleNamespace(
            status="COMPLETED",
            result=SimpleNamespace(type=self.labels[index]),
        )


@pytest.mark.asyncio
async def test_verify_matches_on_first_page_without_a_second_call(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("VERIFY_MAX_PAGES", "2")
    uploaded: dict[str, bytes] = {}

    async def fake_download(client, file_id: str) -> bytes:
        del client, file_id
        return _pdf_with_pages(3)

    async def fake_upload(
        client, *, filename, pdf_bytes, external_file_id=None, purpose=None
    ):
        del client, filename, external_file_id, purpose
        uploaded["pdf"] = pdf_bytes
        return "file-uploaded"

    monkeypatch.setattr(
        "extraction_review.process_file._download_file_bytes", fake_download
    )
    monkeypatch.setattr("extraction_review.process_file._upload_slot_pdf", fake_upload)
    classify = _FakeClassify(["Cover Page"])
    result = await _verify_one_document(
        SimpleNamespace(classify=classify),
        item=FilingPartIn(
            filename="cover_page.pdf",
            file_id="dfl-verify",
            slot_id="cover_page",
        ),
        catalog=type_catalog("SLP_CIVIL"),
        split_config=_split_config(),
        semaphore=asyncio.Semaphore(1),
    )
    assert result.name == "cover_page.pdf"
    assert result.match is True
    assert len(classify.calls) == 1
    assert classify.calls[0]["configuration"]["parsing_configuration"]["max_pages"] == 1
    assert classify.calls[0]["file_input"] == "file-uploaded"
    assert _page_count(uploaded["pdf"]) == 2


@pytest.mark.asyncio
async def test_verify_falls_back_to_max_pages_when_page_one_does_not_match(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("VERIFY_MAX_PAGES", "2")

    async def fake_download(client, file_id: str) -> bytes:
        del client, file_id
        return _pdf_with_pages(4)

    async def fake_upload(
        client, *, filename, pdf_bytes, external_file_id=None, purpose=None
    ):
        del client, filename, pdf_bytes, external_file_id, purpose
        return "file-uploaded"

    monkeypatch.setattr(
        "extraction_review.process_file._download_file_bytes", fake_download
    )
    monkeypatch.setattr("extraction_review.process_file._upload_slot_pdf", fake_upload)
    classify = _FakeClassify([None, "Cover Page"])
    result = await _verify_one_document(
        SimpleNamespace(classify=classify),
        item=FilingPartIn(
            filename="cover_page.pdf",
            file_id="dfl-verify",
            slot_id="cover_page",
        ),
        catalog=type_catalog("SLP_CIVIL"),
        split_config=_split_config(),
        semaphore=asyncio.Semaphore(1),
    )
    assert result.match is True
    assert len(classify.calls) == 2
    assert classify.calls[0]["configuration"]["parsing_configuration"] == {
        "max_pages": 1,
        "target_pages": "1",
    }
    assert classify.calls[1]["configuration"]["parsing_configuration"] == {
        "max_pages": 2,
        "target_pages": "1-2",
    }


@pytest.mark.asyncio
async def test_verify_does_not_read_past_max_pages(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("VERIFY_MAX_PAGES", "1")

    async def fake_download(client, file_id: str) -> bytes:
        del client, file_id
        return _pdf_with_pages(5)

    async def fake_upload(
        client, *, filename, pdf_bytes, external_file_id=None, purpose=None
    ):
        del client, filename, external_file_id, purpose
        return pdf_bytes

    monkeypatch.setattr(
        "extraction_review.process_file._download_file_bytes", fake_download
    )
    monkeypatch.setattr("extraction_review.process_file._upload_slot_pdf", fake_upload)
    classify = _FakeClassify(["Vakalatnama"])
    result = await _verify_one_document(
        SimpleNamespace(classify=classify),
        item=FilingPartIn(
            filename="cover_page.pdf",
            file_id="dfl-verify",
            slot_id="cover_page",
        ),
        catalog=type_catalog("SLP_CIVIL"),
        split_config=_split_config(),
        semaphore=asyncio.Semaphore(1),
    )
    assert result.match is False
    assert len(classify.calls) == 1
    assert classify.calls[0]["file_input"]
    assert _page_count(classify.calls[0]["file_input"]) == 1
    assert classify.calls[0]["configuration"]["parsing_configuration"]["max_pages"] == 1


def test_annexure_family_label_matches_numbered_slot() -> None:
    catalog = type_catalog("ARBITRATION_PETITION")
    assert _label_matches_expected_slot(
        "Annexures", expected_slot="annexure_a5", catalog=catalog
    )
    assert _label_matches_expected_slot(
        "Annexures", expected_slot="annexure_a6", catalog=catalog
    )
    assert not _label_matches_expected_slot(
        "Cover Page", expected_slot="annexure_a5", catalog=catalog
    )
    assert _label_matches_expected_slot(
        "Application", expected_slot="application_2", catalog=catalog
    )


def test_classify_label_matches_expected_slot() -> None:
    catalog = type_catalog("SLP_CIVIL")
    assert _label_matches_expected_slot(
        "Cover Page", expected_slot="cover_page", catalog=catalog
    )
    assert not _label_matches_expected_slot(
        None, expected_slot="cover_page", catalog=catalog
    )


@pytest.mark.asyncio
async def test_classify_document_label_reads_completed_type() -> None:
    classify = _FakeClassify(["Index"])
    label = await _classify_document_label(
        SimpleNamespace(classify=classify),
        file_id="file-1",
        split_config=_split_config(),
        pages=1,
    )
    assert label == "Index"
    assert (
        classify.calls[0]["configuration"]["parsing_configuration"]["target_pages"]
        == "1"
    )


def test_parse_edited_flag_yes_no() -> None:
    assert parse_edited_flag("yes") is True
    assert parse_edited_flag("no") is False
    assert parse_edited_flag("true") is True
    assert parse_edited_flag(False) is False
    assert parse_edited_flag(None) is False
