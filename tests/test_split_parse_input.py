import hashlib
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import AsyncMock

import pymupdf
import pytest

from extraction_review import process_file
from extraction_review.config import Config


def _parse_config():
    config = Config.model_validate_json(
        (Path(__file__).resolve().parents[1] / "configs/config.json").read_text()
    )
    assert config.parse is not None
    return config.parse


def _parse_result(page_numbers: list[int]):
    payload = {
        "job": {"status": "COMPLETED", "tier": "agentic"},
        "usage": {"credits": 2.0},
    }
    return SimpleNamespace(
        job=SimpleNamespace(status="COMPLETED", tier="agentic"),
        markdown=SimpleNamespace(
            pages=[
                SimpleNamespace(page_number=number, markdown=f"page {number}")
                for number in page_numbers
            ]
        ),
        text=None,
        model_dump=lambda **_kwargs: payload,
    )


def test_split_parse_request_enforces_ocr_layout_and_table_quality():
    request = process_file._split_parse_create_kwargs(
        _parse_config(), file_id="dfl-source"
    )
    configuration = request

    assert request["file_id"] == "dfl-source"
    assert configuration["tier"] == "agentic"
    assert set(configuration["output_options"]["granular_bboxes"]) == {
        "word",
        "line",
        "cell",
    }
    assert configuration["output_options"]["extract_printed_page_number"] is True
    assert configuration["output_options"]["markdown"]["tables"] == {
        "output_tables_as_markdown": True,
        "merge_continued_tables": False,
    }
    assert configuration["processing_options"]["aggressive_table_extraction"] is True
    assert configuration["processing_options"]["ocr_parameters"]["languages"] == ["en"]
    assert configuration["processing_control"]["job_failure_conditions"] == {
        "allowed_page_failure_ratio": 0.0001,
        "fail_on_image_ocr_error": True,
        "fail_on_markdown_reconstruction_error": True,
    }


@pytest.mark.asyncio
async def test_split_parse_rejects_missing_physical_pages(monkeypatch):
    result = _parse_result([1, 3])
    client = SimpleNamespace(
        parsing=SimpleNamespace(
            create=AsyncMock(return_value=SimpleNamespace(id="parse-job")),
            get=AsyncMock(return_value=result),
        )
    )
    monkeypatch.setattr(process_file, "collect_parse_usage", AsyncMock())

    with pytest.raises(RuntimeError, match=r"missing=\[2\]"):
        await process_file._parse_bundle_for_split(
            client,
            parse_config=_parse_config(),
            file_id="dfl-source",
            page_count=3,
        )


def test_page_text_falls_back_to_text_expansion_without_overwriting_markdown():
    result = SimpleNamespace(
        markdown=SimpleNamespace(
            pages=[SimpleNamespace(page_number=1, markdown="markdown one")]
        ),
        text=SimpleNamespace(
            pages=[
                SimpleNamespace(page_number=1, text="plain one"),
                SimpleNamespace(page_number=2, text="plain two"),
            ]
        ),
    )

    assert process_file._extract_page_markdown(result) == {
        1: "markdown one",
        2: "plain two",
    }


def test_page_text_preserves_distinct_llamaparse_header_and_footer():
    result = SimpleNamespace(
        markdown=SimpleNamespace(
            pages=[
                SimpleNamespace(
                    page_number=126,
                    header="118 | Annexure A/16 | P.W.15",
                    markdown="DEPOSITION OF WITNESS",
                    footer="TRUE COPY",
                )
            ]
        ),
        text=None,
    )

    assert process_file._extract_page_markdown(result) == {
        126: "118 | Annexure A/16 | P.W.15\nDEPOSITION OF WITNESS\nTRUE COPY"
    }


def test_parse_page_metadata_keeps_folio_separate_from_handwritten_label():
    result = SimpleNamespace(
        metadata=SimpleNamespace(
            pages=[
                SimpleNamespace(
                    page_number=126,
                    printed_page_number="118",
                    confidence=0.952,
                )
            ]
        )
    )

    folios, confidences = process_file._extract_parse_page_metadata(result)

    assert folios == {126: "118"}
    assert confidences == {126: 0.952}


def _pdf_bytes(page_count: int) -> bytes:
    document = pymupdf.open()
    try:
        for _ in range(page_count):
            document.new_page()
        return document.tobytes()
    finally:
        document.close()


@pytest.mark.asyncio
async def test_matching_extract_parse_is_reused_without_creating_another(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    pdf = _pdf_bytes(2)
    digest = hashlib.sha256(pdf).hexdigest()
    created: list[object] = []

    class Parsing:
        async def create(self, **_kwargs: object) -> None:
            created.append(_kwargs)
            raise AssertionError("parsing.create")

        async def get(self, job_id: str, **_kwargs: object) -> SimpleNamespace:
            assert job_id == "pjb-saved"
            return _parse_result([1, 2])

    class AgentData:
        async def get(self, item_id: str) -> SimpleNamespace:
            assert item_id == "agd-1"
            return SimpleNamespace(
                data={
                    "metadata": {
                        "source_parse_job_id": "pjb-saved",
                        "source_file_hash": digest,
                    }
                }
            )

    class Client:
        parsing = Parsing()
        beta = SimpleNamespace(agent_data=AgentData())

    monkeypatch.setattr(
        "extraction_review.s3_artifacts.download_json_object",
        lambda _key: None,
    )
    reused = await process_file._load_reused_split_parse(
        Client(),
        agent_data_id="agd-1",
        pdf_bytes=pdf,
    )
    assert created == []
    assert reused is not None
    job_id, pages, _result, _usage = reused
    assert job_id == "pjb-saved"
    assert pages[1] == "page 1"
    assert pages[2] == "page 2"


@pytest.mark.asyncio
async def test_different_pdf_does_not_reuse_the_saved_parse() -> None:
    pdf = _pdf_bytes(1)

    class Parsing:
        async def get(self, *_args: object, **_kwargs: object) -> None:
            raise AssertionError("hash mismatch must not read the parse job")

    class AgentData:
        async def get(self, _item_id: str) -> SimpleNamespace:
            return SimpleNamespace(
                data={
                    "metadata": {
                        "source_parse_job_id": "pjb-saved",
                        "source_file_hash": "a" * 64,
                    }
                }
            )

    class Client:
        parsing = Parsing()
        beta = SimpleNamespace(agent_data=AgentData())

    reused = await process_file._load_reused_split_parse(
        Client(),
        agent_data_id="agd-1",
        pdf_bytes=pdf,
    )
    assert reused is None
