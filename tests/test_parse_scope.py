"""Parse-scope skip/reuse without calling LlamaCloud."""

from __future__ import annotations

import asyncio
from types import SimpleNamespace

import pytest

import extraction_review.process_split_files as split_workflow
from extraction_review.config import ParseConfig
from extraction_review.process_split_files import _parse_labeled_files, _parse_one_file
from extraction_review.split_upload import (
    SplitPartInput,
    extract_source_parts,
    type_catalog,
)


class _Ctx:
    def write_event_to_stream(self, _event: object) -> None:
        return None


@pytest.mark.asyncio
async def test_full_parse_requests_expanded_structured_outputs(monkeypatch) -> None:
    requested: list[str] = []

    class Parsing:
        async def create(self, **_kwargs: object) -> SimpleNamespace:
            return SimpleNamespace(id="parse-1")

        async def get(self, _job_id: str, **kwargs: object) -> object:
            expand = kwargs.get("expand")
            assert isinstance(expand, list)
            requested.extend(str(value) for value in expand)
            page = {"page_number": 1}
            payload = {
                "markdown": {"pages": [{**page, "markdown": "# DECLARATION"}]},
                "text": {"pages": [{**page, "text": "DECLARATION"}]},
                "items": {
                    "pages": [
                        {
                            **page,
                            "success": True,
                            "items": [
                                {"type": "heading", "level": 1, "md": "# DECLARATION"}
                            ],
                        }
                    ]
                },
                "metadata": {"pages": [{**page, "confidence": 1.0}]},
                "usage": {},
            }
            return SimpleNamespace(
                markdown=SimpleNamespace(
                    pages=[SimpleNamespace(page_number=1, markdown="# DECLARATION")]
                ),
                model_dump=lambda **_kwargs: payload,
            )

    async def no_wait(*_args: object, **_kwargs: object) -> None:
        return None

    async def no_sidecar(_url: str | None) -> list[object]:
        return []

    async def usage(*_args: object, **_kwargs: object) -> dict[str, object]:
        return {}

    monkeypatch.setattr(split_workflow, "_wait_for_parse", no_wait)
    monkeypatch.setattr(split_workflow, "download_sidecar_pages", no_sidecar)
    monkeypatch.setattr(split_workflow, "collect_parse_usage", usage)
    result = await _parse_one_file(
        SimpleNamespace(parsing=Parsing()),
        parse_config=ParseConfig(
            product_type="parse_v2", tier="agentic", version="latest"
        ),
        part=SplitPartInput(
            slot_id="petition",
            file_id="dfl-petition",
            document_parts=("Main Petition",),
        ),
        source_parts={"Main Petition"},
        semaphore=asyncio.Semaphore(1),
        ctx=_Ctx(),  # type: ignore[arg-type]
    )
    assert requested == ["markdown", "text", "items", "metadata", "usage"]
    assert result[1] == {1: "# DECLARATION"}
    assert result[2][1]["items"][0]["type"] == "heading"


@pytest.mark.asyncio
async def test_unparsed_scope_does_not_create_parse_for_reused_slot() -> None:
    created: list[str] = []

    class Parsing:
        async def create(self, **kwargs: object) -> SimpleNamespace:
            created.append(str(kwargs.get("file_id")))
            raise AssertionError("reused slot must not call parsing.create")

    catalog = type_catalog("SLP_CIVIL")
    pages, structured, jobs, _layouts, _usage = await _parse_labeled_files(
        SimpleNamespace(parsing=Parsing()),
        parse_config=SimpleNamespace(),
        parts=[
            SplitPartInput(
                slot_id="petition",
                file_id="dfl-petition",
                document_parts=("Main Petition",),
            )
        ],
        source_parts=extract_source_parts(catalog),
        catalog=catalog,
        parse_scope="unparsed",
        reuse_slots={"petition"},
        reuse_pages={"petition": {1: "kept markdown"}},
        reuse_layouts={"petition": {1: {"words": []}}},
        reuse_job_ids={"petition": "old-parse"},
        ctx=_Ctx(),  # type: ignore[arg-type]
    )
    assert created == []
    assert pages["petition"][1] == "kept markdown"
    assert structured["petition"] == {}
    assert jobs["petition"] == "old-parse"


@pytest.mark.asyncio
async def test_extract_sources_scope_skips_annexure_without_create() -> None:
    created: list[str] = []

    class Parsing:
        async def create(self, **kwargs: object) -> SimpleNamespace:
            created.append(str(kwargs.get("file_id")))
            raise AssertionError("skipped extract-source slot must not parse")

    catalog = type_catalog("SLP_CIVIL")
    pages, structured, jobs, _layouts, _usage = await _parse_labeled_files(
        SimpleNamespace(parsing=Parsing()),
        parse_config=SimpleNamespace(),
        parts=[
            SplitPartInput(
                slot_id="annexure_p1",
                file_id="dfl-ann",
                document_parts=("Annexure P-1",),
            )
        ],
        source_parts=extract_source_parts(catalog),
        catalog=catalog,
        parse_scope="extract_sources",
        reuse_slots=set(),
        ctx=_Ctx(),  # type: ignore[arg-type]
    )
    assert created == []
    assert pages["annexure_p1"] == {}
    assert structured["annexure_p1"] == {}
    assert jobs == {}
