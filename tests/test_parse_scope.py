"""Parse-scope skip/reuse without calling LlamaCloud."""

from __future__ import annotations

from types import SimpleNamespace

import pytest

from extraction_review.process_split_files import _parse_labeled_files
from extraction_review.split_upload import (
    SplitPartInput,
    extract_source_parts,
    type_catalog,
)


class _Ctx:
    def write_event_to_stream(self, _event: object) -> None:
        return None


@pytest.mark.asyncio
async def test_unparsed_scope_does_not_create_parse_for_reused_slot() -> None:
    created: list[str] = []

    class Parsing:
        async def create(self, **kwargs: object) -> SimpleNamespace:
            created.append(str(kwargs.get("file_id")))
            raise AssertionError("reused slot must not call parsing.create")

    catalog = type_catalog("SLP_CIVIL")
    pages, jobs, _layouts, _usage = await _parse_labeled_files(
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
    assert jobs["petition"] == "old-parse"


@pytest.mark.asyncio
async def test_extract_sources_scope_skips_annexure_without_create() -> None:
    created: list[str] = []

    class Parsing:
        async def create(self, **kwargs: object) -> SimpleNamespace:
            created.append(str(kwargs.get("file_id")))
            raise AssertionError("skipped extract-source slot must not parse")

    catalog = type_catalog("SLP_CIVIL")
    pages, jobs, _layouts, _usage = await _parse_labeled_files(
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
    assert jobs == {}
