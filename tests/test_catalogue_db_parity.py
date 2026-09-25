"""Database catalogue matches the JSON catalogue, when a database is configured."""

from __future__ import annotations

import os

import pytest

from extraction_review.config import JUBEEX_FILING_TYPES
from extraction_review.scrutiny.catalogue_db import catalogue_from_rows
from extraction_review.scrutiny.catalogue_export import legacy_selected_ids
from extraction_review.scrutiny.rules import (
    Defect,
    defects_for_filing_type,
    get_case_types,
    get_catalogue,
    normalize_filing_type,
    refresh_catalogue,
)


def test_rows_become_an_explicit_catalogue() -> None:
    catalogue = catalogue_from_rows(
        [
            {
                "check_id": "D-9",
                "defect_version": 3,
                "defect": "Missing index",
                "requirement": "An index is required",
                "where_to_look": ["Index"],
                "how_to_cure": ["Add an index"],
                "trigger_words": ["index", "contents"],
                "main_category": "General/Global",
                "special_category": None,
                "special_category_mode": None,
                "applies_to_all_petition_types": True,
                "agent_filing_types": ["SLP_CIVIL"],
                "parent_check_id": None,
                "is_enabled": True,
                "inspect_parts": ["Index"],
                "context_parts": [],
                "exclude_parts": None,
                "applicable_rule": None,
                "location_source": "SCI_RULES_2013",
                "notes": None,
                "overlap_note": None,
            }
        ],
        [
            {
                "source_id": "SCI_RULES_2013",
                "title": "Supreme Court Rules",
                "authority_type": "rules",
                "url": None,
                "alternate_urls": [],
                "filename_aliases": [],
                "issued_date": None,
                "effective_date": None,
                "checksum": None,
                "locators": {},
            }
        ],
        [{"agent_filing_type": "SLP_CIVIL", "special_category": "Tax Matters"}],
        12,
    )
    assert catalogue.catalogue_version == "db:12"
    defect = catalogue.defect("D-9")
    assert defect.use_explicit_applicability is True
    assert defect.defect_version == 3
    assert defect.trigger_words == "index; contents"
    assert defect.agent_filing_types == ["slp_civil"]
    assert catalogue.petition_specials["slp_civil"] == ["Tax Matters"]


def test_database_failure_falls_back_to_json_file(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("CATALOGUE_SOURCE", "db")
    monkeypatch.delenv("CATALOGUE_DATABASE_URL", raising=False)
    get_catalogue.cache_clear()
    missing_url = refresh_catalogue()
    assert missing_url.catalogue_id == "sci_registry_defects"
    assert missing_url.catalogue_version == "3.0.1"
    assert len(missing_url.defects) == 321

    def unavailable() -> int:
        raise OSError("connection refused")

    monkeypatch.setenv("CATALOGUE_DATABASE_URL", "postgresql://parse:secret@127.0.0.1:1/none")
    monkeypatch.setattr(
        "extraction_review.scrutiny.catalogue_db.fetch_catalogue_revision",
        unavailable,
    )
    get_catalogue.cache_clear()
    loaded = refresh_catalogue()
    assert loaded.catalogue_version == "3.0.1"
    assert len(loaded.defects) == 321
    assert loaded.defect_by_id("D-1") is not None
    get_catalogue.cache_clear()


def test_refresh_reloads_only_when_revision_changes(monkeypatch: pytest.MonkeyPatch) -> None:
    loads: list[int] = []

    def fake_revision() -> int:
        return revisions[0]

    def fake_load(revision: int) -> object:
        loads.append(revision)
        return get_catalogue()

    revisions = [4]
    monkeypatch.setenv("CATALOGUE_SOURCE", "db")
    monkeypatch.setattr(
        "extraction_review.scrutiny.catalogue_db.fetch_catalogue_revision",
        fake_revision,
    )
    monkeypatch.setattr(
        "extraction_review.scrutiny.catalogue_db.load_catalogue_from_db",
        fake_load,
    )
    get_catalogue.cache_clear()
    # Prime the file catalogue, then switch the cache into database mode.
    monkeypatch.setenv("CATALOGUE_SOURCE", "file")
    first = refresh_catalogue()
    assert first.catalogue_version == "3.0.1"

    monkeypatch.setenv("CATALOGUE_SOURCE", "db")
    # The patched loader returns the already-built file catalogue and records the revision.
    loaded = refresh_catalogue()
    assert loaded is first
    assert loads == [4]
    again = refresh_catalogue()
    assert again is first
    assert loads == [4]
    revisions[0] = 5
    refresh_catalogue()
    assert loads == [4, 5]
    get_catalogue.cache_clear()


def _specials_for(filing_type: str) -> list[str | None]:
    spec = get_case_types().get(normalize_filing_type(filing_type)) or {}
    labels = [str(label) for label in spec.get("special_categories") or []]
    return [None, "N/A", *labels]


@pytest.mark.skipif(
    not os.getenv("CATALOGUE_DATABASE_URL"),
    reason="CATALOGUE_DATABASE_URL is not set",
)
def test_database_selection_matches_file_catalogue(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("SCRUTINY_DEFECTS", "all")
    os.environ["SCRUTINY_DEFECTS"] = "all"
    monkeypatch.setenv("CATALOGUE_SOURCE", "file")
    get_catalogue.cache_clear()
    file_ids = {
        filing: {
            special: legacy_selected_ids(filing, special)
            for special in _specials_for(filing)
        }
        for filing in JUBEEX_FILING_TYPES
    }

    monkeypatch.setenv("CATALOGUE_SOURCE", "db")
    get_catalogue.cache_clear()
    refresh_catalogue()
    mismatches: list[str] = []
    for filing in JUBEEX_FILING_TYPES:
        for special in _specials_for(filing):
            selected = [defect.check_id for defect in defects_for_filing_type(filing, special)]
            expected = file_ids[filing][special]
            if selected != expected:
                mismatches.append(f"{filing} / {special}: file {len(expected)} db {len(selected)}")
    get_catalogue.cache_clear()
    assert not mismatches, "\n".join(mismatches[:20])


def test_live_selector_matches_legacy(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("CATALOGUE_SOURCE", "file")
    monkeypatch.setenv("SCRUTINY_DEFECTS", "all")
    os.environ["SCRUTINY_DEFECTS"] = "all"
    get_catalogue.cache_clear()
    filing = "SLP_CIVIL"
    assert [d.check_id for d in defects_for_filing_type(filing)] == legacy_selected_ids(filing, None)
    sample = Defect.model_validate(
        {
            "check_id": "D-1",
            "main_category": "General/Global",
            "defect": "x",
            "requirement": "y",
            "where_to_look": ["z"],
            "how_to_cure": ["c"],
            "location_source": "s",
        }
    )
    assert sample.use_explicit_applicability is False
