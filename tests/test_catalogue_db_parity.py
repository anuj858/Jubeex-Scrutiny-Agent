"""Database catalogue matches the JSON catalogue, when a database is configured."""

from __future__ import annotations

import os

import pytest

from extraction_review.scrutiny.catalogue_db import catalogue_from_rows
from extraction_review.scrutiny.catalogue_export import legacy_selected_ids
from extraction_review.scrutiny.rules import (
    Defect,
    defects_for_filing_type,
    get_catalogue,
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


def test_database_failure_does_not_use_the_json_file(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("CATALOGUE_DATABASE_URL", raising=False)
    get_catalogue.cache_clear()
    with pytest.raises(RuntimeError, match="CATALOGUE_DATABASE_URL"):
        refresh_catalogue()

    def unavailable() -> int:
        raise OSError("connection refused")

    monkeypatch.setenv("CATALOGUE_DATABASE_URL", "postgresql://parse:secret@127.0.0.1:1/none")
    monkeypatch.setattr(
        "extraction_review.scrutiny.catalogue_db.fetch_catalogue_revision",
        unavailable,
    )
    get_catalogue.cache_clear()
    with pytest.raises(RuntimeError, match="could not be loaded from the database"):
        refresh_catalogue()
    get_catalogue.cache_clear()


def test_refresh_reloads_only_when_revision_changes(monkeypatch: pytest.MonkeyPatch) -> None:
    loads: list[int] = []
    revisions = [4]

    def fake_revision() -> int:
        return revisions[0]

    def fake_load(revision: int):
        loads.append(revision)
        return catalogue_from_rows([], [], [], revision)

    monkeypatch.setenv("CATALOGUE_DATABASE_URL", "postgresql://parse:secret@127.0.0.1:1/none")
    monkeypatch.setattr(
        "extraction_review.scrutiny.catalogue_db.fetch_catalogue_revision",
        fake_revision,
    )
    monkeypatch.setattr(
        "extraction_review.scrutiny.catalogue_db.load_catalogue_from_db",
        fake_load,
    )
    get_catalogue.cache_clear()
    first = refresh_catalogue()
    assert first.catalogue_version == "db:4"
    assert loads == [4]
    again = refresh_catalogue()
    assert again is first
    assert loads == [4]
    revisions[0] = 5
    refreshed = refresh_catalogue()
    assert refreshed.catalogue_version == "db:5"
    assert loads == [4, 5]
    get_catalogue.cache_clear()


@pytest.mark.skipif(
    not os.getenv("CATALOGUE_DATABASE_URL"),
    reason="CATALOGUE_DATABASE_URL is not set",
)
def test_database_selection_matches_file_catalogue(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("SCRUTINY_DEFECTS", "all")
    os.environ["SCRUTINY_DEFECTS"] = "all"
    get_catalogue.cache_clear()
    try:
        catalogue = refresh_catalogue()
    except RuntimeError as exc:
        get_catalogue.cache_clear()
        pytest.skip(str(exc))
    sample = catalogue.defects[0] if catalogue.defects else None
    if sample is None or not sample.court_code:
        get_catalogue.cache_clear()
        pytest.skip("database catalogue has no court on the parse view")

    plain = defects_for_filing_type("SLP_CIVIL", None)
    indigent = defects_for_filing_type("SLP_CIVIL", "Indigent person")
    get_catalogue.cache_clear()
    assert plain
    assert all(defect.release_stage == "production" for defect in plain)
    assert all(not defect.special_category for defect in plain)
    assert any(defect.is_global for defect in plain)
    assert any(
        (defect.special_category or "").lower().startswith("indigent") for defect in indigent
    )
    assert set(defect.check_id for defect in plain).issubset(
        defect.check_id for defect in indigent
    )


def test_live_selector_matches_legacy(monkeypatch: pytest.MonkeyPatch) -> None:
    from extraction_review.scrutiny.rules import _load_file_catalogue

    catalogue = _load_file_catalogue()
    monkeypatch.setattr(
        "extraction_review.scrutiny.rules.get_catalogue", lambda: catalogue
    )
    monkeypatch.setattr(
        "extraction_review.scrutiny.catalogue_export.get_catalogue", lambda: catalogue
    )
    monkeypatch.setenv("SCRUTINY_DEFECTS", "all")
    os.environ["SCRUTINY_DEFECTS"] = "all"
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
