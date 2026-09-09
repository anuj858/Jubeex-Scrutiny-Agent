"""Petition-type filtering: Global/General + family + civil/criminal side."""

from __future__ import annotations

import extraction_review.scrutiny.rules as rules_mod
from extraction_review.scrutiny.prompts import _filing_phrase
from extraction_review.scrutiny.rules import (
    Catalogue,
    Defect,
    _applies_to_filing,
    categories_for_filing_type,
    defects_for_filing_type,
    normalize_filing_type,
)


def _defect(check_id: str, main_category: str) -> Defect:
    return Defect(
        check_id=check_id,
        serial_no=int(check_id[1:]),
        main_category=main_category,
        defect="The petition is missing a required form or endorsement.",
        requirement="The required form or endorsement must be filed.",
        where_to_look=["Check the Cover Page"],
        how_to_cure=["File the missing form"],
        location_source="SCI Handbook",
    )


def test_normalize_slp_aliases() -> None:
    assert normalize_filing_type("SLP_CIVIL") == "slp_civil"
    assert normalize_filing_type("SLP (Civil)") == "slp_civil"
    assert normalize_filing_type("SLP(Civil)") == "slp_civil"
    assert normalize_filing_type("Special Leave Petition (Civil)") == "slp_civil"
    assert normalize_filing_type("SLP_CRIMINAL") == "slp_criminal"
    assert normalize_filing_type("SLP (Criminal)") == "slp_criminal"
    assert normalize_filing_type("SLP") == "slp"
    assert normalize_filing_type("General/Global") == "global"
    assert normalize_filing_type("Global/General") == "global"


def test_slp_civil_runs_global_family_and_civil_side() -> None:
    assert categories_for_filing_type("SLP_CIVIL") == frozenset(
        {"global", "slp", "slp_civil"}
    )
    filing = "slp_civil"
    assert _applies_to_filing(_defect("D100", "General/Global"), filing)
    assert _applies_to_filing(_defect("D101", "Global/General"), filing)
    assert _applies_to_filing(_defect("D102", "SLP"), filing)
    assert _applies_to_filing(_defect("D103", "SLP (Civil)"), filing)
    assert _applies_to_filing(_defect("D104", "SLP(Civil)"), filing)
    assert not _applies_to_filing(_defect("D105", "SLP (Criminal)"), filing)
    assert not _applies_to_filing(_defect("D106", "Transfer Petition (Civil)"), filing)


def test_slp_criminal_runs_global_family_and_criminal_side() -> None:
    assert categories_for_filing_type("SLP_CRIMINAL") == frozenset(
        {"global", "slp", "slp_criminal"}
    )
    filing = "slp_criminal"
    assert _applies_to_filing(_defect("D100", "General/Global"), filing)
    assert _applies_to_filing(_defect("D101", "SLP"), filing)
    assert _applies_to_filing(_defect("D102", "SLP (Criminal)"), filing)
    assert not _applies_to_filing(_defect("D103", "SLP (Civil)"), filing)


def test_family_pattern_extends_to_later_petition_types() -> None:
    assert categories_for_filing_type("TRANSFER_PETITION_CIVIL") == frozenset(
        {"global", "transfer_petition", "transfer_petition_civil"}
    )
    assert categories_for_filing_type("WRIT_PETITION_CRIMINAL") == frozenset(
        {"global", "writ_petition", "writ_petition_criminal"}
    )
    filing = "transfer_petition_civil"
    assert _applies_to_filing(_defect("D200", "Transfer Petition"), filing)
    assert _applies_to_filing(_defect("D201", "Transfer Petition (Civil)"), filing)
    assert not _applies_to_filing(
        _defect("D202", "Transfer Petition (Criminal)"), filing
    )
    assert not _applies_to_filing(_defect("D203", "SLP"), filing)


def test_defects_for_filing_type_selects_matching_main_categories(
    monkeypatch,
) -> None:
    catalogue = Catalogue(
        catalogue_id="test",
        schema_version="1",
        catalogue_version="1",
        jurisdiction="Supreme Court of India",
        defects=[
            _defect("D100", "General/Global"),
            _defect("D101", "SLP"),
            _defect("D102", "SLP (Civil)"),
            _defect("D103", "SLP (Criminal)"),
            _defect("D104", "Transfer Petition (Civil)"),
        ],
    )
    monkeypatch.setattr(rules_mod, "get_catalogue", lambda: catalogue)
    monkeypatch.setenv("SCRUTINY_DEFECTS", "all")

    civil = [d.check_id for d in defects_for_filing_type("SLP_CIVIL")]
    criminal = [d.check_id for d in defects_for_filing_type("SLP_CRIMINAL")]
    assert civil == ["D100", "D101", "D102"]
    assert criminal == ["D100", "D101", "D103"]


def test_extracted_petition_type_label_selects_slp_civil(monkeypatch) -> None:
    catalogue = Catalogue(
        catalogue_id="test",
        schema_version="1",
        catalogue_version="1",
        jurisdiction="Supreme Court of India",
        defects=[
            _defect("D100", "General/Global"),
            _defect("D102", "SLP (Civil)"),
            _defect("D103", "SLP (Criminal)"),
        ],
    )
    monkeypatch.setattr(rules_mod, "get_catalogue", lambda: catalogue)
    monkeypatch.setenv("SCRUTINY_DEFECTS", "all")

    ids = [
        d.check_id for d in defects_for_filing_type("Special Leave Petition (Civil)")
    ]
    assert ids == ["D100", "D102"]


def test_filing_phrase_distinguishes_global_family_and_side() -> None:
    assert "every petition type" in _filing_phrase("General/Global")
    assert "shared across civil and criminal" in _filing_phrase("SLP")
    assert _filing_phrase("SLP (Civil)") == "this SLP (Civil) filing"
    phrase = _filing_phrase(
        "SLP (Civil), SLP (Criminal), Civil Appeal, Criminal Appeal, "
        "Writ Petition (Civil), Writ Petition (Criminal), Transfer Petition (Civil)"
    )
    assert phrase.startswith("this filing (the check applies to ")
    assert "SLP (Civil)" in phrase
    assert "Civil Appeal" in phrase


def test_comma_separated_main_category_matches_any_listed_type() -> None:
    listed = (
        "SLP (Civil), SLP (Criminal), Civil Appeal, Criminal Appeal, "
        "Writ Petition (Civil),Writ Petition (Criminal), Transfer Petition (Civil)"
    )
    defect = _defect("D300", listed)
    assert defect.main_categories == (
        "SLP (Civil)",
        "SLP (Criminal)",
        "Civil Appeal",
        "Criminal Appeal",
        "Writ Petition (Civil)",
        "Writ Petition (Criminal)",
        "Transfer Petition (Civil)",
    )
    assert _applies_to_filing(defect, "slp_civil")
    assert _applies_to_filing(defect, "slp_criminal")
    assert _applies_to_filing(defect, "writ_petition_civil")
    assert _applies_to_filing(defect, "writ_petition_criminal")
    assert _applies_to_filing(defect, "transfer_petition_civil")
    assert _applies_to_filing(defect, "civil_appeal")
    assert _applies_to_filing(defect, "criminal_appeal")
    assert not _applies_to_filing(defect, "transfer_petition_criminal")
    assert not _applies_to_filing(defect, "arbitration_petition")


def test_main_category_json_array_is_accepted() -> None:
    defect = Defect(
        check_id="D301",
        serial_no=301,
        main_category=["SLP (Civil)", "SLP (Criminal)"],
        defect="The petition is missing a required form or endorsement.",
        requirement="The required form or endorsement must be filed.",
        where_to_look=["Check the Cover Page"],
        how_to_cure=["File the missing form"],
        location_source="SCI Handbook",
    )
    assert defect.main_category == "SLP (Civil), SLP (Criminal)"
    assert _applies_to_filing(defect, "slp_civil")
    assert _applies_to_filing(defect, "slp_criminal")
    assert not _applies_to_filing(defect, "writ_petition_civil")


def test_defects_for_filing_type_includes_multi_category_rows(monkeypatch) -> None:
    catalogue = Catalogue(
        catalogue_id="test",
        schema_version="1",
        catalogue_version="1",
        jurisdiction="Supreme Court of India",
        defects=[
            _defect("D100", "General/Global"),
            _defect(
                "D300",
                "SLP (Civil), SLP (Criminal), Civil Appeal, Criminal Appeal, "
                "Writ Petition (Civil), Writ Petition (Criminal), "
                "Transfer Petition (Civil)",
            ),
            _defect("D301", "SLP (Criminal)"),
        ],
    )
    monkeypatch.setattr(rules_mod, "get_catalogue", lambda: catalogue)
    monkeypatch.setenv("SCRUTINY_DEFECTS", "all")

    civil = [d.check_id for d in defects_for_filing_type("SLP_CIVIL")]
    criminal = [d.check_id for d in defects_for_filing_type("SLP_CRIMINAL")]
    writ = [d.check_id for d in defects_for_filing_type("WRIT_PETITION_CIVIL")]
    tp_criminal = [
        d.check_id for d in defects_for_filing_type("TRANSFER_PETITION_CRIMINAL")
    ]
    assert civil == ["D100", "D300"]
    assert criminal == ["D100", "D300", "D301"]
    assert writ == ["D100", "D300"]
    assert tp_criminal == ["D100"]


def test_slash_separated_main_category_matches_both_slp_sides() -> None:
    defect = _defect("D231", "SLP (Civil)/SLP (Criminal)")
    assert defect.main_categories == ("SLP (Civil)", "SLP (Criminal)")
    assert _applies_to_filing(defect, "slp_civil")
    assert _applies_to_filing(defect, "slp_criminal")
    assert not _applies_to_filing(defect, "writ_petition_civil")


def test_review_contempt_and_original_suit_aliases() -> None:
    assert normalize_filing_type("Review Petition (Civil)") == "review_petition_civil"
    assert (
        normalize_filing_type("Contempt Petition (Criminal)")
        == "contempt_petition_criminal"
    )
    assert (
        normalize_filing_type("Election Petition (Civil)") == "election_petition_civil"
    )
    assert (
        normalize_filing_type("Curative Petition (Civil)") == "curative_petition_civil"
    )
    assert normalize_filing_type("Original Suit (Civil)") == "original_suit_civil"
    assert categories_for_filing_type("REVIEW_PETITION_CIVIL") == frozenset(
        {"global", "review_petition", "review_petition_civil"}
    )


def test_imported_csv_defects_select_by_slp_side(monkeypatch) -> None:
    rules_mod.get_catalogue.cache_clear()
    monkeypatch.setenv("SCRUTINY_DEFECTS", "all")

    catalogue = rules_mod.get_catalogue()
    assert catalogue.catalogue_version == "2.3.0"
    assert len(catalogue.defects) == 94

    by_serial = {str(d.serial_no): d.check_id for d in catalogue.defects}
    form28 = {by_serial[s] for s in ("162", "163", "164", "165", "166", "167")}
    shared_slp = {by_serial[s] for s in ("231", "232")}
    jail_ia = {by_serial[s] for s in ("199", "200", "201", "202", "203", "204", "205")}
    criminal_only = {by_serial[s] for s in ("270", "271", "272", "273")}

    civil = {d.check_id for d in defects_for_filing_type("SLP_CIVIL")}
    criminal = {d.check_id for d in defects_for_filing_type("SLP_CRIMINAL")}

    assert form28 <= civil
    assert form28.isdisjoint(criminal)
    assert criminal_only <= criminal
    assert criminal_only.isdisjoint(civil)
    assert shared_slp <= civil & criminal
    assert jail_ia <= civil & criminal

    d092 = catalogue.defect(by_serial["231"])
    assert d092.main_category == "SLP (Civil), SLP (Criminal)"
    assert d092.inspect_parts == ["AOR's Certificate"]
    d079 = catalogue.defect(by_serial["162"])
    assert d079.category_id == "filing_formalities"
    assert d079.inspect_parts == ["Main Petition"]
