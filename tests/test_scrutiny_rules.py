"""Petition-type filtering: Global/General + family + civil/criminal side."""

from __future__ import annotations

import json

import jsonschema

import extraction_review.scrutiny.rules as rules_mod
from extraction_review.scrutiny.prompts import _filing_phrase, build_defect_prompt
from extraction_review.scrutiny.rules import (
    Catalogue,
    Defect,
    _applies_to_filing,
    catalogue_schema_path,
    categories_for_filing_type,
    defects_for_filing_type,
    normalize_filing_type,
    normalize_special_category,
    rewrite_location_source,
    special_categories_for_catalog,
)


def _defect(
    check_id: str,
    main_category: str,
    special_category: str | None = None,
    notes: str | None = None,
) -> Defect:
    serial = check_id[1:]
    serial_no: int | str = int(serial) if serial.isdigit() else serial
    return Defect(
        check_id=check_id,
        serial_no=serial_no,
        main_category=main_category,
        special_category=special_category,
        defect="The petition is missing a required form or endorsement.",
        requirement="The required form or endorsement must be filed.",
        where_to_look=["Check the Cover Page"],
        how_to_cure=["File the missing form"],
        location_source="SCI Handbook",
        notes=notes,
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
        {"global", "slp_civil"}
    )
    filing = "slp_civil"
    assert _applies_to_filing(_defect("D100", "General/Global"), filing)
    assert _applies_to_filing(_defect("D101", "Global/General"), filing)
    assert not _applies_to_filing(_defect("D102", "SLP"), filing)
    assert _applies_to_filing(_defect("D103", "SLP (Civil)"), filing)
    assert _applies_to_filing(_defect("D104", "SLP(Civil)"), filing)
    assert not _applies_to_filing(_defect("D105", "SLP (Criminal)"), filing)
    assert not _applies_to_filing(_defect("D106", "Transfer Petition (Civil)"), filing)


def test_slp_criminal_runs_global_family_and_criminal_side() -> None:
    assert categories_for_filing_type("SLP_CRIMINAL") == frozenset(
        {"global", "slp_criminal"}
    )
    filing = "slp_criminal"
    assert _applies_to_filing(_defect("D100", "General/Global"), filing)
    assert not _applies_to_filing(_defect("D101", "SLP"), filing)
    assert _applies_to_filing(_defect("D102", "SLP (Criminal)"), filing)
    assert not _applies_to_filing(_defect("D103", "SLP (Civil)"), filing)


def test_family_pattern_extends_to_later_petition_types() -> None:
    assert categories_for_filing_type("TRANSFER_PETITION_CIVIL") == frozenset(
        {"global", "transfer_petition_civil"}
    )
    assert categories_for_filing_type("WRIT_PETITION_CRIMINAL") == frozenset(
        {"global", "writ_petition", "writ_petition_criminal"}
    )
    filing = "transfer_petition_civil"
    assert not _applies_to_filing(_defect("D200", "Transfer Petition"), filing)
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
    assert civil == ["D100", "D102"]
    assert criminal == ["D100", "D103"]


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
    assert normalize_filing_type("Civil Appeal") == "civil_appeal"
    assert normalize_filing_type("Criminal Appeal") == "criminal_appeal"
    assert (
        normalize_filing_type("Miscellaneous Application")
        == "miscellaneous_application"
    )
    assert normalize_filing_type("Interlocutory Application") == (
        "interlocutory_application"
    )
    assert categories_for_filing_type("REVIEW_PETITION_CIVIL") == frozenset(
        {"global", "review_petition_civil"}
    )
    assert categories_for_filing_type("CIVIL_APPEAL") == frozenset(
        {"global", "civil_appeal", "appeals"}
    )
    assert categories_for_filing_type("CRIMINAL_APPEAL") == frozenset(
        {"global", "criminal_appeal", "appeals"}
    )
    assert categories_for_filing_type("ORIGINAL_SUIT_CIVIL") == frozenset(
        {"global", "original_suit_civil"}
    )
    assert categories_for_filing_type("MISCELLANEOUS_APPLICATION") == frozenset(
        {"global", "miscellaneous_application"}
    )
    assert categories_for_filing_type("CURATIVE_PETITION_CRIMINAL") == frozenset(
        {"global", "curative_petition", "curative_petition_criminal"}
    )
    assert _applies_to_filing(
        _defect("D400", "Civil Appeal"), "civil_appeal"
    )
    assert _applies_to_filing(
        _defect("D401", "Miscellaneous Application"), "miscellaneous_application"
    )
    assert _applies_to_filing(
        _defect("D404", "Appeals"), "civil_appeal"
    )
    assert not _applies_to_filing(
        _defect("D402", "Interlocutory Application"), "miscellaneous_application"
    )
    assert not _applies_to_filing(
        _defect("D403", "Civil Appeal"), "miscellaneous_application"
    )


OPAQUE_SOURCE_FILES = {
    "2024011691-1.pdf": "SCI_RULES_2013",
    "Court Processes Handbook.pdf": "SCI_COURT_PROCESSES_HANDBOOK",
    "Form 28 - SLP.pdf": "SCI_FORM_28",
    "2024011779.pdf": "SCI_FORM_28",
    "2025010980.pdf": "SCI_CHECKLIST_2025",
    "Defect List.pdf": "SCI_CHECKLIST_2025",
    "2024042371.pdf": "SCI_COMPENDIUM_CIRCULARS",
    "12032020_071455.pdf": "SCI_CIRCULAR_2020",
    "Advocate's checklist.pdf": "SCBA_ADVOCATE_CHECKLIST",
    "circular_02.06.2017.pdf": "SCI_CIRCULAR_2017_06_02",
    "2024042339.pdf": "SCI_CIRCULARS_GUIDELINES_COMPENDIUM",
    "2024021053.pdf": "SCI_CIRCULAR_2024_02_07",
    "2024042323-1.pdf": "SCI_CIRCULAR_2008_11_21",
    "Armed Forces Tribunal Act.pdf": "AFT_ACT",
    "2026032086.pdf": "SCI_CIRCULAR_2026",
    "sc-amendment-rules-2019.pdf": "SCI_AMENDMENT_RULES_2019",
    "23082023_120659.pdf": "SCI_CIRCULAR_2022_58",
    "181015150934.pdf": "SCI_CIRCULAR_2018",
    "2025010360.pdf": "SCI_CIRCULAR_2025",
}


def test_imported_csv_catalogue(monkeypatch) -> None:
    rules_mod.get_catalogue.cache_clear()
    monkeypatch.setenv("SCRUTINY_DEFECTS", "all")

    catalogue = rules_mod.get_catalogue()
    assert catalogue.catalogue_version == "2.6.0"
    assert len(catalogue.defects) == 327
    schema = json.loads(catalogue_schema_path().read_text(encoding="utf-8"))
    jsonschema.validate(
        json.loads(rules_mod.catalogue_path().read_text(encoding="utf-8")),
        schema,
    )
    mapped = {
        alias: source.source_id
        for source in catalogue.sources
        for alias in source.filename_aliases
    }
    for filename, source_id in OPAQUE_SOURCE_FILES.items():
        assert mapped[filename] == source_id

    d063 = catalogue.defect("D063")
    assert d063.serial_no == 63
    assert d063.main_category == "General/Global"
    assert d063.category_id == "advocate_checklist"
    assert "Advocate's Checklist" in d063.inspect_parts
    assert "SCI_CHECKLIST_2025" in d063.location_source
    assert "Defect List.pdf" not in d063.location_source

    d162 = catalogue.defect("D162")
    assert d162.serial_no == 162
    assert d162.main_category == "SLP (Civil)"
    assert d162.category_id == "filing_formalities"
    assert "Main Petition" in d162.inspect_parts
    assert "SCI_FORM_28" in d162.location_source
    assert "SCI_RULES_2013" in d162.location_source
    assert "Form 28 - SLP.pdf" not in d162.location_source
    assert "2024011691-1.pdf" not in d162.location_source

    d001 = catalogue.defect("D001")
    assert d001.special_category == "Appeal (Armed Forces)"
    assert d001.main_category == "Appeals"

    assert catalogue.defect("D058A").serial_no == "58A"
    assert catalogue.defect("D331").serial_no == 331
    for dropped in ("D024", "D055", "D116", "D182"):
        assert catalogue.defect_by_id(dropped) is None

    civil = {d.check_id for d in defects_for_filing_type("SLP_CIVIL")}
    criminal = {d.check_id for d in defects_for_filing_type("SLP_CRIMINAL")}
    assert "D162" in civil
    assert "D162" not in criminal
    assert "D063" in civil
    assert "D063" in criminal
    assert "D001" not in civil
    motor = [
        d.check_id
        for d in catalogue.defects
        if d.special_category == "Motor Vehical Act"
    ]
    assert motor
    assert not (set(motor) & civil)

    noted = next(d for d in catalogue.defects if d.notes)
    prompt = build_defect_prompt(noted, record={}, chunks=[], catalogue=catalogue)
    assert "## Notes" in prompt
    assert noted.notes.splitlines()[0][:20] in prompt

    for defect in catalogue.defects:
        assert ".pdf" not in defect.location_source.lower()


def test_rewrite_location_source_maps_opaque_pdf_names() -> None:
    rules_mod.get_catalogue.cache_clear()
    catalogue = rules_mod.get_catalogue()
    for filename, source_id in OPAQUE_SOURCE_FILES.items():
        rewritten = rewrite_location_source(
            f"Page 30 of the PDF {filename}",
            catalogue.sources,
        )
        assert source_id in rewritten
        assert filename not in rewritten

    both = rewrite_location_source(
        "SCR, 2013, Page 30 of the PDF      2024011691-1.pdf\n"
        "Form No. 28  Form 28 - SLP.pdf",
        catalogue.sources,
    )
    assert "SCI_RULES_2013" in both
    assert "SCI_FORM_28" in both
    assert "2024011691-1.pdf" not in both
    assert "Form 28 - SLP.pdf" not in both


def test_normalize_special_treats_na_as_empty() -> None:
    assert normalize_special_category(None) == ""
    assert normalize_special_category("") == ""
    assert normalize_special_category("N/A") == ""
    assert normalize_special_category("n.a.") == ""
    assert normalize_special_category("Motor Vehicle Act") == "motor_vehicle_act"
    assert normalize_special_category("Motor Vehical Act") == "motor_vehicle_act"
    assert (
        normalize_special_category("Bail Applications/Bail Matters") == "bail_matters"
    )


def test_special_category_null_skips_tagged_defects(monkeypatch) -> None:
    catalogue = Catalogue(
        catalogue_id="test",
        schema_version="1",
        catalogue_version="1",
        jurisdiction="Supreme Court of India",
        defects=[
            _defect("D100", "SLP (Civil)"),
            _defect("D101", "SLP (Civil)", "Motor Vehical Act"),
            _defect("D102", "SLP (Civil)", "Eviction Matters"),
        ],
    )
    monkeypatch.setattr(rules_mod, "get_catalogue", lambda: catalogue)
    monkeypatch.setenv("SCRUTINY_DEFECTS", "all")

    none = [d.check_id for d in defects_for_filing_type("SLP_CIVIL")]
    na = [d.check_id for d in defects_for_filing_type("SLP_CIVIL", "N/A")]
    motor = [
        d.check_id
        for d in defects_for_filing_type("SLP_CIVIL", "Motor Vehical Act")
    ]
    assert none == ["D100"]
    assert na == ["D100"]
    assert motor == ["D100", "D101"]


def test_miscellaneous_application_overlay_from_special(monkeypatch) -> None:
    catalogue = Catalogue(
        catalogue_id="test",
        schema_version="1",
        catalogue_version="1",
        jurisdiction="Supreme Court of India",
        defects=[
            _defect("D100", "SLP (Civil)"),
            _defect("D101", "Miscellaneous Application"),
            _defect("D102", "Refiling Defects"),
        ],
    )
    monkeypatch.setattr(rules_mod, "get_catalogue", lambda: catalogue)
    monkeypatch.setenv("SCRUTINY_DEFECTS", "all")

    slp_none = [d.check_id for d in defects_for_filing_type("SLP_CIVIL")]
    slp_ma = [
        d.check_id
        for d in defects_for_filing_type("SLP_CIVIL", "Miscellaneous Application")
    ]
    slp_refile = [
        d.check_id for d in defects_for_filing_type("SLP_CIVIL", "Re-filing defect")
    ]
    ma_none = [
        d.check_id for d in defects_for_filing_type("MISCELLANEOUS_APPLICATION")
    ]
    assert slp_none == ["D100"]
    assert slp_ma == ["D100", "D101"]
    assert slp_refile == ["D100", "D102"]
    assert ma_none == ["D101"]


def test_special_category_respects_case_type_allow_list(monkeypatch) -> None:
    rules_mod.get_catalogue.cache_clear()
    monkeypatch.setenv("SCRUTINY_DEFECTS", "all")
    catalogue = rules_mod.get_catalogue()
    motor_ids = {
        d.check_id
        for d in catalogue.defects
        if d.special_category == "Motor Vehical Act"
    }
    armed = {
        d.check_id
        for d in catalogue.defects
        if d.special_category == "Appeal (Armed Forces)"
    }
    civil_none = {d.check_id for d in defects_for_filing_type("SLP_CIVIL")}
    civil_motor = {
        d.check_id
        for d in defects_for_filing_type("SLP_CIVIL", "Motor Vehical Act")
    }
    criminal_motor = {
        d.check_id
        for d in defects_for_filing_type("SLP_CRIMINAL", "Motor Vehical Act")
    }
    appeal_none = {d.check_id for d in defects_for_filing_type("CIVIL_APPEAL")}
    appeal_armed = {
        d.check_id
        for d in defects_for_filing_type("CIVIL_APPEAL", "Appeal (Armed Forces)")
    }
    appeal_bail = {
        d.check_id
        for d in defects_for_filing_type(
            "CIVIL_APPEAL", "Bail Applications/Bail Matters"
        )
    }
    assert motor_ids
    assert motor_ids.isdisjoint(civil_none)
    assert motor_ids <= civil_motor
    assert motor_ids.isdisjoint(criminal_motor)
    assert armed
    assert armed.isdisjoint(appeal_none)
    assert armed <= appeal_armed
    assert "D001" in appeal_armed
    assert "D001" not in appeal_none
    bail = {
        d.check_id
        for d in catalogue.defects
        if d.special_category == "Bail Applications/Bail Matters"
    }
    assert bail.isdisjoint(appeal_bail)

    listed = special_categories_for_catalog()
    assert "Motor Vehical Act" in listed["SLP_CIVIL"]
    assert "Motor Vehical Act" not in listed["SLP_CRIMINAL"]

