"""Named-file verify: LlamaSplit labels vs the filename's expected slot."""

import pytest

from extraction_review.process_file import (
    parse_edited_flag,
    split_labels_match_expected_slot,
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


@pytest.mark.parametrize(
    "expected_slot", ["application_1", "application_2", "application_17"]
)
def test_generic_application_verifies_any_upload_ordinal(expected_slot: str) -> None:
    assert split_labels_match_expected_slot(
        expected_slot=expected_slot,
        catalog=type_catalog("SLP_CIVIL"),
        page_parts={1: ["Application"], 2: ["Application"]},
    )


@pytest.mark.parametrize(
    ("labels", "matches"),
    [
        ({1: ["Application 2"], 2: ["Application 2"]}, True),
        ({1: ["Application 1"], 2: ["Application 1"]}, False),
        ({1: ["Application 01"]}, False),
        ({1: ["Application", "Application 1"]}, False),
        ({1: ["Application"], 2: ["Main Petition"], 3: ["Main Petition"]}, False),
        ({1: ["Application"], 2: ["Application"], 3: ["Main Petition"]}, True),
        ({1: ["Application"], 2: ["Application 1"], 3: ["Application 1"]}, False),
        ({1: ["Application"], 2: ["Application"], 3: ["Application 1"]}, True),
        ({1: ["Affidavit"], 2: ["Affidavit"]}, False),
    ],
    ids=[
        "explicit_match",
        "explicit_mismatch",
        "explicit_leading_zero_mismatch",
        "explicit_overrides_generic_on_same_page",
        "petition_dominates",
        "generic_application_dominates_petition",
        "explicit_application_dominates",
        "generic_application_dominates_explicit",
        "different_family",
    ],
)
def test_application_verification_preserves_identity_and_dominant_category(
    labels: dict[int, list[str]], matches: bool
) -> None:
    assert (
        split_labels_match_expected_slot(
            expected_slot="application_2",
            catalog=type_catalog("SLP_CIVIL"),
            page_parts=labels,
        )
        is matches
    )


def test_overall_match_is_false_if_any_file_fails() -> None:
    matches = [True, False, True]
    overall = all(matches)
    assert overall is False
    assert all([True, True, True]) is True


def test_parse_edited_flag_yes_no() -> None:
    assert parse_edited_flag("yes") is True
    assert parse_edited_flag("no") is False
    assert parse_edited_flag("true") is True
    assert parse_edited_flag(False) is False
    assert parse_edited_flag(None) is False
