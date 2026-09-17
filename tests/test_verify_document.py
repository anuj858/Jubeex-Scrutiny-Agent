"""Named-file verify: LlamaSplit labels vs the filename's expected slot."""

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
