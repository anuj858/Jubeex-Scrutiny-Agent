"""Explicit applicability selects the same defects as the string-based rules."""

from __future__ import annotations

import os

import pytest

from extraction_review.config import JUBEEX_FILING_TYPES
from extraction_review.scrutiny.catalogue_export import (
    build_seed,
    explicit_selected_ids,
    legacy_selected_ids,
)
from extraction_review.scrutiny.rules import get_case_types, normalize_filing_type


@pytest.fixture(autouse=True)
def _all_defects_enabled(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("SCRUTINY_DEFECTS", "all")
    # Import-time readers and a developer .env must not narrow the comparison.
    os.environ["SCRUTINY_DEFECTS"] = "all"


def _specials_for(filing_type: str) -> list[str | None]:
    spec = get_case_types().get(normalize_filing_type(filing_type)) or {}
    labels = [str(label) for label in spec.get("special_categories") or []]
    return [None, "N/A", "Not A Real Category", *labels]


def test_explicit_applicability_matches_legacy_selector() -> None:
    mismatches: list[str] = []
    for filing_type in JUBEEX_FILING_TYPES:
        for special in _specials_for(filing_type):
            legacy = legacy_selected_ids(filing_type, special)
            explicit = explicit_selected_ids(filing_type, special)
            if legacy != explicit:
                missing = sorted(set(legacy) - set(explicit))
                extra = sorted(set(explicit) - set(legacy))
                mismatches.append(
                    f"{filing_type} special={special!r} "
                    f"missing={missing[:8]} extra={extra[:8]} "
                    f"legacy={len(legacy)} explicit={len(explicit)}"
                )
    assert not mismatches, "\n".join(mismatches)


def test_seed_covers_every_defect_and_source() -> None:
    seed = build_seed()
    assert len(seed["defects"]) == 321
    assert len(seed["sources"]) == 21
    assert len(seed["petition_types"]) == len(JUBEEX_FILING_TYPES)
    codes = [row["code"] for row in seed["defects"]]
    assert len(codes) == len(set(codes))
    assert any(row["create"] for row in seed["document_types"])
    assert any(
        row["code"] == "election-petition-civil" for row in seed["petition_types"]
    )
