import pytest

from extraction_review.config import ExtractConfig, dump_api_configuration
from extraction_review.local_extraction import LocalParseExtractor, extraction_provider


@pytest.fixture(autouse=True)
def _json_catalogue(monkeypatch):
    monkeypatch.setenv("SCRUTINY_CATALOGUE_SOURCE", "json")


def test_local_extraction_is_default_and_legacy_provider_is_explicit(monkeypatch):
    monkeypatch.delenv("EXTRACTION_PROVIDER", raising=False)
    assert extraction_provider() == "local"
    monkeypatch.setenv("EXTRACTION_PROVIDER", "llamacloud")
    assert extraction_provider() == "llamacloud"


def test_local_rules_are_not_forwarded_to_llamacloud():
    config = ExtractConfig.model_validate(
        {
            "product_type": "extract_v2",
            "data_schema": {},
            "local_extraction": {"enabled_fields": ["court"]},
        }
    )
    assert "local_extraction" not in dump_api_configuration(config)


def test_local_extraction_uses_full_parse_and_physical_page_provenance():
    pages = {
        1: """
# IN THE SUPREME COURT OF INDIA
SPECIAL LEAVE PETITION (CIVIL) NO. 10 OF 2026
(Against the final judgment and order dated 28.01.2021 passed by the
High Court of Delhi in W.P.(C) No. 8749 of 2020)
""",
        2: """
1. Meera Krishnan ... Petitioner
VERSUS
1. Union of India ... Respondent
""",
        37: """
MAIN PRAYER
It is therefore most respectfully prayed that this Hon'ble Court may grant
special leave to appeal against the impugned judgment and pass such other order.

Date Drafted: 16.11.2026
DRAWN & FILED BY
Ravi Kumar
ADVOCATE-ON-RECORD
AOR Code: 1234
Chamber 22, Supreme Court Compound, New Delhi 110001
Email: ravi@example.com Ph.: 23386175
""",
    }
    parts = {1: ["Cover Page"], 2: ["Main Petition"], 37: ["Main Petition"]}

    result = LocalParseExtractor().extract(
        page_markdown=pages,
        page_parts=parts,
        filing_type="SLP_CIVIL",
        front_extract=True,
    )

    assert result.data["court"] == "Supreme Court of India"
    assert result.data["petition_type"] == "Special Leave Petition (Civil)"
    assert result.data["petition_date"] == "16.11.2026"
    assert "special leave to appeal" in result.data["relief_sort"]
    assert result.data["impugned_orders"][0]["order_date"] == "28.01.2021"
    assert (
        result.data["impugned_orders"][0]["case_number"] == "W.P.(C) No. 8749 of 2020"
    )
    assert result.data["impugned_orders"][0]["Forum"] == "High Court of Delhi"
    assert result.data["advocates_on_record"][0]["email"] == "ravi@example.com"
    assert result.provenance["court"][0]["physical_page"] == 1
    assert result.provenance["relief_sort"][0]["physical_page"] == 37


def test_local_extraction_does_not_invent_missing_values():
    result = LocalParseExtractor().extract(
        page_markdown={1: "Scanned page without readable filing fields"},
        page_parts={1: ["Unidentified"]},
        filing_type=None,
        front_extract=True,
    )

    assert "court" not in result.data
    assert result.data.get("advocates_on_record", []) == []
    assert result.data.get("impugned_orders", []) == []
    assert result.data.get("petitioners", []) == []
    assert result.data.get("respondents", []) == []


def test_enabled_fields_are_configurable():
    extractor = LocalParseExtractor({"enabled_fields": ["court"]})
    result = extractor.extract(
        page_markdown={1: "IN THE SUPREME COURT OF INDIA\nCIVIL APPEAL NO. 1 OF 2026"},
        page_parts={1: ["Cover Page"]},
        filing_type=None,
    )

    assert result.data["court"] == "Supreme Court of India"
    assert "petition_type" not in result.data
