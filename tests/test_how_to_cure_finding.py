"""Findings copy hub_defects cure sentences and buttons without rewriting them."""

from extraction_review.scrutiny.catalogue_db import catalogue_from_rows
from extraction_review.scrutiny.schema import failed_finding

_D4_STEP = (
    "Option for the User to upload the Original arbitration agreement "
    "or a duly certified copy of the same."
)
_D4_ACTIONS = [
    {
        "kind": "upload_document",
        "label": "Upload Original arbitration agreement",
        "constant": "arbitration agreement",
        "input": {"type": "file", "accept": ["application/pdf"], "multiple": False},
        "auto_resolve": True,
    },
    {
        "kind": "upload_document",
        "label": "Upload Certified copy",
        "constant": "Certified copy",
        "input": {"type": "file", "accept": ["application/pdf"], "multiple": False},
        "auto_resolve": True,
    },
    {
        "kind": "replace_document",
        "label": "Replace Original arbitration agreement",
        "constant": "arbitration agreement",
        "input": {"type": "file", "accept": ["application/pdf"], "multiple": False},
        "auto_resolve": True,
    },
    {
        "kind": "replace_document",
        "label": "Replace Certified copy",
        "constant": "Certified copy",
        "input": {"type": "file", "accept": ["application/pdf"], "multiple": False},
        "auto_resolve": True,
    },
    {
        "kind": "manual_complete",
        "label": "Mark as complete",
        "constant": "mark as complete",
        "input": {"type": "confirmation"},
        "auto_resolve": True,
    },
]


def test_d4_finding_copies_stored_cure_buttons(monkeypatch) -> None:
    catalogue = catalogue_from_rows(
        [
            {
                "check_id": "D-4",
                "defect": "Arbitration agreement is not on record",
                "requirement": "The arbitration agreement must be on record",
                "where_to_look": ["Petition"],
                "how_to_cure": [_D4_STEP],
                "remediation_config": {"mode": "any", "actions": _D4_ACTIONS},
                "main_category": "General/Global",
                "location_source": "",
                "is_enabled": True,
            }
        ],
        [],
        [],
        1,
    )
    defect = catalogue.defect("D-4")
    assert defect.remediation_config["actions"] == _D4_ACTIONS
    monkeypatch.setattr(
        "extraction_review.scrutiny.schema.get_catalogue",
        lambda: catalogue,
    )

    finding = failed_finding(defect, "not run")

    assert finding.check_id == "D-4"
    assert finding.status == "not_determined"
    assert finding.defect == "Arbitration agreement is not on record"
    assert finding.evidence == []
    assert finding.location == "Filing page missing — this check did not run."
    assert finding.how_to_cure.model_dump() == {
        "steps": [_D4_STEP],
        "mode": "any",
        "actions": _D4_ACTIONS,
    }
