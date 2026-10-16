"""Synthetic tests for the staged boundary and full-segment split pipeline."""

from pathlib import Path

import pytest

from extraction_review.config import Config
from extraction_review.process_file import (
    _boundary_detection_configuration,
    _boundary_proposals_from_exchange,
    _classification_call_evidence,
    _classification_configuration,
    _split_page_review_reasons,
    _segment_classification_report,
)
from extraction_review.structure_split import PageUnit, classify_complete_segments


def test_boundary_request_is_separate_from_canonical_classification_rules() -> None:
    units = [
        PageUnit(
            pdf_page=1,
            text="ANNEXURE P-9",
            headings=("ANNEXURE P-9",),
            annexure_labels=("Annexure P-9",),
            printed_page="112",
        ),
        PageUnit(
            pdf_page=2,
            text="continues",
            paragraph_continues_previous=True,
            sentence_continues_previous=True,
        ),
    ]

    config = _boundary_detection_configuration(units, page_count=2)

    assert config["product_type"] == "split_v1"
    assert [category["name"] for category in config["categories"]] == [
        "Document Segment"
    ]
    instructions = config["splitting_strategy"]["custom_instructions"]
    assert "BOUNDARY DETECTION ONLY" in instructions
    assert "PDF page 1" in instructions
    assert "Annexure P-9" in instructions
    assert "continues_sentence_from_previous" in instructions
    assert len(instructions) <= 5000


def test_boundary_request_samples_late_annexures_in_long_bundle() -> None:
    units = [
        PageUnit(
            pdf_page=page,
            text=f"ordinary continuation text on page {page}",
            headings=(f"ANNEXURE A-{page // 3}",) if page in {123, 153, 209} else (),
            annexure_labels=(f"Annexure A-{page // 3}",) if page in {123, 153, 209} else (),
        )
        for page in range(1, 229)
    ]

    instructions = _boundary_detection_configuration(
        units, page_count=228
    )["splitting_strategy"]["custom_instructions"]

    assert len(instructions) <= 5000
    assert "PDF page 123:" in instructions
    assert "PDF page 153:" in instructions
    assert "PDF page 209:" in instructions
    assert "PDF page 228:" in instructions
    assert "Page cues are sampled" in instructions


def test_boundary_response_proposes_exact_physical_spans_and_gaps_for_review() -> None:
    exchange = {
        "returned": {
            "result": {
                "segments": [
                    {"category": "Document Segment", "pages": [1, 2]},
                    {"category": "Document Segment", "pages": [4, 5]},
                ]
            }
        }
    }

    validation = _boundary_proposals_from_exchange(exchange, page_count=5)
    proposals = validation["proposals"]

    assert [(item["start_page"], item["end_page"]) for item in proposals] == [
        (1, 2),
        (4, 5),
    ]
    assert validation["missing_pages"] == [3]
    assert "boundary_pages_missing" in {issue["code"] for issue in validation["issues"]}
    assert all(item["final"] is False for item in proposals)


@pytest.mark.parametrize(
    ("segments", "page_count", "expected_issue"),
    [
        ([{"pages": [1, 1, 2]}], 2, "duplicate_page_within_segment"),
        ([{"pages": [1, 3]}], 3, "segment_pages_not_contiguous"),
        ([{"pages": [0, 1]}], 2, "page_out_of_range"),
        ([{"pages": [1, 2], "start_page": 2, "end_page": 1}], 2, "segment_range_invalid"),
        ([{"pages": [1, "2"]}], 2, "invalid_page_number"),
        ([{"pages": [2, 1]}], 2, "pages_out_of_order"),
    ],
)
def test_boundary_proposals_reject_malformed_original_page_lists(
    segments, page_count, expected_issue
):
    validation = _boundary_proposals_from_exchange(
        {"returned": {"result": {"segments": segments}}},
        page_count=page_count,
    )

    assert validation["proposals"] == []
    assert expected_issue in {issue["code"] for issue in validation["issues"]}
    assert validation["status"] == "needs_review"


def test_boundary_proposals_report_missing_pages_and_contradictory_overlap():
    validation = _boundary_proposals_from_exchange(
        {
            "returned": {
                "result": {"segments": [{"pages": [1, 2, 3]}, {"pages": [2, 3, 4]}]}
            }
        },
        page_count=5,
    )

    issue_codes = {issue["code"] for issue in validation["issues"]}
    assert validation["missing_pages"] == [5]
    assert "contradictory_boundary_overlap" in issue_codes
    assert validation["shared_boundary_pages"] == []
    assert _classification_configuration(
        Config.model_validate_json(
            (Path(__file__).resolve().parents[1] / "configs/config.json").read_text()
        ).split,
        validation["proposals"],
        proposal_issues=validation["issues"],
    ) == (None, "boundary_proposals_invalid")


def test_adjacent_shared_boundary_is_explicit_and_reviewable():
    validation = _boundary_proposals_from_exchange(
        {
            "returned": {
                "result": {"segments": [{"pages": [1, 2]}, {"pages": [2, 3]}]}
            }
        },
        page_count=3,
    )

    assert validation["shared_boundary_pages"] == [2]
    assert validation["status"] == "needs_review"
    config = Config.model_validate_json(
        (Path(__file__).resolve().parents[1] / "configs/config.json").read_text()
    ).split
    adapted, status = _classification_configuration(
        config, validation["proposals"], proposal_issues=validation["issues"]
    )
    assert adapted is not None
    assert status == "sent_with_shared_boundary_pages"
    assert "boundary-1=[1, 2]; boundary-2=[2, 3]" in adapted["splitting_strategy"]["custom_instructions"]


def test_boundary_and_classification_uncertainty_maps_to_physical_pages():
    reasons = _split_page_review_reasons(
        {
            "missing_pages": [4],
            "issues": [
                {"code": "shared_boundary_page", "source_page": 2},
                {"code": "segment_pages_not_contiguous", "pages": [6, 8]},
            ],
        },
        {
            "missing_pages": [5],
            "overlapping_pages": [3],
            "out_of_range_pages": [12],
        },
    )

    assert set(reasons) == {2, 3, 4, 5, 6, 8, 12}
    assert "boundary:shared_boundary_page" in reasons[2]
    assert "boundary_page_missing" in reasons[4]
    assert "classification_page_overlap" in reasons[3]


def test_classification_call_receives_boundaries_and_keeps_canonical_categories() -> None:
    config = Config.model_validate_json(
        (Path(__file__).resolve().parents[1] / "configs/config.json").read_text()
    ).split
    adapted, status = _classification_configuration(
        config,
        [
            {
                "segment_id": "boundary-1",
                "start_page": 1,
                "end_page": 7,
                "pages": list(range(1, 8)),
            },
            {
                "segment_id": "boundary-2",
                "start_page": 8,
                "end_page": 10,
                "pages": list(range(8, 11)),
            },
        ],
    )

    assert status == "sent"
    assert adapted is not None
    assert adapted["categories"] == [item.model_dump(exclude_none=True) for item in config.categories]
    instructions = adapted["splitting_strategy"]["custom_instructions"]
    assert "classify each proposed complete segment" in instructions
    assert "boundary-1=[1, 2, 3, 4, 5, 6, 7]; boundary-2=[8, 9, 10]" in instructions
    assert len(instructions) <= 5000


def test_large_classification_call_compacts_boundary_ranges_instead_of_dropping_them() -> None:
    config = Config.model_validate_json(
        (Path(__file__).resolve().parents[1] / "configs/config.json").read_text()
    ).split
    proposals = [
        {
            "segment_id": f"boundary-{number}",
            "start_page": number,
            "end_page": number,
            "pages": [number],
        }
        for number in range(1, 96)
    ]

    adapted, status = _classification_configuration(config, proposals)

    assert status == "sent"
    assert adapted is not None
    instructions = adapted["splitting_strategy"]["custom_instructions"]
    assert instructions.startswith("CLASSIFICATION PASS.")
    assert "boundary-1:1" in instructions
    assert "boundary-95:95" in instructions
    assert len(instructions) <= 5000


def test_remote_classification_configuration_fallback_is_explicit():
    config = Config.model_validate_json(
        (Path(__file__).resolve().parents[1] / "configs/config.json").read_text()
    ).split.model_copy(update={"configuration_id": "remote-split-config"})

    adapted, status = _classification_configuration(
        config, [{"segment_id": "boundary-1", "pages": [1, 2]}]
    )
    evidence = _classification_call_evidence(
        {"sent": {"configuration_id": config.configuration_id}},
        boundary_guidance_status=status,
    )

    assert adapted is None
    assert status == "remote_configuration_cannot_be_augmented"
    assert evidence == {
        "kind": "remote_configuration_id",
        "configuration_id": "remote-split-config",
        "boundary_guidance_sent": False,
        "boundary_guidance_status": "remote_configuration_cannot_be_augmented",
    }


def test_segment_classifier_keeps_complete_contiguous_physical_ranges() -> None:
    page_parts = {
        1: ["Main Petition"],
        2: ["Main Petition"],
        3: ["Annexure P-9"],
        4: ["Annexure P-9"],
    }
    units = [
        PageUnit(pdf_page=1, text="FORM 28\nPetition opening"),
        PageUnit(pdf_page=2, text="Prayer and verification"),
        PageUnit(pdf_page=3, text="ANNEXURE P-9\nOrder starts"),
        PageUnit(pdf_page=4, text="Order continuation"),
    ]

    classified = classify_complete_segments(page_parts, units)

    assert [(item.start_page, item.end_page) for item in classified] == [(1, 2), (3, 4)]
    assert [item.document_type for item in classified] == [
        "Main Petition",
        "Annexure P-9",
    ]


def test_llama_complete_segment_classification_validates_exact_page_ownership() -> None:
    report = _segment_classification_report(
        {
            "returned": {
                "result": {
                    "segments": [
                        {"category": "Main Petition", "pages": [1, 2]},
                        {"category": "Annexure P-9", "pages": [3, 4]},
                    ]
                }
            }
        },
        page_count=4,
    )

    assert report["status"] == "clear"
    assert [(s["document_type"], s["source_pages"]) for s in report["segments"]] == [
        ("Main Petition", [1, 2]),
        ("Annexure P-9", [3, 4]),
    ]


def test_llama_complete_segment_classification_flags_omission_and_overlap() -> None:
    report = _segment_classification_report(
        {
            "returned": {
                "result": {
                    "segments": [
                        {"category": "Main Petition", "pages": [1, 2]},
                        {"category": "Annexure P-9", "pages": [2, 4]},
                    ]
                }
            }
        },
        page_count=4,
    )

    assert report["status"] == "needs_review"
    assert report["missing_pages"] == [3]
    assert report["overlapping_pages"] == [2]
    assert "segment_pages_not_contiguous" in report["issues"]


def test_classification_report_rejects_duplicate_and_out_of_range_pages():
    report = _segment_classification_report(
        {
            "returned": {
                "result": {
                    "segments": [
                        {"category": "Main Petition", "pages": [1, 1]},
                        {"category": "Annexure P-9", "pages": [3, 8]},
                    ]
                }
            }
        },
        page_count=3,
    )

    assert "duplicate_page_within_segment" in report["issues"]
    assert "page_out_of_range" in report["issues"]
    assert report["missing_pages"] == [2]
