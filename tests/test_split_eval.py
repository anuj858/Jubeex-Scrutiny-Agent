"""Offline tests for scoring captured production split artifacts."""

import asyncio
import json
from copy import deepcopy

import httpx
import pytest

from extraction_review.split_eval import (
    ANNOTATION_SCHEMA,
    AnnotationError,
    evaluate_case,
    evaluate_dataset,
    main,
    run_existing_api_case,
    validate_annotations,
)

_HASH = "a" * 64


def _case():
    return {
        "case_id": "synthetic_case_01",
        "source": {"sha256": _HASH},
        "physical_page_count": 5,
        "input_quality": "mixed",
        "documents": [
            {"document_id": "doc_index", "category": "Index", "start_page": 1, "end_page": 1},
            {"document_id": "doc_petition", "category": "Main Petition", "start_page": 2, "end_page": 3},
            {"document_id": "doc_affidavit", "category": "Affidavit", "start_page": 4, "end_page": 4},
            {
                "document_id": "doc_annexure",
                "category": "Annexure P-1",
                "start_page": 5,
                "end_page": 5,
                "reviewer_notes": "private note that must not leak",
                "ambiguity_flags": ["faint_stamp"],
            },
        ],
    }


def _artifact(segments=None, *, page_count=5, missing=(), duplicate_slots=()):
    if segments is None:
        segments = [
            ("Index", 1, 1),
            ("Main Petition", 2, 3),
            ("Affidavit", 4, 4),
            ("Annexure P-1", 5, 5),
        ]
    rows = []
    for page in range(1, page_count + 1):
        if page in missing:
            continue
        selected = next((item for item in segments if item[1] <= page <= item[2]), None)
        category = selected[0] if selected else "Unidentified"
        segment_id = f"segment_{selected[1]}" if selected else f"unidentified_{page}"
        slots = [category]
        if page in duplicate_slots:
            slots.append("other_slot")
        rows.append(
            {
                "source_page": page,
                "segment_id": segment_id,
                "assigned_labels": [category],
                "effective_owner": category,
                "output_slots": slots,
                "classification": {"confidence": 0.8},
                "segment_classification": {
                    "document_type": category,
                    "confidence": 0.84,
                    "signals": ["synthetic_fixture"],
                },
                "review_status": "needs_review" if page == 5 else "clear",
                "review_reasons": ["uncertain_boundary"] if page == 5 else [],
            }
        )
    return {
        "config": {
            "config_id": "jubeex_parse",
            "config_version": "test-version",
            "split": {"config_version": "test-split"},
        },
        "split_audit": {
            "source_pdf_sha256": _HASH,
            "review_status": "needs_review",
            "structural_validation": {"review_status": "clear", "overlapping_ownership": []},
            "page_manifest": {
                "schema_version": "1.0",
                "source_page_count": page_count,
                "source_sha256": _HASH,
                "pages": rows,
            },
        },
    }


def _annotations(case=None):
    return {"schema_version": ANNOTATION_SCHEMA, "cases": [case or _case()]}


def test_perfect_predictions_score_exact_boundaries_ranges_types_and_ownership():
    result = evaluate_case(_case(), _artifact())
    metrics = result["metrics"]
    assert metrics["boundary_precision"] == 1.0
    assert metrics["boundary_recall"] == 1.0
    assert metrics["boundary_f1"] == 1.0
    assert metrics["exact_document_range_accuracy"] == 1.0
    assert metrics["page_document_ownership_accuracy"] == 1.0
    assert metrics["canonical_document_type_accuracy"] == 1.0
    assert metrics["missing_page_count"] == 0
    assert metrics["multiply_assigned_page_count"] == 0
    assert metrics["manual_review_rate"] == 0.25
    assert result["ambiguous_document_count"] == 1
    assert "private note" not in str(result)


def test_boundary_error_is_measured_without_relying_on_segment_order():
    guess = _artifact(
        [
            ("Index", 1, 1),
            ("Main Petition", 2, 4),
            ("Annexure P-1", 5, 5),
        ]
    )
    result = evaluate_case(_case(), guess)
    assert result["metrics"]["boundary_precision"] == 1.0
    assert result["metrics"]["boundary_recall"] == pytest.approx(2 / 3)
    assert result["metrics"]["exact_document_range_accuracy"] == 0.5
    assert result["metrics"]["page_document_ownership_accuracy"] == 0.8


def test_reordered_segments_match_by_page_overlap_not_list_position():
    reordered = _artifact(
        [
            ("Annexure P-1", 5, 5),
            ("Affidavit", 4, 4),
            ("Main Petition", 2, 3),
            ("Index", 1, 1),
        ]
    )
    metrics = evaluate_case(_case(), reordered)["metrics"]
    assert metrics["exact_document_range_accuracy"] == 1.0
    assert metrics["canonical_document_type_accuracy"] == 1.0
    assert metrics["page_document_ownership_accuracy"] == 1.0


def test_final_repeatable_output_slots_keep_adjacent_generic_documents_separate():
    case = {
        "case_id": "repeatable_segments",
        "source": {"sha256": _HASH},
        "physical_page_count": 2,
        "documents": [
            {"document_id": "annexure_a", "category": "Annexures", "start_page": 1, "end_page": 1},
            {"document_id": "annexure_b", "category": "Annexures", "start_page": 2, "end_page": 2},
        ],
    }
    artifact = _artifact(
        [("Annexures", 1, 1), ("Annexures", 2, 2)], page_count=2
    )
    rows = artifact["split_audit"]["page_manifest"]["pages"]
    rows[0].update(segment_id="generic_annexures", output_slots=["annexure_1"])
    rows[1].update(segment_id="generic_annexures", output_slots=["annexure_2"])

    result = evaluate_case(case, artifact)
    assert result["metrics"]["exact_document_range_accuracy"] == 1.0
    assert len(result["predicted_segments"]) == 2


def test_split_and_merged_predictions_remain_visible_as_unmatched_segments():
    split = evaluate_case(
        _case(),
        _artifact(
            [
                ("Index", 1, 1),
                ("Main Petition", 2, 2),
                ("Main Petition", 3, 3),
                ("Affidavit", 4, 4),
                ("Annexure P-1", 5, 5),
            ]
        ),
    )["metrics"]
    assert split["unmatched_predicted_segment_count"] == 1
    assert split["unmatched_expected_document_count"] == 0
    assert split["exact_document_range_accuracy"] == 0.75

    merged = evaluate_case(
        _case(),
        _artifact(
            [
                ("Index", 1, 1),
                ("Main Petition", 2, 4),
                ("Annexure P-1", 5, 5),
            ]
        ),
    )["metrics"]
    assert merged["unmatched_expected_document_count"] == 1
    assert merged["unmatched_predicted_segment_count"] == 0
    assert merged["exact_document_range_accuracy"] == 0.5


def test_wrong_category_is_scored_separately_from_correct_ranges():
    guess = _artifact(
        [
            ("Index", 1, 1),
            ("Main Petition", 2, 3),
            ("Vakalatnama", 4, 4),
            ("Annexure P-1", 5, 5),
        ]
    )
    result = evaluate_case(_case(), guess)
    assert result["metrics"]["exact_document_range_accuracy"] == 1.0
    assert result["metrics"]["canonical_document_type_accuracy"] == 0.75
    assert result["metrics"]["page_document_ownership_accuracy"] == 1.0


def test_missing_page_and_multiple_output_owners_are_counted():
    result = evaluate_case(
        _case(), _artifact(missing={5}, duplicate_slots={4})
    )
    assert result["metrics"]["missing_page_count"] == 1
    assert result["metrics"]["multiply_assigned_page_count"] == 1
    assert result["metrics"]["page_document_ownership_accuracy"] == 0.6


def test_duplicate_output_owner_penalizes_page_ownership_accuracy():
    result = evaluate_case(_case(), _artifact(duplicate_slots={4}))
    assert result["metrics"]["multiply_assigned_page_count"] == 1
    assert result["metrics"]["page_document_ownership_accuracy"] == 0.8


def test_duplicate_manifest_page_row_is_counted_as_multiple_ownership():
    artifact = _artifact()
    duplicate = deepcopy(artifact["split_audit"]["page_manifest"]["pages"][3])
    artifact["split_audit"]["page_manifest"]["pages"].append(duplicate)
    result = evaluate_case(_case(), artifact)
    assert result["metrics"]["multiply_assigned_page_count"] == 1
    assert result["metrics"]["page_document_ownership_accuracy"] == 0.8


def test_out_of_range_assignment_is_counted_and_penalized():
    artifact = _artifact()
    extra = deepcopy(artifact["split_audit"]["page_manifest"]["pages"][-1])
    extra.update(source_page=6, segment_id="out_of_range", output_slots=["annexure_p1"])
    artifact["split_audit"]["page_manifest"]["pages"].append(extra)
    result = evaluate_case(_case(), artifact)
    assert result["metrics"]["out_of_range_page_count"] == 1
    assert result["metrics"]["page_count_denominator"] == 6
    assert result["metrics"]["page_document_ownership_accuracy"] == pytest.approx(5 / 6)
    # A nonexistent cut after the last physical page is not a boundary.
    assert result["metrics"]["boundary_predicted_count"] == 3


def test_boundary_metric_excludes_before_first_and_after_last_page():
    result = evaluate_case(_case(), _artifact())
    assert result["metrics"]["boundary_expected_count"] == 3
    assert result["metrics"]["boundary_predicted_count"] == 3


def test_boundary_f1_is_zero_when_expected_cuts_are_all_missed():
    result = evaluate_case(
        _case(), _artifact([("Main Petition", 1, 5)])
    )["metrics"]
    assert result["boundary_precision"] is None
    assert result["boundary_recall"] == 0.0
    assert result["boundary_f1"] == 0.0


def test_boundary_f1_is_zero_when_predictions_and_truth_have_no_common_cut():
    result = evaluate_case(
        _case(),
        _artifact(
            [
                ("Main Petition", 1, 2),
                ("Affidavit", 3, 5),
            ]
        ),
    )["metrics"]
    assert result["boundary_precision"] == 0.0
    assert result["boundary_recall"] == 0.0
    assert result["boundary_f1"] == 0.0


def test_invalid_manifest_page_number_fails_closed():
    artifact = _artifact()
    artifact["split_audit"]["page_manifest"]["pages"][0]["source_page"] = "unknown"
    with pytest.raises(TypeError, match="invalid source_page"):
        evaluate_case(_case(), artifact)


@pytest.mark.parametrize(
    ("mutate", "message"),
    [
        (lambda case: case["documents"][1].update(start_page=0), "invalid inclusive page range"),
        (lambda case: case["documents"][1].update(category="Made Up"), "non-canonical"),
        (lambda case: case["documents"].pop(), "leaves source pages unannotated"),
        (lambda case: case["documents"][1].update(start_page=1), "overlaps page"),
        (lambda case: case["documents"][1].update(document_id="doc_index"), "duplicate document_id"),
        (lambda case: case["documents"][1].update(end_page=6), "invalid inclusive page range"),
    ],
)
def test_invalid_ground_truth_is_rejected(mutate, message):
    case = deepcopy(_case())
    mutate(case)
    with pytest.raises(AnnotationError, match=message):
        validate_annotations(_annotations(case))


def test_case_id_must_be_a_safe_non_sensitive_filename_key():
    case = deepcopy(_case())
    case["case_id"] = "../private-petition"
    with pytest.raises(AnnotationError, match="safe non-sensitive"):
        validate_annotations(_annotations(case))


def test_checksum_mismatch_rejected_and_quality_breakdown_reported():
    artifact = _artifact()
    artifact["split_audit"]["page_manifest"]["source_sha256"] = "b" * 64
    with pytest.raises(ValueError, match="checksum"):
        evaluate_case(_case(), artifact)

    report = evaluate_dataset(_annotations(), {"synthetic_case_01": _artifact()})
    assert report["by_input_quality"]["mixed"]["case_count"] == 1
    assert report["pipeline"]["split_config_version"]
    assert report["cases"][0]["pipeline_config_identity"]["config_version"] == "test-version"
    assert report["cases"][0]["prediction_artifact_sha256"]
    assert "text_excerpt" not in str(report)


def test_empty_dataset_produces_null_rates_not_zero_accuracy():
    report = evaluate_dataset(
        {"schema_version": ANNOTATION_SCHEMA, "cases": []}, {}
    )
    assert report["cases"] == []
    assert report["global_metrics"]["boundary_f1"] is None
    assert report["global_metrics"]["page_document_ownership_accuracy"] is None
    assert report["validation"]["case_count"] == 0


def test_existing_api_runner_captures_production_split_artifact_without_live_calls(tmp_path):
    artifact = _artifact()
    requests = []

    def handler(request):
        requests.append(request)
        if request.method == "POST":
            return httpx.Response(202, json={"job_id": "job_test", "poll_url": "/v1/jobs/job_test"})
        if request.url.path == "/v1/jobs/job_test":
            return httpx.Response(
                200,
                json={
                    "status": "completed",
                    "result": {"artifacts": {"split": {"url": "https://artifact.test/split.json"}}},
                },
            )
        return httpx.Response(200, json=artifact)

    client = httpx.AsyncClient(transport=httpx.MockTransport(handler))
    output = tmp_path / "predictions" / "synthetic_case_01.json"
    result = asyncio.run(
        run_existing_api_case(
            _case(),
            {
                "download_url": "https://source.test/reviewed.pdf",
                "filing_type": "SLP_CIVIL",
                "organization_id": "evaluation_org",
                "workspace_id": "evaluation_workspace",
            },
            api_base_url="https://agent.test",
            api_key="synthetic-api-key",
            output_path=output,
            allow_remote=True,
            client=client,
            poll_interval_s=0,
        )
    )
    asyncio.run(client.aclose())

    assert result == artifact
    assert json.loads(output.read_text()) == artifact
    assert requests[0].url.path == "/v1/filings"
    payload = json.loads(requests[0].content)
    assert payload["job_type"] == "upload_compiled"
    assert payload["organization_id"] == "evaluation_org"
    assert requests[1].url.path == "/v1/jobs/job_test"
    assert requests[2].url.host == "artifact.test"


def test_remote_api_runner_requires_authorization_before_network_call(tmp_path):
    async def fail_if_called(_request):
        raise AssertionError("network should not be called without authorization")

    client = httpx.AsyncClient(transport=httpx.MockTransport(fail_if_called))
    with pytest.raises(ValueError, match="requires JUBEEX_API_KEY"):
        asyncio.run(
            run_existing_api_case(
                _case(),
                {
                    "download_url": "https://source.test/reviewed.pdf",
                    "filing_type": "SLP_CIVIL",
                    "organization_id": "evaluation_org",
                    "workspace_id": "evaluation_workspace",
                },
                api_base_url="https://agent.test",
                api_key="",
                output_path=tmp_path / "split.json",
                allow_remote=True,
                client=client,
            )
        )
    asyncio.run(client.aclose())


def test_remote_api_runner_requires_explicit_opt_in(tmp_path):
    client = httpx.AsyncClient(
        transport=httpx.MockTransport(
            lambda _request: (_ for _ in ()).throw(
                AssertionError("network should not be called without opt-in")
            )
        )
    )
    with pytest.raises(ValueError, match="explicit allow_remote opt-in"):
        asyncio.run(
            run_existing_api_case(
                _case(),
                {
                    "download_url": "https://source.test/reviewed.pdf",
                    "filing_type": "SLP_CIVIL",
                    "organization_id": "evaluation_org",
                    "workspace_id": "evaluation_workspace",
                },
                api_base_url="https://agent.test",
                api_key="synthetic-api-key",
                output_path=tmp_path / "split.json",
                allow_remote=False,
                client=client,
            )
        )
    asyncio.run(client.aclose())


def test_remote_cli_requires_explicit_flags_and_authorization(tmp_path, monkeypatch):
    annotations_path = tmp_path / "annotations.json"
    sources_path = tmp_path / "sources.json"
    annotations_path.write_text(json.dumps(_annotations()))
    sources_path.write_text(json.dumps({"synthetic_case_01": {}}))
    monkeypatch.delenv("JUBEEX_API_KEY", raising=False)

    result = main(
        [
            "--annotations",
            str(annotations_path),
            "--predictions-dir",
            str(tmp_path / "predictions"),
            "--output",
            str(tmp_path / "report.json"),
            "--run-existing-pipeline",
            "--allow-remote-run",
            "--api-base-url",
            "https://agent.test",
            "--sources",
            str(sources_path),
        ]
    )
    assert result == 2
    assert not (tmp_path / "predictions" / "synthetic_case_01.json").exists()


def test_cli_scores_captured_artifact_and_writes_privacy_reduced_report(tmp_path):
    annotations_path = tmp_path / "annotations.json"
    predictions_dir = tmp_path / "predictions"
    output_path = tmp_path / "report.json"
    predictions_dir.mkdir()
    annotations_path.write_text(json.dumps(_annotations()))
    (predictions_dir / "synthetic_case_01.json").write_text(json.dumps(_artifact()))

    assert main(
        [
            "--annotations",
            str(annotations_path),
            "--predictions-dir",
            str(predictions_dir),
            "--output",
            str(output_path),
        ]
    ) == 0
    report = json.loads(output_path.read_text())
    assert report["global_metrics"]["exact_document_range_accuracy"] == 1.0
    assert "private note" not in output_path.read_text()
