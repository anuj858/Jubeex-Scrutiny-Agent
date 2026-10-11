"""Deterministic transactional tests for bounded page-map reconciliation."""

from extraction_review.structure_split import (
    PageClassification,
    PageUnit,
    reconcile_compiled_split,
    validate_reconciliation_page_map,
)


def _strong(page: int, document_type: str, signal: str) -> PageClassification:
    return PageClassification(
        page=page,
        document_type=document_type,
        confidence=0.97,
        signals=(signal,),
    )


def test_merges_adjacent_same_document_segments_and_audits_operation() -> None:
    operations = []
    result, changes = reconcile_compiled_split(
        {1: ["Main Petition"], 2: ["Application 1"]},
        {1: ["Main Petition"], 2: ["Main Petition"]},
        [_strong(2, "Main Petition", "form28_lookalike")],
        {1: "FORM 28", 2: "continuation of petition"},
        page_count=2,
        operation_log=operations,
        page_units=[
            PageUnit(pdf_page=1, text="FORM 28"),
            PageUnit(pdf_page=2, text="continuation of petition"),
        ],
    )

    assert result == {1: ["Main Petition"], 2: ["Main Petition"]}
    assert [change.page for change in changes] == [2]
    operation = next(item for item in operations if item.status == "accepted")
    assert operation.operation_type == "merge_segments"
    assert operation.validation["structural"]["single_physical_owner"] is True


def test_splits_merged_segment_at_explicit_outer_annexure_boundary() -> None:
    operations = []
    result, _ = reconcile_compiled_split(
        {1: ["Main Petition"], 2: ["Main Petition"], 3: ["Main Petition"]},
        {1: ["Main Petition"], 2: ["Main Petition"], 3: ["Annexure P-9"]},
        [_strong(3, "Annexure P-9", "annexure_stamp")],
        {1: "FORM 28", 2: "petition continuation", 3: "ANNEXURE P-9"},
        page_count=3,
        operation_log=operations,
        page_units=[
            PageUnit(pdf_page=1, text="FORM 28"),
            PageUnit(pdf_page=2, text="petition continuation"),
            PageUnit(pdf_page=3, text="ANNEXURE P-9"),
        ],
    )

    assert result[3] == ["Annexure P-9"]
    assert any(item.operation_type == "split_segment" for item in operations)


def test_relabels_unidentified_complete_segment_from_outer_boundary() -> None:
    operations = []
    result, _ = reconcile_compiled_split(
        {1: ["Unidentified"], 2: ["Unidentified"]},
        {1: ["Annexure P-9"], 2: ["Annexure P-9"]},
        [_strong(1, "Annexure P-9", "annexure_stamp")],
        {1: "ANNEXURE P-9", 2: "order continuation"},
        page_count=2,
        operation_log=operations,
        page_units=[
            PageUnit(pdf_page=1, text="ANNEXURE P-9"),
            PageUnit(pdf_page=2, text="order continuation"),
        ],
    )

    assert result == {1: ["Annexure P-9"], 2: ["Annexure P-9"]}
    assert any(item.operation_type == "relabel_segment" for item in operations)


def test_moves_captionless_signature_continuation_to_anchored_segment() -> None:
    operations = []
    result, _ = reconcile_compiled_split(
        {1: ["Main Petition"], 2: ["Affidavit"]},
        {1: ["Main Petition"], 2: ["Main Petition"]},
        [_strong(1, "Main Petition", "form28_lookalike")],
        {1: "FORM 28", 2: "Signed by petitioner"},
        page_count=2,
        operation_log=operations,
        page_units=[
            PageUnit(pdf_page=1, text="FORM 28"),
            PageUnit(
                pdf_page=2,
                text="Signed by petitioner",
                signature_detected=True,
                paragraph_continues_previous=True,
            ),
        ],
    )

    assert result[2] == ["Main Petition"]
    assert any(item.operation_type == "boundary_adjustment" for item in operations)


def test_historical_order_heading_inside_annexure_is_not_split() -> None:
    operations = []
    baseline = {1: ["Annexure P-9"], 2: ["Annexure P-9"]}
    result, _ = reconcile_compiled_split(
        baseline,
        baseline,
        [_strong(2, "Impugned Order", "centered_title")],
        {1: "ANNEXURE P-9", 2: "JUDGMENT OF THE HIGH COURT"},
        page_count=2,
        operation_log=operations,
        page_units=[
            PageUnit(pdf_page=1, text="ANNEXURE P-9"),
            PageUnit(pdf_page=2, text="JUDGMENT OF THE HIGH COURT"),
        ],
    )

    assert result == baseline
    assert not [item for item in operations if item.status == "accepted"]


def test_page_map_validator_routes_missing_pages_and_rejects_overlap_or_range_errors() -> None:
    missing = validate_reconciliation_page_map({1: ["Index"]}, page_count=2)
    assert missing["valid"] is True
    assert missing["unlabelled_pages_routed_to_unidentified"] == [2]

    overlap = validate_reconciliation_page_map(
        {1: ["Main Petition", "Affidavit"]}, page_count=1
    )
    # Combined category labels are a supported single-page alias; output-slot
    # ownership is enforced separately after actual slicing.
    assert overlap["valid"] is True
    assert overlap["combined_labels_require_slot_validation"] is True
    assert overlap["multiple_label_pages"] == [1]

    outside = validate_reconciliation_page_map({0: ["Index"]}, page_count=2)
    assert outside["valid"] is False
    assert outside["invalid_pages"] == [0]


def test_out_of_range_candidate_is_rejected_before_mutation() -> None:
    operations = []
    baseline = {1: ["Main Petition"]}
    result, changes = reconcile_compiled_split(
        baseline,
        {1: ["Main Petition"], 4: ["Annexure P-9"]},
        [],
        {},
        page_count=2,
        operation_log=operations,
    )

    assert result == baseline
    assert changes == []
    assert operations[0].status == "rejected"
    assert operations[0].validation["proposed"]["invalid_pages"] == [4]


def test_failed_semantic_revalidation_rolls_back_and_iteration_is_bounded(monkeypatch) -> None:
    import extraction_review.structure_split as module

    monkeypatch.setattr(
        module,
        "_index_supported_map",
        lambda *_args: {1: ["Annexure P-9"]},
    )
    operations = []
    baseline = {1: ["Main Petition"]}
    result, changes = reconcile_compiled_split(
        baseline,
        baseline,
        [_strong(1, "Impugned Order", "outer_anchor")],
        {1: "IN THE SUPREME COURT OF INDIA"},
        page_count=1,
        operation_log=operations,
        max_iterations=99,
        page_units=[PageUnit(pdf_page=1, text="Impugned order")],
    )

    assert result == baseline
    assert changes == []
    assert any(item.status == "rolled_back" for item in operations)
    stop = operations[-1]
    assert stop.status == "stopped"
    assert "max_iterations=3" in stop.reason


def test_repeated_state_guard_detects_repeated_page_map() -> None:
    from extraction_review.structure_split import _record_reconciliation_state

    seen = {"page-map-A"}
    assert _record_reconciliation_state("page-map-B", seen) is True
    assert _record_reconciliation_state("page-map-A", seen) is False


def test_fresh_second_pass_corrects_issue_exposed_by_first_mutation() -> None:
    baseline = {
        1: ["Main Petition"],
        2: ["Application 1"],
        3: ["Unidentified"],
    }
    first_proposal = {
        1: ["Main Petition"],
        2: ["Main Petition"],
        3: ["Unidentified"],
    }
    second_proposals = []

    def reassess(current, pass_index):
        second_proposals.append((pass_index, {page: list(labels) for page, labels in current.items()}))
        return {
            **current,
            3: ["Annexure P-9"],
        }

    operations = []
    result, changes = reconcile_compiled_split(
        baseline,
        first_proposal,
        [
            _strong(2, "Main Petition", "form28_lookalike"),
            _strong(3, "Annexure P-9", "annexure_stamp"),
        ],
        {1: "FORM 28", 2: "petition continuation", 3: "ANNEXURE P-9"},
        page_count=3,
        operation_log=operations,
        proposal_generator=reassess,
        page_units=[
            PageUnit(pdf_page=1, text="FORM 28"),
            PageUnit(pdf_page=2, text="petition continuation"),
            PageUnit(pdf_page=3, text="ANNEXURE P-9"),
        ],
    )

    assert result == {
        1: ["Main Petition"],
        2: ["Main Petition"],
        3: ["Annexure P-9"],
    }
    assert [change.page for change in changes] == [2, 3]
    assert second_proposals[0][0] == 2
    assert second_proposals[0][1][2] == ["Main Petition"]
    assert [item.pass_index for item in operations if item.status == "accepted"] == [1, 2]
    assert [item.pass_index for item in operations if item.operation_type == "reconciliation_pass"] == [1, 2]


def test_no_fresh_proposal_pass_when_no_actionable_change_exists() -> None:
    baseline = {1: ["Main Petition"]}
    generator_calls = []
    result, changes = reconcile_compiled_split(
        baseline,
        baseline,
        [],
        {1: "FORM 28"},
        page_count=1,
        proposal_generator=lambda current, index: generator_calls.append(index) or current,
        page_units=[PageUnit(pdf_page=1, text="FORM 28")],
    )

    assert result == baseline
    assert changes == []
    assert generator_calls == []


def test_fresh_reassessment_stops_when_updated_map_has_no_actionable_issues() -> None:
    baseline = {1: ["Main Petition"], 2: ["Unidentified"]}
    proposals = []

    def reassess(current, pass_index):
        proposals.append(pass_index)
        # Recompute returns the already accepted state: no further mutation.
        return current

    result, changes = reconcile_compiled_split(
        baseline,
        {1: ["Main Petition"], 2: ["Annexure P-9"]},
        [_strong(2, "Annexure P-9", "annexure_stamp")],
        {1: "FORM 28", 2: "ANNEXURE P-9"},
        page_count=2,
        proposal_generator=reassess,
        page_units=[
            PageUnit(pdf_page=1, text="FORM 28"),
            PageUnit(pdf_page=2, text="ANNEXURE P-9"),
        ],
    )

    assert result[2] == ["Annexure P-9"]
    assert [change.page for change in changes] == [2]
    assert proposals == [2]


def test_fresh_passes_obey_iteration_and_zero_model_call_budgets() -> None:
    baseline = {1: ["Main Petition"], 2: ["Unidentified"], 3: ["Unidentified"]}
    proposals = []

    def reassess(current, pass_index):
        proposals.append(pass_index)
        return {**current, 3: ["Annexure P-9"]}

    operations = []
    result, changes = reconcile_compiled_split(
        baseline,
        {1: ["Main Petition"], 2: ["Annexure P-8"], 3: ["Unidentified"]},
        [
            _strong(2, "Annexure P-8", "annexure_stamp"),
            _strong(3, "Annexure P-9", "annexure_stamp"),
        ],
        {1: "FORM 28", 2: "ANNEXURE P-8", 3: "ANNEXURE P-9"},
        page_count=3,
        operation_log=operations,
        max_iterations=2,
        max_model_calls=0,
        proposal_generator=reassess,
        page_units=[
            PageUnit(pdf_page=1, text="FORM 28"),
            PageUnit(pdf_page=2, text="ANNEXURE P-8"),
            PageUnit(pdf_page=3, text="ANNEXURE P-9"),
        ],
    )

    assert result[2] == ["Annexure P-8"]
    assert result[3] == ["Annexure P-9"]
    assert len(changes) == 2
    assert proposals == [2]
    assert sum(item.operation_type == "reconciliation_pass" for item in operations) == 2
    stop = operations[-1]
    assert stop.status == "stopped"
    assert "max_iterations=2" in stop.reason
    assert "model_calls=0/0" in operations[-2].reason


def test_repeated_followup_state_is_rejected_and_current_map_is_preserved() -> None:
    baseline = {1: ["Main Petition"], 2: ["Unidentified"]}
    result, changes = reconcile_compiled_split(
        baseline,
        {1: ["Main Petition"], 2: ["Annexure P-9"]},
        [_strong(2, "Annexure P-9", "annexure_stamp")],
        {1: "FORM 28", 2: "ANNEXURE P-9"},
        page_count=2,
        operation_log=(operations := []),
        proposal_generator=lambda _current, _index: baseline,
        page_units=[
            PageUnit(pdf_page=1, text="FORM 28"),
            PageUnit(pdf_page=2, text="ANNEXURE P-9"),
        ],
    )

    assert result == {1: ["Main Petition"], 2: ["Annexure P-9"]}
    assert len(changes) == 1
    assert any(item.status == "rejected" and item.pass_index == 2 for item in operations)
    assert "repeated_state_oscillation_guard" in operations[-1].reason


def test_invalid_fresh_proposal_is_rejected_without_corrupting_page_ownership() -> None:
    baseline = {1: ["Main Petition"], 2: ["Unidentified"]}
    operations = []
    result, changes = reconcile_compiled_split(
        baseline,
        {1: ["Main Petition"], 2: ["Annexure P-9"]},
        [_strong(2, "Annexure P-9", "annexure_stamp")],
        {1: "FORM 28", 2: "ANNEXURE P-9"},
        page_count=2,
        operation_log=operations,
        proposal_generator=lambda current, _index: {**current, 3: ["Affidavit"]},
        page_units=[
            PageUnit(pdf_page=1, text="FORM 28"),
            PageUnit(pdf_page=2, text="ANNEXURE P-9"),
        ],
    )

    assert result == {1: ["Main Petition"], 2: ["Annexure P-9"]}
    assert [change.page for change in changes] == [2]
    rejected = next(
        item
        for item in operations
        if item.status == "rejected" and item.pass_index == 2
    )
    assert rejected.validation["proposed"]["invalid_pages"] == [3]
    assert operations[-1].status == "stopped"
    assert validate_reconciliation_page_map(result, page_count=2)["valid"] is True
