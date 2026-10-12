"""Regression tests for boundary-first compiled-PDF page ownership."""

from extraction_review.process_file import _document_first_page_ownership


def _proposal(number: int, start: int, end: int) -> dict:
    return {
        "segment_id": f"boundary-{number}",
        "start_page": start,
        "end_page": end,
        "pages": list(range(start, end + 1)),
        "source": "llama_boundary_detection",
    }


def _report(*segments: tuple[str, list[int]]) -> dict:
    return {
        "segments": [
            {
                "category": category,
                "document_type": category,
                "source_pages": pages,
            }
            for category, pages in segments
        ]
    }


def test_internal_cover_page_heading_does_not_split_annexure() -> None:
    page_parts, segments, issues = _document_first_page_ownership(
        [_proposal(1, 1, 3)],
        _report(("Annexure P-1", [1, 2]), ("Cover Page", [3])),
        page_count=3,
    )

    assert page_parts == {page: ["Annexure P-1"] for page in (1, 2, 3)}
    assert segments[0]["pages"] == [1, 2, 3]
    assert segments[0]["review_reasons"] == ["conflicting_categories_inside_boundary"]
    assert issues[0]["code"] == "boundary_classification_conflict"


def test_petition_word_in_annexure_body_does_not_change_owner() -> None:
    page_parts, _, _ = _document_first_page_ownership(
        [_proposal(1, 10, 12)],
        _report(("Annexure P-4", [10, 11]), ("Main Petition", [12])),
        page_count=12,
    )

    assert all(page_parts[page] == ["Annexure P-4"] for page in range(10, 13))


def test_captionless_annexure_continuations_keep_complete_candidate_range() -> None:
    page_parts, segments, _ = _document_first_page_ownership(
        [_proposal(1, 1, 5)],
        _report(("Annexure P-7", [1])),
        page_count=5,
    )

    assert set(page_parts) == set(range(1, 6))
    assert {tuple(labels) for labels in page_parts.values()} == {("Annexure P-7",)}
    assert segments[0]["type_evidence"] == ["consistent_overlapping_classification"]


def test_consecutive_similar_annexures_remain_distinct_by_boundary() -> None:
    page_parts, segments, _ = _document_first_page_ownership(
        [_proposal(1, 1, 3), _proposal(2, 4, 6)],
        _report(("Annexure P-1", [1, 2, 3]), ("Annexure P-2", [4, 5, 6])),
        page_count=6,
    )

    assert [segment["category"] for segment in segments] == [
        "Annexure P-1",
        "Annexure P-2",
    ]
    assert page_parts[3] == ["Annexure P-1"]
    assert page_parts[4] == ["Annexure P-2"]


def test_real_document_after_annexure_uses_next_candidate_boundary() -> None:
    page_parts, _, _ = _document_first_page_ownership(
        [_proposal(1, 1, 3), _proposal(2, 4, 5)],
        _report(("Annexure P-1", [1, 2, 3]), ("Affidavit", [4, 5])),
        page_count=5,
    )

    assert page_parts[3] == ["Annexure P-1"]
    assert page_parts[4] == ["Affidavit"]
    assert page_parts[5] == ["Affidavit"]


def test_boundary_without_title_still_governs_document_ownership() -> None:
    page_parts, _, _ = _document_first_page_ownership(
        [_proposal(1, 1, 2), _proposal(2, 3, 4)],
        _report(("Synopsis", [1, 2]), ("List of Dates & Events", [3, 4])),
        page_count=4,
    )

    assert page_parts == {
        1: ["Synopsis"],
        2: ["Synopsis"],
        3: ["List of Dates & Events"],
        4: ["List of Dates & Events"],
    }


def test_signature_blank_and_verification_pages_follow_candidate_document() -> None:
    page_parts, _, _ = _document_first_page_ownership(
        [_proposal(1, 1, 4)],
        _report(("Affidavit", [1]), ("Main Petition", [2, 3, 4])),
        page_count=4,
    )

    assert all(page_parts[page] == ["Affidavit"] for page in range(1, 5))


def test_incorrect_one_page_llama_segment_cannot_fragment_document() -> None:
    page_parts, segments, _ = _document_first_page_ownership(
        [_proposal(1, 1, 6)],
        _report(
            ("Main Petition", [1, 2, 3]),
            ("Cover Page", [4]),
            ("Main Petition", [5, 6]),
        ),
        page_count=6,
    )

    # The malformed non-contiguous classifier evidence is still only type
    # evidence; the accepted physical boundary remains one complete range.
    assert all(page_parts[page] == ["Main Petition"] for page in range(1, 7))
    assert segments[0]["pages"] == list(range(1, 7))


def test_missing_or_misleading_classification_is_left_for_review() -> None:
    page_parts, segments, issues = _document_first_page_ownership(
        [_proposal(1, 1, 2)],
        _report(),
        page_count=4,
    )

    assert page_parts == {
        1: ["Undefined"],
        2: ["Undefined"],
        3: ["Undefined"],
        4: ["Undefined"],
    }
    assert [segment["pages"] for segment in segments] == [[1, 2], [3, 4]]
    assert {issue["code"] for issue in issues} == {
        "boundary_classification_conflict",
        "unresolved_boundary_pages",
    }


def test_reconciliation_preserves_order_complete_ownership_and_audited_share() -> None:
    page_parts, segments, _ = _document_first_page_ownership(
        [_proposal(1, 1, 3), _proposal(2, 3, 5)],
        _report(("Annexure P-1", [1, 2, 3]), ("Annexure P-2", [3, 4, 5])),
        page_count=5,
    )

    assert [segment["pages"] for segment in segments] == [[1, 2, 3], [3, 4, 5]]
    assert page_parts[3] == ["Annexure P-1", "Annexure P-2"]
    assert set(page_parts) == set(range(1, 6))
