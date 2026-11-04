"""Regression coverage captured from split job 79860ddd (Defect File_005)."""

from __future__ import annotations

import io

from pypdf import PdfWriter

from extraction_review.structure_split import PageUnit, structure_aware_split


def _blank_pdf(page_count: int) -> bytes:
    writer = PdfWriter()
    for _ in range(page_count):
        writer.add_blank_page(width=612, height=792)
    output = io.BytesIO()
    writer.write(output)
    return output.getvalue()


def test_defect_005_index_and_p7_ranges_remain_continuous() -> None:
    page_count = 72
    texts = {page: "" for page in range(1, page_count + 1)}
    texts.update(
        {
            5: "INDEX\nS.No. Particulars Page No.",
            6: "| 17. | Annexure P-6: copy of notice | 48-53 |",
            # LlamaParse inserts blank lines inside markdown cells on this real
            # page. The Annexure names are Index inventory, not page owners.
            7: (
                "| 18. | Annexure P-7\n\nTrue translated copy of the notice "
                'dated 17.06.2015 published in "Amar Ujjala" newspaper. | 4-56 |\n'
                "| 19. | Annexure P-8\n\nTrue typed copy of the Letter dated "
                "21.7.2015. | 57-59 |\n"
                "| 20. | Annexure P-9\n\nTrue typed copy of the Sale "
                "Certificate. | 60-62 |"
            ),
            8: "| 23. | Application for exemption | 83-85 |",
            69: "ANNEXURE P-7\nNewspaper notice dated 17.06.2015\n54",
            70: "Newspaper notice continued\n55",
            71: (
                "Newspaper notice and Andhra Bank sale notice continued\n"
                "The secured property will be sold under Rule 9\n56"
            ),
            72: "ANNEXURE P-8\nTrue typed copy of letter dated 21.07.2015\n57",
        }
    )
    llama_parts = {page: ["Undefined"] for page in range(1, page_count + 1)}
    for page in range(5, 9):
        llama_parts[page] = ["Index"]
    llama_parts[69] = llama_parts[70] = ["Annexure P-7"]
    llama_parts[71] = ["Annexures"]
    llama_parts[72] = ["Annexure P-8"]

    units = [
        PageUnit(
            pdf_page=page,
            text=text,
            layout_fingerprint=("table",) if page in {5, 6, 7, 8, 71} else (),
            sentence_continues_previous=page in {6, 7, 8, 70, 71},
            paragraph_continues_previous=page in {6, 7, 8, 70, 71},
        )
        for page, text in texts.items()
    ]
    authoritative_segments = [
        {
            "segment_id": "boundary-index",
            "category": "Index",
            "start_page": 5,
            "end_page": 8,
            "protected": True,
            "confidence": 1.0,
            "type_evidence": ["exact_complete_segment_classification"],
            "review_reasons": [],
        },
        {
            "segment_id": "boundary-p7",
            "category": "Annexure P-7",
            "start_page": 69,
            "end_page": 70,
            "protected": False,
            "confidence": 0.9,
            "type_evidence": ["consistent_overlapping_classification"],
            "review_reasons": ["classifier_range_differs_from_boundary"],
        },
        {
            "segment_id": "boundary-generic",
            "category": "Annexures",
            "start_page": 71,
            "end_page": 71,
            "protected": False,
            "confidence": 0.9,
            "type_evidence": ["consistent_overlapping_classification"],
            "review_reasons": ["classifier_range_differs_from_boundary"],
        },
        {
            "segment_id": "boundary-p8",
            "category": "Annexure P-8",
            "start_page": 72,
            "end_page": 72,
            "protected": True,
            "confidence": 1.0,
            "type_evidence": ["exact_complete_segment_classification"],
            "review_reasons": [],
        },
    ]

    result = structure_aware_split(
        _blank_pdf(page_count),
        llama_page_parts=llama_parts,
        page_units=units,
        reconciliation_mode="guarded",
        authoritative_boundary_pages=[5, 69, 71, 72],
        authoritative_segments=authoritative_segments,
    )

    assert all(result.page_parts[page] == ["Index"] for page in range(5, 9))
    assert all(result.page_parts[page] == ["Annexure P-7"] for page in range(69, 72))
    assert result.page_parts[72] == ["Annexure P-8"]
    assert not any(
        change.page in {7, 8} and change.new_labels == ("Annexure P-7",)
        for change in result.reconciliation_changes
    )


def test_defect_005_generic_annexure_fragments_join_numbered_owner() -> None:
    page_count = 69
    texts = {page: "" for page in range(1, page_count + 1)}
    llama_parts = {page: ["Undefined"] for page in range(1, page_count + 1)}
    segments: list[dict[str, object]] = []
    ranges = [
        ("P-4", 51, 53, 54, 55),
        ("P-5", 56, 60, 61, 62),
        ("P-6", 63, 66, 67, 68),
    ]
    boundary_pages: list[int] = []
    for label, owned_start, owned_end, generic_start, generic_end in ranges:
        owner = f"Annexure {label}"
        texts[owned_start] = f"ANNEXURE {label}\nSale notice begins"
        for page in range(owned_start + 1, generic_end + 1):
            texts[page] = "Sale notice continuation without a new outer document"
        for page in range(owned_start, owned_end + 1):
            llama_parts[page] = [owner]
        for page in range(generic_start, generic_end + 1):
            llama_parts[page] = ["Annexures"]
        boundary_pages.extend((owned_start, generic_start))
        segments.extend(
            (
                {
                    "segment_id": f"boundary-{label}",
                    "category": owner,
                    "start_page": owned_start,
                    "end_page": owned_end,
                    "protected": True,
                    "confidence": 1.0,
                    "type_evidence": ["exact_complete_segment_classification"],
                    "review_reasons": [],
                },
                {
                    "segment_id": f"boundary-generic-after-{label}",
                    "category": "Annexures",
                    "start_page": generic_start,
                    "end_page": generic_end,
                    "protected": False,
                    "confidence": 1.0,
                    "type_evidence": ["exact_complete_segment_classification"],
                    "review_reasons": [],
                },
            )
        )

    units = [
        PageUnit(
            pdf_page=page,
            text=text,
            sentence_continues_previous=any(
                owned_start < page <= generic_end
                for _, owned_start, _, _, generic_end in ranges
            ),
            paragraph_continues_previous=any(
                owned_start < page <= generic_end
                for _, owned_start, _, _, generic_end in ranges
            ),
        )
        for page, text in texts.items()
    ]
    result = structure_aware_split(
        _blank_pdf(page_count),
        llama_page_parts=llama_parts,
        page_units=units,
        reconciliation_mode="guarded",
        authoritative_boundary_pages=boundary_pages,
        authoritative_segments=segments,
    )

    for label, owned_start, _, _, generic_end in ranges:
        assert all(
            result.page_parts[page] == [f"Annexure {label}"]
            for page in range(owned_start, generic_end + 1)
        )


def test_reconciliation_modes_have_distinct_ownership_authority() -> None:
    text = "ANNEXURE P-1\nTrue copy of order"
    units = [PageUnit(pdf_page=1, text=text)]
    baseline = {1: ["Index"]}
    protected = [
        {
            "segment_id": "protected-index",
            "category": "Index",
            "start_page": 1,
            "end_page": 1,
            "protected": True,
            "confidence": 1.0,
            "type_evidence": ["exact_complete_segment_classification"],
            "review_reasons": [],
        }
    ]

    llama_only = structure_aware_split(
        _blank_pdf(1),
        llama_page_parts=baseline,
        page_units=units,
        run_hybrid_repair=False,
        reconciliation_mode="llama_only",
        authoritative_boundary_pages=[1],
        authoritative_segments=protected,
    )
    guarded = structure_aware_split(
        _blank_pdf(1),
        llama_page_parts=baseline,
        page_units=units,
        reconciliation_mode="guarded",
        authoritative_boundary_pages=[1],
        authoritative_segments=protected,
    )
    legacy = structure_aware_split(
        _blank_pdf(1),
        llama_page_parts=baseline,
        page_units=units,
        reconciliation_mode="legacy",
        authoritative_boundary_pages=[1],
        authoritative_segments=protected,
    )

    assert llama_only.page_parts == {1: ["Index"]}
    assert guarded.page_parts == {1: ["Index"]}
    assert legacy.page_parts == {1: ["Annexure P-1"]}
    assert guarded.reconciliation_summary["status"] == "completed"
    assert legacy.reconciliation_summary["status"] == "completed_legacy_unchecked"


def test_defect_005_registry_notice_is_not_rop_or_application() -> None:
    """Physical page 1 must fall through to Unidentified, not Application 1."""
    from extraction_review.bundle_slicer import slice_bundle_pdf
    from extraction_review.split_upload import type_catalog

    notice = (
        "Delivery Mode: Registered\n"
        "IN THE SUPREME COURT OF INDIA\n"
        "CIVIL APPELLATE JURISDICTION\n"
        "PETITION FOR SPECIAL LEAVE TO APPEAL (C) NO. 12345 OF 2025\n"
        "A.B. ... PETITIONER\nVERSUS\nC.D. ... RESPONDENT\n"
        "WHEREAS the petition above-mentioned has been filed in this Court.\n"
        "Issue notice returnable on 10.02.2026.\n"
        "TAKE NOTICE that the matter will be listed for hearing.\n"
        "ASSISTANT REGISTRAR"
    )
    texts = {
        1: notice,
        2: (
            "IN THE SUPREME COURT OF INDIA\n"
            "I.A. NO. OF 2025\n"
            "APPLICATION FOR EXEMPTION FROM FILING OFFICIAL TRANSLATION\n"
            "MOST RESPECTFULLY SHOWETH"
        ),
        3: "PRAYER\nADVOCATE FOR THE PETITIONER",
    }
    segments = [
        {
            "segment_id": "llama-rop",
            "category": "Record of Proceedings",
            "start_page": 1,
            "end_page": 1,
            "protected": False,
            "confidence": 0.5,
            "type_evidence": ["segment_classification_uncategorized"],
            "review_reasons": ["boundary_label_differs_from_classification"],
        },
        {
            "segment_id": "llama-application",
            "category": "Application 1",
            "start_page": 2,
            "end_page": 3,
            "protected": True,
            "confidence": 1.0,
            "type_evidence": ["exact_complete_segment_classification"],
            "review_reasons": [],
        },
    ]
    pdf = _blank_pdf(3)
    result = structure_aware_split(
        pdf,
        llama_page_parts={
            1: ["Record of Proceedings"],
            2: ["Application 1"],
            3: ["Application 1"],
        },
        page_units=[PageUnit(pdf_page=page, text=text) for page, text in texts.items()],
        reconciliation_mode="guarded",
        authoritative_boundary_pages=[1, 2],
        authoritative_segments=segments,
    )

    assert 1 not in result.page_parts
    assert result.page_parts[2] == ["Application 1"]
    assert result.page_parts[3] == ["Application 1"]
    assert result.classifications[0].document_type == "other"
    assert "llama_rop_rejected" in result.classifications[0].signals
    assert any(
        event["stage"] == "registry_notice_rop_exclusion"
        and event["status"] == "committed"
        for event in result.assignment_events[1]
    )

    slices = slice_bundle_pdf(pdf, type_catalog("SLP_CIVIL"), result.page_parts)
    assert next(item for item in slices if item.label == "Undefined").pages == (1,)
    assert next(item for item in slices if item.label == "Application 1").pages == (
        2,
        3,
    )
