"""Tests for structure-aware page classify → boundary → logical document split."""

from __future__ import annotations

from extraction_review.structure_split import (
    DOCUMENT_TYPES,
    PageClassification,
    PageUnit,
    build_logical_documents,
    classify_complete_segments,
    classify_page,
    detect_boundaries,
    extract_page_units,
    reconcile_compiled_split,
    structure_aware_split,
)


def _blank_pdf(page_count: int) -> bytes:
    import io

    from pypdf import PdfWriter

    writer = PdfWriter()
    for _ in range(page_count):
        writer.add_blank_page(width=612, height=792)
    buffer = io.BytesIO()
    writer.write(buffer)
    return buffer.getvalue()


def test_document_types_are_controlled_taxonomy() -> None:
    assert "Main Petition" in DOCUMENT_TYPES
    assert "cover_page" not in DOCUMENT_TYPES  # canonical Split names only
    assert "other" in DOCUMENT_TYPES


def test_llama_boundaries_are_proposed_before_local_classification(monkeypatch) -> None:
    import extraction_review.structure_split as module

    events: list[str] = []
    original = module.classify_pages

    def record_classification(*args, **kwargs):
        events.append("classify")
        return original(*args, **kwargs)

    monkeypatch.setattr(module, "classify_pages", record_classification)
    original_propose = module.propose_llama_boundaries

    def record_boundary(*args, **kwargs):
        events.append("boundary")
        return original_propose(*args, **kwargs)

    monkeypatch.setattr(module, "propose_llama_boundaries", record_boundary)
    result = module.structure_aware_split(
        _blank_pdf(3),
        page_units=[
            PageUnit(pdf_page=1, text="SYNOPSIS\nSummary"),
            PageUnit(pdf_page=2, text="Summary continuation"),
            PageUnit(pdf_page=3, text="ANNEXURE P-1\nOrder"),
        ],
        llama_page_parts={1: ["Synopsis"], 2: ["Synopsis"], 3: ["Annexures"]},
        run_hybrid_repair=False,
    )

    assert events[:2] == ["boundary", "classify"]
    assert [proposal.page for proposal in result.llama_boundary_proposals] == [1, 3]
    assert result.llama_boundary_proposals[1].previous_labels == ("Synopsis",)
    assert result.llama_boundary_proposals[1].proposed_labels == ("Annexures",)
    assert any(
        "llama_segment_transition" in boundary.signals
        for boundary in result.boundaries
        if boundary.page == 3
    )


def test_complete_segment_classification_uses_full_physical_span() -> None:
    page_parts = {
        1: ["Synopsis"],
        2: ["Synopsis"],
        3: ["Annexure P-1"],
        4: ["Annexure P-1"],
    }
    units = [
        PageUnit(pdf_page=1, text="SYNOPSIS\nSummary"),
        PageUnit(pdf_page=2, text="Summary continued"),
        PageUnit(pdf_page=3, text="ANNEXURE P-1\nCertified order"),
        PageUnit(pdf_page=4, text="Order continuation"),
    ]

    segments = classify_complete_segments(page_parts, units)

    assert [(item.start_page, item.end_page) for item in segments] == [(1, 2), (3, 4)]
    assert [item.document_type for item in segments] == ["Synopsis", "Annexure P-1"]


def test_classify_page_prefers_annexure_stamp() -> None:
    unit = PageUnit(
        pdf_page=10,
        text="ANNEXURE P-1\nCertified copy of the High Court order\n",
    )
    result = classify_page(unit)
    assert result.document_type == "Annexure P-1"
    assert result.confidence >= 0.9
    assert "annexure_stamp" in result.signals


def test_classify_page_uses_llama_on_ocr_needed_empty_page() -> None:
    unit = PageUnit(pdf_page=5, text="  ", requires_ocr=True, char_count=0)
    result = classify_page(unit, llama_hint="Main Petition")
    assert result.document_type == "Main Petition"
    assert "llama_hint" in result.signals
    assert result.confidence < 0.7


def test_llamaparse_letter_spaced_index_activates_markdown_continuations() -> None:
    pages = {
        1: (
            "I N D E X\n"
            "| S.No. | PARTICULARS | Page No. of part to which it belongs |\n"
            "| --- | --- | --- |\n"
            "| 1. | Office Report on Limitation | A |\n"
            "| 2. | Listing Proforma | A1-A2 |\n"
            "| 3. | Cover Page | A3 |"
        ),
        2: (
            "| 11. | APPENDIX statutory provisions | 23A-23D |\n"
            "| --- | --- | --- |\n"
            "| 12. | ANNEXURE-P/1 Circular | 24-54 |\n"
            "| 13. | ANNEXURE-P/2 Order | 55-62 |"
        ),
        3: "OFFICE REPORT ON LIMITATION\nThe petition is within time.",
    }

    units = extract_page_units(
        _blank_pdf(3),
        page_texts=pages,
        use_local_ocr=False,
        supplied_text_source="llamaparse",
    )

    assert units[0].text.startswith("INDEX\nS.No. Particulars Page No.")
    assert units[1].text.startswith("INDEX\nS.No. Particulars Page No.")
    assert classify_page(units[0]).document_type == "Index"
    assert classify_page(units[1]).document_type == "Index"
    assert classify_page(units[2]).document_type == "Office Report on Limitation"


def test_targeted_reconciliation_preserves_supported_llama_segment() -> None:
    baseline = {1: ["Synopsis"], 2: ["Synopsis"]}
    proposed = {1: ["Annexure P-7"], 2: ["Annexure P-7"]}
    classifications = [
        PageClassification(
            page=page,
            document_type="Synopsis",
            confidence=0.55,
            signals=("llama_hint_only",),
            llama_hint="Synopsis",
        )
        for page in (1, 2)
    ]

    result, changes = reconcile_compiled_split(
        baseline,
        proposed,
        classifications,
        {1: "Synopsis narrative", 2: "Narrative continuation"},
        page_count=2,
    )

    assert result == baseline
    assert changes == []


def test_reconciliation_rejects_internal_cover_heading_in_annexure() -> None:
    baseline = {page: ["Annexure P-1"] for page in (1, 2, 3)}
    proposed = {**baseline, 2: ["Cover Page"]}
    classification = PageClassification(
        page=2,
        document_type="Cover Page",
        confidence=0.95,
        signals=("outer_anchor",),
        llama_hint="Annexure P-1",
    )
    operations = []

    result, changes = reconcile_compiled_split(
        baseline,
        proposed,
        [classification],
        {2: "IN THE SUPREME COURT OF INDIA\nPAPERBOOK"},
        page_count=3,
        operation_log=operations,
    )

    assert result == baseline
    assert changes == []
    assert any(
        operation.original_pages == (2,)
        and operation.status == "left_for_review"
        and "not independent evidence" in operation.reason
        for operation in operations
    )


def test_targeted_reconciliation_uses_index_range_over_main_span(monkeypatch) -> None:
    import extraction_review.structure_split as structure_split_module

    monkeypatch.setattr(
        structure_split_module,
        "_index_supported_map",
        lambda baseline, page_text, page_count: {
            1: ["Main Petition"],
            2: ["Annexure P-12"],
            3: ["Annexure P-12"],
        },
    )
    baseline = {page: ["Main Petition"] for page in range(1, 4)}

    result, changes = reconcile_compiled_split(
        baseline,
        baseline,
        [],
        {},
        page_count=3,
    )

    assert result == {
        1: ["Main Petition"],
        2: ["Annexure P-12"],
        3: ["Annexure P-12"],
    }
    assert [change.page for change in changes] == [2, 3]
    assert all("master_index_and_folio" in change.evidence for change in changes)


def test_application_prayer_mentioning_impugned_order_is_not_new_order() -> None:
    text = (
        "209\napplication for exemption from filing certified copy of the\n"
        "impugned judgment. The present application is made bonafide.\n"
        "PRAYER\nexempt the petitioner from filing the certified copy of the\n"
        "Impugned Judgment and Final Order dated 18.03.2026.\n209"
    )
    classification = classify_page(PageUnit(pdf_page=2, text=text))
    assert classification.document_type != "Impugned Order"


def test_numbered_application_does_not_revert_to_generic_on_reassessment() -> None:
    baseline = {1: ["Application"], 2: ["Application"]}
    numbered = {1: ["Application 1"], 2: ["Application 1"]}
    classifications = [
        PageClassification(
            page=1,
            document_type="Application 1",
            confidence=0.9,
            signals=("application_start",),
            llama_hint="Application",
        )
    ]
    result, _changes = reconcile_compiled_split(
        baseline,
        numbered,
        classifications,
        {1: "APPLICATION FOR EXEMPTION", 2: "PRAYER"},
        page_count=2,
        proposal_generator=lambda _current, _pass: baseline,
    )
    assert result == numbered


def test_targeted_reconciliation_applies_anchored_document_continuation() -> None:
    baseline = {
        1: ["Main Petition"],
        2: ["Main Petition"],
        3: ["Main Petition"],
    }
    proposed = {1: ["Synopsis"], 2: ["Synopsis"], 3: ["Main Petition"]}
    classifications = [
        PageClassification(
            page=1,
            document_type="Synopsis",
            confidence=0.9,
            signals=("outer_anchor",),
            llama_hint="Main Petition",
        ),
        PageClassification(
            page=2,
            document_type="Main Petition",
            confidence=0.5,
            signals=("llama_hint_only",),
            llama_hint="Main Petition",
        ),
        PageClassification(
            page=3,
            document_type="Main Petition",
            confidence=0.55,
            signals=("llama_hint_only",),
            llama_hint="Main Petition",
        ),
    ]

    result, changes = reconcile_compiled_split(
        baseline,
        proposed,
        classifications,
        {1: "SYNOPSIS\nCase summary", 2: "Summary continuation", 3: "PETITION"},
        page_count=3,
    )

    assert result == proposed
    assert [change.page for change in changes] == [1, 2]
    assert "explicit_outer_boundary" in changes[0].evidence
    assert "anchored_continuation" in changes[1].evidence

    repeated, repeated_changes = reconcile_compiled_split(
        result,
        proposed,
        classifications,
        {1: "SYNOPSIS\nCase summary", 2: "Summary continuation", 3: "PETITION"},
        page_count=3,
    )
    assert repeated == result
    assert repeated_changes == []


def test_targeted_reconciliation_does_not_swallow_multiple_llama_documents() -> None:
    baseline = {
        1: ["Synopsis"],
        2: ["Main Petition"],
        3: ["Affidavit"],
    }
    proposed = {
        1: ["Main Petition"],
        2: ["Main Petition"],
        3: ["Main Petition"],
    }
    classifications = [
        PageClassification(
            page=1,
            document_type="Main Petition",
            confidence=0.9,
            signals=("form28_lookalike",),
            llama_hint="Synopsis",
        )
    ]

    result, changes = reconcile_compiled_split(
        baseline,
        proposed,
        classifications,
        {1: "FORM 28", 2: "PETITION", 3: "AFFIDAVIT"},
        page_count=3,
    )

    assert result[1] == ["Main Petition"]
    assert result[2] == ["Main Petition"]
    assert result[3] == ["Affidavit"]
    assert [change.page for change in changes] == [1]


def test_targeted_reconciliation_fills_only_missing_pages_without_evidence() -> None:
    baseline = {1: ["Index"], 3: ["Main Petition"]}
    proposed = {
        1: ["Index"],
        2: ["Synopsis"],
        3: ["Main Petition"],
    }

    result, changes = reconcile_compiled_split(
        baseline,
        proposed,
        [],
        {1: "INDEX", 2: "continuation", 3: "PETITION"},
        page_count=3,
    )

    assert result == proposed
    assert len(changes) == 1
    assert changes[0].evidence == ("missing_or_unidentified",)


def test_boundary_detector_scores_heading_change() -> None:
    units = [
        PageUnit(pdf_page=1, text="SYNOPSIS\nThe instant Special Leave Petition"),
        PageUnit(
            pdf_page=2,
            text=(
                "IN THE SUPREME COURT OF INDIA\nCIVIL APPELLATE JURISDICTION\n"
                "SPECIAL LEAVE PETITION (CIVIL) NO. ______ OF 2026\n"
                "POSITION OF PARTIES\nMOST RESPECTFULLY SHOWETH:\n1. The instant"
            ),
        ),
    ]
    from extraction_review.structure_split import classify_pages

    classifications = classify_pages(units)
    boundaries = detect_boundaries(classifications, units)
    assert boundaries[0].page == 1
    assert any(b.page == 2 and b.action in {"auto", "verify"} for b in boundaries)


def test_logical_documents_preserve_page_ranges() -> None:
    labels = {1: "Cover Page", 2: "Main Petition", 3: "Main Petition", 5: "Affidavit"}
    from extraction_review.structure_split import PageClassification

    classifications = [
        PageClassification(page=p, document_type=t, confidence=0.9)
        for p, t in labels.items()
    ]
    docs = build_logical_documents(labels, classifications)
    assert docs[0].document_type == "Cover Page"
    assert docs[0].start_page == 1 and docs[0].end_page == 1
    assert docs[1].document_type == "Main Petition"
    assert docs[1].start_page == 2 and docs[1].end_page == 3
    assert docs[2].document_type == "Affidavit"


def test_structure_aware_split_keeps_synopsis_out_of_main() -> None:
    """End-to-end: Llama tags Synopsis as Main; structure + repair must separate."""
    import io

    from pypdf import PdfWriter

    # Build a tiny multi-page PDF with extractable text via reportlab-less path:
    # structure_aware_split accepts page_texts override, so empty PDF pages + texts.
    writer = PdfWriter()
    for _ in range(8):
        writer.add_blank_page(width=612, height=792)
    buf = io.BytesIO()
    writer.write(buf)
    pdf_bytes = buf.getvalue()

    texts = {
        1: (
            "IN THE SUPREME COURT OF INDIA\nCIVIL APPELLATE JURISDICTION\n"
            "SPECIAL LEAVE PETITION (CIVIL) NO. _______OF 2026\n"
            "PAPER BOOK\n(FOR INDEX KINDLY SEE INSIDE)\nADVOCATE FOR THE PETITIONER"
        ),
        2: "INDEX\n1. Synopsis 2\n2. Petition 4",
        3: (
            "SYNOPSIS\nThe instant Special Leave Petition under Article 136 "
            "is filed being aggrieved"
        ),
        4: "continuation of synopsis narrative",
        5: "LIST OF DATES AND EVENTS\n1964 On or about",
        6: (
            "IN THE SUPREME COURT OF INDIA\nCIVIL APPELLATE JURISDICTION\n"
            "SPECIAL LEAVE PETITION (CIVIL) NO. ______ OF 2026\n"
            "POSITION OF PARTIES\nMOST RESPECTFULLY SHOWETH:\n1. The instant"
        ),
        7: "Grounds A. Because the High Court erred",
        8: (
            "IN THE SUPREME COURT OF INDIA\nCERTIFICATE\n"
            "Certified that the Special Leave Petition is confined only to the pleadings"
        ),
    }
    llama = {page: ["Main Petition"] for page in range(3, 8)}
    llama[1] = ["Cover Page"]
    result = structure_aware_split(
        pdf_bytes,
        llama_page_parts=llama,
        page_texts=texts,
        run_hybrid_repair=True,
    )
    assert result.page_parts[3] == ["Synopsis"]
    assert result.page_parts[6] == ["Main Petition"]
    assert result.page_parts[7] == ["Main Petition"]
    assert any(d.document_type == "Synopsis" for d in result.logical_documents)
    assert any(d.document_type == "Main Petition" for d in result.logical_documents)
    report = result.report()
    assert report["architecture"] == "structure_aware_v1"
    assert "logical_documents" in report


def test_structure_aware_split_reuses_precomputed_page_units(monkeypatch) -> None:
    import extraction_review.structure_split as module

    units = [PageUnit(pdf_page=1, text="SYNOPSIS\nCase summary")]
    monkeypatch.setattr(
        module,
        "extract_page_units",
        lambda *_args, **_kwargs: (_ for _ in ()).throw(
            AssertionError("page units should not be extracted twice")
        ),
    )

    result = structure_aware_split(
        _blank_pdf(1),
        page_units=units,
        llama_page_parts={1: ["Synopsis"]},
    )

    assert result.page_units == units
    assert result.page_parts[1] == ["Synopsis"]


def test_structure_split_cuts_sci_application_out_of_annexure_carry() -> None:
    """A fresh I.A. caption must end an incorrectly carried annexure label."""
    from extraction_review.bundle_slicer import slice_bundle_pdf
    from extraction_review.split_upload import type_catalog

    texts = {
        1: ("ANNEXURE P-5\nIN THE HIGH COURT OF JUDICATURE AT BOMBAY\nORDER\n96"),
        2: "Continuation of the High Court order.\n97",
        3: "Continuation and conclusion of the High Court order.\n98",
        4: (
            "IN THE SUPREME COURT OF INDIA\n"
            "CIVIL APPELLATE JURISDICTION\n"
            "I.A. NO. ____ OF 2025\n"
            "IN SPECIAL LEAVE PETITION (CIVIL) NO. ____ OF 2025\n"
            "APPLICATION FOR EXEMPTION FROM FILING OFFICIAL TRANSLATION"
        ),
        5: (
            "4. The applicant seeks exemption from filing official translation.\n"
            "PRAYER\nIt is therefore most respectfully prayed.\n"
            "ADVOCATE FOR THE PETITIONER"
        ),
        6: "IN THE SUPREME COURT OF INDIA\nVAKALATNAMA",
    }
    # Reproduce the bad classifier result from the bundle: Annexure P-5 was
    # carried over both application pages.
    llama = {page: ["Annexure P-5"] for page in range(1, 6)}
    llama[6] = ["Vakalatnama"]
    pdf_bytes = _blank_pdf(6)

    result = structure_aware_split(
        pdf_bytes,
        llama_page_parts=llama,
        page_texts=texts,
        run_hybrid_repair=True,
    )

    assert all(result.page_parts[page] == ["Annexure P-5"] for page in range(1, 4))
    assert result.page_parts[4] == ["Application 1"]
    assert result.page_parts[5] == ["Application 1"]
    assert result.page_parts[6] == ["Vakalatnama"]

    slices = {
        item.slot_id: item
        for item in slice_bundle_pdf(
            pdf_bytes,
            type_catalog("SLP_CIVIL"),
            result.page_parts,
        )
    }
    assert slices["annexure_p5"].pages == (1, 2, 3)
    assert slices["application_1"].pages == (4, 5)
    assert slices["application_1"].filename == "Application 1.pdf"


def test_annexure_continuation_waits_for_next_strong_heading() -> None:
    """A body-like page cannot leave an opened annexure on Llama label alone."""
    texts = {
        1: "ANNEXURE P-1\nTrue copy of the registered lease deed\n26",
        2: (
            "Schedule of property\nBounded on the north by the public road\n"
            "continued terms and conditions\n27"
        ),
        3: "ANNEXURE P-2\nTrue copy of the subsequent agreement\n28",
    }
    llama = {
        1: ["Annexure P-1"],
        # Reproduce the bad page-wise classification reported in production.
        2: ["Listing Proforma"],
        3: ["Annexure P-2"],
    }

    result = structure_aware_split(
        _blank_pdf(3),
        llama_page_parts=llama,
        page_texts=texts,
        run_hybrid_repair=True,
    )

    assert result.page_parts[1] == ["Annexure P-1"]
    assert result.page_parts[2] == ["Annexure P-1"]
    assert result.page_parts[3] == ["Annexure P-2"]


def test_structure_split_continuity_gate_applies_to_non_annexure_documents() -> None:
    """The same stateful boundary rule protects every multi-page document."""
    texts = {
        1: "SYNOPSIS\nThe present petition challenges the impugned judgment.",
        2: "The controversy arose when the respondent issued the notice.",
        3: "LIST OF DATES AND EVENTS\n01.01.2024 Notice was issued.",
    }
    llama = {
        1: ["Synopsis"],
        2: ["Main Petition"],
        3: ["List of Dates & Events"],
    }

    result = structure_aware_split(
        _blank_pdf(3),
        llama_page_parts=llama,
        page_texts=texts,
        run_hybrid_repair=True,
    )

    assert result.page_parts[1] == ["Synopsis"]
    assert result.page_parts[2] == ["Synopsis"]
    assert result.page_parts[3] == ["List of Dates & Events"]


def test_folio_prefixed_synopsis_opens_after_listing_proforma() -> None:
    """A body High-Court reference cannot hide a real top-page SYNOPSIS."""
    texts = {
        1: "A-1\nPROFORMA FOR FIRST LISTING\nSECTION IV-A",
        2: "A-2\nPolice Station: Bihariganj\nSentence awarded: under trial",
        3: (
            "B\nSYNOPSIS OF THE CASE\nThe petition challenges the order passed "
            "by the Hon'ble High Court of Judicature at Patna."
        ),
        4: "The petitioner therefore seeks special leave to appeal.",
        5: "MAIN PETITION\nIN THE SUPREME COURT OF INDIA\nPOSITION OF PARTIES",
    }
    llama = {
        1: ["Listing Proforma"],
        2: ["Listing Proforma"],
        3: ["Listing Proforma"],
        4: ["Listing Proforma"],
        5: ["Main Petition"],
    }

    result = structure_aware_split(
        _blank_pdf(5),
        llama_page_parts=llama,
        page_texts=texts,
        run_hybrid_repair=True,
    )

    assert result.page_parts[1] == ["Listing Proforma"]
    assert result.page_parts[2] == ["Listing Proforma"]
    assert result.page_parts[3] == ["Synopsis"]
    assert result.page_parts[4] == ["Synopsis"]


def test_unassigned_annexure_continuation_does_not_reset_owner() -> None:
    """A missing Llama assignment remains inside the opened document."""
    texts = {
        1: "ANNEXURE P-1\nTrue copy of the registered lease deed",
        2: "Terms and conditions continued",
        3: "Signature of lessor\nTRUE COPY",
        4: "ANNEXURE P-2\nTrue copy of the subsequent agreement",
    }
    llama = {
        1: ["Annexure P-1"],
        3: ["Listing Proforma"],
        4: ["Annexure P-2"],
    }

    result = structure_aware_split(
        _blank_pdf(4),
        llama_page_parts=llama,
        page_texts=texts,
        run_hybrid_repair=True,
    )

    assert result.page_parts[1] == ["Annexure P-1"]
    assert result.page_parts[2] == ["Annexure P-1"]
    assert result.page_parts[3] == ["Annexure P-1"]
    assert result.page_parts[4] == ["Annexure P-2"]


def test_structure_split_accepts_verified_heading_as_real_boundary() -> None:
    """Continuity must not swallow a genuinely headed next document."""
    texts = {
        1: "ANNEXURE P-1\nTrue copy of the registered lease deed\n26",
        2: "LISTING PROFORMA\nSECTION: IV-A\nThe case pertains to service law.",
    }
    llama = {1: ["Annexure P-1"], 2: ["Listing Proforma"]}

    result = structure_aware_split(
        _blank_pdf(2),
        llama_page_parts=llama,
        page_texts=texts,
        run_hybrid_repair=True,
    )

    assert result.page_parts[1] == ["Annexure P-1"]
    assert result.page_parts[2] == ["Listing Proforma"]


def test_dedicated_boundary_pass_survives_weak_page_text() -> None:
    """Classification cannot erase a boundary established in the first pass."""
    texts = {
        1: "SYNOPSIS\nThe present petition challenges the impugned judgment.",
        2: "Synopsis narrative continues.",
        3: "01.01.2024 Notice was issued without a repeated chronology title.",
        4: "02.01.2024 The respondent replied.",
    }
    labels = {
        1: ["Synopsis"],
        2: ["Synopsis"],
        3: ["List of Dates & Events"],
        4: ["List of Dates & Events"],
    }

    result = structure_aware_split(
        _blank_pdf(4),
        llama_page_parts=labels,
        page_texts=texts,
        run_hybrid_repair=True,
        authoritative_boundary_pages=[3],
    )

    assert result.page_parts[1] == ["Synopsis"]
    assert result.page_parts[2] == ["Synopsis"]
    assert result.page_parts[3] == ["List of Dates & Events"]
    assert result.page_parts[4] == ["List of Dates & Events"]


def test_scanned_index_evidence_survives_targeted_reconciliation() -> None:
    """A verified Index span should not be lost when several page labels disagree."""
    texts = {
        1: "INDEX OF RECORD OF PROCEEDINGS",
        2: "ITEM NO.42 COURT NO.1\nSUPREME COURT OF INDIA\nRECORD OF PROCEEDINGS",
        3: "Applications for substitution are allowed. Cause title be amended.",
        4: (
            "INDEX\nS.No. Particulars Page No.\n"
            "10.\tAgainst the impugned judgment dated 01.05.2025 in "
            "RFA No. 665 of 2014\t1-2\n"
            "11.\tSpecial Leave Petition with affidavit\t3-4\n"
            "12.\tAnnexure P-1: order dated 04.03.2014 in "
            "Land Acquisition Case No 1209 of 2010\t5-6\n"
            "13.\tApplication for condonation of delay in Refiling "
            "the petition\t7-9"
        ),
        5: (
            "IN THE HIGH COURT OF PUNJAB AND HARYANA AT CHANDIGARH\n"
            "RFA-665-2018\nDate of Decision: 01.05.2025\n"
            "CORAM: HON'BLE MR. JUSTICE A\nORDER"
        ),
        6: "RFA No.8145 of 2014\nEx.P3\nJudgment continued",
        7: (
            "IN THE SUPREME COURT OF INDIA\nCIVIL APPELLATE JURISDICTION\n"
            "SPECIAL LEAVE PETITION (Civil) NO. OF 2025\n"
            "WITH PRAYER FOR INTERIM RELIEF\n"
            "BEFORE HIGH COURT BEFORE SUPREME COURT\n"
            "1. Vinod, s/o Kawara\nRespondent No. Petitioner\nVersus\n"
            "1. State of Haryana\nPetitioner No. Respondent No."
        ),
        8: "PRAYER in the special leave petition",
        9: "AFFIDAVIT\nDEPONENT",
        10: "Certificate supporting the petition",
        11: (
            "IN THE COURT OF ADDITIONAL DISTRICT JUDGE, ROHTAK\n"
            "Land Acquisition Case No. 1209/2010\n"
            "Date of Order: 04.03.2014\nJUDGMENT"
        ),
        12: "District-court judgment continues and concludes.",
        13: (
            "IN THE SUPREME COURT OF INDIA\nCIVIL APPELLATE JURISDICTION\n"
            "APPLICATION FOR CONDONATION OF DELAY IN REFILING SLP\n"
            "MOST RESPECTFULLY SHOWETH"
        ),
        14: "3. That the delay was not deliberate.\nPRAYER\nCondone the delay.",
        15: "AFFIDAVIT\nI verify the accompanying application.\nDEPONENT",
    }
    labels = {
        1: ["Record of Proceedings"], 2: ["Record of Proceedings"],
        3: ["Impugned Order"], 4: ["Index"],
        5: ["Impugned Order"], 6: ["Annexure P-3"],
        7: ["Main Petition"], 8: ["Main Petition"],
        9: ["Affidavit"], 10: ["AOR's Certificate"],
        11: ["Impugned Order"], 12: ["Main Petition"],
        13: ["Application 5"], 14: ["Application 6"],
        15: ["Application 6"],
    }

    result = structure_aware_split(
        _blank_pdf(15),
        page_units=[PageUnit(pdf_page=page, text=text) for page, text in texts.items()],
        llama_page_parts=labels,
        run_hybrid_repair=True,
        authoritative_boundary_pages=[14],
    )

    assert result.page_parts[3] == ["Record of Proceedings"]
    assert result.page_parts[5] == result.page_parts[6] == ["Impugned Order"]
    assert result.page_parts[11] == result.page_parts[12] == ["Annexure P-1"]
    assert result.page_parts[13] == result.page_parts[14] == result.page_parts[15] == ["Application 5"]
    assert any(
        event["stage"] == "strong_index_content_spans"
        and event["to_labels"] == ["Annexure P-1"]
        for event in result.assignment_events[11]
    )


def test_application_prayer_continuation_does_not_become_new_application() -> None:
    """A wrong dedicated boundary cannot promote an IA's prayer to a new IA."""
    texts = {
        1: (
            "IN THE SUPREME COURT OF INDIA\nI. A. NO. OF 2026\n"
            "AN APPLICATION FOR EXEMPTION FROM FILING CERTIFIED COPY\n"
            "MOST RESPECTFULLY SHOWETH:\n1. That the petitioner applies for exemption.\n208"
        ),
        2: (
            "2. The certified copy is not readily available.\n"
            "3. The present application is made bonafide.\nPRAYER\n"
            "The petitioner prays for exemption.\nFILED BY ADVOCATE\n209"
        ),
        3: (
            "IN THE SUPREME COURT OF INDIA\nI. A. NO. OF 2026\n"
            "AN APPLICATION FOR EXEMPTION FROM FILING OFFICIAL TRANSLATION\n"
            "MOST RESPECTFULLY SHOWETH:\n1. That the petitioner seeks leave.\n"
            "2. That Annexure P/4 is in Tamil.\n210"
        ),
        4: (
            "3. That due to paucity of time the translation could not be arranged.\n"
            "4. That the annexure was translated by an Advocate.\n"
            "5. The present application is made bonafide in the interest of justice.\n"
            "PRAYER\nIt is most respectfully prayed that this Hon'ble Court be pleased "
            "to exempt the petitioner from filing the official translation.\n"
            "FILED BY ADVOCATE\n211"
        ),
        5: (
            "IN THE SUPREME COURT OF INDIA\nI. A. NO. OF 2026\n"
            "AN APPLICATION FOR PERMISSION TO FILE ADDITIONAL DOCUMENTS\n"
            "MOST RESPECTFULLY SHOWETH:\n1. That the petitioner seeks permission.\n212"
        ),
    }
    result = structure_aware_split(
        _blank_pdf(5),
        page_units=[PageUnit(pdf_page=page, text=text) for page, text in texts.items()],
        llama_page_parts={
            1: ["Application 1"],
            2: ["Application 1"],
            3: ["Application 2"],
            4: ["Application 1"],
            5: ["Application 3"],
        },
        run_hybrid_repair=True,
        authoritative_boundary_pages=[3, 4, 5],
    )

    assert result.page_parts[1] == result.page_parts[2] == ["Application 1"]
    assert result.page_parts[3] == result.page_parts[4] == ["Application 2"]
    assert result.page_parts[5] == ["Application 3"]


def test_application_body_without_folio_is_not_a_confirmed_boundary_override() -> None:
    from extraction_review.structure_split import _confirmed_application_continuation

    current = {1: ["Application 2"], 2: ["Application 1"]}
    texts = {
        1: "APPLICATION FOR EXEMPTION\n2. That translation is required.",
        2: "3. That the translation could not be arranged.\nPRAYER",
    }

    assert not _confirmed_application_continuation(
        2, ("Application 1",), ("Application 2",), current, texts
    )


def test_scanned_old_case_petition_stays_in_stamped_annexure_until_outer_start() -> None:
    """Sparse scanned exhibit folios are not a new Main Petition island."""
    texts = {
        1: "IN THE SUPREME COURT OF INDIA\nSPECIAL LEAVE PETITION\nPRAYER\n214",
        2: "ANNEXURE P/12\nTrue Copy of W.P. (C) No. 28199 of 2012\n215",
        3: (
            "IN THE HIGH COURT OF KERALA AT ERNAKULAM\n"
            "W.P. (C) NO. 28199 OF 2012\nPETITION\n216"
        ),
        4: "217",
        5: "218",
        6: "219",
        7: "IN THE SUPREME COURT OF INDIA\nFILING MEMO\n233",
    }
    result = structure_aware_split(
        _blank_pdf(7),
        page_units=[PageUnit(pdf_page=page, text=text) for page, text in texts.items()],
        llama_page_parts={
            1: ["Main Petition"],
            2: ["Annexure P-12"],
            3: ["Main Petition"],
            4: ["Main Petition"],
            5: ["Main Petition"],
            6: ["Main Petition"],
            7: ["Filing Memo"],
        },
        run_hybrid_repair=True,
        authoritative_boundary_pages=[2, 3, 7],
    )

    assert result.page_parts[1] == ["Main Petition"]
    assert all(result.page_parts[page] == ["Annexure P-12"] for page in range(2, 7))
    assert result.page_parts[7] == ["Filing Memo"]


def test_appendix_heading_inside_annexure_remains_nested() -> None:
    texts = {
        1: "ANNEXURE P-1\nTrue copy of the High Court petition\n26",
        2: "APPENDIX\nDocuments accompanying the High Court writ petition\n27",
        3: "Appendix schedule continued\n28",
        4: "ANNEXURE P-2\nTrue copy of the High Court order\n29",
    }
    result = structure_aware_split(
        _blank_pdf(4),
        page_units=[PageUnit(pdf_page=page, text=text) for page, text in texts.items()],
        llama_page_parts={
            1: ["Annexure P-1"],
            2: ["Appendix"],
            3: ["Appendix"],
            4: ["Annexure P-2"],
        },
        run_hybrid_repair=True,
        authoritative_boundary_pages=[2, 4],
    )

    assert all(result.page_parts[page] == ["Annexure P-1"] for page in (1, 2, 3))
    assert result.page_parts[4] == ["Annexure P-2"]


def test_llama_only_preserves_remote_page_assignments_without_python_gate() -> None:
    """The compatibility mode remains a true Llama-only ownership path."""
    texts = {
        1: "ANNEXURE P-1\nTrue copy of an order",
        2: "Continuation of the order",
    }
    llama = {1: ["Annexure P-1"], 2: ["Listing Proforma"]}

    result = structure_aware_split(
        _blank_pdf(2),
        llama_page_parts=llama,
        page_texts=texts,
        run_hybrid_repair=False,
    )

    assert result.page_parts == llama


def test_extract_page_units_prefers_clean_layout_over_control_corrupted_date(
    monkeypatch,
) -> None:
    import extraction_review.structure_split as module

    pdf_bytes = _blank_pdf(1)
    clean = "IN THE SUPREME COURT OF INDIA\nCERTIFICATE\nDated:21.01.2026"
    monkeypatch.setattr(
        module,
        "extract_split_layout",
        lambda _pdf: {1: (clean, None, None)},
    )
    monkeypatch.setattr(module, "pages_with_large_images", lambda _pdf: set())
    monkeypatch.setattr(module, "ocr_sparse_pages", lambda _pdf, _pages: {})

    units = module.extract_page_units(
        pdf_bytes,
        page_texts={
            1: (
                "IN THE SUPREME COURT OF INDIA\nCERTIFICATE\n"
                "Dated:\x15\x14.\x13\x14.202\x19 extra padding"
            )
        },
    )

    assert "Dated:21.01.2026" in units[0].text
    assert "\x15" not in units[0].text
