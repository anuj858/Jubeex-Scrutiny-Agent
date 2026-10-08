"""Tests for structure-aware page classify → boundary → logical document split."""

from __future__ import annotations

from extraction_review.structure_split import (
    DOCUMENT_TYPES,
    PageUnit,
    build_logical_documents,
    classify_page,
    detect_boundaries,
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
