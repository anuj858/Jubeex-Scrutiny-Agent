import json
import os
import sys
from pathlib import Path

repo_root = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(repo_root))

from extraction_review.bundle_slicer import slice_bundle_pdf
from extraction_review.split_upload import type_catalog
from extraction_review.structure_split import structure_aware_split

out_dir = repo_root / "output_samples"
out_dir.mkdir(parents=True, exist_ok=True)

docs = [
    (
        "Writ-Petition-1",
        "Writ-Petition-1.pdf",
        "doc1_writ_petition",
    ),
    (
        "7c16530c-aa05-4fd1-bce8-70a27d86e9eb-compiled-petition-final-slp-sahyog-hospitality-vs-sunder-lal-gupta-v001 (1)",
        "7c16530c-compiled-petition.pdf",
        "doc2_slp_sahyog",
    ),
]

for stem, display_name, base_out in docs:
    art_path = Path(f"/Users/laptopbazaar/Documents/Petition_Folder/split_outputs/{stem}.split_artifact.json")
    pdf_path = Path(f"/Users/laptopbazaar/Documents/Petition_Folder/{stem}.pdf")
    if not pdf_path.exists():
        import glob
        matches = glob.glob(f"/Users/laptopbazaar/Documents/Petition_Folder/{stem}*.pdf")
        if matches:
            pdf_path = Path(matches[0])

    print(f"\nProcessing {base_out} ({pdf_path.name})...")
    with open(art_path) as f:
        art = json.load(f)
    with open(pdf_path, "rb") as f:
        pdf_bytes = f.read()

    audit = art["split_audit"]
    catalog = type_catalog(art["filing_type"])
    llama_output = audit.get("llama_segment_classification", {}).get("segments", [])

    # Reconstruct llama page parts from LlamaSplit segments
    llama_parts = {}
    for s in llama_output:
        cat = s["document_type"]
        for p in s["pages"]:
            llama_parts.setdefault(p, []).append(cat)

    manifest = audit["page_manifest"]["pages"]
    texts = {p["source_page"]: p.get("text_excerpt", "") for p in manifest}

    # Run structure_aware_split with updated rules (including Record of Proceedings fix)
    structured = structure_aware_split(
        pdf_bytes,
        llama_page_parts=llama_parts,
        page_texts=texts,
        run_hybrid_repair=True,
    )

    # Slice the bundle PDF chronologically
    slices = slice_bundle_pdf(pdf_bytes, catalog, structured.page_parts)

    has_dups = any(s.duplicate for s in slices)
    s3_payload = {
        "job_type": art.get("job_type", "split_petition"),
        "filing_type": art["filing_type"],
        "filename": display_name,
        "file_id": art["file_id"],
        "has_duplicates": has_dups,
        "parts": [
            {
                "slot_id": s.slot_id,
                "label": s.label,
                "filename": s.filename,
                "page_span": [dict(r) for r in s.page_span],
                "duplicate": s.duplicate,
                "file_hash": s.file_hash,
                "file_id": f"file_{s.slot_id}",
            }
            for s in slices
        ],
    }

    llama_file = out_dir / f"{base_out}_llamaspit_output.json"
    s3_file = out_dir / f"{base_out}_s3_response.json"

    with open(llama_file, "w") as f:
        json.dump(llama_output, f, indent=2)
    with open(s3_file, "w") as f:
        json.dump(s3_payload, f, indent=2)

    print(f"Saved: {llama_file.name}")
    print(f"Saved: {s3_file.name}")
    print(f"Total parts: {len(s3_payload['parts'])}, has_duplicates: {has_dups}")
    print("Parts overview:")
    for i, p in enumerate(s3_payload["parts"][:15]):
        span_str = ", ".join(f"{r['start']}-{r['end']}" if r['start'] != r['end'] else f"{r['start']}" for r in p['page_span'])
        print(f"  {i+1:2d}. {p['slot_id']:<25} | {p['label']:<30} | pp. {span_str:<10} | duplicate: {p['duplicate']}")
    if len(s3_payload["parts"]) > 15:
        print(f"  ... ({len(s3_payload['parts']) - 15} more parts) ...")

print("\nDone generating all samples!")
