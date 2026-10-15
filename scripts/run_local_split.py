#!/usr/bin/env python3
"""Run JubeeX production LlamaSplit pipeline locally on a folder of petition PDFs.

Extracts the exact page-to-document-type mapping and document section boundaries
for every petition using the full production pipeline (Classify -> Boundary Detection
-> Segment Classification -> Structure Rules & Repair -> Page Audit & Slicing).
"""

from __future__ import annotations

import argparse
import asyncio
import json
import logging
import os
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

# Ensure project root is in sys.path
REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from dotenv import load_dotenv

# Load local environment
load_dotenv(REPO_ROOT / ".env")

# Prevent config.py from overriding SPLIT_OUTPUT_DOCUMENTS on import
import src.extraction_review.config as config_mod

config_mod.load_repo_dotenv = lambda: REPO_ROOT / ".env"

# Configure defaults for local split run
os.environ["SPLIT_OUTPUT_DOCUMENTS"] = "All"
os.environ["SKIP_SLOT_PDF_UPLOAD"] = "true"

import pymupdf
from src.extraction_review.clients import get_llama_cloud_client
import src.extraction_review.process_file as process_file_mod
from src.extraction_review.process_file import (
    FileEvent,
    _upload_slot_pdf,
    remember_file_bytes,
)

logger = logging.getLogger("run_local_split")

# Global capture container for split artifacts
_CAPTURED_SPLIT_ARTIFACTS: dict[str, dict[str, Any]] = {}

# Hook process_file.upload_step_json to intercept the full production split artifact
_original_upload_step_json = process_file_mod.upload_step_json


def _intercepting_upload_step_json(step: str, payload: Any, **kwargs: Any) -> Any:
    if step == "split" and isinstance(payload, dict):
        fname = payload.get("filename") or "current"
        _CAPTURED_SPLIT_ARTIFACTS[fname] = payload
    return _original_upload_step_json(step, payload, **kwargs)


process_file_mod.upload_step_json = _intercepting_upload_step_json


def format_page_range(pages: list[int]) -> str:
    """Format a list of sorted page numbers into compact range string, e.g. '1-3, 5, 7-10'."""
    if not pages:
        return "none"
    ranges: list[str] = []
    start = pages[0]
    prev = pages[0]
    for p in pages[1:]:
        if p == prev + 1:
            prev = p
        else:
            ranges.append(f"{start}-{prev}" if start != prev else f"{start}")
            start = p
            prev = p
    ranges.append(f"{start}-{prev}" if start != prev else f"{start}")
    return ", ".join(ranges)


def build_page_mappings(
    artifact: dict[str, Any],
    total_pages: int,
) -> tuple[dict[int, str], list[dict[str, Any]]]:
    """Extract page_to_doc_type dict and contiguous document sections from artifact."""
    page_to_doc: dict[int, str] = {}
    page_manifest = (artifact.get("split_audit") or {}).get("page_manifest", {})
    manifest_rows = page_manifest.get("pages") or []

    # Priority 1: page manifest effective_owner
    for row in manifest_rows:
        if isinstance(row, dict):
            try:
                page_num = int(row.get("source_page"))
                owner = (row.get("effective_owner") or "").strip()
                labels = row.get("assigned_labels") or []
                if not owner or owner.lower() == "unidentified":
                    if labels:
                        owner = labels[0]
                    else:
                        owner = "Unidentified"
                page_to_doc[page_num] = owner
            except (TypeError, ValueError):
                continue

    # Priority 2: fallback to parts / slot_pages if page manifest had gaps
    if len(page_to_doc) < total_pages:
        parts = artifact.get("parts") or []
        for part in parts:
            if isinstance(part, dict):
                label = part.get("label") or part.get("slot_id") or "Unknown"
                spans = part.get("page_span") or []
                for span in spans:
                    if isinstance(span, dict):
                        start = int(span.get("start", 0))
                        end = int(span.get("end", 0))
                        for p in range(start, end + 1):
                            if p not in page_to_doc:
                                page_to_doc[p] = label

    # Priority 3: incorporate boundary proposals if page was Unidentified
    proposals = (artifact.get("split_audit") or {}).get("llama_boundary_proposals", [])
    sorted_props = sorted(
        [p for p in proposals if isinstance(p, dict) and p.get("page")],
        key=lambda x: x["page"],
    )
    for i, prop in enumerate(sorted_props):
        st = prop["page"]
        en = sorted_props[i + 1]["page"] - 1 if i + 1 < len(sorted_props) else total_pages
        lbls = prop.get("proposed_labels") or []
        if lbls and lbls[0].lower() not in {"undefined", "unidentified", "uncategorized"}:
            lbl = lbls[0]
            for p in range(st, min(en + 1, total_pages + 1)):
                if page_to_doc.get(p) in {None, "Unidentified", "Undefined", "uncategorized"}:
                    page_to_doc[p] = lbl

    # Fill any remaining unassigned pages
    for p in range(1, total_pages + 1):
        if p not in page_to_doc:
            page_to_doc[p] = "Unidentified"

    # Stabilize petition mapping: enforce single continuation for outer docs & bridge annexures
    from src.extraction_review.document_parts import _contiguous_groups

    cleaned = dict(page_to_doc)
    main_pages = sorted(p for p, d in cleaned.items() if d == "Main Petition")
    if main_pages:
        outer_main_start = main_pages[0]
        shell_parts = {
            "Main Petition",
            "AOR's Certificate",
            "AOR's Declaration",
            "Affidavit",
            "Appendix",
        }
        outer_shell_end = outer_main_start
        for p in range(outer_main_start, total_pages + 1):
            d = cleaned.get(p)
            if d in shell_parts:
                outer_shell_end = p
            elif d and d.lower() not in {"unidentified", "undefined", "uncategorized"}:
                break

        # Bridge any undefined/unidentified within main petition shell
        for p in range(outer_main_start, outer_shell_end + 1):
            d = cleaned.get(p)
            if not d or d.lower() in {"unidentified", "undefined", "uncategorized"}:
                cleaned[p] = "Main Petition"

        annexure_zone_start = outer_shell_end + 1

        outer_parts = {
            "Advocate's Checklist",
            "Cover Page",
            "Index",
            "Office Report on Limitation",
            "Listing Proforma",
            "Synopsis",
            "List of Dates & Events",
            "Impugned Order",
            "Main Petition",
            "AOR's Certificate",
            "AOR's Declaration",
            "Affidavit",
            "Appendix",
            "Record of Proceedings",
        }

        # In outer shell (before Main Petition): no Annexures or Applications allowed
        for p in range(1, annexure_zone_start):
            d = cleaned.get(p, "")
            if "annexure" in d.lower() or "application" in d.lower():
                prev_outer = [cleaned.get(x) for x in range(p - 1, 0, -1) if cleaned.get(x) in outer_parts]
                cleaned[p] = prev_outer[0] if prev_outer else "Index"

        # Outer shell: keep only first run of each outer part
        for part in outer_parts:
            part_pages = sorted(
                p for p in range(1, annexure_zone_start) if cleaned.get(p) == part
            )
            if part_pages:
                groups = _contiguous_groups(part_pages)
                if len(groups) > 1:
                    first_st, first_en = groups[0]
                    for p in part_pages:
                        if not (first_st <= p <= first_en):
                            cleaned[p] = "Unidentified"

        # Find true back matter start from closing administrative filings
        tail_closing_markers = {
            "Vakalatnama",
            "Filing Memo",
            "Memo of Appearance",
            "PoA/BR",
            "Memo of Parties",
        }
        tail_candidates = [
            p for p in range(annexure_zone_start, total_pages + 1)
            if cleaned.get(p) in tail_closing_markers
        ]
        if tail_candidates:
            tail_start = tail_candidates[0]
            if tail_start > 1 and cleaned.get(tail_start - 1) == "Index":
                tail_start -= 1
                cleaned[tail_start] = "Filing Memo"
            elif tail_start > 2 and cleaned.get(tail_start - 2) == "Index":
                cleaned[tail_start - 2] = "Filing Memo"
                tail_start -= 2
            if tail_start > 1 and cleaned.get(tail_start - 1) == "Proof of Service":
                tail_start -= 1
        else:
            tail_start = total_pages + 1

        # Post-petition zone (annexure_zone_start to tail_start - 1)
        # Demote any outer doc types appearing in this zone
        for p in range(annexure_zone_start, tail_start):
            d = cleaned.get(p, "")
            if d in outer_parts or d == "Proof of Service":
                cleaned[p] = "Annexures"

        # Handle Applications in post-petition zone:
        # A genuine outer Application must be >= 2 pages and not followed by a massive annexure block (> 15 pages)
        app_pages = sorted(
            p for p in range(annexure_zone_start, tail_start)
            if "application" in (cleaned.get(p) or "").lower()
        )
        if app_pages:
            for st, en in _contiguous_groups(app_pages):
                count = en - st + 1
                annex_after = sum(
                    1 for p in range(en + 1, tail_start)
                    if "annexure" in (cleaned.get(p) or "").lower()
                    or (cleaned.get(p) or "").lower() in {"unidentified", "undefined", "uncategorized"}
                )
                if count < 2 or annex_after > 15:
                    for p in range(st, en + 1):
                        cleaned[p] = "Annexures"

        # Fill remaining gaps in post-petition zone with active annexure label
        active_annex = "Annexures"
        for p in range(annexure_zone_start, tail_start):
            d = cleaned.get(p, "")
            if "application" in d.lower():
                continue
            if "annexure" in d.lower():
                active_annex = d
                continue
            cleaned[p] = active_annex

        # Back matter fill (tail_start to total_pages)
        for p in range(tail_start, total_pages + 1):
            d = cleaned.get(p, "")
            if d == "Index":
                cleaned[p] = "Filing Memo"
            elif not d or d.lower() in {"unidentified", "undefined", "uncategorized"}:
                prev = cleaned.get(p - 1)
                cleaned[p] = prev if prev else "Vakalatnama"

        page_to_doc = cleaned

    # Build contiguous document sections
    sections: list[dict[str, Any]] = []
    if total_pages > 0:
        curr_doc = page_to_doc.get(1, "Unidentified")
        curr_start = 1
        for p in range(2, total_pages + 1):
            doc = page_to_doc.get(p, "Unidentified")
            if doc != curr_doc:
                sections.append({
                    "document_type": curr_doc,
                    "start_page": curr_start,
                    "end_page": p - 1,
                    "page_count": (p - 1) - curr_start + 1,
                    "page_range": f"{curr_start}-{p - 1}" if curr_start != p - 1 else f"{curr_start}",
                })
                curr_doc = doc
                curr_start = p
        sections.append({
            "document_type": curr_doc,
            "start_page": curr_start,
            "end_page": total_pages,
            "page_count": total_pages - curr_start + 1,
            "page_range": f"{curr_start}-{total_pages}" if curr_start != total_pages else f"{curr_start}",
        })

    return page_to_doc, sections


async def process_single_pdf(
    pdf_path: Path,
    output_dir: Path,
    *,
    filing_type_override: str | None = None,
    client: Any,
) -> dict[str, Any]:
    """Run full LlamaSplit pipeline on one PDF and write structured outputs."""
    filename = pdf_path.name
    print(f"\n{'='*80}")
    print(f"PROCESSING: {filename}")
    print(f"{'='*80}")

    with pymupdf.open(pdf_path) as doc:
        total_pages = doc.page_count
        file_size_mb = pdf_path.stat().st_size / (1024 * 1024)

    print(f"File Size: {file_size_mb:.2f} MB | Total Physical Pages: {total_pages}")
    pdf_bytes = pdf_path.read_bytes()

    # Step 1: Upload to LlamaCloud for pipeline processing
    t0 = time.time()
    print("-> Uploading PDF to LlamaCloud...")
    file_id = await _upload_slot_pdf(
        client,
        filename=filename,
        pdf_bytes=pdf_bytes,
    )
    print(f"-> Uploaded successfully. LlamaCloud file_id: {file_id}")
    remember_file_bytes(file_id, pdf_bytes)

    # Step 2: Initialize ProcessFileWorkflow
    event_kwargs: dict[str, Any] = {
        "file_id": file_id,
        "filename": filename,
        "job_type": "upload_compiled" if not filing_type_override else "split_petition",
    }
    if filing_type_override:
        event_kwargs["filing_type"] = filing_type_override

    event = FileEvent(**event_kwargs)
    print("-> Launching JubeeX ProcessFileWorkflow...")
    handler = process_file_mod.workflow.run(start_event=event)

    # Stream real-time progress events
    stream = getattr(handler, "stream_events", None)
    if callable(stream):
        async for ev in stream():
            msg = getattr(ev, "message", str(ev))
            level = getattr(ev, "level", "info").upper()
            if msg and msg != "None":
                print(f"   [{level}] {msg}")

    result = await handler
    elapsed = time.time() - t0
    print(f"-> Pipeline completed in {elapsed:.1f}s!")

    # Retrieve intercepted split artifact
    artifact = _CAPTURED_SPLIT_ARTIFACTS.get(filename) or {}
    classified_filing_type = (
        getattr(result, "filing_type", None)
        or artifact.get("filing_type")
        or filing_type_override
        or "Unknown"
    )

    page_to_doc, sections = build_page_mappings(artifact, total_pages)

    # Invert mapping: document type -> list of pages
    doc_to_pages: dict[str, list[int]] = {}
    for p, doc_name in page_to_doc.items():
        doc_to_pages.setdefault(doc_name, []).append(p)

    # Summary payload for this file
    summary = {
        "filename": filename,
        "filepath": str(pdf_path.resolve()),
        "file_size_mb": round(file_size_mb, 2),
        "total_pages": total_pages,
        "filing_type": classified_filing_type,
        "processing_time_seconds": round(elapsed, 1),
        "processed_at": datetime.now(timezone.utc).isoformat(),
        "llama_cloud_file_id": file_id,
        "document_sections": sections,
        "doc_type_summary": [
            {
                "document_type": doc_name,
                "pages": sorted(pages),
                "page_count": len(pages),
                "page_range": format_page_range(sorted(pages)),
            }
            for doc_name, pages in sorted(doc_to_pages.items())
        ],
        "page_to_doc_type": {str(k): v for k, v in sorted(page_to_doc.items())},
        "review_status": (artifact.get("split_review") or {}).get("status", "unknown"),
        "audit_flags_count": (artifact.get("split_audit") or {}).get("flag_counts", {}).get("total", 0),
    }

    # Write output files
    stem = pdf_path.stem
    output_dir.mkdir(parents=True, exist_ok=True)

    # 1. Full Production Artifact JSON
    artifact_path = output_dir / f"{stem}.split_artifact.json"
    with open(artifact_path, "w", encoding="utf-8") as f:
        json.dump(artifact, f, indent=2, ensure_ascii=False)

    # 2. Page Mapping Summary JSON
    summary_path = output_dir / f"{stem}_page_mapping.json"
    with open(summary_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)

    # 3. Clean Human-Readable Text Report
    text_path = output_dir / f"{stem}_page_mapping.txt"
    with open(text_path, "w", encoding="utf-8") as f:
        f.write("=" * 80 + "\n")
        f.write(f"JUBEEX SPLIT REPORT: {filename}\n")
        f.write("=" * 80 + "\n\n")
        f.write(f"Filing Type       : {classified_filing_type}\n")
        f.write(f"Total Pages       : {total_pages}\n")
        f.write(f"File Size         : {file_size_mb:.2f} MB\n")
        f.write(f"Processing Time   : {elapsed:.1f}s\n")
        f.write(f"Review Status     : {summary['review_status']}\n")
        f.write(f"Audit Flags       : {summary['audit_flags_count']}\n\n")

        f.write("-" * 80 + "\n")
        f.write("DOCUMENT SECTIONS SUMMARY\n")
        f.write("-" * 80 + "\n")
        f.write(f"{'Document Type':<45} | {'Pages':<12} | {'Count':<6}\n")
        f.write("-" * 80 + "\n")
        for item in summary["doc_type_summary"]:
            f.write(f"{item['document_type']:<45} | {item['page_range']:<12} | {item['page_count']:<6}\n")
        f.write("-" * 80 + "\n\n")

        f.write("-" * 80 + "\n")
        f.write("PAGE-BY-PAGE MAPPING\n")
        f.write("-" * 80 + "\n")
        f.write(f"{'Page #':<8} | {'Document Type'}\n")
        f.write("-" * 80 + "\n")
        for p in range(1, total_pages + 1):
            f.write(f"Page {p:<3}  | {page_to_doc.get(p, 'Unidentified')}\n")
        f.write("-" * 80 + "\n")

    print(f"\n[OK] Results saved:")
    print(f"  - Text Report    : {text_path.name}")
    print(f"  - Mapping JSON   : {summary_path.name}")
    print(f"  - Artifact JSON  : {artifact_path.name}")

    # Print summary table to console
    print("\nDOCUMENT TYPE BREAKDOWN:")
    for item in summary["doc_type_summary"]:
        print(f"  * {item['document_type']:<40}: pp. {item['page_range']:<10} ({item['page_count']} pages)")

    return summary


def write_master_reports(all_summaries: list[dict[str, Any]], output_dir: Path) -> None:
    """Generate master markdown and JSON aggregating all processed petitions."""
    master_json_path = output_dir / "master_split_summary.json"
    with open(master_json_path, "w", encoding="utf-8") as f:
        json.dump(all_summaries, f, indent=2, ensure_ascii=False)

    master_md_path = output_dir / "MASTER_SPLIT_SUMMARY.md"
    with open(master_md_path, "w", encoding="utf-8") as f:
        f.write("# JubeeX LlamaSplit - Batch Split Summary Report\n\n")
        f.write(f"Generated on: {datetime.now(timezone.utc).strftime('%Y-%m-%d %H:%M:%S UTC')}\n\n")
        f.write(f"Total Petitions Processed: **{len(all_summaries)}**\n\n")

        f.write("## Overview of Processed Petitions\n\n")
        f.write("| # | Petition Filename | Filing Type | Pages | Size (MB) | Doc Types Count | Time (s) | Status |\n")
        f.write("|---|---|---|---|---|---|---|---|\n")
        for idx, item in enumerate(all_summaries, 1):
            f.write(
                f"| {idx} | `{item['filename']}` | **{item['filing_type']}** | "
                f"{item['total_pages']} | {item['file_size_mb']} MB | "
                f"{len(item['doc_type_summary'])} | {item['processing_time_seconds']}s | "
                f"`{item['review_status']}` |\n"
            )

        f.write("\n---\n\n")
        f.write("## Detailed Document Breakdown Per Petition\n\n")

        for idx, item in enumerate(all_summaries, 1):
            f.write(f"### {idx}. {item['filename']}\n\n")
            f.write(f"- **Filing Type**: `{item['filing_type']}`\n")
            f.write(f"- **Physical Pages**: {item['total_pages']}\n")
            f.write(f"- **Processing Time**: {item['processing_time_seconds']} seconds\n")
            f.write(f"- **Review Status**: `{item['review_status']}`\n\n")

            f.write("| Document Section | Page Range | Page Count |\n")
            f.write("|---|---|---|\n")
            for doc_item in item["doc_type_summary"]:
                f.write(f"| **{doc_item['document_type']}** | {doc_item['page_range']} | {doc_item['page_count']} |\n")
            f.write("\n")

            f.write("<details>\n<summary>Click to view full page-by-page mapping</summary>\n\n")
            f.write("| Page | Document Type |\n")
            f.write("|---|---|\n")
            for p_str, doc_name in item["page_to_doc_type"].items():
                f.write(f"| Page {p_str} | {doc_name} |\n")
            f.write("\n</details>\n\n---\n\n")

    print(f"\n{'='*80}")
    print(f"BATCH RUN COMPLETE! Master reports generated:")
    print(f"  - Master Markdown Report : {master_md_path}")
    print(f"  - Master JSON Summary    : {master_json_path}")
    print(f"{'='*80}\n")


async def main_async() -> int:
    parser = argparse.ArgumentParser(
        description="Run local JubeeX LlamaSplit pipeline on a directory of PDFs."
    )
    parser.add_argument(
        "--input-dir",
        "-i",
        type=Path,
        default=Path("/Users/laptopbazaar/Documents/Petition_Folder"),
        help="Directory containing PDF files (default: /Users/laptopbazaar/Documents/Petition_Folder)",
    )
    parser.add_argument(
        "--output-dir",
        "-o",
        type=Path,
        default=None,
        help="Directory to save split outputs (default: <input-dir>/split_outputs)",
    )
    parser.add_argument(
        "--file",
        "-f",
        type=str,
        default=None,
        help="Optional: name or path of a single PDF file to run",
    )
    parser.add_argument(
        "--filing-type",
        type=str,
        default=None,
        help="Optional: override filing type (e.g., SLP_CIVIL) to skip classification step",
    )
    parser.add_argument(
        "--upload-slices",
        action="store_true",
        default=False,
        help="If set, uploads sliced slot PDFs to LlamaCloud (default: skip slot upload to save time)",
    )
    parser.add_argument(
        "--split-output-docs",
        type=str,
        default="All",
        help="Configured SPLIT_OUTPUT_DOCUMENTS filter (default: 'All')",
    )

    parser.add_argument(
        "--skip-existing",
        action="store_true",
        default=False,
        help="If set, skips processing any PDF that already has a completed summary JSON in output-dir",
    )
    parser.add_argument(
        "--regenerate",
        action="store_true",
        default=False,
        help="If set, regenerates page mapping JSON, text reports, and master summary directly from existing *.split_artifact.json files without re-calling LlamaCloud",
    )

    args = parser.parse_args()

    input_dir = args.input_dir.expanduser().resolve()
    if not input_dir.exists():
        print(f"Error: Input directory does not exist: {input_dir}", file=sys.stderr)
        return 1

    output_dir = args.output_dir
    if output_dir is None:
        output_dir = input_dir / "split_outputs"
    output_dir = output_dir.expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    # Fast regeneration mode from existing artifacts
    if args.regenerate:
        print("=" * 80)
        print("JUBEEX LOCAL SPLIT - REGENERATION FROM EXISTING ARTIFACTS")
        print("=" * 80)
        print(f"Output Directory: {output_dir}")
        all_summaries: list[dict[str, Any]] = []
        artifact_files = sorted(output_dir.glob("*.split_artifact.json"))
        if args.file:
            target_stem = Path(args.file).stem
            artifact_files = [p for p in artifact_files if p.name.startswith(target_stem)]
        print(f"Found {len(artifact_files)} artifact(s) to regenerate.")

        for idx, art_path in enumerate(artifact_files, 1):
            stem = art_path.name.replace(".split_artifact.json", "")
            print(f"\n[{idx}/{len(artifact_files)}] Regenerating: {stem}...")
            with open(art_path, "r", encoding="utf-8") as f:
                artifact = json.load(f)
            pdf_path = input_dir / f"{stem}.pdf"
            if pdf_path.exists():
                with pymupdf.open(pdf_path) as doc:
                    total_pages = doc.page_count
                    file_size_mb = pdf_path.stat().st_size / (1024 * 1024)
            else:
                total_pages = len((artifact.get("split_audit") or {}).get("page_manifest", {}).get("pages", []))
                if total_pages == 0:
                    total_pages = (artifact.get("split_audit") or {}).get("total_pages") or 0
                file_size_mb = 0.0

            page_to_doc, sections = build_page_mappings(artifact, total_pages)
            doc_to_pages: dict[str, list[int]] = {}
            for p, doc_name in page_to_doc.items():
                doc_to_pages.setdefault(doc_name, []).append(p)

            classified_filing_type = (
                artifact.get("filing_type")
                or args.filing_type
                or "Unknown"
            )
            summary = {
                "filename": f"{stem}.pdf",
                "filepath": str(pdf_path.resolve()) if pdf_path.exists() else str(art_path.resolve()),
                "file_size_mb": round(file_size_mb, 2),
                "total_pages": total_pages,
                "filing_type": classified_filing_type,
                "processing_time_seconds": (artifact.get("split_review") or {}).get("duration_seconds", 0.0),
                "processed_at": datetime.now(timezone.utc).isoformat(),
                "llama_cloud_file_id": (artifact.get("split_audit") or {}).get("source_document_id", ""),
                "document_sections": sections,
                "doc_type_summary": [
                    {
                        "document_type": doc_name,
                        "pages": sorted(pages),
                        "page_count": len(pages),
                        "page_range": format_page_range(sorted(pages)),
                    }
                    for doc_name, pages in sorted(doc_to_pages.items())
                ],
                "page_to_doc_type": {str(k): v for k, v in sorted(page_to_doc.items())},
                "review_status": (artifact.get("split_review") or {}).get("status", "completed"),
                "audit_flags_count": (artifact.get("split_audit") or {}).get("flag_counts", {}).get("total", 0),
            }

            summary_path = output_dir / f"{stem}_page_mapping.json"
            with open(summary_path, "w", encoding="utf-8") as f:
                json.dump(summary, f, indent=2, ensure_ascii=False)

            text_path = output_dir / f"{stem}_page_mapping.txt"
            with open(text_path, "w", encoding="utf-8") as f:
                f.write("=" * 80 + "\n")
                f.write(f"JUBEEX SPLIT REPORT: {stem}.pdf\n")
                f.write("=" * 80 + "\n\n")
                f.write(f"Filing Type       : {classified_filing_type}\n")
                f.write(f"Total Pages       : {total_pages}\n")
                f.write(f"File Size         : {file_size_mb:.2f} MB\n")
                f.write(f"Review Status     : {summary['review_status']}\n")
                f.write(f"Audit Flags       : {summary['audit_flags_count']}\n\n")

                f.write("-" * 80 + "\n")
                f.write("DOCUMENT SECTIONS SUMMARY\n")
                f.write("-" * 80 + "\n")
                f.write(f"{'Document Type':<45} | {'Pages':<12} | {'Count':<6}\n")
                f.write("-" * 80 + "\n")
                for item in summary["doc_type_summary"]:
                    f.write(f"{item['document_type']:<45} | {item['page_range']:<12} | {item['page_count']:<6}\n")
                f.write("-" * 80 + "\n\n")

                f.write("-" * 80 + "\n")
                f.write("PAGE-BY-PAGE MAPPING\n")
                f.write("-" * 80 + "\n")
                f.write(f"{'Page #':<8} | {'Document Type'}\n")
                f.write("-" * 80 + "\n")
                for p in range(1, total_pages + 1):
                    f.write(f"Page {p:<3}  | {page_to_doc.get(p, 'Unidentified')}\n")
                f.write("-" * 80 + "\n")

            print(f"  [OK] Saved {summary_path.name} and {text_path.name}")
            for item in summary["doc_type_summary"]:
                print(f"    * {item['document_type']:<35}: pp. {item['page_range']:<12} ({item['page_count']} pages)")
            all_summaries.append(summary)

        if all_summaries:
            write_master_reports(all_summaries, output_dir)
        return 0

    # Set pipeline runtime settings
    os.environ["SPLIT_OUTPUT_DOCUMENTS"] = args.split_output_docs
    if args.upload_slices:
        os.environ["SKIP_SLOT_PDF_UPLOAD"] = "false"
    else:
        os.environ["SKIP_SLOT_PDF_UPLOAD"] = "true"

    # Identify PDF files to process
    if args.file:
        single_path = Path(args.file)
        if not single_path.is_absolute():
            single_path = input_dir / single_path
        if not single_path.exists() or single_path.suffix.lower() != ".pdf":
            print(f"Error: File not found or not a PDF: {single_path}", file=sys.stderr)
            return 1
        pdf_files = [single_path]
    else:
        pdf_files = sorted(
            [p for p in input_dir.iterdir() if p.is_file() and p.suffix.lower() == ".pdf" and not p.name.startswith(".")]
        )

    if not pdf_files:
        print(f"No PDF files found in {input_dir}", file=sys.stderr)
        return 1

    print("=" * 80)
    print("JUBEEX PRODUCTION SPLIT RUNNER")
    print("=" * 80)
    print(f"Input Directory  : {input_dir}")
    print(f"Output Directory : {output_dir}")
    print(f"Found {len(pdf_files)} PDF(s) to process:")
    for idx, p in enumerate(pdf_files, 1):
        print(f"  {idx}. {p.name}")
    print(f"Split Output Filter : {os.getenv('SPLIT_OUTPUT_DOCUMENTS')}")
    print(f"Upload Slices       : {'Enabled' if args.upload_slices else 'Disabled (fast local mode)'}")
    print(f"Skip Existing       : {'Enabled' if args.skip_existing else 'Disabled'}")
    print("=" * 80)

    client = get_llama_cloud_client()
    all_summaries: list[dict[str, Any]] = []
    failures: list[tuple[str, str]] = []

    for idx, pdf_path in enumerate(pdf_files, 1):
        summary_path = output_dir / f"{pdf_path.stem}_page_mapping.json"
        if args.skip_existing and summary_path.exists():
            try:
                with open(summary_path, "r", encoding="utf-8") as f:
                    cached_summary = json.load(f)
                print(f"\n>>> [{idx}/{len(pdf_files)}] [SKIP] Already processed: {pdf_path.name} (loaded existing output)")
                all_summaries.append(cached_summary)
                continue
            except Exception:
                pass

        print(f"\n>>> Running petition {idx}/{len(pdf_files)}: {pdf_path.name}")
        try:
            summary = await process_single_pdf(
                pdf_path,
                output_dir,
                filing_type_override=args.filing_type,
                client=client,
            )
            all_summaries.append(summary)
        except Exception as exc:
            logger.exception("Failed processing %s", pdf_path.name)
            print(f"\n[ERROR] Failed processing {pdf_path.name}: {exc}", file=sys.stderr)
            failures.append((pdf_path.name, str(exc)))

    if all_summaries:
        write_master_reports(all_summaries, output_dir)

    print("\n" + "=" * 80)
    print("EXECUTION SUMMARY")
    print("=" * 80)
    print(f"Successfully processed : {len(all_summaries)} / {len(pdf_files)}")
    if failures:
        print(f"Failed petitions       : {len(failures)}")
        for name, err in failures:
            print(f"  - {name}: {err}")
    print("=" * 80)

    return 0 if not failures else 2


def main() -> int:
    return asyncio.run(main_async())


if __name__ == "__main__":
    sys.exit(main())
