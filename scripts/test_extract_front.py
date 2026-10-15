#!/usr/bin/env python3
"""Testing script for POST /v1/filings/extract-front endpoint on petition PDFs.

Reads petition PDFs from /Users/laptopbazaar/Documents/Petition_Folder (or a specified directory),
invokes the extract-front API endpoint, polls the job until completion, and saves:
1. <stem>.extract_front.json (complete raw & finalized extraction JSON)
2. <stem>_front_extract_summary.txt (human-readable summary card)
3. MASTER_EXTRACT_FRONT_SUMMARY.md (comprehensive markdown overview report)
4. master_extract_front_summary.json (aggregated machine-readable summary)
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

import httpx
import pymupdf

from extraction_review.api import JOBS, app
from extraction_review.clients import get_llama_cloud_client
from extraction_review.config import FrontExtractRecord, LegalExtractRecord
from extraction_review.extract_record import unwrap_extracted_record
from extraction_review.process_file import (
    FRONT_EXTRACT_PAGE_LIMIT,
    _upload_slot_pdf,
    remember_file_bytes,
)
import extraction_review.s3_artifacts as s3_artifacts_mod

logger = logging.getLogger("test_extract_front")

# Global container for intercepting the exact extract artifact uploaded to S3
_CAPTURED_EXTRACT_ARTIFACTS: dict[str, dict[str, Any]] = {}

_original_upload_step_json = s3_artifacts_mod.upload_step_json


def _intercepting_upload_step_json(step: str, payload: Any, **kwargs: Any) -> Any:
    if step in ("extract", "extract_front") and isinstance(payload, dict):
        fname = payload.get("filename") or payload.get("file_name") or "current"
        _CAPTURED_EXTRACT_ARTIFACTS[fname] = payload
    return _original_upload_step_json(step, payload, **kwargs)


s3_artifacts_mod.upload_step_json = _intercepting_upload_step_json


def infer_filing_type(filename: str) -> str:
    """Infer the default JubeeX filing type based on filename keywords."""
    lower = filename.lower()
    if "writ" in lower:
        return "WRIT_PETITION_CIVIL"
    if "transfer" in lower:
        return "TRANSFER_PETITION_CIVIL"
    if "bail" in lower or "criminal" in lower:
        return "SLP_CRIMINAL"
    if "review" in lower:
        return "REVIEW_PETITION_CIVIL"
    if "arbitration" in lower:
        return "ARBITRATION_PETITION"
    # Default Supreme Court SLP
    return "SLP_CIVIL"


def extract_key_fields(data_dict: dict[str, Any]) -> dict[str, Any]:
    """Extract and format core legal fields from the finalized extract record."""
    inner = unwrap_extracted_record(data_dict)
    if isinstance(inner, dict) and "data" in inner and isinstance(inner["data"], dict):
        inner = inner["data"]

    court = inner.get("court")
    petition_type = inner.get("petition_type")
    special_category = inner.get("special_category")

    # Cause title
    cause_title = inner.get("cause_title") or {}
    formatted_title = (
        cause_title.get("formatted_title")
        or cause_title.get("petitioner_title")
        or "Not Extracted"
    )

    # Petitioners
    petitioners_raw = inner.get("petitioners") or []
    petitioners: list[str] = []
    for p in petitioners_raw:
        if isinstance(p, dict):
            name = p.get("name") or p.get("raw_text") or ""
            serial = p.get("serial")
            prefix = f"{serial}. " if serial else ""
            if name:
                petitioners.append(f"{prefix}{name}")
        elif isinstance(p, str):
            petitioners.append(p)

    # Respondents
    respondents_raw = inner.get("respondents") or []
    respondents: list[str] = []
    for r in respondents_raw:
        if isinstance(r, dict):
            name = r.get("name") or r.get("raw_text") or ""
            serial = r.get("serial")
            prefix = f"{serial}. " if serial else ""
            if name:
                respondents.append(f"{prefix}{name}")
        elif isinstance(r, str):
            respondents.append(r)

    # Advocates on record
    aor_raw = inner.get("advocates_on_record") or []
    advocates: list[dict[str, Any]] = []
    for a in aor_raw:
        if isinstance(a, dict):
            advocates.append({
                "name": a.get("name"),
                "code": a.get("code") or a.get("aor_code"),
                "state_bar": a.get("state_bar") or a.get("state_bar_council"),
                "raw_text": a.get("raw_text"),
            })
        elif isinstance(a, str):
            advocates.append({"name": a})

    # Impugned orders
    impugned_raw = inner.get("impugned_orders") or []
    impugned_orders: list[dict[str, Any]] = []
    for o in impugned_raw:
        if isinstance(o, dict):
            impugned_orders.append({
                "court": o.get("court") or o.get("court_name"),
                "date": o.get("date") or o.get("order_date"),
                "case_number": o.get("case_number") or o.get("appeal_number"),
                "judge": o.get("judge") or o.get("judges"),
            })

    # Confidence and metadata
    metadata = data_dict.get("metadata") or inner.get("metadata") or {}
    confidence = metadata.get("extract_confidence") or {}
    overall_confidence = confidence.get("overall")

    # Timing
    timing = data_dict.get("timing") or inner.get("timing") or {}

    return {
        "court": court,
        "petition_type": petition_type,
        "special_category": special_category,
        "formatted_title": formatted_title,
        "petitioner_title": cause_title.get("petitioner_title"),
        "respondent_title": cause_title.get("respondent_title"),
        "petitioners": petitioners,
        "respondents": respondents,
        "advocates": advocates,
        "impugned_orders": impugned_orders,
        "overall_confidence": overall_confidence,
        "field_confidence": confidence.get("fields") or {},
        "timing": timing,
        "raw_record": inner,
    }


def write_human_summary_card(
    summary: dict[str, Any],
    output_path: Path,
) -> None:
    """Write a clean, readable text report for quick manual inspection."""
    kf = summary["extracted_fields"]
    with open(output_path, "w", encoding="utf-8") as f:
        f.write("=" * 80 + "\n")
        f.write("JUBEEX EXTRACT-FRONT (FIRST 30 PAGES) SUMMARY CARD\n")
        f.write("=" * 80 + "\n\n")

        f.write(f"File Name           : {summary['filename']}\n")
        f.write(f"Original Pages      : {summary['total_pages']}\n")
        f.write(f"Pages Parsed        : 1 to {min(summary['total_pages'], FRONT_EXTRACT_PAGE_LIMIT)} (Front Slice)\n")
        f.write(f"File Size           : {summary['file_size_mb']:.2f} MB\n")
        f.write(f"Processing Time     : {summary['processing_time_seconds']}s\n")
        f.write(f"Job ID              : {summary['job_id']}\n")
        f.write(f"Agent Data ID       : {summary.get('agent_data_id') or 'N/A'}\n")
        f.write(f"Job Status          : {summary['status']}\n")
        f.write("-" * 80 + "\n\n")

        f.write("CORE CLASSIFICATION & SPECIAL CATEGORY:\n")
        f.write(f"  * Court             : {kf.get('court') or 'Not Extracted'}\n")
        f.write(f"  * Petition Type     : {kf.get('petition_type') or 'Not Extracted'}\n")
        f.write(f"  * Special Category  : {kf.get('special_category') or 'None (Standard/Unspecified)'}\n\n")

        f.write("CAUSE TITLE:\n")
        f.write(f"  * Formatted Title   : {kf.get('formatted_title')}\n")
        if kf.get("petitioner_title"):
            f.write(f"  * Petitioner Title  : {kf.get('petitioner_title')}\n")
        if kf.get("respondent_title"):
            f.write(f"  * Respondent Title  : {kf.get('respondent_title')}\n")
        f.write("\n")

        f.write(f"PETITIONERS ({len(kf.get('petitioners', []))} found):\n")
        if kf.get("petitioners"):
            for p in kf["petitioners"]:
                f.write(f"  - {p}\n")
        else:
            f.write("  (None extracted)\n")
        f.write("\n")

        f.write(f"RESPONDENTS ({len(kf.get('respondents', []))} found):\n")
        if kf.get("respondents"):
            for r in kf["respondents"]:
                f.write(f"  - {r}\n")
        else:
            f.write("  (None extracted)\n")
        f.write("\n")

        f.write(f"ADVOCATES ON RECORD ({len(kf.get('advocates', []))} found):\n")
        if kf.get("advocates"):
            for a in kf["advocates"]:
                details = []
                if a.get("code"):
                    details.append(f"Code: {a['code']}")
                if a.get("state_bar"):
                    details.append(f"Bar: {a['state_bar']}")
                suffix = f" ({', '.join(details)})" if details else ""
                f.write(f"  - {a.get('name') or 'Unknown'}{suffix}\n")
        else:
            f.write("  (None extracted)\n")
        f.write("\n")

        f.write(f"IMPUGNED ORDERS ({len(kf.get('impugned_orders', []))} found):\n")
        if kf.get("impugned_orders"):
            for idx, o in enumerate(kf["impugned_orders"], 1):
                f.write(f"  {idx}. Court       : {o.get('court') or 'N/A'}\n")
                f.write(f"     Order Date  : {o.get('date') or 'N/A'}\n")
                f.write(f"     Case Number : {o.get('case_number') or 'N/A'}\n")
                if o.get("judge"):
                    f.write(f"     Judge(s)    : {o['judge']}\n")
        else:
            f.write("  (None extracted)\n")
        f.write("\n")

        if kf.get("overall_confidence") is not None:
            f.write(f"Overall Confidence  : {kf['overall_confidence']:.2f}\n")
        f.write("=" * 80 + "\n")


def write_master_reports(all_summaries: list[dict[str, Any]], output_dir: Path) -> None:
    """Generate aggregated Markdown and JSON master reports for review."""
    master_json_path = output_dir / "master_extract_front_summary.json"
    with open(master_json_path, "w", encoding="utf-8") as f:
        json.dump(all_summaries, f, indent=2, ensure_ascii=False)

    master_md_path = output_dir / "MASTER_EXTRACT_FRONT_SUMMARY.md"
    with open(master_md_path, "w", encoding="utf-8") as f:
        f.write("# JubeeX Front Extract (`POST /v1/filings/extract-front`) - Test Summary\n\n")
        f.write(f"Generated on: {datetime.now(timezone.utc).strftime('%Y-%m-%d %H:%M:%S UTC')}\n\n")
        f.write(f"- Total Petitions Tested: **{len(all_summaries)}**\n")
        successful = sum(1 for s in all_summaries if s.get("status") == "completed")
        f.write(f"- Successful: **{successful}** | Failed: **{len(all_summaries) - successful}**\n\n")

        f.write("## Overview Table\n\n")
        f.write("| # | Petition Filename | Total Pages | Filing Type | Special Category | Cause Title | Lead Advocate | Orders | Time (s) | Status |\n")
        f.write("|---|---|---|---|---|---|---|---|---|---|\n")

        for idx, item in enumerate(all_summaries, 1):
            kf = item.get("extracted_fields") or {}
            advocates = kf.get("advocates") or []
            lead_adv = advocates[0].get("name", "N/A") if advocates else "N/A"
            cause = kf.get("formatted_title") or "N/A"
            if len(cause) > 50:
                cause = cause[:47] + "..."
            spec_cat = kf.get("special_category") or "—"
            f_type = kf.get("petition_type") or item.get("filing_type") or "N/A"
            orders_cnt = len(kf.get("impugned_orders") or [])

            f.write(
                f"| {idx} | `{item['filename']}` | {item['total_pages']} (slice: 30) | "
                f"**{f_type}** | `{spec_cat}` | {cause} | {lead_adv} | "
                f"{orders_cnt} | {item['processing_time_seconds']}s | `{item['status']}` |\n"
            )

        f.write("\n---\n\n")
        f.write("## Detailed Extraction Breakdown Per Petition\n\n")

        for idx, item in enumerate(all_summaries, 1):
            kf = item.get("extracted_fields") or {}
            f.write(f"### {idx}. `{item['filename']}`\n\n")
            f.write(f"- **Status**: `{item['status']}`\n")
            f.write(f"- **Original Pages**: {item['total_pages']} (parsed first 30 pages)\n")
            f.write(f"- **Processing Time**: {item['processing_time_seconds']} seconds\n")
            f.write(f"- **Court**: `{kf.get('court') or 'N/A'}`\n")
            f.write(f"- **Petition Type**: `{kf.get('petition_type') or 'N/A'}`\n")
            f.write(f"- **Special Category**: `{kf.get('special_category') or 'None'}`\n")
            f.write(f"- **Cause Title**: `{kf.get('formatted_title') or 'N/A'}`\n\n")

            # Advocates table
            f.write("#### Advocates on Record\n\n")
            if kf.get("advocates"):
                f.write("| Name | Code | State Bar |\n|---|---|---|\n")
                for a in kf["advocates"]:
                    f.write(f"| {a.get('name') or 'N/A'} | {a.get('code') or '—'} | {a.get('state_bar') or '—'} |\n")
            else:
                f.write("*None extracted*\n")
            f.write("\n")

            # Impugned Orders table
            f.write("#### Impugned Orders\n\n")
            if kf.get("impugned_orders"):
                f.write("| Court | Order Date | Case Number | Judge(s) |\n|---|---|---|---|\n")
                for o in kf["impugned_orders"]:
                    f.write(f"| {o.get('court') or 'N/A'} | {o.get('date') or '—'} | {o.get('case_number') or '—'} | {o.get('judge') or '—'} |\n")
            else:
                f.write("*None extracted*\n")
            f.write("\n")

            # Parties
            f.write("#### Parties\n\n")
            f.write(f"- **Petitioners**: {', '.join(kf.get('petitioners', [])) or 'None'}\n")
            f.write(f"- **Respondents**: {', '.join(kf.get('respondents', [])) or 'None'}\n\n")

            # Collapsible Full JSON
            raw_json = json.dumps(item.get("raw_record") or {}, indent=2, ensure_ascii=False)
            f.write("<details>\n<summary>Click to view full extracted JSON payload</summary>\n\n")
            f.write("```json\n" + raw_json + "\n```\n")
            f.write("\n</details>\n\n---\n\n")

    print(f"\n{'='*80}")
    print(f"MASTER REPORTS GENERATED:")
    print(f"  - Master Markdown Summary : {master_md_path}")
    print(f"  - Master JSON Summary     : {master_json_path}")
    print(f"{'='*80}\n")


async def process_single_pdf(
    pdf_path: Path,
    output_dir: Path,
    *,
    filing_type_override: str | None = None,
    court: str = "sci",
    api_url: str | None = None,
    api_key: str | None = None,
    client: Any,
    poll_interval: float = 2.0,
    timeout_s: float = 600.0,
) -> dict[str, Any]:
    """Execute POST /v1/filings/extract-front for a single PDF and save outputs."""
    filename = pdf_path.name
    print(f"\n{'='*80}")
    print(f"TESTING EXTRACT-FRONT: {filename}")
    print(f"{'='*80}")

    with pymupdf.open(pdf_path) as doc:
        total_pages = doc.page_count

    file_size_bytes = pdf_path.stat().st_size
    file_size_mb = file_size_bytes / (1024 * 1024)

    filing_type = filing_type_override or infer_filing_type(filename)
    print(f"  - Total Pages        : {total_pages}")
    print(f"  - Sliced For Extract : Pages 1 to {min(total_pages, FRONT_EXTRACT_PAGE_LIMIT)}")
    print(f"  - File Size          : {file_size_mb:.2f} MB")
    print(f"  - Selected Type      : {filing_type}")
    print(f"  - Court Code         : {court}")

    start_time = time.perf_counter()

    # Step 1: Upload source PDF to LlamaCloud
    print(f"\n[1/3] Uploading PDF to LlamaCloud for pipeline processing...")
    pdf_bytes = pdf_path.read_bytes()
    raw_file_id = await _upload_slot_pdf(
        client,
        filename=filename,
        pdf_bytes=pdf_bytes,
    )
    file_id = raw_file_id if raw_file_id.startswith("dfl-") else f"dfl-{raw_file_id}"
    remember_file_bytes(file_id, pdf_bytes)
    remember_file_bytes(raw_file_id, pdf_bytes)
    print(f"      Uploaded file_id: {raw_file_id} (pipeline ref: {file_id})")

    # Step 2: Post to /v1/filings/extract-front
    print(f"\n[2/3] Calling POST /v1/filings/extract-front...")
    request_payload = {
        "filing_type": filing_type,
        "court": court,
        "documents": [
            {
                "name": filename,
                "filename": filename,
                "file_id": file_id,
                "download_url": f"file://{pdf_path.resolve()}",
            }
        ],
    }

    headers = {"Content-Type": "application/json"}
    if api_key:
        headers["X-API-Key"] = api_key

    transport = None
    base_url = "http://testserver"
    if api_url:
        base_url = api_url.rstrip("/")
    else:
        # Run in-process with FastAPI ASGI transport
        transport = httpx.ASGITransport(app=app)

    async with httpx.AsyncClient(
        transport=transport,
        base_url=base_url,
        timeout=httpx.Timeout(timeout_s),
    ) as http:
        response = await http.post(
            "/v1/filings/extract-front",
            json=request_payload,
            headers=headers,
        )

        if response.status_code != 202:
            raise RuntimeError(
                f"API returned status {response.status_code}: {response.text}"
            )

        job_info = response.json()
        job_id = job_info["job_id"]
        poll_url = job_info.get("poll_url") or f"/v1/jobs/{job_id}"
        print(f"      Accepted! Job ID: {job_id}")
        print(f"      Polling URL: {poll_url}")

        # Step 3: Poll job status until completed
        print(f"\n[3/3] Polling job status...")
        poll_start = time.time()
        last_stage_msg = ""
        job_record: dict[str, Any] = {}

        while True:
            if time.time() - poll_start > timeout_s:
                raise TimeoutError(f"Job {job_id} timed out after {timeout_s}s")

            # Check job status via API
            poll_resp = await http.get(poll_url, headers=headers)
            if poll_resp.status_code == 200:
                job_record = poll_resp.json()
            elif job_id in JOBS:
                job_record = JOBS[job_id].as_out().model_dump(mode="json")
            else:
                logger.warning("Poll returned %s", poll_resp.status_code)

            status = job_record.get("status", "running")
            progress = job_record.get("progress", 0)
            stage = job_record.get("stage") or "running"
            msg = job_record.get("stage_message") or ""

            if msg != last_stage_msg:
                elapsed = int(time.time() - poll_start)
                print(f"      [{elapsed:>3}s] [{progress:>2}%] [{stage:<12}] {msg}")
                last_stage_msg = msg

            if status == "completed":
                print(f"      [DONE] Job completed successfully!")
                break
            elif status == "failed":
                err = job_record.get("error") or "Unknown error"
                raise RuntimeError(f"Job {job_id} failed: {err}")

            await asyncio.sleep(poll_interval)

    duration = round(time.perf_counter() - start_time, 2)
    agent_data_id = job_record.get("agent_data_id")

    # Step 4: Extract and finalize records
    captured = _CAPTURED_EXTRACT_ARTIFACTS.get(filename) or {}
    raw_record = captured.get("data") or {}

    # If captured is empty, attempt to fetch from LlamaCloud Agent Data
    if not raw_record and agent_data_id:
        try:
            item = await client.beta.agent_data.get(agent_data_id)
            if item and hasattr(item, "data") and isinstance(item.data, dict):
                raw_record = item.data.get("data") or item.data
        except Exception as exc:
            logger.warning("Could not fetch agent_data %s: %s", agent_data_id, exc)

    if not raw_record:
        # Fallback to job_record result
        raw_record = (job_record.get("result") or {}).get("report") or {}

    extracted_fields = extract_key_fields({"data": raw_record, "metadata": captured.get("metadata")})

    summary = {
        "filename": filename,
        "total_pages": total_pages,
        "file_size_mb": round(file_size_mb, 2),
        "filing_type": filing_type,
        "job_id": job_id,
        "agent_data_id": agent_data_id,
        "status": job_record.get("status", "completed"),
        "processing_time_seconds": duration,
        "extracted_fields": extracted_fields,
        "raw_record": raw_record,
        "job_record": job_record,
    }

    # Save output artifacts
    stem = pdf_path.stem
    json_path = output_dir / f"{stem}.extract_front.json"
    card_path = output_dir / f"{stem}_front_extract_summary.txt"

    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)

    write_human_summary_card(summary, card_path)

    print(f"\n[OK] Results saved:")
    print(f"  - Full Extraction JSON : {json_path.name}")
    print(f"  - Readable Text Card   : {card_path.name}")

    # Print summary highlights to console
    kf = extracted_fields
    print("\nEXTRACTION HIGHLIGHTS:")
    print(f"  * Court             : {kf.get('court')}")
    print(f"  * Petition Type     : {kf.get('petition_type')}")
    print(f"  * Special Category  : {kf.get('special_category')}")
    print(f"  * Cause Title       : {kf.get('formatted_title')}")
    print(f"  * Advocates (AOR)   : {len(kf.get('advocates', []))} advocate(s)")
    print(f"  * Impugned Orders   : {len(kf.get('impugned_orders', []))} order(s)")
    print(f"  * Total Time        : {duration}s")

    return summary


async def main_async() -> int:
    parser = argparse.ArgumentParser(
        description="Run JubeeX extract-front (POST /v1/filings/extract-front) on petition PDFs."
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
        help="Directory to save extract outputs (default: <input-dir>/extract_front_outputs)",
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
        help="Optional override for filing type (e.g. SLP_CIVIL, WRIT_PETITION_CIVIL)",
    )
    parser.add_argument(
        "--court",
        type=str,
        default="sci",
        help="Court code (default: sci)",
    )
    parser.add_argument(
        "--api-url",
        type=str,
        default=None,
        help="Optional URL of a live running API server (e.g. http://localhost:8000). If omitted, runs in-process via ASGI.",
    )
    parser.add_argument(
        "--api-key",
        type=str,
        default=os.getenv("JUBEEX_API_KEY", ""),
        help="API Key for X-API-Key header",
    )
    parser.add_argument(
        "--poll-interval",
        type=float,
        default=2.0,
        help="Seconds between job status polls (default: 2.0)",
    )
    parser.add_argument(
        "--timeout",
        type=float,
        default=600.0,
        help="Timeout in seconds per petition (default: 600.0)",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        default=False,
        help="If set, reprocesses petitions even if their output JSON already exists",
    )

    args = parser.parse_args()

    input_dir = args.input_dir.expanduser().resolve()
    if not input_dir.exists():
        print(f"Error: Input directory does not exist: {input_dir}", file=sys.stderr)
        return 1

    output_dir = args.output_dir
    if output_dir is None:
        output_dir = input_dir / "extract_front_outputs"
    output_dir = output_dir.expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

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
            [
                p
                for p in input_dir.iterdir()
                if p.is_file() and p.suffix.lower() == ".pdf" and not p.name.startswith(".")
            ]
        )

    if not pdf_files:
        print(f"No PDF files found in {input_dir}", file=sys.stderr)
        return 1

    print("=" * 80)
    print("JUBEEX POST /v1/filings/extract-front TEST RUNNER")
    print("=" * 80)
    print(f"Input Directory  : {input_dir}")
    print(f"Output Directory : {output_dir}")
    print(f"Execution Target : {args.api_url or 'In-Process FastAPI ASGI'}")
    print(f"Found {len(pdf_files)} PDF(s) to process:")
    for idx, p in enumerate(pdf_files, 1):
        print(f"  {idx}. {p.name}")
    print("=" * 80)

    client = get_llama_cloud_client()
    all_summaries: list[dict[str, Any]] = []
    failures: list[tuple[str, str]] = []

    for idx, pdf_path in enumerate(pdf_files, 1):
        existing_json = output_dir / f"{pdf_path.stem}.extract_front.json"
        if existing_json.exists() and not args.force:
            print(f"\n>>> [{idx}/{len(pdf_files)}] Reusing existing result for: {pdf_path.name}")
            try:
                with open(existing_json, encoding="utf-8") as f:
                    cached_summary = json.load(f)
                all_summaries.append(cached_summary)
                print(f"      Loaded: {existing_json.name} (use --force to reprocess)")
                continue
            except Exception as exc:
                logger.warning("Could not read %s, will reprocess: %s", existing_json.name, exc)

        print(f"\n>>> Running petition {idx}/{len(pdf_files)}: {pdf_path.name}")
        try:
            summary = await process_single_pdf(
                pdf_path,
                output_dir,
                filing_type_override=args.filing_type,
                court=args.court,
                api_url=args.api_url,
                api_key=args.api_key,
                client=client,
                poll_interval=args.poll_interval,
                timeout_s=args.timeout,
            )
            all_summaries.append(summary)
            # Update master reports incrementally
            write_master_reports(all_summaries, output_dir)
        except Exception as exc:
            logger.exception("Failed testing extract-front on %s", pdf_path.name)
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
