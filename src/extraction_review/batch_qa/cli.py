"""Command-line entry point for deployed batch QA."""

from __future__ import annotations

import argparse
import logging
import os
import sys
from pathlib import Path

from .emailer import send_report_email
from .evaluate import load_case_manifest
from .reporting import write_all_reports
from .runner import BatchQaRunner, BatchRunConfig, write_raw_case


def parser() -> argparse.ArgumentParser:
    result = argparse.ArgumentParser(
        description="Run PDF split, extraction, assembly, and scrutiny sequentially."
    )
    result.add_argument("--input-dir", type=Path, required=True)
    result.add_argument("--report-dir", type=Path, default=Path("qa_reports"))
    result.add_argument("--expected", type=Path, help="Optional expected-results JSON")
    result.add_argument("--api-url", default=os.getenv("QA_API_URL"))
    result.add_argument(
        "--api-key", default=os.getenv("QA_API_KEY") or os.getenv("JUBEEX_API_KEY")
    )
    result.add_argument(
        "--bucket", default=os.getenv("QA_INPUT_BUCKET") or os.getenv("AWS_S3_BUCKET")
    )
    result.add_argument("--region", default=os.getenv("AWS_REGION", "ap-south-1"))
    result.add_argument("--filing-type", default=os.getenv("QA_FILING_TYPE"))
    result.add_argument("--special-category", default=os.getenv("QA_SPECIAL_CATEGORY"))
    result.add_argument("--poll-interval", type=float, default=3)
    result.add_argument("--job-timeout", type=float, default=7200)
    result.add_argument("--request-timeout", type=float, default=60)
    result.add_argument("--delete-staged-inputs", action="store_true")
    result.add_argument("--dry-run", action="store_true")
    result.add_argument("--email", action="store_true")
    result.add_argument("--email-to", default=os.getenv("QA_REPORT_TO"))
    result.add_argument("--email-from", default=os.getenv("QA_REPORT_FROM"))
    return result


def main(argv: list[str] | None = None) -> int:
    args = parser().parse_args(argv)
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(message)s",
    )
    if not args.api_url:
        raise SystemExit("--api-url or QA_API_URL is required")
    if not args.bucket and not args.dry_run:
        raise SystemExit("--bucket, QA_INPUT_BUCKET, or AWS_S3_BUCKET is required")
    manifest = load_case_manifest(args.expected)
    config = BatchRunConfig(
        input_dir=args.input_dir,
        api_url=args.api_url,
        bucket=args.bucket or "dry-run",
        region=args.region,
        api_key=args.api_key,
        filing_type=args.filing_type,
        special_category=args.special_category,
        poll_interval=args.poll_interval,
        job_timeout=args.job_timeout,
        request_timeout=args.request_timeout,
        cleanup_staged_inputs=args.delete_staged_inputs,
        dry_run=args.dry_run,
    )
    runner = BatchQaRunner(config, manifest=manifest)
    try:
        batch = runner.run()
    finally:
        runner.close()
    raw_dir = args.report_dir / batch.run_id / "cases"
    for case in batch.cases:
        write_raw_case(case, raw_dir / Path(case.filename).stem)
    paths = write_all_reports(batch, args.report_dir / batch.run_id)
    print(f"JSON report: {paths['json']}")
    print(f"Excel report: {paths['xlsx']}")
    print(f"HTML report: {paths['html']}")
    if args.email:
        if not args.email_to or not args.email_from:
            raise SystemExit("--email requires --email-to and --email-from")
        recipients = [item.strip() for item in args.email_to.split(",") if item.strip()]
        message_id = send_report_email(
            batch,
            sender=args.email_from,
            recipients=recipients,
            region=args.region,
            html_path=paths["html"],
            xlsx_path=paths["xlsx"],
        )
        print(f"Email sent: {message_id}")
    return 1 if batch.failed_count else 0


if __name__ == "__main__":
    sys.exit(main())
