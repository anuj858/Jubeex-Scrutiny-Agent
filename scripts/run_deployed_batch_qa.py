#!/usr/bin/env python3
"""Run the deployed batch QA command without installing a separate package."""

from extraction_review.batch_qa.cli import main

if __name__ == "__main__":
    raise SystemExit(main())
