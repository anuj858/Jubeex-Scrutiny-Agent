"""Bounded local OCR for scanned split evidence; never change source PDF pages."""

from __future__ import annotations

import logging
import os
import shutil
import subprocess
import tempfile
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

logger = logging.getLogger(__name__)


def ocr_sparse_pages(pdf_bytes: bytes, pages: list[int]) -> dict[int, str]:
    if not pages:
        return {}
    executable = shutil.which("tesseract")
    if not executable:
        logger.warning(
            "Split OCR unavailable: install tesseract; %d pages unreadable", len(pages)
        )
        return {}
    import pymupdf

    # Render on one thread: PyMuPDF documents must not be shared across threads.
    # Only subprocess OCR runs concurrently, with one OpenMP thread per process.
    with tempfile.TemporaryDirectory(prefix="split-ocr-") as directory:
        root = Path(directory).resolve()
        results: dict[int, str] = {}

        def recognize(number: int) -> tuple[int, str]:
            try:
                result = subprocess.run(
                    [
                        executable,
                        str(root / f"{number}.png"),
                        "stdout",
                        "-l",
                        "eng",
                        "--psm",
                        "3",
                    ],
                    capture_output=True,
                    text=True,
                    timeout=30,
                    env={**os.environ, "OMP_THREAD_LIMIT": "1"},
                    check=True,
                )
                return number, result.stdout
            except (subprocess.SubprocessError, OSError) as exc:
                logger.warning(
                    "Split OCR failed for page %d: %s", number, type(exc).__name__
                )
                return number, ""
            finally:
                (root / f"{number}.png").unlink(missing_ok=True)

        with (
            pymupdf.open(stream=pdf_bytes, filetype="pdf") as document,
            ThreadPoolExecutor(max_workers=2) as pool,
        ):
            # Batches bound rendered-image storage and outstanding work.
            for offset in range(0, len(pages), 4):
                batch = pages[offset : offset + 4]
                for number in batch:
                    page = document[number - 1]
                    page.get_pixmap(dpi=180, colorspace=pymupdf.csGRAY).save(
                        root / f"{number}.png"
                    )
                results.update(pool.map(recognize, batch))
        logger.info(
            "Split OCR read %d/%d sparse pages",
            sum(bool(t.strip()) for t in results.values()),
            len(pages),
        )
        return results
