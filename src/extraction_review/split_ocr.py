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


def pages_with_large_images(
    pdf_bytes: bytes, *, minimum_page_coverage: float = 0.40
) -> set[int]:
    """Return pages containing a scan-sized raster image.

    A scanned page may already have a small or stale OCR text layer. Character
    count alone would then skip OCR even though the heading or handwritten
    Annexure stamp exists only in the image. Small logos, seals and signatures
    stay below the coverage threshold.
    """
    if not pdf_bytes:
        return set()
    try:
        import pymupdf

        document = pymupdf.open(stream=pdf_bytes, filetype="pdf")
    except Exception:  # noqa: BLE001 - scan detection must fail open
        return set()

    pages: set[int] = set()
    with document:
        for number in range(document.page_count):
            page = document[number]
            page_area = float(page.rect.width * page.rect.height)
            if page_area <= 0:
                continue
            try:
                image_area = sum(
                    max(0.0, float(block["bbox"][2] - block["bbox"][0]))
                    * max(0.0, float(block["bbox"][3] - block["bbox"][1]))
                    for block in page.get_text("dict").get("blocks", [])
                    if block.get("type") == 1 and block.get("bbox")
                )
            except Exception:  # noqa: BLE001 - malformed image blocks fail open
                continue
            if min(image_area / page_area, 1.0) >= minimum_page_coverage:
                pages.add(number + 1)
    return pages


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
