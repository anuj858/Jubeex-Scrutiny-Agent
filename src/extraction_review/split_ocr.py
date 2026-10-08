"""Bounded local OCR for scanned split evidence; never change source PDF pages."""

from __future__ import annotations

import csv
import io
import logging
import os
import re
import shutil
import subprocess
import tempfile
from concurrent.futures import ThreadPoolExecutor
from itertools import pairwise
from pathlib import Path

logger = logging.getLogger(__name__)

DEFAULT_SPLIT_OCR_CONCURRENCY = 2

_FOLIO_TOKEN_RE = re.compile(r"(?:\d{1,4}[A-Za-z]?|A\d{1,2}|[A-Z]{1,2})")
_INDEX_PAGE_SPAN_RE = re.compile(
    r"(?:\d{1,4}[A-Za-z]?(?:[-–—]\d{1,4}[A-Za-z]?)?|"
    r"A\d{1,2}(?:[-–—]A?-?\d{1,2})?|[A-Z]{1,2}(?:[-–—][A-Z]{1,2})?)"
)


def split_ocr_concurrency() -> int:
    """Bound local Tesseract processes to the CPU allocated to the worker."""
    raw = (os.getenv("SPLIT_OCR_CONCURRENCY") or "").strip()
    try:
        return max(1, min(int(raw), 8)) if raw else DEFAULT_SPLIT_OCR_CONCURRENCY
    except ValueError:
        return DEFAULT_SPLIT_OCR_CONCURRENCY


def _cluster_positions(values: list[float], *, tolerance: float) -> list[float]:
    clusters: list[list[float]] = []
    for value in sorted(values):
        if not clusters or value - clusters[-1][-1] > tolerance:
            clusters.append([value])
        else:
            clusters[-1].append(value)
    return [sum(cluster) / len(cluster) for cluster in clusters]


def index_table_rows_from_tsv(tsv: str) -> str:
    """Rebuild scanned filing-Index rows from Tesseract word geometry.

    Tesseract's plain text commonly emits the particulars and printed-folio
    columns as unrelated blocks. The Index parser then sees ``Annexure P-4``
    without its ``73-144`` range. Scanned Supreme Court indexes retain the
    table rules in TSV, so use those rules to emit stable three-cell rows.
    """
    try:
        rows = list(csv.DictReader(io.StringIO(tsv), delimiter="\t"))
    except (csv.Error, TypeError):
        return ""
    page_row = next((row for row in rows if row.get("level") == "1"), None)
    if not page_row:
        return ""
    try:
        page_width = int(page_row.get("width") or 0)
        page_height = int(page_row.get("height") or 0)
    except ValueError:
        return ""
    if page_width <= 0 or page_height <= 0:
        return ""

    boxes: list[tuple[int, int, int, int, str]] = []
    for row in rows:
        if row.get("level") != "5":
            continue
        try:
            left = int(row.get("left") or 0)
            top = int(row.get("top") or 0)
            width = int(row.get("width") or 0)
            height = int(row.get("height") or 0)
        except ValueError:
            continue
        boxes.append((left, top, width, height, (row.get("text") or "").strip()))

    verticals = _cluster_positions(
        [
            left + width / 2
            for left, _top, width, height, _text in boxes
            if width <= max(12, page_width * 0.012) and height >= page_height * 0.25
        ],
        tolerance=max(8, page_width * 0.008),
    )
    horizontals = _cluster_positions(
        [
            top + height / 2
            for _left, top, width, height, _text in boxes
            if height <= max(12, page_height * 0.012) and width >= page_width * 0.40
        ],
        tolerance=max(8, page_height * 0.006),
    )
    if len(verticals) < 4 or len(horizontals) < 2:
        return ""

    # The filing Index begins with serial, particulars, and printed-page
    # columns. Later columns (file-only page and remarks) are irrelevant.
    table_left, serial_right, particulars_right, page_right = verticals[:4]
    if not (
        table_left < serial_right < particulars_right < page_right
        and particulars_right - serial_right >= page_width * 0.12
    ):
        return ""

    words = [box for box in boxes if box[4]]

    def cell_text(
        top_bound: float, bottom_bound: float, left_bound: float, right_bound: float
    ) -> str:
        selected: list[tuple[float, int, str]] = []
        for left, top, width, height, text in words:
            center_x = left + width / 2
            center_y = top + height / 2
            if (
                top_bound < center_y < bottom_bound
                and left_bound < center_x < right_bound
            ):
                selected.append((center_y, left, text))
        lines: list[list[tuple[float, int, str]]] = []
        line_tolerance = max(10.0, page_height * 0.010)
        for word in sorted(selected):
            if not lines or word[0] - lines[-1][-1][0] > line_tolerance:
                lines.append([word])
            else:
                lines[-1].append(word)
        return " ".join(
            text
            for line in lines
            for _center_y, _left, text in sorted(line, key=lambda item: item[1])
        ).strip()

    rebuilt: list[str] = []
    for top_bound, bottom_bound in pairwise(horizontals):
        serial_text = cell_text(top_bound, bottom_bound, table_left, serial_right)
        serial_match = re.search(r"\d{1,3}", serial_text)
        if not serial_match:
            continue
        particulars = cell_text(
            top_bound, bottom_bound, serial_right, particulars_right
        )
        raw_span = re.sub(
            r"\s+",
            "",
            cell_text(top_bound, bottom_bound, particulars_right, page_right),
        )
        span_match = _INDEX_PAGE_SPAN_RE.fullmatch(raw_span.strip(".,:;|"))
        if not particulars or not span_match:
            continue
        particulars = re.sub(r"(?i)\b([PR])\s*[-–—]\s*(\d+)\b", r"\1-\2", particulars)
        rebuilt.append(f"{serial_match.group()}.\t{particulars}\t{span_match.group()}")
    return "\n".join(rebuilt)


def margin_folio_from_tsv(tsv: str) -> str | None:
    """Return a standalone top/bottom-margin folio from Tesseract geometry.

    Plain OCR puts a visible top folio near the beginning of the text, while
    the split range parser deliberately accepts only a final isolated line.
    Recover the geometry here and append the verified token after OCR.
    """
    try:
        rows = list(csv.DictReader(io.StringIO(tsv), delimiter="\t"))
    except (csv.Error, TypeError):
        return None
    page_row = next((row for row in rows if row.get("level") == "1"), None)
    if not page_row:
        return None
    try:
        page_width = int(page_row.get("width") or 0)
        page_height = int(page_row.get("height") or 0)
    except ValueError:
        return None
    if page_width <= 0 or page_height <= 0:
        return None

    lines: dict[tuple[str, str, str], list[dict[str, str]]] = {}
    for row in rows:
        text = (row.get("text") or "").strip()
        if row.get("level") != "5" or not text:
            continue
        key = (
            row.get("block_num") or "",
            row.get("par_num") or "",
            row.get("line_num") or "",
        )
        lines.setdefault(key, []).append(row)

    candidates: list[tuple[int, float, str]] = []
    for words in lines.values():
        if len(words) != 1:
            continue
        token = (words[0].get("text") or "").strip(" .,:;|_-")
        if not _FOLIO_TOKEN_RE.fullmatch(token):
            continue
        try:
            left = int(words[0].get("left") or 0)
            top = int(words[0].get("top") or 0)
            width = int(words[0].get("width") or 0)
            height = int(words[0].get("height") or 0)
        except ValueError:
            continue
        center_x = (left + width / 2) / page_width
        center_y = (top + height / 2) / page_height
        top_margin = center_y <= 0.11
        bottom_margin = center_y >= 0.91
        centered_or_right = 0.30 <= center_x <= 0.70 or center_x >= 0.78
        if not (centered_or_right and (top_margin or bottom_margin)):
            continue
        if token.isdigit() and 1900 <= int(token) <= 2099:
            continue
        # A one/two-letter folio is credible only at the top and near centre;
        # this excludes footer initials and short words in court headings.
        if token.isalpha() and not (top_margin and 0.38 <= center_x <= 0.62):
            continue
        edge_distance = center_y if top_margin else 1.0 - center_y
        # Compiled paper books normally add their outer folio at top-centre,
        # while a reproduced order may retain a different internal page
        # number at top-right. Prefer the centred compilation folio before
        # comparing distance from the edge.
        zone_priority = 0 if 0.30 <= center_x <= 0.70 else 1
        candidates.append((zone_priority, edge_distance, token.upper()))
    if not candidates:
        return None
    return min(candidates, key=lambda item: (item[0], item[1]))[2]


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
            except Exception:  # noqa: BLE001, S112 - malformed blocks fail open
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
            output_base = root / f"{number}-ocr"
            try:
                subprocess.run(
                    [
                        executable,
                        str(root / f"{number}.png"),
                        str(output_base),
                        "-l",
                        "eng",
                        "--psm",
                        "3",
                        "txt",
                        "tsv",
                    ],
                    capture_output=True,
                    text=True,
                    timeout=30,
                    env={**os.environ, "OMP_THREAD_LIMIT": "1"},
                    check=True,
                )
                text = output_base.with_suffix(".txt").read_text(
                    encoding="utf-8", errors="replace"
                )
                tsv = output_base.with_suffix(".tsv").read_text(
                    encoding="utf-8", errors="replace"
                )
                index_rows = index_table_rows_from_tsv(tsv)
                if index_rows:
                    text = text.rstrip() + "\n" + index_rows + "\n"
                folio = margin_folio_from_tsv(tsv)
                if folio and (text.rstrip().splitlines()[-1:] != [folio]):
                    text = text.rstrip() + "\n" + folio + "\n"
                return number, text
            except (subprocess.SubprocessError, OSError) as exc:
                logger.warning(
                    "Split OCR failed for page %d: %s", number, type(exc).__name__
                )
                return number, ""
            finally:
                (root / f"{number}.png").unlink(missing_ok=True)
                output_base.with_suffix(".txt").unlink(missing_ok=True)
                output_base.with_suffix(".tsv").unlink(missing_ok=True)

        workers = split_ocr_concurrency()
        with (
            pymupdf.open(stream=pdf_bytes, filetype="pdf") as document,
            ThreadPoolExecutor(max_workers=workers) as pool,
        ):
            # Batches bound rendered-image storage and outstanding work.
            batch_size = max(4, workers * 2)
            for offset in range(0, len(pages), batch_size):
                batch = pages[offset : offset + batch_size]
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
