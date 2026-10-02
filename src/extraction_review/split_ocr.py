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
import time
from concurrent.futures import ThreadPoolExecutor
from itertools import pairwise
from pathlib import Path

logger = logging.getLogger(__name__)

_FOLIO_TOKEN_RE = re.compile(r"(?:\d{1,4}[A-Za-z]?|A\d{1,2}|[A-Z]{1,2})")
_INDEX_PAGE_SPAN_RE = re.compile(
    r"(?:\d{1,4}[A-Za-z]?(?:[-–—]\d{1,4}[A-Za-z]?)?|"
    r"A\d{1,2}(?:[-–—]A?-?\d{1,2})?|[A-Z]{1,2}(?:[-–—][A-Z]{1,2})?)"
)
_ANNEXURE_MARGIN_TOKEN_RE = re.compile(
    r"^(?P<series>[PRE])[-/]?(?P<number>\d{1,3})$", re.IGNORECASE
)
_LOWER_COURT_START_RE = re.compile(
    r"\b(?:high\s+court|district\s+court|national\s+company\s+law\s+"
    r"(?:appellate\s+)?tribunal|nclat|nclt|consumer\s+(?:commission|forum)|"
    r"court\s+of\s+the\s+\w+)\b",
    re.IGNORECASE,
)

DEFAULT_SPLIT_OCR_WORKERS = 2
DEFAULT_SPLIT_OCR_DPI = 180


def _needs_table_ocr_retry(text: str) -> bool:
    """Retry sparse Index/table OCR with a column-aware page mode.

    Automatic segmentation (PSM 3) can see ``INDEX`` and ``Pages`` while
    dropping the wide middle header cell entirely.  That cell identifies the
    blank Record of Proceedings table, so retry only this narrow sparse-table
    shape rather than doubling OCR work for every scanned page.
    """
    folded = re.sub(r"\s+", " ", text or "").strip().casefold()
    readable = len(re.sub(r"[^a-z0-9]+", "", folded))
    return readable < 80 and "index" in folded and "pages" in folded


def _positive_int_env(name: str, default: int, *, maximum: int) -> int:
    raw = (os.getenv(name) or "").strip()
    try:
        value = int(raw) if raw else default
    except ValueError:
        value = default
    return min(max(value, 1), maximum)


def split_ocr_workers() -> int:
    """Number of single-threaded Tesseract processes to run concurrently."""
    return _positive_int_env(
        "SPLIT_OCR_WORKERS", DEFAULT_SPLIT_OCR_WORKERS, maximum=16
    )


def split_ocr_dpi() -> int:
    """Render resolution used by split-only OCR."""
    return _positive_int_env("SPLIT_OCR_DPI", DEFAULT_SPLIT_OCR_DPI, maximum=300)


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

    candidates: list[tuple[float, str]] = []
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
        candidates.append((edge_distance, token.upper()))
    if not candidates:
        return None
    return min(candidates, key=lambda item: item[0])[1]


def annexure_stamp_from_margin_tsv(tsv: str) -> str | None:
    """Recover an isolated handwritten ``P2``/``P-2`` above a court title.

    Tesseract often reads the small handwritten ``Annx.`` prefix as arbitrary
    text while still recognizing its P/R/E number.  Accept that number only
    from a short line above an ``IN THE ... COURT/TRIBUNAL`` caption.  This
    geometry guard prevents petition references and ordinary body text from
    being promoted to outer paper-book Annexure stamps.
    """
    try:
        rows = list(csv.DictReader(io.StringIO(tsv), delimiter="\t"))
    except (csv.Error, TypeError):
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

    ordered: list[tuple[int, list[dict[str, str]]]] = []
    for words in lines.values():
        try:
            top = min(int(word.get("top") or 0) for word in words)
        except ValueError:
            continue
        ordered.append(
            (top, sorted(words, key=lambda word: int(word.get("left") or 0)))
        )
    ordered.sort(key=lambda item: item[0])

    caption_top: int | None = None
    for top, words in ordered:
        line = " ".join((word.get("text") or "").strip() for word in words)
        folded = re.sub(r"\s+", " ", line).casefold()
        if "in the" in folded and ("court" in folded or "tribunal" in folded):
            caption_top = top
            break
    if caption_top is None:
        return None

    for top, words in ordered:
        if top >= caption_top or len(words) > 4:
            continue
        tokens = [(word.get("text") or "").strip(" .,:;|_()[]{}") for word in words]
        line = " ".join(tokens).casefold()
        if re.search(r"\b(?:page|part|petition|para|item|form|case|no)\b", line):
            continue
        for token in tokens:
            match = _ANNEXURE_MARGIN_TOKEN_RE.fullmatch(token)
            if not match:
                continue
            series = match.group("series").upper()
            number = int(match.group("number"))
            if number >= 1:
                return f"ANNEXURE {series}-{number}"
    return None


def _needs_annexure_margin_retry(text: str) -> bool:
    """Use expensive handwriting recovery only on plausible record starts."""
    folded = re.sub(r"\s+", " ", text or "").strip()
    return bool(
        folded
        and not re.search(r"\b(?:annexure|annx\.?)\b", folded, re.IGNORECASE)
        and _LOWER_COURT_START_RE.search(folded[:1600])
    )


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

    started = time.perf_counter()
    workers = split_ocr_workers()
    dpi = split_ocr_dpi()

    # Render on one thread: PyMuPDF documents must not be shared across threads.
    # Only subprocess OCR runs concurrently, with one OpenMP thread per process.
    with tempfile.TemporaryDirectory(prefix="split-ocr-") as directory:
        root = Path(directory).resolve()
        results: dict[int, str] = {}

        def recognize(number: int) -> tuple[int, str]:
            output_base = root / f"{number}-ocr"
            output_bases = [output_base]

            def run_tesseract(
                base: Path, *, psm: int, image_path: Path | None = None
            ) -> tuple[str, str]:
                subprocess.run(
                    [
                        executable,
                        str(image_path or root / f"{number}.png"),
                        str(base),
                        "-l",
                        "eng",
                        "--psm",
                        str(psm),
                        "txt",
                        "tsv",
                    ],
                    capture_output=True,
                    text=True,
                    timeout=30,
                    env={**os.environ, "OMP_THREAD_LIMIT": "1"},
                    check=True,
                )
                return (
                    base.with_suffix(".txt").read_text(
                        encoding="utf-8", errors="replace"
                    ),
                    base.with_suffix(".tsv").read_text(
                        encoding="utf-8", errors="replace"
                    ),
                )

            try:
                text, tsv = run_tesseract(output_base, psm=3)
                if _needs_annexure_margin_retry(text):
                    margin_image = root / f"{number}-margin.png"
                    margin_base = root / f"{number}-margin-ocr"
                    output_bases.append(margin_base)
                    try:
                        # Re-render only a plausible lower-court record start
                        # at higher resolution. PSM 3 misses faint handwritten
                        # paper-book stamps such as ``Annx. P2`` even when it
                        # reads the body perfectly. Keep the whole page for
                        # PSM 11 because its layout context materially improves
                        # recognition of the small marginal number.
                        with pymupdf.open(stream=pdf_bytes, filetype="pdf") as source:
                            page = source[number - 1]
                            page.get_pixmap(
                                dpi=max(300, dpi),
                                colorspace=pymupdf.csGRAY,
                            ).save(margin_image)
                        _margin_text, margin_tsv = run_tesseract(
                            margin_base, psm=11, image_path=margin_image
                        )
                        margin_stamp = annexure_stamp_from_margin_tsv(margin_tsv)
                        if margin_stamp:
                            text = margin_stamp + "\n" + text.lstrip()
                    except (subprocess.SubprocessError, OSError, RuntimeError):
                        # The ordinary OCR result remains usable when the
                        # optional handwriting pass cannot be completed.
                        pass
                    finally:
                        margin_image.unlink(missing_ok=True)
                if _needs_table_ocr_retry(text):
                    table_base = root / f"{number}-table-ocr"
                    output_bases.append(table_base)
                    try:
                        table_text, table_tsv = run_tesseract(table_base, psm=4)
                    except (subprocess.SubprocessError, OSError):
                        pass
                    else:
                        primary_score = len(re.sub(r"[^A-Za-z0-9]+", "", text))
                        table_score = len(
                            re.sub(r"[^A-Za-z0-9]+", "", table_text)
                        )
                        if table_score > primary_score:
                            text, tsv = table_text, table_tsv
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
                for base in output_bases:
                    base.with_suffix(".txt").unlink(missing_ok=True)
                    base.with_suffix(".tsv").unlink(missing_ok=True)

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
                    page.get_pixmap(dpi=dpi, colorspace=pymupdf.csGRAY).save(
                        root / f"{number}.png"
                    )
                results.update(pool.map(recognize, batch))
        logger.info(
            "Split OCR read %d/%d sparse pages in %.1fs "
            "(workers=%d dpi=%d)",
            sum(bool(t.strip()) for t in results.values()),
            len(pages),
            time.perf_counter() - started,
            workers,
            dpi,
        )
        return results
