"""Read split evidence from PDF geometry rather than text insertion order."""

from __future__ import annotations

import re

_INDEX_SERIAL_WORD_RE = re.compile(r"(?P<serial>\d{1,3})[.)]")
_INDEX_SPAN_CELL_RE = re.compile(
    r"(?:\d{1,4}[A-Za-z]?|[A-Za-z]{1,2}|A\d{1,2})"
    r"(?:\s*[-–—]\s*(?:\d{1,4}[A-Za-z]?|[A-Za-z]{1,2}|A\d{1,2}))?",
    re.IGNORECASE,
)
_ANNEXURE_CITED_RANGE_RE = re.compile(
    r"\bannexure\s*[-–—:]?\s*(?P<prefix>[a-z])\s*[-/–—]\s*"
    r"(?P<number>\d{1,4})\s*\(\s*(?:kindly|please)\s+see\s+pages?"
    r"[\s_]*(?P<start>\d{1,4})[\s_]*to[\s_]*(?P<end>\d{1,4})[\s_]*\)",
    re.IGNORECASE,
)


def annexure_cited_page_ranges(text: str) -> list[tuple[str, int, int]]:
    """Read explicit narrative page-range references, never document starts.

    Both numeric endpoints must appear within the complete parenthetical
    ``Annexure P-9 (Kindly see Pages 189 to 199)`` reference. Filled underline
    placeholders are allowed; detached page numbers and bare exhibit headings
    are not evidence. The caller must establish the citing document's owner
    before using these ranges to identify an outer annexure.
    """
    ranges: list[tuple[str, int, int]] = []
    for match in _ANNEXURE_CITED_RANGE_RE.finditer(text or ""):
        number = int(match.group("number"))
        start, end = int(match.group("start")), int(match.group("end"))
        if number < 1 or not 1 <= start <= end:
            continue
        label = f"Annexure {match.group('prefix').upper()}-{number}"
        reference = (label, start, end)
        if reference not in ranges:
            ranges.append(reference)
    return ranges


def _append_geometry_annexure_references(page: object, text: str) -> str:
    """Recover inline numbers inserted separately without reordering prose."""
    try:
        words = page.get_text("words", sort=True)  # type: ignore[attr-defined]
        sorted_text = " ".join(str(word[4]) for word in words)
    except Exception:  # noqa: BLE001 - optional geometry enrichment
        return text
    existing = set(annexure_cited_page_ranges(text))
    references = [
        f"Reference to {label} (Kindly see Pages {start} to {end})."
        for label, start, end in annexure_cited_page_ranges(sorted_text)
        if (label, start, end) not in existing
    ]
    if not references:
        return text
    separator = "" if text.endswith("\n") else "\n"
    return text + separator + "\n".join(references)


def _is_margin_folio_candidate(
    value: str,
    bbox: tuple[float, float, float, float],
    *,
    width: float,
    height: float,
) -> bool:
    """Return whether a compact token is physically printed in a page margin.

    Paper books commonly put lettered front-matter folios (B, P, AA) in the
    upper-right corner, while Annexure page references such as ``26`` occur at
    the bottom of List-of-Dates prose.  Text order alone cannot distinguish
    them, so only geometry-confirmed margin tokens are accepted.
    """
    token = value.strip()
    if not re.fullmatch(
        r"(?:\d{1,4}[A-Za-z]?|[A-Za-z]{1,2}|A\d{1,2})", token
    ) or re.fullmatch(r"(?:19|20)\d{2}", token):
        return False
    x0, y0, x1, y1 = (float(item) for item in bbox)
    center_x = (x0 + x1) / 2
    horizontally_centered = width * 0.30 <= center_x <= width * 0.70
    in_horizontal_corner = x1 <= width * 0.25 or x0 >= width * 0.75
    numeric_folio = bool(re.fullmatch(r"(?:\d{1,4}[A-Za-z]?|A\d{1,2})", token))
    return y0 > height * 0.92 or (
        y1 < height * 0.11
        and (horizontally_centered or in_horizontal_corner or numeric_folio)
    )


def _geometry_index_rows(page: object) -> list[str]:
    """Rebuild rows when a PDF emits an Index column-by-column.

    ``Page.find_tables()`` works for ruled tables whose vector lines survive
    compilation. Some SCI paper-books instead contain a visually intact
    table whose text layer is ordered as: serial column, particulars column,
    then every page-number cell. The plain text is therefore unreadable as
    rows even though word coordinates still identify the three columns.

    This fallback is deliberately restricted to a narrow serial column on an
    Index page. It never runs over arbitrary body tables or annexed records.
    """
    try:
        width = float(page.rect.width)  # type: ignore[attr-defined]
        words = list(page.get_text("words"))  # type: ignore[attr-defined]
    except Exception:  # noqa: BLE001 - optional geometry enrichment
        return []

    serials: list[tuple[float, int, str]] = []
    for word in words:
        value = str(word[4]).strip()
        match = _INDEX_SERIAL_WORD_RE.fullmatch(value)
        if not match or float(word[0]) > width * 0.22:
            continue
        serials.append((float(word[1]), int(match.group("serial")), value))
    serials.sort()
    if not serials:
        return []

    def line_text(cells: list[tuple]) -> list[str]:
        lines: list[list[tuple]] = []
        for word in sorted(cells, key=lambda item: (float(item[1]), float(item[0]))):
            if not lines or abs(float(word[1]) - float(lines[-1][0][1])) > 3.0:
                lines.append([word])
            else:
                lines[-1].append(word)
        return [
            " ".join(
                str(word[4]).strip()
                for word in sorted(line, key=lambda item: float(item[0]))
            )
            for line in lines
        ]

    rows: list[str] = []
    body_left = width * 0.18
    page_column_left = width * 0.46
    page_column_right = width * 0.75
    for index, (start_y, serial, _raw) in enumerate(serials):
        end_y = (
            serials[index + 1][0]
            if index + 1 < len(serials)
            else float(page.rect.height)  # type: ignore[attr-defined]
        )
        in_row = [
            word
            for word in words
            if start_y - 2.5 <= float(word[1]) < end_y - 2.5
        ]
        body_words = [
            word
            for word in in_row
            if body_left <= float(word[0]) < page_column_left
        ]
        body = " ".join(line_text(body_words)).strip()
        if not body:
            continue

        span = ""
        span_words = [
            word
            for word in in_row
            if page_column_left <= float(word[0]) < page_column_right
        ]
        for candidate in line_text(span_words):
            normalized = re.sub(
                r"\s*[-–—]\s*", "-", candidate.strip(" .,:;")
            )
            if _INDEX_SPAN_CELL_RE.fullmatch(normalized):
                span = normalized
                break
        rows.append(f"{serial}.\t{body}\t{span}")
    return rows


def letter_folio_number(value: str) -> int:
    """Sortable value for A..Z, AA..ZZ while preserving A == 65."""
    number = 0
    for char in value.strip().upper():
        if not ("A" <= char <= "Z"):
            raise ValueError(f"Invalid letter folio: {value!r}")
        number = number * 26 + ord(char) - ord("A") + 1
    return 64 + number


def extract_split_layout(
    pdf_bytes: bytes,
) -> dict[int, tuple[str, str | None, str | None]]:
    """Return native text, a verified Index table and an isolated margin folio.

    Only contiguous, serial-numbered tables following an Index heading are
    normalized. Arbitrary tables in annexed records remain ordinary text.
    No OCR or external service is invoked here.
    """
    try:
        import pymupdf
    except ImportError:
        return {}
    result = {}
    try:
        document = pymupdf.open(stream=pdf_bytes, filetype="pdf")
    except Exception:  # noqa: BLE001 - optional PDF enrichment must fail open
        return {}
    with document:
        index_active = False
        index_seen = False
        last_serial = 0
        for number in range(1, document.page_count + 1):
            try:
                page = document.load_page(number - 1)
                text = page.get_text("text") or ""
                folios = set()
                for block in page.get_text("dict")["blocks"]:
                    for line in block.get("lines", []):
                        value = "".join(span["text"] for span in line["spans"]).strip()
                        _x0, y0, _x1, y1 = line["bbox"]
                        if not _is_margin_folio_candidate(
                            value,
                            (float(_x0), float(y0), float(_x1), float(y1)),
                            width=float(page.rect.width),
                            height=float(page.rect.height),
                        ):
                            continue
                        folios.add(value)
                folio = next(iter(folios)) if len(folios) == 1 else None
                is_heading = bool(
                    re.search(r"(?mi)^\s*(?:master\s+)?index\s*$", text)
                ) and bool(
                    re.search(r"particulars|page\s+no", text, re.IGNORECASE)
                )
                table_text = None
                volume_index = bool(
                    is_heading
                    and re.search(
                        r"volume\s*[-–—]?\s*(?:i{1,3}|[1-3])\b",
                        text,
                        re.IGNORECASE,
                    )
                )
                if (is_heading and (not index_seen or volume_index)) or index_active:
                    rows = []
                    continuation = None
                    for table in page.find_tables().tables:
                        for cells in table.extract():
                            if len(cells) < 3:
                                continue
                            serial = " ".join((cells[0] or "").split())
                            body = " ".join((cells[1] or "").split())
                            # Some SCI Indexes place Filing Memo/Vakalatnama
                            # folios in the Part-II (fourth) column while the
                            # normal Part-I page column is blank.
                            page_cells = [
                                " ".join((cell or "").split()) for cell in cells[2:4]
                            ]
                            span = next((cell for cell in page_cells if cell), "")
                            if re.fullmatch(r"\d{1,3}[.)]", serial) and body:
                                rows.append(f"{serial}\t{body}\t{span}")
                            elif not serial and body and rows == []:
                                # Continuation of the preceding page's last row.
                                prior = result.get(number - 1)
                                if index_active and prior and prior[1]:
                                    lines = prior[1].splitlines()
                                    previous = lines[-1].split("\t")
                                    if len(previous) == 3:
                                        previous[1] += " " + body
                                        previous[2] = previous[2] or span
                                        lines[-1] = "\t".join(previous)
                                        continuation = (
                                            prior[0],
                                            "\n".join(lines),
                                            prior[2],
                                        )
                    geometry_rows = _geometry_index_rows(page)
                    # Prefer the coordinate reconstruction when table-line
                    # detection failed or recovered fewer serial-numbered
                    # rows. Both sources are confined to the active outer
                    # Index, so this does not reinterpret annexed tables.
                    if len(geometry_rows) > len(rows):
                        rows = geometry_rows
                    serials = [int(row.split("\t")[0][:-1]) for row in rows]
                    ordered = serials == sorted(set(serials))
                    follows = bool(serials and serials[0] > last_serial)
                    if ordered and (
                        (not index_active and len(rows) >= 2)
                        or (index_active and (follows or not rows and continuation))
                    ):
                        table_text = "INDEX\nS.No. Particulars Page No.\n" + "\n".join(
                            rows
                        )
                        if serials:
                            last_serial = serials[-1]
                        if continuation:
                            result[number - 1] = continuation
                index_active = table_text is not None
                index_seen = index_seen or index_active
                if table_text is None and not re.search(
                    r"(?mi)^\s*(?:master\s+)?index\s*$", text
                ):
                    text = _append_geometry_annexure_references(page, text)
                result[number] = (text, table_text, folio)
            except Exception:  # noqa: BLE001 - optional PDF enrichment must fail open
                # Unsupported/damaged layout must not prevent text splitting.
                index_active = False
    return result


_FOLIO_NUM_RE = re.compile(r"^(?P<n>\d{1,4})(?P<suffix>[A-Za-z])?$")
_FOLIO_LETTER_RE = re.compile(r"^[A-Za-z]{1,2}$")


def printed_folio(text: str) -> tuple[str, int, str] | None:
    """Corner folio such as ``36``, ``63B``, or ``B``.

    Only the bottom line counts. A page number cited in the paragraph above
    it (``36`` then ``37 to 48``) is not the folio of this sheet.
    """
    lines = [line.strip() for line in (text or "").splitlines() if line.strip()]
    # Digitally signed judgments can obscure the corner folio with a stamp,
    # while retaining an internal footer such as ``Page 1 of 9``. Prefer a
    # clean isolated outer folio below and use this only as a fallback.
    page_counter = None
    for line in reversed(lines[-18:]):
        match = re.search(
            r"\b(?:page|pg\.?)\s*(?:no\.?\s*)?(\d{1,4})\s*(?:of|/)\s*\d{1,4}\b",
            line,
            re.IGNORECASE,
        )
        if match:
            page_counter = ("number", int(match.group(1)), "")
            break
    skipped_word = False
    for line in reversed(lines):
        if re.fullmatch(r"[\W_]+", line):
            continue
        prefixed = re.fullmatch(r"A(\d{1,2})", line, re.IGNORECASE)
        if prefixed:
            return ("prefixed", int(prefixed.group(1)), "")
        numbered = _FOLIO_NUM_RE.fullmatch(line)
        if numbered:
            number = int(numbered.group("n"))
            if 1900 <= number <= 2099:
                break
            return ("number", number, (numbered.group("suffix") or "").upper())
        if _FOLIO_LETTER_RE.fullmatch(line):
            return ("letter", letter_folio_number(line), "")
        # "61" then "December": the folio sits just above a one-word signature.
        if not skipped_word and re.fullmatch(r"[A-Za-z]{3,}", line):
            skipped_word = True
            continue
        break
    return page_counter
