"""Read split evidence from PDF geometry rather than text insertion order."""

from __future__ import annotations

import re


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
                        if not (
                            y1 < page.rect.height * 0.08 or y0 > page.rect.height * 0.92
                        ):
                            continue
                        if re.fullmatch(
                            r"(?:\d{1,4}[A-Za-z]?|[A-Za-z]|A\d{1,2})", value
                        ) and not re.fullmatch(r"(?:19|20)\d{2}", value):
                            folios.add(value)
                folio = next(iter(folios)) if len(folios) == 1 else None
                is_heading = bool(re.search(r"(?mi)^\s*index\s*$", text)) and bool(
                    re.search(r"particulars|page\s+no", text, re.IGNORECASE)
                )
                table_text = None
                if (is_heading and not index_seen) or index_active:
                    rows = []
                    continuation = None
                    for table in page.find_tables().tables:
                        for cells in table.extract():
                            if len(cells) < 3:
                                continue
                            serial, body, span = (
                                " ".join((cell or "").split()) for cell in cells[:3]
                            )
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
                result[number] = (text, table_text, folio)
            except Exception:  # noqa: BLE001 - optional PDF enrichment must fail open
                # Unsupported/damaged layout must not prevent text splitting.
                index_active = False
    return result


_FOLIO_NUM_RE = re.compile(r"^(?P<n>\d{1,4})(?P<suffix>[A-Za-z])?$")
_FOLIO_LETTER_RE = re.compile(r"^[A-Za-z]$")


def printed_folio(text: str) -> tuple[str, int, str] | None:
    """Corner folio such as ``36``, ``63B``, or ``B``.

    Only the bottom line counts. A page number cited in the paragraph above
    it (``36`` then ``37 to 48``) is not the folio of this sheet.
    """
    lines = [line.strip() for line in (text or "").splitlines() if line.strip()]
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
                return None
            return ("number", number, (numbered.group("suffix") or "").upper())
        if _FOLIO_LETTER_RE.fullmatch(line):
            return ("letter", ord(line.upper()), "")
        # "61" then "December": the folio sits just above a one-word signature.
        if not skipped_word and re.fullmatch(r"[A-Za-z]{3,}", line):
            skipped_word = True
            continue
        return None
    return None
