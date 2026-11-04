"""Render a PDF page to a JPEG preview for the vision model."""

from __future__ import annotations

DEFAULT_PREVIEW_DPI = 160


def preview_dpi(raw: str | None = None) -> int:
    text = (raw or "").strip()
    try:
        value = int(text) if text else DEFAULT_PREVIEW_DPI
    except ValueError:
        value = DEFAULT_PREVIEW_DPI
    return min(240, max(72, value))


def render_page_jpeg(
    pdf_bytes: bytes,
    local_page: int,
    *,
    dpi: int = DEFAULT_PREVIEW_DPI,
) -> bytes:
    """Rasterize one 1-indexed local PDF page. Requires PyMuPDF."""
    import pymupdf

    document = pymupdf.open(stream=pdf_bytes, filetype="pdf")
    try:
        index = local_page - 1
        if index < 0 or index >= document.page_count:
            raise ValueError(
                f"local page {local_page} is outside 1..{document.page_count}"
            )
        page = document[index]
        pixmap = page.get_pixmap(dpi=dpi, colorspace=pymupdf.csRGB, alpha=False)
        try:
            return pixmap.tobytes("jpeg", jpg_quality=82)
        except TypeError:
            return pixmap.tobytes("jpeg")
    finally:
        document.close()


def jpeg_data_url(image_bytes: bytes) -> str:
    import base64

    encoded = base64.b64encode(image_bytes).decode("ascii")
    media = (
        "image/jpeg"
        if image_bytes.startswith(b"\xff\xd8\xff")
        else "image/webp"
        if image_bytes.startswith(b"RIFF")
        else "image/png"
    )
    return f"data:{media};base64,{encoded}"
