"""Page-image ink detection for signatures, notary seals, and court-fee stamps.

LlamaParse stores OCR words. This package renders selected formality pages and
asks a vision model for tight boxes, then scrutiny attaches those marks onto
existing catalogue defects.
"""

from .attach import attach_visual_localizations, visual_needs_for_defect
from .detect import detect_visual_marks, visual_detection_enabled
from .pages import select_formality_pages
from .schema import VisualIndex, VisualMark
from .store import (
    VISUAL_ARTIFACT_KEY_KEY,
    VISUAL_ARTIFACT_URL_KEY,
    VISUAL_SUMMARY_KEY,
    coerce_visual_index,
    dump_visual_index,
    load_visual_index,
    visual_summary,
)

__all__ = [
    "VISUAL_ARTIFACT_KEY_KEY",
    "VISUAL_ARTIFACT_URL_KEY",
    "VISUAL_SUMMARY_KEY",
    "VisualIndex",
    "VisualMark",
    "attach_visual_localizations",
    "coerce_visual_index",
    "detect_visual_marks",
    "dump_visual_index",
    "load_visual_index",
    "select_formality_pages",
    "visual_detection_enabled",
    "visual_needs_for_defect",
    "visual_summary",
]
