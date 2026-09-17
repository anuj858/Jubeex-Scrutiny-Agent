"""Stored ink marks and the vision-model response shapes."""

from __future__ import annotations

from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field

PROMPT_VERSION = "visual-v10-legally-relevant-evidence"
VISUAL_SCHEMA = "visual_index_v1"
VisualStatus = Literal["ok", "skipped", "error"]

SUPPORTED_TYPES = frozenset(
    {
        "notary_seal",
        "government_stamp",
        "revenue_stamp",
        "signature_like_mark",
        "handwritten_field_value",
        "ordinary_seal_or_stamp",
        "table",
        "image",
        "figure",
        "logo",
        "watermark",
        "unknown_relevant_visual",
    }
)
TYPE_ALIASES = {
    "seal": "ordinary_seal_or_stamp",
    "notary_stamp": "notary_seal",
    "notarial_seal": "notary_seal",
    "government_seal": "government_stamp",
    "stamp": "ordinary_seal_or_stamp",
    "revenue_seal": "revenue_stamp",
    "signature": "signature_like_mark",
    "signed_mark": "signature_like_mark",
    "handwriting": "handwritten_field_value",
    "photo": "image",
    "photograph": "image",
    "diagram": "figure",
}
SIGNATURE_ROLES = frozenset(
    {
        "advocate",
        "deponent",
        "petitioner",
        "respondent",
        "unknown",
        "not_applicable",
    }
)
PageSelector = Literal["all", "last"]
D093_SPLIT_TYPES = (
    "Advocate's Checklist",
    "Listing Proforma",
    "Main Petition",
    "AOR's Certificate",
    "Application",
    "Filing Memo",
    "Memo of Parties",
    "Vakalatnama",
    "Memo of Appearance",
    "Annexures",
)

_VISUAL_INDEX_EXAMPLE = {
    "schema": VISUAL_SCHEMA,
    "prompt_version": PROMPT_VERSION,
    "status": "ok",
    "error": None,
    "pages": [20],
    "targets": [{"page": 20, "document_types": ["Main Petition"]}],
    "failures": [],
    "marks": [
        {
            "page": 20,
            "document_type": "Main Petition",
            "marking_type": "signature_like_mark",
            "signature_role": "advocate",
            "bbox": {"x": 0.62, "y": 0.81, "w": 0.28, "h": 0.08},
            "confidence": 0.86,
        }
    ],
}


class NormalizedBBox(BaseModel):
    """Page-relative box. ``x``/``y``/``width``/``height`` are 0–1."""

    model_config = ConfigDict(extra="forbid")

    x: float
    y: float
    width: float
    height: float


class VisualMark(BaseModel):
    """One detected ink mark on a global filing page."""

    page: int
    document_type: str
    marking_type: str
    signature_role: str = "not_applicable"
    bbox: dict[str, float]
    confidence: float = 0.5
    visible_text: str = ""
    associated_label: str = ""
    slot_id: str | None = None
    local_page: int | None = None


class VisualPageTarget(BaseModel):
    """One formality page to render and send to vision."""

    page: int
    slot_id: str
    local_page: int
    document_types: list[str] = Field(
        default_factory=list,
        description=(
            "Split labels on this page. D093 last-page types: "
            + ", ".join(D093_SPLIT_TYPES)
            + ". Affidavit last is included for notary. Absent slots are omitted."
        ),
    )
    file_url: str | None = None
    file_id: str | None = None
    markdown: str = ""


class VisualIndex(BaseModel):
    schema_name: str = Field(default=VISUAL_SCHEMA, alias="schema")
    prompt_version: str = PROMPT_VERSION
    status: VisualStatus = "ok"
    error: str | None = None
    pages: list[int] = Field(
        default_factory=list,
        description="Unique target page numbers actually sent to vision.",
    )
    targets: list[dict[str, Any]] = Field(
        default_factory=list,
        description=(
            "Inventory of pages sent. Missing D093 types are omitted, "
            "never listed as null."
        ),
    )
    failures: list[dict[str, Any]] = Field(
        default_factory=list,
        description=(
            "Per-page load/render/vision errors. Absent when every page succeeded."
        ),
    )
    marks: list[VisualMark] = Field(default_factory=list)

    model_config = ConfigDict(
        populate_by_name=True,
        json_schema_extra={"example": _VISUAL_INDEX_EXAMPLE},
    )


def coerce_visual_status(status: str | None) -> VisualStatus:
    if status in {"ok", "skipped", "error"}:
        return status
    if status in {"missing", None, ""}:
        return "skipped" if status == "missing" else "ok"
    return "error"


def empty_visual_index(
    *, status: str = "skipped", error: str | None = None
) -> dict[str, Any]:
    resolved = coerce_visual_status(status)
    if status == "missing" and error is None:
        error = "missing"
    return {
        "schema": VISUAL_SCHEMA,
        "prompt_version": PROMPT_VERSION,
        "status": resolved,
        "error": error,
        "pages": [],
        "targets": [],
        "failures": [],
        "marks": [],
    }
