"""The `scrutiny_finding_v1` response contract.

`DefectResponse` is the narrow shape the model must return for a single defect.
Catalogue fields (defect, requirement, how to cure, rule, source) are copied
onto the finding in Python so they stay aligned with the defect API payload.
"""

from __future__ import annotations

import logging
import os
import re
from collections import Counter
from datetime import UTC, datetime
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field, ValidationError, model_validator

from ..document_parts import (
    _part_match,
    allows_index_evidence,
    chunks_cover_part,
    missing_required_parts,
    parts_named_in_where_to_look,
    parts_on_page,
    preferred_parts_for_defect,
    required_parts_for_defect,
)
from ..layout_index import (
    boxes_on_pages,
    page_entry,
    part_for_page,
    quote_body,
    quote_in_text,
)
from .prompts import (
    compose_reasoning,
    location_for_finding,
    validated_summary,
)
from .rules import (
    Catalogue,
    CatalogueSource,
    Defect,
    get_catalogue,
    source_match_tokens,
)

logger = logging.getLogger(__name__)

ResultState = Literal[
    "defect_found",
    "compliant",
    "not_applicable",
    "not_determined",
    "needs_review",
    "rule_verification_pending",
]

ModelResultState = Literal[
    "defect_found",
    "compliant",
    "not_applicable",
    "not_determined",
    "needs_review",
]

BoxesStatus = Literal["matched", "page_only", "unavailable"]


class EvidenceRef(BaseModel):
    """A pointer back into the source document for one observation.

    This is the model-facing shape. Do not add coordinates here — the LLM
    schema is derived from DefectResponse. The model returns the excerpt
    id and a quote. The server adds the document page and boxes later.
    """

    model_config = ConfigDict(extra="forbid")

    chunk_id: str | None = Field(
        default=None,
        description=(
            "Excerpt label copied exactly from the header, such as c1. "
            "Null when the quote is not from an excerpt. Do not invent an id."
        ),
    )
    page: int | None = Field(
        default=None,
        description=(
            "Leave null. The document page is taken from chunk_id. "
            "Never copy a page number from Authority or location_source."
        ),
    )
    quote: str = Field(description="Verbatim excerpt supporting the finding")


class BoundingBox(BaseModel):
    """One highlight strip on a PDF page. Coordinates are normalized 0–1."""

    model_config = ConfigDict(extra="forbid")

    page: int
    x: float
    y: float
    w: float
    h: float


class FindingEvidence(BaseModel):
    """Evidence on a server-assembled finding, including highlight boxes."""

    model_config = ConfigDict(extra="forbid")

    page: int | None = None
    quote: str
    bounding_boxes: list[BoundingBox] = Field(default_factory=list)
    boxes_status: BoxesStatus = "unavailable"
    document_part: str | None = None
    slot_id: str | None = None


class VisualLocalization(BaseModel):
    """Ink-mark box attached to a visual catalogue defect after detection."""

    model_config = ConfigDict(
        json_schema_extra={
            "example": {
                "page": 20,
                "document_type": "Main Petition",
                "marking_type": "advocate_on_record_signature",
                "signature_role": "advocate",
                "bounding_boxes": [
                    {"page": 20, "x": 0.62, "y": 0.81, "w": 0.28, "h": 0.08}
                ],
                "boxes_status": "matched",
                "confidence": 0.86,
            }
        }
    )

    page: int | None = None
    document_type: str | None = None
    marking_type: str = Field(
        description=(
            "Attached values: advocate_on_record_signature, "
            "executant_signature, or notary_seal."
        )
    )
    signature_role: str | None = None
    bounding_boxes: list[BoundingBox] = Field(default_factory=list)
    boxes_status: BoxesStatus = "unavailable"
    confidence: float | None = None


ElementState = Literal["present", "missing", "incomplete", "not_visible"]

GapKind = Literal[
    "part_not_in_excerpts",
    "element_page_not_in_excerpts",
    "unreadable",
    "ambiguous",
    "conflict",
    "low_confidence",
    "legal_rule_unverified",
]


class ElementFinding(BaseModel):
    """One element of this row, as the Note for AI lists them."""

    model_config = ConfigDict(extra="forbid")

    name: str = Field(
        description=(
            "The element named in the Note for AI or the Requirement, "
            "such as paragraph 1A or the English text of a cited provision."
        )
    )
    state: ElementState
    detail: str = Field(
        default="",
        description=(
            "The specific item, such as the provision that is missing. "
            "Empty when there is nothing to add."
        ),
    )


class DefectResponse(BaseModel):
    """Exactly what the model returns for one defect."""

    model_config = ConfigDict(extra="forbid")

    check_id: str
    # Governance-only states are assigned after model validation and must not
    # be offered to the model as filing-classification choices.
    status: ModelResultState
    confidence: float = Field(ge=0.0, le=1.0)
    summary: str = Field(
        description=(
            "One plain sentence about this filing and this one defect. "
            "Do not copy the Standard paragraph or mention rulebook pages."
        )
    )
    observed: str = Field(
        default="",
        description=(
            "One or two sentences on what this filing shows for this row. "
            "Name the document and the clause. Do not mention page numbers, "
            "excerpt ids, chunks, search, retrieval, or OCR."
        ),
    )
    elements: list[ElementFinding] = Field(
        default_factory=list,
        description=(
            "One item for each element the Note for AI or the Requirement "
            "names, in the same order. Element names must omit source-list "
            "labels such as 'a.', 'b.', '(a)', or '(b)'."
        ),
    )
    gap: GapKind | None = Field(
        default=None,
        description=(
            "Why the row is needs_review. Null for every other status. "
            "part_not_in_excerpts: the document type was not shown. "
            "element_page_not_in_excerpts: the document was shown but not "
            "the clause this row checks. unreadable: the clause cannot be "
            "read. ambiguous: the wording does not decide the row. "
            "conflict: the text and a visual mark disagree."
        ),
    )
    gap_detail: str = Field(
        default="",
        description=(
            "The missing element, or the visible mark that conflicts. "
            "Empty when gap is null."
        ),
    )
    conclusion: str = Field(
        default="",
        description=(
            "One sentence. Cite a rule only when that citation is written "
            "in the Requirement."
        ),
    )
    evidence: list[EvidenceRef]

    @property
    def reasoning(self) -> str:
        """Backward-compatible view of the structured observation/conclusion."""
        return " ".join(
            part for part in (self.observed, self.conclusion) if part
        ).strip()

    @model_validator(mode="before")
    @classmethod
    def _legacy_reasoning(cls, data: Any) -> Any:
        """Older callers passed a free-text reasoning string."""
        if isinstance(data, dict) and "reasoning" in data:
            data = dict(data)
            legacy = data.pop("reasoning")
            if not str(data.get("observed") or "").strip() and isinstance(legacy, str):
                data["observed"] = legacy
        return data


class Coverage(BaseModel):
    """How much of the document backed this finding."""

    model_config = ConfigDict(extra="forbid")

    chunks_reviewed: int = 0
    pages_reviewed: list[int] = Field(default_factory=list)
    structured_record_available: bool = False
    document_found: bool = False
    required_scope_found: bool = False
    scope_complete: bool = False
    prompt_truncated: bool = False
    evidence_complete: bool = False
    visual_required: bool = False
    visual_complete: bool = False
    visual_pages_reviewed: list[int] = Field(default_factory=list)


class LlmUsage(BaseModel):
    """Spend copied from OpenRouter `usage` on /chat/completions — never from the model JSON."""

    prompt_tokens: int = 0
    completion_tokens: int = 0
    total_tokens: int = 0
    cached_tokens: int = 0
    reasoning_tokens: int = 0
    calls: int = 0
    cost_usd: float | None = None
    model: str | None = None
    generation_id: str | None = None
    generation_ids: list[str] = Field(default_factory=list)

    def plus(self, other: LlmUsage) -> LlmUsage:
        cost: float | None = None
        if self.cost_usd is not None or other.cost_usd is not None:
            cost = (self.cost_usd or 0.0) + (other.cost_usd or 0.0)
        total = self.total_tokens + other.total_tokens
        if not total:
            total = (
                self.prompt_tokens
                + other.prompt_tokens
                + self.completion_tokens
                + other.completion_tokens
            )
        ids = [*(self.generation_ids or []), *(other.generation_ids or [])]
        last = other.generation_id or self.generation_id
        if last and last not in ids:
            ids.append(last)
        return LlmUsage(
            prompt_tokens=self.prompt_tokens + other.prompt_tokens,
            completion_tokens=self.completion_tokens + other.completion_tokens,
            total_tokens=total,
            cached_tokens=self.cached_tokens + other.cached_tokens,
            reasoning_tokens=self.reasoning_tokens + other.reasoning_tokens,
            calls=self.calls + other.calls,
            cost_usd=cost,
            model=other.model or self.model,
            generation_id=last,
            generation_ids=ids,
        )


class UsageByCheck(BaseModel):
    """One row in the cost breakdown, sorted highest charge first."""

    check_id: str
    cost_usd: float | None = None
    prompt_tokens: int = 0
    completion_tokens: int = 0
    total_tokens: int = 0
    llm_calls: int = 0
    share: float | None = None


class UsageSummary(BaseModel):
    """Roll-up of OpenRouter spend for the whole defect check."""

    cost_usd: float | None = None
    prompt_tokens: int = 0
    completion_tokens: int = 0
    total_tokens: int = 0
    cached_tokens: int = 0
    reasoning_tokens: int = 0
    llm_calls: int = 0
    model: str | None = None
    highest_cost_check_id: str | None = None
    highest_cost_usd: float | None = None
    by_check: list[UsageByCheck] = Field(default_factory=list)
    note: str = (
        "Exact OpenRouter usage.cost from each /chat/completions reply, "
        "summed for this check. The model is not asked for cost. LlamaParse, "
        "extract, classify, and Pinecone are billed separately."
    )


class SourceLocation(BaseModel):
    """Official rulebook for this defect: its link and the page in that document."""

    model_config = ConfigDict(extra="forbid")

    url: str | None = None
    page: int | None = None


_SOURCE_PAGE = re.compile(r"\bpages?\s+(\d+)\b", re.IGNORECASE)


def _source_page_number(text: str) -> int | None:
    match = _SOURCE_PAGE.search(text or "")
    if match is None:
        return None
    try:
        return int(match.group(1))
    except ValueError:
        return None


def _sources_named_in(
    text: str, sources: list[CatalogueSource]
) -> list[CatalogueSource]:
    folded = text.casefold()
    cited: list[CatalogueSource] = []
    for source in sources:
        if source.source_id and source.source_id in text:
            cited.append(source)
            continue
        if any(url and url in text for url in source.urls()):
            cited.append(source)
            continue
        if any(token.casefold() in folded for token in source_match_tokens(source)):
            cited.append(source)
    return cited


def official_source_locations(
    defect: Defect, catalogue: Catalogue | None
) -> list[SourceLocation]:
    """Link and page of each official source named on this defect."""
    raw = (defect.location_text or defect.location_source or "").strip()
    if not raw or "did not give" in raw.lower():
        return []
    sources = list(catalogue.sources) if catalogue is not None else []
    found: list[SourceLocation] = []
    seen: set[tuple[str | None, int | None]] = set()
    lines = [line.strip() for line in re.split(r"[\n;]+", raw) if line.strip()]
    for line in lines:
        page = _source_page_number(line)
        matched = _sources_named_in(line, sources)
        if not matched:
            if page is None:
                continue
            key = (None, page)
            if key not in seen:
                seen.add(key)
                found.append(SourceLocation(page=page))
            continue
        for source in matched:
            key = (source.url, page)
            if key in seen:
                continue
            seen.add(key)
            found.append(SourceLocation(url=source.url, page=page))
    return found


class HowToCure(BaseModel):
    """Sentences and buttons copied from the catalogue row for this check."""

    steps: list[str] = Field(default_factory=list)
    mode: str = "any"
    actions: list[Any] = Field(default_factory=list)


def cure_from_defect(defect: Defect) -> HowToCure:
    """Copy how_to_cure and remediation_config. Do not rewrite stored buttons."""
    config = (
        defect.remediation_config if isinstance(defect.remediation_config, dict) else {}
    )
    mode = config.get("mode")
    if not isinstance(mode, str) or not mode.strip():
        mode = "any"
    actions = config.get("actions")
    if not isinstance(actions, list):
        actions = []
    return HowToCure(
        steps=list(defect.how_to_cure or []), mode=mode, actions=list(actions)
    )


class DefectFinding(BaseModel):
    """Server-assembled finding for one catalogue defect."""

    check_id: str
    defect: str = Field(
        description=(
            "Exact catalogue defect text from sci_registry_defects.v1.json. "
            "Copied in Python; the model is not asked for this string."
        )
    )
    requirement: str = Field(
        description=(
            "Exact catalogue requirement text from sci_registry_defects.v1.json. "
            "Copied in Python; the model is not asked for this string."
        )
    )
    main_category: str
    special_category: str | None = None
    status: ResultState
    summary: str
    confidence: float
    review_reason_code: GapKind | None = None
    review_reason: str | None = Field(
        default=None,
        description=(
            "Plain-language explanation of why a needs_review finding was not "
            "published as a confirmed defect or compliance result."
        ),
    )
    reasoning: str
    evidence: list[FindingEvidence] = Field(default_factory=list)
    how_to_cure: HowToCure = Field(default_factory=HowToCure)
    applicable_rule: str | None = None
    location: str | None = Field(
        default=None,
        description=(
            "Page inside the document type, e.g. 'Vakalatnama, page 1.' "
            "This is not the stitched filing page. If the page is unknown: "
            "'Page missing — …'."
        ),
    )
    location_source: list[SourceLocation] = Field(default_factory=list)
    evidence_ids: list[str] = Field(default_factory=list)
    visual_localizations: list[VisualLocalization] = Field(default_factory=list)
    coverage: Coverage = Field(default_factory=Coverage)
    usage: LlmUsage | None = None
    error: str | None = None
    defect_version: int = 1


# A published defect is a high-impact assertion.  Prefer an explicit review
# over a false positive; individual deployments can still tune this after
# calibrating against a labelled filing set.
DEFAULT_REVIEW_CONFIDENCE = 0.6


def _safe_page_number(value: Any) -> int | None:
    try:
        if value is None or isinstance(value, bool):
            return None
        return int(value)
    except (TypeError, ValueError):
        return None


def review_confidence_threshold() -> float:
    raw = os.getenv("SCRUTINY_REVIEW_CONFIDENCE", "")
    try:
        value = float(raw) if raw else DEFAULT_REVIEW_CONFIDENCE
    except ValueError:
        value = DEFAULT_REVIEW_CONFIDENCE
    return min(1.0, max(0.0, value))


def review_reason_for_response(
    response: DefectResponse,
    coverage: Coverage | None = None,
) -> str | None:
    """Explain a review outcome without exposing internal prompt terminology."""
    if response.status not in {"needs_review", "rule_verification_pending"}:
        return None

    detail = " ".join((response.gap_detail or "").split()).strip().rstrip(".")
    if response.gap == "low_confidence":
        score = round(response.confidence * 100)
        threshold = round(review_confidence_threshold() * 100)
        if response.confidence < review_confidence_threshold():
            reason = (
                f"The evidence confidence is {score}%, below the {threshold}% threshold "
                "required to classify this as a confirmed defect or compliance result."
            )
        else:
            reason = (
                f"The evidence was not conclusive enough for a confirmed outcome. "
                f"The {score}% score reflects confidence in sending this finding for "
                "manual review, not confidence that the defect exists."
            )
    elif response.gap == "part_not_in_excerpts":
        reason = "The required document was not available in the pages reviewed."
    elif response.gap == "element_page_not_in_excerpts":
        reason = "The specific page, clause, or field needed for this check was not available."
    elif response.gap == "unreadable":
        reason = "The relevant text or visual field could not be read reliably."
    elif response.gap == "ambiguous":
        reason = (
            "The available evidence supports more than one reasonable interpretation."
        )
    elif response.gap == "conflict":
        reason = "The extracted text and the visible document evidence do not agree."
    elif response.gap == "legal_rule_unverified":
        reason = (
            "This check is awaiting verification of its controlling legal source."
        )
    else:
        reason = "The available evidence is insufficient for a confirmed outcome."

    if detail and response.gap != "low_confidence":
        reason = f"{reason} Detail: {detail}."
    if coverage and coverage.visual_required and not coverage.visual_complete:
        reason += " The required visual inspection was not completed."
    return reason


def _retrieved_pages(chunks: list[dict[str, Any]]) -> set[int]:
    pages: set[int] = set()
    for chunk in chunks:
        if chunk.get("chunk_kind") in {"summary", "page_manifest"}:
            continue
        page_texts = chunk.get("page_texts")
        if isinstance(page_texts, dict):
            for page in page_texts:
                number = _safe_page_number(page)
                if number is not None:
                    pages.add(number)
        start = chunk.get("page")
        if start is None:
            continue
        try:
            start_i = int(start)
        except (TypeError, ValueError):
            continue
        end = chunk.get("page_end")
        try:
            end_i = int(end) if end is not None else start_i
        except (TypeError, ValueError):
            end_i = start_i
        end_i = max(end_i, start_i)
        pages.update(range(start_i, end_i + 1))
    return pages


def _chunk_page_for_quote(quote: str, chunk: dict[str, Any]) -> int | None:
    if chunk.get("chunk_kind") == "page_manifest":
        return None
    page_texts = chunk.get("page_texts")
    if isinstance(page_texts, dict):
        for raw_page, page_text in page_texts.items():
            if quote_in_text(quote, str(page_text or "")):
                return _safe_page_number(raw_page)
    text = chunk.get("text") or ""
    if not quote_in_text(quote, text):
        return None
    start = chunk.get("page")
    try:
        return int(start) if start is not None else None
    except (TypeError, ValueError):
        return None


def _chunk_part_rank(chunk: dict[str, Any], preferred: list[str]) -> int:
    """Higher is better: target part beat Index listing lines."""
    names = [n.lower() for n in parts_on_page(chunk.get("document_part"))]
    if not names:
        return 0
    if preferred and _part_match(chunk, preferred):
        return 2
    if set(names) <= {"index"}:
        return -1
    return 1


def _page_allowed_for_evidence(
    page_chunks: list[dict[str, Any]],
    page: int | None,
    preferred: list[str],
    allow_index: bool,
) -> bool:
    if page is None:
        return False
    for chunk in page_chunks:
        page_texts = chunk.get("page_texts")
        if isinstance(page_texts, dict) and any(
            _safe_page_number(raw_page) == page for raw_page in page_texts
        ):
            if allow_index or _chunk_part_rank(chunk, preferred) >= 0:
                return True
        try:
            chunk_page = int(chunk["page"]) if chunk.get("page") is not None else None
        except (TypeError, ValueError):
            continue
        if chunk_page != page:
            continue
        if allow_index or _chunk_part_rank(chunk, preferred) >= 0:
            return True
    return False


_SNIPPET_MAX = 280


def _snippet_for_boxes(text: str) -> str:
    collapsed = " ".join((text or "").split())
    if len(collapsed) <= _SNIPPET_MAX:
        return collapsed
    return collapsed[-_SNIPPET_MAX:]


def _evidence_from_chunks(
    chunks: list[dict[str, Any]], defect: Defect
) -> list[EvidenceRef]:
    """When the model cites nothing, box the inspected pages we actually retrieved."""
    preferred = parts_named_in_where_to_look(defect) or preferred_parts_for_defect(
        defect
    )
    page_chunks = [chunk for chunk in chunks if chunk.get("chunk_kind") == "page"]
    ranked = sorted(
        page_chunks,
        key=lambda chunk: (
            -_chunk_part_rank(chunk, preferred),
            _safe_page_number(chunk.get("page")) or 999999,
        ),
    )
    refs: list[EvidenceRef] = []
    seen: set[int] = set()
    for chunk in ranked:
        if preferred and _chunk_part_rank(chunk, preferred) < 2:
            continue
        try:
            page = _safe_page_number(chunk.get("page"))
        except (TypeError, ValueError):
            page = None
        if page is None or page in seen:
            continue
        quote = _snippet_for_boxes(str(chunk.get("text") or ""))
        if not quote:
            continue
        seen.add(page)
        refs.append(
            EvidenceRef(
                page=page,
                quote=quote,
                chunk_id=str(chunk.get("excerpt_id") or "") or None,
            )
        )
        break
    return refs


def _as_bounding_boxes(raw: Any) -> list[BoundingBox]:
    boxes: list[BoundingBox] = []
    if not isinstance(raw, list):
        return boxes
    for item in raw:
        try:
            box = BoundingBox.model_validate(item)
        except ValidationError:
            continue
        x = float(box.x)
        y = float(box.y)
        width = float(box.w)
        height = float(box.h)
        if width <= 0 or height <= 0:
            continue
        max_x = x + width
        max_y = y + height
        if x > 1.5 or y > 1.5 or max_x > 1.5 or max_y > 1.5:
            scale_x = max(max_x, 1.0)
            scale_y = max(max_y, 1.0)
            x /= scale_x
            width /= scale_x
            y /= scale_y
            height /= scale_y
        x = min(max(x, 0.0), 1.0)
        y = min(max(y, 0.0), 1.0)
        width = min(max(width, 0.0), round(1.0 - x, 6))
        height = min(max(height, 0.0), round(1.0 - y, 6))
        if width <= 0:
            width = min(0.001, 1.0 - x) if x < 1.0 else 0.001
            x = min(x, 1.0 - width)
        if height <= 0:
            height = min(0.001, 1.0 - y) if y < 1.0 else 0.001
            y = min(y, 1.0 - height)
        boxes.append(
            BoundingBox(
                page=int(box.page),
                x=round(x, 6),
                y=round(y, 6),
                w=round(width, 6),
                h=round(height, 6),
            )
        )
    return boxes


def _local_pages(
    layout: dict[int, dict[str, Any]] | None,
    global_pages: list[int],
) -> list[int]:
    """Map stitched filing pages to each document's own page. Drop unmapped pages."""
    if not layout:
        return []
    local: list[int] = []
    for global_page in global_pages:
        meta = page_entry(layout, global_page)
        number = _safe_page_number((meta or {}).get("local_page")) if meta else None
        if number is not None and number not in local:
            local.append(number)
    return local


def _document_type_for_page(
    page_meta: dict[str, Any] | None,
    chunks: list[dict[str, Any]] | None,
    global_page: int | None,
) -> str | None:
    """Document type for this layout page. The slot label wins over the chunk."""
    raw = (page_meta or {}).get("document_part")
    if isinstance(raw, list):
        names = [str(name).strip() for name in raw if str(name).strip()]
        if len(names) == 1:
            return names[0]
        if names:
            return ", ".join(names)
    text = str(raw or "").strip()
    if text:
        return text
    return part_for_page(chunks, global_page)


def _publish_local_evidence(
    *,
    local_page: int | None,
    boxes: list[BoundingBox],
    boxes_status: BoxesStatus,
) -> tuple[int | None, list[BoundingBox], BoxesStatus]:
    """API page is the page inside the document type.

    The stitched filing page is never returned. A missing box does not clear
    a known document page. Coordinates are sent only when the quote matched.
    """
    if local_page is None:
        return None, [], "unavailable"
    if boxes_status != "matched" or not boxes:
        return local_page, [], "page_only"
    return (
        local_page,
        [box.model_copy(update={"page": local_page}) for box in boxes],
        "matched",
    )


def _excerpt_label(chunk: dict[str, Any]) -> str:
    return str(chunk.get("excerpt_id") or "").strip()


def _chunk_span(chunk: dict[str, Any]) -> list[int]:
    """The chunk's own page.

    Borrowed overlap is marked ``--- from page N ---`` for retrieval only.
    A quote taken from that margin still cites this page, not the neighbour.
    """
    start = _safe_page_number(chunk.get("page"))
    if start is None:
        return []
    return [start]


def _choose_evidence_chunk(
    ref: EvidenceRef,
    page_chunks: list[dict[str, Any]],
) -> dict[str, Any] | None:
    """Pick the sent chunk that owns this citation.

    A matching quote wins over a wrong excerpt id. A known id is kept when
    the quote is not in any sent chunk, so the document page can still be
    returned. An unknown id with no quote match returns nothing.
    """
    cited_id = re.sub(r"[\[\]]", "", (ref.chunk_id or "")).split("|")[0].strip().lower()
    cited = None
    if cited_id:
        for chunk in page_chunks:
            if _excerpt_label(chunk).strip().lower() == cited_id:
                cited = chunk
                break
    quote = (ref.quote or "").strip()
    quote_hits = [
        chunk
        for chunk in page_chunks
        if quote and quote_in_text(quote, str(chunk.get("text") or ""))
    ]
    if cited is not None and cited in quote_hits:
        return cited
    if quote_hits:
        for chunk in quote_hits:
            if ref.page is not None and _chunk_page_for_quote(quote, chunk) == ref.page:
                return chunk
        return quote_hits[0]
    if cited is not None:
        return cited
    return None


def attach_evidence_boxes(
    refs: list[EvidenceRef],
    *,
    layout: dict[int, dict[str, Any]] | None = None,
    chunks: list[dict[str, Any]] | None = None,
) -> list[FindingEvidence]:
    """Open the cited chunk's pages only. Never scan the rest of the filing."""
    attached: list[FindingEvidence] = []
    page_chunks = [
        chunk for chunk in (chunks or []) if chunk.get("chunk_kind") != "summary"
    ]
    for ref in refs:
        try:
            chunk = _choose_evidence_chunk(ref, page_chunks)
            if chunk is None:
                attached.append(
                    FindingEvidence(
                        page=None,
                        quote=ref.quote or "",
                        bounding_boxes=[],
                        boxes_status="unavailable",
                    )
                )
                continue
            boxes, status, page = boxes_on_pages(
                ref.quote or "",
                _chunk_span(chunk),
                layout,
            )
            boxes_status: BoxesStatus = (
                status
                if status in ("matched", "page_only", "unavailable")
                else "unavailable"
            )
            page_meta = page_entry(layout, page) if layout else None
            local_page = None
            slot_id = None
            if page_meta is not None:
                slot_id = str(page_meta.get("slot_id") or "").strip() or None
                local_page = _safe_page_number(page_meta.get("local_page"))
            document_part = _document_type_for_page(page_meta, chunks, page)
            if not document_part:
                document_part = str(chunk.get("document_part") or "").strip() or None
            published_page, published_boxes, boxes_status = _publish_local_evidence(
                local_page=local_page,
                boxes=_as_bounding_boxes(boxes),
                boxes_status=boxes_status,
            )
            attached.append(
                FindingEvidence(
                    page=published_page,
                    quote=ref.quote or "",
                    bounding_boxes=published_boxes,
                    boxes_status=boxes_status,
                    document_part=document_part,
                    slot_id=slot_id,
                )
            )
        except Exception:
            logger.warning(
                "Failed to attach highlight boxes; leaving citation unboxed",
                exc_info=True,
            )
            attached.append(
                FindingEvidence(
                    page=None,
                    quote=(getattr(ref, "quote", None) or ""),
                    bounding_boxes=[],
                    boxes_status="unavailable",
                )
            )
    return attached


def _one_published_evidence(
    evidence: list[FindingEvidence],
) -> list[FindingEvidence]:
    """The API finding carries one citation. A matched box wins."""
    if len(evidence) <= 1:
        return evidence

    def rank(index: int, item: FindingEvidence) -> tuple[int, int, int]:
        matched = 1 if item.boxes_status == "matched" and item.bounding_boxes else 0
        has_page = 1 if item.page is not None else 0
        return (matched, has_page, -index)

    chosen = max(range(len(evidence)), key=lambda index: rank(index, evidence[index]))
    return [evidence[chosen]]


def _display_evidence_quote(value: str) -> str:
    """Remove retrieval overlap and markup noise from a displayed citation."""
    text = re.split(r"---\s*from\s+page\s+\d+\s*---", value or "", maxsplit=1)[0]
    text = re.sub(r"<[^>]+>", " ", text)
    text = re.sub(r"\|(?:\s*:?-+:?\s*\|)+", " ", text)
    text = re.sub(r"\s*\|\s*", " · ", text)
    text = re.sub(r"\s+", " ", text).strip(" ·")
    if len(text) > 500:
        text = text[:497].rstrip() + "…"
    return text


def apply_evidence_pages(
    response: DefectResponse,
    chunks: list[dict[str, Any]],
    defect: Defect | None = None,
) -> DefectResponse:
    """Snap evidence.page to retrieved filing pages; drop rulebook / Index noise."""
    if not response.evidence:
        return response
    page_chunks = [c for c in chunks if c.get("chunk_kind") == "page"]
    preferred: list[str] = []
    allow_index = True
    if defect is not None:
        preferred = parts_named_in_where_to_look(defect) or preferred_parts_for_defect(
            defect
        )
        allow_index = allows_index_evidence(defect)
    grounded: list[EvidenceRef] = []
    for ref in response.evidence:
        quote = ref.quote or ""
        matches = [
            chunk
            for chunk in page_chunks
            if _chunk_page_for_quote(quote, chunk) is not None
        ]
        if not allow_index:
            content_matches = [
                chunk for chunk in matches if _chunk_part_rank(chunk, preferred) >= 0
            ]
            # Index listing lines are not proof for content checks — drop them.
            if matches and not content_matches:
                continue
            matches = content_matches
        page: int | None = None
        if matches:
            matches.sort(
                key=lambda chunk: (
                    _chunk_part_rank(chunk, preferred),
                    1 if ref.page == _chunk_page_for_quote(quote, chunk) else 0,
                ),
                reverse=True,
            )
            page = _chunk_page_for_quote(quote, matches[0])
        elif ref.page in _retrieved_pages(chunks) and _page_allowed_for_evidence(
            page_chunks, ref.page, preferred, allow_index
        ):
            # Keep a retrieved page so layout matching can still box the quote
            # when Pinecone OCR differs from the model quote.
            page = ref.page
        grounded.append(EvidenceRef(page=page, quote=quote, chunk_id=ref.chunk_id))
    response.evidence = grounded
    return response


def apply_status_policy(response: DefectResponse) -> DefectResponse:
    """Move low-confidence defect/compliant calls to needs_review."""
    if (
        response.status in ("defect_found", "compliant")
        and response.confidence < review_confidence_threshold()
    ):
        return _move_to_review(response, gap="low_confidence")
    return response


def apply_review_reason_consistency_policy(
    response: DefectResponse,
) -> DefectResponse:
    """Keep the structured review reason consistent with its numeric score.

    A model may be highly confident that human review is appropriate and still
    label that outcome ``low_confidence``.  That is not the meaning of this
    reason code: it is reserved exclusively for scores below the configured
    publication threshold.
    """
    if (
        response.status == "needs_review"
        and response.gap == "low_confidence"
        and response.confidence >= review_confidence_threshold()
    ):
        response.gap = "ambiguous"
        response.gap_detail = (
            "The evidence is inconclusive even though the model reported a "
            "high confidence in routing the check for review."
        )
    return response


def apply_legal_verification_policy(
    defect: Defect, response: DefectResponse
) -> DefectResponse:
    """Preserve the filing outcome regardless of source-audit metadata.

    ``legal_verification_status`` and source locators are catalogue governance
    fields.  The approved operational requirement, its applicability and the
    filing evidence determine the filing status.  Keep this compatibility
    helper as a no-op so older callers cannot silently reinstate the former
    pre-retrieval gate.
    """
    del defect
    return response


def apply_scope_coverage_policy(
    response: DefectResponse,
    *,
    scope_complete: bool,
    prompt_truncated: bool = False,
) -> DefectResponse:
    """Do not publish collection-wide compliance from a semantic sample."""
    if response.status != "compliant" or scope_complete:
        return response
    detail = (
        "The complete relevant evidence exceeded the prompt evidence limit."
        if prompt_truncated
        else "The complete relevant document scope was not inspected."
    )
    return _move_to_review(
        response,
        gap="element_page_not_in_excerpts",
        detail=detail,
    )


_PARTY_ROLES = re.compile(
    r"\b(petitioner(?:s)?|appellant(?:s)?|respondent(?:s)?|intervenor(?:s)?)\b",
    re.IGNORECASE,
)


def apply_internal_contradiction_policy(
    defect: Defect, response: DefectResponse
) -> DefectResponse:
    """Treat an affirmative filing contradiction as a defect, not uncertainty."""
    if response.status != "needs_review" or response.gap != "conflict":
        return response
    wording = f"{defect.defect} {defect.requirement}".casefold()
    if not any(term in wording for term in ("specify whether", "inconsistent", "correctly")):
        return response
    roles = {
        match.group(1).casefold().rstrip("s")
        for ref in response.evidence
        for match in _PARTY_ROLES.finditer(ref.quote or "")
    }
    if len(roles) < 2:
        return response
    response.status = "defect_found"
    response.confidence = max(response.confidence, 0.9)
    response.gap = None
    response.gap_detail = ""
    response.summary = "The filing gives inconsistent party-side descriptions."
    response.observed = "The relevant filing text identifies more than one incompatible represented side."
    response.conclusion = "The required represented side is internally inconsistent."
    return response


def apply_visual_field_policy(
    defect: Defect,
    response: DefectResponse,
    visual_index: dict[str, Any] | None,
) -> DefectResponse:
    """Resolve owned date-presence failures from completed field review."""
    if not isinstance(visual_index, dict):
        return response
    wording = f"{defect.defect} {defect.requirement}".casefold()
    simple_date_presence = (
        "date" in wording
        and any(term in wording for term in ("mention the date", "does not mention", "date is blank"))
        and not any(term in wording for term in ("same as", "prior date", "earlier than", "later than"))
    )
    check_id = re.sub(r"[^A-Z0-9]", "", defect.check_id.upper())
    vakalatnama_date_owner = check_id == "D57"
    if not simple_date_presence and not vakalatnama_date_owner:
        return response
    required = {part.casefold() for part in defect.inspect_parts}
    assessments = []
    for item in visual_index.get("field_assessments") or []:
        if not isinstance(item, dict) or item.get("field") != "execution_date":
            continue
        document_type = str(item.get("document_type") or "").casefold()
        if vakalatnama_date_owner and "vakalat" not in document_type:
            continue
        if required and not any(
            part in document_type or document_type in part for part in required
        ):
            continue
        assessments.append(item)
    if not assessments:
        blank_date = any(
            re.search(r"(?im)^\s*(?:dated|date)\s*[:\-]\s*$", ref.quote or "")
            for ref in response.evidence
        )
        positive_date_mark = any(
            isinstance(mark, dict)
            and str(mark.get("marking_type") or "").casefold()
            in {"handwritten_field_value", "date_field_value"}
            and "date" in str(mark.get("associated_label") or "").casefold()
            for mark in visual_index.get("marks") or []
        )
        target_pages = {
            _safe_page_number(item.get("page"))
            for item in visual_index.get("targets") or []
            if isinstance(item, dict)
        }
        reviewed_pages = {
            _safe_page_number(page) for page in visual_index.get("pages") or []
        }
        visual_scope_complete = bool(
            str(visual_index.get("status") or "").casefold() == "ok"
            and target_pages
            and None not in target_pages
            and target_pages.issubset(reviewed_pages)
            and not visual_index.get("failures")
        )
        if blank_date and visual_scope_complete and not positive_date_mark:
            assessments = [
                {
                    "field": "execution_date",
                    "readable": True,
                    "value_present": False,
                }
            ]
    if not assessments or not all(bool(item.get("readable")) for item in assessments):
        return response
    if all(bool(item.get("value_present")) for item in assessments):
        # D-57 still needs the actual chronology comparison.  Presence alone
        # cannot establish compliance, so leave that decision to the grounded
        # evaluator.
        if vakalatnama_date_owner:
            return response
        response.status = "compliant"
        response.confidence = max(response.confidence, 0.9)
        response.summary = "The execution date is present in the inspected date field."
        response.observed = response.summary
        response.conclusion = "The date requirement is satisfied."
    else:
        response.status = "defect_found"
        response.confidence = max(response.confidence, 0.9)
        response.summary = "The inspected execution-date field is blank."
        response.observed = "The complete relevant page shows the date label without an entered value."
        response.conclusion = "The required execution date is missing."
    response.gap = None
    response.gap_detail = ""
    return response


def apply_margin_layout_policy(
    defect: Defect,
    response: DefectResponse,
    layout: dict[int, dict[str, Any]] | None,
) -> DefectResponse:
    """Measure D-89 margins from normalized PDF/OCR word geometry.

    The check states fixed dimensions, so a deterministic measurement is more
    reliable than asking the model to infer centimetres from a screenshot.
    A small tolerance absorbs box-rounding noise without excusing a real miss.
    """
    if _normalized_check_id(defect) != "D89" or not layout:
        return response
    wanted = {part.casefold() for part in defect.inspect_parts}
    measurements: list[tuple[int, float, float, float, float, str]] = []
    incomplete = False
    for raw_page, entry in sorted(layout.items()):
        if not isinstance(entry, dict):
            incomplete = True
            continue
        names = {part.casefold() for part in parts_on_page(entry.get("document_part"))}
        if wanted and not any(
            expected in actual or actual in expected
            for expected in wanted
            for actual in names
        ):
            continue
        words = [item for item in entry.get("words") or [] if isinstance(item, dict)]
        try:
            width = float(entry.get("width") or 0)
            height = float(entry.get("height") or 0)
            page = int(raw_page)
        except (TypeError, ValueError):
            incomplete = True
            continue
        if width <= 1 or height <= 1 or not words:
            incomplete = True
            continue
        boxes = []
        for word in words:
            try:
                x = float(word.get("x"))
                y = float(word.get("y"))
                w = float(word.get("w"))
                h = float(word.get("h"))
            except (TypeError, ValueError):
                continue
            if w <= 0 or h <= 0:
                continue
            boxes.append((x, y, x + w, y + h, str(word.get("text") or word.get("t") or "")))
        if not boxes:
            incomplete = True
            continue
        left_box = min(boxes, key=lambda box: box[0])
        right_box = max(boxes, key=lambda box: box[2])
        top_box = min(boxes, key=lambda box: box[1])
        bottom_box = max(boxes, key=lambda box: box[3])
        margins = (
            left_box[0] * width,
            (1.0 - right_box[2]) * width,
            top_box[1] * height,
            (1.0 - bottom_box[3]) * height,
        )
        nearest = min(
            (left_box, right_box, top_box, bottom_box),
            key=lambda box: min(box[0], 1 - box[2], box[1], 1 - box[3]),
        )
        measurements.append((page, *margins, nearest[4]))
    if not measurements:
        return response
    horizontal = 4.0 * 72.0 / 2.54
    vertical = 2.0 * 72.0 / 2.54
    tolerance = 3.0
    failures = [
        row
        for row in measurements
        if row[1] + tolerance < horizontal
        or row[2] + tolerance < horizontal
        or row[3] + tolerance < vertical
        or row[4] + tolerance < vertical
    ]
    if failures:
        page, left, right, top, bottom, token = failures[0]
        response.status = "defect_found"
        response.confidence = max(response.confidence, 0.95)
        response.gap = None
        response.gap_detail = ""
        response.summary = (
            f"Page {page} measures approximately {left:.1f}pt left, "
            f"{right:.1f}pt right, {top:.1f}pt top and {bottom:.1f}pt bottom; "
            "at least one required margin is undersized."
        )
        response.observed = response.summary
        response.conclusion = "The required 4 cm horizontal and 2 cm vertical margins are not met."
        response.evidence = [EvidenceRef(page=page, quote=token or "Measured page content boundary")]
    elif not incomplete:
        response.status = "compliant"
        response.confidence = max(response.confidence, 0.95)
        response.gap = None
        response.gap_detail = ""
        response.summary = "All covered petition and synopsis pages meet the measured margin thresholds."
        response.observed = response.summary
        response.conclusion = "The margin requirement is satisfied."
    return response


_AOR_EXECUTION_ENDORSEMENT = re.compile(
    r"\b(?:executed\s+in\s+my\s+presence|satisf(?:ied|y).{0,80}due\s+execution|"
    r"due\s+execution.{0,80}(?:certif|satisf)|identified\s+and\s+accepted)\b",
    re.IGNORECASE | re.DOTALL,
)


def apply_deterministic_postclassification_policy(
    defect: Defect,
    response: DefectResponse,
    chunks: list[dict[str, Any]],
    visual_index: dict[str, Any] | None,
    *,
    scope_complete: bool,
) -> DefectResponse:
    """Resolve complete, field-specific evidence after all retrieval retries."""
    if _normalized_check_id(defect) != "D75" or not scope_complete:
        return response
    if response.status not in {"needs_review", "not_determined"}:
        return response
    text = "\n".join(str(chunk.get("text") or "") for chunk in chunks)
    if _AOR_EXECUTION_ENDORSEMENT.search(text):
        return response
    if not isinstance(visual_index, dict):
        return response
    target_pages = {
        _safe_page_number(item.get("page"))
        for item in visual_index.get("targets") or []
        if isinstance(item, dict)
    }
    reviewed_pages = {
        _safe_page_number(page) for page in visual_index.get("pages") or []
    }
    visual_complete = bool(
        str(visual_index.get("status") or "").casefold() == "ok"
        and target_pages
        and None not in target_pages
        and target_pages.issubset(reviewed_pages)
        and not visual_index.get("failures")
    )
    if not visual_complete:
        return response
    page = next(
        (
            _safe_page_number(chunk.get("page"))
            for chunk in chunks
            if str(chunk.get("text") or "").strip()
        ),
        None,
    )
    response.status = "defect_found"
    response.confidence = max(response.confidence, 0.9)
    response.gap = None
    response.gap_detail = ""
    response.summary = (
        "The complete Vakalatnama contains no AOR certification that execution "
        "occurred in the AOR's presence and no endorsement of due execution."
    )
    response.observed = response.summary
    response.conclusion = "The required AOR execution endorsement is missing."
    if not response.evidence and page is not None:
        response.evidence = [EvidenceRef(page=page, quote="Vakalatnama execution block")]
    return response


def apply_conditional_manifest_policy(
    defect: Defect,
    response: DefectResponse,
    layout: dict[int, dict[str, Any]] | None,
) -> DefectResponse:
    """Return N/A when a complete manifest affirmatively lacks the trigger.

    This is deliberately limited to requirements phrased as a condition
    (``Where ...``) and catalogue-provided trigger phrases. It never turns an
    incomplete or unlabelled manifest into an absence finding.
    """
    if not str(defect.requirement or "").lstrip().casefold().startswith("where "):
        return response
    if response.status == "defect_found":
        return response
    raw_terms = str(getattr(defect, "trigger_words", "") or "")
    terms = [
        " ".join(item.casefold().split())
        for item in re.split(r"[;\n|]+", raw_terms)
        if len(" ".join(item.split())) >= 5
    ]
    if not terms or not layout:
        return response
    pages: list[str] = []
    for entry in layout.values():
        if not isinstance(entry, dict) or not parts_on_page(entry.get("document_part")):
            return response
        direct = str(entry.get("text") or entry.get("markdown") or "").strip()
        if direct:
            pages.append(direct)
            continue
        words = [
            str(item.get("text") or item.get("t") or "").strip()
            for item in entry.get("words") or []
            if isinstance(item, dict)
        ]
        pages.append(" ".join(item for item in words if item))
    corpus = " ".join(" ".join(pages).casefold().split())
    if not corpus or any(term in corpus for term in terms):
        return response
    response.status = "not_applicable"
    response.confidence = 1.0
    response.gap = None
    response.gap_detail = ""
    response.summary = (
        "Not applicable because the complete document manifest contains no "
        "evidence of the factual condition that activates this requirement."
    )
    response.observed = response.summary
    response.conclusion = "The conditional requirement is not activated."
    return response


def _move_to_review(
    response: DefectResponse,
    *,
    gap: GapKind,
    detail: str = "",
) -> DefectResponse:
    """Downgrade an unsafe conclusion. The published sentence is composed later."""
    response.status = "needs_review"
    review_confidence = max(0.0, review_confidence_threshold() - 0.01)
    response.confidence = min(response.confidence, review_confidence)
    response.summary = (
        "Needs review because the available evidence does not safely prove this defect."
    )
    response.gap = gap
    if detail:
        response.gap_detail = detail
    return response


_ADDRESS_TOPIC = re.compile(r"\baddresses?\b", re.IGNORECASE)
_ADDRESS_SHOWN = re.compile(
    r"\b(?:r\s*/\s*o|resident\s+of|residing\s+at|gut\s+no|plot\s+no|"
    r"house\s+no|street|locality|district|taluka|tehsil)\b",
    re.IGNORECASE,
)
_PARTY_NAME = re.compile(
    r"\b(?:col\.|mr\.|mrs\.|ms\.|shri|smt)\b|\b\d{1,2}\s*[.)]",
    re.IGNORECASE,
)
_OMITTED_PHRASE = re.compile(
    r"\b(?:omits?|missing|lacks?|does not contain|does not give|does not show|"
    r"without)\s+(?:the\s+|a\s+|an\s+|its\s+|their\s+)?"
    r"(?P<what>[^.;:]{3,80})",
    re.IGNORECASE,
)


def _claim_text(defect: Defect, response: DefectResponse) -> str:
    parts = [defect.defect, defect.requirement, defect.ai_note or ""]
    parts.extend(item.name for item in response.elements)
    parts.extend(item.detail for item in response.elements)
    return " ".join(parts)


def _address_claim(defect: Defect, response: DefectResponse) -> bool:
    return _ADDRESS_TOPIC.search(_claim_text(defect, response)) is not None


def _omitted_phrases(*texts: str) -> list[str]:
    found: list[str] = []
    for text in texts:
        for match in _OMITTED_PHRASE.finditer(text or ""):
            phrase = " ".join(match.group("what").split()).strip(" .")
            if len(phrase) >= 3 and phrase.casefold() not in {
                item.casefold() for item in found
            }:
                found.append(phrase)
    return found


def _place_fragment(text: str) -> bool:
    """A city or PIN line, with no party name."""
    compact = " ".join((text or "").split())
    if not compact or _PARTY_NAME.search(compact) or _ADDRESS_SHOWN.search(compact):
        return False
    tokens = re.findall(r"[A-Za-z]+|\d+", compact)
    if not tokens or len(tokens) > 6:
        return False
    return any(token.isalpha() for token in tokens) and any(
        token.isdigit() and len(token) >= 3 for token in tokens
    )


def _quote_already_shows_claim(
    quote: str,
    *,
    address_claim: bool,
    omitted: list[str],
) -> bool:
    body = quote_body(quote)
    folded = body.casefold()
    if address_claim and (_ADDRESS_SHOWN.search(body) or _place_fragment(body)):
        return True
    return any(len(phrase) >= 3 and phrase.casefold() in folded for phrase in omitted)


def _scoped_claim(address_claim: bool) -> tuple[str, str]:
    if address_claim:
        return (
            "The cited entry gives a name and no residential or office address.",
            "The cited entries do not give a residential or office address.",
        )
    return (
        "The cited entry does not show the required element.",
        "The cited entries do not meet this requirement.",
    )


def apply_supported_evidence_policy(
    defect: Defect, response: DefectResponse
) -> DefectResponse:
    """Drop a quote that already shows the thing the finding says is missing.

    This applies to every check. An address row that already names a house,
    locality, district, or taluka is not proof that the address is absent, and
    a quote that contains a phrase the finding calls missing is not proof of
    that gap. The published wording is then limited to the quotes that remain.
    """
    if response.status != "defect_found" or not response.evidence:
        return response
    address_claim = _address_claim(defect, response)
    omitted = _omitted_phrases(
        response.observed,
        response.conclusion,
        response.summary,
        *(item.detail for item in response.elements),
    )
    kept = [
        ref
        for ref in response.evidence
        if not _quote_already_shows_claim(
            ref.quote or "",
            address_claim=address_claim,
            omitted=omitted,
        )
    ]
    if len(kept) == len(response.evidence):
        return response
    response.evidence = kept
    if not kept:
        return response
    detail, closing = _scoped_claim(address_claim)
    for item in response.elements:
        if item.state in ("missing", "incomplete"):
            item.detail = detail
    response.observed = detail
    response.conclusion = closing
    response.summary = closing
    return response


def apply_grounded_defect_policy(response: DefectResponse) -> DefectResponse:
    """Never publish a defect without a grounded filing-page observation.

    A quotation may show an incomplete field or contradictory value.  A model
    assertion that something is absent, with no grounded page observation, is
    not proof of absence and must remain a review item.
    """
    if response.status != "defect_found":
        return response
    grounded = [
        ref for ref in response.evidence if ref.page is not None and ref.quote.strip()
    ]
    if grounded:
        return response
    return _move_to_review(
        response,
        gap="element_page_not_in_excerpts",
        detail="No filing-page quotation was grounded as proof of the defect.",
    )


def apply_visual_conflict_policy(
    defect: Defect,
    response: DefectResponse,
    visual_index: dict[str, Any] | None,
) -> DefectResponse:
    """Route text/vision conflicts to review instead of asserting absence."""
    # A detected mark supports a compliant signature finding; it is not a
    # conflict.  Only a confirmed absence claim can conflict with vision.
    if response.status != "defect_found":
        return response
    wording = f"{defect.defect} {defect.requirement}".casefold()
    missing_claim = " ".join(
        [
            response.observed or "",
            response.conclusion or "",
            response.summary or "",
            *(item.detail for item in response.elements if item.state != "present"),
        ]
    ).casefold()
    relevant_types: set[str] = set()
    signature_absence_claimed = bool(
        re.search(
            r"(?:missing|absent|no|without|unsigned|not\s+(?:signed|executed))"
            r"[^.;]{0,60}(?:signature|thumb impression)|"
            r"(?:signature|thumb impression)[^.;]{0,60}"
            r"(?:missing|absent|not shown|blank)",
            missing_claim,
            re.IGNORECASE,
        )
    )
    if signature_absence_claimed and any(
        term in wording for term in ("signature", "signed", "executed")
    ):
        relevant_types.update(
            {
                "signature_like_mark",
                "advocate_on_record_signature",
                "executant_signature",
            }
        )
    if any(term in wording for term in ("stamp", "seal", "notary", "sworn")) or (
        "date" in wording and "blank" in wording
    ):
        relevant_types.update(
            {"notary_seal", "stamp", "handwritten_field_value", "date_field_value"}
        )
    if not relevant_types:
        return response
    if not isinstance(visual_index, dict):
        return _move_to_review(
            response,
            gap="unreadable",
            detail="Required visual evidence was not inspected.",
        )
    targets = {part.casefold() for part in defect.inspect_parts}
    expected_signature_roles: set[str] = set()
    if any(term in wording for term in ("petitioner", "executant", "thumb impression")):
        expected_signature_roles.add("executant")
    if any(term in wording for term in ("advocate", "aor", "accepted by")):
        expected_signature_roles.add("advocate")
    evidence_pages = {
        int(ref.page) for ref in response.evidence if ref.page is not None
    }
    detected_signature_roles: set[str] = set()
    conflicts: list[str] = []
    for mark in visual_index.get("marks") or []:
        if not isinstance(mark, dict):
            continue
        marking_type = str(mark.get("marking_type") or "").strip().casefold()
        if marking_type not in relevant_types:
            continue
        mark_page = _safe_page_number(mark.get("page"))
        if (
            evidence_pages
            and marking_type
            in {
                "signature_like_mark",
                "advocate_on_record_signature",
                "executant_signature",
            }
            and mark_page not in evidence_pages
        ):
            continue
        document_type = str(mark.get("document_type") or "").strip()
        if targets and not any(
            target in document_type.casefold() or document_type.casefold() in target
            for target in targets
        ):
            continue
        signature_role = str(mark.get("signature_role") or "").strip().casefold()
        if marking_type in {
            "signature_like_mark",
            "advocate_on_record_signature",
            "executant_signature",
        }:
            if not signature_role:
                # An untyped mark elsewhere on the document is not enough to
                # contradict a role-specific absence finding.
                continue
            normalized_role = (
                "advocate"
                if signature_role in {"advocate", "aor", "advocate_on_record"}
                else "executant"
                if signature_role in {"executant", "petitioner", "deponent"}
                else signature_role
            )
            if expected_signature_roles and normalized_role not in expected_signature_roles:
                continue
            detected_signature_roles.add(normalized_role)
        visible = str(mark.get("visible_text") or "").strip()
        label = str(mark.get("associated_label") or "").strip()
        detail = visible or label or marking_type.replace("_", " ")
        conflicts.append(f"{document_type or 'target document'}: {detail}")
    if expected_signature_roles and not expected_signature_roles.issubset(
        detected_signature_roles
    ):
        # One AOR mark cannot erase a separately missing executant signature,
        # and vice versa.  Keep the grounded defect for the unmet role.
        conflicts = []
    if not conflicts:
        status = str(visual_index.get("status") or "").casefold()
        pages = visual_index.get("pages") or []
        target_pages = {
            _safe_page_number(item.get("page"))
            for item in visual_index.get("targets") or []
            if isinstance(item, dict)
            and _safe_page_number(item.get("page")) is not None
        }
        reviewed_pages = {
            page for raw in pages if (page := _safe_page_number(raw)) is not None
        }
        failures = visual_index.get("failures") or []
        if (
            status != "ok"
            or not target_pages
            or not target_pages.issubset(reviewed_pages)
            or failures
        ):
            original_error = str(visual_index.get("error") or "").strip()
            failure_details = [
                str(item.get("error") or item.get("reason") or "").strip()
                for item in failures
                if isinstance(item, dict)
            ]
            reason = next((item for item in failure_details if item), original_error)
            return _move_to_review(
                response,
                gap="unreadable",
                detail=(
                    "Required visual evidence has incomplete page coverage"
                    + (f": {reason}" if reason else ".")
                ),
            )
        return response
    return _move_to_review(
        response,
        gap="conflict",
        detail="; ".join(conflicts[:3]),
    )


def _normalized_check_id(defect: Defect) -> str:
    return re.sub(r"[^A-Z0-9]", "", defect.check_id.upper())


def apply_evidence_capability_policy(
    defect: Defect,
    response: DefectResponse,
    visual_index: dict[str, Any] | None,
) -> DefectResponse:
    """Do not publish legibility findings without page-quality evidence."""
    if _normalized_check_id(defect) != "D81":
        return response
    assessments = (
        visual_index.get("quality_assessments")
        if isinstance(visual_index, dict)
        else None
    )
    if isinstance(assessments, list) and assessments:
        targets = {
            _safe_page_number(item.get("page"))
            for item in (visual_index or {}).get("targets") or []
            if isinstance(item, dict)
        }
        assessed = {
            _safe_page_number(item.get("page"))
            for item in assessments
            if isinstance(item, dict)
        }
        complete = bool(
            str((visual_index or {}).get("status") or "").casefold() == "ok"
            and targets
            and None not in targets
            and targets.issubset(assessed)
            and not (visual_index or {}).get("failures")
        )
        if complete and all(bool(item.get("readable")) for item in assessments):
            response.status = "compliant"
            response.confidence = max(response.confidence, 0.9)
            response.gap = None
            response.gap_detail = ""
            response.summary = "Every covered page was visually readable."
            response.observed = response.summary
            response.conclusion = "The legibility requirement is satisfied."
        return response
    if response.status not in ("defect_found", "compliant", "needs_review"):
        return response
    response = _move_to_review(
        response,
        gap="unreadable",
        detail="No page-quality assessment is available for this visual check.",
    )
    response.observed = (
        "The current evidence contains no page-quality assessment for the alleged "
        "dim, blurred, clipped, or illegible page."
    )
    response.evidence = []
    return response


def _page_text(page: dict[str, Any]) -> str:
    direct = str(page.get("text") or page.get("markdown") or "").strip()
    if direct:
        return direct
    words = page.get("words")
    if not isinstance(words, list):
        return ""
    return " ".join(
        str(word.get("t") or word.get("text") or "").strip()
        for word in words
        if isinstance(word, dict)
    ).strip()


def _memo_of_parties_present(
    chunks: list[dict[str, Any]],
    layout: dict[int, dict[str, Any]] | None,
) -> bool:
    pages = [page for page in (layout or {}).values() if isinstance(page, dict)]
    pages.extend(chunk for chunk in chunks if chunk.get("chunk_kind") != "summary")
    for page in pages:
        names = parts_on_page(page.get("document_part"))
        if any(name.casefold() == "memo of parties" for name in names):
            return True
        if not any(name.casefold() == "main petition" for name in names):
            continue
        text = re.sub(r"\s+", " ", _page_text(page)).casefold()
        if "in the matter of" in text and any(
            role in text for role in ("petitioner", "respondent", "appellant")
        ):
            return True
    return False


def apply_presence_only_policy(
    defect: Defect,
    response: DefectResponse,
    chunks: list[dict[str, Any]],
    layout: dict[int, dict[str, Any]] | None,
) -> DefectResponse:
    """Keep D-93 presence separate from D-63 address completeness."""
    if _normalized_check_id(defect) != "D93" or not _memo_of_parties_present(
        chunks, layout
    ):
        return response
    response.status = "compliant"
    response.confidence = max(response.confidence, 0.95)
    response.summary = "The filing contains a Memo of Parties section."
    response.observed = "The filing contains party particulars in the Main Petition or a separately classified Memo of Parties."
    response.conclusion = "The presence requirement is met; missing address particulars are assessed separately under D-63."
    return response


def missing_parts_in_layout(
    defect: Defect,
    layout: dict[int, dict[str, Any]] | None,
) -> list[str] | None:
    """Required parts with no classified page in a complete layout.

    None when absence cannot be proved: there is no layout, or any page is
    unlabelled. An empty list means every required part has a page.
    """
    if not layout:
        return None
    pages: list[dict[str, Any]] = []
    for page in layout.values():
        if not isinstance(page, dict):
            return None
        names = parts_on_page(page.get("document_part"))
        if not names:
            return None
        pages.append({"document_part": names, "chunk_kind": "page"})
    if not pages:
        return None
    required = required_parts_for_defect(defect)
    if not required:
        return None
    return [part for part in required if not chunks_cover_part(pages, part)]


def manifest_absence_proven(
    defect: Defect,
    layout: dict[int, dict[str, Any]] | None,
) -> bool:
    """Whether a complete layout proves a required part of a switched row absent."""
    if not getattr(defect, "manifest_absence_proves_defect", False):
        return False
    missing = missing_parts_in_layout(defect, layout)
    owned = required_parts_for_defect(defect)
    return bool(missing) and any(part in owned for part in missing)


def missing_document_sentence(parts: list[str]) -> str:
    """The only sentence published when a required document type is absent."""
    label = _english_join([part for part in parts if part])
    if not label:
        label = "required document"
    plural = (
        len([part for part in parts if part]) > 1 or label.casefold() == "annexures"
    )
    verb = "are missing" if plural else "is missing"
    return f"The {label} {verb} from the case file."


def _english_join(parts: list[str]) -> str:
    if not parts:
        return ""
    if len(parts) == 1:
        return parts[0]
    if len(parts) == 2:
        return f"{parts[0]} and {parts[1]}"
    return ", ".join(parts[:-1]) + f" and {parts[-1]}"


def apply_manifest_absence_policy(
    defect: Defect,
    response: DefectResponse,
    layout: dict[int, dict[str, Any]] | None,
) -> DefectResponse:
    """Record a wholly missing document. The model is not asked to explain it."""
    if not manifest_absence_proven(defect, layout):
        return response
    missing = missing_parts_in_layout(defect, layout) or []
    owned = required_parts_for_defect(defect)
    sentence = missing_document_sentence(
        [part for part in missing if part in owned]
    )
    response.status = "defect_found"
    response.confidence = 1.0
    response.summary = sentence
    response.observed = ""
    response.conclusion = sentence
    response.gap = None
    response.evidence = []
    return response


def apply_retrieval_policy(
    defect: Defect,
    response: DefectResponse,
    chunks: list[dict[str, Any]],
) -> DefectResponse:
    """Do not treat a missing document part in the excerpt set as a defect."""
    if response.status != "defect_found":
        return response
    missing = missing_required_parts(defect, chunks)
    if missing:
        return _move_to_review(
            response,
            gap="part_not_in_excerpts",
            detail=", ".join(missing),
        )
    return response


def apply_undetermined_policy(
    defect: Defect,
    response: DefectResponse,
    chunks: list[dict[str, Any]],
) -> DefectResponse:
    """Uncertain evidence must never be promoted into a published defect."""
    if response.status != "not_determined":
        return response
    missing = missing_required_parts(defect, chunks)
    if missing:
        return _move_to_review(
            response,
            gap="part_not_in_excerpts",
            detail=", ".join(missing),
        )
    return _move_to_review(response, gap="unreadable")


def finalize_public_response(response: DefectResponse) -> DefectResponse:
    """Preserve an unresolved result and its original evidence-gap reason.

    An inability to verify a check is not affirmative proof of compliance.
    Callers may omit unresolved rows from the public findings list, but must
    never rewrite them as ``compliant``.
    """
    return response


class ScrutinySummary(BaseModel):
    total_defects: int
    defects_found: int
    compliant: int
    needs_review: int
    rule_verification_pending: int = 0
    not_determined: int
    not_applicable: int
    overall_confidence: float


class ScrutinyReport(BaseModel):
    schema_name: Literal["scrutiny_finding_v1"] = "scrutiny_finding_v1"
    catalogue_id: str
    catalogue_version: str
    agent_data_id: str | None = None
    file_hash: str | None = None
    petition_type: str | None = None
    model: str | None = None
    index_generation: str | None = None
    vector_schema_version: str | None = None
    prompt_version: str = "scrutiny-v2-multi-evidence"
    generated_at: str = Field(default_factory=lambda: datetime.now(UTC).isoformat())
    disclaimer: str | None = None
    findings: list[DefectFinding] = Field(default_factory=list)
    summary: ScrutinySummary
    usage: UsageSummary | None = None
    planned_checks: int | None = None
    stopped_early: bool = False


def scrutiny_report_is_current(
    report: dict[str, Any] | ScrutinyReport,
    *,
    file_hash: str | None,
    index_generation: str | None,
) -> bool:
    """Return whether a stored report belongs to the active source/index run."""
    data = (
        report.model_dump(mode="json") if isinstance(report, ScrutinyReport) else report
    )
    if not isinstance(data, dict):
        return False
    if file_hash and str(data.get("file_hash") or "") != file_hash:
        return False
    if index_generation:
        return str(data.get("index_generation") or "") == index_generation
    return True


def _pages_from_chunks(chunks: list[dict[str, Any]] | None) -> list[int]:
    pages: list[int] = []
    for chunk in chunks or []:
        if chunk.get("chunk_kind") == "summary":
            continue
        try:
            page = chunk.get("page")
            if page is None:
                continue
            page_i = int(page)
        except (TypeError, ValueError):
            continue
        if page_i not in pages:
            pages.append(page_i)
    return pages


def build_finding(
    defect: Defect,
    response: DefectResponse,
    *,
    evidence_ids: list[str],
    coverage: Coverage,
    usage: LlmUsage | None = None,
    chunks: list[dict[str, Any]] | None = None,
    layout: dict[int, dict[str, Any]] | None = None,
    visual_index: dict[str, Any] | None = None,
    page_parts: dict[int, list[str]] | None = None,
    record: dict[str, Any] | None = None,
) -> DefectFinding:
    catalogue = get_catalogue()
    refs = list(response.evidence)
    if not refs and chunks:
        refs = _evidence_from_chunks(chunks, defect)
    evidence = attach_evidence_boxes(refs, layout=layout, chunks=chunks)
    evidence_pages = [ref.page for ref in evidence if ref.page is not None]
    reviewed_global = list(coverage.pages_reviewed) or _pages_from_chunks(chunks)
    reviewed_local = _local_pages(layout, reviewed_global)
    pages = reviewed_local
    location = location_for_finding(
        evidence=evidence,
        chunks=chunks,
        layout=layout,
        reviewed_pages=reviewed_global,
    )
    coverage = coverage.model_copy(update={"pages_reviewed": reviewed_local})
    from ..visual.attach import attach_visual_localizations

    visual_localizations = attach_visual_localizations(
        defect,
        visual_index=visual_index,
        page_parts=page_parts,
        record=record,
    )
    return DefectFinding(
        check_id=defect.check_id,
        defect=defect.defect,
        requirement=defect.requirement,
        main_category=defect.main_category,
        special_category=defect.special_category,
        status=response.status,
        summary=validated_summary(
            defect,
            response.summary,
            response.status,
            pages=pages,
            evidence_pages=evidence_pages,
        ),
        confidence=response.confidence,
        review_reason_code=(
            response.gap
            if response.status in {"needs_review", "rule_verification_pending"}
            else None
        ),
        review_reason=review_reason_for_response(response, coverage),
        reasoning=compose_reasoning(
            defect,
            response,
            chunks=chunks,
            layout=layout,
            evidence=evidence,
        ),
        evidence=evidence,
        how_to_cure=cure_from_defect(defect),
        applicable_rule=defect.applicable_rule,
        location=location,
        location_source=official_source_locations(defect, catalogue),
        evidence_ids=evidence_ids,
        visual_localizations=visual_localizations,
        coverage=coverage,
        usage=usage,
        defect_version=defect.defect_version,
    )


def failed_finding(
    defect: Defect,
    error: str,
    usage: LlmUsage | None = None,
    *,
    visual_index: dict[str, Any] | None = None,
    page_parts: dict[int, list[str]] | None = None,
    record: dict[str, Any] | None = None,
) -> DefectFinding:
    """Placeholder when a defect could not be evaluated at all."""
    catalogue = get_catalogue()
    from ..visual.attach import attach_visual_localizations

    visual_localizations = attach_visual_localizations(
        defect,
        visual_index=visual_index,
        page_parts=page_parts,
        record=record,
    )
    return DefectFinding(
        check_id=defect.check_id,
        defect=defect.defect,
        requirement=defect.requirement,
        main_category=defect.main_category,
        special_category=defect.special_category,
        status="not_determined",
        summary=f"This check could not be completed: {error}",
        confidence=0.0,
        reasoning=f"This check did not run: {error}",
        evidence=[],
        visual_localizations=visual_localizations,
        how_to_cure=cure_from_defect(defect),
        applicable_rule=defect.applicable_rule,
        location="Filing page missing — this check did not run.",
        location_source=official_source_locations(defect, catalogue),
        error=error,
        usage=usage,
        defect_version=defect.defect_version,
    )


def missing_document_finding(
    defect: Defect,
    parts: list[str],
    *,
    visual_index: dict[str, Any] | None = None,
    page_parts: dict[int, list[str]] | None = None,
    record: dict[str, Any] | None = None,
) -> DefectFinding:
    """Publish a missing document type without calling the model."""
    sentence = missing_document_sentence(parts)
    try:
        catalogue = get_catalogue()
    except RuntimeError:
        catalogue = None
    try:
        sources = official_source_locations(defect, catalogue)
    except AttributeError:
        sources = []
    try:
        cure = cure_from_defect(defect)
    except AttributeError:
        cure = HowToCure(steps=list(getattr(defect, "how_to_cure", None) or []))
    from ..visual.attach import attach_visual_localizations

    visual_localizations = attach_visual_localizations(
        defect,
        visual_index=visual_index,
        page_parts=page_parts,
        record=record,
    )
    return DefectFinding(
        check_id=str(getattr(defect, "check_id", "") or ""),
        defect=str(getattr(defect, "defect", "") or ""),
        requirement=str(getattr(defect, "requirement", "") or ""),
        main_category=str(getattr(defect, "main_category", "") or ""),
        special_category=getattr(defect, "special_category", None),
        status="defect_found",
        summary=sentence,
        confidence=1.0,
        reasoning=sentence,
        evidence=[],
        visual_localizations=visual_localizations,
        how_to_cure=cure,
        applicable_rule=getattr(defect, "applicable_rule", None),
        location="Not found in the case file.",
        location_source=sources,
        coverage=Coverage(
            structured_record_available=isinstance(record, dict) and bool(record),
            evidence_complete=True,
        ),
        defect_version=int(getattr(defect, "defect_version", 1) or 1),
    )


def missing_dependency_finding(
    defect: Defect,
    parts: list[str],
    *,
    record: dict[str, Any] | None = None,
) -> DefectFinding:
    """Skip a content/formality check when its source document is absent.

    The absence itself is reported once by the document-presence owner check.
    Signature, stamp, date, execution and content checks cannot truthfully be
    evaluated against a document which the complete manifest proves is absent.
    """
    label = _english_join([part for part in parts if part]) or "required document"
    sentence = (
        f"Not applicable because the {label} is absent from the complete "
        "document manifest; its dependent check was skipped."
    )
    try:
        catalogue = get_catalogue()
        sources = official_source_locations(defect, catalogue)
    except (RuntimeError, AttributeError):
        sources = []
    try:
        cure = cure_from_defect(defect)
    except AttributeError:
        cure = HowToCure(steps=list(getattr(defect, "how_to_cure", None) or []))
    return DefectFinding(
        check_id=str(getattr(defect, "check_id", "") or ""),
        defect=str(getattr(defect, "defect", "") or ""),
        requirement=str(getattr(defect, "requirement", "") or ""),
        main_category=str(getattr(defect, "main_category", "") or ""),
        special_category=getattr(defect, "special_category", None),
        status="not_applicable",
        summary=sentence,
        confidence=1.0,
        reasoning=sentence,
        evidence=[],
        how_to_cure=cure,
        applicable_rule=getattr(defect, "applicable_rule", None),
        location="Required document absent from the complete manifest.",
        location_source=sources,
        coverage=Coverage(
            structured_record_available=isinstance(record, dict) and bool(record),
            evidence_complete=True,
        ),
        defect_version=int(getattr(defect, "defect_version", 1) or 1),
    )


def summarize_usage(
    findings: list[DefectFinding], *, model: str | None
) -> UsageSummary:
    combined = LlmUsage(model=model)
    for finding in findings:
        if finding.usage:
            combined = combined.plus(finding.usage)

    rows: list[UsageByCheck] = []
    for finding in findings:
        usage = finding.usage or LlmUsage()
        share = None
        if combined.cost_usd and usage.cost_usd is not None:
            share = round(usage.cost_usd / combined.cost_usd, 4)
        elif combined.total_tokens and usage.total_tokens:
            share = round(usage.total_tokens / combined.total_tokens, 4)
        rows.append(
            UsageByCheck(
                check_id=finding.check_id,
                cost_usd=usage.cost_usd,
                prompt_tokens=usage.prompt_tokens,
                completion_tokens=usage.completion_tokens,
                total_tokens=usage.total_tokens,
                llm_calls=usage.calls,
                share=share,
            )
        )

    def _rank(row: UsageByCheck) -> tuple[float, int]:
        return (row.cost_usd if row.cost_usd is not None else -1.0, row.total_tokens)

    rows.sort(key=_rank, reverse=True)
    top = rows[0] if rows else None
    has_cost = top is not None and top.cost_usd is not None
    return UsageSummary(
        cost_usd=combined.cost_usd,
        prompt_tokens=combined.prompt_tokens,
        completion_tokens=combined.completion_tokens,
        total_tokens=combined.total_tokens,
        cached_tokens=combined.cached_tokens,
        reasoning_tokens=combined.reasoning_tokens,
        llm_calls=combined.calls,
        model=combined.model or model,
        highest_cost_check_id=top.check_id if has_cost else None,
        highest_cost_usd=top.cost_usd if has_cost else None,
        by_check=rows,
    )


def summarize(findings: list[DefectFinding]) -> ScrutinySummary:
    counts = Counter(f.status for f in findings)
    confidences = [f.confidence for f in findings] or [0.0]
    return ScrutinySummary(
        total_defects=len(findings),
        defects_found=counts.get("defect_found", 0),
        compliant=counts.get("compliant", 0),
        needs_review=counts.get("needs_review", 0),
        rule_verification_pending=counts.get("rule_verification_pending", 0),
        not_determined=counts.get("not_determined", 0),
        not_applicable=counts.get("not_applicable", 0),
        overall_confidence=round(sum(confidences) / len(confidences), 3),
    )
