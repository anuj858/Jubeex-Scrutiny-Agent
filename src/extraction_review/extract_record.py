"""Assemble the stored compiled-petition record after LlamaExtract."""

from __future__ import annotations

import re
from datetime import datetime
from difflib import SequenceMatcher
from typing import Any
from zoneinfo import ZoneInfo

from .document_parts import (
    FILING_TYPE_LABELS,
    document_spans_from_page_parts,
)
from .scrutiny.rules import normalize_filing_type

SCHEMA_VERSION = "extraction-v1"
JOB_TYPE = "compiled_petition"
LEGAL_EXTRACT_FIELDS = (
    "court",
    "petition_type",
    "cause_title",
    "petitioners",
    "respondents",
    "advocates_on_record",
    "impugned_orders",
    "relief_sort",
    "inconsistencies",
)

_ORG_PREFIXES = (
    "government of",
    "ministry of",
    "department of",
    "messrs",
    "m/s.",
    "m/s",
    "union",
)
_ORG_SUFFIXES = (
    "private limited",
    "pvt. ltd.",
    "pvt ltd",
    "limited",
    "ltd",
    "llp",
    "llc",
    "inc.",
    "corp.",
    "corporation",
    "co.",
    "company",
    "foundation",
    "trust",
    "society",
    "association",
)
_CONFIDENCE_OBJECT_KEYS = ("cause_title", "inconsistencies")
_CONFIDENCE_LIST_KEYS = (
    "petitioners",
    "respondents",
    "advocates_on_record",
    "impugned_orders",
)
_NAME_SPLIT = re.compile(r"[^a-z0-9]+")
_QUOTED_TEXT = re.compile(r'"([^"]+)"')
_QUOTED_TEXT_SINGLE = re.compile(r"'([^']+)'")
_VS_SPLIT = re.compile(r"\bvs\.?\b", re.IGNORECASE)
_ROLE_TAIL = re.compile(
    r"(?:[\s.·…]+|\s+)*(?:petitioners?|respondents?)(?:\s*\(\s*s\s*\))?\s*$",
    re.IGNORECASE,
)
_ANR_ORS_TAIL = re.compile(
    r"(?:\s*[,&]?\s*(?:and\s+)?(?:anrs?|ors|another|others)\.?)+\s*$",
    re.IGNORECASE,
)
_LIST_SERIAL = re.compile(r"^\s*(?:\(\s*)?\d+\s*[.)]\s*")
_TITLE_TOKEN = re.compile(
    r"\b(?:smt|shri|sh|mr|mrs|ms|km|kumari|dr|sri)\b\.?",
    re.IGNORECASE,
)
_ANR_ORS_MARK = re.compile(
    r"\b(?:and\s+)?(?:anr|ors|another|others)\b",
    re.IGNORECASE,
)
_EXTRA_PARTY_LABEL = re.compile(
    r"\b(?:petitioner|respondent)\s*(?:no\.?\s*)?(?:[2-9]|[1-9]\d)\b",
    re.IGNORECASE,
)
_CAPTION_SOURCE = re.compile(
    r"cover page|vakalatnama|affidavit|aor's declaration|aors declaration",
    re.IGNORECASE,
)
_PRAYER_HEADING_LINE = re.compile(
    r"^\s*(?:\d+\s*[.)]\s*)?(?:</?[^>]+>|\*{1,3}|_{1,3}|`+|#+\s*)*"
    r"(?:main\s+)?prayer"
    r"(?:</?[^>]+>|\*{1,3}|_{1,3}|`+)*\s*:?\s*$",
    re.IGNORECASE,
)
_PRAYER_HEADING_PREFIX = re.compile(
    r"^\s*(?:\d+\s*[.)]\s*)?(?:</?[^>]+>|\*{1,3}|_{1,3}|`+|#+\s*)*"
    r"(?:main\s+)?prayer"
    r"(?:</?[^>]+>|\*{1,3}|_{1,3}|`+)*\s*:?\s*",
    re.IGNORECASE,
)
_PRAYER_MARKUP = re.compile(r"</?[^>]+>|\*{1,3}|_{2}")


def _as_dict(value: Any) -> dict[str, Any]:
    if isinstance(value, dict):
        return value
    dump = getattr(value, "model_dump", None)
    if callable(dump):
        payload = dump()
        return payload if isinstance(payload, dict) else {}
    return {}


def _named(value: Any) -> str | None:
    if isinstance(value, str):
        text = value.strip()
        return text or None
    data = _as_dict(value)
    name = data.get("name")
    if isinstance(name, str) and name.strip():
        return name.strip()
    return None


def _pages_int(value: Any) -> list[int]:
    pages: list[int] = []
    if value is None:
        return pages
    if isinstance(value, (int, float)):
        pages.append(int(value))
        return pages
    if isinstance(value, str):
        for bit in value.replace(";", ",").split(","):
            bit = bit.strip()
            if not bit:
                continue
            try:
                pages.append(int(bit))
            except ValueError:
                continue
        return pages
    if isinstance(value, (list, tuple)):
        for item in value:
            pages.extend(_pages_int(item))
    return pages


def _walk_for_confidence(value: Any) -> list[float]:
    scores: list[float] = []
    if isinstance(value, (int, float)) and not isinstance(value, bool):
        number = float(value)
        if 0.0 <= number <= 1.0:
            scores.append(number)
        return scores
    if isinstance(value, dict):
        for key, nested in value.items():
            lowered = str(key).lower()
            if lowered in {"confidence", "score", "overall_confidence"}:
                scores.extend(_walk_for_confidence(nested))
            elif isinstance(nested, (dict, list)):
                scores.extend(_walk_for_confidence(nested))
        return scores
    if isinstance(value, list):
        for item in value:
            scores.extend(_walk_for_confidence(item))
    return scores


def overall_confidence_from_job(job: Any) -> float | None:
    dumped = _as_dict(job)
    for key in ("extract_metadata", "extraction_metadata", "metadata"):
        blob = dumped.get(key)
        scores = _walk_for_confidence(blob)
        if scores:
            return round(sum(scores) / len(scores), 3)
    extra = getattr(job, "extract_metadata", None) or getattr(
        job, "extraction_metadata", None
    )
    scores = _walk_for_confidence(_as_dict(extra) or extra)
    if scores:
        return round(sum(scores) / len(scores), 3)
    return None


def field_confidence_from_job(job: Any) -> dict[str, Any]:
    dumped = _as_dict(job)
    for key in ("extract_metadata", "extraction_metadata", "metadata"):
        blob = dumped.get(key)
        if not isinstance(blob, dict):
            continue
        for nested_key in ("confidence", "confidence_scores", "field_confidence"):
            nested = blob.get(nested_key)
            if isinstance(nested, dict) and nested:
                return nested
    return {}


def confidence_percent_string(value: Any) -> str | None:
    """Turn 0.91, 91, or '91%' into '91%'."""
    if value is None or isinstance(value, bool):
        return None
    if isinstance(value, str):
        text = value.strip()
        if not text:
            return None
        if text.endswith("%"):
            number_text = text[:-1].strip()
            try:
                number = float(number_text)
            except ValueError:
                return text
            return f"{int(round(number))}%"
        try:
            number = float(text)
        except ValueError:
            return None
    elif isinstance(value, (int, float)):
        number = float(value)
    elif isinstance(value, dict):
        for key in ("overall_confidence", "confidence", "score"):
            if key in value:
                return confidence_percent_string(value.get(key))
        scores = _walk_for_confidence(value)
        if not scores:
            return None
        number = sum(scores) / len(scores)
    else:
        return None
    if 0.0 <= number <= 1.0:
        return f"{int(round(number * 100))}%"
    if 0.0 <= number <= 100.0:
        return f"{int(round(number))}%"
    return None


def clean_relief_sort(text: str | None) -> str | None:
    """Keep the Main Prayer body; drop heading, clause number, and markdown."""
    if not isinstance(text, str):
        return None
    blob = text.replace("\r\n", "\n").strip()
    if not blob:
        return None
    lines = blob.split("\n")
    while lines and _PRAYER_HEADING_LINE.match(lines[0]):
        lines.pop(0)
    blob = "\n".join(lines).strip()
    blob = _PRAYER_HEADING_PREFIX.sub("", blob, count=1).strip()
    blob = _PRAYER_MARKUP.sub("", blob)
    blob = re.sub(r"\n{3,}", "\n\n", blob).strip()
    return blob or None


def strip_anr_ors_suffix(text: str | None) -> str:
    """Drop trailing And Anr / And Ors / & Anr. already printed on the Cover Page."""
    cleaned = strip_party_role_label(text)
    while True:
        updated = _ANR_ORS_TAIL.sub("", cleaned).rstrip(" .,&")
        if updated == cleaned:
            break
        cleaned = updated
    return cleaned


def format_side_title(main_name: str | None, count: int) -> str | None:
    """Build one side of formatted_title: Name / Name and Anr. / Name and Ors."""
    name = strip_anr_ors_suffix(main_name)
    if not name:
        return None
    if count <= 1:
        return name
    if count == 2:
        return f"{name} and Anr."
    return f"{name} and Ors."


def build_formatted_title(
    main_petitioner: str | None,
    petitioner_count: int,
    main_respondent: str | None,
    respondent_count: int,
) -> str | None:
    left = format_side_title(main_petitioner, petitioner_count)
    right = format_side_title(main_respondent, respondent_count)
    if left and right:
        return f"{left} VS {right}"
    return left or right


def _name_tokens(name: str) -> str:
    return _NAME_SPLIT.sub(" ", name.strip().lower()).strip()


def is_organization_name(name: str | None) -> bool:
    text = (name or "").strip()
    if not text:
        return False
    original_lower = text.lower()
    if original_lower == "the" or original_lower.startswith("the "):
        return True
    for prefix in _ORG_PREFIXES:
        if original_lower == prefix:
            return True
        if original_lower.startswith(prefix) and (
            len(original_lower) == len(prefix)
            or not original_lower[len(prefix)].isalnum()
        ):
            return True
    compact = re.sub(r"\s+", " ", original_lower).rstrip(" .")
    for suffix in _ORG_SUFFIXES:
        needle = suffix.rstrip(".")
        if compact == needle or compact.endswith(" " + needle):
            return True
    return False


def normalize_party_kind(name: str | None, kind: str | None) -> str | None:
    if is_organization_name(name):
        return "ORGANIZATION"
    if isinstance(kind, str):
        text = kind.strip().upper()
        if text in {"ORGANISATION", "ORG"}:
            return "ORGANIZATION"
        if text in {"INDIVIDUAL", "ORGANIZATION"}:
            return text
    if (name or "").strip():
        return "INDIVIDUAL"
    return None


def _party_name(party: Any) -> str | None:
    data = _as_dict(party) if not isinstance(party, dict) else party
    name = data.get("name")
    if isinstance(name, str) and name.strip():
        return name.strip()
    return None


def _names_match(left: str | None, right: str | None) -> bool:
    a = _name_tokens(left or "")
    b = _name_tokens(right or "")
    return bool(a and b and (a == b or a in b or b in a))


def strip_party_role_label(text: str | None) -> str:
    """Drop trailing Petitioner/Respondent caption marks, with or without dots."""
    cleaned = re.sub(r"\s+", " ", (text or "").replace("…", "...")).strip()
    while True:
        updated = _ROLE_TAIL.sub("", cleaned).rstrip(" .")
        if updated == cleaned:
            break
        cleaned = updated
    return cleaned


def _inconsistency_compare_texts(raw_text: str | None) -> list[str]:
    blob = (raw_text or "").strip()
    if not blob:
        return []
    quoted = _QUOTED_TEXT.findall(blob)
    if len(quoted) < 2:
        quoted = _QUOTED_TEXT_SINGLE.findall(blob)
    if len(quoted) >= 2:
        return quoted
    parts = [bit.strip() for bit in _VS_SPLIT.split(blob) if bit.strip()]
    if len(parts) < 2:
        return []
    sides: list[str] = []
    for part in parts:
        if ":" in part:
            part = part.split(":", 1)[1].strip()
        if part:
            sides.append(part)
    return sides


def is_party_role_label_mismatch(raw_text: str | None) -> bool:
    """True when quoted sides differ only by caption labels, Anr/Ors, or letter case."""
    return is_cosmetic_name_mismatch(raw_text)


def normalize_compared_name(text: str | None) -> str:
    """Name used to decide if two inconsistency quotes are the same person."""
    cleaned = _LIST_SERIAL.sub("", strip_anr_ors_suffix(text)).strip()
    return re.sub(r"\s+", " ", cleaned.casefold()).strip()


def _core_person_name(text: str | None) -> str:
    cleaned = _TITLE_TOKEN.sub(" ", normalize_compared_name(text))
    return re.sub(r"\s+", " ", cleaned).strip()


def names_are_spelling_variants(left: str | None, right: str | None) -> bool:
    """True when two strings look like the same person with a spelling slip."""
    a = _core_person_name(left)
    b = _core_person_name(right)
    if not a or not b:
        return False
    if a == b or a in b or b in a:
        return True
    tokens_a = a.split()
    tokens_b = b.split()
    last_a = tokens_a[-1] if tokens_a else ""
    last_b = tokens_b[-1] if tokens_b else ""
    given_a = " ".join(tokens_a[:-1]) if len(tokens_a) > 1 else a
    given_b = " ".join(tokens_b[:-1]) if len(tokens_b) > 1 else b
    if last_a and last_b and last_a == last_b:
        return SequenceMatcher(None, given_a, given_b).ratio() >= 0.75
    return SequenceMatcher(None, a, b).ratio() >= 0.85


def is_extra_party_caption_mismatch(
    raw_text: str | None, label: str | None = None
) -> bool:
    """True when Cover Page 'And Anr/Ors' is compared to a later-listed party."""
    sides = _inconsistency_compare_texts(raw_text)
    blob = raw_text or ""
    has_shorthand = any(_ANR_ORS_MARK.search(side) for side in sides)
    caption_context = bool(_CAPTION_SOURCE.search(blob))
    extra_label = bool(_EXTRA_PARTY_LABEL.search(label or ""))
    if extra_label and (has_shorthand or caption_context):
        return True
    if len(sides) < 2:
        return False
    if not has_shorthand and not caption_context:
        return False
    cores = [name for name in (_core_person_name(side) for side in sides) if name]
    distinct = list(dict.fromkeys(cores))
    if len(distinct) < 2:
        return False
    base = distinct[0]
    if all(names_are_spelling_variants(base, other) for other in distinct[1:]):
        return False
    return True


def is_cosmetic_name_mismatch(raw_text: str | None) -> bool:
    """True when compared names match after ignoring case, role labels, and Anr/Ors."""
    sides = _inconsistency_compare_texts(raw_text)
    if len(sides) < 2:
        return False
    normalized = [normalize_compared_name(side) for side in sides]
    if not all(normalized):
        return False
    return len(set(normalized)) == 1


def _inconsistency_name_key(raw_text: str | None) -> frozenset[str] | None:
    names = {
        normalize_compared_name(side)
        for side in _inconsistency_compare_texts(raw_text)
    }
    names.discard("")
    if len(names) < 2:
        return None
    return frozenset(names)


def _drop_role_label_inconsistencies(payload: dict[str, Any]) -> None:
    blob = payload.get("inconsistencies")
    if not isinstance(blob, dict):
        return
    items = blob.get("items")
    if not isinstance(items, list):
        return
    kept: list[Any] = []
    seen_names: dict[frozenset[str], int] = {}
    for item in items:
        data = item if isinstance(item, dict) else _as_dict(item)
        raw = data.get("raw_text") or data.get("detail")
        raw_text = raw if isinstance(raw, str) else None
        if is_cosmetic_name_mismatch(raw_text):
            continue
        if is_extra_party_caption_mismatch(
            raw_text, data.get("label") if isinstance(data.get("label"), str) else None
        ):
            continue
        key = _inconsistency_name_key(raw_text)
        if key is not None and key in seen_names:
            existing = kept[seen_names[key]]
            existing_raw = ""
            if isinstance(existing, dict):
                existing_raw = str(existing.get("raw_text") or existing.get("detail") or "")
            if raw_text and (not existing_raw or len(raw_text) < len(existing_raw)):
                kept[seen_names[key]] = item
            continue
        if key is not None:
            seen_names[key] = len(kept)
        kept.append(item)
    for index, item in enumerate(kept, start=1):
        if isinstance(item, dict):
            item["id"] = str(index)
    blob["items"] = kept
    if not kept:
        blob["source_part"] = []
        blob["source_pages"] = []


def _next_inconsistency_id(items: list[Any]) -> str:
    used: set[int] = set()
    for item in items:
        data = _as_dict(item) if not isinstance(item, dict) else item
        raw = str(data.get("id") or "").strip()
        if raw.isdigit():
            used.add(int(raw))
    next_id = 1
    while next_id in used:
        next_id += 1
    return str(next_id)


def _normalize_legacy_keys(payload: dict[str, Any]) -> None:
    if not payload.get("relief_sort"):
        relief = payload.get("relief")
        sought = None
        if isinstance(relief, dict):
            sought = relief.get("sought")
        elif isinstance(relief, str):
            sought = relief
        if isinstance(sought, str) and sought.strip():
            payload["relief_sort"] = sought.strip()
    if "relief_sort" in payload:
        payload["relief_sort"] = clean_relief_sort(
            payload.get("relief_sort") if isinstance(payload.get("relief_sort"), str) else None
        )
    for aor in payload.get("advocates_on_record") or []:
        if isinstance(aor, dict) and not aor.get("office_address") and aor.get("office"):
            aor["office_address"] = aor.get("office")
    for order in payload.get("impugned_orders") or []:
        if not isinstance(order, dict):
            continue
        if not order.get("Forum") and not order.get("forum") and order.get("court_name"):
            order["Forum"] = order.get("court_name")
    blob = payload.get("inconsistencies")
    if not isinstance(blob, dict):
        return
    for item in blob.get("items") or []:
        if isinstance(item, dict) and not item.get("raw_text") and item.get("detail"):
            item["raw_text"] = item.get("detail")
            item.pop("detail", None)


def _normalize_parties(payload: dict[str, Any]) -> None:
    cause = payload.get("cause_title")
    cause_data = cause if isinstance(cause, dict) else _as_dict(cause)
    main_petitioner = (
        cause_data.get("main_petitioner") if isinstance(cause_data, dict) else None
    )
    main_respondent = (
        cause_data.get("main_respondent") if isinstance(cause_data, dict) else None
    )
    if not isinstance(main_petitioner, str):
        main_petitioner = None
    else:
        main_petitioner = strip_anr_ors_suffix(main_petitioner) or None
    if not isinstance(main_respondent, str):
        main_respondent = None
    else:
        main_respondent = strip_anr_ors_suffix(main_respondent) or None
    if isinstance(cause, dict):
        if main_petitioner:
            cause["main_petitioner"] = main_petitioner
        if main_respondent:
            cause["main_respondent"] = main_respondent

    for key, main_name in (
        ("petitioners", main_petitioner),
        ("respondents", main_respondent),
    ):
        parties = payload.get(key)
        if not isinstance(parties, list):
            continue
        for index, party in enumerate(parties):
            if not isinstance(party, dict):
                continue
            name = _party_name(party)
            party["kind"] = normalize_party_kind(name, party.get("kind"))
            if main_name:
                party["is_primary"] = _names_match(name, main_name)
            elif party.get("is_primary") is None:
                party["is_primary"] = index == 0

    if isinstance(cause, dict):
        formatted = build_formatted_title(
            main_petitioner,
            len(payload.get("petitioners") or []),
            main_respondent,
            len(payload.get("respondents") or []),
        )
        if formatted:
            cause["formatted_title"] = formatted


def _append_missing_acting_through(payload: dict[str, Any]) -> None:
    missing: list[str] = []
    pages: list[int] = []
    parts: list[str] = []
    for key in ("petitioners", "respondents"):
        for party in payload.get(key) or []:
            if not isinstance(party, dict):
                continue
            if party.get("kind") != "ORGANIZATION":
                continue
            acting = party.get("acting_through")
            if isinstance(acting, str) and acting.strip():
                continue
            name = _party_name(party) or "unnamed party"
            side = "petitioner" if key == "petitioners" else "respondent"
            missing.append(f"ORGANIZATION {side} '{name}' has no acting_through.")
            pages.extend(_pages_int(party.get("source_pages")))
            part = party.get("source_part")
            if isinstance(part, str) and part.strip():
                parts.append(part.strip())

    if not missing:
        return
    blob = payload.get("inconsistencies")
    if not isinstance(blob, dict):
        blob = {"items": [], "source_part": [], "source_pages": []}
        payload["inconsistencies"] = blob
    items = blob.get("items")
    if not isinstance(items, list):
        items = []
        blob["items"] = items
    existing = " ".join(
        str((_as_dict(item) if not isinstance(item, dict) else item).get("raw_text") or "")
        for item in items
    ).lower()
    for text in missing:
        if text.lower() in existing:
            continue
        items.append(
            {
                "id": _next_inconsistency_id(items),
                "label": "Missing acting through",
                "raw_text": text,
            }
        )
    part_list = blob.get("source_part")
    if not isinstance(part_list, list):
        part_list = []
        blob["source_part"] = part_list
    for part in parts:
        if part not in part_list:
            part_list.append(part)
    blob["source_pages"] = sorted(set(_pages_int(blob.get("source_pages")) + pages))


def _score_from_field_confidence(scores: Any) -> Any:
    if isinstance(scores, dict):
        nested = scores.get("confidence") or scores.get("score")
        if nested is not None:
            return nested
    return scores


def _stamp_object_confidence(node: dict[str, Any], scores: Any) -> None:
    existing = node.get("confidence")
    converted = confidence_percent_string(existing)
    if converted:
        node["confidence"] = converted
        return
    converted = confidence_percent_string(_score_from_field_confidence(scores))
    if converted:
        node["confidence"] = converted


def stamp_confidence(
    record: dict[str, Any],
    *,
    field_confidence: dict[str, Any] | None = None,
) -> None:
    scores = field_confidence if isinstance(field_confidence, dict) else {}
    for key in _CONFIDENCE_OBJECT_KEYS:
        node = record.get(key)
        if isinstance(node, dict):
            _stamp_object_confidence(node, scores.get(key))
    for key in _CONFIDENCE_LIST_KEYS:
        items = record.get(key)
        item_scores = scores.get(key)
        if not isinstance(items, list):
            continue
        for index, item in enumerate(items):
            if not isinstance(item, dict):
                continue
            per_item = None
            if isinstance(item_scores, list) and index < len(item_scores):
                per_item = item_scores[index]
            elif not isinstance(item_scores, list):
                per_item = item_scores
            _stamp_object_confidence(item, per_item)


def _petition_type_name(record: dict[str, Any], filing_type: str | None) -> str | None:
    named = _named(record.get("petition_type"))
    if named:
        return named
    key = normalize_filing_type(filing_type)
    if key in FILING_TYPE_LABELS:
        return FILING_TYPE_LABELS[key]
    raw = (filing_type or "").strip()
    return raw or None


def _drop_removed_fields(payload: dict[str, Any]) -> None:
    for key in ("classification", "applications", "relief"):
        payload.pop(key, None)
    for key in ("petitioners", "respondents"):
        for party in payload.get(key) or []:
            if isinstance(party, dict):
                party.pop("age", None)
    for aor in payload.get("advocates_on_record") or []:
        if isinstance(aor, dict):
            aor.pop("office", None)
    for order in payload.get("impugned_orders") or []:
        if isinstance(order, dict):
            for extra in ("court_name", "petition_type_name", "judges", "lower_court_name"):
                order.pop(extra, None)
    blob = payload.get("inconsistencies")
    if isinstance(blob, dict):
        blob.pop("summary", None)


def apply_extract_envelope(
    record: dict[str, Any] | None,
    *,
    page_parts: dict[int, Any] | None = None,
    filing_type: str | None = None,
    overall_confidence: float | str | None = None,
    field_confidence: dict[str, Any] | None = None,
    generated_at: str | None = None,
) -> dict[str, Any]:
    """Add envelope keys, stitch documents, and normalize legal fields."""
    payload = dict(record or {})
    spans = document_spans_from_page_parts(page_parts or {})
    petition_name = _petition_type_name(payload, filing_type)
    if petition_name:
        payload["petition_type"] = petition_name
    court_name = _named(payload.get("court"))
    if court_name:
        payload["court"] = court_name

    _normalize_legacy_keys(payload)
    _normalize_parties(payload)
    _append_missing_acting_through(payload)
    _drop_role_label_inconsistencies(payload)
    stamp_confidence(payload, field_confidence=field_confidence)
    _drop_removed_fields(payload)

    payload["schema_version"] = SCHEMA_VERSION
    payload["job_type"] = JOB_TYPE
    payload["organization_id"] = None
    payload["workspace_id"] = None
    payload["user_id"] = None
    payload["primary_document_id"] = None
    payload["documents"] = spans
    payload["document_counts"] = {
        "processed": len(spans),
        "failed": 0,
    }
    payload["overall_confidence"] = confidence_percent_string(overall_confidence)
    if generated_at is None:
        generated_at = datetime.now(ZoneInfo("Asia/Kolkata")).isoformat(
            timespec="seconds"
        )
    payload["generated_at"] = generated_at
    return payload


def unwrap_extracted_record(extracted_data: Any) -> dict[str, Any]:
    """Agent Data stores ExtractedData; legal fields live on `.data`."""
    dumped = _as_dict(extracted_data)
    inner = dumped.get("data")
    if isinstance(inner, dict):
        return inner
    return dumped


def stamp_source_pages(record: dict[str, Any]) -> None:
    """Coerce source_pages on nested objects to integer lists."""

    def visit(node: Any) -> None:
        if isinstance(node, dict):
            if "source_pages" in node:
                node["source_pages"] = _pages_int(node.get("source_pages"))
            for value in node.values():
                visit(value)
        elif isinstance(node, list):
            for item in node:
                visit(item)

    visit(record)
