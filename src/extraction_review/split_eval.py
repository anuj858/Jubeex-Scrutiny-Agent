"""Evaluate production compiled-split artifacts against reviewed annotations.

This module deliberately does not split PDFs. Run the existing ``upload_compiled``
pipeline, export its persisted STEP_SPLIT JSON artifact, then score that artifact
against a versioned, page-level ground-truth manifest.
"""

from __future__ import annotations

import argparse
import asyncio
import hashlib
import json
import os
import re
import sys
from collections import Counter, defaultdict
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import httpx

from .document_parts import normalize_part_name
from .structure_split import DOCUMENT_TYPES

ANNOTATION_SCHEMA = "jubeex.split-ground-truth.v1"
REPORT_SCHEMA = "jubeex.split-evaluation-report.v1"
_SHA256_RE = re.compile(r"^[0-9a-f]{64}$", re.IGNORECASE)
_CASE_ID_RE = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._-]{0,119}$")
_DYNAMIC_CATEGORY_RE = re.compile(r"^(?:Annexure [PAR]-[1-9]\d*|Application [1-9]\d*)$")


class AnnotationError(ValueError):
    """Ground truth is malformed or violates physical-page ownership rules."""


def _canonical_categories() -> set[str]:
    config_path = Path(__file__).resolve().parents[2] / "configs" / "config.json"
    payload = json.loads(config_path.read_text(encoding="utf-8"))
    categories = {
        str(item.get("name") or "").strip()
        for item in payload.get("split", {}).get("categories", [])
        if isinstance(item, dict)
    }
    categories.update(DOCUMENT_TYPES)
    categories.add("Unidentified")
    return {item for item in categories if item}


def _is_canonical_category(value: Any, allowed: set[str]) -> bool:
    if not isinstance(value, str) or not value.strip():
        return False
    category = value.strip()
    if category != value:
        return False
    if category in allowed or _DYNAMIC_CATEGORY_RE.fullmatch(category):
        return True
    # Dynamic canonical names are intentionally accepted only in the exact
    # spelling emitted by the existing part normalizer.
    return normalize_part_name(category) == category and category in allowed


def validate_annotations(payload: Any) -> dict[str, Any]:
    """Validate schema, types, unique identities, exact coverage and no overlap.

    Product ownership is single-owner: uncertain pages must be annotated as
    ``Unidentified`` rather than omitted or assigned to multiple documents.
    """
    if not isinstance(payload, dict) or payload.get("schema_version") != ANNOTATION_SCHEMA:
        raise AnnotationError(f"schema_version must be {ANNOTATION_SCHEMA!r}")
    cases = payload.get("cases")
    if not isinstance(cases, list):
        raise AnnotationError("cases must be a JSON array")
    allowed = _canonical_categories()
    seen_cases: set[str] = set()
    total_documents = 0
    for case_index, case in enumerate(cases):
        where = f"cases[{case_index}]"
        if not isinstance(case, dict):
            raise AnnotationError(f"{where} must be an object")
        case_id = case.get("case_id")
        if not isinstance(case_id, str) or not case_id.strip():
            raise AnnotationError(f"{where}.case_id is required")
        case_id = case_id.strip()
        if not _CASE_ID_RE.fullmatch(case_id) or case_id in {".", ".."}:
            raise AnnotationError(f"{where}.case_id must be a safe non-sensitive filename key")
        if case_id in seen_cases:
            raise AnnotationError(f"duplicate case_id {case_id!r}")
        seen_cases.add(case_id)
        source = case.get("source")
        if not isinstance(source, dict) or not any(
            isinstance(source.get(key), str) and source[key].strip()
            for key in ("pdf_id", "sha256")
        ):
            raise AnnotationError(f"{where}.source needs pdf_id or sha256")
        if source.get("sha256") and not _SHA256_RE.fullmatch(str(source["sha256"])):
            raise AnnotationError(f"{where}.source.sha256 must be 64 hexadecimal characters")
        page_count = case.get("physical_page_count")
        if isinstance(page_count, bool) or not isinstance(page_count, int) or page_count < 1:
            raise AnnotationError(f"{where}.physical_page_count must be a positive integer")
        quality = case.get("input_quality")
        if quality is not None and (not isinstance(quality, str) or not quality.strip()):
            raise AnnotationError(f"{where}.input_quality must be a non-empty string")
        docs = case.get("documents")
        if not isinstance(docs, list) or not docs:
            raise AnnotationError(f"{where}.documents must be a non-empty array")
        seen_doc_ids: set[str] = set()
        seen_doc_spans: set[tuple[str, int, int]] = set()
        owners: dict[int, str] = {}
        for doc_index, doc in enumerate(docs):
            doc_where = f"{where}.documents[{doc_index}]"
            if not isinstance(doc, dict):
                raise AnnotationError(f"{doc_where} must be an object")
            doc_id = doc.get("document_id")
            if not isinstance(doc_id, str) or not doc_id.strip():
                raise AnnotationError(f"{doc_where}.document_id is required")
            doc_id = doc_id.strip()
            if doc_id in seen_doc_ids:
                raise AnnotationError(f"{where} has duplicate document_id {doc_id!r}")
            seen_doc_ids.add(doc_id)
            category = doc.get("category")
            if not _is_canonical_category(category, allowed):
                raise AnnotationError(
                    f"{doc_where}.category is unsupported or non-canonical: {category!r}"
                )
            start, end = doc.get("start_page"), doc.get("end_page")
            if (
                isinstance(start, bool)
                or isinstance(end, bool)
                or not isinstance(start, int)
                or not isinstance(end, int)
                or start < 1
                or end < start
                or end > page_count
            ):
                raise AnnotationError(
                    f"{doc_where} has invalid inclusive page range {start!r}-{end!r}"
                )
            span_key = (str(category), start, end)
            if span_key in seen_doc_spans:
                raise AnnotationError(f"{where} has duplicate document range {span_key}")
            seen_doc_spans.add(span_key)
            for page in range(start, end + 1):
                if page in owners:
                    raise AnnotationError(
                        f"{where} overlaps page {page}: {owners[page]!r} and {doc_id!r}"
                    )
                owners[page] = doc_id
            flags = doc.get("ambiguity_flags", [])
            if not isinstance(flags, list) or any(not isinstance(flag, str) for flag in flags):
                raise AnnotationError(f"{doc_where}.ambiguity_flags must be a string array")
            if "reviewer_notes" in doc and not isinstance(doc["reviewer_notes"], str):
                raise AnnotationError(f"{doc_where}.reviewer_notes must be a string")
        missing = sorted(set(range(1, page_count + 1)) - set(owners))
        if missing:
            raise AnnotationError(
                f"{where} leaves source pages unannotated: {missing}; use Unidentified when needed"
            )
        total_documents += len(docs)
    return {
        "case_count": len(cases),
        "document_count": total_documents,
        "canonical_categories": sorted(allowed),
    }


def _prediction_segments(artifact: dict[str, Any]) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    audit = artifact.get("split_audit")
    if not isinstance(audit, dict):
        raise TypeError("production artifact is missing split_audit")
    manifest = audit.get("page_manifest")
    if not isinstance(manifest, dict) or not isinstance(manifest.get("pages"), list):
        raise TypeError("production artifact is missing split_audit.page_manifest.pages")
    pages = manifest["pages"]
    rows: list[dict[str, Any]] = []
    for item in pages:
        if not isinstance(item, dict):
            raise TypeError("prediction manifest page entries must be objects")
        raw_number = item.get("source_page")
        if isinstance(raw_number, bool) or not isinstance(raw_number, int):
            raise TypeError("prediction manifest contains an invalid source_page")
        number = raw_number
        labels = item.get("assigned_labels")
        if not isinstance(labels, list):
            labels = []
        labels = [str(label) for label in labels if str(label).strip()]
        output_slots = item.get("output_slots")
        if not isinstance(output_slots, list):
            output_slots = []
        output_slots = sorted({str(slot) for slot in output_slots if str(slot).strip()})
        rows.append(
            {
                **item,
                "source_page": number,
                "assigned_labels": labels,
                "output_slots": output_slots,
            }
        )
    rows.sort(key=lambda row: row["source_page"])
    segments: list[dict[str, Any]] = []
    for row in rows:
        segment_id = row.get("segment_id")
        label_tuple = tuple(row["assigned_labels"])
        if (
            segments
            and segments[-1]["segment_id"] == segment_id
            and segments[-1]["categories"] == list(label_tuple)
            and segments[-1]["output_slots"] == set(row["output_slots"])
            and row["source_page"] == segments[-1]["end_page"] + 1
        ):
            segment = segments[-1]
            segment["end_page"] = row["source_page"]
            segment["pages"].append(row["source_page"])
            segment["review_required"] |= row.get("review_status") == "needs_review"
            segment["review_reasons"].extend(row.get("review_reasons") or [])
            segment["output_slots"].update(row.get("output_slots") or [])
            classification = row.get("segment_classification") or {}
            if classification.get("confidence") is not None:
                segment["confidence"] = classification["confidence"]
        else:
            classification = row.get("segment_classification") or {}
            segments.append(
                {
                    "segment_id": str(segment_id or f"page_{row['source_page']}"),
                    "start_page": row["source_page"],
                    "end_page": row["source_page"],
                    "pages": [row["source_page"]],
                    "categories": list(label_tuple),
                    "category": label_tuple[0] if len(label_tuple) == 1 else None,
                    "confidence": classification.get("confidence"),
                    "review_required": row.get("review_status") == "needs_review",
                    "review_reasons": list(row.get("review_reasons") or []),
                    "output_slots": set(row.get("output_slots") or []),
                }
            )
    for segment in segments:
        segment["output_slots"] = sorted(segment["output_slots"])
        segment["review_reasons"] = sorted(set(segment["review_reasons"]))
    return segments, {"manifest": manifest, "page_rows": rows, "audit": audit}


def _maximum_overlap_matching(
    expected: list[dict[str, Any]], predicted: list[dict[str, Any]]
) -> list[tuple[int, int]]:
    """Maximum-weight one-to-one matching; overlap leads, identity breaks ties."""
    size = max(len(expected), len(predicted))
    if not size:
        return []
    weights = [[0] * size for _ in range(size)]
    for i, truth in enumerate(expected):
        truth_pages = set(range(truth["start_page"], truth["end_page"] + 1))
        for j, guess in enumerate(predicted):
            intersection = len(truth_pages.intersection(guess["pages"]))
            if not intersection:
                continue
            same_type = truth["category"] == guess["category"]
            exact = truth["start_page"] == guess["start_page"] and truth["end_page"] == guess["end_page"]
            weights[i][j] = intersection * 1_000_000 + int(same_type) * 1_000 + int(exact)
    # Hungarian minimum-cost assignment, applied to negative weights.
    u = [0] * (size + 1)
    v = [0] * (size + 1)
    p = [0] * (size + 1)
    way = [0] * (size + 1)
    for i in range(1, size + 1):
        p[0] = i
        j0 = 0
        minv = [10**30] * (size + 1)
        used = [False] * (size + 1)
        while True:
            used[j0] = True
            i0 = p[j0]
            delta = 10**30
            j1 = 0
            for j in range(1, size + 1):
                if used[j]:
                    continue
                cur = -weights[i0 - 1][j - 1] - u[i0] - v[j]
                if cur < minv[j]:
                    minv[j] = cur
                    way[j] = j0
                if minv[j] < delta:
                    delta = minv[j]
                    j1 = j
            for j in range(size + 1):
                if used[j]:
                    u[p[j]] += delta
                    v[j] -= delta
                else:
                    minv[j] -= delta
            j0 = j1
            if p[j0] == 0:
                break
        while True:
            j1 = way[j0]
            p[j0] = p[j1]
            j0 = j1
            if j0 == 0:
                break
    pairs = []
    for j in range(1, size + 1):
        i = p[j] - 1
        if 0 <= i < len(expected) and j - 1 < len(predicted) and weights[i][j - 1] > 0:
            pairs.append((i, j - 1))
    return pairs


def _safe_ratio(numerator: int, denominator: int) -> float | None:
    return round(numerator / denominator, 6) if denominator else None


def _f1(precision: float | None, recall: float | None) -> float | None:
    if precision is None and recall is None:
        return None
    if precision is None or recall is None:
        return 0.0
    if precision + recall == 0:
        return 0.0
    return round(2 * precision * recall / (precision + recall), 6)


def evaluate_case(case: dict[str, Any], artifact: dict[str, Any]) -> dict[str, Any]:
    expected = case["documents"]
    predicted, data = _prediction_segments(artifact)
    manifest = data["manifest"]
    page_rows = data["page_rows"]
    page_count = int(case["physical_page_count"])
    if int(manifest.get("source_page_count", -1)) != page_count:
        raise ValueError("prediction page count does not match ground truth")
    wanted_hash = (case.get("source") or {}).get("sha256")
    actual_hash = manifest.get("source_sha256") or data["audit"].get("source_pdf_sha256")
    if wanted_hash and str(actual_hash or "").lower() != str(wanted_hash).lower():
        raise ValueError("prediction source checksum does not match ground truth")
    wanted_pdf_id = (case.get("source") or {}).get("pdf_id")
    actual_pdf_id = manifest.get("source_document_id") or data["audit"].get("source_document_id")
    if wanted_pdf_id and str(wanted_pdf_id) != str(actual_pdf_id or ""):
        raise ValueError("prediction source PDF identifier does not match ground truth")

    matches = _maximum_overlap_matching(expected, predicted)
    expected_to_predicted = {i: j for i, j in matches}
    predicted_to_expected = {j: i for i, j in matches}
    gt_boundaries = {doc["end_page"] for doc in expected if 1 <= doc["end_page"] < page_count}
    # A boundary is a cut between adjacent physical pages assigned to distinct
    # predicted segments. Segment endpoints alone can invent cuts around gaps
    # and out-of-range pages, so derive the set from the page map.
    segment_by_page: dict[int, set[int]] = defaultdict(set)
    for segment_index, segment in enumerate(predicted):
        for page in segment["pages"]:
            if 1 <= page <= page_count:
                segment_by_page[page].add(segment_index)
    pred_boundaries = {
        page
        for page in range(1, page_count)
        if len(segment_by_page.get(page, set())) == 1
        and len(segment_by_page.get(page + 1, set())) == 1
        and next(iter(segment_by_page[page])) != next(iter(segment_by_page[page + 1]))
    }
    boundary_tp = len(gt_boundaries & pred_boundaries)
    boundary_precision = _safe_ratio(boundary_tp, len(pred_boundaries))
    boundary_recall = _safe_ratio(boundary_tp, len(gt_boundaries))
    exact_ranges = sum(
        expected[i]["start_page"] == predicted[j]["start_page"]
        and expected[i]["end_page"] == predicted[j]["end_page"]
        for i, j in matches
    )
    type_correct = sum(expected[i]["category"] == predicted[j]["category"] for i, j in matches)

    expected_owner = {
        page: i
        for i, doc in enumerate(expected)
        for page in range(doc["start_page"], doc["end_page"] + 1)
    }
    row_by_page: dict[int, list[dict[str, Any]]] = defaultdict(list)
    for row in page_rows:
        row_by_page[row["source_page"]].append(row)
    missing_pages = 0
    multiply_assigned_pages = 0
    out_of_range_pages = sorted(
        {row["source_page"] for row in page_rows if not 1 <= row["source_page"] <= page_count}
    )
    review_pages: set[int] = set()
    for page in range(1, page_count + 1):
        rows = row_by_page.get(page, [])
        if not rows:
            missing_pages += 1
            continue
        slot_sets = []
        for row in rows:
            slots = row.get("output_slots")
            if slots is None:
                slots = row.get("assigned_labels") or []
            slot_sets.append({str(slot) for slot in slots if str(slot).strip()})
        count_slots = len(set().union(*slot_sets))
        if count_slots == 0:
            missing_pages += 1
        elif count_slots > 1 or len(rows) > 1:
            multiply_assigned_pages += 1
        if any(row.get("review_status") == "needs_review" for row in rows):
            review_pages.add(page)

    correctly_owned = 0
    for page, truth_index in expected_owner.items():
        rows = row_by_page.get(page, [])
        if len(rows) != 1:
            continue
        slots = rows[0].get("output_slots")
        if slots is None:
            slots = rows[0].get("assigned_labels") or []
        if len({str(slot) for slot in slots if str(slot).strip()}) != 1:
            continue
        owners = [
            predicted_to_expected.get(index)
            for index, segment in enumerate(predicted)
            if page in segment["pages"]
        ]
        if owners == [truth_index]:
            correctly_owned += 1

    per_type: dict[str, dict[str, int]] = {}
    for i, doc in enumerate(expected):
        bucket = per_type.setdefault(
            doc["category"], {"expected_documents": 0, "matched_documents": 0, "exact_ranges": 0, "type_correct": 0}
        )
        bucket["expected_documents"] += 1
        j = expected_to_predicted.get(i)
        if j is not None:
            bucket["matched_documents"] += 1
            bucket["exact_ranges"] += int(
                doc["start_page"] == predicted[j]["start_page"]
                and doc["end_page"] == predicted[j]["end_page"]
            )
            bucket["type_correct"] += int(doc["category"] == predicted[j]["category"])
    for bucket in per_type.values():
        bucket["exact_range_rate"] = _safe_ratio(bucket["exact_ranges"], bucket["expected_documents"])
        bucket["type_accuracy"] = _safe_ratio(bucket["type_correct"], bucket["matched_documents"])

    reviewed_segments = sum(bool(doc["review_required"]) for doc in predicted)
    audit = data["audit"]
    artifact_config = artifact.get("config") or {}
    split_config_identity = artifact_config.get("split")
    if not isinstance(split_config_identity, dict) or not split_config_identity.get("config_version"):
        raise ValueError("production artifact is missing split config identity/version")
    safe_page_assignments = [
        {
            "source_page": row["source_page"],
            "segment_id": row.get("segment_id"),
            "assigned_labels": row.get("assigned_labels", []),
            "effective_owner": row.get("effective_owner"),
            "output_slots": row.get("output_slots") or [],
            "classification_confidence": (row.get("classification") or {}).get("confidence"),
            "segment_classification": row.get("segment_classification"),
            "review_status": row.get("review_status"),
            "review_reasons": row.get("review_reasons") or [],
        }
        for row in page_rows
    ]
    return {
        "case_id": case["case_id"],
        "source_pdf_id": actual_pdf_id,
        "source_sha256": actual_hash,
        "physical_page_count": page_count,
        "metrics": {
            "boundary_precision": boundary_precision,
            "boundary_recall": boundary_recall,
            "boundary_f1": _f1(boundary_precision, boundary_recall),
            "boundary_true_positives": boundary_tp,
            "boundary_predicted_count": len(pred_boundaries),
            "boundary_expected_count": len(gt_boundaries),
            "exact_document_range_accuracy": _safe_ratio(exact_ranges, len(expected)),
            "exact_document_ranges": exact_ranges,
            "expected_document_count": len(expected),
            "page_document_ownership_accuracy": _safe_ratio(
                correctly_owned, page_count + len(out_of_range_pages)
            ),
            "correctly_owned_pages": correctly_owned,
            "page_count_denominator": page_count + len(out_of_range_pages),
            "canonical_document_type_accuracy": _safe_ratio(type_correct, len(matches)),
            "correct_document_types": type_correct,
            "matched_document_count": len(matches),
            "unmatched_expected_document_count": len(expected) - len(matches),
            "unmatched_predicted_segment_count": len(predicted) - len(matches),
            "missing_page_count": missing_pages,
            "multiply_assigned_page_count": multiply_assigned_pages,
            "out_of_range_page_count": len(out_of_range_pages),
            "manual_review_rate": _safe_ratio(reviewed_segments, len(predicted)),
            "manual_review_segments": reviewed_segments,
            "predicted_segment_count": len(predicted),
            "manual_review_page_rate": _safe_ratio(len(review_pages), page_count),
            "review_required_pages": len(review_pages),
            "prediction_review_status": audit.get("review_status"),
            "case_manual_review_required": audit.get("review_status") == "needs_review",
            "structural_validation": audit.get("structural_validation"),
        },
        "by_document_type": per_type,
        "input_quality": str(case.get("input_quality") or "unlabeled").strip(),
        "ambiguous_document_count": sum(bool(doc.get("ambiguity_flags")) for doc in expected),
        "prediction_artifact_sha256": hashlib.sha256(
            json.dumps(artifact, sort_keys=True, separators=(",", ":")).encode("utf-8")
        ).hexdigest(),
        "pipeline_config_identity": artifact_config,
        # Safe, text-free predictions for review; confidence is model output and
        # is copied verbatim rather than inferred from structural checks.
        "predicted_segments": [
            {key: value for key, value in doc.items() if key != "pages"}
            for doc in predicted
        ],
        "predicted_page_assignments": safe_page_assignments,
        "predicted_boundary_proposals": [
            {
                key: item.get(key)
                for key in ("page", "previous_labels", "proposed_labels", "source")
                if key in item
            }
            for item in audit.get("llama_boundary_proposals", [])
            if isinstance(item, dict)
        ],
        "predicted_segment_classifications": [
            {
                key: item.get(key)
                for key in ("segment_id", "start_page", "end_page", "document_type", "confidence", "signals")
                if key in item
            }
            for item in audit.get("segment_classifications", [])
            if isinstance(item, dict)
        ],
    }


def evaluate_dataset(
    annotations: dict[str, Any],
    predictions: dict[str, dict[str, Any]],
    *,
    config_path: Path | None = None,
) -> dict[str, Any]:
    validation = validate_annotations(annotations)
    cases = annotations["cases"]
    if set(predictions) != {case["case_id"] for case in cases}:
        missing = sorted({case["case_id"] for case in cases} - set(predictions))
        extra = sorted(set(predictions) - {case["case_id"] for case in cases})
        raise ValueError(f"prediction case ids mismatch; missing={missing}, extra={extra}")
    config_file = config_path or Path(__file__).resolve().parents[2] / "configs" / "config.json"
    config_bytes = config_file.read_bytes()
    config = json.loads(config_bytes)
    results = [evaluate_case(case, predictions[case["case_id"]]) for case in cases]
    totals = Counter()
    for result in results:
        for key in (
            "boundary_true_positives",
            "boundary_predicted_count",
            "boundary_expected_count",
            "exact_document_ranges",
            "expected_document_count",
            "correctly_owned_pages",
            "page_count_denominator",
            "correct_document_types",
            "matched_document_count",
            "missing_page_count",
            "multiply_assigned_page_count",
            "out_of_range_page_count",
            "manual_review_segments",
            "predicted_segment_count",
            "review_required_pages",
            "case_manual_review_required",
        ):
            totals[key] += int(bool(result["metrics"][key])) if key == "case_manual_review_required" else int(result["metrics"][key])
    precision = _safe_ratio(totals["boundary_true_positives"], totals["boundary_predicted_count"])
    recall = _safe_ratio(totals["boundary_true_positives"], totals["boundary_expected_count"])
    global_metrics = {
        "boundary_precision": precision,
        "boundary_recall": recall,
        "boundary_f1": _f1(precision, recall),
        "exact_document_range_accuracy": _safe_ratio(totals["exact_document_ranges"], totals["expected_document_count"]),
        "page_document_ownership_accuracy": _safe_ratio(totals["correctly_owned_pages"], totals["page_count_denominator"]),
        "canonical_document_type_accuracy": _safe_ratio(totals["correct_document_types"], totals["matched_document_count"]),
        "missing_page_count": totals["missing_page_count"],
        "multiply_assigned_page_count": totals["multiply_assigned_page_count"],
        "out_of_range_page_count": totals["out_of_range_page_count"],
        "manual_review_rate": _safe_ratio(totals["manual_review_segments"], totals["predicted_segment_count"]),
        "manual_review_page_rate": _safe_ratio(totals["review_required_pages"], totals["page_count_denominator"]),
        "manual_review_case_rate": _safe_ratio(totals["case_manual_review_required"], len(results)),
        "denominators": dict(totals),
    }
    by_quality: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for result in results:
        by_quality[result["input_quality"]].append(result["metrics"])
    quality_summary = {
        quality: {
            "case_count": len(rows),
            "page_ownership_accuracy_mean": _safe_ratio(
                sum(row["correctly_owned_pages"] for row in rows),
                sum(row["page_count_denominator"] for row in rows),
            ),
            "exact_document_range_accuracy_mean": _safe_ratio(
                sum(row["exact_document_ranges"] for row in rows),
                sum(row["expected_document_count"] for row in rows),
            ),
        }
        for quality, rows in sorted(by_quality.items())
    }
    by_type_totals: dict[str, Counter] = defaultdict(Counter)
    for result in results:
        for category, metrics in result["by_document_type"].items():
            for metric in ("expected_documents", "matched_documents", "exact_ranges", "type_correct"):
                by_type_totals[category][metric] += metrics[metric]
    by_type_summary = {
        category: {
            **dict(counts),
            "exact_range_accuracy": _safe_ratio(counts["exact_ranges"], counts["expected_documents"]),
            "type_accuracy": _safe_ratio(counts["type_correct"], counts["matched_documents"]),
        }
        for category, counts in sorted(by_type_totals.items())
    }
    return {
        "schema_version": REPORT_SCHEMA,
        "generated_at_utc": datetime.now(UTC).isoformat(),
        "annotation_schema_version": annotations["schema_version"],
        "validation": validation,
        "pipeline": {
            "config_id": config.get("config_id"),
            "config_version": config.get("config_version"),
            "split_config_version": (config.get("pipeline_versions", {}).get("split") or {}).get("config_version"),
            "evaluator_config_sha256": hashlib.sha256(config_bytes).hexdigest(),
        },
        "metric_definitions": {
            "boundary": "Exact cut positions after physical pages; no tolerance. Precision denominator is predicted cuts; recall denominator is ground-truth cuts.",
            "document_matching": "One-to-one maximum-overlap assignment; overlap is the primary weight, canonical type and exact range are deterministic tie-breakers. Zero-overlap pairs are not matches.",
            "exact_document_range_accuracy": "Matched predictions with both inclusive endpoints exact / total ground-truth documents; unmatched ground-truth documents count incorrect.",
            "page_document_ownership_accuracy": "Correctly owned in-range pages / (all physical pages + distinct out-of-range page assignments). Missing pages, multiple output slots, duplicate manifest rows, and pages mapped to the wrong ground-truth document count incorrect; out-of-range assignments add errors to the denominator.",
            "canonical_document_type_accuracy": "Exact category matches among one-to-one document matches; unmatched segments are not in this denominator and are reported through match counts/range accuracy.",
            "missing_and_multiple_ownership": "Count in-range physical pages with zero or more than one output slot in the production page manifest; labels alone do not imply multiple owners. Out-of-range source-page rows are counted separately.",
            "manual_review_rate": "Predicted segments with at least one page marked needs_review / all predicted segments; page-level rate is also reported. A separate case-level flag/rate reflects the persisted overall review status.",
            "structural_validation": "Reported separately; a valid page map is not semantic correctness.",
            "ambiguity": "Reviewer flags are reported; chosen ground-truth labels remain scored and are not silently excluded.",
        },
        "global_metrics": global_metrics,
        "by_document_type": by_type_summary,
        "by_input_quality": quality_summary,
        "cases": results,
    }


def _read_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


async def run_existing_api_case(
    case: dict[str, Any],
    source: dict[str, Any],
    *,
    api_base_url: str,
    api_key: str,
    output_path: Path,
    allow_remote: bool,
    client: httpx.AsyncClient | None = None,
    poll_interval_s: float = 2.0,
    timeout_s: float = 1800.0,
) -> dict[str, Any]:
    """Submit one PDF to the existing upload_compiled API and save its STEP_SPLIT artifact.

    This is an explicit remote operation that uploads petition bytes through a
    caller-provided download URL and creates a filing job/artifact in the
    caller-provided organization/workspace. Call only for authorized data.
    """
    if not allow_remote:
        raise ValueError("remote evaluation requires explicit allow_remote opt-in")
    required = ("download_url", "filing_type", "organization_id", "workspace_id")
    absent = [key for key in required if not str(source.get(key) or "").strip()]
    if absent:
        raise ValueError(f"source mapping for {case['case_id']} missing: {', '.join(absent)}")
    if not api_key.strip():
        raise ValueError("remote evaluation requires JUBEEX_API_KEY authorization")
    owns_client = client is None
    http = client or httpx.AsyncClient(timeout=60.0, follow_redirects=True)
    base = api_base_url.rstrip("/")
    headers = {"X-API-Key": api_key}
    try:
        response = await http.post(
            f"{base}/v1/filings",
            headers=headers,
            json={
                "job_type": "upload_compiled",
                "filing_type": source["filing_type"],
                "organization_id": source["organization_id"],
                "workspace_id": source["workspace_id"],
                "documents": [
                    {
                        "name": f"{case['case_id']}.pdf",
                        "download_url": source["download_url"],
                    }
                ],
            },
        )
        response.raise_for_status()
        accepted = response.json()
        poll_url = httpx.URL(str(accepted.get("poll_url") or f"/v1/jobs/{accepted['job_id']}"))
        if not poll_url.is_absolute_url:
            poll_url = httpx.URL(base).join(poll_url)
        deadline = asyncio.get_running_loop().time() + timeout_s
        job: dict[str, Any] = {}
        while asyncio.get_running_loop().time() < deadline:
            response = await http.get(str(poll_url), headers=headers)
            response.raise_for_status()
            job = response.json()
            status = str(job.get("status") or "").lower()
            if status == "completed":
                break
            if status == "failed":
                raise RuntimeError("production split job failed; inspect the authorized job record")
            await asyncio.sleep(poll_interval_s)
        else:
            raise TimeoutError("production split job did not complete before timeout")
        artifacts = ((job.get("result") or {}).get("artifacts") or {})
        split_artifact = artifacts.get("split")
        artifact_url = split_artifact.get("url") if isinstance(split_artifact, dict) else None
        if not artifact_url:
            raise RuntimeError("completed job did not expose a split artifact URL")
        response = await http.get(str(artifact_url))
        response.raise_for_status()
        artifact = response.json()
        if not isinstance(artifact, dict):
            raise TypeError("split artifact must contain a JSON object")
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(json.dumps(artifact, indent=2) + "\n", encoding="utf-8")
        return artifact
    finally:
        if owns_client:
            await http.aclose()


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--annotations", required=True, type=Path)
    parser.add_argument("--predictions-dir", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--run-existing-pipeline", action="store_true")
    parser.add_argument("--allow-remote-run", action="store_true")
    parser.add_argument("--api-base-url")
    parser.add_argument("--sources", type=Path, help="Secure case_id→source URL/filing/workspace mapping JSON")
    args = parser.parse_args(argv)
    try:
        annotations = _read_json(args.annotations)
        validate_annotations(annotations)
        if args.run_existing_pipeline:
            if not args.allow_remote_run or not args.api_base_url or not args.sources:
                raise ValueError(
                    "--run-existing-pipeline requires --allow-remote-run, --api-base-url, and --sources"
                )
            source_map = _read_json(args.sources)
            if not isinstance(source_map, dict):
                raise ValueError("--sources must be an object keyed by case_id")
            api_key = os.getenv("JUBEEX_API_KEY", "")
            if not api_key.strip():
                raise ValueError("--run-existing-pipeline requires JUBEEX_API_KEY authorization")
            for case in annotations["cases"]:
                case_id = case["case_id"]
                if case_id not in source_map or not isinstance(source_map[case_id], dict):
                    raise ValueError(f"--sources has no object for case_id {case_id!r}")
                asyncio.run(
                    run_existing_api_case(
                        case,
                        source_map[case_id],
                        api_base_url=args.api_base_url,
                        api_key=api_key,
                        output_path=args.predictions_dir / f"{case_id}.json",
                        allow_remote=True,
                    )
                )
        predictions = {
            case["case_id"]: _read_json(args.predictions_dir / f"{case['case_id']}.json")
            for case in annotations.get("cases", [])
        }
        report = evaluate_dataset(annotations, predictions)
    except (OSError, ValueError, TypeError, KeyError) as exc:
        print(f"split evaluation failed: {exc}", file=sys.stderr)
        return 2
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps({"cases": len(report["cases"]), "global_metrics": report["global_metrics"]}, indent=2))
    return 0


if __name__ == "__main__":  # pragma: no cover - exercised by CLI tests via main()
    raise SystemExit(main())
