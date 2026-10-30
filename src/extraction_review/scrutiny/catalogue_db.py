"""Read-only catalogue loader for the shared Postgres views.

Parse never writes these tables. Defect checks always read ``v_parse_*`` with
``CATALOGUE_DATABASE_URL``.
"""

from __future__ import annotations

import json
import logging
import os
from datetime import date, datetime
from typing import Any

from psycopg import errors

from .rules import (
    Catalogue,
    CatalogueSource,
    Defect,
    check_id_sort_key,
    normalize_filing_type,
    rewrite_location_source,
)

logger = logging.getLogger(__name__)


def catalogue_database_url() -> str:
    url = (os.getenv("CATALOGUE_DATABASE_URL") or "").strip()
    if not url:
        raise RuntimeError(
            "CATALOGUE_DATABASE_URL is required "
            "(read-only role, SELECT on v_parse_* and hub_catalogue_state)"
        )
    return url


def _connect():
    import psycopg

    return psycopg.connect(catalogue_database_url(), connect_timeout=10)


def fetch_catalogue_revision() -> int:
    with _connect() as conn:
        row = conn.execute(
            "SELECT revision FROM hub_catalogue_state WHERE id = 1"
        ).fetchone()
    if row is None or row[0] is None:
        return 0
    return int(row[0])


_ACTIVE_COLUMNS = ("is_active", "active")
_STATUS_COLUMN = "status"
_ORDER_COLUMNS = ("display_order", "sort_order", "position")
_FLAG_TRUE = frozenset({"1", "true", "t", "yes", "y", "active", "enabled"})
_FLAG_FALSE = frozenset({"0", "false", "f", "no", "n", "inactive", "disabled"})
_FLAG_ALL = _FLAG_TRUE | _FLAG_FALSE


def _column_name(columns: list[str], *candidates: str) -> str | None:
    lookup = {column.lower(): column for column in columns}
    for candidate in candidates:
        found = lookup.get(candidate)
        if found:
            return found
    return None


def _flag_is_true(value: object) -> bool:
    if isinstance(value, bool):
        return value
    if isinstance(value, (int, float)):
        return value != 0
    return str(value or "").strip().lower() in _FLAG_TRUE


def _usable_flag_column(rows: list[dict[str, Any]], column: str) -> bool:
    seen = [
        str(row.get(column)).strip().lower()
        for row in rows
        if row.get(column) is not None and str(row.get(column)).strip()
    ]
    return bool(seen) and all(item in _FLAG_ALL for item in seen)


def special_category_names_from_rows(
    rows: list[dict[str, Any]],
    columns: list[str] | None = None,
) -> list[str]:
    """Exact special_category strings, active rows first when that column exists."""
    columns = list(columns or (list(rows[0]) if rows else []))
    active_column = _column_name(columns, *_ACTIVE_COLUMNS)
    status_column = _column_name(columns, _STATUS_COLUMN)
    order_column = _column_name(columns, *_ORDER_COLUMNS)
    flag_column = active_column
    if flag_column is None and status_column and _usable_flag_column(rows, status_column):
        flag_column = status_column
    kept = [
        row
        for row in rows
        if flag_column is None or _flag_is_true(row.get(flag_column))
    ]
    ordered = list(enumerate(kept))
    if order_column:
        def sort_key(item: tuple[int, dict[str, Any]]) -> tuple[int, int, int]:
            index, row = item
            raw = row.get(order_column)
            try:
                return (0, int(raw), index)
            except (TypeError, ValueError):
                return (1, index, index)

        ordered.sort(key=sort_key)
    names: list[str] = []
    for _index, row in ordered:
        label = str(row.get("special_category") or "").strip()
        if label and label not in names:
            names.append(label)
    return names


def _names_from_case_types(case_types: dict[str, Any]) -> list[str]:
    """Exact special-category labels from the local case-type catalogue."""
    names: list[str] = []
    for spec in case_types.values():
        if not isinstance(spec, dict):
            continue
        for label in spec.get("special_categories") or []:
            text = str(label).strip()
            if text and text not in names:
                names.append(text)
    return names


def fetch_special_category_names() -> list[str]:
    """Distinct catalogue names, with the stored spelling unchanged.

    ``CATALOGUE_DATABASE_URL=local`` uses the checked-in case-type list and
    does not open a database. A real URL reads ``v_parse_petition_specials``.
    Extract-front does not substitute the old hardcoded list.
    """
    from .rules import _use_json_catalogue, get_case_types

    file_catalogue = _use_json_catalogue() or (
        (os.getenv("SCRUTINY_CATALOGUE_SOURCE") or "db").strip().lower() == "json"
    )
    if file_catalogue:
        names = _names_from_case_types(get_case_types())
        if not names:
            raise RuntimeError("Local catalogue has no special categories")
        return names
    with _connect() as conn, conn.cursor() as cur:
        cur.execute("SELECT * FROM v_parse_petition_specials")
        columns = [desc.name for desc in cur.description]
        rows = [dict(zip(columns, row, strict=True)) for row in cur.fetchall()]
    names = special_category_names_from_rows(rows, columns)
    if not names:
        raise RuntimeError(
            "Catalogue database returned no special categories "
            "from v_parse_petition_specials"
        )
    return names


def load_catalogue_from_db(revision: int | None = None) -> Catalogue:
    if revision is None:
        revision = fetch_catalogue_revision()
    with _connect() as conn, conn.cursor() as cur:
        cur.execute("SELECT * FROM v_parse_defects")
        defect_cols = [desc.name for desc in cur.description]
        defect_rows = [dict(zip(defect_cols, row, strict=True)) for row in cur.fetchall()]
        _attach_remediation_config(cur, defect_rows)
        cur.execute("SELECT * FROM v_parse_sources")
        source_cols = [desc.name for desc in cur.description]
        source_rows = [dict(zip(source_cols, row, strict=True)) for row in cur.fetchall()]
        cur.execute("SELECT * FROM v_parse_petition_specials")
        special_cols = [desc.name for desc in cur.description]
        special_rows = [dict(zip(special_cols, row, strict=True)) for row in cur.fetchall()]
    return catalogue_from_rows(defect_rows, source_rows, special_rows, revision)


def catalogue_from_rows(
    defect_rows: list[dict[str, Any]],
    source_rows: list[dict[str, Any]],
    special_rows: list[dict[str, Any]],
    revision: int,
) -> Catalogue:
    sources = [_source_from_row(row) for row in source_rows]
    defects = []
    for row in sorted(defect_rows, key=lambda item: check_id_sort_key(str(item.get("check_id") or ""))):
        defect = _defect_from_row(row)
        original = defect.location_source
        defects.append(
            defect.model_copy(
                update={
                    "location_text": original,
                    "location_source": rewrite_location_source(original, sources),
                }
            )
        )
    specials: dict[str, list[str]] = {}
    for row in special_rows:
        filing = normalize_filing_type(str(row.get("agent_filing_type") or ""))
        label = str(row.get("special_category") or "").strip()
        if not filing or not label:
            continue
        bucket = specials.setdefault(filing, [])
        if label not in bucket:
            bucket.append(label)
    return Catalogue(
        catalogue_id="sci_registry_defects",
        schema_version="db",
        catalogue_version=f"db:{revision}",
        jurisdiction="sci",
        sources=sources,
        defects=defects,
        petition_specials=specials,
    )


def _attach_remediation_config(cur: Any, defect_rows: list[dict[str, Any]]) -> None:
    """Copy hub_defects.remediation_config onto rows the view left behind."""
    if not defect_rows or "remediation_config" in defect_rows[0]:
        return
    codes = [
        str(row.get("check_id") or "").strip()
        for row in defect_rows
        if str(row.get("check_id") or "").strip()
    ]
    if not codes:
        return
    try:
        # A nested transaction creates a savepoint, so a denied optional lookup
        # cannot poison the transaction used for the remaining catalogue views.
        with cur.connection.transaction():
            cur.execute(
                "SELECT code, remediation_config FROM hub_defects WHERE code = ANY(%s)",
                (codes,),
            )
            by_code = {str(code): config for code, config in cur.fetchall()}
    except errors.InsufficientPrivilege:
        logger.warning(
            "[Scrutiny] Catalogue reader lacks remediation_config access; "
            "continuing without remediation actions until migration 0085 is applied"
        )
        return
    for row in defect_rows:
        code = str(row.get("check_id") or "").strip()
        if code in by_code:
            row["remediation_config"] = by_code[code]


def _remediation_config(value: Any) -> dict[str, Any]:
    if value is None:
        return {}
    if isinstance(value, str):
        text = value.strip()
        if not text:
            return {}
        try:
            value = json.loads(text)
        except json.JSONDecodeError:
            return {}
    if isinstance(value, dict):
        return value
    return {}


def _string_list(value: Any) -> list[str]:
    if value is None:
        return []
    if isinstance(value, str):
        text = value.strip()
        return [text] if text else []
    return [str(item).strip() for item in value if str(item).strip()]


def _optional_string_list(value: Any) -> list[str] | None:
    if value is None:
        return None
    return _string_list(value)


def _trigger_words(value: Any) -> str | None:
    if value is None:
        return None
    if isinstance(value, list):
        parts = [str(item).strip() for item in value if str(item).strip()]
        return "; ".join(parts) or None
    text = str(value).strip()
    return text or None


def _text(value: Any) -> str | None:
    if value is None:
        return None
    if isinstance(value, (date, datetime)):
        return value.isoformat()
    text = str(value).strip()
    return text or None


def _defect_from_row(row: dict[str, Any]) -> Defect:
    special_name = _text(row.get("special_category"))
    filing_types = [
        normalize_filing_type(str(item))
        for item in _string_list(row.get("agent_filing_types"))
        if normalize_filing_type(str(item))
    ]
    return Defect(
        check_id=str(row.get("check_id") or "").strip(),
        main_category=str(row.get("main_category") or ""),
        special_category=special_name,
        parent_check_id=_text(row.get("parent_check_id")),
        overlap_note=_text(row.get("overlap_note")),
        defect=str(row.get("defect") or ""),
        requirement=str(row.get("requirement") or ""),
        trigger_words=_trigger_words(row.get("trigger_words")),
        where_to_look=_string_list(row.get("where_to_look")),
        inspect_parts=_string_list(row.get("inspect_parts")),
        manifest_absence_proves_defect=bool(
            row.get("missing_document_is_defect")
            or row.get("manifest_absence_proves_defect")
        ),
        legal_verification_status=str(
            row.get("legal_verification_status") or "not_required"
        ).strip().lower(),
        context_parts=_string_list(row.get("context_parts")),
        exclude_parts=_optional_string_list(row.get("exclude_parts")),
        how_to_cure=_string_list(row.get("how_to_cure")),
        remediation_config=_remediation_config(row.get("remediation_config")),
        applicable_rule=_text(row.get("applicable_rule")),
        location_source=str(row.get("location_source") or ""),
        notes=_text(row.get("notes")),
        ivan_comment=_text(row.get("ivan_comment")),
        notes_2=_text(row.get("notes_2")),
        ai_note=_text(row.get("ai_note")),
        applies_to_all_petition_types=bool(row.get("applies_to_all_petition_types")),
        is_global=(
            bool(row.get("is_global"))
            if "is_global" in row
            else bool(row.get("applies_to_all_petition_types"))
        ),
        agent_filing_types=filing_types,
        special_category_mode=_text(row.get("special_category_mode")),
        special_category_key="",
        is_enabled=bool(row.get("is_enabled", True)),
        selection_source="db",
        release_stage=str(row.get("release_stage") or "production"),
        court_code=_text(row.get("court_code")),
        inspect_part_groups=_string_list(row.get("inspect_part_groups")),
        defect_version=int(row.get("defect_version") or 1),
        use_explicit_applicability=True,
    )


def _source_from_row(row: dict[str, Any]) -> CatalogueSource:
    locators = row.get("locators") or {}
    if not isinstance(locators, dict):
        locators = {}
    return CatalogueSource(
        source_id=str(row.get("source_id") or ""),
        title=str(row.get("title") or ""),
        authority_type=str(row.get("authority_type") or ""),
        url=_text(row.get("url")),
        issued_date=_text(row.get("issued_date")),
        effective_date=_text(row.get("effective_date")),
        checksum=_text(row.get("checksum")),
        locators={str(key): str(value) for key, value in locators.items()},
        alternate_urls=_string_list(row.get("alternate_urls")),
        filename_aliases=_string_list(row.get("filename_aliases")),
    )
