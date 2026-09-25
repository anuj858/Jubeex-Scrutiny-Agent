"""Read-only catalogue loader for the shared Postgres views.

Parse never writes these tables. ``CATALOGUE_SOURCE=file`` (the default) keeps
the JSON catalogue. ``CATALOGUE_SOURCE=db`` reads ``v_parse_*`` with
``CATALOGUE_DATABASE_URL``. A failed database read falls back to the JSON file.
"""

from __future__ import annotations

import os
from datetime import date, datetime
from typing import Any

from .rules import (
    Catalogue,
    CatalogueSource,
    Defect,
    check_id_sort_key,
    normalize_filing_type,
    rewrite_location_source,
)


def catalogue_uses_database() -> bool:
    return os.getenv("CATALOGUE_SOURCE", "file").strip().lower() == "db"


def catalogue_database_url() -> str:
    url = (os.getenv("CATALOGUE_DATABASE_URL") or "").strip()
    if not url:
        raise RuntimeError(
            "CATALOGUE_SOURCE=db requires CATALOGUE_DATABASE_URL "
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


def load_catalogue_from_db(revision: int | None = None) -> Catalogue:
    if revision is None:
        revision = fetch_catalogue_revision()
    with _connect() as conn, conn.cursor() as cur:
        cur.execute("SELECT * FROM v_parse_defects")
        defect_cols = [desc.name for desc in cur.description]
        defect_rows = [dict(zip(defect_cols, row, strict=True)) for row in cur.fetchall()]
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
        context_parts=_string_list(row.get("context_parts")),
        exclude_parts=_optional_string_list(row.get("exclude_parts")),
        how_to_cure=_string_list(row.get("how_to_cure")),
        applicable_rule=_text(row.get("applicable_rule")),
        location_source=str(row.get("location_source") or ""),
        notes=_text(row.get("notes")),
        ivan_comment=_text(row.get("ivan_comment")),
        notes_2=_text(row.get("notes_2")),
        applies_to_all_petition_types=bool(row.get("applies_to_all_petition_types")),
        agent_filing_types=filing_types,
        special_category_mode=_text(row.get("special_category_mode")),
        special_category_key="",
        is_enabled=bool(row.get("is_enabled", True)),
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
