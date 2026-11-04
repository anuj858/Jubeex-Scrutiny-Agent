#!/usr/bin/env python3
"""Generate a guarded SQL backfill for scrutiny inspect-part links.

The catalogue keeps logical/atomic inspect labels while the filing catalogue may
represent multiple logical documents with one combined frontend document type.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_CATALOGUE = ROOT / "scrutiny_harness/rules/sci_registry_defects.v1.json"
DEFAULT_OUTPUT = ROOT / "outputs/db/sync_inspect_parts_from_catalogue.sql"

DOCUMENT_TYPE_CODES = {
    "Advocate's Checklist": "advocate-s-checklist",
    "Affidavit": "affidavit",
    "Annexures": "annexures",
    "AOR's Certificate": "aor-s-certificate",
    "AOR's Declaration": "aor-s-declaration",
    "Appendix": "appendix",
    "Application": "application",
    "Cover Page": "cover-page",
    "Custody Certificate": "custody-certificate",
    "Filing Memo": "filing-memo",
    "Impugned Order": "impugned-order",
    "Index": "index",
    "Listing Proforma": "listing-proforma",
    "Main Petition": "petition",
    "Memo of Appearance": "vakalatnama",
    "Memo of Parties": "memo-of-parties",
    "Office Report on Limitation": "office-report-on-limitation",
    "PoA/BR": "poa-board-resolution",
    "Record of Proceedings": "record-of-proceedings",
    "Synopsis": "synopsis",
    "List of Dates & Events": "synopsis",
    "Vakalatnama": "vakalatnama",
}


def sql_text(value: str) -> str:
    return "'" + value.replace("'", "''") + "'"


def values_block(rows: list[tuple[object, ...]]) -> str:
    rendered = []
    for row in rows:
        values = [str(value) if isinstance(value, int) else sql_text(str(value)) for value in row]
        rendered.append("  (" + ", ".join(values) + ")")
    return ",\n".join(rendered)


def generate(catalogue_path: Path) -> str:
    payload = json.loads(catalogue_path.read_text(encoding="utf-8"))
    defects = payload.get("defects")
    if not isinstance(defects, list) or not defects:
        raise ValueError("Catalogue does not contain a non-empty defects array")

    target_rows: list[tuple[str]] = []
    link_rows: list[tuple[str, str, str, int]] = []
    seen_codes: set[str] = set()
    for defect in defects:
        code = str(defect.get("check_id") or "").strip()
        if not code or code in seen_codes:
            raise ValueError(f"Missing or repeated defect code: {code!r}")
        seen_codes.add(code)
        target_rows.append((code,))
        parts = defect.get("inspect_parts")
        if not isinstance(parts, list) or not parts:
            raise ValueError(f"{code} has no inspect_parts")
        for order, raw_label in enumerate(parts):
            label = str(raw_label).strip()
            document_type_code = DOCUMENT_TYPE_CODES.get(label)
            if document_type_code is None:
                raise ValueError(f"{code} has unmapped inspect label: {label!r}")
            link_rows.append((code, label, document_type_code, order))

    return f"""\\set ON_ERROR_STOP on

-- Generated from: {catalogue_path}
-- Defects: {len(target_rows)}; desired inspect links: {len(link_rows)}
-- This script changes only role='inspect' associations, defect version metadata,
-- and the catalogue revision. Frontend document_types names are not changed.

BEGIN;

LOCK TABLE hub_defect_document_types IN SHARE ROW EXCLUSIVE MODE;

CREATE TEMP TABLE desired_inspect_defects (
  defect_code text PRIMARY KEY
) ON COMMIT PRESERVE ROWS;

INSERT INTO desired_inspect_defects (defect_code) VALUES
{values_block(target_rows)};

CREATE TEMP TABLE desired_inspect_links (
  defect_code text NOT NULL,
  label text NOT NULL,
  document_type_code text NOT NULL,
  display_order integer NOT NULL,
  PRIMARY KEY (defect_code, label)
) ON COMMIT PRESERVE ROWS;

INSERT INTO desired_inspect_links (
  defect_code, label, document_type_code, display_order
) VALUES
{values_block(link_rows)};

-- Fail before modifying data if the deployed DB catalogue is incomplete.
DO $$
DECLARE
  missing_defects text;
  missing_document_types text;
BEGIN
  SELECT string_agg(target.defect_code, ', ' ORDER BY target.defect_code)
    INTO missing_defects
    FROM desired_inspect_defects AS target
    LEFT JOIN hub_defects AS defect ON defect.code = target.defect_code
   WHERE defect.id IS NULL;

  IF missing_defects IS NOT NULL THEN
    RAISE EXCEPTION 'Missing hub_defects rows: %', missing_defects;
  END IF;

  SELECT string_agg(missing.document_type_code, ', ' ORDER BY missing.document_type_code)
    INTO missing_document_types
    FROM (
      SELECT DISTINCT desired.document_type_code
        FROM desired_inspect_links AS desired
        LEFT JOIN document_types AS document_type
          ON document_type.code = desired.document_type_code
         AND document_type.organization_id IS NULL
       WHERE document_type.id IS NULL
    ) AS missing;

  IF missing_document_types IS NOT NULL THEN
    RAISE EXCEPTION 'Missing shared document_types rows: %', missing_document_types;
  END IF;
END $$;

-- Remove only the inspect links covered by this catalogue. Context links and
-- frontend document types remain untouched.
DELETE FROM hub_defect_document_types AS link
USING hub_defects AS defect, desired_inspect_defects AS target
WHERE link.defect_id = defect.id
  AND defect.code = target.defect_code
  AND link.role = 'inspect';

INSERT INTO hub_defect_document_types (
  id, defect_id, document_type_id, role, label, display_order
)
SELECT
  gen_random_uuid(),
  defect.id,
  document_type.id,
  'inspect',
  desired.label,
  desired.display_order
FROM desired_inspect_links AS desired
JOIN hub_defects AS defect ON defect.code = desired.defect_code
JOIN document_types AS document_type
  ON document_type.code = desired.document_type_code
 AND document_type.organization_id IS NULL
ORDER BY desired.defect_code, desired.display_order;

-- Follow the existing catalogue migrations: mark affected defects as changed
-- and invalidate worker-side catalogue caches.
UPDATE hub_defects AS defect
   SET version = defect.version + 1,
       updated_at = now()
  FROM desired_inspect_defects AS target
 WHERE defect.code = target.defect_code;

UPDATE hub_catalogue_state
   SET revision = revision + 1
 WHERE id = 1;

-- Final assertions run inside the transaction.
DO $$
DECLARE
  expected_count integer;
  actual_count integer;
BEGIN
  SELECT count(*) INTO expected_count FROM desired_inspect_links;
  SELECT count(*)
    INTO actual_count
    FROM hub_defect_document_types AS link
    JOIN hub_defects AS defect ON defect.id = link.defect_id
    JOIN desired_inspect_defects AS target ON target.defect_code = defect.code
   WHERE link.role = 'inspect';

  IF actual_count <> expected_count THEN
    RAISE EXCEPTION 'Inspect-link count mismatch: expected %, found %',
      expected_count, actual_count;
  END IF;
END $$;

COMMIT;

-- Verification report returned after commit.
SELECT
  defect.code,
  link.label AS logical_inspect_target,
  document_type.code AS document_type_code,
  document_type.name AS frontend_section_name,
  link.display_order
FROM hub_defect_document_types AS link
JOIN hub_defects AS defect ON defect.id = link.defect_id
JOIN document_types AS document_type ON document_type.id = link.document_type_id
WHERE link.role = 'inspect'
  AND defect.code IN (SELECT defect_code FROM desired_inspect_defects)
ORDER BY defect.code, link.display_order;

DROP TABLE desired_inspect_links;
DROP TABLE desired_inspect_defects;
"""


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--catalogue", type=Path, default=DEFAULT_CATALOGUE)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()
    sql = generate(args.catalogue.resolve())
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(sql, encoding="utf-8")
    print(f"Wrote {args.output} ({len(sql):,} characters)")


if __name__ == "__main__":
    main()
