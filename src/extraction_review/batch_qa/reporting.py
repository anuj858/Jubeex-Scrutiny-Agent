"""JSON, HTML, CSV, and Excel reports for deployed QA batches."""

from __future__ import annotations

import csv
import html
import json
from collections.abc import Iterable
from pathlib import Path
from typing import Any

import xlsxwriter
from xlsxwriter.format import Format

from .evaluate import finding_code, finding_status, report_findings
from .models import BatchResult, CaseResult


def _text(value: Any) -> str:
    if value is None:
        return ""
    if isinstance(value, (dict, list, tuple)):
        rendered = json.dumps(value, ensure_ascii=False, sort_keys=True)
    else:
        rendered = str(value)
    return rendered if len(rendered) <= 32000 else f"{rendered[:31980]}… [truncated]"


def _flatten(value: Any, prefix: str = "") -> Iterable[tuple[str, Any]]:
    if isinstance(value, dict):
        for key, item in value.items():
            path = f"{prefix}.{key}" if prefix else str(key)
            yield from _flatten(item, path)
        return
    if isinstance(value, list):
        yield prefix, _text(value)
        return
    yield prefix, value


def _part_rows(batch: BatchResult) -> Iterable[list[Any]]:
    for case in batch.cases:
        for part in case.split_parts:
            yield [
                case.filename,
                part.get("slot_id"),
                part.get("label"),
                _text(part.get("page_span")),
                part.get("filename") or part.get("name"),
                part.get("reason"),
                bool(part.get("file_id")),
            ]


def _defect_rows(batch: BatchResult) -> Iterable[list[Any]]:
    for case in batch.cases:
        for finding in report_findings(case.scrutiny_report):
            yield [
                case.filename,
                finding_code(finding),
                finding_status(finding),
                finding.get("defect") or finding.get("title"),
                finding.get("document")
                or finding.get("document_name")
                or finding.get("location"),
                finding.get("page")
                or finding.get("page_number")
                or _first_evidence_page(finding),
                finding.get("reasoning") or finding.get("reason"),
                _text(finding.get("evidence")),
            ]


def _first_evidence_page(finding: dict[str, Any]) -> Any:
    evidence = finding.get("evidence")
    if isinstance(evidence, list):
        for item in evidence:
            if isinstance(item, dict) and item.get("page") is not None:
                return item["page"]
    return None


def write_json(batch: BatchResult, path: Path) -> None:
    path.write_text(
        json.dumps(batch.as_dict(), ensure_ascii=False, indent=2), encoding="utf-8"
    )


def _duration(case: CaseResult, stage: str) -> float | None:
    result = case.stages.get(stage)
    return result.duration_seconds if result else None


def write_csv(batch: BatchResult, path: Path) -> None:
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(
            [
                "file",
                "status",
                "verification",
                "pages",
                "filing_type",
                "split_seconds",
                "extraction_seconds",
                "assemble_scrutiny_seconds",
                "total_seconds",
                "split_parts",
                "defects",
                "first_failure_stage",
                "error",
            ]
        )
        for case in batch.cases:
            writer.writerow(
                [
                    case.filename,
                    case.status,
                    case.verification_status,
                    case.page_count,
                    case.filing_type,
                    _duration(case, "split"),
                    _duration(case, "extraction"),
                    _duration(case, "assemble_scrutiny"),
                    case.total_seconds,
                    len(case.split_parts),
                    len(report_findings(case.scrutiny_report)),
                    case.first_failure_stage,
                    case.error,
                ]
            )


def write_html(batch: BatchResult, path: Path) -> None:
    rows = []
    for case in batch.cases:
        rows.append(
            "<tr>"
            f"<td>{html.escape(case.filename)}</td>"
            f"<td>{html.escape(case.status)}</td>"
            f"<td>{html.escape(case.verification_status)}</td>"
            f"<td>{case.page_count}</td>"
            f"<td>{len(case.split_parts)}</td>"
            f"<td>{len(report_findings(case.scrutiny_report))}</td>"
            f"<td>{case.total_seconds or 0:.1f}</td>"
            f"<td>{html.escape(case.error or '')}</td>"
            "</tr>"
        )
    document = f"""<!doctype html>
<html lang="en"><head><meta charset="utf-8"><title>JubeeX batch QA</title>
<style>
body{{font-family:Arial,sans-serif;color:#171717;margin:32px}}
h1{{font-size:24px;margin-bottom:4px}} p{{color:#666}}
table{{border-collapse:collapse;width:100%;font-size:13px}}
th{{background:#171717;color:white;text-align:left;padding:9px}}
td{{border-bottom:1px solid #ddd;padding:8px;vertical-align:top}}
</style></head><body>
<h1>JubeeX batch QA</h1>
<p>Run {html.escape(batch.run_id)}. {batch.completed_count} completed, 
{batch.failed_count} failed.</p>
<table><thead><tr><th>File</th><th>Run status</th><th>Verification</th>
<th>Pages</th><th>Split parts</th><th>Findings</th><th>Total seconds</th>
<th>Error</th></tr></thead><tbody>{"".join(rows)}</tbody></table>
</body></html>"""
    path.write_text(document, encoding="utf-8")


def _write_table_sheet(
    workbook: xlsxwriter.Workbook,
    name: str,
    columns: list[str],
    rows: list[list[Any]],
    header: Format,
    wrapped: Format,
    *,
    widths: list[int],
) -> None:
    sheet = workbook.add_worksheet(name)
    sheet.hide_gridlines(2)
    sheet.write_row(0, 0, columns, header)
    for row_number, row in enumerate(rows, start=1):
        for column_number, value in enumerate(row):
            sheet.write(row_number, column_number, value, wrapped)
    sheet.autofilter(0, 0, max(1, len(rows)), len(columns) - 1)
    sheet.freeze_panes(1, 1)
    for index, width in enumerate(widths):
        sheet.set_column(index, index, width)


def write_xlsx(batch: BatchResult, path: Path) -> None:
    workbook = xlsxwriter.Workbook(path)
    workbook.set_properties(
        {"title": "JubeeX batch QA", "subject": f"Run {batch.run_id}"}
    )
    header = workbook.add_format(
        {
            "bold": True,
            "font_color": "#FFFFFF",
            "bg_color": "#171717",
            "align": "center",
            "valign": "vcenter",
        }
    )
    title = workbook.add_format({"bold": True, "font_size": 16})
    note = workbook.add_format({"font_color": "#666666", "italic": True})
    number = workbook.add_format({"num_format": "0.0"})
    wrapped = workbook.add_format({"text_wrap": True, "valign": "top"})
    failed = workbook.add_format({"font_color": "#C62828", "bg_color": "#FFEBEE"})
    passed = workbook.add_format({"font_color": "#216E39", "bg_color": "#E6F4EA"})

    summary = workbook.add_worksheet("Summary")
    summary.hide_gridlines(2)
    summary.write("A1", "JubeeX batch QA", title)
    summary.write("A2", f"Run {batch.run_id} · API {batch.api_url}", note)
    summary.write_row(
        "A4",
        [
            "File",
            "Run status",
            "Verification",
            "Pages",
            "Filing type",
            "Split seconds",
            "Extraction seconds",
            "Assemble + scrutiny seconds",
            "Total seconds",
            "Split parts",
            "Findings",
            "First failure stage",
            "Error",
        ],
        header,
    )
    for row, case in enumerate(batch.cases, start=4):
        values = [
            case.filename,
            case.status,
            case.verification_status,
            case.page_count,
            case.filing_type,
            _duration(case, "split"),
            _duration(case, "extraction"),
            _duration(case, "assemble_scrutiny"),
            case.total_seconds,
            len(case.split_parts),
            len(report_findings(case.scrutiny_report)),
            case.first_failure_stage,
            case.error,
        ]
        summary.write_row(row, 0, values)
        for column in (5, 6, 7, 8):
            if isinstance(values[column], (int, float)):
                summary.write_number(row, column, values[column], number)
    last_summary_row = max(4, 3 + len(batch.cases))
    summary.autofilter(3, 0, last_summary_row, 12)
    summary.freeze_panes(4, 1)
    summary.set_column("A:A", 32)
    summary.set_column("B:C", 15)
    summary.set_column("D:L", 18)
    summary.set_column("M:M", 48, wrapped)
    summary.conditional_format(
        4,
        1,
        last_summary_row,
        2,
        {
            "type": "text",
            "criteria": "containing",
            "value": "failed",
            "format": failed,
        },
    )
    summary.conditional_format(
        4,
        1,
        last_summary_row,
        2,
        {
            "type": "text",
            "criteria": "containing",
            "value": "passed",
            "format": passed,
        },
    )

    _write_table_sheet(
        workbook,
        "Split results",
        [
            "File",
            "Slot",
            "Label",
            "Page ranges",
            "Output file",
            "Reason",
            "File ID available",
        ],
        list(_part_rows(batch)),
        header,
        wrapped,
        widths=[32, 24, 26, 24, 34, 54, 18],
    )
    _write_table_sheet(
        workbook,
        "All defects",
        [
            "File",
            "Defect code",
            "Status",
            "Title",
            "Document",
            "Page",
            "Reasoning",
            "Evidence",
        ],
        list(_defect_rows(batch)),
        header,
        wrapped,
        widths=[32, 14, 16, 44, 28, 10, 64, 64],
    )
    extraction_rows = [
        [case.filename, path, _text(value)]
        for case in batch.cases
        for path, value in _flatten(case.extraction_data)
    ]
    _write_table_sheet(
        workbook,
        "Extraction data",
        ["File", "Field path", "Extracted value"],
        extraction_rows,
        header,
        wrapped,
        widths=[32, 54, 100],
    )
    timing_rows = []
    for case in batch.cases:
        for stage_name, stage in case.stages.items():
            timing_rows.append(
                [
                    case.filename,
                    stage_name,
                    stage.status,
                    stage.duration_seconds,
                    stage.job_id,
                    stage.error,
                    _text(stage.observations),
                ]
            )
    _write_table_sheet(
        workbook,
        "Timings",
        [
            "File",
            "Stage",
            "Status",
            "Seconds",
            "Job ID",
            "Error",
            "Progress observations",
        ],
        timing_rows,
        header,
        wrapped,
        widths=[32, 22, 14, 12, 38, 54, 80],
    )
    comparison_rows = []
    for case in batch.cases:
        for category, result in (
            ("split", case.split_evaluation),
            ("extraction", case.extraction_evaluation),
            ("scrutiny", case.scrutiny_evaluation),
        ):
            for difference in result.get("differences", []):
                comparison_rows.append(
                    [case.filename, category, result.get("status"), _text(difference)]
                )
    _write_table_sheet(
        workbook,
        "Expected comparison",
        ["File", "Category", "Status", "Difference"],
        comparison_rows,
        header,
        wrapped,
        widths=[32, 18, 16, 100],
    )
    workbook.close()


def write_all_reports(batch: BatchResult, report_dir: Path) -> dict[str, Path]:
    report_dir.mkdir(parents=True, exist_ok=True)
    paths = {
        "json": batch.output_path(report_dir, "json"),
        "csv": batch.output_path(report_dir, "csv"),
        "html": batch.output_path(report_dir, "html"),
        "xlsx": batch.output_path(report_dir, "xlsx"),
    }
    write_json(batch, paths["json"])
    write_csv(batch, paths["csv"])
    write_html(batch, paths["html"])
    write_xlsx(batch, paths["xlsx"])
    return paths
