"""AWS SES delivery for batch QA reports."""

from __future__ import annotations

from email.message import EmailMessage
from pathlib import Path

import boto3

from .models import BatchResult


def send_report_email(
    batch: BatchResult,
    *,
    sender: str,
    recipients: list[str],
    region: str,
    html_path: Path,
    xlsx_path: Path,
) -> str:
    message = EmailMessage()
    message["Subject"] = (
        f"JubeeX batch QA {batch.run_id}: "
        f"{batch.completed_count} completed, {batch.failed_count} failed"
    )
    message["From"] = sender
    message["To"] = ", ".join(recipients)
    message.set_content(
        f"JubeeX batch QA {batch.run_id}\n"
        f"Completed: {batch.completed_count}\nFailed: {batch.failed_count}\n"
        "See the attached report."
    )
    message.add_alternative(html_path.read_text(encoding="utf-8"), subtype="html")
    message.add_attachment(
        xlsx_path.read_bytes(),
        maintype="application",
        subtype="vnd.openxmlformats-officedocument.spreadsheetml.sheet",
        filename=xlsx_path.name,
    )
    response = boto3.client("ses", region_name=region).send_raw_email(
        Source=sender,
        Destinations=recipients,
        RawMessage={"Data": message.as_bytes()},
    )
    return str(response.get("MessageId") or "")
