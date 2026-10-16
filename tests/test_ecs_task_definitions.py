"""Deployment task definitions keep required runtime secrets wired."""

from __future__ import annotations

import json
from pathlib import Path

import pytest


REPO_ROOT = Path(__file__).resolve().parents[1]


@pytest.mark.parametrize(
    "definition_name",
    [
        "worker-task-definition.json",
        "ecs-scrutiny-worker-task.json",
        "ecs-scrutiny-api-task.json",
    ],
)
def test_scrutiny_capable_tasks_receive_catalogue_database_url(
    definition_name: str,
) -> None:
    definition = json.loads((REPO_ROOT / definition_name).read_text(encoding="utf-8"))
    container = definition["containerDefinitions"][0]

    plain_environment = {
        entry["name"]: entry.get("value") for entry in container.get("environment", [])
    }
    secret_environment = {
        entry["name"]: entry.get("valueFrom") for entry in container.get("secrets", [])
    }

    assert "CATALOGUE_DATABASE_URL" not in plain_environment
    assert secret_environment.get("CATALOGUE_DATABASE_URL", "").endswith(
        ":CATALOGUE_DATABASE_URL::"
    )


@pytest.mark.parametrize(
    "definition_name",
    ["worker-task-definition.json", "ecs-scrutiny-worker-task.json"],
)
def test_workers_prepare_google_adc_before_starting(definition_name: str) -> None:
    definition = json.loads((REPO_ROOT / definition_name).read_text(encoding="utf-8"))
    container = definition["containerDefinitions"][0]

    assert container["entryPoint"] == ["/app/scripts/start.sh"]
    assert container["command"] == ["uv", "run", "jubeex-sqs-worker"]
