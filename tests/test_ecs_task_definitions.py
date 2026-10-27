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


def test_api_prepares_google_adc_before_starting() -> None:
    dockerfile = (REPO_ROOT / "Dockerfile.api").read_text(encoding="utf-8")
    assert 'ENTRYPOINT ["/app/scripts/start.sh"]' in dockerfile
    assert 'CMD ["uv", "run", "uvicorn", "extraction_review.api:app"' in dockerfile


@pytest.mark.parametrize(
    "definition_name", ["worker-task-definition.json", "ecs-scrutiny-worker-task.json"]
)
def test_workers_use_shared_vertex_pacing(definition_name: str) -> None:
    definition = json.loads(
        (REPO_ROOT / definition_name).read_text(encoding="utf-8")
    )
    environment = {
        entry["name"]: entry.get("value")
        for entry in definition["containerDefinitions"][0].get("environment", [])
    }

    assert environment["VERTEX_SHARED_RATE_LIMIT"] == "true"
    assert environment["VERTEX_REQUESTS_PER_MINUTE"] == "8"
    assert environment["AWS_S3_BUCKET"]
    assert environment["VERTEX_MAX_ATTEMPTS"] == "6"
    assert environment["SCRUTINY_CONCURRENCY"] == "8"
    assert environment["SCRUTINY_CATALOGUE_SOURCE"] == "json"
    assert environment["SCRUTINY_DEFECTS"] == "all"
    assert environment["OPENROUTER_VISION_ENABLED"] == "true"
