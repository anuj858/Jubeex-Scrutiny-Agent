from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from scripts.render_ecs_task_for_environment import render


def test_render_rewrites_environment_and_secret_references() -> None:
    payload = {
        "taskDefinition": {
            "family": "old",
            "revision": 4,
            "status": "ACTIVE",
            "containerDefinitions": [
                {
                    "name": "worker",
                    "image": "old:tag",
                    "environment": [
                        {"name": "AWS_S3_BUCKET", "value": "old-bucket"},
                        {"name": "JUBEEX_CALLBACK_URL", "value": "old-callback"},
                        {"name": "SCRUTINY_DEFECTS", "value": "D-1"},
                    ],
                    "secrets": [
                        {"name": "OPENROUTER_API_KEY", "valueFrom": "old"},
                        {"name": "LLAMA_CLOUD_API_KEY", "valueFrom": "old"},
                        {"name": "LLAMA_CLOUD_TIER", "valueFrom": "old"},
                        {"name": "JUBEEX_CALLBACK_URL", "valueFrom": "old"},
                        {"name": "JUBEEX_API_KEY", "valueFrom": "old"},
                    ],
                    "logConfiguration": {
                        "logDriver": "awslogs",
                        "options": {"awslogs-group": "/old"},
                    },
                }
            ],
        }
    }

    task = render(
        payload,
        family="jubeex-prod-worker",
        image="example@sha256:" + "a" * 64,
        execution_role_arn="arn:execution",
        task_role_arn="arn:task",
        log_group="/ecs/prod",
        bucket="prod-bucket",
        ingestion_queue_url="https://queue/ingestion",
        scrutiny_queue_url="https://queue/scrutiny",
        callback_url="http://10.0.0.4:8000/api/v1/webhooks/ai-agent",
        pinecone_namespace="jubeex-prod-filings",
        agent_secret_arn="arn:agent",
        llama_secret_arn="arn:llama",
        api_secret_arn="arn:api",
    )

    assert task["family"] == "jubeex-prod-worker"
    assert "revision" not in task
    container = task["containerDefinitions"][0]
    environment = {item["name"]: item["value"] for item in container["environment"]}
    assert environment["AWS_S3_BUCKET"] == "prod-bucket"
    assert environment["SCRUTINY_DEFECTS"] == "all"
    assert "JUBEEX_CALLBACK_URL" not in environment
    secrets = {item["name"]: item["valueFrom"] for item in container["secrets"]}
    assert secrets["OPENROUTER_API_KEY"] == "arn:agent:OPENROUTER_API_KEY::"
    assert secrets["LLAMA_CLOUD_API_KEY"] == "arn:llama:LLAMA_CLOUD_API_KEY::"
    assert secrets["LLAMA_CLOUD_TIER"] == "arn:llama:LLAMA_CLOUD_TIER::"
    assert secrets["JUBEEX_API_KEY"] == "arn:api:JUBEEX_API_KEY::"
    assert container["logConfiguration"]["options"]["awslogs-group"] == "/ecs/prod"
