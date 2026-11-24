"""Render a sanitized ECS task definition for a deployment environment.

The source may be a checked-in task definition or the output of
``aws ecs describe-task-definition``. Secret values are never read; only secret
ARN references are rewritten.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

READ_ONLY_KEYS = {
    "compatibilities",
    "deregisteredAt",
    "registeredAt",
    "registeredBy",
    "requiresAttributes",
    "revision",
    "status",
    "taskDefinitionArn",
}

RUNTIME_SECRET_NAMES = {
    "EXTRACTION_PROVIDER",
    "LLAMA_CLOUD_TIER",
    "LLAMA_PARSE_COST_OPTIMIZER",
    "OPENROUTER_BATCH_DELETE_AFTER_READ",
    "OPENROUTER_BATCH_ENABLED",
    "OPENROUTER_BATCH_MAX_REQUESTS",
    "OPENROUTER_BATCH_POLL_SECONDS",
    "OPENROUTER_BATCH_SYNC_FALLBACK",
    "OPENROUTER_BATCH_TIMEOUT_S",
    "OPENROUTER_PROMPT_CACHE",
    "OPENROUTER_RESPONSE_CACHE",
    "OPENROUTER_RESPONSE_CACHE_TTL_S",
}


def _set_environment(container: dict[str, Any], values: dict[str, str]) -> None:
    environment = {
        str(item["name"]): str(item["value"])
        for item in container.get("environment", [])
    }
    environment.update(values)
    container["environment"] = [
        {"name": name, "value": value} for name, value in sorted(environment.items())
    ]


def _rewrite_secrets(
    container: dict[str, Any],
    *,
    agent_secret_arn: str,
    llama_secret_arn: str,
    api_secret_arn: str,
    extra_agent_secret_names: set[str] | None = None,
) -> None:
    rewritten: list[dict[str, str]] = []
    names = (
        {str(item["name"]) for item in container.get("secrets", [])}
        | RUNTIME_SECRET_NAMES
        | (extra_agent_secret_names or set())
    )
    for name in names:
        if name.startswith("LLAMA_CLOUD_"):
            arn = llama_secret_arn
        elif name == "JUBEEX_API_KEY":
            arn = api_secret_arn
        else:
            arn = agent_secret_arn
        rewritten.append({"name": name, "valueFrom": f"{arn}:{name}::"})
    container["secrets"] = sorted(rewritten, key=lambda item: item["name"])
    secret_names = {item["name"] for item in rewritten}
    container["environment"] = [
        item
        for item in container.get("environment", [])
        if item.get("name") not in secret_names
    ]


def render(
    payload: dict[str, Any],
    *,
    family: str,
    image: str,
    execution_role_arn: str,
    task_role_arn: str,
    log_group: str,
    bucket: str,
    ingestion_queue_url: str,
    scrutiny_queue_url: str,
    callback_url: str,
    pinecone_namespace: str,
    agent_secret_arn: str,
    llama_secret_arn: str,
    api_secret_arn: str,
    catalogue_source: str = "db",
    catalogue_source_from_secret: bool = False,
) -> dict[str, Any]:
    if catalogue_source not in {"db", "json"}:
        raise ValueError("catalogue_source must be 'db' or 'json'")
    task = dict(payload.get("taskDefinition", payload))
    for key in READ_ONLY_KEYS:
        task.pop(key, None)
    task["family"] = family
    task["executionRoleArn"] = execution_role_arn
    task["taskRoleArn"] = task_role_arn

    containers = task.get("containerDefinitions") or []
    if len(containers) != 1:
        raise ValueError("expected exactly one container definition")
    container = containers[0]
    container["image"] = image
    _set_environment(
        container,
        {
            "AWS_REGION": "ap-south-1",
            "AWS_S3_BUCKET": bucket,
            "JUBEEX_CALLBACK_URL": callback_url,
            "JUBEEX_REMOTE_FILE_HOSTS": (
                f"{bucket}.s3.ap-south-1.amazonaws.com,"
                f"{bucket}.s3.amazonaws.com,"
                "s3.ap-south-1.amazonaws.com"
            ),
            "JUBEEX_SQS_ENABLED": "true",
            "JUBEEX_SQS_INGESTION_QUEUE_URL": ingestion_queue_url,
            "JUBEEX_SQS_SCRUTINY_QUEUE_URL": scrutiny_queue_url,
            "PINECONE_NAMESPACE": pinecone_namespace,
            "SCRUTINY_CATALOGUE_SOURCE": catalogue_source,
            "SCRUTINY_DEFECTS": "all",
        },
    )
    _rewrite_secrets(
        container,
        agent_secret_arn=agent_secret_arn,
        llama_secret_arn=llama_secret_arn,
        api_secret_arn=api_secret_arn,
        extra_agent_secret_names=(
            {"SCRUTINY_CATALOGUE_SOURCE"} if catalogue_source_from_secret else None
        ),
    )
    log_configuration = container.setdefault("logConfiguration", {})
    log_configuration["logDriver"] = "awslogs"
    options = log_configuration.setdefault("options", {})
    options["awslogs-group"] = log_group
    options["awslogs-region"] = "ap-south-1"
    return task


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--family", required=True)
    parser.add_argument("--image", required=True)
    parser.add_argument("--execution-role-arn", required=True)
    parser.add_argument("--task-role-arn", required=True)
    parser.add_argument("--log-group", required=True)
    parser.add_argument("--bucket", required=True)
    parser.add_argument("--ingestion-queue-url", required=True)
    parser.add_argument("--scrutiny-queue-url", required=True)
    parser.add_argument("--callback-url", required=True)
    parser.add_argument("--pinecone-namespace", required=True)
    parser.add_argument("--agent-secret-arn", required=True)
    parser.add_argument("--llama-secret-arn", required=True)
    parser.add_argument("--api-secret-arn", required=True)
    parser.add_argument(
        "--catalogue-source",
        choices=("db", "json"),
        default="db",
    )
    parser.add_argument(
        "--catalogue-source-from-secret",
        action="store_true",
        help="Read SCRUTINY_CATALOGUE_SOURCE from the agent secret",
    )
    args = parser.parse_args()

    payload = json.loads(args.source.read_text(encoding="utf-8"))
    rendered = render(
        payload,
        family=args.family,
        image=args.image,
        execution_role_arn=args.execution_role_arn,
        task_role_arn=args.task_role_arn,
        log_group=args.log_group,
        bucket=args.bucket,
        ingestion_queue_url=args.ingestion_queue_url,
        scrutiny_queue_url=args.scrutiny_queue_url,
        callback_url=args.callback_url,
        pinecone_namespace=args.pinecone_namespace,
        agent_secret_arn=args.agent_secret_arn,
        llama_secret_arn=args.llama_secret_arn,
        api_secret_arn=args.api_secret_arn,
        catalogue_source=args.catalogue_source,
        catalogue_source_from_secret=args.catalogue_source_from_secret,
    )
    args.output.write_text(json.dumps(rendered, indent=2) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
