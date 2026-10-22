"""Render reviewed ECS task definitions with immutable images and safe defaults.

This prepares local JSON files only. It never registers a task or updates ECS.
"""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path
from urllib.parse import urlparse

ROOT = Path(__file__).resolve().parents[1]
TEMPLATES = {
    "api": "ecs-scrutiny-api-task.json",
    "ingestion": "worker-task-definition.json",
    "scrutiny": "ecs-scrutiny-worker-task.json",
}
IMAGE_DIGEST = re.compile(r"^[^\s@]+@sha256:[0-9a-f]{64}$")


def _environment(container: dict) -> dict[str, str]:
    return {str(item["name"]): str(item["value"]) for item in container["environment"]}


def _set_environment(container: dict, name: str, value: str) -> None:
    for item in container["environment"]:
        if item["name"] == name:
            item["value"] = value
            return
    container["environment"].append({"name": name, "value": value})


def _validate_callback_url(value: str) -> None:
    parsed = urlparse(value)
    if parsed.username or parsed.password or not parsed.hostname or not parsed.path:
        raise ValueError("callback URL must have a host and path without credentials")
    if parsed.scheme == "https":
        return
    if parsed.scheme == "http" and parsed.hostname.endswith(".internal"):
        return
    raise ValueError("callback URL must use HTTPS or approved private .internal HTTP")


def prepare(
    *,
    api_image: str,
    worker_image: str,
    callback_url: str,
    output_dir: Path,
    template_dir: Path = ROOT,
) -> list[Path]:
    """Write release JSON after validating immutable images and worker roles."""
    for image in (api_image, worker_image):
        if not IMAGE_DIGEST.fullmatch(image):
            raise ValueError("ECS release images must be immutable @sha256 digests")
    _validate_callback_url(callback_url)
    if output_dir.resolve() == template_dir.resolve():
        raise ValueError("Release output must not overwrite checked-in templates")

    rendered: list[tuple[str, dict]] = []
    for role, filename in TEMPLATES.items():
        task = json.loads((template_dir / filename).read_text(encoding="utf-8"))
        containers = task.get("containerDefinitions") or []
        if len(containers) != 1:
            raise ValueError(f"{filename}: expected exactly one container")
        container = containers[0]
        env = _environment(container)
        if env.get("SCRUTINY_CATALOGUE_SOURCE") != "db":
            raise ValueError(f"{filename}: production requires the DB catalogue")
        if env.get("SCRUTINY_DEFECTS", "").lower() != "all":
            raise ValueError(f"{filename}: production cannot use a defect allowlist")
        if env.get("JUBEEX_SQS_ENABLED", "").lower() != "true":
            raise ValueError(f"{filename}: SQS must be enabled")
        secrets = {item["name"] for item in container.get("secrets") or []}
        if not {"CATALOGUE_DATABASE_URL", "JUBEEX_CALLBACK_SECRET"} <= secrets:
            raise ValueError(f"{filename}: missing catalogue or callback secret")
        if role != "api":
            expected = "process_file" if role == "ingestion" else "scrutiny"
            if env.get("JUBEEX_WORKER_KINDS") != expected:
                raise ValueError(f"{filename}: expected worker kind {expected}")
            _set_environment(container, "JUBEEX_WORKER_MAX_JOB_SECONDS", "5400")
        if role != "api":
            if env.get("LLM_PROVIDER", "vertex") != "vertex":
                raise ValueError(f"{filename}: worker must use Vertex")
            if env.get("VERTEX_SHARED_RATE_LIMIT", "").lower() != "true":
                raise ValueError(f"{filename}: shared Vertex rate limit must be enabled")
            try:
                vertex_rpm = int(env.get("VERTEX_REQUESTS_PER_MINUTE", ""))
            except ValueError as exc:
                raise ValueError(f"{filename}: invalid Vertex request limit") from exc
            if vertex_rpm <= 0:
                raise ValueError(f"{filename}: Vertex request limit must be positive")
            if env.get("LLM_REQUESTS_PER_MINUTE"):
                raise ValueError(
                    f"{filename}: LLM_REQUESTS_PER_MINUTE overrides the Vertex limit"
                )
            if not (env.get("AWS_S3_BUCKET") or env.get("JUBEEX_ARTIFACT_BUCKET")):
                raise ValueError(f"{filename}: shared Vertex limiter requires an S3 bucket")
        container["image"] = api_image if role == "api" else worker_image
        _set_environment(container, "JUBEEX_CALLBACK_URL", callback_url)
        rendered.append((filename, task))

    if any((output_dir / filename).exists() for filename, _ in rendered):
        raise FileExistsError("Release output already exists; use a new directory")
    output_dir.mkdir(parents=True, exist_ok=True)
    paths: list[Path] = []
    for filename, task in rendered:
        target = output_dir / filename
        target.write_text(json.dumps(task, indent=2) + "\n", encoding="utf-8")
        paths.append(target)
    return paths


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--api-image", required=True)
    parser.add_argument("--worker-image", required=True)
    parser.add_argument("--callback-url", required=True)
    parser.add_argument("--output-dir", required=True, type=Path)
    args = parser.parse_args()
    for path in prepare(
        api_image=args.api_image,
        worker_image=args.worker_image,
        callback_url=args.callback_url,
        output_dir=args.output_dir,
    ):
        print(path)


if __name__ == "__main__":
    main()
