import importlib.util
import json
import shutil
from pathlib import Path

import pytest

SCRIPT = Path(__file__).resolve().parents[1] / "scripts" / "prepare_ecs_release.py"
spec = importlib.util.spec_from_file_location("prepare_ecs_release", SCRIPT)
assert spec and spec.loader
release = importlib.util.module_from_spec(spec)
spec.loader.exec_module(release)
prepare = release.prepare


API_IMAGE = "registry.example/api@sha256:" + "a" * 64
WORKER_IMAGE = "registry.example/worker@sha256:" + "b" * 64


def test_prepare_release_pins_images_and_full_db_catalogue(tmp_path: Path) -> None:
    paths = prepare(
        api_image=API_IMAGE,
        worker_image=WORKER_IMAGE,
        callback_url="https://backend.example/api/v1/webhooks/ai-agent",
        output_dir=tmp_path / "release",
    )
    assert len(paths) == 3
    for path in paths:
        task = json.loads(path.read_text())
        container = task["containerDefinitions"][0]
        env = {item["name"]: item["value"] for item in container["environment"]}
        assert env["SCRUTINY_CATALOGUE_SOURCE"] == "db"
        assert env["SCRUTINY_DEFECTS"] == "all"
        assert env["JUBEEX_CALLBACK_URL"].startswith("https://")
        assert container["image"] == (API_IMAGE if "api" in path.name else WORKER_IMAGE)


@pytest.mark.parametrize("image", ["registry.example/api:latest", "api@sha256:short"])
def test_prepare_release_rejects_mutable_image(tmp_path: Path, image: str) -> None:
    with pytest.raises(ValueError, match="immutable"):
        prepare(
            api_image=image,
            worker_image=WORKER_IMAGE,
            callback_url="https://backend.example/webhook",
            output_dir=tmp_path / "release",
        )
    assert not (tmp_path / "release").exists()


def test_prepare_release_rejects_public_http_callback(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="HTTPS"):
        prepare(
            api_image=API_IMAGE,
            worker_image=WORKER_IMAGE,
            callback_url="http://3.111.86.61/webhook",
            output_dir=tmp_path / "release",
        )


def test_prepare_release_will_not_overwrite_templates() -> None:
    with pytest.raises(ValueError, match="must not overwrite"):
        prepare(
            api_image=API_IMAGE,
            worker_image=WORKER_IMAGE,
            callback_url="https://backend.example/webhook",
            output_dir=release.ROOT,
        )


def test_prepare_release_will_not_overwrite_existing_output(tmp_path: Path) -> None:
    output_dir = tmp_path / "release"
    output_dir.mkdir()
    existing = output_dir / "ecs-scrutiny-api-task.json"
    existing.write_text("keep me")
    with pytest.raises(FileExistsError, match="already exists"):
        prepare(
            api_image=API_IMAGE,
            worker_image=WORKER_IMAGE,
            callback_url="https://backend.example/webhook",
            output_dir=output_dir,
        )
    assert existing.read_text() == "keep me"


@pytest.mark.parametrize(
    ("name", "value", "message"),
    [
        ("VERTEX_SHARED_RATE_LIMIT", "false", "shared Vertex rate limit"),
        ("VERTEX_REQUESTS_PER_MINUTE", "0", "must be positive"),
        ("VERTEX_REQUESTS_PER_MINUTE", "bad", "invalid Vertex request limit"),
        ("AWS_S3_BUCKET", "", "requires an S3 bucket"),
        ("LLM_REQUESTS_PER_MINUTE", "16", "overrides the Vertex limit"),
    ],
)
@pytest.mark.parametrize(
    "worker_filename", ["ecs-scrutiny-worker-task.json", "worker-task-definition.json"]
)
def test_prepare_release_rejects_broken_scrutiny_pacing(
    tmp_path: Path, worker_filename: str, name: str, value: str, message: str
) -> None:
    template_dir = tmp_path / "templates"
    template_dir.mkdir()
    for filename in release.TEMPLATES.values():
        shutil.copyfile(release.ROOT / filename, template_dir / filename)
    worker_path = template_dir / worker_filename
    task = json.loads(worker_path.read_text())
    environment = task["containerDefinitions"][0]["environment"]
    entry = next((item for item in environment if item["name"] == name), None)
    if entry:
        entry["value"] = value
    else:
        environment.append({"name": name, "value": value})
    worker_path.write_text(json.dumps(task))

    with pytest.raises(ValueError, match=message):
        prepare(
            api_image=API_IMAGE,
            worker_image=WORKER_IMAGE,
            callback_url="https://backend.example/webhook",
            output_dir=tmp_path / "release",
            template_dir=template_dir,
        )
    assert not (tmp_path / "release").exists()


@pytest.mark.parametrize(
    "filename",
    [
        "agent-task.json",
        "ecs-ai-api-task.json",
        "ecs-scrutiny-api-task.json",
        "ecs-scrutiny-worker-task.json",
        "worker-task-definition.json",
    ],
)
def test_checked_in_task_catalogue_is_not_test_subset(filename: str) -> None:
    task = json.loads((release.ROOT / filename).read_text())
    env = {
        item["name"]: item["value"]
        for item in task["containerDefinitions"][0]["environment"]
    }
    assert env["SCRUTINY_CATALOGUE_SOURCE"] == "db"
    assert env["SCRUTINY_DEFECTS"] == "all"
