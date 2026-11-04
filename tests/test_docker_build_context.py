from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def test_worker_docker_context_is_deny_by_default_with_runtime_allowlist():
    rules = (ROOT / ".dockerignore").read_text(encoding="utf-8").splitlines()
    rules = [line.strip() for line in rules if line.strip() and not line.lstrip().startswith("#")]

    assert rules[0] == "**"
    for required in (
        "!README.md",
        "!pyproject.toml",
        "!uv.lock",
        "!scripts/start.sh",
        "!configs/config.json",
        "!src/extraction_review/**",
        "!scrutiny_harness/rules/**",
    ):
        assert required in rules

    for forbidden in (
        "**/.env",
        "**/.env.*",
        "**/*.pem",
        "**/*.key",
        "**/*credentials*.json",
        "**/*secret*.json",
        "**/.aws/**",
        "**/.ssh/**",
        "**/.venv/**",
        "**/__pycache__/**",
        "**/.pytest_cache/**",
        "**/.ruff_cache/**",
        "**/tmp/**",
        "**/artifacts/**",
        "**/qa_reports/**",
        "**/tests/**",
        "**/evaluation/**",
        "**/*.db",
        "**/*.sqlite",
        "**/*.sqlite3",
    ):
        assert forbidden in rules
