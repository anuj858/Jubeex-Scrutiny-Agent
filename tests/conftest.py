"""Pytest configuration: install the LlamaCloud fake server when compatible."""

import logging
import sys

import pytest

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)],
)

_fake = None
_fake_error: BaseException | None = None
try:
    from llama_cloud_fake import FakeLlamaCloudServer

    _fake = FakeLlamaCloudServer().install()
except Exception as exc:  # pragma: no cover - environment-dependent
    _fake_error = exc


@pytest.fixture
def fake():
    if _fake is None:
        pytest.skip(f"llama-cloud-fake is incompatible with this llama-cloud: {_fake_error}")
    return _fake
