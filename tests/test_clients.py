from unittest.mock import MagicMock

import pytest

from extraction_review.clients import (
    DEFAULT_LLAMA_CLOUD_REQUEST_TIMEOUT_S,
    get_llama_cloud_client,
    llama_cloud_request_timeout_seconds,
)


def test_llama_cloud_timeout_defaults_for_large_uploads(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.delenv("LLAMA_CLOUD_REQUEST_TIMEOUT_S", raising=False)
    assert (
        llama_cloud_request_timeout_seconds()
        == DEFAULT_LLAMA_CLOUD_REQUEST_TIMEOUT_S
        == 900.0
    )


def test_llama_cloud_client_uses_configured_read_and_write_timeout(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    factory = MagicMock(return_value=MagicMock())
    monkeypatch.setenv("LLAMA_CLOUD_REQUEST_TIMEOUT_S", "1200")
    monkeypatch.setattr("extraction_review.clients.AsyncLlamaCloud", factory)

    get_llama_cloud_client()

    timeout = factory.call_args.kwargs["timeout"]
    assert timeout.connect == 30.0
    assert timeout.read == 1200.0
    assert timeout.write == 1200.0
    assert timeout.pool == 60.0


@pytest.mark.parametrize("value", ["invalid", "59", "3601"])
def test_llama_cloud_timeout_rejects_unsafe_values(
    monkeypatch: pytest.MonkeyPatch, value: str
) -> None:
    monkeypatch.setenv("LLAMA_CLOUD_REQUEST_TIMEOUT_S", value)
    assert llama_cloud_request_timeout_seconds() == 900.0
