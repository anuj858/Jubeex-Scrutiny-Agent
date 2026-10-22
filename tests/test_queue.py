import pytest

from extraction_review.queue import (
    DEFAULT_VISIBILITY_TIMEOUT,
    _client_kwargs,
    _queue_url,
    extend_job_visibility,
    queue_kind,
    queue_name,
    release_job_visibility,
    sqs_enabled,
    visibility_timeout,
)


def test_queue_kind_splits_ingestion_and_scrutiny() -> None:
    assert queue_kind("scrutiny") == "scrutiny"
    assert queue_kind("annexure_index") == "scrutiny"
    assert queue_kind("process_file") == "process_file"
    assert queue_kind("INGESTION") == "process_file"


def test_queue_names_default() -> None:
    assert queue_name("process_file") == "jubeex-ingestion-jobs"
    assert queue_name("scrutiny") == "jubeex-scrutiny-jobs"


def test_visibility_timeout_defaults_to_ten_minutes(monkeypatch) -> None:
    monkeypatch.delenv("JUBEEX_SQS_VISIBILITY_TIMEOUT", raising=False)
    assert DEFAULT_VISIBILITY_TIMEOUT == 600
    assert visibility_timeout() == 600


def test_client_uses_explicit_env_credentials(monkeypatch) -> None:
    monkeypatch.setenv("AWS_REGION", "ap-south-1")
    monkeypatch.setenv("AWS_ACCESS_KEY_ID", "AKIAEXAMPLE")
    monkeypatch.setenv("AWS_SECRET_ACCESS_KEY", "secret")
    monkeypatch.delenv("AWS_SESSION_TOKEN", raising=False)
    kwargs = _client_kwargs()
    assert kwargs["region_name"] == "ap-south-1"
    assert kwargs["aws_access_key_id"] == "AKIAEXAMPLE"
    assert kwargs["aws_secret_access_key"] == "secret"


def test_client_uses_jubeex_credential_aliases(monkeypatch) -> None:
    monkeypatch.delenv("AWS_REGION", raising=False)
    monkeypatch.delenv("AWS_DEFAULT_REGION", raising=False)
    monkeypatch.delenv("AWS_ACCESS_KEY_ID", raising=False)
    monkeypatch.delenv("AWS_SECRET_ACCESS_KEY", raising=False)
    monkeypatch.delenv("AWS_SESSION_TOKEN", raising=False)
    monkeypatch.setenv("JUBEEX_AWS_REGION", "ap-south-1")
    monkeypatch.setenv("JUBEEX_AWS_ACCESS_KEY_ID", "AKIAALIAS")
    monkeypatch.setenv("JUBEEX_AWS_SECRET_ACCESS_KEY", "alias-secret")
    kwargs = _client_kwargs()
    assert kwargs["region_name"] == "ap-south-1"
    assert kwargs["aws_access_key_id"] == "AKIAALIAS"
    assert kwargs["aws_secret_access_key"] == "alias-secret"


def test_sqs_disabled_without_env(monkeypatch) -> None:
    monkeypatch.delenv("JUBEEX_SQS_ENABLED", raising=False)
    monkeypatch.delenv("JUBEEX_SQS_INGESTION_QUEUE_URL", raising=False)
    monkeypatch.delenv("JUBEEX_SQS_SCRUTINY_QUEUE_URL", raising=False)
    assert sqs_enabled() is False
    monkeypatch.setenv("JUBEEX_SQS_ENABLED", "true")
    assert sqs_enabled() is True


def test_extend_job_visibility_renews_receipt(monkeypatch) -> None:
    class FakeClient:
        def __init__(self) -> None:
            self.kwargs = None

        def change_message_visibility(self, **kwargs) -> None:
            self.kwargs = kwargs

    client = FakeClient()
    monkeypatch.setattr("extraction_review.queue._client", lambda: client)

    extend_job_visibility(
        {"_queue_url": "https://sqs/queue", "_receipt_handle": "receipt"},
        timeout_seconds=900,
    )

    assert client.kwargs == {
        "QueueUrl": "https://sqs/queue",
        "ReceiptHandle": "receipt",
        "VisibilityTimeout": 900,
    }


def test_release_job_visibility_retries_immediately(monkeypatch) -> None:
    class FakeClient:
        def __init__(self) -> None:
            self.kwargs = None

        def change_message_visibility(self, **kwargs) -> None:
            self.kwargs = kwargs

    client = FakeClient()
    monkeypatch.setattr("extraction_review.queue._client", lambda: client)
    release_job_visibility(
        {"_queue_url": "https://sqs/queue", "_receipt_handle": "receipt"}
    )
    assert client.kwargs == {
        "QueueUrl": "https://sqs/queue",
        "ReceiptHandle": "receipt",
        "VisibilityTimeout": 0,
    }


def test_missing_queue_is_not_created_without_dlq(monkeypatch) -> None:
    class MissingQueueClient:
        def get_queue_url(self, **kwargs):
            raise RuntimeError("QueueDoesNotExist")

        def create_queue(self, **kwargs):
            pytest.fail("worker must not create an unprotected SQS queue")

    monkeypatch.delenv("JUBEEX_SQS_SCRUTINY_QUEUE_URL", raising=False)
    monkeypatch.setattr("extraction_review.queue._client", MissingQueueClient)
    with pytest.raises(RuntimeError, match="QueueDoesNotExist"):
        _queue_url("scrutiny")
