from extraction_review.queue import _client_kwargs, queue_kind, queue_name, sqs_enabled


def test_queue_kind_splits_ingestion_and_scrutiny() -> None:
    assert queue_kind("scrutiny") == "scrutiny"
    assert queue_kind("process_file") == "process_file"
    assert queue_kind("INGESTION") == "process_file"


def test_queue_names_default() -> None:
    assert queue_name("process_file") == "jubeex-ingestion-jobs"
    assert queue_name("scrutiny") == "jubeex-scrutiny-jobs"


def test_client_uses_explicit_env_credentials(monkeypatch) -> None:
    monkeypatch.setenv("AWS_REGION", "ap-south-1")
    monkeypatch.setenv("AWS_ACCESS_KEY_ID", "AKIAEXAMPLE")
    monkeypatch.setenv("AWS_SECRET_ACCESS_KEY", "secret")
    monkeypatch.delenv("AWS_SESSION_TOKEN", raising=False)
    kwargs = _client_kwargs()
    assert kwargs["region_name"] == "ap-south-1"
    assert kwargs["aws_access_key_id"] == "AKIAEXAMPLE"
    assert kwargs["aws_secret_access_key"] == "secret"


def test_sqs_disabled_without_env(monkeypatch) -> None:
    monkeypatch.delenv("JUBEEX_SQS_ENABLED", raising=False)
    monkeypatch.delenv("JUBEEX_SQS_INGESTION_QUEUE_URL", raising=False)
    monkeypatch.delenv("JUBEEX_SQS_SCRUTINY_QUEUE_URL", raising=False)
    assert sqs_enabled() is False
    monkeypatch.setenv("JUBEEX_SQS_ENABLED", "true")
    assert sqs_enabled() is True
