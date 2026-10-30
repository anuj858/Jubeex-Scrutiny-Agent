import io
import json
from unittest.mock import AsyncMock

import pytest
from botocore.exceptions import ClientError
from pydantic import BaseModel

from extraction_review.llm import (
    LLMError,
    _complete_structured_fields,
    _extend_shared_vertex_cooldown,
    _is_retryable_vertex_error,
    _is_vertex_resource_exhausted,
    _OpenRouterRateLimiter,
    _parse_json,
    _reserve_shared_vertex_slot,
    _shared_vertex_cooldown_remaining,
    _vertex_retry_after_seconds,
    gemini_max_attempts,
    gemini_request_timeout_seconds,
    get_genai_client,
    genai_shared_rate_limit_enabled,
    llm_requests_per_minute,
    openrouter_requests_per_minute,
    vertex_max_attempts,
    vertex_request_timeout_seconds,
    vertex_shared_rate_limit_enabled,
)


def test_parse_json_object() -> None:
    assert _parse_json('{"check_id": "D018", "status": "compliant"}') == {
        "check_id": "D018",
        "status": "compliant",
    }


def test_parse_json_fenced_block() -> None:
    raw = """```json
{"check_id": "D018", "status": "compliant"}
```"""
    assert _parse_json(raw)["check_id"] == "D018"


def test_parse_json_truncated_object_is_repaired() -> None:
    parsed = _parse_json('{\n  "check_id": "D018",\n  "status": "compliant",\n')
    assert parsed["check_id"] == "D018"
    assert parsed["status"] == "compliant"


def test_a_cut_off_sentence_is_not_a_complete_answer() -> None:
    from extraction_review.llm import (
        ANSWER_TOKEN_BUDGET,
        DEFAULT_MAX_TOKENS,
        _answer_was_cut,
        _hard_token_limit,
        _loads_complete_json,
        _with_output_budget,
    )

    cut = """{
  "check_id": "D-68",
  "status": "needs_review",
  "summary": "The language of Annexure P-6 cannot be identified.",
  "observed": "Annexures P-1 through P-5 appear in English, but Annexure P-"""
    assert _loads_complete_json(cut) is None
    assert _answer_was_cut(cut) is True
    assert DEFAULT_MAX_TOKENS == 9000
    assert ANSWER_TOKEN_BUDGET == 8200
    assert _hard_token_limit(None) == 9000
    assert _hard_token_limit(20000) == 9000
    told = _with_output_budget("Decide this defect.")
    assert "8200" in told
    assert _with_output_budget(told) == told


def test_parse_json_truncated_reasoning_is_salvaged() -> None:
    parsed = _parse_json(
        """{
  "check_id": "D007",
  "status": "needs_review",
  "confidence": 0.5,
  "summary": "The filing parts required to identify whether an interlocutory or miscellaneous application has been filed beyond limitation are not available in the excerpts.",
  "reasoning": "The inspect target Record of Proceedings appears at pages 2–3, but the filing excerpts do not contain any interlocutory or miscellaneou"""
    )
    assert parsed["check_id"] == "D007"
    assert parsed["status"] == "needs_review"
    assert parsed["confidence"] == 0.5
    assert parsed["reasoning"].startswith("The inspect target")
    completed = _complete_structured_fields(parsed)
    assert completed["evidence"] == []


def test_complete_structured_fields_fills_missing_confidence_and_summary() -> None:
    completed = _complete_structured_fields(
        {
            "check_id": "D-73",
            "status": "needs_review",
            "reasoning": "The Vakalatnama was checked on page 40.",
        }
    )
    assert completed["confidence"] == 0.5
    assert completed["summary"] == "The Vakalatnama was checked on page 40."
    assert completed["evidence"] == []


def test_parse_json_garbage_raises() -> None:
    with pytest.raises(LLMError, match="truncated or not JSON"):
        _parse_json("not json at all")


def test_openrouter_requests_per_minute_env(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("OPENROUTER_REQUESTS_PER_MINUTE", "20")
    assert openrouter_requests_per_minute() == 20
    monkeypatch.setenv("OPENROUTER_REQUESTS_PER_MINUTE", "0")
    assert openrouter_requests_per_minute() == 0
    monkeypatch.delenv("OPENROUTER_REQUESTS_PER_MINUTE", raising=False)
    assert openrouter_requests_per_minute() == 0


def test_llm_provider_defaults_to_openrouter(monkeypatch: pytest.MonkeyPatch) -> None:
    from extraction_review.llm import llm_provider, openrouter_model

    monkeypatch.delenv("LLM_PROVIDER", raising=False)
    monkeypatch.setenv("OPENROUTER_MODEL", "google/gemini-4-argon")
    assert llm_provider() == "openrouter"
    assert openrouter_model() == "google/gemini-4-argon"


def test_gemini_provider_uses_unprefixed_developer_api_model(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from extraction_review.llm import llm_provider, openrouter_enabled, openrouter_model

    monkeypatch.setenv("LLM_PROVIDER", "gemini")
    monkeypatch.setenv("GEMINI_API_KEY", "developer-api-key")
    monkeypatch.setenv("GEMINI_MODEL", "google/gemini-4-argon")

    assert llm_provider() == "gemini"
    assert openrouter_enabled() is True
    assert openrouter_model() == "gemini-4-argon"


def test_openrouter_limit_does_not_throttle_vertex(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("LLM_PROVIDER", "vertex")
    monkeypatch.delenv("LLM_REQUESTS_PER_MINUTE", raising=False)
    monkeypatch.delenv("VERTEX_REQUESTS_PER_MINUTE", raising=False)
    monkeypatch.setenv("OPENROUTER_REQUESTS_PER_MINUTE", "25")
    assert llm_requests_per_minute() == 0


def test_active_provider_rate_limit(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("LLM_PROVIDER", "vertex")
    monkeypatch.setenv("VERTEX_REQUESTS_PER_MINUTE", "120")
    assert llm_requests_per_minute() == 120
    monkeypatch.setenv("LLM_REQUESTS_PER_MINUTE", "80")
    assert llm_requests_per_minute() == 80


def test_vertex_shared_rate_limit_defaults_on_for_sqs_with_bucket(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.delenv("VERTEX_SHARED_RATE_LIMIT", raising=False)
    monkeypatch.setenv("JUBEEX_SQS_ENABLED", "true")
    monkeypatch.setenv("AWS_S3_BUCKET", "artifacts")
    assert vertex_shared_rate_limit_enabled() is True
    monkeypatch.setenv("VERTEX_SHARED_RATE_LIMIT", "false")
    assert vertex_shared_rate_limit_enabled() is False


def test_gemini_shared_rate_limit_defaults_on_for_sqs_with_bucket(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("LLM_PROVIDER", "gemini")
    monkeypatch.delenv("GEMINI_SHARED_RATE_LIMIT", raising=False)
    monkeypatch.setenv("JUBEEX_SQS_ENABLED", "true")
    monkeypatch.setenv("AWS_S3_BUCKET", "artifacts")
    assert genai_shared_rate_limit_enabled() is True
    monkeypatch.setenv("GEMINI_SHARED_RATE_LIMIT", "false")
    assert genai_shared_rate_limit_enabled() is False


def test_shared_vertex_limiter_reserves_sequential_s3_slots(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    class FakeS3:
        body: bytes | None = None
        etag = 0

        def get_object(self, **_kwargs):
            if self.body is None:
                raise ClientError(
                    {"Error": {"Code": "NoSuchKey", "Message": "missing"}},
                    "GetObject",
                )
            return {"ETag": f'"{self.etag}"', "Body": io.BytesIO(self.body)}

        def put_object(self, **kwargs):
            if self.body is None:
                assert kwargs.get("IfNoneMatch") == "*"
            else:
                assert kwargs.get("IfMatch") == f'"{self.etag}"'
            self.etag += 1
            self.body = kwargs["Body"]
            return {"ETag": f'"{self.etag}"'}

    client = FakeS3()
    monkeypatch.setattr("extraction_review.llm.time.time", lambda: 1_000.0)
    first = _reserve_shared_vertex_slot(
        client=client,
        bucket="artifacts",
        key="rate.json",
        interval_s=7.5,
    )
    second = _reserve_shared_vertex_slot(
        client=client,
        bucket="artifacts",
        key="rate.json",
        interval_s=7.5,
    )
    assert first == 0
    assert second == 7.5
    assert json.loads(client.body or b"{}")["next_allowed_at"] == 1_015.0


def test_shared_vertex_limiter_respects_global_cooldown(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    class FakeS3:
        body = json.dumps(
            {"next_allowed_at": 1_000.0, "cooldown_until": 1_060.0}
        ).encode()
        etag = 1

        def get_object(self, **_kwargs):
            return {"ETag": f'"{self.etag}"', "Body": io.BytesIO(self.body)}

        def put_object(self, **kwargs):
            assert kwargs.get("IfMatch") == f'"{self.etag}"'
            self.etag += 1
            self.body = kwargs["Body"]
            return {"ETag": f'"{self.etag}"'}

    client = FakeS3()
    monkeypatch.setattr("extraction_review.llm.time.time", lambda: 1_000.0)
    wait_s = _reserve_shared_vertex_slot(
        client=client,
        bucket="artifacts",
        key="rate.json",
        interval_s=7.5,
    )
    assert wait_s == 60.0
    state = json.loads(client.body)
    assert state["next_allowed_at"] == 1_067.5
    assert state["cooldown_until"] == 1_060.0


def test_shared_vertex_429_cooldown_is_global_and_adaptive(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    class FakeS3:
        body: bytes | None = None
        etag = 0

        def get_object(self, **_kwargs):
            if self.body is None:
                raise ClientError(
                    {"Error": {"Code": "NoSuchKey", "Message": "missing"}},
                    "GetObject",
                )
            return {"ETag": f'"{self.etag}"', "Body": io.BytesIO(self.body)}

        def put_object(self, **kwargs):
            if self.body is None:
                assert kwargs.get("IfNoneMatch") == "*"
            else:
                assert kwargs.get("IfMatch") == f'"{self.etag}"'
            self.etag += 1
            self.body = kwargs["Body"]
            return {"ETag": f'"{self.etag}"'}

    client = FakeS3()
    monkeypatch.setattr("extraction_review.llm.time.time", lambda: 1_000.0)
    first = _extend_shared_vertex_cooldown(
        client=client,
        bucket="artifacts",
        key="rate.json",
    )
    second = _extend_shared_vertex_cooldown(
        client=client,
        bucket="artifacts",
        key="rate.json",
    )
    assert first == 30.0
    assert second == 60.0
    assert (
        _shared_vertex_cooldown_remaining(
            client=client,
            bucket="artifacts",
            key="rate.json",
        )
        == 60.0
    )
    state = json.loads(client.body or b"{}")
    assert state["consecutive_429s"] == 2
    assert state["cooldown_until"] == 1_060.0


def test_vertex_resource_exhausted_detection() -> None:
    assert _is_vertex_resource_exhausted(RuntimeError("429 RESOURCE_EXHAUSTED"))
    assert not _is_vertex_resource_exhausted(RuntimeError("503 unavailable"))


@pytest.mark.asyncio
async def test_vertex_vision_retries_after_shared_429_cooldown(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import extraction_review.llm as llm_module

    response = type("Response", (), {"text": '{"ok": true}'})()
    generate = AsyncMock(side_effect=[RuntimeError("429 RESOURCE_EXHAUSTED"), response])
    models = type("Models", (), {"generate_content": generate})()
    client = type("Client", (), {"aio": type("Aio", (), {"models": models})()})()
    acquire = AsyncMock()
    publish = AsyncMock(return_value=30.0)
    sleep = AsyncMock()

    monkeypatch.setenv("LLM_PROVIDER", "vertex")
    monkeypatch.setenv("VERTEX_MAX_ATTEMPTS", "2")
    monkeypatch.setattr(llm_module, "get_genai_client", lambda: client)
    monkeypatch.setattr(llm_module, "acquire_openrouter_slot", acquire)
    monkeypatch.setattr(llm_module, "_publish_vertex_cooldown", publish)
    monkeypatch.setattr(llm_module.asyncio, "sleep", sleep)

    content, _usage = await llm_module.generate_vision_json(
        system_prompt="system",
        user_prompt="user",
        images=[],
        model="gemini-3.8-flash",
    )

    assert content == '{"ok": true}'
    assert generate.await_count == 2
    assert acquire.await_count == 2
    publish.assert_awaited_once()
    sleep.assert_awaited_once_with(30.0)


def test_genai_client_disables_sdk_retries(monkeypatch: pytest.MonkeyPatch) -> None:
    from google import genai

    import extraction_review.llm as llm_module

    captured: dict[str, object] = {}

    def fake_client(**kwargs):
        captured.update(kwargs)
        return object()

    monkeypatch.setattr(llm_module, "_genai_client", None)
    monkeypatch.setenv("LLM_PROVIDER", "vertex")
    monkeypatch.setenv("VERTEX_REQUEST_TIMEOUT_S", "900")
    monkeypatch.setattr(genai, "Client", fake_client)
    get_genai_client()
    http_options = captured["http_options"]
    assert http_options.retry_options.attempts == 1
    assert http_options.timeout == 900000


def test_gemini_developer_client_uses_api_key_not_vertex_credentials(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from google import genai

    import extraction_review.llm as llm_module

    captured: dict[str, object] = {}

    def fake_client(**kwargs):
        captured.update(kwargs)
        return object()

    monkeypatch.setattr(llm_module, "_genai_client", None)
    monkeypatch.setattr(llm_module, "_genai_client_key", None)
    monkeypatch.setenv("LLM_PROVIDER", "gemini")
    monkeypatch.setenv("GEMINI_API_KEY", "developer-api-key")
    monkeypatch.setenv("GEMINI_REQUEST_TIMEOUT_S", "600")
    monkeypatch.setattr(genai, "Client", fake_client)

    get_genai_client()

    assert captured["api_key"] == "developer-api-key"
    assert "project" not in captured
    assert "location" not in captured
    assert "credentials" not in captured
    assert captured["http_options"].retry_options.attempts == 1
    assert captured["http_options"].timeout == 600000


def test_vertex_request_timeout_must_be_bounded(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.delenv("VERTEX_REQUEST_TIMEOUT_S", raising=False)
    assert vertex_request_timeout_seconds() == 900
    monkeypatch.setenv("VERTEX_REQUEST_TIMEOUT_S", "30")
    with pytest.raises(ValueError, match="between 60 and 3600"):
        vertex_request_timeout_seconds()


def test_vertex_retry_settings(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("VERTEX_MAX_ATTEMPTS", raising=False)
    assert vertex_max_attempts() == 6
    monkeypatch.setenv("VERTEX_MAX_ATTEMPTS", "4")
    assert vertex_max_attempts() == 4
    assert _is_retryable_vertex_error(RuntimeError("429 RESOURCE_EXHAUSTED"))
    assert not _is_retryable_vertex_error(RuntimeError("invalid request"))


def test_gemini_retry_settings(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("GEMINI_MAX_ATTEMPTS", raising=False)
    monkeypatch.delenv("GEMINI_REQUEST_TIMEOUT_S", raising=False)
    assert gemini_max_attempts() == 4
    assert gemini_request_timeout_seconds() == 900
    monkeypatch.setenv("GEMINI_MAX_ATTEMPTS", "2")
    monkeypatch.setenv("GEMINI_REQUEST_TIMEOUT_S", "600")
    assert gemini_max_attempts() == 2
    assert gemini_request_timeout_seconds() == 600


@pytest.mark.asyncio
async def test_call_structured_uses_gemini_developer_api_client(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import extraction_review.llm as llm_module

    class Result(BaseModel):
        value: str

    response = type(
        "Response",
        (),
        {
            "text": '{"value":"ok"}',
            "parsed": None,
            "candidates": [],
            "usage_metadata": None,
        },
    )()
    generate = AsyncMock(return_value=response)
    models = type("Models", (), {"generate_content": generate})()
    client = type("Client", (), {"aio": type("Aio", (), {"models": models})()})()

    monkeypatch.setenv("LLM_PROVIDER", "gemini")
    monkeypatch.setenv("GEMINI_API_KEY", "developer-api-key")
    monkeypatch.setenv("GEMINI_MODEL", "gemini-4-argon")
    monkeypatch.setattr(llm_module, "get_genai_client", lambda: client)
    monkeypatch.setattr(llm_module, "acquire_openrouter_slot", AsyncMock())

    result, usage = await llm_module.call_structured(
        system_prompt="Return JSON.",
        user_prompt="Return ok.",
        response_model=Result,
    )

    assert result.value == "ok"
    assert usage.calls == 1
    assert generate.await_args.kwargs["model"] == "gemini-4-argon"
    config = generate.await_args.kwargs["config"]
    assert config.response_mime_type == "application/json"
    assert config.response_json_schema is not None


def test_vertex_retry_after_header_is_honoured() -> None:
    response = type("Response", (), {"headers": {"retry-after": "12"}})()
    error = RuntimeError("429")
    error.response = response  # type: ignore[attr-defined]
    assert _vertex_retry_after_seconds(error, 1) == 12


def test_llm_provider_openrouter_keeps_prefixed_model(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from extraction_review.llm import llm_provider, openrouter_model

    monkeypatch.setenv("LLM_PROVIDER", "openrouter")
    monkeypatch.setenv("OPENROUTER_MODEL", "google/gemini-3.8-flash")
    assert llm_provider() == "openrouter"
    assert openrouter_model() == "google/gemini-3.8-flash"


def test_scrutiny_model_defaults_to_gemini_4_argon_for_vertex(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from extraction_review.llm import scrutiny_model

    monkeypatch.setenv("LLM_PROVIDER", "vertex")
    monkeypatch.delenv("SCRUTINY_MODEL", raising=False)
    assert scrutiny_model() == "gemini-4-argon"


def test_scrutiny_model_normalizes_vertex_and_preserves_openrouter_prefix(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from extraction_review.llm import scrutiny_model

    monkeypatch.setenv("SCRUTINY_MODEL", "google/gemini-3.8-flash")
    monkeypatch.setenv("LLM_PROVIDER", "vertex")
    assert scrutiny_model() == "gemini-3.8-flash"
    monkeypatch.setenv("LLM_PROVIDER", "openrouter")
    assert scrutiny_model() == "google/gemini-3.8-flash"
    monkeypatch.setenv("LLM_PROVIDER", "gemini")
    assert scrutiny_model() == "gemini-3.8-flash"


@pytest.mark.asyncio
async def test_openrouter_rate_limiter_paces_requests(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("LLM_PROVIDER", "openrouter")
    monkeypatch.setenv("OPENROUTER_REQUESTS_PER_MINUTE", "2")
    limiter = _OpenRouterRateLimiter()
    slept: list[float] = []

    async def fake_sleep(seconds: float) -> None:
        slept.append(seconds)

    monkeypatch.setattr("extraction_review.llm.asyncio.sleep", fake_sleep)
    await limiter.acquire()
    await limiter.acquire()
    assert slept == [pytest.approx(30.0, abs=0.1)]
