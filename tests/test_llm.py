import pytest

from extraction_review.llm import (
    LLMError,
    _complete_structured_fields,
    _is_retryable_vertex_error,
    _OpenRouterRateLimiter,
    _parse_json,
    _vertex_retry_after_seconds,
    llm_requests_per_minute,
    openrouter_requests_per_minute,
    vertex_max_attempts,
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


def test_llm_provider_defaults_to_vertex(monkeypatch: pytest.MonkeyPatch) -> None:
    from extraction_review.llm import llm_provider, openrouter_model

    monkeypatch.delenv("LLM_PROVIDER", raising=False)
    monkeypatch.setenv("VERTEX_MODEL", "gemini-3.8-flash")
    assert llm_provider() == "vertex"
    assert openrouter_model() == "gemini-3.8-flash"


def test_openrouter_limit_does_not_throttle_vertex(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.delenv("LLM_PROVIDER", raising=False)
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


def test_vertex_retry_settings(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("VERTEX_MAX_ATTEMPTS", raising=False)
    assert vertex_max_attempts() == 6
    monkeypatch.setenv("VERTEX_MAX_ATTEMPTS", "4")
    assert vertex_max_attempts() == 4
    assert _is_retryable_vertex_error(RuntimeError("429 RESOURCE_EXHAUSTED"))
    assert not _is_retryable_vertex_error(RuntimeError("invalid request"))


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


def test_scrutiny_model_defaults_to_gemini_3_8_for_vertex(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from extraction_review.llm import scrutiny_model

    monkeypatch.setenv("LLM_PROVIDER", "vertex")
    monkeypatch.delenv("SCRUTINY_MODEL", raising=False)
    assert scrutiny_model() == "gemini-3.8-flash"


def test_scrutiny_model_normalizes_vertex_and_preserves_openrouter_prefix(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from extraction_review.llm import scrutiny_model

    monkeypatch.setenv("SCRUTINY_MODEL", "google/gemini-3.8-flash")
    monkeypatch.setenv("LLM_PROVIDER", "vertex")
    assert scrutiny_model() == "gemini-3.8-flash"
    monkeypatch.setenv("LLM_PROVIDER", "openrouter")
    assert scrutiny_model() == "google/gemini-3.8-flash"


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
