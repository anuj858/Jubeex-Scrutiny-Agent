import pytest

from extraction_review.llm import (
    LLMError,
    _OpenRouterRateLimiter,
    _complete_structured_fields,
    _parse_json,
    openrouter_requests_per_minute,
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
            "suggested_fix": None,
            "fix_rationale": None,
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


def test_llm_provider_openrouter_keeps_prefixed_model(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from extraction_review.llm import llm_provider, openrouter_model

    monkeypatch.setenv("LLM_PROVIDER", "openrouter")
    monkeypatch.setenv("OPENROUTER_MODEL", "google/gemini-3.8-flash")
    assert llm_provider() == "openrouter"
    assert openrouter_model() == "google/gemini-3.8-flash"


@pytest.mark.asyncio
async def test_openrouter_rate_limiter_paces_requests(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("OPENROUTER_REQUESTS_PER_MINUTE", "2")
    limiter = _OpenRouterRateLimiter()
    await limiter.acquire()
    await limiter.acquire()
    # Third acquire must wait — patch sleep to avoid a real 60s pause.
    slept: list[float] = []

    async def fake_sleep(seconds: float) -> None:
        slept.append(seconds)
        # Expire the oldest stamp so the next loop can proceed.
        if limiter._timestamps:
            limiter._timestamps.popleft()

    monkeypatch.setattr("extraction_review.llm.asyncio.sleep", fake_sleep)
    await limiter.acquire()
    assert slept and slept[0] > 0
