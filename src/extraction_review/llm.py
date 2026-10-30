"""Provider-neutral Gemini/OpenRouter structured-output client.

OpenRouter exposes an OpenAI-compatible chat completions endpoint, so this is a
thin `httpx` wrapper rather than a new SDK dependency. Not every model honours
`response_format: json_schema` with `strict: true`, so the client degrades to
plain JSON mode and validates with Pydantic either way.
"""

from __future__ import annotations

import asyncio
import email.utils
import hashlib
import json
import logging
import os
import random
import time
from datetime import UTC, datetime
from typing import Any

import httpx
from pydantic import BaseModel, ValidationError

from .scrutiny.schema import LlmUsage

logger = logging.getLogger(__name__)

DEFAULT_BASE_URL = "https://openrouter.ai/api/v1"
DEFAULT_VERTEX_PROJECT = "scrutiny-jubeex"
DEFAULT_VERTEX_LOCATION = "global"
DEFAULT_VERTEX_MODEL = "gemini-4-argon"
DEFAULT_GEMINI_MODEL = "gemini-4-argon"
# DEFAULT_MODEL kept for OpenRouter ids.
DEFAULT_MODEL = "google/gemini-4-argon"
DEFAULT_SCRUTINY_VERTEX_MODEL = "gemini-4-argon"
DEFAULT_SCRUTINY_OPENROUTER_MODEL = "google/gemini-4-argon"
DEFAULT_TIMEOUT_S = 180.0
DEFAULT_VERTEX_REQUEST_TIMEOUT_S = 900
DEFAULT_GEMINI_REQUEST_TIMEOUT_S = 900
# Hard ceiling sent to the model. It is never raised.
DEFAULT_MAX_TOKENS = 9000
# The model is told to finish the whole JSON inside this. The gap up to
# 9000 is spare so a finished sentence is not cut at the ceiling.
ANSWER_TOKEN_BUDGET = 8200
MAX_ATTEMPTS = 3
DEFAULT_OPENROUTER_MAX_ATTEMPTS = 3
DEFAULT_REQUESTS_PER_MINUTE = 0
DEFAULT_VERTEX_MAX_ATTEMPTS = 6
DEFAULT_GEMINI_MAX_ATTEMPTS = 4
MAX_VERTEX_RETRY_DELAY_S = 60.0
MAX_SHARED_RATE_LIMIT_QUEUE_S = 600.0
DEFAULT_VERTEX_SHARED_COOLDOWN_BASE_S = 30.0
DEFAULT_VERTEX_SHARED_COOLDOWN_MAX_S = 300.0

_genai_client: Any = None
_genai_client_key: tuple[str, ...] | None = None
_genai_client_lock = asyncio.Lock()
_vertex_rate_limit_s3_client: Any = None


class LLMError(RuntimeError):
    """Raised when the model cannot produce a valid structured response."""

    def __init__(self, message: str, usage: LlmUsage | None = None) -> None:
        super().__init__(message)
        self.usage = usage or LlmUsage()


class LLMFatalError(LLMError):
    """Auth / billing / permission failures that must not be retried."""


def llm_provider() -> str:
    """Active backend: Gemini Developer API, Vertex, or OpenRouter."""
    raw = (os.getenv("LLM_PROVIDER") or "openrouter").strip().casefold()
    if raw in {"openrouter", "or"}:
        return "openrouter"
    if raw in {"vertex", "vertex_ai", "vertexai"}:
        return "vertex"
    if raw in {
        "google",
        "gemini",
        "genai",
        "google_ai",
        "google_ai_studio",
        "ai_studio",
        "gemini_developer_api",
    }:
        return "gemini"
    return "openrouter"


def openrouter_max_attempts() -> int:
    """Maximum OpenRouter attempts, including the initial request."""
    raw = (os.getenv("OPENROUTER_MAX_ATTEMPTS") or "").strip()
    try:
        return max(1, int(raw)) if raw else DEFAULT_OPENROUTER_MAX_ATTEMPTS
    except ValueError:
        return DEFAULT_OPENROUTER_MAX_ATTEMPTS


def openrouter_api_key() -> str | None:
    return os.getenv("OPENROUTER_API_KEY")


def gemini_api_key() -> str | None:
    """Developer API key; GOOGLE_API_KEY remains an SDK-compatible fallback."""
    return os.getenv("GEMINI_API_KEY") or os.getenv("GOOGLE_API_KEY")


def google_cloud_project() -> str:
    return (
        os.getenv("GOOGLE_CLOUD_PROJECT")
        or os.getenv("GCP_PROJECT")
        or os.getenv("GCLOUD_PROJECT")
        or DEFAULT_VERTEX_PROJECT
    ).strip()


def google_cloud_location() -> str:
    return (
        os.getenv("GOOGLE_CLOUD_LOCATION")
        or os.getenv("VERTEX_LOCATION")
        or DEFAULT_VERTEX_LOCATION
    ).strip() or DEFAULT_VERTEX_LOCATION


def mask_openrouter_api_key(key: str | None = None) -> str:
    """Safe console fingerprint — never log the full secret."""
    if llm_provider() == "vertex":
        return f"vertex:{google_cloud_project()}/{google_cloud_location()}"
    provider = llm_provider()
    value = (
        key
        if key is not None
        else (gemini_api_key() if provider == "gemini" else openrouter_api_key())
    )
    if not value:
        return "(unset)"
    if len(value) <= 16:
        return f"{provider}:{value[:4]}…(len={len(value)})"
    return f"{provider}:{value[:12]}…{value[-4:]} (len={len(value)})"


def _strip_google_model_prefix(model: str) -> str:
    raw = (model or "").strip()
    if raw.startswith("google/"):
        return raw[len("google/") :]
    return raw


def openrouter_model() -> str:
    """Resolved model id for the active provider."""
    if llm_provider() == "vertex":
        raw = (
            os.getenv("VERTEX_MODEL")
            or os.getenv("GOOGLE_MODEL")
            or os.getenv("OPENROUTER_MODEL")
            or DEFAULT_VERTEX_MODEL
        )
        return _strip_google_model_prefix(raw)
    if llm_provider() == "gemini":
        raw = (
            os.getenv("GEMINI_MODEL")
            or os.getenv("GOOGLE_MODEL")
            or os.getenv("OPENROUTER_MODEL")
            or DEFAULT_GEMINI_MODEL
        )
        return _strip_google_model_prefix(raw)
    return os.getenv("OPENROUTER_MODEL", DEFAULT_MODEL)


def scrutiny_model() -> str:
    """Model used only for registry defect checks."""
    raw = (os.getenv("SCRUTINY_MODEL") or "").strip()
    if llm_provider() in {"vertex", "gemini"}:
        return _strip_google_model_prefix(raw or DEFAULT_SCRUTINY_VERTEX_MODEL)
    return raw or DEFAULT_SCRUTINY_OPENROUTER_MODEL


def openrouter_enabled() -> bool:
    if llm_provider() == "vertex":
        return bool(google_cloud_project())
    if llm_provider() == "gemini":
        return bool(gemini_api_key())
    return bool(openrouter_api_key())


def llm_enabled() -> bool:
    """Whether the selected provider has its required authentication config."""
    return openrouter_enabled()


def llm_model() -> str:
    """Provider-neutral alias for the resolved default model."""
    return openrouter_model()


def mask_llm_credential(key: str | None = None) -> str:
    """Provider-neutral, non-secret credential fingerprint for diagnostics."""
    return mask_openrouter_api_key(key)


def openrouter_requests_per_minute() -> int:
    """Max LLM calls per rolling 60s (0 = unlimited)."""
    raw = os.getenv("OPENROUTER_REQUESTS_PER_MINUTE", str(DEFAULT_REQUESTS_PER_MINUTE))
    try:
        return max(0, int(str(raw).strip() or "0"))
    except (TypeError, ValueError):
        return DEFAULT_REQUESTS_PER_MINUTE


def llm_requests_per_minute() -> int:
    """Rate limit for the active provider.

    ``OPENROUTER_REQUESTS_PER_MINUTE`` must not throttle Vertex calls.  A
    provider-neutral override is available for installations that deliberately
    want one limit regardless of backend.
    """
    generic = (os.getenv("LLM_REQUESTS_PER_MINUTE") or "").strip()
    if generic:
        try:
            return max(0, int(generic))
        except ValueError:
            return DEFAULT_REQUESTS_PER_MINUTE
    provider = llm_provider()
    if provider == "openrouter":
        return openrouter_requests_per_minute()
    if provider == "gemini":
        raw = (os.getenv("GEMINI_REQUESTS_PER_MINUTE") or "").strip()
        try:
            return max(0, int(raw)) if raw else DEFAULT_REQUESTS_PER_MINUTE
        except ValueError:
            return DEFAULT_REQUESTS_PER_MINUTE
    raw = (os.getenv("VERTEX_REQUESTS_PER_MINUTE") or "").strip()
    try:
        return max(0, int(raw)) if raw else DEFAULT_REQUESTS_PER_MINUTE
    except ValueError:
        return DEFAULT_REQUESTS_PER_MINUTE


def vertex_max_attempts() -> int:
    """Maximum Vertex attempts, including the initial request."""
    raw = (os.getenv("VERTEX_MAX_ATTEMPTS") or "").strip()
    try:
        return max(1, int(raw)) if raw else DEFAULT_VERTEX_MAX_ATTEMPTS
    except ValueError:
        return DEFAULT_VERTEX_MAX_ATTEMPTS


def vertex_request_timeout_seconds() -> int:
    """Bound a single Argon request while allowing long reasoning responses."""
    raw = (os.getenv("VERTEX_REQUEST_TIMEOUT_S") or "").strip()
    if not raw:
        return DEFAULT_VERTEX_REQUEST_TIMEOUT_S
    try:
        timeout = int(raw)
    except ValueError as exc:
        raise ValueError("VERTEX_REQUEST_TIMEOUT_S must be an integer") from exc
    if not 60 <= timeout <= 3600:
        raise ValueError("VERTEX_REQUEST_TIMEOUT_S must be between 60 and 3600")
    return timeout


def gemini_max_attempts() -> int:
    """Maximum Developer API attempts, including the initial request."""
    raw = (os.getenv("GEMINI_MAX_ATTEMPTS") or "").strip()
    try:
        return max(1, int(raw)) if raw else DEFAULT_GEMINI_MAX_ATTEMPTS
    except ValueError:
        return DEFAULT_GEMINI_MAX_ATTEMPTS


def gemini_request_timeout_seconds() -> int:
    """Bound one Gemini Developer API request."""
    raw = (os.getenv("GEMINI_REQUEST_TIMEOUT_S") or "").strip()
    if not raw:
        return DEFAULT_GEMINI_REQUEST_TIMEOUT_S
    try:
        timeout = int(raw)
    except ValueError as exc:
        raise ValueError("GEMINI_REQUEST_TIMEOUT_S must be an integer") from exc
    if not 60 <= timeout <= 3600:
        raise ValueError("GEMINI_REQUEST_TIMEOUT_S must be between 60 and 3600")
    return timeout


def genai_max_attempts() -> int:
    return (
        gemini_max_attempts() if llm_provider() == "gemini" else vertex_max_attempts()
    )


def genai_request_timeout_seconds() -> int:
    if llm_provider() == "gemini":
        return gemini_request_timeout_seconds()
    return vertex_request_timeout_seconds()


class _OpenRouterRateLimiter:
    """Process-local, evenly paced fallback limiter."""

    def __init__(self) -> None:
        self._lock = asyncio.Lock()
        self._next_allowed_at = 0.0

    async def acquire(
        self,
        *,
        rpm: int | None = None,
        request_label: str | None = None,
    ) -> None:
        rpm = llm_requests_per_minute() if rpm is None else max(0, rpm)
        if rpm <= 0:
            return
        interval_s = 60.0 / rpm
        async with self._lock:
            now = time.monotonic()
            wait_s = max(0.0, self._next_allowed_at - now)
            self._next_allowed_at = max(now, self._next_allowed_at) + interval_s
        if wait_s > 0:
            logger.info(
                "[LLM]%s local pacing at %s req/min; waiting %.1fs",
                f"[{request_label}]" if request_label else "",
                rpm,
                wait_s,
            )
            await asyncio.sleep(max(wait_s, 0.05))


_openrouter_rate_limiter = _OpenRouterRateLimiter()


def vertex_shared_rate_limit_enabled() -> bool:
    """Use the S3-backed project limiter when workers share an artifact bucket."""
    raw = (os.getenv("VERTEX_SHARED_RATE_LIMIT") or "").strip().casefold()
    if raw:
        return raw in {"1", "true", "yes", "on"}
    bucket = (os.getenv("AWS_S3_BUCKET") or "").strip() or (
        os.getenv("JUBEEX_ARTIFACT_BUCKET") or ""
    ).strip()
    sqs = (os.getenv("JUBEEX_SQS_ENABLED") or "").strip().casefold()
    return bool(bucket) and sqs in {"1", "true", "yes", "on"}


def genai_shared_rate_limit_enabled() -> bool:
    """Coordinate Gemini/Vertex calls across every ECS worker via S3."""
    if llm_provider() != "gemini":
        return vertex_shared_rate_limit_enabled()
    raw = (os.getenv("GEMINI_SHARED_RATE_LIMIT") or "").strip().casefold()
    if raw:
        return raw in {"1", "true", "yes", "on"}
    bucket = (os.getenv("AWS_S3_BUCKET") or "").strip() or (
        os.getenv("JUBEEX_ARTIFACT_BUCKET") or ""
    ).strip()
    sqs = (os.getenv("JUBEEX_SQS_ENABLED") or "").strip().casefold()
    return bool(bucket) and sqs in {"1", "true", "yes", "on"}


def _vertex_rate_limit_bucket() -> str:
    return (os.getenv("AWS_S3_BUCKET") or "").strip() or (
        os.getenv("JUBEEX_ARTIFACT_BUCKET") or ""
    ).strip()


def _vertex_rate_limit_key(model_name: str | None) -> str:
    provider = llm_provider()
    credential_scope = ""
    if provider == "gemini" and gemini_api_key():
        credential_scope = hashlib.sha256(gemini_api_key().encode()).hexdigest()[:16]
    scope = "|".join(
        (
            provider,
            google_cloud_project(),
            google_cloud_location(),
            _strip_google_model_prefix(model_name or openrouter_model()),
            credential_scope,
        )
    )
    digest = hashlib.sha256(scope.encode("utf-8")).hexdigest()[:24]
    return f"agent-jobs/rate-limits/vertex-{digest}.json"


def _get_vertex_rate_limit_s3_client() -> Any:
    global _vertex_rate_limit_s3_client
    if _vertex_rate_limit_s3_client is not None:
        return _vertex_rate_limit_s3_client
    import boto3

    from .queue import _client_kwargs

    _vertex_rate_limit_s3_client = boto3.client("s3", **_client_kwargs())
    return _vertex_rate_limit_s3_client


def _s3_error_code(error: Exception) -> str:
    response = getattr(error, "response", None)
    if not isinstance(response, dict):
        return ""
    detail = response.get("Error")
    return str(detail.get("Code") or "") if isinstance(detail, dict) else ""


def _reserve_shared_vertex_slot(
    *,
    client: Any,
    bucket: str,
    key: str,
    interval_s: float,
) -> float:
    """Atomically reserve one globally paced request slot in S3."""
    missing_codes = {"404", "NoSuchKey", "NotFound"}
    conflict_codes = {"409", "412", "ConditionalRequestConflict", "PreconditionFailed"}
    for _attempt in range(100):
        now = time.time()
        etag: str | None = None
        next_allowed_at = now
        payload: dict[str, Any] = {}
        try:
            current = client.get_object(Bucket=bucket, Key=key)
            etag = str(current.get("ETag") or "").strip() or None
            stream = current["Body"]
            try:
                loaded = json.loads(stream.read())
                if isinstance(loaded, dict):
                    payload = loaded
            finally:
                close = getattr(stream, "close", None)
                if callable(close):
                    close()
            stored = float(payload.get("next_allowed_at") or now)
            if now <= stored <= now + MAX_SHARED_RATE_LIMIT_QUEUE_S:
                next_allowed_at = stored
            cooldown_until = float(payload.get("cooldown_until") or now)
            if now <= cooldown_until <= now + MAX_SHARED_RATE_LIMIT_QUEUE_S:
                next_allowed_at = max(next_allowed_at, cooldown_until)
        except Exception as exc:
            if _s3_error_code(exc) not in missing_codes:
                raise

        reserved_at = max(now, next_allowed_at)
        payload["next_allowed_at"] = reserved_at + interval_s
        body = json.dumps(payload, separators=(",", ":")).encode("utf-8")
        kwargs: dict[str, Any] = {
            "Bucket": bucket,
            "Key": key,
            "Body": body,
            "ContentType": "application/json",
        }
        if etag:
            kwargs["IfMatch"] = etag
        else:
            kwargs["IfNoneMatch"] = "*"
        try:
            client.put_object(**kwargs)
            return max(0.0, reserved_at - now)
        except Exception as exc:
            if _s3_error_code(exc) not in conflict_codes:
                raise
            time.sleep(random.uniform(0.01, 0.05))
    raise RuntimeError("Could not reserve a shared Vertex rate-limit slot")


def _shared_vertex_cooldown_remaining(
    *,
    client: Any,
    bucket: str,
    key: str,
) -> float:
    """Return the current project-wide Vertex cooldown, if any."""
    try:
        current = client.get_object(Bucket=bucket, Key=key)
        stream = current["Body"]
        try:
            payload = json.loads(stream.read())
        finally:
            close = getattr(stream, "close", None)
            if callable(close):
                close()
    except Exception as exc:
        if _s3_error_code(exc) in {"404", "NoSuchKey", "NotFound"}:
            return 0.0
        raise
    now = time.time()
    cooldown_until = float(payload.get("cooldown_until") or 0.0)
    return max(0.0, min(MAX_SHARED_RATE_LIMIT_QUEUE_S, cooldown_until - now))


def _extend_shared_vertex_cooldown(
    *,
    client: Any,
    bucket: str,
    key: str,
    base_delay_s: float = DEFAULT_VERTEX_SHARED_COOLDOWN_BASE_S,
    max_delay_s: float = DEFAULT_VERTEX_SHARED_COOLDOWN_MAX_S,
) -> float:
    """Atomically publish adaptive 429 backoff for every ECS worker."""
    missing_codes = {"404", "NoSuchKey", "NotFound"}
    conflict_codes = {"409", "412", "ConditionalRequestConflict", "PreconditionFailed"}
    for _attempt in range(100):
        now = time.time()
        etag: str | None = None
        payload: dict[str, Any] = {}
        try:
            current = client.get_object(Bucket=bucket, Key=key)
            etag = str(current.get("ETag") or "").strip() or None
            stream = current["Body"]
            try:
                loaded = json.loads(stream.read())
                if isinstance(loaded, dict):
                    payload = loaded
            finally:
                close = getattr(stream, "close", None)
                if callable(close):
                    close()
        except Exception as exc:
            if _s3_error_code(exc) not in missing_codes:
                raise

        previous_at = float(payload.get("last_429_at") or 0.0)
        previous_count = int(payload.get("consecutive_429s") or 0)
        if now - previous_at > max_delay_s:
            previous_count = 0
        consecutive = min(previous_count + 1, 16)
        delay_s = min(max_delay_s, base_delay_s * (2 ** (consecutive - 1)))
        cooldown_until = max(
            float(payload.get("cooldown_until") or 0.0),
            now + delay_s,
        )
        next_allowed_at = max(
            float(payload.get("next_allowed_at") or now),
            cooldown_until,
        )
        body = json.dumps(
            {
                "next_allowed_at": next_allowed_at,
                "cooldown_until": cooldown_until,
                "last_429_at": now,
                "consecutive_429s": consecutive,
            },
            separators=(",", ":"),
        ).encode("utf-8")
        kwargs: dict[str, Any] = {
            "Bucket": bucket,
            "Key": key,
            "Body": body,
            "ContentType": "application/json",
        }
        if etag:
            kwargs["IfMatch"] = etag
        else:
            kwargs["IfNoneMatch"] = "*"
        try:
            client.put_object(**kwargs)
            return max(0.0, cooldown_until - now)
        except Exception as exc:
            if _s3_error_code(exc) not in conflict_codes:
                raise
            time.sleep(random.uniform(0.01, 0.05))
    raise RuntimeError("Could not publish shared Vertex cooldown")


def _is_vertex_resource_exhausted(error: Exception) -> bool:
    text = str(error).casefold()
    return "429" in text or "resource_exhausted" in text or "resource exhausted" in text


async def _publish_vertex_cooldown(
    *,
    model_name: str,
    request_label: str | None,
) -> float:
    """Pause all workers after the active Google provider reports a 429."""
    if not genai_shared_rate_limit_enabled():
        return 0.0
    bucket = _vertex_rate_limit_bucket()
    if not bucket:
        return 0.0
    try:
        delay_s = await asyncio.to_thread(
            _extend_shared_vertex_cooldown,
            client=_get_vertex_rate_limit_s3_client(),
            bucket=bucket,
            key=_vertex_rate_limit_key(model_name),
        )
    except Exception as exc:  # noqa: BLE001 - cooldown failure must not mask LLM error
        logger.warning("Could not publish shared Vertex cooldown: %s", exc)
        return 0.0
    logger.warning(
        "[LLM]%s %s capacity exhausted; pausing all workers for %.1fs",
        f"[{request_label}]" if request_label else "",
        llm_provider(),
        delay_s,
    )
    return delay_s


async def acquire_openrouter_slot(
    *,
    model_name: str | None = None,
    request_label: str | None = None,
) -> None:
    """Wait until the active provider may fire exactly one HTTP request."""
    rpm = llm_requests_per_minute()
    if rpm <= 0:
        return
    provider = llm_provider()
    if provider not in {"vertex", "gemini"} or not genai_shared_rate_limit_enabled():
        await _openrouter_rate_limiter.acquire(rpm=rpm, request_label=request_label)
        return

    bucket = _vertex_rate_limit_bucket()
    if not bucket:
        raise LLMError("Shared GenAI rate limiting requires an artifact S3 bucket")
    client = _get_vertex_rate_limit_s3_client()
    key = _vertex_rate_limit_key(model_name)
    while True:
        try:
            wait_s = await asyncio.to_thread(
                _reserve_shared_vertex_slot,
                client=client,
                bucket=bucket,
                key=key,
                interval_s=60.0 / rpm,
            )
        except Exception as exc:
            # Failing closed prevents an S3 outage from turning every ECS task into
            # an independent burst source and worsening Vertex exhaustion.
            raise LLMError(f"Shared GenAI rate limiter unavailable: {exc}") from exc
        if wait_s > 0:
            logger.info(
                "[LLM]%s shared %s pacing at %s req/min; waiting %.1fs",
                f"[{request_label}]" if request_label else "",
                provider,
                rpm,
                wait_s,
            )
            await asyncio.sleep(max(wait_s, 0.05))
        try:
            cooldown_s = await asyncio.to_thread(
                _shared_vertex_cooldown_remaining,
                client=client,
                bucket=bucket,
                key=key,
            )
        except Exception as exc:
            raise LLMError(f"Shared GenAI rate limiter unavailable: {exc}") from exc
        if cooldown_s <= 0:
            return
        logger.info(
            "[LLM]%s shared Vertex cooldown extended; rescheduling after %.1fs",
            f"[{request_label}]" if request_label else "",
            cooldown_s,
        )


def _vertex_retry_after_seconds(error: Exception, attempt: int) -> float:
    """Return a bounded Retry-After or exponential-backoff delay."""
    response = getattr(error, "response", None)
    headers = getattr(response, "headers", None)
    retry_after = headers.get("retry-after") if headers else None
    if retry_after:
        try:
            return min(MAX_VERTEX_RETRY_DELAY_S, max(0.0, float(retry_after)))
        except (TypeError, ValueError):
            try:
                retry_at = email.utils.parsedate_to_datetime(str(retry_after))
                if retry_at.tzinfo is None:
                    retry_at = retry_at.replace(tzinfo=UTC)
                delay = (retry_at - datetime.now(UTC)).total_seconds()
                return min(MAX_VERTEX_RETRY_DELAY_S, max(0.0, delay))
            except (TypeError, ValueError, OverflowError):
                pass
    return min(MAX_VERTEX_RETRY_DELAY_S, float(2**attempt)) + random.uniform(0, 1)


def _is_retryable_vertex_error(error: Exception) -> bool:
    text = str(error).casefold()
    return any(
        marker in text
        for marker in (
            "429",
            "resource_exhausted",
            "resource exhausted",
            "408",
            "409",
            "500",
            "502",
            "503",
            "504",
            "deadline",
            "temporarily unavailable",
            "timeout",
        )
    )


def get_genai_client() -> Any:
    """Lazy Google Gen AI client for Developer API key or Vertex credentials."""
    global _genai_client, _genai_client_key
    provider = llm_provider()
    if provider == "gemini":
        api_key = gemini_api_key()
        if not api_key:
            raise LLMFatalError("GEMINI_API_KEY is required for LLM_PROVIDER=gemini")
        client_key = (provider, hashlib.sha256(api_key.encode()).hexdigest()[:12])
    elif provider == "vertex":
        client_key = (provider, google_cloud_project(), google_cloud_location())
    else:
        raise LLMError("Google Gen AI client requires LLM_PROVIDER=gemini or vertex")
    if _genai_client is not None and _genai_client_key == client_key:
        return _genai_client
    try:
        from google import genai
        from google.genai import types
    except ImportError as exc:
        raise LLMError(
            "google-genai is required for LLM_PROVIDER=gemini or vertex "
            "(pip install google-genai)"
        ) from exc
    http_options = types.HttpOptions(
        # Keep a single retry layer. SDK retries happen below our limiter and
        # would otherwise multiply bursts across concurrent worker processes.
        retry_options=types.HttpRetryOptions(attempts=1),
        timeout=genai_request_timeout_seconds() * 1000,
    )
    if provider == "gemini":
        _genai_client = genai.Client(
            api_key=api_key,
            http_options=http_options,
        )
        logger.info("[LLM] GenAI client ready provider=gemini_developer_api")
    else:
        project = google_cloud_project()
        location = google_cloud_location()
        from .aws_wif import ecs_google_credentials

        _genai_client = genai.Client(
            enterprise=True,
            project=project,
            location=location,
            credentials=ecs_google_credentials(),
            http_options=http_options,
        )
        logger.info(
            "[LLM] GenAI client ready provider=vertex project=%s location=%s",
            project,
            location,
        )
    _genai_client_key = client_key
    return _genai_client


def parse_genai_usage(response: Any, *, model: str | None = None) -> LlmUsage:
    """Map google-genai usage_metadata onto LlmUsage (cost usually unset)."""
    meta = getattr(response, "usage_metadata", None)
    if meta is None:
        return LlmUsage(model=model, calls=1)
    prompt = int(getattr(meta, "prompt_token_count", None) or 0)
    completion = int(getattr(meta, "candidates_token_count", None) or 0)
    total = int(getattr(meta, "total_token_count", None) or 0) or (prompt + completion)
    reasoning = int(getattr(meta, "thoughts_token_count", None) or 0)
    cached = int(getattr(meta, "cached_content_token_count", None) or 0)
    return LlmUsage(
        prompt_tokens=prompt,
        completion_tokens=completion,
        total_tokens=total,
        cached_tokens=cached,
        reasoning_tokens=reasoning,
        calls=1,
        cost_usd=None,
        model=model,
    )


def _headers() -> dict[str, str]:
    key = openrouter_api_key()
    if not key:
        raise LLMError("OPENROUTER_API_KEY is not set")
    headers = {
        "Authorization": f"Bearer {key}",
        "Content-Type": "application/json",
        "X-Title": os.getenv("OPENROUTER_APP_TITLE", "JubeeX Scrutiny"),
    }
    referer = os.getenv("OPENROUTER_SITE_URL")
    if referer:
        headers["HTTP-Referer"] = referer
    return headers


def strict_json_schema(model: type[BaseModel]) -> dict[str, Any]:
    """Pydantic JSON schema tightened to satisfy OpenAI-style strict mode.

    Strict mode requires every object to forbid extra properties and to list all
    of its properties as required; optional fields must be expressed as nullable
    instead.
    """
    schema = model.model_json_schema()

    def tighten(node: Any) -> None:
        if isinstance(node, list):
            for entry in node:
                tighten(entry)
            return
        if not isinstance(node, dict):
            return
        if node.get("type") == "object" or "properties" in node:
            properties = node.get("properties")
            if isinstance(properties, dict):
                node["additionalProperties"] = False
                node["required"] = list(properties.keys())
        for key, value in list(node.items()):
            if key in ("properties", "$defs", "definitions") and isinstance(
                value, dict
            ):
                for sub in value.values():
                    tighten(sub)
            elif key in ("items", "anyOf", "oneOf", "allOf", "prefixItems"):
                tighten(value)

    tighten(schema)
    return schema


def parse_openrouter_usage(
    payload: dict[str, Any], *, model: str | None = None
) -> LlmUsage:
    """Take `usage.cost` and token counts from the /chat/completions body.

    Do not estimate from list prices. Do not read cost from the model JSON.
    """
    if not isinstance(payload, dict):
        return LlmUsage(model=model)
    generation_id = payload.get("id")
    generation_id = generation_id if isinstance(generation_id, str) else None
    raw = payload.get("usage")
    if not isinstance(raw, dict):
        return LlmUsage(
            model=payload.get("model") or model,
            generation_id=generation_id,
            generation_ids=[generation_id] if generation_id else [],
        )

    prompt_details = raw.get("prompt_tokens_details") or {}
    completion_details = raw.get("completion_tokens_details") or {}
    if not isinstance(prompt_details, dict):
        prompt_details = {}
    if not isinstance(completion_details, dict):
        completion_details = {}

    cost: float | None = None
    if raw.get("cost") is not None:
        try:
            cost = float(raw["cost"])
        except (TypeError, ValueError):
            cost = None

    prompt = int(raw.get("prompt_tokens") or 0)
    completion = int(raw.get("completion_tokens") or 0)
    total = int(raw.get("total_tokens") or 0) or (prompt + completion)
    return LlmUsage(
        prompt_tokens=prompt,
        completion_tokens=completion,
        total_tokens=total,
        cached_tokens=int(prompt_details.get("cached_tokens") or 0),
        reasoning_tokens=int(completion_details.get("reasoning_tokens") or 0),
        calls=1,
        cost_usd=cost,
        model=payload.get("model") or model,
        generation_id=generation_id,
        generation_ids=[generation_id] if generation_id else [],
    )


def _extract_content(payload: dict[str, Any]) -> str:
    choices = payload.get("choices") or []
    if not choices:
        raise LLMError(f"No choices in response: {json.dumps(payload)[:400]}")
    message = choices[0].get("message") or {}
    content = message.get("content")
    if isinstance(content, list):
        # Some providers return content parts rather than a plain string.
        content = "".join(
            part.get("text", "") for part in content if isinstance(part, dict)
        )
    if not content or not str(content).strip():
        raise LLMError("Model returned empty content")
    return str(content)


def _close_truncated_json(text: str) -> str | None:
    """Close a cut-off JSON object or array so salvageable fields can still parse."""
    starts = [index for index in (text.find("{"), text.find("[")) if index != -1]
    if not starts:
        return None
    start = min(starts)
    fragment = text[start:]
    in_string = False
    escape = False
    stack: list[str] = []
    for char in fragment:
        if in_string:
            if escape:
                escape = False
            elif char == "\\":
                escape = True
            elif char == '"':
                in_string = False
            continue
        if char == '"':
            in_string = True
        elif char == "{":
            stack.append("}")
        elif char == "[":
            stack.append("]")
        elif char in ("}", "]") and stack:
            stack.pop()
    repaired = fragment
    if in_string:
        repaired += '"'
    repaired = repaired.rstrip()
    if repaired.endswith(","):
        repaired = repaired[:-1].rstrip()
    while stack:
        repaired += stack.pop()
    return repaired


def _complete_structured_fields(data: dict[str, Any]) -> dict[str, Any]:
    completed = dict(data)
    legacy = completed.get("reasoning")
    if not str(completed.get("observed") or "").strip():
        if isinstance(legacy, str) and legacy.strip():
            completed["observed"] = legacy.strip()
    if not completed.get("summary"):
        source = completed.get("observed") or ""
        if isinstance(source, str) and source.strip():
            first = source.strip().split(".")[0].strip()
            completed["summary"] = f"{first}." if first else "Model omitted summary."
        else:
            completed["summary"] = "Model omitted summary."
    if not str(completed.get("observed") or "").strip():
        completed["observed"] = (
            completed.get("summary") or "Model response was truncated."
        )
    if completed.get("confidence") is None:
        status = str(completed.get("status") or "").strip().lower()
        # Conservative defaults when the model drops the field.
        if status in {"compliant", "defect_found"}:
            completed["confidence"] = 0.7
        else:
            completed["confidence"] = 0.5
    completed.setdefault("elements", [])
    completed.setdefault("conclusion", "")
    completed.setdefault("evidence", [])
    return completed


def _loads_complete_json(content: str) -> dict[str, Any] | None:
    """Parse a finished JSON object. A reply cut off mid-sentence is not one."""
    text = content.strip()
    if text.startswith("```"):
        text = text.strip("`").strip()
        text = text.removeprefix("json").strip()
    candidates = [text]
    start, end = text.find("{"), text.rfind("}")
    if start != -1 and end > start:
        candidates.append(text[start : end + 1])
    for candidate in candidates:
        try:
            parsed = json.loads(candidate)
        except json.JSONDecodeError:
            continue
        if isinstance(parsed, dict):
            return parsed
    return None


def _answer_was_cut(content: str, *, finish_reason: str = "") -> bool:
    """True when the token budget stopped the reply before the JSON closed."""
    if _loads_complete_json(content) is None:
        return True
    return finish_reason.strip().upper() in {"MAX_TOKENS", "LENGTH"}


def output_budget_instruction() -> str:
    """Tell the model the whole structured reply must finish inside the budget."""
    return (
        f"The entire JSON object must be complete in {ANSWER_TOKEN_BUDGET} "
        "tokens or fewer. Finish every sentence and close every field inside "
        "that budget. Do not write past it."
    )


def _with_output_budget(system_prompt: str) -> str:
    note = output_budget_instruction()
    base = (system_prompt or "").rstrip()
    if note in base:
        return base
    return f"{base}\n\n{note}" if base else note


def _hard_token_limit(limit: int | None) -> int:
    """Never send a structured call more than DEFAULT_MAX_TOKENS."""
    if limit is None or limit <= 0:
        return DEFAULT_MAX_TOKENS
    return min(int(limit), DEFAULT_MAX_TOKENS)


def _finish_reason_name(response: Any) -> str:
    candidates = getattr(response, "candidates", None) or []
    if not candidates:
        return ""
    reason = getattr(candidates[0], "finish_reason", None)
    name = getattr(reason, "name", None)
    return str(name or reason or "")


def _parse_json(content: str) -> dict[str, Any]:
    text = content.strip()
    if text.startswith("```"):
        text = text.strip("`").strip()
        text = text.removeprefix("json").strip()
    candidates = [text]
    start, end = text.find("{"), text.rfind("}")
    if start != -1 and end > start:
        candidates.append(text[start : end + 1])
    repaired = _close_truncated_json(text)
    if repaired:
        candidates.append(repaired)
    last_error: Exception | None = None
    for candidate in candidates:
        try:
            parsed = json.loads(candidate)
        except json.JSONDecodeError as exc:
            last_error = exc
            continue
        if isinstance(parsed, dict):
            return parsed
        last_error = LLMError(f"Response JSON was not an object: {content[:400]}")
    raise LLMError(
        "Response was truncated or not JSON: " + content[:400]
    ) from last_error


async def _call_structured_genai[T: BaseModel](
    *,
    system_prompt: str,
    user_prompt: str,
    response_model: type[T],
    model_name: str,
    temperature: float,
    token_limit: int | None,
    request_label: str | None,
) -> tuple[T, LlmUsage]:
    """Structured output through Google GenAI (Developer API or Vertex)."""
    from google.genai import types

    client = get_genai_client()
    schema = strict_json_schema(response_model)
    usage = LlmUsage(model=model_name)
    last_error: Exception | None = None
    contents = user_prompt
    use_schema = True
    provider = llm_provider()
    provider_label = "Gemini Developer API" if provider == "gemini" else "Vertex"
    max_attempts = genai_max_attempts()
    attempts_made = 0

    for attempt in range(1, max_attempts + 1):
        attempts_made = attempt
        try:
            await acquire_openrouter_slot(
                model_name=model_name,
                request_label=request_label,
            )
            config: dict[str, Any] = {
                "temperature": temperature,
                "system_instruction": system_prompt,
                "response_mime_type": "application/json",
            }
            if token_limit:
                config["max_output_tokens"] = token_limit
            if use_schema:
                config["response_json_schema"] = schema

            response = await client.aio.models.generate_content(
                model=model_name,
                contents=contents,
                config=types.GenerateContentConfig(**config),
            )
            usage = usage.plus(parse_genai_usage(response, model=model_name))
            content = (getattr(response, "text", None) or "").strip()
            if not content and getattr(response, "parsed", None) is not None:
                parsed_obj = response.parsed
                if isinstance(parsed_obj, dict):
                    parsed = _complete_structured_fields(parsed_obj)
                    return response_model.model_validate(parsed), usage
                if isinstance(parsed_obj, BaseModel):
                    return response_model.model_validate(parsed_obj.model_dump()), usage
            if not content:
                raise LLMError(f"{provider_label} response had empty text")
            reason = _finish_reason_name(response)
            if _answer_was_cut(content, finish_reason=reason):
                logger.warning(
                    "[LLM] %s reply was cut off (%s); retrying inside "
                    "max_output_tokens=%s",
                    provider_label,
                    reason or "incomplete JSON",
                    token_limit,
                )
                if attempt < MAX_ATTEMPTS:
                    contents = (
                        f"{user_prompt}\n\n{output_budget_instruction()} "
                        "The previous reply was cut off. Return the complete "
                        "JSON inside that budget."
                    )
                    continue
                complete = _loads_complete_json(content)
                if complete is None:
                    raise LLMError(
                        "The model reply was cut off before the sentence finished"
                    )
            else:
                complete = _loads_complete_json(content)
            if complete is None:
                raise LLMError(f"{provider_label} response was not a JSON object")
            parsed = _complete_structured_fields(complete)
            return response_model.model_validate(parsed), usage
        except ValidationError as e:
            last_error = e
            logger.warning(
                "[LLM]%s %s attempt %s/%s malformed: %s",
                f"[{request_label}]" if request_label else "",
                provider_label,
                attempt,
                max_attempts,
                str(e)[:300],
            )
            if attempt < min(MAX_ATTEMPTS, max_attempts):
                contents = (
                    f"{user_prompt}\n\nYour previous response did not match the "
                    f"required schema:\n{str(e)[:1500]}\n\nReturn corrected JSON "
                    "matching the schema exactly. No prose, no markdown."
                )
            else:
                break
        except Exception as e:
            last_error = e
            msg = str(e)
            lower = msg.casefold()
            schema_retry = "schema" in lower and use_schema
            if any(
                token in lower
                for token in ("permission", "unauthenticated", "403", "401", "billing")
            ):
                raise LLMFatalError(
                    f"{provider_label} auth/billing: {msg[:300]}"
                ) from e
            logger.warning(
                "[LLM]%s %s attempt %s/%s failed: %s",
                f"[{request_label}]" if request_label else "",
                provider_label,
                attempt,
                max_attempts,
                msg[:300],
            )
            if schema_retry:
                use_schema = False
                logger.warning("[LLM] Falling back to JSON mime without schema")
            retryable = _is_retryable_vertex_error(e)
            shared_cooldown_s = 0.0
            if _is_vertex_resource_exhausted(e):
                shared_cooldown_s = await _publish_vertex_cooldown(
                    model_name=model_name,
                    request_label=request_label,
                )
            if attempt < max_attempts and (retryable or schema_retry):
                if schema_retry and not retryable:
                    continue
                delay_s = max(
                    _vertex_retry_after_seconds(e, attempt),
                    shared_cooldown_s,
                )
                logger.info(
                    "[LLM]%s Retrying %s after %.1fs (attempt %s/%s)",
                    f"[{request_label}]" if request_label else "",
                    provider_label,
                    delay_s,
                    attempt,
                    max_attempts,
                )
                await asyncio.sleep(delay_s)
            elif attempt >= MAX_ATTEMPTS or not retryable:
                break

    if isinstance(last_error, LLMFatalError):
        raise last_error
    raise LLMError(
        f"{model_name} failed after {attempts_made} attempts: {last_error}",
        usage=usage,
    ) from last_error


async def generate_vision_json(
    *,
    system_prompt: str,
    user_prompt: str,
    images: list[tuple[bytes, str]],
    model: str | None = None,
    max_tokens: int | None = None,
) -> tuple[str, LlmUsage]:
    """Multimodal JSON generation through Gemini Developer API or Vertex.

    ``images`` is a list of ``(jpeg_bytes, mime_type)`` including the page and
    optional reference sheets.
    """
    provider = llm_provider()
    if provider not in {"vertex", "gemini"}:
        raise LLMError("generate_vision_json requires LLM_PROVIDER=gemini or vertex")
    provider_label = "Gemini Developer API" if provider == "gemini" else "Vertex"
    from google.genai import types

    model_name = _strip_google_model_prefix(model or openrouter_model())
    token_limit = max_tokens
    if token_limit is None:
        raw_limit = os.getenv("VISION_MAX_OUTPUT_TOKENS", "").strip()
        token_limit = int(raw_limit) if raw_limit else 4096

    parts: list[Any] = [types.Part.from_text(text=user_prompt)]
    for data, mime in images:
        parts.append(types.Part.from_bytes(data=data, mime_type=mime or "image/jpeg"))

    client = get_genai_client()
    max_attempts = genai_max_attempts()
    last_error: Exception | None = None
    for attempt in range(1, max_attempts + 1):
        try:
            await acquire_openrouter_slot(
                model_name=model_name,
                request_label="vision",
            )
            response = await client.aio.models.generate_content(
                model=model_name,
                contents=parts,
                config=types.GenerateContentConfig(
                    temperature=0.0,
                    system_instruction=system_prompt,
                    max_output_tokens=token_limit,
                    response_mime_type="application/json",
                ),
            )
            break
        except Exception as exc:
            last_error = exc
            retryable = _is_retryable_vertex_error(exc)
            shared_cooldown_s = 0.0
            if _is_vertex_resource_exhausted(exc):
                shared_cooldown_s = await _publish_vertex_cooldown(
                    model_name=model_name,
                    request_label="vision",
                )
            if attempt >= max_attempts or not retryable:
                raise LLMError(
                    f"{model_name} vision failed after {attempt} attempts: {exc}"
                ) from exc
            delay_s = max(
                _vertex_retry_after_seconds(exc, attempt),
                shared_cooldown_s,
            )
            logger.info(
                "[LLM][vision] Retrying %s after %.1fs (attempt %s/%s)",
                provider_label,
                delay_s,
                attempt,
                max_attempts,
            )
            await asyncio.sleep(delay_s)
    else:  # pragma: no cover - the loop either succeeds or raises
        raise LLMError(f"{model_name} vision failed: {last_error}") from last_error
    usage = parse_genai_usage(response, model=model_name)
    content = (getattr(response, "text", None) or "").strip()
    if not content:
        raise LLMError(f"{provider_label} vision response had empty text", usage=usage)
    return content, usage


async def call_structured[T: BaseModel](
    *,
    system_prompt: str,
    user_prompt: str,
    response_model: type[T],
    model: str | None = None,
    temperature: float = 0.0,
    max_tokens: int | None = None,
    client: httpx.AsyncClient | None = None,
    request_label: str | None = None,
) -> tuple[T, LlmUsage]:
    """Call Gemini Developer API, Vertex, or OpenRouter with validated JSON."""
    model_name = model or openrouter_model()
    system_prompt = _with_output_budget(system_prompt)
    if llm_provider() in {"vertex", "gemini"}:
        model_name = _strip_google_model_prefix(model_name)
        token_limit = max_tokens
        if token_limit is None:
            raw_limit = os.getenv("OPENROUTER_MAX_TOKENS", "").strip()
            token_limit = int(raw_limit) if raw_limit else DEFAULT_MAX_TOKENS
        return await _call_structured_genai(
            system_prompt=system_prompt,
            user_prompt=user_prompt,
            response_model=response_model,
            model_name=model_name,
            temperature=temperature,
            token_limit=token_limit,
            request_label=request_label,
        )

    schema = strict_json_schema(response_model)
    base_url = os.getenv("OPENROUTER_BASE_URL", DEFAULT_BASE_URL).rstrip("/")
    timeout = float(os.getenv("OPENROUTER_TIMEOUT_S", DEFAULT_TIMEOUT_S))
    token_limit = max_tokens
    if token_limit is None:
        raw_limit = os.getenv("OPENROUTER_MAX_TOKENS", "").strip()
        token_limit = int(raw_limit) if raw_limit else DEFAULT_MAX_TOKENS
    token_limit = _hard_token_limit(token_limit)

    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": user_prompt},
    ]

    def body(use_json_schema: bool) -> dict[str, Any]:
        payload: dict[str, Any] = {
            "model": model_name,
            "messages": messages,
            "temperature": temperature,
        }
        if token_limit:
            payload["max_tokens"] = token_limit
        if use_json_schema:
            payload["response_format"] = {
                "type": "json_schema",
                "json_schema": {
                    "name": response_model.__name__,
                    "strict": True,
                    "schema": schema,
                },
            }
        else:
            payload["response_format"] = {"type": "json_object"}
        return payload

    owns_client = client is None
    http = client or httpx.AsyncClient(timeout=timeout)
    use_json_schema = True
    max_attempts = openrouter_max_attempts()
    last_error: Exception | None = None
    usage = LlmUsage(model=model_name)

    try:
        for attempt in range(1, max_attempts + 1):
            try:
                await acquire_openrouter_slot(
                    model_name=model_name,
                    request_label=request_label,
                )
                response = await http.post(
                    f"{base_url}/chat/completions",
                    headers=_headers(),
                    json=body(use_json_schema),
                )

                if response.status_code == 400 and use_json_schema:
                    detail = response.text[:300]
                    logger.warning(
                        "[LLM] %s rejected json_schema mode, falling back to "
                        "json_object: %s",
                        model_name,
                        detail,
                    )
                    use_json_schema = False
                    continue

                if response.status_code in (408, 409, 429) or (
                    response.status_code >= 500
                ):
                    raise httpx.HTTPStatusError(
                        f"Retryable status {response.status_code}: "
                        f"{response.text[:300]}",
                        request=response.request,
                        response=response,
                    )

                if response.status_code in (401, 402, 403):
                    logger.error(
                        "[LLM] OpenRouter fatal %s body=%s",
                        response.status_code,
                        response.text[:2000],
                    )
                    raise LLMFatalError(
                        f"OpenRouter {response.status_code} "
                        f"(auth/billing — not retried): {response.text[:300]}"
                    )

                if response.status_code >= 400:
                    logger.error(
                        "[LLM] OpenRouter error body: %s",
                        response.text,
                    )
                    raise LLMError(f"OpenRouter error: {response.text[:300]}")
                payload = response.json()
                if isinstance(payload, dict):
                    usage = usage.plus(
                        parse_openrouter_usage(payload, model=model_name)
                    )
                content = _extract_content(payload)
                finish = ""
                if isinstance(payload, dict):
                    choices = payload.get("choices") or []
                    if choices and isinstance(choices[0], dict):
                        finish = str(choices[0].get("finish_reason") or "")
                if _answer_was_cut(content, finish_reason=finish):
                    logger.warning(
                        "[LLM] %s reply was cut off (%s); retrying inside max_tokens=%s",
                        model_name,
                        finish or "incomplete JSON",
                        token_limit,
                    )
                    if attempt < max_attempts:
                        messages.append(
                            {
                                "role": "user",
                                "content": (
                                    f"{output_budget_instruction()} "
                                    "The previous reply was cut off. Return the "
                                    "complete JSON inside that budget."
                                ),
                            }
                        )
                        continue
                    complete = _loads_complete_json(content)
                    if complete is None:
                        raise LLMError(
                            "The model reply was cut off before the sentence finished"
                        )
                else:
                    complete = _loads_complete_json(content)
                if complete is None:
                    raise LLMError("Response was not a JSON object")
                parsed = _complete_structured_fields(complete)
                value = response_model.model_validate(parsed)
                return value, usage

            except ValidationError as e:
                last_error = e
                logger.warning(
                    "[LLM] Attempt %s/%s returned malformed data: %s",
                    attempt,
                    max_attempts,
                    str(e)[:300],
                )
                if attempt < max_attempts:
                    messages.append(
                        {
                            "role": "user",
                            "content": (
                                "Your previous response did not match the required "
                                f"schema:\n{str(e)[:1500]}\n\nReturn corrected JSON "
                                "matching the schema exactly. No prose, no markdown."
                            ),
                        }
                    )
            except LLMFatalError as e:
                last_error = e
                logger.error(
                    "[LLM] Fatal OpenRouter error (no retry): %s", str(e)[:300]
                )
                break
            except (httpx.HTTPError, LLMError) as e:
                last_error = e
                logger.warning(
                    "[LLM] Attempt %s/%s failed: %s",
                    attempt,
                    max_attempts,
                    str(e)[:300],
                )
                if isinstance(e, LLMError) and attempt < max_attempts:
                    messages.append(
                        {
                            "role": "user",
                            "content": (
                                "Your previous response was not valid JSON "
                                "(it may have been cut off). Return one complete "
                                "JSON object matching the schema. No prose, "
                                "no markdown."
                            ),
                        }
                    )

            if attempt < max_attempts:
                await asyncio.sleep(min(2**attempt, 8) + random.uniform(0, 0.5))
    finally:
        if owns_client:
            await http.aclose()

    if isinstance(last_error, LLMFatalError):
        raise last_error
    raise LLMError(
        f"{model_name} failed after {max_attempts} attempts: {last_error}",
        usage=usage,
    ) from last_error
