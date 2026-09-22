"""Minimal OpenRouter client for structured (JSON-schema constrained) output.

OpenRouter exposes an OpenAI-compatible chat completions endpoint, so this is a
thin `httpx` wrapper rather than a new SDK dependency. Not every model honours
`response_format: json_schema` with `strict: true`, so the client degrades to
plain JSON mode and validates with Pydantic either way.
"""

from __future__ import annotations

import asyncio
import json
import logging
import os
import random
import time
from collections import deque
from typing import Any

import httpx
from pydantic import BaseModel, ValidationError

from .scrutiny.schema import LlmUsage

logger = logging.getLogger(__name__)

DEFAULT_BASE_URL = "https://openrouter.ai/api/v1"
DEFAULT_VERTEX_PROJECT = "scrutiny-jubeex"
DEFAULT_VERTEX_LOCATION = "global"
DEFAULT_VERTEX_MODEL = "gemini-3.8-flash"
# DEFAULT_MODEL kept for OpenRouter ids.
DEFAULT_MODEL = "google/gemini-3.8-flash"
DEFAULT_TIMEOUT_S = 180.0
DEFAULT_MAX_TOKENS = 2800
MAX_ATTEMPTS = 3
DEFAULT_REQUESTS_PER_MINUTE = 0
_RATE_WINDOW_S = 60.0

_genai_client: Any = None
_genai_client_lock = asyncio.Lock()


class LLMError(RuntimeError):
    """Raised when the model cannot produce a valid structured response."""

    def __init__(self, message: str, usage: LlmUsage | None = None) -> None:
        super().__init__(message)
        self.usage = usage or LlmUsage()


class LLMFatalError(LLMError):
    """Auth / billing / permission failures that must not be retried."""


def llm_provider() -> str:
    """Active backend: ``vertex`` (GCP Gen AI SDK) or ``openrouter``."""
    raw = (os.getenv("LLM_PROVIDER") or "vertex").strip().casefold()
    if raw in {"openrouter", "or"}:
        return "openrouter"
    if raw in {"google", "gemini", "google_ai", "ai_studio"}:
        # Treat AI Studio key mode as openrouter-incompatible; use vertex enterprise.
        return "vertex"
    return "vertex"


def openrouter_api_key() -> str | None:
    return os.getenv("OPENROUTER_API_KEY")


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
    value = key if key is not None else openrouter_api_key()
    if not value:
        return "(unset)"
    if len(value) <= 16:
        return f"{value[:4]}…(len={len(value)})"
    return f"{value[:12]}…{value[-4:]} (len={len(value)})"


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
    return os.getenv("OPENROUTER_MODEL", DEFAULT_MODEL)


def openrouter_enabled() -> bool:
    if llm_provider() == "vertex":
        return bool(google_cloud_project())
    return bool(openrouter_api_key())


def openrouter_requests_per_minute() -> int:
    """Max LLM calls per rolling 60s (0 = unlimited)."""
    raw = os.getenv("OPENROUTER_REQUESTS_PER_MINUTE", str(DEFAULT_REQUESTS_PER_MINUTE))
    try:
        return max(0, int(str(raw).strip() or "0"))
    except (TypeError, ValueError):
        return DEFAULT_REQUESTS_PER_MINUTE


class _OpenRouterRateLimiter:
    """Process-wide sliding window for LLM completion calls."""

    def __init__(self) -> None:
        self._lock = asyncio.Lock()
        self._timestamps: deque[float] = deque()

    async def acquire(self) -> None:
        rpm = openrouter_requests_per_minute()
        if rpm <= 0:
            return
        while True:
            wait_s = 0.0
            async with self._lock:
                now = time.monotonic()
                while (
                    self._timestamps and now - self._timestamps[0] >= _RATE_WINDOW_S
                ):
                    self._timestamps.popleft()
                if len(self._timestamps) < rpm:
                    self._timestamps.append(now)
                    return
                wait_s = _RATE_WINDOW_S - (now - self._timestamps[0]) + 0.05
            logger.info(
                "[LLM] rate limit %s req/min reached; waiting %.1fs",
                rpm,
                wait_s,
            )
            await asyncio.sleep(max(wait_s, 0.05))


_openrouter_rate_limiter = _OpenRouterRateLimiter()


async def acquire_openrouter_slot() -> None:
    """Wait until this process may fire another LLM request."""
    await _openrouter_rate_limiter.acquire()


def get_genai_client() -> Any:
    """Lazy Google Gen AI client (Vertex / Agent Platform, ADC or WIF)."""
    global _genai_client
    if _genai_client is not None:
        return _genai_client
    try:
        from google import genai
    except ImportError as exc:
        raise LLMError(
            "google-genai is required for LLM_PROVIDER=vertex "
            "(pip install google-genai)"
        ) from exc
    project = google_cloud_project()
    location = google_cloud_location()
    _genai_client = genai.Client(
        enterprise=True,
        project=project,
        location=location,
    )
    logger.info(
        "[LLM] GenAI client ready provider=vertex project=%s location=%s",
        project,
        location,
    )
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
    if not completed.get("summary"):
        reasoning = completed.get("reasoning")
        if isinstance(reasoning, str) and reasoning.strip():
            first = reasoning.strip().split(".")[0].strip()
            completed["summary"] = f"{first}." if first else "Model omitted summary."
        else:
            completed["summary"] = "Model omitted summary."
    if completed.get("confidence") is None:
        status = str(completed.get("status") or "").strip().lower()
        # Conservative defaults when the model drops the field.
        if status in {"compliant", "defect_found"}:
            completed["confidence"] = 0.7
        else:
            completed["confidence"] = 0.5
    if not completed.get("reasoning"):
        completed["reasoning"] = (
            completed.get("summary") or "Model response was truncated."
        )
    completed.setdefault("evidence", [])
    completed.setdefault("suggested_fix", None)
    completed.setdefault("fix_rationale", None)
    return completed


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


async def _call_structured_vertex[T: BaseModel](
    *,
    system_prompt: str,
    user_prompt: str,
    response_model: type[T],
    model_name: str,
    temperature: float,
    token_limit: int | None,
) -> tuple[T, LlmUsage]:
    """Vertex / Agent Platform via google-genai (ADC locally, WIF on ECS)."""
    from google.genai import types

    client = get_genai_client()
    schema = strict_json_schema(response_model)
    usage = LlmUsage(model=model_name)
    last_error: Exception | None = None
    contents = user_prompt
    use_schema = True

    for attempt in range(1, MAX_ATTEMPTS + 1):
        try:
            await acquire_openrouter_slot()
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
                    return response_model.model_validate(
                        parsed_obj.model_dump()
                    ), usage
            if not content:
                raise LLMError("Vertex response had empty text")
            parsed = _complete_structured_fields(_parse_json(content))
            return response_model.model_validate(parsed), usage
        except ValidationError as e:
            last_error = e
            logger.warning(
                "[LLM] Vertex attempt %s/%s malformed: %s",
                attempt,
                MAX_ATTEMPTS,
                str(e)[:300],
            )
            if attempt < MAX_ATTEMPTS:
                contents = (
                    f"{user_prompt}\n\nYour previous response did not match the "
                    f"required schema:\n{str(e)[:1500]}\n\nReturn corrected JSON "
                    "matching the schema exactly. No prose, no markdown."
                )
        except Exception as e:
            last_error = e
            msg = str(e)
            lower = msg.casefold()
            if any(
                token in lower
                for token in ("permission", "unauthenticated", "403", "401", "billing")
            ):
                raise LLMFatalError(f"Vertex auth/billing: {msg[:300]}") from e
            logger.warning(
                "[LLM] Vertex attempt %s/%s failed: %s",
                attempt,
                MAX_ATTEMPTS,
                msg[:300],
            )
            if "schema" in lower and use_schema:
                use_schema = False
                logger.warning("[LLM] Falling back to JSON mime without schema")
            if attempt < MAX_ATTEMPTS:
                await asyncio.sleep(min(2**attempt, 8) + random.uniform(0, 0.5))

    if isinstance(last_error, LLMFatalError):
        raise last_error
    raise LLMError(
        f"{model_name} failed after {MAX_ATTEMPTS} attempts: {last_error}",
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
    """Multimodal JSON generation for visual marks (Vertex Gen AI SDK).

    ``images`` is a list of ``(jpeg_bytes, mime_type)`` including the page and
    optional reference sheets.
    """
    if llm_provider() != "vertex":
        raise LLMError("generate_vision_json requires LLM_PROVIDER=vertex")
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
    await acquire_openrouter_slot()
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
    usage = parse_genai_usage(response, model=model_name)
    content = (getattr(response, "text", None) or "").strip()
    if not content:
        raise LLMError("Vertex vision response had empty text", usage=usage)
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
) -> tuple[T, LlmUsage]:
    """Call Vertex (default) or OpenRouter and return validated JSON + usage."""
    model_name = model or openrouter_model()
    if llm_provider() == "vertex":
        model_name = _strip_google_model_prefix(model_name)
        token_limit = max_tokens
        if token_limit is None:
            raw_limit = os.getenv("OPENROUTER_MAX_TOKENS", "").strip()
            token_limit = int(raw_limit) if raw_limit else DEFAULT_MAX_TOKENS
        return await _call_structured_vertex(
            system_prompt=system_prompt,
            user_prompt=user_prompt,
            response_model=response_model,
            model_name=model_name,
            temperature=temperature,
            token_limit=token_limit,
        )

    schema = strict_json_schema(response_model)
    base_url = os.getenv("OPENROUTER_BASE_URL", DEFAULT_BASE_URL).rstrip("/")
    timeout = float(os.getenv("OPENROUTER_TIMEOUT_S", DEFAULT_TIMEOUT_S))
    token_limit = max_tokens
    if token_limit is None:
        raw_limit = os.getenv("OPENROUTER_MAX_TOKENS", "").strip()
        token_limit = int(raw_limit) if raw_limit else DEFAULT_MAX_TOKENS

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
    last_error: Exception | None = None
    usage = LlmUsage(model=model_name)

    try:
        for attempt in range(1, MAX_ATTEMPTS + 1):
            try:
                await acquire_openrouter_slot()
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
                parsed = _complete_structured_fields(_parse_json(content))
                finish = ""
                if isinstance(payload, dict):
                    choices = payload.get("choices") or []
                    if choices and isinstance(choices[0], dict):
                        finish = str(choices[0].get("finish_reason") or "")
                if finish == "length":
                    logger.warning(
                        "[LLM] %s hit max_tokens; salvaged truncated JSON",
                        model_name,
                    )
                value = response_model.model_validate(parsed)
                return value, usage

            except ValidationError as e:
                last_error = e
                logger.warning(
                    "[LLM] Attempt %s/%s returned malformed data: %s",
                    attempt,
                    MAX_ATTEMPTS,
                    str(e)[:300],
                )
                if attempt < MAX_ATTEMPTS:
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
                logger.error("[LLM] Fatal OpenRouter error (no retry): %s", str(e)[:300])
                break
            except (httpx.HTTPError, LLMError) as e:
                last_error = e
                logger.warning(
                    "[LLM] Attempt %s/%s failed: %s",
                    attempt,
                    MAX_ATTEMPTS,
                    str(e)[:300],
                )
                if isinstance(e, LLMError) and attempt < MAX_ATTEMPTS:
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

            if attempt < MAX_ATTEMPTS:
                await asyncio.sleep(min(2**attempt, 8) + random.uniform(0, 0.5))
    finally:
        if owns_client:
            await http.aclose()

    if isinstance(last_error, LLMFatalError):
        raise last_error
    raise LLMError(
        f"{model_name} failed after {MAX_ATTEMPTS} attempts: {last_error}",
        usage=usage,
    ) from last_error
