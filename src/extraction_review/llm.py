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
from typing import Any

import httpx
from pydantic import BaseModel, ValidationError

logger = logging.getLogger(__name__)

DEFAULT_BASE_URL = "https://openrouter.ai/api/v1"
DEFAULT_MODEL = "openai/gpt-5.5"
DEFAULT_TIMEOUT_S = 180.0
MAX_ATTEMPTS = 3


class LLMError(RuntimeError):
    """Raised when the model cannot produce a valid structured response."""


def openrouter_api_key() -> str | None:
    return os.getenv("OPENROUTER_API_KEY")


def openrouter_model() -> str:
    return os.getenv("OPENROUTER_MODEL", DEFAULT_MODEL)


def openrouter_enabled() -> bool:
    return bool(openrouter_api_key())


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


def _parse_json(content: str) -> dict[str, Any]:
    text = content.strip()
    if text.startswith("```"):
        text = text.strip("`").strip()
        text = text.removeprefix("json").strip()
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        # Fall back to the outermost JSON object in the response.
        start, end = text.find("{"), text.rfind("}")
        if start == -1 or end <= start:
            raise LLMError(f"Response was not JSON: {content[:400]}")
        return json.loads(text[start : end + 1])


async def call_structured[T: BaseModel](
    *,
    system_prompt: str,
    user_prompt: str,
    response_model: type[T],
    model: str | None = None,
    temperature: float = 0.0,
    max_tokens: int | None = None,
    client: httpx.AsyncClient | None = None,
) -> T:
    """Call OpenRouter and return a validated instance of `response_model`."""
    model_name = model or openrouter_model()
    schema = strict_json_schema(response_model)
    base_url = os.getenv("OPENROUTER_BASE_URL", DEFAULT_BASE_URL).rstrip("/")
    timeout = float(os.getenv("OPENROUTER_TIMEOUT_S", DEFAULT_TIMEOUT_S))

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
        if max_tokens:
            payload["max_tokens"] = max_tokens
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

    try:
        for attempt in range(1, MAX_ATTEMPTS + 1):
            try:
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

                response.raise_for_status()
                content = _extract_content(response.json())
                return response_model.model_validate(_parse_json(content))

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
            except (httpx.HTTPError, LLMError) as e:
                last_error = e
                logger.warning(
                    "[LLM] Attempt %s/%s failed: %s",
                    attempt,
                    MAX_ATTEMPTS,
                    str(e)[:300],
                )

            if attempt < MAX_ATTEMPTS:
                await asyncio.sleep(min(2**attempt, 8) + random.uniform(0, 0.5))
    finally:
        if owns_client:
            await http.aclose()

    raise LLMError(
        f"{model_name} failed after {MAX_ATTEMPTS} attempts: {last_error}"
    ) from last_error
