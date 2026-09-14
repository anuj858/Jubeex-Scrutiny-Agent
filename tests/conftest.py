"""Pytest configuration: install the LlamaCloud fake server when compatible."""

from __future__ import annotations

import json
import logging
import sys
from typing import Any

import httpx
import pytest

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)],
)


def _install_llama_cloud_compat(server: object) -> None:
    """llama-cloud-fake 0.1.x still mocks the older files/split URLs.

    llama-cloud 2.14 retrieve is GET /api/v1/beta/files/{id} and split is
    /api/v1/split/jobs with file_input instead of document_input.
    """
    from llama_cloud_fake import FakeLlamaCloudServer

    if not isinstance(server, FakeLlamaCloudServer):
        return

    def _handle_retrieve(request: httpx.Request) -> httpx.Response:
        file_id = request.url.path.rstrip("/").split("/")[-1]
        stored = server.files.get(file_id)
        if stored is None:
            return server.json_response({"detail": "File not found"}, status_code=404)
        return server.json_response(stored.file.model_dump(mode="json"))

    def _rewrite_json(request: httpx.Request, payload: dict[str, Any]) -> httpx.Request:
        return httpx.Request(
            method=request.method,
            url=str(request.url),
            headers=request.headers,
            content=json.dumps(payload).encode("utf-8"),
        )

    def _split_request_to_beta(payload: dict[str, Any]) -> dict[str, Any]:
        adapted = dict(payload)
        file_input = payload.get("file_input")
        if file_input and not payload.get("document_input"):
            adapted["document_input"] = {"type": "file_id", "value": file_input}
        return adapted

    def _split_job_to_v1(payload: dict[str, Any]) -> dict[str, Any]:
        document = payload.get("document_input") or {}
        adapted = dict(payload)
        adapted.pop("document_input", None)
        adapted["document_input_type"] = document.get("type") or "file_id"
        adapted["file_input"] = document.get("value") or payload.get("file_input") or ""
        result = adapted.get("result")
        if isinstance(result, dict):
            segments = result.get("segments") or []
            if not segments:
                categories = adapted.get("categories") or []
                name = None
                if categories and isinstance(categories[0], dict):
                    name = categories[0].get("name")
                elif categories:
                    name = getattr(categories[0], "name", None)
                if name:
                    adapted["result"] = {
                        **result,
                        "segments": [
                            {
                                "category": name,
                                "confidence_category": "high",
                                "pages": [1],
                            }
                        ],
                    }
        return adapted

    def _handle_split_create(request: httpx.Request) -> httpx.Response:
        payload = json.loads(request.content.decode("utf-8") or "{}")
        beta_request = _rewrite_json(request, _split_request_to_beta(payload))
        response = server.split._create_split_job(beta_request)
        if response.status_code >= 400:
            return response
        body = json.loads(response.content.decode("utf-8") or "{}")
        return server.json_response(_split_job_to_v1(body), status_code=response.status_code)

    def _handle_split_get(request: httpx.Request) -> httpx.Response:
        response = server.split._get_split_job_result(request)
        if response.status_code >= 400:
            return response
        body = json.loads(response.content.decode("utf-8") or "{}")
        return server.json_response(_split_job_to_v1(body))

    server.add_route(
        "GET",
        "/api/v1/beta/files/{file_id}",
        _handle_retrieve,
        namespace="files",
        alias="retrieve",
    )
    server.add_route(
        "POST",
        "/api/v1/split/jobs",
        _handle_split_create,
        namespace="split",
        alias="create_v1",
    )
    server.add_route(
        "GET",
        "/api/v1/split/jobs/{split_job_id}",
        _handle_split_get,
        namespace="split",
        alias="get_v1",
    )


_fake = None
_fake_error: BaseException | None = None
try:
    from llama_cloud_fake import FakeLlamaCloudServer

    _fake = FakeLlamaCloudServer().install()
    _install_llama_cloud_compat(_fake)
except Exception as exc:  # pragma: no cover - environment-dependent
    _fake_error = exc


@pytest.fixture
def fake():
    if _fake is None:
        pytest.skip(f"llama-cloud-fake is incompatible with this llama-cloud: {_fake_error}")
    return _fake
