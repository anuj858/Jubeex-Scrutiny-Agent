"""Production /v1 HTTP API (no LlamaCloud workflow run)."""

from __future__ import annotations

from types import SimpleNamespace

import pytest
from fastapi.testclient import TestClient

from extraction_review.api import JOBS, app
from extraction_review.process_file import BundlePrepared


class ImmediateHandler:
    def __init__(self, result: object) -> None:
        self._result = result

    def __await__(self):
        return self._done().__await__()

    async def _done(self):
        return self._result


@pytest.fixture
def client() -> TestClient:
    JOBS.clear()
    return TestClient(app)


def test_health(client: TestClient) -> None:
    response = client.get("/v1/health")
    assert response.status_code == 200
    assert response.json() == {"status": "ok"}


def test_catalog(client: TestClient) -> None:
    response = client.get("/v1/catalog")
    assert response.status_code == 200
    body = response.json()
    assert "SLP_CIVIL" in body["filing_types"]
    assert "upload_separate" in body["job_types"]
    assert "SLP_CIVIL" in body["split_upload_types"]


def test_create_filing_rejects_empty_body(client: TestClient) -> None:
    response = client.post("/v1/filings", json={})
    assert response.status_code == 422


def test_create_filing_rejects_swagger_placeholder_filing_type(
    client: TestClient,
) -> None:
    response = client.post(
        "/v1/filings",
        json={
            "job_type": "upload_separate",
            "filing_type": "string",
            "documents": [
                {
                    "name": "01_Petition.pdf",
                    "download_url": "https://example.com/01_Petition.pdf",
                }
            ],
        },
    )
    assert response.status_code == 422
    assert "SLP_CIVIL" in response.text


def test_create_filing_rejects_unknown_filing_type(client: TestClient) -> None:
    response = client.post(
        "/v1/filings",
        json={
            "job_type": "upload_separate",
            "filing_type": "test123",
            "documents": [
                {
                    "name": "01_Petition.pdf",
                    "download_url": "https://example.com/01_Petition.pdf",
                }
            ],
        },
    )
    assert response.status_code == 422
    assert "test123" in response.text
    assert "SLP_CIVIL" in response.text


def test_create_filing_rejects_swagger_slot_id_without_mappable_name(
    client: TestClient,
) -> None:
    response = client.post(
        "/v1/filings",
        json={
            "job_type": "upload_separate",
            "filing_type": "SLP_CIVIL",
            "organization_id": "test123",
            "workspace_id": "test123",
            "user_id": "test123",
            "documents": [
                {
                    "name": "test",
                    "document_id": "test",
                    "download_url": (
                        "https://example.com/uploads/"
                        "uuid-Defect_SLP_Civil_-3_.pdf"
                    ),
                    "slot_id": "string",
                    "file_id": "string",
                    "filename": "string",
                    "file_url": "string",
                }
            ],
        },
    )
    assert response.status_code == 422
    assert "slot" in response.text.lower() or "upload_compiled" in response.text


def test_create_filing_split_one_pdf_requires_compiled(client: TestClient) -> None:
    response = client.post(
        "/v1/filings",
        json={
            "job_type": "upload_separate",
            "filing_type": "SLP_CIVIL",
            "documents": [
                {
                    "slot_id": "cover_page",
                    "name": "Defect_SLP_Civil.pdf",
                    "download_url": "https://example.com/Defect_SLP_Civil.pdf",
                    "file_id": "test123",
                }
            ],
        },
    )
    assert response.status_code == 422
    assert "upload_compiled" in response.text
    assert "Missing required documents" in response.text


def test_create_filing_and_poll(
    client: TestClient, monkeypatch: pytest.MonkeyPatch
) -> None:
    prepared = BundlePrepared(
        filing_type="SLP_CIVIL",
        agent_data_id="agd-test-1",
        organization_id="org-1",
        workspace_id="ws-1",
        user_id="user-1",
        result="agd-test-1",
    )
    monkeypatch.setattr(
        "extraction_review.api.process_file_workflow.run",
        lambda start_event: ImmediateHandler(prepared),
    )
    response = client.post(
        "/v1/filings",
        json={
            "job_type": "upload_compiled",
            "organization_id": "org-1",
            "workspace_id": "ws-1",
            "user_id": "user-1",
            "documents": [
                {
                    "name": "Defect_SLP_Civil.pdf",
                    "document_id": "doc-1",
                    "download_url": "https://example.com/compiled.pdf",
                }
            ],
        },
    )
    assert response.status_code == 202
    job_id = response.json()["job_id"]
    polled = None
    for _ in range(40):
        polled = client.get(f"/v1/jobs/{job_id}")
        if polled.json()["status"] != "running":
            break
    assert polled is not None
    body = polled.json()
    assert body["status"] == "completed"
    assert body["agent_data_id"] == "agd-test-1"
    assert body["organization_id"] == "org-1"


def test_unknown_job_is_404(client: TestClient) -> None:
    response = client.get("/v1/jobs/does-not-exist")
    assert response.status_code == 404


def test_api_key_required(monkeypatch: pytest.MonkeyPatch) -> None:
    JOBS.clear()
    monkeypatch.setenv("JUBEEX_API_KEY", "secret-key")
    guarded = TestClient(app)
    denied = guarded.get("/v1/catalog")
    assert denied.status_code == 401
    allowed = guarded.get("/v1/catalog", headers={"X-API-Key": "secret-key"})
    assert allowed.status_code == 200
    bearer = guarded.get("/v1/catalog", headers={"Authorization": "Bearer secret-key"})
    assert bearer.status_code == 200
    health = guarded.get("/v1/health")
    assert health.status_code == 200
    monkeypatch.delenv("JUBEEX_API_KEY", raising=False)


def test_get_filing_not_found(
    client: TestClient, monkeypatch: pytest.MonkeyPatch
) -> None:
    class FakeClient:
        class beta:
            class agent_data:
                @staticmethod
                async def get(_item_id: str):
                    raise RuntimeError("missing")

    monkeypatch.setattr(
        "extraction_review.api.get_llama_cloud_client",
        lambda: FakeClient(),
    )
    response = client.get("/v1/filings/agd-missing")
    assert response.status_code == 404


def test_create_scrutiny(client: TestClient, monkeypatch: pytest.MonkeyPatch) -> None:
    report = SimpleNamespace(
        model_dump=lambda mode=None: {
            "agent_data_id": "agd-scr-1",
            "summary": {"defects_found": 0},
        }
    )
    result = SimpleNamespace(
        model_dump=lambda mode=None: {
            "report": report.model_dump(),
            "type": "ScrutinyResponse",
        }
    )
    monkeypatch.setattr(
        "extraction_review.api.scrutiny_workflow.run",
        lambda start_event: ImmediateHandler(result),
    )
    response = client.post("/v1/filings/agd-scr-1/scrutiny", json={})
    assert response.status_code == 202
    job_id = response.json()["job_id"]
    polled = None
    for _ in range(40):
        polled = client.get(f"/v1/jobs/{job_id}")
        if polled.json()["status"] != "running":
            break
    assert polled is not None
    assert polled.json()["status"] == "completed"
    assert polled.json()["agent_data_id"] == "agd-scr-1"


def test_create_scrutiny_accepts_hash_and_url(
    client: TestClient, monkeypatch: pytest.MonkeyPatch
) -> None:
    captured: dict[str, object] = {}
    result = SimpleNamespace(
        model_dump=lambda mode=None: {
            "report": {"agent_data_id": "agd-scr-2"},
            "type": "ScrutinyResponse",
        }
    )

    def fake_run(start_event):
        captured["event"] = start_event
        return ImmediateHandler(result)

    monkeypatch.setattr(
        "extraction_review.api.scrutiny_workflow.run",
        fake_run,
    )
    response = client.post(
        "/v1/filings/agd-scr-2/scrutiny",
        json={
            "file_hash": "abc123hash",
            "file_url": "https://example.com/Defect_SLP_Civil.pdf",
        },
    )
    assert response.status_code == 202
    event = captured["event"]
    assert getattr(event, "file_hash") == "abc123hash"
    assert getattr(event, "file_url") == "https://example.com/Defect_SLP_Civil.pdf"


def test_create_scrutiny_accepts_download_url_alias(
    client: TestClient, monkeypatch: pytest.MonkeyPatch
) -> None:
    captured: dict[str, object] = {}
    result = SimpleNamespace(
        model_dump=lambda mode=None: {"report": {}, "type": "ScrutinyResponse"}
    )

    def fake_run(start_event):
        captured["event"] = start_event
        return ImmediateHandler(result)

    monkeypatch.setattr(
        "extraction_review.api.scrutiny_workflow.run",
        fake_run,
    )
    response = client.post(
        "/v1/filings/agd-scr-4/scrutiny",
        json={"download_url": "https://example.com/compiled.pdf"},
    )
    assert response.status_code == 202
    assert getattr(captured["event"], "file_url") == "https://example.com/compiled.pdf"


def test_create_scrutiny_drops_swagger_placeholders(
    client: TestClient, monkeypatch: pytest.MonkeyPatch
) -> None:
    captured: dict[str, object] = {}
    result = SimpleNamespace(
        model_dump=lambda mode=None: {"report": {}, "type": "ScrutinyResponse"}
    )

    def fake_run(start_event):
        captured["event"] = start_event
        return ImmediateHandler(result)

    monkeypatch.setattr(
        "extraction_review.api.scrutiny_workflow.run",
        fake_run,
    )
    response = client.post(
        "/v1/filings/agd-scr-3/scrutiny",
        json={"file_hash": "string", "file_url": "string"},
    )
    assert response.status_code == 202
    event = captured["event"]
    assert getattr(event, "file_hash") is None
    assert getattr(event, "file_url") is None
