"""Production /v1 HTTP API (no LlamaCloud workflow run)."""

from __future__ import annotations

from types import SimpleNamespace

import pytest
from fastapi.testclient import TestClient

from extraction_review.api import JOBS, app
from extraction_review.config import JUBEEX_FILING_TYPES, JUBEEX_UPLOAD_FILING_TYPES
from extraction_review.process_file import BundlePrepared


class ImmediateHandler:
    def __init__(self, result: object) -> None:
        self._result = result

    def __await__(self):
        return self._done().__await__()

    async def _done(self):
        return self._result


@pytest.fixture
def client(monkeypatch: pytest.MonkeyPatch) -> TestClient:
    JOBS.clear()
    monkeypatch.delenv("JUBEEX_API_KEY", raising=False)
    monkeypatch.delenv("JUBEEX_SQS_ENABLED", raising=False)
    monkeypatch.delenv("JUBEEX_SQS_INGESTION_QUEUE_URL", raising=False)
    monkeypatch.delenv("JUBEEX_SQS_SCRUTINY_QUEUE_URL", raising=False)

    async def _skip_prepare(_agent_data_id: str) -> None:
        return None

    monkeypatch.setattr(
        "extraction_review.api.prepare_extracted_filing_for_scrutiny",
        _skip_prepare,
    )
    return TestClient(app)


def test_health(client: TestClient) -> None:
    response = client.get("/v1/health")
    assert response.status_code == 200
    assert response.json() == {"status": "ok"}


def test_catalog(client: TestClient) -> None:
    response = client.get("/v1/catalog")
    assert response.status_code == 200
    body = response.json()
    assert body["filing_types"] == list(JUBEEX_FILING_TYPES)
    assert "CIVIL_APPEAL" in body["filing_types"]
    assert "MISCELLANEOUS_APPLICATION" in body["filing_types"]
    assert "Motor Vehical Act" in body["special_categories"]["SLP_CIVIL"]
    assert "Appeal (Armed Forces)" in body["special_categories"]["CIVIL_APPEAL"]
    assert "Motor Vehical Act" not in body["special_categories"]["SLP_CRIMINAL"]
    assert set(body["split_upload_types"]) == set(JUBEEX_UPLOAD_FILING_TYPES)
    assert "CIVIL_APPEAL" in body["split_upload_types"]
    assert "MISCELLANEOUS_APPLICATION" in body["split_upload_types"]
    assert "upload_separate" in body["job_types"]
    assert "SLP_CIVIL" in body["split_upload_types"]
    assert "TRANSFER_PETITION_CIVIL" in body["split_upload_types"]
    assert body["config"]["config_id"] == "jubeex_parse"
    assert body["config"]["classify"]["config_version"] == "1.0.0"
    assert body["config"]["extract"]["config_version"] == "1.0.0"
    assert body["config"]["split"]["config_version"] == "1.0.0"


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


def test_create_filing_maps_unlabeled_application_to_undefined(
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
                    "name": "Petition.pdf",
                    "download_url": "https://example.com/Petition.pdf",
                    "slot_id": "petition",
                },
                {
                    "name": "Application 3.pdf",
                    "download_url": "https://example.com/Application%203.pdf",
                },
            ],
        },
    )
    assert response.status_code == 202


def test_create_filing_split_accepts_whatever_documents_backend_sends(
    client: TestClient,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    prepared = BundlePrepared(
        filing_type="SLP_CIVIL",
        agent_data_id="agd-test-cover",
        result="agd-test-cover",
    )
    monkeypatch.setattr(
        "extraction_review.api.process_file_workflow.run",
        lambda start_event: ImmediateHandler(prepared),
    )
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
    assert response.status_code == 202


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


def test_approve_filing(
    client: TestClient, monkeypatch: pytest.MonkeyPatch
) -> None:
    stored = {
        "status": "pending_review",
        "file_name": "Cover Page.pdf",
        "data": {"court": None},
    }

    class FakeAgentData:
        @staticmethod
        async def get(_item_id: str):
            return SimpleNamespace(id="agd-approve-1", data=dict(stored))

        @staticmethod
        async def update(_item_id: str, data: dict | None = None, **_: object):
            stored.clear()
            stored.update(data or {})
            return SimpleNamespace(id="agd-approve-1", data=dict(stored))

    class FakeClient:
        beta = SimpleNamespace(agent_data=FakeAgentData())

    monkeypatch.setattr(
        "extraction_review.api.get_llama_cloud_client",
        lambda: FakeClient(),
    )
    response = client.patch(
        "/v1/filings/agd-approve-1",
        json={"status": "approved"},
    )
    assert response.status_code == 200
    assert response.json()["data"]["status"] == "approved"
    assert stored["status"] == "approved"


def test_approve_filing_rejects_unknown_status(client: TestClient) -> None:
    response = client.patch(
        "/v1/filings/agd-approve-1",
        json={"status": "done"},
    )
    assert response.status_code == 422


def test_get_filing_is_removed(client: TestClient) -> None:
    response = client.get("/v1/filings/agd-missing")
    assert response.status_code == 405


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


@pytest.mark.asyncio
async def test_prepare_extracted_filing_normalizes_llamaextract_error(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    stored = {"status": "error", "file_name": "Cover_Page.pdf"}

    class FakeAgentData:
        async def get(self, item_id: str):
            return SimpleNamespace(id=item_id, data=dict(stored))

        async def update(self, item_id: str, data=None, **_):
            stored.clear()
            stored.update(data or {})
            return SimpleNamespace(id=item_id, data=dict(stored))

    class FakeClient:
        def __init__(self) -> None:
            self.beta = SimpleNamespace(agent_data=FakeAgentData())

    monkeypatch.setattr(
        "extraction_review.api.get_llama_cloud_client",
        lambda: FakeClient(),
    )
    from extraction_review.api import prepare_extracted_filing_for_scrutiny

    await prepare_extracted_filing_for_scrutiny("agd-err-1")
    assert stored["status"] == "pending_review"
    assert stored["metadata"]["extract_status"] == "error"


def _doc(name: str, url: str = "https://example.com/file.pdf") -> dict[str, str]:
    return {"name": name, "download_url": url}


def test_serialize_verify_result_is_only_match_documents() -> None:
    from extraction_review.api import _serialize_result

    payload = _serialize_result(
        BundlePrepared(
            filing_type="SLP_CIVIL",
            match=False,
            verified_documents=[
                {"name": "cover_page.pdf", "match": True},
                {"name": "vakalatnama.pdf", "match": False},
            ],
            result={"match": False, "documents": []},
        )
    )
    assert payload == {
        "type": "DocumentsVerified",
        "match": False,
        "documents": [
            {"name": "cover_page.pdf", "match": True},
            {"name": "vakalatnama.pdf", "match": False},
        ],
    }


def test_catalog_lists_new_job_types(client: TestClient) -> None:
    body = client.get("/v1/catalog").json()
    for job_type in (
        "split_petition",
        "verify_document",
        "extract_only",
        "index_parsed",
    ):
        assert job_type in body["job_types"]


def test_split_petition_requires_filing_type(client: TestClient) -> None:
    response = client.post(
        "/v1/filings/split-petition",
        json={"documents": [_doc("Main_Petition.pdf")]},
    )
    assert response.status_code == 422


def test_split_petition_accepts(client: TestClient, monkeypatch: pytest.MonkeyPatch) -> None:
    prepared = BundlePrepared(filing_type="SLP_CIVIL", parsed_slots=[])
    monkeypatch.setattr(
        "extraction_review.api.process_file_workflow.run",
        lambda start_event: ImmediateHandler(prepared),
    )
    captured: dict[str, object] = {}

    def fake_run(start_event):
        captured["event"] = start_event
        return ImmediateHandler(prepared)

    monkeypatch.setattr(
        "extraction_review.api.process_file_workflow.run",
        fake_run,
    )
    response = client.post(
        "/v1/filings/split-petition",
        json={
            "filing_type": "SLP_CIVIL",
            "documents": [_doc("Main_Petition.pdf")],
        },
    )
    assert response.status_code == 202
    assert getattr(captured["event"], "job_type") == "split_petition"


def test_split_petition_post_is_not_shadowed_by_filing_id(client: TestClient) -> None:
    response = client.post("/v1/filings/split-petition", json={})
    assert response.status_code != 405


def test_verify_document_accepts_multiple_files(
    client: TestClient, monkeypatch: pytest.MonkeyPatch
) -> None:
    verified = BundlePrepared(
        filing_type="SLP_CIVIL",
        match=False,
        verified_documents=[
            {"name": "cover_page.pdf", "match": True},
            {"name": "vakalatnama.pdf", "match": False},
        ],
        result={
            "match": False,
            "documents": [
                {"name": "cover_page.pdf", "match": True},
                {"name": "vakalatnama.pdf", "match": False},
            ],
        },
    )
    monkeypatch.setattr(
        "extraction_review.api.process_file_workflow.run",
        lambda start_event: ImmediateHandler(verified),
    )
    response = client.post(
        "/v1/filings/verify-document",
        json={
            "filing_type": "SLP_CIVIL",
            "documents": [
                _doc("cover_page.pdf"),
                _doc("vakalatnama.pdf"),
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
    result = body["result"]
    assert result["match"] is False
    names = {item["name"]: item["match"] for item in result["documents"]}
    assert names["cover_page.pdf"] is True
    assert names["vakalatnama.pdf"] is False
    assert result["type"] == "DocumentsVerified"
    assert set(result.keys()) <= {"type", "match", "documents", "artifacts"}
    assert {"type", "match", "documents"} <= set(result.keys())


def test_extract_only_accepts(client: TestClient, monkeypatch: pytest.MonkeyPatch) -> None:
    prepared = BundlePrepared(
        filing_type="SLP_CIVIL",
        agent_data_id="agd-extract-1",
        parsed_slots=["petition", "cover_page"],
        result="agd-extract-1",
    )
    captured: dict[str, object] = {}

    def fake_run(start_event):
        captured["event"] = start_event
        return ImmediateHandler(prepared)

    monkeypatch.setattr(
        "extraction_review.api.process_file_workflow.run",
        fake_run,
    )
    response = client.post(
        "/v1/filings/extract",
        json={
            "filing_type": "SLP_CIVIL",
            "documents": [
                _doc("01_Petition.pdf"),
                _doc("cover_page.pdf"),
            ],
        },
    )
    assert response.status_code == 202
    assert getattr(captured["event"], "job_type") == "extract_only"


def test_index_accepts_edited_yes_and_parsed_slots(
    client: TestClient, monkeypatch: pytest.MonkeyPatch
) -> None:
    prepared = BundlePrepared(
        filing_type="SLP_CIVIL",
        agent_data_id="agd-index-1",
        result="agd-index-1",
    )
    captured: dict[str, object] = {}

    def fake_run(start_event):
        captured["event"] = start_event
        return ImmediateHandler(prepared)

    monkeypatch.setattr(
        "extraction_review.api.process_file_workflow.run",
        fake_run,
    )
    response = client.post(
        "/v1/filings/agd-index-1/index",
        json={
            "filing_type": "SLP_CIVIL",
            "edited": "yes",
            "parsed_slots": ["cover_page", "petition"],
            "documents": [
                _doc("cover_page.pdf"),
                _doc("01_Petition.pdf"),
                _doc("annexure_p1.pdf"),
            ],
        },
    )
    assert response.status_code == 202
    event = captured["event"]
    assert getattr(event, "job_type") == "index_parsed"
    assert getattr(event, "agent_data_id") == "agd-index-1"
    assert getattr(event, "edited") is True
    assert getattr(event, "parsed_slots") == ["cover_page", "petition"]


def test_index_edited_false(client: TestClient, monkeypatch: pytest.MonkeyPatch) -> None:
    prepared = BundlePrepared(filing_type="SLP_CIVIL", agent_data_id="agd-index-2")
    captured: dict[str, object] = {}
    monkeypatch.setattr(
        "extraction_review.api.process_file_workflow.run",
        lambda start_event: captured.update(event=start_event)
        or ImmediateHandler(prepared),
    )
    response = client.post(
        "/v1/filings/agd-index-2/index",
        json={
            "filing_type": "SLP_CIVIL",
            "edited": False,
            "parsed_slots": ["petition"],
            "documents": [_doc("01_Petition.pdf"), _doc("index.pdf")],
        },
    )
    assert response.status_code == 202
    assert getattr(captured["event"], "edited") is False
