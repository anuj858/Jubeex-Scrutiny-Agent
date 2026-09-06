import json

import pytest

from extraction_review.callbacks import notify_job_finished, sign_callback


def test_callback_signature_is_stable() -> None:
    body = b'{"agent_job_id":"abc"}'
    first = sign_callback(body, secret="secret", timestamp="100")
    second = sign_callback(body, secret="secret", timestamp="100")
    assert first == second
    assert first != sign_callback(body, secret="other", timestamp="100")


@pytest.mark.asyncio
async def test_notify_job_finished_includes_artifact_urls(monkeypatch) -> None:
    captured: dict[str, object] = {}

    class FakeResponse:
        def raise_for_status(self) -> None:
            return None

    class FakeClient:
        def __init__(self, timeout: object = None) -> None:
            pass

        async def __aenter__(self):
            return self

        async def __aexit__(self, *args):
            return False

        async def post(self, url, content, headers):
            captured["url"] = url
            captured["body"] = json.loads(content)
            captured["headers"] = headers
            return FakeResponse()

    monkeypatch.setattr("extraction_review.callbacks.httpx.AsyncClient", FakeClient)
    await notify_job_finished(
        callback_url="https://backend.example/api/v1/webhooks/ai-agent",
        job_id="job-9",
        kind="process_file",
        status="completed",
        agent_data_id="agd-1",
        organization_id="org-1",
        workspace_id="ws-1",
        error=None,
        result={"filing_type": "SLP_CIVIL"},
        artifacts={
            "extract": {
                "step": "extract",
                "url": "https://s3.example/extract.json",
                "key": "org/org-1/filing-workspace/ws-1/job-9-v001-agent-extract-x.json",
            }
        },
        event_id="evt-1",
    )
    body = captured["body"]
    assert body["task_id"] == "job-9"
    assert body["workspace_id"] == "ws-1"
    assert body["organization_id"] == "org-1"
    assert body["artifacts"]["extract"]["url"].endswith("extract.json")
    assert body["result"]["artifacts"]["extract"]["url"].endswith("extract.json")
