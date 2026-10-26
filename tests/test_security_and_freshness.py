from __future__ import annotations

import socket
from types import SimpleNamespace
from unittest.mock import AsyncMock

import pytest
from fastapi import HTTPException

from extraction_review import scrutiny_workflow
from extraction_review.api import _assert_filing_scope, require_api_key
from extraction_review.remote_download import download_remote_pdf, validate_remote_url
from extraction_review.scrutiny.rules import Defect
from extraction_review.scrutiny.schema import (
    DefectResponse,
    EvidenceRef,
    scrutiny_report_is_current,
)


@pytest.mark.asyncio
async def test_remote_url_rejects_http_and_unlisted_hosts(monkeypatch) -> None:
    monkeypatch.setenv("JUBEEX_REMOTE_FILE_HOSTS", "files.example.com")
    with pytest.raises(ValueError, match="HTTPS"):
        await validate_remote_url("http://files.example.com/filing.pdf")
    with pytest.raises(ValueError, match="not in"):
        await validate_remote_url("https://untrusted.example/filing.pdf")


@pytest.mark.asyncio
async def test_remote_url_rejects_private_dns(monkeypatch) -> None:
    monkeypatch.setenv("JUBEEX_REMOTE_FILE_HOSTS", "files.example.com")
    monkeypatch.setattr(
        socket,
        "getaddrinfo",
        lambda *_args, **_kwargs: [
            (socket.AF_INET, socket.SOCK_STREAM, 6, "", ("127.0.0.1", 443))
        ],
    )
    with pytest.raises(ValueError, match="private or reserved"):
        await validate_remote_url("https://files.example.com/filing.pdf")


@pytest.mark.asyncio
async def test_remote_download_enforces_streaming_size_limit(monkeypatch) -> None:
    monkeypatch.setenv("JUBEEX_MAX_PDF_BYTES", "5")

    async def allow(url: str) -> str:
        return url

    monkeypatch.setattr("extraction_review.remote_download.validate_remote_url", allow)

    class Response:
        is_redirect = False

        def __init__(self) -> None:
            self.headers = {"content-type": "application/pdf"}

        def raise_for_status(self) -> None:
            return None

        async def aiter_bytes(self):
            yield b"%PDF-"
            yield b"too-large"

        async def aclose(self) -> None:
            return None

    class Client:
        async def get(self, _url: str) -> Response:
            return Response()

    with pytest.raises(ValueError, match="size limit"):
        await download_remote_pdf(
            "https://files.example.com/filing.pdf",
            timeout_seconds=1,
            client=Client(),  # type: ignore[arg-type]
        )


def test_authentication_fails_closed_without_configuration(monkeypatch) -> None:
    monkeypatch.delenv("JUBEEX_API_KEY", raising=False)
    monkeypatch.delenv("JUBEEX_ALLOW_UNAUTHENTICATED", raising=False)
    with pytest.raises(HTTPException) as error:
        require_api_key(None, None)
    assert error.value.status_code == 503


def test_filing_scope_rejects_cross_workspace_access() -> None:
    data = {
        "metadata": {
            "vector_index": {
                "organization_id": "org-1",
                "workspace_id": "workspace-1",
            }
        }
    }
    with pytest.raises(HTTPException) as error:
        _assert_filing_scope(data, organization_id="org-1", workspace_id="workspace-2")
    assert error.value.status_code == 403


def test_scrutiny_report_freshness_binds_hash_and_generation() -> None:
    report = {"file_hash": "a" * 64, "index_generation": "generation-2"}
    assert scrutiny_report_is_current(
        report, file_hash="a" * 64, index_generation="generation-2"
    )
    assert not scrutiny_report_is_current(
        report, file_hash="b" * 64, index_generation="generation-2"
    )
    assert not scrutiny_report_is_current(
        report, file_hash="a" * 64, index_generation="generation-3"
    )


def _missing_part_defect(*, shortcut: bool) -> Defect:
    return Defect(
        check_id="D-TEST",
        main_category="General/Global",
        defect="Appendix is missing",
        requirement="An Appendix must be filed.",
        where_to_look=["Appendix"],
        inspect_parts=["Appendix"],
        manifest_absence_proves_defect=shortcut,
        how_to_cure=["File the Appendix."],
        location_source="",
    )


@pytest.mark.asyncio
async def test_missing_layout_part_is_not_automatic_defect_without_switch(
    monkeypatch,
) -> None:
    response = DefectResponse(
        check_id="D-TEST",
        status="needs_review",
        confidence=0.4,
        summary="The available evidence is incomplete.",
        evidence=[],
    )
    call = AsyncMock(return_value=(response, None))
    monkeypatch.setattr(scrutiny_workflow, "call_structured", call)
    monkeypatch.setattr(scrutiny_workflow, "build_system_prompt", lambda *_args: "")
    monkeypatch.setattr(
        scrutiny_workflow, "build_defect_prompt", lambda *_args, **_kwargs: ""
    )
    monkeypatch.setattr(
        scrutiny_workflow,
        "build_finding",
        lambda _defect, result, **_kwargs: SimpleNamespace(status=result.status),
    )

    finding = await scrutiny_workflow._run_defect(
        _missing_part_defect(shortcut=False),
        catalogue=SimpleNamespace(),
        record={},
        chunks=[],
        filing_type="SLP_CIVIL",
        layout={1: {"document_part": "Main Petition"}},
    )

    assert finding.status == "needs_review"
    call.assert_awaited_once()


def test_sanitization_preserves_multiple_grounded_evidence_items() -> None:
    defect = _missing_part_defect(shortcut=False)
    chunks = [
        {
            "chunk_kind": "page",
            "page": 1,
            "document_part": "Appendix",
            "text": "Clause A is present.",
        },
        {
            "chunk_kind": "page",
            "page": 2,
            "document_part": "Appendix",
            "text": "Clause B is present.",
        },
    ]
    response = DefectResponse(
        check_id="D-TEST",
        status="compliant",
        confidence=0.95,
        summary="Both required clauses are present.",
        evidence=[
            EvidenceRef(chunk_id="c1", quote="Clause A is present."),
            EvidenceRef(chunk_id="c2", quote="Clause B is present."),
        ],
    )

    sanitized = scrutiny_workflow._sanitize_response(defect, response, chunks)

    assert [item.page for item in sanitized.evidence] == [1, 2]
