"""Annexure index descriptions from List of Dates and annexure pages."""

from __future__ import annotations

from types import SimpleNamespace

import pytest
from fastapi import HTTPException
from fastapi.testclient import TestClient

from extraction_review.annexure_index import (
    AnnexureDescription,
    AnnexureIndexResponse,
    _mentions_annexure,
    annexure_query_terms,
    clean_index_description,
    describe_annexures,
    resolve_annexure_token,
    resolve_page_base_id,
)
from extraction_review.api import JOBS, app
from extraction_review.scrutiny.schema import LlmUsage
from extraction_review.vector_store import (
    fetch_page_texts,
    page_record_id,
    scoped_record_base_id,
)

LOD_PAGE = """
29.11.2020 ANNEXURE P-1: A true copy of the Facility Agreement dated 29.11.2020 executed between the Petitioner and Respondent No. 2 to 4. (Pg. 37-48)

15.08.2021 ANNEXURE P-3: 15.08.2021 Board Resolution. True copy. (Pages 70-72)
""".strip()

P2_FIRST_PAGE = (
    "IN THE HIGH COURT OF DELHI CS(COMM) 112/2020. "
    "Order dated 12.03.2021. Certified copy."
)

DOCUMENTS = [
    {"name": "List of Dates & Events", "start_page": 5, "end_page": 5},
    {"name": "Annexure P-1", "start_page": 37, "end_page": 48},
    {"name": "Annexure P-2", "start_page": 49, "end_page": 61},
    {"name": "Annexure P-3", "start_page": 70, "end_page": 72},
]

PAGES = {
    5: LOD_PAGE,
    37: "Facility Agreement dated 29.11.2020.",
    48: "End of the facility agreement.",
    49: P2_FIRST_PAGE,
    61: "Last page of the order.",
    70: "BOARD RESOLUTION dated 15.08.2021.",
    72: "Certified as a true copy.",
}


def _pages(*, base_id: str, pages: list[int]) -> dict[int, str]:
    assert base_id == "abc123"
    return {page: PAGES[page] for page in pages if page in PAGES}


async def _fake_llm(*, system_prompt: str, user_prompt: str, response_model, **_kwargs):
    assert "Do not include a page number" in system_prompt
    assert "Do not invent a party" in system_prompt
    assert "exact wording" in system_prompt
    assert "first page" in system_prompt.casefold()
    if "Annexure id: annexure_p1" in user_prompt:
        assert "true copy" in user_prompt.casefold()
        assert "Petitioner and Respondent No. 2 to 4" in user_prompt
        return response_model(
            source="list_of_dates",
            source_location=(
                "A true copy of the Facility Agreement dated 29.11.2020 "
                "executed between the Petitioner and Respondent No. 2 to 4. "
                "(Pg. 37-48)"
            ),
            confidence=0.94,
            description=(
                "ANNEXURE P-1: A true copy of the Facility Agreement dated "
                "29.11.2020 executed between the Petitioner and Respondent "
                "No. 2 to 4. (Pg. 37-48)"
            ),
        ), LlmUsage()
    if "Annexure id: annexure_p2" in user_prompt:
        lod, _, rest = user_prompt.partition("Annexure first page")
        assert "ANNEXURE P-2" not in lod
        assert "Certified copy" in rest
        return response_model(
            source="annexure",
            source_location=P2_FIRST_PAGE,
            confidence=0.81,
            description=(
                "A certified copy of the order dated 12.03.2021 passed by "
                "the High Court of Delhi in CS(COMM) 112/2020."
            ),
        ), LlmUsage()
    if "Annexure id: annexure_p3" in user_prompt:
        return response_model(
            source="list_of_dates",
            source_location="15.08.2021 Board Resolution. True copy. (Pages 70-72)",
            confidence=0.9,
            description="A true copy of the Board Resolution dated 15.08.2021. (Pages 70-72)",
        ), LlmUsage()
    raise AssertionError(user_prompt)


def test_clean_index_description_drops_label_and_pages() -> None:
    cleaned = clean_index_description(
        "ANNEXURE P-1: A true copy of the Facility Agreement dated 29.11.2020 "
        "executed between the Petitioner and Respondent No. 2 to 4. (Pg. 37-48)"
    )
    assert cleaned.startswith("A true copy")
    assert "ANNEXURE" not in cleaned
    assert "37" not in cleaned
    assert "Petitioner and Respondent No. 2 to 4" in cleaned


def test_annexure_query_accepts_the_rule_spellings() -> None:
    terms = annexure_query_terms("Annexure P-3")
    assert "Annexure P-3" in terms
    assert "Annexure P3" in terms
    assert "annexure 3" in terms
    for wording in ("annexure p3", "annexure p-3", "annexure 3", "ANNEXURE-P/3"):
        assert _mentions_annexure(wording, "Annexure P-3", "annexure_p3")
    assert not _mentions_annexure("annexure 30", "Annexure P-3", "annexure_p3")
    assert not _mentions_annexure("annexure r-3", "Annexure P-3", "annexure_p3")


def test_resolve_annexure_token_accepts_slot_and_label() -> None:
    slot, label, names = resolve_annexure_token("Annexure P-1")
    assert slot == "annexure_p1"
    assert label == "Annexure P-1"
    assert "Annexure P-1" in names
    assert resolve_annexure_token("annexure_r2")[0] == "annexure_r2"


def test_fetch_page_texts_follows_a_thin_first_window(monkeypatch) -> None:
    calls: list[list[str]] = []

    def fake_fetch(ids: list[str]) -> dict[str, dict[str, object]]:
        calls.append(list(ids))
        found: dict[str, dict[str, object]] = {}
        for record_id in ids:
            if record_id.endswith(":10:0"):
                found[record_id] = {"metadata": {"normalized_text": "short"}}
            elif record_id.endswith(":10:1"):
                found[record_id] = {"metadata": {"normalized_text": "x" * 600}}
        return found

    monkeypatch.setattr("extraction_review.vector_store._fetch_id_batch", fake_fetch)
    monkeypatch.setattr(
        "extraction_review.vector_store.resolve_text_field",
        lambda: "normalized_text",
    )
    pages = fetch_page_texts(base_id="abc", pages=[10], thin_chars=500, max_windows=3)
    assert page_record_id("abc", 10, 0) == "abc:page:10:0"
    assert calls == [["abc:page:10:0"], ["abc:page:10:1"]]
    assert pages[10].startswith("short")
    assert len(pages[10]) > 500


def _metadata_chunks(query: str, **kwargs: object) -> list[dict[str, object]]:
    assert query
    subtype = kwargs.get("document_subtype_id")
    part = kwargs.get("document_part")
    labels = kwargs.get("part_labels") or []
    if part == "Synopsis" or "Synopsis" in labels:
        assert kwargs.get("document_subtype_id") is None
        return [
            {
                "text": LOD_PAGE,
                "page": 5,
                "document_part": ["Synopsis", "List of Dates & Events"],
                "document_part_id": "synopsis",
            }
        ]
    pages = {
        "annexure_p1": [
            ("Facility Agreement dated 29.11.2020.", 37),
            ("End of the facility agreement.", 48),
        ],
        "annexure_p2": [
            (P2_FIRST_PAGE, 49),
            ("Last page of the order.", 61),
        ],
        "annexure_p3": [
            ("BOARD RESOLUTION dated 15.08.2021.", 70),
            ("Certified as a true copy.", 72),
        ],
    }
    rows = pages.get(str(subtype or ""))
    if not rows:
        return []
    return [
        {
            "text": text,
            "page": page,
            "document_part": f"Annexure {str(subtype)[-2:].upper()}",
            "document_part_id": "annexures",
            "document_subtype_id": subtype,
            "document_subtype_ids": [subtype],
        }
        for text, page in rows
    ]


@pytest.mark.asyncio
async def test_describe_annexures_lod_copy_type_and_fallback(monkeypatch) -> None:
    monkeypatch.setattr(
        "extraction_review.annexure_index.call_structured",
        _fake_llm,
    )
    rows = await describe_annexures(
        annexures=["annexure_p1", "annexure_p2", "annexure_p3"],
        documents=DOCUMENTS,
        file_hash="abc123",
        search_chunks=_metadata_chunks,
    )
    assert [row.annexure for row in rows] == [
        "annexure_p1",
        "annexure_p2",
        "annexure_p3",
    ]
    first, second, third = rows
    assert first.source == "list_of_dates"
    assert first.source_location.endswith("(Pg. 37-48)")
    assert "List of Dates" not in first.source_location
    assert first.description == (
        "A true copy of the Facility Agreement dated 29.11.2020 "
        "executed between the Petitioner and Respondent No. 2 to 4."
    )
    assert "ANNEXURE" not in first.description
    assert "37" not in first.description
    assert second.source == "annexure"
    assert second.source_location == P2_FIRST_PAGE
    assert second.description.startswith("A certified copy")
    assert third.source == "list_of_dates"
    assert "(Pages 70-72)" in third.source_location
    assert "Petitioner" not in third.description
    assert "Respondent" not in third.description
    assert "70" not in third.description
    assert third.description == "A true copy of the Board Resolution dated 15.08.2021."
    for row in rows:
        assert set(row.model_dump()) == {
            "annexure",
            "source",
            "source_location",
            "confidence",
            "description",
        }


def test_resolve_page_base_id_prefers_the_active_generation() -> None:
    file_hash = "a" * 64
    active = scoped_record_base_id(
        file_hash=file_hash,
        organization_id="org",
        workspace_id="ws",
        index_generation="active-gen",
    )
    payload = {
        "file_hash": file_hash,
        "metadata": {
            "vector_index": {
                "file_hash": file_hash,
                "organization_id": "org",
                "workspace_id": "ws",
                "generation": "active-gen",
            }
        },
    }

    def fetch(*, base_id: str, pages: list[int]) -> dict[int, str]:
        if base_id == active:
            return {pages[0]: "indexed page"}
        return {}

    def search(**_kwargs: object) -> list[dict[str, object]]:
        raise AssertionError("search is not used when the active generation has text")

    assert (
        resolve_page_base_id(
            payload,
            probe_pages=[33],
            fetch_pages=fetch,
            search_pages=search,
        )
        == active
    )


def test_resolve_page_base_id_uses_the_generation_that_has_pages() -> None:
    file_hash = "a" * 64
    live = "b" * 64
    payload = {
        "file_hash": file_hash,
        "metadata": {
            "vector_index": {
                "file_hash": file_hash,
                "organization_id": "org",
                "workspace_id": "ws",
                "generation": "empty-gen",
            }
        },
    }

    def fetch(*, base_id: str, pages: list[int]) -> dict[int, str]:
        if base_id == live:
            return {pages[0]: "annexure text"}
        return {}

    def search(**_kwargs: object) -> list[dict[str, object]]:
        return [{"record_id": f"{live}:page:33:0"}]

    assert (
        resolve_page_base_id(
            payload,
            probe_pages=[33],
            fetch_pages=fetch,
            search_pages=search,
        )
        == live
    )


@pytest.mark.asyncio
async def test_describe_annexures_rejects_unknown_id() -> None:
    with pytest.raises(ValueError, match="annexure_p9"):
        await describe_annexures(
            annexures=["annexure_p9"],
            documents=DOCUMENTS,
            file_hash="abc123",
            search_chunks=lambda *_args, **_kwargs: [],
        )


@pytest.fixture
def client(monkeypatch: pytest.MonkeyPatch) -> TestClient:
    JOBS.clear()
    monkeypatch.delenv("JUBEEX_API_KEY", raising=False)
    monkeypatch.setenv("JUBEEX_ALLOW_UNAUTHENTICATED", "true")
    monkeypatch.delenv("JUBEEX_SQS_ENABLED", raising=False)
    monkeypatch.delenv("JUBEEX_SQS_INGESTION_QUEUE_URL", raising=False)
    monkeypatch.delenv("JUBEEX_SQS_SCRUTINY_QUEUE_URL", raising=False)
    return TestClient(app)


def test_annexure_index_requires_annexures(client: TestClient) -> None:
    missing = client.post("/v1/filings/agd-1/annexure-index", json={})
    assert missing.status_code == 422
    empty = client.post("/v1/filings/agd-1/annexure-index", json={"annexures": []})
    assert empty.status_code == 422


def test_annexure_index_requires_scrutiny(
    client: TestClient, monkeypatch: pytest.MonkeyPatch
) -> None:
    async def _no_report(_agent_data_id: str):
        return SimpleNamespace(data={"metadata": {}, "file_hash": "abc123"})

    monkeypatch.setattr("extraction_review.api._load_filing_item", _no_report)
    response = client.post(
        "/v1/filings/agd-1/annexure-index",
        json={"annexures": ["annexure_p1"]},
    )
    assert response.status_code == 409
    assert "scrutiny" in response.text.casefold()


def test_annexure_index_unknown_filing(
    client: TestClient, monkeypatch: pytest.MonkeyPatch
) -> None:
    async def _missing(_agent_data_id: str):
        raise HTTPException(status_code=404, detail="Filing not found: agd-missing")

    monkeypatch.setattr("extraction_review.api._load_filing_item", _missing)
    response = client.post(
        "/v1/filings/agd-missing/annexure-index",
        json={"annexures": ["annexure_p1"]},
    )
    assert response.status_code == 404


def test_annexure_index_returns_one_row_per_annexure(
    client: TestClient, monkeypatch: pytest.MonkeyPatch
) -> None:
    async def _ready(_agent_data_id: str):
        return SimpleNamespace(data={"metadata": {"scrutiny_report": {"findings": []}}})

    result = AnnexureIndexResponse(
        agent_data_id="agd-1",
        annexures=[
            AnnexureDescription(
                annexure="annexure_p1",
                source="list_of_dates",
                source_location="A true copy of the agreement. (Pg. 37-48)",
                confidence=0.94,
                description="A true copy of the agreement.",
            )
        ],
    )

    def fake_run(start_event):
        assert start_event.annexures == ["annexure_p1", "annexure_p2"]
        assert start_event.agent_data_id == "agd-1"

        class _Handler:
            def __await__(self):
                async def _done():
                    return result

                return _done().__await__()

        return _Handler()

    monkeypatch.setattr("extraction_review.api._load_filing_item", _ready)
    monkeypatch.setattr("extraction_review.api.annexure_index_workflow.run", fake_run)
    response = client.post(
        "/v1/filings/agd-1/annexure-index",
        json={"annexures": ["annexure_p1", "annexure_p2"]},
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
    assert body["kind"] == "annexure_index"
    assert body["result"]["annexures"] == [
        {
            "annexure": "annexure_p1",
            "source": "list_of_dates",
            "source_location": "A true copy of the agreement. (Pg. 37-48)",
            "confidence": 0.94,
            "description": "A true copy of the agreement.",
        }
    ]
