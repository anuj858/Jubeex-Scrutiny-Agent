from __future__ import annotations

from types import SimpleNamespace

import httpx
import pytest
from llama_cloud import NotFoundError

from extraction_review.process_split_files import _upsert_agent_data


def _not_found(item_id: str) -> NotFoundError:
    return NotFoundError(
        f"Agent Data {item_id} not found",
        response=httpx.Response(
            404,
            request=httpx.Request(
                "GET", f"https://api.cloud.llamaindex.ai/agent-data/{item_id}"
            ),
        ),
        body={"detail": "Agent data not found"},
    )


class _AgentDataStore:
    def __init__(self) -> None:
        self.records: dict[str, dict[str, object]] = {}
        self.search_filters: list[dict[str, dict[str, str]]] = []

    async def update(self, item_id: str, *, data: dict[str, object]):
        if item_id not in self.records:
            raise _not_found(item_id)
        self.records[item_id] = dict(data)
        return SimpleNamespace(id=item_id, data=dict(data))

    async def search(self, **kwargs: object):
        filters = kwargs.get("filter")
        assert isinstance(filters, dict)
        self.search_filters.append(filters)
        items = []
        for item_id, data in self.records.items():
            if all(
                data.get(key) == condition.get("eq")
                for key, condition in filters.items()
            ):
                items.append(SimpleNamespace(id=item_id, data=dict(data)))
        return SimpleNamespace(items=items[:2])

    async def create(self, *, data: dict[str, object], **_kwargs: object):
        item_id = f"agd-{len(self.records) + 1}"
        self.records[item_id] = dict(data)
        return SimpleNamespace(id=item_id, data=dict(data))


def _client(store: _AgentDataStore) -> SimpleNamespace:
    return SimpleNamespace(beta=SimpleNamespace(agent_data=store))


@pytest.mark.asyncio
async def test_reextract_preserves_id_and_same_file_in_other_workspace_is_safe() -> (
    None
):
    store = _AgentDataStore()
    client = _client(store)
    first = {
        "organization_id": "org-1",
        "workspace_id": "ws-1",
        "file_hash": "same-hash",
        "version": 1,
    }
    created = await _upsert_agent_data(
        client,  # type: ignore[arg-type]
        data=first,
        agent_data_id=None,
        organization_id="org-1",
        workspace_id="ws-1",
    )

    updated = await _upsert_agent_data(
        client,  # type: ignore[arg-type]
        data={**first, "version": 2},
        agent_data_id=created.id,
        organization_id="org-1",
        workspace_id="ws-1",
    )
    other = await _upsert_agent_data(
        client,  # type: ignore[arg-type]
        data={**first, "workspace_id": "ws-2"},
        agent_data_id=None,
        organization_id="org-1",
        workspace_id="ws-2",
    )

    assert updated.id == created.id
    assert other.id != created.id
    assert store.records[created.id]["version"] == 2
    assert set(store.records) == {created.id, other.id}
    assert store.search_filters == [
        {
            "workspace_id": {"eq": "ws-1"},
            "organization_id": {"eq": "org-1"},
        },
        {
            "workspace_id": {"eq": "ws-2"},
            "organization_id": {"eq": "org-1"},
        },
    ]


@pytest.mark.asyncio
async def test_stale_id_creates_replacement_and_returns_it() -> None:
    store = _AgentDataStore()
    replacement = await _upsert_agent_data(
        _client(store),  # type: ignore[arg-type]
        data={"organization_id": "org-1", "workspace_id": "ws-1"},
        agent_data_id="agd-deleted",
        organization_id="org-1",
        workspace_id="ws-1",
    )

    assert replacement.id == "agd-1"
    assert set(store.records) == {"agd-1"}
