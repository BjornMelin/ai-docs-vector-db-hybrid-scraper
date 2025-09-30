"""Tests for the search tool registrations."""

from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from src.mcp_tools.models.requests import SearchRequest
from src.mcp_tools.models.responses import SearchResult
from src.mcp_tools.tools import search


class FakeMcp:
    """Collect functions registered via the FastMCP decorator."""

    def __init__(self) -> None:
        self.registered: dict[str, object] = {}

    def tool(self, *_, **__):  # noqa: D401 - signature mirrors FastMCP.tool
        def decorator(func):
            self.registered[func.__name__] = func
            return func

        return decorator


@pytest.fixture
def fake_mcp():
    """Provide a fake FastMCP instance capturing registered functions."""
    return FakeMcp()


@pytest.fixture
def client_manager():
    """Provide a minimal async client manager for tool wiring tests."""
    manager = AsyncMock()
    manager.get_qdrant_service = AsyncMock()
    return manager


@pytest.fixture
def context():
    """Provide a context stub matching the runtime protocol."""
    ctx = MagicMock()
    ctx.info = AsyncMock()
    ctx.debug = AsyncMock()
    ctx.warning = AsyncMock()
    ctx.error = AsyncMock()
    return ctx


@pytest.mark.asyncio
async def test_register_tools_binds_search_documents(fake_mcp, client_manager, context):
    """search_documents should delegate to search_documents_core."""
    expected = [SearchResult(id="1", content="hit", score=0.9)]
    request = SearchRequest(query="q")

    with patch(
        "src.mcp_tools.tools.search.search_documents_core",
        new=AsyncMock(return_value=expected),
    ) as mock_core:
        search.register_tools(fake_mcp, client_manager)

        result = await fake_mcp.registered["search_documents"](request, context)

    assert result == expected
    mock_core.assert_awaited_once_with(request, client_manager, context)


@pytest.mark.asyncio
async def test_search_similar_returns_results(fake_mcp, client_manager, context):
    """search_similar should retrieve a source vector and return similar results."""
    qdrant = AsyncMock()
    qdrant.get_points = AsyncMock(
        return_value=[SimpleNamespace(vector=SimpleNamespace(dense=[0.1, 0.2]))]
    )
    qdrant.hybrid_search = AsyncMock(
        return_value=[
            {
                "id": "source_doc",
                "score": 1.0,
                "payload": {"content": "source", "url": "https://example.com/src"},
            },
            {"id": "similar", "score": 0.8, "payload": {"content": "match"}},
        ]
    )
    client_manager.get_qdrant_service.return_value = qdrant

    search.register_tools(fake_mcp, client_manager)

    results = await fake_mcp.registered["search_similar"](
        query_id="source_doc",
        collection="docs",
        limit=1,
        score_threshold=0.5,
        ctx=context,
    )

    assert [result.id for result in results] == ["similar"]
    qdrant.get_points.assert_awaited_once_with(
        collection_name="docs",
        point_ids=["source_doc"],
        with_vectors=True,
        with_payload=True,
    )
    qdrant.hybrid_search.assert_awaited_once()


@pytest.mark.asyncio
async def test_search_similar_raises_for_missing_source(
    fake_mcp, client_manager, context
):
    """search_similar should raise a ValueError when the source document is missing."""
    qdrant = AsyncMock()
    qdrant.get_points = AsyncMock(return_value=[])
    client_manager.get_qdrant_service.return_value = qdrant

    search.register_tools(fake_mcp, client_manager)

    with pytest.raises(ValueError, match="Document missing-id not found"):
        await fake_mcp.registered["search_similar"](
            query_id="missing-id", collection="docs", limit=1, ctx=context
        )
