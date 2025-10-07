"""Tests for the search tool registrations."""

from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

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
    """search_documents should call the vector service and filter by score."""

    vector_service = AsyncMock()
    vector_service.search_documents.return_value = [
        SimpleNamespace(id="keep", score=0.92, payload={"content": "hit"}),
        SimpleNamespace(id="drop", score=0.5, payload={"content": "low"}),
    ]

    patched_get = AsyncMock(return_value=vector_service)

    with (
        patch("src.mcp_tools.tools.search._get_vector_service", patched_get),
        patch(
            "src.mcp_tools.tools.search._format_match",
            side_effect=lambda match, method=None: {
                "id": match.id,
                "score": match.score,
                "payload": match.payload,
            },
        ),
    ):
        search.register_tools(fake_mcp, client_manager)

        result = await fake_mcp.registered["search_documents"](
            query="test",
            collection="docs",
            limit=5,
            score_threshold=0.9,
            ctx=context,
        )

    patched_get.assert_awaited()
    vector_service.search_documents.assert_awaited_once_with(
        "docs", "test", limit=5, filters=None
    )
    assert result == [{"id": "keep", "score": 0.92, "payload": {"content": "hit"}}]


@pytest.mark.asyncio
async def test_recommend_similar_returns_results(fake_mcp, client_manager, context):
    """recommend_similar should return formatted matches excluding the seed."""

    vector_service = AsyncMock()
    vector_service.get_document.return_value = SimpleNamespace(
        id="seed",
        payload={"content": "src"},
    )
    vector_service.recommend.return_value = [
        SimpleNamespace(id="seed", score=1.0, payload={}),
        SimpleNamespace(id="match", score=0.95, payload={"content": "match"}),
    ]

    patched_get = AsyncMock(return_value=vector_service)

    with (
        patch("src.mcp_tools.tools.search._get_vector_service", patched_get),
        patch(
            "src.mcp_tools.tools.search._format_match",
            side_effect=lambda match, method=None: {
                "id": match.id,
                "score": match.score,
                "payload": match.payload,
                "method": method,
            },
        ),
    ):
        search.register_tools(fake_mcp, client_manager)

        results = await fake_mcp.registered["recommend_similar"](
            point_id="seed",
            collection="docs",
            limit=1,
            score_threshold=0.5,
            ctx=context,
        )

    vector_service.get_document.assert_awaited_once_with("docs", "seed")
    vector_service.recommend.assert_awaited_once()
    assert results == [
        {
            "id": "match",
            "score": 0.95,
            "payload": {"content": "match"},
            "method": "recommend_similar",
        }
    ]


@pytest.mark.asyncio
async def test_recommend_similar_raises_for_missing_source(
    fake_mcp, client_manager, context
):
    """recommend_similar should raise when the source document is missing."""

    vector_service = AsyncMock()
    vector_service.get_document.return_value = None

    patched_get = AsyncMock(return_value=vector_service)

    with patch("src.mcp_tools.tools.search._get_vector_service", patched_get):
        search.register_tools(fake_mcp, client_manager)

        with pytest.raises(ValueError, match="Document missing-id not found"):
            await fake_mcp.registered["recommend_similar"](
                point_id="missing-id", collection="docs", limit=1, ctx=context
            )

    vector_service.get_document.assert_awaited_once_with("docs", "missing-id")
