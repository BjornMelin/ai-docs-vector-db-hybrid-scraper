"""Tests for the simplified MCP search tools."""

from __future__ import annotations

from collections.abc import Callable
from unittest.mock import AsyncMock, MagicMock, Mock

import pytest

from src.config import SearchStrategy
from src.mcp_tools.models.requests import (
    FilteredSearchRequest,
    HyDESearchRequest,
    MultiStageSearchRequest,
    SearchRequest,
)
from src.mcp_tools.models.responses import SearchResult
from src.mcp_tools.tools.search_tools import register_tools
from src.services.vector_db.adapter_base import VectorMatch


@pytest.fixture
def mock_vector_service() -> Mock:
    service = Mock()
    service.is_initialized.return_value = True
    service.search_documents = AsyncMock()
    service.hybrid_search = AsyncMock()
    service.list_documents = AsyncMock()
    service.collection_stats = AsyncMock()
    service.list_collections = AsyncMock()
    service.ensure_collection = AsyncMock()
    return service


@pytest.fixture
async def mock_client_manager(mock_vector_service: Mock) -> Mock:
    manager = Mock()
    manager.get_vector_store_service = AsyncMock(return_value=mock_vector_service)
    return manager


@pytest.fixture
def register(mock_client_manager: Mock) -> dict[str, Callable]:
    mock_mcp = MagicMock()
    registered_tools: dict[str, Callable] = {}

    def capture(func: Callable) -> Callable:
        registered_tools[func.__name__] = func
        return func

    mock_mcp.tool.return_value = capture
    register_tools(mock_mcp, mock_client_manager)
    return registered_tools


@pytest.fixture
async def mock_context() -> Mock:
    ctx = Mock()
    ctx.info = AsyncMock()
    ctx.debug = AsyncMock()
    ctx.warning = AsyncMock()
    ctx.error = AsyncMock()
    return ctx


def _match(content: str, score: float = 0.5) -> VectorMatch:
    return VectorMatch(id=content, score=score, payload={"content": content})


@pytest.mark.asyncio
async def test_registers_expected_tools(register: dict[str, Callable]) -> None:
    assert {
        "search_documents",
        "multi_stage_search",
        "hyde_search",
        "filtered_search",
    } <= set(register)


@pytest.mark.asyncio
async def test_search_documents_dense(
    register: dict[str, Callable],
    mock_client_manager: Mock,
    mock_vector_service: Mock,
    mock_context: Mock,
) -> None:
    mock_vector_service.search_documents.return_value = [_match("result", 0.9)]

    request = SearchRequest(
        query="test",
        collection="docs",
        strategy=SearchStrategy.DENSE,
        limit=3,
    )
    results = await register["search_documents"](request, mock_context)

    mock_vector_service.search_documents.assert_awaited_once_with(
        "docs", "test", limit=3, filters=None
    )
    expected = SearchResult(
        id="result",
        content="result",
        score=0.9,
        metadata={"content": "result"},
    )
    assert results == [expected]


@pytest.mark.asyncio
async def test_search_documents_hybrid(
    register: dict[str, Callable], mock_vector_service: Mock, mock_context: Mock
) -> None:
    mock_vector_service.hybrid_search.return_value = [_match("hybrid")]

    request = SearchRequest(query="a", collection="docs", limit=2)
    results = await register["search_documents"](request, mock_context)

    mock_vector_service.hybrid_search.assert_awaited_once_with(
        "docs", "a", limit=2, filters=None
    )
    assert len(results) == 1


@pytest.mark.asyncio
async def test_multi_stage_search_deduplicates(
    register: dict[str, Callable], mock_vector_service: Mock, mock_context: Mock
) -> None:
    mock_vector_service.hybrid_search.side_effect = [
        [_match("doc-1", 0.7), _match("doc-2", 0.6)],
        [_match("doc-2", 0.8), _match("doc-3", 0.5)],
    ]

    request = MultiStageSearchRequest(
        query="multi",
        collection="docs",
        stages=[{"limit": 2}, {"filters": {"type": "guide"}}],
        limit=3,
    )
    results = await register["multi_stage_search"](request, mock_context)

    assert [result.id for result in results] == ["doc-2", "doc-1", "doc-3"]
    assert mock_vector_service.hybrid_search.await_count == 2


@pytest.mark.asyncio
async def test_filtered_search(
    register: dict[str, Callable], mock_vector_service: Mock, mock_context: Mock
) -> None:
    mock_vector_service.hybrid_search.return_value = [_match("filtered")]

    request = FilteredSearchRequest(
        query="filters",
        collection="docs",
        filters={"type": "api"},
    )
    results = await register["filtered_search"](request, mock_context)

    mock_vector_service.hybrid_search.assert_awaited_once_with(
        "docs", "filters", limit=10, filters={"type": "api"}
    )
    assert results[0].metadata == {"content": "filtered"}


@pytest.mark.asyncio
async def test_hyde_search_uses_hybrid(
    register: dict[str, Callable], mock_vector_service: Mock, mock_context: Mock
) -> None:
    mock_vector_service.hybrid_search.return_value = [_match("hyde", 0.4)]

    request = HyDESearchRequest(query="hyde", collection="docs", limit=1)
    results = await register["hyde_search"](request, mock_context)

    mock_vector_service.hybrid_search.assert_awaited_once()
    assert results[0].score == 0.4
