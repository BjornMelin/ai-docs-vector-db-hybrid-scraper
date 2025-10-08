"""Tests for MCP retrieval tools covering hybrid search helpers."""

from __future__ import annotations

import sys
import types
from collections.abc import Callable
from unittest.mock import AsyncMock, MagicMock, Mock

import pytest

from src.config.models import SearchStrategy
from src.mcp_tools.models.requests import MultiStageSearchRequest, SearchRequest
from src.mcp_tools.models.responses import SearchResult
from src.mcp_tools.tools import retrieval
from src.services.vector_db.types import VectorMatch


if "src.infrastructure.client_manager" not in sys.modules:
    client_manager_stub = types.ModuleType("src.infrastructure.client_manager")

    class ClientManager:  # pragma: no cover - test stub for imports
        """Minimal client manager stub used for tool import isolation."""

        async def initialize(self) -> None:  # pragma: no cover - unused stub
            return None

    client_manager_stub.ClientManager = ClientManager  # type: ignore[attr-defined]
    sys.modules["src.infrastructure.client_manager"] = client_manager_stub


if "src.services.crawling" not in sys.modules:
    crawling_stub = types.ModuleType("src.services.crawling")

    async def crawl_page(*_args, **_kwargs) -> dict[str, object]:  # pragma: no cover
        return {}

    crawling_stub.crawl_page = crawl_page  # type: ignore[attr-defined]
    sys.modules["src.services.crawling"] = crawling_stub


@pytest.fixture
def mock_vector_service() -> Mock:
    """Provide a vector service mock primed for hybrid search calls."""

    service = Mock()
    service.is_initialized.return_value = True
    service.search_documents = AsyncMock()
    return service


@pytest.fixture
def mock_client_manager(mock_vector_service: Mock) -> Mock:
    """Return a client manager stub returning the vector service mock."""

    manager = Mock()
    manager.get_vector_store_service = AsyncMock(return_value=mock_vector_service)
    return manager


@pytest.fixture
def registered_tools(mock_client_manager: Mock) -> dict[str, Callable]:
    """Register retrieval tools and capture the exposed callables."""

    mock_mcp = MagicMock()
    tools: dict[str, Callable] = {}

    def capture(func: Callable) -> Callable:
        tools[func.__name__] = func
        return func

    mock_mcp.tool.return_value = capture
    retrieval.register_tools(mock_mcp, mock_client_manager)
    return tools


@pytest.fixture
def mock_context() -> Mock:
    """Build an MCP context mock used for logging within tools."""

    ctx = Mock()
    ctx.info = AsyncMock()
    ctx.error = AsyncMock()
    return ctx


@pytest.mark.asyncio
async def test_search_documents_returns_hybrid_results(
    registered_tools: dict[str, Callable],
    mock_vector_service: Mock,
    mock_context: Mock,
) -> None:
    """Ensure hybrid search converts matches into SearchResult records."""

    mock_vector_service.search_documents.return_value = [
        VectorMatch(
            id="doc-1",
            score=0.87,
            payload={
                "content": "body",
                "title": "Hybrid",
                "url": "https://example.com/doc",
            },
        ),
        VectorMatch(
            id="doc-2",
            score=0.45,
            payload={"content": "ignored"},
        ),
    ]

    request = SearchRequest(
        query="hybrid",
        collection="docs",
        limit=1,
        strategy=SearchStrategy.HYBRID,
        include_metadata=True,
        filters={"lang": "en"},
    )

    results = await registered_tools["search_documents"](request, mock_context)

    mock_vector_service.search_documents.assert_awaited_once_with(
        "docs", "hybrid", limit=1, filters={"lang": "en"}
    )
    assert len(results) == 1
    assert isinstance(results[0], SearchResult)
    assert results[0].id == "doc-1"
    assert results[0].score == pytest.approx(0.87)
    assert results[0].metadata == {
        "content": "body",
        "title": "Hybrid",
        "url": "https://example.com/doc",
    }
    assert results[0].content == "body"
    assert results[0].title == "Hybrid"
    assert results[0].url == "https://example.com/doc"
    mock_context.info.assert_awaited_once()


@pytest.mark.asyncio
async def test_multi_stage_search_merges_deduped_matches(
    registered_tools: dict[str, Callable],
    mock_vector_service: Mock,
    mock_context: Mock,
) -> None:
    """Verify multi-stage hybrid search dedupes IDs and keeps highest score."""

    mock_vector_service.search_documents.side_effect = [
        [
            VectorMatch(
                id="doc-1",
                score=0.55,
                payload={"content": "stage-one"},
            ),
            VectorMatch(
                id="doc-2",
                score=0.52,
                payload={"content": "first-pass"},
            ),
        ],
        [
            VectorMatch(
                id="doc-1",
                score=0.91,
                payload={"content": "stage-two"},
            ),
            VectorMatch(
                id="doc-3",
                score=0.64,
                payload={"content": "new"},
            ),
        ],
    ]

    request = MultiStageSearchRequest(
        query="hybrid",
        collection="docs",
        limit=2,
        stages=[{"limit": 2}, {"limit": 3}],
        include_metadata=True,
    )

    results = await registered_tools["multi_stage_search"](request, mock_context)

    assert mock_vector_service.search_documents.await_count == 2
    assert len(results) == 2
    assert [result.id for result in results] == ["doc-1", "doc-2"]
    assert results[0].score == pytest.approx(0.91)
    assert results[0].metadata == {"content": "stage-two"}
    assert results[1].metadata == {"content": "first-pass"}
    mock_context.info.assert_awaited()
