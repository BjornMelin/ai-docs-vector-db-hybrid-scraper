"""Tests for MCP retrieval tools covering hybrid search helpers."""

from __future__ import annotations

from collections.abc import Callable
from typing import Any
from unittest.mock import AsyncMock, MagicMock, Mock

import pytest

from src.config.models import SearchStrategy
from src.contracts.retrieval import SearchRecord
from src.mcp_tools.tools import retrieval
from src.models.search import SearchRequest


class StubMCP:
    """Stub FastMCP server recording registered tools."""

    def __init__(self) -> None:
        """Initialize the stub MCP with an empty tool registry."""
        self.tools: dict[str, Callable[..., Any]] = {}

    def tool(self, *_, **__):  # pragma: no cover
        """Decorator to register a tool function by name."""

        def decorator(func: Callable[..., Any]) -> Callable[..., Any]:
            self.tools[func.__name__] = func
            return func

        return decorator


@pytest.fixture
def mock_vector_service() -> Mock:
    """Provide a vector service mock primed for hybrid search calls."""
    service = Mock()
    service.is_initialized.return_value = True
    service.initialize = AsyncMock()
    service.search_documents = AsyncMock()
    service.ensure_payload_indexes = AsyncMock()
    return service


@pytest.fixture
def registered_tools(mock_vector_service: Mock) -> dict[str, Callable[..., Any]]:
    """Register retrieval tools and capture the exposed callables."""
    mcp = StubMCP()
    retrieval.register_tools(mcp, vector_service=mock_vector_service)
    return mcp.tools


@pytest.fixture
def mock_context() -> MagicMock:
    """Build an MCP context mock used for logging within tools."""
    ctx = MagicMock()
    ctx.info = AsyncMock()
    ctx.error = AsyncMock()
    return ctx


@pytest.mark.asyncio
async def test_search_documents_returns_hybrid_results(
    registered_tools: dict[str, Callable[..., Any]],
    mock_vector_service: Mock,
    mock_context: MagicMock,
) -> None:
    """Ensure hybrid search returns canonical search records."""
    mock_vector_service.search_documents.return_value = [
        SearchRecord(
            id="doc-1",
            score=0.87,
            content="body",
            title="Hybrid",
            url="https://example.com/doc",
            metadata={
                "content": "body",
                "title": "Hybrid",
                "url": "https://example.com/doc",
            },
        ),
        SearchRecord(
            id="doc-2",
            score=0.45,
            content="ignored",
            metadata={"content": "ignored"},
        ),
    ]

    request = SearchRequest(
        query="hybrid",
        collection="docs",
        limit=1,
        offset=0,
        search_strategy=SearchStrategy.HYBRID,
        include_metadata=True,
        filters={"lang": "en"},
    )

    results = await registered_tools["search_documents"](request, mock_context)

    mock_vector_service.search_documents.assert_awaited_once_with(
        "docs", "hybrid", limit=1, filters={"lang": "en"}
    )
    assert len(results) == 1
    assert isinstance(results[0], SearchRecord)
    assert results[0].id == "doc-1"
    assert results[0].metadata["title"] == "Hybrid"


@pytest.mark.asyncio
async def test_multi_stage_search_merges_deduped_matches(
    registered_tools: dict[str, Callable[..., Any]],
    mock_vector_service: Mock,
    mock_context: MagicMock,
) -> None:
    """Verify multi-stage hybrid search dedupes IDs and keeps highest score."""
    mock_vector_service.search_documents.side_effect = [
        [
            SearchRecord(
                id="doc-1",
                score=0.55,
                content="stage-one",
                metadata={"content": "stage-one"},
            ),
            SearchRecord(
                id="doc-2",
                score=0.52,
                content="first-pass",
                metadata={"content": "first-pass"},
            ),
        ],
        [
            SearchRecord(
                id="doc-1",
                score=0.91,
                content="stage-two",
                metadata={"content": "stage-two"},
            ),
            SearchRecord(
                id="doc-3",
                score=0.64,
                content="new",
                metadata={"content": "new"},
            ),
        ],
    ]

    payload = retrieval.MultiStageSearchPayload(
        query="hybrid",
        collection="docs",
        limit=2,
        stages=[{"limit": 2}, {"limit": 3}],
        include_metadata=True,
    )

    results = await registered_tools["multi_stage_search"](payload, mock_context)

    assert mock_vector_service.search_documents.await_count == 2
    assert len(results) == 2
    assert [result.id for result in results] == ["doc-1", "doc-2"]
    assert results[0].score == pytest.approx(0.91)
    assert results[0].metadata == {"content": "stage-two"}


@pytest.mark.asyncio
async def test_filtered_search_forces_hybrid_strategy(
    registered_tools: dict[str, Callable[..., Any]],
    mock_vector_service: Mock,
    mock_context: MagicMock,
) -> None:
    """Filtered search should normalise to hybrid strategy."""
    mock_vector_service.search_documents.return_value = []
    request = SearchRequest(
        query="term",
        collection="docs",
        limit=5,
        offset=0,
        search_strategy=SearchStrategy.DENSE,
        include_metadata=False,
        filters={"lang": "en"},
    )

    await registered_tools["filtered_search"](request, mock_context)

    args = mock_vector_service.search_documents.await_args[0]
    assert args == ("docs", "term")


@pytest.mark.asyncio
async def test_search_documents_strips_metadata_when_disabled(
    registered_tools: dict[str, Callable[..., Any]],
    mock_vector_service: Mock,
    mock_context: MagicMock,
) -> None:
    """Metadata stripping should remove metadata payloads."""
    mock_vector_service.search_documents.return_value = [
        SearchRecord(
            id="doc-1",
            score=0.5,
            content="body",
            metadata={"foo": "bar"},
        )
    ]

    request = SearchRequest(
        query="q",
        collection="docs",
        limit=1,
        offset=0,
        search_strategy=SearchStrategy.HYBRID,
        include_metadata=False,
    )

    results = await registered_tools["search_documents"](request, mock_context)

    assert results[0].metadata is None
