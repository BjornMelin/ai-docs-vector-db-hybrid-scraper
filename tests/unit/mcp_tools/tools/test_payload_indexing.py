"""Tests for MCP payload indexing tools with VectorStoreService integration."""

from __future__ import annotations

from unittest.mock import AsyncMock, MagicMock, Mock

import pytest

from src.contracts.retrieval import SearchRecord
from src.mcp_tools.tools.payload_indexing import register_tools


@pytest.fixture(autouse=True)
def mock_security_validator(monkeypatch):
    """Provide a permissive MLSecurityValidator stub for all tests."""

    validator = Mock()
    validator.validate_collection_name.side_effect = lambda value: value
    validator.validate_query_string.side_effect = lambda value: value

    monkeypatch.setattr(
        "src.mcp_tools.tools.payload_indexing.MLSecurityValidator.from_unified_config",
        Mock(return_value=validator),
    )
    return validator


@pytest.fixture
def mock_vector_service():
    """Create a mock VectorStoreService."""

    service = Mock()
    service.is_initialized.return_value = True
    service.initialize = AsyncMock()
    service.list_collections = AsyncMock(return_value=["test_collection"])
    service.ensure_payload_indexes = AsyncMock(
        return_value={
            "indexed_fields_count": 5,
            "indexed_fields": [
                "site_name",
                "embedding_model",
                "title",
                "word_count",
                "crawl_timestamp",
            ],
            "points_count": 1000,
            "payload_schema": {},
        }
    )
    service.get_payload_index_summary = AsyncMock(
        return_value={
            "indexed_fields_count": 5,
            "indexed_fields": [
                "site_name",
                "embedding_model",
                "title",
                "word_count",
                "crawl_timestamp",
            ],
            "points_count": 1000,
            "payload_schema": {},
        }
    )
    service.drop_payload_indexes = AsyncMock()
    service.collection_stats = AsyncMock(return_value={"points_count": 5000})
    service.search_documents = AsyncMock(
        return_value=[
            SearchRecord(
                id="doc1",
                score=0.9,
                content="Result 1",
                metadata={"content": "Result 1"},
            ),
            SearchRecord(
                id="doc2",
                score=0.8,
                content="Result 2",
                metadata={"content": "Result 2"},
            ),
        ]
    )
    return service


@pytest.fixture
def mock_client_manager(mock_vector_service):
    """Provide the mocked vector service directly for tool registration."""

    return mock_vector_service


@pytest.fixture
def mock_context():
    """Asynchronous MCP context stub capturing log calls."""

    ctx = Mock()
    ctx.info = AsyncMock()
    ctx.debug = AsyncMock()
    ctx.warning = AsyncMock()
    ctx.error = AsyncMock()
    return ctx


@pytest.mark.asyncio
async def test_tool_registration(mock_client_manager):
    """Verify the payload indexing tools register correctly."""

    mock_mcp = MagicMock()
    registered = {}

    def capture(func):
        registered[func.__name__] = func
        return func

    mock_mcp.tool.return_value = capture

    register_tools(mock_mcp, mock_client_manager)

    assert {
        "create_payload_indexes",
        "list_payload_indexes",
        "reindex_collection",
        "benchmark_filtered_search",
    }.issubset(registered.keys())


@pytest.mark.asyncio
async def test_create_payload_indexes(
    mock_client_manager, mock_vector_service, mock_context
):
    """Ensure create_payload_indexes provisions indexes and returns metadata."""

    mock_mcp = MagicMock()
    registered = {}
    mock_mcp.tool.return_value = lambda func: registered.setdefault(func.__name__, func)

    register_tools(mock_mcp, mock_client_manager)

    response = await registered["create_payload_indexes"](
        collection_name="test_collection",
        ctx=mock_context,
    )

    mock_vector_service.ensure_payload_indexes.assert_awaited_once()
    assert response.indexes_created == 5
    assert response.indexed_fields == [
        "site_name",
        "embedding_model",
        "title",
        "word_count",
        "crawl_timestamp",
    ]
    assert response.total_points == 1000
    assert "_total_points" not in response.model_dump()


@pytest.mark.asyncio
async def test_create_payload_indexes_missing_collection(
    mock_client_manager, mock_vector_service, mock_context
):
    """Raise ValueError when the requested collection cannot be found."""

    mock_vector_service.list_collections.return_value = []

    mock_mcp = MagicMock()
    registered = {}
    mock_mcp.tool.return_value = lambda func: registered.setdefault(func.__name__, func)

    register_tools(mock_mcp, mock_client_manager)

    with pytest.raises(ValueError, match="Collection 'test_collection' not found"):
        await registered["create_payload_indexes"](
            collection_name="test_collection",
            ctx=mock_context,
        )


@pytest.mark.asyncio
async def test_list_payload_indexes(mock_client_manager, mock_context):
    """Return summary metadata for existing payload indexes."""

    mock_mcp = MagicMock()
    registered = {}
    mock_mcp.tool.return_value = lambda func: registered.setdefault(func.__name__, func)

    register_tools(mock_mcp, mock_client_manager)

    response = await registered["list_payload_indexes"](
        collection_name="test_collection", ctx=mock_context
    )

    assert response.indexes_created == 5
    assert response.indexed_fields_count == 5
    assert response.total_points == 1000
    assert "_total_points" not in response.model_dump()


@pytest.mark.asyncio
async def test_reindex_collection(
    mock_client_manager, mock_vector_service, mock_context
):
    """Drop and recreate indexes while reporting post-state summary."""

    mock_mcp = MagicMock()
    registered = {}
    mock_mcp.tool.return_value = lambda func: registered.setdefault(func.__name__, func)

    register_tools(mock_mcp, mock_client_manager)

    response = await registered["reindex_collection"](
        collection_name="test_collection", ctx=mock_context
    )

    mock_vector_service.drop_payload_indexes.assert_awaited_once()
    mock_vector_service.ensure_payload_indexes.assert_awaited()
    assert response.reindexed_count == 5
    assert response.details["indexes_after"] == 5
    assert "_total_points" not in response.details


@pytest.mark.asyncio
async def test_benchmark_filtered_search(mock_client_manager, mock_context):
    """Benchmark filtered search and return performance metrics."""

    mock_mcp = MagicMock()
    registered = {}
    mock_mcp.tool.return_value = lambda func: registered.setdefault(func.__name__, func)

    register_tools(mock_mcp, mock_client_manager)

    response = await registered["benchmark_filtered_search"](
        collection_name="test_collection",
        test_filters={"site_name": "example.com"},
        query="docs",
        ctx=mock_context,
    )

    assert response.results_found == 2
    assert "_total_points" not in response.model_dump()
    assert response.performance_estimate == "10-100x faster than unindexed"
    assert response.total_points == 5000
    assert response.indexed_fields == [
        "site_name",
        "embedding_model",
        "title",
        "word_count",
        "crawl_timestamp",
    ]
    assert response.results[0]["metadata"] == {"content": "Result 1"}
    assert "payload" not in response.results[0]
