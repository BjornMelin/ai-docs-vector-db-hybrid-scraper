"""Tests for MCP payload indexing tools."""

from unittest.mock import AsyncMock
from unittest.mock import MagicMock
from unittest.mock import Mock
from unittest.mock import patch

import pytest
from src.mcp_tools.models.responses import GenericDictResponse
from src.mcp_tools.models.responses import ReindexCollectionResponse


@pytest.fixture
def mock_context():
    """Create a mock context for testing."""
    context = Mock()
    context.info = AsyncMock()
    context.debug = AsyncMock()
    context.warning = AsyncMock()
    context.error = AsyncMock()
    return context


@pytest.fixture
def mock_client_manager():
    """Create a mock client manager."""
    manager = Mock()

    # Mock QdrantService
    mock_qdrant = Mock()
    mock_qdrant.list_collections = AsyncMock()
    mock_qdrant.create_payload_indexes = AsyncMock()
    mock_qdrant.get_payload_index_stats = AsyncMock()
    mock_qdrant.reindex_collection = AsyncMock()
    mock_qdrant.get_collection_info = AsyncMock()
    mock_qdrant.filtered_search = AsyncMock()
    manager.qdrant_service = mock_qdrant

    # Mock embedding manager
    mock_embedding = Mock()
    mock_embedding.generate_embeddings = AsyncMock()
    manager.embedding_manager = mock_embedding

    return manager


@pytest.fixture
def mock_index_stats():
    """Create mock index statistics."""
    return {
        "indexed_fields_count": 5,
        "indexed_fields": ["site_name", "embedding_model", "title", "word_count", "crawl_timestamp"],
        "total_points": 1000
    }


@pytest.mark.asyncio
async def test_payload_indexing_tools_registration(mock_client_manager, mock_context):
    """Test that payload indexing tools are properly registered."""
    from src.mcp_tools.tools.payload_indexing import register_tools

    mock_mcp = MagicMock()
    registered_tools = {}

    def capture_tool(func):
        registered_tools[func.__name__] = func
        return func

    mock_mcp.tool.return_value = capture_tool

    register_tools(mock_mcp, mock_client_manager)

    assert "create_payload_indexes" in registered_tools
    assert "list_payload_indexes" in registered_tools
    assert "reindex_collection" in registered_tools
    assert "benchmark_filtered_search" in registered_tools


@pytest.mark.asyncio
async def test_create_payload_indexes_success(mock_client_manager, mock_context, mock_index_stats):
    """Test successful payload index creation."""
    from src.mcp_tools.tools.payload_indexing import register_tools

    mock_mcp = MagicMock()
    registered_tools = {}

    def capture_tool(func):
        registered_tools[func.__name__] = func
        return func

    mock_mcp.tool.return_value = capture_tool

    # Setup mocks
    mock_client_manager.qdrant_service.list_collections.return_value = ["test_collection", "other_collection"]
    mock_client_manager.qdrant_service.get_payload_index_stats.return_value = mock_index_stats

    # Mock security validator
    with patch('src.mcp_tools.tools.payload_indexing.SecurityValidator') as mock_security:
        mock_validator = Mock()
        mock_validator.validate_collection_name.return_value = "test_collection"
        mock_security.from_unified_config.return_value = mock_validator

        register_tools(mock_mcp, mock_client_manager)

        # Test create_payload_indexes function
        result = await registered_tools["create_payload_indexes"](
            collection_name="test_collection",
            ctx=mock_context
        )

        assert isinstance(result, GenericDictResponse)
        assert result.collection_name == "test_collection"
        assert result.status == "success"
        assert result.indexes_created == 5
        assert result.indexed_fields == ["site_name", "embedding_model", "title", "word_count", "crawl_timestamp"]
        assert result.total_points == 1000
        assert hasattr(result, "request_id")

        # Verify services were called correctly
        mock_client_manager.qdrant_service.list_collections.assert_called_once()
        mock_client_manager.qdrant_service.create_payload_indexes.assert_called_once_with("test_collection")
        mock_client_manager.qdrant_service.get_payload_index_stats.assert_called_once_with("test_collection")

        # Verify context logging
        mock_context.info.assert_called()


@pytest.mark.asyncio
async def test_create_payload_indexes_collection_not_found(mock_client_manager, mock_context):
    """Test payload index creation when collection doesn't exist."""
    from src.mcp_tools.tools.payload_indexing import register_tools

    mock_mcp = MagicMock()
    registered_tools = {}

    def capture_tool(func):
        registered_tools[func.__name__] = func
        return func

    mock_mcp.tool.return_value = capture_tool

    # Setup mocks - collection not found
    mock_client_manager.qdrant_service.list_collections.return_value = ["other_collection"]

    # Mock security validator
    with patch('src.mcp_tools.tools.payload_indexing.SecurityValidator') as mock_security:
        mock_validator = Mock()
        mock_validator.validate_collection_name.return_value = "missing_collection"
        mock_security.from_unified_config.return_value = mock_validator

        register_tools(mock_mcp, mock_client_manager)

        # Test with non-existent collection
        with pytest.raises(ValueError, match="Collection 'missing_collection' not found"):
            await registered_tools["create_payload_indexes"](
                collection_name="missing_collection",
                ctx=mock_context
            )

        # Verify error logging
        mock_context.error.assert_called()


@pytest.mark.asyncio
async def test_list_payload_indexes_success(mock_client_manager, mock_context, mock_index_stats):
    """Test successful payload index listing."""
    from src.mcp_tools.tools.payload_indexing import register_tools

    mock_mcp = MagicMock()
    registered_tools = {}

    def capture_tool(func):
        registered_tools[func.__name__] = func
        return func

    mock_mcp.tool.return_value = capture_tool

    # Setup mocks
    mock_client_manager.qdrant_service.get_payload_index_stats.return_value = mock_index_stats

    # Mock security validator
    with patch('src.mcp_tools.tools.payload_indexing.SecurityValidator') as mock_security:
        mock_validator = Mock()
        mock_validator.validate_collection_name.return_value = "test_collection"
        mock_security.from_unified_config.return_value = mock_validator

        register_tools(mock_mcp, mock_client_manager)

        # Test list_payload_indexes function
        result = await registered_tools["list_payload_indexes"](
            collection_name="test_collection",
            ctx=mock_context
        )

        assert isinstance(result, GenericDictResponse)
        assert result.indexed_fields_count == 5
        assert result.indexed_fields == ["site_name", "embedding_model", "title", "word_count", "crawl_timestamp"]
        assert result.total_points == 1000

        # Verify service was called correctly
        mock_client_manager.qdrant_service.get_payload_index_stats.assert_called_once_with("test_collection")

        # Verify context logging
        mock_context.info.assert_called()


@pytest.mark.asyncio
async def test_reindex_collection_success(mock_client_manager, mock_context, mock_index_stats):
    """Test successful collection reindexing."""
    from src.mcp_tools.tools.payload_indexing import register_tools

    mock_mcp = MagicMock()
    registered_tools = {}

    def capture_tool(func):
        registered_tools[func.__name__] = func
        return func

    mock_mcp.tool.return_value = capture_tool

    # Setup mocks - before and after stats
    stats_before = {
        "indexed_fields_count": 3,
        "indexed_fields": ["title", "url", "content"],
        "total_points": 800
    }

    stats_after = mock_index_stats  # More indexes after reindexing

    mock_client_manager.qdrant_service.get_payload_index_stats.side_effect = [stats_before, stats_after]

    # Mock security validator
    with patch('src.mcp_tools.tools.payload_indexing.SecurityValidator') as mock_security:
        mock_validator = Mock()
        mock_validator.validate_collection_name.return_value = "test_collection"
        mock_security.from_unified_config.return_value = mock_validator

        register_tools(mock_mcp, mock_client_manager)

        # Test reindex_collection function
        result = await registered_tools["reindex_collection"](
            collection_name="test_collection",
            ctx=mock_context
        )

        assert isinstance(result, ReindexCollectionResponse)
        assert result.status == "success"
        assert result.collection == "test_collection"
        assert result.reindexed_count == 5

        # Verify details
        details = result.details
        assert details["indexes_before"] == 3
        assert details["indexes_after"] == 5
        assert details["indexed_fields"] == ["site_name", "embedding_model", "title", "word_count", "crawl_timestamp"]
        assert details["total_points"] == 1000
        assert "request_id" in details

        # Verify services were called correctly
        assert mock_client_manager.qdrant_service.get_payload_index_stats.call_count == 2
        mock_client_manager.qdrant_service.reindex_collection.assert_called_once_with("test_collection")

        # Verify context logging
        mock_context.info.assert_called()


@pytest.mark.asyncio
async def test_benchmark_filtered_search_success(mock_client_manager, mock_context):
    """Test successful filtered search benchmarking."""
    from src.mcp_tools.tools.payload_indexing import register_tools

    mock_mcp = MagicMock()
    registered_tools = {}

    def capture_tool(func):
        registered_tools[func.__name__] = func
        return func

    mock_mcp.tool.return_value = capture_tool

    # Setup mocks
    mock_embedding_result = {
        "embeddings": [[0.1, 0.2, 0.3, 0.4]]
    }
    mock_client_manager.embedding_manager.generate_embeddings.return_value = mock_embedding_result

    mock_search_results = [
        {"id": "doc1", "score": 0.9, "payload": {"content": "Result 1"}},
        {"id": "doc2", "score": 0.8, "payload": {"content": "Result 2"}}
    ]
    mock_client_manager.qdrant_service.filtered_search.return_value = mock_search_results

    mock_collection_info = {"points_count": 5000}
    mock_client_manager.qdrant_service.get_collection_info.return_value = mock_collection_info

    mock_index_stats = {
        "indexed_fields": ["site_name", "title", "category"]
    }
    mock_client_manager.qdrant_service.get_payload_index_stats.return_value = mock_index_stats

    # Mock security validator
    with patch('src.mcp_tools.tools.payload_indexing.SecurityValidator') as mock_security:
        mock_validator = Mock()
        mock_validator.validate_collection_name.return_value = "test_collection"
        mock_validator.validate_query_string.return_value = "test query"
        mock_security.from_unified_config.return_value = mock_validator

        register_tools(mock_mcp, mock_client_manager)

        # Test benchmark_filtered_search function
        test_filters = {"site_name": "example.com", "category": "documentation"}

        result = await registered_tools["benchmark_filtered_search"](
            collection_name="test_collection",
            test_filters=test_filters,
            query="test query",
            ctx=mock_context
        )

        assert isinstance(result, GenericDictResponse)
        assert result.collection_name == "test_collection"
        assert result.query == "test query"
        assert result.filters_applied == test_filters
        assert result.results_found == 2
        assert result.total_points == 5000
        assert result.indexed_fields == ["site_name", "title", "category"]
        assert result.performance_estimate == "10-100x faster than unindexed"
        assert hasattr(result, "search_time_ms")
        assert hasattr(result, "benchmark_timestamp")

        # Verify services were called correctly
        mock_client_manager.embedding_manager.generate_embeddings.assert_called_once_with(
            ["test query"], generate_sparse=False
        )
        mock_client_manager.qdrant_service.filtered_search.assert_called_once_with(
            collection_name="test_collection",
            query_vector=[0.1, 0.2, 0.3, 0.4],
            filters=test_filters,
            limit=10,
            search_accuracy="balanced"
        )

        # Verify context logging
        mock_context.info.assert_called()


@pytest.mark.asyncio
async def test_benchmark_filtered_search_no_indexes(mock_client_manager, mock_context):
    """Test filtered search benchmarking with no indexes."""
    from src.mcp_tools.tools.payload_indexing import register_tools

    mock_mcp = MagicMock()
    registered_tools = {}

    def capture_tool(func):
        registered_tools[func.__name__] = func
        return func

    mock_mcp.tool.return_value = capture_tool

    # Setup mocks - no indexes
    mock_embedding_result = {
        "embeddings": [[0.1, 0.2, 0.3, 0.4]]
    }
    mock_client_manager.embedding_manager.generate_embeddings.return_value = mock_embedding_result

    mock_search_results = []
    mock_client_manager.qdrant_service.filtered_search.return_value = mock_search_results

    mock_collection_info = {"points_count": 1000}
    mock_client_manager.qdrant_service.get_collection_info.return_value = mock_collection_info

    mock_index_stats = {
        "indexed_fields": []  # No indexed fields
    }
    mock_client_manager.qdrant_service.get_payload_index_stats.return_value = mock_index_stats

    # Mock security validator
    with patch('src.mcp_tools.tools.payload_indexing.SecurityValidator') as mock_security:
        mock_validator = Mock()
        mock_validator.validate_collection_name.return_value = "test_collection"
        mock_validator.validate_query_string.return_value = "test query"
        mock_security.from_unified_config.return_value = mock_validator

        register_tools(mock_mcp, mock_client_manager)

        # Test benchmark_filtered_search function
        result = await registered_tools["benchmark_filtered_search"](
            collection_name="test_collection",
            test_filters={"category": "docs"},
            query="test query",
            ctx=mock_context
        )

        assert isinstance(result, GenericDictResponse)
        assert result.results_found == 0
        assert result.indexed_fields == []
        assert result.performance_estimate == "No indexes detected"


@pytest.mark.asyncio
async def test_benchmark_filtered_search_default_query(mock_client_manager, mock_context):
    """Test filtered search benchmarking with default query."""
    from src.mcp_tools.tools.payload_indexing import register_tools

    mock_mcp = MagicMock()
    registered_tools = {}

    def capture_tool(func):
        registered_tools[func.__name__] = func
        return func

    mock_mcp.tool.return_value = capture_tool

    # Setup mocks
    mock_embedding_result = {
        "embeddings": [[0.1, 0.2, 0.3, 0.4]]
    }
    mock_client_manager.embedding_manager.generate_embeddings.return_value = mock_embedding_result
    mock_client_manager.qdrant_service.filtered_search.return_value = []
    mock_client_manager.qdrant_service.get_collection_info.return_value = {"points_count": 100}
    mock_client_manager.qdrant_service.get_payload_index_stats.return_value = {"indexed_fields": ["title"]}

    # Mock security validator
    with patch('src.mcp_tools.tools.payload_indexing.SecurityValidator') as mock_security:
        mock_validator = Mock()
        mock_validator.validate_collection_name.return_value = "test_collection"
        mock_validator.validate_query_string.return_value = "documentation search test"
        mock_security.from_unified_config.return_value = mock_validator

        register_tools(mock_mcp, mock_client_manager)

        # Test without providing query (should use default)
        result = await registered_tools["benchmark_filtered_search"](
            collection_name="test_collection",
            test_filters={"type": "guide"},
            ctx=mock_context
        )

        assert isinstance(result, GenericDictResponse)
        assert result.query == "documentation search test"  # Default query

        # Verify embedding generation was called with default query
        mock_client_manager.embedding_manager.generate_embeddings.assert_called_once_with(
            ["documentation search test"], generate_sparse=False
        )


@pytest.mark.asyncio
async def test_benchmark_filtered_search_without_context(mock_client_manager):
    """Test filtered search benchmarking without context parameter."""
    from src.mcp_tools.tools.payload_indexing import register_tools

    mock_mcp = MagicMock()
    registered_tools = {}

    def capture_tool(func):
        registered_tools[func.__name__] = func
        return func

    mock_mcp.tool.return_value = capture_tool

    # Setup mocks
    mock_embedding_result = {
        "embeddings": [[0.1, 0.2, 0.3, 0.4]]
    }
    mock_client_manager.embedding_manager.generate_embeddings.return_value = mock_embedding_result
    mock_client_manager.qdrant_service.filtered_search.return_value = []
    mock_client_manager.qdrant_service.get_collection_info.return_value = {"points_count": 100}
    mock_client_manager.qdrant_service.get_payload_index_stats.return_value = {"indexed_fields": ["title"]}

    # Mock security validator
    with patch('src.mcp_tools.tools.payload_indexing.SecurityValidator') as mock_security:
        mock_validator = Mock()
        mock_validator.validate_collection_name.return_value = "test_collection"
        mock_validator.validate_query_string.return_value = "test query"
        mock_security.from_unified_config.return_value = mock_validator

        register_tools(mock_mcp, mock_client_manager)

        # Test without ctx parameter (None)
        result = await registered_tools["benchmark_filtered_search"](
            collection_name="test_collection",
            test_filters={"category": "docs"},
            query="test query",
            ctx=None
        )

        assert isinstance(result, GenericDictResponse)
        assert result.query == "test query"


@pytest.mark.asyncio
async def test_create_payload_indexes_error_handling(mock_client_manager, mock_context):
    """Test error handling in create_payload_indexes."""
    from src.mcp_tools.tools.payload_indexing import register_tools

    mock_mcp = MagicMock()
    registered_tools = {}

    def capture_tool(func):
        registered_tools[func.__name__] = func
        return func

    mock_mcp.tool.return_value = capture_tool

    # Mock security validator to raise exception
    with patch('src.mcp_tools.tools.payload_indexing.SecurityValidator') as mock_security:
        mock_validator = Mock()
        mock_validator.validate_collection_name.side_effect = Exception("Validation error")
        mock_security.from_unified_config.return_value = mock_validator

        register_tools(mock_mcp, mock_client_manager)

        # Test that exception is properly handled and re-raised
        with pytest.raises(Exception, match="Validation error"):
            await registered_tools["create_payload_indexes"](
                collection_name="test_collection",
                ctx=mock_context
            )

        # Verify error logging
        mock_context.error.assert_called()


@pytest.mark.asyncio
async def test_list_payload_indexes_error_handling(mock_client_manager, mock_context):
    """Test error handling in list_payload_indexes."""
    from src.mcp_tools.tools.payload_indexing import register_tools

    mock_mcp = MagicMock()
    registered_tools = {}

    def capture_tool(func):
        registered_tools[func.__name__] = func
        return func

    mock_mcp.tool.return_value = capture_tool

    # Mock service to raise exception
    mock_client_manager.qdrant_service.get_payload_index_stats.side_effect = Exception("Service error")

    # Mock security validator
    with patch('src.mcp_tools.tools.payload_indexing.SecurityValidator') as mock_security:
        mock_validator = Mock()
        mock_validator.validate_collection_name.return_value = "test_collection"
        mock_security.from_unified_config.return_value = mock_validator

        register_tools(mock_mcp, mock_client_manager)

        # Test that exception is properly handled and re-raised
        with pytest.raises(Exception, match="Service error"):
            await registered_tools["list_payload_indexes"](
                collection_name="test_collection",
                ctx=mock_context
            )

        # Verify error logging
        mock_context.error.assert_called()


@pytest.mark.asyncio
async def test_reindex_collection_error_handling(mock_client_manager, mock_context):
    """Test error handling in reindex_collection."""
    from src.mcp_tools.tools.payload_indexing import register_tools

    mock_mcp = MagicMock()
    registered_tools = {}

    def capture_tool(func):
        registered_tools[func.__name__] = func
        return func

    mock_mcp.tool.return_value = capture_tool

    # Mock service to raise exception during reindexing
    mock_client_manager.qdrant_service.reindex_collection.side_effect = Exception("Reindex failed")

    # Mock security validator
    with patch('src.mcp_tools.tools.payload_indexing.SecurityValidator') as mock_security:
        mock_validator = Mock()
        mock_validator.validate_collection_name.return_value = "test_collection"
        mock_security.from_unified_config.return_value = mock_validator

        register_tools(mock_mcp, mock_client_manager)

        # Test that exception is properly handled and re-raised
        with pytest.raises(Exception, match="Reindex failed"):
            await registered_tools["reindex_collection"](
                collection_name="test_collection",
                ctx=mock_context
            )

        # Verify error logging
        mock_context.error.assert_called()


@pytest.mark.asyncio
async def test_benchmark_filtered_search_error_handling(mock_client_manager, mock_context):
    """Test error handling in benchmark_filtered_search."""
    from src.mcp_tools.tools.payload_indexing import register_tools

    mock_mcp = MagicMock()
    registered_tools = {}

    def capture_tool(func):
        registered_tools[func.__name__] = func
        return func

    mock_mcp.tool.return_value = capture_tool

    # Mock embedding manager to raise exception
    mock_client_manager.embedding_manager.generate_embeddings.side_effect = Exception("Embedding failed")

    # Mock security validator
    with patch('src.mcp_tools.tools.payload_indexing.SecurityValidator') as mock_security:
        mock_validator = Mock()
        mock_validator.validate_collection_name.return_value = "test_collection"
        mock_validator.validate_query_string.return_value = "test query"
        mock_security.from_unified_config.return_value = mock_validator

        register_tools(mock_mcp, mock_client_manager)

        # Test that exception is properly handled and re-raised
        with pytest.raises(Exception, match="Embedding failed"):
            await registered_tools["benchmark_filtered_search"](
                collection_name="test_collection",
                test_filters={"category": "docs"},
                query="test query",
                ctx=mock_context
            )

        # Verify error logging
        mock_context.error.assert_called()
