"""Tests for MCP advanced search tools."""

from unittest.mock import AsyncMock, MagicMock, Mock, patch

import pytest

from src.config.enums import FusionAlgorithm, SearchAccuracy, SearchStrategy
from src.mcp_tools.models.requests import (
    FilteredSearchRequest,
    HyDESearchRequest,
    MultiStageSearchRequest,
    SearchRequest,
)
from src.mcp_tools.models.responses import HyDEAdvancedResponse, SearchResult
from src.mcp_tools.tools.search_tools import _perform_ab_test_search, register_tools


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
    """Create a mock client manager with all required services."""
    manager = Mock()

    # Mock QdrantService
    mock_qdrant = Mock()
    mock_qdrant.multi_stage_search = AsyncMock()
    mock_qdrant.hybrid_search = AsyncMock()
    mock_qdrant.filtered_search = AsyncMock()
    manager.get_qdrant_service = AsyncMock(return_value=mock_qdrant)

    # Mock HyDE engine
    mock_hyde = Mock()
    mock_hyde.enhanced_search = AsyncMock()
    manager.get_hyde_engine = AsyncMock(return_value=mock_hyde)

    # Mock embedding manager
    mock_embedding = Mock()
    mock_embedding.generate_embeddings = AsyncMock()
    mock_embedding.rerank_results = AsyncMock()
    manager.get_embedding_manager = AsyncMock(return_value=mock_embedding)

    return manager


@pytest.fixture
def mock_security_validator():
    """Create a mock security validator."""
    with patch("src.mcp_tools.tools.search_tools.SecurityValidator") as mock_security:
        mock_validator = Mock()
        mock_validator.validate_collection_name.side_effect = lambda x: x
        mock_validator.validate_query_string.side_effect = lambda x: x
        mock_security.from_unified_config.return_value = mock_validator
        yield mock_validator


@pytest.mark.asyncio
async def test_advanced_search_tools_registration(mock_client_manager, _mock_context):
    """Test that advanced search tools are properly registered."""
    mock_mcp = MagicMock()
    registered_tools = {}

    def capture_tool(func):
        registered_tools[func.__name__] = func
        return func

    mock_mcp.tool.return_value = capture_tool

    register_tools(mock_mcp, mock_client_manager)

    expected_tools = [
        "multi_stage_search",
        "hyde_search",
        "hyde_search_advanced",
        "filtered_search",
    ]

    for tool in expected_tools:
        assert tool in registered_tools


@pytest.mark.asyncio
async def test_multi_stage_search_success(mock_client_manager, mock_context):
    """Test successful multi-stage search."""
    mock_mcp = MagicMock()
    registered_tools = {}

    def capture_tool(func):
        registered_tools[func.__name__] = func
        return func

    mock_mcp.tool.return_value = capture_tool

    # Setup mock results
    mock_search_results = [
        {
            "id": "doc1",
            "score": 0.95,
            "payload": {
                "content": "Test content 1",
                "url": "https://example.com/1",
                "title": "Test Document 1",
                "type": "documentation",
            },
        },
        {
            "id": "doc2",
            "score": 0.85,
            "payload": {
                "content": "Test content 2",
                "url": "https://example.com/2",
                "title": "Test Document 2",
                "type": "guide",
            },
        },
    ]

    qdrant_service = await mock_client_manager.get_qdrant_service()
    qdrant_service.multi_stage_search.return_value = mock_search_results

    register_tools(mock_mcp, mock_client_manager)

    # Test multi_stage_search function
    stage1 = {
        "query_vector": [0.1, 0.2, 0.3, 0.4],
        "vector_name": "dense",
        "vector_type": "dense",
        "limit": 50,
        "filters": {"type": "documentation"},
    }

    stage2 = {
        "query_vector": [0.5, 0.6, 0.7, 0.8],
        "vector_name": "sparse",
        "vector_type": "sparse",
        "limit": 20,
        "filters": {},
    }

    request = MultiStageSearchRequest(
        query="test query",
        collection="test_collection",
        stages=[stage1, stage2],
        limit=10,
        fusion_algorithm=FusionAlgorithm.RRF,
        search_accuracy=SearchAccuracy.BALANCED,
    )

    result = await registered_tools["multi_stage_search"](request, mock_context)

    # Verify results
    assert len(result) == 2
    assert isinstance(result[0], SearchResult)
    assert result[0].id == "doc1"
    assert result[0].content == "Test content 1"
    assert result[0].score == 0.95
    assert result[1].id == "doc2"

    # Verify service calls
    qdrant_service.multi_stage_search.assert_called_once()
    call_args = qdrant_service.multi_stage_search.call_args[1]
    assert call_args["collection_name"] == "test_collection"
    assert call_args["limit"] == 10
    assert call_args["fusion_algorithm"] == "rrf"
    assert call_args["search_accuracy"] == "balanced"
    assert len(call_args["stages"]) == 2

    # Verify stage conversion
    stages = call_args["stages"]
    assert stages[0]["query_vector"] == [0.1, 0.2, 0.3, 0.4]
    assert stages[0]["vector_name"] == "dense"
    assert stages[0]["vector_type"] == "dense"
    assert stages[0]["limit"] == 50
    assert stages[0]["filter"] == {"type": "documentation"}

    # Verify context logging
    mock_context.info.assert_called()


@pytest.mark.asyncio
async def test_multi_stage_search_error_handling(mock_client_manager, mock_context):
    """Test error handling in multi-stage search."""
    mock_mcp = MagicMock()
    registered_tools = {}

    def capture_tool(func):
        registered_tools[func.__name__] = func
        return func

    mock_mcp.tool.return_value = capture_tool

    # Setup service to raise exception
    qdrant_service = await mock_client_manager.get_qdrant_service()
    qdrant_service.multi_stage_search.side_effect = Exception("Search service error")

    register_tools(mock_mcp, mock_client_manager)

    request = MultiStageSearchRequest(
        query="test query",
        collection="test_collection",
        stages=[
            {
                "query_vector": [0.1, 0.2, 0.3, 0.4],
                "vector_name": "dense",
                "vector_type": "dense",
                "limit": 10,
                "filters": None,
            }
        ],
        limit=5,
        fusion_algorithm=FusionAlgorithm.RRF,
        search_accuracy=SearchAccuracy.BALANCED,
    )

    with pytest.raises(Exception, match="Search service error"):
        await registered_tools["multi_stage_search"](request, mock_context)

    # Verify error logging
    mock_context.error.assert_called()


@pytest.mark.asyncio
async def test_hyde_search_success(mock_client_manager, mock_context):
    """Test successful HyDE search."""
    mock_mcp = MagicMock()
    registered_tools = {}

    def capture_tool(func):
        registered_tools[func.__name__] = func
        return func

    mock_mcp.tool.return_value = capture_tool

    # Setup mock HyDE results
    mock_hyde_results = [
        {
            "id": "hyde_doc1",
            "content": "HyDE generated content 1",
            "score": 0.92,
            "url": "https://example.com/hyde1",
            "title": "HyDE Document 1",
            "metadata": {"generated": True},
        },
        {
            "id": "hyde_doc2",
            "content": "HyDE generated content 2",
            "score": 0.88,
            "url": "https://example.com/hyde2",
            "title": "HyDE Document 2",
            "metadata": {"generated": True},
        },
    ]

    hyde_engine = await mock_client_manager.get_hyde_engine()
    hyde_engine.enhanced_search.return_value = mock_hyde_results

    # Setup reranking mock
    embedding_manager = await mock_client_manager.get_embedding_manager()
    embedding_manager.rerank_results.return_value = [
        {
            "original": SearchResult(
                id="hyde_doc1",
                content="HyDE generated content 1",
                score=0.92,
                url="https://example.com/hyde1",
                title="HyDE Document 1",
                metadata={"generated": True},
            )
        }
    ]

    register_tools(mock_mcp, mock_client_manager)

    # Test HyDE search
    request = HyDESearchRequest(
        query="machine learning algorithms",
        collection="ml_docs",
        domain="artificial intelligence",
        num_generations=3,
        limit=5,
        enable_reranking=True,
        include_metadata=True,
        fusion_algorithm=FusionAlgorithm.RRF,
        search_accuracy=SearchAccuracy.ACCURATE,
    )

    result = await registered_tools["hyde_search"](request, mock_context)

    # Verify results
    assert len(result) == 1  # After reranking
    assert isinstance(result[0], SearchResult)
    assert result[0].id == "hyde_doc1"
    assert result[0].content == "HyDE generated content 1"

    # Verify service calls
    hyde_engine.enhanced_search.assert_called_once()
    call_args = hyde_engine.enhanced_search.call_args[1]
    assert call_args["query"] == "machine learning algorithms"
    assert call_args["collection_name"] == "ml_docs"
    assert call_args["limit"] == 15  # 5 * 3 for reranking
    assert call_args["domain"] == "artificial intelligence"
    assert call_args["search_accuracy"] == "accurate"
    assert call_args["use_cache"] is True
    assert call_args["force_hyde"] is True

    # Verify reranking was called
    embedding_manager.rerank_results.assert_called_once()

    # Verify context logging
    mock_context.info.assert_called()
    mock_context.debug.assert_called()


@pytest.mark.asyncio
async def test_hyde_search_fallback(mock_client_manager, mock_context):
    """Test HyDE search fallback to regular search."""
    mock_mcp = MagicMock()
    registered_tools = {}

    def capture_tool(func):
        registered_tools[func.__name__] = func
        return func

    mock_mcp.tool.return_value = capture_tool

    # Make HyDE engine unavailable
    mock_client_manager.get_hyde_engine.side_effect = Exception("HyDE not available")

    # Mock fallback search
    with patch(
        "src.mcp_tools.tools._search_utils.search_documents_core",
        new_callable=AsyncMock,
    ) as mock_fallback:
        mock_fallback_results = [
            SearchResult(
                id="fallback_doc1",
                content="Fallback content",
                score=0.8,
                url="https://example.com/fallback",
                title="Fallback Document",
            )
        ]
        mock_fallback.return_value = mock_fallback_results

        register_tools(mock_mcp, mock_client_manager)

        request = HyDESearchRequest(
            query="test query",
            collection="test_collection",
            limit=5,
            enable_reranking=False,
        )

        result = await registered_tools["hyde_search"](request, mock_context)

        # Verify fallback was used
        assert len(result) == 1
        assert result[0].id == "fallback_doc1"
        assert result[0].content == "Fallback content"

        # Verify warning was logged
        mock_context.warning.assert_called()

        # Verify fallback search was called with correct parameters
        mock_fallback.assert_called_once()
        fallback_request = mock_fallback.call_args[0][0]
        assert isinstance(fallback_request, SearchRequest)
        assert fallback_request.query == "test query"
        assert fallback_request.collection == "test_collection"
        assert fallback_request.limit == 5
        assert fallback_request.strategy == SearchStrategy.HYBRID


@pytest.mark.asyncio
async def test_hyde_search_qdrant_point_format(mock_client_manager, mock_context):
    """Test HyDE search with Qdrant point object format."""
    mock_mcp = MagicMock()
    registered_tools = {}

    def capture_tool(func):
        registered_tools[func.__name__] = func
        return func

    mock_mcp.tool.return_value = capture_tool

    # Mock Qdrant point objects
    mock_point = Mock()
    mock_point.id = "point_doc1"
    mock_point.score = 0.95
    mock_point.payload = {
        "content": "Point content",
        "url": "https://example.com/point",
        "title": "Point Document",
    }

    hyde_engine = await mock_client_manager.get_hyde_engine()
    hyde_engine.enhanced_search.return_value = [mock_point]

    register_tools(mock_mcp, mock_client_manager)

    request = HyDESearchRequest(
        query="test query",
        collection="test_collection",
        limit=5,
        enable_reranking=False,
        include_metadata=True,
    )

    result = await registered_tools["hyde_search"](request, mock_context)

    # Verify Qdrant point was converted correctly
    assert len(result) == 1
    assert result[0].id == "point_doc1"
    assert result[0].content == "Point content"
    assert result[0].score == 0.95
    assert result[0].url == "https://example.com/point"
    assert result[0].title == "Point Document"
    assert result[0].metadata == mock_point.payload


@pytest.mark.asyncio
async def test_hyde_search_advanced_success(mock_client_manager, mock_context):
    """Test successful advanced HyDE search."""
    mock_mcp = MagicMock()
    registered_tools = {}

    def capture_tool(func):
        registered_tools[func.__name__] = func
        return func

    mock_mcp.tool.return_value = capture_tool

    # Setup mock HyDE results
    mock_hyde_results = [
        {
            "id": "advanced_doc1",
            "content": "Advanced HyDE content",
            "score": 0.93,
            "url": "https://example.com/advanced",
            "title": "Advanced Document",
            "metadata": {"advanced": True},
        }
    ]

    hyde_engine = await mock_client_manager.get_hyde_engine()
    hyde_engine.enhanced_search.return_value = mock_hyde_results

    # Setup reranking mock
    embedding_manager = await mock_client_manager.get_embedding_manager()
    embedding_manager.rerank_results.return_value = [{"original": mock_hyde_results[0]}]

    register_tools(mock_mcp, mock_client_manager)

    # Test advanced HyDE search
    result = await registered_tools["hyde_search_advanced"](
        query="advanced machine learning",
        collection="advanced_docs",
        domain="AI research",
        num_generations=7,
        generation_temperature=0.8,
        limit=3,
        enable_reranking=True,
        enable_ab_testing=False,
        use_cache=True,
        ctx=mock_context,
    )

    # Verify response
    assert isinstance(result, HyDEAdvancedResponse)
    assert result.query == "advanced machine learning"
    assert result.collection == "advanced_docs"
    assert len(result.results) == 1
    assert result.results[0]["id"] == "advanced_doc1"
    assert result.results[0]["content"] == "Advanced HyDE content"

    # Verify configuration
    assert result.hyde_config.domain == "AI research"
    assert result.hyde_config.num_generations == 7
    assert result.hyde_config.temperature == 0.8
    assert result.hyde_config.enable_ab_testing is False
    assert result.hyde_config.use_cache is True

    # Verify metrics
    assert hasattr(result.metrics, "search_time_ms")
    assert result.metrics.results_found == 1
    assert result.metrics.reranking_applied is True
    assert result.metrics.cache_used is True
    assert result.metrics.generation_parameters["num_generations"] == 7
    assert result.metrics.generation_parameters["temperature"] == 0.8
    assert result.metrics.generation_parameters["domain"] == "AI research"

    # Verify service calls
    hyde_engine.enhanced_search.assert_called_once()
    call_args = hyde_engine.enhanced_search.call_args[1]
    assert call_args["query"] == "advanced machine learning"
    assert call_args["collection_name"] == "advanced_docs"
    assert call_args["limit"] == 9  # 3 * 3 for reranking
    assert call_args["domain"] == "AI research"
    assert call_args["use_cache"] is True
    assert call_args["force_hyde"] is True


@pytest.mark.asyncio
async def test_hyde_search_advanced_with_ab_testing(mock_client_manager, mock_context):
    """Test advanced HyDE search with A/B testing."""
    mock_mcp = MagicMock()
    registered_tools = {}

    def capture_tool(func):
        registered_tools[func.__name__] = func
        return func

    mock_mcp.tool.return_value = capture_tool

    # Mock A/B test function
    with patch(
        "src.mcp_tools.tools.search_tools._perform_ab_test_search",
        new_callable=AsyncMock,
    ) as mock_ab_test:
        mock_search_results = [
            {"id": "ab_doc1", "content": "AB test content", "score": 0.9}
        ]
        mock_ab_results = {
            "hyde_count": 5,
            "regular_count": 4,
            "hyde_avg_score": 0.85,
            "regular_avg_score": 0.78,
        }
        mock_ab_test.return_value = (mock_search_results, mock_ab_results)

        register_tools(mock_mcp, mock_client_manager)

        result = await registered_tools["hyde_search_advanced"](
            query="ab test query",
            collection="ab_test_collection",
            limit=5,
            enable_ab_testing=True,
            enable_reranking=False,
            ctx=mock_context,
        )

        # Verify A/B test results are included
        assert result.ab_test_results is not None
        assert result.ab_test_results["hyde_count"] == 5
        assert result.ab_test_results["regular_count"] == 4
        assert result.ab_test_results["hyde_avg_score"] == 0.85
        assert result.ab_test_results["regular_avg_score"] == 0.78

        # Verify A/B test function was called
        mock_ab_test.assert_called_once()


@pytest.mark.asyncio
async def test_hyde_search_advanced_engine_unavailable(
    mock_client_manager, mock_context
):
    """Test advanced HyDE search when engine is unavailable."""
    mock_mcp = MagicMock()
    registered_tools = {}

    def capture_tool(func):
        registered_tools[func.__name__] = func
        return func

    mock_mcp.tool.return_value = capture_tool

    # Make HyDE engine unavailable
    mock_client_manager.get_hyde_engine.side_effect = Exception("HyDE not available")

    register_tools(mock_mcp, mock_client_manager)

    with pytest.raises(ValueError, match="HyDE engine not initialized"):
        await registered_tools["hyde_search_advanced"](
            query="test query", collection="test_collection", ctx=mock_context
        )

    # Verify error logging
    mock_context.error.assert_called()


@pytest.mark.asyncio
async def test_filtered_search_success(mock_client_manager, mock_context):
    """Test successful filtered search."""
    mock_mcp = MagicMock()
    registered_tools = {}

    def capture_tool(func):
        registered_tools[func.__name__] = func
        return func

    mock_mcp.tool.return_value = capture_tool

    # Setup embedding generation mock
    embedding_manager = await mock_client_manager.get_embedding_manager()
    mock_embedding_result = Mock()
    mock_embedding_result.embeddings = [[0.1, 0.2, 0.3, 0.4]]
    embedding_manager.generate_embeddings.return_value = mock_embedding_result

    # Setup filtered search results
    mock_filtered_results = [
        {
            "id": "filtered_doc1",
            "score": 0.9,
            "payload": {
                "content": "Filtered content 1",
                "url": "https://example.com/filtered1",
                "title": "Filtered Document 1",
                "category": "technical",
            },
        },
        {
            "id": "filtered_doc2",
            "score": 0.8,
            "payload": {
                "content": "Filtered content 2",
                "url": "https://example.com/filtered2",
                "title": "Filtered Document 2",
                "category": "technical",
            },
        },
    ]

    qdrant_service = await mock_client_manager.get_qdrant_service()
    qdrant_service.filtered_search.return_value = mock_filtered_results

    register_tools(mock_mcp, mock_client_manager)

    # Test filtered search
    request = FilteredSearchRequest(
        query="technical documentation",
        collection="docs_collection",
        filters={"category": "technical"},
        limit=10,
        search_accuracy=SearchAccuracy.ACCURATE,
        include_metadata=True,
    )

    result = await registered_tools["filtered_search"](request, mock_context)

    # Verify results
    assert len(result) == 2
    assert isinstance(result[0], SearchResult)
    assert result[0].id == "filtered_doc1"
    assert result[0].content == "Filtered content 1"
    assert result[0].score == 0.9
    assert result[0].metadata is not None
    assert result[0].metadata["category"] == "technical"

    # Verify service calls
    embedding_manager.generate_embeddings.assert_called_once_with(
        ["technical documentation"], generate_sparse=False
    )

    qdrant_service.filtered_search.assert_called_once()
    call_args = qdrant_service.filtered_search.call_args[1]
    assert call_args["collection_name"] == "docs_collection"
    assert call_args["query_vector"] == [0.1, 0.2, 0.3, 0.4]
    assert call_args["filters"] == {"category": "technical"}
    assert call_args["limit"] == 10
    assert call_args["search_accuracy"] == "accurate"

    # Verify context logging
    mock_context.info.assert_called()
    mock_context.debug.assert_called()


@pytest.mark.asyncio
async def test_filtered_search_without_metadata(mock_client_manager, mock_context):
    """Test filtered search without including metadata."""
    mock_mcp = MagicMock()
    registered_tools = {}

    def capture_tool(func):
        registered_tools[func.__name__] = func
        return func

    mock_mcp.tool.return_value = capture_tool

    # Setup mocks
    embedding_manager = await mock_client_manager.get_embedding_manager()
    mock_embedding_result = Mock()
    mock_embedding_result.embeddings = [[0.1, 0.2, 0.3, 0.4]]
    embedding_manager.generate_embeddings.return_value = mock_embedding_result

    mock_filtered_results = [
        {
            "id": "doc1",
            "score": 0.9,
            "payload": {
                "content": "Content 1",
                "url": "https://example.com/1",
                "title": "Document 1",
                "category": "test",
            },
        }
    ]

    qdrant_service = await mock_client_manager.get_qdrant_service()
    qdrant_service.filtered_search.return_value = mock_filtered_results

    register_tools(mock_mcp, mock_client_manager)

    # Test with include_metadata=False
    request = FilteredSearchRequest(
        query="test query",
        collection="test_collection",
        filters={"category": "test"},
        limit=5,
        search_accuracy=SearchAccuracy.BALANCED,
        include_metadata=False,
    )

    result = await registered_tools["filtered_search"](request, mock_context)

    # Verify metadata is not included
    assert len(result) == 1
    assert result[0].metadata is None


@pytest.mark.asyncio
async def test_filtered_search_error_handling(mock_client_manager, mock_context):
    """Test error handling in filtered search."""
    mock_mcp = MagicMock()
    registered_tools = {}

    def capture_tool(func):
        registered_tools[func.__name__] = func
        return func

    mock_mcp.tool.return_value = capture_tool

    # Make embedding generation fail
    embedding_manager = await mock_client_manager.get_embedding_manager()
    embedding_manager.generate_embeddings.side_effect = Exception(
        "Embedding generation failed"
    )

    register_tools(mock_mcp, mock_client_manager)

    request = FilteredSearchRequest(
        query="test query",
        collection="test_collection",
        filters={"type": "test"},
        limit=5,
        search_accuracy=SearchAccuracy.BALANCED,
    )

    with pytest.raises(Exception, match="Embedding generation failed"):
        await registered_tools["filtered_search"](request, mock_context)

    # Verify error logging
    mock_context.error.assert_called()


@pytest.mark.asyncio
async def test_perform_ab_test_search_success(mock_client_manager, mock_context):
    """Test _perform_ab_test_search function."""

    # Setup HyDE engine mock
    hyde_engine = await mock_client_manager.get_hyde_engine()
    mock_hyde_results = [{"id": "hyde1", "score": 0.9}, {"id": "hyde2", "score": 0.8}]
    hyde_engine.enhanced_search.return_value = mock_hyde_results

    # Setup Qdrant service mock
    qdrant_service = await mock_client_manager.get_qdrant_service()
    mock_regular_results = [Mock(score=0.85), Mock(score=0.75)]
    qdrant_service.hybrid_search.return_value = mock_regular_results

    # Test A/B search
    search_results, ab_test_results = await _perform_ab_test_search(
        query="test query",
        collection="test_collection",
        limit=5,
        domain="test",
        use_cache=True,
        client_manager=mock_client_manager,
        ctx=mock_context,
    )

    # Verify results
    assert len(search_results) == 2
    assert search_results == mock_hyde_results

    # Verify A/B test metrics
    assert ab_test_results["hyde_count"] == 2
    assert ab_test_results["regular_count"] == 2
    assert abs(ab_test_results["hyde_avg_score"] - 0.85) < 0.01  # (0.9 + 0.8) / 2
    assert ab_test_results["regular_avg_score"] == 0.8  # (0.85 + 0.75) / 2


@pytest.mark.asyncio
async def test_perform_ab_test_search_with_exceptions(
    mock_client_manager, mock_context
):
    """Test _perform_ab_test_search with service exceptions."""

    # Make HyDE engine fail
    hyde_engine = await mock_client_manager.get_hyde_engine()
    hyde_engine.enhanced_search.side_effect = Exception("HyDE failed")

    # Setup working regular search
    qdrant_service = await mock_client_manager.get_qdrant_service()
    mock_regular_results = [Mock(score=0.8)]
    qdrant_service.hybrid_search.return_value = mock_regular_results

    # Test with HyDE failure
    search_results, ab_test_results = await _perform_ab_test_search(
        query="test query",
        collection="test_collection",
        limit=5,
        domain=None,
        use_cache=False,
        client_manager=mock_client_manager,
        ctx=mock_context,
    )

    # Should fall back to regular search results
    assert len(search_results) == 1
    assert search_results == mock_regular_results

    # Verify A/B test metrics reflect the failure
    assert ab_test_results["hyde_count"] == 0
    assert ab_test_results["regular_count"] == 1
    assert ab_test_results["hyde_avg_score"] == 0
    assert ab_test_results["regular_avg_score"] == 0.8

    # Verify warning was logged
    mock_context.warning.assert_called()


@pytest.mark.asyncio
async def test_hyde_search_advanced_without_context(mock_client_manager):
    """Test advanced HyDE search without context parameter."""
    mock_mcp = MagicMock()
    registered_tools = {}

    def capture_tool(func):
        registered_tools[func.__name__] = func
        return func

    mock_mcp.tool.return_value = capture_tool

    # Setup mock HyDE results
    hyde_engine = await mock_client_manager.get_hyde_engine()
    hyde_engine.enhanced_search.return_value = [
        {"id": "doc1", "content": "Content", "score": 0.9}
    ]

    register_tools(mock_mcp, mock_client_manager)

    # Test without context (ctx=None)
    result = await registered_tools["hyde_search_advanced"](
        query="test query",
        collection="test_collection",
        limit=5,
        enable_reranking=False,
        ctx=None,
    )

    # Should still work without context
    assert isinstance(result, HyDEAdvancedResponse)
    assert len(result.results) == 1


@pytest.mark.asyncio
async def test_hyde_search_with_search_accuracy_enum(mock_client_manager, mock_context):
    """Test HyDE search with SearchAccuracy enum value."""
    mock_mcp = MagicMock()
    registered_tools = {}

    def capture_tool(func):
        registered_tools[func.__name__] = func
        return func

    mock_mcp.tool.return_value = capture_tool

    # Setup mock HyDE results
    hyde_engine = await mock_client_manager.get_hyde_engine()
    hyde_engine.enhanced_search.return_value = [
        {"id": "doc1", "content": "Content", "score": 0.9}
    ]

    register_tools(mock_mcp, mock_client_manager)

    # Test with SearchAccuracy enum
    request = HyDESearchRequest(
        query="test query",
        collection="test_collection",
        limit=5,
        search_accuracy=SearchAccuracy.ACCURATE,  # Enum value
        enable_reranking=False,
    )

    result = await registered_tools["hyde_search"](request, mock_context)

    # Verify it works with enum value
    assert len(result) == 1

    # Verify the enum value was converted to string
    hyde_engine.enhanced_search.assert_called_once()
    call_args = hyde_engine.enhanced_search.call_args[1]
    assert call_args["search_accuracy"] == "accurate"


@pytest.mark.asyncio
async def test_hyde_search_error_with_failed_fallback(
    mock_client_manager, mock_context
):
    """Test HyDE search error handling when both main and fallback fail."""
    mock_mcp = MagicMock()
    registered_tools = {}

    def capture_tool(func):
        registered_tools[func.__name__] = func
        return func

    mock_mcp.tool.return_value = capture_tool

    # Make HyDE engine fail
    hyde_engine = await mock_client_manager.get_hyde_engine()
    hyde_engine.enhanced_search.side_effect = Exception("HyDE failed")

    # Make fallback search also fail
    with patch(
        "src.mcp_tools.tools._search_utils.search_documents_core",
        new_callable=AsyncMock,
    ) as mock_fallback:
        mock_fallback.side_effect = Exception("Fallback failed")

        register_tools(mock_mcp, mock_client_manager)

        request = HyDESearchRequest(
            query="test query", collection="test_collection", limit=5
        )

        # Should raise the original HyDE error
        with pytest.raises(Exception, match="HyDE failed"):
            await registered_tools["hyde_search"](request, mock_context)

        # Verify both error messages were logged
        assert mock_context.error.call_count >= 2
        mock_context.warning.assert_called()
