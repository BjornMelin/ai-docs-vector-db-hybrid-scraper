"""Tests for query_processing_tools.py to achieve coverage."""

from unittest.mock import AsyncMock, MagicMock

import pytest
from pydantic import ValidationError

from src.mcp_tools.models.responses import SearchResult
from src.mcp_tools.tools.query_processing_tools import (
    ClusteredSearchRequest,
    FederatedSearchRequest,
    OrchestrationRequest,
    PersonalizedSearchRequest,
    QueryExpansionRequest,
    clustered_search_tool,
    create_orchestrator,
    federated_search_tool,
    orchestrated_search_tool,
    personalized_search_tool,
    query_expansion_tool,
    register_query_processing_tools,
)


class MockContext:
    """Mock context for testing."""

    async def info(self, msg: str) -> None:
        pass

    async def debug(self, msg: str) -> None:
        pass

    async def warning(self, msg: str) -> None:
        pass

    async def error(self, msg: str) -> None:
        pass


class TestQueryProcessingTools:
    """Test query processing tools."""

    @pytest.fixture
    def mock_context(self):
        """Create mock context."""
        return MockContext()

    @pytest.fixture
    def mock_orchestrator(self):
        """Create mock orchestrator."""
        mock = MagicMock()
        mock.search = AsyncMock()
        mock.initialize = AsyncMock()

        # Mock search result
        mock_result = MagicMock()
        mock_result.results = [
            MagicMock(
                id="result_1",
                title="Test Result",
                content="Test content",
                score=0.9,
                content_type="documentation",
                published_date="2024-01-01T00:00:00Z",
                metadata={"source": "test"},
            )
        ]
        mock_result._total_results = 1
        mock_result.processing_time_ms = 100.0
        mock_result.metadata = {"confidence": 0.9}
        mock_result.quality_score = 0.85
        mock_result.pipeline = MagicMock(value="balanced")
        mock.search.return_value = mock_result

        return mock

    def test_create_orchestrator(self):
        """Test orchestrator creation."""
        orchestrator = create_orchestrator()
        assert orchestrator is not None

    async def test_query_expansion_tool(self, mock_context, mock_orchestrator):
        """Test query expansion tool."""
        request = QueryExpansionRequest(
            collection_name="test_collection",
            query="machine learning",
            expansion_methods=["synonyms"],
            expansion_depth=2,
            limit=5,
        )

        results = await query_expansion_tool(request, mock_context, mock_orchestrator)

        assert isinstance(results, list)
        assert len(results) > 0
        assert all(isinstance(r, SearchResult) for r in results)
        mock_orchestrator.search.assert_called_once()

    async def test_query_expansion_error_handling(self, mock_context):
        """Test query expansion error handling."""
        # Create a fresh mock that fails
        error_orchestrator = MagicMock()
        error_orchestrator.search = AsyncMock(side_effect=Exception("Search failed"))

        request = QueryExpansionRequest(
            collection_name="test_collection",
            query="test query",
            expansion_methods=["invalid_method"],
            limit=5,
        )

        with pytest.raises(Exception, match="Search failed"):
            await query_expansion_tool(request, mock_context, error_orchestrator)

    async def test_clustered_search_tool(self, mock_context, mock_orchestrator):
        """Test clustered search tool."""
        request = ClusteredSearchRequest(
            collection_name="test_collection",
            query="data science",
            clustering_method="kmeans",
            num_clusters=3,
            limit=10,
        )

        results = await clustered_search_tool(request, mock_context, mock_orchestrator)

        assert isinstance(results, list)
        assert len(results) > 0
        mock_orchestrator.search.assert_called_once()

    async def test_clustered_search_error_handling(self, mock_context):
        """Test clustered search error handling."""
        # Create a fresh mock that fails
        error_orchestrator = MagicMock()
        error_orchestrator.search = AsyncMock(
            side_effect=Exception("Clustering failed")
        )

        request = ClusteredSearchRequest(
            collection_name="test_collection",
            query="test query",
            clustering_method="invalid",
            limit=5,
        )

        with pytest.raises(Exception, match="Clustering failed"):
            await clustered_search_tool(request, mock_context, error_orchestrator)

    async def test_federated_search_tool(self, mock_context, mock_orchestrator):
        """Test federated search tool."""
        request = FederatedSearchRequest(
            collections=["collection1", "collection2"],
            query="artificial intelligence",
            merge_strategy="score_weighted",
            limit=15,
        )

        results = await federated_search_tool(request, mock_context, mock_orchestrator)

        assert isinstance(results, list)
        assert len(results) > 0
        mock_orchestrator.search.assert_called()

    async def test_federated_search_error_handling(self, mock_context):
        """Test federated search error handling."""
        # Create a fresh mock that fails
        error_orchestrator = MagicMock()
        error_orchestrator.search = AsyncMock(
            side_effect=Exception("Federated search failed")
        )

        request = FederatedSearchRequest(
            collections=["collection1"],
            query="test query",
            merge_strategy="invalid",
            limit=5,
        )

        with pytest.raises(Exception, match="Federated search failed"):
            await federated_search_tool(request, mock_context, error_orchestrator)

    async def test_personalized_search_tool(self, mock_context, mock_orchestrator):
        """Test personalized search tool."""
        request = PersonalizedSearchRequest(
            collection_name="test_collection",
            query="machine learning tutorial",
            user_id="user123",
            user_context={"interests": ["AI", "programming"]},
            personalization_weight=0.3,
            limit=10,
        )

        results = await personalized_search_tool(
            request, mock_context, mock_orchestrator
        )

        assert isinstance(results, list)
        assert len(results) > 0
        mock_orchestrator.search.assert_called_once()

    async def test_personalized_search_tool_minimal_user(
        self, mock_context, mock_orchestrator
    ):
        """Test personalized search tool with minimal user data."""
        request = PersonalizedSearchRequest(
            collection_name="test_collection",
            query="machine learning tutorial",
            user_id="minimal_user",
            limit=10,
        )

        results = await personalized_search_tool(
            request, mock_context, mock_orchestrator
        )

        assert isinstance(results, list)
        assert len(results) > 0
        mock_orchestrator.search.assert_called_once()

    async def test_personalized_search_error_handling(self, mock_context):
        """Test personalized search error handling."""
        # Create a fresh mock that fails
        error_orchestrator = MagicMock()
        error_orchestrator.search = AsyncMock(
            side_effect=Exception("Personalization failed")
        )

        request = PersonalizedSearchRequest(
            collection_name="test_collection",
            query="test query",
            user_id="user123",
            limit=5,
        )

        with pytest.raises(Exception, match="Personalization failed"):
            await personalized_search_tool(request, mock_context, error_orchestrator)

    async def test_orchestrated_search_tool(self, mock_context, mock_orchestrator):
        """Test orchestrated search tool."""
        request = OrchestrationRequest(
            collection_name="test_collection",
            query="deep learning networks",
            pipeline="balanced",
            enable_preprocessing=True,
            enable_intent_classification=True,
            enable_expansion=True,
            enable_clustering=True,
            enable_personalization=False,
            limit=20,
        )

        results = await orchestrated_search_tool(
            request, mock_context, mock_orchestrator
        )

        assert isinstance(results, list)
        assert len(results) > 0
        mock_orchestrator.search.assert_called_once()

    async def test_orchestrated_search_tool_minimal_config(
        self, mock_context, mock_orchestrator
    ):
        """Test orchestrated search tool with minimal configuration."""
        request = OrchestrationRequest(
            collection_name="test_collection", query="simple query", limit=5
        )

        results = await orchestrated_search_tool(
            request, mock_context, mock_orchestrator
        )

        assert isinstance(results, list)
        assert len(results) > 0
        mock_orchestrator.search.assert_called_once()

    async def test_orchestrated_search_error_handling(self, mock_context):
        """Test orchestrated search error handling."""
        # Create a fresh mock that fails
        error_orchestrator = MagicMock()
        error_orchestrator.search = AsyncMock(
            side_effect=Exception("Orchestration failed")
        )

        request = OrchestrationRequest(
            collection_name="test_collection", query="test query", limit=5
        )

        with pytest.raises(Exception, match="Orchestration failed"):
            await orchestrated_search_tool(request, mock_context, error_orchestrator)

    def test_request_validation(self):
        """Test request model validation."""

        # Test QueryExpansionRequest validation
        with pytest.raises(ValidationError):
            QueryExpansionRequest(
                collection_name="test",
                query="test",
                expansion_depth=10,  # Too high (max 5)
            )

        # Test ClusteredSearchRequest validation
        with pytest.raises(ValidationError):
            ClusteredSearchRequest(
                collection_name="test",
                query="test",
                num_clusters=1,  # Too low (min 2)
            )

        # Test FederatedSearchRequest validation - collections is required
        with pytest.raises(ValidationError):
            FederatedSearchRequest(
                query="test"  # Missing collections field
            )

    def test_search_result_conversion(self):
        """Test search result conversion logic."""
        mock_result = MagicMock()
        mock_result.id = "test_id"
        mock_result.title = "Test Title"
        mock_result.content = "Test content"
        mock_result.score = 0.85
        mock_result.content_type = "documentation"
        mock_result.published_date = "2024-01-01T00:00:00Z"
        mock_result.metadata = {"source": "test"}

        # Test conversion logic
        search_result = SearchResult(
            id=mock_result.id,
            title=mock_result.title,
            content=mock_result.content,
            score=mock_result.score,
            content_type=mock_result.content_type,
            published_date=mock_result.published_date,
            metadata=mock_result.metadata,
        )

        assert search_result.id == "test_id"
        assert search_result.title == "Test Title"
        assert search_result.score == 0.85

    def test_register_query_processing_tools(self):
        """Test tool registration function."""
        mock_mcp = MagicMock()
        mock_client_manager = MagicMock()

        register_query_processing_tools(mock_mcp, mock_client_manager)

        # Verify tools were registered
        assert mock_mcp.tool.call_count > 0
