"""Tests for filtering tools."""

from unittest.mock import AsyncMock, Mock

import pytest

from src.mcp_tools.tools.filtering_tools import (
    CompositeFilterRequest,
    ContentTypeFilterRequest,
    MetadataFilterRequest,
    SimilarityFilterRequest,
    TemporalFilterRequest,
    composite_filter_tool,
    content_type_filter_tool,
    create_orchestrator,
    metadata_filter_tool,
    similarity_filter_tool,
    temporal_filter_tool,
)
from src.services.query_processing.orchestrator import (
    SearchResult as AdvancedSearchResult,
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


class TestFilteringTools:
    """Test filtering tools."""

    @pytest.fixture
    def mock_context(self):
        """Create mock context."""
        return MockContext()

    @pytest.fixture
    def mock_orchestrator(self):
        """Create mock orchestrator."""
        orchestrator = AsyncMock()
        orchestrator.search.return_value = AdvancedSearchResult(
            results=[
                {
                    "id": "doc1",
                    "title": "Test Document",
                    "content": "Test content",
                    "score": 0.9,
                    "metadata": {"source": "test"},
                }
            ],
            total_results=1,
            query_processed="test query",
            processing_time_ms=100.0,
            features_used=["temporal_filter", "content_type_filter"],
        )
        return orchestrator

    async def test_temporal_filter_tool(self, mock_context, mock_orchestrator):
        """Test temporal filter tool."""
        request = TemporalFilterRequest(
            collection_name="documents",
            query="test query",
            start_date="2024-01-01",
            end_date="2024-12-31",
            time_window="30d",
            freshness_weight=0.2,
            limit=10,
        )

        results = await temporal_filter_tool(request, mock_context, mock_orchestrator)

        assert isinstance(results, list)
        assert len(results) == 1
        assert results[0].id == "doc1"
        mock_orchestrator.search.assert_called_once()

    async def test_content_type_filter_tool(self, mock_context, mock_orchestrator):
        """Test content type filter tool."""
        request = ContentTypeFilterRequest(
            collection_name="documents",
            query="test query",
            allowed_types=["documentation", "code"],
            exclude_types=["deprecated"],
            priority_types=["documentation"],
            limit=10,
        )

        results = await content_type_filter_tool(
            request, mock_context, mock_orchestrator
        )

        assert isinstance(results, list)
        assert len(results) == 1
        mock_orchestrator.search.assert_called_once()

    async def test_metadata_filter_tool(self, mock_context, mock_orchestrator):
        """Test metadata filter tool."""
        request = MetadataFilterRequest(
            collection_name="documents",
            query="test query",
            metadata_filters={"category": "tutorial", "level": "beginner"},
            filter_operator="AND",
            exact_match=True,
            case_sensitive=False,
            limit=10,
        )

        results = await metadata_filter_tool(request, mock_context, mock_orchestrator)

        assert isinstance(results, list)
        assert len(results) == 1
        mock_orchestrator.search.assert_called_once()

    async def test_similarity_filter_tool(self, mock_context, mock_orchestrator):
        """Test similarity filter tool."""
        request = SimilarityFilterRequest(
            collection_name="documents",
            query="test query",
            min_similarity=0.7,
            max_similarity=1.0,
            similarity_metric="cosine",
            adaptive_threshold=True,
            boost_recent=True,
            limit=10,
        )

        results = await similarity_filter_tool(request, mock_context, mock_orchestrator)

        assert isinstance(results, list)
        assert len(results) == 1
        mock_orchestrator.search.assert_called_once()

    async def test_composite_filter_tool(self, mock_context, mock_orchestrator):
        """Test composite filter tool."""
        request = CompositeFilterRequest(
            collection_name="documents",
            query="test query",
            temporal_config={
                "start_date": "2024-01-01",
                "end_date": "2024-12-31",
            },
            content_type_config={
                "allowed_types": ["documentation"],
            },
            metadata_config={
                "category": "tutorial",
            },
            similarity_config={
                "min_similarity": 0.7,
            },
            operator="AND",
            nested_logic=False,
            optimize_order=True,
            limit=10,
        )

        results = await composite_filter_tool(request, mock_context, mock_orchestrator)

        assert isinstance(results, list)
        assert len(results) == 1
        mock_orchestrator.search.assert_called_once()

    async def test_create_orchestrator(self):
        """Test orchestrator creation."""
        orchestrator = create_orchestrator()

        assert orchestrator is not None
        assert orchestrator.enable_performance_optimization is True

    async def test_temporal_filter_error_handling(self, mock_context):
        """Test error handling in temporal filter."""
        # Create an orchestrator that will fail
        orchestrator = AsyncMock()
        orchestrator.search.side_effect = Exception("Search failed")

        request = TemporalFilterRequest(
            collection_name="documents",
            query="test query",
            limit=10,
        )

        with pytest.raises(Exception, match="Search failed"):
            await temporal_filter_tool(request, mock_context, orchestrator)

    async def test_content_type_filter_with_empty_types(
        self, mock_context, mock_orchestrator
    ):
        """Test content type filter with empty allowed types."""
        request = ContentTypeFilterRequest(
            collection_name="documents",
            query="test query",
            allowed_types=[],  # Empty allowed types
            limit=10,
        )

        results = await content_type_filter_tool(
            request, mock_context, mock_orchestrator
        )

        assert isinstance(results, list)
        mock_orchestrator.search.assert_called_once()

    async def test_metadata_filter_or_operator(self, mock_context, mock_orchestrator):
        """Test metadata filter with OR operator."""
        request = MetadataFilterRequest(
            collection_name="documents",
            query="test query",
            metadata_filters={"category": "tutorial", "level": "advanced"},
            filter_operator="OR",
            limit=10,
        )

        results = await metadata_filter_tool(request, mock_context, mock_orchestrator)

        assert isinstance(results, list)
        mock_orchestrator.search.assert_called_once()

    async def test_similarity_filter_with_adaptive_threshold(
        self, mock_context, mock_orchestrator
    ):
        """Test similarity filter with adaptive threshold."""
        request = SimilarityFilterRequest(
            collection_name="documents",
            query="test query",
            min_similarity=0.5,
            adaptive_threshold=True,
            boost_recent=True,
            limit=10,
        )

        results = await similarity_filter_tool(request, mock_context, mock_orchestrator)

        assert isinstance(results, list)
        mock_orchestrator.search.assert_called_once()

    async def test_composite_filter_or_operator(self, mock_context, mock_orchestrator):
        """Test composite filter with OR operator."""
        request = CompositeFilterRequest(
            collection_name="documents",
            query="test query",
            temporal_config={"time_window": "7d"},
            content_type_config={"allowed_types": ["code"]},
            operator="OR",
            limit=10,
        )

        results = await composite_filter_tool(request, mock_context, mock_orchestrator)

        assert isinstance(results, list)
        mock_orchestrator.search.assert_called_once()

    async def test_composite_filter_minimal_config(
        self, mock_context, mock_orchestrator
    ):
        """Test composite filter with minimal configuration."""
        request = CompositeFilterRequest(
            collection_name="documents",
            query="test query",
            operator="AND",
            limit=5,
        )

        results = await composite_filter_tool(request, mock_context, mock_orchestrator)

        assert isinstance(results, list)
        mock_orchestrator.search.assert_called_once()

    async def test_filter_request_validation(self):
        """Test filter request model validation."""
        # Test valid temporal filter request
        request = TemporalFilterRequest(
            collection_name="documents",
            query="test",
            limit=5,
        )
        assert request.collection_name == "documents"
        assert request.limit == 5

        # Test invalid limit
        with pytest.raises(ValueError):
            TemporalFilterRequest(
                collection_name="documents",
                query="test",
                limit=0,  # Should be >= 1
            )

    async def test_search_result_conversion(self, mock_context, mock_orchestrator):
        """Test conversion from AdvancedSearchResult to SearchResult list."""
        # Configure mock to return specific results
        mock_orchestrator.search.return_value = AdvancedSearchResult(
            results=[
                {
                    "id": "doc1",
                    "title": "Document 1",
                    "content": "Content 1",
                    "score": 0.95,
                    "metadata": {"category": "tutorial"},
                },
                {
                    "id": "doc2",
                    "title": "Document 2",
                    "content": "Content 2",
                    "score": 0.85,
                    "metadata": {"category": "guide"},
                },
            ],
            total_results=2,
            query_processed="test query",
            processing_time_ms=120.0,
            features_used=["temporal_filter", "content_type_filter"],
            cache_hit=False,
        )

        request = TemporalFilterRequest(
            collection_name="documents",
            query="test query",
            limit=10,
        )

        results = await temporal_filter_tool(request, mock_context, mock_orchestrator)

        assert len(results) == 2
        assert results[0].id == "doc1"
        assert results[0].score == 0.95
        assert results[1].id == "doc2"
        assert results[1].score == 0.85

    async def test_temporal_filter_error_with_exception(self):
        """Test temporal filter error handling with exception."""
        mock_context = MockContext()
        error_orchestrator = Mock()
        error_orchestrator.search = AsyncMock(side_effect=Exception("Search failed"))

        request = TemporalFilterRequest(
            collection_name="test_collection",
            query="test query",
            start_date="2024-01-01",
            limit=5,
        )

        with pytest.raises(Exception, match="Search failed"):
            await temporal_filter_tool(request, mock_context, error_orchestrator)

    async def test_content_type_filter_error_with_exception(self):
        """Test content type filter error handling with exception."""
        mock_context = MockContext()
        error_orchestrator = Mock()
        error_orchestrator.search = AsyncMock(side_effect=Exception("Search failed"))

        request = ContentTypeFilterRequest(
            collection_name="test_collection",
            query="test query",
            allowed_types=["documentation"],
            limit=5,
        )

        with pytest.raises(Exception, match="Search failed"):
            await content_type_filter_tool(request, mock_context, error_orchestrator)

    async def test_metadata_filter_error_with_exception(self):
        """Test metadata filter error handling with exception."""
        mock_context = MockContext()
        error_orchestrator = Mock()
        error_orchestrator.search = AsyncMock(side_effect=Exception("Search failed"))

        request = MetadataFilterRequest(
            collection_name="test_collection",
            query="test query",
            metadata_filters={"author": "test"},
            limit=5,
        )

        with pytest.raises(Exception, match="Search failed"):
            await metadata_filter_tool(request, mock_context, error_orchestrator)

    async def test_similarity_filter_error_with_exception(self):
        """Test similarity filter error handling with exception."""
        mock_context = MockContext()
        error_orchestrator = Mock()
        error_orchestrator.search = AsyncMock(side_effect=Exception("Search failed"))

        request = SimilarityFilterRequest(
            collection_name="test_collection",
            query="test query",
            similarity_threshold=0.8,
            limit=5,
        )

        with pytest.raises(Exception, match="Search failed"):
            await similarity_filter_tool(request, mock_context, error_orchestrator)

    async def test_composite_filter_error_with_exception(self):
        """Test composite filter error handling with exception."""
        mock_context = MockContext()
        error_orchestrator = Mock()
        error_orchestrator.search = AsyncMock(side_effect=Exception("Search failed"))

        request = CompositeFilterRequest(
            collection_name="test_collection",
            query="test query",
            temporal_config={"start_date": "2024-01-01"},
            limit=5,
        )

        with pytest.raises(Exception, match="Search failed"):
            await composite_filter_tool(request, mock_context, error_orchestrator)

    def test_register_filtering_tools(self):
        """Test registration of filtering tools."""
        from src.mcp_tools.tools.filtering_tools import register_filtering_tools  # noqa: PLC0415

        mock_mcp = Mock()
        mock_client_manager = Mock()

        register_filtering_tools(mock_mcp, mock_client_manager)

        # Verify that tool decorator was called multiple times
        assert mock_mcp.tool.call_count >= 5  # Should have 5 tools registered
