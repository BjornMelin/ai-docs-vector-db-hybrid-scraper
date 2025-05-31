"""Tests for QdrantSearch service."""

from unittest.mock import AsyncMock, MagicMock

import pytest
from qdrant_client import models
from src.config import UnifiedConfig
from src.config.enums import SearchAccuracy, VectorType
from src.services.errors import QdrantServiceError
from src.services.vector_db.search import QdrantSearch


class TestQdrantSearch:
    """Test QdrantSearch service."""

    @pytest.fixture
    def config(self):
        """Create test configuration."""
        return UnifiedConfig()

    @pytest.fixture
    def mock_client(self):
        """Create mock Qdrant client."""
        return AsyncMock()

    @pytest.fixture
    def search_service(self, config, mock_client):
        """Create search service."""
        return QdrantSearch(mock_client, config)

    async def test_hybrid_search_success(self, search_service, mock_client):
        """Test successful hybrid search."""
        # Mock search results
        mock_point = MagicMock()
        mock_point.id = "test-1"
        mock_point.score = 0.95
        mock_point.payload = {"title": "Test Document"}

        mock_results = MagicMock()
        mock_results.points = [mock_point]
        mock_client.query_points.return_value = mock_results

        # Test hybrid search
        results = await search_service.hybrid_search(
            collection_name="test_collection",
            query_vector=[0.1] * 1536,
            sparse_vector={1: 0.5, 2: 0.3},
            limit=10,
        )

        assert len(results) == 1
        assert results[0]["id"] == "test-1"
        assert results[0]["score"] == 0.95
        assert results[0]["payload"]["title"] == "Test Document"

        # Verify query_points was called with fusion
        mock_client.query_points.assert_called_once()
        call_args = mock_client.query_points.call_args
        assert "prefetch" in call_args.kwargs
        assert len(call_args.kwargs["prefetch"]) == 2  # Dense + sparse

    async def test_hybrid_search_dense_only(self, search_service, mock_client):
        """Test hybrid search with dense vector only."""
        mock_point = MagicMock()
        mock_point.id = "test-1"
        mock_point.score = 0.85
        mock_point.payload = {}

        mock_results = MagicMock()
        mock_results.points = [mock_point]
        mock_client.query_points.return_value = mock_results

        results = await search_service.hybrid_search(
            collection_name="test_collection",
            query_vector=[0.1] * 1536,
            limit=5,
        )

        assert len(results) == 1
        assert results[0]["id"] == "test-1"

        # Should use single query without fusion
        call_args = mock_client.query_points.call_args
        assert call_args.kwargs["query"] == [0.1] * 1536
        assert call_args.kwargs["using"] == "dense"

    async def test_multi_stage_search_success(self, search_service, mock_client):
        """Test successful multi-stage search."""
        mock_point = MagicMock()
        mock_point.id = "multi-1"
        mock_point.score = 0.88
        mock_point.payload = {"type": "multi-stage"}

        mock_results = MagicMock()
        mock_results.points = [mock_point]
        mock_client.query_points.return_value = mock_results

        stages = [
            {
                "query_vector": [0.1] * 1536,
                "vector_name": "dense",
                "vector_type": "dense",
                "limit": 20,
            },
            {
                "query_vector": [0.2] * 1536,
                "vector_name": "sparse",
                "vector_type": "sparse",
                "limit": 10,
            },
        ]

        results = await search_service.multi_stage_search(
            collection_name="test_collection",
            stages=stages,
            limit=5,
        )

        assert len(results) == 1
        assert results[0]["id"] == "multi-1"

        # Verify prefetch queries were built correctly
        call_args = mock_client.query_points.call_args
        assert "prefetch" in call_args.kwargs
        prefetch_queries = call_args.kwargs["prefetch"]
        assert len(prefetch_queries) == 1  # All but final stage

    async def test_multi_stage_search_validation(self, search_service):
        """Test multi-stage search input validation."""
        # Empty stages
        with pytest.raises(ValueError, match="Stages list cannot be empty"):
            await search_service.multi_stage_search("test", [], 5)

        # Invalid stage format
        with pytest.raises(ValueError, match="Stage 0 must be a dictionary"):
            await search_service.multi_stage_search("test", ["invalid"], 5)

        # Missing required fields
        with pytest.raises(ValueError, match="Stage 0 must contain 'query_vector'"):
            await search_service.multi_stage_search("test", [{}], 5)

    async def test_hyde_search_success(self, search_service, mock_client):
        """Test successful HyDE search."""
        mock_point = MagicMock()
        mock_point.id = "hyde-1"
        mock_point.score = 0.92
        mock_point.payload = {"method": "hyde"}

        mock_results = MagicMock()
        mock_results.points = [mock_point]
        mock_client.query_points.return_value = mock_results

        query_embedding = [0.1] * 1536
        hypothetical_embeddings = [[0.2] * 1536, [0.3] * 1536, [0.4] * 1536]

        results = await search_service.hyde_search(
            collection_name="test_collection",
            query="test query",
            query_embedding=query_embedding,
            hypothetical_embeddings=hypothetical_embeddings,
            limit=10,
        )

        assert len(results) == 1
        assert results[0]["id"] == "hyde-1"

        # Verify HyDE prefetch was used
        call_args = mock_client.query_points.call_args
        assert "prefetch" in call_args.kwargs
        prefetch_queries = call_args.kwargs["prefetch"]
        assert len(prefetch_queries) == 2  # HyDE + original query

    async def test_filtered_search_success(self, search_service, mock_client):
        """Test successful filtered search."""
        mock_point = MagicMock()
        mock_point.id = "filtered-1"
        mock_point.score = 0.89
        mock_point.payload = {"doc_type": "api"}

        mock_results = MagicMock()
        mock_results.points = [mock_point]
        mock_client.query_points.return_value = mock_results

        filters = {
            "doc_type": "api",
            "language": "python",
            "min_word_count": 100,
        }

        results = await search_service.filtered_search(
            collection_name="test_collection",
            query_vector=[0.1] * 1536,
            filters=filters,
            limit=10,
        )

        assert len(results) == 1
        assert results[0]["id"] == "filtered-1"

        # Verify filter was applied
        call_args = mock_client.query_points.call_args
        assert "filter" in call_args.kwargs
        assert call_args.kwargs["filter"] is not None

    async def test_filtered_search_validation(self, search_service):
        """Test filtered search input validation."""
        # Invalid query vector
        with pytest.raises(ValueError, match="query_vector must be a list"):
            await search_service.filtered_search("test", "invalid", {})

        # Empty query vector
        with pytest.raises(ValueError, match="query_vector cannot be empty"):
            await search_service.filtered_search("test", [], {})

        # Invalid filters type
        with pytest.raises(ValueError, match="filters must be a dictionary"):
            await search_service.filtered_search("test", [0.1] * 1536, "invalid")

        # Wrong vector dimension
        with pytest.raises(ValueError, match="query_vector dimension"):
            await search_service.filtered_search("test", [0.1] * 100, {})

    async def test_prefetch_limit_calculation(self, search_service):
        """Test prefetch limit calculation for different vector types."""
        # Test different vector types
        dense_limit = search_service._calculate_prefetch_limit(VectorType.DENSE, 10)
        sparse_limit = search_service._calculate_prefetch_limit(VectorType.SPARSE, 10)
        hyde_limit = search_service._calculate_prefetch_limit(VectorType.HYDE, 10)

        # Sparse should have highest multiplier
        assert sparse_limit > hyde_limit > dense_limit

        # Test maximum limits
        large_sparse = search_service._calculate_prefetch_limit(VectorType.SPARSE, 1000)
        assert large_sparse <= 500  # Max sparse limit

    async def test_search_params_generation(self, search_service):
        """Test search parameters generation for different accuracy levels."""
        fast_params = search_service._get_search_params(SearchAccuracy.FAST)
        balanced_params = search_service._get_search_params(SearchAccuracy.BALANCED)
        accurate_params = search_service._get_search_params(SearchAccuracy.ACCURATE)
        exact_params = search_service._get_search_params(SearchAccuracy.EXACT)

        # Fast should have lowest ef
        assert fast_params.hnsw_ef < balanced_params.hnsw_ef < accurate_params.hnsw_ef

        # Exact should disable HNSW
        assert exact_params.exact is True

    async def test_filter_building(self, search_service):
        """Test filter building for different types of conditions."""
        # Test keyword filters
        keyword_filters = {"doc_type": "api", "language": "python"}
        filter_obj = search_service._build_filter(keyword_filters)
        assert filter_obj is not None
        assert len(filter_obj.must) == 2

        # Test text filters
        text_filters = {"title": "test document"}
        filter_obj = search_service._build_filter(text_filters)
        assert filter_obj is not None

        # Test range filters
        range_filters = {"min_word_count": 100, "max_word_count": 1000}
        filter_obj = search_service._build_filter(range_filters)
        assert filter_obj is not None

        # Test empty filters
        assert search_service._build_filter({}) is None
        assert search_service._build_filter(None) is None

    async def test_error_handling(self, search_service, mock_client):
        """Test error handling in search operations."""
        # Mock client error
        mock_client.query_points.side_effect = Exception("Connection error")

        with pytest.raises(QdrantServiceError):
            await search_service.hybrid_search(
                collection_name="test_collection",
                query_vector=[0.1] * 1536,
            )

    async def test_collection_not_found_error(self, search_service, mock_client):
        """Test collection not found error handling."""
        mock_client.query_points.side_effect = Exception("Collection not found")

        with pytest.raises(QdrantServiceError, match="Collection .* not found"):
            await search_service.hybrid_search(
                collection_name="nonexistent",
                query_vector=[0.1] * 1536,
            )

    async def test_vector_dimension_mismatch_error(self, search_service, mock_client):
        """Test vector dimension mismatch error handling."""
        mock_client.query_points.side_effect = Exception("wrong vector size")

        with pytest.raises(QdrantServiceError, match="Vector dimension mismatch"):
            await search_service.hybrid_search(
                collection_name="test_collection",
                query_vector=[0.1] * 1536,
            )

    async def test_search_timeout_error(self, search_service, mock_client):
        """Test search timeout error handling."""
        mock_client.query_points.side_effect = Exception("timeout")

        with pytest.raises(QdrantServiceError, match="Search request timed out"):
            await search_service.hybrid_search(
                collection_name="test_collection",
                query_vector=[0.1] * 1536,
            )