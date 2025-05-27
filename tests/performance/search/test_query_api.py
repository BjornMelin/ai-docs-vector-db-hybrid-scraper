"""Tests for new Query API methods in QdrantService."""

from unittest.mock import AsyncMock
from unittest.mock import MagicMock
from unittest.mock import patch

import pytest
from src.config.models import QdrantConfig
from src.config.models import UnifiedConfig
from src.services.qdrant_service import QdrantService


@pytest.fixture
def config():
    """Create test configuration."""
    return UnifiedConfig(
        qdrant=QdrantConfig(
            url="http://localhost:6333",
            api_key="test-key",
        )
    )


@pytest.fixture
def qdrant_service(config):
    """Create Qdrant service instance."""
    return QdrantService(config)


class TestQueryAPI:
    """Test new Query API methods."""

    @pytest.mark.asyncio
    async def test_multi_stage_search_success(self, qdrant_service):
        """Test successful multi-stage search."""
        with patch("src.services.qdrant_service.AsyncQdrantClient") as mock_client:
            mock_instance = AsyncMock()
            mock_instance.get_collections.return_value = MagicMock(collections=[])

            # Mock successful query_points response
            mock_result = MagicMock()
            mock_result.id = "doc1"
            mock_result.score = 0.95
            mock_result.payload = {"content": "Test content"}
            mock_instance.query_points.return_value = MagicMock(points=[mock_result])

            mock_client.return_value = mock_instance

            await qdrant_service.initialize()

            stages = [
                {
                    "query_vector": [0.1] * 1536,
                    "vector_name": "dense",
                    "vector_type": "dense",
                    "limit": 20,
                },
                {
                    "query_vector": [0.2] * 1536,
                    "vector_name": "dense",
                    "vector_type": "dense",
                    "limit": 10,
                },
            ]

            results = await qdrant_service.multi_stage_search(
                collection_name="test_collection",
                stages=stages,
                limit=5,
                fusion_algorithm="rrf",
                search_accuracy="balanced",
            )

            assert len(results) == 1
            assert results[0]["id"] == "doc1"
            assert results[0]["score"] == 0.95
            mock_instance.query_points.assert_called_once()

    @pytest.mark.asyncio
    async def test_multi_stage_search_validation(self, qdrant_service):
        """Test multi-stage search input validation."""
        with patch("src.services.qdrant_service.AsyncQdrantClient") as mock_client:
            mock_instance = AsyncMock()
            mock_instance.get_collections.return_value = MagicMock(collections=[])
            mock_client.return_value = mock_instance

            await qdrant_service.initialize()

            # Test empty stages
            with pytest.raises(ValueError, match="Stages list cannot be empty"):
                await qdrant_service.multi_stage_search(
                    collection_name="test_collection",
                    stages=[],
                    limit=5,
                )

            # Test invalid stage format
            with pytest.raises(ValueError, match="Stage 0 must contain 'query_vector'"):
                await qdrant_service.multi_stage_search(
                    collection_name="test_collection",
                    stages=[{"invalid": "stage"}],
                    limit=5,
                )

    @pytest.mark.asyncio
    async def test_hyde_search_success(self, qdrant_service):
        """Test successful HyDE search."""
        with patch("src.services.qdrant_service.AsyncQdrantClient") as mock_client:
            mock_instance = AsyncMock()
            mock_instance.get_collections.return_value = MagicMock(collections=[])

            # Mock successful query_points response
            mock_result = MagicMock()
            mock_result.id = "doc1"
            mock_result.score = 0.88
            mock_result.payload = {"content": "HyDE test content"}
            mock_instance.query_points.return_value = MagicMock(points=[mock_result])

            mock_client.return_value = mock_instance

            await qdrant_service.initialize()

            results = await qdrant_service.hyde_search(
                collection_name="test_collection",
                query="test query",
                query_embedding=[0.1] * 1536,
                hypothetical_embeddings=[[0.2] * 1536, [0.3] * 1536],
                limit=5,
                fusion_algorithm="rrf",
                search_accuracy="balanced",
            )

            assert len(results) == 1
            assert results[0]["id"] == "doc1"
            assert results[0]["score"] == 0.88
            mock_instance.query_points.assert_called_once()

    @pytest.mark.asyncio
    async def test_filtered_search_success(self, qdrant_service):
        """Test successful filtered search."""
        with patch("src.services.qdrant_service.AsyncQdrantClient") as mock_client:
            mock_instance = AsyncMock()
            mock_instance.get_collections.return_value = MagicMock(collections=[])

            # Mock successful query_points response
            mock_result = MagicMock()
            mock_result.id = "doc1"
            mock_result.score = 0.92
            mock_result.payload = {"content": "Filtered content", "doc_type": "guide"}
            mock_instance.query_points.return_value = MagicMock(points=[mock_result])

            mock_client.return_value = mock_instance

            await qdrant_service.initialize()

            results = await qdrant_service.filtered_search(
                collection_name="test_collection",
                query_vector=[0.1] * 1536,
                filters={"doc_type": "guide"},
                limit=5,
                search_accuracy="balanced",
            )

            assert len(results) == 1
            assert results[0]["id"] == "doc1"
            assert results[0]["score"] == 0.92
            mock_instance.query_points.assert_called_once()

    @pytest.mark.asyncio
    async def test_filtered_search_validation(self, qdrant_service):
        """Test filtered search input validation."""
        with patch("src.services.qdrant_service.AsyncQdrantClient") as mock_client:
            mock_instance = AsyncMock()
            mock_instance.get_collections.return_value = MagicMock(collections=[])
            mock_client.return_value = mock_instance

            await qdrant_service.initialize()

            # Test invalid vector dimension
            with pytest.raises(ValueError, match="query_vector dimension"):
                await qdrant_service.filtered_search(
                    collection_name="test_collection",
                    query_vector=[0.1] * 100,  # Wrong dimension
                    filters={"doc_type": "guide"},
                    limit=5,
                )

            # Test empty query vector
            with pytest.raises(ValueError, match="query_vector cannot be empty"):
                await qdrant_service.filtered_search(
                    collection_name="test_collection",
                    query_vector=[],
                    filters={"doc_type": "guide"},
                    limit=5,
                )

            # Test invalid filter type
            with pytest.raises(ValueError, match="filters must be a dictionary"):
                await qdrant_service.filtered_search(
                    collection_name="test_collection",
                    query_vector=[0.1] * 1536,
                    filters="invalid",
                    limit=5,
                )

    @pytest.mark.asyncio
    async def test_prefetch_limit_calculations(self, qdrant_service):
        """Test prefetch limit calculations."""
        # Test dense vector multiplier (2x)
        dense_limit = qdrant_service._calculate_prefetch_limit("dense", 10)
        assert dense_limit == 20

        # Test sparse vector multiplier (5x)
        sparse_limit = qdrant_service._calculate_prefetch_limit("sparse", 10)
        assert sparse_limit == 50

        # Test HyDE vector multiplier (3x)
        hyde_limit = qdrant_service._calculate_prefetch_limit("hyde", 10)
        assert hyde_limit == 30

        # Test default (dense) for unknown type
        default_limit = qdrant_service._calculate_prefetch_limit("unknown", 10)
        assert default_limit == 20

    @pytest.mark.asyncio
    async def test_search_params_optimization(self, qdrant_service):
        """Test search parameter optimization for different accuracy levels."""
        # Test fast parameters
        fast_params = qdrant_service._get_search_params("fast")
        assert fast_params.hnsw_ef == 50
        assert fast_params.exact is False

        # Test balanced parameters
        balanced_params = qdrant_service._get_search_params("balanced")
        assert balanced_params.hnsw_ef == 100
        assert balanced_params.exact is False

        # Test accurate parameters
        accurate_params = qdrant_service._get_search_params("accurate")
        assert accurate_params.hnsw_ef == 200
        assert accurate_params.exact is False

        # Test exact parameters
        exact_params = qdrant_service._get_search_params("exact")
        assert exact_params.exact is True

        # Test default (balanced) for unknown level
        default_params = qdrant_service._get_search_params("unknown")
        assert default_params.hnsw_ef == 100

    @pytest.mark.asyncio
    async def test_filter_building(self, qdrant_service):
        """Test filter building with validation."""
        # Test valid filters
        filters = {"doc_type": "guide", "language": "en"}
        filter_obj = qdrant_service._build_filter(filters)
        assert filter_obj is not None

        # Test empty filters
        empty_filter = qdrant_service._build_filter({})
        assert empty_filter is None

        # Test None filters
        none_filter = qdrant_service._build_filter(None)
        assert none_filter is None

        # Test invalid filter value type
        with pytest.raises(
            ValueError, match="Filter value for doc_type must be a simple type"
        ):
            qdrant_service._build_filter({"doc_type": {"nested": "object"}})
