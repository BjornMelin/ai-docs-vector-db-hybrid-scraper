"""Test suite for QdrantService advanced functionality - search and upsert operations."""

from unittest.mock import AsyncMock
from unittest.mock import MagicMock

import pytest
from qdrant_client.http.exceptions import ResponseHandlingException
from src.config import UnifiedConfig
from src.services.errors import QdrantServiceError
from src.services.qdrant_service import QdrantService


@pytest.fixture
async def qdrant_service():
    """Create QdrantService instance with mocked client."""
    config = UnifiedConfig()
    service = QdrantService(config)

    # Mock the client
    mock_client = AsyncMock()
    service._client = mock_client
    service._initialized = True

    return service


@pytest.fixture
def sample_points():
    """Sample points for upsert testing."""
    return [
        {
            "id": "1",
            "vector": [0.1] * 1536,
            "payload": {"content": "Test content 1", "site_name": "FastAPI"},
        },
        {
            "id": "2",
            "vector": [0.2] * 1536,
            "payload": {"content": "Test content 2", "site_name": "Django"},
        },
    ]


class TestPointOperations:
    """Test point upsert and management operations."""

    @pytest.mark.asyncio
    async def test_upsert_points_success(self, qdrant_service, sample_points):
        """Test successful point upsert."""
        collection_name = "test_collection"

        qdrant_service._client.upsert = AsyncMock()

        result = await qdrant_service.upsert_points(
            collection_name=collection_name, points=sample_points, batch_size=100
        )

        assert result is True
        qdrant_service._client.upsert.assert_called_once()

        # Verify point structure conversion
        call_args = qdrant_service._client.upsert.call_args
        assert call_args.kwargs["collection_name"] == collection_name
        assert len(call_args.kwargs["points"]) == 2
        assert call_args.kwargs["wait"] is True

    @pytest.mark.asyncio
    async def test_upsert_points_batching(self, qdrant_service):
        """Test point upsert with batching."""
        collection_name = "test_collection"

        # Create more points than batch size
        points = [
            {
                "id": str(i),
                "vector": [0.1] * 1536,
                "payload": {"content": f"Content {i}"},
            }
            for i in range(5)
        ]

        qdrant_service._client.upsert = AsyncMock()

        result = await qdrant_service.upsert_points(
            collection_name=collection_name,
            points=points,
            batch_size=2,  # Force batching
        )

        assert result is True
        # Should be called 3 times (2+2+1)
        assert qdrant_service._client.upsert.call_count == 3

    @pytest.mark.asyncio
    async def test_upsert_points_collection_not_found(
        self, qdrant_service, sample_points
    ):
        """Test upsert with collection not found error."""
        collection_name = "nonexistent_collection"

        qdrant_service._client.upsert.side_effect = ResponseHandlingException(
            "collection not found"
        )

        with pytest.raises(
            QdrantServiceError, match="Collection 'nonexistent_collection' not found"
        ):
            await qdrant_service.upsert_points(collection_name, sample_points)

    @pytest.mark.asyncio
    async def test_upsert_points_vector_size_mismatch(
        self, qdrant_service, sample_points
    ):
        """Test upsert with vector size mismatch."""
        collection_name = "test_collection"

        qdrant_service._client.upsert.side_effect = ResponseHandlingException(
            "wrong vector size"
        )

        with pytest.raises(QdrantServiceError, match="Vector dimension mismatch"):
            await qdrant_service.upsert_points(collection_name, sample_points)

    @pytest.mark.asyncio
    async def test_upsert_points_payload_too_large(self, qdrant_service, sample_points):
        """Test upsert with payload too large error."""
        collection_name = "test_collection"

        qdrant_service._client.upsert.side_effect = ResponseHandlingException(
            "payload too large"
        )

        with pytest.raises(QdrantServiceError, match="Payload too large"):
            await qdrant_service.upsert_points(collection_name, sample_points)


class TestHybridSearch:
    """Test hybrid search functionality."""

    @pytest.mark.asyncio
    async def test_hybrid_search_dense_only(self, qdrant_service):
        """Test hybrid search with dense vector only."""
        collection_name = "test_collection"
        query_vector = [0.1] * 1536

        # Mock search results
        mock_results = MagicMock()
        mock_point = MagicMock()
        mock_point.id = "1"
        mock_point.score = 0.95
        mock_point.payload = {"content": "Test content"}
        mock_results.points = [mock_point]

        qdrant_service._client.query_points = AsyncMock(return_value=mock_results)

        results = await qdrant_service.hybrid_search(
            collection_name=collection_name, query_vector=query_vector, limit=10
        )

        assert len(results) == 1
        assert results[0]["id"] == "1"
        assert results[0]["score"] == 0.95
        assert results[0]["payload"] == {"content": "Test content"}

    @pytest.mark.asyncio
    async def test_hybrid_search_with_sparse_vector(self, qdrant_service):
        """Test hybrid search with both dense and sparse vectors."""
        collection_name = "test_collection"
        query_vector = [0.1] * 1536
        sparse_vector = {1: 0.5, 2: 0.3, 5: 0.8}

        # Mock search results
        mock_results = MagicMock()
        mock_point = MagicMock()
        mock_point.id = "1"
        mock_point.score = 0.95
        mock_point.payload = {"content": "Test content"}
        mock_results.points = [mock_point]

        qdrant_service._client.query_points = AsyncMock(return_value=mock_results)

        results = await qdrant_service.hybrid_search(
            collection_name=collection_name,
            query_vector=query_vector,
            sparse_vector=sparse_vector,
            limit=10,
            fusion_type="rrf",
        )

        assert len(results) == 1
        # Verify fusion query was used
        call_args = qdrant_service._client.query_points.call_args
        assert "prefetch" in call_args.kwargs
        assert len(call_args.kwargs["prefetch"]) == 2  # Dense + sparse

    @pytest.mark.asyncio
    async def test_hybrid_search_collection_not_found(self, qdrant_service):
        """Test hybrid search with collection not found."""
        collection_name = "nonexistent_collection"
        query_vector = [0.1] * 1536

        qdrant_service._client.query_points.side_effect = ResponseHandlingException(
            "collection not found"
        )

        with pytest.raises(
            QdrantServiceError, match="Collection 'nonexistent_collection' not found"
        ):
            await qdrant_service.hybrid_search(collection_name, query_vector)

    @pytest.mark.asyncio
    async def test_hybrid_search_vector_size_mismatch(self, qdrant_service):
        """Test hybrid search with wrong vector size."""
        collection_name = "test_collection"
        query_vector = [0.1] * 1536

        qdrant_service._client.query_points.side_effect = ResponseHandlingException(
            "wrong vector size"
        )

        with pytest.raises(QdrantServiceError, match="Vector dimension mismatch"):
            await qdrant_service.hybrid_search(collection_name, query_vector)

    @pytest.mark.asyncio
    async def test_hybrid_search_timeout(self, qdrant_service):
        """Test hybrid search timeout."""
        collection_name = "test_collection"
        query_vector = [0.1] * 1536

        qdrant_service._client.query_points.side_effect = ResponseHandlingException(
            "timeout"
        )

        with pytest.raises(QdrantServiceError, match="Search request timed out"):
            await qdrant_service.hybrid_search(collection_name, query_vector)


class TestFilteredSearch:
    """Test filtered search functionality."""

    @pytest.mark.asyncio
    async def test_filtered_search_success(self, qdrant_service):
        """Test successful filtered search."""
        collection_name = "test_collection"
        query_vector = [0.1] * 1536
        filters = {"site_name": "FastAPI Documentation", "min_word_count": 100}

        # Mock search results
        mock_results = MagicMock()
        mock_point = MagicMock()
        mock_point.id = "1"
        mock_point.score = 0.95
        mock_point.payload = {"content": "Test content"}
        mock_results.points = [mock_point]

        qdrant_service._client.query_points = AsyncMock(return_value=mock_results)

        results = await qdrant_service.filtered_search(
            collection_name=collection_name,
            query_vector=query_vector,
            filters=filters,
            limit=10,
        )

        assert len(results) == 1
        assert results[0]["id"] == "1"

        # Verify filter was applied
        call_args = qdrant_service._client.query_points.call_args
        assert call_args.kwargs["filter"] is not None

    @pytest.mark.asyncio
    async def test_filtered_search_validation_errors(self, qdrant_service):
        """Test filtered search input validation."""
        collection_name = "test_collection"

        # Test empty query vector
        with pytest.raises(ValueError, match="query_vector cannot be empty"):
            await qdrant_service.filtered_search(collection_name, [], {})

        # Test non-list query vector
        with pytest.raises(ValueError, match="query_vector must be a list"):
            await qdrant_service.filtered_search(collection_name, "invalid", {})

        # Test wrong vector dimension
        with pytest.raises(ValueError, match="query_vector dimension"):
            await qdrant_service.filtered_search(collection_name, [0.1] * 100, {})

        # Test invalid filters type
        with pytest.raises(ValueError, match="filters must be a dictionary"):
            await qdrant_service.filtered_search(
                collection_name, [0.1] * 1536, "invalid"
            )


class TestAdvancedQueryAPI:
    """Test advanced Query API methods."""

    @pytest.mark.asyncio
    async def test_multi_stage_search_success(self, qdrant_service):
        """Test multi-stage search success."""
        collection_name = "test_collection"
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

        # Mock search results
        mock_results = MagicMock()
        mock_point = MagicMock()
        mock_point.id = "1"
        mock_point.score = 0.95
        mock_point.payload = {"content": "Test content"}
        mock_results.points = [mock_point]

        qdrant_service._client.query_points = AsyncMock(return_value=mock_results)

        results = await qdrant_service.multi_stage_search(
            collection_name=collection_name, stages=stages, limit=10
        )

        assert len(results) == 1
        assert results[0]["id"] == "1"

    @pytest.mark.asyncio
    async def test_multi_stage_search_validation_errors(self, qdrant_service):
        """Test multi-stage search validation."""
        collection_name = "test_collection"

        # Test empty stages
        with pytest.raises(ValueError, match="Stages list cannot be empty"):
            await qdrant_service.multi_stage_search(collection_name, [])

        # Test non-list stages
        with pytest.raises(ValueError, match="Stages must be a list"):
            await qdrant_service.multi_stage_search(collection_name, "invalid")

        # Test invalid stage structure
        invalid_stages = [{"invalid": "stage"}]
        with pytest.raises(ValueError, match="Stage 0 must contain 'query_vector'"):
            await qdrant_service.multi_stage_search(collection_name, invalid_stages)

    @pytest.mark.asyncio
    async def test_hyde_search_success(self, qdrant_service):
        """Test HyDE search success."""
        collection_name = "test_collection"
        query = "What is FastAPI?"
        query_embedding = [0.1] * 1536
        hypothetical_embeddings = [[0.2] * 1536, [0.3] * 1536]

        # Mock search results
        mock_results = MagicMock()
        mock_point = MagicMock()
        mock_point.id = "1"
        mock_point.score = 0.95
        mock_point.payload = {"content": "FastAPI is a web framework"}
        mock_results.points = [mock_point]

        qdrant_service._client.query_points = AsyncMock(return_value=mock_results)

        # Skip test if numpy not available
        pytest.importorskip("numpy")

        results = await qdrant_service.hyde_search(
            collection_name=collection_name,
            query=query,
            query_embedding=query_embedding,
            hypothetical_embeddings=hypothetical_embeddings,
            limit=10,
        )

        assert len(results) == 1
        assert results[0]["id"] == "1"

    @pytest.mark.asyncio
    async def test_hyde_search_with_numpy_available(self, qdrant_service):
        """Test HyDE search when numpy is available."""
        collection_name = "test_collection"
        query = "What is FastAPI?"
        query_embedding = [0.1] * 1536
        hypothetical_embeddings = [[0.2] * 1536]

        # Mock search results
        mock_results = MagicMock()
        mock_point = MagicMock()
        mock_point.id = "1"
        mock_point.score = 0.95
        mock_point.payload = {"content": "FastAPI is a web framework"}
        mock_results.points = [mock_point]

        qdrant_service._client.query_points = AsyncMock(return_value=mock_results)

        # Skip if numpy not available
        pytest.importorskip("numpy")

        results = await qdrant_service.hyde_search(
            collection_name, query, query_embedding, hypothetical_embeddings
        )

        assert len(results) == 1
        assert results[0]["id"] == "1"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
