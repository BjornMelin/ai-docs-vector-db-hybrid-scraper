"""Comprehensive tests for QdrantService."""

from unittest.mock import AsyncMock
from unittest.mock import MagicMock
from unittest.mock import patch

import pytest
from qdrant_client.http.exceptions import ResponseHandlingException
from qdrant_client.http.exceptions import UnexpectedResponse
from qdrant_client.models import CollectionInfo
from qdrant_client.models import CollectionStatus
from qdrant_client.models import Distance
from qdrant_client.models import HnswConfigDiff
from qdrant_client.models import PointIdsList
from qdrant_client.models import PointStruct
from qdrant_client.models import QuantizationSearchParams
from qdrant_client.models import Record
from qdrant_client.models import ScalarQuantizationConfig
from qdrant_client.models import ScoredPoint
from qdrant_client.models import SparseVector
from qdrant_client.models import SparseVectorParams
from qdrant_client.models import UpdateResult
from qdrant_client.models import UpdateStatus
from src.config.models import QdrantConfig
from src.config.models import UnifiedConfig
from src.services.core.qdrant_service import QdrantService
from src.services.errors import QdrantServiceError


@pytest.fixture
def config():
    """Create test configuration."""
    return UnifiedConfig(
        qdrant=QdrantConfig(
            url="http://localhost:6333",
            api_key="test-api-key",
            timeout=30,
            prefer_grpc=False,
        )
    )


@pytest.fixture
def service(config):
    """Create QdrantService instance."""
    return QdrantService(config)


@pytest.fixture
def mock_client():
    """Create mock Qdrant client."""
    client = AsyncMock()
    client.get_collections = AsyncMock()
    client.create_collection = AsyncMock()
    client.get_collection = AsyncMock()
    client.delete_collection = AsyncMock()
    client.upsert = AsyncMock()
    client.get_points = AsyncMock()
    client.search = AsyncMock()
    client.search_batch = AsyncMock()
    client.query_points = AsyncMock()
    client.delete = AsyncMock()
    client.update_collection = AsyncMock()
    client.create_payload_index = AsyncMock()
    client.count = AsyncMock()
    client.close = AsyncMock()
    return client


class TestQdrantServiceInitialization:
    """Test service initialization and configuration."""

    def test_service_creation(self, service, config):
        """Test basic service creation."""
        assert service.config == config
        assert service._client is None
        assert service._initialized is False

    @pytest.mark.asyncio
    async def test_initialize_success(self, service, mock_client):
        """Test successful initialization."""
        mock_collections = MagicMock(collections=[])
        mock_client.get_collections.return_value = mock_collections

        with patch(
            "src.services.core.qdrant_service.AsyncQdrantClient",
            return_value=mock_client,
        ):
            await service.initialize()

        assert service._initialized is True
        assert service._client == mock_client
        mock_client.get_collections.assert_called_once()

    @pytest.mark.asyncio
    async def test_initialize_connection_failure(self, service, mock_client):
        """Test initialization with connection failure."""
        mock_client.get_collections.side_effect = Exception("Connection refused")

        with (
            patch(
                "src.services.core.qdrant_service.AsyncQdrantClient",
                return_value=mock_client,
            ),
            pytest.raises(QdrantServiceError, match="Qdrant connection check failed"),
        ):
            await service.initialize()

        assert service._initialized is False
        assert service._client is None

    @pytest.mark.asyncio
    async def test_initialize_client_creation_failure(self, service):
        """Test initialization when client creation fails."""
        with (
            patch(
                "src.services.core.qdrant_service.AsyncQdrantClient",
                side_effect=Exception("Invalid config"),
            ),
            pytest.raises(
                QdrantServiceError, match="Failed to initialize Qdrant client"
            ),
        ):
            await service.initialize()

        assert service._initialized is False

    @pytest.mark.asyncio
    async def test_initialize_idempotent(self, service, mock_client):
        """Test that initialization is idempotent."""
        mock_client.get_collections.return_value = MagicMock(collections=[])

        with patch(
            "src.services.core.qdrant_service.AsyncQdrantClient",
            return_value=mock_client,
        ):
            await service.initialize()
            await service.initialize()  # Second call

        assert mock_client.get_collections.call_count == 1

    @pytest.mark.asyncio
    async def test_cleanup(self, service, mock_client):
        """Test cleanup process."""
        service._client = mock_client
        service._initialized = True

        await service.cleanup()

        mock_client.close.assert_called_once()
        assert service._client is None
        assert service._initialized is False

    @pytest.mark.asyncio
    async def test_cleanup_not_initialized(self, service):
        """Test cleanup when not initialized."""
        await service.cleanup()  # Should not raise


class TestCollectionOperations:
    """Test collection management operations."""

    @pytest.mark.asyncio
    async def test_create_collection_basic(self, service, mock_client):
        """Test basic collection creation."""
        service._client = mock_client
        service._initialized = True

        # Mock collection doesn't exist
        mock_client.get_collections.return_value = MagicMock(collections=[])
        mock_client.create_collection.return_value = True

        result = await service.create_collection(
            collection_name="test_collection", vector_size=1536, distance="Cosine"
        )

        assert result is True
        mock_client.create_collection.assert_called_once()

        # Verify collection config
        call_args = mock_client.create_collection.call_args
        assert call_args[0][0] == "test_collection"

        # Check vectors config
        vectors_config = call_args[1]["vectors_config"]
        assert isinstance(vectors_config, dict)
        assert "dense" in vectors_config
        assert vectors_config["dense"].size == 1536
        assert vectors_config["dense"].distance == Distance.COSINE

    @pytest.mark.asyncio
    async def test_create_collection_already_exists(self, service, mock_client):
        """Test creation when collection already exists."""
        service._client = mock_client
        service._initialized = True

        # Mock collection exists
        existing_collection = MagicMock(name="test_collection")
        mock_client.get_collections.return_value = MagicMock(
            collections=[existing_collection]
        )

        result = await service.create_collection(
            collection_name="test_collection", vector_size=1536
        )

        assert result is True
        mock_client.create_collection.assert_not_called()

    @pytest.mark.asyncio
    async def test_create_collection_with_sparse_vectors(self, service, mock_client):
        """Test collection creation with sparse vectors for hybrid search."""
        service._client = mock_client
        service._initialized = True

        mock_client.get_collections.return_value = MagicMock(collections=[])

        result = await service.create_collection(
            collection_name="hybrid_collection",
            vector_size=384,
            distance="Dot",
            sparse_vector_name="text-sparse",
            enable_quantization=True,
        )

        assert result is True

        call_args = mock_client.create_collection.call_args
        vectors_config = call_args[1]["vectors_config"]

        # Check sparse vector config
        assert "text-sparse" in vectors_config
        assert isinstance(vectors_config["text-sparse"], SparseVectorParams)

    @pytest.mark.asyncio
    async def test_create_collection_with_optimized_hnsw(self, service, mock_client):
        """Test collection creation with optimized HNSW parameters."""
        service._client = mock_client
        service._initialized = True

        mock_client.get_collections.return_value = MagicMock(collections=[])

        result = await service.create_collection(
            collection_name="api_docs", vector_size=768, collection_type="api_reference"
        )

        assert result is True

        call_args = mock_client.create_collection.call_args
        hnsw_config = call_args[1]["hnsw_config"]

        # Should have optimized settings for API reference
        assert hnsw_config.m == 32  # High connectivity
        assert hnsw_config.ef_construct == 200  # High quality
        assert hnsw_config.full_scan_threshold == 20000

    @pytest.mark.asyncio
    async def test_create_collection_not_initialized(self, service):
        """Test creation when service not initialized."""
        with pytest.raises(QdrantServiceError, match="Service not initialized"):
            await service.create_collection("test", 1536)

    @pytest.mark.asyncio
    async def test_create_collection_error_handling(self, service, mock_client):
        """Test error handling during collection creation."""
        service._client = mock_client
        service._initialized = True

        mock_client.get_collections.return_value = MagicMock(collections=[])
        mock_client.create_collection.side_effect = Exception("Creation failed")

        with pytest.raises(QdrantServiceError, match="Failed to create collection"):
            await service.create_collection("test", 1536)

    @pytest.mark.asyncio
    async def test_get_collection_info(self, service, mock_client):
        """Test getting collection information."""
        service._client = mock_client
        service._initialized = True

        mock_info = CollectionInfo(
            status=CollectionStatus.GREEN,
            optimizer_status=MagicMock(ok=True),
            vectors_count=1000,
            points_count=1000,
            segments_count=2,
            config=MagicMock(
                params=MagicMock(vectors=MagicMock(size=384, distance=Distance.COSINE))
            ),
        )
        mock_client.get_collection.return_value = mock_info

        result = await service.get_collection_info("test_collection")

        assert result == mock_info
        mock_client.get_collection.assert_called_once_with("test_collection")

    @pytest.mark.asyncio
    async def test_delete_collection(self, service, mock_client):
        """Test collection deletion."""
        service._client = mock_client
        service._initialized = True

        mock_client.delete_collection.return_value = True

        result = await service.delete_collection("old_collection")

        assert result is True
        mock_client.delete_collection.assert_called_once_with("old_collection")

    @pytest.mark.asyncio
    async def test_delete_collection_not_found(self, service, mock_client):
        """Test deleting non-existent collection."""
        service._client = mock_client
        service._initialized = True

        mock_client.delete_collection.side_effect = UnexpectedResponse(
            status_code=404, content=b"Collection not found"
        )

        # Should handle gracefully
        result = await service.delete_collection("nonexistent")
        assert result is True


class TestPointOperations:
    """Test point/vector operations."""

    @pytest.mark.asyncio
    async def test_upsert_points_basic(self, service, mock_client):
        """Test basic point upsert."""
        service._client = mock_client
        service._initialized = True

        mock_client.upsert.return_value = UpdateResult(
            operation_id=123, status=UpdateStatus.COMPLETED
        )

        points = [
            {
                "id": "doc1",
                "vector": [0.1, 0.2, 0.3],
                "payload": {"text": "Test document", "category": "test"},
            },
            {
                "id": "doc2",
                "vector": [0.4, 0.5, 0.6],
                "payload": {"text": "Another document", "category": "test"},
            },
        ]

        result = await service.upsert_points("test_collection", points)

        assert result.status == UpdateStatus.COMPLETED
        mock_client.upsert.assert_called_once()

        # Verify point structures
        call_args = mock_client.upsert.call_args
        assert call_args[0][0] == "test_collection"
        upserted_points = call_args[1]["points"]
        assert len(upserted_points) == 2
        assert all(isinstance(p, PointStruct) for p in upserted_points)

    @pytest.mark.asyncio
    async def test_upsert_points_with_sparse_vectors(self, service, mock_client):
        """Test upsert with both dense and sparse vectors."""
        service._client = mock_client
        service._initialized = True

        mock_client.upsert.return_value = UpdateResult(
            operation_id=124, status=UpdateStatus.COMPLETED
        )

        points = [
            {
                "id": "hybrid1",
                "vector": {
                    "dense": [0.1, 0.2, 0.3],
                    "text-sparse": {"indices": [0, 10, 20], "values": [0.5, 0.3, 0.2]},
                },
                "payload": {"text": "Hybrid search document"},
            }
        ]

        result = await service.upsert_points("hybrid_collection", points)

        assert result.status == UpdateStatus.COMPLETED

        # Verify sparse vector handling
        call_args = mock_client.upsert.call_args
        upserted_points = call_args[1]["points"]
        point = upserted_points[0]
        assert "dense" in point.vector
        assert "text-sparse" in point.vector
        assert isinstance(point.vector["text-sparse"], SparseVector)

    @pytest.mark.asyncio
    async def test_upsert_points_batch_size_handling(self, service, mock_client):
        """Test batch size handling for large upserts."""
        service._client = mock_client
        service._initialized = True

        mock_client.upsert.return_value = UpdateResult(
            operation_id=125, status=UpdateStatus.COMPLETED
        )

        # Create 150 points (should be split into batches of 100)
        points = [
            {"id": f"doc{i}", "vector": [0.1 * i] * 384, "payload": {"index": i}}
            for i in range(150)
        ]

        result = await service.upsert_points("test_collection", points, batch_size=100)

        assert result.status == UpdateStatus.COMPLETED
        assert mock_client.upsert.call_count == 2  # Two batches

    @pytest.mark.asyncio
    async def test_get_points(self, service, mock_client):
        """Test retrieving points by IDs."""
        service._client = mock_client
        service._initialized = True

        mock_points = [
            Record(
                id="doc1",
                vector={"dense": [0.1, 0.2, 0.3]},
                payload={"text": "Document 1"},
            ),
            Record(
                id="doc2",
                vector={"dense": [0.4, 0.5, 0.6]},
                payload={"text": "Document 2"},
            ),
        ]
        mock_client.get_points.return_value = mock_points

        result = await service.get_points(
            collection_name="test_collection", ids=["doc1", "doc2"]
        )

        assert result == mock_points
        mock_client.get_points.assert_called_once_with(
            collection_name="test_collection",
            ids=["doc1", "doc2"],
            with_payload=True,
            with_vectors=True,
        )

    @pytest.mark.asyncio
    async def test_delete_points(self, service, mock_client):
        """Test deleting points."""
        service._client = mock_client
        service._initialized = True

        mock_client.delete.return_value = UpdateResult(
            operation_id=126, status=UpdateStatus.COMPLETED
        )

        result = await service.delete_points(
            collection_name="test_collection", point_ids=["doc1", "doc2", "doc3"]
        )

        assert result.status == UpdateStatus.COMPLETED
        mock_client.delete.assert_called_once()

        call_args = mock_client.delete.call_args
        assert call_args[0][0] == "test_collection"
        assert isinstance(call_args[1]["points_selector"], PointIdsList)


class TestSearchOperations:
    """Test search and query operations."""

    @pytest.mark.asyncio
    async def test_search_basic(self, service, mock_client):
        """Test basic vector search."""
        service._client = mock_client
        service._initialized = True

        mock_results = [
            ScoredPoint(
                id="doc1",
                score=0.95,
                payload={"text": "Relevant document", "category": "api"},
            ),
            ScoredPoint(
                id="doc2",
                score=0.87,
                payload={"text": "Another result", "category": "tutorial"},
            ),
        ]
        mock_client.search.return_value = mock_results

        query_vector = [0.1, 0.2, 0.3]
        results = await service.search(
            collection_name="test_collection", query_vector=query_vector, limit=10
        )

        assert results == mock_results
        mock_client.search.assert_called_once_with(
            collection_name="test_collection",
            query_vector=query_vector,
            limit=10,
            with_payload=True,
            with_vectors=False,
        )

    @pytest.mark.asyncio
    async def test_search_with_filter(self, service, mock_client):
        """Test search with metadata filters."""
        service._client = mock_client
        service._initialized = True

        mock_client.search.return_value = []

        query_vector = [0.1, 0.2, 0.3]
        filter_dict = {"category": "api", "status": "published"}

        await service.search(
            collection_name="test_collection",
            query_vector=query_vector,
            filter=filter_dict,
            limit=5,
        )

        call_args = mock_client.search.call_args
        assert call_args[1]["filter"] == filter_dict

    @pytest.mark.asyncio
    async def test_search_hybrid(self, service, mock_client):
        """Test hybrid search with dense and sparse vectors."""
        service._client = mock_client
        service._initialized = True

        mock_client.search.return_value = []

        query_vectors = {
            "dense": [0.1, 0.2, 0.3],
            "text-sparse": {"indices": [0, 5, 10], "values": [0.5, 0.3, 0.2]},
        }

        await service.search_hybrid(
            collection_name="hybrid_collection", query_vectors=query_vectors, limit=10
        )

        # Should use query_points for hybrid search
        mock_client.query_points.assert_called_once()

    @pytest.mark.asyncio
    async def test_search_batch(self, service, mock_client):
        """Test batch search operations."""
        service._client = mock_client
        service._initialized = True

        mock_batch_results = [
            [ScoredPoint(id="doc1", score=0.9, payload={})],
            [ScoredPoint(id="doc2", score=0.8, payload={})],
        ]
        mock_client.search_batch.return_value = mock_batch_results

        queries = [
            {"vector": [0.1, 0.2, 0.3], "limit": 5},
            {"vector": [0.4, 0.5, 0.6], "limit": 5},
        ]

        results = await service.search_batch(
            collection_name="test_collection", requests=queries
        )

        assert results == mock_batch_results
        mock_client.search_batch.assert_called_once()

    @pytest.mark.asyncio
    async def test_search_with_score_threshold(self, service, mock_client):
        """Test search with score threshold."""
        service._client = mock_client
        service._initialized = True

        mock_client.search.return_value = []

        await service.search(
            collection_name="test_collection",
            query_vector=[0.1, 0.2, 0.3],
            score_threshold=0.8,
            limit=10,
        )

        call_args = mock_client.search.call_args
        assert call_args[1]["score_threshold"] == 0.8

    @pytest.mark.asyncio
    async def test_search_with_vectors(self, service, mock_client):
        """Test search returning vectors in results."""
        service._client = mock_client
        service._initialized = True

        mock_client.search.return_value = []

        await service.search(
            collection_name="test_collection",
            query_vector=[0.1, 0.2, 0.3],
            with_vectors=True,
        )

        call_args = mock_client.search.call_args
        assert call_args[1]["with_vectors"] is True


class TestOptimizationOperations:
    """Test collection optimization operations."""

    @pytest.mark.asyncio
    async def test_update_collection_hnsw(self, service, mock_client):
        """Test updating HNSW parameters."""
        service._client = mock_client
        service._initialized = True

        mock_client.update_collection.return_value = True

        result = await service.update_collection(
            collection_name="test_collection",
            hnsw_config=HnswConfigDiff(
                m=32, ef_construct=200, full_scan_threshold=20000
            ),
        )

        assert result is True
        mock_client.update_collection.assert_called_once()

    @pytest.mark.asyncio
    async def test_create_payload_index(self, service, mock_client):
        """Test creating payload index."""
        service._client = mock_client
        service._initialized = True

        mock_client.create_payload_index.return_value = UpdateResult(
            operation_id=127, status=UpdateStatus.COMPLETED
        )

        result = await service.create_payload_index(
            collection_name="test_collection",
            field_name="category",
            field_type="keyword",
        )

        assert result.status == UpdateStatus.COMPLETED
        mock_client.create_payload_index.assert_called_once()

        call_args = mock_client.create_payload_index.call_args
        assert call_args[0][0] == "test_collection"
        assert call_args[0][1] == "category"

    @pytest.mark.asyncio
    async def test_optimize_collection_for_search(self, service, mock_client):
        """Test collection optimization for search."""
        service._client = mock_client
        service._initialized = True

        # Mock get_collection to return current config
        mock_info = CollectionInfo(
            status=CollectionStatus.GREEN,
            optimizer_status=MagicMock(ok=True),
            vectors_count=50000,
            points_count=50000,
            segments_count=5,
            config=MagicMock(),
        )
        mock_client.get_collection.return_value = mock_info
        mock_client.update_collection.return_value = True
        mock_client.create_payload_index.return_value = UpdateResult(
            operation_id=128, status=UpdateStatus.COMPLETED
        )

        result = await service.optimize_collection_for_search(
            collection_name="large_collection",
            expected_point_count=50000,
            payload_indices=["category", "type", "status"],
        )

        assert result is True
        # Should update HNSW config
        mock_client.update_collection.assert_called()
        # Should create payload indices
        assert mock_client.create_payload_index.call_count == 3


class TestMetricsAndMonitoring:
    """Test metrics and monitoring operations."""

    @pytest.mark.asyncio
    async def test_get_collection_stats(self, service, mock_client):
        """Test getting collection statistics."""
        service._client = mock_client
        service._initialized = True

        mock_info = CollectionInfo(
            status=CollectionStatus.GREEN,
            optimizer_status=MagicMock(ok=True),
            vectors_count=10000,
            points_count=10000,
            segments_count=3,
            config=MagicMock(
                params=MagicMock(vectors=MagicMock(size=384), shard_number=1)
            ),
        )
        mock_client.get_collection.return_value = mock_info

        stats = await service.get_collection_stats("test_collection")

        assert stats["status"] == "green"
        assert stats["vectors_count"] == 10000
        assert stats["points_count"] == 10000
        assert stats["segments_count"] == 3
        assert stats["vector_size"] == 384

    @pytest.mark.asyncio
    async def test_count_points(self, service, mock_client):
        """Test counting points with filters."""
        service._client = mock_client
        service._initialized = True

        mock_client.count.return_value = MagicMock(count=150)

        count = await service.count_points(
            collection_name="test_collection", filter={"category": "api"}
        )

        assert count == 150
        mock_client.count.assert_called_once_with(
            collection_name="test_collection", count_filter={"category": "api"}
        )

    @pytest.mark.asyncio
    async def test_health_check(self, service, mock_client):
        """Test health check operation."""
        service._client = mock_client
        service._initialized = True

        mock_collections = MagicMock(
            collections=[MagicMock(name="col1"), MagicMock(name="col2")]
        )
        mock_client.get_collections.return_value = mock_collections

        health = await service.health_check()

        assert health["status"] == "healthy"
        assert health["collections_count"] == 2
        assert "response_time_ms" in health


class TestErrorHandling:
    """Test error handling scenarios."""

    @pytest.mark.asyncio
    async def test_operation_not_initialized(self, service):
        """Test operations when service not initialized."""
        with pytest.raises(QdrantServiceError, match="Service not initialized"):
            await service.search("collection", [0.1, 0.2, 0.3])

    @pytest.mark.asyncio
    async def test_search_error_handling(self, service, mock_client):
        """Test error handling in search operations."""
        service._client = mock_client
        service._initialized = True

        mock_client.search.side_effect = ResponseHandlingException("Search failed")

        with pytest.raises(QdrantServiceError, match="Search failed"):
            await service.search("collection", [0.1, 0.2, 0.3])

    @pytest.mark.asyncio
    async def test_upsert_error_handling(self, service, mock_client):
        """Test error handling in upsert operations."""
        service._client = mock_client
        service._initialized = True

        mock_client.upsert.side_effect = Exception("Upsert failed")

        with pytest.raises(QdrantServiceError, match="Failed to upsert points"):
            await service.upsert_points("collection", [{"id": "1", "vector": [0.1]}])

    @pytest.mark.asyncio
    async def test_collection_not_found_handling(self, service, mock_client):
        """Test handling of collection not found errors."""
        service._client = mock_client
        service._initialized = True

        mock_client.get_collection.side_effect = UnexpectedResponse(
            status_code=404, content=b"Collection not found"
        )

        with pytest.raises(QdrantServiceError, match="not found"):
            await service.get_collection_info("nonexistent")


class TestBatchProcessing:
    """Test batch processing operations."""

    @pytest.mark.asyncio
    async def test_batch_upsert_with_retry(self, service, mock_client):
        """Test batch upsert with retry on failure."""
        service._client = mock_client
        service._initialized = True

        # First call fails, second succeeds
        mock_client.upsert.side_effect = [
            Exception("Temporary failure"),
            UpdateResult(operation_id=129, status=UpdateStatus.COMPLETED),
        ]

        points = [{"id": "1", "vector": [0.1, 0.2, 0.3]}]

        # Should retry and succeed
        result = await service.upsert_points(
            "test_collection", points, wait=True, retry_on_failure=True
        )

        assert result.status == UpdateStatus.COMPLETED
        assert mock_client.upsert.call_count == 2

    @pytest.mark.asyncio
    async def test_parallel_batch_processing(self, service, mock_client):
        """Test parallel batch processing."""
        service._client = mock_client
        service._initialized = True

        mock_client.upsert.return_value = UpdateResult(
            operation_id=130, status=UpdateStatus.COMPLETED
        )

        # Create multiple batches
        batches = [
            [{"id": f"batch1_{i}", "vector": [0.1] * 384} for i in range(50)],
            [{"id": f"batch2_{i}", "vector": [0.2] * 384} for i in range(50)],
            [{"id": f"batch3_{i}", "vector": [0.3] * 384} for i in range(50)],
        ]

        # Process batches in parallel
        results = await service.upsert_batches_parallel(
            "test_collection", batches, max_parallel=2
        )

        assert all(r.status == UpdateStatus.COMPLETED for r in results)
        assert len(results) == 3


class TestAdvancedFeatures:
    """Test advanced Qdrant features."""

    @pytest.mark.asyncio
    async def test_scroll_points(self, service, mock_client):
        """Test scrolling through all points."""
        service._client = mock_client
        service._initialized = True

        # Mock paginated results
        page1 = MagicMock(
            points=[Record(id=f"doc{i}", payload={"i": i}) for i in range(100)],
            next_page_offset=100,
        )
        page2 = MagicMock(
            points=[Record(id=f"doc{i}", payload={"i": i}) for i in range(100, 150)],
            next_page_offset=None,
        )

        mock_client.scroll.side_effect = [page1, page2]

        all_points = []
        async for points in service.scroll_points("test_collection", limit=100):
            all_points.extend(points)

        assert len(all_points) == 150
        assert mock_client.scroll.call_count == 2

    @pytest.mark.asyncio
    async def test_recommend_points(self, service, mock_client):
        """Test recommendation based on positive/negative examples."""
        service._client = mock_client
        service._initialized = True

        mock_results = [
            ScoredPoint(id="rec1", score=0.85, payload={"text": "Recommended"}),
            ScoredPoint(id="rec2", score=0.82, payload={"text": "Also good"}),
        ]
        mock_client.recommend.return_value = mock_results

        results = await service.recommend(
            collection_name="test_collection",
            positive=["doc1", "doc2"],
            negative=["doc3"],
            limit=10,
        )

        assert results == mock_results
        mock_client.recommend.assert_called_once()

    @pytest.mark.asyncio
    async def test_snapshot_operations(self, service, mock_client):
        """Test snapshot creation and recovery."""
        service._client = mock_client
        service._initialized = True

        # Test creating snapshot
        mock_client.create_snapshot.return_value = MagicMock(
            name="snapshot_20240101_120000", status="completed"
        )

        snapshot = await service.create_snapshot("test_collection")
        assert snapshot.status == "completed"

        # Test listing snapshots
        mock_client.list_snapshots.return_value = [
            MagicMock(name="snapshot1"),
            MagicMock(name="snapshot2"),
        ]

        snapshots = await service.list_snapshots("test_collection")
        assert len(snapshots) == 2

    @pytest.mark.asyncio
    async def test_alias_operations(self, service, mock_client):
        """Test collection alias operations."""
        service._client = mock_client
        service._initialized = True

        # Test creating alias
        mock_client.update_collection_aliases.return_value = True

        result = await service.create_alias(
            alias_name="current_docs", collection_name="docs_v2"
        )
        assert result is True

        # Test switching alias
        result = await service.switch_alias(
            alias_name="current_docs",
            from_collection="docs_v1",
            to_collection="docs_v2",
        )
        assert result is True


class TestPerformanceOptimization:
    """Test performance optimization features."""

    @pytest.mark.asyncio
    async def test_quantization_configuration(self, service, mock_client):
        """Test configuring quantization for storage optimization."""
        service._client = mock_client
        service._initialized = True

        mock_client.update_collection.return_value = True

        result = await service.enable_quantization(
            collection_name="large_collection",
            quantization_config=ScalarQuantizationConfig(
                type="int8", quantile=0.99, always_ram=True
            ),
        )

        assert result is True
        mock_client.update_collection.assert_called_once()

    @pytest.mark.asyncio
    async def test_search_with_quantization(self, service, mock_client):
        """Test search with quantization parameters."""
        service._client = mock_client
        service._initialized = True

        mock_client.search.return_value = []

        await service.search(
            collection_name="quantized_collection",
            query_vector=[0.1, 0.2, 0.3],
            search_params={
                "quantization": QuantizationSearchParams(
                    ignore=False, rescore=True, oversampling=2.0
                )
            },
        )

        call_args = mock_client.search.call_args
        assert "search_params" in call_args[1]

    @pytest.mark.asyncio
    async def test_optimize_indexing_performance(self, service, mock_client):
        """Test optimizing indexing performance."""
        service._client = mock_client
        service._initialized = True

        mock_client.update_collection.return_value = True

        # Optimize for indexing (lower m, ef_construct)
        result = await service.optimize_for_indexing(collection_name="new_collection")

        assert result is True

        call_args = mock_client.update_collection.call_args
        hnsw_config = call_args[1]["hnsw_config"]
        assert hnsw_config.m <= 16  # Lower for faster indexing
        assert hnsw_config.ef_construct <= 100


class TestMigrationAndMaintenance:
    """Test migration and maintenance operations."""

    @pytest.mark.asyncio
    async def test_migrate_collection(self, service, mock_client):
        """Test collection migration."""
        service._client = mock_client
        service._initialized = True

        # Mock source collection info
        source_info = CollectionInfo(
            status=CollectionStatus.GREEN,
            vectors_count=10000,
            config=MagicMock(
                params=MagicMock(vectors=MagicMock(size=384, distance=Distance.COSINE))
            ),
        )
        mock_client.get_collection.return_value = source_info

        # Mock scroll for data migration
        mock_client.scroll.return_value = MagicMock(
            points=[
                Record(id=f"doc{i}", vector=[0.1] * 384, payload={"i": i})
                for i in range(100)
            ],
            next_page_offset=None,
        )

        mock_client.create_collection.return_value = True
        mock_client.upsert.return_value = UpdateResult(
            operation_id=131, status=UpdateStatus.COMPLETED
        )

        result = await service.migrate_collection(
            source_collection="old_collection",
            target_collection="new_collection",
            batch_size=100,
        )

        assert result is True
        mock_client.create_collection.assert_called_once()
        mock_client.upsert.assert_called()

    @pytest.mark.asyncio
    async def test_reindex_collection(self, service, mock_client):
        """Test collection reindexing."""
        service._client = mock_client
        service._initialized = True

        mock_client.update_collection.return_value = True

        result = await service.reindex_collection(
            collection_name="test_collection", wait_for_completion=True
        )

        assert result is True
        # Should trigger optimizer
        mock_client.update_collection.assert_called()

    @pytest.mark.asyncio
    async def test_validate_collection_consistency(self, service, mock_client):
        """Test collection consistency validation."""
        service._client = mock_client
        service._initialized = True

        mock_info = CollectionInfo(
            status=CollectionStatus.GREEN,
            optimizer_status=MagicMock(ok=True),
            vectors_count=1000,
            points_count=1000,
            segments_count=2,
        )
        mock_client.get_collection.return_value = mock_info

        is_consistent, issues = await service.validate_collection_consistency(
            "test_collection"
        )

        assert is_consistent is True
        assert len(issues) == 0

        # Test with issues
        mock_info.status = CollectionStatus.YELLOW
        mock_info.optimizer_status.ok = False

        is_consistent, issues = await service.validate_collection_consistency(
            "test_collection"
        )

        assert is_consistent is False
        assert len(issues) > 0
