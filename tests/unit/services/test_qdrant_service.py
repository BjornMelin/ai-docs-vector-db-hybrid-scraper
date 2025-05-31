"""Tests for QdrantService with complete coverage."""

import pytest
from unittest.mock import AsyncMock, MagicMock, patch
from qdrant_client.http.exceptions import ResponseHandlingException, UnexpectedResponse
from qdrant_client.models import (
    CollectionInfo,
    CollectionStatus,
    Distance,
    HnswConfigDiff,
    PointIdsList,
    PointStruct,
    QuantizationSearchParams,
    ScoredPoint,
    SparseVector,
    UpdateResult,
    UpdateStatus,
    VectorParams,
    PayloadSchemaType,
    Filter,
    FieldCondition,
    MatchValue,
    MatchText,
    Range,
    Prefetch,
    FusionQuery,
    Fusion,
    QueryResponse,
    SearchParams,
)

from src.config.models import QdrantConfig, UnifiedConfig
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
    # Mock all methods we'll use
    client.get_collections = AsyncMock()
    client.create_collection = AsyncMock()
    client.get_collection = AsyncMock()
    client.delete_collection = AsyncMock()
    client.upsert = AsyncMock()
    client.delete = AsyncMock()
    client.update_collection = AsyncMock()
    client.create_payload_index = AsyncMock()
    client.delete_payload_index = AsyncMock()
    client.count = AsyncMock()
    client.query_points = AsyncMock()
    client.update_collection_aliases = AsyncMock()
    client.close = AsyncMock()
    return client


class TestQdrantServiceLifecycle:
    """Test service initialization and cleanup."""

    def test_service_creation(self, service, config):
        """Test service is created properly."""
        assert service.config == config
        assert service._client is None
        assert service._initialized is False

    @pytest.mark.asyncio
    async def test_initialize_success(self, service):
        """Test successful initialization."""
        with patch("src.services.core.qdrant_service.AsyncQdrantClient") as mock_class:
            mock_client = AsyncMock()
            mock_client.get_collections.return_value = MagicMock(collections=[])
            mock_class.return_value = mock_client
            
            await service.initialize()
            
            assert service._initialized is True
            assert service._client == mock_client
            mock_client.get_collections.assert_called_once()

    @pytest.mark.asyncio
    async def test_initialize_connection_failure(self, service):
        """Test initialization when connection fails."""
        with patch("src.services.core.qdrant_service.AsyncQdrantClient") as mock_class:
            mock_client = AsyncMock()
            mock_client.get_collections.side_effect = Exception("Connection refused")
            mock_class.return_value = mock_client
            
            with pytest.raises(QdrantServiceError, match="Qdrant connection check failed"):
                await service.initialize()
            
            assert service._initialized is False
            assert service._client is None

    @pytest.mark.asyncio
    async def test_initialize_idempotent(self, service):
        """Test multiple initialization calls."""
        with patch("src.services.core.qdrant_service.AsyncQdrantClient") as mock_class:
            mock_client = AsyncMock()
            mock_client.get_collections.return_value = MagicMock(collections=[])
            mock_class.return_value = mock_client
            
            await service.initialize()
            await service.initialize()  # Second call
            
            # Should only connect once
            assert mock_class.call_count == 1

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

    @pytest.mark.asyncio
    async def test_validate_initialized(self, service):
        """Test operation fails when not initialized."""
        with pytest.raises(QdrantServiceError, match="Service not initialized"):
            await service.create_collection("test", 1536)


class TestCollectionManagement:
    """Test collection operations."""

    @pytest.mark.asyncio
    async def test_create_collection_basic(self, service, mock_client):
        """Test basic collection creation."""
        service._client = mock_client
        service._initialized = True
        
        mock_client.get_collections.return_value = MagicMock(collections=[])
        mock_client.create_collection.return_value = True
        
        result = await service.create_collection(
            collection_name="test_collection",
            vector_size=1536,
            distance="Cosine"
        )
        
        assert result is True
        mock_client.create_collection.assert_called_once()
        
        # Verify call arguments
        call_args = mock_client.create_collection.call_args
        assert call_args.kwargs["collection_name"] == "test_collection"
        assert "vectors_config" in call_args.kwargs
        assert "dense" in call_args.kwargs["vectors_config"]

    @pytest.mark.asyncio
    async def test_create_collection_already_exists(self, service, mock_client):
        """Test creation when collection exists."""
        service._client = mock_client
        service._initialized = True
        
        existing = MagicMock()
        existing.name = "test_collection"
        mock_client.get_collections.return_value = MagicMock(collections=[existing])
        
        result = await service.create_collection("test_collection", 1536)
        
        assert result is True
        mock_client.create_collection.assert_not_called()

    @pytest.mark.asyncio
    async def test_create_collection_with_sparse_vectors(self, service, mock_client):
        """Test collection with sparse vectors for hybrid search."""
        service._client = mock_client
        service._initialized = True
        
        mock_client.get_collections.return_value = MagicMock(collections=[])
        
        result = await service.create_collection(
            collection_name="hybrid_collection",
            vector_size=384,
            sparse_vector_name="text-sparse",
            enable_quantization=True
        )
        
        assert result is True
        
        call_args = mock_client.create_collection.call_args
        assert "sparse_vectors_config" in call_args.kwargs
        assert "text-sparse" in call_args.kwargs["sparse_vectors_config"]

    @pytest.mark.asyncio
    async def test_delete_collection(self, service, mock_client):
        """Test collection deletion."""
        service._client = mock_client
        service._initialized = True
        
        mock_client.delete_collection.return_value = True
        
        result = await service.delete_collection("test_collection")
        
        assert result is True
        mock_client.delete_collection.assert_called_once_with("test_collection")

    @pytest.mark.asyncio
    async def test_delete_collection_not_found(self, service, mock_client):
        """Test deleting non-existent collection."""
        service._client = mock_client
        service._initialized = True
        
        mock_client.delete_collection.side_effect = Exception("Collection not found")
        
        with pytest.raises(QdrantServiceError, match="Failed to delete collection"):
            await service.delete_collection("nonexistent")

    @pytest.mark.asyncio
    async def test_get_collection_info(self, service, mock_client):
        """Test getting collection information."""
        service._client = mock_client
        service._initialized = True
        
        mock_info = MagicMock()
        mock_info.status = CollectionStatus.GREEN
        mock_info.vectors_count = 1000
        mock_info.points_count = 1000
        mock_info.config = MagicMock()
        mock_client.get_collection.return_value = mock_info
        
        result = await service.get_collection_info("test_collection")
        
        assert result["status"] == CollectionStatus.GREEN
        assert result["points_count"] == 1000
        assert result["vectors_count"] == 1000

    @pytest.mark.asyncio
    async def test_list_collections(self, service, mock_client):
        """Test listing collections."""
        service._client = mock_client
        service._initialized = True
        
        col1 = MagicMock()
        col1.name = "col1"
        col2 = MagicMock()
        col2.name = "col2"
        mock_client.get_collections.return_value = MagicMock(collections=[col1, col2])
        
        result = await service.list_collections()
        
        assert result == ["col1", "col2"]

    @pytest.mark.asyncio
    async def test_count_points(self, service, mock_client):
        """Test counting points in collection."""
        service._client = mock_client
        service._initialized = True
        
        mock_client.count.return_value = MagicMock(count=42)
        
        result = await service.count_points("test_collection")
        
        assert result == 42
        mock_client.count.assert_called_once_with(
            collection_name="test_collection",
            exact=True
        )


class TestPointOperations:
    """Test point/vector operations."""

    @pytest.mark.asyncio
    async def test_upsert_points_basic(self, service, mock_client):
        """Test basic point upsert."""
        service._client = mock_client
        service._initialized = True
        
        mock_client.upsert.return_value = UpdateResult(
            operation_id=1,
            status=UpdateStatus.COMPLETED
        )
        
        points = [
            {
                "id": "doc1",
                "vector": [0.1, 0.2, 0.3],
                "payload": {"text": "Test document"}
            }
        ]
        
        result = await service.upsert_points("test_collection", points)
        
        assert result is True
        mock_client.upsert.assert_called_once()

    @pytest.mark.asyncio
    async def test_upsert_points_with_sparse(self, service, mock_client):
        """Test upsert with sparse vectors."""
        service._client = mock_client
        service._initialized = True
        
        mock_client.upsert.return_value = UpdateResult(
            operation_id=1,
            status=UpdateStatus.COMPLETED
        )
        
        points = [
            {
                "id": "doc1",
                "vector": {
                    "dense": [0.1, 0.2, 0.3],
                    "text-sparse": {"indices": [0, 10], "values": [0.5, 0.3]}
                },
                "payload": {"text": "Hybrid document"}
            }
        ]
        
        result = await service.upsert_points("test_collection", points)
        
        assert result is True

    @pytest.mark.asyncio
    async def test_upsert_points_batch_handling(self, service, mock_client):
        """Test batch size handling."""
        service._client = mock_client
        service._initialized = True
        
        mock_client.upsert.return_value = UpdateResult(
            operation_id=1,
            status=UpdateStatus.COMPLETED
        )
        
        # Create 150 points (should be split into batches)
        points = [
            {"id": f"doc{i}", "vector": [0.1] * 384, "payload": {"idx": i}}
            for i in range(150)
        ]
        
        result = await service.upsert_points("test_collection", points, batch_size=100)
        
        assert result is True
        assert mock_client.upsert.call_count == 2  # Two batches

    @pytest.mark.asyncio
    async def test_delete_points(self, service, mock_client):
        """Test deleting points."""
        service._client = mock_client
        service._initialized = True
        
        mock_client.delete.return_value = UpdateResult(
            operation_id=1,
            status=UpdateStatus.COMPLETED
        )
        
        result = await service.delete_points(
            collection_name="test_collection",
            point_ids=["doc1", "doc2"]
        )
        
        assert result.status == UpdateStatus.COMPLETED
        mock_client.delete.assert_called_once()


class TestSearchOperations:
    """Test search operations."""

    @pytest.mark.asyncio
    async def test_hybrid_search_basic(self, service, mock_client):
        """Test basic hybrid search."""
        service._client = mock_client
        service._initialized = True
        
        mock_response = QueryResponse(
            points=[
                ScoredPoint(
                    id="doc1",
                    score=0.95,
                    payload={"text": "Result"}
                )
            ]
        )
        mock_client.query_points.return_value = mock_response
        
        result = await service.hybrid_search(
            collection_name="test_collection",
            query_vector=[0.1, 0.2, 0.3],
            limit=10
        )
        
        assert len(result) == 1
        assert result[0]["id"] == "doc1"
        assert result[0]["score"] == 0.95

    @pytest.mark.asyncio
    async def test_hybrid_search_with_sparse(self, service, mock_client):
        """Test hybrid search with sparse vectors."""
        service._client = mock_client
        service._initialized = True
        
        mock_response = QueryResponse(points=[])
        mock_client.query_points.return_value = mock_response
        
        result = await service.hybrid_search(
            collection_name="test_collection",
            query_vector=[0.1, 0.2, 0.3],
            sparse_vector={"indices": [0, 5], "values": [0.5, 0.3]},
            limit=10
        )
        
        # Should create prefetch queries for both dense and sparse
        call_args = mock_client.query_points.call_args
        assert "prefetch" in call_args.kwargs
        assert len(call_args.kwargs["prefetch"]) == 2

    @pytest.mark.asyncio
    async def test_filtered_search(self, service, mock_client):
        """Test filtered search."""
        service._client = mock_client
        service._initialized = True
        
        mock_response = QueryResponse(points=[])
        mock_client.query_points.return_value = mock_response
        
        result = await service.filtered_search(
            collection_name="test_collection",
            query_vector=[0.1] * 1536,
            filters={"doc_type": "api", "language": "python"},
            limit=10
        )
        
        # Should apply filters
        call_args = mock_client.query_points.call_args
        assert "filter" in call_args.kwargs

    @pytest.mark.asyncio
    async def test_multi_stage_search(self, service, mock_client):
        """Test multi-stage search."""
        service._client = mock_client
        service._initialized = True
        
        mock_response = QueryResponse(points=[])
        mock_client.query_points.return_value = mock_response
        
        stages = [
            {
                "query_vector": [0.1, 0.2, 0.3],
                "vector_name": "dense",
                "limit": 50
            },
            {
                "query_vector": [0.4, 0.5, 0.6],
                "vector_name": "dense",
                "limit": 20
            }
        ]
        
        result = await service.multi_stage_search(
            collection_name="test_collection",
            stages=stages,
            limit=10
        )
        
        # Should create prefetch queries
        call_args = mock_client.query_points.call_args
        assert "prefetch" in call_args.kwargs

    @pytest.mark.asyncio
    async def test_hyde_search(self, service, mock_client):
        """Test HyDE search."""
        service._client = mock_client
        service._initialized = True
        
        mock_response = QueryResponse(points=[])
        mock_client.query_points.return_value = mock_response
        
        result = await service.hyde_search(
            collection_name="test_collection",
            query="test query",
            query_embedding=[0.1, 0.2, 0.3],
            hypothetical_embeddings=[[0.2, 0.3, 0.4], [0.3, 0.4, 0.5]],
            limit=10
        )
        
        # Should use query_points with fusion
        mock_client.query_points.assert_called_once()


class TestPayloadIndexing:
    """Test payload indexing operations."""

    @pytest.mark.asyncio
    async def test_create_payload_indexes(self, service, mock_client):
        """Test creating payload indexes."""
        service._client = mock_client
        service._initialized = True
        
        await service.create_payload_indexes("test_collection")
        
        # Should create multiple indexes
        assert mock_client.create_payload_index.call_count > 10
        
        # Check some key indexes were created
        call_args_list = mock_client.create_payload_index.call_args_list
        indexed_fields = [call.kwargs["field_name"] for call in call_args_list]
        
        assert "doc_type" in indexed_fields
        assert "language" in indexed_fields
        assert "title" in indexed_fields
        assert "created_at" in indexed_fields

    @pytest.mark.asyncio
    async def test_list_payload_indexes(self, service, mock_client):
        """Test listing payload indexes."""
        service._client = mock_client
        service._initialized = True
        
        mock_info = MagicMock()
        mock_info.payload_schema = {
            "doc_type": MagicMock(index=True),
            "language": MagicMock(index=True),
            "title": MagicMock(index=False)
        }
        mock_client.get_collection.return_value = mock_info
        
        result = await service.list_payload_indexes("test_collection")
        
        assert "doc_type" in result
        assert "language" in result
        assert "title" not in result

    @pytest.mark.asyncio
    async def test_drop_payload_index(self, service, mock_client):
        """Test dropping payload index."""
        service._client = mock_client
        service._initialized = True
        
        await service.drop_payload_index("test_collection", "doc_type")
        
        mock_client.delete_payload_index.assert_called_once_with(
            collection_name="test_collection",
            field_name="doc_type",
            wait=True
        )

    @pytest.mark.asyncio
    async def test_validate_index_health(self, service, mock_client):
        """Test index health validation."""
        service._client = mock_client
        service._initialized = True
        
        # Mock collection info
        mock_info = MagicMock()
        mock_info.points_count = 1000
        mock_info.status = CollectionStatus.GREEN
        mock_client.get_collection.return_value = mock_info
        
        # Mock indexed fields
        with patch.object(service, "list_payload_indexes", return_value=["doc_type", "language"]):
            result = await service.validate_index_health("test_collection")
        
        assert "status" in result
        assert "health_score" in result
        assert "recommendations" in result


class TestOptimizationOperations:
    """Test optimization operations."""

    @pytest.mark.asyncio
    async def test_update_collection(self, service, mock_client):
        """Test updating collection parameters."""
        service._client = mock_client
        service._initialized = True
        
        mock_client.update_collection.return_value = True
        
        result = await service.update_collection(
            collection_name="test_collection",
            hnsw_config=HnswConfigDiff(m=32, ef_construct=200)
        )
        
        assert result is True
        mock_client.update_collection.assert_called_once()

    @pytest.mark.asyncio
    async def test_create_payload_index(self, service, mock_client):
        """Test creating single payload index."""
        service._client = mock_client
        service._initialized = True
        
        mock_client.create_payload_index.return_value = UpdateResult(
            operation_id=1,
            status=UpdateStatus.COMPLETED
        )
        
        result = await service.create_payload_index(
            collection_name="test_collection",
            field_name="category",
            field_type="keyword"
        )
        
        assert result.status == UpdateStatus.COMPLETED

    @pytest.mark.asyncio
    async def test_trigger_collection_optimization(self, service, mock_client):
        """Test triggering collection optimization."""
        service._client = mock_client
        service._initialized = True
        
        # Mock get_collection_info
        with patch.object(service, "get_collection_info", return_value={}):
            result = await service.trigger_collection_optimization("test_collection")
        
        assert result is True
        mock_client.update_collection_aliases.assert_called_once()


class TestErrorHandling:
    """Test error handling scenarios."""

    @pytest.mark.asyncio
    async def test_operation_not_initialized(self, service):
        """Test operations fail when not initialized."""
        with pytest.raises(QdrantServiceError, match="Service not initialized"):
            await service.hybrid_search("test", [0.1, 0.2], limit=10)

    @pytest.mark.asyncio
    async def test_search_collection_not_found(self, service, mock_client):
        """Test search with non-existent collection."""
        service._client = mock_client
        service._initialized = True
        
        mock_client.query_points.side_effect = Exception("collection not found")
        
        with pytest.raises(QdrantServiceError, match="Collection"):
            await service.hybrid_search("nonexistent", [0.1, 0.2], limit=10)

    @pytest.mark.asyncio
    async def test_upsert_dimension_mismatch(self, service, mock_client):
        """Test upsert with wrong vector dimension."""
        service._client = mock_client
        service._initialized = True
        
        mock_client.upsert.side_effect = Exception("wrong vector size")
        
        with pytest.raises(QdrantServiceError, match="dimension"):
            await service.upsert_points("test", [{"id": "1", "vector": [0.1]}])


class TestAdvancedFeatures:
    """Test advanced Qdrant features."""

    @pytest.mark.asyncio
    async def test_search_with_adaptive_ef(self, service, mock_client):
        """Test search with adaptive ef parameter."""
        service._client = mock_client
        service._initialized = True
        
        # Mock HNSWOptimizer
        with patch("src.services.core.qdrant_service.HNSWOptimizer") as mock_optimizer_class:
            mock_optimizer = AsyncMock()
            mock_optimizer.adaptive_ef_retrieve.return_value = {
                "results": [],
                "filtered_count": 0
            }
            mock_optimizer_class.return_value = mock_optimizer
            
            result = await service.search_with_adaptive_ef(
                collection_name="test_collection",
                query_vector=[0.1, 0.2, 0.3],
                limit=10,
                time_budget_ms=100
            )
            
            assert "results" in result

    @pytest.mark.asyncio
    async def test_get_hnsw_configuration_info(self, service):
        """Test getting HNSW configuration info."""
        service._initialized = True
        
        result = service.get_hnsw_configuration_info("api_reference")
        
        assert "collection_type" in result
        assert "hnsw_parameters" in result
        assert result["collection_type"] == "api_reference"

    @pytest.mark.asyncio
    async def test_list_collections_details(self, service, mock_client):
        """Test listing collections with details."""
        service._client = mock_client
        service._initialized = True
        
        col1 = MagicMock()
        col1.name = "col1"
        mock_client.get_collections.return_value = MagicMock(collections=[col1])
        
        # Mock get_collection_info
        with patch.object(service, "get_collection_info", return_value={
            "vectors_count": 100,
            "points_count": 100,
            "status": "GREEN"
        }):
            result = await service.list_collections_details()
        
        assert len(result) == 1
        assert result[0]["name"] == "col1"
        assert result[0]["vector_count"] == 100


class TestUtilityMethods:
    """Test utility methods."""

    def test_calculate_prefetch_limit(self, service):
        """Test prefetch limit calculation."""
        service._initialized = True
        
        # Test different vector types
        assert service._calculate_prefetch_limit("sparse", 10) == 50  # 5x multiplier
        assert service._calculate_prefetch_limit("hyde", 10) == 30    # 3x multiplier
        assert service._calculate_prefetch_limit("dense", 10) == 20   # 2x multiplier
        
        # Test max limits
        assert service._calculate_prefetch_limit("sparse", 200) == 500  # Max 500

    def test_get_search_params(self, service):
        """Test search parameter generation."""
        service._initialized = True
        
        # Test different accuracy levels
        fast = service._get_search_params("fast")
        assert fast.hnsw_ef == 50
        assert fast.exact is False
        
        exact = service._get_search_params("exact")
        assert exact.exact is True

    def test_build_filter(self, service):
        """Test filter building."""
        service._initialized = True
        
        # Test empty filter
        assert service._build_filter({}) is None
        
        # Test with filters
        filter_obj = service._build_filter({
            "doc_type": "api",
            "language": "python",
            "created_after": "2024-01-01"
        })
        
        assert filter_obj is not None
        assert len(filter_obj.must) > 0