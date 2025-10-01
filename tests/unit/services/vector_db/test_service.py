"""Tests for QdrantService with ClientManager integration."""

from unittest.mock import AsyncMock, MagicMock

import pytest

from src.config import Config
from src.services.errors import QdrantServiceError
from src.services.vector_db.service import QdrantService


@pytest.fixture
def mock_config():
    """Create mock unified config."""
    config = MagicMock(spec=Config)
    config.qdrant = MagicMock()
    config.qdrant.url = "http://localhost:6333"
    config.qdrant.timeout = 30
    return config


@pytest.fixture
def mock_client_manager():
    """Create mock ClientManager."""
    manager = AsyncMock()
    manager.get_qdrant_client = AsyncMock()
    return manager


@pytest.fixture
def mock_qdrant_client():
    """Create mock Qdrant client."""
    client = AsyncMock()
    client.get_collections = AsyncMock(return_value=[])
    return client


@pytest.fixture
async def qdrant_service(mock_config, mock_client_manager):
    """Create QdrantService instance."""
    return QdrantService(mock_config, client_manager=mock_client_manager)


class TestQdrantServiceInitialization:
    """Test QdrantService initialization."""

    @pytest.mark.asyncio
    async def test_initialization_with_client_manager(
        self, mock_config, mock_client_manager, mock_qdrant_client
    ):
        """Test service initialization with ClientManager."""
        mock_client_manager.get_qdrant_client.return_value = mock_qdrant_client

        service = QdrantService(mock_config, client_manager=mock_client_manager)
        await service.initialize()

        assert service._initialized
        assert service._collections is not None
        assert service._search is not None
        assert service._indexing is not None
        assert service._documents is not None
        mock_client_manager.get_qdrant_client.assert_called_once()

    @pytest.mark.asyncio
    async def test_double_initialization(
        self, qdrant_service, mock_client_manager, mock_qdrant_client
    ):
        """Test that double initialization is safe."""
        mock_client_manager.get_qdrant_client.return_value = mock_qdrant_client

        await qdrant_service.initialize()
        await qdrant_service.initialize()  # Should not raise error

        assert qdrant_service._initialized

    @pytest.mark.asyncio
    async def test_initialization_failure(self, mock_config, mock_client_manager):
        """Test initialization failure handling."""
        mock_client_manager.get_qdrant_client.side_effect = Exception(
            "Connection failed"
        )

        service = QdrantService(mock_config, client_manager=mock_client_manager)

        with pytest.raises(QdrantServiceError, match="Failed to get Qdrant client"):
            await service.initialize()

        assert not service._initialized

    @pytest.mark.asyncio
    async def test_cleanup(
        self, qdrant_service, mock_client_manager, mock_qdrant_client
    ):
        """Test service cleanup."""
        mock_client_manager.get_qdrant_client.return_value = mock_qdrant_client

        await qdrant_service.initialize()
        await qdrant_service.cleanup()

        assert not qdrant_service._initialized
        assert qdrant_service._collections is None
        assert qdrant_service._search is None
        assert qdrant_service._indexing is None
        assert qdrant_service._documents is None


class TestQdrantServiceCollectionAPI:
    """Test collection management API delegation."""

    @pytest.mark.asyncio
    async def test_create_collection(
        self, qdrant_service, mock_client_manager, mock_qdrant_client
    ):
        """Test create collection with payload indexing."""
        mock_client_manager.get_qdrant_client.return_value = mock_qdrant_client
        await qdrant_service.initialize()

        # Mock the collections module
        qdrant_service._collections.create_collection = AsyncMock(return_value=True)
        qdrant_service._indexing.create_payload_indexes = AsyncMock()

        result = await qdrant_service.create_collection(
            collection_name="test_collection", vector_size=1536, distance="Cosine"
        )

        assert result is True
        qdrant_service._collections.create_collection.assert_called_once_with(
            collection_name="test_collection",
            vector_size=1536,
            distance="Cosine",
            sparse_vector_name=None,
            enable_quantization=True,
            collection_type="general",
        )
        qdrant_service._indexing.create_payload_indexes.assert_called_once_with(
            "test_collection"
        )

    @pytest.mark.asyncio
    async def test_create_collection_indexing_failure(
        self, qdrant_service, mock_client_manager, mock_qdrant_client
    ):
        """Test create collection when payload indexing fails."""
        mock_client_manager.get_qdrant_client.return_value = mock_qdrant_client
        await qdrant_service.initialize()

        qdrant_service._collections.create_collection = AsyncMock(return_value=True)
        qdrant_service._indexing.create_payload_indexes = AsyncMock(
            side_effect=Exception("Index failed")
        )

        result = await qdrant_service.create_collection(
            collection_name="test_collection", vector_size=1536
        )

        # Should still return True even if indexing fails
        assert result is True

    @pytest.mark.asyncio
    async def test_delete_collection(
        self, qdrant_service, mock_client_manager, mock_qdrant_client
    ):
        """Test delete collection delegation."""
        mock_client_manager.get_qdrant_client.return_value = mock_qdrant_client
        await qdrant_service.initialize()

        qdrant_service._collections.delete_collection = AsyncMock(return_value=True)

        result = await qdrant_service.delete_collection("test_collection")

        assert result is True
        qdrant_service._collections.delete_collection.assert_called_once_with(
            "test_collection"
        )

    @pytest.mark.asyncio
    async def test_list_collections(
        self, qdrant_service, mock_client_manager, mock_qdrant_client
    ):
        """Test list collections delegation."""
        mock_client_manager.get_qdrant_client.return_value = mock_qdrant_client
        await qdrant_service.initialize()

        expected_collections = ["collection1", "collection2"]
        qdrant_service._collections.list_collections = AsyncMock(
            return_value=expected_collections
        )

        result = await qdrant_service.list_collections()

        assert result == expected_collections
        qdrant_service._collections.list_collections.assert_called_once()

    @pytest.mark.asyncio
    async def test_get_collection_info(
        self, qdrant_service, mock_client_manager, mock_qdrant_client
    ):
        """Test get collection info delegation."""
        mock_client_manager.get_qdrant_client.return_value = mock_qdrant_client
        await qdrant_service.initialize()

        expected_info = {"name": "test", "vector_count": 100, "vector_size": 1536}
        qdrant_service._collections.get_collection_info = AsyncMock(
            return_value=expected_info
        )

        result = await qdrant_service.get_collection_info("test_collection")

        assert result == expected_info
        qdrant_service._collections.get_collection_info.assert_called_once_with(
            "test_collection"
        )


class TestQdrantServiceCollectionAPIExtended:
    """Extended tests for collection management API delegation."""

    @pytest.mark.asyncio
    async def test_list_collections_details(
        self, qdrant_service, mock_client_manager, mock_qdrant_client
    ):
        """Test list collections with details delegation."""
        mock_client_manager.get_qdrant_client.return_value = mock_qdrant_client
        await qdrant_service.initialize()

        expected_details = [
            {"name": "collection1", "vector_count": 100, "vector_size": 1536},
            {"name": "collection2", "vector_count": 50, "vector_size": 768},
        ]
        qdrant_service._collections.list_collections_details = AsyncMock(
            return_value=expected_details
        )

        result = await qdrant_service.list_collections_details()

        assert result == expected_details
        qdrant_service._collections.list_collections_details.assert_called_once()

    @pytest.mark.asyncio
    async def test_trigger_collection_optimization(
        self, qdrant_service, mock_client_manager, mock_qdrant_client
    ):
        """Test trigger collection optimization delegation."""
        mock_client_manager.get_qdrant_client.return_value = mock_qdrant_client
        await qdrant_service.initialize()

        qdrant_service._collections.trigger_collection_optimization = AsyncMock(
            return_value=True
        )

        result = await qdrant_service.trigger_collection_optimization("test_collection")

        assert result is True
        qdrant_service._collections.trigger_collection_optimization.assert_called_once_with(
            "test_collection"
        )


class TestQdrantServiceSearchAPI:
    """Test search API delegation."""

    @pytest.mark.asyncio
    async def test_hybrid_search(
        self, qdrant_service, mock_client_manager, mock_qdrant_client
    ):
        """Test hybrid search delegation."""
        mock_client_manager.get_qdrant_client.return_value = mock_qdrant_client
        await qdrant_service.initialize()

        query_vector = [0.1] * 1536
        sparse_vector = {0: 0.5, 1: 0.3}
        expected_results = [{"id": 1, "score": 0.9, "payload": {}}]

        qdrant_service._search.hybrid_search = AsyncMock(return_value=expected_results)

        result = await qdrant_service.hybrid_search(
            collection_name="test_collection",
            query_vector=query_vector,
            sparse_vector=sparse_vector,
            limit=10,
        )

        assert result == expected_results
        qdrant_service._search.hybrid_search.assert_called_once_with(
            collection_name="test_collection",
            query_vector=query_vector,
            sparse_vector=sparse_vector,
            limit=10,
            score_threshold=0.0,
            fusion_type="rrf",
            search_accuracy="balanced",
        )

    @pytest.mark.asyncio
    async def test_multi_stage_search(
        self, qdrant_service, mock_client_manager, mock_qdrant_client
    ):
        """Test multi-stage search delegation."""
        mock_client_manager.get_qdrant_client.return_value = mock_qdrant_client
        await qdrant_service.initialize()

        stages = [
            {"type": "vector", "vector": [0.1] * 1536, "limit": 20},
            {"type": "rerank", "model": "bge-reranker", "limit": 10},
        ]
        expected_results = [{"id": 1, "score": 0.9, "payload": {}}]

        qdrant_service._search.multi_stage_search = AsyncMock(
            return_value=expected_results
        )

        result = await qdrant_service.multi_stage_search(
            collection_name="test_collection",
            stages=stages,
            limit=10,
            fusion_algorithm="rrf",
        )

        assert result == expected_results
        qdrant_service._search.multi_stage_search.assert_called_once_with(
            collection_name="test_collection",
            stages=stages,
            limit=10,
            fusion_algorithm="rrf",
            search_accuracy="balanced",
        )

    @pytest.mark.asyncio
    async def test_hyde_search(
        self, qdrant_service, mock_client_manager, mock_qdrant_client
    ):
        """Test HyDE search delegation."""
        mock_client_manager.get_qdrant_client.return_value = mock_qdrant_client
        await qdrant_service.initialize()

        query = "How to implement vector search?"
        query_embedding = [0.1] * 1536
        hypothetical_embeddings = [[0.2] * 1536, [0.3] * 1536]
        expected_results = [{"id": 1, "score": 0.9, "payload": {}}]

        qdrant_service._search.hyde_search = AsyncMock(return_value=expected_results)

        result = await qdrant_service.hyde_search(
            collection_name="test_collection",
            query=query,
            query_embedding=query_embedding,
            hypothetical_embeddings=hypothetical_embeddings,
            limit=5,
        )

        assert result == expected_results
        qdrant_service._search.hyde_search.assert_called_once_with(
            collection_name="test_collection",
            query=query,
            query_embedding=query_embedding,
            hypothetical_embeddings=hypothetical_embeddings,
            limit=5,
            fusion_algorithm="rrf",
            search_accuracy="balanced",
        )

    @pytest.mark.asyncio
    async def test_filtered_search(
        self, qdrant_service, mock_client_manager, mock_qdrant_client
    ):
        """Test filtered search delegation."""
        mock_client_manager.get_qdrant_client.return_value = mock_qdrant_client
        await qdrant_service.initialize()

        query_vector = [0.1] * 1536
        filters = {"category": "test"}
        expected_results = [{"id": 1, "score": 0.9, "payload": {}}]

        qdrant_service._search.filtered_search = AsyncMock(
            return_value=expected_results
        )

        result = await qdrant_service.filtered_search(
            collection_name="test_collection",
            query_vector=query_vector,
            filters=filters,
            limit=5,
        )

        assert result == expected_results
        qdrant_service._search.filtered_search.assert_called_once_with(
            collection_name="test_collection",
            query_vector=query_vector,
            filters=filters,
            limit=5,
            search_accuracy="balanced",
            score_threshold=None,
        )


class TestQdrantServiceDocumentAPI:
    """Test document API delegation."""

    @pytest.mark.asyncio
    async def test_upsert_points(
        self, qdrant_service, mock_client_manager, mock_qdrant_client
    ):
        """Test upsert points delegation."""
        mock_client_manager.get_qdrant_client.return_value = mock_qdrant_client
        await qdrant_service.initialize()

        points = [{"id": 1, "vector": [0.1] * 1536, "payload": {"text": "test"}}]
        qdrant_service._documents.upsert_points = AsyncMock(return_value=True)

        result = await qdrant_service.upsert_points(
            collection_name="test_collection", points=points, batch_size=50
        )

        assert result is True
        qdrant_service._documents.upsert_points.assert_called_once_with(
            collection_name="test_collection", points=points, batch_size=50
        )

    @pytest.mark.asyncio
    async def test_get_points(
        self, qdrant_service, mock_client_manager, mock_qdrant_client
    ):
        """Test get points delegation."""
        mock_client_manager.get_qdrant_client.return_value = mock_qdrant_client
        await qdrant_service.initialize()

        point_ids = [1, 2, 3]
        expected_points = [
            {"id": 1, "payload": {"text": "test1"}},
            {"id": 2, "payload": {"text": "test2"}},
        ]
        qdrant_service._documents.get_points = AsyncMock(return_value=expected_points)

        result = await qdrant_service.get_points(
            collection_name="test_collection",
            point_ids=point_ids,
            with_payload=True,
            with_vectors=False,
        )

        assert result == expected_points
        qdrant_service._documents.get_points.assert_called_once_with(
            collection_name="test_collection",
            point_ids=point_ids,
            with_payload=True,
            with_vectors=False,
        )

    @pytest.mark.asyncio
    async def test_delete_points(
        self, qdrant_service, mock_client_manager, mock_qdrant_client
    ):
        """Test delete points delegation."""
        mock_client_manager.get_qdrant_client.return_value = mock_qdrant_client
        await qdrant_service.initialize()

        point_ids = [1, 2, 3]
        qdrant_service._documents.delete_points = AsyncMock(return_value=True)

        result = await qdrant_service.delete_points(
            collection_name="test_collection", point_ids=point_ids
        )

        assert result is True
        qdrant_service._documents.delete_points.assert_called_once_with(
            collection_name="test_collection",
            point_ids=point_ids,
            filter_condition=None,
        )

    @pytest.mark.asyncio
    async def test_update_point_payload(
        self, qdrant_service, mock_client_manager, mock_qdrant_client
    ):
        """Test update point payload delegation."""
        mock_client_manager.get_qdrant_client.return_value = mock_qdrant_client
        await qdrant_service.initialize()

        point_id = 1
        payload = {"category": "updated", "status": "active"}
        qdrant_service._documents.update_point_payload = AsyncMock(return_value=True)

        result = await qdrant_service.update_point_payload(
            collection_name="test_collection",
            point_id=point_id,
            payload=payload,
            replace=False,
        )

        assert result is True
        qdrant_service._documents.update_point_payload.assert_called_once_with(
            collection_name="test_collection",
            point_id=point_id,
            payload=payload,
            replace=False,
        )

    @pytest.mark.asyncio
    async def test_count_points(
        self, qdrant_service, mock_client_manager, mock_qdrant_client
    ):
        """Test count points delegation."""
        mock_client_manager.get_qdrant_client.return_value = mock_qdrant_client
        await qdrant_service.initialize()

        expected_count = 150
        qdrant_service._documents.count_points = AsyncMock(return_value=expected_count)

        result = await qdrant_service.count_points(
            collection_name="test_collection", filter_condition={"category": "test"}
        )

        assert result == expected_count
        qdrant_service._documents.count_points.assert_called_once_with(
            collection_name="test_collection",
            filter_condition={"category": "test"},
            exact=True,
        )

    @pytest.mark.asyncio
    async def test_scroll_points(
        self, qdrant_service, mock_client_manager, mock_qdrant_client
    ):
        """Test scroll points delegation."""
        mock_client_manager.get_qdrant_client.return_value = mock_qdrant_client
        await qdrant_service.initialize()

        expected_result = {
            "points": [{"id": 1, "payload": {"text": "test"}}],
            "next_page_offset": "next_token",
        }
        qdrant_service._documents.scroll_points = AsyncMock(
            return_value=expected_result
        )

        result = await qdrant_service.scroll_points(
            collection_name="test_collection",
            limit=50,
            offset="some_token",
            filter_condition={"category": "test"},
            with_payload=True,
            with_vectors=False,
        )

        assert result == expected_result
        qdrant_service._documents.scroll_points.assert_called_once_with(
            collection_name="test_collection",
            limit=50,
            offset="some_token",
            filter_condition={"category": "test"},
            with_payload=True,
            with_vectors=False,
        )

    @pytest.mark.asyncio
    async def test_clear_collection(
        self, qdrant_service, mock_client_manager, mock_qdrant_client
    ):
        """Test clear collection delegation."""
        mock_client_manager.get_qdrant_client.return_value = mock_qdrant_client
        await qdrant_service.initialize()

        qdrant_service._documents.clear_collection = AsyncMock(return_value=True)

        result = await qdrant_service.clear_collection("test_collection")

        assert result is True
        qdrant_service._documents.clear_collection.assert_called_once_with(
            "test_collection"
        )


class TestQdrantServiceIndexingAPI:
    """Test indexing API delegation."""

    @pytest.mark.asyncio
    async def test_create_payload_indexes(
        self, qdrant_service, mock_client_manager, mock_qdrant_client
    ):
        """Test create payload indexes delegation."""
        mock_client_manager.get_qdrant_client.return_value = mock_qdrant_client
        await qdrant_service.initialize()

        qdrant_service._indexing.create_payload_indexes = AsyncMock()

        await qdrant_service.create_payload_indexes("test_collection")

        qdrant_service._indexing.create_payload_indexes.assert_called_once_with(
            "test_collection"
        )

    @pytest.mark.asyncio
    async def test_list_payload_indexes(
        self, qdrant_service, mock_client_manager, mock_qdrant_client
    ):
        """Test list payload indexes delegation."""
        mock_client_manager.get_qdrant_client.return_value = mock_qdrant_client
        await qdrant_service.initialize()

        expected_indexes = ["url", "title", "category"]
        qdrant_service._indexing.list_payload_indexes = AsyncMock(
            return_value=expected_indexes
        )

        result = await qdrant_service.list_payload_indexes("test_collection")

        assert result == expected_indexes
        qdrant_service._indexing.list_payload_indexes.assert_called_once_with(
            "test_collection"
        )

    @pytest.mark.asyncio
    async def test_drop_payload_index(
        self, qdrant_service, mock_client_manager, mock_qdrant_client
    ):
        """Test drop payload index delegation."""
        mock_client_manager.get_qdrant_client.return_value = mock_qdrant_client
        await qdrant_service.initialize()

        qdrant_service._indexing.drop_payload_index = AsyncMock()

        await qdrant_service.drop_payload_index("test_collection", "category")

        qdrant_service._indexing.drop_payload_index.assert_called_once_with(
            "test_collection", "category"
        )

    @pytest.mark.asyncio
    async def test_reindex_collection(
        self, qdrant_service, mock_client_manager, mock_qdrant_client
    ):
        """Test reindex collection delegation."""
        mock_client_manager.get_qdrant_client.return_value = mock_qdrant_client
        await qdrant_service.initialize()

        qdrant_service._indexing.reindex_collection = AsyncMock()

        await qdrant_service.reindex_collection("test_collection")

        qdrant_service._indexing.reindex_collection.assert_called_once_with(
            "test_collection"
        )

    @pytest.mark.asyncio
    async def test_get_payload_index_stats(
        self, qdrant_service, mock_client_manager, mock_qdrant_client
    ):
        """Test get payload index stats delegation."""
        mock_client_manager.get_qdrant_client.return_value = mock_qdrant_client
        await qdrant_service.initialize()

        expected_stats = {
            "indexes": {"url": {"count": 1000}, "title": {"count": 950}},
            "_total_indexes": 2,
        }
        qdrant_service._indexing.get_payload_index_stats = AsyncMock(
            return_value=expected_stats
        )

        result = await qdrant_service.get_payload_index_stats("test_collection")

        assert result == expected_stats
        qdrant_service._indexing.get_payload_index_stats.assert_called_once_with(
            "test_collection"
        )

    @pytest.mark.asyncio
    async def test_validate_index_health(
        self, qdrant_service, mock_client_manager, mock_qdrant_client
    ):
        """Test validate index health delegation."""
        mock_client_manager.get_qdrant_client.return_value = mock_qdrant_client
        await qdrant_service.initialize()

        expected_health = {
            "status": "healthy",
            "indexes": {"url": "ok", "title": "ok"},
            "issues": [],
        }
        qdrant_service._indexing.validate_index_health = AsyncMock(
            return_value=expected_health
        )

        result = await qdrant_service.validate_index_health("test_collection")

        assert result == expected_health
        qdrant_service._indexing.validate_index_health.assert_called_once_with(
            "test_collection"
        )

    @pytest.mark.asyncio
    async def test_get_index_usage_stats(
        self, qdrant_service, mock_client_manager, mock_qdrant_client
    ):
        """Test get index usage stats delegation."""
        mock_client_manager.get_qdrant_client.return_value = mock_qdrant_client
        await qdrant_service.initialize()

        expected_usage = {
            "query_count": 1500,
            "index_usage": {"url": 800, "title": 700},
            "performance_metrics": {"avg_query_time": 0.05},
        }
        qdrant_service._indexing.get_index_usage_stats = AsyncMock(
            return_value=expected_usage
        )

        result = await qdrant_service.get_index_usage_stats("test_collection")

        assert result == expected_usage
        qdrant_service._indexing.get_index_usage_stats.assert_called_once_with(
            "test_collection"
        )


class TestQdrantServiceHNSWOptimization:
    """Test HNSW optimization API delegation."""

    @pytest.mark.asyncio
    async def test_create_collection_with_hnsw_optimization(
        self, qdrant_service, mock_client_manager, mock_qdrant_client
    ):
        """Test create collection with HNSW optimization delegation."""
        mock_client_manager.get_qdrant_client.return_value = mock_qdrant_client
        await qdrant_service.initialize()

        qdrant_service._collections.create_collection = AsyncMock(return_value=True)

        result = await qdrant_service.create_collection_with_hnsw_optimization(
            collection_name="test_collection",
            vector_size=1536,
            collection_type="api_reference",
            distance="Cosine",
            sparse_vector_name="sparse",
            enable_quantization=True,
        )

        assert result is True
        qdrant_service._collections.create_collection.assert_called_once_with(
            collection_name="test_collection",
            vector_size=1536,
            distance="Cosine",
            sparse_vector_name="sparse",
            enable_quantization=True,
            collection_type="api_reference",
        )

    def test_get_hnsw_configuration_info(self, qdrant_service):
        """Test get HNSW configuration info delegation."""
        # Mock the collections instance for the validation check
        qdrant_service._initialized = True
        qdrant_service._collections = AsyncMock()

        expected_config = {
            "m": 20,
            "ef_construct": 300,
            "full_scan_threshold": 5000,
            "min_ef": 100,
            "balanced_ef": 150,
            "max_ef": 200,
        }
        qdrant_service._collections.get_hnsw_configuration_info = MagicMock(
            return_value=expected_config
        )

        result = qdrant_service.get_hnsw_configuration_info("api_reference")

        assert result == expected_config
        qdrant_service._collections.get_hnsw_configuration_info.assert_called_once_with(
            "api_reference"
        )


class TestQdrantServiceValidation:
    """Test service validation."""

    @pytest.mark.asyncio
    async def test_validate_initialized_success(
        self, qdrant_service, mock_client_manager, mock_qdrant_client
    ):
        """Test validation when service is properly initialized."""
        mock_client_manager.get_qdrant_client.return_value = mock_qdrant_client
        await qdrant_service.initialize()

        # Should not raise error
        qdrant_service._validate_initialized()

    @pytest.mark.asyncio
    async def test_validate_initialized_not_initialized(self, qdrant_service):
        """Test validation when service is not initialized."""
        with pytest.raises(QdrantServiceError, match="Service not initialized"):
            qdrant_service._validate_initialized()

    @pytest.mark.asyncio
    async def test_validate_initialized_missing_collections(self, qdrant_service):
        """Test validation when collections module is missing."""
        qdrant_service._initialized = True
        qdrant_service._collections = None

        with pytest.raises(QdrantServiceError, match="Service not initialized"):
            qdrant_service._validate_initialized()

    @pytest.mark.asyncio
    async def test_api_methods_require_initialization(self, qdrant_service):
        """Test that API methods require initialization."""
        with pytest.raises(QdrantServiceError, match="Service not initialized"):
            await qdrant_service.list_collections()

        with pytest.raises(QdrantServiceError, match="Service not initialized"):
            await qdrant_service.create_collection("test", 1536)

        with pytest.raises(QdrantServiceError, match="Service not initialized"):
            await qdrant_service.hybrid_search("test", [0.1] * 1536)
        with pytest.raises(QdrantServiceError, match="Service not initialized"):
            await qdrant_service.list_collections()

        with pytest.raises(QdrantServiceError, match="Service not initialized"):
            await qdrant_service.create_collection("test", 1536)

        with pytest.raises(QdrantServiceError, match="Service not initialized"):
            await qdrant_service.hybrid_search("test", [0.1] * 1536)
