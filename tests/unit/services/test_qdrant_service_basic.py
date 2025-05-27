"""Test suite for QdrantService basic functionality - increasing coverage to 60%+."""

from unittest.mock import AsyncMock
from unittest.mock import MagicMock
from unittest.mock import patch

import pytest
from qdrant_client import models
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
def mock_config():
    """Create mock configuration."""
    config = MagicMock()
    config.qdrant.url = "http://localhost:6333"
    config.qdrant.api_key = None
    config.qdrant.timeout = 60
    config.qdrant.prefer_grpc = False
    return config


class TestQdrantServiceInitialization:
    """Test service initialization and cleanup."""

    @pytest.mark.asyncio
    async def test_initialize_success(self, mock_config):
        """Test successful service initialization."""
        service = QdrantService(mock_config)

        with patch(
            "src.services.qdrant_service.AsyncQdrantClient"
        ) as mock_client_class:
            mock_client = AsyncMock()
            mock_client_class.return_value = mock_client
            mock_client.get_collections.return_value = MagicMock()

            await service.initialize()

            assert service._initialized is True
            assert service._client is mock_client
            mock_client.get_collections.assert_called_once()

    @pytest.mark.asyncio
    async def test_initialize_already_initialized(self, mock_config):
        """Test initialization when already initialized."""
        service = QdrantService(mock_config)
        service._initialized = True
        service._client = AsyncMock()

        # Should return early without reinitializing
        await service.initialize()
        assert service._initialized is True

    @pytest.mark.asyncio
    async def test_initialize_connection_failure(self, mock_config):
        """Test initialization with connection failure."""
        service = QdrantService(mock_config)

        with patch(
            "src.services.qdrant_service.AsyncQdrantClient"
        ) as mock_client_class:
            mock_client = AsyncMock()
            mock_client_class.return_value = mock_client
            mock_client.get_collections.side_effect = Exception("Connection failed")

            with pytest.raises(
                QdrantServiceError, match="Qdrant connection check failed"
            ):
                await service.initialize()

            assert service._initialized is False
            assert service._client is None

    @pytest.mark.asyncio
    async def test_initialize_client_creation_failure(self, mock_config):
        """Test initialization with client creation failure."""
        service = QdrantService(mock_config)

        with patch(
            "src.services.qdrant_service.AsyncQdrantClient"
        ) as mock_client_class:
            mock_client_class.side_effect = Exception("Client creation failed")

            with pytest.raises(
                QdrantServiceError, match="Failed to initialize Qdrant client"
            ):
                await service.initialize()

            assert service._initialized is False
            assert service._client is None

    @pytest.mark.asyncio
    async def test_cleanup(self, qdrant_service):
        """Test service cleanup."""
        # Ensure client has close method and store reference
        mock_close = AsyncMock()
        qdrant_service._client.close = mock_close

        await qdrant_service.cleanup()

        mock_close.assert_called_once()
        assert qdrant_service._client is None
        assert qdrant_service._initialized is False

    @pytest.mark.asyncio
    async def test_cleanup_no_client(self, mock_config):
        """Test cleanup when no client exists."""
        service = QdrantService(mock_config)
        service._client = None

        # Should not raise error
        await service.cleanup()
        assert service._client is None
        assert service._initialized is False


class TestCollectionOperations:
    """Test collection CRUD operations."""

    @pytest.mark.asyncio
    async def test_create_collection_success(self, qdrant_service):
        """Test successful collection creation."""
        collection_name = "test_collection"
        vector_size = 1536

        # Mock collection doesn't exist
        qdrant_service._client.get_collections.return_value = MagicMock(collections=[])
        qdrant_service._client.create_collection = AsyncMock()
        qdrant_service.create_payload_indexes = AsyncMock()

        result = await qdrant_service.create_collection(
            collection_name=collection_name, vector_size=vector_size, distance="Cosine"
        )

        assert result is True
        qdrant_service._client.create_collection.assert_called_once()
        qdrant_service.create_payload_indexes.assert_called_once_with(collection_name)

    @pytest.mark.asyncio
    async def test_create_collection_already_exists(self, qdrant_service):
        """Test collection creation when it already exists."""
        collection_name = "test_collection"

        # Mock collection exists
        existing_collection = MagicMock()
        existing_collection.name = collection_name
        qdrant_service._client.get_collections.return_value = MagicMock(
            collections=[existing_collection]
        )

        result = await qdrant_service.create_collection(
            collection_name=collection_name, vector_size=1536
        )

        assert result is True
        qdrant_service._client.create_collection.assert_not_called()

    @pytest.mark.asyncio
    async def test_create_collection_with_sparse_vectors(self, qdrant_service):
        """Test collection creation with sparse vectors."""
        collection_name = "test_collection"

        qdrant_service._client.get_collections.return_value = MagicMock(collections=[])
        qdrant_service._client.create_collection = AsyncMock()
        qdrant_service.create_payload_indexes = AsyncMock()

        result = await qdrant_service.create_collection(
            collection_name=collection_name,
            vector_size=1536,
            sparse_vector_name="sparse",
            enable_quantization=True,
        )

        assert result is True
        # Verify create_collection was called with sparse vectors config
        call_args = qdrant_service._client.create_collection.call_args
        assert "sparse_vectors_config" in call_args.kwargs
        assert "quantization_config" in call_args.kwargs

    @pytest.mark.asyncio
    async def test_create_collection_error_handling(self, qdrant_service):
        """Test collection creation error handling."""
        collection_name = "test_collection"

        qdrant_service._client.get_collections.return_value = MagicMock(collections=[])

        # Test invalid distance by causing AttributeError
        with pytest.raises(
            AttributeError, match="type object 'Distance' has no attribute"
        ):
            await qdrant_service.create_collection(
                collection_name=collection_name,
                vector_size=1536,
                distance="InvalidDistance",
            )

    @pytest.mark.asyncio
    async def test_delete_collection_success(self, qdrant_service):
        """Test successful collection deletion."""
        collection_name = "test_collection"

        qdrant_service._client.delete_collection = AsyncMock()

        result = await qdrant_service.delete_collection(collection_name)

        assert result is True
        qdrant_service._client.delete_collection.assert_called_once_with(
            collection_name
        )

    @pytest.mark.asyncio
    async def test_delete_collection_failure(self, qdrant_service):
        """Test collection deletion failure."""
        collection_name = "test_collection"

        qdrant_service._client.delete_collection.side_effect = Exception(
            "Deletion failed"
        )

        with pytest.raises(QdrantServiceError, match="Failed to delete collection"):
            await qdrant_service.delete_collection(collection_name)

    @pytest.mark.asyncio
    async def test_get_collection_info_success(self, qdrant_service):
        """Test successful collection info retrieval."""
        collection_name = "test_collection"

        mock_info = MagicMock()
        mock_info.status = "green"
        mock_info.vectors_count = 1000
        mock_info.points_count = 500
        mock_info.config = MagicMock()
        mock_info.config.model_dump.return_value = {"vector_size": 1536}

        qdrant_service._client.get_collection.return_value = mock_info

        result = await qdrant_service.get_collection_info(collection_name)

        assert result["status"] == "green"
        assert result["vectors_count"] == 1000
        assert result["points_count"] == 500
        assert result["config"] == {"vector_size": 1536}

    @pytest.mark.asyncio
    async def test_get_collection_info_failure(self, qdrant_service):
        """Test collection info retrieval failure."""
        collection_name = "test_collection"

        qdrant_service._client.get_collection.side_effect = Exception(
            "Info retrieval failed"
        )

        with pytest.raises(QdrantServiceError, match="Failed to get collection info"):
            await qdrant_service.get_collection_info(collection_name)

    @pytest.mark.asyncio
    async def test_count_points_success(self, qdrant_service):
        """Test successful point counting."""
        collection_name = "test_collection"

        mock_result = MagicMock()
        mock_result.count = 1500
        qdrant_service._client.count.return_value = mock_result

        result = await qdrant_service.count_points(collection_name, exact=True)

        assert result == 1500
        qdrant_service._client.count.assert_called_once_with(
            collection_name=collection_name, exact=True
        )

    @pytest.mark.asyncio
    async def test_count_points_failure(self, qdrant_service):
        """Test point counting failure."""
        collection_name = "test_collection"

        qdrant_service._client.count.side_effect = Exception("Count failed")

        with pytest.raises(QdrantServiceError, match="Failed to count points"):
            await qdrant_service.count_points(collection_name)

    @pytest.mark.asyncio
    async def test_list_collections_success(self, qdrant_service):
        """Test successful collection listing."""
        mock_collections = MagicMock()
        collection1 = MagicMock()
        collection1.name = "collection1"
        collection2 = MagicMock()
        collection2.name = "collection2"
        mock_collections.collections = [collection1, collection2]

        qdrant_service._client.get_collections.return_value = mock_collections

        result = await qdrant_service.list_collections()

        assert result == ["collection1", "collection2"]

    @pytest.mark.asyncio
    async def test_list_collections_failure(self, qdrant_service):
        """Test collection listing failure."""
        qdrant_service._client.get_collections.side_effect = Exception("List failed")

        with pytest.raises(QdrantServiceError, match="Failed to list collections"):
            await qdrant_service.list_collections()

    @pytest.mark.asyncio
    async def test_list_collections_details_success(self, qdrant_service):
        """Test successful detailed collection listing."""
        # Mock collections list
        mock_collections = MagicMock()
        collection1 = MagicMock()
        collection1.name = "collection1"
        mock_collections.collections = [collection1]
        qdrant_service._client.get_collections.return_value = mock_collections

        # Mock get_collection_info
        qdrant_service.get_collection_info = AsyncMock(
            return_value={
                "vectors_count": 1000,
                "points_count": 500,
                "status": "green",
                "config": {"vector_size": 1536},
            }
        )

        result = await qdrant_service.list_collections_details()

        assert len(result) == 1
        assert result[0]["name"] == "collection1"
        assert result[0]["vector_count"] == 1000
        assert result[0]["indexed_count"] == 500
        assert result[0]["status"] == "green"

    @pytest.mark.asyncio
    async def test_list_collections_details_with_error(self, qdrant_service):
        """Test detailed collection listing with error for one collection."""
        # Mock collections list
        mock_collections = MagicMock()
        collection1 = MagicMock()
        collection1.name = "collection1"
        mock_collections.collections = [collection1]
        qdrant_service._client.get_collections.return_value = mock_collections

        # Mock get_collection_info to fail
        qdrant_service.get_collection_info = AsyncMock(
            side_effect=Exception("Info failed")
        )

        result = await qdrant_service.list_collections_details()

        assert len(result) == 1
        assert result[0]["name"] == "collection1"
        assert "error" in result[0]


class TestUtilityMethods:
    """Test utility and helper methods."""

    @pytest.mark.asyncio
    async def test_trigger_collection_optimization_success(self, qdrant_service):
        """Test successful collection optimization trigger."""
        collection_name = "test_collection"

        # Mock get_collection_info
        qdrant_service.get_collection_info = AsyncMock(return_value={"status": "green"})
        qdrant_service._client.update_collection_aliases = AsyncMock()

        result = await qdrant_service.trigger_collection_optimization(collection_name)

        assert result is True
        qdrant_service.get_collection_info.assert_called_once_with(collection_name)
        qdrant_service._client.update_collection_aliases.assert_called_once()

    @pytest.mark.asyncio
    async def test_trigger_collection_optimization_failure(self, qdrant_service):
        """Test collection optimization trigger failure."""
        collection_name = "test_collection"

        qdrant_service.get_collection_info = AsyncMock(
            side_effect=Exception("Collection not found")
        )

        with pytest.raises(QdrantServiceError, match="Failed to optimize collection"):
            await qdrant_service.trigger_collection_optimization(collection_name)

    def test_calculate_prefetch_limit(self, qdrant_service):
        """Test prefetch limit calculation."""
        # Test different vector types
        assert qdrant_service._calculate_prefetch_limit("sparse", 10) == 50
        assert qdrant_service._calculate_prefetch_limit("hyde", 10) == 30
        assert qdrant_service._calculate_prefetch_limit("dense", 10) == 20
        assert qdrant_service._calculate_prefetch_limit("unknown", 10) == 20

        # Test maximum limits
        assert (
            qdrant_service._calculate_prefetch_limit("sparse", 200) == 500
        )  # Max for sparse
        assert (
            qdrant_service._calculate_prefetch_limit("hyde", 100) == 150
        )  # Max for hyde
        assert (
            qdrant_service._calculate_prefetch_limit("dense", 200) == 200
        )  # Max for dense

    def test_get_search_params(self, qdrant_service):
        """Test search parameters generation."""
        # Test different accuracy levels
        fast_params = qdrant_service._get_search_params("fast")
        assert fast_params.hnsw_ef == 50
        assert fast_params.exact is False

        balanced_params = qdrant_service._get_search_params("balanced")
        assert balanced_params.hnsw_ef == 100
        assert balanced_params.exact is False

        accurate_params = qdrant_service._get_search_params("accurate")
        assert accurate_params.hnsw_ef == 200
        assert accurate_params.exact is False

        exact_params = qdrant_service._get_search_params("exact")
        assert exact_params.exact is True

        # Test default fallback
        default_params = qdrant_service._get_search_params("unknown")
        assert default_params.hnsw_ef == 100
        assert default_params.exact is False


class TestFilterBuildingMethods:
    """Test the new filter building helper methods."""

    def test_build_keyword_conditions(self, qdrant_service):
        """Test keyword conditions building."""
        filters = {
            "doc_type": "api",
            "language": "python",
            "site_name": "FastAPI Documentation",
        }

        conditions = qdrant_service._build_keyword_conditions(filters)

        assert len(conditions) == 3
        for condition in conditions:
            assert isinstance(condition, models.FieldCondition)
            assert condition.key in filters
            assert isinstance(condition.match, models.MatchValue)

    def test_build_keyword_conditions_invalid_value(self, qdrant_service):
        """Test keyword conditions with invalid value type."""
        filters = {"doc_type": ["invalid", "list"]}

        with pytest.raises(
            ValueError, match="Filter value for doc_type must be a simple type"
        ):
            qdrant_service._build_keyword_conditions(filters)

    def test_build_text_conditions(self, qdrant_service):
        """Test text conditions building."""
        filters = {"title": "API Documentation", "content_preview": "FastAPI tutorial"}

        conditions = qdrant_service._build_text_conditions(filters)

        assert len(conditions) == 2
        for condition in conditions:
            assert isinstance(condition, models.FieldCondition)
            assert isinstance(condition.match, models.MatchText)

    def test_build_text_conditions_invalid_value(self, qdrant_service):
        """Test text conditions with invalid value type."""
        filters = {"title": 123}

        with pytest.raises(
            ValueError, match="Text filter value for title must be a string"
        ):
            qdrant_service._build_text_conditions(filters)

    def test_build_timestamp_conditions(self, qdrant_service):
        """Test timestamp conditions building."""
        filters = {
            "created_after": 1640995200,
            "updated_before": 1641081600,
            "scraped_after": 1640995200,
        }

        conditions = qdrant_service._build_timestamp_conditions(filters)

        assert len(conditions) == 3
        for condition in conditions:
            assert isinstance(condition, models.FieldCondition)
            assert hasattr(condition, "range")

    def test_build_content_metric_conditions(self, qdrant_service):
        """Test content metric conditions building."""
        filters = {
            "min_word_count": 100,
            "max_word_count": 1000,
            "min_quality_score": 0.8,
        }

        conditions = qdrant_service._build_content_metric_conditions(filters)

        assert len(conditions) == 3
        for condition in conditions:
            assert isinstance(condition, models.FieldCondition)
            assert hasattr(condition, "range")

    def test_build_structural_conditions(self, qdrant_service):
        """Test structural conditions building."""
        filters = {
            "chunk_index": 5,
            "depth": 2,
            "min_total_chunks": 10,
            "max_links_count": 50,
        }

        conditions = qdrant_service._build_structural_conditions(filters)

        assert len(conditions) == 4
        # Check exact match conditions
        chunk_conditions = [c for c in conditions if c.key == "chunk_index"]
        assert len(chunk_conditions) == 1
        assert isinstance(chunk_conditions[0].match, models.MatchValue)

        # Check range conditions
        range_conditions = [
            c for c in conditions if hasattr(c, "range") and c.range is not None
        ]
        assert len(range_conditions) == 2


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
