"""Tests for QdrantCollections service."""

from unittest.mock import AsyncMock, MagicMock, PropertyMock, patch

import pytest
from qdrant_client import AsyncQdrantClient
from qdrant_client.http.exceptions import ResponseHandlingException

from src.config import Config
from src.services.base import BaseService
from src.services.errors import APIError, QdrantServiceError
from src.services.vector_db.collections import QdrantCollections


class TestQdrantCollections:
    """Test cases for QdrantCollections service."""

    @pytest.fixture
    def mock_config(self):
        """Create mock configuration."""
        config = MagicMock(spec=Config)

        # Mock HNSW configuration
        hnsw_config = MagicMock()
        hnsw_config.m = 16
        hnsw_config.ef_construct = 200
        hnsw_config.full_scan_threshold = 10000
        hnsw_config.max_indexing_threads = 0
        hnsw_config.on_disk = False

        collection_configs = MagicMock()
        collection_configs.general = hnsw_config
        collection_configs.api_reference = hnsw_config
        collection_configs.tutorials = hnsw_config
        collection_configs.blog_posts = hnsw_config
        collection_configs.code_examples = hnsw_config

        config.qdrant = MagicMock()
        config.qdrant.collection_hnsw_configs = collection_configs
        config.search = MagicMock()
        config.search.hnsw = MagicMock()
        config.search.hnsw.enable_adaptive_ef = True

        return config

    @pytest.fixture
    def mock_client(self):
        """Create mock AsyncQdrantClient."""
        return AsyncMock(spec=AsyncQdrantClient)

    @pytest.fixture
    def collections_service(self, mock_config, mock_client):
        """Create QdrantCollections instance."""
        return QdrantCollections(mock_config, mock_client)

    async def test_initialization_success(self, collections_service):
        """Test successful service initialization."""
        assert collections_service._initialized is True

    async def test_initialize_already_initialized(self, collections_service):
        """Test initialization when already initialized."""
        await collections_service.initialize()
        # Should complete without error

    async def test_initialize_no_client(self, mock_config):
        """Test initialization with no client."""
        service = QdrantCollections(mock_config, None)
        service._initialized = False

        with pytest.raises(QdrantServiceError, match="QdrantClient must be provided"):
            await service.initialize()

    async def test_cleanup_success(self, collections_service):
        """Test successful cleanup."""
        await collections_service.cleanup()
        assert collections_service._initialized is False

    async def test_create_collection_success(self, collections_service, mock_client):
        """Test successful collection creation."""
        mock_collections = MagicMock()
        mock_collections.collections = []
        mock_client.get_collections.return_value = mock_collections

        result = await collections_service.create_collection(
            collection_name="test_collection",
            vector_size=1536,
            distance="Cosine",
            sparse_vector_name="sparse",
            enable_quantization=True,
            collection_type="general",
        )

        assert result is True
        mock_client.create_collection.assert_called_once()

    async def test_create_collection_already_exists(
        self, collections_service, mock_client
    ):
        """Test collection creation when collection already exists."""
        existing_collection = MagicMock()
        existing_collection.name = "test_collection"
        mock_collections = MagicMock()
        mock_collections.collections = [existing_collection]
        mock_client.get_collections.return_value = mock_collections

        result = await collections_service.create_collection(
            collection_name="test_collection", vector_size=1536
        )

        assert result is True
        mock_client.create_collection.assert_not_called()

    async def test_create_collection_with_sparse_vectors(
        self, collections_service, mock_client
    ):
        """Test collection creation with sparse vectors."""
        mock_collections = MagicMock()
        mock_collections.collections = []
        mock_client.get_collections.return_value = mock_collections

        await collections_service.create_collection(
            collection_name="test_collection",
            vector_size=1536,
            sparse_vector_name="sparse_field",
        )

        # Verify sparse vectors config was passed
        call_args = mock_client.create_collection.call_args
        assert call_args.kwargs["sparse_vectors_config"] is not None

    async def test_create_collection_without_quantization(
        self, collections_service, mock_client
    ):
        """Test collection creation without quantization."""
        mock_collections = MagicMock()
        mock_collections.collections = []
        mock_client.get_collections.return_value = mock_collections

        await collections_service.create_collection(
            collection_name="test_collection",
            vector_size=1536,
            enable_quantization=False,
        )

        # Verify quantization config is None
        call_args = mock_client.create_collection.call_args
        assert call_args.kwargs["quantization_config"] is None

    async def test_create_collection_invalid_distance(
        self, collections_service, mock_client
    ):
        """Test collection creation with invalid distance metric."""
        mock_collections = MagicMock()
        mock_collections.collections = []
        mock_client.get_collections.return_value = mock_collections
        mock_client.create_collection.side_effect = ResponseHandlingException(
            "invalid distance"
        )

        with pytest.raises(QdrantServiceError, match="Invalid distance metric"):
            await collections_service.create_collection(
                collection_name="test_collection",
                vector_size=1536,
                distance="InvalidDistance",
            )

    async def test_create_collection_unauthorized(
        self, collections_service, mock_client
    ):
        """Test collection creation with unauthorized error."""
        mock_collections = MagicMock()
        mock_collections.collections = []
        mock_client.get_collections.return_value = mock_collections
        mock_client.create_collection.side_effect = ResponseHandlingException(
            "unauthorized"
        )

        with pytest.raises(QdrantServiceError, match="Unauthorized access to Qdrant"):
            await collections_service.create_collection(
                collection_name="test_collection", vector_size=1536
            )

    async def test_create_collection_already_exists_error(
        self, collections_service, mock_client
    ):
        """Test collection creation with 'already exists' error handling."""
        mock_collections = MagicMock()
        mock_collections.collections = []
        mock_client.get_collections.return_value = mock_collections
        mock_client.create_collection.side_effect = ResponseHandlingException(
            "already exists"
        )

        result = await collections_service.create_collection(
            collection_name="test_collection", vector_size=1536
        )

        assert result is True  # Should handle gracefully

    async def test_create_collection_payload_index_failure(
        self, collections_service, mock_client, caplog
    ):
        """Test collection creation with payload index creation failure."""
        mock_collections = MagicMock()
        mock_collections.collections = []
        mock_client.get_collections.return_value = mock_collections

        # Note: Payload index creation is now handled by QdrantService
        result = await collections_service.create_collection(
            collection_name="test_collection", vector_size=1536
        )

        assert result is True  # Collection creation should succeed

    async def test_delete_collection_success(self, collections_service, mock_client):
        """Test successful collection deletion."""
        result = await collections_service.delete_collection("test_collection")

        assert result is True
        mock_client.delete_collection.assert_called_once_with("test_collection")

    async def test_delete_collection_error(self, collections_service, mock_client):
        """Test collection deletion error."""
        mock_client.delete_collection.side_effect = Exception("Delete failed")

        with pytest.raises(QdrantServiceError, match="Failed to delete collection"):
            await collections_service.delete_collection("test_collection")

    async def test_get_collection_info_success(self, collections_service, mock_client):
        """Test successful collection info retrieval."""
        mock_info = MagicMock()
        mock_info.status = "green"
        mock_info.vectors_count = 1000
        mock_info.points_count = 1000
        mock_info.config = MagicMock()
        mock_info.config.model_dump.return_value = {"vector_size": 1536}
        mock_client.get_collection.return_value = mock_info

        result = await collections_service.get_collection_info("test_collection")

        assert result["status"] == "green"
        assert result["vectors_count"] == 1000
        assert result["points_count"] == 1000
        assert result["config"] == {"vector_size": 1536}

    async def test_get_collection_info_no_config(
        self, collections_service, mock_client
    ):
        """Test collection info retrieval with no config."""
        mock_info = MagicMock()
        mock_info.status = "green"
        mock_info.vectors_count = 1000
        mock_info.points_count = 1000
        mock_info.config = None
        mock_client.get_collection.return_value = mock_info

        result = await collections_service.get_collection_info("test_collection")

        assert result["config"] == {}

    async def test_get_collection_info_error(self, collections_service, mock_client):
        """Test collection info retrieval error."""
        mock_client.get_collection.side_effect = Exception("Info failed")

        with pytest.raises(QdrantServiceError, match="Failed to get collection info"):
            await collections_service.get_collection_info("test_collection")

    async def test_list_collections_success(self, collections_service, mock_client):
        """Test successful collection listing."""
        mock_collection1 = MagicMock()
        mock_collection1.name = "collection1"
        mock_collection2 = MagicMock()
        mock_collection2.name = "collection2"

        mock_collections = MagicMock()
        mock_collections.collections = [mock_collection1, mock_collection2]
        mock_client.get_collections.return_value = mock_collections

        result = await collections_service.list_collections()

        assert result == ["collection1", "collection2"]

    async def test_list_collections_error(self, collections_service, mock_client):
        """Test collection listing error."""
        mock_client.get_collections.side_effect = Exception("List failed")

        with pytest.raises(QdrantServiceError, match="Failed to list collections"):
            await collections_service.list_collections()

    async def test_list_collections_details_success(
        self, collections_service, mock_client
    ):
        """Test successful detailed collection listing."""
        mock_collection = MagicMock()
        mock_collection.name = "test_collection"

        mock_collections = MagicMock()
        mock_collections.collections = [mock_collection]
        mock_client.get_collections.return_value = mock_collections

        with patch.object(
            collections_service,
            "get_collection_info",
            return_value={
                "vectors_count": 1000,
                "points_count": 1000,
                "status": "green",
                "config": {},
            },
        ):
            result = await collections_service.list_collections_details()

        assert len(result) == 1
        assert result[0]["name"] == "test_collection"
        assert result[0]["vector_count"] == 1000
        assert result[0]["indexed_count"] == 1000

    async def test_list_collections_details_with_error(
        self, collections_service, mock_client
    ):
        """Test detailed collection listing with error for one collection."""
        mock_collection = MagicMock()
        mock_collection.name = "test_collection"

        mock_collections = MagicMock()
        mock_collections.collections = [mock_collection]
        mock_client.get_collections.return_value = mock_collections

        with patch.object(
            collections_service,
            "get_collection_info",
            side_effect=Exception("Info failed"),
        ):
            result = await collections_service.list_collections_details()

        assert len(result) == 1
        assert result[0]["name"] == "test_collection"
        assert "error" in result[0]

    async def test_list_collections_details_error(
        self, collections_service, mock_client
    ):
        """Test detailed collection listing error."""
        mock_client.get_collections.side_effect = Exception("List failed")

        with pytest.raises(
            QdrantServiceError, match="Failed to list collection details"
        ):
            await collections_service.list_collections_details()

    async def test_trigger_collection_optimization_success(
        self, collections_service, mock_client
    ):
        """Test successful collection optimization trigger."""
        with patch.object(
            collections_service, "get_collection_info", return_value={"status": "green"}
        ):
            result = await collections_service.trigger_collection_optimization(
                "test_collection"
            )

        assert result is True
        mock_client.update_collection_aliases.assert_called_once_with(
            change_aliases_operations=[]
        )

    async def test_trigger_collection_optimization_error(
        self, collections_service, mock_client
    ):
        """Test collection optimization trigger error."""
        mock_client.update_collection_aliases.side_effect = Exception(
            "Optimization failed"
        )

        with (
            patch.object(
                collections_service,
                "get_collection_info",
                return_value={"status": "green"},
            ),
            pytest.raises(QdrantServiceError, match="Failed to optimize collection"),
        ):
            await collections_service.trigger_collection_optimization("test_collection")

    def test_get_hnsw_configuration_info(self, collections_service):
        """Test HNSW configuration info retrieval."""
        config_info = collections_service.get_hnsw_configuration_info("api_reference")

        assert config_info["collection_type"] == "api_reference"
        assert "hnsw_parameters" in config_info
        assert "m" in config_info["hnsw_parameters"]
        assert "ef_construct" in config_info["hnsw_parameters"]
        assert "description" in config_info

    async def test_get_hnsw_config_for_collection_type(self, collections_service):
        """Test HNSW config retrieval for different collection types."""
        api_config = collections_service._get_hnsw_config_for_collection_type(
            "api_reference"
        )
        assert api_config is not None

        tutorial_config = collections_service._get_hnsw_config_for_collection_type(
            "tutorials"
        )
        assert tutorial_config is not None

        unknown_config = collections_service._get_hnsw_config_for_collection_type(
            "unknown"
        )
        assert unknown_config is not None  # Should fall back to general

    async def test_validate_hnsw_configuration_success(self, collections_service):
        """Test successful HNSW configuration validation."""
        mock_collection_info = MagicMock()
        mock_hnsw_config = MagicMock()
        mock_hnsw_config.m = 16
        mock_hnsw_config.ef_construct = 200
        mock_hnsw_config.on_disk = False
        mock_collection_info.config.hnsw_config = mock_hnsw_config
        mock_collection_info.points_count = 1000

        result = await collections_service._validate_hnsw_configuration(
            "test_collection", mock_collection_info
        )

        assert result["health_score"] > 0
        assert "current_configuration" in result
        assert "optimal_configuration" in result

    async def test_validate_hnsw_configuration_no_config(self, collections_service):
        """Test HNSW configuration validation with no config."""
        mock_collection_info = MagicMock()
        mock_collection_info.config = None
        mock_collection_info.points_count = 1000

        result = await collections_service._validate_hnsw_configuration(
            "test_collection", mock_collection_info
        )

        # Should use defaults
        assert result["current_configuration"]["m"] == 16
        assert result["current_configuration"]["ef_construct"] == 200

    async def test_validate_hnsw_configuration_error(self, collections_service):
        """Test HNSW configuration validation with error."""

        mock_collection_info = MagicMock()
        # Create a property that raises an exception when accessed
        type(mock_collection_info).points_count = PropertyMock(
            side_effect=Exception("Mock error")
        )

        # Should handle gracefully and return default healthy status
        result = await collections_service._validate_hnsw_configuration(
            "test_collection", mock_collection_info
        )

        assert result["health_score"] == 85.0

    async def test_infer_collection_type(self, collections_service):
        """Test collection type inference from name."""
        assert (
            collections_service._infer_collection_type("api_reference")
            == "api_reference"
        )
        assert (
            collections_service._infer_collection_type("tutorial_docs") == "tutorials"
        )
        assert collections_service._infer_collection_type("blog_posts") == "blog_posts"
        assert (
            collections_service._infer_collection_type("code_examples")
            == "code_examples"
        )
        assert (
            collections_service._infer_collection_type("random_name") == "general_docs"
        )

    async def test_calculate_hnsw_optimality_score(
        self, collections_service, mock_config
    ):
        """Test HNSW optimality score calculation."""
        optimal_config = mock_config.qdrant.collection_hnsw_configs.general

        # Perfect match
        current_config = {"m": 16, "ef_construct": 200, "on_disk": False}
        score = collections_service._calculate_hnsw_optimality_score(
            current_config, optimal_config
        )
        assert score == 100.0

        # Suboptimal configuration
        current_config = {"m": 32, "ef_construct": 100, "on_disk": True}
        score = collections_service._calculate_hnsw_optimality_score(
            current_config, optimal_config
        )
        assert score < 100.0

    async def test_inheritance_from_base_service(self, collections_service):
        """Test that QdrantCollections inherits from BaseService."""

        assert isinstance(collections_service, BaseService)

    async def test_not_initialized_check(self, mock_config, mock_client):
        """Test validation when service is not initialized."""
        service = QdrantCollections(mock_config, mock_client)
        service._initialized = False

        with pytest.raises(APIError, match="not initialized"):
            await service.create_collection("test", 1536)
