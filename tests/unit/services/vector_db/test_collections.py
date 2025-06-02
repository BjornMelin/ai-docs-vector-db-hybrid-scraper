"""Tests for QdrantCollections service."""

import logging
from unittest.mock import AsyncMock
from unittest.mock import MagicMock
from unittest.mock import patch

import pytest
from qdrant_client import AsyncQdrantClient
from qdrant_client.http.exceptions import ResponseHandlingException
from src.config import UnifiedConfig
from src.services.errors import QdrantServiceError
from src.services.vector_db.collections import QdrantCollections


class TestQdrantCollections:
    """Test cases for QdrantCollections service."""

    @pytest.fixture
    def mock_config(self):
        """Create mock configuration."""
        config = MagicMock(spec=UnifiedConfig)

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

        with patch.object(
            collections_service, "create_payload_indexes"
        ) as mock_create_indexes:
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
        mock_create_indexes.assert_called_once_with("test_collection")

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

        with patch.object(collections_service, "create_payload_indexes"):
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

        with patch.object(collections_service, "create_payload_indexes"):
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

        with patch.object(
            collections_service,
            "create_payload_indexes",
            side_effect=Exception("Index failed"),
        ):
            with caplog.at_level(logging.WARNING):
                result = await collections_service.create_collection(
                    collection_name="test_collection", vector_size=1536
                )

        assert result is True  # Collection creation should succeed
        assert "Failed to create payload indexes" in caplog.text

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

        with patch.object(
            collections_service, "get_collection_info", return_value={"status": "green"}
        ):
            with pytest.raises(
                QdrantServiceError, match="Failed to optimize collection"
            ):
                await collections_service.trigger_collection_optimization(
                    "test_collection"
                )

    async def test_create_payload_indexes_success(
        self, collections_service, mock_client
    ):
        """Test successful payload index creation."""
        await collections_service.create_payload_indexes("test_collection")

        # Should create multiple indexes
        assert mock_client.create_payload_index.call_count > 0

    async def test_create_payload_indexes_error(self, collections_service, mock_client):
        """Test payload index creation error."""
        mock_client.create_payload_index.side_effect = Exception("Index failed")

        with pytest.raises(
            QdrantServiceError, match="Failed to create payload indexes"
        ):
            await collections_service.create_payload_indexes("test_collection")

    async def test_list_payload_indexes_success(self, collections_service, mock_client):
        """Test successful payload index listing."""
        mock_collection_info = MagicMock()
        mock_field_info = MagicMock()
        mock_field_info.index = True
        mock_collection_info.payload_schema = {
            "field1": mock_field_info,
            "field2": MagicMock(index=False),
        }
        mock_client.get_collection.return_value = mock_collection_info

        result = await collections_service.list_payload_indexes("test_collection")

        assert "field1" in result
        assert "field2" not in result

    async def test_list_payload_indexes_no_schema(
        self, collections_service, mock_client
    ):
        """Test payload index listing with no schema."""
        mock_collection_info = MagicMock()
        mock_collection_info.payload_schema = None
        mock_client.get_collection.return_value = mock_collection_info

        result = await collections_service.list_payload_indexes("test_collection")

        assert result == []

    async def test_list_payload_indexes_error(self, collections_service, mock_client):
        """Test payload index listing error."""
        mock_client.get_collection.side_effect = Exception("List failed")

        with pytest.raises(QdrantServiceError, match="Failed to list payload indexes"):
            await collections_service.list_payload_indexes("test_collection")

    async def test_drop_payload_index_success(self, collections_service, mock_client):
        """Test successful payload index drop."""
        await collections_service.drop_payload_index("test_collection", "test_field")

        mock_client.delete_payload_index.assert_called_once_with(
            collection_name="test_collection", field_name="test_field", wait=True
        )

    async def test_drop_payload_index_error(self, collections_service, mock_client):
        """Test payload index drop error."""
        mock_client.delete_payload_index.side_effect = Exception("Drop failed")

        with pytest.raises(QdrantServiceError, match="Failed to drop payload index"):
            await collections_service.drop_payload_index(
                "test_collection", "test_field"
            )

    async def test_validate_index_health_healthy(
        self, collections_service, mock_client
    ):
        """Test index health validation with healthy status."""
        mock_collection_info = MagicMock()
        mock_collection_info.points_count = 1000
        mock_client.get_collection.return_value = mock_collection_info

        # Mock all expected indexes are present
        expected_indexes = [
            "doc_type",
            "language",
            "framework",
            "version",
            "crawl_source",
            "site_name",
            "embedding_model",
            "embedding_provider",
            "title",
            "content_preview",
            "created_at",
            "last_updated",
            "word_count",
            "char_count",
        ]

        with (
            patch.object(
                collections_service,
                "list_payload_indexes",
                return_value=expected_indexes,
            ),
            patch.object(
                collections_service,
                "_validate_hnsw_configuration",
                return_value={"health_score": 95.0, "recommendations": []},
            ),
        ):
            result = await collections_service.validate_index_health("test_collection")

        assert result["status"] == "healthy"
        assert result["health_score"] >= 95

    async def test_validate_index_health_missing_indexes(
        self, collections_service, mock_client
    ):
        """Test index health validation with missing indexes."""
        mock_collection_info = MagicMock()
        mock_collection_info.points_count = 1000
        mock_client.get_collection.return_value = mock_collection_info

        # Mock missing some indexes
        partial_indexes = ["doc_type", "language"]

        with patch.object(
            collections_service, "list_payload_indexes", return_value=partial_indexes
        ):
            result = await collections_service.validate_index_health("test_collection")

        assert result["status"] in ["warning", "critical"]
        assert len(result["payload_indexes"]["missing_indexes"]) > 0

    async def test_validate_index_health_extra_indexes(
        self, collections_service, mock_client
    ):
        """Test index health validation with extra indexes."""
        mock_collection_info = MagicMock()
        mock_collection_info.points_count = 1000
        mock_client.get_collection.return_value = mock_collection_info

        # Mock having extra indexes
        indexes_with_extra = [
            "doc_type",
            "language",
            "framework",
            "version",
            "crawl_source",
            "site_name",
            "embedding_model",
            "embedding_provider",
            "title",
            "content_preview",
            "created_at",
            "last_updated",
            "word_count",
            "char_count",
            "extra_field1",
            "extra_field2",
        ]

        with patch.object(
            collections_service, "list_payload_indexes", return_value=indexes_with_extra
        ):
            result = await collections_service.validate_index_health("test_collection")

        assert len(result["payload_indexes"]["extra_indexes"]) == 2

    async def test_validate_index_health_error(self, collections_service, mock_client):
        """Test index health validation error."""
        mock_client.get_collection.side_effect = Exception("Health check failed")

        with pytest.raises(QdrantServiceError, match="Failed to validate index health"):
            await collections_service.validate_index_health("test_collection")

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
        from unittest.mock import PropertyMock

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

    async def test_generate_comprehensive_recommendations_healthy(
        self, collections_service
    ):
        """Test recommendation generation for healthy state."""
        hnsw_health = {"recommendations": []}
        recommendations = collections_service._generate_comprehensive_recommendations(
            [], [], "healthy", hnsw_health
        )

        assert any("optimally configured" in rec for rec in recommendations)

    async def test_generate_comprehensive_recommendations_missing(
        self, collections_service
    ):
        """Test recommendation generation with missing indexes."""
        hnsw_health = {"recommendations": []}
        recommendations = collections_service._generate_comprehensive_recommendations(
            ["missing_field"], [], "warning", hnsw_health
        )

        assert any("Create missing indexes" in rec for rec in recommendations)

    async def test_generate_comprehensive_recommendations_extra(
        self, collections_service
    ):
        """Test recommendation generation with extra indexes."""
        hnsw_health = {"recommendations": []}
        recommendations = collections_service._generate_comprehensive_recommendations(
            [], ["extra_field"], "warning", hnsw_health
        )

        assert any("removing unused indexes" in rec for rec in recommendations)

    async def test_generate_comprehensive_recommendations_critical(
        self, collections_service
    ):
        """Test recommendation generation for critical state."""
        hnsw_health = {"recommendations": []}
        recommendations = collections_service._generate_comprehensive_recommendations(
            ["missing"], ["extra"], "critical", hnsw_health
        )

        assert any("Critical:" in rec for rec in recommendations)

    async def test_inheritance_from_base_service(self, collections_service):
        """Test that QdrantCollections inherits from BaseService."""
        from src.services.base import BaseService

        assert isinstance(collections_service, BaseService)

    async def test_not_initialized_check(self, mock_config, mock_client):
        """Test validation when service is not initialized."""
        service = QdrantCollections(mock_config, mock_client)
        service._initialized = False

        from src.services.errors import APIError

        with pytest.raises(APIError, match="not initialized"):
            await service.create_collection("test", 1536)
