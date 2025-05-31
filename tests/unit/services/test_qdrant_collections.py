"""Tests for QdrantCollections service."""

from unittest.mock import AsyncMock
from unittest.mock import MagicMock

import pytest
from qdrant_client import models
from src.config import UnifiedConfig
from src.services.errors import QdrantServiceError
from src.services.vector_db.collections import QdrantCollections


class TestQdrantCollections:
    """Test QdrantCollections service."""

    @pytest.fixture
    def config(self):
        """Create test configuration."""
        return UnifiedConfig()

    @pytest.fixture
    def mock_client(self):
        """Create mock Qdrant client."""
        return AsyncMock()

    @pytest.fixture
    async def collections_service(self, config, mock_client):
        """Create collections service."""
        service = QdrantCollections(config, mock_client)
        await service.initialize()
        return service

    async def test_service_initialization(self, config, mock_client):
        """Test service can be initialized."""
        service = QdrantCollections(config, mock_client)
        assert service.config == config
        assert service._client == mock_client
        assert service._initialized is True

        await service.initialize()
        assert service._initialized is True

    async def test_create_collection_basic(self, collections_service, mock_client):
        """Test basic collection creation."""
        # Mock empty collections list
        mock_collections_response = MagicMock()
        mock_collections_response.collections = []
        mock_client.get_collections.return_value = mock_collections_response

        # Mock successful creation
        mock_client.create_collection.return_value = None
        mock_client.create_payload_index.return_value = None

        result = await collections_service.create_collection(
            collection_name="test_collection", vector_size=1536, distance="Cosine"
        )

        assert result is True
        mock_client.create_collection.assert_called_once()

        # Verify HNSW configuration was applied
        call_args = mock_client.create_collection.call_args
        assert "vectors_config" in call_args.kwargs
        assert "dense" in call_args.kwargs["vectors_config"]

    async def test_create_collection_already_exists(
        self, collections_service, mock_client
    ):
        """Test collection creation when collection already exists."""
        # Mock existing collection
        mock_collection = MagicMock()
        mock_collection.name = "test_collection"
        mock_collections_response = MagicMock()
        mock_collections_response.collections = [mock_collection]
        mock_client.get_collections.return_value = mock_collections_response

        result = await collections_service.create_collection(
            collection_name="test_collection", vector_size=1536
        )

        assert result is True
        mock_client.create_collection.assert_not_called()

    async def test_delete_collection(self, collections_service, mock_client):
        """Test collection deletion."""
        mock_client.delete_collection.return_value = None

        result = await collections_service.delete_collection("test_collection")

        assert result is True
        mock_client.delete_collection.assert_called_once_with("test_collection")

    async def test_list_collections(self, collections_service, mock_client):
        """Test listing collections."""
        # Mock collections response
        mock_collection1 = MagicMock()
        mock_collection1.name = "collection1"
        mock_collection2 = MagicMock()
        mock_collection2.name = "collection2"

        mock_response = MagicMock()
        mock_response.collections = [mock_collection1, mock_collection2]
        mock_client.get_collections.return_value = mock_response

        result = await collections_service.list_collections()

        assert result == ["collection1", "collection2"]

    async def test_get_collection_info(self, collections_service, mock_client):
        """Test getting collection information."""
        # Mock collection info response
        mock_info = MagicMock()
        mock_info.status = "green"
        mock_info.vectors_count = 1000
        mock_info.points_count = 1000
        mock_info.config = MagicMock()
        mock_info.config.model_dump.return_value = {"test": "config"}

        mock_client.get_collection.return_value = mock_info

        result = await collections_service.get_collection_info("test_collection")

        assert result["status"] == "green"
        assert result["vectors_count"] == 1000
        assert result["points_count"] == 1000
        assert result["config"] == {"test": "config"}

    async def test_create_payload_indexes(self, collections_service, mock_client):
        """Test payload index creation."""
        mock_client.create_payload_index.return_value = None

        await collections_service.create_payload_indexes("test_collection")

        # Verify multiple index types were created
        calls = mock_client.create_payload_index.call_args_list
        assert len(calls) > 0

        # Check that different field types were indexed
        field_types = {call.kwargs["field_schema"] for call in calls}
        assert models.PayloadSchemaType.KEYWORD in field_types
        assert models.PayloadSchemaType.TEXT in field_types
        assert models.PayloadSchemaType.INTEGER in field_types

    async def test_list_payload_indexes(self, collections_service, mock_client):
        """Test listing payload indexes."""
        # Mock collection info with payload schema
        mock_field_info = MagicMock()
        mock_field_info.index = True

        mock_info = MagicMock()
        mock_info.payload_schema = {"indexed_field": mock_field_info}
        mock_client.get_collection.return_value = mock_info

        result = await collections_service.list_payload_indexes("test_collection")

        assert "indexed_field" in result

    async def test_hnsw_config_for_collection_type(self, collections_service):
        """Test HNSW configuration for different collection types."""
        api_config = collections_service._get_hnsw_config_for_collection_type(
            "api_reference"
        )
        tutorial_config = collections_service._get_hnsw_config_for_collection_type(
            "tutorials"
        )
        general_config = collections_service._get_hnsw_config_for_collection_type(
            "general"
        )

        # API reference should have higher quality settings
        assert api_config.m >= tutorial_config.m
        assert api_config.ef_construct >= tutorial_config.ef_construct

        # Unknown types should default to general
        unknown_config = collections_service._get_hnsw_config_for_collection_type(
            "unknown"
        )
        assert unknown_config.m == general_config.m

    async def test_collection_type_inference(self, collections_service):
        """Test collection type inference from names."""
        assert collections_service._infer_collection_type("api-docs") == "api_reference"
        assert (
            collections_service._infer_collection_type("tutorial-collection")
            == "tutorials"
        )
        assert collections_service._infer_collection_type("blog-posts") == "blog_posts"
        assert (
            collections_service._infer_collection_type("code-examples")
            == "code_examples"
        )
        assert (
            collections_service._infer_collection_type("random-name") == "general_docs"
        )

    async def test_validate_index_health(self, collections_service, mock_client):
        """Test index health validation."""
        # Mock collection info
        mock_info = MagicMock()
        mock_info.points_count = 1000
        mock_info.config = MagicMock()
        mock_client.get_collection.return_value = mock_info

        # Mock payload schema with some indexed fields
        mock_field_info = MagicMock()
        mock_field_info.index = True
        mock_info.payload_schema = {
            "doc_type": mock_field_info,
            "language": mock_field_info,
        }

        result = await collections_service.validate_index_health("test_collection")

        assert "collection_name" in result
        assert "status" in result
        assert "health_score" in result
        assert "payload_indexes" in result
        assert "hnsw_configuration" in result
        assert "recommendations" in result

    async def test_error_handling(self, collections_service, mock_client):
        """Test error handling."""
        mock_client.get_collections.side_effect = Exception("Connection error")

        with pytest.raises(QdrantServiceError):
            await collections_service.list_collections()

    async def test_cleanup(self, collections_service):
        """Test service cleanup."""
        await collections_service.cleanup()
        assert collections_service._initialized is False
