"""Tests for Qdrant service."""

from unittest.mock import AsyncMock
from unittest.mock import MagicMock
from unittest.mock import patch

import pytest
from src.services.config import APIConfig
from src.services.qdrant_service import QdrantService


@pytest.fixture
def api_config():
    """Create test API config."""
    return APIConfig(
        qdrant_url="http://localhost:6333",
        qdrant_api_key="test-key",
    )


@pytest.fixture
def qdrant_service(api_config):
    """Create Qdrant service instance."""
    return QdrantService(api_config)


class TestQdrantService:
    """Test Qdrant service functionality."""

    @pytest.mark.asyncio
    async def test_initialize(self, qdrant_service):
        """Test service initialization."""
        with patch("src.services.qdrant_service.AsyncQdrantClient") as mock_client:
            await qdrant_service.initialize()

            assert qdrant_service._initialized
            mock_client.assert_called_once_with(
                url="http://localhost:6333",
                api_key="test-key",
                timeout=30.0,
                prefer_grpc=False,
            )

    @pytest.mark.asyncio
    async def test_cleanup(self, qdrant_service):
        """Test service cleanup."""
        # Initialize first
        with patch("src.services.qdrant_service.AsyncQdrantClient") as mock_client:
            mock_instance = AsyncMock()
            mock_client.return_value = mock_instance

            await qdrant_service.initialize()
            await qdrant_service.cleanup()

            mock_instance.close.assert_called_once()
            assert not qdrant_service._initialized

    @pytest.mark.asyncio
    async def test_create_collection(self, qdrant_service):
        """Test collection creation."""
        with patch("src.services.qdrant_service.AsyncQdrantClient") as mock_client:
            mock_instance = AsyncMock()
            mock_client.return_value = mock_instance

            # Mock get_collections to return empty list
            mock_collections = MagicMock()
            mock_collections.collections = []
            mock_instance.get_collections.return_value = mock_collections

            await qdrant_service.initialize()

            result = await qdrant_service.create_collection(
                collection_name="test_collection",
                vector_size=384,
                distance="Cosine",
                enable_quantization=True,
            )

            assert result is True
            mock_instance.create_collection.assert_called_once()

    @pytest.mark.asyncio
    async def test_create_collection_already_exists(self, qdrant_service):
        """Test collection creation when already exists."""
        with patch("src.services.qdrant_service.AsyncQdrantClient") as mock_client:
            mock_instance = AsyncMock()
            mock_client.return_value = mock_instance

            # Mock get_collections to return existing collection
            mock_collection = MagicMock()
            mock_collection.name = "test_collection"
            mock_collections = MagicMock()
            mock_collections.collections = [mock_collection]
            mock_instance.get_collections.return_value = mock_collections

            await qdrant_service.initialize()

            result = await qdrant_service.create_collection(
                collection_name="test_collection",
                vector_size=384,
            )

            assert result is True
            mock_instance.create_collection.assert_not_called()

    @pytest.mark.asyncio
    async def test_hybrid_search(self, qdrant_service):
        """Test hybrid search functionality."""
        with patch("src.services.qdrant_service.AsyncQdrantClient") as mock_client:
            mock_instance = AsyncMock()
            mock_client.return_value = mock_instance

            # Mock search results
            mock_point = MagicMock()
            mock_point.id = "123"
            mock_point.score = 0.95
            mock_point.payload = {"text": "test"}

            mock_result = MagicMock()
            mock_result.points = [mock_point]
            mock_instance.query_points.return_value = mock_result

            await qdrant_service.initialize()

            results = await qdrant_service.hybrid_search(
                collection_name="test_collection",
                query_vector=[0.1, 0.2, 0.3],
                limit=10,
            )

            assert len(results) == 1
            assert results[0]["id"] == "123"
            assert results[0]["score"] == 0.95
            assert results[0]["payload"] == {"text": "test"}

    @pytest.mark.asyncio
    async def test_hybrid_search_with_sparse(self, qdrant_service):
        """Test hybrid search with sparse vectors."""
        with patch("src.services.qdrant_service.AsyncQdrantClient") as mock_client:
            mock_instance = AsyncMock()
            mock_client.return_value = mock_instance

            # Mock search results
            mock_result = MagicMock()
            mock_result.points = []
            mock_instance.query_points.return_value = mock_result

            await qdrant_service.initialize()

            await qdrant_service.hybrid_search(
                collection_name="test_collection",
                query_vector=[0.1, 0.2, 0.3],
                sparse_vector={1: 0.5, 10: 0.8},
                fusion_type="rrf",
            )

            # Verify prefetch queries were created
            call_args = mock_instance.query_points.call_args
            assert "prefetch" in call_args.kwargs
            assert len(call_args.kwargs["prefetch"]) == 2

    @pytest.mark.asyncio
    async def test_upsert_points(self, qdrant_service):
        """Test point upsert functionality."""
        with patch("src.services.qdrant_service.AsyncQdrantClient") as mock_client:
            mock_instance = AsyncMock()
            mock_client.return_value = mock_instance

            await qdrant_service.initialize()

            points = [
                {
                    "id": "1",
                    "vector": [0.1, 0.2, 0.3],
                    "payload": {"text": "test1"},
                },
                {
                    "id": "2",
                    "vector": {"dense": [0.4, 0.5, 0.6]},
                    "payload": {"text": "test2"},
                },
            ]

            result = await qdrant_service.upsert_points(
                collection_name="test_collection",
                points=points,
                batch_size=1,
            )

            assert result is True
            assert mock_instance.upsert.call_count == 2

    @pytest.mark.asyncio
    async def test_service_not_initialized(self, qdrant_service):
        """Test error when service not initialized."""
        from src.services.errors import APIError

        with pytest.raises(APIError, match="not initialized"):
            await qdrant_service.hybrid_search(
                collection_name="test",
                query_vector=[0.1, 0.2],
            )

    @pytest.mark.asyncio
    async def test_delete_collection(self, qdrant_service):
        """Test collection deletion."""
        with patch("src.services.qdrant_service.AsyncQdrantClient") as mock_client:
            mock_instance = AsyncMock()
            mock_client.return_value = mock_instance

            await qdrant_service.initialize()

            result = await qdrant_service.delete_collection("test_collection")

            assert result is True
            mock_instance.delete_collection.assert_called_once_with("test_collection")

    @pytest.mark.asyncio
    async def test_get_collection_info(self, qdrant_service):
        """Test getting collection information."""
        with patch("src.services.qdrant_service.AsyncQdrantClient") as mock_client:
            mock_instance = AsyncMock()
            mock_client.return_value = mock_instance

            # Mock collection info
            mock_info = MagicMock()
            mock_info.status = "green"
            mock_info.vectors_count = 1000
            mock_info.points_count = 1000
            mock_info.config = MagicMock()
            mock_info.config.model_dump.return_value = {"params": "test"}

            mock_instance.get_collection.return_value = mock_info

            await qdrant_service.initialize()

            info = await qdrant_service.get_collection_info("test_collection")

            assert info["status"] == "green"
            assert info["vectors_count"] == 1000
            assert info["points_count"] == 1000
            assert info["config"] == {"params": "test"}

    @pytest.mark.asyncio
    async def test_count_points(self, qdrant_service):
        """Test counting points in collection."""
        with patch("src.services.qdrant_service.AsyncQdrantClient") as mock_client:
            mock_instance = AsyncMock()
            mock_client.return_value = mock_instance

            # Mock count result
            mock_result = MagicMock()
            mock_result.count = 500
            mock_instance.count.return_value = mock_result

            await qdrant_service.initialize()

            count = await qdrant_service.count_points("test_collection")

            assert count == 500
            mock_instance.count.assert_called_once_with(
                collection_name="test_collection",
                exact=True,
            )
