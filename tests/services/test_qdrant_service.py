"""Tests for Qdrant service."""

from unittest.mock import AsyncMock
from unittest.mock import MagicMock
from unittest.mock import patch

import pytest
from src.config.models import UnifiedConfig
from src.services.qdrant_service import QdrantService


@pytest.fixture
def config():
    """Create test configuration."""
    # When using nested config objects directly
    from src.config.models import QdrantConfig

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


@pytest.fixture
def mock_qdrant_client():
    """Create a properly mocked Qdrant client."""
    with patch("src.services.qdrant_service.AsyncQdrantClient") as mock_client:
        mock_instance = AsyncMock()
        # Mock connection check
        mock_instance.get_collections = AsyncMock(
            return_value=MagicMock(collections=[])
        )
        # Mock close method
        mock_instance.close = AsyncMock()
        mock_client.return_value = mock_instance
        yield mock_client, mock_instance


class TestQdrantService:
    """Test Qdrant service functionality."""

    @pytest.mark.asyncio
    async def test_initialize(self, qdrant_service):
        """Test service initialization."""
        with patch("src.services.qdrant_service.AsyncQdrantClient") as mock_client:
            # Create a mock instance with async get_collections method
            mock_instance = MagicMock()
            mock_instance.get_collections = AsyncMock(
                return_value=MagicMock(collections=[])
            )
            mock_client.return_value = mock_instance

            await qdrant_service.initialize()

            assert qdrant_service._initialized
            assert qdrant_service._client is not None
            mock_client.assert_called_once_with(
                url="http://localhost:6333",
                api_key="test-key",
                timeout=30.0,
                prefer_grpc=False,
            )

    @pytest.mark.asyncio
    async def test_create_collection_success(self, qdrant_service):
        """Test successful collection creation."""
        with patch("src.services.qdrant_service.AsyncQdrantClient") as mock_client:
            mock_instance = AsyncMock()
            mock_instance.get_collections.return_value = MagicMock(collections=[])
            mock_instance.create_collection = AsyncMock()
            mock_client.return_value = mock_instance

            await qdrant_service.initialize()

            # Create collection
            await qdrant_service.create_collection(
                collection_name="test_collection",
                vector_size=1536,
                distance="cosine",
            )

            mock_instance.create_collection.assert_called_once()
            call_args = mock_instance.create_collection.call_args[1]
            assert call_args["collection_name"] == "test_collection"

    @pytest.mark.asyncio
    async def test_delete_collection(self, qdrant_service):
        """Test collection deletion."""
        with patch("src.services.qdrant_service.AsyncQdrantClient") as mock_client:
            mock_instance = AsyncMock()
            mock_instance.get_collections.return_value = MagicMock(collections=[])
            mock_instance.delete_collection = AsyncMock()
            mock_client.return_value = mock_instance

            await qdrant_service.initialize()

            await qdrant_service.delete_collection("test_collection")

            mock_instance.delete_collection.assert_called_once_with(
                collection_name="test_collection"
            )

    @pytest.mark.asyncio
    async def test_list_collections(self, qdrant_service):
        """Test listing collections."""
        with patch("src.services.qdrant_service.AsyncQdrantClient") as mock_client:
            mock_instance = AsyncMock()
            # Mock collections list
            mock_collection = MagicMock()
            mock_collection.name = "existing_collection"
            mock_instance.get_collections.return_value = MagicMock(
                collections=[mock_collection]
            )
            mock_client.return_value = mock_instance

            await qdrant_service.initialize()

            # List collections
            collections = await qdrant_service.list_collections()
            assert "existing_collection" in collections

    @pytest.mark.asyncio
    async def test_upsert_points(self, qdrant_service):
        """Test document upsertion."""
        with patch("src.services.qdrant_service.AsyncQdrantClient") as mock_client:
            mock_instance = AsyncMock()
            mock_instance.get_collections.return_value = MagicMock(collections=[])
            mock_instance.upsert = AsyncMock()
            mock_client.return_value = mock_instance

            await qdrant_service.initialize()

            # Prepare documents
            documents = [
                {
                    "id": "doc1",
                    "content": "Test content 1",
                    "embedding": [0.1] * 1536,
                    "metadata": {"title": "Doc 1"},
                },
                {
                    "id": "doc2",
                    "content": "Test content 2",
                    "embedding": [0.2] * 1536,
                    "metadata": {"title": "Doc 2"},
                },
            ]

            await qdrant_service.upsert_points(
                collection_name="test_collection",
                points=documents,
            )

            mock_instance.upsert.assert_called_once()
            call_args = mock_instance.upsert.call_args[1]
            assert call_args["collection_name"] == "test_collection"
            assert len(call_args["points"]) == 2

    @pytest.mark.asyncio
    async def test_search(self, qdrant_service):
        """Test vector search."""
        with patch("src.services.qdrant_service.AsyncQdrantClient") as mock_client:
            mock_instance = AsyncMock()
            mock_instance.get_collections.return_value = MagicMock(collections=[])

            # Mock search results
            mock_result = MagicMock()
            mock_result.id = "doc1"
            mock_result.score = 0.95
            mock_result.payload = {
                "content": "Test content",
                "metadata": {"title": "Test"},
            }
            mock_instance.query_points.return_value = MagicMock(points=[mock_result])

            mock_client.return_value = mock_instance

            await qdrant_service.initialize()

            query_vector = [0.1] * 1536
            results = await qdrant_service.hybrid_search(
                collection_name="test_collection",
                query_vector=query_vector,
                limit=5,
            )

            assert len(results) == 1
            assert results[0].id == "doc1"
            assert results[0].score == 0.95
            mock_instance.query_points.assert_called_once()

    @pytest.mark.asyncio
    async def test_search_with_filter(self, qdrant_service):
        """Test vector search with metadata filter."""
        with patch("src.services.qdrant_service.AsyncQdrantClient") as mock_client:
            mock_instance = AsyncMock()
            mock_instance.get_collections.return_value = MagicMock(collections=[])
            mock_instance.query_points.return_value = MagicMock(points=[])
            mock_client.return_value = mock_instance

            await qdrant_service.initialize()

            query_vector = [0.1] * 1536
            filters = {"metadata.category": "documentation"}

            await qdrant_service.hybrid_search(
                collection_name="test_collection",
                query_vector=query_vector,
                filters=filters,
                limit=5,
            )

            mock_instance.search.assert_called_once()
            call_args = mock_instance.search.call_args[1]
            assert call_args["query_filter"] is not None

    @pytest.mark.asyncio
    async def test_get_collection_info(self, qdrant_service):
        """Test getting collection information."""
        with patch("src.services.qdrant_service.AsyncQdrantClient") as mock_client:
            mock_instance = AsyncMock()
            mock_instance.get_collections.return_value = MagicMock(collections=[])

            # Mock collection info
            mock_info = MagicMock()
            mock_info.points_count = 100
            mock_info.indexed_vectors_count = 100
            mock_info.status = "green"
            mock_instance.get_collection.return_value = mock_info

            mock_client.return_value = mock_instance

            await qdrant_service.initialize()

            info = await qdrant_service.get_collection_info("test_collection")

            assert info.points_count == 100
            assert info.status == "green"

    @pytest.mark.asyncio
    async def test_count_points(self, qdrant_service):
        """Test counting points in collection."""
        with patch("src.services.qdrant_service.AsyncQdrantClient") as mock_client:
            mock_instance = AsyncMock()
            mock_instance.get_collections.return_value = MagicMock(collections=[])
            mock_instance.count.return_value = MagicMock(count=100)
            mock_client.return_value = mock_instance

            await qdrant_service.initialize()

            count = await qdrant_service.count_points(
                collection_name="test_collection"
            )

            assert count == 100
            mock_instance.count.assert_called_once()

    @pytest.mark.asyncio
    async def test_hybrid_search(self, qdrant_service):
        """Test hybrid search with dense and sparse vectors."""
        with patch("src.services.qdrant_service.AsyncQdrantClient") as mock_client:
            mock_instance = AsyncMock()
            mock_instance.get_collections.return_value = MagicMock(collections=[])

            # Mock collection with sparse vectors
            mock_collection = MagicMock()
            mock_collection.config.params.sparse_vectors = {
                "text-sparse": MagicMock(index=MagicMock(on_disk=False))
            }
            mock_instance.get_collection.return_value = mock_collection

            # Mock search batch results
            mock_result = MagicMock()
            mock_result.id = "doc1"
            mock_result.score = 0.9
            mock_result.payload = {"content": "Test"}
            mock_instance.search_batch.return_value = [[mock_result]]

            mock_client.return_value = mock_instance

            await qdrant_service.initialize()

            results = await qdrant_service.hybrid_search(
                collection_name="test_collection",
                query_vector=[0.1] * 1536,
                sparse_vector={"indices": [1, 2, 3], "values": [0.5, 0.3, 0.2]},
                limit=5,
            )

            assert len(results) > 0
            mock_instance.search_batch.assert_called()

    @pytest.mark.asyncio
    async def test_collection_info_method(self, qdrant_service):
        """Test getting collection info method."""
        with patch("src.services.qdrant_service.AsyncQdrantClient") as mock_client:
            mock_instance = AsyncMock()
            mock_instance.get_collections.return_value = MagicMock(collections=[])
            mock_info = MagicMock()
            mock_info.points_count = 100
            mock_info.status = "green"
            mock_instance.get_collection.return_value = mock_info
            mock_client.return_value = mock_instance

            await qdrant_service.initialize()

            info = await qdrant_service.get_collection_info("test_collection")

            assert info["points_count"] == 100
            assert info["status"] == "green"
            mock_instance.get_collection.assert_called_once()

    @pytest.mark.asyncio
    async def test_cleanup(self, qdrant_service):
        """Test service cleanup."""
        with patch("src.services.qdrant_service.AsyncQdrantClient") as mock_client:
            mock_instance = AsyncMock()
            mock_instance.get_collections.return_value = MagicMock(collections=[])
            mock_instance.close = AsyncMock()
            mock_client.return_value = mock_instance

            await qdrant_service.initialize()
            assert qdrant_service._initialized

            await qdrant_service.cleanup()

            assert not qdrant_service._initialized
            assert qdrant_service._client is None
            mock_instance.close.assert_called_once()

    @pytest.mark.asyncio
    async def test_error_handling(self, qdrant_service):
        """Test error handling in operations."""
        with patch("src.services.qdrant_service.AsyncQdrantClient") as mock_client:
            mock_instance = AsyncMock()
            mock_instance.get_collections.return_value = MagicMock(collections=[])

            # Mock search error
            mock_instance.query_points.side_effect = Exception("Search failed")
            mock_client.return_value = mock_instance

            await qdrant_service.initialize()

            # Search should raise an error
            with pytest.raises(Exception, match="Search failed"):
                await qdrant_service.hybrid_search(
                    collection_name="test_collection",
                    query_vector=[0.1] * 1536,
                    limit=5,
                )

    @pytest.mark.asyncio
    async def test_batch_operations(self, qdrant_service):
        """Test batch operations."""
        with patch("src.services.qdrant_service.AsyncQdrantClient") as mock_client:
            mock_instance = AsyncMock()
            mock_instance.get_collections.return_value = MagicMock(collections=[])
            mock_instance.upsert = AsyncMock()
            mock_client.return_value = mock_instance

            await qdrant_service.initialize()

            # Create large batch of documents
            documents = [
                {
                    "id": f"doc_{i}",
                    "content": f"Content {i}",
                    "embedding": [0.1] * 1536,
                    "metadata": {"index": i},
                }
                for i in range(150)  # More than typical batch size
            ]

            await qdrant_service.upsert_points(
                collection_name="test_collection",
                points=documents,
            )

            # Should be called multiple times for batching
            assert mock_instance.upsert.call_count >= 2  # At least 2 batches
