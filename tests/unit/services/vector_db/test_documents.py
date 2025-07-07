"""Tests for QdrantDocuments service."""

from unittest.mock import AsyncMock, MagicMock

import pytest
from qdrant_client import AsyncQdrantClient, models

from src.config import Config
from src.services.errors import QdrantServiceError
from src.services.vector_db.documents import QdrantDocuments


class TestQdrantDocuments:
    """Test cases for QdrantDocuments service."""

    @pytest.fixture
    def mock_config(self):
        """Create mock configuration."""
        return MagicMock(spec=Config)

    @pytest.fixture
    def mock_client(self):
        """Create mock AsyncQdrantClient."""
        return AsyncMock(spec=AsyncQdrantClient)

    @pytest.fixture
    def documents_service(self, mock_client, mock_config):
        """Create QdrantDocuments instance."""
        return QdrantDocuments(mock_client, mock_config)

    @pytest.fixture
    def sample_points(self):
        """Sample points for testing."""
        return [
            {
                "id": "point1",
                "vector": [0.1, 0.2, 0.3],
                "payload": {"title": "Test Document 1", "category": "test"},
            },
            {
                "id": "point2",
                "vector": {"dense": [0.4, 0.5, 0.6]},
                "payload": {"title": "Test Document 2", "category": "test"},
            },
            {
                "id": "point3",
                "vector": [0.7, 0.8, 0.9],
                "payload": {"title": "Test Document 3", "category": "example"},
            },
        ]

    @pytest.mark.asyncio
    async def test_upsert_points_success(
        self, documents_service, mock_client, sample_points
    ):
        """Test successful point upsert."""
        result = await documents_service.upsert_points(
            collection_name="test_collection", points=sample_points, batch_size=2
        )

        assert result is True
        # Should be called twice due to batch size of 2 with 3 points
        assert mock_client.upsert.call_count == 2

    @pytest.mark.asyncio
    async def test_upsert_points_single_batch(
        self, documents_service, mock_client, sample_points
    ):
        """Test point upsert with single batch."""
        result = await documents_service.upsert_points(
            collection_name="test_collection", points=sample_points, batch_size=100
        )

        assert result is True
        assert mock_client.upsert.call_count == 1

    @pytest.mark.asyncio
    async def test_upsert_points_vector_normalization(
        self, documents_service, mock_client
    ):
        """Test point upsert with vector normalization."""
        points = [
            {
                "id": "point1",
                "vector": [0.1, 0.2, 0.3],  # Should be converted to {"dense": [...]}
                "payload": {"title": "Test"},
            }
        ]

        await documents_service.upsert_points("test_collection", points)

        call_args = mock_client.upsert.call_args
        point_struct = call_args._kwargs["points"][0]
        assert isinstance(point_struct.vector, dict)
        assert "dense" in point_struct.vector

    @pytest.mark.asyncio
    async def test_upsert_points_collection_not_found(
        self, documents_service, mock_client, sample_points
    ):
        """Test point upsert with collection not found error."""
        mock_client.upsert.side_effect = Exception("collection not found")

        with pytest.raises(
            QdrantServiceError, match="Collection 'test_collection' not found"
        ):
            await documents_service.upsert_points("test_collection", sample_points)

    @pytest.mark.asyncio
    async def test_upsert_points_wrong_vector_size(
        self, documents_service, mock_client, sample_points
    ):
        """Test point upsert with wrong vector size error."""
        mock_client.upsert.side_effect = Exception("wrong vector size")

        with pytest.raises(QdrantServiceError, match="Vector dimension mismatch"):
            await documents_service.upsert_points("test_collection", sample_points)

    @pytest.mark.asyncio
    async def test_upsert_points_payload_too_large(
        self, documents_service, mock_client, sample_points
    ):
        """Test point upsert with payload too large error."""
        mock_client.upsert.side_effect = Exception("payload too large")

        with pytest.raises(QdrantServiceError, match="Payload too large"):
            await documents_service.upsert_points("test_collection", sample_points)

    @pytest.mark.asyncio
    async def test_upsert_points_generic_error(
        self, documents_service, mock_client, sample_points
    ):
        """Test point upsert with generic error."""
        mock_client.upsert.side_effect = Exception("Generic error")

        with pytest.raises(QdrantServiceError, match="Failed to upsert points"):
            await documents_service.upsert_points("test_collection", sample_points)

    @pytest.mark.asyncio
    async def test_get_points_success(self, documents_service, mock_client):
        """Test successful point retrieval."""
        mock_point1 = MagicMock()
        mock_point1.id = "point1"
        mock_point1.payload = {"title": "Test 1"}
        mock_point1.vector = {"dense": [0.1, 0.2, 0.3]}

        mock_point2 = MagicMock()
        mock_point2.id = "point2"
        mock_point2.payload = {"title": "Test 2"}
        mock_point2.vector = {"dense": [0.4, 0.5, 0.6]}

        mock_client.retrieve.return_value = [mock_point1, mock_point2]

        result = await documents_service.get_points(
            collection_name="test_collection",
            point_ids=["point1", "point2"],
            with_payload=True,
            with_vectors=True,
        )

        assert len(result) == 2
        assert result[0]["id"] == "point1"
        assert result[0]["payload"]["title"] == "Test 1"
        assert result[0]["vector"] == {"dense": [0.1, 0.2, 0.3]}

    @pytest.mark.asyncio
    async def test_get_points_payload_only(self, documents_service, mock_client):
        """Test point retrieval with payload only."""
        mock_point = MagicMock()
        mock_point.id = "point1"
        mock_point.payload = {"title": "Test 1"}
        mock_point.vector = None

        mock_client.retrieve.return_value = [mock_point]

        result = await documents_service.get_points(
            collection_name="test_collection",
            point_ids=["point1"],
            with_payload=True,
            with_vectors=False,
        )

        assert result[0]["payload"]["title"] == "Test 1"
        assert "vector" not in result[0]

    @pytest.mark.asyncio
    async def test_get_points_no_payload(self, documents_service, mock_client):
        """Test point retrieval without payload."""
        mock_point = MagicMock()
        mock_point.id = "point1"
        mock_point.payload = None
        mock_point.vector = None

        mock_client.retrieve.return_value = [mock_point]

        result = await documents_service.get_points(
            collection_name="test_collection",
            point_ids=["point1"],
            with_payload=False,
            with_vectors=False,
        )

        assert result[0]["id"] == "point1"
        assert "payload" not in result[0]
        assert "vector" not in result[0]

    @pytest.mark.asyncio
    async def test_get_points_error(self, documents_service, mock_client):
        """Test point retrieval error."""
        mock_client.retrieve.side_effect = Exception("Retrieve failed")

        with pytest.raises(QdrantServiceError, match="Failed to retrieve points"):
            await documents_service.get_points("test_collection", ["point1"])

    @pytest.mark.asyncio
    async def test_delete_points_by_ids(self, documents_service, mock_client):
        """Test point deletion by IDs."""
        result = await documents_service.delete_points(
            collection_name="test_collection", point_ids=["point1", "point2"]
        )

        assert result is True
        mock_client.delete.assert_called_once()
        call_args = mock_client.delete.call_args
        assert isinstance(call_args._kwargs["points_selector"], models.PointIdsList)

    @pytest.mark.asyncio
    async def test_delete_points_by_filter(self, documents_service, mock_client):
        """Test point deletion by filter."""
        filter_condition = {"doc_type": "test"}  # Use a valid keyword field

        result = await documents_service.delete_points(
            collection_name="test_collection", filter_condition=filter_condition
        )

        assert result is True
        mock_client.delete.assert_called_once()
        call_args = mock_client.delete.call_args
        assert isinstance(call_args._kwargs["points_selector"], models.FilterSelector)

    @pytest.mark.asyncio
    async def test_delete_points_no_criteria(self, documents_service, _mock_client):
        """Test point deletion with no criteria provided."""
        with pytest.raises(
            ValueError, match="Either point_ids or filter_condition must be provided"
        ):
            await documents_service.delete_points("test_collection")

    @pytest.mark.asyncio
    async def test_delete_points_error(self, documents_service, mock_client):
        """Test point deletion error."""
        mock_client.delete.side_effect = Exception("Delete failed")

        with pytest.raises(QdrantServiceError, match="Failed to delete points"):
            await documents_service.delete_points(
                "test_collection", point_ids=["point1"]
            )

    @pytest.mark.asyncio
    async def test_update_point_payload_merge(self, documents_service, mock_client):
        """Test point payload update with merge."""
        result = await documents_service.update_point_payload(
            collection_name="test_collection",
            point_id="point1",
            payload={"new_field": "new_value"},
            replace=False,
        )

        assert result is True
        mock_client.set_payload.assert_called_once()

    @pytest.mark.asyncio
    async def test_update_point_payload_replace(self, documents_service, mock_client):
        """Test point payload update with replace."""
        result = await documents_service.update_point_payload(
            collection_name="test_collection",
            point_id="point1",
            payload={"new_field": "new_value"},
            replace=True,
        )

        assert result is True
        mock_client.overwrite_payload.assert_called_once()

    @pytest.mark.asyncio
    async def test_update_point_payload_error(self, documents_service, mock_client):
        """Test point payload update error."""
        mock_client.set_payload.side_effect = Exception("Update failed")

        with pytest.raises(QdrantServiceError, match="Failed to update point payload"):
            await documents_service.update_point_payload(
                "test_collection", "point1", {"field": "value"}
            )

    @pytest.mark.asyncio
    async def test_count_points_all(self, documents_service, mock_client):
        """Test point counting without filter."""
        mock_result = MagicMock()
        mock_result.count = 1000
        mock_client.count.return_value = mock_result

        result = await documents_service.count_points("test_collection")

        assert result == 1000
        mock_client.count.assert_called_once_with(
            collection_name="test_collection", count_filter=None, exact=True
        )

    @pytest.mark.asyncio
    async def test_count_points_with_filter(self, documents_service, mock_client):
        """Test point counting with filter."""
        mock_result = MagicMock()
        mock_result.count = 500
        mock_client.count.return_value = mock_result

        filter_condition = {"category": "test"}

        result = await documents_service.count_points(
            collection_name="test_collection",
            filter_condition=filter_condition,
            exact=False,
        )

        assert result == 500

    @pytest.mark.asyncio
    async def test_count_points_error(self, documents_service, mock_client):
        """Test point counting error."""
        mock_client.count.side_effect = Exception("Count failed")

        with pytest.raises(QdrantServiceError, match="Failed to count points"):
            await documents_service.count_points("test_collection")

    @pytest.mark.asyncio
    async def test_scroll_points_success(self, documents_service, mock_client):
        """Test successful point scrolling."""
        mock_point1 = MagicMock()
        mock_point1.id = "point1"
        mock_point1.payload = {"title": "Test 1"}
        mock_point1.vector = None

        mock_point2 = MagicMock()
        mock_point2.id = "point2"
        mock_point2.payload = {"title": "Test 2"}
        mock_point2.vector = None

        mock_client.scroll.return_value = (
            [mock_point1, mock_point2],
            "next_offset_123",
        )

        result = await documents_service.scroll_points(
            collection_name="test_collection",
            limit=100,
            offset="offset_123",
            with_payload=True,
            with_vectors=False,
        )

        assert len(result["points"]) == 2
        assert result["next_offset"] == "next_offset_123"
        assert result["points"][0]["id"] == "point1"

    @pytest.mark.asyncio
    async def test_scroll_points_with_filter(self, documents_service, mock_client):
        """Test point scrolling with filter."""
        mock_client.scroll.return_value = ([], None)

        filter_condition = {"doc_type": "test"}  # Use a valid keyword field

        await documents_service.scroll_points(
            collection_name="test_collection", filter_condition=filter_condition
        )

        call_args = mock_client.scroll.call_args
        assert call_args._kwargs["scroll_filter"] is not None

    @pytest.mark.asyncio
    async def test_scroll_points_with_vectors(self, documents_service, mock_client):
        """Test point scrolling with vectors."""
        mock_point = MagicMock()
        mock_point.id = "point1"
        mock_point.payload = {"title": "Test"}
        mock_point.vector = {"dense": [0.1, 0.2, 0.3]}

        mock_client.scroll.return_value = ([mock_point], None)

        result = await documents_service.scroll_points(
            collection_name="test_collection", with_vectors=True
        )

        assert result["points"][0]["vector"] == {"dense": [0.1, 0.2, 0.3]}

    @pytest.mark.asyncio
    async def test_scroll_points_error(self, documents_service, mock_client):
        """Test point scrolling error."""
        mock_client.scroll.side_effect = Exception("Scroll failed")

        with pytest.raises(QdrantServiceError, match="Failed to scroll points"):
            await documents_service.scroll_points("test_collection")

    @pytest.mark.asyncio
    async def test_clear_collection_success(self, documents_service, mock_client):
        """Test successful collection clearing."""
        result = await documents_service.clear_collection("test_collection")

        assert result is True
        mock_client.delete.assert_called_once()
        call_args = mock_client.delete.call_args
        assert isinstance(call_args._kwargs["points_selector"], models.FilterSelector)

    @pytest.mark.asyncio
    async def test_clear_collection_error(self, documents_service, mock_client):
        """Test collection clearing error."""
        mock_client.delete.side_effect = Exception("Clear failed")

        with pytest.raises(QdrantServiceError, match="Failed to clear collection"):
            await documents_service.clear_collection("test_collection")

    @pytest.mark.asyncio
    async def test_initialization_and_config(
        self, documents_service, mock_client, mock_config
    ):
        """Test service initialization and configuration."""
        assert documents_service.client is mock_client
        assert documents_service.config is mock_config

    @pytest.mark.asyncio
    async def test_upsert_points_empty_list(self, documents_service, mock_client):
        """Test point upsert with empty list."""
        result = await documents_service.upsert_points("test_collection", [])

        assert result is True
        mock_client.upsert.assert_not_called()

    @pytest.mark.asyncio
    async def test_get_points_empty_ids(self, documents_service, mock_client):
        """Test point retrieval with empty ID list."""
        mock_client.retrieve.return_value = []

        result = await documents_service.get_points("test_collection", [])

        assert result == []

    @pytest.mark.asyncio
    async def test_point_struct_creation(self, documents_service, mock_client):
        """Test PointStruct creation with various vector formats."""
        points = [
            {
                "id": "point1",
                "vector": [0.1, 0.2, 0.3],  # List format
                "payload": {"test": "value"},
            },
            {
                "id": "point2",
                "vector": {"dense": [0.4, 0.5, 0.6]},  # Dense only dict format
                "payload": {},
            },
            {
                "id": "point3",
                "vector": {"dense": [0.7, 0.8, 0.9]},  # Dense only dict
            },
        ]

        await documents_service.upsert_points("test_collection", points)

        call_args = mock_client.upsert.call_args
        point_structs = call_args._kwargs["points"]

        # First point: list should be converted to {"dense": [...]}
        assert isinstance(point_structs[0].vector, dict)
        assert "dense" in point_structs[0].vector

        # Second point: dict should remain as dict
        assert isinstance(point_structs[1].vector, dict)

        # Third point: payload defaults to empty dict
        assert point_structs[2].payload == {}

    @pytest.mark.asyncio
    async def test_error_handling_consistency(self, documents_service, mock_client):
        """Test consistent error handling across methods."""
        mock_client.upsert.side_effect = Exception("Test error")
        mock_client.retrieve.side_effect = Exception("Test error")
        mock_client.delete.side_effect = Exception("Test error")
        mock_client.count.side_effect = Exception("Test error")
        mock_client.scroll.side_effect = Exception("Test error")

        # All methods should raise QdrantServiceError
        with pytest.raises(QdrantServiceError):
            await documents_service.upsert_points(
                "test", [{"id": "1", "vector": [0.1]}]
            )

        with pytest.raises(QdrantServiceError):
            await documents_service.get_points("test", ["1"])

        with pytest.raises(QdrantServiceError):
            await documents_service.delete_points("test", point_ids=["1"])

        with pytest.raises(QdrantServiceError):
            await documents_service.count_points("test")

        with pytest.raises(QdrantServiceError):
            await documents_service.scroll_points("test")
