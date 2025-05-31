"""Tests for QdrantDocuments service."""

from unittest.mock import AsyncMock
from unittest.mock import MagicMock

import pytest
from qdrant_client import models
from src.config import UnifiedConfig
from src.services.errors import QdrantServiceError
from src.services.vector_db.documents import QdrantDocuments


class TestQdrantDocuments:
    """Test QdrantDocuments service."""

    @pytest.fixture
    def config(self):
        """Create test configuration."""
        return UnifiedConfig()

    @pytest.fixture
    def mock_client(self):
        """Create mock Qdrant client."""
        return AsyncMock()

    @pytest.fixture
    def documents_service(self, config, mock_client):
        """Create documents service."""
        return QdrantDocuments(mock_client, config)

    async def test_upsert_points_success(self, documents_service, mock_client):
        """Test successful point upsertion."""
        mock_client.upsert.return_value = None

        points = [
            {
                "id": "doc-1",
                "vector": [0.1] * 1536,
                "payload": {"title": "Test Document 1"},
            },
            {
                "id": "doc-2",
                "vector": {"dense": [0.2] * 1536},
                "payload": {"title": "Test Document 2"},
            },
        ]

        result = await documents_service.upsert_points(
            collection_name="test_collection",
            points=points,
            batch_size=100,
        )

        assert result is True
        mock_client.upsert.assert_called_once()

        # Verify points were converted to PointStruct
        call_args = mock_client.upsert.call_args
        assert call_args.kwargs["collection_name"] == "test_collection"
        assert len(call_args.kwargs["points"]) == 2

    async def test_upsert_points_batching(self, documents_service, mock_client):
        """Test point upsertion with batching."""
        mock_client.upsert.return_value = None

        # Create 5 points with batch size of 2
        points = [
            {
                "id": f"doc-{i}",
                "vector": [0.1 * i] * 1536,
                "payload": {"title": f"Document {i}"},
            }
            for i in range(5)
        ]

        result = await documents_service.upsert_points(
            collection_name="test_collection",
            points=points,
            batch_size=2,
        )

        assert result is True
        # Should be called 3 times: 2+2+1
        assert mock_client.upsert.call_count == 3

    async def test_get_points_success(self, documents_service, mock_client):
        """Test successful point retrieval."""
        # Mock retrieved points
        mock_point1 = MagicMock()
        mock_point1.id = "doc-1"
        mock_point1.payload = {"title": "Document 1"}
        mock_point1.vector = {"dense": [0.1] * 1536}

        mock_point2 = MagicMock()
        mock_point2.id = "doc-2"
        mock_point2.payload = {"title": "Document 2"}
        mock_point2.vector = {"dense": [0.2] * 1536}

        mock_client.retrieve.return_value = [mock_point1, mock_point2]

        results = await documents_service.get_points(
            collection_name="test_collection",
            point_ids=["doc-1", "doc-2"],
            with_payload=True,
            with_vectors=True,
        )

        assert len(results) == 2
        assert results[0]["id"] == "doc-1"
        assert results[0]["payload"]["title"] == "Document 1"
        assert results[0]["vector"]["dense"] == [0.1] * 1536

        mock_client.retrieve.assert_called_once_with(
            collection_name="test_collection",
            ids=["doc-1", "doc-2"],
            with_payload=True,
            with_vectors=True,
        )

    async def test_get_points_minimal(self, documents_service, mock_client):
        """Test point retrieval with minimal data."""
        mock_point = MagicMock()
        mock_point.id = "doc-1"
        mock_point.payload = None
        mock_point.vector = None

        mock_client.retrieve.return_value = [mock_point]

        results = await documents_service.get_points(
            collection_name="test_collection",
            point_ids=["doc-1"],
            with_payload=False,
            with_vectors=False,
        )

        assert len(results) == 1
        assert results[0]["id"] == "doc-1"
        assert "payload" not in results[0]
        assert "vector" not in results[0]

    async def test_delete_points_by_ids(self, documents_service, mock_client):
        """Test point deletion by IDs."""
        mock_client.delete.return_value = None

        result = await documents_service.delete_points(
            collection_name="test_collection",
            point_ids=["doc-1", "doc-2", "doc-3"],
        )

        assert result is True
        mock_client.delete.assert_called_once()

        call_args = mock_client.delete.call_args
        assert call_args.kwargs["collection_name"] == "test_collection"
        assert isinstance(call_args.kwargs["points_selector"], models.PointIdsList)

    async def test_delete_points_by_filter(self, documents_service, mock_client):
        """Test point deletion by filter."""
        mock_client.delete.return_value = None

        filter_condition = {"doc_type": "temporary"}

        result = await documents_service.delete_points(
            collection_name="test_collection",
            filter_condition=filter_condition,
        )

        assert result is True
        mock_client.delete.assert_called_once()

        call_args = mock_client.delete.call_args
        assert isinstance(call_args.kwargs["points_selector"], models.FilterSelector)

    async def test_delete_points_validation(self, documents_service):
        """Test delete points input validation."""
        with pytest.raises(ValueError, match="Either point_ids or filter_condition"):
            await documents_service.delete_points(collection_name="test_collection")

    async def test_update_point_payload_merge(self, documents_service, mock_client):
        """Test point payload update with merge."""
        mock_client.set_payload.return_value = None

        result = await documents_service.update_point_payload(
            collection_name="test_collection",
            point_id="doc-1",
            payload={"new_field": "new_value"},
            replace=False,
        )

        assert result is True
        mock_client.set_payload.assert_called_once()
        mock_client.overwrite_payload.assert_not_called()

    async def test_update_point_payload_replace(self, documents_service, mock_client):
        """Test point payload update with replacement."""
        mock_client.overwrite_payload.return_value = None

        result = await documents_service.update_point_payload(
            collection_name="test_collection",
            point_id="doc-1",
            payload={"title": "New Title"},
            replace=True,
        )

        assert result is True
        mock_client.overwrite_payload.assert_called_once()
        mock_client.set_payload.assert_not_called()

    async def test_count_points_no_filter(self, documents_service, mock_client):
        """Test point counting without filter."""
        mock_result = MagicMock()
        mock_result.count = 1000
        mock_client.count.return_value = mock_result

        count = await documents_service.count_points(
            collection_name="test_collection",
            exact=True,
        )

        assert count == 1000
        mock_client.count.assert_called_once()

        call_args = mock_client.count.call_args
        assert call_args.kwargs["collection_name"] == "test_collection"
        assert call_args.kwargs["count_filter"] is None
        assert call_args.kwargs["exact"] is True

    async def test_count_points_with_filter(self, documents_service, mock_client):
        """Test point counting with filter."""
        mock_result = MagicMock()
        mock_result.count = 50
        mock_client.count.return_value = mock_result

        filter_condition = {"doc_type": "api"}

        count = await documents_service.count_points(
            collection_name="test_collection",
            filter_condition=filter_condition,
            exact=False,
        )

        assert count == 50
        call_args = mock_client.count.call_args
        assert call_args.kwargs["count_filter"] is not None

    async def test_scroll_points_success(self, documents_service, mock_client):
        """Test successful point scrolling."""
        mock_point1 = MagicMock()
        mock_point1.id = "doc-1"
        mock_point1.payload = {"title": "Document 1"}
        mock_point1.vector = None

        mock_point2 = MagicMock()
        mock_point2.id = "doc-2"
        mock_point2.payload = {"title": "Document 2"}
        mock_point2.vector = None

        # Mock scroll returns tuple (points, next_offset)
        mock_client.scroll.return_value = ([mock_point1, mock_point2], "next_token")

        result = await documents_service.scroll_points(
            collection_name="test_collection",
            limit=100,
            offset="start_token",
            with_payload=True,
            with_vectors=False,
        )

        assert len(result["points"]) == 2
        assert result["points"][0]["id"] == "doc-1"
        assert result["points"][1]["id"] == "doc-2"
        assert result["next_offset"] == "next_token"

        mock_client.scroll.assert_called_once()

    async def test_scroll_points_with_filter(self, documents_service, mock_client):
        """Test point scrolling with filter."""
        mock_client.scroll.return_value = ([], None)

        filter_condition = {"doc_type": "tutorial"}

        result = await documents_service.scroll_points(
            collection_name="test_collection",
            filter_condition=filter_condition,
            limit=50,
        )

        assert result["points"] == []
        assert result["next_offset"] is None

        call_args = mock_client.scroll.call_args
        assert call_args.kwargs["scroll_filter"] is not None

    async def test_clear_collection_success(self, documents_service, mock_client):
        """Test successful collection clearing."""
        mock_client.delete.return_value = None

        result = await documents_service.clear_collection("test_collection")

        assert result is True
        mock_client.delete.assert_called_once()

        call_args = mock_client.delete.call_args
        assert call_args.kwargs["collection_name"] == "test_collection"
        assert isinstance(call_args.kwargs["points_selector"], models.FilterSelector)

    async def test_build_filter_keyword_fields(self, documents_service):
        """Test filter building for keyword fields."""
        filters = {
            "doc_type": "api",
            "language": "python",
            "framework": "fastapi",
        }

        filter_obj = documents_service._build_filter(filters)

        assert filter_obj is not None
        assert len(filter_obj.must) == 3

        # Check that all conditions are FieldConditions with MatchValue
        for condition in filter_obj.must:
            assert isinstance(condition, models.FieldCondition)
            assert isinstance(condition.match, models.MatchValue)

    async def test_build_filter_text_fields(self, documents_service):
        """Test filter building for text fields."""
        filters = {
            "title": "API documentation",
            "content_preview": "This document describes",
        }

        filter_obj = documents_service._build_filter(filters)

        assert filter_obj is not None
        assert len(filter_obj.must) == 2

        # Check that text conditions use MatchText
        for condition in filter_obj.must:
            assert isinstance(condition, models.FieldCondition)
            assert isinstance(condition.match, models.MatchText)

    async def test_build_filter_range_fields(self, documents_service):
        """Test filter building for range fields."""
        filters = {
            "created_after": 1640995200,  # Jan 1, 2022
            "min_word_count": 100,
            "max_word_count": 1000,
        }

        filter_obj = documents_service._build_filter(filters)

        assert filter_obj is not None
        assert len(filter_obj.must) == 3

        # Check that range conditions use Range
        for condition in filter_obj.must:
            assert isinstance(condition, models.FieldCondition)
            assert isinstance(condition.range, models.Range)

    async def test_build_filter_structural_fields(self, documents_service):
        """Test filter building for structural fields."""
        filters = {
            "chunk_index": 0,
            "depth": 2,
        }

        filter_obj = documents_service._build_filter(filters)

        assert filter_obj is not None
        assert len(filter_obj.must) == 2

    async def test_build_filter_validation(self, documents_service):
        """Test filter building validation."""
        # Invalid keyword field type
        with pytest.raises(
            ValueError, match="Filter value for .* must be a simple type"
        ):
            documents_service._build_filter({"doc_type": ["invalid", "list"]})

        # Invalid text field type
        with pytest.raises(
            ValueError, match="Text filter value for .* must be a string"
        ):
            documents_service._build_filter({"title": 123})

    async def test_build_filter_empty(self, documents_service):
        """Test filter building with empty input."""
        assert documents_service._build_filter(None) is None
        assert documents_service._build_filter({}) is None

    async def test_error_handling_upsert(self, documents_service, mock_client):
        """Test error handling in upsert operations."""
        mock_client.upsert.side_effect = Exception("Collection not found")

        with pytest.raises(QdrantServiceError, match="Collection .* not found"):
            await documents_service.upsert_points(
                "nonexistent",
                [{"id": "test", "vector": [0.1] * 1536}],
            )

    async def test_error_handling_vector_size(self, documents_service, mock_client):
        """Test vector size error handling."""
        mock_client.upsert.side_effect = Exception("wrong vector size")

        with pytest.raises(QdrantServiceError, match="Vector dimension mismatch"):
            await documents_service.upsert_points(
                "test_collection",
                [{"id": "test", "vector": [0.1] * 100}],  # Wrong size
            )

    async def test_error_handling_payload_size(self, documents_service, mock_client):
        """Test payload size error handling."""
        mock_client.upsert.side_effect = Exception("payload too large")

        with pytest.raises(QdrantServiceError, match="Payload too large"):
            await documents_service.upsert_points(
                "test_collection",
                [
                    {
                        "id": "test",
                        "vector": [0.1] * 1536,
                        "payload": {"large": "x" * 10000},
                    }
                ],
            )

    async def test_error_handling_generic(self, documents_service, mock_client):
        """Test generic error handling."""
        mock_client.retrieve.side_effect = Exception("Network error")

        with pytest.raises(QdrantServiceError):
            await documents_service.get_points("test_collection", ["doc-1"])
