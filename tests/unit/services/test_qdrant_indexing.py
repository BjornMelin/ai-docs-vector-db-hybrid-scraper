"""Tests for QdrantIndexing service."""

from unittest.mock import AsyncMock, MagicMock

import pytest
from qdrant_client import models
from src.config import UnifiedConfig
from src.services.errors import QdrantServiceError
from src.services.vector_db.indexing import QdrantIndexing


class TestQdrantIndexing:
    """Test QdrantIndexing service."""

    @pytest.fixture
    def config(self):
        """Create test configuration."""
        return UnifiedConfig()

    @pytest.fixture
    def mock_client(self):
        """Create mock Qdrant client."""
        return AsyncMock()

    @pytest.fixture
    def indexing_service(self, config, mock_client):
        """Create indexing service."""
        return QdrantIndexing(mock_client, config)

    async def test_create_payload_indexes_success(self, indexing_service, mock_client):
        """Test successful payload index creation."""
        mock_client.create_payload_index.return_value = None

        await indexing_service.create_payload_indexes("test_collection")

        # Verify multiple index types were created
        calls = mock_client.create_payload_index.call_args_list
        assert len(calls) > 0

        # Check that different field types were indexed
        field_types = {call.kwargs["field_schema"] for call in calls}
        assert models.PayloadSchemaType.KEYWORD in field_types
        assert models.PayloadSchemaType.TEXT in field_types
        assert models.PayloadSchemaType.INTEGER in field_types

        # Verify specific expected fields
        field_names = {call.kwargs["field_name"] for call in calls}
        expected_keyword_fields = {
            "doc_type", "language", "framework", "version", "crawl_source",
            "site_name", "embedding_model", "embedding_provider"
        }
        expected_text_fields = {"title", "content_preview"}
        expected_integer_fields = {
            "created_at", "last_updated", "word_count", "char_count"
        }

        # Check that core fields are present
        assert expected_keyword_fields.issubset(field_names)
        assert expected_text_fields.issubset(field_names)
        assert expected_integer_fields.issubset(field_names)

    async def test_list_payload_indexes_success(self, indexing_service, mock_client):
        """Test successful payload index listing."""
        # Mock collection info with payload schema
        mock_field_info_indexed = MagicMock()
        mock_field_info_indexed.index = True

        mock_field_info_not_indexed = MagicMock()
        mock_field_info_not_indexed.index = False

        mock_info = MagicMock()
        mock_info.payload_schema = {
            "indexed_field": mock_field_info_indexed,
            "not_indexed_field": mock_field_info_not_indexed,
        }
        mock_client.get_collection.return_value = mock_info

        result = await indexing_service.list_payload_indexes("test_collection")

        assert "indexed_field" in result
        assert "not_indexed_field" not in result

    async def test_list_payload_indexes_no_schema(self, indexing_service, mock_client):
        """Test payload index listing with no schema."""
        mock_info = MagicMock()
        mock_info.payload_schema = None
        mock_client.get_collection.return_value = mock_info

        result = await indexing_service.list_payload_indexes("test_collection")

        assert result == []

    async def test_drop_payload_index_success(self, indexing_service, mock_client):
        """Test successful payload index dropping."""
        mock_client.delete_payload_index.return_value = None

        await indexing_service.drop_payload_index("test_collection", "test_field")

        mock_client.delete_payload_index.assert_called_once_with(
            collection_name="test_collection",
            field_name="test_field",
            wait=True,
        )

    async def test_reindex_collection_success(self, indexing_service, mock_client):
        """Test successful collection reindexing."""
        # Mock existing indexes
        mock_field_info = MagicMock()
        mock_field_info.index = True
        mock_info = MagicMock()
        mock_info.payload_schema = {
            "existing_field": mock_field_info,
        }
        mock_client.get_collection.return_value = mock_info

        mock_client.delete_payload_index.return_value = None
        mock_client.create_payload_index.return_value = None

        await indexing_service.reindex_collection("test_collection")

        # Verify that existing indexes were dropped
        mock_client.delete_payload_index.assert_called_with(
            collection_name="test_collection",
            field_name="existing_field",
            wait=True,
        )

        # Verify that new indexes were created
        create_calls = mock_client.create_payload_index.call_args_list
        assert len(create_calls) > 0

    async def test_get_payload_index_stats_success(self, indexing_service, mock_client):
        """Test successful payload index stats retrieval."""
        # Mock collection info
        mock_info = MagicMock()
        mock_info.points_count = 1000

        mock_field_info = MagicMock()
        mock_field_info.index = True
        mock_field_info.data_type = "keyword"

        mock_info.payload_schema = {
            "doc_type": mock_field_info,
        }
        mock_client.get_collection.return_value = mock_info

        stats = await indexing_service.get_payload_index_stats("test_collection")

        assert stats["collection_name"] == "test_collection"
        assert stats["total_points"] == 1000
        assert stats["indexed_fields_count"] == 1
        assert "doc_type" in stats["indexed_fields"]
        assert stats["payload_schema"]["doc_type"]["indexed"] is True

    async def test_validate_index_health_healthy(self, indexing_service, mock_client):
        """Test index health validation for healthy collection."""
        # Mock collection with all expected indexes
        mock_info = MagicMock()
        mock_info.points_count = 1000

        mock_field_info = MagicMock()
        mock_field_info.index = True

        # Create payload schema with all expected core fields
        expected_fields = [
            "doc_type", "language", "framework", "version", "crawl_source",
            "site_name", "embedding_model", "embedding_provider",
            "title", "content_preview",
            "created_at", "last_updated", "word_count", "char_count",
        ]
        
        mock_info.payload_schema = {
            field: mock_field_info for field in expected_fields
        }
        mock_client.get_collection.return_value = mock_info

        health_report = await indexing_service.validate_index_health("test_collection")

        assert health_report["collection_name"] == "test_collection"
        assert health_report["status"] == "healthy"
        assert health_report["health_score"] >= 95
        assert len(health_report["payload_indexes"]["missing_indexes"]) == 0

    async def test_validate_index_health_critical(self, indexing_service, mock_client):
        """Test index health validation for critical collection."""
        # Mock collection with no indexes
        mock_info = MagicMock()
        mock_info.points_count = 1000
        mock_info.payload_schema = {}
        mock_client.get_collection.return_value = mock_info

        health_report = await indexing_service.validate_index_health("test_collection")

        assert health_report["status"] == "critical"
        assert health_report["health_score"] < 80
        assert len(health_report["payload_indexes"]["missing_indexes"]) > 0

    async def test_validate_index_health_warning(self, indexing_service, mock_client):
        """Test index health validation for warning status."""
        # Mock collection with some indexes
        mock_info = MagicMock()
        mock_info.points_count = 1000

        mock_field_info = MagicMock()
        mock_field_info.index = True

        # Only include some core fields
        partial_fields = ["doc_type", "language", "title", "created_at"]
        mock_info.payload_schema = {
            field: mock_field_info for field in partial_fields
        }
        mock_client.get_collection.return_value = mock_info

        health_report = await indexing_service.validate_index_health("test_collection")

        assert health_report["status"] in ["warning", "critical"]
        assert 0 < len(health_report["payload_indexes"]["missing_indexes"])

    async def test_get_index_usage_stats_success(self, indexing_service, mock_client):
        """Test successful index usage stats retrieval."""
        # Mock collection info
        mock_info = MagicMock()
        mock_info.points_count = 1000

        mock_field_info = MagicMock()
        mock_field_info.index = True

        mock_info.payload_schema = {
            "doc_type": mock_field_info,  # Keyword
            "title": mock_field_info,     # Text
            "created_at": mock_field_info, # Integer
        }
        mock_client.get_collection.return_value = mock_info

        stats = await indexing_service.get_index_usage_stats("test_collection")

        assert stats["collection_name"] == "test_collection"
        assert stats["collection_stats"]["total_points"] == 1000
        assert stats["collection_stats"]["indexed_fields_count"] == 3

        # Verify categorization
        assert stats["index_details"]["keyword_indexes"]["count"] >= 1
        assert stats["index_details"]["text_indexes"]["count"] >= 1
        assert stats["index_details"]["integer_indexes"]["count"] >= 1

    async def test_get_index_usage_stats_large_collection(self, indexing_service, mock_client):
        """Test index usage stats for large collection."""
        mock_info = MagicMock()
        mock_info.points_count = 150000  # Large collection
        mock_info.payload_schema = {}
        mock_client.get_collection.return_value = mock_info

        stats = await indexing_service.get_index_usage_stats("test_collection")

        # Should include optimization suggestion for large collection
        suggestions = stats["optimization_suggestions"]
        assert any("Large collection detected" in suggestion for suggestion in suggestions)

    async def test_get_index_usage_stats_many_indexes(self, indexing_service, mock_client):
        """Test index usage stats with many indexes."""
        mock_info = MagicMock()
        mock_info.points_count = 1000

        mock_field_info = MagicMock()
        mock_field_info.index = True

        # Create many indexed fields
        many_fields = {f"field_{i}": mock_field_info for i in range(20)}
        mock_info.payload_schema = many_fields
        mock_client.get_collection.return_value = mock_info

        stats = await indexing_service.get_index_usage_stats("test_collection")

        # Should include suggestion about too many indexes
        suggestions = stats["optimization_suggestions"]
        assert any("Many indexes detected" in suggestion for suggestion in suggestions)

    async def test_generate_index_recommendations_missing(self, indexing_service):
        """Test recommendation generation for missing indexes."""
        missing_indexes = ["doc_type", "language", "title"]
        extra_indexes = []
        status = "warning"

        recommendations = indexing_service._generate_index_recommendations(
            missing_indexes, extra_indexes, status
        )

        assert any("Create missing indexes" in rec for rec in recommendations)
        assert any("doc_type" in rec for rec in recommendations)

    async def test_generate_index_recommendations_extra(self, indexing_service):
        """Test recommendation generation for extra indexes."""
        missing_indexes = []
        extra_indexes = ["unused_field", "legacy_field"]
        status = "healthy"

        recommendations = indexing_service._generate_index_recommendations(
            missing_indexes, extra_indexes, status
        )

        assert any("removing unused indexes" in rec for rec in recommendations)

    async def test_generate_index_recommendations_critical(self, indexing_service):
        """Test recommendation generation for critical status."""
        missing_indexes = ["doc_type", "language"]
        extra_indexes = []
        status = "critical"

        recommendations = indexing_service._generate_index_recommendations(
            missing_indexes, extra_indexes, status
        )

        assert any("Critical" in rec for rec in recommendations)
        assert any("migration script" in rec for rec in recommendations)

    async def test_generate_index_recommendations_healthy(self, indexing_service):
        """Test recommendation generation for healthy status."""
        missing_indexes = []
        extra_indexes = []
        status = "healthy"

        recommendations = indexing_service._generate_index_recommendations(
            missing_indexes, extra_indexes, status
        )

        assert any("optimally configured" in rec for rec in recommendations)

    async def test_error_handling_create_indexes(self, indexing_service, mock_client):
        """Test error handling in index creation."""
        mock_client.create_payload_index.side_effect = Exception("Index creation failed")

        with pytest.raises(QdrantServiceError):
            await indexing_service.create_payload_indexes("test_collection")

    async def test_error_handling_list_indexes(self, indexing_service, mock_client):
        """Test error handling in index listing."""
        mock_client.get_collection.side_effect = Exception("Collection not found")

        with pytest.raises(QdrantServiceError):
            await indexing_service.list_payload_indexes("nonexistent_collection")

    async def test_error_handling_drop_index(self, indexing_service, mock_client):
        """Test error handling in index dropping."""
        mock_client.delete_payload_index.side_effect = Exception("Index not found")

        with pytest.raises(QdrantServiceError):
            await indexing_service.drop_payload_index("test_collection", "nonexistent_field")

    async def test_error_handling_reindex(self, indexing_service, mock_client):
        """Test error handling in reindexing."""
        mock_client.get_collection.side_effect = Exception("Connection error")

        with pytest.raises(QdrantServiceError):
            await indexing_service.reindex_collection("test_collection")

    async def test_error_handling_stats(self, indexing_service, mock_client):
        """Test error handling in stats retrieval."""
        mock_client.get_collection.side_effect = Exception("Stats unavailable")

        with pytest.raises(QdrantServiceError):
            await indexing_service.get_payload_index_stats("test_collection")

    async def test_error_handling_health_validation(self, indexing_service, mock_client):
        """Test error handling in health validation."""
        mock_client.get_collection.side_effect = Exception("Health check failed")

        with pytest.raises(QdrantServiceError):
            await indexing_service.validate_index_health("test_collection")