"""Tests for QdrantIndexing service."""

from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from qdrant_client import AsyncQdrantClient, models

from src.config import Config
from src.services.errors import QdrantServiceError
from src.services.vector_db.indexing import QdrantIndexing


class TestQdrantIndexing:
    """Test cases for QdrantIndexing service."""

    @pytest.fixture
    def mock_config(self):
        """Create mock configuration."""
        return MagicMock(spec=Config)

    @pytest.fixture
    def mock_client(self):
        """Create mock AsyncQdrantClient."""
        return AsyncMock(spec=AsyncQdrantClient)

    @pytest.fixture
    def indexing_service(self, mock_client, mock_config):
        """Create QdrantIndexing instance."""
        return QdrantIndexing(mock_client, mock_config)

    async def test_create_payload_indexes_success(self, indexing_service, mock_client):
        """Test successful payload index creation."""
        await indexing_service.create_payload_indexes("test_collection")

        # Should create multiple indexes (keyword + text + integer fields)
        assert mock_client.create_payload_index.call_count > 20

        # Verify keyword indexes were created
        keyword_calls = [
            call
            for call in mock_client.create_payload_index.call_args_list
            if call.kwargs.get("field_schema") == models.PayloadSchemaType.KEYWORD
        ]
        assert len(keyword_calls) >= 9  # doc_type, language, framework, etc.

        # Verify text indexes were created
        text_calls = [
            call
            for call in mock_client.create_payload_index.call_args_list
            if call.kwargs.get("field_schema") == models.PayloadSchemaType.TEXT
        ]
        assert len(text_calls) >= 2  # title, content_preview

        # Verify integer indexes were created
        integer_calls = [
            call
            for call in mock_client.create_payload_index.call_args_list
            if call.kwargs.get("field_schema") == models.PayloadSchemaType.INTEGER
        ]
        assert len(integer_calls) >= 10  # created_at, last_updated, word_count, etc.

    async def test_create_payload_indexes_error(self, indexing_service, mock_client):
        """Test payload index creation error."""
        mock_client.create_payload_index.side_effect = Exception(
            "Index creation failed"
        )

        with pytest.raises(
            QdrantServiceError, match="Failed to create payload indexes"
        ):
            await indexing_service.create_payload_indexes("test_collection")

    async def test_list_payload_indexes_success(self, indexing_service, mock_client):
        """Test successful payload index listing."""
        # Mock collection info with payload schema
        mock_field_info1 = MagicMock()
        mock_field_info1.index = True

        mock_field_info2 = MagicMock()
        mock_field_info2.index = False

        mock_field_info3 = MagicMock()
        mock_field_info3.index = True

        mock_collection_info = MagicMock()
        mock_collection_info.payload_schema = {
            "indexed_field1": mock_field_info1,
            "non_indexed_field": mock_field_info2,
            "indexed_field2": mock_field_info3,
        }
        mock_client.get_collection.return_value = mock_collection_info

        result = await indexing_service.list_payload_indexes("test_collection")

        assert "indexed_field1" in result
        assert "indexed_field2" in result
        assert "non_indexed_field" not in result
        assert len(result) == 2

    async def test_list_payload_indexes_no_schema(self, indexing_service, mock_client):
        """Test payload index listing with no schema."""
        mock_collection_info = MagicMock()
        mock_collection_info.payload_schema = None
        mock_client.get_collection.return_value = mock_collection_info

        result = await indexing_service.list_payload_indexes("test_collection")

        assert result == []

    async def test_list_payload_indexes_no_index_attribute(
        self, indexing_service, mock_client
    ):
        """Test payload index listing with fields without index attribute."""
        mock_field_info = MagicMock()
        del mock_field_info.index  # Remove index attribute

        mock_collection_info = MagicMock()
        mock_collection_info.payload_schema = {"field_without_index": mock_field_info}
        mock_client.get_collection.return_value = mock_collection_info

        result = await indexing_service.list_payload_indexes("test_collection")

        assert result == []

    async def test_list_payload_indexes_error(self, indexing_service, mock_client):
        """Test payload index listing error."""
        mock_client.get_collection.side_effect = Exception("Get collection failed")

        with pytest.raises(QdrantServiceError, match="Failed to list payload indexes"):
            await indexing_service.list_payload_indexes("test_collection")

    async def test_drop_payload_index_success(self, indexing_service, mock_client):
        """Test successful payload index drop."""
        await indexing_service.drop_payload_index("test_collection", "test_field")

        mock_client.delete_payload_index.assert_called_once_with(
            collection_name="test_collection", field_name="test_field", wait=True
        )

    async def test_drop_payload_index_error(self, indexing_service, mock_client):
        """Test payload index drop error."""
        mock_client.delete_payload_index.side_effect = Exception("Drop failed")

        with pytest.raises(QdrantServiceError, match="Failed to drop payload index"):
            await indexing_service.drop_payload_index("test_collection", "test_field")

    async def test_reindex_collection_success(self, indexing_service, mock_client):
        """Test successful collection reindexing."""
        # Mock existing indexes
        existing_indexes = ["old_field1", "old_field2", "old_field3"]

        with (
            patch.object(
                indexing_service, "list_payload_indexes", return_value=existing_indexes
            ) as mock_list,
            patch.object(indexing_service, "drop_payload_index") as mock_drop,
            patch.object(indexing_service, "create_payload_indexes") as mock_create,
        ):
            await indexing_service.reindex_collection("test_collection")

            mock_list.assert_called_once_with("test_collection")
            assert mock_drop.call_count == 3  # Should drop all existing indexes
            mock_create.assert_called_once_with("test_collection")

    async def test_reindex_collection_drop_error(
        self, indexing_service, mock_client, caplog
    ):
        """Test collection reindexing with drop error."""
        existing_indexes = ["field1", "field2"]

        with (
            patch.object(
                indexing_service, "list_payload_indexes", return_value=existing_indexes
            ),
            patch.object(
                indexing_service,
                "drop_payload_index",
                side_effect=Exception("Drop failed"),
            ) as mock_drop,
            patch.object(indexing_service, "create_payload_indexes") as mock_create,
        ):
            await indexing_service.reindex_collection("test_collection")

            # Should continue despite drop errors
            assert mock_drop.call_count == 2
            mock_create.assert_called_once()
            assert "Failed to drop index" in caplog.text

    async def test_reindex_collection_error(self, indexing_service, mock_client):
        """Test collection reindexing error."""
        with (
            patch.object(
                indexing_service,
                "list_payload_indexes",
                side_effect=Exception("List failed"),
            ),
            pytest.raises(QdrantServiceError, match="Failed to reindex collection"),
        ):
            await indexing_service.reindex_collection("test_collection")

    async def test_get_payload_index_stats_success(self, indexing_service, mock_client):
        """Test successful payload index stats retrieval."""
        mock_collection_info = MagicMock()
        mock_collection_info.points_count = 1000

        # Mock payload schema
        mock_field_info1 = MagicMock()
        mock_field_info1.data_type = "string"
        mock_field_info2 = MagicMock()
        mock_field_info2.data_type = "integer"

        mock_collection_info.payload_schema = {
            "field1": mock_field_info1,
            "field2": mock_field_info2,
        }

        mock_client.get_collection.return_value = mock_collection_info

        indexed_fields = ["field1"]

        with patch.object(
            indexing_service, "list_payload_indexes", return_value=indexed_fields
        ):
            result = await indexing_service.get_payload_index_stats("test_collection")

        assert result["collection_name"] == "test_collection"
        assert result["total_points"] == 1000
        assert result["indexed_fields_count"] == 1
        assert result["indexed_fields"] == ["field1"]
        assert "payload_schema" in result
        assert result["payload_schema"]["field1"]["indexed"] is True
        assert result["payload_schema"]["field2"]["indexed"] is False

    async def test_get_payload_index_stats_no_schema(
        self, indexing_service, mock_client
    ):
        """Test payload index stats with no schema."""
        mock_collection_info = MagicMock()
        mock_collection_info.points_count = 500
        mock_collection_info.payload_schema = None

        mock_client.get_collection.return_value = mock_collection_info

        with patch.object(indexing_service, "list_payload_indexes", return_value=[]):
            result = await indexing_service.get_payload_index_stats("test_collection")

        assert result["payload_schema"] == {}

    async def test_get_payload_index_stats_error(self, indexing_service, mock_client):
        """Test payload index stats error."""
        mock_client.get_collection.side_effect = Exception("Stats failed")

        with pytest.raises(
            QdrantServiceError, match="Failed to get payload index stats"
        ):
            await indexing_service.get_payload_index_stats("test_collection")

    async def test_validate_index_health_healthy(self, indexing_service, mock_client):
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

        with patch.object(
            indexing_service, "list_payload_indexes", return_value=expected_indexes
        ):
            result = await indexing_service.validate_index_health("test_collection")

        assert result["status"] == "healthy"
        assert result["health_score"] == 100.0
        assert result["collection_name"] == "test_collection"
        assert result["total_points"] == 1000
        assert len(result["payload_indexes"]["missing_indexes"]) == 0
        assert len(result["payload_indexes"]["extra_indexes"]) == 0

    async def test_validate_index_health_missing_indexes(
        self, indexing_service, mock_client
    ):
        """Test index health validation with missing indexes."""
        mock_collection_info = MagicMock()
        mock_collection_info.points_count = 1000
        mock_client.get_collection.return_value = mock_collection_info

        # Mock missing some indexes
        partial_indexes = ["doc_type", "language", "title"]

        with patch.object(
            indexing_service, "list_payload_indexes", return_value=partial_indexes
        ):
            result = await indexing_service.validate_index_health("test_collection")

        assert result["status"] in ["warning", "critical"]
        assert result["health_score"] < 100.0
        assert len(result["payload_indexes"]["missing_indexes"]) > 0

    async def test_validate_index_health_extra_indexes(
        self, indexing_service, mock_client
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
            indexing_service, "list_payload_indexes", return_value=indexes_with_extra
        ):
            result = await indexing_service.validate_index_health("test_collection")

        assert len(result["payload_indexes"]["extra_indexes"]) == 2
        assert "extra_field1" in result["payload_indexes"]["extra_indexes"]
        assert "extra_field2" in result["payload_indexes"]["extra_indexes"]

    async def test_validate_index_health_critical_status(
        self, indexing_service, mock_client
    ):
        """Test index health validation with critical status."""
        mock_collection_info = MagicMock()
        mock_collection_info.points_count = 1000
        mock_client.get_collection.return_value = mock_collection_info

        # Mock very few indexes present (< 80% health score)
        minimal_indexes = ["doc_type"]

        with patch.object(
            indexing_service, "list_payload_indexes", return_value=minimal_indexes
        ):
            result = await indexing_service.validate_index_health("test_collection")

        assert result["status"] == "critical"
        assert result["health_score"] < 80.0

    async def test_validate_index_health_warning_status(
        self, indexing_service, mock_client
    ):
        """Test index health validation with warning status."""
        mock_collection_info = MagicMock()
        mock_collection_info.points_count = 1000
        mock_client.get_collection.return_value = mock_collection_info

        # Mock moderate number of indexes (80-95% health score)
        # Need 12-13 out of 14 expected fields for 85-92% score
        moderate_indexes = [
            "doc_type",
            "language",
            "framework",
            "version",
            "site_name",
            "embedding_model",
            "title",
            "content_preview",
            "created_at",
            "last_updated",
            "word_count",
            "char_count",
        ]

        with patch.object(
            indexing_service, "list_payload_indexes", return_value=moderate_indexes
        ):
            result = await indexing_service.validate_index_health("test_collection")

        assert result["status"] == "warning"
        assert 80.0 <= result["health_score"] < 95.0

    async def test_validate_index_health_with_timestamp(
        self, indexing_service, mock_client
    ):
        """Test index health validation includes timestamp."""
        mock_collection_info = MagicMock()
        mock_collection_info.points_count = 1000
        mock_client.get_collection.return_value = mock_collection_info

        with (
            patch.object(indexing_service, "list_payload_indexes", return_value=[]),
            patch("time.time", return_value=1234567890),
        ):
            result = await indexing_service.validate_index_health("test_collection")

        assert result["validation_timestamp"] == 1234567890

    async def test_validate_index_health_error(self, indexing_service, mock_client):
        """Test index health validation error."""
        mock_client.get_collection.side_effect = Exception("Health check failed")

        with pytest.raises(QdrantServiceError, match="Failed to validate index health"):
            await indexing_service.validate_index_health("test_collection")

    async def test_get_index_usage_stats_success(self, indexing_service, mock_client):
        """Test successful index usage stats retrieval."""
        mock_collection_info = MagicMock()
        mock_collection_info.points_count = 50000
        mock_client.get_collection.return_value = mock_collection_info

        # Mock mixed types of indexes
        indexed_fields = [
            "doc_type",
            "language",
            "framework",  # keyword
            "title",
            "content_preview",  # text
            "created_at",
            "word_count",  # integer
        ]

        with (
            patch.object(
                indexing_service, "list_payload_indexes", return_value=indexed_fields
            ),
            patch("time.time", return_value=1234567890),
        ):
            result = await indexing_service.get_index_usage_stats("test_collection")

        assert result["collection_name"] == "test_collection"
        assert result["collection_stats"]["total_points"] == 50000
        assert result["collection_stats"]["indexed_fields_count"] == 7

        # Check index categorization
        assert result["index_details"]["keyword_indexes"]["count"] == 3
        assert result["index_details"]["text_indexes"]["count"] == 2
        assert result["index_details"]["integer_indexes"]["count"] == 2

        assert result["generated_at"] == 1234567890

    async def test_get_index_usage_stats_large_collection(
        self, indexing_service, mock_client
    ):
        """Test index usage stats for large collection."""
        mock_collection_info = MagicMock()
        mock_collection_info.points_count = 150000  # Large collection
        mock_client.get_collection.return_value = mock_collection_info

        with patch.object(
            indexing_service, "list_payload_indexes", return_value=["doc_type"]
        ):
            result = await indexing_service.get_index_usage_stats("test_collection")

        assert any(
            "Large collection detected" in suggestion
            for suggestion in result["optimization_suggestions"]
        )

    async def test_get_index_usage_stats_many_indexes(
        self, indexing_service, mock_client
    ):
        """Test index usage stats with many indexes."""
        mock_collection_info = MagicMock()
        mock_collection_info.points_count = 1000
        mock_client.get_collection.return_value = mock_collection_info

        # Mock many indexes
        many_indexes = [f"field_{i}" for i in range(20)]  # 20 indexes (> 15)

        with patch.object(
            indexing_service, "list_payload_indexes", return_value=many_indexes
        ):
            result = await indexing_service.get_index_usage_stats("test_collection")

        assert any(
            "Many indexes detected" in suggestion
            for suggestion in result["optimization_suggestions"]
        )

    async def test_get_index_usage_stats_no_keyword_indexes(
        self, indexing_service, mock_client
    ):
        """Test index usage stats with no keyword indexes."""
        mock_collection_info = MagicMock()
        mock_collection_info.points_count = 1000
        mock_client.get_collection.return_value = mock_collection_info

        # Mock only non-keyword indexes
        non_keyword_indexes = ["title", "created_at"]

        with patch.object(
            indexing_service, "list_payload_indexes", return_value=non_keyword_indexes
        ):
            result = await indexing_service.get_index_usage_stats("test_collection")

        assert any(
            "No keyword indexes found" in suggestion
            for suggestion in result["optimization_suggestions"]
        )

    async def test_get_index_usage_stats_optimal_config(
        self, indexing_service, mock_client
    ):
        """Test index usage stats with optimal configuration."""
        mock_collection_info = MagicMock()
        mock_collection_info.points_count = 1000
        mock_client.get_collection.return_value = mock_collection_info

        # Mock optimal number of indexes with good distribution
        optimal_indexes = ["doc_type", "language", "title", "created_at"]

        with patch.object(
            indexing_service, "list_payload_indexes", return_value=optimal_indexes
        ):
            result = await indexing_service.get_index_usage_stats("test_collection")

        assert any(
            "optimal" in suggestion for suggestion in result["optimization_suggestions"]
        )

    async def test_get_index_usage_stats_error(self, indexing_service, mock_client):
        """Test index usage stats error."""
        mock_client.get_collection.side_effect = Exception("Stats failed")

        with pytest.raises(QdrantServiceError, match="Failed to get index usage stats"):
            await indexing_service.get_index_usage_stats("test_collection")

    async def test_generate_index_recommendations_missing(self, indexing_service):
        """Test recommendation generation with missing indexes."""
        recommendations = indexing_service._generate_index_recommendations(
            missing_indexes=["doc_type", "language"], extra_indexes=[], status="warning"
        )

        assert any("Create missing indexes" in rec for rec in recommendations)
        assert any("doc_type, language" in rec for rec in recommendations)

    async def test_generate_index_recommendations_extra(self, indexing_service):
        """Test recommendation generation with extra indexes."""
        recommendations = indexing_service._generate_index_recommendations(
            missing_indexes=[],
            extra_indexes=["unused_field1", "unused_field2"],
            status="healthy",
        )

        assert any("removing unused indexes" in rec for rec in recommendations)
        assert any("unused_field1, unused_field2" in rec for rec in recommendations)

    async def test_generate_index_recommendations_critical(self, indexing_service):
        """Test recommendation generation for critical status."""
        recommendations = indexing_service._generate_index_recommendations(
            missing_indexes=["doc_type"], extra_indexes=[], status="critical"
        )

        assert any("Critical:" in rec for rec in recommendations)

    async def test_generate_index_recommendations_warning(self, indexing_service):
        """Test recommendation generation for warning status."""
        recommendations = indexing_service._generate_index_recommendations(
            missing_indexes=["doc_type"], extra_indexes=[], status="warning"
        )

        assert any("Warning:" in rec for rec in recommendations)

    async def test_generate_index_recommendations_healthy(self, indexing_service):
        """Test recommendation generation for healthy status."""
        recommendations = indexing_service._generate_index_recommendations(
            missing_indexes=[], extra_indexes=[], status="healthy"
        )

        assert any("All indexes are healthy" in rec for rec in recommendations)

    async def test_initialization_and_config(
        self, indexing_service, mock_client, mock_config
    ):
        """Test service initialization and configuration."""
        assert indexing_service.client is mock_client
        assert indexing_service.config is mock_config

    async def test_create_payload_indexes_field_types(
        self, indexing_service, mock_client
    ):
        """Test that different field types are created with correct schemas."""
        await indexing_service.create_payload_indexes("test_collection")

        # Check specific field types were called
        calls = mock_client.create_payload_index.call_args_list

        # Find keyword field calls
        keyword_calls = [
            call
            for call in calls
            if call.kwargs["field_schema"] == models.PayloadSchemaType.KEYWORD
        ]
        keyword_fields = [call.kwargs["field_name"] for call in keyword_calls]
        assert "doc_type" in keyword_fields
        assert "language" in keyword_fields

        # Find text field calls
        text_calls = [
            call
            for call in calls
            if call.kwargs["field_schema"] == models.PayloadSchemaType.TEXT
        ]
        text_fields = [call.kwargs["field_name"] for call in text_calls]
        assert "title" in text_fields
        assert "content_preview" in text_fields

        # Find integer field calls
        integer_calls = [
            call
            for call in calls
            if call.kwargs["field_schema"] == models.PayloadSchemaType.INTEGER
        ]
        integer_fields = [call.kwargs["field_name"] for call in integer_calls]
        assert "created_at" in integer_fields
        assert "word_count" in integer_fields

    async def test_index_health_calculation_edge_cases(
        self, indexing_service, mock_client
    ):
        """Test index health calculation edge cases."""
        mock_collection_info = MagicMock()
        mock_collection_info.points_count = 0  # Empty collection
        mock_client.get_collection.return_value = mock_collection_info

        # Test with no expected indexes (edge case)
        with patch.object(indexing_service, "list_payload_indexes", return_value=[]):
            result = await indexing_service.validate_index_health("test_collection")

        # Should handle division by zero gracefully
        assert result["health_score"] >= 0
        assert result["total_points"] == 0

    async def test_stats_with_missing_data_type(self, indexing_service, mock_client):
        """Test stats generation with missing data type attributes."""
        mock_collection_info = MagicMock()
        mock_collection_info.points_count = 1000

        # Mock field info without data_type attribute
        mock_field_info = MagicMock()
        del mock_field_info.data_type

        mock_collection_info.payload_schema = {"field_without_type": mock_field_info}
        mock_client.get_collection.return_value = mock_collection_info

        with patch.object(indexing_service, "list_payload_indexes", return_value=[]):
            result = await indexing_service.get_payload_index_stats("test_collection")

        # Should handle missing data_type gracefully
        assert result["payload_schema"]["field_without_type"]["type"] == "unknown"
