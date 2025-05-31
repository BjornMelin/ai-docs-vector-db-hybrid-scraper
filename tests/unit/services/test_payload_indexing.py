"""Test suite for payload indexing functionality (Issue #56)."""

import time
from unittest.mock import AsyncMock
from unittest.mock import MagicMock

import pytest
from qdrant_client import models
from src.config import UnifiedConfig
from src.services.errors import APIError
from src.services.errors import QdrantServiceError
from src.services.vector_db.service import QdrantService


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
def sample_payload_data():
    """Sample payload data for testing indexing."""
    return {
        "content": "Sample documentation content",
        "title": "Sample Title",
        # Core documented fields
        "doc_type": "api",
        "language": "python",
        "framework": "fastapi",
        "version": "0.104.1",
        "crawl_source": "crawl4ai",
        # System fields
        "site_name": "FastAPI Documentation",
        "embedding_model": "text-embedding-3-small",
        "embedding_provider": "openai",
        "search_strategy": "hybrid",
        "scraper_version": "3.0-Advanced",
        # Metrics
        "word_count": 150,
        "char_count": 800,
        "quality_score": 85,
        "chunk_index": 0,
        "total_chunks": 5,
        "depth": 2,
        "links_count": 10,
        # Timestamps
        "created_at": 1640995200,  # Unix timestamp
        "last_updated": 1640995200,
        "scraped_at": 1640995200,
    }


class TestPayloadIndexCreation:
    """Test payload index creation functionality."""

    @pytest.mark.asyncio
    async def test_create_payload_indexes_success(self, qdrant_service):
        """Test successful payload index creation."""
        collection_name = "test_collection"

        # Mock successful index creation
        qdrant_service._client.create_payload_index = AsyncMock()

        # Execute
        await qdrant_service.create_payload_indexes(collection_name)

        # Verify all expected index types were created
        create_calls = qdrant_service._client.create_payload_index.call_args_list

        # Should create 10 keyword + 2 text + 10 integer = 22 indexes
        assert len(create_calls) == 22

        # Check keyword indexes
        keyword_fields = [
            "doc_type",
            "language",
            "framework",
            "version",
            "crawl_source",
            "site_name",
            "embedding_model",
            "embedding_provider",
            "search_strategy",
            "scraper_version",
        ]
        text_fields = ["title", "content_preview"]
        integer_fields = [
            "created_at",
            "last_updated",
            "scraped_at",
            "word_count",
            "char_count",
            "quality_score",
            "chunk_index",
            "total_chunks",
            "depth",
            "links_count",
        ]

        # Verify keyword indexes
        keyword_calls = [
            call
            for call in create_calls
            if call.kwargs["field_schema"] == models.PayloadSchemaType.KEYWORD
        ]
        assert len(keyword_calls) == len(keyword_fields)

        # Verify text indexes
        text_calls = [
            call
            for call in create_calls
            if call.kwargs["field_schema"] == models.PayloadSchemaType.TEXT
        ]
        assert len(text_calls) == len(text_fields)

        # Verify integer indexes
        integer_calls = [
            call
            for call in create_calls
            if call.kwargs["field_schema"] == models.PayloadSchemaType.INTEGER
        ]
        assert len(integer_calls) == len(integer_fields)

        # Verify all calls used wait=True
        for call in create_calls:
            assert call.kwargs["wait"] is True
            assert call.kwargs["collection_name"] == collection_name

    @pytest.mark.asyncio
    async def test_create_payload_indexes_failure(self, qdrant_service):
        """Test payload index creation failure handling."""
        collection_name = "test_collection"

        # Mock failure
        qdrant_service._client.create_payload_index = AsyncMock(
            side_effect=Exception("Index creation failed")
        )

        # Should raise QdrantServiceError
        with pytest.raises(
            QdrantServiceError, match="Failed to create payload indexes"
        ):
            await qdrant_service.create_payload_indexes(collection_name)

    @pytest.mark.asyncio
    async def test_create_payload_indexes_not_initialized(self, qdrant_service):
        """Test behavior when service not initialized."""
        qdrant_service._initialized = False

        with pytest.raises(APIError, match="Service not initialized"):
            await qdrant_service.create_payload_indexes("test_collection")


class TestPayloadIndexManagement:
    """Test payload index management functionality."""

    @pytest.mark.asyncio
    async def test_list_payload_indexes_success(self, qdrant_service):
        """Test successful payload index listing."""
        collection_name = "test_collection"

        # Mock collection info with payload schema
        mock_collection_info = MagicMock()
        mock_collection_info.payload_schema = {
            "site_name": MagicMock(index=True),
            "title": MagicMock(index=True),
            "unindexed_field": MagicMock(index=False),
        }
        qdrant_service._client.get_collection = AsyncMock(
            return_value=mock_collection_info
        )

        # Execute
        result = await qdrant_service.list_payload_indexes(collection_name)

        # Should return only indexed fields
        assert result == ["site_name", "title"]

    @pytest.mark.asyncio
    async def test_list_payload_indexes_no_schema(self, qdrant_service):
        """Test listing indexes when no payload schema exists."""
        collection_name = "test_collection"

        # Mock collection info without payload schema
        mock_collection_info = MagicMock()
        mock_collection_info.payload_schema = None
        qdrant_service._client.get_collection = AsyncMock(
            return_value=mock_collection_info
        )

        # Execute
        result = await qdrant_service.list_payload_indexes(collection_name)

        # Should return empty list
        assert result == []

    @pytest.mark.asyncio
    async def test_drop_payload_index_success(self, qdrant_service):
        """Test successful payload index deletion."""
        collection_name = "test_collection"
        field_name = "site_name"

        # Mock successful deletion
        qdrant_service._client.delete_payload_index = AsyncMock()

        # Execute
        await qdrant_service.drop_payload_index(collection_name, field_name)

        # Verify deletion was called correctly
        qdrant_service._client.delete_payload_index.assert_called_once_with(
            collection_name=collection_name, field_name=field_name, wait=True
        )

    @pytest.mark.asyncio
    async def test_reindex_collection_success(self, qdrant_service):
        """Test successful collection reindexing."""
        collection_name = "test_collection"

        # Mock existing indexes
        qdrant_service.list_payload_indexes = AsyncMock(
            return_value=["old_field1", "old_field2"]
        )
        qdrant_service.drop_payload_index = AsyncMock()
        qdrant_service.create_payload_indexes = AsyncMock()

        # Execute
        await qdrant_service.reindex_collection(collection_name)

        # Verify process
        assert qdrant_service.drop_payload_index.call_count == 2
        qdrant_service.create_payload_indexes.assert_called_once_with(collection_name)


class TestPayloadIndexStats:
    """Test payload index statistics functionality."""

    @pytest.mark.asyncio
    async def test_get_payload_index_stats_success(self, qdrant_service):
        """Test successful stats retrieval."""
        collection_name = "test_collection"

        # Mock collection info
        mock_collection_info = MagicMock()
        mock_collection_info.points_count = 1000
        mock_collection_info.payload_schema = {
            "site_name": MagicMock(data_type="keyword"),
            "word_count": MagicMock(data_type="integer"),
        }

        qdrant_service._client.get_collection = AsyncMock(
            return_value=mock_collection_info
        )
        qdrant_service.list_payload_indexes = AsyncMock(
            return_value=["site_name", "word_count"]
        )

        # Execute
        result = await qdrant_service.get_payload_index_stats(collection_name)

        # Verify structure
        assert result["collection_name"] == collection_name
        assert result["total_points"] == 1000
        assert result["indexed_fields_count"] == 2
        assert result["indexed_fields"] == ["site_name", "word_count"]
        assert "payload_schema" in result


class TestFilteredSearchWithIndexes:
    """Test filtered search functionality with payload indexes."""

    @pytest.mark.asyncio
    async def test_filtered_search_with_indexed_fields(self, qdrant_service):
        """Test filtered search using indexed fields."""
        collection_name = "test_collection"
        query_vector = [0.1] * 1536  # OpenAI embedding size
        filters = {
            "site_name": "FastAPI Documentation",
            "min_word_count": 100,
            "scraped_after": 1640995200,
        }

        # Mock successful search
        mock_results = MagicMock()
        mock_results.points = [
            MagicMock(
                id="1",
                score=0.95,
                payload={"content": "test", "site_name": "FastAPI Documentation"},
            )
        ]
        qdrant_service._client.query_points = AsyncMock(return_value=mock_results)

        # Execute
        results = await qdrant_service.filtered_search(
            collection_name=collection_name,
            query_vector=query_vector,
            filters=filters,
            limit=10,
        )

        # Verify results
        assert len(results) == 1
        assert results[0]["id"] == "1"
        assert results[0]["score"] == 0.95

        # Verify query_points was called with proper filter
        call_args = qdrant_service._client.query_points.call_args
        assert call_args.kwargs["collection_name"] == collection_name
        assert call_args.kwargs["query"] == query_vector
        assert call_args.kwargs["filter"] is not None

    @pytest.mark.asyncio
    async def test_filtered_search_input_validation(self, qdrant_service):
        """Test input validation for filtered search."""
        collection_name = "test_collection"

        # Test empty query vector
        with pytest.raises(ValueError, match="query_vector cannot be empty"):
            await qdrant_service.filtered_search(collection_name, [], {}, 10)

        # Test wrong vector dimension
        wrong_size_vector = [0.1] * 100  # Wrong size
        with pytest.raises(ValueError, match="query_vector dimension"):
            await qdrant_service.filtered_search(
                collection_name, wrong_size_vector, {}, 10
            )

        # Test invalid filters type
        with pytest.raises(ValueError, match="filters must be a dictionary"):
            await qdrant_service.filtered_search(
                collection_name, [0.1] * 1536, "invalid", 10
            )


class TestFilterBuilding:
    """Test filter building functionality."""

    @pytest.mark.asyncio
    async def test_build_filter_keyword_fields(self, qdrant_service):
        """Test filter building for keyword fields."""
        filters = {
            "site_name": "FastAPI Documentation",
            "embedding_model": "text-embedding-3-small",
            "search_strategy": "hybrid",
        }

        # Execute
        filter_obj = qdrant_service._build_filter(filters)

        # Verify filter structure
        assert filter_obj is not None
        assert len(filter_obj.must) == 3

        # Check each condition
        for condition in filter_obj.must:
            assert isinstance(condition, models.FieldCondition)
            assert condition.key in filters
            assert isinstance(condition.match, models.MatchValue)

    @pytest.mark.asyncio
    async def test_build_filter_range_fields(self, qdrant_service):
        """Test filter building for range fields."""
        filters = {
            "min_word_count": 100,
            "max_word_count": 1000,
            "scraped_after": 1640995200,
            "scraped_before": 1641081600,
        }

        # Execute
        filter_obj = qdrant_service._build_filter(filters)

        # Verify filter structure
        assert filter_obj is not None
        assert len(filter_obj.must) == 4

        # Check range conditions
        for condition in filter_obj.must:
            assert isinstance(condition, models.FieldCondition)
            assert hasattr(condition, "range")

    @pytest.mark.asyncio
    async def test_build_filter_text_fields(self, qdrant_service):
        """Test filter building for text fields."""
        filters = {"title": "API Documentation", "content_preview": "FastAPI tutorial"}

        # Execute
        filter_obj = qdrant_service._build_filter(filters)

        # Verify filter structure
        assert filter_obj is not None
        assert len(filter_obj.must) == 2

        # Check text conditions
        for condition in filter_obj.must:
            assert isinstance(condition, models.FieldCondition)
            assert isinstance(condition.match, models.MatchText)

    @pytest.mark.asyncio
    async def test_build_filter_invalid_values(self, qdrant_service):
        """Test filter building with invalid values."""
        # Test invalid filter value type
        filters = {"site_name": ["invalid", "list"]}

        with pytest.raises(
            ValueError, match="Filter value for site_name must be a simple type"
        ):
            qdrant_service._build_filter(filters)

    @pytest.mark.asyncio
    async def test_build_filter_empty(self, qdrant_service):
        """Test filter building with empty filters."""
        # Empty filters should return None
        assert qdrant_service._build_filter({}) is None
        assert qdrant_service._build_filter(None) is None


class TestPerformanceBenchmarks:
    """Test performance benchmarking functionality."""

    @pytest.mark.asyncio
    async def test_benchmark_search_performance(
        self, qdrant_service, sample_payload_data
    ):
        """Test search performance with and without indexes."""
        collection_name = "test_collection"
        query_vector = [0.1] * 1536
        filters = {"site_name": "FastAPI Documentation"}

        # Mock search results
        mock_results = MagicMock()
        mock_results.points = [
            MagicMock(id="1", score=0.95, payload=sample_payload_data)
        ]
        qdrant_service._client.query_points = AsyncMock(return_value=mock_results)

        # Simulate timing (mock should be fast)
        start_time = time.time()
        results = await qdrant_service.filtered_search(
            collection_name, query_vector, filters, 10
        )
        search_time = (time.time() - start_time) * 1000

        # Verify performance (mocked, should be very fast)
        assert search_time < 100  # Should be under 100ms
        assert len(results) == 1

    @pytest.mark.asyncio
    async def test_benchmark_multiple_filters(self, qdrant_service):
        """Test performance with complex multi-field filters."""
        collection_name = "test_collection"
        query_vector = [0.1] * 1536
        complex_filters = {
            "site_name": "FastAPI Documentation",
            "embedding_model": "text-embedding-3-small",
            "min_word_count": 100,
            "scraped_after": 1640995200,
            "search_strategy": "hybrid",
        }

        # Mock empty results
        mock_results = MagicMock()
        mock_results.points = []
        qdrant_service._client.query_points = AsyncMock(return_value=mock_results)

        # Execute complex filtered search
        results = await qdrant_service.filtered_search(
            collection_name, query_vector, complex_filters, 10
        )

        # Verify call was made with complex filter
        call_args = qdrant_service._client.query_points.call_args
        assert call_args.kwargs["filter"] is not None

        # Should handle multiple conditions without error
        assert isinstance(results, list)


class TestIntegrationScenarios:
    """Test realistic integration scenarios."""

    @pytest.mark.asyncio
    async def test_full_indexing_workflow(self, qdrant_service):
        """Test complete workflow: create indexes -> search -> reindex."""
        collection_name = "test_collection"

        # Mock all operations
        qdrant_service._client.create_payload_index = AsyncMock()
        qdrant_service._client.get_collection = AsyncMock(
            return_value=MagicMock(points_count=100, payload_schema={})
        )
        qdrant_service.list_payload_indexes = AsyncMock(return_value=[])
        qdrant_service._client.delete_payload_index = AsyncMock()
        qdrant_service._client.query_points = AsyncMock(
            return_value=MagicMock(points=[])
        )

        # 1. Create indexes
        await qdrant_service.create_payload_indexes(collection_name)

        # 2. Perform filtered search
        results = await qdrant_service.filtered_search(
            collection_name, [0.1] * 1536, {"site_name": "test"}, 10
        )

        # 3. Reindex collection
        await qdrant_service.reindex_collection(collection_name)

        # Verify all operations completed without error
        assert qdrant_service._client.create_payload_index.called
        assert qdrant_service._client.query_points.called
        assert isinstance(results, list)

    @pytest.mark.asyncio
    async def test_error_recovery_scenarios(self, qdrant_service):
        """Test error recovery in various scenarios."""
        collection_name = "test_collection"

        # Test partial index creation failure
        call_count = 0

        def mock_create_index(*args, **kwargs):
            nonlocal call_count
            call_count += 1
            if call_count == 3:  # Fail on 3rd call
                raise Exception("Index creation failed")
            return AsyncMock()

        qdrant_service._client.create_payload_index = AsyncMock(
            side_effect=mock_create_index
        )

        # Should fail and raise QdrantServiceError
        with pytest.raises(QdrantServiceError):
            await qdrant_service.create_payload_indexes(collection_name)

        # Verify it attempted multiple index creations before failing
        assert call_count == 3

    @pytest.mark.asyncio
    async def test_collection_creation_with_auto_indexing(self, qdrant_service):
        """Test that collection creation automatically creates indexes."""
        collection_name = "test_collection"

        # Mock collection creation workflow
        qdrant_service._client.get_collections = AsyncMock(
            return_value=MagicMock(collections=[])
        )
        qdrant_service._client.create_collection = AsyncMock()
        qdrant_service.create_payload_indexes = AsyncMock()

        # Execute collection creation
        await qdrant_service.create_collection(
            collection_name=collection_name, vector_size=1536, distance="Cosine"
        )

        # Verify indexes were automatically created
        qdrant_service.create_payload_indexes.assert_called_once_with(collection_name)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
