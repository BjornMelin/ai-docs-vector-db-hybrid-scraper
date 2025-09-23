"""Unit tests for API contract models."""

import pytest
from pydantic import ValidationError

from src.models.api_contracts import (
    AnalyticsRequest,
    AnalyticsResponse,
    BulkDocumentRequest,
    BulkDocumentResponse,
    CacheRequest,
    CacheResponse,
    CollectionInfo,
    CollectionRequest,
    CollectionResponse,
    DocumentRequest,
    DocumentResponse,
    ErrorResponse,
    HealthCheckResponse,
    ListCollectionsResponse,
    MCPRequest,
    MCPResponse,
    MetricData,
    SearchRequest,
    SearchResponse,
    SearchResultItem,
    ValidationRequest,
    ValidationResponse,
)


class TestMCPRequest:
    """Test MCPRequest base model."""

    def test_forbids_extra_fields(self):
        """Test that extra fields are forbidden."""
        with pytest.raises(ValidationError):
            MCPRequest(extra_field="not allowed")

    def test_empty_request(self):
        """Test creating empty request."""
        request = MCPRequest()
        assert isinstance(request, MCPRequest)


class TestMCPResponse:
    """Test MCPResponse base model."""

    def test_required_fields(self):
        """Test that required fields are enforced."""
        # Missing success
        with pytest.raises(ValidationError) as exc_info:
            MCPResponse(timestamp=123.45)
        assert "success" in str(exc_info.value)

        # Missing timestamp
        with pytest.raises(ValidationError) as exc_info:
            MCPResponse(success=True)
        assert "timestamp" in str(exc_info.value)

    def test_valid_response(self):
        """Test creating valid response."""
        response = MCPResponse(success=True, timestamp=123.45)
        assert response.success is True
        assert response.timestamp == 123.45

    def test_forbids_extra_fields(self):
        """Test that extra fields are forbidden."""
        with pytest.raises(ValidationError):
            MCPResponse(success=True, timestamp=123.45, extra_field="not allowed")


class TestErrorResponse:
    """Test ErrorResponse model."""

    def test_default_values(self):
        """Test default field values."""
        response = ErrorResponse(error="Test error", timestamp=123.45)
        assert response.success is False
        assert response.error == "Test error"
        assert response.error_type == "general"
        assert response.context == {}

    def test_custom_values(self):
        """Test custom field values."""
        context = {"file": "test.py", "line": 42}
        response = ErrorResponse(
            error="File not found",
            error_type="io_error",
            context=context,
            timestamp=123.45,
        )
        assert response.error == "File not found"
        assert response.error_type == "io_error"
        assert response.context == context

    def test_inherits_from_mcp_response(self):
        """Test that ErrorResponse inherits from MCPResponse."""
        response = ErrorResponse(error="Test", timestamp=123.45)
        assert isinstance(response, MCPResponse)


class TestSearchRequest:
    """Test SearchRequest model."""

    def test_default_values(self):
        """Test default field values."""
        request = SearchRequest(query="test")
        assert request.query == "test"
        assert request.collection_name == "documents"
        assert request.search_strategy == "hybrid"
        assert request.limit == 10
        assert request.accuracy_level == "balanced"
        assert request.enable_reranking is False
        assert request.hyde_config is None
        assert request.filters is None

    def test_search_strategy_field(self):
        """Test search_strategy field."""
        # Any string is accepted as search strategy
        request = SearchRequest(query="test", search_strategy="dense")
        assert request.search_strategy == "dense"

        request = SearchRequest(query="test", search_strategy="multi_stage")
        assert request.search_strategy == "multi_stage"

    def test_accuracy_level_field(self):
        """Test accuracy_level field."""
        # Any string is accepted as accuracy level
        request = SearchRequest(query="test", accuracy_level="fast")
        assert request.accuracy_level == "fast"

        request = SearchRequest(query="test", accuracy_level="exact")
        assert request.accuracy_level == "exact"

    def test_hyde_config_field(self):
        """Test hyde_config field accepts any dict."""
        hyde_config = {"temperature": 0.7, "max_tokens": 100}
        request = SearchRequest(query="test", hyde_config=hyde_config)
        assert request.hyde_config == hyde_config


class TestSearchResultItem:
    """Test SearchResultItem model."""

    def test_required_fields(self):
        """Test required fields."""
        # Valid item
        item = SearchResultItem(id="123", score=0.95)
        assert item.id == "123"
        assert item.score == 0.95

        # Missing required fields
        with pytest.raises(ValidationError):
            SearchResultItem(score=0.95)  # Missing id
        with pytest.raises(ValidationError):
            SearchResultItem(id="123")  # Missing score

    def test_optional_fields(self):
        """Test optional fields with defaults."""
        item = SearchResultItem(id="123", score=0.95)
        assert item.title is None
        assert item.content is None
        assert item.url is None
        assert item.doc_type is None
        assert item.language is None
        assert item.metadata == {}

    def test_all_fields(self):
        """Test creating item with all fields."""
        metadata = {"author": "John Doe", "date": "2024-01-01"}
        item = SearchResultItem(
            id="123",
            score=0.95,
            title="Test Document",
            content="This is the content",
            url="https://example.com",
            doc_type="article",
            language="en",
            metadata=metadata,
        )
        assert item.title == "Test Document"
        assert item.content == "This is the content"
        assert item.url == "https://example.com"
        assert item.doc_type == "article"
        assert item.language == "en"
        assert item.metadata == metadata


class TestSearchResponse:
    """Test SearchResponse model."""

    def test_default_values(self):
        """Test default field values."""
        response = SearchResponse(timestamp=123.45)
        assert response.success is True
        assert response.results == []
        assert response._total_count == 0
        assert response.query_time_ms == 0.0
        assert response.search_strategy == "unknown"
        assert response.cache_hit is False

    def test_with_results(self):
        """Test response with search results."""
        results = [
            SearchResultItem(id="1", score=0.9),
            SearchResultItem(id="2", score=0.8),
        ]
        response = SearchResponse(
            timestamp=123.45,
            results=results,
            _total_count=2,
            query_time_ms=15.5,
            search_strategy="hybrid",
            cache_hit=True,
        )
        assert len(response.results) == 2
        assert response._total_count == 2
        assert response.query_time_ms == 15.5
        assert response.search_strategy == "hybrid"
        assert response.cache_hit is True


class TestDocumentRequest:
    """Test DocumentRequest model."""

    def test_required_fields(self):
        """Test required fields."""
        request = DocumentRequest(url="https://example.com")
        assert request.url == "https://example.com"

        # Missing required field
        with pytest.raises(ValidationError):
            DocumentRequest()

    def test_default_values(self):
        """Test default field values."""
        request = DocumentRequest(url="https://example.com")
        assert request.collection_name == "documents"
        assert request.doc_type is None
        assert request.metadata == {}
        assert request.force_recrawl is False

    def test_custom_values(self):
        """Test custom field values."""
        metadata = {"tags": ["python", "tutorial"]}
        request = DocumentRequest(
            url="https://example.com",
            collection_name="tutorials",
            doc_type="tutorial",
            metadata=metadata,
            force_recrawl=True,
        )
        assert request.collection_name == "tutorials"
        assert request.doc_type == "tutorial"
        assert request.metadata == metadata
        assert request.force_recrawl is True


class TestBulkDocumentRequest:
    """Test BulkDocumentRequest model."""

    def test_required_fields(self):
        """Test required fields."""
        urls = ["https://example1.com", "https://example2.com"]
        request = BulkDocumentRequest(urls=urls)
        assert request.urls == urls

        # Missing required field
        with pytest.raises(ValidationError):
            BulkDocumentRequest()

    def test_urls_constraints(self):
        """Test urls field constraints."""
        # Valid: 1-100 URLs
        BulkDocumentRequest(urls=["https://example.com"])
        BulkDocumentRequest(urls=["https://example.com"] * 100)

        # Invalid: empty list
        with pytest.raises(ValidationError):
            BulkDocumentRequest(urls=[])

        # Invalid: more than 100 URLs
        with pytest.raises(ValidationError):
            BulkDocumentRequest(urls=["https://example.com"] * 101)

    def test_max_concurrent_constraints(self):
        """Test max_concurrent field constraints."""
        # Valid values
        BulkDocumentRequest(urls=["url"], max_concurrent=1)
        BulkDocumentRequest(urls=["url"], max_concurrent=20)

        # Invalid values
        with pytest.raises(ValidationError):
            BulkDocumentRequest(urls=["url"], max_concurrent=0)
        with pytest.raises(ValidationError):
            BulkDocumentRequest(urls=["url"], max_concurrent=21)

    def test_default_values(self):
        """Test default field values."""
        request = BulkDocumentRequest(urls=["url"])
        assert request.collection_name == "documents"
        assert request.doc_type is None
        assert request.metadata == {}
        assert request.force_recrawl is False
        assert request.max_concurrent == 5


class TestDocumentResponse:
    """Test DocumentResponse model."""

    def test_required_fields(self):
        """Test required fields."""
        response = DocumentResponse(
            timestamp=123.45,
            document_id="doc123",
            url="https://example.com",
        )
        assert response.document_id == "doc123"
        assert response.url == "https://example.com"

    def test_default_values(self):
        """Test default field values."""
        response = DocumentResponse(
            timestamp=123.45,
            document_id="doc123",
            url="https://example.com",
        )
        assert response.success is True
        assert response.chunks_created == 0
        assert response.processing_time_ms == 0.0
        assert response.status == "processed"

    def test_custom_values(self):
        """Test custom field values."""
        response = DocumentResponse(
            timestamp=123.45,
            success=False,
            document_id="doc123",
            url="https://example.com",
            chunks_created=5,
            processing_time_ms=150.5,
            status="failed",
        )
        assert response.success is False
        assert response.chunks_created == 5
        assert response.processing_time_ms == 150.5
        assert response.status == "failed"


class TestBulkDocumentResponse:
    """Test BulkDocumentResponse model."""

    def test_default_values(self):
        """Test default field values."""
        response = BulkDocumentResponse(timestamp=123.45)
        assert response.success is True
        assert response.processed_count == 0
        assert response.failed_count == 0
        assert response._total_chunks == 0
        assert response.processing_time_ms == 0.0
        assert response.results == []
        assert response.errors == []

    def test_with_results(self):
        """Test response with results."""
        results = [
            DocumentResponse(
                timestamp=123.45,
                document_id="doc1",
                url="https://example1.com",
                chunks_created=3,
            ),
            DocumentResponse(
                timestamp=123.46,
                document_id="doc2",
                url="https://example2.com",
                chunks_created=5,
            ),
        ]
        errors = ["Failed to process https://example3.com"]

        response = BulkDocumentResponse(
            timestamp=123.45,
            processed_count=2,
            failed_count=1,
            _total_chunks=8,
            processing_time_ms=500.0,
            results=results,
            errors=errors,
        )
        assert response.processed_count == 2
        assert response.failed_count == 1
        assert response._total_chunks == 8
        assert len(response.results) == 2
        assert len(response.errors) == 1


class TestCollectionRequest:
    """Test CollectionRequest model."""

    def test_required_fields(self):
        """Test required fields."""
        request = CollectionRequest(collection_name="my_collection")
        assert request.collection_name == "my_collection"

    def test_collection_name_validation(self):
        """Test collection_name validation."""
        # Empty name not allowed
        with pytest.raises(ValidationError):
            CollectionRequest(collection_name="")

    def test_vector_size_constraints(self):
        """Test vector_size constraints."""
        # Valid values
        CollectionRequest(collection_name="test", vector_size=1)
        CollectionRequest(collection_name="test", vector_size=4096)

        # Invalid value
        with pytest.raises(ValidationError):
            CollectionRequest(collection_name="test", vector_size=0)

    def test_default_values(self):
        """Test default field values."""
        request = CollectionRequest(collection_name="test")
        assert request.vector_size is None
        assert request.distance_metric == "Cosine"
        assert request.enable_hybrid is True
        assert request.hnsw_config is None

    def test_hnsw_config(self):
        """Test hnsw_config field."""
        hnsw_config = {"m": 16, "ef_construct": 200}
        request = CollectionRequest(
            collection_name="test",
            hnsw_config=hnsw_config,
        )
        assert request.hnsw_config == hnsw_config


class TestCollectionInfo:
    """Test CollectionInfo model."""

    def test_required_fields(self):
        """Test required fields."""
        info = CollectionInfo(name="my_collection")
        assert info.name == "my_collection"

    def test_default_values(self):
        """Test default field values."""
        info = CollectionInfo(name="test")
        assert info.points_count == 0
        assert info.vectors_count == 0
        assert info.indexed_fields == []
        assert info.status == "unknown"
        assert info.config == {}

    def test_all_fields(self):
        """Test creating info with all fields."""
        config = {"vector_size": 384, "distance": "Cosine"}
        info = CollectionInfo(
            name="test",
            points_count=1000,
            vectors_count=1000,
            indexed_fields=["title", "content"],
            status="green",
            config=config,
        )
        assert info.points_count == 1000
        assert info.vectors_count == 1000
        assert info.indexed_fields == ["title", "content"]
        assert info.status == "green"
        assert info.config == config


class TestCollectionResponse:
    """Test CollectionResponse model."""

    def test_required_fields(self):
        """Test required fields."""
        response = CollectionResponse(
            timestamp=123.45,
            collection_name="test",
            operation="create",
        )
        assert response.collection_name == "test"
        assert response.operation == "create"

    def test_default_values(self):
        """Test default field values."""
        response = CollectionResponse(
            timestamp=123.45,
            collection_name="test",
            operation="create",
        )
        assert response.success is True
        assert response.details == {}

    def test_with_details(self):
        """Test response with details."""
        details = {"vector_size": 384, "indexed_fields": ["title"]}
        response = CollectionResponse(
            timestamp=123.45,
            collection_name="test",
            operation="update",
            details=details,
        )
        assert response.details == details


class TestListCollectionsResponse:
    """Test ListCollectionsResponse model."""

    def test_default_values(self):
        """Test default field values."""
        response = ListCollectionsResponse(timestamp=123.45)
        assert response.success is True
        assert response.collections == []
        assert response._total_count == 0

    def test_with_collections(self):
        """Test response with collections."""
        collections = [
            CollectionInfo(name="col1", points_count=100),
            CollectionInfo(name="col2", points_count=200),
        ]
        response = ListCollectionsResponse(
            timestamp=123.45,
            collections=collections,
            _total_count=2,
        )
        assert len(response.collections) == 2
        assert response._total_count == 2


class TestAnalyticsRequest:
    """Test AnalyticsRequest model."""

    def test_default_values(self):
        """Test default field values."""
        request = AnalyticsRequest()
        assert request.collection_name is None
        assert request.time_range == "24h"
        assert request.metric_types == []

    def test_custom_values(self):
        """Test custom field values."""
        request = AnalyticsRequest(
            collection_name="test",
            time_range="7d",
            metric_types=["searches", "documents"],
        )
        assert request.collection_name == "test"
        assert request.time_range == "7d"
        assert request.metric_types == ["searches", "documents"]


class TestMetricData:
    """Test MetricData model."""

    def test_required_fields(self):
        """Test required fields."""
        metric = MetricData(name="cpu_usage", value=75.5)
        assert metric.name == "cpu_usage"
        assert metric.value == 75.5

    def test_value_types(self):
        """Test different value types."""
        # Integer value
        metric = MetricData(name="count", value=100)
        assert metric.value == 100

        # Float value
        metric = MetricData(name="percentage", value=99.9)
        assert metric.value == 99.9

        # String value
        metric = MetricData(name="status", value="healthy")
        assert metric.value == "healthy"

    def test_optional_fields(self):
        """Test optional fields."""
        metric = MetricData(name="test", value=42)
        assert metric.unit is None
        assert metric.timestamp is None

        # With optional fields
        metric = MetricData(
            name="memory",
            value=2048,
            unit="MB",
            timestamp=123.45,
        )
        assert metric.unit == "MB"
        assert metric.timestamp == 123.45


class TestAnalyticsResponse:
    """Test AnalyticsResponse model."""

    def test_required_fields(self):
        """Test required fields."""
        response = AnalyticsResponse(
            timestamp=123.45,
            time_range="24h",
            generated_at=123.45,
        )
        assert response.time_range == "24h"
        assert response.generated_at == 123.45

    def test_default_values(self):
        """Test default field values."""
        response = AnalyticsResponse(
            timestamp=123.45,
            time_range="24h",
            generated_at=123.45,
        )
        assert response.success is True
        assert response.metrics == []

    def test_with_metrics(self):
        """Test response with metrics."""
        metrics = [
            MetricData(name="searches", value=1000),
            MetricData(name="latency_p95", value=150.5, unit="ms"),
        ]
        response = AnalyticsResponse(
            timestamp=123.45,
            metrics=metrics,
            time_range="24h",
            generated_at=123.45,
        )
        assert len(response.metrics) == 2


class TestHealthCheckResponse:
    """Test HealthCheckResponse model."""

    def test_default_values(self):
        """Test default field values."""
        response = HealthCheckResponse(timestamp=123.45)
        assert response.success is True
        assert response.status == "healthy"
        assert response.services == {}
        assert response.uptime_seconds == 0.0
        assert response.version == "unknown"

    def test_custom_values(self):
        """Test custom field values."""
        services = {
            "database": "healthy",
            "cache": "degraded",
            "search": "healthy",
        }
        response = HealthCheckResponse(
            timestamp=123.45,
            status="degraded",
            services=services,
            uptime_seconds=3600.0,
            version="1.0.0",
        )
        assert response.status == "degraded"
        assert response.services == services
        assert response.uptime_seconds == 3600.0
        assert response.version == "1.0.0"


class TestCacheRequest:
    """Test CacheRequest model."""

    def test_required_fields(self):
        """Test required fields."""
        request = CacheRequest(operation="clear")
        assert request.operation == "clear"

    def test_optional_fields(self):
        """Test optional fields."""
        request = CacheRequest(operation="stats")
        assert request.cache_type is None
        assert request.keys is None

        # With optional fields
        request = CacheRequest(
            operation="warm",
            cache_type="search",
            keys=["key1", "key2"],
        )
        assert request.cache_type == "search"
        assert request.keys == ["key1", "key2"]


class TestCacheResponse:
    """Test CacheResponse model."""

    def test_required_fields(self):
        """Test required fields."""
        response = CacheResponse(
            timestamp=123.45,
            operation="clear",
        )
        assert response.operation == "clear"

    def test_default_values(self):
        """Test default field values."""
        response = CacheResponse(
            timestamp=123.45,
            operation="clear",
        )
        assert response.success is True
        assert response.affected_keys == 0
        assert response.cache_stats == {}

    def test_with_stats(self):
        """Test response with cache stats."""
        stats = {
            "_total_keys": 1000,
            "memory_used": "512MB",
            "hit_rate": 0.85,
        }
        response = CacheResponse(
            timestamp=123.45,
            operation="stats",
            affected_keys=50,
            cache_stats=stats,
        )
        assert response.affected_keys == 50
        assert response.cache_stats == stats


class TestValidationRequest:
    """Test ValidationRequest model."""

    def test_default_values(self):
        """Test default field values."""
        request = ValidationRequest()
        assert request.config_section is None
        assert request.validate_connections is True

    def test_custom_values(self):
        """Test custom field values."""
        request = ValidationRequest(
            config_section="database",
            validate_connections=False,
        )
        assert request.config_section == "database"
        assert request.validate_connections is False


class TestValidationResponse:
    """Test ValidationResponse model."""

    def test_required_fields(self):
        """Test required fields."""
        response = ValidationResponse(
            timestamp=123.45,
            valid=True,
        )
        assert response.valid is True

    def test_default_values(self):
        """Test default field values."""
        response = ValidationResponse(
            timestamp=123.45,
            valid=True,
        )
        assert response.success is True
        assert response.issues == []
        assert response.warnings == []
        assert response.tested_services == []

    def test_validation_failure(self):
        """Test validation failure response."""
        response = ValidationResponse(
            timestamp=123.45,
            valid=False,
            issues=["Database connection failed", "Invalid API key"],
            warnings=["Cache service not configured"],
            tested_services=["database", "api", "cache"],
        )
        assert response.valid is False
        assert len(response.issues) == 2
        assert len(response.warnings) == 1
        assert len(response.tested_services) == 3
