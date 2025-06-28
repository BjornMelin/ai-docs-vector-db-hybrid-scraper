"""JSON schema validation tests for API contracts.

This module tests JSON schema validation for all request/response models
and ensures data structure compliance.
"""

from datetime import UTC, datetime

import pytest

from src.models.api_contracts import (
    AdvancedSearchRequest,
    BulkDocumentRequest,
    CollectionRequest,
    DocumentRequest,
    ErrorResponse,
    HealthCheckResponse,
    SearchRequest,
    SearchResponse,
)


class TestPydanticModelValidation:
    """Test Pydantic model validation for API contracts."""

    @pytest.mark.schema_validation
    def test_search_request_validation(self):
        """Test SearchRequest model validation."""
        # Valid request
        valid_data = {
            "query": "test query",
            "collection_name": "documents",
            "limit": 10,
            "score_threshold": 0.5,
        }

        request = SearchRequest(**valid_data)
        assert request.query == "test query"
        assert request.collection_name == "documents"
        assert request.limit == 10
        assert request.score_threshold == 0.5

        # Test defaults
        minimal_request = SearchRequest(query="test")
        assert minimal_request.collection_name == "documents"
        assert minimal_request.limit == 10
        assert minimal_request.score_threshold == 0.0
        assert minimal_request.enable_hyde is False

        # Invalid data
        with pytest.raises(ValueError):
            SearchRequest(query="")  # Empty query

        with pytest.raises(ValueError):
            SearchRequest(query="test", limit=0)  # Invalid limit

        with pytest.raises(ValueError):
            SearchRequest(query="test", score_threshold=1.5)  # Invalid threshold

    @pytest.mark.schema_validation
    def test_advanced_search_request_validation(self):
        """Test AdvancedSearchRequest model validation."""
        # Valid request
        valid_data = {
            "query": "advanced query",
            "search_strategy": "hybrid",
            "accuracy_level": "balanced",
            "enable_reranking": True,
            "hyde_config": {"temperature": 0.7, "max_tokens": 100},
        }

        request = AdvancedSearchRequest(**valid_data)
        assert request.search_strategy == "hybrid"
        assert request.accuracy_level == "balanced"
        assert request.enable_reranking is True
        assert request.hyde_config == {"temperature": 0.7, "max_tokens": 100}

    @pytest.mark.schema_validation
    def test_search_response_validation(self):
        """Test SearchResponse model validation."""
        # Valid response
        valid_data = {
            "success": True,
            "timestamp": datetime.now(tz=UTC).timestamp(),
            "results": [
                {
                    "id": "doc1",
                    "score": 0.95,
                    "title": "Test Document",
                    "content": "Test content",
                    "metadata": {"source": "test"},
                }
            ],
            "_total_count": 1,
            "query_time_ms": 50.0,
            "search_strategy": "hybrid",
            "cache_hit": False,
        }

        response = SearchResponse(**valid_data)
        assert response.success is True
        assert len(response.results) == 1
        assert response.results[0].id == "doc1"
        assert response.results[0].score == 0.95
        assert response._total_count == 1
        assert response.search_strategy == "hybrid"

    @pytest.mark.schema_validation
    def test_document_request_validation(self):
        """Test DocumentRequest model validation."""
        # Valid request
        valid_data = {
            "url": "https://example.com/doc",
            "collection_name": "test_collection",
            "doc_type": "webpage",
            "metadata": {"category": "documentation"},
            "force_recrawl": True,
        }

        request = DocumentRequest(**valid_data)
        assert request.url == "https://example.com/doc"
        assert request.collection_name == "test_collection"
        assert request.doc_type == "webpage"
        assert request.force_recrawl is True

    @pytest.mark.schema_validation
    def test_bulk_document_request_validation(self):
        """Test BulkDocumentRequest model validation."""
        # Valid request
        valid_data = {
            "urls": [
                "https://example.com/doc1",
                "https://example.com/doc2",
                "https://example.com/doc3",
            ],
            "collection_name": "bulk_collection",
            "max_concurrent": 3,
        }

        request = BulkDocumentRequest(**valid_data)
        assert len(request.urls) == 3
        assert request.max_concurrent == 3

        # Invalid data
        with pytest.raises(ValueError):
            BulkDocumentRequest(urls=[])  # Empty URLs

        with pytest.raises(ValueError):
            BulkDocumentRequest(urls=["url"], max_concurrent=0)  # Invalid concurrent

        # Too many URLs
        too_many_urls = [f"https://example.com/doc{i}" for i in range(101)]
        with pytest.raises(ValueError):
            BulkDocumentRequest(urls=too_many_urls)

    @pytest.mark.schema_validation
    def test_collection_request_validation(self):
        """Test CollectionRequest model validation."""
        # Valid request
        valid_data = {
            "collection_name": "test_collection",
            "vector_size": 1024,
            "distance_metric": "Cosine",
            "enable_hybrid": True,
            "hnsw_config": {"m": 16, "ef_construct": 100},
        }

        request = CollectionRequest(**valid_data)
        assert request.collection_name == "test_collection"
        assert request.vector_size == 1024
        assert request.distance_metric == "Cosine"
        assert request.enable_hybrid is True

        # Invalid data
        with pytest.raises(ValueError):
            CollectionRequest(collection_name="")  # Empty name

        with pytest.raises(ValueError):
            CollectionRequest(collection_name="test", vector_size=0)  # Invalid size

    @pytest.mark.schema_validation
    def test_error_response_validation(self):
        """Test ErrorResponse model validation."""
        # Valid error response
        valid_data = {
            "success": False,
            "timestamp": datetime.now(tz=UTC).timestamp(),
            "error": "Invalid query parameter",
            "error_type": "validation_error",
            "context": {"parameter": "limit", "value": -1},
        }

        response = ErrorResponse(**valid_data)
        assert response.success is False
        assert response.error == "Invalid query parameter"
        assert response.error_type == "validation_error"
        assert "parameter" in response.context

    @pytest.mark.schema_validation
    def test_health_check_response_validation(self):
        """Test HealthCheckResponse model validation."""
        # Valid health response
        valid_data = {
            "success": True,
            "timestamp": datetime.now(tz=UTC).timestamp(),
            "status": "healthy",
            "services": {
                "qdrant": "healthy",
                "redis": "healthy",
                "embedding_service": "healthy",
            },
            "uptime_seconds": 3600.0,
            "version": "1.0.0",
        }

        response = HealthCheckResponse(**valid_data)
        assert response.status == "healthy"
        assert "qdrant" in response.services
        assert response.uptime_seconds == 3600.0
        assert response.version == "1.0.0"


class TestJSONSchemaGeneration:
    """Test JSON schema generation from Pydantic models."""

    @pytest.mark.schema_validation
    def test_generate_schemas(self, _json_schema_validator):
        """Test generating JSON schemas from Pydantic models."""
        # Generate schema for SearchRequest
        search_schema = SearchRequest.model_json_schema()

        # Validate schema structure
        assert "type" in search_schema
        assert search_schema["type"] == "object"
        assert "properties" in search_schema
        assert "required" in search_schema

        # Check required fields
        assert "query" in search_schema["required"]

        # Check property definitions
        properties = search_schema["properties"]
        assert "query" in properties
        assert "collection_name" in properties
        assert "limit" in properties

        # Validate query property
        query_prop = properties["query"]
        assert query_prop["type"] == "string"
        assert "minLength" in query_prop

        # Register and test validation
        _json_schema_validator.register_schema("search_request", search_schema)

        # Test valid data
        valid_data = {"query": "test query"}
        result = _json_schema_validator.validate_data(valid_data, "search_request")
        assert result["valid"]

        # Test invalid data
        invalid_data = {"query": ""}  # Empty query
        result = _json_schema_validator.validate_data(invalid_data, "search_request")
        assert not result["valid"]

    @pytest.mark.schema_validation
    def test_nested_schema_validation(self, _json_schema_validator):
        """Test validation of nested schema structures."""
        # Generate schema for SearchResponse
        response_schema = SearchResponse.model_json_schema()
        _json_schema_validator.register_schema("search_response", response_schema)

        # Test valid nested data
        valid_data = {
            "success": True,
            "timestamp": datetime.now(tz=UTC).timestamp(),
            "results": [
                {
                    "id": "doc1",
                    "score": 0.95,
                    "title": "Test Document",
                    "metadata": {"source": "test"},
                }
            ],
            "_total_count": 1,
            "query_time_ms": 50.0,
            "search_strategy": "hybrid",
            "cache_hit": False,
        }

        result = _json_schema_validator.validate_data(valid_data, "search_response")
        assert result["valid"], f"Validation errors: {result['errors']}"

        # Test invalid nested data (missing required field in result item)
        invalid_data = {
            "success": True,
            "timestamp": datetime.now(tz=UTC).timestamp(),
            "results": [
                {
                    "score": 0.95,  # Missing 'id' field
                    "title": "Test Document",
                }
            ],
            "_total_count": 1,
            "query_time_ms": 50.0,
            "search_strategy": "hybrid",
            "cache_hit": False,
        }

        result = _json_schema_validator.validate_data(invalid_data, "search_response")
        assert not result["valid"]


class TestSchemaCompatibility:
    """Test schema compatibility and evolution."""

    @pytest.mark.schema_validation
    def test_schema_evolution(self, _json_schema_validator):
        """Test schema evolution scenarios."""
        # Version 1 schema
        v1_schema = {
            "type": "object",
            "properties": {
                "id": {"type": "string"},
                "title": {"type": "string"},
                "score": {"type": "number"},
            },
            "required": ["id", "title"],
        }

        # Version 2 schema (backward compatible - added optional field)
        v2_schema = {
            "type": "object",
            "properties": {
                "id": {"type": "string"},
                "title": {"type": "string"},
                "score": {"type": "number"},
                "description": {"type": "string"},  # New optional field
            },
            "required": ["id", "title"],
        }

        # Data valid in v1
        v1_data = {"id": "doc1", "title": "Test Document", "score": 0.95}

        # Should be valid in both versions
        result_v1 = _json_schema_validator.validate_against_schema(v1_data, v1_schema)
        assert result_v1["valid"]

        result_v2 = _json_schema_validator.validate_against_schema(v1_data, v2_schema)
        assert result_v2["valid"]

        # Data with new field
        v2_data = {
            "id": "doc1",
            "title": "Test Document",
            "score": 0.95,
            "description": "Test description",
        }

        # Should be valid in v2
        result_v2_extended = _json_schema_validator.validate_against_schema(
            v2_data, v2_schema
        )
        assert result_v2_extended["valid"]

    @pytest.mark.schema_validation
    def test_schema_breaking_changes(self, _json_schema_validator):
        """Test detection of schema breaking changes."""
        # Original schema
        original_schema = {
            "type": "object",
            "properties": {
                "id": {"type": "string"},
                "title": {"type": "string"},
                "score": {"type": "number"},
            },
            "required": ["id"],
        }

        # Breaking change: new required field
        breaking_schema = {
            "type": "object",
            "properties": {
                "id": {"type": "string"},
                "title": {"type": "string"},
                "score": {"type": "number"},
                "required_field": {"type": "string"},
            },
            "required": ["id", "required_field"],  # Added required field
        }

        # Data valid in original
        test_data = {"id": "doc1", "title": "Test Document"}

        # Should be valid in original
        result_original = _json_schema_validator.validate_against_schema(
            test_data, original_schema
        )
        assert result_original["valid"]

        # Should be invalid in breaking change
        result_breaking = _json_schema_validator.validate_against_schema(
            test_data, breaking_schema
        )
        assert not result_breaking["valid"]
        assert any("required_field" in error for error in result_breaking["errors"])


class TestDataValidationEdgeCases:
    """Test edge cases for data validation."""

    @pytest.mark.schema_validation
    def test_boundary_values(self):
        """Test boundary value validation."""
        # Test limit boundaries
        min_request = SearchRequest(query="test", limit=1)
        assert min_request.limit == 1

        max_request = SearchRequest(query="test", limit=100)
        assert max_request.limit == 100

        # Invalid boundaries
        with pytest.raises(ValueError):
            SearchRequest(query="test", limit=0)

        with pytest.raises(ValueError):
            SearchRequest(query="test", limit=101)

        # Test score threshold boundaries
        min_score = SearchRequest(query="test", score_threshold=0.0)
        assert min_score.score_threshold == 0.0

        max_score = SearchRequest(query="test", score_threshold=1.0)
        assert max_score.score_threshold == 1.0

        # Invalid score boundaries
        with pytest.raises(ValueError):
            SearchRequest(query="test", score_threshold=-0.1)

        with pytest.raises(ValueError):
            SearchRequest(query="test", score_threshold=1.1)

    @pytest.mark.schema_validation
    def test_unicode_and_special_characters(self):
        """Test handling of Unicode and special characters."""
        # Unicode query
        unicode_request = SearchRequest(query="测试查询")
        assert unicode_request.query == "测试查询"

        # Special characters
        special_request = SearchRequest(query="test@#$%^&*()")
        assert special_request.query == "test@#$%^&*()"

        # Emoji
        emoji_request = SearchRequest(query="test 🔍 search")
        assert emoji_request.query == "test 🔍 search"

    @pytest.mark.schema_validation
    def test_null_and_empty_values(self):
        """Test handling of null and empty values."""
        # Optional fields can be None
        request = SearchRequest(query="test", filters=None)
        assert request.filters is None

        # Empty metadata dict is allowed
        doc_request = DocumentRequest(url="https://example.com", metadata={})
        assert doc_request.metadata == {}

        # Empty string in query should fail
        with pytest.raises(ValueError):
            SearchRequest(query="")

        # Whitespace-only query should fail
        with pytest.raises(ValueError):
            SearchRequest(query="   ")
