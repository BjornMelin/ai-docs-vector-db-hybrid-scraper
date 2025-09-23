"""Unit tests for secure vector search models."""

import pytest
from pydantic import ValidationError

from src.models.vector_search import (
    AdvancedHybridSearchRequest,
    BasicSearchRequest,
    DimensionError,
    FilterValidationError,
    FusionAlgorithm,
    HyDESearchRequest,
    MultiStageSearchRequest,
    SearchAccuracy,
    SearchResponse,
    SearchStage,
    SecureFilterModel,
    SecurePayloadModel,
    SecureSearchResult,
    SecureVectorModel,
)


class TestSecureVectorModel:
    """Test SecureVectorModel."""

    def test_valid_vector(self):
        """Test valid vector creation."""
        vector = SecureVectorModel(values=[0.1, 0.2, 0.3, 0.4])
        assert vector.values == [0.1, 0.2, 0.3, 0.4]
        assert vector.dimension == 4

    def test_empty_vector_rejected(self):
        """Test empty vector is rejected."""
        with pytest.raises(ValidationError) as exc_info:
            SecureVectorModel(values=[])
        assert "Vector cannot be empty" in str(exc_info.value)

    def test_oversized_vector_rejected(self):
        """Test oversized vector is rejected."""
        large_vector = [0.1] * 5000  # Exceeds 4096 limit
        with pytest.raises(DimensionError) as exc_info:
            SecureVectorModel(values=large_vector)
        assert "Vector dimensions exceed allowed" in str(exc_info.value)

    def test_invalid_values_rejected(self):
        """Test invalid values are rejected."""
        with pytest.raises(ValidationError) as exc_info:
            SecureVectorModel(values=[0.1, float("nan"), 0.3])
        assert "Invalid vector value" in str(exc_info.value)

        with pytest.raises(ValidationError) as exc_info:
            SecureVectorModel(values=[0.1, float("inf"), 0.3])
        assert "Invalid vector value" in str(exc_info.value)


class TestSecureFilterModel:
    """Test SecureFilterModel."""

    def test_valid_filter(self):
        """Test valid filter creation."""
        filter_model = SecureFilterModel(
            field="category", operator="eq", value="documentation"
        )
        assert filter_model.field == "category"
        assert filter_model.operator == "eq"
        assert filter_model.value == "documentation"

    def test_invalid_field_name_rejected(self):
        """Test invalid field names are rejected."""
        with pytest.raises(ValidationError) as exc_info:
            SecureFilterModel(field="invalid-field-name!", operator="eq", value="test")
        assert "String should match pattern" in str(exc_info.value)

    def test_sql_injection_prevented(self):
        """Test SQL injection patterns are prevented."""
        with pytest.raises(FilterValidationError) as exc_info:
            SecureFilterModel(
                field="category", operator="eq", value="'; DROP TABLE users; --"
            )
        assert "Potentially dangerous pattern" in str(exc_info.value)


class TestBasicSearchRequest:
    """Test BasicSearchRequest."""

    def test_valid_request(self):
        """Test valid search request creation."""
        vector = SecureVectorModel(values=[0.1, 0.2, 0.3])
        request = BasicSearchRequest(query_vector=vector)
        assert request.query_vector == vector
        assert request.include_metadata is True
        assert request.include_vectors is False

    def test_with_filters(self):
        """Test request with filters."""
        vector = SecureVectorModel(values=[0.1, 0.2, 0.3])
        filter_model = SecureFilterModel(field="category", operator="eq", value="docs")
        request = BasicSearchRequest(query_vector=vector, filters=[filter_model])
        assert len(request.filters) == 1
        assert request.filters[0] == filter_model


class TestAdvancedHybridSearchRequest:
    """Test AdvancedHybridSearchRequest."""

    def test_weight_normalization(self):
        """Test weight normalization."""
        vector = SecureVectorModel(values=[0.1, 0.2, 0.3])
        request = AdvancedHybridSearchRequest(
            query_vector=vector,
            dense_weight=0.6,
            sparse_weight=0.6,  # Total > 1.0, should be normalized
        )
        # Weights should be normalized to sum to 1.0
        assert abs(request.dense_weight + request.sparse_weight - 1.0) < 0.01


class TestSearchStage:
    """Test SearchStage model."""

    def test_valid_stage(self):
        """Test valid search stage creation."""
        vector = SecureVectorModel(values=[0.1, 0.2, 0.3])
        stage = SearchStage(stage_name="test_stage", query_vector=vector, weight=0.8)
        assert stage.stage_name == "test_stage"
        assert stage.query_vector == vector
        assert stage.weight == 0.8

    def test_invalid_stage_name_rejected(self):
        """Test invalid stage names are rejected."""
        vector = SecureVectorModel(values=[0.1, 0.2, 0.3])
        with pytest.raises(ValidationError) as exc_info:
            SearchStage(
                stage_name="invalid stage name!",  # Contains spaces and special chars
                query_vector=vector,
            )
        assert "String should match pattern" in str(exc_info.value)


class TestSecureSearchResult:
    """Test SecureSearchResult."""

    def test_valid_result(self):
        """Test valid search result creation."""
        # Build the model to resolve forward references
        SecurePayloadModel.model_rebuild()

        payload = SecurePayloadModel(content="Test content")
        result = SecureSearchResult(id="test-id", score=0.95, payload=payload)
        assert result.id == "test-id"
        assert result.score == 0.95
        assert result.relevance_tier == "high"

    def test_relevance_tiers(self):
        """Test relevance tier calculation."""
        # Build the model to resolve forward references
        SecurePayloadModel.model_rebuild()

        payload = SecurePayloadModel(content="Test content")

        # High relevance
        result_high = SecureSearchResult(id="1", score=0.9, payload=payload)
        assert result_high.relevance_tier == "high"

        # Medium relevance
        result_medium = SecureSearchResult(id="2", score=0.7, payload=payload)
        assert result_medium.relevance_tier == "medium"

        # Low relevance
        result_low = SecureSearchResult(id="3", score=0.4, payload=payload)
        assert result_low.relevance_tier == "low"


class TestSearchResponse:
    """Test SearchResponse."""

    def test_valid_response(self):
        """Test valid search response creation."""
        # Build the model to resolve forward references
        SecurePayloadModel.model_rebuild()

        payload = SecurePayloadModel(content="Test content")
        result = SecureSearchResult(id="1", score=0.9, payload=payload)

        response = SearchResponse(
            results=[result],
            total_count=1,
            search_time_ms=150,
            accuracy=SearchAccuracy.HIGH,
        )
        assert len(response.results) == 1
        assert response.total_count == 1
        assert response.search_time_ms == 150
        assert response.accuracy == SearchAccuracy.HIGH

    def test_performance_score_calculation(self):
        """Test performance score calculation."""
        # Build the model to resolve forward references
        SecurePayloadModel.model_rebuild()

        payload = SecurePayloadModel(content="Test content")
        result = SecureSearchResult(id="1", score=0.9, payload=payload)

        # Fast search with good cache efficiency
        response = SearchResponse(
            results=[result],
            total_count=1,
            search_time_ms=100,  # Fast search
            accuracy=SearchAccuracy.HIGH,
            vector_ops=10,
            cache_hits=8,  # Good cache efficiency
        )
        performance_score = response.performance_score
        assert 0.0 <= performance_score <= 1.0
        assert performance_score > 0.8  # Should be high for fast, cached search


class TestMultiStageSearchRequest:
    """Test MultiStageSearchRequest."""

    def test_weight_normalization(self):
        """Test stage weight normalization."""
        vector1 = SecureVectorModel(values=[0.1, 0.2, 0.3])
        vector2 = SecureVectorModel(values=[0.4, 0.5, 0.6])

        stage1 = SearchStage(stage_name="stage1", query_vector=vector1, weight=0.3)
        stage2 = SearchStage(stage_name="stage2", query_vector=vector2, weight=0.9)

        request = MultiStageSearchRequest(
            stages=[stage1, stage2], fusion_algorithm=FusionAlgorithm.RRF
        )

        # Weights should be normalized to sum to 1.0
        total_weight = sum(stage.weight for stage in request.stages)
        assert abs(total_weight - 1.0) < 0.01


class TestHyDESearchRequest:
    """Test HyDESearchRequest."""

    def test_valid_hyde_request(self):
        """Test valid HyDE search request creation."""
        vector = SecureVectorModel(values=[0.1, 0.2, 0.3])
        request = HyDESearchRequest(
            query_vector=vector, query_text="What is vector search?", num_hypotheses=5
        )
        assert request.query_text == "What is vector search?"
        assert request.num_hypotheses == 5
        assert request.hypothesis_weight == 0.5

    def test_short_query_rejected(self):
        """Test short queries are rejected."""
        vector = SecureVectorModel(values=[0.1, 0.2, 0.3])
        with pytest.raises(ValidationError) as exc_info:
            HyDESearchRequest(
                query_vector=vector,
                query_text="hi",  # Too short
                num_hypotheses=3,
            )
        assert "Query too short for meaningful search" in str(exc_info.value)
