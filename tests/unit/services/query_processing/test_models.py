"""Tests for query processing models."""

import pytest
from pydantic import ValidationError

from src.services.query_processing.models import (
    MatryoshkaDimension,
    QueryComplexity,
    QueryIntent,
    QueryIntentClassification,
    QueryPreprocessingResult,
    QueryProcessingRequest,
    QueryProcessingResponse,
    SearchStrategy,
    SearchStrategySelection,
)


class TestQueryIntent:
    """Test the QueryIntent enum."""

    def test_basic_intents(self):
        """Test basic intent categories."""
        assert QueryIntent.CONCEPTUAL == "conceptual"
        assert QueryIntent.PROCEDURAL == "procedural"
        assert QueryIntent.FACTUAL == "factual"
        assert QueryIntent.TROUBLESHOOTING == "troubleshooting"

    def test_advanced_intents(self):
        """Test advanced intent categories."""
        assert QueryIntent.COMPARATIVE == "comparative"
        assert QueryIntent.ARCHITECTURAL == "architectural"
        assert QueryIntent.PERFORMANCE == "performance"
        assert QueryIntent.SECURITY == "security"
        assert QueryIntent.INTEGRATION == "integration"
        assert QueryIntent.BEST_PRACTICES == "best_practices"
        assert QueryIntent.CODE_REVIEW == "code_review"
        assert QueryIntent.MIGRATION == "migration"
        assert QueryIntent.DEBUGGING == "debugging"
        assert QueryIntent.CONFIGURATION == "configuration"

    def test_all_intents_count(self):
        """Test that we have exactly 14 intent categories."""
        all_intents = list(QueryIntent)
        assert len(all_intents) == 14


class TestQueryComplexity:
    """Test the QueryComplexity enum."""

    def test_complexity_levels(self):
        """Test complexity level values."""
        assert QueryComplexity.SIMPLE == "simple"
        assert QueryComplexity.MODERATE == "moderate"
        assert QueryComplexity.COMPLEX == "complex"
        assert QueryComplexity.EXPERT == "expert"


class TestSearchStrategy:
    """Test the SearchStrategy enum."""

    def test_strategy_values(self):
        """Test search strategy values."""
        assert SearchStrategy.SEMANTIC == "semantic"
        assert SearchStrategy.HYBRID == "hybrid"
        assert SearchStrategy.HYDE == "hyde"
        assert SearchStrategy.MULTI_STAGE == "multi_stage"
        assert SearchStrategy.FILTERED == "filtered"
        assert SearchStrategy.RERANKED == "reranked"
        assert SearchStrategy.ADAPTIVE == "adaptive"


class TestMatryoshkaDimension:
    """Test the MatryoshkaDimension enum."""

    def test_dimension_values(self):
        """Test dimension values."""
        assert MatryoshkaDimension.SMALL.value == 512
        assert MatryoshkaDimension.MEDIUM.value == 768
        assert MatryoshkaDimension.LARGE.value == 1536


class TestQueryIntentClassification:
    """Test the QueryIntentClassification model."""

    def test_valid_classification(self):
        """Test creating a valid classification."""
        classification = QueryIntentClassification(
            primary_intent=QueryIntent.CONCEPTUAL,
            secondary_intents=[QueryIntent.PROCEDURAL],
            confidence_scores={
                QueryIntent.CONCEPTUAL: 0.8,
                QueryIntent.PROCEDURAL: 0.3,
            },
            complexity_level=QueryComplexity.MODERATE,
            classification_reasoning="Test reasoning",
        )

        assert classification.primary_intent == QueryIntent.CONCEPTUAL
        assert len(classification.secondary_intents) == 1
        assert classification.confidence_scores[QueryIntent.CONCEPTUAL] == 0.8
        assert classification.complexity_level == QueryComplexity.MODERATE

    def test_optional_fields(self):
        """Test optional fields have defaults."""
        classification = QueryIntentClassification(
            primary_intent=QueryIntent.FACTUAL,
            secondary_intents=[],
            confidence_scores={QueryIntent.FACTUAL: 0.9},
            complexity_level=QueryComplexity.SIMPLE,
            classification_reasoning="Simple factual query",
        )

        assert classification.domain_category is None
        assert classification.requires_context is False
        assert classification.suggested_followups == []


class TestQueryPreprocessingResult:
    """Test the QueryPreprocessingResult model."""

    def test_valid_preprocessing_result(self):
        """Test creating a valid preprocessing result."""
        result = QueryPreprocessingResult(
            original_query="what is phython?",
            processed_query="what is python?",
            corrections_applied=["phython → python"],
            expansions_added=["py → python"],
            normalization_applied=True,
            context_extracted={"programming_language": ["python"]},
            preprocessing_time_ms=15.5,
        )

        assert result.original_query == "what is phython?"
        assert result.processed_query == "what is python?"
        assert len(result.corrections_applied) == 1
        assert result.preprocessing_time_ms == 15.5

    def test_defaults(self):
        """Test default values."""
        result = QueryPreprocessingResult(
            original_query="test",
            processed_query="test",
        )

        assert result.corrections_applied == []
        assert result.expansions_added == []
        assert result.normalization_applied is False
        assert result.context_extracted == {}
        assert result.preprocessing_time_ms == 0.0


class TestSearchStrategySelection:
    """Test the SearchStrategySelection model."""

    def test_valid_strategy_selection(self):
        """Test creating a valid strategy selection."""
        selection = SearchStrategySelection(
            primary_strategy=SearchStrategy.HYDE,
            fallback_strategies=[SearchStrategy.SEMANTIC, SearchStrategy.HYBRID],
            matryoshka_dimension=MatryoshkaDimension.LARGE,
            confidence=0.85,
            reasoning="HyDE works well for procedural queries",
            estimated_quality=0.9,
            estimated_latency_ms=200.0,
        )

        assert selection.primary_strategy == SearchStrategy.HYDE
        assert len(selection.fallback_strategies) == 2
        assert selection.matryoshka_dimension == MatryoshkaDimension.LARGE
        assert selection.confidence == 0.85

    def test_confidence_validation(self):
        """Test confidence score validation."""
        with pytest.raises(ValidationError):
            SearchStrategySelection(
                primary_strategy=SearchStrategy.SEMANTIC,
                fallback_strategies=[],
                matryoshka_dimension=MatryoshkaDimension.MEDIUM,
                confidence=1.5,  # Invalid: > 1.0
                reasoning="Test",
                estimated_quality=0.8,
                estimated_latency_ms=100.0,
            )


class TestQueryProcessingRequest:
    """Test the QueryProcessingRequest model."""

    def test_minimal_valid_request(self):
        """Test creating a minimal valid request."""
        request = QueryProcessingRequest(
            query="What is machine learning?",
            collection_name="docs",
            limit=10,
        )

        assert request.query == "What is machine learning?"
        assert request.collection_name == "docs"
        assert request.limit == 10
        assert request.enable_preprocessing is True
        assert request.enable_intent_classification is True

    def test_full_request_configuration(self):
        """Test request with all options."""
        request = QueryProcessingRequest(
            query="How to optimize database performance?",
            collection_name="documentation",
            limit=20,
            enable_preprocessing=False,
            enable_intent_classification=True,
            enable_strategy_selection=True,
            force_strategy=SearchStrategy.RERANKED,
            force_dimension=MatryoshkaDimension.LARGE,
            user_context={"urgency": "high"},
            filters={"category": "database"},
            max_processing_time_ms=5000,
        )

        assert request.enable_preprocessing is False
        assert request.force_strategy == SearchStrategy.RERANKED
        assert request.user_context["urgency"] == "high"
        assert request.max_processing_time_ms == 5000

    def test_query_validation(self):
        """Test query string validation."""
        with pytest.raises(ValidationError):
            QueryProcessingRequest(
                query="",  # Empty query should fail
                collection_name="docs",
                limit=10,
            )

    def test_limit_validation(self):
        """Test limit validation."""
        with pytest.raises(ValidationError):
            QueryProcessingRequest(
                query="test query",
                collection_name="docs",
                limit=0,  # Invalid limit
            )


class TestQueryProcessingResponse:
    """Test the QueryProcessingResponse model."""

    def test_successful_response(self):
        """Test creating a successful response."""
        response = QueryProcessingResponse(
            success=True,
            results=[{"id": "1", "content": "test", "score": 0.9}],
            _total_results=1,
            _total_processing_time_ms=150.5,
            confidence_score=0.85,
            quality_score=0.9,
        )

        assert response.success is True
        assert len(response.results) == 1
        assert response._total_results == 1
        assert response.confidence_score == 0.85

    def test_error_response(self):
        """Test creating an error response."""
        response = QueryProcessingResponse(
            success=False,
            results=[],
            _total_results=0,
            error="Processing failed",
        )

        assert response.success is False
        assert response.results == []
        assert response.error == "Processing failed"

    def test_response_with_processing_details(self):
        """Test response with processing details."""
        intent_classification = QueryIntentClassification(
            primary_intent=QueryIntent.PERFORMANCE,
            secondary_intents=[QueryIntent.TROUBLESHOOTING],
            confidence_scores={QueryIntent.PERFORMANCE: 0.9},
            complexity_level=QueryComplexity.COMPLEX,
            classification_reasoning="Performance optimization query",
        )

        preprocessing_result = QueryPreprocessingResult(
            original_query="optimize db performance",
            processed_query="optimize database performance",
            corrections_applied=[],
            expansions_added=["db → database"],
        )

        response = QueryProcessingResponse(
            success=True,
            results=[],
            _total_results=0,
            intent_classification=intent_classification,
            preprocessing_result=preprocessing_result,
            processing_steps=["preprocessing", "intent_classification", "search"],
            fallback_used=False,
        )

        assert response.intent_classification.primary_intent == QueryIntent.PERFORMANCE
        assert (
            response.preprocessing_result.processed_query
            == "optimize database performance"
        )
        assert len(response.processing_steps) == 3
        assert response.fallback_used is False

    def test_defaults(self):
        """Test default field values."""
        response = QueryProcessingResponse(
            success=True,
            results=[],
            _total_results=0,
        )

        assert response._total_processing_time_ms == 0.0
        assert response.search_time_ms == 0.0
        assert response.confidence_score == 0.0
        assert response.quality_score == 0.0
        assert response.processing_steps == []
        assert response.fallback_used is False
        assert response.cache_hit is False
        assert response.error is None
