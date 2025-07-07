"""Tests for ResponseConverter.

- Real-world functionality focus
- Proper data validation
- Zero flaky tests
- Modern pytest patterns
"""

from unittest.mock import Mock

import pytest

from src.mcp_tools.models.responses import (
    AdvancedQueryProcessingResponse,
    QueryIntentResult,
    QueryPreprocessingResult,
    SearchResult,
    SearchStrategyResult,
)
from src.mcp_tools.tools.helpers.response_converter import ResponseConverter
from src.services.query_processing.models import (
    MatryoshkaDimension,
    QueryComplexity,
    QueryIntent,
    SearchStrategy,
)


@pytest.fixture
def response_converter():
    """Create response converter instance."""
    return ResponseConverter()


@pytest.fixture
def sample_intent_data():
    """Create realistic intent classification data."""
    mock_intent = Mock()
    mock_intent.primary_intent = QueryIntent.PROCEDURAL
    mock_intent.secondary_intents = [QueryIntent.FACTUAL]
    mock_intent.confidence_scores = {
        QueryIntent.PROCEDURAL: 0.85,
        QueryIntent.FACTUAL: 0.65,
    }
    mock_intent.complexity_level = QueryComplexity.MODERATE
    mock_intent.domain_category = "web_development"
    mock_intent.classification_reasoning = "Procedural query about implementation"
    mock_intent.requires_context = False
    mock_intent.suggested_followups = ["What are best practices?", "How to test?"]
    return mock_intent


@pytest.fixture
def sample_preprocessing_data():
    """Create realistic preprocessing data."""
    mock_preprocessing = Mock()
    mock_preprocessing.original_query = "How to implement authentication in FastAPI?"
    mock_preprocessing.processed_query = "implement authentication fastapi"
    mock_preprocessing.corrections_applied = ["FastAPI"]
    mock_preprocessing.expansions_added = ["auth", "security"]
    mock_preprocessing.normalization_applied = True
    mock_preprocessing.context_extracted = {"framework": "web", "language": "python"}
    mock_preprocessing.preprocessing_time_ms = 25.5
    return mock_preprocessing


@pytest.fixture
def sample_strategy_data():
    """Create realistic strategy selection data."""
    mock_strategy = Mock()
    mock_strategy.primary_strategy = SearchStrategy.HYBRID
    mock_strategy.fallback_strategies = [SearchStrategy.SEMANTIC]
    mock_strategy.matryoshka_dimension = MatryoshkaDimension.MEDIUM
    mock_strategy.confidence = 0.88
    mock_strategy.reasoning = "Hybrid search for procedural query"
    mock_strategy.estimated_quality = 0.92
    mock_strategy.estimated_latency_ms = 200.0
    return mock_strategy


@pytest.fixture
def sample_search_results():
    """Create realistic search results data."""
    return [
        {
            "id": "doc1",
            "content": "FastAPI authentication implementation guide",
            "score": 0.95,
            "url": "https://fastapi.tiangolo.com/tutorial/security/",
            "title": "FastAPI Security Tutorial",
            "metadata": {
                "type": "documentation",
                "category": "security",
                "language": "python",
            },
        },
        {
            "id": "doc2",
            "content": "JWT token implementation in FastAPI",
            "score": 0.88,
            "url": "https://example.com/jwt-fastapi",
            "title": "JWT with FastAPI",
            "metadata": {
                "type": "tutorial",
                "category": "auth",
                "difficulty": "intermediate",
            },
        },
        {
            # Minimal result to test defaults
            "content": "Basic auth example",
            "score": 0.75,
        },
    ]


class TestIntentClassificationConversion:
    """Test intent classification conversion functionality."""

    def test_intent_classification_success(
        self, response_converter, sample_intent_data
    ):
        """Test successful intent classification conversion."""
        result = response_converter.convert_intent_classification(sample_intent_data)

        assert isinstance(result, QueryIntentResult)
        assert result.primary_intent == "procedural"
        assert "factual" in result.secondary_intents
        assert result.confidence_scores["procedural"] == 0.85
        assert result.confidence_scores["factual"] == 0.65
        assert result.complexity_level == "moderate"
        assert result.domain_category == "web_development"
        assert (
            result.classification_reasoning == "Procedural query about implementation"
        )
        assert result.requires_context is False
        assert len(result.suggested_followups) == 2

    def test_intent_classification_empty_secondaries(self, response_converter):
        """Test intent classification with no secondary intents."""
        mock_intent = Mock()
        mock_intent.primary_intent = QueryIntent.FACTUAL
        mock_intent.secondary_intents = []
        mock_intent.confidence_scores = {QueryIntent.FACTUAL: 0.9}
        mock_intent.complexity_level = QueryComplexity.SIMPLE
        mock_intent.domain_category = "general"
        mock_intent.classification_reasoning = "Simple factual query"
        mock_intent.requires_context = False
        mock_intent.suggested_followups = []

        result = response_converter.convert_intent_classification(mock_intent)

        assert result.primary_intent == "factual"
        assert result.secondary_intents == []
        assert result.complexity_level == "simple"

    def test_intent_classification_none_input(self, response_converter):
        """Test intent classification with None input."""
        result = response_converter.convert_intent_classification(None)
        assert result is None


class TestPreprocessingResultConversion:
    """Test preprocessing result conversion functionality."""

    def test_preprocessing_success(self, response_converter, sample_preprocessing_data):
        """Test successful preprocessing result conversion."""
        result = response_converter.convert_preprocessing_result(
            sample_preprocessing_data
        )

        assert isinstance(result, QueryPreprocessingResult)
        assert result.original_query == "How to implement authentication in FastAPI?"
        assert result.processed_query == "implement authentication fastapi"
        assert "FastAPI" in result.corrections_applied
        assert "auth" in result.expansions_added
        assert "security" in result.expansions_added
        assert result.normalization_applied is True
        assert result.context_extracted["framework"] == "web"
        assert result.context_extracted["language"] == "python"
        assert result.preprocessing_time_ms == 25.5

    def test_preprocessing_minimal_data(self, response_converter):
        """Test preprocessing with minimal data."""
        mock_preprocessing = Mock()
        mock_preprocessing.original_query = "test query"
        mock_preprocessing.processed_query = "test query"
        mock_preprocessing.corrections_applied = []
        mock_preprocessing.expansions_added = []
        mock_preprocessing.normalization_applied = False
        mock_preprocessing.context_extracted = {}
        mock_preprocessing.preprocessing_time_ms = 0.0

        result = response_converter.convert_preprocessing_result(mock_preprocessing)

        assert result.original_query == "test query"
        assert result.corrections_applied == []
        assert result.normalization_applied is False
        assert result.context_extracted == {}

    def test_preprocessing_none_input(self, response_converter):
        """Test preprocessing with None input."""
        result = response_converter.convert_preprocessing_result(None)
        assert result is None


class TestStrategySelectionConversion:
    """Test strategy selection conversion functionality."""

    def test_strategy_selection_success(self, response_converter, sample_strategy_data):
        """Test successful strategy selection conversion."""
        result = response_converter.convert_strategy_selection(sample_strategy_data)

        assert isinstance(result, SearchStrategyResult)
        assert result.primary_strategy == "hybrid"
        assert "semantic" in result.fallback_strategies
        assert result.matryoshka_dimension == 768  # MatryoshkaDimension.MEDIUM value
        assert result.confidence == 0.88
        assert result.reasoning == "Hybrid search for procedural query"
        assert result.estimated_quality == 0.92
        assert result.estimated_latency_ms == 200.0

    def test_strategy_selection_no_fallbacks(self, response_converter):
        """Test strategy selection with no fallback strategies."""
        mock_strategy = Mock()
        mock_strategy.primary_strategy = SearchStrategy.SEMANTIC
        mock_strategy.fallback_strategies = []
        mock_strategy.matryoshka_dimension = MatryoshkaDimension.LARGE
        mock_strategy.confidence = 0.95
        mock_strategy.reasoning = "Simple semantic search"
        mock_strategy.estimated_quality = 0.85
        mock_strategy.estimated_latency_ms = 100.0

        result = response_converter.convert_strategy_selection(mock_strategy)

        assert result.primary_strategy == "semantic"
        assert result.fallback_strategies == []
        assert result.matryoshka_dimension == 1536  # MatryoshkaDimension.LARGE value

    def test_strategy_selection_none_input(self, response_converter):
        """Test strategy selection with None input."""
        result = response_converter.convert_strategy_selection(None)
        assert result is None


class TestSearchResultsConversion:
    """Test search results conversion functionality."""

    def test_search_results_with_analytics(
        self, response_converter, sample_search_results
    ):
        """Test search results conversion with analytics enabled."""
        results = response_converter.convert_search_results(
            sample_search_results, include_analytics=True
        )

        assert len(results) == 3

        # Test first result (complete data)
        result1 = results[0]
        assert isinstance(result1, SearchResult)
        assert result1.id == "doc1"
        assert result1.content == "FastAPI authentication implementation guide"
        assert result1.score == 0.95
        assert result1.url == "https://fastapi.tiangolo.com/tutorial/security/"
        assert result1.title == "FastAPI Security Tutorial"
        assert result1.metadata is not None
        assert result1.metadata["type"] == "documentation"

        # Test second result
        result2 = results[1]
        assert result2.id == "doc2"
        assert result2.metadata["difficulty"] == "intermediate"

        # Test minimal result (tests defaults)
        result3 = results[2]
        assert result3.content == "Basic auth example"
        assert result3.score == 0.75
        assert result3.id is not None  # Generated UUID when no ID provided
        assert len(result3.id) > 0  # Should be a valid UUID string
        assert result3.url is None
        assert result3.title is None

    def test_search_results_without_analytics(
        self, response_converter, sample_search_results
    ):
        """Test search results conversion without analytics."""
        results = response_converter.convert_search_results(
            sample_search_results, include_analytics=False
        )

        assert len(results) == 3

        # Verify metadata is excluded when analytics is disabled
        for result in results:
            assert result.metadata is None

    def test_search_results_empty_list(self, response_converter):
        """Test search results conversion with empty list."""
        results = response_converter.convert_search_results([])
        assert results == []


class TestCompleteResponseConversion:
    """Test complete MCP response conversion."""

    def test_complete_response_conversion(
        self,
        response_converter,
        sample_intent_data,
        sample_preprocessing_data,
        sample_strategy_data,
    ):
        """Test conversion of complete response with all components."""
        # Create mock complete response
        mock_response = Mock()
        mock_response.success = True
        mock_response.results = [
            {
                "id": "test1",
                "content": "Test content",
                "score": 0.9,
                "url": "https://test.com",
                "title": "Test Document",
                "metadata": {"category": "test"},
            }
        ]
        mock_response._total_results = 1
        mock_response._total_processing_time_ms = 150.5
        mock_response.search_time_ms = 120.0
        mock_response.strategy_selection_time_ms = 30.5
        mock_response.confidence_score = 0.88
        mock_response.quality_score = 0.92
        mock_response.processing_steps = ["preprocessing", "classification", "search"]
        mock_response.fallback_used = False
        mock_response.cache_hit = True
        mock_response.error = None
        mock_response.intent_classification = sample_intent_data
        mock_response.preprocessing_result = sample_preprocessing_data
        mock_response.strategy_selection = sample_strategy_data

        # Convert to MCP response
        result = response_converter.convert_to_mcp_response(
            mock_response, include_analytics=True
        )

        # Verify complete response structure
        assert isinstance(result, AdvancedQueryProcessingResponse)
        assert result.success is True
        assert result._total_results == 1
        assert len(result.results) == 1
        assert result._total_processing_time_ms == 150.5
        assert result.confidence_score == 0.88
        assert result.quality_score == 0.92
        assert result.fallback_used is False
        assert result.cache_hit is True
        assert result.error is None

        # Verify sub-components were converted
        assert result.intent_classification is not None
        assert result.intent_classification.primary_intent == "procedural"
        assert result.preprocessing_result is not None
        assert result.preprocessing_result.normalization_applied is True
        assert result.strategy_selection is not None
        assert result.strategy_selection.primary_strategy == "hybrid"

        # Verify search results with metadata (analytics enabled)
        assert result.results[0].metadata is not None

    def test_error_response_conversion(self, response_converter):
        """Test conversion of error response."""
        mock_response = Mock()
        mock_response.success = False
        mock_response.results = []
        mock_response._total_results = 0
        mock_response._total_processing_time_ms = 0.0
        mock_response.search_time_ms = 0.0
        mock_response.strategy_selection_time_ms = 0.0
        mock_response.confidence_score = 0.0
        mock_response.quality_score = 0.0
        mock_response.processing_steps = []
        mock_response.fallback_used = False
        mock_response.cache_hit = False
        mock_response.error = "Processing failed"
        mock_response.intent_classification = None
        mock_response.preprocessing_result = None
        mock_response.strategy_selection = None

        result = response_converter.convert_to_mcp_response(
            mock_response, include_analytics=False
        )

        assert result.success is False
        assert result._total_results == 0
        assert result.error == "Processing failed"
        assert result.intent_classification is None
        assert result.preprocessing_result is None
        assert result.strategy_selection is None
