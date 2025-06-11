"""Tests for search strategy selector."""

import pytest
from src.services.query_processing.models import MatryoshkaDimension
from src.services.query_processing.models import QueryComplexity
from src.services.query_processing.models import QueryIntent
from src.services.query_processing.models import QueryIntentClassification
from src.services.query_processing.models import SearchStrategy
from src.services.query_processing.models import SearchStrategySelection
from src.services.query_processing.strategy_selector import SearchStrategySelector


@pytest.fixture
def strategy_selector():
    """Create a strategy selector instance."""
    return SearchStrategySelector()


@pytest.fixture
async def initialized_selector(strategy_selector):
    """Create an initialized strategy selector."""
    await strategy_selector.initialize()
    return strategy_selector


@pytest.fixture
def sample_intent_classification():
    """Create a sample intent classification."""
    return QueryIntentClassification(
        primary_intent=QueryIntent.CONCEPTUAL,
        secondary_intents=[QueryIntent.PROCEDURAL],
        confidence_scores={
            QueryIntent.CONCEPTUAL: 0.8,
            QueryIntent.PROCEDURAL: 0.3,
        },
        complexity_level=QueryComplexity.MODERATE,
        classification_reasoning="Test classification",
    )


class TestSearchStrategySelector:
    """Test the SearchStrategySelector class."""

    def test_initialization(self, strategy_selector):
        """Test strategy selector initialization."""
        assert strategy_selector._initialized is False
        assert len(strategy_selector._intent_strategy_map) == 14  # All 14 intents
        assert (
            len(strategy_selector._complexity_adjustments) == 4
        )  # All complexity levels

    async def test_initialize(self, strategy_selector):
        """Test strategy selector initialization."""
        await strategy_selector.initialize()
        assert strategy_selector._initialized is True

    async def test_strategy_mapping_coverage(self, initialized_selector):
        """Test that all intents have strategy mappings."""
        for intent in QueryIntent:
            assert intent in initialized_selector._intent_strategy_map
            config = initialized_selector._intent_strategy_map[intent]
            assert "primary" in config
            assert "fallbacks" in config
            assert "dimension" in config
            assert "reasoning" in config

    async def test_conceptual_intent_strategy(
        self, initialized_selector, sample_intent_classification
    ):
        """Test strategy selection for conceptual intent."""
        classification = sample_intent_classification
        classification.primary_intent = QueryIntent.CONCEPTUAL

        selection = await initialized_selector.select_strategy(classification)

        assert isinstance(selection, SearchStrategySelection)
        assert selection.primary_strategy == SearchStrategy.SEMANTIC
        assert selection.matryoshka_dimension == MatryoshkaDimension.MEDIUM
        assert selection.confidence > 0.0
        assert "semantic understanding" in selection.reasoning.lower()

    async def test_procedural_intent_strategy(self, initialized_selector):
        """Test strategy selection for procedural intent."""
        classification = QueryIntentClassification(
            primary_intent=QueryIntent.PROCEDURAL,
            secondary_intents=[],
            confidence_scores={QueryIntent.PROCEDURAL: 0.9},
            complexity_level=QueryComplexity.MODERATE,
            classification_reasoning="Procedural query",
        )

        selection = await initialized_selector.select_strategy(classification)

        assert selection.primary_strategy == SearchStrategy.HYDE
        assert selection.matryoshka_dimension == MatryoshkaDimension.LARGE
        assert "hypothetical document generation" in selection.reasoning.lower()

    async def test_factual_intent_strategy(self, initialized_selector):
        """Test strategy selection for factual intent."""
        classification = QueryIntentClassification(
            primary_intent=QueryIntent.FACTUAL,
            secondary_intents=[],
            confidence_scores={QueryIntent.FACTUAL: 0.85},
            complexity_level=QueryComplexity.SIMPLE,
            classification_reasoning="Factual query",
        )

        selection = await initialized_selector.select_strategy(classification)

        assert selection.primary_strategy == SearchStrategy.SEMANTIC
        assert selection.matryoshka_dimension == MatryoshkaDimension.SMALL
        assert "keyword + semantic matching" in selection.reasoning.lower()

    async def test_troubleshooting_intent_strategy(self, initialized_selector):
        """Test strategy selection for troubleshooting intent."""
        classification = QueryIntentClassification(
            primary_intent=QueryIntent.TROUBLESHOOTING,
            secondary_intents=[],
            confidence_scores={QueryIntent.TROUBLESHOOTING: 0.9},
            complexity_level=QueryComplexity.COMPLEX,
            classification_reasoning="Troubleshooting query",
        )

        selection = await initialized_selector.select_strategy(classification)

        assert selection.primary_strategy == SearchStrategy.RERANKED
        assert selection.matryoshka_dimension == MatryoshkaDimension.LARGE
        assert "reranking for relevance" in selection.reasoning.lower()

    async def test_advanced_intent_strategies(self, initialized_selector):
        """Test strategy selection for advanced intent categories."""
        test_cases = [
            (QueryIntent.COMPARATIVE, SearchStrategy.MULTI_STAGE),
            (QueryIntent.ARCHITECTURAL, SearchStrategy.HYDE),
            (QueryIntent.PERFORMANCE, SearchStrategy.RERANKED),
            (QueryIntent.SECURITY, SearchStrategy.FILTERED),
            (QueryIntent.INTEGRATION, SearchStrategy.HYBRID),
            (QueryIntent.BEST_PRACTICES, SearchStrategy.RERANKED),
            (QueryIntent.CODE_REVIEW, SearchStrategy.MULTI_STAGE),
            (QueryIntent.MIGRATION, SearchStrategy.HYDE),
            (QueryIntent.DEBUGGING, SearchStrategy.FILTERED),
            (QueryIntent.CONFIGURATION, SearchStrategy.FILTERED),
        ]

        for intent, expected_strategy in test_cases:
            classification = QueryIntentClassification(
                primary_intent=intent,
                secondary_intents=[],
                confidence_scores={intent: 0.8},
                complexity_level=QueryComplexity.MODERATE,
                classification_reasoning=f"{intent.value} query",
            )

            selection = await initialized_selector.select_strategy(classification)

            assert selection.primary_strategy == expected_strategy, (
                f"Intent {intent} should use {expected_strategy}"
            )

    async def test_complexity_adjustments(self, initialized_selector):
        """Test complexity-based strategy adjustments."""
        base_classification = QueryIntentClassification(
            primary_intent=QueryIntent.CONCEPTUAL,
            secondary_intents=[],
            confidence_scores={QueryIntent.CONCEPTUAL: 0.8},
            complexity_level=QueryComplexity.MODERATE,
            classification_reasoning="Test query",
        )

        # Test simple complexity
        simple_classification = base_classification.model_copy()
        simple_classification.complexity_level = QueryComplexity.SIMPLE
        simple_selection = await initialized_selector.select_strategy(
            simple_classification
        )

        # Test expert complexity
        expert_classification = base_classification.model_copy()
        expert_classification.complexity_level = QueryComplexity.EXPERT
        expert_selection = await initialized_selector.select_strategy(
            expert_classification
        )

        # Expert queries should have larger dimensions or additional fallbacks
        assert (
            expert_selection.matryoshka_dimension.value
            >= simple_selection.matryoshka_dimension.value
        )
        assert len(expert_selection.fallback_strategies) >= len(
            simple_selection.fallback_strategies
        )

    async def test_performance_requirements_latency(
        self, initialized_selector, sample_intent_classification
    ):
        """Test performance requirements for latency constraints."""
        performance_requirements = {"max_latency_ms": 50}

        selection = await initialized_selector.select_strategy(
            sample_intent_classification,
            performance_requirements=performance_requirements,
        )

        # Should prefer faster strategies when latency is constrained
        fast_strategies = [SearchStrategy.SEMANTIC, SearchStrategy.FILTERED]
        assert selection.primary_strategy in fast_strategies
        # Check that latency consideration is reflected in estimated latency
        assert selection.estimated_latency_ms <= 100  # Should be optimized for speed

    async def test_performance_requirements_quality(
        self, initialized_selector, sample_intent_classification
    ):
        """Test performance requirements for quality constraints."""
        performance_requirements = {"min_quality": 0.9}

        selection = await initialized_selector.select_strategy(
            sample_intent_classification,
            performance_requirements=performance_requirements,
        )

        # Should prefer higher quality strategies
        high_quality_strategies = [SearchStrategy.HYDE, SearchStrategy.RERANKED]
        assert selection.primary_strategy in high_quality_strategies or any(
            strategy in high_quality_strategies
            for strategy in selection.fallback_strategies
        )
        assert "quality" in selection.reasoning.lower()

    async def test_context_adjustments(
        self, initialized_selector, sample_intent_classification
    ):
        """Test context-based strategy adjustments."""
        context_cases = [
            (
                {"programming_language": ["python", "javascript"]},
                SearchStrategy.SEMANTIC,
            ),
            ({"framework": ["react", "django"]}, SearchStrategy.FILTERED),
            ({"error_code": ["404", "500"]}, SearchStrategy.FILTERED),
            ({"urgency": "high"}, SearchStrategy.SEMANTIC),
        ]

        for context, expected_strategy_type in context_cases:
            selection = await initialized_selector.select_strategy(
                sample_intent_classification, context=context
            )

            # Should include the expected strategy in primary or fallbacks
            all_strategies = [
                selection.primary_strategy,
                *selection.fallback_strategies,
            ]
            if expected_strategy_type == SearchStrategy.SEMANTIC:
                assert SearchStrategy.SEMANTIC in all_strategies
            elif expected_strategy_type == SearchStrategy.FILTERED:
                assert SearchStrategy.FILTERED in all_strategies

    async def test_secondary_intents_incorporation(self, initialized_selector):
        """Test incorporation of secondary intents into fallback strategies."""
        classification = QueryIntentClassification(
            primary_intent=QueryIntent.CONCEPTUAL,
            secondary_intents=[QueryIntent.TROUBLESHOOTING, QueryIntent.PERFORMANCE],
            confidence_scores={
                QueryIntent.CONCEPTUAL: 0.7,
                QueryIntent.TROUBLESHOOTING: 0.4,
                QueryIntent.PERFORMANCE: 0.3,
            },
            complexity_level=QueryComplexity.MODERATE,
            classification_reasoning="Multi-intent query",
        )

        selection = await initialized_selector.select_strategy(classification)

        # Should include strategies from secondary intents in fallbacks
        troubleshooting_strategy = SearchStrategy.RERANKED
        performance_strategy = SearchStrategy.RERANKED

        all_strategies = [selection.primary_strategy, *selection.fallback_strategies]
        assert (
            troubleshooting_strategy in all_strategies
            or performance_strategy in all_strategies
        )

    async def test_confidence_calculation(
        self, initialized_selector, sample_intent_classification
    ):
        """Test confidence score calculation."""
        # High confidence intent classification
        high_confidence_classification = sample_intent_classification.model_copy()
        high_confidence_classification.confidence_scores[QueryIntent.CONCEPTUAL] = 0.95

        high_confidence_selection = await initialized_selector.select_strategy(
            high_confidence_classification
        )

        # Low confidence intent classification
        low_confidence_classification = sample_intent_classification.model_copy()
        low_confidence_classification.confidence_scores[QueryIntent.CONCEPTUAL] = 0.3

        low_confidence_selection = await initialized_selector.select_strategy(
            low_confidence_classification
        )

        # Higher intent confidence should result in higher strategy confidence
        assert (
            high_confidence_selection.confidence > low_confidence_selection.confidence
        )

    async def test_estimated_metrics(
        self, initialized_selector, sample_intent_classification
    ):
        """Test estimated quality and latency metrics."""
        selection = await initialized_selector.select_strategy(
            sample_intent_classification
        )

        # Should provide reasonable estimates
        assert 0.0 <= selection.estimated_quality <= 1.0
        assert selection.estimated_latency_ms > 0
        assert selection.estimated_latency_ms < 1000  # Should be reasonable

    async def test_fallback_strategy_limits(self, initialized_selector):
        """Test that fallback strategies are properly limited."""
        classification = QueryIntentClassification(
            primary_intent=QueryIntent.CONCEPTUAL,
            secondary_intents=[
                QueryIntent.PROCEDURAL,
                QueryIntent.FACTUAL,
                QueryIntent.TROUBLESHOOTING,
            ],
            confidence_scores={
                QueryIntent.CONCEPTUAL: 0.7,
                QueryIntent.PROCEDURAL: 0.5,
                QueryIntent.FACTUAL: 0.4,
                QueryIntent.TROUBLESHOOTING: 0.3,
            },
            complexity_level=QueryComplexity.SIMPLE,  # Simple should limit fallbacks
            classification_reasoning="Multi-intent query",
        )

        selection = await initialized_selector.select_strategy(classification)

        # Simple complexity should limit fallbacks (allow flexibility in implementation)
        assert (
            len(selection.fallback_strategies) <= 3
        )  # Allow reasonable number of fallbacks

    async def test_dimension_boost_for_complexity(self, initialized_selector):
        """Test dimension boost for complex queries."""
        base_classification = QueryIntentClassification(
            primary_intent=QueryIntent.FACTUAL,  # Normally uses SMALL dimension
            secondary_intents=[],
            confidence_scores={QueryIntent.FACTUAL: 0.8},
            complexity_level=QueryComplexity.MODERATE,
            classification_reasoning="Test query",
        )

        # Simple complexity
        simple_classification = base_classification.model_copy()
        simple_classification.complexity_level = QueryComplexity.SIMPLE
        simple_selection = await initialized_selector.select_strategy(
            simple_classification
        )

        # Complex complexity
        complex_classification = base_classification.model_copy()
        complex_classification.complexity_level = QueryComplexity.COMPLEX
        complex_selection = await initialized_selector.select_strategy(
            complex_classification
        )

        # Complex should have same or larger dimension
        assert (
            complex_selection.matryoshka_dimension.value
            >= simple_selection.matryoshka_dimension.value
        )

    async def test_strategy_performance_estimates(self, strategy_selector):
        """Test strategy performance estimate consistency."""
        # All strategies should have performance estimates
        for _strategy, perf in strategy_selector._strategy_performance.items():
            assert "latency" in perf
            assert "quality" in perf
            assert 0 < perf["latency"] < 1000  # Reasonable latency range
            assert 0.0 <= perf["quality"] <= 1.0  # Valid quality range

    async def test_uninitialized_selector_error(
        self, strategy_selector, sample_intent_classification
    ):
        """Test error when using uninitialized selector."""
        with pytest.raises(RuntimeError, match="not initialized"):
            await strategy_selector.select_strategy(sample_intent_classification)

    async def test_cleanup(self, initialized_selector):
        """Test strategy selector cleanup."""
        await initialized_selector.cleanup()
        assert initialized_selector._initialized is False

    async def test_reasoning_content(
        self, initialized_selector, sample_intent_classification
    ):
        """Test that reasoning provides meaningful explanations."""
        selection = await initialized_selector.select_strategy(
            sample_intent_classification
        )

        assert selection.reasoning is not None
        assert len(selection.reasoning) > 10  # Should be descriptive
        # Should mention the intent type
        assert (
            sample_intent_classification.primary_intent.value
            in selection.reasoning.lower()
        )

    async def test_dimension_multiplier_effects(
        self, initialized_selector, sample_intent_classification
    ):
        """Test that dimension affects latency and quality estimates."""
        # Force different dimensions to test multiplier effects
        classification_small = sample_intent_classification.model_copy()
        classification_small.primary_intent = (
            QueryIntent.FACTUAL
        )  # Uses SMALL dimension

        classification_large = sample_intent_classification.model_copy()
        classification_large.primary_intent = (
            QueryIntent.PROCEDURAL
        )  # Uses LARGE dimension

        small_selection = await initialized_selector.select_strategy(
            classification_small
        )
        large_selection = await initialized_selector.select_strategy(
            classification_large
        )

        # Larger dimensions should have higher latency and quality
        assert (
            large_selection.estimated_latency_ms >= small_selection.estimated_latency_ms
        )
        assert large_selection.estimated_quality >= small_selection.estimated_quality
