"""Tests for the model selector service.

This module contains comprehensive tests for the ModelSelector
including model selection, performance tracking, and cost optimization.
"""

from unittest.mock import MagicMock

import pytest

from src.config import Config
from src.config.enums import (
    EmbeddingModel,
    ModelType,
    OptimizationStrategy,
    QueryComplexity,
    QueryType,
)
from src.models.vector_search import ModelSelectionStrategy, QueryClassification
from src.services.vector_db.model_selector import ModelSelector


class TestModelSelector:
    """Test suite for ModelSelector."""

    @pytest.fixture
    def mock_config(self):
        """Mock unified configuration."""
        config = MagicMock(spec=Config)
        config.embedding_cost_budget = 1000.0
        return config

    @pytest.fixture
    def selector(self, mock_config):
        """Create ModelSelector instance."""
        return ModelSelector(mock_config)

    @pytest.fixture
    def sample_query_classification(self):
        """Create sample query classification."""
        return QueryClassification(
            query_type=QueryType.CODE,
            complexity_level=QueryComplexity.MODERATE,
            domain="programming",
            programming_language="python",
            is_multimodal=False,
            confidence=0.85,
            features={"has_code_keywords": True},
        )

    async def test_initialization(self, selector):
        """Test model selector initialization."""
        assert selector.config is not None
        assert len(selector.model_registry) > 0
        assert selector.cost_budget == 1000.0
        assert isinstance(selector.performance_history, dict)

    async def test_model_registry_structure(self, selector):
        """Test model registry structure and required fields."""
        for model_info in selector.model_registry.values():
            # Check required fields
            assert "type" in model_info
            assert "dimensions" in model_info
            assert "cost_per_1k_tokens" in model_info
            assert "latency_ms" in model_info
            assert "quality_score" in model_info
            assert "specializations" in model_info
            assert "provider" in model_info
            assert "description" in model_info

            # Check data types
            assert isinstance(model_info["type"], ModelType)
            assert isinstance(model_info["dimensions"], int)
            assert isinstance(model_info["cost_per_1k_tokens"], int | float)
            assert isinstance(model_info["latency_ms"], int | float)
            assert isinstance(model_info["quality_score"], int | float)
            assert isinstance(model_info["specializations"], list)

    @pytest.mark.parametrize(
        "query_type,expected_models",
        [
            (QueryType.CODE, ["code-search-net", EmbeddingModel.NV_EMBED_V2.value]),
            (QueryType.MULTIMODAL, ["clip-vit-base-patch32"]),
            (QueryType.CONCEPTUAL, [EmbeddingModel.TEXT_EMBEDDING_3_LARGE.value]),
            (
                QueryType.API_REFERENCE,
                ["code-search-net", EmbeddingModel.NV_EMBED_V2.value],
            ),
        ],
    )
    async def test_candidate_model_selection(
        self, selector, query_type, expected_models
    ):
        """Test candidate model selection based on query type."""
        classification = QueryClassification(
            query_type=query_type,
            complexity_level=QueryComplexity.MODERATE,
            domain="general",
            programming_language=None,
            is_multimodal=(query_type == QueryType.MULTIMODAL),
            confidence=0.8,
            features={},
        )

        candidates = selector._get_candidate_models(classification)

        # Check that expected models are in candidates
        for expected_model in expected_models:
            if expected_model in selector.model_registry:
                assert expected_model in candidates

    @pytest.mark.parametrize(
        "optimization_strategy,expected_weight_priority",
        [
            (OptimizationStrategy.QUALITY_OPTIMIZED, "quality_score"),
            (OptimizationStrategy.SPEED_OPTIMIZED, "latency_ms"),
            (OptimizationStrategy.COST_OPTIMIZED, "cost_per_1k_tokens"),
            (OptimizationStrategy.BALANCED, "balanced"),
        ],
    )
    async def test_optimization_strategy_weighting(
        self,
        selector,
        sample_query_classification,
        optimization_strategy,
        expected_weight_priority,
    ):
        """Test that optimization strategies apply correct weighting."""
        selection = await selector.select_optimal_model(
            sample_query_classification, optimization_strategy
        )

        assert isinstance(selection, ModelSelectionStrategy)
        assert selection.primary_model in selector.model_registry

        # Verify the selection rationale mentions the optimization focus
        if expected_weight_priority == "quality_score":
            assert "quality" in selection.selection_rationale.lower()
        elif expected_weight_priority == "latency_ms":
            assert (
                "speed" in selection.selection_rationale.lower()
                or "fastest" in selection.selection_rationale.lower()
            )
        elif expected_weight_priority == "cost_per_1k_tokens":
            assert "cost" in selection.selection_rationale.lower()

    async def test_specialization_scoring(self, selector):
        """Test specialization scoring for different query types."""
        # Code query should prefer code-specialized models
        code_classification = QueryClassification(
            query_type=QueryType.CODE,
            complexity_level=QueryComplexity.MODERATE,
            domain="programming",
            programming_language="python",
            is_multimodal=False,
            confidence=0.8,
            features={},
        )

        # Test specialization scoring
        code_model_info = selector.model_registry.get("code-search-net", {})
        general_model_info = selector.model_registry.get(
            EmbeddingModel.BGE_SMALL_EN_V15.value, {}
        )

        if code_model_info:
            code_score = selector._calculate_specialization_score(
                code_model_info, code_classification
            )
            general_score = selector._calculate_specialization_score(
                general_model_info, code_classification
            )
            assert code_score > general_score

    async def test_historical_performance_tracking(
        self, selector, sample_query_classification
    ):
        """Test historical performance tracking and retrieval."""
        model_id = EmbeddingModel.TEXT_EMBEDDING_3_SMALL.value
        performance_score = 0.85

        # Update performance history
        await selector.update_performance_history(
            model_id, sample_query_classification, performance_score
        )

        # Check that history was stored
        assert model_id in selector.performance_history

        # Retrieve historical performance
        historical_score = selector._get_historical_performance(
            model_id, sample_query_classification
        )
        assert historical_score > 0.5  # Should be influenced by the update

    async def test_weighted_score_calculation(self, selector):
        """Test weighted score calculation for different strategies."""
        quality_score = 0.9
        speed_score = 0.6
        cost_score = 0.8
        specialization_score = 0.7
        historical_score = 0.75

        # Test different strategies
        quality_weighted = selector._calculate_weighted_score(
            quality_score,
            speed_score,
            cost_score,
            specialization_score,
            historical_score,
            OptimizationStrategy.QUALITY_OPTIMIZED,
        )

        speed_weighted = selector._calculate_weighted_score(
            quality_score,
            speed_score,
            cost_score,
            specialization_score,
            historical_score,
            OptimizationStrategy.SPEED_OPTIMIZED,
        )

        # Quality-optimized should weight quality more heavily when quality is high
        # Speed-optimized should weight speed more when considering the same scores
        assert quality_weighted != speed_weighted

    async def test_ensemble_weights_calculation(self, selector):
        """Test ensemble weights calculation."""
        scored_candidates = [
            {"model_id": "model1", "total_score": 0.8},
            {"model_id": "model2", "total_score": 0.6},
            {"model_id": "model3", "total_score": 0.1},  # Below threshold
        ]

        weights = selector._calculate_ensemble_weights(scored_candidates)

        # Should include models above threshold
        assert "model1" in weights
        assert "model2" in weights
        # Should exclude model below threshold
        assert "model3" not in weights

        # Weights should sum to less than or equal to 1
        assert sum(weights.values()) <= 1.0

    async def test_fallback_strategy(self, selector, sample_query_classification):
        """Test fallback strategy when selection fails."""
        fallback = selector._get_fallback_strategy(sample_query_classification)

        assert isinstance(fallback, ModelSelectionStrategy)
        assert fallback.primary_model in selector.model_registry
        assert fallback.model_type == ModelType.GENERAL_PURPOSE
        assert "fallback" in fallback.selection_rationale.lower()

    async def test_cost_estimation(self, selector):
        """Test monthly cost estimation."""
        model_id = EmbeddingModel.TEXT_EMBEDDING_3_SMALL.value
        estimated_queries = 10000
        avg_tokens = 100

        cost = selector.estimate_monthly_cost(model_id, estimated_queries, avg_tokens)

        assert isinstance(cost, float)
        assert cost >= 0

    async def test_cost_optimized_recommendations(self, selector):
        """Test cost-optimized model recommendations."""
        monthly_budget = 50.0
        estimated_queries = 10000

        recommendations = selector.get_cost_optimized_recommendations(
            monthly_budget, estimated_queries
        )

        assert isinstance(recommendations, list)
        for rec in recommendations:
            assert rec["estimated_monthly_cost"] <= monthly_budget
            assert "cost_efficiency" in rec
            assert "quality_score" in rec

    async def test_model_info_retrieval(self, selector):
        """Test model information retrieval."""
        model_id = EmbeddingModel.TEXT_EMBEDDING_3_SMALL.value

        info = selector.get_model_info(model_id)

        assert info is not None
        assert "quality_score" in info
        assert "cost_per_1k_tokens" in info

    async def test_available_models_listing(self, selector):
        """Test listing available models."""
        # All models
        all_models = selector.list_available_models()
        assert len(all_models) > 0

        # Filter by type
        general_models = selector.list_available_models(ModelType.GENERAL_PURPOSE)
        selector.list_available_models(ModelType.CODE_SPECIALIZED)

        assert len(general_models) > 0
        for model in general_models:
            assert model["type"] == ModelType.GENERAL_PURPOSE

    async def test_complex_query_model_selection(self, selector):
        """Test model selection for complex queries."""
        complex_classification = QueryClassification(
            query_type=QueryType.CONCEPTUAL,
            complexity_level=QueryComplexity.COMPLEX,
            domain="general",
            programming_language=None,
            is_multimodal=False,
            confidence=0.9,
            features={},
        )

        selection = await selector.select_optimal_model(complex_classification)

        assert isinstance(selection, ModelSelectionStrategy)
        # Complex queries should prefer higher-quality models
        model_info = selector.model_registry[selection.primary_model]
        assert model_info["quality_score"] > 0.7

    async def test_simple_query_optimization(self, selector):
        """Test optimization for simple queries."""
        simple_classification = QueryClassification(
            query_type=QueryType.DOCUMENTATION,
            complexity_level=QueryComplexity.SIMPLE,
            domain="general",
            programming_language=None,
            is_multimodal=False,
            confidence=0.8,
            features={},
        )

        # Speed-optimized selection for simple query
        selection = await selector.select_optimal_model(
            simple_classification, OptimizationStrategy.SPEED_OPTIMIZED
        )

        assert isinstance(selection, ModelSelectionStrategy)
        # Should prefer faster models for simple queries
        model_info = selector.model_registry[selection.primary_model]
        assert model_info["latency_ms"] < 200  # Reasonable threshold

    async def test_multimodal_query_handling(self, selector):
        """Test handling of multimodal queries."""
        multimodal_classification = QueryClassification(
            query_type=QueryType.MULTIMODAL,
            complexity_level=QueryComplexity.MODERATE,
            domain="general",
            programming_language=None,
            is_multimodal=True,
            confidence=0.8,
            features={},
        )

        selection = await selector.select_optimal_model(multimodal_classification)

        assert isinstance(selection, ModelSelectionStrategy)
        # Should select multimodal-capable model if available
        model_info = selector.model_registry[selection.primary_model]
        assert model_info[
            "type"
        ] == ModelType.MULTIMODAL or "multimodal" in model_info.get(
            "specializations", []
        )

    async def test_performance_history_exponential_moving_average(
        self, selector, sample_query_classification
    ):
        """Test exponential moving average in performance history updates."""
        model_id = EmbeddingModel.TEXT_EMBEDDING_3_SMALL.value

        # First update
        await selector.update_performance_history(
            model_id, sample_query_classification, 0.8
        )
        first_score = selector.performance_history[model_id][
            f"{sample_query_classification.query_type}_{sample_query_classification.complexity_level}"
        ]

        # Second update with different score
        await selector.update_performance_history(
            model_id, sample_query_classification, 0.6
        )
        second_score = selector.performance_history[model_id][
            f"{sample_query_classification.query_type}_{sample_query_classification.complexity_level}"
        ]

        # Score should have moved toward the new value but not completely
        assert second_score != first_score
        assert second_score != 0.6  # Should be averaged with previous

    async def test_error_handling_in_model_selection(
        self, selector, sample_query_classification
    ):
        """Test error handling during model selection."""
        # Mock an error in candidate selection
        original_method = selector._get_candidate_models

        def raise_index_error(x):
            raise IndexError("Test error")

        selector._get_candidate_models = raise_index_error

        try:
            selection = await selector.select_optimal_model(sample_query_classification)
            assert isinstance(selection, ModelSelectionStrategy)
            assert "fallback" in selection.selection_rationale.lower()
        finally:
            selector._get_candidate_models = original_method

    async def test_selection_rationale_generation(
        self, selector, sample_query_classification
    ):
        """Test selection rationale generation."""
        selection = await selector.select_optimal_model(
            sample_query_classification, OptimizationStrategy.QUALITY_OPTIMIZED
        )

        assert isinstance(selection.selection_rationale, str)
        assert len(selection.selection_rationale) > 0
        assert selection.primary_model in selection.selection_rationale

    async def test_cost_efficiency_calculation(
        self, selector, sample_query_classification
    ):
        """Test cost efficiency calculation in model selection."""
        selection = await selector.select_optimal_model(sample_query_classification)

        assert hasattr(selection, "cost_efficiency")
        assert isinstance(selection.cost_efficiency, int | float)
        assert selection.cost_efficiency >= 0

    async def test_programming_language_specific_selection(self, selector):
        """Test model selection based on programming language."""
        python_classification = QueryClassification(
            query_type=QueryType.CODE,
            complexity_level=QueryComplexity.MODERATE,
            domain="programming",
            programming_language="python",
            is_multimodal=False,
            confidence=0.8,
            features={},
        )

        selection = await selector.select_optimal_model(python_classification)

        assert isinstance(selection, ModelSelectionStrategy)
        # Should prefer code-specialized models for programming queries
        model_info = selector.model_registry[selection.primary_model]
        specializations = model_info.get("specializations", [])
        assert (
            "code" in specializations
            or model_info["type"] == ModelType.CODE_SPECIALIZED
            or "programming" in specializations
        )

    async def test_model_fallback_list(self, selector, sample_query_classification):
        """Test that fallback models are provided."""
        selection = await selector.select_optimal_model(sample_query_classification)

        assert isinstance(selection.fallback_models, list)
        assert len(selection.fallback_models) <= 2  # As per implementation
        for fallback_model in selection.fallback_models:
            assert fallback_model in selector.model_registry

    async def test_empty_candidates_handling(
        self, selector, sample_query_classification
    ):
        """Test handling when no candidates are found."""
        # Mock empty candidates
        original_method = selector._get_candidate_models
        selector._get_candidate_models = lambda x: []

        try:
            selection = await selector.select_optimal_model(sample_query_classification)
            # Should still return a valid selection (fallback)
            assert isinstance(selection, ModelSelectionStrategy)
        finally:
            selector._get_candidate_models = original_method

    @pytest.mark.parametrize(
        "model_type",
        [ModelType.GENERAL_PURPOSE, ModelType.CODE_SPECIALIZED, ModelType.MULTIMODAL],
    )
    async def test_model_type_filtering(self, selector, model_type):
        """Test model listing with type filtering."""
        models = selector.list_available_models(model_type)

        for model in models:
            assert model["type"] == model_type
