"""Tests for the adaptive fusion tuner service.

This module contains comprehensive tests for the AdaptiveFusionTuner
including effectiveness scoring, multi-armed bandit optimization, and user feedback integration.
"""

import time
import uuid
from unittest.mock import MagicMock

import pytest

from src.config import UnifiedConfig
from src.config.enums import QueryComplexity, QueryType
from src.models.vector_search import (
    AdaptiveFusionWeights, EffectivenessScore, QueryClassification
)
from src.services.vector_db.adaptive_fusion_tuner import AdaptiveFusionTuner


class TestAdaptiveFusionTuner:
    """Test suite for AdaptiveFusionTuner."""

    @pytest.fixture
    def mock_config(self):
        """Mock unified configuration."""
        return MagicMock(spec=UnifiedConfig)

    @pytest.fixture
    def tuner(self, mock_config):
        """Create AdaptiveFusionTuner instance."""
        return AdaptiveFusionTuner(mock_config)

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
            features={"has_code_keywords": True}
        )

    @pytest.fixture
    def sample_dense_results(self):
        """Create sample dense search results."""
        return [
            {"id": "doc1", "score": 0.9, "payload": {"title": "Python Functions"}},
            {"id": "doc2", "score": 0.85, "payload": {"title": "Async Programming"}},
            {"id": "doc3", "score": 0.8, "payload": {"title": "Error Handling"}},
        ]

    @pytest.fixture
    def sample_sparse_results(self):
        """Create sample sparse search results."""
        return [
            {"id": "doc2", "score": 0.88, "payload": {"title": "Async Programming"}},
            {"id": "doc4", "score": 0.82, "payload": {"title": "Python Syntax"}},
            {"id": "doc1", "score": 0.78, "payload": {"title": "Python Functions"}},
        ]

    async def test_initialization(self, tuner):
        """Test adaptive fusion tuner initialization."""
        assert tuner.config is not None
        assert tuner.epsilon == 0.1
        assert tuner.learning_rate == 0.01
        assert tuner.min_weight == 0.1
        assert tuner.max_weight == 0.9
        assert isinstance(tuner.effectiveness_history, dict)
        assert isinstance(tuner.weight_history, dict)
        assert isinstance(tuner.arm_counts, dict)
        assert isinstance(tuner.arm_rewards, dict)

    async def test_basic_weight_computation(self, tuner, sample_query_classification):
        """Test basic adaptive weight computation."""
        query_id = str(uuid.uuid4())
        
        weights = await tuner.compute_adaptive_weights(
            sample_query_classification, query_id
        )
        
        assert isinstance(weights, AdaptiveFusionWeights)
        assert 0.0 <= weights.dense_weight <= 1.0
        assert 0.0 <= weights.sparse_weight <= 1.0
        assert abs(weights.dense_weight + weights.sparse_weight - 1.0) < 0.01  # Should sum to ~1
        assert weights.query_classification == sample_query_classification

    async def test_effectiveness_score_calculation(self, tuner, sample_dense_results, sample_sparse_results):
        """Test effectiveness score calculation."""
        query_id = str(uuid.uuid4())
        
        effectiveness = await tuner._calculate_effectiveness_scores(
            query_id, sample_dense_results, sample_sparse_results
        )
        
        assert isinstance(effectiveness, EffectivenessScore)
        assert 0.0 <= effectiveness.dense_effectiveness <= 1.0
        assert 0.0 <= effectiveness.sparse_effectiveness <= 1.0
        assert 0.0 <= effectiveness.hybrid_effectiveness <= 1.0
        assert effectiveness.query_id == query_id
        assert effectiveness.evaluation_method == "top_result_with_feedback"

    async def test_effectiveness_with_user_feedback(self, tuner, sample_dense_results, sample_sparse_results):
        """Test effectiveness calculation with user feedback."""
        query_id = str(uuid.uuid4())
        user_feedback = {
            "dense_satisfaction": 0.8,
            "sparse_satisfaction": 0.6,
            "dense_clicked": True,
            "sparse_clicked": False,
            "dwell_time": 45
        }
        
        effectiveness = await tuner._calculate_effectiveness_scores(
            query_id, sample_dense_results, sample_sparse_results, user_feedback
        )
        
        assert isinstance(effectiveness, EffectivenessScore)
        # Dense should be rated higher due to better feedback
        assert effectiveness.dense_effectiveness > effectiveness.sparse_effectiveness

    async def test_top_result_quality_evaluation(self, tuner):
        """Test top result quality evaluation."""
        high_score_results = [{"score": 0.95}, {"score": 0.85}]
        low_score_results = [{"score": 0.25}, {"score": 0.20}]
        empty_results = []
        
        high_quality = tuner._evaluate_top_result_quality(high_score_results)
        low_quality = tuner._evaluate_top_result_quality(low_score_results)
        empty_quality = tuner._evaluate_top_result_quality(empty_results)
        
        assert high_quality > low_quality
        assert empty_quality == 0.0
        assert high_quality > 0.8  # High score should get bonus

    async def test_user_feedback_adjustment(self, tuner):
        """Test user feedback adjustment of effectiveness scores."""
        base_effectiveness = 0.7
        
        # Positive feedback
        positive_feedback = {
            "dense_satisfaction": 0.9,
            "dense_clicked": True,
            "dwell_time": 60
        }
        
        # Negative feedback
        negative_feedback = {
            "dense_satisfaction": 0.2,
            "dense_clicked": False,
            "dwell_time": 3
        }
        
        positive_adjusted = tuner._adjust_with_user_feedback(
            base_effectiveness, positive_feedback, "dense"
        )
        negative_adjusted = tuner._adjust_with_user_feedback(
            base_effectiveness, negative_feedback, "dense"
        )
        
        assert positive_adjusted > base_effectiveness
        assert negative_adjusted < base_effectiveness

    async def test_historical_weights_retrieval(self, tuner, sample_query_classification):
        """Test historical weights retrieval and storage."""
        query_type_key = f"{sample_query_classification.query_type}_{sample_query_classification.complexity_level}"
        
        # Initially should return defaults
        weights = tuner._get_historical_weights(query_type_key)
        assert weights["dense"] == 0.7
        assert weights["sparse"] == 0.3
        
        # Store some historical weights
        adaptive_weights = AdaptiveFusionWeights(
            dense_weight=0.8,
            sparse_weight=0.2,
            hybrid_weight=1.0,
            confidence=0.9,
            learning_rate=0.01,
            query_classification=sample_query_classification
        )
        
        await tuner._store_weights_for_learning(query_type_key, adaptive_weights)
        
        # Should now return stored weights
        updated_weights = tuner._get_historical_weights(query_type_key)
        assert updated_weights["dense"] == 0.8
        assert updated_weights["sparse"] == 0.2

    async def test_rule_based_weight_computation(self, tuner):
        """Test rule-based weight computation for different query types."""
        test_cases = [
            (QueryType.CODE, QueryComplexity.MODERATE, "code queries should balance dense/sparse"),
            (QueryType.CONCEPTUAL, QueryComplexity.SIMPLE, "conceptual queries should favor dense"),
            (QueryType.API_REFERENCE, QueryComplexity.MODERATE, "API queries should favor sparse"),
        ]
        
        for query_type, complexity, description in test_cases:
            classification = QueryClassification(
                query_type=query_type,
                complexity_level=complexity,
                domain="general",
                programming_language=None,
                is_multimodal=False,
                confidence=0.8,
                features={}
            )
            
            weights = tuner._compute_rule_based_weights(classification)
            
            assert 0.1 <= weights["dense"] <= 0.9
            assert 0.1 <= weights["sparse"] <= 0.9
            assert abs(weights["dense"] + weights["sparse"] - 1.0) < 0.01

    async def test_effectiveness_based_weight_computation(self, tuner):
        """Test effectiveness-based weight computation."""
        # Dense better than sparse
        dense_better = EffectivenessScore(
            dense_effectiveness=0.9,
            sparse_effectiveness=0.6,
            hybrid_effectiveness=0.8,
            query_id="test",
            timestamp=time.time(),
            evaluation_method="test"
        )
        
        # Sparse better than dense
        sparse_better = EffectivenessScore(
            dense_effectiveness=0.5,
            sparse_effectiveness=0.8,
            hybrid_effectiveness=0.7,
            query_id="test",
            timestamp=time.time(),
            evaluation_method="test"
        )
        
        dense_weights = tuner._compute_effectiveness_based_weights(dense_better)
        sparse_weights = tuner._compute_effectiveness_based_weights(sparse_better)
        
        assert dense_weights["dense"] > sparse_weights["dense"]
        assert dense_weights["sparse"] < sparse_weights["sparse"]

    async def test_strategy_combination(self, tuner):
        """Test combination of multiple weight strategies."""
        strategies = [
            {"weights": {"dense": 0.8, "sparse": 0.2, "confidence": 0.7}, "confidence": 0.8},
            {"weights": {"dense": 0.6, "sparse": 0.4, "confidence": 0.6}, "confidence": 0.6},
            {"weights": {"dense": 0.7, "sparse": 0.3, "confidence": 0.8}, "confidence": 0.9},
        ]
        
        combined = tuner._combine_strategies(strategies)
        
        assert 0.0 <= combined["dense"] <= 1.0
        assert 0.0 <= combined["sparse"] <= 1.0
        assert abs(combined["dense"] + combined["sparse"] - 1.0) < 0.01
        assert 0.0 <= combined["confidence"] <= 1.0

    async def test_multi_armed_bandit_optimization(self, tuner, sample_query_classification):
        """Test multi-armed bandit optimization."""
        base_weights = {"dense": 0.7, "sparse": 0.3, "confidence": 0.8}
        
        # Simulate multiple optimizations
        optimized_weights_list = []
        for _ in range(10):
            optimized = tuner._apply_bandit_optimization(
                base_weights, sample_query_classification, None
            )
            optimized_weights_list.append(optimized)
        
        # All should be valid weights
        for weights in optimized_weights_list:
            assert 0.0 <= weights["dense"] <= 1.0
            assert 0.0 <= weights["sparse"] <= 1.0
            assert abs(weights["dense"] + weights["sparse"] - 1.0) < 0.01

    async def test_bandit_arm_selection(self, tuner):
        """Test bandit arm selection logic."""
        # Initially should select balanced due to lack of data
        arm = tuner._select_best_arm()
        assert arm in ["dense", "sparse", "balanced"]
        
        # Add some performance data
        tuner.arm_counts = {"dense": 10, "sparse": 5, "balanced": 8}
        tuner.arm_rewards = {"dense": 8.0, "sparse": 2.0, "balanced": 6.0}
        tuner.query_count = 23
        
        # Should select dense (best performing)
        arm = tuner._select_best_arm()
        assert arm == "dense"

    async def test_bandit_statistics_update(self, tuner):
        """Test bandit statistics update with decay."""
        initial_count = tuner.arm_counts["dense"]
        initial_reward = tuner.arm_rewards["dense"]
        
        tuner._update_bandit_statistics("dense", 0.8)
        
        assert tuner.arm_counts["dense"] > initial_count
        assert tuner.arm_rewards["dense"] > initial_reward

    async def test_weight_storage_and_cleanup(self, tuner, sample_query_classification):
        """Test weight storage and automatic cleanup."""
        query_type_key = f"{sample_query_classification.query_type}_{sample_query_classification.complexity_level}"
        
        # Store many weights to trigger cleanup
        for i in range(110):  # More than the 100 limit
            weights = AdaptiveFusionWeights(
                dense_weight=0.7,
                sparse_weight=0.3,
                hybrid_weight=1.0,
                confidence=0.8,
                learning_rate=0.01,
                query_classification=sample_query_classification
            )
            await tuner._store_weights_for_learning(query_type_key, weights)
        
        # Should have been cleaned up to 50 most recent
        assert len(tuner.weight_history[query_type_key]) == 50

    async def test_fallback_weights(self, tuner, sample_query_classification):
        """Test fallback weights when computation fails."""
        fallback = tuner._get_fallback_weights(sample_query_classification)
        
        assert isinstance(fallback, AdaptiveFusionWeights)
        assert fallback.dense_weight == 0.7
        assert fallback.sparse_weight == 0.3
        assert fallback.confidence == 0.5
        assert fallback.query_classification == sample_query_classification

    async def test_user_feedback_integration(self, tuner, sample_query_classification):
        """Test user feedback integration for continuous learning."""
        query_id = str(uuid.uuid4())
        weights = AdaptiveFusionWeights(
            dense_weight=0.7,
            sparse_weight=0.3,
            hybrid_weight=1.0,
            confidence=0.8,
            learning_rate=0.01,
            query_classification=sample_query_classification,
            effectiveness_score=EffectivenessScore(
                dense_effectiveness=0.8,
                sparse_effectiveness=0.6,
                hybrid_effectiveness=0.75,
                query_id=query_id,
                timestamp=time.time(),
                evaluation_method="test"
            )
        )
        
        feedback = {
            "satisfaction": 0.9,
            "clicked": True,
            "dwell_time": 45,
            "relevance": 0.85
        }
        
        await tuner.update_with_feedback(query_id, feedback, weights)
        
        # Should have stored effectiveness history
        assert query_id in tuner.effectiveness_history

    async def test_reward_extraction_from_feedback(self, tuner):
        """Test reward signal extraction from user feedback."""
        # Positive feedback
        positive_feedback = {
            "clicked": True,
            "satisfaction": 0.9,
            "dwell_time": 60,
            "relevance": 0.8
        }
        
        # Negative feedback
        negative_feedback = {
            "clicked": False,
            "satisfaction": 0.2,
            "dwell_time": 2,
            "relevance": 0.3
        }
        
        positive_reward = tuner._extract_reward_from_feedback(positive_feedback)
        negative_reward = tuner._extract_reward_from_feedback(negative_feedback)
        
        assert positive_reward > 0
        assert negative_reward < 0
        assert -1.0 <= positive_reward <= 1.0
        assert -1.0 <= negative_reward <= 1.0

    async def test_performance_statistics(self, tuner):
        """Test performance statistics generation."""
        # Add some data
        tuner.query_count = 100
        tuner.total_reward = 75.0
        tuner.arm_counts = {"dense": 40, "sparse": 30, "balanced": 30}
        tuner.arm_rewards = {"dense": 32.0, "sparse": 20.0, "balanced": 23.0}
        
        stats = tuner.get_performance_stats()
        
        assert stats["total_queries"] == 100
        assert stats["total_reward"] == 75.0
        assert stats["average_reward"] == 0.75
        assert "arm_counts" in stats
        assert "arm_average_rewards" in stats
        assert stats["exploration_rate"] == tuner.epsilon

    async def test_weight_bounds_enforcement(self, tuner, sample_query_classification):
        """Test that weight bounds are properly enforced."""
        # Test multiple computations to ensure bounds
        for _ in range(20):
            weights = await tuner.compute_adaptive_weights(
                sample_query_classification, str(uuid.uuid4())
            )
            
            assert tuner.min_weight <= weights.dense_weight <= tuner.max_weight
            assert tuner.min_weight <= weights.sparse_weight <= tuner.max_weight

    async def test_error_handling_in_weight_computation(self, tuner, sample_query_classification):
        """Test error handling during weight computation."""
        # Mock an error in strategy computation
        original_method = tuner._compute_rule_based_weights
        tuner._compute_rule_based_weights = lambda x: None.__getattribute__("error")
        
        try:
            weights = await tuner.compute_adaptive_weights(
                sample_query_classification, str(uuid.uuid4())
            )
            # Should return fallback weights
            assert isinstance(weights, AdaptiveFusionWeights)
            assert weights.dense_weight == 0.7  # Fallback value
        finally:
            tuner._compute_rule_based_weights = original_method

    async def test_effectiveness_score_zero_handling(self, tuner):
        """Test handling of zero effectiveness scores."""
        zero_effectiveness = EffectivenessScore(
            dense_effectiveness=0.0,
            sparse_effectiveness=0.0,
            hybrid_effectiveness=0.0,
            query_id="test",
            timestamp=time.time(),
            evaluation_method="test"
        )
        
        weights = tuner._compute_effectiveness_based_weights(zero_effectiveness)
        
        assert weights["dense"] == 0.5
        assert weights["sparse"] == 0.5
        assert weights["confidence"] == 0.3

    async def test_multiple_strategies_integration(self, tuner, sample_query_classification, sample_dense_results, sample_sparse_results):
        """Test integration of multiple weight computation strategies."""
        query_id = str(uuid.uuid4())
        
        # Add some historical data
        query_type_key = f"{sample_query_classification.query_type}_{sample_query_classification.complexity_level}"
        historical_weights = AdaptiveFusionWeights(
            dense_weight=0.75,
            sparse_weight=0.25,
            hybrid_weight=1.0,
            confidence=0.85,
            learning_rate=0.01,
            query_classification=sample_query_classification
        )
        await tuner._store_weights_for_learning(query_type_key, historical_weights)
        
        # Compute weights with all strategies
        weights = await tuner.compute_adaptive_weights(
            sample_query_classification, query_id, sample_dense_results, sample_sparse_results
        )
        
        assert isinstance(weights, AdaptiveFusionWeights)
        assert weights.effectiveness_score is not None
        assert 0.0 <= weights.confidence <= 1.0

    @pytest.mark.parametrize("query_type,expected_dense_bias", [
        (QueryType.CODE, False),  # Should be more balanced
        (QueryType.CONCEPTUAL, True),  # Should favor dense
        (QueryType.API_REFERENCE, False),  # Should favor sparse
    ])
    async def test_query_type_specific_weighting(self, tuner, query_type, expected_dense_bias):
        """Test query type specific weight computation."""
        classification = QueryClassification(
            query_type=query_type,
            complexity_level=QueryComplexity.MODERATE,
            domain="general",
            programming_language=None,
            is_multimodal=False,
            confidence=0.8,
            features={}
        )
        
        weights = tuner._compute_rule_based_weights(classification)
        
        if expected_dense_bias:
            assert weights["dense"] > 0.6
        else:
            assert weights["dense"] <= 0.7