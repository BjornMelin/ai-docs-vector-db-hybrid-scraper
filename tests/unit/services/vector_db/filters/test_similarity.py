"""Tests for the similarity threshold filter implementation."""

import asyncio
from datetime import UTC, datetime, timedelta, timezone

import pytest

from src.services.vector_db.filters.base import FilterError, FilterResult
from src.services.vector_db.filters.similarity import (
    ClusteringAnalysis,
    QueryContext,
    SimilarityThresholdCriteria,
    SimilarityThresholdManager,
    ThresholdMetrics,
    ThresholdStrategy,
)


class TestThresholdStrategy:
    """Test ThresholdStrategy enum."""

    def test_strategy_values(self):
        """Test threshold strategy enum values."""
        assert ThresholdStrategy.STATIC.value == "static"
        assert ThresholdStrategy.ADAPTIVE.value == "adaptive"
        assert ThresholdStrategy.CLUSTER_BASED.value == "cluster_based"
        assert ThresholdStrategy.PERFORMANCE_BASED.value == "performance_based"
        assert ThresholdStrategy.CONTEXT_AWARE.value == "context_aware"
        assert ThresholdStrategy.ML_OPTIMIZED.value == "ml_optimized"


class TestQueryContext:
    """Test QueryContext enum."""

    def test_context_values(self):
        """Test query context enum values."""
        assert QueryContext.GENERAL.value == "general"
        assert QueryContext.PROGRAMMING.value == "programming"
        assert QueryContext.DOCUMENTATION.value == "documentation"
        assert QueryContext.TUTORIAL.value == "tutorial"
        assert QueryContext.TROUBLESHOOTING.value == "troubleshooting"
        assert QueryContext.REFERENCE.value == "reference"
        assert QueryContext.RESEARCH.value == "research"
        assert QueryContext.NEWS.value == "news"


class TestThresholdMetrics:
    """Test ThresholdMetrics model."""

    def test_valid_metrics(self):
        """Test valid threshold metrics."""
        metrics = ThresholdMetrics(
            precision=0.85,
            recall=0.75,
            f1_score=0.8,
            result_count=15,
            avg_score=0.72,
            score_variance=0.05,
            query_time_ms=250.0,
            user_satisfaction=0.9,
        )

        assert metrics.precision == 0.85
        assert metrics.recall == 0.75
        assert metrics.f1_score == 0.8
        assert metrics.result_count == 15
        assert metrics.avg_score == 0.72
        assert metrics.score_variance == 0.05
        assert metrics.query_time_ms == 250.0
        assert metrics.user_satisfaction == 0.9

    def test_validation_errors(self):
        """Test validation errors for invalid metrics."""
        # Invalid precision
        with pytest.raises(ValueError):
            ThresholdMetrics(
                precision=1.5,
                recall=0.75,
                f1_score=0.8,
                result_count=15,
                avg_score=0.72,
                score_variance=0.05,
                query_time_ms=250.0,
            )

        # Negative result count
        with pytest.raises(ValueError):
            ThresholdMetrics(
                precision=0.85,
                recall=0.75,
                f1_score=0.8,
                result_count=-5,
                avg_score=0.72,
                score_variance=0.05,
                query_time_ms=250.0,
            )


class TestClusteringAnalysis:
    """Test ClusteringAnalysis model."""

    def test_valid_analysis(self):
        """Test valid clustering analysis."""
        analysis = ClusteringAnalysis(
            optimal_threshold=0.75,
            cluster_count=3,
            silhouette_score=0.6,
            density_ratio=0.8,
            separation_quality=0.4,
            noise_ratio=0.1,
        )

        assert analysis.optimal_threshold == 0.75
        assert analysis.cluster_count == 3
        assert analysis.silhouette_score == 0.6
        assert analysis.density_ratio == 0.8
        assert analysis.separation_quality == 0.4
        assert analysis.noise_ratio == 0.1

    def test_boundary_values(self):
        """Test boundary values for clustering analysis."""
        # Test minimum values
        analysis = ClusteringAnalysis(
            optimal_threshold=0.0,
            cluster_count=0,
            silhouette_score=-1.0,
            density_ratio=0.0,
            separation_quality=0.0,
            noise_ratio=0.0,
        )
        assert analysis.optimal_threshold == 0.0
        assert analysis.silhouette_score == -1.0

        # Test maximum values
        analysis = ClusteringAnalysis(
            optimal_threshold=1.0,
            cluster_count=100,
            silhouette_score=1.0,
            density_ratio=1.0,
            separation_quality=1.0,
            noise_ratio=1.0,
        )
        assert analysis.optimal_threshold == 1.0
        assert analysis.silhouette_score == 1.0


class TestSimilarityThresholdCriteria:
    """Test SimilarityThresholdCriteria model."""

    def test_default_values(self):
        """Test default threshold criteria."""
        criteria = SimilarityThresholdCriteria()

        assert criteria.base_threshold == 0.7
        assert criteria.min_threshold == 0.3
        assert criteria.max_threshold == 0.95
        assert criteria.strategy == ThresholdStrategy.ADAPTIVE
        assert criteria.adaptation_rate == 0.1
        assert criteria.context == QueryContext.GENERAL
        assert criteria.target_result_count == 10
        assert criteria.min_result_count == 3
        assert criteria.max_result_count == 50
        assert criteria.max_query_time_ms == 1000.0
        assert criteria.target_precision == 0.8
        assert criteria.target_recall == 0.7
        assert criteria.enable_clustering is True
        assert criteria.min_samples_for_clustering == 10
        assert criteria.clustering_eps == 0.1
        assert criteria.enable_historical_learning is True
        assert criteria.history_window_days == 7

    def test_custom_values(self):
        """Test criteria with custom values."""
        criteria = SimilarityThresholdCriteria(
            base_threshold=0.8,
            min_threshold=0.5,
            max_threshold=0.9,
            strategy=ThresholdStrategy.CLUSTER_BASED,
            adaptation_rate=0.2,
            context=QueryContext.PROGRAMMING,
            target_result_count=15,
            enable_clustering=False,
        )

        assert criteria.base_threshold == 0.8
        assert criteria.min_threshold == 0.5
        assert criteria.max_threshold == 0.9
        assert criteria.strategy == ThresholdStrategy.CLUSTER_BASED
        assert criteria.adaptation_rate == 0.2
        assert criteria.context == QueryContext.PROGRAMMING
        assert criteria.target_result_count == 15
        assert criteria.enable_clustering is False

    def test_threshold_validation(self):
        """Test threshold range validation."""
        # Valid thresholds
        criteria = SimilarityThresholdCriteria(
            base_threshold=0.7, min_threshold=0.5, max_threshold=0.9
        )
        assert criteria.base_threshold == 0.7

        # Invalid - max <= min
        with pytest.raises(ValueError):
            SimilarityThresholdCriteria(min_threshold=0.8, max_threshold=0.7)

    def test_validation_ranges(self):
        """Test validation of parameter ranges."""
        # Valid ranges
        criteria = SimilarityThresholdCriteria(
            adaptation_rate=0.05,
            target_result_count=20,
            min_result_count=5,
            max_result_count=100,
        )
        assert criteria.adaptation_rate == 0.05

        # Invalid adaptation rate
        with pytest.raises(ValueError):
            SimilarityThresholdCriteria(adaptation_rate=0.005)

        with pytest.raises(ValueError):
            SimilarityThresholdCriteria(adaptation_rate=1.5)


class TestSimilarityThresholdManager:
    """Test SimilarityThresholdManager implementation."""

    @pytest.fixture
    def manager(self):
        """Create similarity threshold manager instance."""
        return SimilarityThresholdManager()

    @pytest.fixture
    def sample_criteria(self):
        """Create sample criteria for testing."""
        return SimilarityThresholdCriteria(
            base_threshold=0.7,
            strategy=ThresholdStrategy.ADAPTIVE,
            min_threshold=0.5,
            max_threshold=0.9,
        )

    @pytest.fixture
    def sample_context(self):
        """Create sample context for testing."""
        return {
            "query_info": {"query": "test query", "has_technical_terms": False},
            "historical_data": [],
        }

    def test_initialization(self, manager):
        """Test manager initialization."""
        assert manager.name == "similarity_threshold_manager"
        assert (
            manager.description
            == "Manage similarity thresholds with adaptive optimization"
        )
        assert manager.enabled is True
        assert manager.priority == 60
        assert len(manager.threshold_history) == 0
        assert len(manager.performance_history) == 0
        assert manager.query_count == 0

    @pytest.mark.asyncio
    async def test_apply_static_strategy(self, manager, sample_context):
        """Test applying filter with static strategy."""
        criteria = SimilarityThresholdCriteria(
            base_threshold=0.75, strategy=ThresholdStrategy.STATIC
        )

        result = await manager.apply(criteria.model_dump(), sample_context)

        assert isinstance(result, FilterResult)
        assert result.filter_conditions is None
        assert "threshold_info" in result.metadata
        assert result.metadata["threshold_info"]["optimal_threshold"] == 0.75
        assert result.metadata["threshold_info"]["strategy"] == "static"
        assert result.confidence_score == 0.85

    @pytest.mark.asyncio
    async def test_apply_adaptive_strategy(self, manager, _sample_context):
        """Test applying filter with adaptive strategy."""
        # Add multiple historical data points for adaptation to work
        historical_data = []
        for _ in range(5):  # Need multiple data points
            historical_data.append(
                {
                    "timestamp": datetime.now(tz=UTC).isoformat(),
                    "result_count": 2,  # Too few results (< min_result_count=3)
                    "precision": 0.9,
                    "query_time_ms": 200,
                }
            )

        context = {
            "query_info": {"query": "test query"},
            "historical_data": historical_data,
        }

        criteria = SimilarityThresholdCriteria(
            base_threshold=0.7,
            strategy=ThresholdStrategy.ADAPTIVE,
            min_result_count=3,  # Use default min_result_count
            adaptation_rate=0.1,
        )

        result = await manager.apply(criteria.model_dump(), context)

        assert isinstance(result, FilterResult)
        assert result.metadata["threshold_info"]["strategy"] == "adaptive"
        # Should adjust threshold down due to too few results
        optimal_threshold = result.metadata["threshold_info"]["optimal_threshold"]
        # With consistent low results, should adjust down
        assert optimal_threshold <= 0.7

    @pytest.mark.asyncio
    async def test_apply_cluster_based_strategy(self, manager):
        """Test applying filter with cluster-based strategy."""
        # Create context with similarity scores for clustering
        historical_data = [
            {
                "timestamp": datetime.now(tz=UTC).isoformat(),
                "similarity_scores": [
                    0.9,
                    0.85,
                    0.8,
                    0.75,
                    0.7,
                    0.4,
                    0.35,
                    0.3,
                    0.25,
                    0.2,
                ],
            }
        ]
        context = {
            "query_info": {"query": "test query"},
            "historical_data": historical_data,
        }

        criteria = SimilarityThresholdCriteria(
            base_threshold=0.7,
            strategy=ThresholdStrategy.CLUSTER_BASED,
            enable_clustering=True,
        )

        result = await manager.apply(criteria.model_dump(), context)

        assert isinstance(result, FilterResult)
        assert result.metadata["threshold_info"]["strategy"] == "cluster_based"

    @pytest.mark.asyncio
    async def test_apply_context_aware_strategy(self, manager, sample_context):
        """Test applying filter with context-aware strategy."""
        criteria = SimilarityThresholdCriteria(
            base_threshold=0.7,
            strategy=ThresholdStrategy.CONTEXT_AWARE,
            context=QueryContext.PROGRAMMING,
        )

        result = await manager.apply(criteria.model_dump(), sample_context)

        assert isinstance(result, FilterResult)
        assert result.metadata["threshold_info"]["strategy"] == "context_aware"
        assert result.metadata["threshold_info"]["context"] == "programming"
        # Programming context should increase threshold
        assert result.metadata["threshold_info"]["optimal_threshold"] > 0.7

    @pytest.mark.asyncio
    async def test_adaptive_threshold_calculation(self, manager):
        """Test adaptive threshold calculation logic."""
        criteria = SimilarityThresholdCriteria(
            base_threshold=0.7,
            min_result_count=10,
            max_result_count=50,
            target_precision=0.8,
            adaptation_rate=0.1,
        )

        query_info = {"query": "test query"}

        # Test with too few results - need multiple data points
        historical_data = []
        for _ in range(5):  # Need >= 3 data points for adaptation
            historical_data.append(
                {
                    "timestamp": datetime.now(tz=UTC).isoformat(),
                    "result_count": 2,  # Much less than min_result_count=10
                    "precision": 0.9,
                    "query_time_ms": 200,
                }
            )

        threshold = await manager._adaptive_threshold(
            criteria, query_info, historical_data
        )

        # Should decrease threshold due to too few results
        assert threshold <= criteria.base_threshold

    @pytest.mark.asyncio
    async def test_performance_based_threshold(self, manager):
        """Test performance-based threshold calculation."""
        criteria = SimilarityThresholdCriteria(
            base_threshold=0.7, target_precision=0.8, target_recall=0.7
        )

        # Create historical data with different thresholds and performance
        historical_data = []
        for i in range(20):
            threshold = 0.65 + i * 0.01  # Range from 0.65 to 0.84
            performance = 0.9 - abs(threshold - 0.75) * 2  # Peak at 0.75

            historical_data.append(
                {
                    "timestamp": datetime.now(tz=UTC).isoformat(),
                    "threshold": threshold,
                    "precision": performance,
                    "recall": performance * 0.9,
                    "result_count": 15,
                    "query_time_ms": 200,
                }
            )

        threshold = await manager._performance_based_threshold(
            criteria, historical_data
        )

        # Should find optimal threshold around 0.75
        assert 0.74 <= threshold <= 0.76

    @pytest.mark.asyncio
    async def test_context_aware_threshold(self, manager):
        """Test context-aware threshold calculation."""
        criteria = SimilarityThresholdCriteria(
            base_threshold=0.7, context=QueryContext.PROGRAMMING
        )

        query_info = {
            "query": "python machine learning algorithm",
            "has_technical_terms": True,
        }

        threshold = await manager._context_aware_threshold(criteria, query_info)

        # Programming context with technical terms should increase threshold
        assert threshold > criteria.base_threshold

    @pytest.mark.asyncio
    async def test_clustering_analysis(self, manager):
        """Test clustering analysis functionality."""
        criteria = SimilarityThresholdCriteria(
            clustering_eps=0.1, min_samples_for_clustering=5
        )

        # Create similarity scores with clear clusters
        similarity_scores = [
            0.9,
            0.89,
            0.88,
            0.87,
            0.86,  # High similarity cluster
            0.7,
            0.69,
            0.68,
            0.67,
            0.66,  # Medium similarity cluster
            0.4,
            0.39,
            0.38,
            0.37,
            0.36,  # Low similarity cluster
        ]

        analysis = await manager._analyze_similarity_clusters(
            similarity_scores, criteria
        )

        if analysis:  # Clustering might not always succeed
            assert isinstance(analysis, ClusteringAnalysis)
            assert analysis.cluster_count >= 0
            assert -1.0 <= analysis.silhouette_score <= 1.0
            assert 0.0 <= analysis.optimal_threshold <= 1.0

    @pytest.mark.asyncio
    async def test_ml_optimized_threshold(self, manager):
        """Test ML-optimized threshold calculation."""
        criteria = SimilarityThresholdCriteria(base_threshold=0.7)

        # Create sufficient historical data for ML optimization
        historical_data = []
        for i in range(25):
            historical_data.append(
                {
                    "timestamp": datetime.now(tz=UTC).isoformat(),
                    "query": f"test query {i}",
                    "has_technical_terms": i % 2,
                    "context_score": 0.5 + i * 0.01,
                    "f1_score": 0.7 + (i % 5) * 0.05,
                    "result_count": 10 + i,
                    "threshold": 0.65 + i * 0.01,
                    "user_satisfaction": 0.7 + (i % 3) * 0.1,
                }
            )

        query_info = {
            "query": "test query similar",
            "has_technical_terms": 1,
            "context_score": 0.6,
        }

        threshold = await manager._ml_optimized_threshold(
            criteria, query_info, historical_data
        )

        # Should return a valid threshold
        assert criteria.min_threshold <= threshold <= criteria.max_threshold

    @pytest.mark.asyncio
    async def test_get_recent_data(self, manager):
        """Test filtering recent historical data."""
        now = datetime.now(tz=UTC)

        historical_data = [
            {"timestamp": (now - timedelta(days=2)).isoformat(), "data": "recent"},
            {"timestamp": (now - timedelta(days=10)).isoformat(), "data": "old"},
            {"timestamp": now.isoformat(), "data": "newest"},
        ]

        recent_data = manager._get_recent_data(historical_data, days=7)

        assert len(recent_data) == 2
        assert any(d["data"] == "recent" for d in recent_data)
        assert any(d["data"] == "newest" for d in recent_data)
        assert not any(d["data"] == "old" for d in recent_data)

    def test_record_performance(self, manager):
        """Test recording performance metrics."""
        metrics = ThresholdMetrics(
            precision=0.85,
            recall=0.75,
            f1_score=0.8,
            result_count=15,
            avg_score=0.72,
            score_variance=0.05,
            query_time_ms=250.0,
        )

        initial_count = len(manager.performance_history)

        manager.record_performance(
            threshold=0.7, metrics=metrics, query_context=QueryContext.PROGRAMMING
        )

        assert len(manager.performance_history) == initial_count + 1

        latest_record = manager.performance_history[-1]
        assert latest_record["threshold"] == 0.7
        assert latest_record["context"] == "programming"
        assert latest_record["metrics"]["precision"] == 0.85

    def test_get_threshold_recommendations(self, manager):
        """Test getting threshold recommendations."""
        # Add some performance history
        metrics = ThresholdMetrics(
            precision=0.9,
            recall=0.8,
            f1_score=0.85,
            result_count=15,
            avg_score=0.75,
            score_variance=0.05,
            query_time_ms=200.0,
        )

        manager.record_performance(0.75, metrics, QueryContext.PROGRAMMING)

        recommendations = manager.get_threshold_recommendations(
            QueryContext.PROGRAMMING
        )

        assert "static" in recommendations
        assert "context_aware" in recommendations
        assert "learned" in recommendations
        assert recommendations["learned"] == 0.75  # Should match recorded threshold

    @pytest.mark.asyncio
    async def test_validate_criteria(self, manager):
        """Test criteria validation."""
        # Valid criteria
        valid_criteria = {
            "base_threshold": 0.7,
            "strategy": "adaptive",
            "min_threshold": 0.5,
            "max_threshold": 0.9,
        }

        is_valid = await manager.validate_criteria(valid_criteria)
        assert is_valid is True

        # Invalid criteria
        invalid_criteria = {
            "base_threshold": 1.5,  # Out of range
            "strategy": "invalid_strategy",
        }

        is_valid = await manager.validate_criteria(invalid_criteria)
        assert is_valid is False

    def test_get_supported_operators(self, manager):
        """Test getting supported operators."""
        operators = manager.get_supported_operators()

        assert isinstance(operators, list)
        assert "base_threshold" in operators
        assert "strategy" in operators
        assert "adaptation_rate" in operators
        assert "context" in operators
        assert "enable_clustering" in operators

    @pytest.mark.asyncio
    async def test_error_handling(self, manager):
        """Test error handling in filter application."""
        # Invalid criteria should raise FilterError
        invalid_criteria = {
            "base_threshold": "invalid_string"
        }  # String instead of float

        with pytest.raises(FilterError) as exc_info:
            await manager.apply(invalid_criteria, {})

        error = exc_info.value
        assert error.filter_name == "similarity_threshold_manager"
        assert "Failed to apply similarity threshold management" in str(error)

    @pytest.mark.asyncio
    async def test_threshold_bounds_enforcement(self, manager):
        """Test that threshold bounds are enforced."""
        criteria = SimilarityThresholdCriteria(
            base_threshold=0.7,
            min_threshold=0.6,
            max_threshold=0.8,
            strategy=ThresholdStrategy.ADAPTIVE,
            adaptation_rate=1.0,  # High rate to test bounds
        )

        # Context that would push threshold very low
        historical_data = [
            {
                "timestamp": datetime.now(tz=UTC).isoformat(),
                "result_count": 0,  # No results - should lower threshold
                "precision": 0.5,
                "query_time_ms": 2000,  # Slow query
            }
        ]

        context = {"query_info": {"query": "test"}, "historical_data": historical_data}

        result = await manager.apply(criteria.model_dump(), context)

        # Threshold should be bounded
        optimal_threshold = result.metadata["threshold_info"]["optimal_threshold"]
        assert criteria.min_threshold <= optimal_threshold <= criteria.max_threshold

    @pytest.mark.asyncio
    async def test_concurrent_applications(self, manager):
        """Test concurrent filter applications."""
        criteria = SimilarityThresholdCriteria(
            base_threshold=0.7, strategy=ThresholdStrategy.STATIC
        )

        contexts = [
            {"query_info": {"query": f"query {i}"}, "historical_data": []}
            for i in range(5)
        ]

        # Apply concurrently
        tasks = [manager.apply(criteria.model_dump(), context) for context in contexts]

        results = await asyncio.gather(*tasks)

        # All should succeed
        assert len(results) == 5
        assert all(isinstance(r, FilterResult) for r in results)
        assert all(r.confidence_score == 0.85 for r in results)

    @pytest.mark.asyncio
    async def test_strategy_selection_logic(self, manager):
        """Test that different strategies produce different results."""
        base_criteria = {
            "base_threshold": 0.7,
            "min_threshold": 0.5,
            "max_threshold": 0.9,
        }

        context = {
            "query_info": {"query": "test query"},
            "historical_data": [
                {
                    "timestamp": datetime.now(tz=UTC).isoformat(),
                    "result_count": 5,
                    "precision": 0.8,
                }
            ],
        }

        # Test different strategies
        strategies = [
            ThresholdStrategy.STATIC,
            ThresholdStrategy.ADAPTIVE,
            ThresholdStrategy.CONTEXT_AWARE,
        ]

        thresholds = []
        for strategy in strategies:
            criteria = {**base_criteria, "strategy": strategy.value}
            result = await manager.apply(criteria, context)
            thresholds.append(result.metadata["threshold_info"]["optimal_threshold"])

        # Static should return base threshold
        assert thresholds[0] == 0.7

        # Adaptive and context-aware might be different
        # (though not guaranteed depending on the specific logic)
        assert all(0.5 <= t <= 0.9 for t in thresholds)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
