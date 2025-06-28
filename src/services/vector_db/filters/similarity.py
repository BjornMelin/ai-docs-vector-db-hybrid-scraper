"""Similarity threshold management for adaptive threshold controls.

This module provides sophisticated similarity threshold management including
adaptive threshold adjustment, cluster-based optimization, performance-based tuning,
and context-aware threshold selection for optimal search results.
"""

import logging
import statistics
from datetime import UTC, datetime, timedelta
from enum import Enum
from typing import Any

import numpy as np
from cachetools import LRUCache
from pydantic import BaseModel, Field, field_validator
from sklearn.cluster import DBSCAN
from sklearn.metrics import silhouette_score

from .base import BaseFilter, FilterError, FilterResult


logger = logging.getLogger(__name__)


class ThresholdStrategy(str, Enum):
    """Strategies for threshold management."""

    STATIC = "static"  # Fixed threshold
    ADAPTIVE = "adaptive"  # Adaptive based on result quality
    CLUSTER_BASED = "cluster_based"  # Based on clustering analysis
    PERFORMANCE_BASED = "performance_based"  # Based on query performance
    CONTEXT_AWARE = "context_aware"  # Based on query context
    ML_OPTIMIZED = "ml_optimized"  # Machine learning optimized


class QueryContext(str, Enum):
    """Query context types for context-aware thresholds."""

    GENERAL = "general"
    PROGRAMMING = "programming"
    DOCUMENTATION = "documentation"
    TUTORIAL = "tutorial"
    TROUBLESHOOTING = "troubleshooting"
    REFERENCE = "reference"
    RESEARCH = "research"
    NEWS = "news"


class ThresholdMetrics(BaseModel):
    """Metrics for threshold performance evaluation."""

    precision: float = Field(..., ge=0.0, le=1.0, description="Precision score")
    recall: float = Field(..., ge=0.0, le=1.0, description="Recall score")
    f1_score: float = Field(..., ge=0.0, le=1.0, description="F1 score")
    result_count: int = Field(..., ge=0, description="Number of results")
    avg_score: float = Field(
        ..., ge=0.0, le=1.0, description="Average similarity score"
    )
    score_variance: float = Field(..., ge=0.0, description="Score variance")
    query_time_ms: float = Field(..., ge=0.0, description="Query execution time")
    user_satisfaction: float | None = Field(
        None, ge=0.0, le=1.0, description="User satisfaction score"
    )


class ClusteringAnalysis(BaseModel):
    """Results of clustering analysis for threshold optimization."""

    optimal_threshold: float = Field(
        ..., ge=0.0, le=1.0, description="Optimal threshold"
    )
    cluster_count: int = Field(..., ge=0, description="Number of clusters found")
    silhouette_score: float = Field(
        ..., ge=-1.0, le=1.0, description="Silhouette score"
    )
    density_ratio: float = Field(..., ge=0.0, le=1.0, description="Density ratio")
    separation_quality: float = Field(
        ..., ge=0.0, le=1.0, description="Cluster separation quality"
    )
    noise_ratio: float = Field(..., ge=0.0, le=1.0, description="Noise ratio")


class SimilarityThresholdCriteria(BaseModel):
    """Criteria for similarity threshold management."""

    # Basic threshold settings
    base_threshold: float = Field(
        0.7, ge=0.0, le=1.0, description="Base similarity threshold"
    )
    min_threshold: float = Field(
        0.3, ge=0.0, le=1.0, description="Minimum allowed threshold"
    )
    max_threshold: float = Field(
        0.95, ge=0.0, le=1.0, description="Maximum allowed threshold"
    )

    # Strategy and adaptation settings
    strategy: ThresholdStrategy = Field(
        ThresholdStrategy.ADAPTIVE, description="Threshold management strategy"
    )
    adaptation_rate: float = Field(
        0.1, ge=0.01, le=1.0, description="Rate of threshold adaptation"
    )
    context: QueryContext = Field(
        QueryContext.GENERAL, description="Query context for threshold selection"
    )

    # Quality targets
    target_result_count: int = Field(
        10, ge=1, le=100, description="Target number of results"
    )
    min_result_count: int = Field(
        3, ge=1, description="Minimum acceptable result count"
    )
    max_result_count: int = Field(
        50, ge=10, description="Maximum acceptable result count"
    )

    # Performance settings
    max_query_time_ms: float = Field(
        1000.0, ge=10.0, description="Maximum acceptable query time"
    )
    target_precision: float = Field(
        0.8, ge=0.0, le=1.0, description="Target precision score"
    )
    target_recall: float = Field(0.7, ge=0.0, le=1.0, description="Target recall score")

    # Clustering settings
    enable_clustering: bool = Field(True, description="Enable clustering analysis")
    min_samples_for_clustering: int = Field(
        10, ge=5, description="Minimum samples needed for clustering"
    )
    clustering_eps: float = Field(
        0.1, ge=0.01, le=1.0, description="DBSCAN eps parameter"
    )

    # Historical analysis
    enable_historical_learning: bool = Field(
        True, description="Enable learning from historical queries"
    )
    history_window_days: int = Field(
        7, ge=1, le=30, description="Days of history to consider"
    )

    @field_validator("min_threshold", "max_threshold")
    @classmethod
    def validate_threshold_ranges(cls, v, info):
        """Validate threshold ranges."""
        if (
            info.field_name == "max_threshold"
            and info.data.get("min_threshold")
            and v <= info.data["min_threshold"]
        ):
            msg = "max_threshold must be > min_threshold"
            raise ValueError(msg)
        return v


class SimilarityThresholdManager(BaseFilter):
    """Advanced similarity threshold management with adaptive optimization."""

    def __init__(
        self,
        name: str = "similarity_threshold_manager",
        description: str = "Manage similarity thresholds with adaptive optimization",
        enabled: bool = True,
        priority: int = 60,
        max_cache_size: int = 1000,
    ):
        """Initialize similarity threshold manager.

        Args:
            name: Filter name
            description: Filter description
            enabled: Whether filter is enabled
            priority: Filter priority (higher = earlier execution)
            max_cache_size: Maximum number of items in clustering cache

        """
        super().__init__(name, description, enabled, priority)

        # Threshold history and learning
        self.threshold_history = []
        self.performance_history = []
        self.context_thresholds = {}

        # Clustering cache with LRU to prevent memory leaks
        self.clustering_cache = LRUCache(maxsize=max_cache_size)
        self.last_clustering_analysis = None
        self.max_cache_size = max_cache_size

        # Performance tracking
        self.query_count = 0
        self.total_query_time = 0.0
        self.adaptation_count = 0

    async def apply(
        self, filter_criteria: dict[str, Any], context: dict[str, Any] | None = None
    ) -> FilterResult:
        """Apply similarity threshold management.

        Args:
            filter_criteria: Threshold management criteria
            context: Optional context with query info and historical data

        Returns:
            FilterResult with optimized similarity threshold

        Raises:
            FilterError: If threshold management fails

        """
        try:
            # Validate and parse criteria
            criteria = SimilarityThresholdCriteria.model_validate(filter_criteria)

            # Get current context information
            query_info = context.get("query_info", {}) if context else {}
            historical_data = context.get("historical_data", []) if context else []

            # Calculate optimal threshold based on strategy
            optimal_threshold = await self._calculate_optimal_threshold(
                criteria, query_info, historical_data
            )

            # Build metadata about threshold decision
            metadata = {
                "threshold_info": {
                    "strategy": criteria.strategy.value,
                    "base_threshold": criteria.base_threshold,
                    "optimal_threshold": optimal_threshold,
                    "context": criteria.context.value,
                    "adaptation_applied": abs(
                        optimal_threshold - criteria.base_threshold
                    )
                    > 0.01,
                },
                "performance_targets": {
                    "target_result_count": criteria.target_result_count,
                    "target_precision": criteria.target_precision,
                    "target_recall": criteria.target_recall,
                },
            }

            # Add clustering analysis if available
            if self.last_clustering_analysis:
                metadata["clustering_analysis"] = (
                    self.last_clustering_analysis.model_dump()
                )

            # Create score threshold filter condition
            filter_condition = None
            if optimal_threshold > 0.0:
                # Note: In actual usage, this would be passed to the search function
                # rather than being a payload filter
                metadata["score_threshold"] = optimal_threshold

            # Calculate performance impact
            performance_impact = "low"  # Threshold management is typically low impact

            self._logger.info(
                f"Applied similarity threshold: {optimal_threshold:.3f} "
                f"(strategy: {criteria.strategy.value}, context: {criteria.context.value})"
            )

            return FilterResult(
                filter_conditions=filter_condition,
                metadata=metadata,
                confidence_score=0.85,
                performance_impact=performance_impact,
            )

        except Exception as e:
            error_msg = "Failed to apply similarity threshold management"
            self._logger.error(error_msg, exc_info=True)
            raise FilterError(
                error_msg,
                filter_name=self.name,
                filter_criteria=filter_criteria,
                underlying_error=e,
            ) from e

    async def _calculate_optimal_threshold(
        self,
        criteria: SimilarityThresholdCriteria,
        query_info: dict[str, Any],
        historical_data: list[dict[str, Any]],
    ) -> float:
        """Calculate optimal threshold based on strategy and context."""
        if criteria.strategy == ThresholdStrategy.STATIC:
            return criteria.base_threshold

        if criteria.strategy == ThresholdStrategy.ADAPTIVE:
            return await self._adaptive_threshold(criteria, query_info, historical_data)

        if criteria.strategy == ThresholdStrategy.CLUSTER_BASED:
            return await self._cluster_based_threshold(
                criteria, query_info, historical_data
            )

        if criteria.strategy == ThresholdStrategy.PERFORMANCE_BASED:
            return await self._performance_based_threshold(criteria, historical_data)

        if criteria.strategy == ThresholdStrategy.CONTEXT_AWARE:
            return await self._context_aware_threshold(criteria, query_info)

        if criteria.strategy == ThresholdStrategy.ML_OPTIMIZED:
            return await self._ml_optimized_threshold(
                criteria, query_info, historical_data
            )

        self._logger.warning("Unknown strategy")
        return criteria.base_threshold

    async def _adaptive_threshold(
        self,
        criteria: SimilarityThresholdCriteria,
        _query_info: dict[str, Any],
        historical_data: list[dict[str, Any]],
    ) -> float:
        """Calculate adaptive threshold based on recent performance."""
        base_threshold = criteria.base_threshold

        if not historical_data:
            return base_threshold

        # Analyze recent performance
        recent_data = self._get_recent_data(
            historical_data, criteria.history_window_days
        )

        if len(recent_data) < 3:
            return base_threshold

        # Calculate performance metrics
        avg_result_count = statistics.mean(
            [d.get("result_count", 0) for d in recent_data]
        )
        avg_precision = statistics.mean(
            [d.get("precision", 0.5) for d in recent_data if "precision" in d]
        )
        avg_query_time = statistics.mean(
            [d.get("query_time_ms", 0) for d in recent_data]
        )

        # Adaptive adjustments
        threshold_adjustment = 0.0

        # Adjust based on result count
        if avg_result_count < criteria.min_result_count:
            # Too few results - lower threshold
            threshold_adjustment -= 0.05
        elif avg_result_count > criteria.max_result_count:
            # Too many results - raise threshold
            threshold_adjustment += 0.03

        # Adjust based on precision
        if avg_precision < criteria.target_precision:
            # Low precision - raise threshold
            threshold_adjustment += 0.02

        # Adjust based on query time
        if avg_query_time > criteria.max_query_time_ms:
            # Slow queries - raise threshold to reduce search space
            threshold_adjustment += 0.01

        # Apply adaptation rate
        threshold_adjustment *= criteria.adaptation_rate

        # Calculate new threshold
        new_threshold = base_threshold + threshold_adjustment

        # Ensure within bounds
        return max(criteria.min_threshold, min(criteria.max_threshold, new_threshold))

    async def _cluster_based_threshold(
        self,
        criteria: SimilarityThresholdCriteria,
        _query_info: dict[str, Any],
        historical_data: list[dict[str, Any]],
    ) -> float:
        """Calculate threshold based on clustering analysis."""
        if not criteria.enable_clustering:
            return criteria.base_threshold

        # Get similarity scores from recent queries
        recent_data = self._get_recent_data(
            historical_data, criteria.history_window_days
        )
        similarity_scores = []

        for data in recent_data:
            if "similarity_scores" in data:
                similarity_scores.extend(data["similarity_scores"])

        if len(similarity_scores) < criteria.min_samples_for_clustering:
            return criteria.base_threshold

        # Perform clustering analysis
        clustering_analysis = await self._analyze_similarity_clusters(
            similarity_scores, criteria
        )

        if clustering_analysis:
            self.last_clustering_analysis = clustering_analysis

            # Use clustering analysis to determine optimal threshold
            optimal_threshold = clustering_analysis.optimal_threshold

            # Ensure within bounds
            return max(
                criteria.min_threshold, min(criteria.max_threshold, optimal_threshold)
            )

        return criteria.base_threshold

    async def _performance_based_threshold(
        self,
        criteria: SimilarityThresholdCriteria,
        historical_data: list[dict[str, Any]],
    ) -> float:
        """Calculate threshold based on performance optimization."""
        if not historical_data:
            return criteria.base_threshold

        # Group historical data by threshold ranges
        threshold_buckets = {}
        bucket_size = 0.05  # 5% buckets

        for data in historical_data:
            threshold = data.get("threshold", criteria.base_threshold)
            bucket = round(threshold / bucket_size) * bucket_size

            if bucket not in threshold_buckets:
                threshold_buckets[bucket] = []
            threshold_buckets[bucket].append(data)

        # Evaluate performance for each bucket
        best_threshold = criteria.base_threshold
        best_score = 0.0

        for threshold, data_points in threshold_buckets.items():
            if len(data_points) < 2:
                continue

            # Calculate composite performance score
            avg_precision = statistics.mean(
                [d.get("precision", 0.5) for d in data_points if "precision" in d]
            )
            avg_recall = statistics.mean(
                [d.get("recall", 0.5) for d in data_points if "recall" in d]
            )
            avg_f1 = (
                2 * (avg_precision * avg_recall) / (avg_precision + avg_recall)
                if (avg_precision + avg_recall) > 0
                else 0
            )
            avg_result_count = statistics.mean(
                [d.get("result_count", 0) for d in data_points]
            )
            avg_query_time = statistics.mean(
                [d.get("query_time_ms", 1000) for d in data_points]
            )

            # Composite score considering multiple factors
            result_count_score = (
                1.0
                if criteria.min_result_count
                <= avg_result_count
                <= criteria.max_result_count
                else 0.5
            )
            time_score = max(0.0, 1.0 - (avg_query_time / criteria.max_query_time_ms))

            composite_score = (
                0.4 * avg_f1
                + 0.3 * result_count_score
                + 0.2 * time_score
                + 0.1 * min(1.0, avg_result_count / criteria.target_result_count)
            )

            if composite_score > best_score:
                best_score = composite_score
                best_threshold = threshold

        return max(criteria.min_threshold, min(criteria.max_threshold, best_threshold))

    async def _context_aware_threshold(
        self, criteria: SimilarityThresholdCriteria, query_info: dict[str, Any]
    ) -> float:
        """Calculate threshold based on query context."""
        context = criteria.context

        # Context-specific threshold mappings
        context_adjustments = {
            QueryContext.PROGRAMMING: 0.05,  # Programming queries need higher precision
            QueryContext.DOCUMENTATION: 0.03,  # Documentation needs good precision
            QueryContext.TUTORIAL: -0.02,  # Tutorials can be more flexible
            QueryContext.TROUBLESHOOTING: 0.02,  # Troubleshooting needs relevance
            QueryContext.REFERENCE: 0.04,  # Reference needs high precision
            QueryContext.RESEARCH: -0.01,  # Research can be more exploratory
            QueryContext.NEWS: -0.03,  # News can be more flexible
            QueryContext.GENERAL: 0.0,  # No adjustment for general queries
        }

        base_threshold = criteria.base_threshold
        adjustment = context_adjustments.get(context, 0.0)

        # Additional adjustments based on query characteristics
        query_length = len(query_info.get("query", "").split())
        if query_length > 10:
            # Longer queries often need lower thresholds
            adjustment -= 0.01
        elif query_length < 3:
            # Very short queries might need higher thresholds
            adjustment += 0.02

        # Check for technical terms
        has_technical_terms = query_info.get("has_technical_terms", False)
        if has_technical_terms:
            adjustment += 0.01

        new_threshold = base_threshold + adjustment

        # Store context-specific threshold for learning
        self.context_thresholds[context.value] = new_threshold

        return max(criteria.min_threshold, min(criteria.max_threshold, new_threshold))

    async def _ml_optimized_threshold(
        self,
        criteria: SimilarityThresholdCriteria,
        query_info: dict[str, Any],
        historical_data: list[dict[str, Any]],
    ) -> float:
        """Calculate threshold using machine learning optimization."""
        # Simplified ML approach - in production, this would use more sophisticated models

        if len(historical_data) < 20:
            # Not enough data for ML optimization
            return await self._adaptive_threshold(criteria, query_info, historical_data)

        # Feature extraction
        features = []
        targets = []

        for data in historical_data:
            # Extract features
            feature_vector = [
                len(data.get("query", "").split()),  # Query length
                data.get("has_technical_terms", 0),  # Technical terms
                data.get("context_score", 0.5),  # Context similarity
                data.get("user_satisfaction", 0.5),  # User feedback
            ]

            # Target is the optimal threshold (based on performance)
            f1_score = data.get("f1_score", 0.5)
            result_count_score = (
                1.0
                if criteria.min_result_count
                <= data.get("result_count", 0)
                <= criteria.max_result_count
                else 0.5
            )
            performance_score = 0.7 * f1_score + 0.3 * result_count_score

            if performance_score > 0.6:  # Good performance
                features.append(feature_vector)
                targets.append(data.get("threshold", criteria.base_threshold))

        if len(features) < 10:
            return criteria.base_threshold

        # Simple weighted average based on performance
        # In production, use proper ML models like random forest or neural networks
        current_features = [
            len(query_info.get("query", "").split()),
            query_info.get("has_technical_terms", 0),
            query_info.get("context_score", 0.5),
            0.5,  # Default user satisfaction
        ]

        # Calculate similarity to historical queries
        similarities = []
        for i, hist_features in enumerate(features):
            # Simple cosine similarity
            similarity = np.dot(current_features, hist_features) / (
                np.linalg.norm(current_features) * np.linalg.norm(hist_features)
            )
            similarities.append((similarity, targets[i]))

        # Weight by similarity and take weighted average
        similarities.sort(reverse=True, key=lambda x: x[0])
        top_similarities = similarities[:5]  # Top 5 similar queries

        if top_similarities:
            weighted_threshold = sum(
                sim * threshold for sim, threshold in top_similarities
            ) / sum(sim for sim, _ in top_similarities)
            return max(
                criteria.min_threshold, min(criteria.max_threshold, weighted_threshold)
            )

        return criteria.base_threshold

    async def _analyze_similarity_clusters(
        self, similarity_scores: list[float], criteria: SimilarityThresholdCriteria
    ) -> ClusteringAnalysis | None:
        """Analyze similarity score clusters to find optimal threshold."""
        if len(similarity_scores) < criteria.min_samples_for_clustering:
            return None

        try:
            # Prepare data for clustering
            scores_array = np.array(similarity_scores).reshape(-1, 1)

            # Perform DBSCAN clustering
            dbscan = DBSCAN(eps=criteria.clustering_eps, min_samples=3)
            cluster_labels = dbscan.fit_predict(scores_array)

            # Analyze clusters
            unique_labels = set(cluster_labels)
            cluster_count = len(unique_labels) - (
                1 if -1 in unique_labels else 0
            )  # Exclude noise

            if cluster_count < 2:
                return None

            # Calculate silhouette score
            if len(set(cluster_labels)) > 1:
                silhouette_avg = silhouette_score(scores_array, cluster_labels)
            else:
                silhouette_avg = 0.0

            # Calculate noise ratio
            noise_count = sum(1 for label in cluster_labels if label == -1)
            noise_ratio = noise_count / len(cluster_labels)

            # Find optimal threshold between clusters
            cluster_centers = []
            for label in unique_labels:
                if label != -1:  # Exclude noise
                    cluster_scores = [
                        scores_array[i][0]
                        for i, label_val in enumerate(cluster_labels)
                        if label_val == label
                    ]
                    cluster_centers.append(statistics.mean(cluster_scores))

            cluster_centers.sort()

            # Optimal threshold is typically between the highest and second-highest clusters
            if len(cluster_centers) >= 2:
                optimal_threshold = (cluster_centers[-1] + cluster_centers[-2]) / 2
            else:
                optimal_threshold = (
                    cluster_centers[0] if cluster_centers else criteria.base_threshold
                )

            # Calculate separation quality
            if len(cluster_centers) >= 2:
                separation_quality = abs(cluster_centers[-1] - cluster_centers[-2])
            else:
                separation_quality = 0.0

            # Calculate density ratio
            density_ratio = 1.0 - noise_ratio

            return ClusteringAnalysis(
                optimal_threshold=optimal_threshold,
                cluster_count=cluster_count,
                silhouette_score=silhouette_avg,
                density_ratio=density_ratio,
                separation_quality=separation_quality,
                noise_ratio=noise_ratio,
            )

        except Exception:
            self._logger.exception("Clustering analysis failed")
            return None

    def _get_recent_data(
        self, historical_data: list[dict[str, Any]], days: int
    ) -> list[dict[str, Any]]:
        """Get recent historical data within specified days."""
        cutoff_date = datetime.now(UTC) - timedelta(days=days)

        recent_data = []
        for data in historical_data:
            timestamp = data.get("timestamp")
            if timestamp:
                if isinstance(timestamp, str):
                    timestamp = datetime.fromisoformat(timestamp.replace("Z", "+00:00"))

                if timestamp >= cutoff_date:
                    recent_data.append(data)

        return recent_data

    def record_performance(
        self, threshold: float, metrics: ThresholdMetrics, query_context: QueryContext
    ) -> None:
        """Record performance metrics for threshold learning."""
        performance_record = {
            "timestamp": datetime.now(UTC),
            "threshold": threshold,
            "context": query_context.value,
            "metrics": metrics.model_dump(),
        }

        self.performance_history.append(performance_record)

        # Keep only recent history to prevent memory bloat
        cutoff_date = datetime.now(UTC) - timedelta(days=30)
        self.performance_history = [
            record
            for record in self.performance_history
            if record["timestamp"] >= cutoff_date
        ]

    def get_threshold_recommendations(
        self, context: QueryContext = QueryContext.GENERAL
    ) -> dict[str, float]:
        """Get threshold recommendations for different strategies."""
        recommendations = {
            "static": 0.7,
            "context_aware": self.context_thresholds.get(context.value, 0.7),
            "learned": 0.7,
        }

        # Calculate learned threshold from performance history
        if self.performance_history:
            context_records = [
                record
                for record in self.performance_history
                if record["context"] == context.value
            ]

            if context_records:
                # Find threshold with best average F1 score
                threshold_performance = {}
                for record in context_records:
                    threshold = record["threshold"]
                    f1_score = record["metrics"].get("f1_score", 0.5)

                    if threshold not in threshold_performance:
                        threshold_performance[threshold] = []
                    threshold_performance[threshold].append(f1_score)

                best_threshold = 0.7
                best_avg_f1 = 0.0

                for threshold, f1_scores in threshold_performance.items():
                    avg_f1 = statistics.mean(f1_scores)
                    if avg_f1 > best_avg_f1:
                        best_avg_f1 = avg_f1
                        best_threshold = threshold

                recommendations["learned"] = best_threshold

        return recommendations

    async def validate_criteria(self, filter_criteria: dict[str, Any]) -> bool:
        """Validate similarity threshold criteria."""
        try:
            SimilarityThresholdCriteria.model_validate(filter_criteria)
            return True
        except Exception:
            self._logger.warning("Invalid similarity threshold criteria")
            return False

    def get_supported_operators(self) -> list[str]:
        """Get supported threshold management operators."""
        return [
            "base_threshold",
            "min_threshold",
            "max_threshold",
            "strategy",
            "adaptation_rate",
            "context",
            "target_result_count",
            "enable_clustering",
            "enable_historical_learning",
        ]

    def get_cache_stats(self) -> dict[str, Any]:
        """Get clustering cache statistics.

        Returns:
            Dictionary with cache statistics

        """
        return {
            "enabled": True,
            "current_size": len(self.clustering_cache),
            "max_size": self.max_cache_size,
            "cache_type": "LRUCache",
        }

    def clear_clustering_cache(self) -> None:
        """Clear the clustering cache to free memory."""
        cache_size = len(self.clustering_cache)
        self.clustering_cache.clear()
        self.last_clustering_analysis = None
        logger.info(f"Cleared clustering cache ({cache_size} items)")  # TODO: Convert f-string to logging format

    def cleanup(self) -> None:
        """Cleanup resources and clear caches."""
        logger.info("Starting similarity threshold manager cleanup")

        # Clear clustering cache
        self.clear_clustering_cache()

        # Clear history if it's too large (keep last 100 entries)
        if len(self.threshold_history) > 100:
            self.threshold_history = self.threshold_history[-100:]
        if len(self.performance_history) > 100:
            self.performance_history = self.performance_history[-100:]

        # Clear context thresholds if too many
        if len(self.context_thresholds) > 50:
            # Keep only the most recently used contexts
            sorted_contexts = sorted(
                self.context_thresholds.items(),
                key=lambda x: x[1].get("last_used", datetime.min.replace(tzinfo=UTC)),
                reverse=True,
            )
            self.context_thresholds = dict(sorted_contexts[:50])

        logger.info("Similarity threshold manager cleanup completed")