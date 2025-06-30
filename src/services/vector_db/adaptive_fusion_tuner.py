"""Adaptive fusion tuner for optimizing hybrid search weight combinations.

This module implements machine learning-based tuning of fusion weights for hybrid search,
optimizing the combination of dense and sparse vector search results.
"""

import logging
import time
from typing import Any

from src.config import Config
from src.services.query_processing.models import QueryIntentClassification


logger = logging.getLogger(__name__)


class AdaptiveFusionTuner:
    """ML-based adaptive fusion tuner for hybrid search optimization."""

    def __init__(self, config: Config):
        """Initialize adaptive fusion tuner.

        Args:
            config: Unified configuration

        """
        self.config = config
        self.performance_history: dict[str, dict[str, float]] = {}
        self.total_queries = 0
        self.successful_optimizations = 0

        # Default weights for different query types
        self.default_weights = {
            "code": {"dense": 0.7, "sparse": 0.3},
            "documentation": {"dense": 0.6, "sparse": 0.4},
            "api_reference": {"dense": 0.5, "sparse": 0.5},
            "conceptual": {"dense": 0.8, "sparse": 0.2},
            "troubleshooting": {"dense": 0.6, "sparse": 0.4},
            "multimodal": {"dense": 0.7, "sparse": 0.3},
            "default": {"dense": 0.7, "sparse": 0.3},
        }

    async def compute_adaptive_weights(
        self,
        query_classification: QueryIntentClassification,
        historical_performance: dict[str, float] | None = None,
        _context: dict[str, Any] | None = None,
    ) -> dict[str, float]:
        """Compute adaptive weights for hybrid search fusion.

        Args:
            query_classification: Classification results for the query
            historical_performance: Historical performance data for similar queries
            context: Additional context for weight computation

        Returns:
            Dictionary with 'dense' and 'sparse' weights that sum to 1.0

        """
        try:
            self.total_queries += 1
            start_time = time.time()

            # Get base weights for query type
            query_type = query_classification.primary_intent.lower()
            base_weights = self.default_weights.get(
                query_type, self.default_weights["default"]
            ).copy()

            # Adjust based on query complexity
            complexity_adjustment = self._compute_complexity_adjustment(
                query_classification
            )

            # Apply complexity adjustment
            dense_weight = base_weights["dense"] + complexity_adjustment
            sparse_weight = 1.0 - dense_weight

            # Ensure weights are within valid range
            dense_weight = max(0.1, min(0.9, dense_weight))
            sparse_weight = 1.0 - dense_weight

            # Apply historical performance adjustments if available
            if historical_performance:
                performance_adjustment = self._compute_performance_adjustment(
                    historical_performance, query_classification
                )
                dense_weight = max(0.1, min(0.9, dense_weight + performance_adjustment))
                sparse_weight = 1.0 - dense_weight

            # Record successful optimization
            self.successful_optimizations += 1

            # Update performance history
            self._update_performance_history(
                query_classification, dense_weight, time.time() - start_time
            )

            weights = {"dense": dense_weight, "sparse": sparse_weight}

            logger.debug("Computed adaptive weights for %s: %s", query_type, weights)

        except OSError:
            logger.exception("Failed to compute adaptive weights")
            # Return default balanced weights on error
            weights = {"dense": 0.7, "sparse": 0.3}

        return weights

    def _compute_complexity_adjustment(
        self,
        query_classification: QueryIntentClassification,
    ) -> float:
        """Compute weight adjustment based on query complexity."""
        complexity = query_classification.complexity_level.lower()

        # More complex queries benefit from sparse search (keyword matching)
        # Simpler queries benefit from dense search (semantic understanding)
        if complexity in ["complex", "high"]:
            return -0.1  # Favor sparse for complex queries
        if complexity in ["simple", "low"]:
            return 0.1  # Favor dense for simple queries
        return 0.0  # No adjustment for moderate complexity

    def _compute_performance_adjustment(
        self,
        historical_performance: dict[str, float],
        _query_classification: QueryIntentClassification,
    ) -> float:
        """Compute adjustment based on historical performance."""
        try:
            # Simple heuristic: if dense performed better historically, favor it
            dense_score = historical_performance.get("dense_score", 0.5)
            sparse_score = historical_performance.get("sparse_score", 0.5)

            score_diff = dense_score - sparse_score

            # Scale the adjustment (max ±0.15)
            return max(-0.15, min(0.15, score_diff * 0.3))

        except OSError:
            logger.debug("Failed to compute performance adjustment")
            return 0.0

    def _update_performance_history(
        self,
        query_classification: QueryIntentClassification,
        weight: float,
        computation_time: float,
    ) -> None:
        """Update performance history for future optimizations."""
        try:
            query_key = (
                f"{query_classification.primary_intent}_"
                f"{query_classification.complexity_level}"
            )

            if query_key not in self.performance_history:
                self.performance_history[query_key] = {}

            # Store weight and computation time
            self.performance_history[query_key]["last_weight"] = weight
            self.performance_history[query_key]["last_computation_time"] = (
                computation_time
            )
            self.performance_history[query_key]["usage_count"] = (
                self.performance_history[query_key].get("usage_count", 0) + 1
            )

        except OSError:
            logger.debug("Failed to update performance history")

    def get_performance_stats(self) -> dict[str, Any]:
        """Get performance statistics for monitoring and debugging."""
        try:
            avg_computation_time = 0.0
            if self.performance_history:
                computation_times = [
                    data.get("last_computation_time", 0.0)
                    for data in self.performance_history.values()
                    if "last_computation_time" in data
                ]
                if computation_times:
                    avg_computation_time = sum(computation_times) / len(
                        computation_times
                    )

            return {
                "total_queries": self.total_queries,
                "successful_optimizations": self.successful_optimizations,
                "optimization_success_rate": (
                    self.successful_optimizations / max(self.total_queries, 1)
                ),
                "avg_computation_time_ms": avg_computation_time * 1000,
                "query_types_seen": len(self.performance_history),
                "performance_history_size": len(self.performance_history),
            }
        except OSError:
            logger.exception("Failed to get performance stats")
            return {
                "total_queries": self.total_queries,
                "successful_optimizations": self.successful_optimizations,
                "optimization_success_rate": 0.0,
                "avg_computation_time_ms": 0.0,
                "query_types_seen": 0,
                "performance_history_size": 0,
            }

    def reset_performance_history(self) -> None:
        """Reset performance history (useful for testing or retraining)."""
        self.performance_history.clear()
        self.total_queries = 0
        self.successful_optimizations = 0
        logger.info("Adaptive fusion tuner performance history reset")

    def get_weight_recommendations(self, query_type: str) -> dict[str, float]:
        """Get weight recommendations for a specific query type."""
        return self.default_weights.get(
            query_type.lower(), self.default_weights["default"]
        ).copy()
