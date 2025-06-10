"""Adaptive fusion weight tuner for hybrid search optimization.

This module implements ML-based adaptive fusion weight tuning using effectiveness
scoring and online learning algorithms like multi-armed bandits.
"""

import logging
import math
import random
import time
from typing import Any

import numpy as np

from ...config import UnifiedConfig
from ...models.vector_search import AdaptiveFusionWeights
from ...models.vector_search import EffectivenessScore
from ...models.vector_search import QueryClassification

logger = logging.getLogger(__name__)


class AdaptiveFusionTuner:
    """Adaptive fusion weight tuner using effectiveness scoring and online learning."""

    def __init__(self, config: UnifiedConfig):
        """Initialize adaptive fusion tuner.

        Args:
            config: Unified configuration
        """
        self.config = config
        self.effectiveness_history: dict[str, list[EffectivenessScore]] = {}
        self.weight_history: dict[str, list[AdaptiveFusionWeights]] = {}

        # Multi-armed bandit parameters
        self.epsilon = 0.1  # Exploration rate for ε-greedy
        self.learning_rate = 0.01
        self.confidence_multiplier = 2.0  # For UCB algorithm

        # Weight adjustment parameters
        self.min_weight = 0.1  # Minimum weight to maintain diversity
        self.max_weight = 0.9  # Maximum weight to maintain balance
        self.adaptation_threshold = 0.05  # Minimum change threshold

        # Performance tracking
        self.query_count = 0
        self.total_reward = 0.0
        self.arm_counts = {"dense": 0, "sparse": 0, "balanced": 0}
        self.arm_rewards = {"dense": 0.0, "sparse": 0.0, "balanced": 0.0}

    async def compute_adaptive_weights(
        self,
        query_classification: QueryClassification,
        query_id: str,
        dense_results: list[dict[str, Any]] | None = None,
        sparse_results: list[dict[str, Any]] | None = None,
        user_feedback: dict[str, Any] | None = None,
    ) -> AdaptiveFusionWeights:
        """Compute adaptive fusion weights based on query characteristics and effectiveness.

        Args:
            query_classification: Classification results for the query
            query_id: Unique identifier for the query
            dense_results: Results from dense vector search (for effectiveness scoring)
            sparse_results: Results from sparse vector search (for effectiveness scoring)
            user_feedback: Optional user feedback for effectiveness evaluation

        Returns:
            AdaptiveFusionWeights with optimized weights and metadata
        """
        try:
            # Get effectiveness scores if results are provided
            effectiveness_score = None
            if dense_results is not None and sparse_results is not None:
                effectiveness_score = await self._calculate_effectiveness_scores(
                    query_id, dense_results, sparse_results, user_feedback
                )

            # Get historical performance for this query type
            query_type_key = f"{query_classification.query_type}_{query_classification.complexity_level}"
            historical_weights = self._get_historical_weights(query_type_key)

            # Compute adaptive weights using multiple strategies
            weights = await self._compute_weights_from_multiple_strategies(
                query_classification, effectiveness_score, historical_weights
            )

            # Apply multi-armed bandit for continuous optimization
            optimized_weights = self._apply_bandit_optimization(
                weights, query_classification, effectiveness_score
            )

            # Create adaptive fusion weights object
            adaptive_weights = AdaptiveFusionWeights(
                dense_weight=optimized_weights["dense"],
                sparse_weight=optimized_weights["sparse"],
                hybrid_weight=1.0,
                confidence=optimized_weights["confidence"],
                learning_rate=self.learning_rate,
                query_classification=query_classification,
                effectiveness_score=effectiveness_score,
            )

            # Store for future learning
            await self._store_weights_for_learning(query_type_key, adaptive_weights)

            return adaptive_weights

        except Exception as e:
            logger.error(f"Adaptive weight computation failed: {e}", exc_info=True)
            # Return balanced fallback weights
            return self._get_fallback_weights(query_classification)

    async def _calculate_effectiveness_scores(
        self,
        query_id: str,
        dense_results: list[dict[str, Any]],
        sparse_results: list[dict[str, Any]],
        user_feedback: dict[str, Any] | None = None,
    ) -> EffectivenessScore:
        """Calculate effectiveness scores for dense and sparse retrieval."""
        timestamp = time.time()

        # Evaluate top-1 result quality (DAT approach)
        dense_effectiveness = self._evaluate_top_result_quality(dense_results)
        sparse_effectiveness = self._evaluate_top_result_quality(sparse_results)

        # Incorporate user feedback if available
        if user_feedback:
            dense_effectiveness = self._adjust_with_user_feedback(
                dense_effectiveness, user_feedback, "dense"
            )
            sparse_effectiveness = self._adjust_with_user_feedback(
                sparse_effectiveness, user_feedback, "sparse"
            )

        # Calculate hybrid effectiveness (weighted combination)
        hybrid_effectiveness = 0.6 * dense_effectiveness + 0.4 * sparse_effectiveness

        return EffectivenessScore(
            dense_effectiveness=dense_effectiveness,
            sparse_effectiveness=sparse_effectiveness,
            hybrid_effectiveness=hybrid_effectiveness,
            query_id=query_id,
            timestamp=timestamp,
            evaluation_method="top_result_with_feedback",
        )

    def _evaluate_top_result_quality(self, results: list[dict[str, Any]]) -> float:
        """Evaluate the quality of the top result."""
        if not results:
            return 0.0

        top_result = results[0]
        score = top_result.get("score", 0.0)

        # Normalize score to 0-1 range (assuming scores are typically 0-1)
        # Add some heuristics for quality assessment
        quality_score = min(score, 1.0)

        # Bonus for higher scores
        if score > 0.8:
            quality_score *= 1.1
        elif score < 0.3:
            quality_score *= 0.8

        return min(quality_score, 1.0)

    def _adjust_with_user_feedback(
        self, effectiveness: float, feedback: dict[str, Any], search_type: str
    ) -> float:
        """Adjust effectiveness score based on user feedback."""
        feedback_score = feedback.get(f"{search_type}_satisfaction", 0.5)
        click_through = feedback.get(f"{search_type}_clicked", False)
        dwell_time = feedback.get("dwell_time", 0)

        # Incorporate different feedback signals
        adjustment = 0.0

        # Click-through adjustment
        if click_through:
            adjustment += 0.1
        else:
            adjustment -= 0.05

        # Satisfaction score adjustment
        adjustment += (feedback_score - 0.5) * 0.2

        # Dwell time adjustment (longer dwell time = better result)
        if dwell_time > 30:  # 30 seconds threshold
            adjustment += 0.05
        elif dwell_time < 5:
            adjustment -= 0.05

        return max(0.0, min(1.0, effectiveness + adjustment))

    def _get_historical_weights(self, query_type_key: str) -> dict[str, float]:
        """Get historical weight performance for a query type."""
        if query_type_key not in self.weight_history:
            return {"dense": 0.7, "sparse": 0.3, "confidence": 0.5}

        # Calculate average weights from recent history
        recent_weights = self.weight_history[query_type_key][-10:]  # Last 10 queries
        if not recent_weights:
            return {"dense": 0.7, "sparse": 0.3, "confidence": 0.5}

        avg_dense = np.mean([w.dense_weight for w in recent_weights])
        avg_sparse = np.mean([w.sparse_weight for w in recent_weights])
        avg_confidence = np.mean([w.confidence for w in recent_weights])

        return {
            "dense": float(avg_dense),
            "sparse": float(avg_sparse),
            "confidence": float(avg_confidence),
        }

    async def _compute_weights_from_multiple_strategies(
        self,
        query_classification: QueryClassification,
        effectiveness_score: EffectivenessScore | None,
        historical_weights: dict[str, float],
    ) -> dict[str, float]:
        """Compute weights using multiple strategies and combine them."""
        strategies = []

        # Strategy 1: Rule-based weights based on query type
        rule_based = self._compute_rule_based_weights(query_classification)
        strategies.append({"weights": rule_based, "confidence": 0.6})

        # Strategy 2: Effectiveness-based weights (DAT approach)
        if effectiveness_score:
            effectiveness_based = self._compute_effectiveness_based_weights(
                effectiveness_score
            )
            strategies.append({"weights": effectiveness_based, "confidence": 0.8})

        # Strategy 3: Historical performance weights
        strategies.append({"weights": historical_weights, "confidence": 0.7})

        # Combine strategies using weighted average
        return self._combine_strategies(strategies)

    def _compute_rule_based_weights(
        self, query_classification: QueryClassification
    ) -> dict[str, float]:
        """Compute weights based on query classification rules."""
        # Default balanced weights
        dense_weight = 0.7
        sparse_weight = 0.3

        # Adjust based on query type
        if query_classification.query_type.value == "code":
            # Code queries often benefit from exact keyword matching
            dense_weight = 0.6
            sparse_weight = 0.4
        elif query_classification.query_type.value == "conceptual":
            # Conceptual queries benefit more from semantic understanding
            dense_weight = 0.8
            sparse_weight = 0.2
        elif query_classification.query_type.value == "api_reference":
            # API queries need precise keyword matching
            dense_weight = 0.5
            sparse_weight = 0.5

        # Adjust based on complexity
        if query_classification.complexity_level.value == "complex":
            # Complex queries might benefit from more semantic understanding
            dense_weight += 0.1
            sparse_weight -= 0.1
        elif query_classification.complexity_level.value == "simple":
            # Simple queries might benefit from keyword matching
            dense_weight -= 0.1
            sparse_weight += 0.1

        # Ensure weights are within bounds and sum to 1
        dense_weight = max(self.min_weight, min(self.max_weight, dense_weight))
        sparse_weight = 1.0 - dense_weight

        return {"dense": dense_weight, "sparse": sparse_weight, "confidence": 0.7}

    def _compute_effectiveness_based_weights(
        self, effectiveness_score: EffectivenessScore
    ) -> dict[str, float]:
        """Compute weights based on effectiveness scores (DAT approach)."""
        total_effectiveness = (
            effectiveness_score.dense_effectiveness
            + effectiveness_score.sparse_effectiveness
        )

        if total_effectiveness == 0:
            return {"dense": 0.5, "sparse": 0.5, "confidence": 0.3}

        # Normalize effectiveness scores to weights
        dense_weight = effectiveness_score.dense_effectiveness / total_effectiveness
        sparse_weight = effectiveness_score.sparse_effectiveness / total_effectiveness

        # Apply smoothing to prevent extreme weights
        smoothing_factor = 0.1
        dense_weight = (1 - smoothing_factor) * dense_weight + smoothing_factor * 0.7
        sparse_weight = 1.0 - dense_weight

        # Confidence based on the difference in effectiveness
        effectiveness_diff = abs(
            effectiveness_score.dense_effectiveness
            - effectiveness_score.sparse_effectiveness
        )
        confidence = min(0.9, 0.5 + effectiveness_diff)

        return {
            "dense": dense_weight,
            "sparse": sparse_weight,
            "confidence": confidence,
        }

    def _combine_strategies(self, strategies: list[dict[str, Any]]) -> dict[str, float]:
        """Combine multiple weight strategies using confidence-weighted averaging."""
        total_confidence = sum(s["confidence"] for s in strategies)

        if total_confidence == 0:
            return {"dense": 0.7, "sparse": 0.3, "confidence": 0.5}

        # Weighted average of dense weights
        dense_weight = (
            sum(s["weights"]["dense"] * s["confidence"] for s in strategies)
            / total_confidence
        )

        # Sparse weight is complementary
        sparse_weight = 1.0 - dense_weight

        # Average confidence
        avg_confidence = total_confidence / len(strategies)

        return {
            "dense": dense_weight,
            "sparse": sparse_weight,
            "confidence": avg_confidence,
        }

    def _apply_bandit_optimization(
        self,
        base_weights: dict[str, float],
        query_classification: QueryClassification,
        effectiveness_score: EffectivenessScore | None,
    ) -> dict[str, float]:
        """Apply multi-armed bandit optimization for continuous learning."""
        self.query_count += 1

        # Define arms (weight configurations)
        arms = {
            "dense": {"dense": 0.8, "sparse": 0.2},
            "sparse": {"dense": 0.3, "sparse": 0.7},
            "balanced": {"dense": 0.6, "sparse": 0.4},
        }

        # ε-greedy arm selection
        if random.random() < self.epsilon:
            # Exploration: random arm
            selected_arm = random.choice(list(arms.keys()))
        else:
            # Exploitation: best performing arm
            selected_arm = self._select_best_arm()

        selected_weights = arms[selected_arm]

        # Combine bandit selection with base weights
        adaptation_rate = 0.3
        final_dense = (1 - adaptation_rate) * base_weights[
            "dense"
        ] + adaptation_rate * selected_weights["dense"]
        final_sparse = 1.0 - final_dense

        # Update bandit statistics if we have effectiveness feedback
        if effectiveness_score:
            reward = effectiveness_score.hybrid_effectiveness
            self._update_bandit_statistics(selected_arm, reward)

        return {
            "dense": final_dense,
            "sparse": final_sparse,
            "confidence": base_weights["confidence"]
            * 0.9,  # Slight confidence reduction for exploration
        }

    def _select_best_arm(self) -> str:
        """Select the best performing arm using UCB (Upper Confidence Bound)."""
        if self.query_count <= len(self.arm_counts):
            # Not enough data, return balanced
            return "balanced"

        ucb_values = {}
        for arm in self.arm_counts:
            if self.arm_counts[arm] == 0:
                ucb_values[arm] = float("inf")
            else:
                avg_reward = self.arm_rewards[arm] / self.arm_counts[arm]
                confidence_bonus = math.sqrt(
                    (self.confidence_multiplier * math.log(self.query_count))
                    / self.arm_counts[arm]
                )
                ucb_values[arm] = avg_reward + confidence_bonus

        return max(ucb_values, key=ucb_values.get)

    def _update_bandit_statistics(self, arm: str, reward: float) -> None:
        """Update bandit statistics with new reward."""
        self.arm_counts[arm] += 1
        self.arm_rewards[arm] += reward
        self.total_reward += reward

        # Decay old statistics to adapt to changing conditions
        decay_factor = 0.995
        for arm_name in self.arm_counts:
            self.arm_counts[arm_name] = int(self.arm_counts[arm_name] * decay_factor)
            self.arm_rewards[arm_name] *= decay_factor

    async def _store_weights_for_learning(
        self, query_type_key: str, weights: AdaptiveFusionWeights
    ) -> None:
        """Store weights for future learning and adaptation."""
        if query_type_key not in self.weight_history:
            self.weight_history[query_type_key] = []

        self.weight_history[query_type_key].append(weights)

        # Keep only recent history to prevent memory bloat
        if len(self.weight_history[query_type_key]) > 100:
            self.weight_history[query_type_key] = self.weight_history[query_type_key][
                -50:
            ]

    def _get_fallback_weights(
        self, query_classification: QueryClassification
    ) -> AdaptiveFusionWeights:
        """Get fallback weights when adaptive computation fails."""
        return AdaptiveFusionWeights(
            dense_weight=0.7,
            sparse_weight=0.3,
            hybrid_weight=1.0,
            confidence=0.5,
            learning_rate=self.learning_rate,
            query_classification=query_classification,
            effectiveness_score=None,
        )

    async def update_with_feedback(
        self,
        query_id: str,
        user_feedback: dict[str, Any],
        weights_used: AdaptiveFusionWeights,
    ) -> None:
        """Update the tuner with user feedback for continuous learning."""
        try:
            # Extract reward signal from feedback
            reward = self._extract_reward_from_feedback(user_feedback)

            # Update effectiveness history
            if query_id not in self.effectiveness_history:
                self.effectiveness_history[query_id] = []

            # Create updated effectiveness score
            if weights_used.effectiveness_score:
                updated_score = EffectivenessScore(
                    dense_effectiveness=weights_used.effectiveness_score.dense_effectiveness
                    * (1 + reward * 0.1),
                    sparse_effectiveness=weights_used.effectiveness_score.sparse_effectiveness
                    * (1 + reward * 0.1),
                    hybrid_effectiveness=weights_used.effectiveness_score.hybrid_effectiveness
                    * (1 + reward * 0.1),
                    query_id=query_id,
                    timestamp=time.time(),
                    evaluation_method="user_feedback_adjusted",
                )
                self.effectiveness_history[query_id].append(updated_score)

            logger.debug(
                f"Updated tuner with feedback for query {query_id}: reward={reward}"
            )

        except Exception as e:
            logger.error(f"Failed to update with feedback: {e}", exc_info=True)

    def _extract_reward_from_feedback(self, feedback: dict[str, Any]) -> float:
        """Extract reward signal from user feedback."""
        # Default reward
        reward = 0.0

        # Click-through reward
        if feedback.get("clicked", False):
            reward += 0.3

        # Satisfaction score
        satisfaction = feedback.get("satisfaction", 0.5)
        reward += (satisfaction - 0.5) * 0.4

        # Dwell time reward
        dwell_time = feedback.get("dwell_time", 0)
        if dwell_time > 30:
            reward += 0.2
        elif dwell_time < 5:
            reward -= 0.1

        # Relevance rating
        relevance = feedback.get("relevance", 0.5)
        reward += (relevance - 0.5) * 0.3

        return max(-1.0, min(1.0, reward))  # Clamp between -1 and 1

    def get_performance_stats(self) -> dict[str, Any]:
        """Get performance statistics for monitoring."""
        return {
            "total_queries": self.query_count,
            "total_reward": self.total_reward,
            "average_reward": self.total_reward / max(self.query_count, 1),
            "arm_counts": dict(self.arm_counts),
            "arm_average_rewards": {
                arm: self.arm_rewards[arm] / max(self.arm_counts[arm], 1)
                for arm in self.arm_counts
            },
            "exploration_rate": self.epsilon,
            "learning_rate": self.learning_rate,
        }
