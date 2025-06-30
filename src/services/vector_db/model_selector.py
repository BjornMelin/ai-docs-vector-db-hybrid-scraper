"""Model selector for dynamic embedding model selection.

This module implements intelligent embedding model selection based on query
characteristics, performance history, and cost optimization.
"""

import logging
from typing import Any

from src.config import (
    Config,
    EmbeddingModel,
    ModelType,
    OptimizationStrategy,
    QueryComplexity,
    QueryType,
)
# TODO: Fix imports - Any  # TODO: Replace with proper ModelSelectionStrategy type and Any  # TODO: Replace with proper QueryClassification type don't exist
# from src.models.vector_search import Any  # TODO: Replace with proper ModelSelectionStrategy type, Any  # TODO: Replace with proper QueryClassification type


logger = logging.getLogger(__name__)


class ModelSelector:
    """Intelligent model selector for optimal embedding model selection."""

    def __init__(self, config: Config):
        """Initialize model selector.

        Args:
            config: Unified configuration

        """
        self.config = config
        self.model_registry = self._initialize_model_registry()
        self.performance_history: dict[str, dict[str, float]] = {}
        self.cost_budget = getattr(
            config, "embedding_cost_budget", 1000.0
        )  # Monthly budget in USD

    def _initialize_model_registry(self) -> dict[str, dict[str, Any]]:
        """Initialize registry of available embedding models with their characteristics."""
        return {
            # OpenAI Models (API-based)
            EmbeddingModel.TEXT_EMBEDDING_3_SMALL.value: {
                "type": ModelType.GENERAL_PURPOSE,
                "dimensions": 1536,
                "cost_per_1k_tokens": 0.00002,  # $0.02 per 1M tokens
                "latency_ms": 150,
                "quality_score": 0.85,
                "specializations": ["general", "conceptual", "documentation"],
                "max_tokens": 8191,
                "provider": "openai",
                "description": "Cost-effective general purpose embedding model",
            },
            EmbeddingModel.TEXT_EMBEDDING_3_LARGE.value: {
                "type": ModelType.GENERAL_PURPOSE,
                "dimensions": 3072,
                "cost_per_1k_tokens": 0.00013,  # $0.13 per 1M tokens
                "latency_ms": 200,
                "quality_score": 0.95,
                "specializations": ["complex", "conceptual", "multimodal"],
                "max_tokens": 8191,
                "provider": "openai",
                "description": "High-quality general purpose embedding model",
            },
            # FastEmbed Models (Local inference)
            EmbeddingModel.NV_EMBED_V2.value: {
                "type": ModelType.GENERAL_PURPOSE,
                "dimensions": 4096,
                "cost_per_1k_tokens": 0.0,  # Local inference
                "latency_ms": 80,
                "quality_score": 0.98,  # #1 on MTEB leaderboard
                "specializations": ["general", "code", "technical"],
                "max_tokens": 32768,
                "provider": "fastembed",
                "description": "Top-performing open source embedding model",
            },
            EmbeddingModel.BGE_SMALL_EN_V15.value: {
                "type": ModelType.GENERAL_PURPOSE,
                "dimensions": 384,
                "cost_per_1k_tokens": 0.0,  # Local inference
                "latency_ms": 30,
                "quality_score": 0.75,
                "specializations": ["general", "fast"],
                "max_tokens": 512,
                "provider": "fastembed",
                "description": "Fast, cost-effective open source model",
            },
            EmbeddingModel.BGE_LARGE_EN_V15.value: {
                "type": ModelType.GENERAL_PURPOSE,
                "dimensions": 1024,
                "cost_per_1k_tokens": 0.0,  # Local inference
                "latency_ms": 60,
                "quality_score": 0.88,
                "specializations": ["general", "balanced"],
                "max_tokens": 512,
                "provider": "fastembed",
                "description": "High-quality open source embedding model",
            },
            # Specialized Models
            "code-search-net": {
                "type": ModelType.CODE_SPECIALIZED,
                "dimensions": 768,
                "cost_per_1k_tokens": 0.0,
                "latency_ms": 100,
                "quality_score": 0.90,
                "specializations": ["code", "programming", "api"],
                "max_tokens": 2048,
                "provider": "local",
                "description": "Code-specialized embedding model",
            },
            "clip-vit-base-patch32": {
                "type": ModelType.MULTIMODAL,
                "dimensions": 512,
                "cost_per_1k_tokens": 0.0001,
                "latency_ms": 120,
                "quality_score": 0.82,
                "specializations": ["multimodal", "visual", "documentation"],
                "max_tokens": 77,
                "provider": "local",
                "description": "Multi-modal vision-text embedding model",
            },
        }

    async def select_optimal_model(
        self,
        query_classification: Any,  # TODO: Replace with proper QueryClassification type
        optimization_strategy: OptimizationStrategy = OptimizationStrategy.BALANCED,
        context: dict[str, Any] | None = None,
    ) -> Any:  # TODO: Replace with proper ModelSelectionStrategy type
        """Select the optimal embedding model based on query characteristics.

        Args:
            query_classification: Classification results for the query
            optimization_strategy: Optimization strategy (speed, quality, cost, balanced)
            context: Additional context (user preferences, system constraints)

        Returns:
            Any  # TODO: Replace with proper ModelSelectionStrategy type with selected model and rationale

        """
        try:
            # Get candidate models based on query characteristics
            candidates = self._get_candidate_models(query_classification)

            # Score candidates based on optimization strategy
            scored_candidates = await self._score_candidates(
                candidates, query_classification, optimization_strategy, context
            )

            # Select the best model
            best_model = max(scored_candidates, key=lambda x: x["total_score"])

            # Determine fallback models
            fallback_models = [
                model["model_id"]
                for model in sorted(
                    scored_candidates, key=lambda x: x["total_score"], reverse=True
                )[1:3]  # Top 2 alternatives
            ]

            # Calculate ensemble weights if using multiple models
            model_weights = self._calculate_ensemble_weights(scored_candidates)

            # TODO: Replace with proper ModelSelectionStrategy instance
            return {
                "primary_model": best_model["model_id"],
                "model_type": self.model_registry[best_model["model_id"]]["type"],
                "fallback_models": fallback_models,
                "model_weights": model_weights,
                "selection_rationale": best_model["rationale"],
                "expected_performance": best_model["quality_score"],
                "cost_efficiency": best_model["cost_efficiency"],
                "query_classification": query_classification,
            }

        except Exception as e:
            logger.error(
                f"Model selection failed: {e}", exc_info=True
            )  # TODO: Convert f-string to logging format
            return self._get_fallback_strategy(query_classification)

    def _get_candidate_models(
        self, query_classification: Any  # TODO: Replace with proper QueryClassification type
    ) -> list[str]:
        """Get candidate models based on query characteristics."""
        candidates = []

        # Add models based on query type
        for model_id, model_info in self.model_registry.items():
            specializations = model_info["specializations"]

            # Map QueryTypes to the properties that make a model suitable.
            query_type_requirements = {
                QueryType.CODE: {
                    "specializations": {"code"},
                    "model_types": {ModelType.CODE_SPECIALIZED},
                },
                QueryType.MULTIMODAL: {
                    "specializations": {"multimodal"},
                    "model_types": {ModelType.MULTIMODAL},
                },
                QueryType.CONCEPTUAL: {
                    "specializations": {"general", "conceptual"},
                    "model_types": set(),  # No specific model type required
                },
                QueryType.DOCUMENTATION: {
                    "specializations": {"general", "conceptual"},
                    "model_types": set(),
                },
                QueryType.API_REFERENCE: {
                    "specializations": {"api", "technical"},
                    "model_types": set(),
                },
            }

            # Check if model is suitable for query type
            current_query_type = query_classification.query_type

            # Get the requirements for the current query type
            requirements = query_type_requirements.get(current_query_type)

            if requirements:
                # A model is a candidate if its type matches OR it has a required specialization.
                # set.isdisjoint() checks for overlap. `not isdisjoint` means there is an overlap.
                has_required_specialization = not set(specializations).isdisjoint(
                    requirements["specializations"]
                )

                has_required_model_type = (
                    model_info["type"] in requirements["model_types"]
                )

                if has_required_specialization or has_required_model_type:
                    candidates.append(model_id)

        # Always include general-purpose models as fallbacks
        for model_id, model_info in self.model_registry.items():
            if (
                model_info["type"] == ModelType.GENERAL_PURPOSE
                and model_id not in candidates
            ):
                candidates.append(model_id)

        return candidates

    async def _score_candidates(
        self,
        candidates: list[str],
        query_classification: Any,  # TODO: Replace with proper QueryClassification type
        optimization_strategy: OptimizationStrategy,
        _context: dict[str, Any] | None,
    ) -> list[dict[str, Any]]:
        """Score candidate models based on multiple criteria."""
        scored_candidates = []

        for model_id in candidates:
            model_info = self.model_registry[model_id]

            # Base scores
            quality_score = model_info["quality_score"]
            speed_score = 1.0 - (model_info["latency_ms"] / 300.0)  # Normalize to 0-1
            cost_score = 1.0 - min(
                model_info["cost_per_1k_tokens"] / 0.001, 1.0
            )  # Normalize

            # Specialization bonus
            specialization_score = self._calculate_specialization_score(
                model_info, query_classification
            )

            # Historical performance adjustment
            historical_score = self._get_historical_performance(
                model_id, query_classification
            )

            # Combine scores based on optimization strategy
            total_score = self._calculate_weighted_score(
                quality_score,
                speed_score,
                cost_score,
                specialization_score,
                historical_score,
                optimization_strategy,
            )

            # Cost efficiency calculation (normalized to 0-1 range)
            raw_cost_efficiency = quality_score / max(
                model_info["cost_per_1k_tokens"], 0.00001
            )
            # Normalize using a sigmoid-like function to keep within 0-1 range
            cost_efficiency = min(
                1.0, raw_cost_efficiency / (raw_cost_efficiency + 10.0)
            )

            # Generate rationale
            rationale = self._generate_selection_rationale(
                model_id, model_info, total_score, optimization_strategy
            )

            scored_candidates.append(
                {
                    "model_id": model_id,
                    "total_score": total_score,
                    "quality_score": quality_score,
                    "speed_score": speed_score,
                    "cost_score": cost_score,
                    "specialization_score": specialization_score,
                    "historical_score": historical_score,
                    "cost_efficiency": cost_efficiency,
                    "rationale": rationale,
                }
            )

        return scored_candidates

    def _calculate_specialization_score(
        self, model_info: dict[str, Any], query_classification: Any  # TODO: Replace with proper QueryClassification type
    ) -> float:
        """Calculate specialization score based on model-query alignment."""
        specializations = model_info["specializations"]
        score = 0.5  # Base score

        # Query type alignment
        if query_classification.query_type == QueryType.CODE:
            if "code" in specializations:
                score += 0.3
            if model_info["type"] == ModelType.CODE_SPECIALIZED:
                score += 0.2
        elif query_classification.query_type == QueryType.MULTIMODAL:
            if "multimodal" in specializations:
                score += 0.4
            if model_info["type"] == ModelType.MULTIMODAL:
                score += 0.3
        elif query_classification.query_type in [
            QueryType.CONCEPTUAL,
            QueryType.DOCUMENTATION,
        ]:
            if "conceptual" in specializations or "documentation" in specializations:
                score += 0.2

        # Complexity alignment
        if (
            query_classification.complexity_level == QueryComplexity.COMPLEX
            and ("complex" in specializations or model_info["dimensions"] > 1000)
        ) or (
            query_classification.complexity_level == QueryComplexity.SIMPLE
            and ("fast" in specializations or model_info["latency_ms"] < 50)
        ):
            score += 0.1

        # Programming language alignment
        if query_classification.programming_language and (
            "code" in specializations or "programming" in specializations
        ):
            score += 0.15

        return min(score, 1.0)

    def _get_historical_performance(
        self, model_id: str, query_classification: Any  # TODO: Replace with proper QueryClassification type
    ) -> float:
        """Get historical performance score for model on similar queries."""
        query_type_key = (
            f"{query_classification.query_type}_{query_classification.complexity_level}"
        )

        if model_id not in self.performance_history:
            return 0.5  # Neutral score for unknown models

        model_history = self.performance_history[model_id]
        if query_type_key not in model_history:
            # Use average performance across all query types
            if model_history:
                return sum(model_history.values()) / len(model_history)
            return 0.5

        return model_history[query_type_key]

    def _calculate_weighted_score(
        self,
        quality_score: float,
        speed_score: float,
        cost_score: float,
        specialization_score: float,
        historical_score: float,
        optimization_strategy: OptimizationStrategy,
    ) -> float:
        """Calculate weighted total score based on optimization strategy."""
        if optimization_strategy == OptimizationStrategy.QUALITY_OPTIMIZED:
            weights = [
                0.4,
                0.1,
                0.1,
                0.25,
                0.15,
            ]  # Quality, Speed, Cost, Specialization, Historical
        elif optimization_strategy == OptimizationStrategy.SPEED_OPTIMIZED:
            weights = [0.2, 0.4, 0.1, 0.15, 0.15]
        elif optimization_strategy == OptimizationStrategy.COST_OPTIMIZED:
            weights = [0.2, 0.1, 0.4, 0.15, 0.15]
        else:  # BALANCED
            weights = [0.25, 0.2, 0.2, 0.2, 0.15]

        scores = [
            quality_score,
            speed_score,
            cost_score,
            specialization_score,
            historical_score,
        ]
        return sum(w * s for w, s in zip(weights, scores, strict=False))

    def _calculate_ensemble_weights(
        self, scored_candidates: list[dict[str, Any]]
    ) -> dict[str, float]:
        """Calculate ensemble weights for multi-model usage."""
        if len(scored_candidates) < 2:
            return {}

        # Simple approach: weight by normalized scores
        total_score = sum(candidate["total_score"] for candidate in scored_candidates)
        if total_score == 0:
            return {}

        weights = {}
        for candidate in scored_candidates:
            weight = candidate["total_score"] / total_score
            if weight > 0.1:  # Only include models with significant contribution
                weights[candidate["model_id"]] = weight

        return weights

    def _generate_selection_rationale(
        self,
        model_id: str,
        model_info: dict[str, Any],
        total_score: float,
        optimization_strategy: OptimizationStrategy,
    ) -> str:
        """Generate human-readable rationale for model selection."""
        rationale_parts = [f"Selected {model_id}"]

        # Add primary reason based on optimization strategy
        if optimization_strategy == OptimizationStrategy.QUALITY_OPTIMIZED:
            rationale_parts.append(
                f"for optimal quality (score: {model_info['quality_score']:.2f})"
            )
        elif optimization_strategy == OptimizationStrategy.SPEED_OPTIMIZED:
            rationale_parts.append(
                f"for fastest response ({model_info['latency_ms']}ms latency)"
            )
        elif optimization_strategy == OptimizationStrategy.COST_OPTIMIZED:
            rationale_parts.append(
                f"for cost efficiency (${model_info['cost_per_1k_tokens']:.5f}/1k tokens)"
            )
        else:
            rationale_parts.append(
                f"for balanced performance (total score: {total_score:.2f})"
            )

        # Add specialization info
        if model_info["specializations"]:
            specializations = ", ".join(model_info["specializations"][:2])
            rationale_parts.append(f"Specializes in: {specializations}")

        # Add provider info
        if model_info["cost_per_1k_tokens"] == 0:
            rationale_parts.append("(local inference, no API costs)")
        else:
            rationale_parts.append(f"({model_info['provider']} API)")

        return ". ".join(rationale_parts)

    def _get_fallback_strategy(
        self, query_classification: Any  # TODO: Replace with proper QueryClassification type
    ) -> Any:  # TODO: Replace with proper ModelSelectionStrategy type
        """Get fallback strategy when selection fails."""
        # Default to balanced general-purpose model
        fallback_model = EmbeddingModel.BGE_LARGE_EN_V15.value

        # TODO: Replace with proper ModelSelectionStrategy instance
        return {
            "primary_model": fallback_model,
            "model_type": ModelType.GENERAL_PURPOSE,
            "fallback_models": [EmbeddingModel.TEXT_EMBEDDING_3_SMALL.value],
            "model_weights": {},
            "selection_rationale": "Fallback to reliable general-purpose model due to selection error",
            "expected_performance": 0.7,
            "cost_efficiency": 0.8,
            "query_classification": query_classification,
        }

    async def update_performance_history(
        self,
        model_id: str,
        query_classification: Any,  # TODO: Replace with proper QueryClassification type
        performance_score: float,
    ) -> None:
        """Update performance history for a model on a specific query type."""
        try:
            query_type_key = f"{query_classification.query_type}_{query_classification.complexity_level}"

            if model_id not in self.performance_history:
                self.performance_history[model_id] = {}

            # Use exponential moving average for performance updates
            alpha = 0.1  # Learning rate
            current_score = self.performance_history[model_id].get(query_type_key, 0.5)
            updated_score = alpha * performance_score + (1 - alpha) * current_score

            self.performance_history[model_id][query_type_key] = updated_score

            logger.debug(
                f"Updated performance history for {model_id} on {query_type_key}: {updated_score:.3f}"
            )

        except Exception as e:
            logger.error(
                f"Failed to update performance history: {e}", exc_info=True
            )  # TODO: Convert f-string to logging format

    def get_model_info(self, model_id: str) -> dict[str, Any] | None:
        """Get information about a specific model."""
        return self.model_registry.get(model_id)

    def list_available_models(
        self, model_type: ModelType | None = None
    ) -> list[dict[str, Any]]:
        """List available models, optionally filtered by type."""
        models = []
        for model_id, model_info in self.model_registry.items():
            if model_type is None or model_info["type"] == model_type:
                models.append(
                    {
                        "id": model_id,
                        "type": model_info["type"],
                        "description": model_info["description"],
                        "quality_score": model_info["quality_score"],
                        "cost_per_1k_tokens": model_info["cost_per_1k_tokens"],
                        "latency_ms": model_info["latency_ms"],
                    }
                )
        return models

    def estimate_monthly_cost(
        self,
        model_id: str,
        estimated_queries_per_month: int,
        avg_tokens_per_query: int = 100,
    ) -> float:
        """Estimate monthly cost for using a specific model."""
        model_info = self.model_registry.get(model_id)
        if not model_info:
            return 0.0

        total_tokens = estimated_queries_per_month * avg_tokens_per_query
        cost_per_token = model_info["cost_per_1k_tokens"] / 1000
        return total_tokens * cost_per_token

    def get_cost_optimized_recommendations(
        self, monthly_budget: float, estimated_queries: int
    ) -> list[dict[str, Any]]:
        """Get model recommendations that fit within a monthly budget."""
        recommendations = []

        for model_id, model_info in self.model_registry.items():
            estimated_cost = self.estimate_monthly_cost(model_id, estimated_queries)
            if estimated_cost <= monthly_budget:
                efficiency = model_info["quality_score"] / max(estimated_cost, 0.01)
                recommendations.append(
                    {
                        "model_id": model_id,
                        "estimated_monthly_cost": estimated_cost,
                        "quality_score": model_info["quality_score"],
                        "cost_efficiency": efficiency,
                        "description": model_info["description"],
                    }
                )

        return sorted(recommendations, key=lambda x: x["cost_efficiency"], reverse=True)
