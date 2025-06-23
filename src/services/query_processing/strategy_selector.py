
"""Intelligent Search Strategy Selection.

This module provides intelligent strategy selection based on query intent,
complexity, and characteristics to optimize search performance and quality.
"""

import logging
from typing import Any

from .models import MatryoshkaDimension
from .models import QueryComplexity
from .models import QueryIntent
from .models import QueryIntentClassification
from .models import SearchStrategy
from .models import SearchStrategySelection

logger = logging.getLogger(__name__)


class SearchStrategySelector:
    """Intelligent search strategy selection based on query analysis.

    Uses query intent, complexity, and domain characteristics to select
    optimal search strategies and Matryoshka embedding dimensions.
    """

    def __init__(self):
        """Initialize the strategy selector."""
        self._initialized = False

        # Strategy selection rules based on query intent
        self._intent_strategy_map = {
            QueryIntent.CONCEPTUAL: {
                "primary": SearchStrategy.SEMANTIC,
                "fallbacks": [SearchStrategy.HYBRID, SearchStrategy.MULTI_STAGE],
                "dimension": MatryoshkaDimension.MEDIUM,
                "reasoning": "Conceptual queries benefit from semantic understanding",
            },
            QueryIntent.PROCEDURAL: {
                "primary": SearchStrategy.HYDE,
                "fallbacks": [SearchStrategy.SEMANTIC, SearchStrategy.HYBRID],
                "dimension": MatryoshkaDimension.LARGE,
                "reasoning": "Step-by-step queries benefit from hypothetical document generation",
            },
            QueryIntent.FACTUAL: {
                "primary": SearchStrategy.HYBRID,
                "fallbacks": [SearchStrategy.SEMANTIC, SearchStrategy.FILTERED],
                "dimension": MatryoshkaDimension.SMALL,
                "reasoning": "Factual queries benefit from precise keyword + semantic matching",
            },
            QueryIntent.TROUBLESHOOTING: {
                "primary": SearchStrategy.RERANKED,
                "fallbacks": [SearchStrategy.HYBRID, SearchStrategy.SEMANTIC],
                "dimension": MatryoshkaDimension.LARGE,
                "reasoning": "Problem-solving queries need comprehensive reranking for relevance",
            },
            QueryIntent.COMPARATIVE: {
                "primary": SearchStrategy.MULTI_STAGE,
                "fallbacks": [SearchStrategy.RERANKED, SearchStrategy.HYBRID],
                "dimension": MatryoshkaDimension.LARGE,
                "reasoning": "Comparison queries benefit from multi-stage retrieval analysis",
            },
            QueryIntent.ARCHITECTURAL: {
                "primary": SearchStrategy.HYDE,
                "fallbacks": [SearchStrategy.MULTI_STAGE, SearchStrategy.RERANKED],
                "dimension": MatryoshkaDimension.LARGE,
                "reasoning": "Architecture queries need comprehensive hypothetical document analysis",
            },
            QueryIntent.PERFORMANCE: {
                "primary": SearchStrategy.RERANKED,
                "fallbacks": [SearchStrategy.HYBRID, SearchStrategy.MULTI_STAGE],
                "dimension": MatryoshkaDimension.LARGE,
                "reasoning": "Performance queries require high-quality ranking for optimization insights",
            },
            QueryIntent.SECURITY: {
                "primary": SearchStrategy.FILTERED,
                "fallbacks": [SearchStrategy.RERANKED, SearchStrategy.HYBRID],
                "dimension": MatryoshkaDimension.LARGE,
                "reasoning": "Security queries benefit from filtered search with strict relevance",
            },
            QueryIntent.INTEGRATION: {
                "primary": SearchStrategy.HYBRID,
                "fallbacks": [SearchStrategy.SEMANTIC, SearchStrategy.FILTERED],
                "dimension": MatryoshkaDimension.MEDIUM,
                "reasoning": "Integration queries need balanced keyword and semantic matching",
            },
            QueryIntent.BEST_PRACTICES: {
                "primary": SearchStrategy.RERANKED,
                "fallbacks": [SearchStrategy.SEMANTIC, SearchStrategy.HYDE],
                "dimension": MatryoshkaDimension.MEDIUM,
                "reasoning": "Best practice queries benefit from quality-focused reranking",
            },
            QueryIntent.CODE_REVIEW: {
                "primary": SearchStrategy.MULTI_STAGE,
                "fallbacks": [SearchStrategy.RERANKED, SearchStrategy.SEMANTIC],
                "dimension": MatryoshkaDimension.LARGE,
                "reasoning": "Code review queries need multi-stage analysis for comprehensive feedback",
            },
            QueryIntent.MIGRATION: {
                "primary": SearchStrategy.HYDE,
                "fallbacks": [SearchStrategy.MULTI_STAGE, SearchStrategy.RERANKED],
                "dimension": MatryoshkaDimension.LARGE,
                "reasoning": "Migration queries benefit from hypothetical scenario generation",
            },
            QueryIntent.DEBUGGING: {
                "primary": SearchStrategy.FILTERED,
                "fallbacks": [SearchStrategy.RERANKED, SearchStrategy.HYBRID],
                "dimension": MatryoshkaDimension.LARGE,
                "reasoning": "Debugging queries need precise filtering for specific problem solutions",
            },
            QueryIntent.CONFIGURATION: {
                "primary": SearchStrategy.FILTERED,
                "fallbacks": [SearchStrategy.HYBRID, SearchStrategy.SEMANTIC],
                "dimension": MatryoshkaDimension.MEDIUM,
                "reasoning": "Configuration queries benefit from filtered search with specific parameters",
            },
        }

        # Complexity adjustments
        self._complexity_adjustments = {
            QueryComplexity.SIMPLE: {
                "dimension_boost": 0,  # Use default dimension
                "prefer_fast": True,
                "max_fallbacks": 1,
            },
            QueryComplexity.MODERATE: {
                "dimension_boost": 0,  # Use default dimension
                "prefer_fast": False,
                "max_fallbacks": 2,
            },
            QueryComplexity.COMPLEX: {
                "dimension_boost": 1,  # Upgrade dimension if possible
                "prefer_fast": False,
                "max_fallbacks": 3,
            },
            QueryComplexity.EXPERT: {
                "dimension_boost": 1,  # Upgrade to highest dimension
                "prefer_fast": False,
                "max_fallbacks": 3,
                "force_reranking": True,
            },
        }

        # Performance estimates (in milliseconds)
        self._strategy_performance = {
            SearchStrategy.SEMANTIC: {"latency": 50, "quality": 0.7},
            SearchStrategy.HYBRID: {"latency": 80, "quality": 0.8},
            SearchStrategy.HYDE: {"latency": 200, "quality": 0.9},
            SearchStrategy.MULTI_STAGE: {"latency": 150, "quality": 0.85},
            SearchStrategy.FILTERED: {"latency": 60, "quality": 0.75},
            SearchStrategy.RERANKED: {"latency": 120, "quality": 0.9},
            SearchStrategy.ADAPTIVE: {"latency": 100, "quality": 0.8},
        }

    async def initialize(self) -> None:
        """Initialize the strategy selector."""
        self._initialized = True
        logger.info("SearchStrategySelector initialized")

    async def select_strategy(
        self,
        intent_classification: QueryIntentClassification,
        context: dict[str, Any] | None = None,
        performance_requirements: dict[str, Any] | None = None,
    ) -> SearchStrategySelection:
        """Select optimal search strategy based on query analysis.

        Args:
            intent_classification: Results from query intent classification
            context: Optional additional context information
            performance_requirements: Optional performance constraints

        Returns:
            SearchStrategySelection: Selected strategy with reasoning

        Raises:
            RuntimeError: If selector not initialized
        """
        if not self._initialized:
            raise RuntimeError("SearchStrategySelector not initialized")

        primary_intent = intent_classification.primary_intent
        complexity = intent_classification.complexity_level

        # Get base strategy configuration for intent
        base_config = self._intent_strategy_map.get(
            primary_intent,
            {
                "primary": SearchStrategy.SEMANTIC,
                "fallbacks": [SearchStrategy.HYBRID],
                "dimension": MatryoshkaDimension.MEDIUM,
                "reasoning": f"Default strategy for {primary_intent.value} queries",
            },
        )

        # Apply complexity adjustments
        adjusted_strategy = self._apply_complexity_adjustments(base_config, complexity)

        # Apply performance requirements if specified
        if performance_requirements:
            adjusted_strategy = self._apply_performance_requirements(
                adjusted_strategy, performance_requirements
            )

        # Apply context-based adjustments
        if context:
            adjusted_strategy = self._apply_context_adjustments(
                adjusted_strategy, context
            )

        # Consider secondary intents for fallback strategies
        if intent_classification.secondary_intents:
            adjusted_strategy = self._incorporate_secondary_intents(
                adjusted_strategy, intent_classification.secondary_intents
            )

        # Calculate confidence based on intent classification confidence
        primary_confidence = intent_classification.confidence_scores.get(
            primary_intent, 0.5
        )
        strategy_confidence = min(primary_confidence * 1.2, 1.0)  # Boost slightly

        # Get performance estimates
        primary_perf = self._strategy_performance[adjusted_strategy["primary"]]
        estimated_latency = primary_perf["latency"]
        estimated_quality = primary_perf["quality"]

        # Adjust estimates based on dimension
        dimension_multiplier = {
            MatryoshkaDimension.SMALL: 0.8,
            MatryoshkaDimension.MEDIUM: 1.0,
            MatryoshkaDimension.LARGE: 1.3,
        }
        multiplier = dimension_multiplier[adjusted_strategy["dimension"]]
        estimated_latency *= multiplier
        estimated_quality = min(estimated_quality * (1 + (multiplier - 1) * 0.5), 1.0)

        return SearchStrategySelection(
            primary_strategy=adjusted_strategy["primary"],
            fallback_strategies=adjusted_strategy["fallbacks"],
            matryoshka_dimension=adjusted_strategy["dimension"],
            confidence=strategy_confidence,
            reasoning=adjusted_strategy["reasoning"],
            estimated_quality=estimated_quality,
            estimated_latency_ms=estimated_latency,
        )

    def _apply_complexity_adjustments(
        self, base_config: dict[str, Any], complexity: QueryComplexity
    ) -> dict[str, Any]:
        """Apply adjustments based on query complexity."""
        adjusted = base_config.copy()
        complexity_config = self._complexity_adjustments[complexity]

        # Upgrade dimension if needed
        current_dim = adjusted["dimension"]
        boost = complexity_config["dimension_boost"]

        if boost > 0:
            if current_dim == MatryoshkaDimension.SMALL:
                adjusted["dimension"] = MatryoshkaDimension.MEDIUM
            elif current_dim == MatryoshkaDimension.MEDIUM:
                adjusted["dimension"] = MatryoshkaDimension.LARGE

        # Adjust strategy for speed if needed
        if complexity_config["prefer_fast"]:
            fast_strategies = [SearchStrategy.SEMANTIC, SearchStrategy.FILTERED]
            if adjusted["primary"] not in fast_strategies:
                # Move current primary to fallback and use semantic
                adjusted["fallbacks"] = [adjusted["primary"]] + adjusted["fallbacks"][
                    :1
                ]
                adjusted["primary"] = SearchStrategy.SEMANTIC
                adjusted["reasoning"] += " (optimized for speed)"

        # Force reranking for expert queries
        if (
            complexity_config.get("force_reranking", False)
            and adjusted["primary"] != SearchStrategy.RERANKED
            and SearchStrategy.RERANKED not in adjusted["fallbacks"]
        ):
            # Add reranking as primary or fallback
            adjusted["fallbacks"] = [SearchStrategy.RERANKED] + adjusted["fallbacks"][
                :2
            ]

        # Limit fallback strategies based on complexity
        max_fallbacks = complexity_config["max_fallbacks"]
        adjusted["fallbacks"] = adjusted["fallbacks"][:max_fallbacks]

        return adjusted

    def _apply_performance_requirements(
        self, config: dict[str, Any], requirements: dict[str, Any]
    ) -> dict[str, Any]:
        """Apply performance-based strategy adjustments."""
        adjusted = config.copy()

        max_latency = requirements.get("max_latency_ms")
        min_quality = requirements.get("min_quality")

        if max_latency:
            # If latency constraint is tight, prefer faster strategies
            primary_latency = self._strategy_performance[adjusted["primary"]]["latency"]

            if primary_latency > max_latency:
                # Find a faster strategy
                fast_strategies = [
                    (strategy, perf)
                    for strategy, perf in self._strategy_performance.items()
                    if perf["latency"] <= max_latency
                ]

                if fast_strategies:
                    # Choose the fastest strategy with best quality
                    best_strategy = max(fast_strategies, key=lambda x: x[1]["quality"])
                    adjusted["fallbacks"] = [adjusted["primary"]] + adjusted[
                        "fallbacks"
                    ][:1]
                    adjusted["primary"] = best_strategy[0]
                    adjusted["reasoning"] += (
                        f" (optimized for <{max_latency}ms latency)"
                    )

        if min_quality:
            # If quality requirement is high, prefer higher quality strategies
            primary_quality = self._strategy_performance[adjusted["primary"]]["quality"]

            if primary_quality < min_quality:
                # Find a higher quality strategy
                quality_strategies = [
                    (strategy, perf)
                    for strategy, perf in self._strategy_performance.items()
                    if perf["quality"] >= min_quality
                ]

                if quality_strategies:
                    # Choose the highest quality strategy with reasonable latency
                    best_strategy = min(
                        quality_strategies, key=lambda x: x[1]["latency"]
                    )
                    adjusted["fallbacks"] = [adjusted["primary"]] + adjusted[
                        "fallbacks"
                    ][:1]
                    adjusted["primary"] = best_strategy[0]
                    adjusted["reasoning"] += f" (optimized for >{min_quality} quality)"

        return adjusted

    def _apply_context_adjustments(
        self, config: dict[str, Any], context: dict[str, Any]
    ) -> dict[str, Any]:
        """Apply context-based strategy adjustments."""
        adjusted = config.copy()

        # Programming language context
        if (
            "programming_language" in context
            and any(
                lang in ["python", "javascript", "java"]
                for lang in context["programming_language"]
            )
            and SearchStrategy.SEMANTIC
            not in [adjusted["primary"]] + adjusted["fallbacks"]
        ):
            # These languages have rich documentation, semantic search works well
            adjusted["fallbacks"] = [SearchStrategy.SEMANTIC] + adjusted["fallbacks"][
                :2
            ]

        # Framework context
        if (
            "framework" in context
            and context["framework"]
            and SearchStrategy.FILTERED
            not in [adjusted["primary"]] + adjusted["fallbacks"]
        ):
            # Framework-specific queries often benefit from filtered search
            adjusted["fallbacks"] = [SearchStrategy.FILTERED] + adjusted["fallbacks"][
                :2
            ]

        # Error code context
        if "error_code" in context and adjusted["primary"] != SearchStrategy.FILTERED:
            # Error codes benefit from precise filtering
            adjusted["fallbacks"] = [adjusted["primary"]] + adjusted["fallbacks"][:1]
            adjusted["primary"] = SearchStrategy.FILTERED
            adjusted["reasoning"] += " (optimized for error code lookup)"

        # Version context
        if (
            "version" in context
            and SearchStrategy.HYBRID
            not in [adjusted["primary"]] + adjusted["fallbacks"]
        ):
            # Version-specific queries often need precise matching
            adjusted["fallbacks"] = [SearchStrategy.HYBRID] + adjusted["fallbacks"][:2]

        # Urgency context
        urgency = context.get("urgency")
        if urgency == "high":
            # High urgency prefers faster strategies
            fast_strategies = [SearchStrategy.SEMANTIC, SearchStrategy.FILTERED]
            if adjusted["primary"] not in fast_strategies:
                adjusted["fallbacks"] = [adjusted["primary"]] + adjusted["fallbacks"][
                    :1
                ]
                adjusted["primary"] = SearchStrategy.SEMANTIC
                adjusted["reasoning"] += " (prioritized for urgency)"

        return adjusted

    def _incorporate_secondary_intents(
        self, config: dict[str, Any], secondary_intents: list[QueryIntent]
    ) -> dict[str, Any]:
        """Incorporate secondary intents into fallback strategy selection."""
        adjusted = config.copy()

        # Add strategies from secondary intents to fallbacks
        for intent in secondary_intents[:2]:  # Limit to top 2 secondary intents
            if intent in self._intent_strategy_map:
                secondary_config = self._intent_strategy_map[intent]
                secondary_strategy = secondary_config["primary"]

                # Add to fallbacks if not already present
                if (
                    secondary_strategy
                    not in [adjusted["primary"]] + adjusted["fallbacks"]
                ):
                    adjusted["fallbacks"].append(secondary_strategy)

        # Limit total fallbacks to 3
        adjusted["fallbacks"] = adjusted["fallbacks"][:3]

        return adjusted

    async def cleanup(self) -> None:
        """Cleanup strategy selector resources."""
        self._initialized = False
        logger.info("SearchStrategySelector cleaned up")
