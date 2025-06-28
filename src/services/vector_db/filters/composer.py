"""Filter composition for combining multiple filters with boolean logic.

This module provides sophisticated filter composition capabilities including
boolean logic operations (AND, OR, NOT), filter orchestration, performance optimization,
and intelligent result merging for complex filtering scenarios.
"""

import asyncio
import logging
import time
from enum import Enum
from typing import Any

from cachetools import LRUCache
from pydantic import BaseModel, Field, field_validator
from qdrant_client import models

from .base import BaseFilter, FilterError, FilterResult


logger = logging.getLogger(__name__)


class CompositionOperator(str, Enum):
    """Boolean operators for filter composition."""

    AND = "and"
    OR = "or"
    NOT = "not"


class FilterReference(BaseModel):
    """Reference to a filter with execution parameters."""

    filter_instance: BaseFilter = Field(..., description="Filter instance to apply")
    criteria: dict[str, Any] = Field(..., description="Criteria for this filter")
    weight: float = Field(
        1.0, ge=0.0, le=1.0, description="Weight for this filter's contribution"
    )
    required: bool = Field(True, description="Whether this filter must succeed")

    class Config:
        arbitrary_types_allowed = True


class CompositionRule(BaseModel):
    """Rule for composing multiple filters."""

    operator: CompositionOperator = Field(..., description="Boolean operator")
    filters: list[FilterReference] = Field(..., description="Filters to compose")
    nested_rules: list["CompositionRule"] | None = Field(
        None, description="Nested composition rules"
    )
    optimization_hints: dict[str, Any] = Field(
        default_factory=dict, description="Hints for query optimization"
    )

    @field_validator("filters")
    @classmethod
    def validate_filters_for_operator(cls, v, info):
        """Validate filter count based on operator."""
        operator = info.data.get("operator")

        if operator == CompositionOperator.NOT:
            if len(v) != 1:
                raise ValueError("NOT operator requires exactly one filter")
        elif (
            operator in [CompositionOperator.AND, CompositionOperator.OR] and len(v) < 2
        ):
            raise ValueError(f"{operator} operator requires at least two filters")

        return v


# Enable forward references for recursive model
CompositionRule.model_rebuild()


class FilterCompositionCriteria(BaseModel):
    """Criteria for filter composition operations."""

    # Main composition rule
    composition_rule: CompositionRule = Field(..., description="Root composition rule")

    # Execution settings
    execution_strategy: str = Field(
        "parallel", description="Execution strategy (parallel, sequential, optimized)"
    )
    fail_fast: bool = Field(
        True, description="Stop execution on first required filter failure"
    )
    optimize_order: bool = Field(
        True, description="Automatically optimize filter execution order"
    )

    # Performance settings
    max_execution_time_ms: float = Field(
        5000.0, ge=100.0, description="Maximum total execution time"
    )
    enable_caching: bool = Field(True, description="Enable filter result caching")

    # Result combination settings
    combine_metadata: bool = Field(
        True, description="Combine metadata from all filters"
    )
    weighted_confidence: bool = Field(
        True, description="Use weighted confidence scores"
    )


class CompositionResult(BaseModel):
    """Result of filter composition execution."""

    success: bool = Field(..., description="Whether composition succeeded")
    final_filter: models.Filter | None = Field(
        None, description="Final composed Qdrant filter"
    )
    execution_metadata: dict[str, Any] = Field(
        default_factory=dict, description="Execution metadata"
    )
    filter_results: dict[str, FilterResult] = Field(
        default_factory=dict, description="Individual filter results"
    )
    composition_confidence: float = Field(
        ..., ge=0.0, le=1.0, description="Overall composition confidence"
    )
    total_execution_time_ms: float = Field(
        ..., ge=0.0, description="Total execution time"
    )


class FilterComposer(BaseFilter):
    """Advanced filter composer for combining multiple filters with boolean logic."""

    def __init__(
        self,
        name: str = "filter_composer",
        description: str = "Compose multiple filters using boolean logic and optimization",
        enabled: bool = True,
        priority: int = 50,
        max_cache_size: int = 1000,
    ):
        """Initialize filter composer.

        Args:
            name: Filter name
            description: Filter description
            enabled: Whether filter is enabled
            priority: Filter priority (higher = earlier execution)
            max_cache_size: Maximum number of items in execution cache
        """
        super().__init__(name, description, enabled, priority)

        # Execution tracking with LRU cache to prevent memory leaks
        self.execution_cache = LRUCache(maxsize=max_cache_size)
        self.performance_stats = {}
        self.optimization_enabled = True
        self.max_cache_size = max_cache_size

        # Filter execution strategies
        self.execution_strategies = {
            "parallel": self._execute_parallel,
            "sequential": self._execute_sequential,
            "optimized": self._execute_optimized,
        }

    async def apply(
        self, filter_criteria: dict[str, Any], context: dict[str, Any] | None = None
    ) -> FilterResult:
        """Apply filter composition with boolean logic.

        Args:
            filter_criteria: Filter composition criteria
            context: Optional context with execution settings and cache

        Returns:
            FilterResult with composed filter conditions

        Raises:
            FilterError: If filter composition fails
        """
        try:
            # Validate and parse criteria
            criteria = FilterCompositionCriteria.model_validate(filter_criteria)

            # Execute composition
            composition_result = await self._execute_composition(criteria, context)

            # Build final filter result
            metadata = {
                "composition_info": {
                    "operator": criteria.composition_rule.operator.value,
                    "filter_count": len(criteria.composition_rule.filters),
                    "execution_strategy": criteria.execution_strategy,
                    "optimized": criteria.optimize_order,
                },
                "execution_metadata": composition_result.execution_metadata,
                "individual_results": {
                    name: result.model_dump()
                    for name, result in composition_result.filter_results.items()
                },
            }

            # Calculate performance impact
            performance_impact = self._estimate_composition_performance(
                composition_result
            )

            self._logger.info(
                f"Applied filter composition with {len(criteria.composition_rule.filters)} filters "
                f"using {criteria.composition_rule.operator.value} operator"
            )

            return FilterResult(
                filter_conditions=composition_result.final_filter,
                metadata=metadata,
                confidence_score=composition_result.composition_confidence,
                performance_impact=performance_impact,
            )

        except Exception as e:
            error_msg = "Failed to apply filter composition"
            self._logger.error(error_msg, exc_info=True)
            raise FilterError(
                error_msg,
                filter_name=self.name,
                filter_criteria=filter_criteria,
                underlying_error=e,
            ) from e

    async def _execute_composition(
        self, criteria: FilterCompositionCriteria, context: dict[str, Any] | None = None
    ) -> CompositionResult:
        """Execute the filter composition based on criteria."""

        start_time = time.time()

        # Get execution strategy
        strategy = self.execution_strategies.get(
            criteria.execution_strategy, self._execute_sequential
        )

        # Optimize filter order if requested
        if criteria.optimize_order:
            criteria.composition_rule = self._optimize_composition_order(
                criteria.composition_rule
            )

        # Execute filters using selected strategy
        try:
            filter_results = await strategy(
                criteria.composition_rule.filters, context, criteria
            )

            # Compose final filter
            final_filter = self._compose_filters(
                criteria.composition_rule, filter_results
            )

            # Calculate composition confidence
            composition_confidence = self._calculate_composition_confidence(
                filter_results, criteria
            )

            execution_time_ms = (time.time() - start_time) * 1000

            return CompositionResult(
                success=True,
                final_filter=final_filter,
                execution_metadata={
                    "strategy": criteria.execution_strategy,
                    "optimized": criteria.optimize_order,
                    "filter_count": len(criteria.composition_rule.filters),
                    "execution_order": [
                        f.filter_instance.name
                        for f in criteria.composition_rule.filters
                    ],
                },
                filter_results=filter_results,
                composition_confidence=composition_confidence,
                total_execution_time_ms=execution_time_ms,
            )

        except Exception as e:
            execution_time_ms = (time.time() - start_time) * 1000

            return CompositionResult(
                success=False,
                final_filter=None,
                execution_metadata={
                    "error": str(e),
                    "execution_time_ms": execution_time_ms,
                },
                filter_results={},
                composition_confidence=0.0,
                total_execution_time_ms=execution_time_ms,
            )

    async def _execute_parallel(
        self,
        filter_refs: list[FilterReference],
        context: dict[str, Any] | None,
        criteria: FilterCompositionCriteria,
    ) -> dict[str, FilterResult]:
        """Execute filters in parallel."""

        results = {}

        # Create tasks for all filters
        tasks = []
        for filter_ref in filter_refs:
            task = asyncio.create_task(
                self._execute_single_filter(filter_ref, context),
                name=filter_ref.filter_instance.name,
            )
            tasks.append((filter_ref.filter_instance.name, task))

        # Wait for all tasks with timeout
        timeout = criteria.max_execution_time_ms / 1000.0

        try:
            done, pending = await asyncio.wait(
                [task for _, task in tasks],
                timeout=timeout,
                return_when=(
                    asyncio.ALL_COMPLETED
                    if not criteria.fail_fast
                    else asyncio.FIRST_EXCEPTION
                ),
            )

            # Cancel pending tasks
            for task in pending:
                task.cancel()

            # Collect results
            for name, task in tasks:
                if task in done:
                    try:
                        result = await task
                        results[name] = result
                    except Exception:
                        self._logger.exception("Filter {name} failed")
                        if criteria.fail_fast:
                            raise
                else:
                    self._logger.warning(f"Filter {name} timed out")

        except TimeoutError as e:
            self._logger.exception("Filter composition timed out")
            raise FilterError("Filter composition execution timed out") from e

        return results

    async def _execute_sequential(
        self,
        filter_refs: list[FilterReference],
        context: dict[str, Any] | None,
        criteria: FilterCompositionCriteria,
    ) -> dict[str, FilterResult]:
        """Execute filters sequentially."""
        results = {}

        for filter_ref in filter_refs:
            try:
                result = await self._execute_single_filter(filter_ref, context)
                results[filter_ref.filter_instance.name] = result

                # Check if this is a required filter that failed
                if (
                    filter_ref.required
                    and not result.filter_conditions
                    and criteria.fail_fast
                ):
                    raise FilterError(
                        f"Required filter {filter_ref.filter_instance.name} failed"
                    )

            except Exception:
                self._logger.exception(
                    "Filter {filter_ref.filter_instance.name} failed"
                )
                if criteria.fail_fast and filter_ref.required:
                    raise

        return results

    async def _execute_optimized(
        self,
        filter_refs: list[FilterReference],
        context: dict[str, Any] | None,
        criteria: FilterCompositionCriteria,
    ) -> dict[str, FilterResult]:
        """Execute filters with performance optimization."""
        # Start with high-priority, low-cost filters
        optimized_refs = sorted(
            filter_refs,
            key=lambda f: (f.filter_instance.priority, -self._estimate_filter_cost(f)),
        )

        # Use parallel execution for low-cost filters, sequential for high-cost
        low_cost_filters = []
        high_cost_filters = []

        for filter_ref in optimized_refs:
            cost = self._estimate_filter_cost(filter_ref)
            if cost < 0.5:  # Threshold for parallel execution
                low_cost_filters.append(filter_ref)
            else:
                high_cost_filters.append(filter_ref)

        results = {}

        # Execute low-cost filters in parallel
        if low_cost_filters:
            parallel_results = await self._execute_parallel(
                low_cost_filters, context, criteria
            )
            results.update(parallel_results)

        # Execute high-cost filters sequentially
        if high_cost_filters:
            sequential_results = await self._execute_sequential(
                high_cost_filters, context, criteria
            )
            results.update(sequential_results)

        return results

    async def _execute_single_filter(
        self, filter_ref: FilterReference, context: dict[str, Any] | None
    ) -> FilterResult:
        """Execute a single filter with error handling."""
        try:
            if not filter_ref.filter_instance.enabled:
                # Return empty result for disabled filters
                return FilterResult(
                    filter_conditions=None,
                    metadata={"disabled": True},
                    confidence_score=0.0,
                    performance_impact="none",
                )

            return await filter_ref.filter_instance.apply(filter_ref.criteria, context)

        except Exception as e:
            self._logger.exception(
                "Filter {filter_ref.filter_instance.name} execution failed"
            )
            raise FilterError(
                f"Filter {filter_ref.filter_instance.name} failed",
                filter_name=filter_ref.filter_instance.name,
                filter_criteria=filter_ref.criteria,
                underlying_error=e,
            ) from e

    def _compose_filters(
        self, rule: CompositionRule, filter_results: dict[str, FilterResult]
    ) -> models.Filter | None:
        """Compose individual filter results into final filter."""
        conditions = self._collect_filter_conditions(rule, filter_results)

        if not conditions:
            return None

        return self._apply_composition_operator(rule.operator, conditions)

    def _collect_filter_conditions(
        self, rule: CompositionRule, filter_results: dict[str, FilterResult]
    ) -> list[models.Filter]:
        """Collect valid filter conditions from rule and nested rules."""
        conditions = []

        # Collect valid filter conditions
        for filter_ref in rule.filters:
            filter_name = filter_ref.filter_instance.name
            if filter_name in filter_results:
                result = filter_results[filter_name]
                if result.filter_conditions:
                    conditions.append(result.filter_conditions)

        # Handle nested rules
        if rule.nested_rules:
            for nested_rule in rule.nested_rules:
                nested_filter = self._compose_filters(nested_rule, filter_results)
                if nested_filter:
                    conditions.append(nested_filter)

        return conditions

    def _apply_composition_operator(
        self, operator: CompositionOperator, conditions: list[models.Filter]
    ) -> models.Filter | None:
        """Apply boolean operator to conditions."""
        if not conditions:
            return None

        if operator == CompositionOperator.NOT:
            # For NOT composition, we need to negate all conditions
            # Create a filter with must_not containing all the original conditions
            return models.Filter(must_not=conditions)

        if len(conditions) == 1 and operator != CompositionOperator.NOT:
            return conditions[0]

        if operator == CompositionOperator.AND:
            return self._create_and_filter(conditions)
        elif operator == CompositionOperator.OR:
            return models.Filter(should=conditions)

        return None

    def _create_and_filter(self, conditions: list[models.Filter]) -> models.Filter:
        """Create AND filter from conditions."""
        all_must_conditions = []
        for condition in conditions:
            if isinstance(condition, models.Filter):
                if condition.must:
                    all_must_conditions.extend(condition.must)
                else:
                    all_must_conditions.append(condition)
            else:
                all_must_conditions.append(condition)
        return models.Filter(must=all_must_conditions)

    def _optimize_composition_order(self, rule: CompositionRule) -> CompositionRule:
        """Optimize filter execution order for better performance."""
        if not self.optimization_enabled:
            return rule

        # Sort filters by priority (high first) and estimated cost (low first)
        optimized_filters = sorted(
            rule.filters,
            key=lambda f: (f.filter_instance.priority, -self._estimate_filter_cost(f)),
        )

        # Recursively optimize nested rules
        optimized_nested = [
            self._optimize_composition_order(nested_rule)
            for nested_rule in (rule.nested_rules or [])
        ]

        return CompositionRule(
            operator=rule.operator,
            filters=optimized_filters,
            nested_rules=optimized_nested if optimized_nested else None,
            optimization_hints=rule.optimization_hints,
        )

    def _estimate_filter_cost(self, filter_ref: FilterReference) -> float:
        """Estimate the computational cost of a filter (0.0 to 1.0)."""
        # Base cost on filter type and criteria complexity
        base_cost = 0.1

        # Increase cost based on criteria complexity
        criteria_complexity = (
            len(str(filter_ref.criteria)) / 1000.0
        )  # Rough approximation
        base_cost += min(0.3, criteria_complexity)

        # Check filter-specific cost factors
        filter_name = filter_ref.filter_instance.__class__.__name__

        cost_multipliers = {
            "TemporalFilter": 0.2,  # Low cost
            "ContentTypeFilter": 0.4,  # Medium cost
            "MetadataFilter": 0.3,  # Medium-low cost
            "SimilarityThresholdManager": 0.6,  # Higher cost due to ML operations
        }

        multiplier = cost_multipliers.get(filter_name, 0.5)
        return min(1.0, base_cost * multiplier)

    def _calculate_composition_confidence(
        self,
        filter_results: dict[str, FilterResult],
        criteria: FilterCompositionCriteria,
    ) -> float:
        """Calculate overall confidence for the composition."""
        if not filter_results:
            return 0.0

        if criteria.weighted_confidence:
            # Weighted average based on filter weights
            total_weight = 0.0
            weighted_confidence = 0.0

            for filter_ref in criteria.composition_rule.filters:
                filter_name = filter_ref.filter_instance.name
                if filter_name in filter_results:
                    result = filter_results[filter_name]
                    weighted_confidence += result.confidence_score * filter_ref.weight
                    total_weight += filter_ref.weight

            return weighted_confidence / total_weight if total_weight > 0 else 0.0
        else:
            # Simple average
            confidences = [
                result.confidence_score for result in filter_results.values()
            ]
            return sum(confidences) / len(confidences)

    def _estimate_composition_performance(self, result: CompositionResult) -> str:
        """Estimate performance impact of the composition."""
        if result.total_execution_time_ms > 2000:
            return "high"
        elif result.total_execution_time_ms > 500:
            return "medium"
        else:
            return "low"

    async def validate_criteria(self, filter_criteria: dict[str, Any]) -> bool:
        """Validate filter composition criteria."""
        try:
            FilterCompositionCriteria.model_validate(filter_criteria)
            return True
        except Exception:
            self._logger.warning("Invalid composition criteria")
            return False

    def get_supported_operators(self) -> list[str]:
        """Get supported composition operators."""
        return [op.value for op in CompositionOperator] + [
            "composition_rule",
            "execution_strategy",
        ]

    def create_simple_composition(
        self,
        filters: list[tuple[BaseFilter, dict[str, Any]]],
        operator: CompositionOperator = CompositionOperator.AND,
    ) -> dict[str, Any]:
        """Create a simple composition criteria from filters and operator.

        Args:
            filters: List of (filter_instance, criteria) tuples
            operator: Boolean operator to use

        Returns:
            Dictionary suitable for FilterCompositionCriteria
        """
        filter_refs = []
        for filter_instance, criteria in filters:
            filter_refs.append(
                FilterReference(
                    filter_instance=filter_instance,
                    criteria=criteria,
                    weight=1.0,
                    required=True,
                )
            )

        composition_rule = CompositionRule(operator=operator, filters=filter_refs)

        return {
            "composition_rule": composition_rule,
            "execution_strategy": "optimized",
            "fail_fast": True,
            "optimize_order": True,
        }

    def explain_composition(self, criteria: FilterCompositionCriteria) -> str:
        """Generate human-readable explanation of the composition.

        Args:
            criteria: Composition criteria to explain

        Returns:
            Human-readable explanation string
        """

        def explain_rule(rule: CompositionRule, indent: int = 0) -> str:
            prefix = "  " * indent

            explanation = f"{prefix}{rule.operator.value.upper()}:\n"
            for filter_ref in rule.filters:
                weight_info = (
                    f" (weight: {filter_ref.weight:.1f})"
                    if filter_ref.weight != 1.0
                    else ""
                )
                required_info = "" if filter_ref.required else " (optional)"
                explanation += f"{prefix}  - {filter_ref.filter_instance.name}{weight_info}{required_info}\n"

            if rule.nested_rules:
                for nested_rule in rule.nested_rules:
                    explanation += explain_rule(nested_rule, indent + 1)

            return explanation

        base_explanation = (
            f"Filter Composition ({criteria.execution_strategy} execution):\n"
        )
        base_explanation += explain_rule(criteria.composition_rule)

        settings = []
        if criteria.fail_fast:
            settings.append("fail-fast enabled")
        if criteria.optimize_order:
            settings.append("order optimization enabled")
        if criteria.weighted_confidence:
            settings.append("weighted confidence")

        if settings:
            base_explanation += "\nSettings"

        return base_explanation

    def get_cache_stats(self) -> dict[str, Any]:
        """Get execution cache statistics.

        Returns:
            Dictionary with cache statistics
        """
        return {
            "enabled": True,
            "current_size": len(self.execution_cache),
            "max_size": self.max_cache_size,
            "cache_type": "LRUCache",
        }

    def clear_execution_cache(self) -> None:
        """Clear the execution cache to free memory."""
        cache_size = len(self.execution_cache)
        self.execution_cache.clear()
        logger.info(f"Cleared execution cache ({cache_size} items)")

    def cleanup(self) -> None:
        """Cleanup resources and clear caches."""
        logger.info("Starting filter composer cleanup")

        # Clear execution cache
        self.clear_execution_cache()

        # Clear performance stats
        self.performance_stats.clear()

        logger.info("Filter composer cleanup completed")
