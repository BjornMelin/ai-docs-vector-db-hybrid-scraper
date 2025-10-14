"""HNSW parameter optimization and adaptive ef_retrieve implementation."""

import asyncio
import logging
import time
from dataclasses import dataclass, field
from typing import Any

import numpy as np
from qdrant_client import models

from src.config.loader import Settings
from src.services.errors import QdrantServiceError


logger = logging.getLogger(__name__)


@dataclass(slots=True)
class AdaptiveEfConfig:
    """Configuration knobs for adaptive EF selection."""

    time_budget_ms: int = 100
    min_ef: int = 50
    max_ef: int = 200
    step_size: int = 25
    target_limit: int = 10


@dataclass(slots=True)
class AdaptiveSearchContext:
    """Container holding search parameters and configuration."""

    collection_name: str
    query_vector: list[float]
    config: AdaptiveEfConfig


@dataclass
class AdaptiveSearchState:
    """Track adaptive EF search progress."""

    best_ef: int
    best_results: Any | None = None
    search_times_ms: list[float] = field(default_factory=list)
    tested_ef_values: list[int] = field(default_factory=list)

    def record(self, ef_value: int, results: Any, elapsed_ms: float) -> None:
        """Update tracking with a new EF attempt."""
        self.best_ef = ef_value
        self.best_results = results
        self.search_times_ms.append(elapsed_ms)
        self.tested_ef_values.append(ef_value)

    @property
    def final_search_time(self) -> float:
        """Return the most recent search time in milliseconds."""
        return self.search_times_ms[-1] if self.search_times_ms else 0.0


class HNSWOptimizer:
    """Intelligent HNSW parameter optimization for Qdrant collections."""

    def __init__(self, config: Settings, qdrant_service: Any):
        """Initialize HNSW optimizer.

        Args:
            config: Unified configuration
            qdrant_service: QdrantService instance
        """
        self.config = config
        self.qdrant_service = qdrant_service
        self.logger = logger

        # Performance tracking
        self.performance_cache = {}
        self.adaptive_ef_cache = {}
        self._initialized = False

    def is_initialized(self) -> bool:
        """Return whether the optimizer has been initialized."""
        return self._initialized

    def _is_qdrant_ready(self) -> bool:
        """Normalize whatever form of readiness the service exposes."""
        readiness = getattr(self.qdrant_service, "is_initialized", None)
        if callable(readiness):
            return bool(readiness())
        if readiness is not None:
            return bool(readiness)
        return bool(getattr(self.qdrant_service, "_initialized", False))

    def _log_or_propagate(
        self, exc: Exception, context: str, *, rethrow_cancelled: bool = True
    ) -> None:
        """Log exception as warning or propagate if cancelled."""
        if rethrow_cancelled and isinstance(exc, asyncio.CancelledError):
            raise exc
        self.logger.warning("%s failed: %s", context, exc)

    async def initialize(self) -> None:
        """Initialize optimizer."""
        if self.is_initialized():
            return

        if not self._is_qdrant_ready():
            msg = "QdrantService must be initialized before HNSWOptimizer"
            raise QdrantServiceError(msg)

        self._initialized = True
        self.logger.info("HNSW optimizer initialized")

    async def adaptive_ef_retrieve(
        self,
        collection_name: str,
        query_vector: list[float],
        *,
        config: AdaptiveEfConfig | None = None,
    ) -> dict[str, Any]:
        """Dynamically adjust ef parameter based on time budget.

        Args:
            collection_name: Collection to search
            query_vector: Query vector
            config: Adaptive EF configuration overrides

        Returns:
            Search results with optimal ef value used
        """
        effective_config = config or AdaptiveEfConfig()
        context = AdaptiveSearchContext(
            collection_name=collection_name,
            query_vector=query_vector,
            config=effective_config,
        )

        cache_key = self._make_adaptive_cache_key(context)
        client = await self.qdrant_service.get_client()

        cached = await self._try_cached_search(client, cache_key, context)
        if cached is not None:
            return cached

        state = AdaptiveSearchState(best_ef=effective_config.min_ef)
        self.logger.debug(
            "Starting adaptive EF for %s with budget %dms",
            collection_name,
            effective_config.time_budget_ms,
        )
        await self._run_adaptive_search(
            client=client,
            state=state,
            context=context,
        )

        self._store_adaptive_cache(cache_key, state)

        return self._build_adaptive_response(
            state=state,
            context=context,
            source="adaptive",
        )

    def _make_adaptive_cache_key(self, context: AdaptiveSearchContext) -> str:
        """Generate cache key for adaptive EF lookups."""
        cfg = context.config
        return (
            f"{context.collection_name}:{cfg.time_budget_ms}:{cfg.min_ef}:{cfg.max_ef}"
        )

    async def _try_cached_search(
        self,
        client: Any,
        cache_key: str,
        context: AdaptiveSearchContext,
    ) -> dict[str, Any] | None:
        """Execute cached EF search if available."""
        cached_entry = self.adaptive_ef_cache.get(cache_key)
        if not cached_entry:
            return None

        cached_ef = cached_entry["optimal_ef"]
        self.logger.debug(
            "Using cached optimal EF %d for collection %s",
            cached_ef,
            context.collection_name,
        )

        timed_query = await self._query_points_with_timing(
            client,
            context,
            cached_ef,
            log_context=f"Cached search at EF {cached_ef}",
        )

        if timed_query is None:
            return None

        search_time_ms, results = timed_query
        return {
            "results": results.points,
            "ef_used": cached_ef,
            "search_time_ms": search_time_ms,
            "time_budget_ms": context.config.time_budget_ms,
            "budget_utilized_percent": self._budget_utilization(
                search_time_ms, context.config.time_budget_ms
            ),
            "source": "cache",
        }

    async def _run_adaptive_search(
        self,
        *,
        client: Any,
        state: AdaptiveSearchState,
        context: AdaptiveSearchContext,
    ) -> None:
        """Iteratively search with increasing EF until the time budget is met."""
        config = context.config
        current_ef = config.min_ef
        while current_ef <= config.max_ef:
            timed_query = await self._query_points_with_timing(
                client,
                context,
                current_ef,
                log_context=f"Search at EF {current_ef}",
            )
            if timed_query is None:
                break

            search_time_ms, results = timed_query
            self.logger.debug("EF %d: %.1fms", current_ef, search_time_ms)
            state.record(current_ef, results, search_time_ms)

            if self._should_stop(search_time_ms, config):
                self.logger.debug("Stopping at EF %d due to time budget", current_ef)
                break

            next_ef = self._next_ef(
                current_ef=current_ef,
                elapsed_ms=search_time_ms,
                config=config,
            )
            if next_ef is None:
                break
            current_ef = next_ef

    async def _query_points_with_timing(
        self,
        client: Any,
        context: AdaptiveSearchContext,
        ef_value: int,
        *,
        log_context: str,
    ) -> tuple[float, Any] | None:
        """Execute a query and capture elapsed time in milliseconds."""
        start_time = time.time()
        try:
            results = await client.query_points(
                collection_name=context.collection_name,
                query=context.query_vector,
                using="dense",
                limit=context.config.target_limit,
                params=models.SearchParams(hnsw_ef=ef_value, exact=False),
                with_payload=True,
                with_vectors=False,
            )
        except (ValueError, TypeError) as exc:
            self._log_or_propagate(exc, log_context)
            return None
        except Exception as exc:
            self._log_or_propagate(exc, log_context)
            return None

        elapsed_ms = (time.time() - start_time) * 1000
        return elapsed_ms, results

    @staticmethod
    def _should_stop(search_time_ms: float, config: AdaptiveEfConfig) -> bool:
        """Determine whether the search should stop based on budget usage."""
        if config.time_budget_ms <= 0:
            return True
        return search_time_ms >= config.time_budget_ms * 0.8

    @staticmethod
    def _next_ef(
        *,
        current_ef: int,
        elapsed_ms: float,
        config: AdaptiveEfConfig,
    ) -> int | None:
        """Calculate the next EF value or signal completion."""
        if current_ef >= config.max_ef:
            return None

        if config.time_budget_ms <= 0 or elapsed_ms < config.time_budget_ms * 0.5:
            increment = config.step_size
        else:
            increment = max(config.step_size // 2, 1)

        next_ef = min(current_ef + increment, config.max_ef)
        if next_ef == current_ef:
            return None
        return next_ef

    def _store_adaptive_cache(self, cache_key: str, state: AdaptiveSearchState) -> None:
        """Persist adaptive search findings and enforce cache size limits."""
        self.adaptive_ef_cache[cache_key] = {
            "optimal_ef": state.best_ef,
            "expected_time_ms": state.final_search_time,
            "tested_ef_values": state.tested_ef_values,
            "tested_times_ms": state.search_times_ms,
            "timestamp": time.time(),
        }

        if len(self.adaptive_ef_cache) <= 100:
            return

        oldest_key = min(
            self.adaptive_ef_cache.keys(),
            key=lambda key: self.adaptive_ef_cache[key]["timestamp"],
        )
        del self.adaptive_ef_cache[oldest_key]

    def _build_adaptive_response(
        self,
        *,
        state: AdaptiveSearchState,
        context: AdaptiveSearchContext,
        source: str,
    ) -> dict[str, Any]:
        """Construct the response payload from adaptive search data."""
        best_results = state.best_results.points if state.best_results else []
        final_time = state.final_search_time
        return {
            "results": best_results,
            "ef_used": state.best_ef,
            "search_time_ms": final_time,
            "time_budget_ms": context.config.time_budget_ms,
            "budget_utilized_percent": self._budget_utilization(
                final_time, context.config.time_budget_ms
            ),
            "ef_progression": state.tested_ef_values,
            "time_progression": state.search_times_ms,
            "source": source,
        }

    @staticmethod
    def _budget_utilization(search_time_ms: float, time_budget_ms: int) -> float:
        """Compute time budget utilization percentage."""
        if time_budget_ms <= 0:
            return 0.0
        return (search_time_ms / time_budget_ms) * 100

    def get_collection_specific_hnsw_config(
        self, collection_type: str
    ) -> dict[str, Any]:
        """Get optimized HNSW config for collection type.

        Args:
            collection_type: Type of collection (api_reference, tutorials,
            blog_posts, etc.)

        Returns:
            HNSW configuration dictionary
        """
        # Collection-specific optimization profiles
        configs = {
            "api_reference": {
                "m": 20,
                "ef_construct": 300,
                "full_scan_threshold": 5000,
                "description": "High accuracy for API documentation",
                "target_use_case": "Precise technical reference lookups",
            },
            "tutorials": {
                "m": 16,
                "ef_construct": 200,
                "full_scan_threshold": 10000,
                "description": "Balanced for tutorial content",
                "target_use_case": "Educational content discovery",
            },
            "blog_posts": {
                "m": 12,
                "ef_construct": 150,
                "full_scan_threshold": 20000,
                "description": "Fast for blog content",
                "target_use_case": "Quick content browsing",
            },
            "code_examples": {
                "m": 18,
                "ef_construct": 250,
                "full_scan_threshold": 8000,
                "description": "Code-specific optimization",
                "target_use_case": "Precise code snippet matching",
            },
            "general": {
                "m": 16,
                "ef_construct": 200,
                "full_scan_threshold": 10000,
                "description": "Default balanced configuration",
                "target_use_case": "General purpose documentation",
            },
        }

        config = configs.get(collection_type, configs["general"])

        # Add runtime ef recommendations based on collection type
        ef_recommendations = {
            "api_reference": {"min_ef": 100, "balanced_ef": 150, "max_ef": 200},
            "tutorials": {"min_ef": 75, "balanced_ef": 100, "max_ef": 150},
            "blog_posts": {"min_ef": 50, "balanced_ef": 75, "max_ef": 100},
            "code_examples": {"min_ef": 100, "balanced_ef": 125, "max_ef": 175},
            "general": {"min_ef": 75, "balanced_ef": 100, "max_ef": 150},
        }

        config["ef_recommendations"] = ef_recommendations.get(
            collection_type, ef_recommendations["general"]
        )

        return config

    async def optimize_collection_hnsw(
        self,
        collection_name: str,
        collection_type: str,
        test_queries: list[list[float]] | None = None,
    ) -> dict[str, Any]:
        """Optimize HNSW parameters for a specific collection.

        Args:
            collection_name: Collection to optimize
            collection_type: Type of collection for optimization profile
            test_queries: Optional test queries for validation

        Returns:
            Optimization results and recommendations
        """
        self.logger.info(
            "Optimizing HNSW parameters for collection %s (type: %s)",
            collection_name,
            collection_type,
        )

        # Get recommended configuration
        recommended_config = self.get_collection_specific_hnsw_config(collection_type)

        # Test current performance if queries provided
        current_performance = None
        if test_queries:
            current_performance = await self._test_search_performance(
                collection_name,
                test_queries,
                recommended_config["ef_recommendations"]["balanced_ef"],
            )

        # Check if collection needs HNSW updates
        client = await self.qdrant_service.get_client()
        collection_info = await client.get_collection(collection_name)
        current_config = self._extract_current_hnsw_config(collection_info)

        needs_update = self._compare_hnsw_configs(current_config, recommended_config)

        optimization_result = {
            "collection_name": collection_name,
            "collection_type": collection_type,
            "current_config": current_config,
            "recommended_config": recommended_config,
            "needs_update": needs_update,
            "current_performance": current_performance,
            "optimization_timestamp": time.time(),
        }

        if needs_update:
            self.logger.info(
                "Collection %s would benefit from HNSW optimization", collection_name
            )
            optimization_result["update_recommendation"] = {
                "action": "recreate_collection",
                "reason": "HNSW parameters can only be set during collection creation",
                "estimated_improvement": self._estimate_performance_improvement(
                    current_config, recommended_config
                ),
            }
        else:
            self.logger.info(
                "Collection %s already has optimal HNSW configuration", collection_name
            )
            optimization_result["update_recommendation"] = {
                "action": "no_update_needed",
                "reason": (
                    "Current configuration is already optimal or close to recommended"
                ),
            }

        return optimization_result

    def _extract_current_hnsw_config(self, collection_info: Any) -> dict[str, Any]:
        """Extract current HNSW configuration from collection info.

        Args:
            collection_info: Collection information from Qdrant

        Returns:
            Current HNSW configuration
        """
        try:
            # Access vector configuration
            vectors_config = collection_info.config.params.vectors

            if hasattr(vectors_config, "dense") and vectors_config.dense:
                hnsw_config = vectors_config.dense.hnsw_config
                if hnsw_config:
                    return {
                        "m": getattr(hnsw_config, "m", 16),
                        "ef_construct": getattr(hnsw_config, "ef_construct", 128),
                        "full_scan_threshold": getattr(
                            hnsw_config, "full_scan_threshold", 10000
                        ),
                    }
        except (ValueError, ConnectionError, TimeoutError, RuntimeError) as e:
            self.logger.debug("Could not extract HNSW config: %s", e)

        # Return defaults if extraction fails
        return {
            "m": 16,
            "ef_construct": 128,
            "full_scan_threshold": 10000,
        }

    def _compare_hnsw_configs(
        self, current: dict[str, Any], recommended: dict[str, Any]
    ) -> bool:
        """Compare current and recommended HNSW configurations.

        Args:
            current: Current HNSW configuration
            recommended: Recommended HNSW configuration

        Returns:
            True if update is recommended
        """
        # Check significant differences
        m_diff = abs(current.get("m", 16) - recommended.get("m", 16))
        ef_construct_diff = abs(
            current.get("ef_construct", 128) - recommended.get("ef_construct", 200)
        )

        # Thresholds for significant differences
        return m_diff >= 4 or ef_construct_diff >= 50

    def _estimate_performance_improvement(
        self, current: dict[str, Any], recommended: dict[str, Any]
    ) -> dict[str, Any]:
        """Estimate performance improvement from HNSW optimization.

        Args:
            current: Current configuration
            recommended: Recommended configuration

        Returns:
            Estimated improvements
        """
        current_m = current.get("m", 16)
        current_ef = current.get("ef_construct", 128)
        recommended_m = recommended.get("m", 16)
        recommended_ef = recommended.get("ef_construct", 200)

        # Rough estimates based on HNSW research
        # Higher m generally improves recall but increases memory usage
        # Higher ef_construct improves index quality but increases build time

        recall_improvement = 0
        if recommended_m > current_m:
            recall_improvement += (
                recommended_m - current_m
            ) * 0.5  # ~0.5% per m increase
        if recommended_ef > current_ef:
            recall_improvement += (
                recommended_ef - current_ef
            ) * 0.02  # ~0.02% per ef_construct increase

        latency_change = 0
        if recommended_m > current_m:
            latency_change += (
                recommended_m - current_m
            ) * 2  # ~2% latency increase per m
        if recommended_ef > current_ef:
            latency_change -= (
                recommended_ef - current_ef
            ) * 0.1  # Better quality may reduce latency slightly

        memory_change = 0
        if recommended_m != current_m:
            memory_change = ((recommended_m / current_m) - 1) * 100  # Proportional to m

        return {
            "estimated_recall_improvement_percent": max(0, recall_improvement),
            "estimated_latency_change_percent": latency_change,
            "estimated_memory_change_percent": memory_change,
            "build_time_change_percent": max(0, (recommended_ef - current_ef) * 0.5),
            "confidence": "medium",  # These are rough estimates
        }

    async def _test_search_performance(
        self, collection_name: str, test_queries: list[list[float]], ef_value: int
    ) -> dict[str, Any]:
        """Test search performance with given parameters.

        Args:
            collection_name: Collection to test
            test_queries: Test query vectors
            ef_value: EF value to use

        Returns:
            Performance metrics
        """
        search_times = []

        for query_vector in test_queries[:10]:  # Limit to 10 queries for performance
            start_time = time.time()

            try:
                client = await self.qdrant_service.get_client()
                result = await client.query_points(
                    collection_name=collection_name,
                    query=query_vector,
                    using="dense",
                    limit=10,
                    params=models.SearchParams(hnsw_ef=ef_value, exact=False),
                    with_payload=True,
                    with_vectors=False,
                )

                if result is None:
                    self.logger.warning(
                        "Performance test query returned no results at EF %d", ef_value
                    )
                    continue

                search_time_ms = (time.time() - start_time) * 1000
                search_times.append(search_time_ms)

            except (ValueError, ConnectionError, TimeoutError, RuntimeError) as exc:
                self._log_or_propagate(exc, "Performance test query")
                continue
            except Exception as exc:
                self._log_or_propagate(exc, "Performance test query")
                continue

        if search_times:
            return {
                "avg_search_time_ms": float(np.mean(search_times)),
                "p95_search_time_ms": float(np.percentile(search_times, 95)),
                "min_search_time_ms": float(np.min(search_times)),
                "max_search_time_ms": float(np.max(search_times)),
                "queries_tested": len(search_times),
                "ef_used": ef_value,
            }
        return {
            "error": "No successful queries",
            "queries_tested": 0,
            "ef_used": ef_value,
        }

    def get_performance_cache_stats(self) -> dict[str, Any]:
        """Get statistics about performance caching.

        Returns:
            Cache statistics
        """
        return {
            "adaptive_ef_cache_size": len(self.adaptive_ef_cache),
            "performance_cache_size": len(self.performance_cache),
            "cache_entries": list(self.adaptive_ef_cache.keys()),
        }

    async def cleanup(self) -> None:
        """Cleanup optimizer resources."""
        self.performance_cache.clear()
        self.adaptive_ef_cache.clear()
        self._initialized = False
        self.logger.info("HNSW optimizer cleaned up")
