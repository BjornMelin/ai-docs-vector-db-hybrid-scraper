"""Component-level benchmarks for individual ML components.

This module provides isolated performance testing for each component
of the advanced hybrid search system.
"""

import asyncio
import logging
import statistics
import time
from collections.abc import Sequence
from typing import Any

from pydantic import BaseModel, Field

from src.benchmarks.performance_profiler import AdvancedHybridSearchService
from src.config import Settings
from src.models.search import SearchRequest


logger = logging.getLogger(__name__)


class ComponentBenchmarkResult(BaseModel):
    """Results for a single component benchmark."""

    component_name: str = Field(..., description="Name of the benchmarked component")
    total_executions: int = Field(..., description="Total number of executions")
    successful_executions: int = Field(
        ..., description="Number of successful executions"
    )
    failed_executions: int = Field(..., description="Number of failed executions")

    # Timing metrics (in milliseconds)
    avg_latency_ms: float = Field(..., description="Average latency")
    p50_latency_ms: float = Field(..., description="50th percentile latency")
    p95_latency_ms: float = Field(..., description="95th percentile latency")
    p99_latency_ms: float = Field(..., description="99th percentile latency")
    min_latency_ms: float = Field(..., description="Minimum latency")
    max_latency_ms: float = Field(..., description="Maximum latency")

    # Quality metrics
    accuracy: float = Field(0.0, description="Component accuracy (if applicable)")
    error_rate: float = Field(..., description="Error rate")

    # Additional metrics
    throughput_per_second: float = Field(..., description="Operations per second")
    memory_usage_mb: float = Field(0.0, description="Peak memory usage in MB")


class ComponentBenchmarks:
    """Individual component benchmark runner."""

    def __init__(self, config: Settings):
        """Initialize component benchmarks.

        Args:
            config: Unified configuration

        """
        self.config = config

    async def run_all_component_benchmarks(
        self,
        search_service: AdvancedHybridSearchService,
        test_queries: Sequence[SearchRequest],
    ) -> dict[str, ComponentBenchmarkResult]:
        """Run benchmarks for all components.

        Args:
            search_service: Advanced hybrid search service
            test_queries: Test queries to use for benchmarking

        Returns:
            Dictionary mapping component names to benchmark results
        """

        queries = list(test_queries)
        results: dict[str, ComponentBenchmarkResult] = {}

        # Query Classifier benchmarks
        logger.info("Benchmarking Query Classifier...")
        results["query_classifier"] = await self.benchmark_query_classifier(
            search_service.query_classifier,
            queries,
        )

        # Model Selector benchmarks
        logger.info("Benchmarking Model Selector...")
        results["model_selector"] = await self.benchmark_model_selector(
            search_service.model_selector,
            search_service.query_classifier,
            queries,
        )

        # Adaptive Fusion Tuner benchmarks
        logger.info("Benchmarking Adaptive Fusion Tuner...")
        results["adaptive_fusion_tuner"] = await self.benchmark_adaptive_fusion_tuner(
            search_service.adaptive_fusion_tuner,
            search_service.query_classifier,
            queries,
        )

        # SPLADE Provider benchmarks
        logger.info("Benchmarking SPLADE Provider...")
        results["splade_provider"] = await self.benchmark_splade_provider(
            search_service.splade_provider,
            queries,
        )

        # End-to-end search benchmarks
        logger.info("Benchmarking End-to-End Search...")
        results["end_to_end_search"] = await self.benchmark_end_to_end_search(
            search_service, queries
        )

        return results

    async def benchmark_query_classifier(  # pylint: disable=too-many-locals
        self,
        query_classifier: Any,
        test_queries: Sequence[SearchRequest],
    ) -> ComponentBenchmarkResult:
        """Benchmark query classifier performance."""
        queries = list(test_queries)
        latencies: list[float] = []
        successes = 0
        failures = 0

        start_time = time.time()

        for query in queries:
            try:
                start = time.perf_counter()
                await query_classifier.classify_query(query.query)
                end = time.perf_counter()

                latency_ms = (end - start) * 1000
                latencies.append(latency_ms)
                successes += 1

            except (ValueError, RuntimeError, OSError) as e:
                logger.debug("Query classification failed: %s", e)
                failures += 1

        end_time = time.time()
        total_time = end_time - start_time

        total_runs = len(queries)
        return ComponentBenchmarkResult(
            component_name="query_classifier",
            total_executions=total_runs,
            successful_executions=successes,
            failed_executions=failures,
            avg_latency_ms=statistics.mean(latencies) if latencies else 0,
            p50_latency_ms=statistics.median(latencies) if latencies else 0,
            p95_latency_ms=self._percentile(latencies, 95) if latencies else 0,
            p99_latency_ms=self._percentile(latencies, 99) if latencies else 0,
            min_latency_ms=min(latencies) if latencies else 0,
            max_latency_ms=max(latencies) if latencies else 0,
            accuracy=0.0,
            error_rate=failures / total_runs if total_runs else 0,
            throughput_per_second=successes / total_time if total_time > 0 else 0,
            memory_usage_mb=0.0,
        )

    async def benchmark_model_selector(  # pylint: disable=too-many-locals
        self,
        model_selector: Any,
        query_classifier: Any,
        test_queries: Sequence[SearchRequest],
    ) -> ComponentBenchmarkResult:
        """Benchmark model selector performance."""
        queries = list(test_queries)
        latencies: list[float] = []
        successes = 0
        failures = 0

        start_time = time.time()

        subset = queries[:20]  # Use subset to avoid quota limits
        for query in subset:
            try:
                # First classify the query
                classification = await query_classifier.classify_query(query.query)

                start = time.perf_counter()
                await model_selector.select_optimal_model(classification)
                end = time.perf_counter()

                latency_ms = (end - start) * 1000
                latencies.append(latency_ms)
                successes += 1

            except (ValueError, RuntimeError, OSError) as e:
                logger.debug("Model selection failed: %s", e)
                failures += 1

        end_time = time.time()
        total_time = end_time - start_time

        total_runs = len(subset)
        return ComponentBenchmarkResult(
            component_name="model_selector",
            total_executions=total_runs,
            successful_executions=successes,
            failed_executions=failures,
            avg_latency_ms=statistics.mean(latencies) if latencies else 0,
            p50_latency_ms=statistics.median(latencies) if latencies else 0,
            p95_latency_ms=self._percentile(latencies, 95) if latencies else 0,
            p99_latency_ms=self._percentile(latencies, 99) if latencies else 0,
            min_latency_ms=min(latencies) if latencies else 0,
            max_latency_ms=max(latencies) if latencies else 0,
            accuracy=0.0,
            error_rate=failures / total_runs if total_runs else 0,
            throughput_per_second=successes / total_time if total_time > 0 else 0,
            memory_usage_mb=0.0,
        )

    async def benchmark_adaptive_fusion_tuner(  # pylint: disable=too-many-locals
        self,
        fusion_tuner: Any,
        query_classifier: Any,
        test_queries: Sequence[SearchRequest],
    ) -> ComponentBenchmarkResult:
        """Benchmark adaptive fusion tuner performance."""
        queries = list(test_queries)
        subset = queries[:15]  # Use smaller subset
        latencies: list[float] = []
        successes = 0
        failures = 0

        start_time = time.time()

        for idx, query in enumerate(subset):
            try:
                # First classify the query
                classification = await query_classifier.classify_query(query.query)

                start = time.perf_counter()
                await fusion_tuner.compute_adaptive_weights(
                    classification, f"benchmark_{idx}"
                )
                end = time.perf_counter()

                latency_ms = (end - start) * 1000
                latencies.append(latency_ms)
                successes += 1

            except (ValueError, RuntimeError, OSError) as e:
                logger.debug("Adaptive fusion failed: %s", e)
                failures += 1

        end_time = time.time()
        total_time = end_time - start_time

        total_runs = len(subset)
        return ComponentBenchmarkResult(
            component_name="adaptive_fusion_tuner",
            total_executions=total_runs,
            successful_executions=successes,
            failed_executions=failures,
            avg_latency_ms=statistics.mean(latencies) if latencies else 0,
            p50_latency_ms=statistics.median(latencies) if latencies else 0,
            p95_latency_ms=self._percentile(latencies, 95) if latencies else 0,
            p99_latency_ms=self._percentile(latencies, 99) if latencies else 0,
            min_latency_ms=min(latencies) if latencies else 0,
            max_latency_ms=max(latencies) if latencies else 0,
            accuracy=0.0,
            error_rate=failures / total_runs if total_runs else 0,
            throughput_per_second=successes / total_time if total_time > 0 else 0,
            memory_usage_mb=0.0,
        )

    async def benchmark_splade_provider(  # pylint: disable=too-many-locals
        self,
        splade_provider: Any,
        test_queries: Sequence[SearchRequest],
    ) -> ComponentBenchmarkResult:
        """Benchmark SPLADE provider performance."""
        queries = list(test_queries)
        subset = queries[:25]  # Use subset for SPLADE
        latencies: list[float] = []
        successes = 0
        failures = 0

        start_time = time.time()

        for query in subset:
            try:
                start = time.perf_counter()
                await splade_provider.generate_sparse_vector(query.query)
                end = time.perf_counter()

                latency_ms = (end - start) * 1000
                latencies.append(latency_ms)
                successes += 1

            except (ValueError, RuntimeError, OSError) as e:
                logger.debug("SPLADE generation failed: %s", e)
                failures += 1

        end_time = time.time()
        total_time = end_time - start_time

        total_runs = len(subset)
        return ComponentBenchmarkResult(
            component_name="splade_provider",
            total_executions=total_runs,
            successful_executions=successes,
            failed_executions=failures,
            avg_latency_ms=statistics.mean(latencies) if latencies else 0,
            p50_latency_ms=statistics.median(latencies) if latencies else 0,
            p95_latency_ms=self._percentile(latencies, 95) if latencies else 0,
            p99_latency_ms=self._percentile(latencies, 99) if latencies else 0,
            min_latency_ms=min(latencies) if latencies else 0,
            max_latency_ms=max(latencies) if latencies else 0,
            accuracy=0.0,
            error_rate=failures / total_runs if total_runs else 0,
            throughput_per_second=successes / total_time if total_time > 0 else 0,
            memory_usage_mb=0.0,
        )

    async def benchmark_end_to_end_search(  # pylint: disable=too-many-locals
        self,
        search_service: AdvancedHybridSearchService,
        test_queries: Sequence[SearchRequest],
    ) -> ComponentBenchmarkResult:
        """Benchmark end-to-end search performance."""
        queries = list(test_queries)
        subset = queries[:10]  # Use small subset for full end-to-end
        latencies: list[float] = []
        successes = 0
        failures = 0

        start_time = time.time()

        for query in subset:
            try:
                start = time.perf_counter()
                await search_service.hybrid_search(query)
                end = time.perf_counter()

                latency_ms = (end - start) * 1000
                latencies.append(latency_ms)
                successes += 1

            except (ValueError, RuntimeError, OSError) as e:
                logger.debug("End-to-end search failed: %s", e)
                failures += 1

        end_time = time.time()
        total_time = end_time - start_time

        total_runs = len(subset)
        return ComponentBenchmarkResult(
            component_name="end_to_end_search",
            total_executions=total_runs,
            successful_executions=successes,
            failed_executions=failures,
            avg_latency_ms=statistics.mean(latencies) if latencies else 0,
            p50_latency_ms=statistics.median(latencies) if latencies else 0,
            p95_latency_ms=self._percentile(latencies, 95) if latencies else 0,
            p99_latency_ms=self._percentile(latencies, 99) if latencies else 0,
            min_latency_ms=min(latencies) if latencies else 0,
            max_latency_ms=max(latencies) if latencies else 0,
            accuracy=0.0,
            error_rate=failures / total_runs if total_runs else 0,
            throughput_per_second=successes / total_time if total_time > 0 else 0,
            memory_usage_mb=0.0,
        )

    def _percentile(self, data: list[float], percentile: float) -> float:
        """Calculate percentile of a list of values."""
        if not data:
            return 0.0
        sorted_data = sorted(data)
        index = int((percentile / 100) * len(sorted_data))
        if index >= len(sorted_data):
            index = len(sorted_data) - 1
        return sorted_data[index]

    async def benchmark_cache_performance(
        self,
        search_service: AdvancedHybridSearchService,
        test_queries: Sequence[SearchRequest],
    ) -> dict[str, float]:
        """Benchmark cache performance specifically."""
        cache_metrics: dict[str, float] = {}
        queries = list(test_queries)

        # Clear caches first
        if hasattr(search_service.splade_provider, "clear_cache"):
            search_service.splade_provider.clear_cache()

        # First run (cold cache)
        cold_latencies: list[float] = []
        for query in queries[:5]:
            try:
                start = time.perf_counter()
                await search_service.splade_provider.generate_sparse_vector(query.query)
                end = time.perf_counter()
                cold_latencies.append((end - start) * 1000)
            except (ValueError, RuntimeError, OSError) as e:
                logger.warning(
                    "SPLADE cold run failed for query '%s': %s", query.query, e
                )
                continue

        # Second run (warm cache) - same queries
        warm_latencies: list[float] = []
        for query in queries[:5]:
            try:
                start = time.perf_counter()
                await search_service.splade_provider.generate_sparse_vector(query.query)
                end = time.perf_counter()
                warm_latencies.append((end - start) * 1000)
            except (ValueError, RuntimeError, OSError) as e:
                logger.warning(
                    "SPLADE warm run failed for query '%s': %s", query.query, e
                )
                continue

        if cold_latencies and warm_latencies:
            cache_metrics["cold_avg_latency_ms"] = statistics.mean(cold_latencies)
            cache_metrics["warm_avg_latency_ms"] = statistics.mean(warm_latencies)
            cache_metrics["cache_speedup_factor"] = statistics.mean(
                cold_latencies
            ) / statistics.mean(warm_latencies)

        # Get cache statistics
        if hasattr(search_service.splade_provider, "get_cache_stats"):
            cache_stats = search_service.splade_provider.get_cache_stats()
            cache_metrics.update(cache_stats)

        return cache_metrics

    async def benchmark_concurrent_component_access(
        self,
        search_service: AdvancedHybridSearchService,
        test_queries: Sequence[SearchRequest],
    ) -> dict[str, ComponentBenchmarkResult]:
        """Benchmark components under concurrent access."""
        concurrent_results: dict[str, ComponentBenchmarkResult] = {}
        queries = list(test_queries)

        # Test query classifier under concurrency
        async def concurrent_classify(query_text: str) -> Any:
            return await search_service.query_classifier.classify_query(query_text)

        subset = queries[:10]
        if not subset:
            return concurrent_results

        tasks = [concurrent_classify(query.query) for query in subset]
        if not tasks:
            return concurrent_results

        start_time = time.time()
        results = await asyncio.gather(*tasks, return_exceptions=True)
        end_time = time.time()

        successes = sum(1 for r in results if not isinstance(r, Exception))
        failures = len(results) - successes
        total_time = end_time - start_time

        avg_latency_ms = (total_time * 1000) / len(tasks) if len(tasks) else 0
        concurrent_results["query_classifier_concurrent"] = ComponentBenchmarkResult(
            component_name="query_classifier_concurrent",
            total_executions=len(tasks),
            successful_executions=successes,
            failed_executions=failures,
            avg_latency_ms=avg_latency_ms,
            p50_latency_ms=avg_latency_ms,
            p95_latency_ms=avg_latency_ms,
            p99_latency_ms=avg_latency_ms,
            min_latency_ms=0,
            max_latency_ms=total_time * 1000,
            accuracy=0.0,
            error_rate=failures / len(tasks) if len(tasks) else 0,
            throughput_per_second=successes / total_time if total_time > 0 else 0,
            memory_usage_mb=0.0,
        )

        return concurrent_results
