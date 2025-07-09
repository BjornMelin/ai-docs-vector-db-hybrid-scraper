"""Performance benchmarking suite for baseline and optimization tracking.

This module provides comprehensive benchmarking capabilities for establishing
performance baselines and tracking optimization improvements across all system components.
"""

import asyncio
import gc
import logging
import time
from contextlib import asynccontextmanager
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import psutil
import pytest
from opentelemetry import trace

from src.config.settings import Settings
from src.infrastructure.clients.qdrant_client import QdrantClientWrapper
from src.services.cache.intelligent import IntelligentCache
from src.services.embeddings.manager import EmbeddingManager
from src.services.monitoring.performance_monitor import RealTimePerformanceMonitor
from src.services.query_processing.pipeline import QueryPipeline


logger = logging.getLogger(__name__)
tracer = trace.get_tracer(__name__)


@dataclass
class BenchmarkResult:
    """Result from a single benchmark run."""

    name: str
    duration_ms: float
    memory_mb: float
    cpu_percent: float
    iterations: int
    throughput: float
    p50_ms: float
    p95_ms: float
    p99_ms: float
    metadata: dict[str, Any]


@dataclass
class BenchmarkSuite:
    """Collection of benchmark results with summary statistics."""

    timestamp: datetime
    python_version: str
    platform: str
    results: list[BenchmarkResult]
    baseline_metrics: dict[str, float] | None = None

    def calculate_improvements(self) -> dict[str, float]:
        """Calculate performance improvements over baseline."""
        if not self.baseline_metrics:
            return {}

        improvements = {}
        for result in self.results:
            if result.name in self.baseline_metrics:
                baseline = self.baseline_metrics[result.name]
                improvement = ((baseline - result.duration_ms) / baseline) * 100
                improvements[result.name] = improvement

        return improvements

    def to_report(self) -> dict[str, Any]:
        """Generate comprehensive performance report."""
        improvements = self.calculate_improvements()

        return {
            "timestamp": self.timestamp.isoformat(),
            "python_version": self.python_version,
            "platform": self.platform,
            "summary": {
                "total_benchmarks": len(self.results),
                "avg_duration_ms": sum(r.duration_ms for r in self.results)
                / len(self.results),
                "avg_memory_mb": sum(r.memory_mb for r in self.results)
                / len(self.results),
                "avg_throughput": sum(r.throughput for r in self.results)
                / len(self.results),
            },
            "improvements": improvements,
            "results": [
                {
                    "name": r.name,
                    "duration_ms": r.duration_ms,
                    "memory_mb": r.memory_mb,
                    "cpu_percent": r.cpu_percent,
                    "throughput": r.throughput,
                    "p50_ms": r.p50_ms,
                    "p95_ms": r.p95_ms,
                    "p99_ms": r.p99_ms,
                    "improvement_percent": improvements.get(r.name, 0),
                    "metadata": r.metadata,
                }
                for r in self.results
            ],
        }


class PerformanceBenchmark:
    """Core performance benchmarking framework."""

    def __init__(
        self, settings: Settings, monitor: RealTimePerformanceMonitor | None = None
    ):
        """Initialize benchmark framework."""
        self.settings = settings
        self.monitor = monitor or RealTimePerformanceMonitor()
        self.results: list[BenchmarkResult] = []

    @asynccontextmanager
    async def measure_performance(
        self, name: str, iterations: int = 100
    ) -> AsyncGenerator[dict[str, Any]]:
        """Context manager for measuring performance metrics."""
        # Force garbage collection before measurement
        gc.collect()
        await asyncio.sleep(0.1)  # Let system settle

        # Record initial state
        process = psutil.Process()
        start_memory = process.memory_info().rss / 1024 / 1024  # MB
        start_cpu = process.cpu_percent()

        # Timing storage
        timings = []
        context = {"timings": timings}

        # Start timing
        start_time = time.perf_counter()

        try:
            yield context
        finally:
            # Calculate metrics
            end_time = time.perf_counter()
            duration_ms = (end_time - start_time) * 1000

            end_memory = process.memory_info().rss / 1024 / 1024
            memory_delta = end_memory - start_memory
            cpu_percent = process.cpu_percent() - start_cpu

            # Calculate timing percentiles if available
            if timings:
                timings_sorted = sorted(timings)
                p50 = timings_sorted[len(timings_sorted) // 2]
                p95 = timings_sorted[int(len(timings_sorted) * 0.95)]
                p99 = timings_sorted[int(len(timings_sorted) * 0.99)]
            else:
                p50 = p95 = p99 = (
                    duration_ms / iterations if iterations > 0 else duration_ms
                )

            # Calculate throughput
            throughput = iterations / (duration_ms / 1000) if duration_ms > 0 else 0

            # Create result
            result = BenchmarkResult(
                name=name,
                duration_ms=duration_ms,
                memory_mb=memory_delta,
                cpu_percent=cpu_percent,
                iterations=iterations,
                throughput=throughput,
                p50_ms=p50,
                p95_ms=p95,
                p99_ms=p99,
                metadata={
                    "start_memory_mb": start_memory,
                    "end_memory_mb": end_memory,
                    "timing_count": len(timings),
                },
            )

            self.results.append(result)
            logger.info(
                f"Benchmark '{name}' completed: "
                f"{duration_ms:.2f}ms, {throughput:.2f} ops/s, "
                f"P95: {p95:.2f}ms"
            )

    async def benchmark_embedding_generation(
        self,
        embedding_manager: EmbeddingManager,
        texts: list[str],
        batch_size: int = 100,
    ) -> BenchmarkResult:
        """Benchmark embedding generation performance."""
        with tracer.start_as_current_span("benchmark_embedding_generation") as span:
            span.set_attribute("batch_size", batch_size)
            span.set_attribute("text_count", len(texts))

            async with self.measure_performance(
                "embedding_generation", iterations=len(texts)
            ) as context:
                for i in range(0, len(texts), batch_size):
                    batch = texts[i : i + batch_size]

                    batch_start = time.perf_counter()
                    embeddings = await embedding_manager.embed_documents(batch)
                    batch_duration = (time.perf_counter() - batch_start) * 1000

                    # Record individual batch timing
                    context["timings"].extend(
                        [batch_duration / len(batch)] * len(batch)
                    )

                    # Validate embeddings
                    assert len(embeddings) == len(batch)
                    assert all(
                        len(emb) == self.settings.EMBEDDING_DIMENSION
                        for emb in embeddings
                    )

            return self.results[-1]

    async def benchmark_vector_search(
        self,
        qdrant_client: QdrantClientWrapper,
        queries: list[list[float]],
        collection_name: str,
        limit: int = 10,
    ) -> BenchmarkResult:
        """Benchmark vector search performance."""
        with tracer.start_as_current_span("benchmark_vector_search") as span:
            span.set_attribute("query_count", len(queries))
            span.set_attribute("collection", collection_name)
            span.set_attribute("limit", limit)

            async with self.measure_performance(
                "vector_search", iterations=len(queries)
            ) as context:
                for query_vector in queries:
                    query_start = time.perf_counter()

                    results = await qdrant_client.search(
                        collection_name=collection_name,
                        query_vector=query_vector,
                        limit=limit,
                    )

                    query_duration = (time.perf_counter() - query_start) * 1000
                    context["timings"].append(query_duration)

                    # Validate results
                    assert len(results) <= limit

            return self.results[-1]

    async def benchmark_cache_operations(
        self, cache: IntelligentCache, operations: int = 1000
    ) -> BenchmarkResult:
        """Benchmark cache operations (get/set/invalidate)."""
        with tracer.start_as_current_span("benchmark_cache_operations") as span:
            span.set_attribute("operations", operations)

            async with self.measure_performance(
                "cache_operations",
                iterations=operations * 3,  # get, set, invalidate
            ) as context:
                # Benchmark set operations
                for i in range(operations):
                    key = f"bench_key_{i}"
                    value = {"data": f"value_{i}", "timestamp": time.time()}

                    set_start = time.perf_counter()
                    await cache.set(key, value, ttl=300)
                    set_duration = (time.perf_counter() - set_start) * 1000
                    context["timings"].append(set_duration)

                # Benchmark get operations
                for i in range(operations):
                    key = f"bench_key_{i}"

                    get_start = time.perf_counter()
                    value = await cache.get(key)
                    get_duration = (time.perf_counter() - get_start) * 1000
                    context["timings"].append(get_duration)

                    assert value is not None

                # Benchmark invalidation
                for i in range(operations):
                    key = f"bench_key_{i}"

                    inv_start = time.perf_counter()
                    await cache.invalidate(key)
                    inv_duration = (time.perf_counter() - inv_start) * 1000
                    context["timings"].append(inv_duration)

            return self.results[-1]

    async def benchmark_query_pipeline(
        self, pipeline: QueryPipeline, queries: list[str], top_k: int = 10
    ) -> BenchmarkResult:
        """Benchmark end-to-end query pipeline performance."""
        with tracer.start_as_current_span("benchmark_query_pipeline") as span:
            span.set_attribute("query_count", len(queries))
            span.set_attribute("top_k", top_k)

            async with self.measure_performance(
                "query_pipeline", iterations=len(queries)
            ) as context:
                for query in queries:
                    pipeline_start = time.perf_counter()

                    results = await pipeline.process_query(
                        query=query,
                        top_k=top_k,
                        enable_hyde=True,
                        enable_reranking=True,
                    )

                    pipeline_duration = (time.perf_counter() - pipeline_start) * 1000
                    context["timings"].append(pipeline_duration)

                    # Validate results
                    assert results is not None
                    assert len(results.documents) <= top_k

            return self.results[-1]

    async def benchmark_database_operations(
        self, operations: int = 100
    ) -> BenchmarkResult:
        """Benchmark database connection pool and query performance."""
        # This would be implemented with actual database operations
        # For now, simulate with async operations
        async with self.measure_performance(
            "database_operations", iterations=operations
        ) as context:
            for _ in range(operations):
                op_start = time.perf_counter()

                # Simulate database operation
                await asyncio.sleep(0.001)  # 1ms simulated query

                op_duration = (time.perf_counter() - op_start) * 1000
                context["timings"].append(op_duration)

        return self.results[-1]

    async def run_baseline_suite(self, components: dict[str, Any]) -> BenchmarkSuite:
        """Run complete baseline benchmark suite."""
        logger.info("Starting baseline benchmark suite")

        # Platform info
        import platform  # noqa: PLC0415
        import sys  # noqa: PLC0415

        # Generate test data
        test_texts = [
            f"This is test document {i} for benchmarking performance."
            for i in range(1000)
        ]
        test_queries = [
            "performance optimization",
            "vector search",
            "embedding generation",
        ]

        # Run benchmarks
        if "embedding_manager" in components:
            await self.benchmark_embedding_generation(
                components["embedding_manager"],
                test_texts[:100],  # Limit for baseline
            )

        if "qdrant_client" in components and "embedding_manager" in components:
            # Generate query embeddings
            query_embeddings = await components["embedding_manager"].embed_documents(
                test_queries
            )
            await self.benchmark_vector_search(
                components["qdrant_client"],
                query_embeddings,
                components.get("collection_name", "documents"),
            )

        if "cache" in components:
            await self.benchmark_cache_operations(components["cache"], operations=100)

        if "query_pipeline" in components:
            await self.benchmark_query_pipeline(
                components["query_pipeline"], test_queries
            )

        # Always run database benchmark
        await self.benchmark_database_operations(operations=50)

        # Create suite
        suite = BenchmarkSuite(
            timestamp=datetime.now(UTC),
            python_version=sys.version,
            platform=platform.platform(),
            results=self.results.copy(),
        )

        logger.info(
            f"Baseline benchmark suite completed with {len(suite.results)} benchmarks"
        )
        return suite

    def save_results(self, suite: BenchmarkSuite, filepath: Path) -> None:
        """Save benchmark results to file."""
        import json  # noqa: PLC0415

        filepath.parent.mkdir(parents=True, exist_ok=True)

        with filepath.open("w") as f:
            json.dump(suite.to_report(), f, indent=2)

        logger.info(f"Benchmark results saved to {filepath}")

    def load_baseline(self, filepath: Path) -> dict[str, float]:
        """Load baseline metrics from file."""
        import json  # noqa: PLC0415

        if not filepath.exists():
            logger.warning(f"Baseline file not found: {filepath}")
            return {}

        with filepath.open() as f:
            data = json.load(f)

        # Extract baseline metrics
        baseline = {}
        for result in data.get("results", []):
            baseline[result["name"]] = result["duration_ms"]

        logger.info(f"Loaded {len(baseline)} baseline metrics")
        return baseline


# Pytest integration
@pytest.mark.benchmark
async def test_performance_baseline(benchmark_components):
    """Pytest benchmark for establishing performance baseline."""
    benchmark = PerformanceBenchmark(benchmark_components["settings"])
    suite = await benchmark.run_baseline_suite(benchmark_components)

    # Save results
    results_path = Path("benchmarks/results/baseline.json")
    benchmark.save_results(suite, results_path)

    # Assert performance targets
    for result in suite.results:
        assert result.p95_ms < 100, f"{result.name} P95 latency exceeds 100ms target"
        assert result.throughput > 10, f"{result.name} throughput below minimum"
