"""Performance Tests for Search Operations with P95 Latency Validation.

This module implements comprehensive performance testing for search operations,
including P95 latency validation, throughput testing, and load characteristics
following 2025 performance testing best practices.
"""

import asyncio
import time
from typing import Any, Dict, List
from unittest.mock import AsyncMock, MagicMock

import pytest

from tests.utils.modern_ai_testing import (
    ModernAITestingUtils,
    PerformanceTestingFramework,
    performance_critical_test,
)
from tests.utils.performance_utils import PerformanceTracker


@pytest.mark.performance
class TestSearchPerformance:
    """Performance tests for search operations with comprehensive metrics."""

    @pytest.fixture
    def performance_framework(self):
        """Provide performance testing framework."""
        return PerformanceTestingFramework()

    @pytest.fixture
    def mock_search_service(self):
        """Mock search service with configurable latency."""
        service = MagicMock()

        async def mock_search(query: str, latency_ms: float = 50.0):
            """Mock search with configurable latency."""
            # Simulate processing time
            await asyncio.sleep(latency_ms / 1000.0)

            # Return mock results
            return {
                "query": query,
                "results": [
                    {
                        "id": f"doc_{i}",
                        "score": 0.9 - (i * 0.1),
                        "title": f"Document {i}",
                        "content": f"Content for {query} - document {i}",
                    }
                    for i in range(5)
                ],
                "total_time_ms": latency_ms,
                "total_results": 5,
            }

        service.search = mock_search
        return service

    @pytest.fixture
    def test_queries(self):
        """Provide realistic test queries for performance testing."""
        return [
            "machine learning algorithms",
            "neural network architecture",
            "natural language processing",
            "computer vision techniques",
            "deep learning frameworks",
            "artificial intelligence applications",
            "data science methods",
            "statistical modeling",
            "reinforcement learning",
            "transformer models",
        ]

    @performance_critical_test(p95_threshold_ms=100.0)
    async def test_search_latency_p95_validation(
        self, performance_framework, mock_search_service, test_queries
    ):
        """Verify P95 search latency meets performance requirements."""

        # Configure mock service for realistic latency
        async def search_func(query: str):
            # Simulate variable latency (most requests fast, some slower)
            import random

            if random.random() < 0.95:  # 95% of requests
                latency = random.uniform(20, 80)  # Fast requests
            else:  # 5% of requests
                latency = random.uniform(80, 150)  # Slower requests

            return await mock_search_service.search(query, latency_ms=latency)

        # Run latency test with 100 concurrent requests
        metrics = await performance_framework.run_latency_test(
            search_func=search_func, queries=test_queries, concurrent_requests=100
        )

        # Validate performance requirements
        performance_framework.assert_performance_requirements(
            metrics=metrics, p95_threshold_ms=100.0, success_rate_threshold=0.95
        )

        # Additional performance assertions
        assert metrics["mean_latency_ms"] < 80.0, (
            f"Mean latency {metrics['mean_latency_ms']:.1f}ms too high"
        )
        assert metrics["success_rate"] >= 0.98, (
            f"Success rate {metrics['success_rate']:.3f} below 98%"
        )

        # Log performance metrics for monitoring
        print("\nPerformance Metrics:")
        print(f"  P95 Latency: {metrics['p95_latency_ms']:.1f}ms")
        print(f"  P99 Latency: {metrics['p99_latency_ms']:.1f}ms")
        print(f"  Mean Latency: {metrics['mean_latency_ms']:.1f}ms")
        print(f"  Success Rate: {metrics['success_rate']:.3f}")

    @performance_critical_test(p95_threshold_ms=200.0)
    async def test_concurrent_search_throughput(
        self, performance_framework, mock_search_service, test_queries
    ):
        """Test system throughput under concurrent load."""

        async def search_func(query: str):
            # Simulate realistic processing time
            return await mock_search_service.search(query, latency_ms=50.0)

        # Test different concurrency levels
        concurrency_levels = [10, 25, 50, 100]
        throughput_results = []

        for concurrent_requests in concurrency_levels:
            start_time = time.perf_counter()

            # Create concurrent search tasks
            tasks = []
            for i in range(concurrent_requests):
                query = test_queries[i % len(test_queries)]
                task = search_func(query)
                tasks.append(task)

            # Execute all searches concurrently
            results = await asyncio.gather(*tasks, return_exceptions=True)

            end_time = time.perf_counter()
            total_time = end_time - start_time

            # Analyze results
            successful_requests = sum(
                1 for r in results if not isinstance(r, Exception)
            )
            throughput = successful_requests / total_time

            throughput_results.append(
                {
                    "concurrency": concurrent_requests,
                    "throughput_rps": throughput,
                    "success_rate": successful_requests / concurrent_requests,
                    "total_time": total_time,
                }
            )

            # Assert minimum throughput requirements
            assert successful_requests >= concurrent_requests * 0.95, (
                f"Success rate below 95% at concurrency {concurrent_requests}"
            )

            if concurrent_requests <= 50:
                # For lower concurrency, expect higher throughput per request
                assert throughput >= 15.0, (
                    f"Throughput {throughput:.1f} RPS too low for concurrency {concurrent_requests}"
                )

        # Verify throughput scaling characteristics
        max_throughput = max(result["throughput_rps"] for result in throughput_results)
        assert max_throughput >= 20.0, (
            f"Peak throughput {max_throughput:.1f} RPS insufficient"
        )

        # Log throughput analysis
        print("\nThroughput Analysis:")
        for result in throughput_results:
            print(
                f"  Concurrency {result['concurrency']:3d}: "
                f"{result['throughput_rps']:6.1f} RPS, "
                f"Success: {result['success_rate']:.3f}"
            )

    @pytest.mark.performance
    async def test_memory_usage_under_load(self, mock_search_service, test_queries):
        """Test memory usage characteristics under sustained load."""
        tracker = PerformanceTracker()

        async def search_func(query: str):
            return await mock_search_service.search(query, latency_ms=30.0)

        # Measure memory usage during sustained load
        with tracker.measure("sustained_search_load"):
            # Run searches for sustained period
            for _round_num in range(10):  # 10 rounds
                tasks = []
                for i in range(20):  # 20 concurrent searches per round
                    query = test_queries[i % len(test_queries)]
                    task = search_func(query)
                    tasks.append(task)

                await asyncio.gather(*tasks)

                # Small delay between rounds
                await asyncio.sleep(0.1)

        # Analyze memory usage
        stats = tracker.get_statistics("sustained_search_load")

        # Memory usage should be reasonable
        assert stats["memory_peak_mb"]["max"] < 100.0, (
            f"Peak memory {stats['memory_peak_mb']['max']:.1f}MB too high"
        )

        # Execution time should be reasonable for sustained load
        assert stats["execution_time"]["mean"] < 10.0, (
            f"Mean execution time {stats['execution_time']['mean']:.2f}s too high"
        )

        print("\nMemory Usage Analysis:")
        print(f"  Peak Memory: {stats['memory_peak_mb']['max']:.1f}MB")
        print(f"  Mean Memory: {stats['memory_peak_mb']['mean']:.1f}MB")
        print(f"  Execution Time: {stats['execution_time']['mean']:.2f}s")

    @pytest.mark.performance
    async def test_search_scalability_characteristics(
        self, mock_search_service, test_queries
    ):
        """Test search performance scalability with increasing load."""

        # Test scalability with different dataset sizes
        dataset_sizes = [100, 500, 1000, 2000]
        scalability_results = []

        for dataset_size in dataset_sizes:
            # Simulate search in dataset of given size
            async def search_func(query: str):
                # Latency increases logarithmically with dataset size (realistic)
                import math

                base_latency = 30.0
                scale_factor = math.log10(dataset_size / 100.0 + 1) * 20.0
                latency = base_latency + scale_factor

                return await mock_search_service.search(query, latency_ms=latency)

            # Measure performance for this dataset size
            start_time = time.perf_counter()

            # Run moderate concurrent load
            tasks = []
            for i in range(50):  # Fixed concurrent load
                query = test_queries[i % len(test_queries)]
                task = search_func(query)
                tasks.append(task)

            results = await asyncio.gather(*tasks)
            end_time = time.perf_counter()

            avg_latency = (end_time - start_time) / len(tasks) * 1000  # Convert to ms

            scalability_results.append(
                {
                    "dataset_size": dataset_size,
                    "avg_latency_ms": avg_latency,
                    "requests_processed": len(results),
                }
            )

        # Analyze scalability characteristics
        # Latency should increase sub-linearly with dataset size
        base_latency = scalability_results[0]["avg_latency_ms"]
        max_latency = scalability_results[-1]["avg_latency_ms"]

        # For 20x dataset increase, latency should not increase more than 3x
        latency_growth_factor = max_latency / base_latency
        assert latency_growth_factor <= 3.0, (
            f"Latency growth factor {latency_growth_factor:.2f} indicates poor scalability"
        )

        print("\nScalability Analysis:")
        for result in scalability_results:
            print(
                f"  Dataset {result['dataset_size']:4d}: "
                f"{result['avg_latency_ms']:6.1f}ms avg latency"
            )
        print(f"  Growth Factor: {latency_growth_factor:.2f}x")

    @pytest.mark.performance
    async def test_search_cache_performance_impact(
        self, mock_search_service, test_queries
    ):
        """Test performance impact of search result caching."""

        # Simulate cache behavior
        cache = {}
        cache_hits = 0
        cache_misses = 0

        async def cached_search_func(query: str):
            nonlocal cache_hits, cache_misses

            if query in cache:
                cache_hits += 1
                # Cache hit - very fast response
                await asyncio.sleep(0.001)  # 1ms for cache hit
                return cache[query]
            else:
                cache_misses += 1
                # Cache miss - normal search latency
                result = await mock_search_service.search(query, latency_ms=60.0)
                cache[query] = result
                return result

        # Test with repeated queries (should benefit from caching)
        repeated_queries = test_queries * 3  # Each query appears 3 times

        start_time = time.perf_counter()

        tasks = []
        for query in repeated_queries:
            task = cached_search_func(query)
            tasks.append(task)

        results = await asyncio.gather(*tasks)
        end_time = time.perf_counter()

        total_time = end_time - start_time
        avg_latency_ms = (total_time / len(results)) * 1000

        # Verify cache effectiveness
        total_requests = len(repeated_queries)
        expected_cache_hits = total_requests - len(
            test_queries
        )  # First occurrence of each query is a miss

        assert cache_hits >= expected_cache_hits * 0.8, (
            f"Cache hit rate too low: {cache_hits}/{total_requests}"
        )

        # With effective caching, average latency should be much lower
        assert avg_latency_ms <= 25.0, (
            f"Average latency {avg_latency_ms:.1f}ms too high with caching"
        )

        cache_hit_rate = cache_hits / total_requests

        print("\nCache Performance Analysis:")
        print(f"  Cache Hits: {cache_hits}")
        print(f"  Cache Misses: {cache_misses}")
        print(f"  Hit Rate: {cache_hit_rate:.3f}")
        print(f"  Avg Latency: {avg_latency_ms:.1f}ms")
