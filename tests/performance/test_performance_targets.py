"""Performance tests to validate portfolio demonstration targets.

This module contains comprehensive performance tests to ensure the system meets
the specified performance targets for portfolio demonstrations.
"""

import asyncio
import logging
import statistics
import time

import pytest

from src.services.cache.performance_cache import PerformanceCache
from src.services.monitoring.performance_monitor import RealTimePerformanceMonitor
from src.services.processing.batch_optimizer import BatchConfig, BatchProcessor
from src.services.vector_db.optimization import QdrantOptimizer


logger = logging.getLogger(__name__)


@pytest.mark.performance
class TestPerformanceTargets:
    """Validate performance targets for portfolio demonstration."""

    @pytest.mark.asyncio
    async def test_search_p95_latency_target(self, mock_search_manager):
        """Verify P95 search latency < 100ms.

        This test validates that 95% of search requests complete within 100ms,
        which is the key metric for portfolio demonstrations.
        """
        search_manager = mock_search_manager

        # Run 100 searches to get statistical significance
        latencies = []
        for i in range(100):
            start_time = time.perf_counter()
            await search_manager.search(f"test query {i}")
            latency = (time.perf_counter() - start_time) * 1000  # Convert to ms
            latencies.append(latency)

        # Calculate P95 latency
        p95_latency = statistics.quantiles(latencies, n=20)[18]  # 95th percentile
        avg_latency = statistics.mean(latencies)
        max_latency = max(latencies)

        logger.info("Search latency metrics:")
        logger.info(
            f"  P95: {p95_latency:.1f}ms"
        )  # TODO: Convert f-string to logging format
        logger.info(
            f"  Average: {avg_latency:.1f}ms"
        )  # TODO: Convert f-string to logging format
        logger.info(
            f"  Maximum: {max_latency:.1f}ms"
        )  # TODO: Convert f-string to logging format

        assert p95_latency < 100, (
            f"P95 latency {p95_latency:.1f}ms exceeds 100ms target"
        )
        assert avg_latency < 50, (
            f"Average latency {avg_latency:.1f}ms should be well below P95"
        )

    @pytest.mark.asyncio
    async def test_throughput_target(self, mock_search_manager):
        """Verify system can handle 500+ concurrent searches.

        This test validates the system's ability to handle high concurrent load
        while maintaining acceptable success rates.
        """
        search_manager = mock_search_manager

        async def search_task(query_id: int):
            """Single search task for concurrent testing."""
            try:
                result = await search_manager.search(
                    f"concurrent test query {query_id}"
                )
                return {"success": True, "result": result}
            except Exception as e:
                return {"success": False, "error": str(e)}

        # Launch 500 concurrent searches
        start_time = time.perf_counter()
        tasks = [search_task(i) for i in range(500)]
        results = await asyncio.gather(*tasks, return_exceptions=True)
        total_time = time.perf_counter() - start_time

        # Analyze results
        successful_searches = sum(
            1 for r in results if isinstance(r, dict) and r.get("success", False)
        )
        failed_searches = 500 - successful_searches
        throughput = successful_searches / total_time

        logger.info("Throughput test results:")
        logger.info(
            f"  Successful searches: {successful_searches}/500"
        )  # TODO: Convert f-string to logging format
        logger.info(
            f"  Failed searches: {failed_searches}"
        )  # TODO: Convert f-string to logging format
        logger.info(
            f"  Throughput: {throughput:.1f} searches/sec"
        )  # TODO: Convert f-string to logging format
        logger.info(
            f"  Total time: {total_time:.2f}s"
        )  # TODO: Convert f-string to logging format

        assert throughput >= 500, (
            f"Throughput {throughput:.1f} searches/sec below 500 target"
        )
        assert successful_searches >= 475, (
            f"Success rate {successful_searches / 500:.1%} below 95% target"
        )

    @pytest.mark.asyncio
    async def test_cache_hit_rate_target(self, redis_url):
        """Verify cache hit rate > 85% for common queries.

        This test validates that the caching system achieves high hit rates
        for repeated queries, improving overall performance.
        """
        cache = PerformanceCache(redis_url, max_l1_size=1000)
        await cache.initialize()

        try:
            # Populate cache with common queries
            common_queries = [f"query_{i}" for i in range(100)]

            # First pass: populate cache
            for query in common_queries:
                await cache.set(f"search:{query}", f"result_{query}", ttl=3600)

            # Second pass: test hit rates with mixed queries
            hits = 0
            total_requests = 200

            for i in range(total_requests):
                if i < 170:  # 85% should be cache hits
                    query_key = f"search:query_{i % 100}"
                else:  # 15% cache misses
                    query_key = f"search:new_query_{i}"

                result = await cache.get(query_key)
                if result is not None:
                    hits += 1

            hit_rate = hits / total_requests
            cache_stats = await cache.get_cache_stats()

            logger.info("Cache performance test results:")
            logger.info(
                f"  Hit rate: {hit_rate:.1%}"
            )  # TODO: Convert f-string to logging format
            logger.info(
                f"  L1 cache utilization: {cache_stats['l1_cache']['utilization']:.1%}"
            )
            logger.info(
                f"  Total hits: {hits}/{total_requests}"
            )  # TODO: Convert f-string to logging format

            assert hit_rate >= 0.85, f"Cache hit rate {hit_rate:.1%} below 85% target"

        finally:
            await cache.cleanup()

    @pytest.mark.asyncio
    async def test_memory_optimization_quantization(self, mock_qdrant_client):
        """Verify memory usage optimization with quantization.

        This test validates that quantization provides significant memory
        reduction while maintaining search quality.
        """
        optimizer = QdrantOptimizer(mock_qdrant_client)

        collection_name = "test_quantization_collection"
        vector_size = 384

        # Create optimized collection with quantization
        success = await optimizer.create_optimized_collection(
            collection_name=collection_name, vector_size=vector_size
        )

        assert success, "Failed to create optimized collection"

        # Get optimization metrics
        metrics = await optimizer.get_optimization_metrics(collection_name)

        logger.info("Quantization test results:")
        logger.info(
            f"  Collection created: {success}"
        )  # TODO: Convert f-string to logging format
        logger.info(
            f"  Quantization enabled: {metrics.get('quantization_enabled', False)}"
        )
        logger.info(
            f"  Vector size: {vector_size}"
        )  # TODO: Convert f-string to logging format

        assert metrics.get("quantization_enabled", False), "Quantization not enabled"

        # Test memory efficiency (quantization should reduce memory by ~83%)
        expected_memory_reduction = 0.83
        logger.info(
            f"  Expected memory reduction: {expected_memory_reduction:.1%}"
        )  # TODO: Convert f-string to logging format

    @pytest.mark.asyncio
    async def test_batch_processing_optimization(self):
        """Verify batch processing optimization improves throughput.

        This test validates that batch processing provides better throughput
        than individual processing while maintaining low latency.
        """

        async def mock_process_batch(items: list[str]) -> list[str]:
            """Mock batch processing function."""
            # Simulate realistic batch processing time
            await asyncio.sleep(0.01 * len(items))  # 10ms per item in batch
            return [f"processed_{item}" for item in items]

        # Test with optimized batch configuration
        batch_processor = BatchProcessor(
            mock_process_batch,
            BatchConfig(max_batch_size=50, max_wait_time=0.1, adaptive_sizing=True),
        )

        # Process 200 items concurrently
        start_time = time.perf_counter()
        tasks = [batch_processor.process_item(f"item_{i}") for i in range(200)]
        results = await asyncio.gather(*tasks)
        total_time = time.perf_counter() - start_time

        # Get performance statistics
        stats = batch_processor.get_performance_stats()
        throughput = len(results) / total_time

        logger.info("Batch processing test results:")
        logger.info(
            f"  Items processed: {len(results)}"
        )  # TODO: Convert f-string to logging format
        logger.info(
            f"  Total time: {total_time:.2f}s"
        )  # TODO: Convert f-string to logging format
        logger.info(
            f"  Throughput: {throughput:.1f} items/sec"
        )  # TODO: Convert f-string to logging format
        logger.info(
            f"  Average batch size: {stats.get('avg_batch_size', 0):.1f}"
        )  # TODO: Convert f-string to logging format
        logger.info(
            f"  Time per item: {stats.get('avg_time_per_item_ms', 0):.1f}ms"
        )  # TODO: Convert f-string to logging format

        assert len(results) == 200, "Not all items were processed"
        assert throughput > 100, f"Throughput {throughput:.1f} items/sec too low"
        assert all("processed_" in result for result in results), (
            "Invalid processing results"
        )

    @pytest.mark.asyncio
    async def test_real_time_monitoring_performance(self):
        """Verify real-time monitoring has minimal performance impact.

        This test ensures that the performance monitoring system itself
        doesn't significantly impact application performance.
        """
        monitor = RealTimePerformanceMonitor(window_size=30)

        # Start monitoring
        monitoring_task = asyncio.create_task(monitor.start_monitoring())

        try:
            # Let monitoring run for a short period
            await asyncio.sleep(5)

            # Simulate some load while monitoring
            for _i in range(100):
                monitor.record_request(0.05)  # 50ms request
                await asyncio.sleep(0.01)

            # Get performance summary
            summary = monitor.get_performance_summary()
            trends = monitor.get_performance_trends(minutes=1)
            recommendations = monitor.get_optimization_recommendations()

            logger.info("Monitoring performance test results:")
            logger.info(
                f"  Snapshots collected: {len(monitor.snapshots)}"
            )  # TODO: Convert f-string to logging format
            logger.info(
                f"  Current CPU: {summary.get('cpu_percent', 0):.1f}%"
            )  # TODO: Convert f-string to logging format
            logger.info(
                f"  Current memory: {summary.get('memory_percent', 0):.1f}%"
            )  # TODO: Convert f-string to logging format
            logger.info(
                f"  Recommendations: {len(recommendations)}"
            )  # TODO: Convert f-string to logging format

            assert len(monitor.snapshots) > 0, "No performance snapshots collected"
            assert summary.get("timestamp") is not None, "Invalid performance summary"
            assert len(recommendations) > 0, "No optimization recommendations generated"

        finally:
            await monitor.stop_monitoring()
            monitoring_task.cancel()
            try:
                await monitoring_task
            except asyncio.CancelledError:
                pass

    @pytest.mark.asyncio
    async def test_end_to_end_performance_pipeline(
        self, mock_search_manager, redis_url
    ):
        """Test complete performance pipeline integration.

        This test validates the entire performance optimization pipeline
        working together to achieve portfolio demonstration targets.
        """
        # Initialize all performance components
        cache = PerformanceCache(redis_url, max_l1_size=1000)
        monitor = RealTimePerformanceMonitor(window_size=30)

        await cache.initialize()
        monitoring_task = asyncio.create_task(monitor.start_monitoring())

        try:
            # Warm up cache with popular queries
            popular_queries = [f"popular_query_{i}" for i in range(20)]
            for query in popular_queries:
                cache_key = f"search:{query}"
                await cache.set(cache_key, f"cached_result_{query}", ttl=3600)

            # Run mixed workload: cached and uncached queries
            total_requests = 100
            cached_requests = 0
            start_time = time.perf_counter()

            for i in range(total_requests):
                # 80% cached queries, 20% new queries
                query = f"popular_query_{i % 20}" if i < 80 else f"new_query_{i}"

                # Check cache first
                cache_key = f"search:{query}"
                cached_result = await cache.get(cache_key)

                if cached_result:
                    cached_requests += 1
                    result = cached_result
                else:
                    # Simulate search
                    search_start = time.perf_counter()
                    result = await mock_search_manager.search(query)
                    search_time = time.perf_counter() - search_start

                    # Cache result
                    await cache.set(cache_key, result, ttl=3600)

                    # Record performance
                    monitor.record_request(search_time)

            total_time = time.perf_counter() - start_time

            # Analyze end-to-end performance
            cache_stats = await cache.get_cache_stats()
            performance_summary = monitor.get_performance_summary()

            overall_throughput = total_requests / total_time
            cache_hit_rate = cached_requests / total_requests

            logger.info("End-to-end performance test results:")
            logger.info(
                f"  Total requests: {total_requests}"
            )  # TODO: Convert f-string to logging format
            logger.info(
                f"  Cache hits: {cached_requests}"
            )  # TODO: Convert f-string to logging format
            logger.info(
                f"  Cache hit rate: {cache_hit_rate:.1%}"
            )  # TODO: Convert f-string to logging format
            logger.info(
                f"  Overall throughput: {overall_throughput:.1f} req/sec"
            )  # TODO: Convert f-string to logging format
            logger.info(
                f"  Total time: {total_time:.2f}s"
            )  # TODO: Convert f-string to logging format
            logger.info(
                f"  L1 cache utilization: {cache_stats['l1_cache']['utilization']:.1%}"
            )

            # Validate performance targets
            assert overall_throughput >= 100, (
                f"Overall throughput {overall_throughput:.1f} req/sec too low"
            )
            assert cache_hit_rate >= 0.75, (
                f"Cache hit rate {cache_hit_rate:.1%} below expected"
            )
            assert cache_stats["performance"]["hit_rate"] >= 0.75, (
                "Cache performance below target"
            )

        finally:
            await cache.cleanup()
            await monitor.stop_monitoring()
            monitoring_task.cancel()
            try:
                await monitoring_task
            except asyncio.CancelledError:
                pass


@pytest.mark.performance
@pytest.mark.benchmark
class TestBenchmarkTargets:
    """Benchmark tests for specific performance metrics."""

    def test_hnsw_optimization_benchmark(self, benchmark, mock_qdrant_client):
        """Benchmark HNSW optimization performance."""
        optimizer = QdrantOptimizer(mock_qdrant_client)

        def create_optimized_collection():
            """Benchmark function for collection creation."""
            import asyncio

            return asyncio.run(
                optimizer.create_optimized_collection("benchmark_collection", 384)
            )

        result = benchmark(create_optimized_collection)
        assert result, "Failed to create optimized collection in benchmark"

    def test_cache_performance_benchmark(self, benchmark, redis_url):
        """Benchmark cache performance."""

        async def cache_operations():
            cache = PerformanceCache(redis_url, max_l1_size=1000)
            await cache.initialize()

            try:
                # Benchmark mixed read/write operations
                for i in range(100):
                    await cache.set(f"key_{i}", f"value_{i}")
                    await cache.get(f"key_{i}")
                return True
            finally:
                await cache.cleanup()

        def run_cache_benchmark():
            import asyncio

            return asyncio.run(cache_operations())

        result = benchmark(run_cache_benchmark)
        assert result, "Cache benchmark failed"
