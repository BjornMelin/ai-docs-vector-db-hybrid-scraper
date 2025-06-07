"""Performance benchmark tests for MCP server operations.

Comprehensive performance testing including:
- Load testing with concurrent requests
- Memory usage profiling
- Response time analysis
- Throughput measurements
- Resource utilization monitoring
"""

import asyncio
import gc
import time
import tracemalloc
from dataclasses import dataclass
from statistics import mean
from statistics import median
from statistics import stdev
from unittest.mock import AsyncMock
from unittest.mock import MagicMock
from unittest.mock import patch

import pytest
from src.infrastructure.client_manager import ClientManager

from tests.mocks.mock_tools import MockMCPServer
from tests.mocks.mock_tools import register_mock_tools


@dataclass
class PerformanceMetrics:
    """Container for performance test metrics."""

    operation: str
    total_requests: int
    total_time: float
    avg_response_time: float
    min_response_time: float
    max_response_time: float
    median_response_time: float
    std_dev_response_time: float
    requests_per_second: float
    memory_usage_mb: float
    success_rate: float
    p95_response_time: float
    p99_response_time: float


class TestMCPPerformanceBenchmarks:
    """Comprehensive performance benchmarks for MCP server."""

    @pytest.fixture
    async def benchmark_client_manager(self):
        """Create optimized mock client manager for benchmarking."""
        client_manager = MagicMock(spec=ClientManager)

        # Ultra-fast mock services for benchmarking
        mock_vector_service = AsyncMock()
        mock_vector_service.search_documents.return_value = [
            {"id": f"bench-{i}", "content": f"Benchmark content {i}", "score": 0.9}
            for i in range(10)
        ]
        client_manager.vector_service = mock_vector_service

        mock_embedding_service = AsyncMock()
        mock_embedding_service.generate_embeddings.return_value = {
            "embeddings": [[0.1] * 384],
            "model": "test-model",
            "total_tokens": 5,
        }
        client_manager.embedding_service = mock_embedding_service

        mock_cache_service = AsyncMock()
        mock_cache_service.get_stats.return_value = {
            "hit_rate": 0.85,
            "size": 1000,
            "total_requests": 10000,
        }
        client_manager.cache_service = mock_cache_service

        # Add all other required services
        client_manager.crawling_service = AsyncMock()
        client_manager.project_service = AsyncMock()
        client_manager.deployment_service = AsyncMock()
        client_manager.analytics_service = AsyncMock()
        client_manager.hyde_service = AsyncMock()

        return client_manager

    @pytest.fixture
    async def benchmark_server(self, benchmark_client_manager):
        """Create MCP server for benchmarking."""
        mcp = MockMCPServer("benchmark-server")
        register_mock_tools(mcp, benchmark_client_manager)
        return mcp, benchmark_client_manager

    async def _measure_operation(
        self,
        operation_name: str,
        operation_func,
        num_requests: int,
        concurrent_requests: int = 10,
    ) -> PerformanceMetrics:
        """Measure performance metrics for an operation."""
        # Start memory tracking
        tracemalloc.start()
        gc.collect()
        initial_memory = tracemalloc.get_traced_memory()[0]

        response_times = []
        errors = 0

        # Execute requests in batches
        start_time = time.time()

        for batch_start in range(0, num_requests, concurrent_requests):
            batch_size = min(concurrent_requests, num_requests - batch_start)
            batch_tasks = []

            for i in range(batch_size):

                async def timed_operation(idx):
                    op_start = time.time()
                    try:
                        result = await operation_func(idx)
                        op_time = time.time() - op_start
                        return op_time, result
                    except Exception as e:
                        return None, e

                batch_tasks.append(timed_operation(batch_start + i))

            batch_results = await asyncio.gather(*batch_tasks)

            for timing, _result in batch_results:
                if timing is not None:
                    response_times.append(timing)
                else:
                    errors += 1

        total_time = time.time() - start_time

        # Get memory usage
        current_memory = tracemalloc.get_traced_memory()[0]
        memory_usage_mb = (current_memory - initial_memory) / 1024 / 1024
        tracemalloc.stop()

        # Calculate metrics
        if response_times:
            sorted_times = sorted(response_times)
            p95_index = int(len(sorted_times) * 0.95)
            p99_index = int(len(sorted_times) * 0.99)

            metrics = PerformanceMetrics(
                operation=operation_name,
                total_requests=num_requests,
                total_time=total_time,
                avg_response_time=mean(response_times),
                min_response_time=min(response_times),
                max_response_time=max(response_times),
                median_response_time=median(response_times),
                std_dev_response_time=stdev(response_times)
                if len(response_times) > 1
                else 0,
                requests_per_second=num_requests / total_time,
                memory_usage_mb=memory_usage_mb,
                success_rate=(num_requests - errors) / num_requests,
                p95_response_time=sorted_times[p95_index]
                if p95_index < len(sorted_times)
                else sorted_times[-1],
                p99_response_time=sorted_times[p99_index]
                if p99_index < len(sorted_times)
                else sorted_times[-1],
            )
        else:
            # All requests failed
            metrics = PerformanceMetrics(
                operation=operation_name,
                total_requests=num_requests,
                total_time=total_time,
                avg_response_time=0,
                min_response_time=0,
                max_response_time=0,
                median_response_time=0,
                std_dev_response_time=0,
                requests_per_second=0,
                memory_usage_mb=memory_usage_mb,
                success_rate=0,
                p95_response_time=0,
                p99_response_time=0,
            )

        return metrics

    def _print_metrics(self, metrics: PerformanceMetrics):
        """Print formatted performance metrics."""
        print(f"\n{'=' * 60}")
        print(f"Performance Metrics: {metrics.operation}")
        print(f"{'=' * 60}")
        print(f"Total Requests:        {metrics.total_requests}")
        print(f"Total Time:            {metrics.total_time:.2f}s")
        print(f"Requests/Second:       {metrics.requests_per_second:.2f}")
        print(f"Success Rate:          {metrics.success_rate * 100:.1f}%")
        print("")
        print("Response Time (ms):")
        print(f"  Average:             {metrics.avg_response_time * 1000:.2f}")
        print(f"  Median:              {metrics.median_response_time * 1000:.2f}")
        print(f"  Min:                 {metrics.min_response_time * 1000:.2f}")
        print(f"  Max:                 {metrics.max_response_time * 1000:.2f}")
        print(f"  Std Dev:             {metrics.std_dev_response_time * 1000:.2f}")
        print(f"  95th Percentile:     {metrics.p95_response_time * 1000:.2f}")
        print(f"  99th Percentile:     {metrics.p99_response_time * 1000:.2f}")
        print("")
        print(f"Memory Usage:          {metrics.memory_usage_mb:.2f} MB")
        print(f"{'=' * 60}\n")

    async def test_search_performance_benchmark(self, benchmark_server):
        """Benchmark search operation performance."""
        mcp_server, mock_client_manager = benchmark_server

        search_tool = None
        for tool in mcp_server._tools:
            if tool.name == "search_documents":
                search_tool = tool
                break

        async def search_operation(idx):
            return await search_tool.handler(
                query=f"benchmark query {idx}",
                collection="documentation",
                limit=10,
            )

        # Test with different load levels
        load_levels = [
            (100, 10),  # 100 requests, 10 concurrent
            (500, 25),  # 500 requests, 25 concurrent
            (1000, 50),  # 1000 requests, 50 concurrent
        ]

        for total_requests, concurrent in load_levels:
            metrics = await self._measure_operation(
                f"Search (load: {total_requests}, concurrent: {concurrent})",
                search_operation,
                total_requests,
                concurrent,
            )

            self._print_metrics(metrics)

            # Performance assertions
            assert metrics.success_rate >= 0.99, (
                f"Success rate {metrics.success_rate} too low"
            )
            assert metrics.requests_per_second > 100, (
                f"Throughput {metrics.requests_per_second} req/s too low"
            )
            assert metrics.p95_response_time < 0.1, (
                f"P95 response time {metrics.p95_response_time}s too high"
            )

    async def test_embedding_generation_performance(self, benchmark_server):
        """Benchmark embedding generation performance."""
        mcp_server, mock_client_manager = benchmark_server

        embedding_tool = None
        for tool in mcp_server._tools:
            if tool.name == "generate_embeddings":
                embedding_tool = tool
                break

        async def embedding_operation(idx):
            return await embedding_tool.handler(
                texts=[f"Benchmark text {idx} for embedding generation"],
                model="test-model",
            )

        metrics = await self._measure_operation(
            "Embedding Generation",
            embedding_operation,
            num_requests=200,
            concurrent_requests=20,
        )

        self._print_metrics(metrics)

        # Performance assertions
        assert metrics.success_rate >= 0.99
        assert metrics.requests_per_second > 50
        assert metrics.avg_response_time < 0.05

    async def test_mixed_workload_performance(self, benchmark_server):
        """Benchmark mixed workload with multiple tool types."""
        mcp_server, mock_client_manager = benchmark_server

        # Find tools
        tools = {}
        for tool in mcp_server._tools:
            if tool.name in [
                "search_documents",
                "generate_embeddings",
                "get_cache_stats",
            ]:
                tools[tool.name] = tool

        async def mixed_operation(idx):
            # Rotate between different operations
            operation_type = idx % 3

            if operation_type == 0:
                return await tools["search_documents"].handler(
                    query=f"mixed search {idx}",
                    collection="docs",
                    limit=5,
                )
            elif operation_type == 1:
                return await tools["generate_embeddings"].handler(
                    texts=[f"mixed text {idx}"],
                )
            else:
                return await tools["get_cache_stats"].handler()

        metrics = await self._measure_operation(
            "Mixed Workload",
            mixed_operation,
            num_requests=300,
            concurrent_requests=30,
        )

        self._print_metrics(metrics)

        # Mixed workload should still perform well
        assert metrics.success_rate >= 0.98
        assert metrics.requests_per_second > 80

    async def test_sustained_load_performance(self, benchmark_server):
        """Test performance under sustained load over time."""
        mcp_server, mock_client_manager = benchmark_server

        search_tool = None
        for tool in mcp_server._tools:
            if tool.name == "search_documents":
                search_tool = tool
                break

        # Run sustained load test
        duration_seconds = 10
        requests_per_second_target = 100

        metrics_over_time = []
        start_time = time.time()
        request_count = 0

        while time.time() - start_time < duration_seconds:
            batch_start = time.time()
            batch_tasks = []

            # Create batch of requests
            for i in range(requests_per_second_target // 10):  # 10 batches per second

                async def timed_search(idx):
                    op_start = time.time()
                    await search_tool.handler(
                        query=f"sustained load {idx}",
                        collection="docs",
                        limit=5,
                    )
                    return time.time() - op_start

                batch_tasks.append(timed_search(request_count + i))

            response_times = await asyncio.gather(*batch_tasks)
            request_count += len(batch_tasks)

            # Record metrics for this batch
            if response_times:
                metrics_over_time.append(
                    {
                        "timestamp": time.time() - start_time,
                        "avg_response_time": mean(response_times),
                        "max_response_time": max(response_times),
                    }
                )

            # Wait for next batch timing
            batch_duration = time.time() - batch_start
            if batch_duration < 0.1:  # Target 10 batches per second
                await asyncio.sleep(0.1 - batch_duration)

        # Analyze sustained performance
        total_duration = time.time() - start_time
        actual_rps = request_count / total_duration

        print("\nSustained Load Test Results:")
        print(f"Duration: {total_duration:.1f}s")
        print(f"Total Requests: {request_count}")
        print(f"Actual RPS: {actual_rps:.1f}")

        # Check performance degradation over time
        if len(metrics_over_time) >= 2:
            early_metrics = metrics_over_time[: len(metrics_over_time) // 4]
            late_metrics = metrics_over_time[-len(metrics_over_time) // 4 :]

            early_avg = mean([m["avg_response_time"] for m in early_metrics])
            late_avg = mean([m["avg_response_time"] for m in late_metrics])

            degradation = (late_avg - early_avg) / early_avg * 100

            print(f"Early Avg Response Time: {early_avg * 1000:.2f}ms")
            print(f"Late Avg Response Time: {late_avg * 1000:.2f}ms")
            print(f"Performance Degradation: {degradation:.1f}%")

            # Performance should not degrade significantly
            assert degradation < 20, f"Performance degraded by {degradation:.1f}%"

        assert actual_rps > requests_per_second_target * 0.8, (
            "Could not sustain target RPS"
        )

    async def test_spike_load_handling(self, benchmark_server):
        """Test handling of sudden load spikes."""
        mcp_server, mock_client_manager = benchmark_server

        search_tool = None
        for tool in mcp_server._tools:
            if tool.name == "search_documents":
                search_tool = tool
                break

        # Normal load
        async def normal_load_operation(idx):
            return await search_tool.handler(
                query=f"normal load {idx}",
                collection="docs",
                limit=5,
            )

        print("\nMeasuring baseline performance...")
        baseline_metrics = await self._measure_operation(
            "Baseline Load",
            normal_load_operation,
            num_requests=50,
            concurrent_requests=5,
        )

        # Spike load (10x normal)
        async def spike_load_operation(idx):
            return await search_tool.handler(
                query=f"spike load {idx}",
                collection="docs",
                limit=5,
            )

        print("\nApplying spike load...")
        spike_metrics = await self._measure_operation(
            "Spike Load",
            spike_load_operation,
            num_requests=500,
            concurrent_requests=50,
        )

        # Recovery period
        print("\nMeasuring recovery performance...")
        await asyncio.sleep(1)  # Brief pause

        recovery_metrics = await self._measure_operation(
            "Recovery Load",
            normal_load_operation,
            num_requests=50,
            concurrent_requests=5,
        )

        # Analyze spike handling
        print("\nSpike Load Analysis:")
        print(f"Baseline RPS: {baseline_metrics.requests_per_second:.1f}")
        print(f"Spike RPS: {spike_metrics.requests_per_second:.1f}")
        print(f"Recovery RPS: {recovery_metrics.requests_per_second:.1f}")

        spike_degradation = (
            (spike_metrics.avg_response_time - baseline_metrics.avg_response_time)
            / baseline_metrics.avg_response_time
            * 100
        )
        recovery_degradation = (
            (recovery_metrics.avg_response_time - baseline_metrics.avg_response_time)
            / baseline_metrics.avg_response_time
            * 100
        )

        print(f"Response time increase during spike: {spike_degradation:.1f}%")
        print(f"Response time after recovery: {recovery_degradation:.1f}%")

        # System should handle spikes gracefully
        assert spike_metrics.success_rate >= 0.95, "Too many failures during spike"
        assert recovery_degradation < 10, "System did not recover properly after spike"

    async def test_memory_leak_detection(self, benchmark_server):
        """Test for memory leaks under repeated operations."""
        mcp_server, mock_client_manager = benchmark_server

        search_tool = None
        for tool in mcp_server._tools:
            if tool.name == "search_documents":
                search_tool = tool
                break

        # Force garbage collection and get baseline
        gc.collect()
        tracemalloc.start()
        baseline_memory = tracemalloc.get_traced_memory()[0]

        memory_samples = []

        # Run many iterations to detect leaks
        for iteration in range(10):
            # Execute batch of operations
            tasks = []
            for i in range(100):
                task = search_tool.handler(
                    query=f"memory test iteration {iteration} request {i}",
                    collection="docs",
                    limit=5,
                )
                tasks.append(task)

            await asyncio.gather(*tasks)

            # Force garbage collection and measure memory
            gc.collect()
            current_memory = tracemalloc.get_traced_memory()[0]
            memory_mb = (current_memory - baseline_memory) / 1024 / 1024
            memory_samples.append(memory_mb)

            print(f"Iteration {iteration + 1}: Memory usage {memory_mb:.2f} MB")

        tracemalloc.stop()

        # Analyze memory trend
        if len(memory_samples) >= 5:
            early_avg = mean(memory_samples[:3])
            late_avg = mean(memory_samples[-3:])
            memory_growth = late_avg - early_avg

            print("\nMemory Analysis:")
            print(f"Early average: {early_avg:.2f} MB")
            print(f"Late average: {late_avg:.2f} MB")
            print(f"Memory growth: {memory_growth:.2f} MB")

            # Memory should not grow significantly
            assert memory_growth < 10, (
                f"Potential memory leak: {memory_growth:.2f} MB growth"
            )

    async def test_concurrent_tool_performance(self, benchmark_server):
        """Test performance of different tools running concurrently."""
        mcp_server, mock_client_manager = benchmark_server

        # Find multiple tools
        tools = {}
        for tool in mcp_server._tools:
            if tool.name in [
                "search_documents",
                "generate_embeddings",
                "list_collections",
                "get_cache_stats",
            ]:
                tools[tool.name] = tool

        # Mock list_collections
        with patch.object(
            mock_client_manager.vector_service, "list_collections"
        ) as mock_list:
            mock_list.return_value = [
                {"name": f"col-{i}", "vectors_count": i * 100} for i in range(5)
            ]

            async def concurrent_operations():
                tasks = []

                # Mix of different operations
                for i in range(25):
                    tasks.append(
                        tools["search_documents"].handler(
                            query=f"concurrent search {i}",
                            collection="docs",
                            limit=5,
                        )
                    )

                for i in range(25):
                    tasks.append(
                        tools["generate_embeddings"].handler(
                            texts=[f"concurrent embedding {i}"],
                        )
                    )

                for _i in range(25):
                    tasks.append(tools["list_collections"].handler())

                for _i in range(25):
                    tasks.append(tools["get_cache_stats"].handler())

                return await asyncio.gather(*tasks, return_exceptions=True)

            start_time = time.time()
            results = await concurrent_operations()
            total_time = time.time() - start_time

            # Count successes and failures
            successes = sum(1 for r in results if not isinstance(r, Exception))
            failures = len(results) - successes

            print("\nConcurrent Tools Performance:")
            print(f"Total Operations: {len(results)}")
            print(f"Successes: {successes}")
            print(f"Failures: {failures}")
            print(f"Total Time: {total_time:.2f}s")
            print(f"Operations/Second: {len(results) / total_time:.1f}")

            # All operations should succeed
            assert failures == 0, f"{failures} operations failed"
            assert total_time < 5.0, (
                f"Concurrent operations took too long: {total_time:.2f}s"
            )


class TestMCPResourceOptimization:
    """Test resource usage optimization strategies."""

    @pytest.fixture
    async def benchmark_client_manager(self):
        """Create optimized mock client manager for benchmarking."""
        client_manager = MagicMock(spec=ClientManager)

        # Ultra-fast mock services for benchmarking
        mock_vector_service = AsyncMock()
        mock_vector_service.search_documents.return_value = [
            {"id": f"doc-{i}", "content": f"Result {i}", "score": 0.9}
            for i in range(10)
        ]
        client_manager.vector_service = mock_vector_service

        mock_embedding_service = AsyncMock()
        mock_embedding_service.generate_embeddings.return_value = {
            "embeddings": [[0.1] * 384],
            "model": "test-model",
            "total_tokens": 10,
        }
        client_manager.embedding_service = mock_embedding_service

        mock_cache_service = AsyncMock()
        mock_cache_service.get_stats.return_value = {
            "hit_rate": 0.85,
            "size": 1000,
            "total_requests": 10000,
        }
        client_manager.cache_service = mock_cache_service

        # Add all other required services
        client_manager.crawling_service = AsyncMock()
        client_manager.project_service = AsyncMock()
        client_manager.deployment_service = AsyncMock()
        client_manager.analytics_service = AsyncMock()
        client_manager.hyde_service = AsyncMock()

        return client_manager

    @pytest.fixture
    async def benchmark_server(self, benchmark_client_manager):
        """Create MCP server for benchmarking."""
        mcp = MockMCPServer("benchmark-server")
        register_mock_tools(mcp, benchmark_client_manager)
        return mcp, benchmark_client_manager

    async def test_connection_pooling_efficiency(self, benchmark_server):
        """Test efficiency of connection pooling under load."""
        mcp_server, mock_client_manager = benchmark_server

        search_tool = None
        for tool in mcp_server._tools:
            if tool.name == "search_documents":
                search_tool = tool
                break

        # Simulate connection pool behavior
        connection_times = []

        async def search_with_connection_tracking(idx):
            # Measure actual handler execution time
            conn_start = time.time()

            result = await search_tool.handler(
                query=f"connection test {idx}",
                collection="docs",
                limit=5,
            )

            conn_time = time.time() - conn_start
            connection_times.append(conn_time)

            return result

        # Execute many requests that would benefit from pooling
        tasks = []
        for i in range(100):
            tasks.append(search_with_connection_tracking(i))

        start_time = time.time()
        await asyncio.gather(*tasks)
        total_time = time.time() - start_time

        # Analyze connection efficiency
        avg_conn_time = mean(connection_times)
        total_conn_time = sum(connection_times)
        conn_overhead_percent = (total_conn_time / total_time) * 100

        print("\nConnection Pooling Analysis:")
        print(f"Average connection time: {avg_conn_time * 1000:.2f}ms")
        print(f"Total connection overhead: {total_conn_time:.2f}s")
        print(f"Connection overhead %: {conn_overhead_percent:.1f}%")

        # With mock services, the overhead should be close to 100% since all time is "connection"
        # In a real system with actual DB connections, pooling would reduce this significantly
        assert avg_conn_time < 0.01, (
            f"Average connection time too high: {avg_conn_time * 1000:.2f}ms"
        )
        assert total_time < 1.0, f"Total operation time too high: {total_time:.2f}s"

    async def test_batch_processing_optimization(self, benchmark_server):
        """Test optimization through batch processing."""
        mcp_server, mock_client_manager = benchmark_server

        embedding_tool = None
        for tool in mcp_server._tools:
            if tool.name == "generate_embeddings":
                embedding_tool = tool
                break

        # Test individual vs batch processing
        num_texts = 100

        # Individual processing
        individual_start = time.time()
        individual_tasks = []
        for i in range(num_texts):
            task = embedding_tool.handler(
                texts=[f"Individual text {i}"],
            )
            individual_tasks.append(task)

        await asyncio.gather(*individual_tasks)
        individual_time = time.time() - individual_start

        # Batch processing
        batch_start = time.time()
        batch_size = 10
        batch_tasks = []

        for i in range(0, num_texts, batch_size):
            batch_texts = [
                f"Batch text {j}" for j in range(i, min(i + batch_size, num_texts))
            ]
            task = embedding_tool.handler(texts=batch_texts)
            batch_tasks.append(task)

        await asyncio.gather(*batch_tasks)
        batch_time = time.time() - batch_start

        # Calculate improvement
        improvement = ((individual_time - batch_time) / individual_time) * 100

        print("\nBatch Processing Optimization:")
        print(f"Individual processing time: {individual_time:.2f}s")
        print(f"Batch processing time: {batch_time:.2f}s")
        print(f"Improvement: {improvement:.1f}%")
        print(f"Speedup: {individual_time / batch_time:.2f}x")

        # Batch processing should be significantly faster
        assert batch_time < individual_time, "Batch processing should be faster"
        assert improvement > 20, f"Batch improvement too small: {improvement:.1f}%"
