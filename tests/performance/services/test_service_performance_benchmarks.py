"""Performance benchmarks for service layer using pytest-benchmark.

Benchmarks service layer performance including:
- Dependency injection overhead
- Circuit breaker performance impact
- Service interaction latency
- Concurrent service execution
- Memory usage patterns
"""

import asyncio
import contextlib
import time
from unittest.mock import AsyncMock, MagicMock

import pytest

from src.services.functional.circuit_breaker import (
    CircuitBreaker,
    CircuitBreakerConfig,
    create_circuit_breaker,
)
from src.services.functional.dependencies import (
    get_cache_client,
    get_client_manager,
    get_config,
)


class TestDependencyInjectionPerformance:
    """Benchmark dependency injection performance."""

    def test_config_dependency_creation_speed(self, benchmark):
        """Benchmark config dependency creation speed."""

        def create_config():
            return get_config()

        result = benchmark(create_config)
        assert result is not None

    @pytest.mark.asyncio
    async def test_async_dependency_resolution_speed(self, benchmark):
        """Benchmark async dependency resolution speed."""
        config = MagicMock()

        async def resolve_client_manager():
            async for client_manager in get_client_manager(config):
                return client_manager

        # Use benchmark with async support
        result = benchmark.pedantic(
            lambda: asyncio.run(resolve_client_manager()), rounds=10, iterations=1
        )
        assert result is not None

    def test_dependency_caching_performance(self, benchmark):
        """Benchmark dependency caching effectiveness."""
        from functools import lru_cache

        @lru_cache(maxsize=128)
        def cached_expensive_operation(param):
            # Simulate expensive operation
            time.sleep(0.001)
            return f"result_{param}"

        def test_cached_calls():
            results = []
            for i in range(10):
                results.append(cached_expensive_operation(i % 3))  # Only 3 unique calls
            return results

        result = benchmark(test_cached_calls)
        assert len(result) == 10

    @pytest.mark.asyncio
    async def test_concurrent_dependency_resolution(self, benchmark):
        """Benchmark concurrent dependency resolution."""
        config = MagicMock()
        config.cache.dragonfly_url = "redis://localhost:6379"
        config.cache.enable_local_cache = True
        config.cache.enable_dragonfly_cache = True
        config.cache.local_max_size = 1000
        config.cache.local_max_memory_mb = 100

        async def concurrent_dependency_creation():
            tasks = []
            for _ in range(10):
                task = asyncio.create_task(get_cache_client(config).__anext__())
                tasks.append(task)

            results = await asyncio.gather(*tasks, return_exceptions=True)
            return len([r for r in results if not isinstance(r, Exception)])

        result = benchmark.pedantic(
            lambda: asyncio.run(concurrent_dependency_creation()),
            rounds=5,
            iterations=1,
        )
        assert result >= 0


class TestCircuitBreakerPerformance:
    """Benchmark circuit breaker performance impact."""

    def test_circuit_breaker_overhead_closed_state(self, benchmark):
        """Benchmark circuit breaker overhead in closed state."""
        config = CircuitBreakerConfig.simple_mode()
        circuit_breaker = CircuitBreaker(config)

        async def successful_operation():
            return "success"

        async def test_circuit_breaker_call():
            return await circuit_breaker.call(successful_operation)

        result = benchmark.pedantic(
            lambda: asyncio.run(test_circuit_breaker_call()), rounds=100, iterations=1
        )
        assert result == "success"

    def test_circuit_breaker_metrics_collection_performance(self, benchmark):
        """Benchmark circuit breaker metrics collection performance."""
        config = CircuitBreakerConfig.enterprise_mode()
        circuit_breaker = CircuitBreaker(config)

        # Generate some metrics data
        circuit_breaker.metrics.total_requests = 1000
        circuit_breaker.metrics.successful_requests = 950
        circuit_breaker.metrics.failed_requests = 50

        def collect_metrics():
            return circuit_breaker.get_metrics()

        result = benchmark(collect_metrics)
        assert result["total_requests"] == 1000

    @pytest.mark.asyncio
    async def test_circuit_breaker_concurrent_access(self, benchmark):
        """Benchmark circuit breaker under concurrent access."""
        config = CircuitBreakerConfig.enterprise_mode()
        circuit_breaker = CircuitBreaker(config)

        async def fast_operation():
            await asyncio.sleep(0.0001)  # Minimal delay
            return "fast_result"

        async def concurrent_circuit_breaker_calls():
            tasks = [circuit_breaker.call(fast_operation) for _ in range(50)]
            results = await asyncio.gather(*tasks)
            return len(results)

        result = benchmark.pedantic(
            lambda: asyncio.run(concurrent_circuit_breaker_calls()),
            rounds=10,
            iterations=1,
        )
        assert result == 50

    def test_circuit_breaker_state_transition_performance(self, benchmark):
        """Benchmark circuit breaker state transition performance."""
        config = CircuitBreakerConfig.simple_mode()
        config.failure_threshold = 1
        circuit_breaker = CircuitBreaker(config)

        async def failing_operation():
            raise Exception("Simulated failure")

        async def state_transition_test():
            # Trigger failure to open circuit
            with contextlib.suppress(Exception):
                await circuit_breaker.call(failing_operation)

            # Simulate timeout for recovery
            circuit_breaker.last_failure_time = 0

            # Test recovery
            async def success_operation():
                return "recovered"

            return await circuit_breaker.call(success_operation)

        result = benchmark.pedantic(
            lambda: asyncio.run(state_transition_test()), rounds=20, iterations=1
        )
        assert result == "recovered"


class TestServiceInteractionPerformance:
    """Benchmark service interaction performance."""

    @pytest.mark.asyncio
    async def test_embedding_cache_interaction_speed(self, benchmark):
        """Benchmark embedding-cache interaction speed."""
        cache_manager = AsyncMock()
        embedding_manager = AsyncMock()

        # Setup fast cache hit
        cache_manager.get.return_value = {
            "embeddings": [[0.1, 0.2, 0.3]],
            "cached": True,
        }

        async def embedding_cache_interaction():
            cache_key = "test_embedding"

            # Check cache
            cached_result = await cache_manager.get(cache_key)
            if cached_result:
                return cached_result

            # Generate if not cached
            result = await embedding_manager.generate_embeddings(["test"])
            await cache_manager.set(cache_key, result)
            return result

        result = benchmark.pedantic(
            lambda: asyncio.run(embedding_cache_interaction()), rounds=100, iterations=1
        )
        assert result["cached"] is True

    @pytest.mark.asyncio
    async def test_pipeline_processing_speed(self, benchmark):
        """Benchmark multi-service pipeline processing speed."""
        # Mock services
        browser_service = AsyncMock()
        content_ai_service = AsyncMock()
        embedding_service = AsyncMock()
        vector_db_service = AsyncMock()

        # Setup mock responses
        browser_service.scrape_url.return_value = {
            "content": "scraped content",
            "success": True,
        }
        content_ai_service.analyze.return_value = {
            "quality_score": 0.9,
            "content_type": "article",
        }
        embedding_service.generate_embeddings.return_value = {
            "embeddings": [[0.1, 0.2, 0.3]]
        }
        vector_db_service.upsert.return_value = {"status": "success"}

        async def process_document_pipeline():
            url = "https://example.com"

            # Step 1: Scrape content
            scraping_result = await browser_service.scrape_url(url)

            # Step 2: Analyze content
            analysis_result = await content_ai_service.analyze(
                scraping_result["content"]
            )

            # Step 3: Generate embeddings
            embedding_result = await embedding_service.generate_embeddings(
                [scraping_result["content"]]
            )

            # Step 4: Store in vector database
            storage_result = await vector_db_service.upsert(
                {
                    "vector": embedding_result["embeddings"][0],
                    "payload": {
                        "content": scraping_result["content"],
                        "quality": analysis_result["quality_score"],
                    },
                }
            )

            return storage_result

        result = benchmark.pedantic(
            lambda: asyncio.run(process_document_pipeline()), rounds=50, iterations=1
        )
        assert result["status"] == "success"

    @pytest.mark.asyncio
    async def test_concurrent_service_calls_performance(self, benchmark):
        """Benchmark concurrent service calls performance."""
        # Mock multiple services
        services = {
            "embedding": AsyncMock(),
            "vector_db": AsyncMock(),
            "cache": AsyncMock(),
            "browser": AsyncMock(),
        }

        # Setup responses
        for service in services.values():
            service.process.return_value = {"status": "success", "data": "result"}

        async def concurrent_service_calls():
            tasks = []
            for service_name, service in services.items():
                task = asyncio.create_task(service.process(f"data_for_{service_name}"))
                tasks.append(task)

            results = await asyncio.gather(*tasks)
            return len(results)

        result = benchmark.pedantic(
            lambda: asyncio.run(concurrent_service_calls()), rounds=25, iterations=1
        )
        assert result == 4

    def test_service_factory_performance(self, benchmark):
        """Benchmark service factory creation performance."""

        def create_circuit_breakers():
            breakers = {}
            service_names = ["embedding", "vector_db", "cache", "browser", "content_ai"]

            for service_name in service_names:
                breakers[service_name] = create_circuit_breaker(
                    "enterprise", failure_threshold=5, recovery_timeout=60
                )

            return breakers

        result = benchmark(create_circuit_breakers)
        assert len(result) == 5


class TestMemoryUsagePatterns:
    """Benchmark memory usage patterns in services."""

    def test_dependency_memory_overhead(self, benchmark):
        """Benchmark memory overhead of dependency injection."""
        import sys

        def measure_dependency_memory():
            # Get initial memory usage
            initial_objects = len(gc.get_objects()) if "gc" in sys.modules else 0

            # Create dependencies
            dependencies = []
            for _i in range(100):
                config = get_config()
                dependencies.append(config)

            # Get final memory usage
            final_objects = len(gc.get_objects()) if "gc" in sys.modules else 0

            return len(dependencies), final_objects - initial_objects

        # Import gc if available
        try:
            import gc
        except ImportError:
            gc = None

        result = benchmark(measure_dependency_memory)
        assert result[0] == 100  # Created 100 dependencies

    @pytest.mark.asyncio
    async def test_circuit_breaker_memory_usage(self, benchmark):
        """Benchmark circuit breaker memory usage with metrics."""

        async def circuit_breaker_memory_test():
            circuit_breakers = []

            for _i in range(10):
                config = CircuitBreakerConfig.enterprise_mode()
                cb = CircuitBreaker(config)

                # Generate some metrics
                for j in range(100):
                    cb.metrics.total_requests += 1
                    if j % 10 == 0:
                        cb.metrics.failed_requests += 1
                    else:
                        cb.metrics.successful_requests += 1

                circuit_breakers.append(cb)

            return len(circuit_breakers)

        result = benchmark.pedantic(
            lambda: asyncio.run(circuit_breaker_memory_test()), rounds=10, iterations=1
        )
        assert result == 10

    def test_service_cache_memory_efficiency(self, benchmark):
        """Benchmark service cache memory efficiency."""
        from functools import lru_cache

        @lru_cache(maxsize=1000)
        def cached_service_operation(data_id):
            # Simulate processing
            return f"processed_{data_id}"

        def test_cache_efficiency():
            # Create lots of requests with repeated data
            results = []
            for i in range(1000):
                data_id = i % 100  # Only 100 unique items
                result = cached_service_operation(data_id)
                results.append(result)

            return len(results)

        result = benchmark(test_cache_efficiency)
        assert result == 1000


class TestServiceReliabilityBenchmarks:
    """Benchmark service reliability and error handling performance."""

    @pytest.mark.asyncio
    async def test_error_handling_performance(self, benchmark):
        """Benchmark error handling performance in services."""

        async def error_handling_test():
            errors_handled = 0

            for i in range(100):
                try:
                    if i % 10 == 0:
                        raise ConnectionError("Simulated connection error")
                    elif i % 15 == 0:
                        raise TimeoutError("Simulated timeout")
                    else:
                        # Successful operation
                        pass
                except (ConnectionError, TimeoutError):
                    errors_handled += 1
                    # Simulate error recovery
                    await asyncio.sleep(0.0001)

            return errors_handled

        result = benchmark.pedantic(
            lambda: asyncio.run(error_handling_test()), rounds=20, iterations=1
        )
        assert result > 0

    @pytest.mark.asyncio
    async def test_fallback_mechanism_performance(self, benchmark):
        """Benchmark fallback mechanism performance."""

        async def primary_service():
            raise Exception("Primary service down")

        async def fallback_service():
            await asyncio.sleep(0.0001)  # Minimal delay
            return "fallback_result"

        async def service_with_fallback():
            try:
                return await primary_service()
            except Exception:
                return await fallback_service()

        async def fallback_performance_test():
            results = []
            for _ in range(50):
                result = await service_with_fallback()
                results.append(result)
            return len(results)

        result = benchmark.pedantic(
            lambda: asyncio.run(fallback_performance_test()), rounds=20, iterations=1
        )
        assert result == 50

    @pytest.mark.asyncio
    async def test_retry_mechanism_performance(self, benchmark):
        """Benchmark retry mechanism performance."""

        async def flaky_service(attempt_count=None):
            if attempt_count is None:
                attempt_count = [0]
            attempt_count[0] += 1
            if attempt_count[0] < 3:
                raise ConnectionError("Service temporarily unavailable")
            attempt_count[0] = 0  # Reset for next test
            return "success_after_retries"

        async def retry_with_backoff(operation, max_retries=3):
            for attempt in range(max_retries):
                try:
                    return await operation()
                except Exception:
                    if attempt == max_retries - 1:
                        raise
                    await asyncio.sleep(0.001 * (2**attempt))  # Exponential backoff

        async def retry_performance_test():
            results = []
            for _ in range(10):
                try:
                    result = await retry_with_backoff(flaky_service)
                    results.append(result)
                except Exception:
                    results.append("failed")
            return len([r for r in results if r != "failed"])

        result = benchmark.pedantic(
            lambda: asyncio.run(retry_performance_test()), rounds=10, iterations=1
        )
        assert result >= 8  # Most should succeed after retries


class TestServiceScalabilityBenchmarks:
    """Benchmark service scalability patterns."""

    @pytest.mark.asyncio
    async def test_connection_pool_scalability(self, benchmark):
        """Benchmark connection pool scalability."""

        class MockConnectionPool:
            def __init__(self, max_size=10):
                self.max_size = max_size
                self.active_connections = 0
                self.pool = []

            async def acquire(self):
                if len(self.pool) > 0:
                    return self.pool.pop()

                if self.active_connections < self.max_size:
                    self.active_connections += 1
                    return f"connection_{self.active_connections}"

                # Pool exhausted
                raise Exception("Pool exhausted")

            async def release(self, connection):
                self.pool.append(connection)

        async def connection_pool_test():
            pool = MockConnectionPool(max_size=20)
            successful_acquisitions = 0

            tasks = []
            for _i in range(50):

                async def acquire_and_release():
                    nonlocal successful_acquisitions
                    try:
                        conn = await pool.acquire()
                        await asyncio.sleep(0.0001)  # Simulate work
                        await pool.release(conn)
                        successful_acquisitions += 1
                    except Exception:
                        pass

                tasks.append(asyncio.create_task(acquire_and_release()))

            await asyncio.gather(*tasks, return_exceptions=True)
            return successful_acquisitions

        result = benchmark.pedantic(
            lambda: asyncio.run(connection_pool_test()), rounds=10, iterations=1
        )
        assert result > 0

    @pytest.mark.asyncio
    async def test_load_balancing_performance(self, benchmark):
        """Benchmark load balancing performance."""

        class LoadBalancer:
            def __init__(self, services):
                self.services = services
                self.current_index = 0

            def get_next_service(self):
                service = self.services[self.current_index]
                self.current_index = (self.current_index + 1) % len(self.services)
                return service

        async def load_balancing_test():
            # Mock services with different latencies
            services = [
                AsyncMock(process=AsyncMock(return_value="service_1_result")),
                AsyncMock(process=AsyncMock(return_value="service_2_result")),
                AsyncMock(process=AsyncMock(return_value="service_3_result")),
            ]

            balancer = LoadBalancer(services)
            results = []

            # Process 100 requests
            for i in range(100):
                service = balancer.get_next_service()
                result = await service.process(f"request_{i}")
                results.append(result)

            return len(results)

        result = benchmark.pedantic(
            lambda: asyncio.run(load_balancing_test()), rounds=20, iterations=1
        )
        assert result == 100
