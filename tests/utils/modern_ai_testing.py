"""Modern AI Testing Utilities Module.

This module provides testing utilities and patterns for AI/ML testing.
"""

import asyncio
import random
import time
from collections.abc import Callable
from functools import wraps
from typing import Any


try:
    import numpy as np
except ImportError:
    np = None


def integration_test(func: Callable) -> Callable:
    """Decorator for integration tests."""

    @wraps(func)
    def wrapper(*args, **kwargs):
        return func(*args, **kwargs)

    return wrapper


class ModernAITestingUtils:
    """Utilities for modern AI testing patterns."""

    @staticmethod
    def create_test_embeddings(
        dimension: int = 384, count: int = 10
    ) -> list[list[float]]:
        """Create test embeddings."""
        if np is None:
            # Fallback to basic Python random when numpy is not available
            return [[random.random() for _ in range(dimension)] for _ in range(count)]

        rng = np.random.default_rng()
        return rng.random((count, dimension)).tolist()

    @staticmethod
    def create_test_documents(count: int = 10) -> list[dict[str, Any]]:
        """Create test documents."""
        return [
            {
                "id": f"doc_{i}",
                "content": f"Test document {i} content",
                "metadata": {"index": i},
            }
            for i in range(count)
        ]

    @staticmethod
    def assert_embeddings_shape(
        embeddings: list[list[float]], expected_count: int, expected_dim: int
    ):
        """Assert embeddings have correct shape."""
        assert len(embeddings) == expected_count
        assert all(len(emb) == expected_dim for emb in embeddings)


class IntegrationTestingPatterns:
    """Patterns for integration testing."""

    @staticmethod
    def setup_test_environment():
        """Setup test environment."""
        return {"test_mode": True, "mock_external_services": True}

    @staticmethod
    def teardown_test_environment():
        """Teardown test environment."""

    @staticmethod
    def create_test_context(settings: dict[str, Any] | None = None) -> dict[str, Any]:
        """Create test context."""
        default_settings = {"test_mode": True, "debug": True}
        if settings:
            default_settings.update(settings)
        return default_settings


class PerformanceTestingFramework:
    """Framework for performance testing with P95 latency validation."""

    def __init__(self):
        self.metrics = []
        self.latencies = []
        self._latency_measurements = []
        self._test_session = TestSession()

    async def run_latency_test(self, search_func, queries, concurrent_requests=10):
        """Run latency test with concurrent requests."""
        self._test_session.start_time = time.perf_counter()
        latencies = []
        successful_requests = 0

        # Create tasks for concurrent execution
        tasks = []
        for i in range(concurrent_requests):
            query = queries[i % len(queries)]
            tasks.append(self._measure_search_latency(search_func, query))

        # Execute all tasks concurrently
        results = await asyncio.gather(*tasks, return_exceptions=True)

        # Process results
        for result in results:
            if isinstance(result, Exception):
                continue

            latency_ms, success = result
            if success:
                successful_requests += 1
                latencies.append(latency_ms)
                self._latency_measurements.append(latency_ms)

        # Calculate metrics
        if latencies:
            latencies.sort()
            return {
                "mean_latency_ms": sum(latencies) / len(latencies),
                "p95_latency_ms": latencies[int(0.95 * len(latencies))]
                if latencies
                else 0.0,
                "p99_latency_ms": latencies[int(0.99 * len(latencies))]
                if latencies
                else 0.0,
                "success_rate": successful_requests / concurrent_requests,
                "total_requests": concurrent_requests,
                "successful_requests": successful_requests,
            }
        return {
            "mean_latency_ms": 0.0,
            "p95_latency_ms": 0.0,
            "p99_latency_ms": 0.0,
            "success_rate": 0.0,
            "total_requests": concurrent_requests,
            "successful_requests": 0,
        }

    async def _measure_search_latency(self, search_func, query):
        """Measure latency of a single search operation."""
        try:
            start_time = time.perf_counter()
            await search_func(query)
            end_time = time.perf_counter()
            latency_ms = (end_time - start_time) * 1000
        except (RuntimeError, TimeoutError, ConnectionError):
            return 0.0, False
        else:
            return latency_ms, True

    def assert_performance_requirements(
        self, metrics, target_p95_ms, success_rate_threshold=0.95
    ):
        """Assert that performance requirements are met."""
        assert metrics["p95_latency_ms"] <= target_p95_ms, (
            f"P95 latency {metrics['p95_latency_ms']:.1f}ms exceeds target {target_p95_ms}ms"
        )
        assert metrics["success_rate"] >= success_rate_threshold, (
            f"Success rate {metrics['success_rate']:.3f} below threshold {success_rate_threshold}"
        )

    def measure_latency(self, operation: Callable) -> float:
        """Measure operation latency."""
        start_time = time.perf_counter()
        operation()
        end_time = time.perf_counter()
        latency = (end_time - start_time) * 1000  # Convert to milliseconds
        self.latencies.append(latency)
        return latency

    def get_p95_latency(self) -> float:
        """Calculate P95 latency."""
        if not self.latencies:
            return 0.0
        sorted_latencies = sorted(self.latencies)
        index = int(0.95 * len(sorted_latencies))
        return sorted_latencies[index]

    def reset_metrics(self):
        """Reset performance metrics."""
        self.metrics.clear()
        self.latencies.clear()
        self._latency_measurements.clear()


class TestSession:
    """Test session for tracking test state."""

    def __init__(self):
        self.start_time = None


def performance_critical_test(target_p95_ms: float = 100.0):
    """Decorator for performance critical tests."""

    def decorator(func: Callable) -> Callable:
        if asyncio.iscoroutinefunction(func):

            @wraps(func)
            async def async_wrapper(*args, **kwargs):
                # The test should provide its own performance framework via fixture
                # This decorator is primarily for documentation and potential future enhancements
                return await func(*args, **kwargs)

            return async_wrapper

        @wraps(func)
        def sync_wrapper(*args, **kwargs):
            # The test should provide its own performance framework via fixture
            # This decorator is primarily for documentation and potential future enhancements
            return func(*args, **kwargs)

        return sync_wrapper

    return decorator
