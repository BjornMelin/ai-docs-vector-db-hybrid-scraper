import typing

"""Endurance testing scenarios for long-duration performance validation.

This module implements endurance tests to detect memory leaks,
performance degradation over time, and system stability issues.
"""

import asyncio
import logging
import time

import pytest

from ..base_load_test import create_load_test_runner
from ..conftest import LoadTestConfig
from ..conftest import LoadTestType
from ..load_profiles import SteadyLoadProfile

logger = logging.getLogger(__name__)


class TestEnduranceLoad:
    """Test suite for endurance load conditions."""

    @pytest.mark.endurance
    @pytest.mark.slow
    def test_long_duration_stability(self, load_test_runner):
        """Test system stability over extended duration (1 hour)."""
        # Configuration for 1-hour endurance test
        config = LoadTestConfig(
            test_type=LoadTestType.ENDURANCE,
            concurrent_users=50,
            requests_per_second=25,
            duration_seconds=3600,  # 1 hour
            success_criteria={
                "max_error_rate_percent": 2.0,  # Stricter for endurance
                "max_avg_response_time_ms": 800.0,
                "max_memory_growth_mb": 100.0,
                "max_performance_degradation_percent": 20.0,
            },
        )

        # Create environment with steady load
        env = create_load_test_runner()
        env.shape_class = SteadyLoadProfile(
            users=50,
            duration=3600,
            spawn_rate=2,
        )

        # Track metrics over time
        time_series_metrics = []
        memory_samples = []

        @env.events.stats_reset.add_listener
        def collect_time_series_metrics(**kwargs):
            """Collect metrics at regular intervals."""
            import os

            import psutil

            current_time = time.time()
            process = psutil.Process(os.getpid())
            memory_mb = process.memory_info().rss / 1024 / 1024

            stats = env.stats
            if stats and stats.total.num_requests > 0:
                time_series_metrics.append(
                    {
                        "timestamp": current_time,
                        "requests": stats.total.num_requests,
                        "failures": stats.total.num_failures,
                        "avg_response_time": stats.total.avg_response_time,
                        "rps": stats.total.current_rps,
                        "memory_mb": memory_mb,
                    }
                )

                memory_samples.append(
                    {
                        "timestamp": current_time,
                        "memory_mb": memory_mb,
                    }
                )

        # Run endurance test
        result = load_test_runner.run_load_test(
            config=config,
            target_function=self._stable_operation,
            environment=env,
        )

        # Analyze endurance metrics
        endurance_analysis = self._analyze_endurance_performance(
            time_series_metrics, memory_samples
        )

        # Assertions
        assert result.success, f"Endurance test failed: {result.bottlenecks_identified}"
        assert not endurance_analysis["memory_leak_detected"], (
            f"Memory leak detected: {endurance_analysis['memory_growth_rate_mb_per_hour']} MB/hour"
        )
        assert endurance_analysis["performance_degradation_percent"] < 20, (
            f"Performance degraded by {endurance_analysis['performance_degradation_percent']}%"
        )
        assert endurance_analysis["stability_score"] > 0.9, (
            f"Low stability score: {endurance_analysis['stability_score']}"
        )

    @pytest.mark.endurance
    @pytest.mark.slow
    def test_memory_leak_detection(self, load_test_runner, mock_load_test_service):
        """Test for memory leaks during extended operation."""
        # Configure service for memory leak simulation
        memory_leak_simulator = MemoryLeakSimulator()

        async def memory_tracking_operation(**kwargs):
            """Operation that tracks memory usage patterns."""
            # Simulate varying memory usage
            data_size = kwargs.get("data_size_mb", 1.0)

            # Add memory leak simulation
            memory_leak_simulator.allocate_memory(data_size)

            result = await mock_load_test_service.process_request(
                data_size_mb=data_size, **kwargs
            )

            # Sometimes release memory (simulating proper cleanup)
            if time.time() % 10 < 2:  # 20% of the time
                memory_leak_simulator.cleanup_some_memory()

            return result

        # Configuration for memory leak detection
        config = LoadTestConfig(
            test_type=LoadTestType.ENDURANCE,
            concurrent_users=30,
            requests_per_second=15,
            duration_seconds=1800,  # 30 minutes
        )

        # Run test with memory tracking
        load_test_runner.run_load_test(
            config=config,
            target_function=memory_tracking_operation,
            data_size_mb=2.0,  # Larger data to stress memory
        )

        # Analyze memory usage patterns
        memory_analysis = memory_leak_simulator.get_memory_analysis()

        # Assertions
        assert memory_analysis["peak_memory_mb"] < 1024, (
            f"Excessive memory usage: {memory_analysis['peak_memory_mb']} MB"
        )
        assert memory_analysis["growth_rate_mb_per_minute"] < 5, (
            f"High memory growth rate: {memory_analysis['growth_rate_mb_per_minute']} MB/min"
        )
        assert memory_analysis["cleanup_effectiveness"] > 0.7, (
            "Poor memory cleanup effectiveness"
        )

    @pytest.mark.endurance
    def test_cache_performance_over_time(self, load_test_runner):
        """Test cache performance and effectiveness over extended periods."""

        # Simulate cache with aging and eviction
        class CacheSimulator:
            def __init__(self, max_size=1000, ttl_seconds=3600):
                self.cache = {}
                self.access_times = {}
                self.hit_count = 0
                self.miss_count = 0
                self.eviction_count = 0
                self.max_size = max_size
                self.ttl_seconds = ttl_seconds
                self.metrics_history = []

            def get(self, key: str) -> typing.Optional[str]:
                """Get value from cache."""
                current_time = time.time()

                # Check if key exists and is not expired
                if key in self.cache:
                    access_time = self.access_times.get(key, 0)
                    if current_time - access_time < self.ttl_seconds:
                        self.hit_count += 1
                        self.access_times[key] = current_time
                        self._record_metrics("hit")
                        return self.cache[key]
                    else:
                        # Expired
                        del self.cache[key]
                        del self.access_times[key]

                # Cache miss
                self.miss_count += 1
                self._record_metrics("miss")
                return None

            def put(self, key: str, value: str):
                """Put value in cache."""
                current_time = time.time()

                # Evict if at capacity
                if len(self.cache) >= self.max_size:
                    self._evict_oldest()

                self.cache[key] = value
                self.access_times[key] = current_time
                self._record_metrics("put")

            def _evict_oldest(self):
                """Evict oldest entry."""
                if not self.access_times:
                    return

                oldest_key = min(
                    self.access_times.keys(), key=lambda k: self.access_times[k]
                )
                del self.cache[oldest_key]
                del self.access_times[oldest_key]
                self.eviction_count += 1

            def _record_metrics(self, operation: str):
                """Record cache metrics."""
                total_operations = self.hit_count + self.miss_count
                hit_rate = (
                    self.hit_count / total_operations if total_operations > 0 else 0
                )

                self.metrics_history.append(
                    {
                        "timestamp": time.time(),
                        "operation": operation,
                        "hit_rate": hit_rate,
                        "cache_size": len(self.cache),
                        "evictions": self.eviction_count,
                    }
                )

            def get_metrics(self) -> Dict:
                """Get cache performance metrics."""
                total_operations = self.hit_count + self.miss_count
                return {
                    "hit_rate": self.hit_count / total_operations
                    if total_operations > 0
                    else 0,
                    "miss_rate": self.miss_count / total_operations
                    if total_operations > 0
                    else 0,
                    "eviction_count": self.eviction_count,
                    "cache_size": len(self.cache),
                    "total_operations": total_operations,
                }

        cache = CacheSimulator(max_size=500, ttl_seconds=1800)

        async def cache_aware_operation(**kwargs):
            """Operation that uses cache."""
            import hashlib
            import random

            # Generate cache key from operation parameters
            query = kwargs.get("query", f"query_{random.randint(1, 100)}")
            cache_key = hashlib.md5(query.encode()).hexdigest()[:8]

            # Try cache first
            cached_result = cache.get(cache_key)
            if cached_result:
                await asyncio.sleep(0.01)  # Fast cache hit
                return {"result": cached_result, "cached": True}

            # Cache miss - simulate expensive operation
            await asyncio.sleep(0.1)
            result = f"result_{query}_{time.time()}"

            # Store in cache
            cache.put(cache_key, result)

            return {"result": result, "cached": False}

        # Configuration for cache endurance test
        config = LoadTestConfig(
            test_type=LoadTestType.ENDURANCE,
            concurrent_users=40,
            requests_per_second=20,
            duration_seconds=1200,  # 20 minutes
        )

        # Run test with cache simulation
        load_test_runner.run_load_test(
            config=config,
            target_function=cache_aware_operation,
        )

        # Analyze cache performance over time
        cache_metrics = cache.get_metrics()
        cache_analysis = self._analyze_cache_endurance(cache.metrics_history)

        # Assertions
        assert cache_metrics["hit_rate"] > 0.6, (
            f"Low cache hit rate: {cache_metrics['hit_rate']}"
        )
        assert cache_analysis["hit_rate_stability"] > 0.8, (
            "Cache hit rate unstable over time"
        )
        assert cache_analysis["eviction_rate"] < 0.1, "High cache eviction rate"

    @pytest.mark.endurance
    def test_connection_pool_endurance(self, load_test_runner):
        """Test database connection pool behavior over extended periods."""

        # Enhanced connection pool simulator
        class EnduranceConnectionPool:
            def __init__(self, min_size=5, max_size=50, idle_timeout=300):
                self.min_size = min_size
                self.max_size = max_size
                self.idle_timeout = idle_timeout
                self.connections = []
                self.active_connections = set()
                self.idle_connections = set()
                self.connection_metrics = []
                self.last_cleanup = time.time()

                # Initialize minimum connections
                for i in range(min_size):
                    conn_id = f"conn_{i}"
                    self.connections.append(
                        {
                            "id": conn_id,
                            "created_at": time.time(),
                            "last_used": time.time(),
                            "use_count": 0,
                        }
                    )
                    self.idle_connections.add(conn_id)

            async def get_connection(self):
                """Get connection from pool."""
                current_time = time.time()

                # Cleanup idle connections periodically
                if current_time - self.last_cleanup > 60:  # Every minute
                    await self._cleanup_idle_connections()

                # Try to get idle connection
                if self.idle_connections:
                    conn_id = self.idle_connections.pop()
                    self.active_connections.add(conn_id)

                    # Update connection stats
                    for conn in self.connections:
                        if conn["id"] == conn_id:
                            conn["last_used"] = current_time
                            conn["use_count"] += 1
                            break

                    return conn_id

                # Create new connection if under limit
                elif len(self.connections) < self.max_size:
                    conn_id = f"conn_{len(self.connections)}"
                    self.connections.append(
                        {
                            "id": conn_id,
                            "created_at": current_time,
                            "last_used": current_time,
                            "use_count": 1,
                        }
                    )
                    self.active_connections.add(conn_id)
                    return conn_id

                # Pool exhausted - wait
                else:
                    await asyncio.sleep(0.1)
                    return await self.get_connection()  # Retry

            def release_connection(self, conn_id: str):
                """Release connection back to pool."""
                if conn_id in self.active_connections:
                    self.active_connections.remove(conn_id)
                    self.idle_connections.add(conn_id)

                    # Record metrics
                    self.connection_metrics.append(
                        {
                            "timestamp": time.time(),
                            "total_connections": len(self.connections),
                            "active_connections": len(self.active_connections),
                            "idle_connections": len(self.idle_connections),
                            "pool_utilization": len(self.active_connections)
                            / len(self.connections),
                        }
                    )

            async def _cleanup_idle_connections(self):
                """Remove idle connections that exceed timeout."""
                current_time = time.time()
                self.last_cleanup = current_time

                # Don't go below minimum size
                if len(self.connections) <= self.min_size:
                    return

                connections_to_remove = []
                for conn in self.connections:
                    if (
                        conn["id"] in self.idle_connections
                        and current_time - conn["last_used"] > self.idle_timeout
                        and len(self.connections) > self.min_size
                    ):
                        connections_to_remove.append(conn)

                # Remove idle connections
                for conn in connections_to_remove:
                    self.connections.remove(conn)
                    self.idle_connections.discard(conn["id"])

                if connections_to_remove:
                    logger.info(
                        f"Cleaned up {len(connections_to_remove)} idle connections"
                    )

            def get_pool_stats(self) -> Dict:
                """Get pool statistics."""
                if not self.connection_metrics:
                    return {"no_data": True}

                recent_metrics = self.connection_metrics[-10:]  # Last 10 samples
                avg_utilization = sum(
                    m["pool_utilization"] for m in recent_metrics
                ) / len(recent_metrics)

                return {
                    "total_connections": len(self.connections),
                    "active_connections": len(self.active_connections),
                    "idle_connections": len(self.idle_connections),
                    "avg_utilization": avg_utilization,
                    "connection_ages": [
                        time.time() - c["created_at"] for c in self.connections
                    ],
                    "connection_use_counts": [c["use_count"] for c in self.connections],
                }

        pool = EnduranceConnectionPool(min_size=10, max_size=100, idle_timeout=300)

        async def database_endurance_operation(**kwargs):
            """Database operation for endurance testing."""
            conn = await pool.get_connection()

            try:
                # Simulate varying database work
                work_time = kwargs.get("db_work_time", 0.05)
                await asyncio.sleep(work_time)

                return {
                    "status": "success",
                    "connection": conn,
                    "work_time": work_time,
                }
            finally:
                pool.release_connection(conn)

        # Configuration for connection pool endurance
        config = LoadTestConfig(
            test_type=LoadTestType.ENDURANCE,
            concurrent_users=60,
            requests_per_second=30,
            duration_seconds=1800,  # 30 minutes
        )

        # Run endurance test
        load_test_runner.run_load_test(
            config=config,
            target_function=database_endurance_operation,
            db_work_time=0.1,  # Longer DB operations
        )

        # Analyze pool behavior
        pool_stats = pool.get_pool_stats()
        pool_analysis = self._analyze_connection_pool_endurance(pool.connection_metrics)

        # Assertions
        assert pool_stats["avg_utilization"] > 0.3, "Low pool utilization"
        assert pool_analysis["pool_stability"] > 0.9, "Unstable pool behavior"
        assert pool_analysis["cleanup_effectiveness"] > 0.8, "Poor connection cleanup"

    def _stable_operation(self, **kwargs):
        """Stable operation for endurance testing."""
        import asyncio
        import random

        # Consistent, predictable operation
        base_time = 0.05
        variation = random.uniform(-0.01, 0.01)  # Small variation

        return asyncio.sleep(base_time + variation)

    def _analyze_endurance_performance(
        self, time_series: list[Dict], memory_samples: list[Dict]
    ) -> Dict:
        """Analyze performance over extended duration."""
        if len(time_series) < 10 or len(memory_samples) < 10:
            return {
                "memory_leak_detected": False,
                "performance_degradation_percent": 0,
                "stability_score": 0,
                "insufficient_data": True,
            }

        # Analyze memory growth
        memory_values = [s["memory_mb"] for s in memory_samples]
        start_memory = memory_values[0]
        end_memory = memory_values[-1]
        duration_hours = (
            memory_samples[-1]["timestamp"] - memory_samples[0]["timestamp"]
        ) / 3600

        memory_growth_rate = (
            (end_memory - start_memory) / duration_hours if duration_hours > 0 else 0
        )
        memory_leak_detected = memory_growth_rate > 50  # 50 MB/hour threshold

        # Analyze performance degradation
        response_times = [
            m["avg_response_time"] for m in time_series if m["avg_response_time"] > 0
        ]
        if response_times:
            early_avg = sum(response_times[: len(response_times) // 4]) / (
                len(response_times) // 4
            )
            late_avg = sum(response_times[-len(response_times) // 4 :]) / (
                len(response_times) // 4
            )
            performance_degradation = (
                ((late_avg - early_avg) / early_avg) * 100 if early_avg > 0 else 0
            )
        else:
            performance_degradation = 0

        # Calculate stability score
        error_rates = []
        for m in time_series:
            if m["requests"] > 0:
                error_rate = (m["failures"] / m["requests"]) * 100
                error_rates.append(error_rate)

        stability_score = 1.0
        if error_rates:
            avg_error_rate = sum(error_rates) / len(error_rates)
            stability_score = max(
                0.0, 1.0 - (avg_error_rate / 10)
            )  # Penalize high error rates

        return {
            "memory_leak_detected": memory_leak_detected,
            "memory_growth_rate_mb_per_hour": memory_growth_rate,
            "performance_degradation_percent": performance_degradation,
            "stability_score": stability_score,
            "duration_hours": duration_hours,
            "avg_error_rate": sum(error_rates) / len(error_rates) if error_rates else 0,
        }

    def _analyze_cache_endurance(self, metrics_history: list[Dict]) -> Dict:
        """Analyze cache performance over time."""
        if len(metrics_history) < 10:
            return {"hit_rate_stability": 0, "eviction_rate": 0}

        # Calculate hit rate stability (variance over time)
        hit_rates = [m["hit_rate"] for m in metrics_history[-100:]]  # Last 100 samples
        if hit_rates:
            hit_rate_variance = sum(
                (x - sum(hit_rates) / len(hit_rates)) ** 2 for x in hit_rates
            ) / len(hit_rates)
            hit_rate_stability = max(0.0, 1.0 - hit_rate_variance)
        else:
            hit_rate_stability = 0

        # Calculate eviction rate
        eviction_operations = len(
            [m for m in metrics_history if m["operation"] == "put" and "evictions" in m]
        )
        total_operations = len(metrics_history)
        eviction_rate = (
            eviction_operations / total_operations if total_operations > 0 else 0
        )

        return {
            "hit_rate_stability": hit_rate_stability,
            "eviction_rate": eviction_rate,
            "cache_efficiency": sum(hit_rates) / len(hit_rates) if hit_rates else 0,
        }

    def _analyze_connection_pool_endurance(self, metrics: list[Dict]) -> Dict:
        """Analyze connection pool behavior over time."""
        if len(metrics) < 10:
            return {"pool_stability": 0, "cleanup_effectiveness": 0}

        # Analyze pool size stability
        pool_sizes = [m["total_connections"] for m in metrics]
        utilizations = [m["pool_utilization"] for m in metrics]

        # Calculate stability (low variance in pool size)
        if pool_sizes:
            avg_size = sum(pool_sizes) / len(pool_sizes)
            size_variance = sum((x - avg_size) ** 2 for x in pool_sizes) / len(
                pool_sizes
            )
            pool_stability = max(0.0, 1.0 - (size_variance / avg_size**2))
        else:
            pool_stability = 0

        # Estimate cleanup effectiveness (pool doesn't grow indefinitely)
        max_size = max(pool_sizes) if pool_sizes else 0
        min_size = min(pool_sizes) if pool_sizes else 0
        size_range = max_size - min_size
        cleanup_effectiveness = (
            max(0.0, 1.0 - (size_range / max_size)) if max_size > 0 else 1.0
        )

        return {
            "pool_stability": pool_stability,
            "cleanup_effectiveness": cleanup_effectiveness,
            "avg_utilization": sum(utilizations) / len(utilizations)
            if utilizations
            else 0,
            "max_pool_size": max_size,
            "min_pool_size": min_size,
        }


class MemoryLeakSimulator:
    """Simulates memory allocation and cleanup patterns."""

    def __init__(self):
        self.allocated_memory = []
        self.memory_samples = []
        self.start_time = time.time()

    def allocate_memory(self, size_mb: float):
        """Simulate memory allocation."""
        import random

        # Simulate memory allocation (using list as proxy)
        data = [random.random() for _ in range(int(size_mb * 1000))]
        self.allocated_memory.append(
            {
                "data": data,
                "size_mb": size_mb,
                "allocated_at": time.time(),
            }
        )

        # Record memory sample
        total_allocated = sum(item["size_mb"] for item in self.allocated_memory)
        self.memory_samples.append(
            {
                "timestamp": time.time(),
                "allocated_mb": total_allocated,
                "allocation_count": len(self.allocated_memory),
            }
        )

    def cleanup_some_memory(self):
        """Simulate memory cleanup."""
        if self.allocated_memory:
            # Remove 10-30% of allocations
            import random

            cleanup_count = random.randint(1, max(1, len(self.allocated_memory) // 3))

            for _ in range(min(cleanup_count, len(self.allocated_memory))):
                self.allocated_memory.pop(0)  # Remove oldest

    def get_memory_analysis(self) -> Dict:
        """Analyze memory usage patterns."""
        if len(self.memory_samples) < 5:
            return {"insufficient_data": True}

        allocated_values = [s["allocated_mb"] for s in self.memory_samples]

        peak_memory = max(allocated_values)
        start_memory = allocated_values[0]
        end_memory = allocated_values[-1]

        duration_minutes = (time.time() - self.start_time) / 60
        growth_rate = (
            (end_memory - start_memory) / duration_minutes
            if duration_minutes > 0
            else 0
        )

        # Calculate cleanup effectiveness
        allocations = len([s for s in self.memory_samples if s["allocation_count"] > 0])
        cleanups = len(self.memory_samples) - allocations
        cleanup_effectiveness = (
            cleanups / len(self.memory_samples) if self.memory_samples else 0
        )

        return {
            "peak_memory_mb": peak_memory,
            "start_memory_mb": start_memory,
            "end_memory_mb": end_memory,
            "growth_rate_mb_per_minute": growth_rate,
            "cleanup_effectiveness": cleanup_effectiveness,
            "duration_minutes": duration_minutes,
        }
