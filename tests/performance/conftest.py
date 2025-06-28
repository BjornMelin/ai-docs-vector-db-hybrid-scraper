"""Performance testing fixtures and configuration.

This module provides pytest fixtures for comprehensive performance testing including
memory profiling, CPU utilization, network latency, database performance,
API response times, and throughput measurement.
"""

import asyncio
import gc
import resource
import time
from contextlib import asynccontextmanager
from dataclasses import dataclass
from typing import Any
from unittest.mock import AsyncMock

import psutil
import pytest


@dataclass
class PerformanceMetrics:
    """Container for performance metrics."""

    cpu_percent: float
    memory_mb: float
    memory_percent: float
    disk_io_read_mb: float
    disk_io_write_mb: float
    network_sent_mb: float
    network_recv_mb: float
    response_time_ms: float
    throughput_ops_per_sec: float
    timestamp: float


@dataclass
class MemorySnapshot:
    """Memory usage snapshot."""

    rss_mb: float  # Resident Set Size
    vms_mb: float  # Virtual Memory Size
    percent: float  # Memory percentage
    available_mb: float
    gc_objects: int
    timestamp: float


@pytest.fixture(scope="session")
def performance_test_config():
    """Provide performance testing configuration."""
    return {
        "thresholds": {
            "api_response_time_ms": 500,
            "database_query_time_ms": 100,
            "memory_usage_mb": 512,
            "cpu_usage_percent": 80,
            "disk_io_rate_mbps": 100,
            "network_latency_ms": 50,
            "throughput_ops_per_sec": 100,
        },
        "test_durations": {
            "load_test_duration_sec": 60,
            "stress_test_duration_sec": 300,
            "endurance_test_duration_sec": 1800,
            "spike_test_duration_sec": 30,
        },
        "sampling": {
            "metrics_interval_sec": 1.0,
            "memory_profile_interval_sec": 0.5,
            "cpu_profile_interval_sec": 0.1,
        },
        "limits": {
            "max_memory_mb": 1024,
            "max_cpu_percent": 90,
            "max_open_files": 1000,
            "max_connections": 100,
        },
    }


@pytest.fixture
def system_monitor():
    """System resource monitoring utilities."""

    class SystemMonitor:
        def __init__(self):
            self.process = psutil.Process()
            self.baseline_metrics = None
            self.metrics_history = []

        def get_current_metrics(self) -> PerformanceMetrics:
            """Get current system performance metrics."""
            cpu_percent = self.process.cpu_percent()
            memory_info = self.process.memory_info()
            memory_percent = self.process.memory_percent()

            # Disk I/O
            try:
                io_counters = self.process.io_counters()
                disk_read_mb = io_counters.read_bytes / (1024 * 1024)
                disk_write_mb = io_counters.write_bytes / (1024 * 1024)
            except (psutil.AccessDenied, AttributeError):
                disk_read_mb = disk_write_mb = 0.0

            # Network I/O (system-wide)
            try:
                net_io = psutil.net_io_counters()
                net_sent_mb = net_io.bytes_sent / (1024 * 1024)
                net_recv_mb = net_io.bytes_recv / (1024 * 1024)
            except AttributeError:
                net_sent_mb = net_recv_mb = 0.0

            return PerformanceMetrics(
                cpu_percent=cpu_percent,
                memory_mb=memory_info.rss / (1024 * 1024),
                memory_percent=memory_percent,
                disk_io_read_mb=disk_read_mb,
                disk_io_write_mb=disk_write_mb,
                network_sent_mb=net_sent_mb,
                network_recv_mb=net_recv_mb,
                response_time_ms=0.0,  # To be filled by specific tests
                throughput_ops_per_sec=0.0,  # To be filled by specific tests
                timestamp=time.time(),
            )

        def set_baseline(self):
            """Set baseline metrics for comparison."""
            self.baseline_metrics = self.get_current_metrics()

        def get_metrics_delta(self) -> PerformanceMetrics | None:
            """Get metrics delta from baseline."""
            if not self.baseline_metrics:
                return None

            current = self.get_current_metrics()
            return PerformanceMetrics(
                cpu_percent=current.cpu_percent - self.baseline_metrics.cpu_percent,
                memory_mb=current.memory_mb - self.baseline_metrics.memory_mb,
                memory_percent=current.memory_percent
                - self.baseline_metrics.memory_percent,
                disk_io_read_mb=current.disk_io_read_mb
                - self.baseline_metrics.disk_io_read_mb,
                disk_io_write_mb=current.disk_io_write_mb
                - self.baseline_metrics.disk_io_write_mb,
                network_sent_mb=current.network_sent_mb
                - self.baseline_metrics.network_sent_mb,
                network_recv_mb=current.network_recv_mb
                - self.baseline_metrics.network_recv_mb,
                response_time_ms=current.response_time_ms,
                throughput_ops_per_sec=current.throughput_ops_per_sec,
                timestamp=current.timestamp,
            )

        def record_metrics(self):
            """Record current metrics to history."""
            metrics = self.get_current_metrics()
            self.metrics_history.append(metrics)

        def get_average_metrics(self) -> PerformanceMetrics | None:
            """Get average metrics from history."""
            if not self.metrics_history:
                return None

            count = len(self.metrics_history)
            return PerformanceMetrics(
                cpu_percent=sum(m.cpu_percent for m in self.metrics_history) / count,
                memory_mb=sum(m.memory_mb for m in self.metrics_history) / count,
                memory_percent=sum(m.memory_percent for m in self.metrics_history)
                / count,
                disk_io_read_mb=sum(m.disk_io_read_mb for m in self.metrics_history)
                / count,
                disk_io_write_mb=sum(m.disk_io_write_mb for m in self.metrics_history)
                / count,
                network_sent_mb=sum(m.network_sent_mb for m in self.metrics_history)
                / count,
                network_recv_mb=sum(m.network_recv_mb for m in self.metrics_history)
                / count,
                response_time_ms=sum(m.response_time_ms for m in self.metrics_history)
                / count,
                throughput_ops_per_sec=sum(
                    m.throughput_ops_per_sec for m in self.metrics_history
                )
                / count,
                timestamp=time.time(),
            )

    return SystemMonitor()


@pytest.fixture
def memory_profiler():
    """Memory profiling utilities."""

    class MemoryProfiler:
        def __init__(self):
            self.snapshots = []
            self.baseline = None

        def take_snapshot(self) -> MemorySnapshot:
            """Take a memory usage snapshot."""
            process = psutil.Process()
            memory_info = process.memory_info()
            virtual_memory = psutil.virtual_memory()

            # Force garbage collection for accurate measurement
            gc.collect()
            gc_objects = len(gc.get_objects())

            snapshot = MemorySnapshot(
                rss_mb=memory_info.rss / (1024 * 1024),
                vms_mb=memory_info.vms / (1024 * 1024),
                percent=process.memory_percent(),
                available_mb=virtual_memory.available / (1024 * 1024),
                gc_objects=gc_objects,
                timestamp=time.time(),
            )

            self.snapshots.append(snapshot)
            return snapshot

        def set_baseline(self):
            """Set baseline memory snapshot."""
            self.baseline = self.take_snapshot()

        def get_memory_growth(self) -> float | None:
            """Get memory growth since baseline in MB."""
            if not self.baseline or not self.snapshots:
                return None

            current = self.snapshots[-1]
            return current.rss_mb - self.baseline.rss_mb

        def detect_memory_leak(self, threshold_mb: float = 50.0) -> bool:
            """Detect potential memory leak."""
            if len(self.snapshots) < 3:
                return False

            # Check if memory consistently grows
            recent_snapshots = self.snapshots[-3:]
            growth_trend = all(
                recent_snapshots[i].rss_mb > recent_snapshots[i - 1].rss_mb
                for i in range(1, len(recent_snapshots))
            )

            if growth_trend:
                total_growth = recent_snapshots[-1].rss_mb - recent_snapshots[0].rss_mb
                return total_growth > threshold_mb

            return False

        def get_peak_memory(self) -> float | None:
            """Get peak memory usage in MB."""
            if not self.snapshots:
                return None
            return max(snapshot.rss_mb for snapshot in self.snapshots)

    return MemoryProfiler()


@pytest.fixture
def performance_timer():
    """High-precision performance timing utilities."""

    class PerformanceTimer:
        def __init__(self):
            self.timings = {}
            self.active_timers = {}

        def start(self, name: str):
            """Start timing an operation."""
            self.active_timers[name] = time.perf_counter()

        def stop(self, name: str) -> float:
            """Stop timing and return duration in seconds."""
            if name not in self.active_timers:
                raise ValueError(f"Timer '{name}' was not started")

            start_time = self.active_timers.pop(name)
            duration = time.perf_counter() - start_time

            if name not in self.timings:
                self.timings[name] = []
            self.timings[name].append(duration)

            return duration

        def get_stats(self, name: str) -> dict[str, float]:
            """Get timing statistics for an operation."""
            if name not in self.timings or not self.timings[name]:
                return {}

            durations = self.timings[name]
            return {
                "count": len(durations),
                "total": sum(durations),
                "average": sum(durations) / len(durations),
                "min": min(durations),
                "max": max(durations),
                "p50": self._percentile(durations, 50),
                "p95": self._percentile(durations, 95),
                "p99": self._percentile(durations, 99),
            }

        @staticmethod
        def _percentile(data: list[float], percentile: int) -> float:
            """Calculate percentile of data."""
            sorted_data = sorted(data)
            index = int(len(sorted_data) * percentile / 100)
            return sorted_data[min(index, len(sorted_data) - 1)]

        @asynccontextmanager
        async def async_timer(self, name: str):
            """Async context manager for timing operations."""
            self.start(name)
            try:
                yield
            finally:
                self.stop(name)

    return PerformanceTimer()


@pytest.fixture
def throughput_calculator():
    """Throughput calculation utilities."""

    class ThroughputCalculator:
        def __init__(self):
            self.operation_counts = {}
            self.start_times = {}

        def start_measurement(self, operation: str):
            """Start measuring throughput for an operation."""
            self.start_times[operation] = time.perf_counter()
            self.operation_counts[operation] = 0

        def record_operation(self, operation: str, count: int = 1):
            """Record completed operations."""
            if operation not in self.operation_counts:
                self.operation_counts[operation] = 0
            self.operation_counts[operation] += count

        def calculate_throughput(self, operation: str) -> float | None:
            """Calculate operations per second."""
            if (
                operation not in self.start_times
                or operation not in self.operation_counts
            ):
                return None

            elapsed = time.perf_counter() - self.start_times[operation]
            if elapsed <= 0:
                return None

            return self.operation_counts[operation] / elapsed

        def get_all_throughputs(self) -> dict[str, float]:
            """Get throughput for all measured operations."""
            return {
                op: self.calculate_throughput(op) or 0.0 for op in self.operation_counts
            }

    return ThroughputCalculator()


@pytest.fixture
def database_performance_monitor():
    """Database performance monitoring utilities."""

    class DatabasePerformanceMonitor:
        def __init__(self):
            self.query_times = {}
            self.connection_pool_stats = {}

        async def monitor_query(self, query_name: str, query_func):
            """Monitor database query performance."""
            start_time = time.perf_counter()
            try:
                result = await query_func()
                success = True
            except Exception:
                result = None
                success = False
                raise
            finally:
                duration = time.perf_counter() - start_time

                if query_name not in self.query_times:
                    self.query_times[query_name] = []

                self.query_times[query_name].append(
                    {
                        "duration": duration,
                        "success": success,
                        "timestamp": time.time(),
                    }
                )

            return result

        def get_query_stats(self, query_name: str) -> dict[str, Any]:
            """Get statistics for a specific query."""
            if query_name not in self.query_times:
                return {}

            timings = self.query_times[query_name]
            successful_timings = [t["duration"] for t in timings if t["success"]]

            if not successful_timings:
                return {"error": "No successful queries recorded"}

            return {
                "total_queries": len(timings),
                "successful_queries": len(successful_timings),
                "average_time": sum(successful_timings) / len(successful_timings),
                "min_time": min(successful_timings),
                "max_time": max(successful_timings),
                "success_rate": len(successful_timings) / len(timings),
            }

        def record_connection_pool_stats(self, **stats):
            """Record connection pool statistics."""
            self.connection_pool_stats[time.time()] = stats

        def get_connection_pool_trends(self) -> dict[str, Any]:
            """Get connection pool performance trends."""
            if not self.connection_pool_stats:
                return {}

            timestamps = sorted(self.connection_pool_stats.keys())
            latest = self.connection_pool_stats[timestamps[-1]]

            return {
                "current": latest,
                "history_count": len(self.connection_pool_stats),
                "monitoring_duration": timestamps[-1] - timestamps[0]
                if len(timestamps) > 1
                else 0,
            }

    return DatabasePerformanceMonitor()


@pytest.fixture
def network_latency_monitor():
    """Network latency monitoring utilities."""

    class NetworkLatencyMonitor:
        def __init__(self):
            self.latency_measurements = {}

        async def measure_latency(self, endpoint: str, request_func) -> float:
            """Measure network latency for a request."""
            start_time = time.perf_counter()

            try:
                await request_func()
                latency = time.perf_counter() - start_time

                if endpoint not in self.latency_measurements:
                    self.latency_measurements[endpoint] = []

                self.latency_measurements[endpoint].append(
                    {
                        "latency": latency,
                        "timestamp": time.time(),
                        "success": True,
                    }
                )

            except Exception:
                latency = time.perf_counter() - start_time

                if endpoint not in self.latency_measurements:
                    self.latency_measurements[endpoint] = []

                self.latency_measurements[endpoint].append(
                    {
                        "latency": latency,
                        "timestamp": time.time(),
                        "success": False,
                        "error": str(e),
                    }
                )

                raise
            else:
                return latency

        def get_latency_stats(self, endpoint: str) -> dict[str, Any]:
            """Get latency statistics for an endpoint."""
            if endpoint not in self.latency_measurements:
                return {}

            measurements = self.latency_measurements[endpoint]
            successful_measurements = [m for m in measurements if m["success"]]

            if not successful_measurements:
                return {"error": "No successful measurements"}

            latencies = [m["latency"] for m in successful_measurements]

            return {
                "total_requests": len(measurements),
                "successful_requests": len(successful_measurements),
                "average_latency": sum(latencies) / len(latencies),
                "min_latency": min(latencies),
                "max_latency": max(latencies),
                "p95_latency": self._percentile(latencies, 95),
                "success_rate": len(successful_measurements) / len(measurements),
            }

        @staticmethod
        def _percentile(data: list[float], percentile: int) -> float:
            """Calculate percentile of data."""
            sorted_data = sorted(data)
            index = int(len(sorted_data) * percentile / 100)
            return sorted_data[min(index, len(sorted_data) - 1)]

    return NetworkLatencyMonitor()


@pytest.fixture
def resource_limit_monitor():
    """Monitor system resource limits."""

    class ResourceLimitMonitor:
        def __init__(self):
            self.initial_limits = self._get_current_limits()

        def _get_current_limits(self) -> dict[str, int]:
            """Get current resource limits."""
            return {
                "max_open_files": resource.getrlimit(resource.RLIMIT_NOFILE)[0],
                "max_memory": resource.getrlimit(resource.RLIMIT_AS)[0],
                "max_cpu_time": resource.getrlimit(resource.RLIMIT_CPU)[0],
            }

        def check_resource_usage(self) -> dict[str, Any]:
            """Check current resource usage against limits."""
            process = psutil.Process()

            try:
                open_files = len(process.open_files())
            except (psutil.AccessDenied, psutil.NoSuchProcess):
                open_files = 0

            memory_info = process.memory_info()
            cpu_times = process.cpu_times()

            return {
                "open_files": {
                    "current": open_files,
                    "limit": self.initial_limits["max_open_files"],
                    "usage_percent": (
                        open_files / self.initial_limits["max_open_files"]
                    )
                    * 100,
                },
                "memory": {
                    "current_mb": memory_info.rss / (1024 * 1024),
                    "limit_mb": self.initial_limits["max_memory"] / (1024 * 1024)
                    if self.initial_limits["max_memory"] != -1
                    else "unlimited",
                },
                "cpu_time": {
                    "user_time": cpu_times.user,
                    "system_time": cpu_times.system,
                    "total_time": cpu_times.user + cpu_times.system,
                },
            }

        def is_approaching_limits(
            self, threshold_percent: float = 80.0
        ) -> dict[str, bool]:
            """Check if resource usage is approaching limits."""
            usage = self.check_resource_usage()

            return {
                "open_files": usage["open_files"]["usage_percent"] > threshold_percent,
                "memory": False,  # Memory limit checking is complex
                "cpu_time": False,  # CPU time is cumulative
            }

    return ResourceLimitMonitor()


@pytest.fixture
async def performance_test_session():
    """Async session for performance testing with cleanup."""
    session_data = {
        "start_time": time.time(),
        "metrics": [],
        "thresholds_exceeded": [],
        "resource_warnings": [],
    }

    yield session_data

    # Cleanup and garbage collection
    gc.collect()
    session_data["end_time"] = time.time()
    session_data["total_duration"] = (
        session_data["end_time"] - session_data["start_time"]
    )


# Mock services for performance testing
@pytest.fixture
def mock_high_performance_service():
    """Mock high-performance service for testing."""
    service = AsyncMock()

    async def fast_operation():
        """Simulate fast operation (< 10ms)."""
        await asyncio.sleep(0.005)  # 5ms
        return {"status": "success", "duration_ms": 5}

    async def medium_operation():
        """Simulate medium operation (50-100ms)."""
        await asyncio.sleep(0.075)  # 75ms
        return {"status": "success", "duration_ms": 75}

    async def slow_operation():
        """Simulate slow operation (> 500ms)."""
        await asyncio.sleep(0.6)  # 600ms
        return {"status": "success", "duration_ms": 600}

    service.fast_operation = AsyncMock(side_effect=fast_operation)
    service.medium_operation = AsyncMock(side_effect=medium_operation)
    service.slow_operation = AsyncMock(side_effect=slow_operation)

    return service


# Pytest markers for performance test categorization
def pytest_configure(config):
    """Configure performance testing markers."""
    config.addinivalue_line("markers", "performance: mark test as performance test")
    config.addinivalue_line("markers", "memory: mark test as memory performance test")
    config.addinivalue_line("markers", "cpu: mark test as CPU performance test")
    config.addinivalue_line("markers", "network: mark test as network performance test")
    config.addinivalue_line(
        "markers", "database: mark test as database performance test"
    )
    config.addinivalue_line("markers", "api_latency: mark test as API latency test")
    config.addinivalue_line("markers", "throughput: mark test as throughput test")
    config.addinivalue_line("markers", "load: mark test as load test")
    config.addinivalue_line("markers", "stress: mark test as stress test")
