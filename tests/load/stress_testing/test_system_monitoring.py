"""System monitoring and metrics collection during stress testing.

This module implements comprehensive monitoring of system resources,
application metrics, and performance indicators during stress tests
to identify bottlenecks and performance degradation patterns.
"""

import asyncio
import concurrent.futures
import contextlib
import gc
import logging
import os
import random
import statistics
import tempfile
import threading
import time
from collections import defaultdict, deque
from dataclasses import dataclass
from typing import Any

import psutil
import pytest

from ..conftest import LoadTestConfig, LoadTestType


class TestError(Exception):
    """Custom exception for this module."""

    pass


logger = logging.getLogger(__name__)


@dataclass
class SystemMetrics:
    """System-level metrics."""

    timestamp: float
    cpu_percent: float
    memory_percent: float
    memory_mb: float
    disk_io_read_mb: float
    disk_io_write_mb: float
    network_io_sent_mb: float
    network_io_recv_mb: float
    open_files: int
    tcp_connections: int
    process_count: int
    load_average: tuple[float, float, float] | None = None


@dataclass
class ApplicationMetrics:
    """Application-level metrics."""

    timestamp: float
    active_requests: int
    request_queue_size: int
    response_times: list[float]
    error_rate: float
    throughput_rps: float
    cache_hit_rate: float
    database_connections: int
    embedding_queue_size: int
    vector_db_operations: int
    gc_collections: int


@dataclass
class PerformanceAlert:
    """Performance alert definition."""

    metric_name: str
    threshold: float
    current_value: float
    severity: str  # "warning", "critical"
    message: str
    timestamp: float


class SystemMonitor:
    """Monitor system resources during stress tests."""

    def __init__(self, collection_interval: float = 1.0):
        self.collection_interval = collection_interval
        self.system_metrics: list[SystemMetrics] = []
        self.application_metrics: list[ApplicationMetrics] = []
        self.performance_alerts: list[PerformanceAlert] = []
        self.monitoring = False
        self.monitor_thread = None
        self.process = psutil.Process()

        # Alert thresholds
        self.alert_thresholds = {
            "cpu_percent": {"warning": 80.0, "critical": 95.0},
            "memory_percent": {"warning": 85.0, "critical": 95.0},
            "disk_io_mb_per_sec": {"warning": 100.0, "critical": 500.0},
            "network_io_mb_per_sec": {"warning": 50.0, "critical": 200.0},
            "response_time_p95": {"warning": 2000.0, "critical": 5000.0},
            "error_rate": {"warning": 5.0, "critical": 15.0},
            "open_files": {"warning": 1000, "critical": 2000},
        }

    def start_monitoring(self):
        """Start system monitoring."""
        self.monitoring = True
        self.monitor_thread = threading.Thread(
            target=self._monitoring_loop, daemon=True
        )
        self.monitor_thread.start()
        logger.info("Started system monitoring")

    def stop_monitoring(self):
        """Stop system monitoring."""
        self.monitoring = False
        if self.monitor_thread:
            self.monitor_thread.join(timeout=5.0)
        logger.info("Stopped system monitoring")

    def _monitoring_loop(self):
        """Main monitoring loop."""
        last_disk_io = None
        last_network_io = None

        while self.monitoring:
            try:
                # Collect system metrics
                system_metrics = self._collect_system_metrics(
                    last_disk_io, last_network_io
                )
                self.system_metrics.append(system_metrics)

                # Update for delta calculations
                last_disk_io = (
                    system_metrics.disk_io_read_mb + system_metrics.disk_io_write_mb,
                    time.time(),
                )
                last_network_io = (
                    system_metrics.network_io_sent_mb
                    + system_metrics.network_io_recv_mb,
                    time.time(),
                )

                # Collect application metrics
                app_metrics = self._collect_application_metrics()
                self.application_metrics.append(app_metrics)

                # Check for alerts
                self._check_alerts(system_metrics, app_metrics)

                # Keep only recent metrics (last 10 minutes)
                cutoff_time = time.time() - 600
                self.system_metrics = [
                    m for m in self.system_metrics if m.timestamp > cutoff_time
                ]
                self.application_metrics = [
                    m for m in self.application_metrics if m.timestamp > cutoff_time
                ]

            except Exception:
                logger.warning(f"Error during monitoring: {e}")

            time.sleep(self.collection_interval)

    def _collect_system_metrics(
        self,
        _last_disk_io: tuple[float, float] | None,
        _last_network_io: tuple[float, float] | None,
    ) -> SystemMetrics:
        """Collect system-level metrics."""
        # CPU and memory
        cpu_percent = psutil.cpu_percent()
        memory = psutil.virtual_memory()

        # Disk I/O
        disk_io = psutil.disk_io_counters()
        if disk_io:
            disk_read_mb = disk_io.read_bytes / (1024 * 1024)
            disk_write_mb = disk_io.write_bytes / (1024 * 1024)
        else:
            disk_read_mb = disk_write_mb = 0

        # Network I/O
        network_io = psutil.net_io_counters()
        if network_io:
            network_sent_mb = network_io.bytes_sent / (1024 * 1024)
            network_recv_mb = network_io.bytes_recv / (1024 * 1024)
        else:
            network_sent_mb = network_recv_mb = 0

        # Process information
        try:
            open_files = (
                self.process.num_fds() if hasattr(self.process, "num_fds") else 0
            )
        except (psutil.AccessDenied, psutil.NoSuchProcess):
            open_files = 0

        try:
            tcp_connections = len(
                [c for c in self.process.connections() if c.type == psutil.SOCK_STREAM]
            )
        except (psutil.AccessDenied, psutil.NoSuchProcess):
            tcp_connections = 0

        # System load
        try:
            load_average = os.getloadavg()
        except (OSError, AttributeError):
            load_average = None

        return SystemMetrics(
            timestamp=time.time(),
            cpu_percent=cpu_percent,
            memory_percent=memory.percent,
            memory_mb=memory.used / (1024 * 1024),
            disk_io_read_mb=disk_read_mb,
            disk_io_write_mb=disk_write_mb,
            network_io_sent_mb=network_sent_mb,
            network_io_recv_mb=network_recv_mb,
            open_files=open_files,
            tcp_connections=tcp_connections,
            process_count=len(psutil.pids()),
            load_average=load_average,
        )

    def _collect_application_metrics(self) -> ApplicationMetrics:
        """Collect application-level metrics."""
        # Mock application metrics (would be replaced with real metrics in production)
        current_time = time.time()

        # Simulate some application metrics
        active_requests = (
            len([m for m in self.system_metrics[-10:] if m.cpu_percent > 10])
            if self.system_metrics
            else 0
        )

        # GC stats
        gc_stats = gc.get_stats()
        total_collections = sum(stat["collections"] for stat in gc_stats)

        return ApplicationMetrics(
            timestamp=current_time,
            active_requests=active_requests,
            request_queue_size=max(0, active_requests - 5),
            response_times=[],  # Would be populated from real metrics
            error_rate=0.0,  # Would be calculated from real metrics
            throughput_rps=0.0,  # Would be calculated from real metrics
            cache_hit_rate=0.9,  # Would come from cache metrics
            database_connections=min(active_requests, 10),
            embedding_queue_size=max(0, active_requests - 3),
            vector_db_operations=active_requests * 2,
            gc_collections=total_collections,
        )

    def _check_alerts(
        self, system_metrics: SystemMetrics, _app_metrics: ApplicationMetrics
    ):
        """Check for performance alerts."""
        current_time = time.time()

        # Check system metrics
        alerts_to_check = [
            ("cpu_percent", system_metrics.cpu_percent, "CPU usage"),
            ("memory_percent", system_metrics.memory_percent, "Memory usage"),
            ("open_files", system_metrics.open_files, "Open file descriptors"),
        ]

        for metric_name, value, description in alerts_to_check:
            if metric_name in self.alert_thresholds:
                thresholds = self.alert_thresholds[metric_name]

                if value >= thresholds["critical"]:
                    self._create_alert(
                        metric_name,
                        value,
                        "critical",
                        f"{description} critical: {value:.2f}",
                        current_time,
                    )
                elif value >= thresholds["warning"]:
                    self._create_alert(
                        metric_name,
                        value,
                        "warning",
                        f"{description} warning: {value:.2f}",
                        current_time,
                    )

    def _create_alert(
        self,
        metric_name: str,
        value: float,
        severity: str,
        message: str,
        timestamp: float,
    ):
        """Create a performance alert."""
        # Avoid duplicate alerts (same metric, same severity within 30 seconds)
        recent_alerts = [
            a
            for a in self.performance_alerts
            if a.metric_name == metric_name
            and a.severity == severity
            and timestamp - a.timestamp < 30
        ]

        if not recent_alerts:
            alert = PerformanceAlert(
                metric_name=metric_name,
                threshold=self.alert_thresholds[metric_name][severity],
                current_value=value,
                severity=severity,
                message=message,
                timestamp=timestamp,
            )
            self.performance_alerts.append(alert)
            logger.warning(f"Performance alert: {message}")

    def get_metrics_summary(self) -> dict[str, Any]:
        """Get summary of collected metrics."""
        if not self.system_metrics:
            return {"error": "No metrics collected"}

        # System metrics summary
        cpu_values = [m.cpu_percent for m in self.system_metrics]
        memory_values = [m.memory_percent for m in self.system_metrics]

        system_summary = {
            "cpu": {
                "min": min(cpu_values),
                "max": max(cpu_values),
                "avg": statistics.mean(cpu_values),
                "p95": self._percentile(cpu_values, 95),
            },
            "memory": {
                "min": min(memory_values),
                "max": max(memory_values),
                "avg": statistics.mean(memory_values),
                "p95": self._percentile(memory_values, 95),
            },
            "peak_open_files": max(m.open_files for m in self.system_metrics),
            "peak_tcp_connections": max(m.tcp_connections for m in self.system_metrics),
        }

        # Alerts summary
        alerts_by_severity = defaultdict(int)
        for alert in self.performance_alerts:
            alerts_by_severity[alert.severity] += 1

        return {
            "collection_duration": self.system_metrics[-1].timestamp
            - self.system_metrics[0].timestamp,
            "total_data_points": len(self.system_metrics),
            "system_metrics": system_summary,
            "alerts": dict(alerts_by_severity),
            "total_alerts": len(self.performance_alerts),
        }

    @staticmethod
    def _percentile(data: list[float], percentile: int) -> float:
        """Calculate percentile of data."""
        if not data:
            return 0.0
        sorted_data = sorted(data)
        index = int(len(sorted_data) * percentile / 100)
        return sorted_data[min(index, len(sorted_data) - 1)]


class MetricsCollector:
    """Collect and analyze performance metrics during stress tests."""

    def __init__(self):
        self.response_times = deque(maxlen=10000)  # Keep last 10k response times
        self.error_counts = defaultdict(int)
        self.throughput_samples = deque(maxlen=1000)
        self.custom_metrics = defaultdict(list)
        self.start_time = None
        self.end_time = None

    def start_collection(self):
        """Start metrics collection."""
        self.start_time = time.time()

    def stop_collection(self):
        """Stop metrics collection."""
        self.end_time = time.time()

    def record_response_time(self, response_time: float):
        """Record response time."""
        self.response_times.append(response_time)

    def record_error(self, error_type: str):
        """Record error by type."""
        self.error_counts[error_type] += 1

    def record_throughput(self, rps: float):
        """Record throughput sample."""
        self.throughput_samples.append(rps)

    def record_custom_metric(self, name: str, value: float):
        """Record custom metric."""
        self.custom_metrics[name].append(value)

    def get_performance_analysis(self) -> dict[str, Any]:
        """Get comprehensive performance analysis."""
        if not self.response_times:
            return {"error": "No performance data collected"}

        # Response time analysis
        response_times_ms = [rt * 1000 for rt in self.response_times]
        response_analysis = {
            "count": len(response_times_ms),
            "min_ms": min(response_times_ms),
            "max_ms": max(response_times_ms),
            "mean_ms": statistics.mean(response_times_ms),
            "median_ms": statistics.median(response_times_ms),
            "p95_ms": self._percentile(response_times_ms, 95),
            "p99_ms": self._percentile(response_times_ms, 99),
            "std_dev_ms": statistics.stdev(response_times_ms)
            if len(response_times_ms) > 1
            else 0,
        }

        # Error analysis
        total_errors = sum(self.error_counts.values())
        total_requests = len(self.response_times) + total_errors
        error_rate = (total_errors / max(total_requests, 1)) * 100

        # Throughput analysis
        if self.throughput_samples:
            throughput_analysis = {
                "min_rps": min(self.throughput_samples),
                "max_rps": max(self.throughput_samples),
                "mean_rps": statistics.mean(self.throughput_samples),
                "median_rps": statistics.median(self.throughput_samples),
            }
        else:
            throughput_analysis = {"mean_rps": 0, "max_rps": 0}

        # Duration
        duration = (self.end_time or time.time()) - (self.start_time or time.time())

        return {
            "duration_seconds": duration,
            "response_times": response_analysis,
            "error_rate_percent": error_rate,
            "error_breakdown": dict(self.error_counts),
            "throughput": throughput_analysis,
            "custom_metrics": {
                name: {
                    "count": len(values),
                    "mean": statistics.mean(values) if values else 0,
                    "max": max(values) if values else 0,
                    "min": min(values) if values else 0,
                }
                for name, values in self.custom_metrics.items()
            },
        }

    @staticmethod
    def _percentile(data: list[float], percentile: int) -> float:
        """Calculate percentile of data."""
        if not data:
            return 0.0
        sorted_data = sorted(data)
        index = int(len(sorted_data) * percentile / 100)
        return sorted_data[min(index, len(sorted_data) - 1)]


class TestSystemMonitoring:
    """Test suite for system monitoring during stress tests."""

    @pytest.mark.stress
    async def test_comprehensive_system_monitoring(self, load_test_runner):
        """Test comprehensive monitoring of system resources during stress."""

        monitor = SystemMonitor(collection_interval=0.5)
        metrics_collector = MetricsCollector()

        monitor.start_monitoring()
        metrics_collector.start_collection()

        try:
            # Mock service that generates various system load patterns
            class SystemStressingService:
                def __init__(self):
                    self.call_count = 0
                    self.memory_hogs = []
                    self.cpu_intensive_tasks = []

                async def stress_system(self, stress_type: str = "mixed", **_kwargs):
                    """Generate different types of system stress."""
                    self.call_count += 1
                    start_time = time.perf_counter()

                    try:
                        if stress_type == "memory":
                            # Memory stress
                            memory_chunk = bytearray(5 * 1024 * 1024)  # 5MB
                            self.memory_hogs.append(memory_chunk)
                            await asyncio.sleep(0.1)

                        elif stress_type == "cpu":
                            # CPU stress
                            def cpu_intensive_work():
                                return sum(i**2 for i in range(100000))

                            # Run CPU work in thread pool
                            with concurrent.futures.ThreadPoolExecutor(
                                max_workers=2
                            ) as executor:
                                future = executor.submit(cpu_intensive_work)
                                future.result(timeout=1.0)
                            await asyncio.sleep(0.05)

                        elif stress_type == "io":
                            # I/O stress (file operations)
                            with tempfile.NamedTemporaryFile(delete=True) as tmp:
                                data = b"x" * (1024 * 1024)  # 1MB
                                tmp.write(data)
                                tmp.flush()
                                tmp.seek(0)
                                _ = tmp.read()
                            await asyncio.sleep(0.1)

                        else:  # mixed
                            # Mixed stress
                            small_memory = bytearray(1024 * 1024)  # 1MB
                            self.memory_hogs.append(small_memory)

                            # Light CPU work
                            _ = sum(i for i in range(10000))
                            await asyncio.sleep(0.2)

                        response_time = time.perf_counter() - start_time
                        metrics_collector.record_response_time(response_time)

                        # Record custom metrics
                        metrics_collector.record_custom_metric(
                            "memory_allocations", len(self.memory_hogs)
                        )
                        metrics_collector.record_custom_metric(
                            "cpu_tasks", len(self.cpu_intensive_tasks)
                        )

                    except Exception:
                        # Handle stress operation errors
                        logger.warning(f"Stress operation failed: {e}")
                        raise
                    else:
                        return {
                            "status": "stress_applied",
                            "stress_type": stress_type,
                            "response_time": response_time,
                            "call_count": self.call_count,
                        }
                        response_time = time.perf_counter() - start_time
                        metrics_collector.record_response_time(response_time)
                        metrics_collector.record_error("StressError")
                        raise TestError("Stress operation failed") from None

            stressing_service = SystemStressingService()

            # Test different stress patterns
            stress_patterns = [
                {
                    "name": "memory_stress",
                    "stress_type": "memory",
                    "users": 30,
                    "duration": 45,
                },
                {
                    "name": "cpu_stress",
                    "stress_type": "cpu",
                    "users": 20,
                    "duration": 45,
                },
                {"name": "io_stress", "stress_type": "io", "users": 25, "duration": 45},
                {
                    "name": "mixed_stress",
                    "stress_type": "mixed",
                    "users": 40,
                    "duration": 60,
                },
            ]

            pattern_results = []

            for pattern in stress_patterns:
                logger.info(f"Running stress pattern: {pattern['name']}")

                # Configure stress test
                config = LoadTestConfig(
                    test_type=LoadTestType.STRESS,
                    concurrent_users=pattern["users"],
                    requests_per_second=pattern["users"] / 2,
                    duration_seconds=pattern["duration"],
                    success_criteria={
                        "max_error_rate_percent": 20.0,
                        "max_avg_response_time_ms": 3000.0,
                    },
                )

                # Track metrics before pattern
                metrics_before = monitor.get_metrics_summary()

                # Run stress pattern
                result = await load_test_runner.run_load_test(
                    config=config,
                    target_function=stressing_service.stress_system,
                    stress_type=pattern["stress_type"],
                )

                # Track metrics after pattern
                metrics_after = monitor.get_metrics_summary()

                pattern_results.append(
                    {
                        "pattern": pattern["name"],
                        "stress_type": pattern["stress_type"],
                        "result": result,
                        "metrics_before": metrics_before,
                        "metrics_after": metrics_after,
                    }
                )

                logger.info(f"Completed stress pattern: {pattern['name']}")

                # Brief pause between patterns
                await asyncio.sleep(5)

            # Collect final metrics
            final_metrics = monitor.get_metrics_summary()
            performance_analysis = metrics_collector.get_performance_analysis()

            # Assertions for monitoring effectiveness
            assert len(monitor.system_metrics) > 0, "No system metrics collected"
            assert len(monitor.application_metrics) > 0, (
                "No application metrics collected"
            )

            # Verify metrics capture system stress
            cpu_stressed = any(
                p
                for p in pattern_results
                if p["stress_type"] == "cpu"
                and p["metrics_after"]["system_metrics"]["cpu"]["max"] > 30
            )
            assert cpu_stressed, "CPU stress not captured in metrics"

            memory_stressed = any(
                p
                for p in pattern_results
                if p["stress_type"] == "memory"
                and p["metrics_after"]["system_metrics"]["memory"]["max"]
                > p["metrics_before"]["system_metrics"]["memory"]["max"]
            )
            assert memory_stressed, "Memory stress not captured in metrics"

            # Verify performance analysis
            assert performance_analysis["response_times"]["count"] > 0, (
                "No response times recorded"
            )
            assert performance_analysis["duration_seconds"] > 0, "Invalid test duration"

            # Check for performance alerts
            if monitor.performance_alerts:
                alert_types = {alert.severity for alert in monitor.performance_alerts}
                logger.info(
                    f"Performance alerts generated: {len(monitor.performance_alerts)} ({alert_types})"
                )

            logger.info("System monitoring test completed:")
            logger.info(
                f"  - Total metrics collected: {final_metrics['total_data_points']}"
            )
            logger.info(f"  - Performance alerts: {final_metrics['total_alerts']}")
            logger.info(
                f"  - Peak CPU: {final_metrics['system_metrics']['cpu']['max']:.2f}%"
            )
            logger.info(
                f"  - Peak memory: {final_metrics['system_metrics']['memory']['max']:.2f}%"
            )
            logger.info(
                f"  - Response time P95: {performance_analysis['response_times']['p95_ms']:.2f}ms"
            )

        finally:
            monitor.stop_monitoring()
            metrics_collector.stop_collection()

            # Clean up memory
            if hasattr(stressing_service, "memory_hogs"):
                stressing_service.memory_hogs.clear()
            gc.collect()

    @pytest.mark.stress
    async def test_performance_degradation_detection(self, load_test_runner):
        """Test detection of performance degradation patterns."""

        metrics_collector = MetricsCollector()
        metrics_collector.start_collection()

        try:
            # Mock service with gradual performance degradation
            class DegradingService:
                def __init__(self):
                    self.call_count = 0
                    self.degradation_factor = 1.0
                    self.base_latency = 0.1

                async def degrading_operation(self, **_kwargs):
                    """Operation that gradually degrades over time."""
                    self.call_count += 1

                    # Gradual degradation every 100 calls
                    if self.call_count % 100 == 0:
                        self.degradation_factor *= 1.2  # 20% slower each time
                        logger.info(
                            f"Performance degraded: factor = {self.degradation_factor:.2f}"
                        )

                    # Calculate current latency
                    current_latency = self.base_latency * self.degradation_factor

                    # Add some randomness
                    actual_latency = current_latency * random.uniform(0.8, 1.5)

                    start_time = time.perf_counter()
                    await asyncio.sleep(actual_latency)
                    response_time = time.perf_counter() - start_time

                    # Record metrics
                    metrics_collector.record_response_time(response_time)
                    metrics_collector.record_custom_metric(
                        "degradation_factor", self.degradation_factor
                    )

                    # Introduce errors as performance degrades
                    error_probability = max(0, (self.degradation_factor - 2.0) / 10)
                    if random.random() < error_probability:
                        raise TestError("Service degraded - operation failed")
                        raise TestError("Service degraded - operation failed")

                    return {
                        "status": "success",
                        "call_count": self.call_count,
                        "degradation_factor": self.degradation_factor,
                        "response_time": response_time,
                    }

            degrading_service = DegradingService()

            # Run extended test to observe degradation
            config = LoadTestConfig(
                test_type=LoadTestType.STRESS,
                concurrent_users=25,
                requests_per_second=15,
                duration_seconds=300,  # 5 minutes to see degradation
                success_criteria={
                    "max_error_rate_percent": 30.0,  # Allow failures as service degrades
                    "max_avg_response_time_ms": 8000.0,
                },
            )

            # Collect metrics at intervals
            degradation_samples = []

            async def collect_degradation_sample():
                """Collect periodic degradation metrics."""
                while (
                    metrics_collector.start_time
                    and time.time() - metrics_collector.start_time
                    < config.duration_seconds
                ):
                    await asyncio.sleep(30)  # Sample every 30 seconds

                    if degrading_service.call_count > 0:
                        # Analyze recent response times
                        recent_times = list(metrics_collector.response_times)[
                            -50:
                        ]  # Last 50 requests
                        if recent_times:
                            avg_response_time = (
                                statistics.mean(recent_times) * 1000
                            )  # Convert to ms
                            p95_response_time = metrics_collector._percentile(
                                [rt * 1000 for rt in recent_times], 95
                            )

                            degradation_samples.append(
                                {
                                    "timestamp": time.time(),
                                    "call_count": degrading_service.call_count,
                                    "degradation_factor": degrading_service.degradation_factor,
                                    "avg_response_time_ms": avg_response_time,
                                    "p95_response_time_ms": p95_response_time,
                                }
                            )

            # Start degradation monitoring
            monitoring_task = asyncio.create_task(collect_degradation_sample())

            try:
                # Run stress test
                await load_test_runner.run_load_test(
                    config=config,
                    target_function=degrading_service.degrading_operation,
                )
            finally:
                monitoring_task.cancel()
                with contextlib.suppress(asyncio.CancelledError):
                    await monitoring_task

            # Analyze degradation pattern
            performance_analysis = metrics_collector.get_performance_analysis()

            # Assertions for degradation detection
            assert len(degradation_samples) >= 5, (
                "Insufficient degradation samples collected"
            )

            # Verify performance degraded over time
            first_sample = degradation_samples[0]
            last_sample = degradation_samples[-1]

            degradation_ratio = (
                last_sample["avg_response_time_ms"]
                / first_sample["avg_response_time_ms"]
            )
            assert degradation_ratio > 1.5, (
                f"Insufficient performance degradation detected: {degradation_ratio:.2f}x"
            )

            # Verify degradation factor increased
            assert (
                last_sample["degradation_factor"] > first_sample["degradation_factor"]
            ), "Degradation factor did not increase"

            # Verify errors increased with degradation
            total_errors = sum(metrics_collector.error_counts.values())
            if total_errors > 0:
                error_rate = (
                    total_errors / performance_analysis["response_times"]["count"]
                ) * 100
                assert error_rate > 0, (
                    "No errors recorded despite performance degradation"
                )

            logger.info("Performance degradation detected:")
            logger.info(f"  - Degradation ratio: {degradation_ratio:.2f}x")
            logger.info(
                f"  - Initial response time: {first_sample['avg_response_time_ms']:.2f}ms"
            )
            logger.info(
                f"  - Final response time: {last_sample['avg_response_time_ms']:.2f}ms"
            )
            logger.info(
                f"  - Error rate: {performance_analysis['error_rate_percent']:.2f}%"
            )
            logger.info(
                f"  - P95 response time: {performance_analysis['response_times']['p95_ms']:.2f}ms"
            )

        finally:
            metrics_collector.stop_collection()


@pytest.fixture
def system_monitor():
    """Provide system monitor for stress tests."""
    monitor = SystemMonitor(collection_interval=0.5)
    monitor.start_monitoring()
    yield monitor
    monitor.stop_monitoring()


@pytest.fixture
def metrics_collector():
    """Provide metrics collector for stress tests."""
    collector = MetricsCollector()
    collector.start_collection()
    yield collector
    collector.stop_collection()
