"""Resource exhaustion stress tests for AI Documentation Vector DB.

This module implements comprehensive resource exhaustion scenarios to test
system behavior under various resource constraints including memory, CPU,
network bandwidth, database connections, and file descriptors.
"""

import asyncio
import gc
import logging
import resource
import tempfile
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from contextlib import contextmanager
from dataclasses import dataclass, field
from pathlib import Path

import psutil
import pytest

from ..conftest import LoadTestConfig, LoadTestType


logger = logging.getLogger(__name__)


@dataclass
class ResourceConstraints:
    """Define resource constraints for testing."""

    max_memory_mb: int | None = None
    max_cpu_percent: float | None = None
    max_connections: int | None = None
    max_file_descriptors: int | None = None
    max_network_bandwidth_mbps: float | None = None
    timeout_seconds: int | None = None


@dataclass
class ResourceMetrics:
    """Track resource usage metrics during tests."""

    memory_usage_mb: list[float] = field(default_factory=list)
    cpu_usage_percent: list[float] = field(default_factory=list)
    connection_count: list[int] = field(default_factory=list)
    file_descriptor_count: list[int] = field(default_factory=list)
    network_io_mb: list[float] = field(default_factory=list)
    gc_collections: list[int] = field(default_factory=list)
    timestamps: list[float] = field(default_factory=list)


class ResourceMonitor:
    """Monitor system resource usage during stress tests."""

    def __init__(self, interval: float = 1.0):
        self.interval = interval
        self.metrics = ResourceMetrics()
        self.monitoring = False
        self.monitor_thread = None
        self.process = psutil.Process()

    def start_monitoring(self):
        """Start resource monitoring."""
        self.monitoring = True
        self.monitor_thread = threading.Thread(target=self._monitor_loop, daemon=True)
        self.monitor_thread.start()
        logger.info("Started resource monitoring")

    def stop_monitoring(self):
        """Stop resource monitoring."""
        self.monitoring = False
        if self.monitor_thread:
            self.monitor_thread.join(timeout=5.0)
        logger.info("Stopped resource monitoring")

    def _monitor_loop(self):
        """Main monitoring loop."""
        while self.monitoring:
            try:
                # Memory usage
                memory_info = self.process.memory_info()
                memory_mb = memory_info.rss / (1024 * 1024)
                self.metrics.memory_usage_mb.append(memory_mb)

                # CPU usage
                cpu_percent = self.process.cpu_percent()
                self.metrics.cpu_usage_percent.append(cpu_percent)

                # Connection count (approximate)
                try:
                    connections = len(self.process.connections())
                    self.metrics.connection_count.append(connections)
                except (psutil.AccessDenied, psutil.NoSuchProcess):
                    self.metrics.connection_count.append(0)

                # File descriptor count
                try:
                    fd_count = (
                        self.process.num_fds()
                        if hasattr(self.process, "num_fds")
                        else 0
                    )
                    self.metrics.file_descriptor_count.append(fd_count)
                except (psutil.AccessDenied, psutil.NoSuchProcess):
                    self.metrics.file_descriptor_count.append(0)

                # Network I/O
                try:
                    net_io = psutil.net_io_counters()
                    if net_io:
                        total_mb = (net_io.bytes_sent + net_io.bytes_recv) / (
                            1024 * 1024
                        )
                        self.metrics.network_io_mb.append(total_mb)
                    else:
                        self.metrics.network_io_mb.append(0)
                except (psutil.AccessDenied, AttributeError):
                    self.metrics.network_io_mb.append(0)

                # Garbage collection stats
                gc_stats = gc.get_stats()
                total_collections = sum(stat["collections"] for stat in gc_stats)
                self.metrics.gc_collections.append(total_collections)

                # Timestamp
                self.metrics.timestamps.append(time.time())

            except Exception as e:
                logger.warning(f"Error collecting resource metrics: {e}")

            time.sleep(self.interval)

    def get_peak_usage(self) -> dict[str, float]:
        """Get peak resource usage values."""
        return {
            "peak_memory_mb": max(self.metrics.memory_usage_mb)
            if self.metrics.memory_usage_mb
            else 0,
            "peak_cpu_percent": max(self.metrics.cpu_usage_percent)
            if self.metrics.cpu_usage_percent
            else 0,
            "peak_connections": max(self.metrics.connection_count)
            if self.metrics.connection_count
            else 0,
            "peak_file_descriptors": max(self.metrics.file_descriptor_count)
            if self.metrics.file_descriptor_count
            else 0,
            "total_network_io_mb": max(self.metrics.network_io_mb)
            if self.metrics.network_io_mb
            else 0,
        }


@contextmanager
def resource_limit(limit_type: str, limit_value: int):
    """Context manager to temporarily set resource limits."""
    if limit_type == "memory":
        # Set memory limit using resource module
        old_limit = resource.getrlimit(resource.RLIMIT_AS)
        new_limit = (limit_value * 1024 * 1024, old_limit[1])  # Convert MB to bytes
        try:
            resource.setrlimit(resource.RLIMIT_AS, new_limit)
            yield
        finally:
            resource.setrlimit(resource.RLIMIT_AS, old_limit)

    elif limit_type == "fds":
        # Set file descriptor limit
        old_limit = resource.getrlimit(resource.RLIMIT_NOFILE)
        new_limit = (limit_value, old_limit[1])
        try:
            resource.setrlimit(resource.RLIMIT_NOFILE, new_limit)
            yield
        finally:
            resource.setrlimit(resource.RLIMIT_NOFILE, old_limit)

    else:
        # No limit set, just yield
        yield


class TestResourceExhaustion:
    """Test suite for resource exhaustion scenarios."""

    @pytest.mark.stress
    @pytest.mark.slow
    async def test_memory_exhaustion_large_documents(self, load_test_runner):
        """Test system behavior when processing large documents that exhaust memory."""
        monitor = ResourceMonitor(interval=0.5)
        monitor.start_monitoring()

        try:
            # Create progressively larger "documents" to simulate memory pressure
            large_documents = []
            memory_exhaustion_detected = False

            async def process_large_document(size_mb: float = 10.0, **_kwargs):
                """Simulate processing of large documents."""
                nonlocal memory_exhaustion_detected

                try:
                    # Allocate large chunk of memory to simulate document processing
                    document_data = bytearray(int(size_mb * 1024 * 1024))
                    large_documents.append(document_data)

                    # Simulate processing time
                    await asyncio.sleep(0.1)

                    # Simulate memory-intensive operations
                    embeddings = [list(range(1000)) for _ in range(int(size_mb * 10))]

                    # Check memory usage
                    memory_info = psutil.Process().memory_info()
                    memory_mb = memory_info.rss / (1024 * 1024)

                    if memory_mb > 1024:  # Over 1GB
                        memory_exhaustion_detected = True
                        logger.warning(
                            f"High memory usage detected: {memory_mb:.2f} MB"
                        )

                    return {
                        "status": "processed",
                        "document_size_mb": size_mb,
                        "memory_used_mb": memory_mb,
                        "embeddings_count": len(embeddings),
                    }

                except MemoryError:
                    memory_exhaustion_detected = True
                    raise Exception("Memory exhausted during document processing")

            # Configure stress test with large document processing
            config = LoadTestConfig(
                test_type=LoadTestType.STRESS,
                concurrent_users=20,
                requests_per_second=5,
                duration_seconds=120,
                data_size_mb=50.0,  # Large documents
                success_criteria={
                    "max_error_rate_percent": 20.0,  # Allow some failures due to memory constraints
                    "max_avg_response_time_ms": 10000.0,  # Allow slower processing
                },
            )

            # Run stress test
            result = await load_test_runner.run_load_test(
                config=config,
                target_function=process_large_document,
                data_size_mb=config.data_size_mb,
            )

            # Analyze results
            peak_usage = monitor.get_peak_usage()

            # Assertions
            assert memory_exhaustion_detected, (
                "Memory exhaustion scenario was not triggered"
            )
            assert peak_usage["peak_memory_mb"] > 512, (
                f"Peak memory usage too low: {peak_usage['peak_memory_mb']} MB"
            )
            assert result.metrics.total_requests > 0, "No requests were processed"

            # Verify graceful degradation under memory pressure
            if result.metrics.failed_requests > 0:
                memory_errors = sum(
                    1 for e in result.metrics.errors if "memory" in e.lower()
                )
                assert memory_errors < result.metrics.total_requests * 0.8, (
                    "Too many memory-related failures"
                )

        finally:
            monitor.stop_monitoring()
            # Clean up memory
            large_documents.clear()
            gc.collect()

    @pytest.mark.stress
    async def test_cpu_saturation_parallel_embeddings(self, load_test_runner):
        """Test system behavior under CPU saturation from parallel embedding generation."""
        monitor = ResourceMonitor(interval=0.5)
        monitor.start_monitoring()

        try:
            cpu_saturation_detected = False

            async def cpu_intensive_embedding_generation(**_kwargs):
                """Simulate CPU-intensive embedding generation."""
                nonlocal cpu_saturation_detected

                # Simulate CPU-intensive work (embedding computation)
                def cpu_intensive_work():
                    # Simulate matrix operations for embedding generation
                    result = 0
                    for i in range(100000):
                        result += i**2
                        if i % 10000 == 0:
                            # Simulate floating point operations
                            result = result**0.5
                    return result

                # Run multiple CPU-intensive tasks in parallel
                with ThreadPoolExecutor(max_workers=4) as executor:
                    futures = [executor.submit(cpu_intensive_work) for _ in range(8)]
                    results = [future.result() for future in futures]

                # Check CPU usage
                cpu_percent = psutil.Process().cpu_percent()
                if cpu_percent > 80:
                    cpu_saturation_detected = True
                    logger.warning(f"High CPU usage detected: {cpu_percent:.2f}%")

                await asyncio.sleep(0.01)  # Small async yield

                return {
                    "status": "generated",
                    "embeddings_computed": len(results),
                    "cpu_usage": cpu_percent,
                }

            # Configure CPU stress test
            config = LoadTestConfig(
                test_type=LoadTestType.STRESS,
                concurrent_users=50,
                requests_per_second=25,
                duration_seconds=60,
                success_criteria={
                    "max_error_rate_percent": 10.0,
                    "max_avg_response_time_ms": 5000.0,
                },
            )

            # Run stress test
            result = await load_test_runner.run_load_test(
                config=config,
                target_function=cpu_intensive_embedding_generation,
            )

            # Analyze results
            peak_usage = monitor.get_peak_usage()

            # Assertions
            assert cpu_saturation_detected, "CPU saturation scenario was not triggered"
            assert peak_usage["peak_cpu_percent"] > 60, (
                f"Peak CPU usage too low: {peak_usage['peak_cpu_percent']}%"
            )
            assert result.metrics.throughput_rps > 0, "No throughput measured"

            # Verify system remained responsive under CPU load
            if result.metrics.response_times:
                p95_response_time = sorted(result.metrics.response_times)[
                    int(len(result.metrics.response_times) * 0.95)
                ]
                assert p95_response_time < 10.0, (
                    f"P95 response time too high under CPU load: {p95_response_time:.2f}s"
                )

        finally:
            monitor.stop_monitoring()

    @pytest.mark.stress
    async def test_network_bandwidth_stress(self, load_test_runner):
        """Test system behavior under network bandwidth saturation."""
        monitor = ResourceMonitor(interval=1.0)
        monitor.start_monitoring()

        try:
            network_stress_detected = False
            large_payload_responses = []

            async def network_intensive_operation(**kwargs):
                """Simulate network-intensive operations with large payloads."""
                nonlocal network_stress_detected

                # Simulate large data transfer (API responses, document downloads)
                payload_size_mb = kwargs.get("payload_size_mb", 5.0)
                large_payload = bytearray(int(payload_size_mb * 1024 * 1024))

                # Simulate network I/O delay
                network_delay = min(
                    payload_size_mb / 10, 2.0
                )  # Scale delay with payload size
                await asyncio.sleep(network_delay)

                large_payload_responses.append(large_payload)

                # Check if we've accumulated enough data to stress network
                total_data_mb = sum(len(p) for p in large_payload_responses) / (
                    1024 * 1024
                )
                if total_data_mb > 100:  # 100MB total
                    network_stress_detected = True
                    logger.warning(
                        f"Network stress detected: {total_data_mb:.2f} MB transferred"
                    )

                return {
                    "status": "transferred",
                    "payload_size_mb": payload_size_mb,
                    "total_transferred_mb": total_data_mb,
                    "network_delay": network_delay,
                }

            # Configure network stress test
            config = LoadTestConfig(
                test_type=LoadTestType.STRESS,
                concurrent_users=30,
                requests_per_second=10,
                duration_seconds=90,
                success_criteria={
                    "max_error_rate_percent": 15.0,
                    "max_avg_response_time_ms": 8000.0,
                },
            )

            # Run stress test
            result = await load_test_runner.run_load_test(
                config=config,
                target_function=network_intensive_operation,
                payload_size_mb=5.0,
            )

            # Analyze results
            monitor.get_peak_usage()

            # Assertions
            assert network_stress_detected, "Network stress scenario was not triggered"
            assert result.metrics.total_requests > 0, "No requests were processed"

            # Verify network handling under stress
            total_data_processed = sum(len(p) for p in large_payload_responses) / (
                1024 * 1024
            )
            assert total_data_processed > 50, (
                f"Insufficient data processed: {total_data_processed:.2f} MB"
            )

        finally:
            monitor.stop_monitoring()
            # Clean up network data
            large_payload_responses.clear()
            gc.collect()

    @pytest.mark.stress
    async def test_database_connection_pool_exhaustion(self, load_test_runner):
        """Test behavior when database connection pool is exhausted."""
        monitor = ResourceMonitor(interval=0.5)
        monitor.start_monitoring()

        try:
            # Simulate database connection pool
            class MockConnectionPool:
                def __init__(self, max_connections: int = 10):
                    self.max_connections = max_connections
                    self.active_connections = 0
                    self.connection_requests = 0
                    self.pool_exhausted_count = 0
                    self._lock = asyncio.Lock()

                async def get_connection(self):
                    async with self._lock:
                        self.connection_requests += 1
                        if self.active_connections >= self.max_connections:
                            self.pool_exhausted_count += 1
                            raise Exception("Connection pool exhausted")

                        self.active_connections += 1
                        return f"connection_{self.active_connections}"

                async def release_connection(self, _connection: str):
                    async with self._lock:
                        self.active_connections = max(0, self.active_connections - 1)

            pool = MockConnectionPool(max_connections=5)  # Small pool for testing
            pool_exhaustion_detected = False

            async def database_intensive_operation(**kwargs):
                """Simulate database-intensive operations that require connections."""
                nonlocal pool_exhaustion_detected

                try:
                    # Get database connection
                    connection = await pool.get_connection()

                    # Simulate database work (hold connection longer under stress)
                    work_duration = kwargs.get("db_work_duration", 0.5)
                    await asyncio.sleep(work_duration)

                    # Release connection
                    await pool.release_connection(connection)

                    return {
                        "status": "db_operation_complete",
                        "connection": connection,
                        "active_connections": pool.active_connections,
                    }

                except Exception as e:
                    if "pool exhausted" in str(e).lower():
                        pool_exhaustion_detected = True
                        logger.warning("Database connection pool exhausted")
                    raise

            # Configure database stress test
            config = LoadTestConfig(
                test_type=LoadTestType.STRESS,
                concurrent_users=20,  # More users than pool connections
                requests_per_second=15,
                duration_seconds=60,
                success_criteria={
                    "max_error_rate_percent": 30.0,  # Allow failures due to pool exhaustion
                    "max_avg_response_time_ms": 3000.0,
                },
            )

            # Run stress test
            result = await load_test_runner.run_load_test(
                config=config,
                target_function=database_intensive_operation,
                db_work_duration=1.0,  # Long-running DB operations
            )

            # Analyze results
            monitor.get_peak_usage()

            # Assertions
            assert pool_exhaustion_detected, (
                "Connection pool exhaustion was not triggered"
            )
            assert pool.pool_exhausted_count > 0, "No pool exhaustion events detected"
            assert pool.connection_requests > pool.max_connections, (
                "Not enough connection requests to stress pool"
            )

            # Verify graceful handling of pool exhaustion
            pool_errors = sum(
                1 for e in result.metrics.errors if "pool exhausted" in e.lower()
            )
            total_requests = result.metrics.total_requests
            if total_requests > 0:
                pool_error_rate = pool_errors / total_requests
                assert pool_error_rate < 0.8, (
                    f"Too many pool exhaustion errors: {pool_error_rate:.2%}"
                )

        finally:
            monitor.stop_monitoring()

    @pytest.mark.stress
    async def test_file_descriptor_exhaustion(self, load_test_runner):
        """Test behavior when file descriptor limits are reached."""
        monitor = ResourceMonitor(interval=0.5)
        monitor.start_monitoring()

        try:
            # Set low file descriptor limit for testing
            with resource_limit("fds", 100):  # Very low limit
                fd_exhaustion_detected = False
                open_files = []

                async def file_intensive_operation(**_kwargs):
                    """Simulate operations that create many file descriptors."""
                    nonlocal fd_exhaustion_detected

                    try:
                        # Create temporary files to consume file descriptors
                        for _i in range(5):  # Try to open multiple files per request
                            temp_file = tempfile.NamedTemporaryFile(delete=False)
                            open_files.append(temp_file)

                            # Write some data
                            temp_file.write(
                                b"test data for file descriptor stress test"
                            )
                            temp_file.flush()

                        # Simulate file processing
                        await asyncio.sleep(0.1)

                        return {
                            "status": "files_processed",
                            "files_opened": len(open_files),
                            "current_fds": len(open_files),
                        }

                    except OSError as e:
                        if "too many open files" in str(e).lower() or e.errno == 24:
                            fd_exhaustion_detected = True
                            logger.warning("File descriptor limit reached")
                        raise Exception(f"File descriptor exhaustion: {e}")

                    except Exception as e:
                        logger.exception(f"Unexpected error in file operation: {e}")
                        raise

                # Configure file descriptor stress test
                config = LoadTestConfig(
                    test_type=LoadTestType.STRESS,
                    concurrent_users=15,
                    requests_per_second=10,
                    duration_seconds=30,  # Shorter test due to resource constraints
                    success_criteria={
                        "max_error_rate_percent": 50.0,  # Allow many failures due to FD limits
                        "max_avg_response_time_ms": 2000.0,
                    },
                )

                try:
                    # Run stress test
                    result = await load_test_runner.run_load_test(
                        config=config,
                        target_function=file_intensive_operation,
                    )

                    # Analyze results
                    peak_usage = monitor.get_peak_usage()

                    # Assertions
                    assert (
                        fd_exhaustion_detected
                        or peak_usage["peak_file_descriptors"] > 50
                    ), "File descriptor stress scenario was not triggered sufficiently"

                    # Verify system handling of FD exhaustion
                    fd_errors = sum(
                        1
                        for e in result.metrics.errors
                        if "file descriptor" in e.lower()
                    )
                    if result.metrics.total_requests > 0:
                        assert fd_errors > 0, (
                            "Expected some file descriptor exhaustion errors"
                        )

                finally:
                    # Clean up open files
                    for temp_file in open_files:
                        try:
                            temp_file.close()
                            Path(temp_file.name).unlink()
                        except (OSError, AttributeError):
                            pass
                    open_files.clear()

        finally:
            monitor.stop_monitoring()

    @pytest.mark.stress
    async def test_cascading_resource_failure(self, load_test_runner):
        """Test behavior when multiple resources fail simultaneously."""
        monitor = ResourceMonitor(interval=0.5)
        monitor.start_monitoring()

        try:
            # Track different failure types
            failure_types = {
                "memory": 0,
                "cpu": 0,
                "network": 0,
                "database": 0,
            }

            cascading_failure_detected = False
            memory_hogs = []

            async def multi_resource_operation(**_kwargs):
                """Operation that stresses multiple resources simultaneously."""
                nonlocal cascading_failure_detected

                try:
                    # Memory pressure
                    memory_chunk = bytearray(10 * 1024 * 1024)  # 10MB
                    memory_hogs.append(memory_chunk)

                    # CPU intensive work
                    def cpu_work():
                        return sum(i**2 for i in range(50000))

                    # Run CPU work in thread
                    with ThreadPoolExecutor(max_workers=2) as executor:
                        cpu_future = executor.submit(cpu_work)
                        cpu_result = cpu_future.result(timeout=2.0)

                    # Network simulation (large data processing)
                    network_data = bytearray(5 * 1024 * 1024)  # 5MB
                    await asyncio.sleep(0.2)  # Simulate network delay

                    # Database simulation (connection holding)
                    await asyncio.sleep(0.3)  # Simulate DB query

                    # Check resource usage
                    process = psutil.Process()
                    memory_mb = process.memory_info().rss / (1024 * 1024)
                    cpu_percent = process.cpu_percent()

                    # Detect potential failures
                    active_failures = []
                    if memory_mb > 500:  # High memory usage
                        failure_types["memory"] += 1
                        active_failures.append("memory")
                    if cpu_percent > 70:  # High CPU usage
                        failure_types["cpu"] += 1
                        active_failures.append("cpu")
                    if len(network_data) > 0:  # Network stress
                        failure_types["network"] += 1
                        active_failures.append("network")

                    # Check for cascading failure (multiple resources stressed)
                    if len(active_failures) >= 2:
                        cascading_failure_detected = True
                        logger.warning(f"Cascading failure detected: {active_failures}")
                        # Simulate system instability
                        if len(active_failures) >= 3:
                            raise Exception(
                                f"System overload: {', '.join(active_failures)}"
                            )

                    return {
                        "status": "multi_resource_complete",
                        "memory_mb": memory_mb,
                        "cpu_percent": cpu_percent,
                        "active_failures": active_failures,
                        "cpu_result": cpu_result,
                    }

                except TimeoutError:
                    failure_types["cpu"] += 1
                    raise Exception("CPU timeout during multi-resource operation")
                except MemoryError:
                    failure_types["memory"] += 1
                    raise Exception("Memory exhausted during multi-resource operation")
                except Exception as e:
                    # Count the failure type
                    error_msg = str(e).lower()
                    if "memory" in error_msg:
                        failure_types["memory"] += 1
                    elif "cpu" in error_msg or "timeout" in error_msg:
                        failure_types["cpu"] += 1
                    elif "network" in error_msg:
                        failure_types["network"] += 1
                    else:
                        failure_types["database"] += 1
                    raise

            # Configure multi-resource stress test
            config = LoadTestConfig(
                test_type=LoadTestType.STRESS,
                concurrent_users=25,
                requests_per_second=12,
                duration_seconds=90,
                success_criteria={
                    "max_error_rate_percent": 40.0,  # Allow failures from resource exhaustion
                    "max_avg_response_time_ms": 8000.0,
                },
            )

            # Run stress test
            await load_test_runner.run_load_test(
                config=config,
                target_function=multi_resource_operation,
            )

            # Analyze results
            peak_usage = monitor.get_peak_usage()

            # Assertions
            assert cascading_failure_detected, (
                "Cascading failure scenario was not triggered"
            )

            # Verify multiple resource types were stressed
            active_failure_types = sum(
                1 for count in failure_types.values() if count > 0
            )
            assert active_failure_types >= 2, (
                f"Not enough resource types stressed: {failure_types}"
            )

            # Verify peak resource usage
            assert peak_usage["peak_memory_mb"] > 100, "Insufficient memory stress"
            assert peak_usage["peak_cpu_percent"] > 30, "Insufficient CPU stress"

            # Verify system behavior under cascading failure
            total_failures = sum(failure_types.values())
            assert total_failures > 0, "No resource failures detected"
            logger.info(f"Resource failure distribution: {failure_types}")

        finally:
            monitor.stop_monitoring()
            # Clean up memory
            memory_hogs.clear()
            gc.collect()


@pytest.fixture
def resource_monitor():
    """Provide resource monitor for stress tests."""
    monitor = ResourceMonitor(interval=0.5)
    monitor.start_monitoring()
    yield monitor
    monitor.stop_monitoring()


@pytest.fixture
def resource_constraints():
    """Provide common resource constraints for testing."""
    return ResourceConstraints(
        max_memory_mb=1024,
        max_cpu_percent=80.0,
        max_connections=20,
        max_file_descriptors=100,
        timeout_seconds=30,
    )
