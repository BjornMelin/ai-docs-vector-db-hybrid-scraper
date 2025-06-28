"""Performance profiler for detailed system resource monitoring.

This module provides comprehensive performance profiling capabilities
including CPU, memory, and I/O monitoring during benchmark execution.
"""

import asyncio
import logging
import time
from dataclasses import dataclass
from datetime import UTC, datetime
from typing import Any

import psutil
from pydantic import BaseModel, Field

from src.models.vector_search import HybridSearchRequest
from src.services.vector_db.hybrid_search import AdvancedHybridSearchService


logger = logging.getLogger(__name__)


@dataclass
class ResourceSnapshot:
    """Single point-in-time resource measurement."""

    timestamp: float
    cpu_percent: float
    memory_mb: float
    memory_percent: float
    disk_io_read_mb: float
    disk_io_write_mb: float
    network_sent_mb: float
    network_recv_mb: float
    active_threads: int


class ProfilingResults(BaseModel):
    """Results from performance profiling session."""

    duration_seconds: float = Field(..., description="Total profiling duration")
    start_time: datetime = Field(..., description="Profiling start time")
    end_time: datetime = Field(..., description="Profiling end time")

    # CPU metrics
    avg_cpu_percent: float = Field(..., description="Average CPU utilization")
    max_cpu_percent: float = Field(..., description="Peak CPU utilization")
    cpu_over_80_percent: float = Field(..., description="Time with CPU > 80%")

    # Memory metrics
    avg_memory_mb: float = Field(..., description="Average memory usage in MB")
    peak_memory_mb: float = Field(..., description="Peak memory usage in MB")
    memory_growth_mb: float = Field(..., description="Total memory growth in MB")
    memory_leak_detected: bool = Field(
        False, description="Whether memory leak detected"
    )

    # I/O metrics
    total_disk_read_mb: float = Field(..., description="Total disk reads in MB")
    total_disk_write_mb: float = Field(..., description="Total disk writes in MB")
    total_network_sent_mb: float = Field(..., description="Total network sent in MB")
    total_network_recv_mb: float = Field(
        ..., description="Total network received in MB"
    )

    # Threading metrics
    avg_thread_count: float = Field(..., description="Average active thread count")
    max_thread_count: int = Field(..., description="Maximum thread count")

    # Performance insights
    bottleneck_periods: list[dict[str, Any]] = Field(
        default_factory=list, description="Identified bottleneck periods"
    )
    resource_warnings: list[str] = Field(
        default_factory=list, description="Resource usage warnings"
    )
    optimization_suggestions: list[str] = Field(
        default_factory=list, description="Performance optimization suggestions"
    )


class PerformanceProfiler:
    """Advanced performance profiler for search service monitoring."""

    def __init__(self, sampling_interval: float = 0.5):
        """Initialize performance profiler.

        Args:
            sampling_interval: Time between resource samples in seconds

        """
        self.sampling_interval = sampling_interval
        self.resource_snapshots: list[ResourceSnapshot] = []
        self.profiling_active = False

        # Get baseline system info
        self.process = psutil.Process()
        self.baseline_memory = self.process.memory_info().rss / 1024 / 1024  # MB

    async def profile_search_service(
        self,
        search_service: AdvancedHybridSearchService,
        test_queries: list[HybridSearchRequest],
    ) -> dict[str, Any]:
        """Profile search service performance during query execution.

        Args:
            search_service: Search service to profile
            test_queries: Test queries to execute during profiling

        Returns:
            Comprehensive profiling results

        """
        logger.info("Starting performance profiling session")

        # Start resource monitoring
        profiling_task = asyncio.create_task(self._monitor_resources())

        start_time = time.time()
        start_datetime = datetime.now(tz=UTC)

        try:
            # Execute test queries while monitoring
            await self._execute_profiled_queries(search_service, test_queries)

        finally:
            # Stop monitoring
            self.profiling_active = False
            await profiling_task

        end_time = time.time()
        end_datetime = datetime.now(tz=UTC)
        duration = end_time - start_time

        # Analyze results
        results = self._analyze_profiling_results(
            duration, start_datetime, end_datetime
        )

        logger.info(
            f"Profiling completed. Duration: {duration:.2f}s, Peak memory: {results.peak_memory_mb:.1f}MB"
        )

        return {
            "profiling_results": results,
            "resource_metrics": {
                "peak_memory_mb": results.peak_memory_mb,
                "avg_cpu_percent": results.avg_cpu_percent,
                "memory_growth_mb": results.memory_growth_mb,
            },
            "raw_snapshots": len(self.resource_snapshots),
            "bottleneck_count": len(results.bottleneck_periods),
        }

    async def _execute_profiled_queries(
        self,
        search_service: AdvancedHybridSearchService,
        test_queries: list[HybridSearchRequest],
    ) -> None:
        """Execute queries while profiling is active."""
        # Execute a subset of queries for profiling
        profiling_queries = test_queries[: min(20, len(test_queries))]

        for i, query in enumerate(profiling_queries):
            try:
                # Add some variety to the load
                if i % 3 == 0:
                    # Concurrent execution for some queries
                    tasks = [search_service.hybrid_search(query) for _ in range(2)]
                    await asyncio.gather(*tasks, return_exceptions=True)
                else:
                    # Sequential execution
                    await search_service.hybrid_search(query)

                # Small delay between queries
                await asyncio.sleep(0.1)

            except Exception as e:
                logger.debug(f"Query execution failed during profiling: {e}")  # TODO: Convert f-string to logging format

    async def _monitor_resources(self) -> None:
        """Monitor system resources continuously."""
        self.profiling_active = True
        self.resource_snapshots = []

        # Get initial I/O counters
        initial_io = psutil.disk_io_counters()
        initial_net = psutil.net_io_counters()

        while self.profiling_active:
            try:
                # CPU and memory
                cpu_percent = self.process.cpu_percent()
                memory_info = self.process.memory_info()
                memory_mb = memory_info.rss / 1024 / 1024

                # System memory
                system_memory = psutil.virtual_memory()
                memory_percent = system_memory.percent

                # I/O counters
                current_io = psutil.disk_io_counters()
                current_net = psutil.net_io_counters()

                disk_read_mb = (
                    (current_io.read_bytes - initial_io.read_bytes) / 1024 / 1024
                )
                disk_write_mb = (
                    (current_io.write_bytes - initial_io.write_bytes) / 1024 / 1024
                )
                net_sent_mb = (
                    (current_net.bytes_sent - initial_net.bytes_sent) / 1024 / 1024
                )
                net_recv_mb = (
                    (current_net.bytes_recv - initial_net.bytes_recv) / 1024 / 1024
                )

                # Thread count
                thread_count = self.process.num_threads()

                snapshot = ResourceSnapshot(
                    timestamp=time.time(),
                    cpu_percent=cpu_percent,
                    memory_mb=memory_mb,
                    memory_percent=memory_percent,
                    disk_io_read_mb=disk_read_mb,
                    disk_io_write_mb=disk_write_mb,
                    network_sent_mb=net_sent_mb,
                    network_recv_mb=net_recv_mb,
                    active_threads=thread_count,
                )

                self.resource_snapshots.append(snapshot)

            except Exception as e:
                logger.warning(f"Error collecting resource snapshot: {e}")  # TODO: Convert f-string to logging format

            await asyncio.sleep(self.sampling_interval)

    def _analyze_profiling_results(
        self, duration: float, start_time: datetime, end_time: datetime
    ) -> ProfilingResults:
        """Analyze collected resource snapshots."""
        if not self.resource_snapshots:
            return ProfilingResults(
                duration_seconds=duration,
                start_time=start_time,
                end_time=end_time,
                avg_cpu_percent=0,
                max_cpu_percent=0,
                cpu_over_80_percent=0,
                avg_memory_mb=self.baseline_memory,
                peak_memory_mb=self.baseline_memory,
                memory_growth_mb=0,
                total_disk_read_mb=0,
                total_disk_write_mb=0,
                total_network_sent_mb=0,
                total_network_recv_mb=0,
                avg_thread_count=1,
                max_thread_count=1,
            )

        # Calculate metrics
        cpu_values = [s.cpu_percent for s in self.resource_snapshots]
        memory_values = [s.memory_mb for s in self.resource_snapshots]
        thread_values = [s.active_threads for s in self.resource_snapshots]

        # CPU analysis
        avg_cpu = sum(cpu_values) / len(cpu_values)
        max_cpu = max(cpu_values)
        high_cpu_samples = len([c for c in cpu_values if c > 80])
        cpu_over_80_time = (high_cpu_samples / len(cpu_values)) * duration

        # Memory analysis
        avg_memory = sum(memory_values) / len(memory_values)
        peak_memory = max(memory_values)
        initial_memory = memory_values[0] if memory_values else self.baseline_memory
        final_memory = memory_values[-1] if memory_values else self.baseline_memory
        memory_growth = final_memory - initial_memory

        # Detect memory leak (steady growth over time)
        memory_leak = self._detect_memory_leak(memory_values)

        # I/O totals
        final_snapshot = self.resource_snapshots[-1]

        # Threading
        avg_threads = sum(thread_values) / len(thread_values)
        max_threads = max(thread_values)

        # Identify bottlenecks
        bottlenecks = self._identify_bottlenecks()

        # Generate warnings and suggestions
        warnings = self._generate_warnings(avg_cpu, peak_memory, memory_growth)
        suggestions = self._generate_optimization_suggestions(
            avg_cpu, peak_memory, bottlenecks
        )

        return ProfilingResults(
            duration_seconds=duration,
            start_time=start_time,
            end_time=end_time,
            avg_cpu_percent=avg_cpu,
            max_cpu_percent=max_cpu,
            cpu_over_80_percent=cpu_over_80_time,
            avg_memory_mb=avg_memory,
            peak_memory_mb=peak_memory,
            memory_growth_mb=memory_growth,
            memory_leak_detected=memory_leak,
            total_disk_read_mb=final_snapshot.disk_io_read_mb,
            total_disk_write_mb=final_snapshot.disk_io_write_mb,
            total_network_sent_mb=final_snapshot.network_sent_mb,
            total_network_recv_mb=final_snapshot.network_recv_mb,
            avg_thread_count=avg_threads,
            max_thread_count=max_threads,
            bottleneck_periods=bottlenecks,
            resource_warnings=warnings,
            optimization_suggestions=suggestions,
        )

    def _detect_memory_leak(self, memory_values: list[float]) -> bool:
        """Detect potential memory leaks based on memory growth pattern."""
        if len(memory_values) < 10:
            return False

        # Check for consistent upward trend
        quarter_size = len(memory_values) // 4
        first_quarter_avg = sum(memory_values[:quarter_size]) / quarter_size
        last_quarter_avg = sum(memory_values[-quarter_size:]) / quarter_size

        growth_rate = (last_quarter_avg - first_quarter_avg) / first_quarter_avg
        return growth_rate > 0.1  # 10% growth suggests potential leak

    def _identify_bottlenecks(self) -> list[dict[str, Any]]:
        """Identify performance bottleneck periods."""
        bottlenecks = []

        if len(self.resource_snapshots) < 5:
            return bottlenecks

        # CPU bottlenecks (> 90% for sustained period)
        cpu_bottleneck_start = None
        for i, snapshot in enumerate(self.resource_snapshots):
            if snapshot.cpu_percent > 90:
                if cpu_bottleneck_start is None:
                    cpu_bottleneck_start = i
            elif cpu_bottleneck_start is not None:
                duration = (i - cpu_bottleneck_start) * self.sampling_interval
                if duration > 2.0:  # At least 2 seconds
                    bottlenecks.append(
                        {
                            "type": "cpu",
                            "start_time": self.resource_snapshots[
                                cpu_bottleneck_start
                            ].timestamp,
                            "duration_seconds": duration,
                            "severity": "high",
                        }
                    )
                cpu_bottleneck_start = None

        # Memory pressure bottlenecks (> 85% system memory)
        memory_bottleneck_start = None
        for i, snapshot in enumerate(self.resource_snapshots):
            if snapshot.memory_percent > 85:
                if memory_bottleneck_start is None:
                    memory_bottleneck_start = i
            elif memory_bottleneck_start is not None:
                duration = (i - memory_bottleneck_start) * self.sampling_interval
                if duration > 1.0:  # At least 1 second
                    bottlenecks.append(
                        {
                            "type": "memory",
                            "start_time": self.resource_snapshots[
                                memory_bottleneck_start
                            ].timestamp,
                            "duration_seconds": duration,
                            "severity": "high",
                        }
                    )
                memory_bottleneck_start = None

        return bottlenecks

    def _generate_warnings(
        self, avg_cpu: float, peak_memory: float, memory_growth: float
    ) -> list[str]:
        """Generate resource usage warnings."""
        warnings = []

        if avg_cpu > 70:
            warnings.append(f"High average CPU usage: {avg_cpu:.1f}%")

        if peak_memory > 4000:  # 4GB
            warnings.append(f"High memory usage: {peak_memory:.1f}MB")

        if memory_growth > 500:  # 500MB growth
            warnings.append(f"Significant memory growth: {memory_growth:.1f}MB")

        if len(self.resource_snapshots) > 0:
            max_threads = max(s.active_threads for s in self.resource_snapshots)
            if max_threads > 50:
                warnings.append(f"High thread count: {max_threads} threads")

        return warnings

    def _generate_optimization_suggestions(
        self, avg_cpu: float, peak_memory: float, bottlenecks: list[dict[str, Any]]
    ) -> list[str]:
        """Generate performance optimization suggestions."""
        suggestions = []

        if avg_cpu > 60:
            suggestions.append(
                "Consider implementing request queuing to smooth CPU load"
            )
            suggestions.append("Profile individual components to identify CPU hotspots")

        if peak_memory > 2000:  # 2GB
            suggestions.append(
                "Implement memory pooling for frequently allocated objects"
            )
            suggestions.append("Consider lazy loading for large ML models")

        cpu_bottlenecks = [b for b in bottlenecks if b["type"] == "cpu"]
        if len(cpu_bottlenecks) > 0:
            suggestions.append("Add async processing to reduce CPU blocking")
            suggestions.append(
                "Consider horizontal scaling for CPU-intensive operations"
            )

        memory_bottlenecks = [b for b in bottlenecks if b["type"] == "memory"]
        if len(memory_bottlenecks) > 0:
            suggestions.append("Implement LRU caching with memory limits")
            suggestions.append("Add memory monitoring and garbage collection tuning")

        if not suggestions:
            suggestions.append(
                "Performance profile looks good - consider load testing at higher concurrency"
            )

        return suggestions

    def get_resource_timeline(self) -> list[dict[str, Any]]:
        """Get detailed resource usage timeline."""
        return [
            {
                "timestamp": snapshot.timestamp,
                "cpu_percent": snapshot.cpu_percent,
                "memory_mb": snapshot.memory_mb,
                "memory_percent": snapshot.memory_percent,
                "threads": snapshot.active_threads,
                "disk_read_mb": snapshot.disk_io_read_mb,
                "disk_write_mb": snapshot.disk_io_write_mb,
            }
            for snapshot in self.resource_snapshots
        ]

    def clear_snapshots(self) -> None:
        """Clear collected resource snapshots."""
        self.resource_snapshots = []