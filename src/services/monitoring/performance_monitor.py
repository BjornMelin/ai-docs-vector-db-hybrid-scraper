"""Real-time performance monitoring for optimization insights.

This module provides comprehensive real-time performance monitoring with automatic
optimization triggers, memory management, and detailed performance analytics.
"""

import asyncio
import gc
import logging
import time
from dataclasses import asdict, dataclass
from datetime import UTC, datetime, timedelta, timezone
from typing import Any, Dict, List, Optional, Tuple

import psutil


logger = logging.getLogger(__name__)


@dataclass
class PerformanceSnapshot:
    """Performance snapshot containing system and application metrics."""

    timestamp: datetime
    cpu_percent: float
    memory_percent: float
    memory_mb: float
    active_connections: int
    request_rate: float
    avg_response_time: float
    p95_response_time: float
    cache_hit_rate: float
    gc_collections: int
    gc_time_ms: float


class RealTimePerformanceMonitor:
    """Real-time performance monitoring for optimization insights."""

    def __init__(self, window_size: int = 60):
        """Initialize the performance monitor.

        Args:
            window_size: Time window in seconds for metric calculations

        """
        self.window_size = window_size
        self.snapshots: list[PerformanceSnapshot] = []
        self.request_times: list[float] = []
        self.last_gc_time = time.time()
        self.monitoring_active = False
        self.gc_stats_start = self._get_gc_stats()

    async def start_monitoring(self) -> None:
        """Start continuous performance monitoring."""
        if self.monitoring_active:
            logger.warning("Performance monitoring already active")
            return

        self.monitoring_active = True
        logger.info("Starting real-time performance monitoring")

        try:
            while self.monitoring_active:
                snapshot = await self._take_snapshot()
                self.snapshots.append(snapshot)

                # Keep only recent snapshots
                cutoff_time = datetime.now(tz=UTC) - timedelta(
                    seconds=self.window_size * 10
                )
                self.snapshots = [
                    s for s in self.snapshots if s.timestamp > cutoff_time
                ]

                # Trigger optimization if needed
                await self._check_optimization_triggers(snapshot)

                await asyncio.sleep(1)  # Take snapshot every second

        except asyncio.CancelledError:
            logger.info("Performance monitoring cancelled")
        except Exception as e:
            logger.exception("Performance monitoring error")
        finally:
            self.monitoring_active = False

    async def stop_monitoring(self) -> None:
        """Stop performance monitoring."""
        self.monitoring_active = False
        logger.info("Performance monitoring stopped")

    async def _take_snapshot(self) -> PerformanceSnapshot:
        """Take comprehensive performance snapshot.

        Returns:
            PerformanceSnapshot containing current performance metrics

        """
        # System metrics
        cpu_percent = psutil.cpu_percent()
        memory = psutil.virtual_memory()

        # Application metrics
        current_time = time.time()
        recent_requests = [
            t for t in self.request_times if current_time - t < self.window_size
        ]

        request_rate = len(recent_requests) / self.window_size

        # Response time metrics
        recent_response_times = recent_requests[-100:]  # Last 100 requests
        avg_response_time = (
            sum(recent_response_times) / len(recent_response_times)
            if recent_response_times
            else 0
        )
        p95_response_time = self._calculate_percentile(recent_response_times, 95)

        # Garbage collection metrics
        current_gc_stats = self._get_gc_stats()
        gc_collections = sum(current_gc_stats) - sum(self.gc_stats_start)
        gc_time_ms = self._estimate_gc_time()

        return PerformanceSnapshot(
            timestamp=datetime.now(tz=UTC),
            cpu_percent=cpu_percent,
            memory_percent=memory.percent,
            memory_mb=memory.used / 1024 / 1024,
            active_connections=self._get_active_connections(),
            request_rate=request_rate,
            avg_response_time=avg_response_time,
            p95_response_time=p95_response_time,
            cache_hit_rate=self._get_cache_hit_rate(),
            gc_collections=gc_collections,
            gc_time_ms=gc_time_ms,
        )

    async def _check_optimization_triggers(self, snapshot: PerformanceSnapshot) -> None:
        """Check if optimization should be triggered based on current metrics.

        Args:
            snapshot: Current performance snapshot

        """
        # Trigger memory optimization if usage is high
        if snapshot.memory_percent > 80:
            await self._optimize_memory()

        # Log performance warnings
        if snapshot.p95_response_time > 100:  # P95 > 100ms
            logger.warning(
                f"High P95 latency detected: {snapshot.p95_response_time:.1f}ms"
            )

        if snapshot.request_rate > 0 and snapshot.request_rate < 10:  # Low throughput
            logger.warning(
                f"Low throughput detected: {snapshot.request_rate:.1f} req/s"
            )

        if snapshot.cache_hit_rate < 0.8:  # Cache hit rate below 80%
            logger.warning(f"Low cache hit rate: {snapshot.cache_hit_rate:.1%}")

    async def _optimize_memory(self) -> None:
        """Trigger memory optimization when usage is high."""
        current_time = time.time()

        # Only run GC every 30 seconds to avoid performance impact
        if current_time - self.last_gc_time > 30:
            logger.info("Triggering garbage collection due to high memory usage")

            # Force garbage collection
            gc.collect()
            self.last_gc_time = current_time

            # Clear old request times to reduce memory usage
            cutoff = current_time - self.window_size
            self.request_times = [t for t in self.request_times if t > cutoff]

    def record_request(self, response_time: float) -> None:
        """Record request completion for metrics.

        Args:
            response_time: Request response time in seconds

        """
        self.request_times.append(time.time())

    def get_performance_summary(self) -> dict[str, Any]:
        """Get current performance summary.

        Returns:
            dict containing current performance metrics

        """
        if not self.snapshots:
            return {"status": "no_data"}

        latest = self.snapshots[-1]
        return asdict(latest)

    def get_performance_trends(self, minutes: int = 5) -> dict[str, Any]:
        """Get performance trends over specified time period.

        Args:
            minutes: Number of minutes to analyze

        Returns:
            dict containing performance trend analysis

        """
        if not self.snapshots:
            return {"status": "no_data"}

        # Filter snapshots to specified time window
        cutoff_time = datetime.now(tz=UTC) - timedelta(minutes=minutes)
        recent_snapshots = [s for s in self.snapshots if s.timestamp > cutoff_time]

        if len(recent_snapshots) < 2:
            return {"status": "insufficient_data"}

        # Calculate trends
        cpu_values = [s.cpu_percent for s in recent_snapshots]
        memory_values = [s.memory_percent for s in recent_snapshots]
        response_time_values = [s.avg_response_time for s in recent_snapshots]
        throughput_values = [s.request_rate for s in recent_snapshots]

        return {
            "time_window_minutes": minutes,
            "sample_count": len(recent_snapshots),
            "trends": {
                "cpu": {
                    "current": cpu_values[-1],
                    "min": min(cpu_values),
                    "max": max(cpu_values),
                    "avg": sum(cpu_values) / len(cpu_values),
                    "trend": self._calculate_trend(cpu_values),
                },
                "memory": {
                    "current": memory_values[-1],
                    "min": min(memory_values),
                    "max": max(memory_values),
                    "avg": sum(memory_values) / len(memory_values),
                    "trend": self._calculate_trend(memory_values),
                },
                "response_time": {
                    "current": response_time_values[-1],
                    "min": min(response_time_values),
                    "max": max(response_time_values),
                    "avg": sum(response_time_values) / len(response_time_values),
                    "trend": self._calculate_trend(response_time_values),
                },
                "throughput": {
                    "current": throughput_values[-1],
                    "min": min(throughput_values),
                    "max": max(throughput_values),
                    "avg": sum(throughput_values) / len(throughput_values),
                    "trend": self._calculate_trend(throughput_values),
                },
            },
        }

    def get_optimization_recommendations(self) -> list[str]:
        """Generate optimization recommendations based on performance data.

        Returns:
            list of optimization recommendations

        """
        if not self.snapshots:
            return ["Insufficient performance data for recommendations"]

        recommendations = []
        latest = self.snapshots[-1]

        # Memory recommendations
        if latest.memory_percent > 85:
            recommendations.append(
                "High memory usage detected. Consider enabling quantization "
                "or increasing memory limits."
            )

        # CPU recommendations
        if latest.cpu_percent > 80:
            recommendations.append(
                "High CPU usage detected. Consider optimizing algorithms "
                "or scaling horizontally."
            )

        # Response time recommendations
        if latest.p95_response_time > 100:
            recommendations.append(
                "P95 latency exceeds 100ms target. Consider optimizing search "
                "parameters or enabling caching."
            )

        # Throughput recommendations
        if latest.request_rate > 0 and latest.request_rate < 50:
            recommendations.append(
                "Low throughput detected. Consider connection pooling "
                "optimization or batch processing."
            )

        # Cache recommendations
        if latest.cache_hit_rate < 0.85:
            recommendations.append(
                "Cache hit rate below 85%. Consider warming popular queries "
                "or increasing cache size."
            )

        # GC recommendations
        if latest.gc_collections > 10:  # High GC activity
            recommendations.append(
                "High garbage collection activity. Consider object pooling "
                "or memory optimization."
            )

        if not recommendations:
            recommendations.append("Performance is within optimal ranges.")

        return recommendations

    def _calculate_percentile(self, values: list[float], percentile: int) -> float:
        """Calculate percentile from list of values.

        Args:
            values: list of values to analyze
            percentile: Percentile to calculate (0-100)

        Returns:
            Percentile value

        """
        if not values:
            return 0.0

        sorted_values = sorted(values)
        index = int((percentile / 100) * len(sorted_values))
        return sorted_values[min(index, len(sorted_values) - 1)]

    def _calculate_trend(self, values: list[float]) -> str:
        """Calculate trend direction for a series of values.

        Args:
            values: list of values to analyze

        Returns:
            Trend direction ('increasing', 'decreasing', 'stable')

        """
        if len(values) < 2:
            return "stable"

        # Simple trend calculation using first and last values
        first_half = sum(values[: len(values) // 2]) / (len(values) // 2)
        second_half = sum(values[len(values) // 2 :]) / (len(values) - len(values) // 2)

        diff_percent = (
            ((second_half - first_half) / first_half) * 100 if first_half > 0 else 0
        )

        if diff_percent > 10:
            return "increasing"
        if diff_percent < -10:
            return "decreasing"
        return "stable"

    def _get_active_connections(self) -> int:
        """Get number of active connections (placeholder).

        Returns:
            Number of active connections

        """
        # This would be implemented based on the actual connection tracking
        # For now, return a placeholder value
        return len(self.request_times)

    def _get_cache_hit_rate(self) -> float:
        """Get cache hit rate (placeholder).

        Returns:
            Cache hit rate as a float between 0 and 1

        """
        # This would be implemented based on actual cache metrics
        # For now, return a placeholder value
        return 0.85

    def _get_gc_stats(self) -> tuple:
        """Get garbage collection statistics.

        Returns:
            tuple of GC collection counts

        """
        return tuple(gc.get_count())

    def _estimate_gc_time(self) -> float:
        """Estimate time spent in garbage collection.

        Returns:
            Estimated GC time in milliseconds

        """
        # This is a rough estimation - in production you'd want more precise tracking
        current_stats = self._get_gc_stats()
        total_collections = sum(current_stats) - sum(self.gc_stats_start)

        # Rough estimate: 1ms per collection (highly variable in reality)
        return total_collections * 1.0

    def cleanup(self) -> None:
        """Cleanup monitor resources."""
        self.monitoring_active = False
        self.snapshots.clear()
        self.request_times.clear()
        logger.info("Performance monitor cleanup completed")
