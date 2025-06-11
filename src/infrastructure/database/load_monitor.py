"""Load monitoring for dynamic database connection pool sizing.

This module provides real-time monitoring of application load metrics
to enable dynamic adjustment of database connection pool sizes.
"""

import asyncio
import logging
import time
from contextlib import suppress
from dataclasses import dataclass
from typing import Any

import psutil
from pydantic import BaseModel
from pydantic import Field

logger = logging.getLogger(__name__)


@dataclass
class LoadMetrics:
    """Current load metrics for dynamic pool sizing calculations."""

    concurrent_requests: int
    memory_usage_percent: float
    cpu_usage_percent: float
    avg_response_time_ms: float
    connection_errors: int
    timestamp: float


class LoadMonitorConfig(BaseModel):
    """Configuration for load monitoring."""

    monitoring_interval: float = Field(
        default=5.0, gt=0, description="Monitoring interval in seconds"
    )
    metrics_window_size: int = Field(
        default=60, gt=0, description="Number of metrics samples to keep"
    )
    response_time_threshold_ms: float = Field(
        default=500.0, gt=0, description="Response time threshold for scaling"
    )
    memory_threshold_percent: float = Field(
        default=70.0, gt=0, le=100, description="Memory usage threshold"
    )
    cpu_threshold_percent: float = Field(
        default=70.0, gt=0, le=100, description="CPU usage threshold"
    )


class LoadMonitor:
    """Monitors application load metrics for dynamic connection pool sizing.

    This class tracks various system and application metrics to provide
    data for intelligent connection pool scaling decisions.
    """

    def __init__(self, config: LoadMonitorConfig | None = None):
        """Initialize the load monitor.

        Args:
            config: Load monitoring configuration
        """
        self.config = config or LoadMonitorConfig()
        self._metrics_history: list[LoadMetrics] = []
        self._current_requests = 0
        self._total_requests = 0
        self._total_response_time = 0.0
        self._connection_errors = 0
        self._is_monitoring = False
        self._monitoring_task: asyncio.Task[Any] | None = None
        self._lock = asyncio.Lock()

    async def start(self) -> None:
        """Start the load monitoring."""
        if self._is_monitoring:
            return

        self._is_monitoring = True
        self._monitoring_task = asyncio.create_task(self._monitoring_loop())
        logger.info("Load monitoring started")

    async def stop(self) -> None:
        """Stop the load monitoring."""
        if not self._is_monitoring:
            return

        self._is_monitoring = False
        if self._monitoring_task:
            self._monitoring_task.cancel()
            with suppress(asyncio.CancelledError):
                await self._monitoring_task
        logger.info("Load monitoring stopped")

    async def get_current_load(self) -> LoadMetrics:
        """Get the current load metrics.

        Returns:
            Current load metrics
        """
        async with self._lock:
            if not self._metrics_history:
                # Return default metrics if no history available
                return LoadMetrics(
                    concurrent_requests=0,
                    memory_usage_percent=0.0,
                    cpu_usage_percent=0.0,
                    avg_response_time_ms=0.0,
                    connection_errors=0,
                    timestamp=time.time(),
                )
            return self._metrics_history[-1]

    async def get_average_load(self, window_minutes: int = 5) -> LoadMetrics:
        """Get average load metrics over a time window.

        Args:
            window_minutes: Time window in minutes

        Returns:
            Average load metrics
        """
        async with self._lock:
            if not self._metrics_history:
                return await self.get_current_load()

            cutoff_time = time.time() - (window_minutes * 60)
            recent_metrics = [
                m for m in self._metrics_history if m.timestamp >= cutoff_time
            ]

            if not recent_metrics:
                return self._metrics_history[-1]

            avg_concurrent_requests = sum(
                m.concurrent_requests for m in recent_metrics
            ) / len(recent_metrics)
            avg_memory = sum(m.memory_usage_percent for m in recent_metrics) / len(
                recent_metrics
            )
            avg_cpu = sum(m.cpu_usage_percent for m in recent_metrics) / len(
                recent_metrics
            )
            avg_response_time = sum(
                m.avg_response_time_ms for m in recent_metrics
            ) / len(recent_metrics)
            total_errors = sum(m.connection_errors for m in recent_metrics)

            return LoadMetrics(
                concurrent_requests=int(avg_concurrent_requests),
                memory_usage_percent=avg_memory,
                cpu_usage_percent=avg_cpu,
                avg_response_time_ms=avg_response_time,
                connection_errors=total_errors,
                timestamp=time.time(),
            )

    async def record_request_start(self) -> None:
        """Record the start of a request."""
        async with self._lock:
            self._current_requests += 1
            self._total_requests += 1

    async def record_request_end(self, response_time_ms: float) -> None:
        """Record the end of a request.

        Args:
            response_time_ms: Request response time in milliseconds
        """
        async with self._lock:
            self._current_requests = max(0, self._current_requests - 1)
            self._total_response_time += response_time_ms

    async def record_connection_error(self) -> None:
        """Record a database connection error."""
        async with self._lock:
            self._connection_errors += 1

    def calculate_load_factor(self, metrics: LoadMetrics) -> float:
        """Calculate a load factor from 0.0 to 1.0 based on current metrics.

        Args:
            metrics: Current load metrics

        Returns:
            Load factor between 0.0 and 1.0
        """
        # Weight different factors
        request_factor = min(
            metrics.concurrent_requests / 50, 1.0
        )  # Normalize to 50 requests
        memory_factor = metrics.memory_usage_percent / 100.0
        cpu_factor = metrics.cpu_usage_percent / 100.0
        response_time_factor = min(
            metrics.avg_response_time_ms / self.config.response_time_threshold_ms, 1.0
        )

        # Error penalty
        error_penalty = min(metrics.connection_errors * 0.1, 0.5)  # Up to 50% penalty

        # Weighted average
        load_factor = (
            request_factor * 0.3
            + memory_factor * 0.2
            + cpu_factor * 0.2
            + response_time_factor * 0.3
        ) + error_penalty

        return min(load_factor, 1.0)

    async def _monitoring_loop(self) -> None:
        """Main monitoring loop."""
        while self._is_monitoring:
            try:
                await self._collect_metrics()
                await asyncio.sleep(self.config.monitoring_interval)
            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"Error in monitoring loop: {e}")
                await asyncio.sleep(1.0)  # Brief pause before retrying

    async def _collect_metrics(self) -> None:
        """Collect current metrics."""
        try:
            # Get system metrics
            memory_info = psutil.virtual_memory()
            cpu_percent = psutil.cpu_percent(interval=None)

            # Calculate average response time
            avg_response_time = 0.0
            if self._total_requests > 0:
                avg_response_time = self._total_response_time / self._total_requests

            async with self._lock:
                metrics = LoadMetrics(
                    concurrent_requests=self._current_requests,
                    memory_usage_percent=memory_info.percent,
                    cpu_usage_percent=cpu_percent,
                    avg_response_time_ms=avg_response_time,
                    connection_errors=self._connection_errors,
                    timestamp=time.time(),
                )

                # Add to history
                self._metrics_history.append(metrics)

                # Trim history to keep only recent metrics
                if len(self._metrics_history) > self.config.metrics_window_size:
                    self._metrics_history = self._metrics_history[
                        -self.config.metrics_window_size :
                    ]

                # Reset error counter periodically
                if len(self._metrics_history) % 10 == 0:
                    self._connection_errors = 0

        except Exception as e:
            logger.error(f"Error collecting metrics: {e}")
