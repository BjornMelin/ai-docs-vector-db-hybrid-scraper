"""Performance monitoring utilities."""

from __future__ import annotations

import logging
import time
from collections import deque
from contextlib import contextmanager
from dataclasses import dataclass, field
from functools import lru_cache
from typing import Any


try:
    import psutil
except ImportError:  # pragma: no cover - optional dependency
    psutil = None  # type: ignore[assignment]

from opentelemetry import metrics
from opentelemetry.metrics import Meter


LOGGER = logging.getLogger(__name__)
DEFAULT_MONITOR_NAME = "ai-docs.performance"


@dataclass
class OperationRecord:
    duration_ms: float
    cpu_percent: float | None = None
    memory_mb: float | None = None
    metadata: dict[str, Any] = field(default_factory=dict)


class PerformanceMonitor:
    """Minimal context-based performance monitor."""

    _RECENT_OPERATION_LIMIT = 256

    def __init__(self, meter: Meter | None = None) -> None:
        self._meter = meter or metrics.get_meter(__name__)
        self._duration_histogram = self._meter.create_histogram(
            "performance.operation.duration",
            description="Duration of monitored operations in milliseconds",
            unit="ms",
        )
        self._cpu_histogram = self._meter.create_histogram(
            "performance.operation.cpu_percent",
            description="CPU utilisation captured at the end of monitored operations",
            unit="%",
        )
        self._memory_histogram = self._meter.create_histogram(
            "performance.operation.memory_mb",
            description="RSS memory usage captured at the end of monitored operations",
            unit="MB",
        )
        self._recent_operations: deque[OperationRecord] = deque(
            maxlen=self._RECENT_OPERATION_LIMIT
        )

    def initialize(self) -> None:
        LOGGER.info("Performance monitor initialized")

    async def cleanup(self) -> None:
        self._recent_operations.clear()
        LOGGER.info("Performance monitor cleaned up")

    @contextmanager
    def monitor_operation(
        self,
        name: str,
        *,
        category: str = "general",
        collect_resources: bool = True,
        metadata: dict[str, Any] | None = None,
    ):
        start = time.perf_counter()
        try:
            yield
        finally:
            duration_ms = (time.perf_counter() - start) * 1000
            cpu_percent: float | None = None
            memory_mb: float | None = None

            if collect_resources and psutil is not None:
                try:
                    process = psutil.Process()
                    cpu_percent = process.cpu_percent(interval=None)
                    memory_mb = process.memory_info().rss / (1024 * 1024)
                except (OSError, psutil.Error):  # pragma: no cover - best effort only
                    LOGGER.debug("Failed to collect psutil metrics", exc_info=True)

            labels = {"operation": name, "category": category}
            self._duration_histogram.record(duration_ms, labels)
            if cpu_percent is not None:
                self._cpu_histogram.record(cpu_percent, labels)
            if memory_mb is not None:
                self._memory_histogram.record(memory_mb, labels)

            record = OperationRecord(
                duration_ms=duration_ms,
                cpu_percent=cpu_percent,
                memory_mb=memory_mb,
                metadata=metadata or {},
            )
            self._recent_operations.append(record)

    def recent_operations(self, limit: int = 10) -> list[OperationRecord]:
        if limit <= 0:
            return []

        operations = list(self._recent_operations)
        if limit >= len(operations):
            return operations

        return operations[-limit:]


@lru_cache(maxsize=1)
def get_performance_monitor() -> PerformanceMonitor:
    monitor = PerformanceMonitor()
    monitor.initialize()
    return monitor


def initialize_performance_monitor() -> PerformanceMonitor:
    return get_performance_monitor()


def monitor_operation(
    name: str,
    category: str = "general",
    *,
    collect_resources: bool = True,
    metadata: dict[str, Any] | None = None,
):
    monitor = get_performance_monitor()
    return monitor.monitor_operation(
        name,
        category=category,
        collect_resources=collect_resources,
        metadata=metadata,
    )


def get_operation_statistics() -> dict[str, float]:
    monitor = get_performance_monitor()
    recent = monitor.recent_operations()
    if not recent:
        return {"count": 0, "avg_duration_ms": 0.0}

    total_duration = sum(record.duration_ms for record in recent)
    return {
        "count": len(recent),
        "avg_duration_ms": total_duration / len(recent),
    }


def get_system_performance_summary() -> dict[str, Any]:
    if psutil is None:
        return {"psutil_available": False}

    process = psutil.Process()
    with process.oneshot():
        memory_info = process.memory_info()
        cpu_percent = process.cpu_percent(interval=None)
    return {
        "psutil_available": True,
        "rss_memory_mb": memory_info.rss / (1024 * 1024),
        "cpu_percent": cpu_percent,
    }
