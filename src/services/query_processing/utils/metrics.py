"""Lightweight performance tracking helpers."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, field
from typing import Any


@dataclass
class PerformanceTracker:
    """Track average processing time and per-strategy usage."""

    total_operations: int = 0
    total_duration_ms: float = 0.0
    counters: dict[str, int] = field(default_factory=dict)

    def record(self, duration_ms: float, *, label: str | None = None) -> None:
        """Record a single execution duration."""

        self.total_operations += 1
        self.total_duration_ms += duration_ms
        if label is not None:
            self.counters[label] = self.counters.get(label, 0) + 1

    @property
    def average_duration_ms(self) -> float:
        """Return the average duration in milliseconds."""

        if self.total_operations == 0:
            return 0.0
        return self.total_duration_ms / self.total_operations


def performance_snapshot(tracker: PerformanceTracker) -> dict[str, Any]:
    """Serialize tracker statistics for external reporting.

    Args:
        tracker: Performance tracker instance to serialize.

    Returns:
        Dictionary containing total operations, average processing time, and
        per-label counters.
    """

    counters: Mapping[str, int] = tracker.counters
    return {
        "total_operations": tracker.total_operations,
        "avg_processing_time": tracker.average_duration_ms,
        "counters": dict(counters),
    }
