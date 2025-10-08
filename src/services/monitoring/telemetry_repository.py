"""
In-memory telemetry repository for counters and histograms.

This module consolidates prior near-duplicate implementations into a single,
thread-safe version while preserving the public API used by the codebase.

Design:
- Tag normalization produces sorted tuples to ensure stable dict keys.
- Counters are integer accumulators per (name, tags).
- Histograms store a bounded FIFO of float observations per (name, tags).
- Sampling can downsample histogram writes.

Note:
This intentionally remains in-memory. Process metrics should be exported via
the Prometheus client in src/services/monitoring/metrics.py where applicable.
See: https://prometheus.io/docs/concepts/metric_types/  # noqa: E501
"""

from __future__ import annotations

import random
from collections.abc import Iterable, Mapping
from dataclasses import dataclass
from threading import RLock
from typing import Final


TagMapping = Mapping[str, str]
_TagKey = tuple[tuple[str, str], ...]


def _normalize_tags(tags: TagMapping | None) -> _TagKey:
    """Normalize a tag mapping into a stable, hashable key.

    Args:
        tags: Mapping of tag names to values.

    Returns:
        Sorted tuple of (key, value) pairs for dict indexing.
    """
    if not tags:
        return ()
    return tuple(sorted((str(k), str(v)) for k, v in tags.items()))


@dataclass(slots=True)
class CounterSample:
    """Snapshot of a counter metric.

    Attributes:
        tags: Normalized tag key.
        value: Accumulated integer value.
    """

    tags: _TagKey
    value: int

    def as_dict(self) -> dict[str, object]:
        """Return a JSON-serializable representation.

        Returns:
            Dictionary with ``tags`` and ``value``.
        """
        return {"tags": dict(self.tags), "value": self.value}


@dataclass(slots=True)
class HistogramSample:
    """Snapshot of histogram observations for one tag set.

    Attributes:
        tags: Normalized tag key.
        count: Number of observations.
        sum: Sum of observations.
        values: Tuple of raw observed values (bounded by retention).
    """

    tags: _TagKey
    count: int
    sum: float
    values: tuple[float, ...]

    def as_dict(self) -> dict[str, object]:
        """Return a JSON-serializable representation.

        Returns:
            Dictionary with ``tags``, ``count``, ``sum``, and ``values``.
        """
        return {
            "tags": dict(self.tags),
            "count": self.count,
            "sum": self.sum,
            "values": list(self.values),
        }


class TelemetryRepository:
    """Thread-safe, in-memory counters and histograms.

    This class maintains minimal state for quick telemetry capture.
    Prefer exporting operational metrics via Prometheus for production.
    """

    _MIN_SAMPLES: Final[int] = 1

    def __init__(
        self,
        *,
        max_histogram_samples: int = 500,
        sample_rate: float = 1.0,
        in_memory_enabled: bool = True,
    ) -> None:
        """Initialize the repository.

        Args:
            max_histogram_samples: Max stored values per histogram series.
            sample_rate: Probability in [0,1] to record a histogram sample.
            in_memory_enabled: Gate all in-memory storage for tests or perf.
        """
        self._counter_storage: dict[str, dict[_TagKey, int]] = {}
        self._histogram_storage: dict[str, dict[_TagKey, list[float]]] = {}
        self._lock = RLock()
        self._max_histogram_samples = max(self._MIN_SAMPLES, max_histogram_samples)
        self._sample_rate = max(0.0, min(sample_rate, 1.0))
        self._in_memory_enabled = in_memory_enabled

    # --- Counters ---------------------------------------------------------

    def increment_counter(
        self,
        name: str,
        *,
        value: int = 1,
        tags: TagMapping | None = None,
    ) -> None:
        """Increment a counter series.

        Args:
            name: Metric name.
            value: Non-negative delta.
            tags: Optional label mapping.

        Raises:
            ValueError: If ``value`` is negative.
        """
        if value < 0:
            raise ValueError("Counter increments must be non-negative")
        if not self._in_memory_enabled:
            return

        key = _normalize_tags(tags)
        with self._lock:
            series = self._counter_storage.setdefault(name, {})
            series[key] = series.get(key, 0) + value

    # --- Histograms -------------------------------------------------------

    def record_observation(
        self,
        name: str,
        value: float,
        *,
        tags: TagMapping | None = None,
    ) -> None:
        """Record one histogram observation.

        Args:
            name: Metric name.
            value: Observation value.
            tags: Optional label mapping.
        """
        if not self._in_memory_enabled:
            return
        if self._sample_rate < 1.0 and random.random() > self._sample_rate:
            return

        key = _normalize_tags(tags)
        with self._lock:
            series = self._histogram_storage.setdefault(name, {})
            buf = series.setdefault(key, [])
            buf.append(float(value))
            if len(buf) > self._max_histogram_samples:
                # Keep newest samples; drop oldest.
                del buf[: len(buf) - self._max_histogram_samples]

    # --- Configuration -----------------------------------------------------

    def configure(
        self,
        *,
        max_histogram_samples: int | None = None,
        sample_rate: float | None = None,
        in_memory_enabled: bool | None = None,
    ) -> None:
        """Update retention, sampling, or enablement.

        Args:
            max_histogram_samples: New retention cap per series.
            sample_rate: New sampling probability in [0,1].
            in_memory_enabled: Toggle storage; clearing when disabled.
        """
        with self._lock:
            if max_histogram_samples is not None:
                self._max_histogram_samples = max(
                    self._MIN_SAMPLES, max_histogram_samples
                )
            if sample_rate is not None:
                self._sample_rate = max(0.0, min(sample_rate, 1.0))
            if in_memory_enabled is not None:
                self._in_memory_enabled = in_memory_enabled
                if not in_memory_enabled:
                    self._counter_storage.clear()
                    self._histogram_storage.clear()

    def reset(self) -> None:
        """Clear all metrics."""
        with self._lock:
            self._counter_storage.clear()
            self._histogram_storage.clear()

    # --- Snapshot ----------------------------------------------------------

    def counter_samples(self, name: str) -> Iterable[CounterSample]:
        """Iterate counter samples for a metric.

        Args:
            name: Metric name.

        Yields:
            CounterSample entries.
        """
        with self._lock:
            entries = list(self._counter_storage.get(name, {}).items())
        for tags, value in entries:
            yield CounterSample(tags, value)

    def histogram_samples(self, name: str) -> Iterable[HistogramSample]:
        """Iterate histogram samples for a metric.

        Args:
            name: Metric name.

        Yields:
            HistogramSample entries with values bounded by retention.
        """
        with self._lock:
            entries = list(self._histogram_storage.get(name, {}).items())
        for tags, values in entries:
            yield HistogramSample(tags, len(values), float(sum(values)), tuple(values))

    def export_snapshot(self) -> dict[str, object]:
        """Return a nested dictionary of all metrics.

        Returns:
            Dict with keys ``counters`` and ``histograms``.
        """
        snapshot: dict[str, object] = {"counters": {}, "histograms": {}}
        counters: dict[str, list[dict[str, object]]] = {}
        histograms: dict[str, list[dict[str, object]]] = {}

        with self._lock:
            for name, series in self._counter_storage.items():
                counters[name] = [
                    CounterSample(tags, value).as_dict()
                    for tags, value in series.items()
                ]
            for name, series in self._histogram_storage.items():
                histograms[name] = [
                    HistogramSample(
                        tags, len(vals), float(sum(vals)), tuple(vals)
                    ).as_dict()
                    for tags, vals in series.items()
                ]

        snapshot["counters"] = counters
        snapshot["histograms"] = histograms
        return snapshot


_repository = TelemetryRepository()


def get_telemetry_repository() -> TelemetryRepository:
    """Return the process-wide repository instance.

    Returns:
        The singleton repository instance.
    """
    return _repository


__all__ = [
    "CounterSample",
    "HistogramSample",
    "TelemetryRepository",
    "get_telemetry_repository",
]
