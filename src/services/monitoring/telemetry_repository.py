"""In-memory telemetry repository for agentic graph execution."""

from __future__ import annotations

import random
from collections.abc import Iterable, Mapping
from dataclasses import dataclass
from threading import RLock


TagMapping = Mapping[str, str]


def _normalize_tags(tags: TagMapping | None) -> tuple[tuple[str, str], ...]:
    """Return tags as a sorted tuple suitable for dictionary keys."""

    if not tags:
        return ()
    return tuple(sorted((str(key), str(value)) for key, value in tags.items()))


@dataclass(slots=True)
class CounterSample:
    """Snapshot of a counter metric."""

    tags: tuple[tuple[str, str], ...]
    value: int

    def as_dict(self) -> dict[str, object]:
        """Return a serializable representation."""

        return {"tags": dict(self.tags), "value": self.value}


@dataclass(slots=True)
class HistogramSample:
    """Snapshot of histogram observations."""

    tags: tuple[tuple[str, str], ...]
    count: int
    sum: float
    values: tuple[float, ...]

    def as_dict(self) -> dict[str, object]:
        """Return a serializable representation."""

        return {
            "tags": dict(self.tags),
            "count": self.count,
            "sum": self.sum,
            "values": list(self.values),
        }


class TelemetryRepository:
    """Thread-safe repository for lightweight counters and histograms."""

    def __init__(
        self,
        *,
        max_histogram_samples: int = 500,
        sample_rate: float = 1.0,
        in_memory_enabled: bool = True,
    ) -> None:
        self._counter_storage: dict[str, dict[tuple[tuple[str, str], ...], int]] = {}
        self._histogram_storage: dict[
            str, dict[tuple[tuple[str, str], ...], list[float]]
        ] = {}
        self._lock = RLock()
        self._max_histogram_samples = max(1, max_histogram_samples)
        self._sample_rate = max(0.0, min(sample_rate, 1.0))
        self._in_memory_enabled = in_memory_enabled

    def increment_counter(
        self,
        name: str,
        *,
        value: int = 1,
        tags: TagMapping | None = None,
    ) -> None:
        """Increase a counter value for the provided name and tags."""

        if value < 0:
            msg = "Counter increments must be non-negative"
            raise ValueError(msg)
        if not self._in_memory_enabled:
            return
        key = _normalize_tags(tags)
        with self._lock:
            counters = self._counter_storage.setdefault(name, {})
            counters[key] = counters.get(key, 0) + value

    def record_observation(
        self,
        name: str,
        value: float,
        *,
        tags: TagMapping | None = None,
    ) -> None:
        """Record a single histogram observation."""

        if not self._in_memory_enabled:
            return
        if self._sample_rate < 1.0 and random.random() > self._sample_rate:
            return
        key = _normalize_tags(tags)
        with self._lock:
            histograms = self._histogram_storage.setdefault(name, {})
            samples = histograms.setdefault(key, [])
            samples.append(float(value))
            if len(samples) > self._max_histogram_samples:
                del samples[: len(samples) - self._max_histogram_samples]

    def configure(
        self,
        *,
        max_histogram_samples: int | None = None,
        sample_rate: float | None = None,
        in_memory_enabled: bool | None = None,
    ) -> None:
        """Update storage retention, sampling, and memory toggle settings."""

        with self._lock:
            if max_histogram_samples is not None:
                self._max_histogram_samples = max(1, max_histogram_samples)
            if sample_rate is not None:
                self._sample_rate = max(0.0, min(sample_rate, 1.0))
            if in_memory_enabled is not None:
                self._in_memory_enabled = in_memory_enabled
                if not self._in_memory_enabled:
                    self._counter_storage.clear()
                    self._histogram_storage.clear()

    def reset(self) -> None:
        """Clear all recorded metrics."""

        with self._lock:
            self._counter_storage.clear()
            self._histogram_storage.clear()

    def counter_samples(self, name: str) -> Iterable[CounterSample]:
        """Yield counter samples for the supplied metric name."""

        with self._lock:
            entries = list(self._counter_storage.get(name, {}).items())
        for tags, value in entries:
            yield CounterSample(tags, value)

    def histogram_samples(self, name: str) -> Iterable[HistogramSample]:
        """Yield histogram samples for the supplied metric name."""

        with self._lock:
            entries = list(self._histogram_storage.get(name, {}).items())
        for tags, values in entries:
            yield HistogramSample(tags, len(values), float(sum(values)), tuple(values))

    def export_snapshot(self) -> dict[str, object]:
        """Return a nested dictionary representing all recorded metrics."""

        snapshot: dict[str, object] = {"counters": {}, "histograms": {}}
        counters: dict[str, list[dict[str, object]]] = {}
        histograms: dict[str, list[dict[str, object]]] = {}
        with self._lock:
            for name, samples in self._counter_storage.items():
                counters[name] = [
                    CounterSample(tags, value).as_dict()
                    for tags, value in samples.items()
                ]
            for name, samples in self._histogram_storage.items():
                histograms[name] = [
                    HistogramSample(
                        tags, len(values), float(sum(values)), tuple(values)
                    ).as_dict()
                    for tags, values in samples.items()
                ]
        snapshot["counters"] = counters
        snapshot["histograms"] = histograms
        return snapshot


_repository = TelemetryRepository()


def get_telemetry_repository() -> TelemetryRepository:
    """Return the process-wide telemetry repository instance."""

    return _repository


__all__ = [
    "CounterSample",
    "HistogramSample",
    "TelemetryRepository",
    "get_telemetry_repository",
]
