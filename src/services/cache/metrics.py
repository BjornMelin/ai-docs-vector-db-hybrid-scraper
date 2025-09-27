"""Simple cache metrics for V1 MVP."""

from collections import defaultdict
from dataclasses import dataclass
from typing import Any


@dataclass
class CacheStats:
    """Basic cache statistics."""

    hits: int = 0
    misses: int = 0
    sets: int = 0
    errors: int = 0

    @property
    def total_requests(self) -> int:
        """Total cache requests."""
        return self.hits + self.misses

    @property
    def hit_rate(self) -> float:
        """Cache hit rate."""
        if self.total_requests == 0:
            return 0.0
        return self.hits / self.total_requests


class CacheMetrics:
    """Simple cache metrics collector for V1.

    V2 will add Prometheus integration, histograms, etc.
    """

    def __init__(self):
        """Initialize metrics collector."""
        # Stats by cache type and layer
        self._stats: dict[str, dict[str, CacheStats]] = defaultdict(
            lambda: defaultdict(CacheStats)
        )

    def record_hit(self, cache_type: str, layer: str, latency: float) -> None:
        """Record cache hit."""
        self._stats[cache_type][layer].hits += 1

    def record_miss(self, cache_type: str, latency: float) -> None:
        """Record cache miss."""
        self._stats[cache_type]["total"].misses += 1

    def record_set(self, cache_type: str, latency: float, success: bool) -> None:
        """Record cache set operation."""
        if success:
            self._stats[cache_type]["total"].sets += 1
        else:
            self._stats[cache_type]["total"].errors += 1

    def get_summary(self) -> dict[str, Any]:
        """Get metrics summary."""
        summary = {}

        for cache_type, layers in self._stats.items():
            summary[cache_type] = {}

            for layer, stats in layers.items():
                summary[cache_type][layer] = {
                    "hits": stats.hits,
                    "misses": stats.misses,
                    "sets": stats.sets,
                    "errors": stats.errors,
                    "hit_rate": stats.hit_rate,
                }

        return summary

    def get_hit_rates(self) -> dict[str, float]:
        """Get hit rates by cache type."""
        hit_rates = {}
        for cache_type, layers in self._stats.items():
            total_hits = sum(layer.hits for layer in layers.values())
            total_requests = sum(layer.total_requests for layer in layers.values())
            hit_rates[cache_type] = (
                total_hits / total_requests if total_requests > 0 else 0.0
            )
        return hit_rates

    def get_latency_stats(self) -> dict[str, dict[str, float]]:
        """Get latency statistics by cache type."""
        # V1 placeholder - V2 will track actual latencies
        return {
            cache_type: {"avg": 0.0, "p95": 0.0, "p99": 0.0}
            for cache_type in self._stats
        }

    def get_operation_counts(self) -> dict[str, dict[str, int]]:
        """Get operation counts by cache type."""
        counts = {}
        for cache_type, layers in self._stats.items():
            total_stats = CacheStats()
            for layer in layers.values():
                total_stats.hits += layer.hits
                total_stats.misses += layer.misses
                total_stats.sets += layer.sets
                total_stats.errors += layer.errors

            counts[cache_type] = {
                "hits": total_stats.hits,
                "misses": total_stats.misses,
                "sets": total_stats.sets,
                "errors": total_stats.errors,
                "total": total_stats.total_requests,
            }
        return counts
