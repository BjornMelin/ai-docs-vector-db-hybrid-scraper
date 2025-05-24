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
