"""Utility helpers for query processing services."""

from .cache import CacheManager, CacheTracker, build_cache_key
from .metrics import PerformanceTracker, performance_snapshot
from .results import deduplicate_results, merge_performance_metadata
from .similarity import distance_for_metric, sklearn_metric_for
from .text import STOP_WORDS


__all__ = [
    "CacheManager",
    "CacheTracker",
    "PerformanceTracker",
    "performance_snapshot",
    "STOP_WORDS",
    "build_cache_key",
    "deduplicate_results",
    "merge_performance_metadata",
    "distance_for_metric",
    "sklearn_metric_for",
]
