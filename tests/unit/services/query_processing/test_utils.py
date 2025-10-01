"""Unit tests for query processing utilities."""

import numpy as np
import pytest
from pydantic import BaseModel

from src.services.query_processing.clustering import SimilarityMetric
from src.services.query_processing.utils import (
    STOP_WORDS,
    CacheManager,
    PerformanceTracker,
    build_cache_key,
    deduplicate_results,
    distance_for_metric,
    merge_performance_metadata,
    performance_snapshot,
    sklearn_metric_for,
)


class _SampleModel(BaseModel):
    value: int


def test_cache_manager_returns_defensive_copy() -> None:
    """Test that CacheManager returns a defensive copy of cached objects."""
    cache = CacheManager(maxsize=2)
    original = _SampleModel(value=1)

    cache.set("sample", original)
    retrieved = cache.get("sample")
    assert retrieved is not None

    assert retrieved is not original
    assert retrieved == original

    retrieved.value = 42
    latest = cache.get("sample")
    assert latest is not None
    assert latest.value == 1


def test_cache_manager_snapshot_returns_clone() -> None:
    """Snapshot should provide a cloned cache view."""

    cache = CacheManager(maxsize=2)
    cache.set("key", _SampleModel(value=7))

    snapshot = cache.snapshot()

    assert snapshot["key"].value == 7
    snapshot["key"].value = 99
    cached_value = cache.get("key")
    assert cached_value is not None
    assert cached_value.value == 7


def test_cache_key_is_stable() -> None:
    """Test that build_cache_key produces stable and unique keys."""
    first = build_cache_key("a", "b", "c")
    second = build_cache_key("a", "b", "c")
    different = build_cache_key("a", "c", "b")

    assert first == second
    assert first != different


def test_performance_tracker_records_average() -> None:
    """Test that PerformanceTracker correctly records and calculates averages."""
    tracker = PerformanceTracker()
    tracker.record(10.0)
    tracker.record(5.0, label="strategy-a")
    tracker.record(5.0, label="strategy-a")

    assert tracker.total_operations == 3
    assert tracker.average_duration_ms == 20.0 / 3
    assert tracker.counters["strategy-a"] == 2


def test_performance_snapshot_serializes_tracker() -> None:
    """Performance snapshot should surface tracker counters."""

    tracker = PerformanceTracker()
    tracker.record(5.0, label="alpha")
    tracker.record(15.0, label="beta")
    tracker.record(20.0, label="beta")

    snapshot = performance_snapshot(tracker)

    assert snapshot["total_operations"] == 3
    assert snapshot["counters"] == {"alpha": 1, "beta": 2}
    assert snapshot["avg_processing_time"] == pytest.approx(40.0 / 3)


def test_distance_helpers_respect_metric() -> None:
    """Test that distance calculations respect the specified metric."""
    vector_a = np.array([1.0, 0.0])
    vector_b = np.array([0.0, 1.0])

    cosine_distance = distance_for_metric(vector_a, vector_b, "cosine")
    euclidean_distance = distance_for_metric(vector_a, vector_b, "euclidean")
    manhattan_distance = distance_for_metric(vector_a, vector_b, "manhattan")

    assert pytest.approx(cosine_distance, 1e-6) == 1.0
    assert pytest.approx(euclidean_distance, 1e-6) == np.sqrt(2.0)
    assert pytest.approx(manhattan_distance, 1e-6) == 2.0
    assert sklearn_metric_for("cosine") == "cosine"
    assert sklearn_metric_for("manhattan") == "manhattan"
    assert sklearn_metric_for("euclidean") == "euclidean"


def test_distance_helpers_support_enum_and_string_metrics() -> None:
    """Test that distance calculations work with both string and enum metric inputs."""

    vector_a = np.array([1.0, 0.0])
    vector_b = np.array([0.0, 1.0])

    # Test with string inputs
    cosine_str = distance_for_metric(vector_a, vector_b, "cosine")
    euclidean_str = distance_for_metric(vector_a, vector_b, "euclidean")

    # Test with enum inputs
    cosine_enum = distance_for_metric(vector_a, vector_b, SimilarityMetric.COSINE)
    euclidean_enum = distance_for_metric(vector_a, vector_b, SimilarityMetric.EUCLIDEAN)

    # Results should be identical
    assert cosine_str == cosine_enum
    assert euclidean_str == euclidean_enum

    # Test sklearn_metric_for with both types
    assert sklearn_metric_for("cosine") == sklearn_metric_for(SimilarityMetric.COSINE)
    assert sklearn_metric_for("euclidean") == sklearn_metric_for(
        SimilarityMetric.EUCLIDEAN
    )


def test_stop_words_contains_common_terms() -> None:
    """Test that STOP_WORDS contains expected common terms."""
    assert "the" in STOP_WORDS
    assert "and" in STOP_WORDS
    """Deduplication should collapse items sharing embeddings."""

    embedding = np.array([1.0, 0.0, 0.0])
    items = [
        {"content": "First result text", "embedding": embedding},
        {"content": "Different text", "embedding": embedding},
        {"content": "Completely distinct", "embedding": np.array([0.0, 1.0, 0.0])},
    ]

    deduped = deduplicate_results(
        items,
        content_getter=lambda item: item["content"],
        embedding_getter=lambda item: item["embedding"],
        threshold=0.95,
    )

    assert len(deduped) == 2
    assert deduped[0]["content"] == "First result text"
    assert deduped[1]["content"] == "Completely distinct"


def test_deduplicate_results_uses_textual_similarity() -> None:
    """Textual overlap should deduplicate when embeddings missing."""

    items = [
        {"content": "Python async guide", "embedding": None},
        {"content": "Guide for python async", "embedding": None},
        {"content": "Rust ownership explained", "embedding": None},
    ]

    deduped = deduplicate_results(
        items,
        content_getter=lambda item: item["content"],
        embedding_getter=lambda item: item["embedding"],
        threshold=0.6,
    )

    assert len(deduped) == 2


def test_merge_performance_metadata_combines_cache_stats() -> None:
    """Merged metadata should include cache counters and size."""

    class _Tracker:
        hits = 3
        misses = 1

    merged = merge_performance_metadata(
        performance_stats={"total_operations": 5, "avg_processing_time": 2.0},
        cache_tracker=_Tracker(),
        cache_size=42,
    )

    assert merged["cache_stats"] == {"hits": 3, "misses": 1}
    assert merged["cache_size"] == 42
    assert merged["total_operations"] == 5
