"""Tests for cache metrics module."""

import pytest
from src.services.cache.metrics import CacheMetrics
from src.services.cache.metrics import CacheStats


class TestCacheStats:
    """Test the CacheStats dataclass."""

    def test_cache_stats_initialization_defaults(self):
        """Test CacheStats initialization with defaults."""
        stats = CacheStats()

        assert stats.hits == 0
        assert stats.misses == 0
        assert stats.sets == 0
        assert stats.errors == 0

    def test_cache_stats_initialization_with_values(self):
        """Test CacheStats initialization with specific values."""
        stats = CacheStats(hits=10, misses=5, sets=15, errors=2)

        assert stats.hits == 10
        assert stats.misses == 5
        assert stats.sets == 15
        assert stats.errors == 2

    def test_cache_stats__total_requests_property(self):
        """Test _total_requests property calculation."""
        stats = CacheStats(hits=10, misses=5)

        assert stats._total_requests == 15

    def test_cache_stats__total_requests_zero(self):
        """Test _total_requests property when both hits and misses are zero."""
        stats = CacheStats()

        assert stats._total_requests == 0

    def test_cache_stats_hit_rate_property(self):
        """Test hit_rate property calculation."""
        stats = CacheStats(hits=8, misses=2)

        assert stats.hit_rate == 0.8

    def test_cache_stats_hit_rate_perfect(self):
        """Test hit_rate property with perfect hit rate."""
        stats = CacheStats(hits=10, misses=0)

        assert stats.hit_rate == 1.0

    def test_cache_stats_hit_rate_zero_hits(self):
        """Test hit_rate property with zero hits."""
        stats = CacheStats(hits=0, misses=10)

        assert stats.hit_rate == 0.0

    def test_cache_stats_hit_rate_no_requests(self):
        """Test hit_rate property when no requests have been made."""
        stats = CacheStats()

        assert stats.hit_rate == 0.0

    def test_cache_stats_hit_rate_precision(self):
        """Test hit_rate property with precision."""
        stats = CacheStats(hits=1, misses=2)

        # 1/3 should be approximately 0.3333...
        assert abs(stats.hit_rate - (1 / 3)) < 0.0001

    def test_cache_stats_dataclass_fields(self):
        """Test that CacheStats has the expected dataclass fields."""
        stats = CacheStats()

        # Test field access
        assert hasattr(stats, "hits")
        assert hasattr(stats, "misses")
        assert hasattr(stats, "sets")
        assert hasattr(stats, "errors")

    def test_cache_stats_immutable_properties(self):
        """Test that computed properties work correctly with field changes."""
        stats = CacheStats(hits=5, misses=5)

        assert stats._total_requests == 10
        assert stats.hit_rate == 0.5

        # Modify fields
        stats.hits = 8
        stats.misses = 2

        # Properties should reflect new values
        assert stats._total_requests == 10
        assert stats.hit_rate == 0.8

    def test_cache_stats_equality(self):
        """Test CacheStats equality comparison."""
        stats1 = CacheStats(hits=10, misses=5, sets=15, errors=2)
        stats2 = CacheStats(hits=10, misses=5, sets=15, errors=2)
        stats3 = CacheStats(hits=8, misses=5, sets=15, errors=2)

        assert stats1 == stats2
        assert stats1 != stats3


class TestCacheMetrics:
    """Test the CacheMetrics class."""

    @pytest.fixture
    def cache_metrics(self):
        """Create a CacheMetrics instance for testing."""
        return CacheMetrics()

    def test_cache_metrics_initialization(self, cache_metrics):
        """Test CacheMetrics initialization."""
        assert hasattr(cache_metrics, "_stats")
        assert len(cache_metrics._stats) == 0

    def test_record_hit_single(self, cache_metrics):
        """Test recording a single cache hit."""
        cache_metrics.record_hit("embedding", "dragonfly", 0.5)

        summary = cache_metrics.get_summary()
        assert "embedding" in summary
        assert "dragonfly" in summary["embedding"]
        assert summary["embedding"]["dragonfly"]["hits"] == 1
        assert summary["embedding"]["dragonfly"]["misses"] == 0

    def test_record_hit_multiple_same_type(self, cache_metrics):
        """Test recording multiple hits for the same cache type and layer."""
        for _ in range(5):
            cache_metrics.record_hit("embedding", "dragonfly", 0.3)

        summary = cache_metrics.get_summary()
        assert summary["embedding"]["dragonfly"]["hits"] == 5

    def test_record_hit_different_layers(self, cache_metrics):
        """Test recording hits for different cache layers."""
        cache_metrics.record_hit("embedding", "dragonfly", 0.5)
        cache_metrics.record_hit("embedding", "local", 0.1)
        cache_metrics.record_hit("search", "dragonfly", 0.4)

        summary = cache_metrics.get_summary()
        assert summary["embedding"]["dragonfly"]["hits"] == 1
        assert summary["embedding"]["local"]["hits"] == 1
        assert summary["search"]["dragonfly"]["hits"] == 1

    def test_record_miss_single(self, cache_metrics):
        """Test recording a single cache miss."""
        cache_metrics.record_miss("embedding", 2.5)

        summary = cache_metrics.get_summary()
        assert summary["embedding"]["_total"]["hits"] == 0
        assert summary["embedding"]["_total"]["misses"] == 1

    def test_record_miss_multiple(self, cache_metrics):
        """Test recording multiple cache misses."""
        for _ in range(3):
            cache_metrics.record_miss("search", 1.2)

        summary = cache_metrics.get_summary()
        assert summary["search"]["_total"]["misses"] == 3

    def test_record_miss_different_types(self, cache_metrics):
        """Test recording misses for different cache types."""
        cache_metrics.record_miss("embedding", 2.0)
        cache_metrics.record_miss("search", 1.5)
        cache_metrics.record_miss("embedding", 1.8)

        summary = cache_metrics.get_summary()
        assert summary["embedding"]["_total"]["misses"] == 2
        assert summary["search"]["_total"]["misses"] == 1

    def test_record_set_success(self, cache_metrics):
        """Test recording successful cache set operations."""
        cache_metrics.record_set("embedding", 0.8, True)
        cache_metrics.record_set("embedding", 0.6, True)

        summary = cache_metrics.get_summary()
        assert summary["embedding"]["_total"]["sets"] == 2
        assert summary["embedding"]["_total"]["errors"] == 0

    def test_record_set_failure(self, cache_metrics):
        """Test recording failed cache set operations."""
        cache_metrics.record_set("embedding", 1.2, False)
        cache_metrics.record_set("search", 0.9, False)

        summary = cache_metrics.get_summary()
        assert summary["embedding"]["_total"]["sets"] == 0
        assert summary["embedding"]["_total"]["errors"] == 1
        assert summary["search"]["_total"]["errors"] == 1

    def test_record_set_mixed_success_failure(self, cache_metrics):
        """Test recording mixed successful and failed set operations."""
        cache_metrics.record_set("embedding", 0.5, True)
        cache_metrics.record_set("embedding", 1.0, False)
        cache_metrics.record_set("embedding", 0.7, True)
        cache_metrics.record_set("embedding", 1.5, False)

        summary = cache_metrics.get_summary()
        assert summary["embedding"]["_total"]["sets"] == 2
        assert summary["embedding"]["_total"]["errors"] == 2

    def test_mixed_operations_single_cache_type(self, cache_metrics):
        """Test mixed operations for a single cache type."""
        # Record various operations
        cache_metrics.record_hit("embedding", "dragonfly", 0.3)
        cache_metrics.record_hit("embedding", "local", 0.1)
        cache_metrics.record_miss("embedding", 2.0)
        cache_metrics.record_miss("embedding", 1.8)
        cache_metrics.record_set("embedding", 0.6, True)
        cache_metrics.record_set("embedding", 1.2, False)

        summary = cache_metrics.get_summary()
        embedding_stats = summary["embedding"]

        # Check layer-specific hits
        assert embedding_stats["dragonfly"]["hits"] == 1
        assert embedding_stats["local"]["hits"] == 1

        # Check _total stats
        assert embedding_stats["_total"]["misses"] == 2
        assert embedding_stats["_total"]["sets"] == 1
        assert embedding_stats["_total"]["errors"] == 1

    def test_mixed_operations_multiple_cache_types(self, cache_metrics):
        """Test mixed operations across multiple cache types."""
        # Embedding operations
        cache_metrics.record_hit("embedding", "dragonfly", 0.5)
        cache_metrics.record_miss("embedding", 2.0)
        cache_metrics.record_set("embedding", 0.8, True)

        # Search operations
        cache_metrics.record_hit("search", "local", 0.2)
        cache_metrics.record_hit("search", "dragonfly", 0.4)
        cache_metrics.record_miss("search", 1.5)
        cache_metrics.record_set("search", 0.9, False)

        # Document operations
        cache_metrics.record_miss("document", 3.0)

        summary = cache_metrics.get_summary()

        # Verify embedding stats
        assert summary["embedding"]["dragonfly"]["hits"] == 1
        assert summary["embedding"]["_total"]["misses"] == 1
        assert summary["embedding"]["_total"]["sets"] == 1

        # Verify search stats
        assert summary["search"]["local"]["hits"] == 1
        assert summary["search"]["dragonfly"]["hits"] == 1
        assert summary["search"]["_total"]["misses"] == 1
        assert summary["search"]["_total"]["errors"] == 1

        # Verify document stats
        assert summary["document"]["_total"]["misses"] == 1

    def test_get_summary_empty(self, cache_metrics):
        """Test get_summary with no recorded metrics."""
        summary = cache_metrics.get_summary()

        assert summary == {}

    def test_get_summary_structure(self, cache_metrics):
        """Test get_summary returns correct structure."""
        cache_metrics.record_hit("embedding", "dragonfly", 0.5)
        cache_metrics.record_miss("embedding", 2.0)

        summary = cache_metrics.get_summary()

        # Check structure
        assert isinstance(summary, dict)
        assert "embedding" in summary
        assert isinstance(summary["embedding"], dict)
        assert "dragonfly" in summary["embedding"]
        assert "_total" in summary["embedding"]

        # Check dragonfly layer stats
        dragonfly_stats = summary["embedding"]["dragonfly"]
        expected_fields = ["hits", "misses", "sets", "errors", "hit_rate"]
        for field in expected_fields:
            assert field in dragonfly_stats

        # Check _total layer stats
        _total_stats = summary["embedding"]["_total"]
        for field in expected_fields:
            assert field in _total_stats

    def test_hit_rate_calculation_in_summary(self, cache_metrics):
        """Test hit rate calculation in summary."""
        # Record 3 hits and 2 misses for dragonfly layer
        for _ in range(3):
            cache_metrics.record_hit("embedding", "dragonfly", 0.3)

        # Misses go to _total layer
        for _ in range(2):
            cache_metrics.record_miss("embedding", 2.0)

        summary = cache_metrics.get_summary()

        # Dragonfly layer should have 100% hit rate (3 hits, 0 misses)
        assert summary["embedding"]["dragonfly"]["hit_rate"] == 1.0

        # Total layer should have 0% hit rate (0 hits, 2 misses)
        assert summary["embedding"]["_total"]["hit_rate"] == 0.0

    def test_hit_rate_calculation_mixed_layer(self, cache_metrics):
        """Test hit rate calculation with mixed layer stats."""
        # Add some hits to a layer first
        cache_metrics.record_hit("search", "local", 0.1)
        cache_metrics.record_hit("search", "local", 0.1)

        # Then add a miss to _total (simulating local cache miss, going to next layer)
        cache_metrics.record_miss("search", 1.0)

        summary = cache_metrics.get_summary()

        # Local layer: 2 hits, 0 misses = 100% hit rate
        assert summary["search"]["local"]["hit_rate"] == 1.0

        # Total layer: 0 hits, 1 miss = 0% hit rate
        assert summary["search"]["_total"]["hit_rate"] == 0.0

    def test_latency_parameter_ignored(self, cache_metrics):
        """Test that latency parameters are accepted but not used in V1."""
        # V1 accepts latency but doesn't store/use it
        cache_metrics.record_hit("embedding", "dragonfly", 999.9)
        cache_metrics.record_miss("embedding", 888.8)
        cache_metrics.record_set("embedding", 777.7, True)

        # Should still work normally
        summary = cache_metrics.get_summary()
        assert summary["embedding"]["dragonfly"]["hits"] == 1
        assert summary["embedding"]["_total"]["misses"] == 1
        assert summary["embedding"]["_total"]["sets"] == 1

    def test_edge_case_very_large_numbers(self, cache_metrics):
        """Test with very large numbers of operations."""
        large_number = 10000

        for _ in range(large_number):
            cache_metrics.record_hit("embedding", "dragonfly", 0.5)

        summary = cache_metrics.get_summary()
        assert summary["embedding"]["dragonfly"]["hits"] == large_number

    def test_edge_case_zero_latency(self, cache_metrics):
        """Test with zero latency values."""
        cache_metrics.record_hit("embedding", "dragonfly", 0.0)
        cache_metrics.record_miss("embedding", 0.0)
        cache_metrics.record_set("embedding", 0.0, True)

        # Should work normally
        summary = cache_metrics.get_summary()
        assert summary["embedding"]["dragonfly"]["hits"] == 1
        assert summary["embedding"]["_total"]["misses"] == 1
        assert summary["embedding"]["_total"]["sets"] == 1

    def test_cache_type_and_layer_naming(self, cache_metrics):
        """Test various cache type and layer names."""
        test_cases = [
            ("embedding", "dragonfly"),
            ("search", "local"),
            ("document", "redis"),
            ("user-cache", "memory"),
            ("cache_type_123", "layer_456"),
            ("", ""),  # Edge case: empty strings
        ]

        for cache_type, layer in test_cases:
            cache_metrics.record_hit(cache_type, layer, 0.1)

        summary = cache_metrics.get_summary()

        for cache_type, layer in test_cases:
            assert cache_type in summary
            assert layer in summary[cache_type]
            assert summary[cache_type][layer]["hits"] == 1

    def test_state_isolation(self):
        """Test that different CacheMetrics instances are isolated."""
        metrics1 = CacheMetrics()
        metrics2 = CacheMetrics()

        metrics1.record_hit("embedding", "dragonfly", 0.5)
        metrics2.record_miss("search", 2.0)

        summary1 = metrics1.get_summary()
        summary2 = metrics2.get_summary()

        # Each should only have its own data
        assert "embedding" in summary1
        assert "search" not in summary1
        assert "search" in summary2
        assert "embedding" not in summary2

    def test_v1_feature_completeness(self, cache_metrics):
        """Test that V1 implementation covers all documented features."""
        # Test all public methods exist and work
        assert hasattr(cache_metrics, "record_hit")
        assert hasattr(cache_metrics, "record_miss")
        assert hasattr(cache_metrics, "record_set")
        assert hasattr(cache_metrics, "get_summary")

        # Test basic functionality works
        cache_metrics.record_hit("test", "layer", 1.0)
        cache_metrics.record_miss("test", 1.0)
        cache_metrics.record_set("test", 1.0, True)
        cache_metrics.record_set("test", 1.0, False)

        summary = cache_metrics.get_summary()

        # Verify all expected metrics are tracked
        assert summary["test"]["layer"]["hits"] == 1
        assert summary["test"]["_total"]["misses"] == 1
        assert summary["test"]["_total"]["sets"] == 1
        assert summary["test"]["_total"]["errors"] == 1
