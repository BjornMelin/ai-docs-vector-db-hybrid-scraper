"""Unit tests for QueryMonitor with comprehensive coverage.

This test module demonstrates modern testing patterns for query monitoring including:
- Query performance tracking and analysis
- Slow query detection and logging
- Query pattern normalization
- Statistics collection and cleanup
"""

import asyncio
import time
from unittest.mock import patch

import pytest
from src.infrastructure.database.query_monitor import QueryMonitor
from src.infrastructure.database.query_monitor import QueryMonitorConfig
from src.infrastructure.database.query_monitor import QueryStats


class TestQueryStats:
    """Test QueryStats dataclass."""

    def test_query_stats_initialization(self):
        """Test basic initialization of QueryStats."""
        stats = QueryStats(
            query_hash="SELECT * FROM users WHERE id = ?",
            query_pattern="SELECT * FROM users WHERE id = ?",
            execution_count=5,
            total_time_ms=500.0,
            min_time_ms=80.0,
            max_time_ms=150.0,
            avg_time_ms=100.0,
            slow_query_count=1,
            last_execution=time.time(),
        )

        assert stats.query_hash == "SELECT * FROM users WHERE id = ?"
        assert stats.query_pattern == "SELECT * FROM users WHERE id = ?"
        assert stats.execution_count == 5
        assert stats.total_time_ms == 500.0
        assert stats.min_time_ms == 80.0
        assert stats.max_time_ms == 150.0
        assert stats.avg_time_ms == 100.0
        assert stats.slow_query_count == 1
        assert stats.last_execution > 0

    def test_query_stats_with_no_slow_queries(self):
        """Test QueryStats with no slow queries."""
        stats = QueryStats(
            query_hash="SELECT COUNT(*) FROM users",
            query_pattern="SELECT COUNT(*) FROM users",
            execution_count=10,
            total_time_ms=200.0,
            min_time_ms=15.0,
            max_time_ms=25.0,
            avg_time_ms=20.0,
            slow_query_count=0,
            last_execution=time.time(),
        )

        assert stats.slow_query_count == 0
        assert stats.avg_time_ms == 20.0


class TestQueryMonitorConfig:
    """Test QueryMonitorConfig validation and defaults."""

    def test_default_configuration(self):
        """Test default configuration values."""
        config = QueryMonitorConfig()

        assert config.enabled is True
        assert config.slow_query_threshold_ms == 100.0
        assert config.max_tracked_queries == 1000
        assert config.stats_window_hours == 24
        assert config.histogram_buckets == [0.01, 0.05, 0.1, 0.5, 1.0, 5.0, 10.0]
        assert config.log_slow_queries is True
        assert config.track_query_patterns is True

    def test_custom_configuration(self):
        """Test custom configuration values."""
        custom_buckets = [0.1, 0.5, 1.0, 2.0]
        config = QueryMonitorConfig(
            enabled=False,
            slow_query_threshold_ms=200.0,
            max_tracked_queries=500,
            stats_window_hours=12,
            histogram_buckets=custom_buckets,
            log_slow_queries=False,
            track_query_patterns=False,
        )

        assert config.enabled is False
        assert config.slow_query_threshold_ms == 200.0
        assert config.max_tracked_queries == 500
        assert config.stats_window_hours == 12
        assert config.histogram_buckets == custom_buckets
        assert config.log_slow_queries is False
        assert config.track_query_patterns is False

    def test_configuration_validation(self):
        """Test configuration validation for invalid values."""
        # Test negative threshold
        with pytest.raises(ValueError):
            QueryMonitorConfig(slow_query_threshold_ms=-1.0)

        # Test zero max queries
        with pytest.raises(ValueError):
            QueryMonitorConfig(max_tracked_queries=0)

        # Test zero stats window
        with pytest.raises(ValueError):
            QueryMonitorConfig(stats_window_hours=0)


class TestQueryMonitor:
    """Test QueryMonitor functionality."""

    @pytest.fixture
    def config(self):
        """Create test configuration."""
        return QueryMonitorConfig(
            enabled=True,
            slow_query_threshold_ms=100.0,
            max_tracked_queries=10,
            stats_window_hours=1,
            histogram_buckets=[0.1, 0.5, 1.0, 2.0],
            log_slow_queries=True,
            track_query_patterns=True,
        )

    @pytest.fixture
    def query_monitor(self, config):
        """Create QueryMonitor instance."""
        return QueryMonitor(config)

    def test_initialization(self, query_monitor):
        """Test QueryMonitor initialization."""
        assert query_monitor._query_stats == {}
        assert query_monitor._latency_histogram == {}
        assert query_monitor._active_queries == {}
        assert query_monitor._total_queries == 0
        assert query_monitor._slow_queries == 0

    def test_initialization_with_default_config(self):
        """Test initialization with default config."""
        monitor = QueryMonitor()
        assert monitor.config.enabled is True
        assert monitor.config.slow_query_threshold_ms == 100.0

    @pytest.mark.asyncio
    async def test_start_query_enabled(self, query_monitor):
        """Test starting query monitoring when enabled."""
        query = "SELECT * FROM users WHERE id = 1"
        query_id = await query_monitor.start_query(query)

        assert query_id.startswith("query_")
        assert query_id in query_monitor._active_queries
        assert query_monitor._total_queries == 1

    @pytest.mark.asyncio
    async def test_start_query_with_custom_id(self, query_monitor):
        """Test starting query with custom ID."""
        query = "SELECT * FROM products"
        custom_id = "custom_query_123"

        query_id = await query_monitor.start_query(query, custom_id)

        assert query_id == custom_id
        assert custom_id in query_monitor._active_queries

    @pytest.mark.asyncio
    async def test_start_query_disabled(self):
        """Test starting query when monitoring is disabled."""
        config = QueryMonitorConfig(enabled=False)
        monitor = QueryMonitor(config)

        query_id = await monitor.start_query("SELECT 1", "test_id")

        assert query_id == "test_id"
        assert "test_id" not in monitor._active_queries
        assert monitor._total_queries == 0

    @pytest.mark.asyncio
    async def test_end_query_success(self, query_monitor):
        """Test ending query successfully."""
        query = "SELECT * FROM users"
        query_id = await query_monitor.start_query(query)

        # Simulate some execution time
        await asyncio.sleep(0.01)

        execution_time = await query_monitor.end_query(query_id, query, success=True)

        assert execution_time > 0
        assert query_id not in query_monitor._active_queries
        assert len(query_monitor._query_stats) == 1

    @pytest.mark.asyncio
    async def test_end_query_failure(self, query_monitor):
        """Test ending query with failure."""
        query = "SELECT * FROM invalid_table"
        query_id = await query_monitor.start_query(query)

        execution_time = await query_monitor.end_query(query_id, query, success=False)

        assert execution_time > 0
        assert query_id not in query_monitor._active_queries
        # Failed queries should not be recorded in stats
        assert len(query_monitor._query_stats) == 0

    @pytest.mark.asyncio
    async def test_end_query_nonexistent_id(self, query_monitor):
        """Test ending query with non-existent ID."""
        execution_time = await query_monitor.end_query("nonexistent", "SELECT 1", True)

        assert execution_time == 0.0

    @pytest.mark.asyncio
    async def test_record_query_directly(self, query_monitor):
        """Test recording query directly without start/end."""
        query = "SELECT COUNT(*) FROM orders"
        execution_time = 50.0

        await query_monitor.record_query(query, execution_time)

        assert len(query_monitor._query_stats) == 1
        stats = next(iter(query_monitor._query_stats.values()))
        assert stats.execution_count == 1
        assert stats.total_time_ms == execution_time
        assert stats.avg_time_ms == execution_time

    @pytest.mark.asyncio
    async def test_slow_query_detection(self, query_monitor):
        """Test slow query detection and logging."""
        query = "SELECT * FROM large_table"
        slow_execution_time = 150.0  # Above threshold of 100ms

        with patch("src.infrastructure.database.query_monitor.logger") as mock_logger:
            await query_monitor.record_query(query, slow_execution_time)

            # Should log slow query
            mock_logger.warning.assert_called_once()
            assert "Slow query detected" in mock_logger.warning.call_args[0][0]

        # Should increment slow query counter
        assert query_monitor._slow_queries == 1

        # Stats should reflect slow query
        stats = next(iter(query_monitor._query_stats.values()))
        assert stats.slow_query_count == 1

    @pytest.mark.asyncio
    async def test_query_pattern_normalization(self, query_monitor):
        """Test query pattern normalization."""
        queries = [
            "SELECT * FROM users WHERE id = 123",
            "SELECT * FROM users WHERE id = 456",
            "SELECT * FROM users WHERE id = 789",
        ]

        for query in queries:
            await query_monitor.record_query(query, 50.0)

        # Should be normalized to one pattern
        assert len(query_monitor._query_stats) == 1

        stats = next(iter(query_monitor._query_stats.values()))
        assert stats.execution_count == 3
        assert stats.query_pattern == "SELECT * FROM users WHERE id = ?"

    @pytest.mark.asyncio
    async def test_string_literal_normalization(self, query_monitor):
        """Test normalization of string literals."""
        queries = [
            "SELECT * FROM users WHERE name = 'John'",
            "SELECT * FROM users WHERE name = 'Jane'",
            "SELECT * FROM users WHERE name = 'Bob'",
        ]

        for query in queries:
            await query_monitor.record_query(query, 30.0)

        assert len(query_monitor._query_stats) == 1
        stats = next(iter(query_monitor._query_stats.values()))
        assert "?" in stats.query_pattern

    @pytest.mark.asyncio
    async def test_in_clause_normalization(self, query_monitor):
        """Test normalization of IN clauses."""
        queries = [
            "SELECT * FROM orders WHERE status IN ('pending', 'processing')",
            "SELECT * FROM orders WHERE status IN ('completed', 'cancelled', 'refunded')",
        ]

        for query in queries:
            await query_monitor.record_query(query, 40.0)

        assert len(query_monitor._query_stats) == 1
        stats = next(iter(query_monitor._query_stats.values()))
        assert "IN (?)" in stats.query_pattern

    @pytest.mark.asyncio
    async def test_latency_histogram(self, query_monitor):
        """Test latency histogram tracking."""
        # Record queries with different execution times
        test_cases = [
            ("SELECT 1", 50.0),  # 0.05s -> 0.1s bucket
            ("SELECT 2", 300.0),  # 0.3s -> 0.5s bucket
            ("SELECT 3", 800.0),  # 0.8s -> 1.0s bucket
            ("SELECT 4", 1500.0),  # 1.5s -> 2.0s bucket
        ]

        for query, exec_time in test_cases:
            await query_monitor.record_query(query, exec_time)

        histogram = await query_monitor.get_latency_histogram()

        assert histogram["≤0.1s"] == 1
        assert histogram["≤0.5s"] == 1
        assert histogram["≤1.0s"] == 1
        assert histogram["≤2.0s"] == 1

    @pytest.mark.asyncio
    async def test_get_query_stats_sorted(self, query_monitor):
        """Test getting query stats sorted by total time."""
        queries = [
            ("SELECT * FROM users", 100.0),
            ("SELECT * FROM orders", 300.0),
            ("SELECT * FROM products", 200.0),
        ]

        for query, exec_time in queries:
            await query_monitor.record_query(query, exec_time)

        stats = await query_monitor.get_query_stats(limit=3)

        assert len(stats) == 3
        # Should be sorted by total_time_ms descending
        assert (
            stats[0].total_time_ms >= stats[1].total_time_ms >= stats[2].total_time_ms
        )

    @pytest.mark.asyncio
    async def test_get_slow_queries(self, query_monitor):
        """Test getting slow queries only."""
        queries = [
            ("SELECT * FROM fast_table", 50.0),  # Fast query
            ("SELECT * FROM slow_table", 150.0),  # Slow query
            ("SELECT * FROM medium_table", 80.0),  # Fast query
        ]

        for query, exec_time in queries:
            await query_monitor.record_query(query, exec_time)

        slow_queries = await query_monitor.get_slow_queries()

        assert len(slow_queries) == 1
        assert slow_queries[0].slow_query_count > 0
        assert (
            slow_queries[0].avg_time_ms >= query_monitor.config.slow_query_threshold_ms
        )

    @pytest.mark.asyncio
    async def test_get_summary_stats_empty(self, query_monitor):
        """Test getting summary stats with no data."""
        stats = await query_monitor.get_summary_stats()

        assert stats["total_queries"] == 0
        assert stats["slow_queries"] == 0
        assert stats["slow_query_percentage"] == 0.0
        assert stats["unique_queries"] == 0
        assert stats["avg_execution_time_ms"] == 0.0
        assert stats["active_queries"] == 0

    @pytest.mark.asyncio
    async def test_get_summary_stats_with_data(self, query_monitor):
        """Test getting summary stats with data."""
        # Record some queries
        await query_monitor.record_query("SELECT * FROM users", 50.0)
        await query_monitor.record_query("SELECT * FROM orders", 150.0)  # Slow
        await query_monitor.record_query("SELECT * FROM products", 75.0)

        stats = await query_monitor.get_summary_stats()

        assert stats["total_queries"] == 3
        assert stats["slow_queries"] == 1
        assert stats["slow_query_percentage"] == 33.33
        assert stats["unique_queries"] == 3
        assert stats["avg_execution_time_ms"] > 0
        assert stats["active_queries"] == 0

    @pytest.mark.asyncio
    async def test_max_tracked_queries_limit(self, query_monitor):
        """Test that old queries are removed when limit is reached."""
        # Monitor has max_tracked_queries = 10

        # Add more queries than the limit
        for i in range(15):
            query = f"SELECT * FROM table_{i}"
            await query_monitor.record_query(query, 50.0)

        # Should only keep the most recent queries
        assert len(query_monitor._query_stats) == 10

    @pytest.mark.asyncio
    async def test_cleanup_old_stats(self, query_monitor):
        """Test cleanup of old statistics."""
        current_time = time.time()
        old_time = current_time - (2 * 3600)  # 2 hours ago

        # Add some old stats manually
        for i in range(3):
            query_hash = f"SELECT * FROM old_table_{i}"
            query_monitor._query_stats[query_hash] = QueryStats(
                query_hash=query_hash,
                query_pattern=query_hash,
                execution_count=1,
                total_time_ms=100.0,
                min_time_ms=100.0,
                max_time_ms=100.0,
                avg_time_ms=100.0,
                slow_query_count=0,
                last_execution=old_time,
            )

        # Add some recent stats
        await query_monitor.record_query("SELECT * FROM recent_table", 50.0)

        # Cleanup old stats (window is 1 hour)
        cleaned_count = await query_monitor.cleanup_old_stats()

        assert cleaned_count == 3
        assert len(query_monitor._query_stats) == 1

    @pytest.mark.asyncio
    async def test_multiple_executions_same_query(self, query_monitor):
        """Test multiple executions of the same query."""
        query = "SELECT COUNT(*) FROM users"
        execution_times = [45.0, 55.0, 65.0, 120.0, 35.0]  # One slow query

        for exec_time in execution_times:
            await query_monitor.record_query(query, exec_time)

        assert len(query_monitor._query_stats) == 1

        stats = next(iter(query_monitor._query_stats.values()))
        assert stats.execution_count == 5
        assert stats.total_time_ms == sum(execution_times)
        assert stats.min_time_ms == min(execution_times)
        assert stats.max_time_ms == max(execution_times)
        assert stats.avg_time_ms == sum(execution_times) / len(execution_times)
        assert stats.slow_query_count == 1  # Only one execution was slow

    @pytest.mark.asyncio
    async def test_query_pattern_tracking_disabled(self):
        """Test query monitoring with pattern tracking disabled."""
        config = QueryMonitorConfig(track_query_patterns=False)
        monitor = QueryMonitor(config)

        query1 = "SELECT * FROM users WHERE id = 123"
        query2 = "SELECT * FROM users WHERE id = 456"

        await monitor.record_query(query1, 50.0)
        await monitor.record_query(query2, 60.0)

        # Should treat as separate queries without normalization
        assert len(monitor._query_stats) == 2

    @pytest.mark.asyncio
    async def test_disabled_monitoring(self):
        """Test that disabled monitoring doesn't record anything."""
        config = QueryMonitorConfig(enabled=False)
        monitor = QueryMonitor(config)

        await monitor.record_query("SELECT * FROM users", 100.0)

        assert len(monitor._query_stats) == 0
        assert monitor._total_queries == 0
        assert monitor._slow_queries == 0

    @pytest.mark.asyncio
    async def test_histogram_edge_case_very_slow_query(self, query_monitor):
        """Test histogram handling for queries slower than all buckets."""
        very_slow_time = 5000.0  # 5 seconds, exceeds all buckets

        await query_monitor.record_query("SELECT * FROM huge_table", very_slow_time)

        histogram = await query_monitor.get_latency_histogram()

        # Should be placed in the last bucket
        assert histogram["≤2.0s"] == 1


class TestQueryMonitorIntegration:
    """Integration tests for QueryMonitor."""

    @pytest.mark.asyncio
    @pytest.mark.integration
    async def test_full_query_lifecycle(self):
        """Test complete query monitoring lifecycle."""
        config = QueryMonitorConfig(
            slow_query_threshold_ms=100.0, max_tracked_queries=5
        )
        monitor = QueryMonitor(config)

        # Start and end queries
        queries = [
            ("SELECT * FROM users WHERE active = true", 80.0),
            ("SELECT COUNT(*) FROM orders", 45.0),
            ("SELECT * FROM products ORDER BY name", 150.0),  # Slow
            ("UPDATE users SET last_login = NOW()", 200.0),  # Slow
        ]

        for query_text, target_time in queries:
            # Use record_query directly with the target execution time
            # This is more reliable than trying to simulate real execution time
            await monitor.record_query(query_text, target_time)

        # Verify statistics
        summary = await monitor.get_summary_stats()
        assert summary["total_queries"] == 4
        assert summary["slow_queries"] == 2
        assert summary["unique_queries"] == 4

        # Check slow queries
        slow_queries = await monitor.get_slow_queries()
        assert len(slow_queries) == 2

        # Check histogram
        histogram = await monitor.get_latency_histogram()
        assert isinstance(histogram, dict)

        # Check individual stats
        stats = await monitor.get_query_stats()
        assert len(stats) == 4

    @pytest.mark.asyncio
    async def test_concurrent_query_monitoring(self):
        """Test monitoring multiple concurrent queries."""
        monitor = QueryMonitor()

        async def simulate_query(query_text: str, execution_time_ms: float):
            query_id = await monitor.start_query(query_text)
            await asyncio.sleep(execution_time_ms / 1000.0)
            return await monitor.end_query(query_id, query_text, success=True)

        # Start multiple queries concurrently
        tasks = [
            simulate_query("SELECT * FROM table1", 50.0),
            simulate_query("SELECT * FROM table2", 75.0),
            simulate_query("SELECT * FROM table3", 100.0),
            simulate_query("SELECT * FROM table4", 125.0),
        ]

        execution_times = await asyncio.gather(*tasks)

        # All queries should have recorded execution times
        assert all(time > 0 for time in execution_times)
        assert len(monitor._query_stats) == 4

        # Check that concurrent tracking worked
        summary = await monitor.get_summary_stats()
        assert summary["total_queries"] == 4
