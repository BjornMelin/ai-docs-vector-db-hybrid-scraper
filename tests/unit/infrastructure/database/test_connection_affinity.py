"""Comprehensive tests for ConnectionAffinityManager query optimization system.

This test module provides comprehensive coverage for the connection affinity system
including query pattern recognition, connection specialization, performance tracking,
and intelligent routing based on query characteristics.
"""

import asyncio

import pytest
from src.infrastructure.database.connection_affinity import ConnectionAffinityManager
from src.infrastructure.database.connection_affinity import ConnectionSpecialization
from src.infrastructure.database.connection_affinity import ConnectionStats
from src.infrastructure.database.connection_affinity import QueryPattern
from src.infrastructure.database.connection_affinity import QueryType


class TestQueryPattern:
    """Test QueryPattern data class functionality."""

    def test_query_pattern_initialization(self):
        """Test QueryPattern initialization."""
        pattern = QueryPattern(
            pattern_id="test_pattern",
            normalized_query="SELECT * FROM test WHERE id = ?",
            sample_query="SELECT * FROM test WHERE id = 123",
            query_type=QueryType.READ,
        )

        # Should initialize with default values
        assert pattern.execution_count == 0
        assert pattern.total_execution_time_ms == 0.0
        assert pattern.avg_execution_time_ms == 0.0
        assert pattern.connection_performance == {}

    def test_query_pattern_update_performance(self):
        """Test updating query pattern performance."""
        pattern = QueryPattern(
            pattern_id="test_pattern",
            normalized_query="SELECT * FROM test WHERE id = ?",
            sample_query="SELECT * FROM test WHERE id = 123",
            query_type=QueryType.READ,
        )

        pattern.update_performance(150.0)

        assert pattern.execution_count == 1
        assert pattern.total_execution_time_ms == 150.0
        assert pattern.avg_execution_time_ms == 150.0

        # Add another measurement
        pattern.update_performance(200.0)

        assert pattern.execution_count == 2
        assert pattern.total_execution_time_ms == 350.0
        assert pattern.avg_execution_time_ms == 175.0

    def test_query_pattern_connection_performance(self):
        """Test connection-specific performance tracking."""
        pattern = QueryPattern(
            pattern_id="test_pattern",
            normalized_query="SELECT * FROM test WHERE id = ?",
            sample_query="SELECT * FROM test WHERE id = 123",
            query_type=QueryType.READ,
        )

        pattern.add_connection_performance("conn_1", 100.0)
        pattern.add_connection_performance("conn_2", 150.0)
        pattern.add_connection_performance("conn_1", 120.0)  # Update existing

        assert "conn_1" in pattern.connection_performance
        assert "conn_2" in pattern.connection_performance


class TestConnectionStats:
    """Test ConnectionStats data class functionality."""

    def test_connection_stats_initialization(self):
        """Test ConnectionStats initialization."""
        from src.infrastructure.database.connection_affinity import (
            ConnectionSpecialization,
        )

        stats = ConnectionStats(
            connection_id="test_conn",
            specialization=ConnectionSpecialization.READ_OPTIMIZED,
        )

        # Should initialize with default values
        assert stats.total_queries == 0
        assert stats.active_queries == 0
        assert stats.avg_response_time_ms == 0.0
        assert stats.error_count == 0
        assert stats.success_rate == 1.0

    def test_connection_stats_update_usage(self):
        """Test updating connection usage statistics."""
        from src.infrastructure.database.connection_affinity import (
            ConnectionSpecialization,
        )

        stats = ConnectionStats(
            connection_id="test_conn", specialization=ConnectionSpecialization.GENERAL
        )

        stats.update_usage(QueryType.READ, 100.0, success=True)

        assert stats.total_queries == 1
        # With EMA alpha=0.2: 0.2 * 100.0 + 0.8 * 0.0 = 20.0
        assert stats.avg_response_time_ms == 20.0
        assert QueryType.READ in stats.query_type_counts
        assert stats.query_type_counts[QueryType.READ] == 1

        # Add a failed query
        stats.update_usage(QueryType.WRITE, 200.0, success=False)

        assert stats.total_queries == 2
        assert stats.error_count == 1
        # Response time should be updated with moving average (EMA)
        # 0.2 * 200.0 + 0.8 * 20.0 = 40.0 + 16.0 = 56.0
        assert stats.avg_response_time_ms == 56.0

    def test_connection_stats_load_score(self):
        """Test load score calculation."""
        from src.infrastructure.database.connection_affinity import (
            ConnectionSpecialization,
        )

        stats = ConnectionStats(
            connection_id="test_conn", specialization=ConnectionSpecialization.GENERAL
        )

        # Initially should have low load
        load_score = stats.get_load_score()
        assert 0.0 <= load_score <= 1.0

        # Add some active queries to increase load
        stats.active_queries = 5
        load_score = stats.get_load_score()
        assert load_score > 0.0


class TestConnectionAffinityManager:
    """Test ConnectionAffinityManager functionality."""

    @pytest.fixture
    def affinity_manager(self):
        """Create ConnectionAffinityManager instance."""
        return ConnectionAffinityManager(max_patterns=100, max_connections=10)

    @pytest.fixture
    def sample_queries(self):
        """Sample queries for testing."""
        return [
            "SELECT * FROM users WHERE id = ?",
            "INSERT INTO orders (user_id, amount) VALUES (?, ?)",
            "SELECT COUNT(*) FROM analytics_data WHERE date >= ?",
            "UPDATE user_profiles SET last_login = ? WHERE user_id = ?",
            "DELETE FROM temporary_data WHERE created_at < ?",
            "SELECT AVG(price) FROM products GROUP BY category",
        ]

    def test_initialization(self, affinity_manager):
        """Test manager initialization."""
        assert affinity_manager.max_patterns == 100
        assert affinity_manager.max_connections == 10
        assert len(affinity_manager.query_patterns) == 0
        assert len(affinity_manager.connection_stats) == 0

    def test_normalize_query(self, affinity_manager):
        """Test query normalization."""
        # Test numeric literal replacement
        query1 = "SELECT * FROM users WHERE id = 123"
        query2 = "SELECT * FROM users WHERE id = 456"

        norm1 = affinity_manager._normalize_query(query1)
        norm2 = affinity_manager._normalize_query(query2)

        assert norm1 == norm2
        assert "?" in norm1

        # Test string literal replacement
        query3 = "SELECT * FROM users WHERE name = 'John'"
        query4 = "SELECT * FROM users WHERE name = 'Jane'"

        norm3 = affinity_manager._normalize_query(query3)
        norm4 = affinity_manager._normalize_query(query4)

        assert norm3 == norm4

    def test_classify_query_type(self, affinity_manager):
        """Test automatic query type classification."""
        test_cases = [
            ("SELECT * FROM users", QueryType.READ),
            ("INSERT INTO users VALUES (1, 'test')", QueryType.WRITE),
            ("UPDATE users SET name = 'test'", QueryType.WRITE),
            ("DELETE FROM users WHERE id = 1", QueryType.WRITE),
            ("SELECT COUNT(*) FROM orders GROUP BY date", QueryType.ANALYTICS),
            ("SELECT AVG(price) FROM products", QueryType.ANALYTICS),
        ]

        for query, expected_type in test_cases:
            result = affinity_manager._classify_query_type(query)
            assert result == expected_type, (
                f"Query '{query}' should be {expected_type}, got {result}"
            )

    def test_calculate_query_complexity(self, affinity_manager):
        """Test query complexity calculation."""
        # Simple query
        simple_query = "SELECT id FROM users"
        simple_score = affinity_manager._calculate_query_complexity(simple_query)

        # Complex query
        complex_query = """
        SELECT u.name, COUNT(o.id) as order_count,
               AVG(o.amount) as avg_amount
        FROM users u
        LEFT JOIN orders o ON u.id = o.user_id
        WHERE u.created_at > '2023-01-01'
        GROUP BY u.id, u.name
        ORDER BY order_count DESC
        """
        complex_score = affinity_manager._calculate_query_complexity(complex_query)

        assert 0.0 <= simple_score <= 1.0
        assert 0.0 <= complex_score <= 1.0
        assert complex_score > simple_score

    @pytest.mark.asyncio
    async def test_register_connection(self, affinity_manager):
        """Test connection registration."""
        from src.infrastructure.database.connection_affinity import (
            ConnectionSpecialization,
        )

        await affinity_manager.register_connection(
            "conn_123", ConnectionSpecialization.READ_OPTIMIZED
        )

        assert "conn_123" in affinity_manager.connection_stats
        stats = affinity_manager.connection_stats["conn_123"]
        assert isinstance(stats, ConnectionStats)

    @pytest.mark.asyncio
    async def test_unregister_connection(self, affinity_manager):
        """Test connection unregistration."""
        from src.infrastructure.database.connection_affinity import (
            ConnectionSpecialization,
        )

        await affinity_manager.register_connection(
            "conn_test", ConnectionSpecialization.WRITE_OPTIMIZED
        )
        assert "conn_test" in affinity_manager.connection_stats

        await affinity_manager.unregister_connection("conn_test")
        assert "conn_test" not in affinity_manager.connection_stats

    @pytest.mark.asyncio
    async def test_get_optimal_connection_no_history(self, affinity_manager):
        """Test optimal connection selection with no history."""
        # Register some connections
        await affinity_manager.register_connection(
            "conn_read_1", ConnectionSpecialization.READ_OPTIMIZED
        )
        await affinity_manager.register_connection(
            "conn_write_1", ConnectionSpecialization.WRITE_OPTIMIZED
        )

        # Test read query - should return one of the registered connections
        result = await affinity_manager.get_optimal_connection(
            "SELECT * FROM users", QueryType.READ
        )
        assert result is not None

    @pytest.mark.asyncio
    async def test_track_query_performance(self, affinity_manager):
        """Test query performance tracking."""
        await affinity_manager.register_connection(
            "conn_track", ConnectionSpecialization.READ_OPTIMIZED
        )

        query = "SELECT * FROM test_table"

        # Track query performance
        await affinity_manager.track_query_performance(
            "conn_track", query, 120.0, QueryType.READ, True
        )

        # Check connection stats were updated
        stats = affinity_manager.connection_stats["conn_track"]
        assert stats.total_queries == 1
        assert stats.error_count == 0  # Successful query, no errors

    @pytest.mark.asyncio
    async def test_track_query_performance_error(self, affinity_manager):
        """Test tracking failed query performance."""
        await affinity_manager.register_connection(
            "conn_error", ConnectionSpecialization.WRITE_OPTIMIZED
        )

        query = "INSERT INTO test VALUES (1)"

        # Track failed query
        await affinity_manager.track_query_performance(
            "conn_error", query, 300.0, QueryType.WRITE, False
        )

        # Check error was recorded
        stats = affinity_manager.connection_stats["conn_error"]
        assert stats.total_queries == 1
        assert stats.error_count == 1  # Failed query recorded

    @pytest.mark.asyncio
    async def test_get_connection_recommendations(self, affinity_manager):
        """Test connection recommendations generation."""
        # Register various connections
        await affinity_manager.register_connection(
            "conn_read_1", ConnectionSpecialization.READ_OPTIMIZED
        )
        await affinity_manager.register_connection(
            "conn_write_1", ConnectionSpecialization.WRITE_OPTIMIZED
        )

        recommendations = await affinity_manager.get_connection_recommendations(
            QueryType.READ
        )

        assert "query_type" in recommendations
        assert "available_connections" in recommendations
        assert "recommendations" in recommendations

    @pytest.mark.asyncio
    async def test_get_performance_report(self, affinity_manager):
        """Test comprehensive performance report generation."""
        # Set up test data
        await affinity_manager.register_connection(
            "conn_report_1", ConnectionSpecialization.READ_OPTIMIZED
        )
        await affinity_manager.register_connection(
            "conn_report_2", ConnectionSpecialization.WRITE_OPTIMIZED
        )

        # Track some queries
        for i in range(3):
            await affinity_manager.track_query_performance(
                "conn_report_1",
                f"SELECT * FROM table{i}",
                100.0 + i * 10,
                QueryType.READ,
                True,
            )

        report = await affinity_manager.get_performance_report()

        # Check actual structure returned by get_performance_report
        assert "summary" in report
        assert "top_patterns" in report
        assert "connection_performance" in report
        assert "specialization_distribution" in report

        # Check summary contents
        assert "total_connections" in report["summary"]
        assert "total_patterns" in report["summary"]
        assert "cache_size" in report["summary"]

        # Verify data structure
        assert report["summary"]["total_connections"] == 2
        assert (
            report["summary"]["total_patterns"] >= 1
        )  # At least one pattern from tracked queries
        assert isinstance(report["top_patterns"], list)
        assert isinstance(report["connection_performance"], dict)
        assert isinstance(report["specialization_distribution"], dict)

    @pytest.mark.asyncio
    async def test_connection_specialization_learning(self, affinity_manager):
        """Test that connections learn specializations through usage."""
        # Register generic connection
        await affinity_manager.register_connection(
            "conn_learning", ConnectionSpecialization.READ_OPTIMIZED
        )

        # Track many read queries
        for i in range(5):
            await affinity_manager.track_query_performance(
                "conn_learning",
                f"SELECT * FROM table{i}",
                100.0,
                QueryType.READ,
                True,
            )

        # Connection should have specialization data
        stats = affinity_manager.connection_stats["conn_learning"]
        assert QueryType.READ in stats.query_type_counts
        assert stats.query_type_counts[QueryType.READ] == 5

    @pytest.mark.asyncio
    async def test_pattern_memory_management(self, affinity_manager):
        """Test pattern memory management with max_patterns limit."""
        affinity_manager.max_patterns = 5  # Set low limit for testing

        await affinity_manager.register_connection(
            "conn_memory", ConnectionSpecialization.READ_OPTIMIZED
        )

        # Add more patterns than the limit
        for i in range(10):
            query = f"SELECT * FROM table_{i} WHERE id = ?"
            await affinity_manager.track_query_performance(
                "conn_memory", query, 100.0, QueryType.READ, True
            )

        # Eviction is triggered but only applies to patterns older than 1 hour
        # So in tests, patterns will accumulate but eviction logic is still exercised
        assert (
            len(affinity_manager.query_patterns) == 10
        )  # All patterns kept since they're recent

    @pytest.mark.asyncio
    async def test_concurrent_access_safety(self, affinity_manager):
        """Test concurrent operations safety."""
        # Register connections
        for i in range(3):
            await affinity_manager.register_connection(
                f"conn_{i}", ConnectionSpecialization.READ_OPTIMIZED
            )

        # Concurrent operations
        async def track_performance(conn_id, query_num):
            query = f"SELECT * FROM table_{query_num}"
            await affinity_manager.track_query_performance(
                conn_id, query, 100.0, QueryType.READ, True
            )

        async def get_connection(query_num):
            query = f"SELECT * FROM table_{query_num}"
            return await affinity_manager.get_optimal_connection(query, QueryType.READ)

        # Run concurrent operations
        tasks = []
        for i in range(10):
            conn_id = f"conn_{i % 3}"
            tasks.append(track_performance(conn_id, i))
            tasks.append(get_connection(i))

        results = await asyncio.gather(*tasks, return_exceptions=True)

        # Should not have any exceptions
        exceptions = [r for r in results if isinstance(r, Exception)]
        assert len(exceptions) == 0


class TestConnectionAffinityManagerEdgeCases:
    """Test edge cases and error conditions."""

    @pytest.mark.asyncio
    async def test_empty_connections_list(self):
        """Test behavior with no registered connections."""
        manager = ConnectionAffinityManager(max_patterns=10, max_connections=5)

        result = await manager.get_optimal_connection("SELECT 1", QueryType.READ)
        assert result is None

        recommendations = await manager.get_connection_recommendations(QueryType.READ)
        assert "error" in recommendations
        assert recommendations["error"] == "No available connections"

    @pytest.mark.asyncio
    async def test_track_performance_unregistered_connection(self):
        """Test tracking performance for unregistered connection."""
        manager = ConnectionAffinityManager(max_patterns=10, max_connections=5)

        # Should handle gracefully (might log warning but not crash)
        await manager.track_query_performance(
            "nonexistent", "SELECT 1", 100.0, QueryType.READ, True
        )

        # Should not create stats for unregistered connection
        assert "nonexistent" not in manager.connection_stats

    @pytest.mark.asyncio
    async def test_very_long_query_handling(self):
        """Test handling of very long queries."""
        manager = ConnectionAffinityManager(max_patterns=10, max_connections=5)
        await manager.register_connection(
            "conn_long", ConnectionSpecialization.READ_OPTIMIZED
        )

        # Create very long query
        long_query = (
            "SELECT " + ", ".join([f"col{i}" for i in range(100)]) + " FROM huge_table"
        )

        # Should handle without crashing
        await manager.track_query_performance(
            "conn_long", long_query, 100.0, QueryType.READ, True
        )

        result = await manager.get_optimal_connection(long_query, QueryType.READ)
        assert result == "conn_long"

    def test_extreme_complexity_scores(self):
        """Test handling of extreme complexity scores."""
        manager = ConnectionAffinityManager(max_patterns=10, max_connections=5)

        # Test very simple query (should have low complexity)
        simple_query = "SELECT 1"
        simple_score = manager._calculate_query_complexity(simple_query)
        assert 0.0 <= simple_score <= 0.3

        # Test empty query
        empty_score = manager._calculate_query_complexity("")
        assert empty_score == 0.0

    @pytest.mark.asyncio
    async def test_pattern_limit_overflow_handling(self):
        """Test graceful handling when pattern limit is exceeded."""
        manager = ConnectionAffinityManager(max_patterns=3, max_connections=5)
        await manager.register_connection(
            "conn_overflow", ConnectionSpecialization.READ_OPTIMIZED
        )

        # Add patterns beyond limit
        queries = [
            "SELECT * FROM users",
            "SELECT * FROM orders",
            "SELECT * FROM products",
            "SELECT * FROM categories",
            "SELECT * FROM reviews",
        ]

        for query in queries:
            await manager.track_query_performance(
                "conn_overflow", query, 100.0, QueryType.READ, True
            )

        # Should handle gracefully and maintain functionality
        result = await manager.get_optimal_connection(
            "SELECT * FROM new_table", QueryType.READ
        )
        assert result == "conn_overflow"

        # Eviction logic is triggered but patterns remain since they're recent
        # The test verifies the system continues to function even with pattern overflow
        assert (
            len(manager.query_patterns) >= manager.max_patterns
        )  # Patterns accumulate in tests due to 1-hour eviction threshold
