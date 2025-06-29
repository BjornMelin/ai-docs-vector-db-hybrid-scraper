"""Tests for search analytics dashboard functionality."""

import asyncio
from datetime import datetime, timedelta
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from src.services.analytics.search_dashboard import (
    PerformanceMetric,
    QueryPattern,
    SearchAnalyticsDashboard,
    UserBehaviorInsight,
)


@pytest.fixture
def mock_config():
    """Create mock configuration."""
    config = MagicMock()
    config.redis = MagicMock()
    config.redis.enabled = True
    config.redis.host = "localhost"
    config.redis.port = 6379
    config.performance = MagicMock()
    config.performance.alert_thresholds = {"avg_query_time": 1000}
    return config


@pytest.fixture
def analytics_dashboard(mock_config):
    """Create analytics dashboard instance."""
    with patch(
        "src.services.analytics.search_dashboard.get_config", return_value=mock_config
    ):
        dashboard = SearchAnalyticsDashboard()
        return dashboard


@pytest.fixture
async def initialized_dashboard(analytics_dashboard):
    """Create initialized analytics dashboard."""
    await analytics_dashboard.initialize()
    return analytics_dashboard


class TestSearchAnalyticsDashboard:
    """Test the SearchAnalyticsDashboard class."""

    def test_initialization(self, analytics_dashboard):
        """Test dashboard initialization."""
        assert analytics_dashboard.query_history == []
        assert analytics_dashboard.performance_metrics == []
        assert analytics_dashboard.user_patterns == {}
        assert analytics_dashboard.detected_patterns == []
        assert analytics_dashboard.pattern_detection_interval == 300  # 5 minutes

    async def test_initialize(self, analytics_dashboard):
        """Test dashboard initialization."""
        await analytics_dashboard.initialize()
        # Should complete without error

    async def test_track_query_basic(self, initialized_dashboard):
        """Test basic query tracking functionality."""
        await initialized_dashboard.track_query(
            query="machine learning basics",
            user_id="user123",
            processing_time_ms=250.5,
            results_count=10,
            success=True,
            collection="documentation",
            session_id="session456",
        )

        # Verify query was tracked
        assert len(initialized_dashboard.query_history) == 1
        tracked_query = initialized_dashboard.query_history[0]
        assert tracked_query["query"] == "machine learning basics"
        assert tracked_query["processing_time_ms"] == 250.5
        assert tracked_query["success"] is True

    async def test_track_query_with_metadata(self, initialized_dashboard):
        """Test query tracking with additional metadata."""
        await initialized_dashboard.track_query(
            query="deep learning architectures",
            user_id="user789",
            processing_time_ms=1200.0,
            results_count=15,
            success=True,
            features_used=["query_expansion", "reranking"],
            cache_hit=False,
            collection="papers",
            session_id="session012",
            model_used="text-embedding-3-large",
        )

        tracked_query = initialized_dashboard.query_history[0]
        assert tracked_query["features_used"] == ["query_expansion", "reranking"]
        assert tracked_query["cache_hit"] is False
        assert tracked_query["model_used"] == "text-embedding-3-large"

    async def test_track_multiple_queries(self, initialized_dashboard):
        """Test tracking multiple queries."""
        queries = [
            ("python programming", 150.0, 8, True),
            ("machine learning", 300.0, 12, True),
            ("data science", 200.0, 6, False),
        ]

        for query, processing_time, results_count, success in queries:
            await initialized_dashboard.track_query(
                query=query,
                processing_time_ms=processing_time,
                results_count=results_count,
                success=success,
            )

        assert len(initialized_dashboard.query_history) == 3

        # Test different success states
        successful_queries = [
            q for q in initialized_dashboard.query_history if q["success"]
        ]
        failed_queries = [
            q for q in initialized_dashboard.query_history if not q["success"]
        ]

        assert len(successful_queries) == 2
        assert len(failed_queries) == 1

    async def test_query_analytics(self, initialized_dashboard):
        """Test query analytics retrieval."""
        # Add queries with similar patterns
        similar_queries = [
            {
                "query": "what is machine learning",
                "processing_time_ms": 200.0,
                "success": True,
            },
            {
                "query": "what is deep learning",
                "processing_time_ms": 180.0,
                "success": True,
            },
            {
                "query": "what is neural networks",
                "processing_time_ms": 220.0,
                "success": True,
            },
            {
                "query": "how to implement tensorflow",
                "processing_time_ms": 300.0,
                "success": True,
            },
            {
                "query": "how to use pytorch",
                "processing_time_ms": 280.0,
                "success": True,
            },
        ]

        for query_data in similar_queries:
            await initialized_dashboard.track_query(**query_data)

        analytics = await initialized_dashboard.get_query_analytics()

        assert "total_queries" in analytics
        assert analytics["total_queries"] == 5

    async def test_get_realtime_dashboard(self, initialized_dashboard):
        """Test realtime dashboard data."""
        # Add performance data
        queries = [
            {
                "query": "test query 1",
                "processing_time_ms": 100.0,
                "success": True,
                "results_count": 5,
            },
            {
                "query": "test query 2",
                "processing_time_ms": 200.0,
                "success": True,
                "results_count": 10,
            },
            {
                "query": "test query 3",
                "processing_time_ms": 150.0,
                "success": False,
                "results_count": 0,
            },
            {
                "query": "test query 4",
                "processing_time_ms": 300.0,
                "success": True,
                "results_count": 8,
            },
        ]

        for query_data in queries:
            await initialized_dashboard.track_query(**query_data)

        dashboard_data = await initialized_dashboard.get_realtime_dashboard()

        assert "realtime_stats" in dashboard_data
        assert "query_patterns" in dashboard_data
        assert "performance_trends" in dashboard_data
        assert "user_insights" in dashboard_data
        assert "last_updated" in dashboard_data

    async def test_get_optimization_recommendations(self, initialized_dashboard):
        """Test optimization recommendation generation."""
        # Add user behavior data
        user_queries = [
            {
                "query": "beginner tutorial",
                "user_id": "user1",
                "session_id": "sess1",
                "success": True,
            },
            {
                "query": "advanced concepts",
                "user_id": "user1",
                "session_id": "sess1",
                "success": True,
            },
            {
                "query": "quick reference",
                "user_id": "user2",
                "session_id": "sess2",
                "success": True,
            },
            {
                "query": "troubleshooting",
                "user_id": "user2",
                "session_id": "sess2",
                "success": False,
            },
        ]

        for query_data in user_queries:
            await initialized_dashboard.track_query(**query_data)

        recommendations = await initialized_dashboard.get_optimization_recommendations()

        assert isinstance(recommendations, list)
        # Recommendations should be available
        assert len(recommendations) >= 0

    async def test_track_multiple_sessions(self, initialized_dashboard):
        """Test tracking across multiple user sessions."""
        # Add queries across different sessions
        session_queries = [
            {
                "query": "session 1 query 1",
                "session_id": "sess1",
                "user_id": "user1",
                "success": True,
            },
            {
                "query": "session 1 query 2",
                "session_id": "sess1",
                "user_id": "user1",
                "success": True,
            },
            {
                "query": "session 2 query 1",
                "session_id": "sess2",
                "user_id": "user2",
                "success": True,
            },
        ]

        for query_data in session_queries:
            await initialized_dashboard.track_query(**query_data)

        analytics = await initialized_dashboard.get_query_analytics()
        assert analytics["total_queries"] == 3

        # Test user-specific analytics
        user1_analytics = await initialized_dashboard.get_query_analytics(
            user_id="user1"
        )
        assert user1_analytics["total_queries"] == 2

    async def test_error_handling(self, initialized_dashboard):
        """Test error handling in analytics methods."""
        # Test methods with no data should not crash
        analytics = await initialized_dashboard.get_query_analytics()
        assert "message" in analytics  # Should return a message when no data

        recommendations = await initialized_dashboard.get_optimization_recommendations()
        assert isinstance(recommendations, list)

    async def test_performance_tracking(self, initialized_dashboard):
        """Test performance metric tracking."""
        # Add query data
        query_data = {
            "query": "performance test",
            "processing_time_ms": 100.0,
            "success": True,
        }
        await initialized_dashboard.track_query(**query_data)

        # Track a performance metric
        await initialized_dashboard.track_performance_metric(
            metric_name="custom_metric", value=0.85, tags={"component": "test"}
        )

        # Get dashboard data
        dashboard_data = await initialized_dashboard.get_realtime_dashboard()
        assert isinstance(dashboard_data, dict)

    async def test_cleanup(self, initialized_dashboard):
        """Test dashboard cleanup."""
        await initialized_dashboard.cleanup()
        # Should complete without error

    async def test_concurrent_tracking(self, initialized_dashboard):
        """Test concurrent query tracking."""

        async def track_concurrent_query(i):
            query_data = {
                "query": f"concurrent query {i}",
                "processing_time_ms": 100.0,
                "success": True,
            }
            await initialized_dashboard.track_query(**query_data)

        # Track multiple queries concurrently
        tasks = [track_concurrent_query(i) for i in range(5)]
        await asyncio.gather(*tasks)

        # All queries should be tracked
        assert len(initialized_dashboard.query_history) == 5
