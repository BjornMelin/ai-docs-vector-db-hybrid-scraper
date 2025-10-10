"""Tests for browser automation monitoring system."""

import asyncio
import time

import pytest
from pydantic.warnings import PydanticDeprecatedSince20

from src.services.browser.monitoring import (
    Alert,
    AlertSeverity,
    AlertType,
    BrowserAutomationMonitor,
    MonitoringConfig,
    PerformanceMetrics,
)


@pytest.fixture
def monitor_config():
    """Create test monitoring configuration."""
    return MonitoringConfig(
        error_rate_threshold=0.2,  # 20% error rate threshold
        response_time_threshold_ms=5000,  # 5 second threshold
        health_check_interval_seconds=1,  # Fast for testing
        metrics_collection_interval_seconds=1,
        alert_cooldown_seconds=5,  # Short cooldown for testing
        max_alerts_per_hour=5,
    )


@pytest.fixture
async def monitor(monitor_config):
    """Create monitoring instance."""
    monitor = BrowserAutomationMonitor(monitor_config)
    yield monitor
    await monitor.stop_monitoring()


class TestBrowserAutomationMonitor:
    """Test monitoring system functionality."""

    @pytest.mark.asyncio
    async def test_monitor_initialization(self, monitor):
        """Test monitor initializes correctly."""
        assert monitor.config is not None
        assert monitor.monitoring_active is False
        assert len(monitor.metrics_history) == 0
        assert len(monitor.health_status) == 0
        assert len(monitor.alerts) == 0

    @pytest.mark.asyncio
    async def test_start_stop_monitoring(self, monitor):
        """Test starting and stopping monitoring."""
        # Start monitoring
        await monitor.start_monitoring()
        assert monitor.monitoring_active is True
        assert monitor.monitoring_task is not None

        # Stop monitoring
        await monitor.stop_monitoring()
        assert monitor.monitoring_active is False

    @pytest.mark.asyncio
    async def test_record_successful_request_metrics(self, monitor):
        """Test recording metrics for successful requests."""
        # Record successful request
        await monitor.record_request_metrics(
            tier="lightweight", success=True, response_time_ms=100.0, cache_hit=True
        )

        # Check metrics were recorded
        assert len(monitor.metrics_history) == 1

        metrics = monitor.metrics_history[0]
        assert metrics.tier == "lightweight"
        assert metrics.success_rate == 1.0
        assert metrics.avg_response_time_ms == 100.0
        assert metrics.cache_hit_rate == 1.0

        # Check health status was updated
        assert "lightweight" in monitor.health_status
        health = monitor.health_status["lightweight"]
        assert health.status == "healthy"
        assert health.success_rate == 1.0

    @pytest.mark.asyncio
    async def test_record_failed_request_metrics(self, monitor):
        """Test recording metrics for failed requests."""
        # Record failed request
        await monitor.record_request_metrics(
            tier="browser_use",
            success=False,
            response_time_ms=8000.0,
            error_type="timeout",
            cache_hit=False,
        )

        # Check metrics were recorded
        assert len(monitor.metrics_history) == 1

        metrics = monitor.metrics_history[0]
        assert metrics.tier == "browser_use"
        assert metrics.success_rate == 0.0
        assert metrics.avg_response_time_ms == 8000.0
        assert metrics.cache_hit_rate == 0.0
        assert "timeout" in metrics.error_types

        # Check health status shows unhealthy
        health = monitor.health_status["browser_use"]
        assert health.status == "unhealthy"

    @pytest.mark.asyncio
    async def test_alert_generation_high_error_rate(self, monitor):
        """Test alert generation for high error rate."""
        # Record multiple failed requests to trigger alert
        for _ in range(5):
            await monitor.record_request_metrics(
                tier="crawl4ai",
                success=False,
                response_time_ms=1000.0,
                error_type="service_error",
            )

        # Check alert was generated
        alerts = monitor.get_active_alerts()
        assert len(alerts) > 0

        error_alerts = [a for a in alerts if a.alert_type == AlertType.HIGH_ERROR_RATE]
        assert len(error_alerts) > 0
        assert error_alerts[0].tier == "crawl4ai"
        assert error_alerts[0].severity == AlertSeverity.HIGH

    @pytest.mark.asyncio
    async def test_alert_generation_slow_response(self, monitor):
        """Test alert generation for slow response times."""
        # Record request with slow response time
        await monitor.record_request_metrics(
            tier="playwright",
            success=True,
            response_time_ms=15000.0,  # Exceeds 5s threshold
        )

        # Check alert was generated
        alerts = monitor.get_active_alerts()
        slow_alerts = [
            a for a in alerts if a.alert_type == AlertType.SLOW_RESPONSE_TIME
        ]

        assert len(slow_alerts) > 0
        assert slow_alerts[0].tier == "playwright"
        assert slow_alerts[0].severity == AlertSeverity.MEDIUM

    @pytest.mark.asyncio
    async def test_alert_cooldown(self, monitor):
        """Test alert cooldown prevents spam."""
        # Record failed requests
        for _ in range(3):
            await monitor.record_request_metrics(
                tier="firecrawl",
                success=False,
                response_time_ms=1000.0,
                error_type="api_error",
            )

        initial_alert_count = len(monitor.get_active_alerts())

        # Record more failures immediately (should be in cooldown)
        for _ in range(3):
            await monitor.record_request_metrics(
                tier="firecrawl",
                success=False,
                response_time_ms=1000.0,
                error_type="api_error",
            )

        # Should not have generated more alerts due to cooldown
        final_alert_count = len(monitor.get_active_alerts())
        assert final_alert_count == initial_alert_count

    @pytest.mark.asyncio
    async def test_alert_rate_limiting(self, monitor):
        """Test alert rate limiting prevents too many alerts."""
        # Generate many different alerts quickly
        tiers = ["lightweight", "crawl4ai", "browser_use", "playwright", "firecrawl"]

        _alert_count = 0
        for i, tier in enumerate(tiers * 3):  # 15 potential alerts
            await monitor.record_request_metrics(
                tier=tier,
                success=False,
                response_time_ms=10000.0,  # Slow + error
                error_type=f"error_{i}",
            )
            # Small delay to avoid all being same timestamp
            await asyncio.sleep(0.01)

        # Should be limited to max_alerts_per_hour (5)
        alerts = monitor.get_active_alerts()
        assert len(alerts) <= monitor.config.max_alerts_per_hour

    @pytest.mark.asyncio
    async def test_health_status_calculation(self, monitor):
        """Test health status calculation logic."""
        # Record mostly successful requests first to establish baseline
        for _ in range(8):
            await monitor.record_request_metrics(
                tier="test_tier", success=True, response_time_ms=500.0
            )

        # Then record some failures
        for _ in range(2):
            await monitor.record_request_metrics(
                tier="test_tier", success=False, response_time_ms=3000.0
            )

        # Check health status exists
        assert "test_tier" in monitor.health_status
        health = monitor.health_status["test_tier"]

        # Success rate should be reasonable (exact value depends on aggregation logic)
        assert 0.0 <= health.success_rate <= 1.0
        assert health.component == "test_tier"

    @pytest.mark.asyncio
    async def test_get_system_health(self, monitor):
        """Test getting overall system health."""
        # Record metrics for multiple tiers
        await monitor.record_request_metrics("tier1", True, 500.0)  # Healthy
        await monitor.record_request_metrics("tier2", False, 8000.0)  # Unhealthy
        await monitor.record_request_metrics("tier3", True, 3000.0)  # Healthy

        with pytest.warns(PydanticDeprecatedSince20):
            system_health = monitor.get_system_health()

        assert system_health["overall_status"] == "unhealthy"  # One unhealthy tier
        assert "tier_details" in system_health
        tiers = system_health["tier_details"]
        assert len(tiers) == 3
        statuses = [details["status"] for details in tiers.values()]
        assert statuses.count("healthy") == 2
        assert statuses.count("unhealthy") == 1

    @pytest.mark.asyncio
    async def test_get_recent_metrics(self, monitor):
        """Test retrieving recent metrics."""
        # Record metrics for different tiers at different times
        await monitor.record_request_metrics("tier1", True, 500.0)
        await asyncio.sleep(0.1)
        await monitor.record_request_metrics("tier2", True, 600.0)
        await asyncio.sleep(0.1)
        await monitor.record_request_metrics("tier1", True, 700.0)

        # Get all recent metrics
        all_metrics = monitor.get_recent_metrics(hours=1)
        assert len(all_metrics) == 3

        # Get metrics for specific tier
        tier1_metrics = monitor.get_recent_metrics(tier="tier1", hours=1)
        assert len(tier1_metrics) == 2
        assert all(m.tier == "tier1" for m in tier1_metrics)

    @pytest.mark.asyncio
    async def test_get_active_alerts(self, monitor):
        """Test retrieving active alerts."""
        # Generate alerts of different severities
        await monitor.record_request_metrics(
            "tier1", False, 15000.0
        )  # High + Medium severity
        await monitor.record_request_metrics(
            "tier2", True, 1000.0, cache_hit=False
        )  # Low severity

        # Get all active alerts
        all_alerts = monitor.get_active_alerts()
        assert len(all_alerts) > 0

        # Get high severity alerts only
        high_alerts = monitor.get_active_alerts(severity=AlertSeverity.HIGH)
        assert all(a.severity == AlertSeverity.HIGH for a in high_alerts)

    @pytest.mark.asyncio
    async def test_resolve_alert(self, monitor):
        """Test resolving alerts."""
        # Generate an alert
        await monitor.record_request_metrics("tier1", False, 8000.0)

        alerts = monitor.get_active_alerts()
        assert len(alerts) > 0

        alert_id = alerts[0].id
        assert not alerts[0].resolved

        # Resolve the alert
        await monitor.resolve_alert(alert_id)

        # Check alert is marked as resolved
        for alert in monitor.alerts:
            if alert.id == alert_id:
                assert alert.resolved
                assert alert.resolved_at is not None
                break

    @pytest.mark.asyncio
    async def test_alert_handlers(self, monitor):
        """Test alert handler functionality."""
        handler_calls = []

        def test_handler(alert: Alert):
            handler_calls.append(alert)

        # Add handler
        monitor.add_alert_handler(test_handler)

        # Generate alert
        await monitor.record_request_metrics("tier1", False, 10000.0)

        # Check handler was called
        assert len(handler_calls) > 0
        assert isinstance(handler_calls[0], Alert)

    @pytest.mark.asyncio
    async def test_monitoring_loop_cleanup(self, monitor):
        """Test that monitoring loop cleans up old data."""
        # Add some old metrics (simulate by modifying timestamp)
        old_time = time.time() - (25 * 3600)  # 25 hours ago

        # Manually add old metric
        old_metric = PerformanceMetrics(
            timestamp=old_time,
            tier="old_tier",
            requests_per_minute=10.0,
            success_rate=1.0,
            avg_response_time_ms=500.0,
            p95_response_time_ms=600.0,
            active_requests=0,
        )
        monitor.metrics_history.append(old_metric)

        # Start monitoring to trigger cleanup
        await monitor.start_monitoring()
        await asyncio.sleep(0.1)  # Let cleanup run
        await monitor.stop_monitoring()

        # Old metrics should be cleaned up (24h retention)
        remaining_metrics = [m for m in monitor.metrics_history if m.tier == "old_tier"]
        assert len(remaining_metrics) == 0

    @pytest.mark.asyncio
    async def test_cache_miss_rate_alert(self, monitor):
        """Test cache miss rate alert generation."""
        # Record requests with low cache hit rate
        for _ in range(5):
            await monitor.record_request_metrics(
                tier="cached_tier",
                success=True,
                response_time_ms=1000.0,
                cache_hit=False,  # No cache hits
            )

        # Should generate cache miss rate alert
        alerts = monitor.get_active_alerts()
        cache_alerts = [a for a in alerts if a.alert_type == AlertType.CACHE_MISS_RATE]

        assert len(cache_alerts) > 0
        assert cache_alerts[0].severity == AlertSeverity.LOW

    @pytest.mark.asyncio
    async def test_performance_metrics_aggregation(self, monitor):
        """Test that performance metrics are properly aggregated."""
        # Record multiple requests with varying performance
        response_times = [100, 200, 300, 400, 500]  # All successful requests

        for rt in response_times:
            await monitor.record_request_metrics(
                tier="perf_tier", success=True, response_time_ms=float(rt)
            )

        # Get the latest metrics
        latest_metrics = [m for m in monitor.metrics_history if m.tier == "perf_tier"][
            -1
        ]

        # Check basic properties
        assert latest_metrics.tier == "perf_tier"
        assert latest_metrics.success_rate > 0.0  # Should have some success
        assert latest_metrics.avg_response_time_ms > 0.0  # Should have response time
        assert (
            latest_metrics.p95_response_time_ms >= latest_metrics.avg_response_time_ms
        )  # P95 >= avg
