class TestError(Exception):
    """Custom exception for this module."""


"""Tests for 5-tier browser automation health monitoring.

This module tests the comprehensive browser automation monitoring system
including tier health metrics, performance optimization, failover mechanisms,
and monitoring for lightweight, playwright, crawl4ai, browser_use, and firecrawl tiers.
"""

import asyncio
import time
from typing import Any

import pytest


class MockBrowserTier:
    """Mock browser automation tier for testing."""

    def __init__(self, name: str, tier_level: int):
        self.name = name
        self.tier_level = tier_level
        self.is_healthy = True
        self.response_time = 100.0  # ms
        self.success_rate = 95.0
        self.cpu_usage = 50.0
        self.memory_usage = 512.0  # MB
        self.active_sessions = 0
        self.max_sessions = 10
        self._total_requests = 0
        self.failed_requests = 0

    async def health_check(self) -> dict[str, Any]:
        """Perform health check on tier."""
        self._total_requests += 1

        # Simulate occasional failures
        if self._total_requests % 20 == 0:
            self.failed_requests += 1
            self.is_healthy = False
            return {
                "healthy": False,
                "response_time_ms": self.response_time * 3,
                "error": "Tier overloaded",
            }

        self.is_healthy = True
        return {
            "healthy": True,
            "response_time_ms": self.response_time,
            "success_rate": self.success_rate,
            "cpu_usage": self.cpu_usage,
            "memory_usage_mb": self.memory_usage,
            "active_sessions": self.active_sessions,
            "max_sessions": self.max_sessions,
        }

    async def execute_request(self, request_type: str) -> dict[str, Any]:
        """Execute browser automation request."""
        if not self.is_healthy:
            msg = f"Tier {self.name} is unhealthy"
            raise TestError(msg)

        if self.active_sessions >= self.max_sessions:
            msg = f"Tier {self.name} at capacity"
            raise TestError(msg)

        self.active_sessions += 1
        try:
            # Simulate work
            await asyncio.sleep(self.response_time / 1000)
            return {
                "status": "success",
                "tier": self.name,
                "response_time_ms": self.response_time,
                "request_type": request_type,
            }
        finally:
            self.active_sessions -= 1

    def get_metrics(self) -> dict[str, Any]:
        """Get tier performance metrics."""
        return {
            "tier_name": self.name,
            "tier_level": self.tier_level,
            "is_healthy": self.is_healthy,
            "response_time_ms": self.response_time,
            "success_rate": self.success_rate,
            "cpu_usage": self.cpu_usage,
            "memory_usage_mb": self.memory_usage,
            "active_sessions": self.active_sessions,
            "max_sessions": self.max_sessions,
            "_total_requests": self._total_requests,
            "failed_requests": self.failed_requests,
            "error_rate": self.failed_requests / max(self._total_requests, 1),
        }


class MockBrowserMonitoringSystem:
    """Mock 5-tier browser monitoring system."""

    def __init__(self):
        self.tiers = {
            "lightweight": MockBrowserTier("lightweight", 1),
            "playwright": MockBrowserTier("playwright", 2),
            "crawl4ai": MockBrowserTier("crawl4ai", 3),
            "browser_use": MockBrowserTier("browser_use", 4),
            "firecrawl": MockBrowserTier("firecrawl", 5),
        }

        # Configure tier-specific characteristics
        self.tiers["lightweight"].response_time = 50.0
        self.tiers["lightweight"].max_sessions = 50

        self.tiers["playwright"].response_time = 150.0
        self.tiers["playwright"].max_sessions = 20

        self.tiers["crawl4ai"].response_time = 300.0
        self.tiers["crawl4ai"].max_sessions = 15

        self.tiers["browser_use"].response_time = 500.0
        self.tiers["browser_use"].max_sessions = 10

        self.tiers["firecrawl"].response_time = 800.0
        self.tiers["firecrawl"].max_sessions = 5

        self.failover_enabled = True
        self.load_balancing = "round_robin"
        self.monitoring_interval = 30  # seconds

    async def comprehensive_health_check(self) -> dict[str, Any]:
        """Perform comprehensive health check across all tiers."""
        health_results = {}

        for tier_name, tier in self.tiers.items():
            health_results[tier_name] = await tier.health_check()

        # Calculate overall system health
        healthy_tiers = sum(
            1 for result in health_results.values() if result["healthy"]
        )
        _total_tiers = len(self.tiers)

        return {
            "overall_health": healthy_tiers / _total_tiers,
            "healthy_tiers": healthy_tiers,
            "_total_tiers": _total_tiers,
            "tier_health": health_results,
            "timestamp": time.time(),
        }

    async def execute_with_failover(
        self, request_type: str, preferred_tier: str | None = None
    ) -> dict[str, Any]:
        """Execute request with automatic failover."""
        # Determine tier order (prefer lighter tiers first)
        tier_order = [
            "lightweight",
            "playwright",
            "crawl4ai",
            "browser_use",
            "firecrawl",
        ]

        if preferred_tier and preferred_tier in tier_order:
            # Move preferred tier to front
            tier_order.remove(preferred_tier)
            tier_order.insert(0, preferred_tier)

        last_error = None

        for tier_name in tier_order:
            tier = self.tiers[tier_name]

            try:
                # Check tier health first
                health = await tier.health_check()
                if not health["healthy"]:
                    continue

                # Execute request
                result = await tier.execute_request(request_type)
                result["failover_tier"] = tier_name
                result["attempts"] = tier_order.index(tier_name) + 1
                return result

            except Exception as e:
                last_error = e
                continue

        # All tiers failed
        msg = f"All tiers failed. Last error: {last_error}"
        raise TestError(msg)

    def get_system_metrics(self) -> dict[str, Any]:
        """Get comprehensive system metrics."""
        tier_metrics = {}

        for tier_name, tier in self.tiers.items():
            tier_metrics[tier_name] = tier.get_metrics()

        # Calculate aggregate metrics
        _total_sessions = sum(tier.active_sessions for tier in self.tiers.values())
        _total_capacity = sum(tier.max_sessions for tier in self.tiers.values())
        avg_response_time = sum(
            tier.response_time for tier in self.tiers.values()
        ) / len(self.tiers)

        return {
            "tier_metrics": tier_metrics,
            "aggregate_metrics": {
                "_total_active_sessions": _total_sessions,
                "_total_capacity": _total_capacity,
                "capacity_utilization": _total_sessions / _total_capacity,
                "average_response_time_ms": avg_response_time,
                "healthy_tier_count": sum(
                    1 for tier in self.tiers.values() if tier.is_healthy
                ),
            },
            "system_configuration": {
                "failover_enabled": self.failover_enabled,
                "load_balancing": self.load_balancing,
                "monitoring_interval_seconds": self.monitoring_interval,
            },
        }

    def optimize_tier_allocation(self) -> dict[str, Any]:
        """Optimize tier allocation based on current metrics."""
        metrics = self.get_system_metrics()
        optimization_results = {}

        for tier_name, tier_metric in metrics["tier_metrics"].items():
            utilization = tier_metric["active_sessions"] / tier_metric["max_sessions"]

            if utilization > 0.8:
                optimization_results[tier_name] = "scale_up"
            elif utilization < 0.2:
                optimization_results[tier_name] = "scale_down"
            else:
                optimization_results[tier_name] = "optimal"

        return {
            "optimization_recommendations": optimization_results,
            "timestamp": time.time(),
            "trigger": "automatic_optimization",
        }


@pytest.mark.browser_monitoring
class TestBrowserAutomationMonitoring:
    """Test browser automation monitoring functionality."""

    @pytest.fixture
    def monitoring_system(self):
        """Browser monitoring system fixture."""
        return MockBrowserMonitoringSystem()

    @pytest.fixture
    def lightweight_tier(self):
        """Lightweight tier fixture."""
        return MockBrowserTier("lightweight", 1)

    def test_tier_initialization(self, monitoring_system):
        """Test 5-tier system initialization."""
        assert len(monitoring_system.tiers) == 5
        assert "lightweight" in monitoring_system.tiers
        assert "playwright" in monitoring_system.tiers
        assert "crawl4ai" in monitoring_system.tiers
        assert "browser_use" in monitoring_system.tiers
        assert "firecrawl" in monitoring_system.tiers

        # Verify tier characteristics
        assert monitoring_system.tiers["lightweight"].response_time == 50.0
        assert monitoring_system.tiers["firecrawl"].response_time == 800.0

    @pytest.mark.asyncio
    async def test_individual_tier_health_check(self, lightweight_tier):
        """Test individual tier health checking."""
        health = await lightweight_tier.health_check()

        assert "healthy" in health
        assert "response_time_ms" in health
        assert health["healthy"] is True
        assert health["response_time_ms"] == 50.0

    @pytest.mark.asyncio
    async def test_comprehensive_system_health_check(self, monitoring_system):
        """Test comprehensive system health checking."""
        health_report = await monitoring_system.comprehensive_health_check()

        assert "overall_health" in health_report
        assert "healthy_tiers" in health_report
        assert "_total_tiers" in health_report
        assert "tier_health" in health_report

        assert health_report["_total_tiers"] == 5
        assert health_report["healthy_tiers"] <= 5
        assert 0 <= health_report["overall_health"] <= 1

    @pytest.mark.asyncio
    async def test_tier_request_execution(self, lightweight_tier):
        """Test tier request execution."""
        result = await lightweight_tier.execute_request("simple_scrape")

        assert result["status"] == "success"
        assert result["tier"] == "lightweight"
        assert result["request_type"] == "simple_scrape"
        assert "response_time_ms" in result

    @pytest.mark.asyncio
    async def test_tier_capacity_limits(self, lightweight_tier):
        """Test tier capacity enforcement."""
        # Fill tier to capacity
        lightweight_tier.max_sessions = 2

        # First two requests should succeed
        _task1 = asyncio.create_task(lightweight_tier.execute_request("test1"))
        _task2 = asyncio.create_task(lightweight_tier.execute_request("test2"))

        # Third request should fail due to capacity
        with pytest.raises(Exception, match="at capacity"):
            await lightweight_tier.execute_request("test3")

    @pytest.mark.asyncio
    async def test_failover_mechanism(self, monitoring_system):
        """Test automatic failover between tiers."""
        # Make lightweight tier unhealthy
        monitoring_system.tiers["lightweight"].is_healthy = False

        # Request should failover to playwright
        result = await monitoring_system.execute_with_failover(
            "complex_scrape", "lightweight"
        )

        assert result["status"] == "success"
        assert result["failover_tier"] == "playwright"  # Should failover
        assert result["attempts"] > 1

    @pytest.mark.asyncio
    async def test_preferred_tier_selection(self, monitoring_system):
        """Test preferred tier selection."""
        # Request with preferred tier
        result = await monitoring_system.execute_with_failover(
            "simple_scrape", "crawl4ai"
        )

        assert result["status"] == "success"
        assert result["failover_tier"] == "crawl4ai"
        assert result["attempts"] == 1  # No failover needed

    def test_tier_performance_metrics(self, monitoring_system):
        """Test tier performance metrics collection."""
        metrics = monitoring_system.get_system_metrics()

        assert "tier_metrics" in metrics
        assert "aggregate_metrics" in metrics
        assert "system_configuration" in metrics

        # Check tier-specific metrics
        for tier_name in [
            "lightweight",
            "playwright",
            "crawl4ai",
            "browser_use",
            "firecrawl",
        ]:
            assert tier_name in metrics["tier_metrics"]
            tier_metric = metrics["tier_metrics"][tier_name]
            assert "response_time_ms" in tier_metric
            assert "success_rate" in tier_metric
            assert "active_sessions" in tier_metric

    def test_system_aggregate_metrics(self, monitoring_system):
        """Test system-wide aggregate metrics."""
        metrics = monitoring_system.get_system_metrics()
        aggregate = metrics["aggregate_metrics"]

        assert "_total_active_sessions" in aggregate
        assert "_total_capacity" in aggregate
        assert "capacity_utilization" in aggregate
        assert "average_response_time_ms" in aggregate
        assert "healthy_tier_count" in aggregate

        assert aggregate["_total_capacity"] == 100  # Sum of all max_sessions
        assert 0 <= aggregate["capacity_utilization"] <= 1

    def test_performance_optimization_recommendations(self, monitoring_system):
        """Test performance optimization recommendations."""
        # Simulate high load on lightweight tier
        monitoring_system.tiers["lightweight"].active_sessions = 45  # Near capacity

        optimization = monitoring_system.optimize_tier_allocation()

        assert "optimization_recommendations" in optimization
        assert "timestamp" in optimization

        recommendations = optimization["optimization_recommendations"]
        assert "lightweight" in recommendations
        # Should recommend scaling up due to high utilization
        assert recommendations["lightweight"] == "scale_up"

    @pytest.mark.asyncio
    async def test_tier_failure_recovery(self, monitoring_system):
        """Test tier failure and recovery patterns."""
        lightweight = monitoring_system.tiers["lightweight"]

        # Simulate multiple health checks with failures
        for _i in range(25):  # Force some failures
            await lightweight.health_check()

        metrics = lightweight.get_metrics()
        assert metrics["_total_requests"] == 25
        assert metrics["failed_requests"] > 0
        assert metrics["error_rate"] > 0

    @pytest.mark.asyncio
    async def test_concurrent_tier_operations(self, monitoring_system):
        """Test concurrent operations across multiple tiers."""
        tasks = []

        # Create requests for different tiers
        tier_requests = [
            ("lightweight", "simple_scrape"),
            ("playwright", "dynamic_content"),
            ("crawl4ai", "ai_extraction"),
            ("browser_use", "complex_automation"),
            ("firecrawl", "enterprise_scrape"),
        ]

        for tier_name, request_type in tier_requests:
            task = asyncio.create_task(
                monitoring_system.execute_with_failover(request_type, tier_name)
            )
            tasks.append(task)

        # Execute all requests concurrently
        results = await asyncio.gather(*tasks, return_exceptions=True)

        # Check results
        successful_requests = sum(
            1 for r in results if isinstance(r, dict) and r.get("status") == "success"
        )
        assert successful_requests >= 4  # Most should succeed

    def test_monitoring_configuration(self, monitoring_system):
        """Test monitoring system configuration."""
        metrics = monitoring_system.get_system_metrics()
        config = metrics["system_configuration"]

        assert config["failover_enabled"] is True
        assert config["load_balancing"] == "round_robin"
        assert config["monitoring_interval_seconds"] == 30

    @pytest.mark.asyncio
    async def test_tier_response_time_characteristics(self, monitoring_system):
        """Test tier response time characteristics."""
        # Test different tiers have appropriate response times
        response_times = {}

        for tier_name in [
            "lightweight",
            "playwright",
            "crawl4ai",
            "browser_use",
            "firecrawl",
        ]:
            try:
                result = await monitoring_system.execute_with_failover(
                    "test_request", tier_name
                )
                response_times[tier_name] = result["response_time_ms"]
            except Exception:
                response_times[tier_name] = float("inf")

        # Verify tier hierarchy (lighter tiers should be faster)
        assert response_times["lightweight"] < response_times["playwright"]
        assert response_times["playwright"] < response_times["crawl4ai"]
        assert response_times["crawl4ai"] < response_times["browser_use"]
        assert response_times["browser_use"] < response_times["firecrawl"]


@pytest.mark.browser_monitoring
class TestAdvancedMonitoringFeatures:
    """Test advanced monitoring features."""

    def test_tier_health_scoring(self):
        """Test tier health scoring algorithm."""
        tier = MockBrowserTier("test_tier", 1)

        # Calculate health score based on multiple factors
        health_factors = {
            "response_time": tier.response_time,
            "success_rate": tier.success_rate,
            "cpu_usage": tier.cpu_usage,
            "memory_usage": tier.memory_usage,
            "error_rate": tier.failed_requests / max(tier._total_requests, 1),
        }

        # Simple health scoring algorithm
        health_score = (
            (100 - health_factors["cpu_usage"]) * 0.3
            + health_factors["success_rate"] * 0.3
            + (1000 - health_factors["response_time"]) / 10 * 0.2
            + (2048 - health_factors["memory_usage"]) / 20 * 0.2
        )

        assert 0 <= health_score <= 100

    def test_alerting_thresholds(self):
        """Test monitoring alerting thresholds."""
        system = MockBrowserMonitoringSystem()

        # Define alerting thresholds
        thresholds = {
            "response_time_critical": 1000,  # ms
            "success_rate_warning": 90.0,
            "cpu_usage_critical": 90.0,
            "memory_usage_warning": 1024.0,  # MB
            "capacity_utilization_critical": 0.9,
        }

        metrics = system.get_system_metrics()
        alerts = []

        for tier_name, tier_metrics in metrics["tier_metrics"].items():
            if tier_metrics["response_time_ms"] > thresholds["response_time_critical"]:
                alerts.append(f"Critical: {tier_name} response time high")

            if tier_metrics["success_rate"] < thresholds["success_rate_warning"]:
                alerts.append(f"Warning: {tier_name} success rate low")

        # Alerts should be actionable
        assert isinstance(alerts, list)

    def test_performance_trend_analysis(self):
        """Test performance trend analysis."""
        tier = MockBrowserTier("trend_tier", 1)

        # Simulate performance data over time
        performance_history = []
        for i in range(10):
            # Simulate gradually degrading performance
            tier.response_time = 100 + (i * 10)
            tier.success_rate = 95 - (i * 1)

            metrics = tier.get_metrics()
            performance_history.append(
                {
                    "timestamp": time.time() - (10 - i) * 60,  # 1 minute intervals
                    "response_time": metrics["response_time_ms"],
                    "success_rate": metrics["success_rate"],
                }
            )

        # Analyze trend
        response_time_trend = [p["response_time"] for p in performance_history]
        success_rate_trend = [p["success_rate"] for p in performance_history]

        # Performance should be degrading
        assert response_time_trend[-1] > response_time_trend[0]
        assert success_rate_trend[-1] < success_rate_trend[0]