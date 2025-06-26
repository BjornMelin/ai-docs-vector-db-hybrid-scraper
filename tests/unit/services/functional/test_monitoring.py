"""Tests for functional monitoring service."""

import asyncio
import time
from unittest.mock import MagicMock

import pytest

from src.config import Config
from src.services.functional.monitoring import (
    check_service_health,
    get_metrics_summary,
    get_performance_report,
    get_system_status,
    increment_counter,
    log_api_call,
    record_timer,
    set_gauge,
    timed,
)


@pytest.fixture
def mock_config():
    """Mock configuration for monitoring tests."""
    config = MagicMock(spec=Config)
    return config


@pytest.fixture(autouse=True)
async def clear_metrics():
    """Clear metrics before each test."""
    from src.services.functional.monitoring import _metrics, _metrics_lock

    async with _metrics_lock:
        _metrics["counters"].clear()
        _metrics["gauges"].clear()
        _metrics["timers"].clear()
        _metrics["health_checks"].clear()

    yield

    # Clear after test as well
    async with _metrics_lock:
        _metrics["counters"].clear()
        _metrics["gauges"].clear()
        _metrics["timers"].clear()
        _metrics["health_checks"].clear()


class TestMetricsCollection:
    """Test metrics collection functions."""

    @pytest.mark.asyncio
    async def test_increment_counter(self):
        """Test counter increment functionality."""
        await increment_counter("test_counter", 5)
        await increment_counter("test_counter", 3)

        summary = await get_metrics_summary()
        assert summary["counters"]["test_counter:"] == 8

    @pytest.mark.asyncio
    async def test_increment_counter_with_tags(self):
        """Test counter increment with tags."""
        await increment_counter(
            "api_calls", 1, tags={"provider": "openai", "status": "success"}
        )
        await increment_counter(
            "api_calls", 2, tags={"provider": "openai", "status": "error"}
        )

        summary = await get_metrics_summary()
        assert summary["counters"]["api_calls:provider=openai,status=success"] == 1
        assert summary["counters"]["api_calls:provider=openai,status=error"] == 2

    @pytest.mark.asyncio
    async def test_set_gauge(self):
        """Test gauge setting functionality."""
        await set_gauge("cpu_usage", 45.5)
        await set_gauge("memory_usage", 78.2, tags={"instance": "web-1"})

        summary = await get_metrics_summary()
        assert summary["gauges"]["cpu_usage:"] == 45.5
        assert summary["gauges"]["memory_usage:instance=web-1"] == 78.2

    @pytest.mark.asyncio
    async def test_record_timer(self):
        """Test timer recording functionality."""
        await record_timer("response_time", 150.5)
        await record_timer("response_time", 200.0)
        await record_timer("response_time", 125.0)

        summary = await get_metrics_summary()
        timer_stats = summary["timers"]["response_time:"]

        assert timer_stats["count"] == 3
        assert timer_stats["avg_ms"] == pytest.approx(158.5, rel=1e-2)
        assert timer_stats["min_ms"] == 125.0
        assert timer_stats["max_ms"] == 200.0

    @pytest.mark.asyncio
    async def test_timer_history_limit(self):
        """Test timer history is limited to prevent memory growth."""
        # Record more than 1000 measurements
        for i in range(1200):
            await record_timer("test_timer", float(i))

        from src.services.functional.monitoring import _metrics, _metrics_lock

        async with _metrics_lock:
            measurements = _metrics["timers"]["test_timer:"]
            assert len(measurements) == 1000  # Should be limited to 1000


class TestHealthChecks:
    """Test health check functionality."""

    @pytest.mark.asyncio
    async def test_check_service_health_vector_db(self, mock_config):
        """Test vector database health check."""
        health = await check_service_health("vector_db", mock_config)

        assert health["service"] == "vector_db"
        assert health["status"] in ["healthy", "unhealthy"]
        assert "checks" in health
        assert "qdrant_connection" in health["checks"]

    @pytest.mark.asyncio
    async def test_check_service_health_cache(self, mock_config):
        """Test cache health check."""
        health = await check_service_health("cache", mock_config)

        assert health["service"] == "cache"
        assert health["status"] in ["healthy", "unhealthy"]
        assert "checks" in health
        assert "dragonfly_connection" in health["checks"]

    @pytest.mark.asyncio
    async def test_check_service_health_embeddings(self, mock_config):
        """Test embeddings service health check."""
        health = await check_service_health("embeddings", mock_config)

        assert health["service"] == "embeddings"
        assert health["status"] in ["healthy", "unhealthy"]
        assert "checks" in health
        assert "openai_available" in health["checks"]
        assert "fastembed_available" in health["checks"]

    @pytest.mark.asyncio
    async def test_check_service_health_unknown(self, mock_config):
        """Test health check for unknown service."""
        health = await check_service_health("unknown_service", mock_config)

        assert health["service"] == "unknown_service"
        assert health["status"] in ["healthy", "unhealthy"]
        assert "checks" in health
        assert "generic" in health["checks"]

    @pytest.mark.asyncio
    async def test_get_system_status(self, mock_config):
        """Test system status aggregation."""
        status = await get_system_status(mock_config)

        assert "status" in status
        assert status["status"] in ["healthy", "degraded", "unhealthy", "error"]
        assert "timestamp" in status
        assert "services" in status
        assert "metrics_summary" in status

        # Should check key services
        assert "vector_db" in status["services"]
        assert "cache" in status["services"]
        assert "embeddings" in status["services"]
        assert "crawling" in status["services"]


class TestAPILogging:
    """Test API call logging functionality."""

    @pytest.mark.asyncio
    async def test_log_api_call_success(self):
        """Test successful API call logging."""
        await log_api_call(
            provider="openai",
            endpoint="embeddings",
            status_code=200,
            duration_ms=150.5,
            request_size=1024,
            response_size=2048,
        )

        summary = await get_metrics_summary()

        # Check counters
        assert (
            "api_calls_total:provider=openai,endpoint=embeddings,status_code=200"
            in summary["counters"]
        )
        assert (
            summary["counters"][
                "api_calls_total:provider=openai,endpoint=embeddings,status_code=200"
            ]
            == 1
        )

        # Check timers
        assert (
            "api_call_duration:provider=openai,endpoint=embeddings,status_code=200"
            in summary["timers"]
        )
        timer_stats = summary["timers"][
            "api_call_duration:provider=openai,endpoint=embeddings,status_code=200"
        ]
        assert timer_stats["avg_ms"] == 150.5

    @pytest.mark.asyncio
    async def test_log_api_call_error(self):
        """Test error API call logging."""
        await log_api_call(
            provider="firecrawl",
            endpoint="crawl",
            status_code=429,
            duration_ms=500.0,
        )

        summary = await get_metrics_summary()

        # Check error counter
        assert (
            "api_errors_total:provider=firecrawl,endpoint=crawl,status_code=429"
            in summary["counters"]
        )
        assert (
            summary["counters"][
                "api_errors_total:provider=firecrawl,endpoint=crawl,status_code=429"
            ]
            == 1
        )


class TestTimerDecorator:
    """Test timer decorator functionality."""

    @pytest.mark.asyncio
    async def test_timed_decorator(self):
        """Test timed decorator."""

        @timed("test_function")
        async def test_function():
            await asyncio.sleep(0.01)  # Sleep for 10ms
            return "success"

        result = await test_function()
        assert result == "success"

        summary = await get_metrics_summary()
        assert "test_function:" in summary["timers"]

        timer_stats = summary["timers"]["test_function:"]
        assert timer_stats["count"] == 1
        assert timer_stats["avg_ms"] >= 10  # Should be at least 10ms

    @pytest.mark.asyncio
    async def test_timed_decorator_with_tags(self):
        """Test timed decorator with tags."""

        @timed("test_function", tags={"operation": "test"})
        async def test_function_with_tags():
            await asyncio.sleep(0.005)  # Sleep for 5ms
            return "success"

        result = await test_function_with_tags()
        assert result == "success"

        summary = await get_metrics_summary()
        assert "test_function:operation=test" in summary["timers"]

    @pytest.mark.asyncio
    async def test_timed_decorator_with_exception(self):
        """Test timed decorator when function raises exception."""

        @timed("test_function_error")
        async def test_function_error():
            await asyncio.sleep(0.005)
            raise ValueError("Test error")

        with pytest.raises(ValueError):
            await test_function_error()

        # Timer should still be recorded
        summary = await get_metrics_summary()
        assert "test_function_error:" in summary["timers"]
        timer_stats = summary["timers"]["test_function_error:"]
        assert timer_stats["count"] == 1


class TestPerformanceReport:
    """Test performance reporting functionality."""

    @pytest.mark.asyncio
    async def test_get_performance_report_empty(self):
        """Test performance report with no data."""
        report = await get_performance_report()

        assert "timestamp" in report
        assert "summary" in report
        assert report["summary"]["total_api_calls"] == 0
        assert report["summary"]["total_errors"] == 0

    @pytest.mark.asyncio
    async def test_get_performance_report_with_data(self):
        """Test performance report with API call data."""
        # Log some API calls
        await log_api_call("openai", "embeddings", 200, 150.0)
        await log_api_call("openai", "embeddings", 200, 200.0)
        await log_api_call("firecrawl", "crawl", 429, 500.0)

        report = await get_performance_report()

        assert report["summary"]["total_api_calls"] == 3
        assert report["summary"]["total_errors"] == 1
        assert report["summary"]["average_response_time"] > 0
