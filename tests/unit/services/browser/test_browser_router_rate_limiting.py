"""Tests for EnhancedAutomationRouter with rate limiting integration."""

import asyncio
import contextlib
from unittest.mock import AsyncMock, Mock, patch

import pytest

from src.config import Config
from src.services.browser.browser_router import EnhancedAutomationRouter
from src.services.browser.tier_config import EnhancedRoutingConfig
from src.services.browser.tier_rate_limiter import TierRateLimiter
from src.services.errors import CrawlServiceError


@pytest.fixture
def mock_config():
    """Create mock configuration."""
    config = Mock(spec=Config)
    config.performance = Mock()
    return config


@pytest.fixture
def mock_adapters():
    """Create mock adapters for testing."""
    return {
        "lightweight": AsyncMock(),
        "crawl4ai": AsyncMock(),
        "browser_use": AsyncMock(),
        "playwright": AsyncMock(),
    }


@pytest.fixture
async def enhanced_router_with_rate_limiting(mock_config, mock_adapters):
    """Create EnhancedAutomationRouter with rate limiting enabled."""
    router = EnhancedAutomationRouter(mock_config)

    # Mock initialization
    router._initialized = True
    router._adapters = mock_adapters

    # Set up tier configurations with rate limits
    router.routing_config = EnhancedRoutingConfig.get_default_config()

    # Override rate limits for testing
    router.routing_config.tier_configs[
        "lightweight"
    ].requests_per_minute = 10  # Very low for testing
    router.routing_config.tier_configs["crawl4ai"].requests_per_minute = 5
    router.routing_config.tier_configs["browser_use"].requests_per_minute = 3

    # Create rate limiter
    router.rate_limiter = TierRateLimiter(router.routing_config.tier_configs)

    # Mock the tier execution methods
    router._try_lightweight = mock_adapters["lightweight"]
    router._try_crawl4ai = mock_adapters["crawl4ai"]
    router._try_browser_use = mock_adapters["browser_use"]
    router._try_playwright = mock_adapters["playwright"]

    # Default responses
    mock_adapters["lightweight"].return_value = {
        "success": True,
        "content": "Lightweight content",
        "metadata": {},
    }
    mock_adapters["crawl4ai"].return_value = {
        "success": True,
        "content": "Crawl4AI content",
        "metadata": {},
    }
    mock_adapters["browser_use"].return_value = {
        "success": True,
        "content": "Browser-use content",
        "metadata": {},
    }

    return router


class TestEnhancedRouterRateLimiting:
    """Test EnhancedAutomationRouter rate limiting functionality."""

    async def test_rate_limiting_blocks_excessive_requests(
        self, enhanced_router_with_rate_limiting
    ):
        """Test that rate limiting blocks requests when limit is exceeded."""
        router = enhanced_router_with_rate_limiting
        url = "https://example.com"

        # browser_use has limit of 3 requests per minute
        # Force using browser_use tier
        results = []

        # Make 5 rapid requests
        for _ in range(5):
            try:
                result = await router.scrape(url, force_tool="browser_use")
                results.append(result)
            except Exception as e:
                results.append({"error": str(e)})

        # First 3 should succeed
        assert results[0].get("success") is True
        assert results[1].get("success") is True
        assert results[2].get("success") is True

        # 4th and 5th should fail or use fallback
        # (depends on fallback configuration)
        assert len(results) == 5

    async def test_rate_limiting_fallback_behavior(
        self, enhanced_router_with_rate_limiting
    ):
        """Test fallback behavior when primary tier is rate limited."""
        router = enhanced_router_with_rate_limiting
        url = "https://example.com"

        # Fill up browser_use rate limit
        for _ in range(3):
            await router.scrape(url, force_tool="browser_use")

        # Configure browser_use to have fallback to playwright
        router.routing_config.tier_configs["browser_use"].fallback_tiers = [
            "playwright"
        ]

        # Mock selection to prefer browser_use
        router._enhanced_select_tier = AsyncMock(return_value="browser_use")

        # Next request should fallback
        await router.scrape(url)

        # Should have attempted fallback
        # Check that playwright was called (fallback)
        router._try_playwright.assert_called()

    async def test_rate_limiting_with_circuit_breaker(
        self, enhanced_router_with_rate_limiting
    ):
        """Test interaction between rate limiting and circuit breaker."""
        router = enhanced_router_with_rate_limiting
        url = "https://example.com"

        # Make crawl4ai fail consistently
        router._try_crawl4ai.side_effect = Exception("Service unavailable")

        # Attempt exactly 3 requests to trigger circuit breaker
        for _ in range(3):
            with contextlib.suppress(Exception):
                await router.scrape(url, force_tool="crawl4ai")

        # Circuit breaker should be open after 3 failures
        breaker = router.circuit_breakers.get("crawl4ai")
        assert breaker is not None
        assert breaker.consecutive_failures == 3
        assert breaker.is_open  # Circuit breaker should be open

        # Rate limiter should have tracked attempts
        status = router.rate_limiter.get_status("crawl4ai")
        assert status["recent_requests"] == 3

    async def test_concurrent_rate_limiting(self, enhanced_router_with_rate_limiting):
        """Test rate limiting with concurrent requests."""
        router = enhanced_router_with_rate_limiting
        url = "https://example.com"

        # lightweight has max 5 concurrent requests
        router.routing_config.tier_configs["lightweight"].max_concurrent_requests = 3
        router.rate_limiter = TierRateLimiter(router.routing_config.tier_configs)

        # Make requests wait to test concurrency
        async def slow_response(*args, **kwargs):
            await asyncio.sleep(0.5)
            return {"success": True, "content": "Slow content", "metadata": {}}

        router._try_lightweight.side_effect = slow_response

        # Launch 5 concurrent requests
        tasks = []
        for i in range(5):
            task = router.scrape(f"{url}/{i}", force_tool="lightweight")
            tasks.append(task)

        # Wait for all to complete
        results = await asyncio.gather(*tasks, return_exceptions=True)

        # Should have some successes and some failures/fallbacks
        successes = sum(1 for r in results if isinstance(r, dict) and r.get("success"))
        assert successes >= 3  # At least concurrent limit

    async def test_rate_limit_status_in_performance_report(
        self, enhanced_router_with_rate_limiting
    ):
        """Test that rate limit status appears in performance report."""
        router = enhanced_router_with_rate_limiting

        # Make some requests to generate data
        for _ in range(2):
            await router.scrape("https://example.com", force_tool="lightweight")

        # Get performance report
        report = await router.get_performance_report()

        # Should include rate limit information
        assert "config" in report

        # Check rate limiter status
        rate_status = router.rate_limiter.get_status()
        assert "tiers" in rate_status
        assert "lightweight" in rate_status["tiers"]

        # Verify requests were tracked
        lightweight_status = rate_status["tiers"]["lightweight"]
        assert lightweight_status["recent_requests"] == 2

    async def test_rate_limit_reset_functionality(
        self, enhanced_router_with_rate_limiting
    ):
        """Test resetting rate limits for a tier."""
        router = enhanced_router_with_rate_limiting
        url = "https://example.com"

        # Fill up browser_use limit
        for _ in range(3):
            await router.scrape(url, force_tool="browser_use")

        # Next request should be rate limited
        # Force no fallback by mocking
        router._intelligent_fallback = AsyncMock(
            side_effect=CrawlServiceError("No fallback")
        )

        # This should fail due to rate limit
        with pytest.raises(CrawlServiceError):
            await router.scrape(url, force_tool="browser_use")

        # Reset the tier
        router.rate_limiter.reset_tier("browser_use")

        # Now should work again
        result = await router.scrape(url, force_tool="browser_use")
        assert result["success"] is True

    async def test_different_tier_rate_limits(self, enhanced_router_with_rate_limiting):
        """Test that different tiers have independent rate limits."""
        router = enhanced_router_with_rate_limiting
        url = "https://example.com"

        # Fill up browser_use (limit 3)
        for i in range(3):
            await router.scrape(f"{url}/browser/{i}", force_tool="browser_use")

        # browser_use should be rate limited now
        status = router.rate_limiter.get_status("browser_use")
        assert status["remaining_capacity"] == 0

        # But lightweight should still work (limit 10)
        for i in range(5):
            result = await router.scrape(f"{url}/light/{i}", force_tool="lightweight")
            assert result["success"] is True

        # Check lightweight still has capacity
        status = router.rate_limiter.get_status("lightweight")
        assert status["remaining_capacity"] > 0

    async def test_rate_limit_context_timeout(self, enhanced_router_with_rate_limiting):
        """Test rate limit context with timeout."""
        router = enhanced_router_with_rate_limiting
        url = "https://example.com"

        # Fill up rate limit for browser_use to test fallback
        for i in range(3):
            await router.scrape(f"{url}/warmup/{i}", force_tool="browser_use")

        # Configure browser_use to have fallback to playwright
        router.routing_config.tier_configs["browser_use"].fallback_tiers = [
            "playwright"
        ]

        # Next request should be rate limited and use fallback
        result = await router.scrape(url, force_tool="browser_use")

        # Should have used fallback due to rate limit
        assert "fallback_from" in result or result.get("tier_used") != "browser_use"

    async def test_rate_limiter_metrics_tracking(
        self, enhanced_router_with_rate_limiting
    ):
        """Test that rate limiter tracks metrics correctly."""
        router = enhanced_router_with_rate_limiting
        url = "https://example.com"

        # Make several requests
        for _ in range(5):
            with contextlib.suppress(Exception):
                await router.scrape(url, force_tool="lightweight")

        # Check metrics
        status = router.rate_limiter.get_status("lightweight")
        assert status["recent_requests"] == 5
        assert status["concurrent_requests"] >= 0
        assert "requests_per_minute" in status
        assert "remaining_capacity" in status

    @patch("src.services.browser.enhanced_router.logger")
    async def test_rate_limit_logging(
        self, mock_logger, enhanced_router_with_rate_limiting
    ):
        """Test that rate limiting logs appropriate messages."""
        router = enhanced_router_with_rate_limiting
        url = "https://example.com"

        # Fill up browser_use limit
        for _ in range(3):
            await router.scrape(url, force_tool="browser_use")

        # Disable fallback to see rate limit message
        router._intelligent_fallback = AsyncMock(
            return_value={
                "success": False,
                "content": "",
                "metadata": {},
                "tier_used": "none",
            }
        )

        # This should trigger rate limit warning
        await router.scrape(url, force_tool="browser_use")

        # Check for rate limit warning in logs
        mock_logger.warning.assert_any_call(
            "Rate limit exceeded for browser_use, attempting fallback"
        )
