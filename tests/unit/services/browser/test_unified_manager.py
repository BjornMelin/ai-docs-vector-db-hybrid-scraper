"""Comprehensive tests for UnifiedBrowserManager."""

from unittest.mock import AsyncMock, Mock, patch

import pytest

from src.config import Config
from src.services.browser.unified_manager import (
    UnifiedBrowserManager,
    UnifiedScrapingRequest,
    UnifiedScrapingResponse,
)
from src.services.cache.browser_cache import BrowserCacheEntry
from src.services.errors import CrawlServiceError


@pytest.fixture
def mock_config():
    """Create mock configuration for testing."""
    config = Mock(spec=Config)
    config.performance = Mock()
    config.cache = Mock()
    config.cache.enable_browser_cache = True
    return config


@pytest.fixture
def mock_client_manager():
    """Create mock client manager."""
    client_manager = Mock()
    client_manager.initialize = AsyncMock()
    client_manager.cleanup = AsyncMock()
    client_manager.get_cache_manager = AsyncMock()
    return client_manager


@pytest.fixture
def mock_automation_router():
    """Create mock automation router."""
    router = Mock()
    router.scrape = AsyncMock()
    router.get_recommended_tool = AsyncMock(return_value="crawl4ai")
    router.get_metrics = Mock(return_value={})
    return router


@pytest.fixture
async def unified_manager(mock_config):
    """Create UnifiedBrowserManager instance for testing."""
    return UnifiedBrowserManager(mock_config)


class TestUnifiedBrowserManagerInitialization:
    """Test initialization and cleanup of UnifiedBrowserManager."""

    @pytest.mark.asyncio
    async def test_initialization_success(
        self, unified_manager, mock_client_manager, mock_automation_router
    ):
        """Test successful initialization."""
        with patch("src.infrastructure.client_manager.ClientManager") as mock_cm:
            mock_cm.return_value = mock_client_manager
            mock_client_manager.get_browser_automation_router = AsyncMock(
                return_value=mock_automation_router
            )

            await unified_manager.initialize()

            assert unified_manager._initialized is True
            assert unified_manager._automation_router is mock_automation_router
            mock_client_manager.initialize.assert_awaited_once()

    @pytest.mark.asyncio
    async def test_initialization_idempotent(
        self, unified_manager, mock_client_manager, mock_automation_router
    ):
        """Test that initialization is idempotent."""
        with patch("src.infrastructure.client_manager.ClientManager") as mock_cm:
            mock_cm.return_value = mock_client_manager
            mock_client_manager.get_browser_automation_router = AsyncMock(
                return_value=mock_automation_router
            )

            await unified_manager.initialize()
            await unified_manager.initialize()  # Second call

            mock_client_manager.initialize.assert_awaited_once()  # Only called once

    @pytest.mark.asyncio
    async def test_initialization_failure(self, unified_manager, mock_client_manager):
        """Test initialization failure handling."""
        with patch("src.infrastructure.client_manager.ClientManager") as mock_cm:
            mock_cm.return_value = mock_client_manager
            mock_client_manager.initialize.side_effect = Exception("Init failed")

            with pytest.raises(CrawlServiceError) as exc_info:
                await unified_manager.initialize()

            assert "Failed to initialize unified browser manager" in str(exc_info.value)

    @pytest.mark.asyncio
    async def test_cleanup(
        self, unified_manager, mock_client_manager, mock_automation_router
    ):
        """Test cleanup process."""
        # Initialize first
        with patch("src.infrastructure.client_manager.ClientManager") as mock_cm:
            mock_cm.return_value = mock_client_manager
            mock_client_manager.get_browser_automation_router = AsyncMock(
                return_value=mock_automation_router
            )

            await unified_manager.initialize()
            await unified_manager.cleanup()

            assert unified_manager._initialized is False
            assert unified_manager._client_manager is None
            assert unified_manager._automation_router is None
            mock_client_manager.cleanup.assert_awaited_once()


class TestUnifiedScrapingAPI:
    """Test the unified scraping API."""

    @pytest.mark.asyncio
    async def test_scrape_with_request_object(
        self, unified_manager, mock_client_manager, mock_automation_router
    ):
        """Test scraping with UnifiedScrapingRequest object."""
        # Initialize manager
        with patch("src.infrastructure.client_manager.ClientManager") as mock_cm:
            mock_cm.return_value = mock_client_manager
            mock_client_manager.get_browser_automation_router = AsyncMock(
                return_value=mock_automation_router
            )
            # Mock cache manager
            mock_cache_manager = Mock()
            mock_cache_manager.local_cache = None
            mock_cache_manager.distributed_cache = None
            mock_client_manager.get_cache_manager = AsyncMock(
                return_value=mock_cache_manager
            )

            await unified_manager.initialize()

            # Setup mock response
            mock_automation_router.scrape.return_value = {
                "success": True,
                "content": "Test content",
                "metadata": {"title": "Test Page"},
                "provider": "crawl4ai",
                "failed_tools": [],
            }

            # Create request
            request = UnifiedScrapingRequest(
                url="https://example.com",
                tier="auto",
                interaction_required=False,
                timeout=30000,
            )

            # Execute
            response = await unified_manager.scrape(request)

            # Verify
            assert isinstance(response, UnifiedScrapingResponse)
            assert response.success is True
            assert response.content == "Test content"
            assert response.url == "https://example.com"
            assert response.title == "Test Page"
            assert response.tier_used == "crawl4ai"
            assert response.quality_score > 0

            # Verify AutomationRouter was called correctly
            mock_automation_router.scrape.assert_awaited_once_with(
                url="https://example.com",
                interaction_required=False,
                custom_actions=None,
                force_tool=None,
                timeout=30000,
            )

    @pytest.mark.asyncio
    async def test_scrape_with_url_parameter(
        self, unified_manager, mock_client_manager, mock_automation_router
    ):
        """Test scraping with simple URL parameter."""
        # Initialize manager
        with patch("src.infrastructure.client_manager.ClientManager") as mock_cm:
            mock_cm.return_value = mock_client_manager
            mock_client_manager.get_browser_automation_router = AsyncMock(
                return_value=mock_automation_router
            )
            await unified_manager.initialize()

            # Setup mock response
            mock_automation_router.scrape.return_value = {
                "success": True,
                "content": "Simple content",
                "metadata": {},
                "provider": "lightweight",
            }

            # Execute
            response = await unified_manager.scrape(url="https://example.com")

            # Verify
            assert response.success is True
            assert response.tier_used == "lightweight"

    @pytest.mark.asyncio
    async def test_scrape_with_forced_tier(
        self, unified_manager, mock_client_manager, mock_automation_router
    ):
        """Test scraping with forced tier selection."""
        # Initialize manager
        with patch("src.infrastructure.client_manager.ClientManager") as mock_cm:
            mock_cm.return_value = mock_client_manager
            mock_client_manager.get_browser_automation_router = AsyncMock(
                return_value=mock_automation_router
            )
            await unified_manager.initialize()

            # Setup mock response
            mock_automation_router.scrape.return_value = {
                "success": True,
                "content": "Browser content",
                "metadata": {},
                "provider": "browser_use",
            }

            # Execute with forced tier
            await unified_manager.scrape(url="https://example.com", tier="browser_use")

            # Verify force_tool was set
            mock_automation_router.scrape.assert_awaited_once()
            call_args = mock_automation_router.scrape.call_args
            assert call_args._kwargs["force_tool"] == "browser_use"

    @pytest.mark.asyncio
    async def test_scrape_with_fallback(
        self, unified_manager, mock_client_manager, mock_automation_router
    ):
        """Test scraping with tier fallback."""
        # Initialize manager
        with patch("src.infrastructure.client_manager.ClientManager") as mock_cm:
            mock_cm.return_value = mock_client_manager
            mock_client_manager.get_browser_automation_router = AsyncMock(
                return_value=mock_automation_router
            )
            await unified_manager.initialize()

            # Setup mock response with fallback
            mock_automation_router.scrape.return_value = {
                "success": True,
                "content": "Fallback content",
                "metadata": {},
                "provider": "playwright",
                "fallback_from": "crawl4ai",
                "failed_tools": ["lightweight", "crawl4ai"],
            }

            # Execute
            response = await unified_manager.scrape(url="https://example.com")

            # Verify fallback tracking
            assert response.fallback_attempted is True
            assert response.failed_tiers == ["lightweight", "crawl4ai"]
            assert response.tier_used == "playwright"

    @pytest.mark.asyncio
    async def test_scrape_error_handling(
        self, unified_manager, mock_client_manager, mock_automation_router
    ):
        """Test error handling during scraping."""
        # Initialize manager
        with patch("src.infrastructure.client_manager.ClientManager") as mock_cm:
            mock_cm.return_value = mock_client_manager
            mock_client_manager.get_browser_automation_router = AsyncMock(
                return_value=mock_automation_router
            )
            await unified_manager.initialize()

            # Setup mock to raise exception
            mock_automation_router.scrape.side_effect = Exception("Scraping failed")

            # Execute
            response = await unified_manager.scrape(url="https://example.com")

            # Verify error response
            assert response.success is False
            assert response.content == ""
            assert response.error == "Scraping failed"
            assert response.tier_used == "none"
            assert response.quality_score == 0.0

    @pytest.mark.asyncio
    async def test_scrape_not_initialized(self, unified_manager):
        """Test scraping when manager not initialized."""
        with pytest.raises(CrawlServiceError) as exc_info:
            await unified_manager.scrape(url="https://example.com")

        assert "UnifiedBrowserManager not initialized" in str(exc_info.value)

    @pytest.mark.asyncio
    async def test_scrape_no_url_or_request(
        self, unified_manager, mock_client_manager, mock_automation_router
    ):
        """Test scraping with neither request object nor URL."""
        # Initialize manager
        with patch("src.infrastructure.client_manager.ClientManager") as mock_cm:
            mock_cm.return_value = mock_client_manager
            mock_client_manager.get_browser_automation_router = AsyncMock(
                return_value=mock_automation_router
            )
            await unified_manager.initialize()

            with pytest.raises(CrawlServiceError) as exc_info:
                await unified_manager.scrape()

            assert "Either request object or url parameter required" in str(
                exc_info.value
            )


class TestMetricsTracking:
    """Test metrics tracking functionality."""

    @pytest.mark.asyncio
    async def test_metrics_initialization(self, unified_manager):
        """Test that metrics are properly initialized."""
        assert len(unified_manager._tier_metrics) == 6
        for tier in [
            "lightweight",
            "crawl4ai",
            "crawl4ai_enhanced",
            "browser_use",
            "playwright",
            "firecrawl",
        ]:
            assert tier in unified_manager._tier_metrics
            metrics = unified_manager._tier_metrics[tier]
            assert metrics.tier_name == tier
            assert metrics._total_requests == 0
            assert metrics.successful_requests == 0
            assert metrics.failed_requests == 0

    @pytest.mark.asyncio
    async def test_metrics_update_on_success(
        self, unified_manager, mock_client_manager, mock_automation_router
    ):
        """Test metrics update on successful scraping."""
        # Initialize manager
        with patch("src.infrastructure.client_manager.ClientManager") as mock_cm:
            mock_cm.return_value = mock_client_manager
            mock_client_manager.get_browser_automation_router = AsyncMock(
                return_value=mock_automation_router
            )
            await unified_manager.initialize()

            # Setup mock response
            mock_automation_router.scrape.return_value = {
                "success": True,
                "content": "Test content",
                "metadata": {},
                "provider": "crawl4ai",
            }

            # Execute
            await unified_manager.scrape(url="https://example.com")

            # Check metrics
            metrics = unified_manager._tier_metrics["crawl4ai"]
            assert metrics._total_requests == 1
            assert metrics.successful_requests == 1
            assert metrics.failed_requests == 0
            assert metrics.success_rate == 1.0

    @pytest.mark.asyncio
    async def test_metrics_update_on_failure(
        self, unified_manager, mock_client_manager, mock_automation_router
    ):
        """Test metrics update on failed scraping."""
        # Initialize manager
        with patch("src.infrastructure.client_manager.ClientManager") as mock_cm:
            mock_cm.return_value = mock_client_manager
            mock_client_manager.get_browser_automation_router = AsyncMock(
                return_value=mock_automation_router
            )
            await unified_manager.initialize()

            # Setup mock to raise exception
            mock_automation_router.scrape.side_effect = Exception("Failed")

            # Execute
            await unified_manager.scrape(url="https://example.com")

            # Check metrics - should track as unknown tier
            metrics = unified_manager._tier_metrics.get("unknown")
            assert metrics is not None
            assert metrics._total_requests == 1
            assert metrics.successful_requests == 0
            assert metrics.failed_requests == 1

    @pytest.mark.asyncio
    async def test_get_tier_metrics(
        self, unified_manager, mock_client_manager, mock_automation_router
    ):
        """Test getting tier metrics."""
        # Initialize and run some scrapes
        with patch("src.infrastructure.client_manager.ClientManager") as mock_cm:
            mock_cm.return_value = mock_client_manager
            mock_client_manager.get_browser_automation_router = AsyncMock(
                return_value=mock_automation_router
            )
            await unified_manager.initialize()

            # Setup varying responses
            mock_automation_router.scrape.side_effect = [
                {
                    "success": True,
                    "content": "Test 1",
                    "metadata": {},
                    "provider": "crawl4ai",
                },
                {
                    "success": True,
                    "content": "Test 2",
                    "metadata": {},
                    "provider": "lightweight",
                },
                {
                    "success": False,
                    "content": "",
                    "metadata": {},
                    "provider": "browser_use",
                },
            ]

            # Execute multiple scrapes
            await unified_manager.scrape(url="https://example1.com")
            await unified_manager.scrape(url="https://example2.com")
            await unified_manager.scrape(url="https://example3.com")

            # Get metrics
            all_metrics = unified_manager.get_tier_metrics()

            # Verify
            assert "crawl4ai" in all_metrics
            assert "lightweight" in all_metrics
            assert all_metrics["crawl4ai"]._total_requests == 1
            assert all_metrics["lightweight"]._total_requests == 1

    @pytest.mark.asyncio
    async def test_average_response_time_calculation(
        self, unified_manager, mock_client_manager, mock_automation_router
    ):
        """Test rolling average response time calculation."""
        # Initialize manager
        with patch("src.infrastructure.client_manager.ClientManager") as mock_cm:
            mock_cm.return_value = mock_client_manager
            mock_client_manager.get_browser_automation_router = AsyncMock(
                return_value=mock_automation_router
            )
            await unified_manager.initialize()

            # Mock responses with varying execution times
            mock_automation_router.scrape.return_value = {
                "success": True,
                "content": "Test",
                "metadata": {},
                "provider": "crawl4ai",
            }

            # Execute multiple scrapes with manual timing

            # Manually update metrics to test averaging
            unified_manager._update_tier_metrics("crawl4ai", True, 100.0)
            unified_manager._update_tier_metrics("crawl4ai", True, 200.0)
            unified_manager._update_tier_metrics("crawl4ai", True, 300.0)

            metrics = unified_manager._tier_metrics["crawl4ai"]
            assert metrics.average_response_time_ms == 200.0  # (100+200+300)/3


class TestURLAnalysis:
    """Test URL analysis functionality."""

    @pytest.mark.asyncio
    async def test_analyze_url_success(
        self, unified_manager, mock_client_manager, mock_automation_router
    ):
        """Test successful URL analysis."""
        # Initialize manager
        with patch("src.infrastructure.client_manager.ClientManager") as mock_cm:
            mock_cm.return_value = mock_client_manager
            mock_client_manager.get_browser_automation_router = AsyncMock(
                return_value=mock_automation_router
            )
            await unified_manager.initialize()

            # Setup mock recommendation
            mock_automation_router.get_recommended_tool.return_value = "lightweight"

            # Execute
            analysis = await unified_manager.analyze_url("https://example.com")

            # Verify
            assert analysis["url"] == "https://example.com"
            assert analysis["domain"] == "example.com"
            assert analysis["recommended_tier"] == "lightweight"
            assert "expected_performance" in analysis
            assert "estimated_time_ms" in analysis["expected_performance"]
            assert "success_rate" in analysis["expected_performance"]

    @pytest.mark.asyncio
    async def test_analyze_url_error_handling(
        self, unified_manager, mock_client_manager, mock_automation_router
    ):
        """Test URL analysis error handling."""
        # Initialize manager
        with patch("src.infrastructure.client_manager.ClientManager") as mock_cm:
            mock_cm.return_value = mock_client_manager
            mock_client_manager.get_browser_automation_router = AsyncMock(
                return_value=mock_automation_router
            )
            await unified_manager.initialize()

            # Setup mock to raise exception
            mock_automation_router.get_recommended_tool.side_effect = Exception(
                "Analysis failed"
            )

            # Execute
            analysis = await unified_manager.analyze_url("https://example.com")

            # Verify fallback behavior
            assert analysis["url"] == "https://example.com"
            assert analysis["error"] == "Analysis failed"
            assert analysis["recommended_tier"] == "crawl4ai"  # Default fallback

    @pytest.mark.asyncio
    async def test_analyze_url_not_initialized(self, unified_manager):
        """Test URL analysis when manager not initialized."""
        with pytest.raises(CrawlServiceError) as exc_info:
            await unified_manager.analyze_url("https://example.com")

        assert "UnifiedBrowserManager not initialized" in str(exc_info.value)


class TestSystemStatus:
    """Test system status reporting."""

    @pytest.mark.asyncio
    async def test_get_system_status_not_initialized(self, unified_manager):
        """Test system status when not initialized."""
        status = unified_manager.get_system_status()

        assert status["status"] == "not_initialized"
        assert status["error"] == "Manager not initialized"

    @pytest.mark.asyncio
    async def test_get_system_status_healthy(
        self, unified_manager, mock_client_manager, mock_automation_router
    ):
        """Test system status when healthy."""
        # Initialize manager
        with patch("src.infrastructure.client_manager.ClientManager") as mock_cm:
            mock_cm.return_value = mock_client_manager
            mock_client_manager.get_browser_automation_router = AsyncMock(
                return_value=mock_automation_router
            )
            await unified_manager.initialize()

            # Add some successful metrics
            unified_manager._update_tier_metrics("crawl4ai", True, 100.0)
            unified_manager._update_tier_metrics("lightweight", True, 50.0)

            # Get status
            status = unified_manager.get_system_status()

            # Verify
            assert status["status"] == "healthy"
            assert status["initialized"] is True
            assert status["_total_requests"] == 2
            assert status["overall_success_rate"] == 1.0
            assert status["tier_count"] == 2
            assert status["router_available"] is True

    @pytest.mark.asyncio
    async def test_get_system_status_degraded(
        self, unified_manager, mock_client_manager, mock_automation_router
    ):
        """Test system status when degraded."""
        # Initialize manager
        with patch("src.infrastructure.client_manager.ClientManager") as mock_cm:
            mock_cm.return_value = mock_client_manager
            mock_client_manager.get_browser_automation_router = AsyncMock(
                return_value=mock_automation_router
            )
            await unified_manager.initialize()

            # Add mixed success/failure metrics
            for _ in range(3):
                unified_manager._update_tier_metrics("crawl4ai", True, 100.0)
            for _ in range(2):
                unified_manager._update_tier_metrics("crawl4ai", False, 100.0)

            # Get status
            status = unified_manager.get_system_status()

            # Verify
            assert status["status"] == "degraded"  # Success rate is 3/5 = 0.6 < 0.8
            assert status["overall_success_rate"] == 0.6


class TestQualityScoring:
    """Test content quality scoring."""

    @pytest.mark.asyncio
    async def test_quality_score_calculation(self, unified_manager):
        """Test quality score calculation based on content length."""
        # Test various content lengths
        test_cases = [
            ({"success": False, "content": ""}, 0.0),
            ({"success": True, "content": ""}, 0.0),
            ({"success": True, "content": "x" * 1000}, 0.2),  # 1000/5000
            ({"success": True, "content": "x" * 2500}, 0.5),  # 2500/5000
            ({"success": True, "content": "x" * 5000}, 1.0),  # 5000/5000
            ({"success": True, "content": "x" * 10000}, 1.0),  # capped at 1.0
        ]

        for result, expected_score in test_cases:
            score = unified_manager._calculate_quality_score(result)
            assert score == expected_score


class TestCustomActions:
    """Test handling of custom actions."""

    @pytest.mark.asyncio
    async def test_scrape_with_custom_actions(
        self, unified_manager, mock_client_manager, mock_automation_router
    ):
        """Test scraping with custom actions."""
        # Initialize manager
        with patch("src.infrastructure.client_manager.ClientManager") as mock_cm:
            mock_cm.return_value = mock_client_manager
            mock_client_manager.get_browser_automation_router = AsyncMock(
                return_value=mock_automation_router
            )
            await unified_manager.initialize()

            # Setup mock response
            mock_automation_router.scrape.return_value = {
                "success": True,
                "content": "Interactive content",
                "metadata": {},
                "provider": "browser_use",
            }

            # Create request with custom actions
            custom_actions = [
                {"type": "click", "selector": "#button"},
                {"type": "wait", "duration": 1000},
            ]
            request = UnifiedScrapingRequest(
                url="https://example.com",
                tier="browser_use",
                interaction_required=True,
                custom_actions=custom_actions,
            )

            # Execute
            await unified_manager.scrape(request)

            # Verify custom actions were passed
            mock_automation_router.scrape.assert_awaited_once()
            call_args = mock_automation_router.scrape.call_args
            assert call_args._kwargs["custom_actions"] == custom_actions
            assert call_args._kwargs["interaction_required"] is True


class TestUnifiedBrowserManagerMonitoring:
    """Test monitoring system integration with UnifiedBrowserManager."""

    @pytest.mark.asyncio
    async def test_monitoring_integration(
        self, unified_manager, mock_client_manager, mock_automation_router
    ):
        """Test that monitoring system integrates correctly."""
        # Enable monitoring
        unified_manager._monitoring_enabled = True

        # Mock monitoring system
        mock_monitor = AsyncMock()
        unified_manager._monitor = mock_monitor

        # Initialize manager
        with patch("src.infrastructure.client_manager.ClientManager") as mock_cm:
            mock_cm.return_value = mock_client_manager
            mock_client_manager.get_browser_automation_router = AsyncMock(
                return_value=mock_automation_router
            )
            # Mock cache manager
            mock_cache_manager = Mock()
            mock_cache_manager.local_cache = None
            mock_cache_manager.distributed_cache = None
            mock_client_manager.get_cache_manager = AsyncMock(
                return_value=mock_cache_manager
            )

            await unified_manager.initialize()

            # Setup mock response
            mock_automation_router.scrape.return_value = {
                "success": True,
                "content": "Test content",
                "metadata": {},
                "provider": "crawl4ai",
            }

            # Perform scraping
            await unified_manager.scrape(url="https://example.com")

            # Verify monitoring was called
            mock_monitor.record_request_metrics.assert_called_once()
            call_args = mock_monitor.record_request_metrics.call_args

            assert call_args[1]["tier"] == "crawl4ai"
            assert call_args[1]["success"] is True
            assert call_args[1]["response_time_ms"] > 0
            assert call_args[1]["cache_hit"] is False

    @pytest.mark.asyncio
    async def test_monitoring_disabled(
        self, unified_manager, mock_client_manager, mock_automation_router
    ):
        """Test behavior when monitoring is disabled."""
        # Disable monitoring
        unified_manager._monitoring_enabled = False

        # Mock monitoring system (shouldn't be called)
        mock_monitor = AsyncMock()
        unified_manager._monitor = mock_monitor

        # Initialize manager
        with patch("src.infrastructure.client_manager.ClientManager") as mock_cm:
            mock_cm.return_value = mock_client_manager
            mock_client_manager.get_browser_automation_router = AsyncMock(
                return_value=mock_automation_router
            )
            # Mock cache manager
            mock_cache_manager = Mock()
            mock_cache_manager.local_cache = None
            mock_cache_manager.distributed_cache = None
            mock_client_manager.get_cache_manager = AsyncMock(
                return_value=mock_cache_manager
            )

            await unified_manager.initialize()

            # Setup mock response
            mock_automation_router.scrape.return_value = {
                "success": True,
                "content": "Test content",
                "metadata": {},
                "provider": "crawl4ai",
            }

            # Perform scraping
            response = await unified_manager.scrape(url="https://example.com")

            # Verify monitoring was NOT called
            mock_monitor.record_request_metrics.assert_not_called()
            assert response.success is True

    @pytest.mark.asyncio
    async def test_monitoring_system_health_in_status(self, unified_manager):
        """Test that monitoring health appears in system status."""
        # Enable monitoring
        unified_manager._monitoring_enabled = True

        # Mock monitoring system health
        mock_monitor = Mock()
        mock_monitor.get_system_health = Mock(
            return_value={
                "overall_status": "healthy",
                "tier_health": {"_total": 2, "healthy": 2},
                "monitoring_active": True,
            }
        )
        unified_manager._monitor = mock_monitor
        unified_manager._initialized = True

        # Get system status
        status = unified_manager.get_system_status()

        # Verify monitoring health is included
        assert status["monitoring_enabled"] is True
        assert "monitoring_health" in status
        assert status["monitoring_health"]["overall_status"] == "healthy"
        mock_monitor.get_system_health.assert_called_once()

    @pytest.mark.asyncio
    async def test_monitoring_error_handling(
        self, unified_manager, mock_client_manager, mock_automation_router
    ):
        """Test graceful handling of monitoring errors."""
        # Enable monitoring
        unified_manager._monitoring_enabled = True

        # Mock monitoring system that raises error
        mock_monitor = AsyncMock()
        mock_monitor.record_request_metrics.side_effect = Exception("Monitoring error")
        unified_manager._monitor = mock_monitor

        # Initialize manager
        with patch("src.infrastructure.client_manager.ClientManager") as mock_cm:
            mock_cm.return_value = mock_client_manager
            mock_client_manager.get_browser_automation_router = AsyncMock(
                return_value=mock_automation_router
            )
            # Mock cache manager
            mock_cache_manager = Mock()
            mock_cache_manager.local_cache = None
            mock_cache_manager.distributed_cache = None
            mock_client_manager.get_cache_manager = AsyncMock(
                return_value=mock_cache_manager
            )

            await unified_manager.initialize()

            # Setup mock response
            mock_automation_router.scrape.return_value = {
                "success": True,
                "content": "Test content",
                "metadata": {},
                "provider": "crawl4ai",
            }

            # Scraping should still work despite monitoring error
            response = await unified_manager.scrape(url="https://example.com")

            # Verify scraping succeeded despite monitoring failure
            assert response.success is True
            mock_monitor.record_request_metrics.assert_called_once()

    @pytest.mark.asyncio
    async def test_monitoring_cache_hit_tracking(self, unified_manager):
        """Test monitoring tracks cache hits correctly."""
        # Enable both caching and monitoring
        unified_manager._cache_enabled = True
        unified_manager._monitoring_enabled = True

        # Mock cache and monitoring
        mock_cache = AsyncMock()
        mock_monitor = AsyncMock()

        # Create cache entry
        cache_entry = BrowserCacheEntry(
            url="https://example.com",
            content="Cached content",
            metadata={"title": "Test"},
            tier_used="lightweight",
        )

        mock_cache.get.return_value = cache_entry
        mock_cache._generate_cache_key.return_value = "test_key"

        unified_manager._browser_cache = mock_cache
        unified_manager._monitor = mock_monitor
        unified_manager._initialized = True

        # Perform scraping that hits cache
        await unified_manager.scrape(url="https://example.com")

        # Verify cache hit was tracked in monitoring
        mock_monitor.record_request_metrics.assert_called_once()
        call_args = mock_monitor.record_request_metrics.call_args

        assert call_args[1]["tier"] == "lightweight"
        assert call_args[1]["success"] is True
        assert call_args[1]["cache_hit"] is True

    @pytest.mark.asyncio
    async def test_monitoring_cleanup_on_manager_cleanup(self, unified_manager):
        """Test monitoring is stopped when manager is cleaned up."""
        # Enable monitoring
        unified_manager._monitoring_enabled = True

        # Mock monitoring system
        mock_monitor = AsyncMock()
        unified_manager._monitor = mock_monitor
        unified_manager._initialized = True

        # Cleanup manager
        await unified_manager.cleanup()

        # Verify monitoring was stopped
        mock_monitor.stop_monitoring.assert_called_once()
