"""Tests for UnifiedBrowserManager with browser caching integration."""

import time
from unittest.mock import AsyncMock, Mock, patch

import pytest

from src.config import Config
from src.services.browser.unified_manager import (
    UnifiedBrowserManager,
    UnifiedScrapingRequest,
)
from src.services.cache.browser_cache import BrowserCache, BrowserCacheEntry
from src.services.errors import CrawlServiceError


@pytest.fixture
def mock_config():
    """Create mock configuration with caching enabled."""
    config = Mock(spec=Config)
    config.performance = Mock()
    config.cache = Mock()
    config.cache.enable_browser_cache = True
    config.cache.browser_cache_ttl = 3600
    config.cache.browser_dynamic_ttl = 300
    config.cache.browser_static_ttl = 86400
    return config


@pytest.fixture
def mock_browser_cache():
    """Create mock browser cache."""
    cache = Mock(spec=BrowserCache)
    cache._generate_cache_key = Mock(
        side_effect=lambda url, tier: f"browser:test:{url}:{tier or 'auto'}"
    )
    cache.get = AsyncMock(return_value=None)
    cache.set = AsyncMock(return_value=True)
    cache.get_stats = Mock(return_value={"hits": 5, "misses": 3, "hit_rate": 0.625})
    return cache


@pytest.fixture
async def unified_manager_with_cache(mock_config):
    """Create UnifiedBrowserManager with caching enabled."""
    manager = UnifiedBrowserManager(mock_config)
    # Don't initialize to avoid dependencies
    manager._initialized = True
    manager._cache_enabled = True
    return manager


class TestUnifiedManagerCaching:
    """Test UnifiedBrowserManager caching functionality."""

    @patch("src.infrastructure.client_manager.ClientManager")
    @patch("src.services.cache.browser_cache.BrowserCache")
    @pytest.mark.asyncio
    async def test_initialization_with_cache(
        self, mock_browser_cache_class, mock_client_manager_class, mock_config
    ):
        """Test manager initializes browser cache when enabled."""
        # Setup mocks
        mock_client_manager = AsyncMock()
        mock_client_manager.initialize = AsyncMock()
        mock_client_manager.get_browser_automation_router = AsyncMock(
            return_value=Mock()
        )

        mock_cache_manager = Mock()
        mock_cache_manager.local_cache = Mock()
        mock_cache_manager.distributed_cache = Mock()
        mock_client_manager.get_cache_manager = AsyncMock(
            return_value=mock_cache_manager
        )

        mock_client_manager_class.return_value = mock_client_manager

        # Create manager
        manager = UnifiedBrowserManager(mock_config)
        await manager.initialize()

        # Verify browser cache was created
        mock_browser_cache_class.assert_called_once_with(
            local_cache=mock_cache_manager.local_cache,
            distributed_cache=mock_cache_manager.distributed_cache,
            default_ttl=3600,
            dynamic_content_ttl=300,
            static_content_ttl=86400,
        )
        assert manager._cache_enabled is True
        assert manager._browser_cache is not None

    @pytest.mark.asyncio
    async def test_scrape_checks_cache_first(
        self, unified_manager_with_cache, mock_browser_cache
    ):
        """Test scraping checks cache before executing."""
        # Setup cache hit
        cached_entry = BrowserCacheEntry(
            url="https://example.com",
            content="Cached content",
            metadata={"title": "Cached Title"},
            tier_used="crawl4ai",
            timestamp=time.time() - 100,  # 100 seconds ago
        )
        mock_browser_cache.get.return_value = cached_entry

        # Setup manager
        unified_manager_with_cache._browser_cache = mock_browser_cache
        unified_manager_with_cache._automation_router = Mock()

        # Execute scrape
        request = UnifiedScrapingRequest(url="https://example.com")
        response = await unified_manager_with_cache.scrape(request)

        # Verify cache was checked
        mock_browser_cache._generate_cache_key.assert_called_with(
            "https://example.com", None
        )
        mock_browser_cache.get.assert_called_once()

        # Verify cached response returned
        assert response.success is True
        assert response.content == "Cached content"
        assert response.title == "Cached Title"
        assert response.metadata["cached"] is True
        assert "cache_age_seconds" in response.metadata
        assert response.tier_used == "crawl4ai"

        # Verify router was NOT called
        unified_manager_with_cache._automation_router.scrape.assert_not_called()

    @pytest.mark.asyncio
    async def test_scrape_caches_successful_result(
        self, unified_manager_with_cache, mock_browser_cache
    ):
        """Test successful scrape results are cached."""
        # Setup cache miss
        mock_browser_cache.get.return_value = None

        # Setup router response
        mock_router = AsyncMock()
        mock_router.scrape = AsyncMock(
            return_value={
                "success": True,
                "content": "Fresh content",
                "metadata": {"title": "Fresh Title"},
                "provider": "browser_use",
            }
        )

        unified_manager_with_cache._browser_cache = mock_browser_cache
        unified_manager_with_cache._automation_router = mock_router

        # Execute scrape
        request = UnifiedScrapingRequest(url="https://example.com")
        response = await unified_manager_with_cache.scrape(request)

        # Verify result
        assert response.success is True
        assert response.content == "Fresh content"

        # Verify caching was attempted
        assert mock_browser_cache.set.call_count == 1
        cache_call_args = mock_browser_cache.set.call_args[0]
        _cache_key = cache_call_args[0]
        cache_entry = cache_call_args[1]

        assert isinstance(cache_entry, BrowserCacheEntry)
        assert cache_entry.url == "https://example.com"
        assert cache_entry.content == "Fresh content"
        assert cache_entry.tier_used == "browser_use"

    @pytest.mark.asyncio
    async def test_scrape_skips_cache_for_interaction(
        self, unified_manager_with_cache, mock_browser_cache
    ):
        """Test caching is skipped when interaction is required."""
        # Setup router response
        mock_router = AsyncMock()
        mock_router.scrape = AsyncMock(
            return_value={
                "success": True,
                "content": "Interactive content",
                "metadata": {},
                "provider": "browser_use",
            }
        )

        unified_manager_with_cache._browser_cache = mock_browser_cache
        unified_manager_with_cache._automation_router = mock_router

        # Execute scrape with interaction
        request = UnifiedScrapingRequest(
            url="https://example.com",
            interaction_required=True,
            custom_actions=[{"action": "click", "selector": "button"}],
        )
        await unified_manager_with_cache.scrape(request)

        # Verify cache was NOT checked or set
        mock_browser_cache.get.assert_not_called()
        mock_browser_cache.set.assert_not_called()

        # Verify router was called
        mock_router.scrape.assert_called_once()

    @pytest.mark.asyncio
    async def test_scrape_does_not_cache_failures(
        self, unified_manager_with_cache, mock_browser_cache
    ):
        """Test failed scrapes are not cached."""
        # Setup cache miss
        mock_browser_cache.get.return_value = None

        # Setup router to fail
        mock_router = AsyncMock()
        mock_router.scrape = AsyncMock(side_effect=CrawlServiceError("Scraping failed"))

        unified_manager_with_cache._browser_cache = mock_browser_cache
        unified_manager_with_cache._automation_router = mock_router

        # Execute scrape
        request = UnifiedScrapingRequest(url="https://example.com")
        response = await unified_manager_with_cache.scrape(request)

        # Verify failure response
        assert response.success is False
        assert response.error == "Scraping failed"

        # Verify result was NOT cached
        mock_browser_cache.set.assert_not_called()

    @pytest.mark.asyncio
    async def test_scrape_with_tier_specific_caching(
        self, unified_manager_with_cache, mock_browser_cache
    ):
        """Test caching respects tier specification."""
        # Setup cache miss
        mock_browser_cache.get.return_value = None

        # Setup router response
        mock_router = AsyncMock()
        mock_router.scrape = AsyncMock(
            return_value={
                "success": True,
                "content": "Lightweight content",
                "metadata": {},
                "provider": "lightweight",
            }
        )

        unified_manager_with_cache._browser_cache = mock_browser_cache
        unified_manager_with_cache._automation_router = mock_router

        # Execute scrape with specific tier
        request = UnifiedScrapingRequest(url="https://example.com", tier="lightweight")
        await unified_manager_with_cache.scrape(request)

        # Verify cache key includes tier
        mock_browser_cache._generate_cache_key.assert_any_call(
            "https://example.com", "lightweight"
        )

    @pytest.mark.asyncio
    async def test_get_system_status_includes_cache_stats(
        self, unified_manager_with_cache, mock_browser_cache
    ):
        """Test system status includes cache statistics."""
        unified_manager_with_cache._browser_cache = mock_browser_cache
        unified_manager_with_cache._automation_router = Mock()
        unified_manager_with_cache._automation_router.get_metrics = Mock(
            return_value={}
        )

        status = unified_manager_with_cache.get_system_status()

        assert status["cache_enabled"] is True
        assert "cache_stats" in status
        assert status["cache_stats"]["hits"] == 5
        assert status["cache_stats"]["misses"] == 3
        assert status["cache_stats"]["hit_rate"] == 0.625

    @pytest.mark.asyncio
    async def test_cache_disabled_behavior(self, mock_config):
        """Test behavior when caching is disabled."""
        # Disable caching in config
        mock_config.cache.enable_browser_cache = False

        manager = UnifiedBrowserManager(mock_config)
        manager._initialized = True
        manager._cache_enabled = False
        manager._browser_cache = None

        # Setup router
        mock_router = AsyncMock()
        mock_router.scrape = AsyncMock(
            return_value={
                "success": True,
                "content": "Content",
                "metadata": {},
                "provider": "crawl4ai",
            }
        )
        manager._automation_router = mock_router

        # Execute scrape
        request = UnifiedScrapingRequest(url="https://example.com")
        response = await manager.scrape(request)

        # Verify no caching operations occurred
        assert response.success is True
        assert response.content == "Content"
        # No cache operations should have been attempted

    @pytest.mark.asyncio
    async def test_cache_error_handling(
        self, unified_manager_with_cache, mock_browser_cache
    ):
        """Test graceful handling of cache errors."""
        # Setup cache to raise error
        mock_browser_cache.get.side_effect = Exception("Cache error")

        # Setup router response
        mock_router = AsyncMock()
        mock_router.scrape = AsyncMock(
            return_value={
                "success": True,
                "content": "Fresh content",
                "metadata": {},
                "provider": "crawl4ai",
            }
        )

        unified_manager_with_cache._browser_cache = mock_browser_cache
        unified_manager_with_cache._automation_router = mock_router

        # Execute scrape - should continue despite cache error
        request = UnifiedScrapingRequest(url="https://example.com")
        response = await unified_manager_with_cache.scrape(request)

        # Verify scraping succeeded despite cache error
        assert response.success is True
        assert response.content == "Fresh content"
        mock_router.scrape.assert_called_once()

    @pytest.mark.asyncio
    async def test_cache_metrics_update(
        self, unified_manager_with_cache, mock_browser_cache
    ):
        """Test tier metrics are updated for cache hits."""
        # Setup cache hit
        cached_entry = BrowserCacheEntry(
            url="https://example.com",
            content="Cached content",
            metadata={},
            tier_used="crawl4ai",
            timestamp=time.time() - 50,
        )
        mock_browser_cache.get.return_value = cached_entry

        unified_manager_with_cache._browser_cache = mock_browser_cache
        unified_manager_with_cache._automation_router = Mock()

        # Execute scrape
        request = UnifiedScrapingRequest(url="https://example.com")
        await unified_manager_with_cache.scrape(request)

        # Verify metrics were updated
        metrics = unified_manager_with_cache._tier_metrics["crawl4ai"]
        assert metrics._total_requests == 1
        assert metrics.successful_requests == 1
        assert metrics.success_rate == 1.0
