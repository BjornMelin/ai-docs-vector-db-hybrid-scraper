"""Comprehensive tests for browser caching functionality."""

import asyncio
import json
import time
from unittest.mock import AsyncMock
from unittest.mock import patch

import pytest
from src.services.cache.base import CacheInterface
from src.services.cache.browser_cache import BrowserCache
from src.services.cache.browser_cache import BrowserCacheEntry


class MockCache(CacheInterface[str]):
    """Mock cache implementation for testing."""

    def __init__(self):
        self.data = {}
        self.get_count = 0
        self.set_count = 0

    async def get(self, key: str) -> str | None:
        self.get_count += 1
        return self.data.get(key)

    async def set(self, key: str, value: str, ttl: int | None = None) -> bool:
        self.set_count += 1
        self.data[key] = value
        return True

    async def delete(self, key: str) -> bool:
        if key in self.data:
            del self.data[key]
            return True
        return False

    async def exists(self, key: str) -> bool:
        return key in self.data

    async def clear(self) -> int:
        count = len(self.data)
        self.data.clear()
        return count

    async def size(self) -> int:
        return len(self.data)

    async def close(self) -> None:
        pass


@pytest.fixture
def mock_local_cache():
    """Create mock local cache."""
    return MockCache()


@pytest.fixture
def mock_distributed_cache():
    """Create mock distributed cache."""
    return MockCache()


@pytest.fixture
def browser_cache(mock_local_cache, mock_distributed_cache):
    """Create BrowserCache instance with mock caches."""
    return BrowserCache(
        local_cache=mock_local_cache,
        distributed_cache=mock_distributed_cache,
        default_ttl=3600,
        dynamic_content_ttl=300,
        static_content_ttl=86400,
    )


@pytest.fixture
def sample_cache_entry():
    """Create sample cache entry for testing."""
    return BrowserCacheEntry(
        url="https://example.com/test",
        content="<html><body>Test content</body></html>",
        metadata={"title": "Test Page", "description": "Test description"},
        tier_used="crawl4ai",
        timestamp=time.time(),
    )


class TestBrowserCacheEntry:
    """Test BrowserCacheEntry functionality."""

    def test_cache_entry_initialization(self):
        """Test cache entry initialization."""
        entry = BrowserCacheEntry(
            url="https://example.com",
            content="Test content",
            metadata={"title": "Test"},
            tier_used="lightweight",
        )

        assert entry.url == "https://example.com"
        assert entry.content == "Test content"
        assert entry.metadata == {"title": "Test"}
        assert entry.tier_used == "lightweight"
        assert isinstance(entry.timestamp, float)

    def test_cache_entry_to_dict(self, sample_cache_entry):
        """Test converting cache entry to dictionary."""
        data = sample_cache_entry.to_dict()

        assert data["url"] == sample_cache_entry.url
        assert data["content"] == sample_cache_entry.content
        assert data["metadata"] == sample_cache_entry.metadata
        assert data["tier_used"] == sample_cache_entry.tier_used
        assert data["timestamp"] == sample_cache_entry.timestamp

    def test_cache_entry_from_dict(self):
        """Test creating cache entry from dictionary."""
        data = {
            "url": "https://example.com",
            "content": "Test content",
            "metadata": {"title": "Test"},
            "tier_used": "browser_use",
            "timestamp": 1234567890.0,
        }

        entry = BrowserCacheEntry.from_dict(data)

        assert entry.url == data["url"]
        assert entry.content == data["content"]
        assert entry.metadata == data["metadata"]
        assert entry.tier_used == data["tier_used"]
        assert entry.timestamp == data["timestamp"]


class TestBrowserCache:
    """Test BrowserCache functionality."""

    def test_initialization(self, browser_cache):
        """Test browser cache initialization."""
        assert browser_cache.default_ttl == 3600
        assert browser_cache.dynamic_content_ttl == 300
        assert browser_cache.static_content_ttl == 86400
        assert browser_cache._cache_stats["hits"] == 0
        assert browser_cache._cache_stats["misses"] == 0

    def test_generate_cache_key_basic(self, browser_cache):
        """Test basic cache key generation."""
        key = browser_cache._generate_cache_key("https://example.com/test")
        assert key.startswith("browser:")
        assert "example.com" in key

    def test_generate_cache_key_with_tier(self, browser_cache):
        """Test cache key generation with tier."""
        key1 = browser_cache._generate_cache_key("https://example.com/test", "crawl4ai")
        key2 = browser_cache._generate_cache_key("https://example.com/test", "browser_use")

        assert key1 != key2
        assert "example.com" in key1
        assert "example.com" in key2

    def test_generate_cache_key_query_params(self, browser_cache):
        """Test cache key generation with query parameters."""
        # Same parameters, different order
        url1 = "https://example.com/test?b=2&a=1"
        url2 = "https://example.com/test?a=1&b=2"

        key1 = browser_cache._generate_cache_key(url1)
        key2 = browser_cache._generate_cache_key(url2)

        # Should be the same after normalization
        assert key1 == key2

    def test_determine_ttl_static_content(self, browser_cache):
        """Test TTL determination for static content."""
        static_urls = [
            "https://example.com/doc.pdf",
            "https://example.com/data.json",
            "https://raw.githubusercontent.com/user/repo/file.txt",
            "https://example.com/docs/api.html",
        ]

        for url in static_urls:
            ttl = browser_cache._determine_ttl(url, 1000)
            assert ttl == browser_cache.static_content_ttl

    def test_determine_ttl_dynamic_content(self, browser_cache):
        """Test TTL determination for dynamic content."""
        dynamic_urls = [
            "https://example.com/search?q=test",
            "https://api.example.com/v1/data",
            "https://twitter.com/user/status/123",
            "https://example.com/feed/latest",
        ]

        for url in dynamic_urls:
            ttl = browser_cache._determine_ttl(url, 1000)
            assert ttl == browser_cache.dynamic_content_ttl

    def test_determine_ttl_default(self, browser_cache):
        """Test default TTL determination."""
        url = "https://example.com/page"
        ttl = browser_cache._determine_ttl(url, 1000)
        assert ttl == browser_cache.default_ttl

    async def test_get_from_local_cache(self, browser_cache, mock_local_cache, sample_cache_entry):
        """Test getting entry from local cache."""
        # Store in local cache
        key = browser_cache._generate_cache_key(sample_cache_entry.url)
        mock_local_cache.data[key] = json.dumps(sample_cache_entry.to_dict())

        # Get from cache
        result = await browser_cache.get(key)

        assert result is not None
        assert result.url == sample_cache_entry.url
        assert result.content == sample_cache_entry.content
        assert browser_cache._cache_stats["hits"] == 1
        assert mock_local_cache.get_count == 1

    async def test_get_from_distributed_cache(
        self, browser_cache, mock_local_cache, mock_distributed_cache, sample_cache_entry
    ):
        """Test getting entry from distributed cache with promotion to local."""
        # Store only in distributed cache
        key = browser_cache._generate_cache_key(sample_cache_entry.url)
        mock_distributed_cache.data[key] = json.dumps(sample_cache_entry.to_dict())

        # Get from cache
        result = await browser_cache.get(key)

        assert result is not None
        assert result.url == sample_cache_entry.url
        assert browser_cache._cache_stats["hits"] == 1
        assert mock_distributed_cache.get_count == 1

        # Should be promoted to local cache
        assert key in mock_local_cache.data

    async def test_get_cache_miss(self, browser_cache):
        """Test cache miss."""
        result = await browser_cache.get("browser:nonexistent:key")

        assert result is None
        assert browser_cache._cache_stats["misses"] == 1

    async def test_set_to_both_caches(
        self, browser_cache, mock_local_cache, mock_distributed_cache, sample_cache_entry
    ):
        """Test setting entry to both caches."""
        key = browser_cache._generate_cache_key(sample_cache_entry.url)
        success = await browser_cache.set(key, sample_cache_entry)

        assert success is True
        assert browser_cache._cache_stats["sets"] == 1
        assert key in mock_local_cache.data
        assert key in mock_distributed_cache.data

    async def test_set_with_custom_ttl(self, browser_cache, sample_cache_entry):
        """Test setting entry with custom TTL."""
        key = browser_cache._generate_cache_key(sample_cache_entry.url)
        success = await browser_cache.set(key, sample_cache_entry, ttl=60)

        assert success is True

    async def test_delete_from_both_caches(
        self, browser_cache, mock_local_cache, mock_distributed_cache, sample_cache_entry
    ):
        """Test deleting from both caches."""
        key = browser_cache._generate_cache_key(sample_cache_entry.url)

        # Set in both caches
        await browser_cache.set(key, sample_cache_entry)
        assert key in mock_local_cache.data
        assert key in mock_distributed_cache.data

        # Delete
        success = await browser_cache.delete(key)
        assert success is True
        assert key not in mock_local_cache.data
        assert key not in mock_distributed_cache.data

    async def test_invalidate_pattern(self, browser_cache, mock_distributed_cache):
        """Test pattern-based cache invalidation."""
        # Mock distributed cache with pattern support
        mock_distributed_cache.invalidate_pattern = AsyncMock(return_value=5)

        count = await browser_cache.invalidate_pattern("example.com")

        assert count == 5
        assert browser_cache._cache_stats["evictions"] == 5
        mock_distributed_cache.invalidate_pattern.assert_called_once_with("browser:*example.com*")

    def test_get_stats(self, browser_cache):
        """Test cache statistics."""
        browser_cache._cache_stats = {
            "hits": 10,
            "misses": 5,
            "sets": 8,
            "evictions": 2,
        }

        stats = browser_cache.get_stats()

        assert stats["hits"] == 10
        assert stats["misses"] == 5
        assert stats["sets"] == 8
        assert stats["evictions"] == 2
        assert stats["total_requests"] == 15
        assert stats["hit_rate"] == 10 / 15

    def test_get_stats_no_requests(self, browser_cache):
        """Test cache statistics with no requests."""
        stats = browser_cache.get_stats()

        assert stats["total_requests"] == 0
        assert stats["hit_rate"] == 0.0

    @patch("src.services.cache.browser_cache.logger")
    async def test_get_with_error_handling(self, mock_logger, browser_cache, mock_local_cache):
        """Test error handling in get operation."""
        # Make local cache raise an exception
        mock_local_cache.get = AsyncMock(side_effect=Exception("Test error"))

        result = await browser_cache.get("test_key")

        assert result is None
        assert browser_cache._cache_stats["misses"] == 1
        mock_logger.warning.assert_called()

    @patch("src.services.cache.browser_cache.logger")
    async def test_set_with_error_handling(self, mock_logger, browser_cache, mock_local_cache, sample_cache_entry):
        """Test error handling in set operation."""
        # Make local cache raise an exception
        mock_local_cache.set = AsyncMock(side_effect=Exception("Test error"))

        key = browser_cache._generate_cache_key(sample_cache_entry.url)
        success = await browser_cache.set(key, sample_cache_entry)

        # Should still return False on error
        assert success is False
        mock_logger.error.assert_called()

    async def test_get_or_fetch_cache_hit(self, browser_cache, sample_cache_entry):
        """Test get_or_fetch with cache hit."""
        # Pre-populate cache
        key = browser_cache._generate_cache_key(sample_cache_entry.url, "crawl4ai")
        await browser_cache.set(key, sample_cache_entry)

        # Mock fetch function (should not be called)
        fetch_func = AsyncMock()

        # Get or fetch
        result, was_cached = await browser_cache.get_or_fetch(
            sample_cache_entry.url, "crawl4ai", fetch_func
        )

        assert was_cached is True
        assert result.url == sample_cache_entry.url
        fetch_func.assert_not_called()

    async def test_get_or_fetch_cache_miss(self, browser_cache):
        """Test get_or_fetch with cache miss."""
        # Mock fetch function
        fetch_result = {
            "success": True,
            "content": "Fresh content",
            "metadata": {"title": "Fresh"},
            "tier_used": "browser_use",
        }
        fetch_func = AsyncMock(return_value=fetch_result)

        # Get or fetch
        result, was_cached = await browser_cache.get_or_fetch(
            "https://example.com/new", "browser_use", fetch_func
        )

        assert was_cached is False
        assert result.content == "Fresh content"
        assert result.tier_used == "browser_use"
        fetch_func.assert_called_once()

        # Should be cached now
        key = browser_cache._generate_cache_key("https://example.com/new", "browser_use")
        cached = await browser_cache.get(key)
        assert cached is not None
        assert cached.content == "Fresh content"

    async def test_get_or_fetch_error_handling(self, browser_cache):
        """Test get_or_fetch error handling."""
        # Mock fetch function that raises error
        fetch_func = AsyncMock(side_effect=Exception("Fetch failed"))

        with pytest.raises(Exception, match="Fetch failed"):
            await browser_cache.get_or_fetch(
                "https://example.com/error", "crawl4ai", fetch_func
            )


class TestBrowserCacheIntegration:
    """Test browser cache integration scenarios."""

    async def test_concurrent_cache_operations(self, browser_cache, sample_cache_entry):
        """Test concurrent cache operations."""
        key = browser_cache._generate_cache_key(sample_cache_entry.url)

        # Concurrent sets
        tasks = [
            browser_cache.set(key, sample_cache_entry) for _ in range(5)
        ]
        results = await asyncio.gather(*tasks)
        assert all(results)

        # Concurrent gets
        tasks = [browser_cache.get(key) for _ in range(5)]
        results = await asyncio.gather(*tasks)
        assert all(r is not None for r in results)

    async def test_cache_expiration_behavior(self, browser_cache):
        """Test cache behavior with different TTLs."""
        # Create entries with different content types
        static_entry = BrowserCacheEntry(
            url="https://example.com/doc.pdf",
            content="PDF content",
            metadata={},
            tier_used="lightweight",
        )

        dynamic_entry = BrowserCacheEntry(
            url="https://api.example.com/data",
            content="API response",
            metadata={},
            tier_used="crawl4ai",
        )

        # Set entries
        static_key = browser_cache._generate_cache_key(static_entry.url)
        dynamic_key = browser_cache._generate_cache_key(dynamic_entry.url)

        await browser_cache.set(static_key, static_entry)  # Will use static TTL
        await browser_cache.set(dynamic_key, dynamic_entry)  # Will use dynamic TTL

        # Verify both are cached
        assert await browser_cache.get(static_key) is not None
        assert await browser_cache.get(dynamic_key) is not None

    async def test_cache_without_distributed(self, mock_local_cache):
        """Test browser cache with only local cache."""
        cache = BrowserCache(
            local_cache=mock_local_cache,
            distributed_cache=None,  # No distributed cache
        )

        entry = BrowserCacheEntry(
            url="https://example.com",
            content="Test",
            metadata={},
            tier_used="crawl4ai",
        )

        key = cache._generate_cache_key(entry.url)

        # Should still work with only local cache
        success = await cache.set(key, entry)
        assert success is True

        result = await cache.get(key)
        assert result is not None
        assert result.url == entry.url

    async def test_cache_without_local(self, mock_distributed_cache):
        """Test browser cache with only distributed cache."""
        cache = BrowserCache(
            local_cache=None,  # No local cache
            distributed_cache=mock_distributed_cache,
        )

        entry = BrowserCacheEntry(
            url="https://example.com",
            content="Test",
            metadata={},
            tier_used="browser_use",
        )

        key = cache._generate_cache_key(entry.url)

        # Should still work with only distributed cache
        success = await cache.set(key, entry)
        assert success is True

        result = await cache.get(key)
        assert result is not None
        assert result.url == entry.url
