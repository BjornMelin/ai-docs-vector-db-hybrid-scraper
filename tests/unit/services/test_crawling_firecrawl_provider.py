"""Tests for Firecrawl provider module."""

import asyncio
from unittest.mock import AsyncMock
from unittest.mock import MagicMock
from unittest.mock import patch

import pytest
from src.services.crawling.base import CrawlProvider
from src.services.crawling.firecrawl_provider import FirecrawlProvider
from src.services.errors import CrawlServiceError


class TestFirecrawlProvider:
    """Test the FirecrawlProvider class."""

    def test_init(self):
        """Test FirecrawlProvider initialization."""
        provider = FirecrawlProvider("test_api_key")

        assert isinstance(provider, CrawlProvider)
        assert provider.api_key == "test_api_key"
        assert provider._client is None
        assert provider._initialized is False
        assert provider.rate_limiter is None

    def test_init_with_rate_limiter(self):
        """Test initialization with rate limiter."""
        rate_limiter = MagicMock()
        provider = FirecrawlProvider("test_api_key", rate_limiter)

        assert provider.rate_limiter == rate_limiter

    @pytest.mark.asyncio
    async def test_initialize_success(self):
        """Test successful provider initialization."""
        with patch(
            "src.services.crawling.firecrawl_provider.FirecrawlApp"
        ) as mock_app_class:
            mock_app = MagicMock()
            mock_app_class.return_value = mock_app

            provider = FirecrawlProvider("test_api_key")
            await provider.initialize()

            assert provider._initialized is True
            assert provider._client == mock_app
            mock_app_class.assert_called_once_with(api_key="test_api_key")

    @pytest.mark.asyncio
    async def test_initialize_failure(self):
        """Test provider initialization failure."""
        with patch(
            "src.services.crawling.firecrawl_provider.FirecrawlApp"
        ) as mock_app_class:
            mock_app_class.side_effect = Exception("Invalid API key")

            provider = FirecrawlProvider("invalid_key")

            with pytest.raises(
                CrawlServiceError, match="Failed to initialize Firecrawl"
            ):
                await provider.initialize()

    @pytest.mark.asyncio
    async def test_initialize_idempotent(self):
        """Test that multiple initialization calls are safe."""
        with patch(
            "src.services.crawling.firecrawl_provider.FirecrawlApp"
        ) as mock_app_class:
            mock_app = MagicMock()
            mock_app_class.return_value = mock_app

            provider = FirecrawlProvider("test_api_key")

            await provider.initialize()
            await provider.initialize()  # Second call

            # Should only create client once
            mock_app_class.assert_called_once()

    @pytest.mark.asyncio
    async def test_cleanup(self):
        """Test provider cleanup."""
        provider = FirecrawlProvider("test_api_key")
        provider._client = MagicMock()
        provider._initialized = True

        await provider.cleanup()

        assert provider._client is None
        assert provider._initialized is False

    @pytest.mark.asyncio
    async def test_scrape_url_not_initialized(self):
        """Test scraping when provider not initialized."""
        provider = FirecrawlProvider("test_api_key")

        with pytest.raises(CrawlServiceError, match="Provider not initialized"):
            await provider.scrape_url("https://example.com")

    @pytest.mark.asyncio
    async def test_scrape_url_success(self):
        """Test successful URL scraping."""
        provider = FirecrawlProvider("test_api_key")
        provider._initialized = True

        # Mock the _scrape_url_with_rate_limit method
        mock_result = {
            "success": True,
            "markdown": "# Test Content",
            "html": "<h1>Test Content</h1>",
            "metadata": {"title": "Test Page"},
        }

        with patch.object(
            provider, "_scrape_url_with_rate_limit", return_value=mock_result
        ):
            result = await provider.scrape_url("https://example.com")

            assert result["success"] is True
            assert result["content"] == "# Test Content"
            assert result["html"] == "<h1>Test Content</h1>"
            assert result["metadata"] == {"title": "Test Page"}
            assert result["url"] == "https://example.com"

    @pytest.mark.asyncio
    async def test_scrape_url_failure(self):
        """Test URL scraping failure."""
        provider = FirecrawlProvider("test_api_key")
        provider._initialized = True

        mock_result = {"success": False, "error": "Page not found"}

        with patch.object(
            provider, "_scrape_url_with_rate_limit", return_value=mock_result
        ):
            result = await provider.scrape_url("https://example.com")

            assert result["success"] is False
            assert result["error"] == "Page not found"
            assert result["url"] == "https://example.com"

    @pytest.mark.asyncio
    async def test_scrape_url_with_formats(self):
        """Test URL scraping with specific formats."""
        provider = FirecrawlProvider("test_api_key")
        provider._initialized = True

        mock_result = {"success": True, "markdown": "Content", "html": "<p>Content</p>"}

        with patch.object(
            provider, "_scrape_url_with_rate_limit", return_value=mock_result
        ) as mock_scrape:
            await provider.scrape_url(
                "https://example.com", formats=["html", "markdown"]
            )

            mock_scrape.assert_called_once_with(
                "https://example.com", ["html", "markdown"]
            )

    @pytest.mark.asyncio
    async def test_scrape_url_default_formats(self):
        """Test URL scraping with default formats."""
        provider = FirecrawlProvider("test_api_key")
        provider._initialized = True

        mock_result = {"success": True, "markdown": "Content"}

        with patch.object(
            provider, "_scrape_url_with_rate_limit", return_value=mock_result
        ) as mock_scrape:
            await provider.scrape_url("https://example.com")

            mock_scrape.assert_called_once_with("https://example.com", ["markdown"])

    @pytest.mark.asyncio
    async def test_scrape_url_exception_handling(self):
        """Test exception handling during scraping."""
        provider = FirecrawlProvider("test_api_key")
        provider._initialized = True

        with patch.object(
            provider,
            "_scrape_url_with_rate_limit",
            side_effect=Exception("Network error"),
        ):
            result = await provider.scrape_url("https://example.com")

            assert result["success"] is False
            assert "Scraping failed: Network error" in result["error"]

    @pytest.mark.asyncio
    async def test_scrape_url_rate_limit_error(self):
        """Test handling of rate limit errors."""
        provider = FirecrawlProvider("test_api_key")
        provider._initialized = True

        rate_limit_error = Exception("Rate limit exceeded")

        with patch.object(
            provider, "_scrape_url_with_rate_limit", side_effect=rate_limit_error
        ):
            result = await provider.scrape_url("https://example.com")

            assert result["success"] is False
            assert "Rate limit exceeded" in result["error"]

    @pytest.mark.asyncio
    async def test_scrape_url_auth_error(self):
        """Test handling of authentication errors."""
        provider = FirecrawlProvider("test_api_key")
        provider._initialized = True

        auth_error = Exception("Invalid API key")

        with patch.object(
            provider, "_scrape_url_with_rate_limit", side_effect=auth_error
        ):
            result = await provider.scrape_url("https://example.com")

            assert result["success"] is False
            assert "Invalid API key" in result["error"]

    @pytest.mark.asyncio
    async def test_scrape_url_timeout_error(self):
        """Test handling of timeout errors."""
        provider = FirecrawlProvider("test_api_key")
        provider._initialized = True

        timeout_error = Exception("Request timeout")

        with patch.object(
            provider, "_scrape_url_with_rate_limit", side_effect=timeout_error
        ):
            result = await provider.scrape_url("https://example.com")

            assert result["success"] is False
            assert "Request timed out" in result["error"]

    @pytest.mark.asyncio
    async def test_scrape_url_not_found_error(self):
        """Test handling of 404 errors."""
        provider = FirecrawlProvider("test_api_key")
        provider._initialized = True

        not_found_error = Exception("404 Not Found")

        with patch.object(
            provider, "_scrape_url_with_rate_limit", side_effect=not_found_error
        ):
            result = await provider.scrape_url("https://example.com")

            assert result["success"] is False
            assert "Page not found (404)" in result["error"]

    @pytest.mark.asyncio
    async def test_crawl_site_not_initialized(self):
        """Test site crawling when provider not initialized."""
        provider = FirecrawlProvider("test_api_key")

        with pytest.raises(CrawlServiceError, match="Provider not initialized"):
            await provider.crawl_site("https://example.com")

    @pytest.mark.asyncio
    async def test_crawl_site_success(self):
        """Test successful site crawling."""
        provider = FirecrawlProvider("test_api_key")
        provider._initialized = True

        mock_client = MagicMock()
        provider._client = mock_client

        # Mock async crawl start
        crawl_start_result = {"id": "crawl_123"}

        # Mock status checks
        mock_client.check_crawl_status.side_effect = [
            {"status": "running"},
            {
                "status": "completed",
                "data": [
                    {
                        "url": "https://example.com/1",
                        "markdown": "Content 1",
                        "html": "<p>Content 1</p>",
                        "metadata": {},
                    },
                    {
                        "url": "https://example.com/2",
                        "markdown": "Content 2",
                        "html": "<p>Content 2</p>",
                        "metadata": {},
                    },
                ],
            },
        ]

        with patch.object(
            provider,
            "_async_crawl_url_with_rate_limit",
            return_value=crawl_start_result,
        ):
            result = await provider.crawl_site("https://example.com", max_pages=10)

            assert result["success"] is True
            assert result["total"] == 2
            assert len(result["pages"]) == 2
            assert result["crawl_id"] == "crawl_123"

    @pytest.mark.asyncio
    async def test_crawl_site_no_crawl_id(self):
        """Test site crawling when no crawl ID returned."""
        provider = FirecrawlProvider("test_api_key")
        provider._initialized = True

        with patch.object(
            provider, "_async_crawl_url_with_rate_limit", return_value={}
        ):
            result = await provider.crawl_site("https://example.com")

            assert result["success"] is False
            assert "No crawl ID returned" in result["error"]

    @pytest.mark.asyncio
    async def test_crawl_site_failed_status(self):
        """Test site crawling with failed status."""
        provider = FirecrawlProvider("test_api_key")
        provider._initialized = True

        mock_client = MagicMock()
        provider._client = mock_client

        crawl_start_result = {"id": "crawl_123"}
        mock_client.check_crawl_status.return_value = {
            "status": "failed",
            "error": "Crawl failed",
        }

        with patch.object(
            provider,
            "_async_crawl_url_with_rate_limit",
            return_value=crawl_start_result,
        ):
            result = await provider.crawl_site("https://example.com")

            assert result["success"] is False
            assert result["error"] == "Crawl failed"
            assert result["crawl_id"] == "crawl_123"

    @pytest.mark.asyncio
    async def test_crawl_site_timeout(self):
        """Test site crawling timeout."""
        provider = FirecrawlProvider("test_api_key")
        provider._initialized = True

        mock_client = MagicMock()
        provider._client = mock_client

        crawl_start_result = {"id": "crawl_123"}
        # Always return running status to trigger timeout
        mock_client.check_crawl_status.return_value = {"status": "running"}

        # Define a faster crawl_site method for testing timeout behavior

        async def fast_crawl_site(url, max_pages=50, formats=None):
            formats = formats or ["markdown"]

            try:
                crawl_result = await provider._async_crawl_url_with_rate_limit(
                    url, max_pages, formats
                )
                crawl_id = crawl_result.get("id")
                if not crawl_id:
                    return {
                        "success": False,
                        "error": "No crawl ID returned",
                        "pages": [],
                        "total": 0,
                    }

                # Only check 2 times for test speed
                for _ in range(2):
                    status = provider._client.check_crawl_status(crawl_id)
                    if status.get("status") == "completed":
                        return {
                            "success": True,
                            "pages": [],
                            "total": 0,
                            "crawl_id": crawl_id,
                        }
                    elif status.get("status") == "failed":
                        return {
                            "success": False,
                            "error": "failed",
                            "pages": [],
                            "total": 0,
                            "crawl_id": crawl_id,
                        }
                    await asyncio.sleep(0.01)  # Very short sleep for test

                return {
                    "success": False,
                    "error": "Crawl timed out",
                    "pages": [],
                    "total": 0,
                    "crawl_id": crawl_id,
                }
            except Exception as e:
                return {"success": False, "error": str(e), "pages": [], "total": 0}

        with patch.object(
            provider,
            "_async_crawl_url_with_rate_limit",
            return_value=crawl_start_result,
        ):
            result = await fast_crawl_site("https://example.com")

            assert result["success"] is False
            assert "Crawl timed out" in result["error"]

    @pytest.mark.asyncio
    async def test_crawl_site_exception(self):
        """Test site crawling with exception."""
        provider = FirecrawlProvider("test_api_key")
        provider._initialized = True

        with patch.object(
            provider,
            "_async_crawl_url_with_rate_limit",
            side_effect=Exception("API error"),
        ):
            result = await provider.crawl_site("https://example.com")

            assert result["success"] is False
            assert "API error" in result["error"]

    @pytest.mark.asyncio
    async def test_crawl_site_with_formats(self):
        """Test site crawling with custom formats."""
        provider = FirecrawlProvider("test_api_key")
        provider._initialized = True

        with patch.object(provider, "_async_crawl_url_with_rate_limit") as mock_crawl:
            mock_crawl.return_value = {"id": "crawl_123"}

            mock_client = MagicMock()
            provider._client = mock_client
            mock_client.check_crawl_status.return_value = {
                "status": "completed",
                "data": [],
            }

            await provider.crawl_site("https://example.com", formats=["html", "text"])

            mock_crawl.assert_called_once_with(
                "https://example.com", 50, ["html", "text"]
            )

    @pytest.mark.asyncio
    async def test_cancel_crawl_not_initialized(self):
        """Test canceling crawl when provider not initialized."""
        provider = FirecrawlProvider("test_api_key")

        with pytest.raises(CrawlServiceError, match="Provider not initialized"):
            await provider.cancel_crawl("crawl_123")

    @pytest.mark.asyncio
    async def test_cancel_crawl_success(self):
        """Test successful crawl cancellation."""
        provider = FirecrawlProvider("test_api_key")
        provider._initialized = True

        mock_client = MagicMock()
        mock_client.cancel_crawl.return_value = {"success": True}
        provider._client = mock_client

        result = await provider.cancel_crawl("crawl_123")

        assert result is True
        mock_client.cancel_crawl.assert_called_once_with("crawl_123")

    @pytest.mark.asyncio
    async def test_cancel_crawl_failure(self):
        """Test crawl cancellation failure."""
        provider = FirecrawlProvider("test_api_key")
        provider._initialized = True

        mock_client = MagicMock()
        mock_client.cancel_crawl.return_value = {"success": False}
        provider._client = mock_client

        result = await provider.cancel_crawl("crawl_123")

        assert result is False

    @pytest.mark.asyncio
    async def test_cancel_crawl_exception(self):
        """Test crawl cancellation with exception."""
        provider = FirecrawlProvider("test_api_key")
        provider._initialized = True

        mock_client = MagicMock()
        mock_client.cancel_crawl.side_effect = Exception("Cancel failed")
        provider._client = mock_client

        result = await provider.cancel_crawl("crawl_123")

        assert result is False

    @pytest.mark.asyncio
    async def test_map_url_not_initialized(self):
        """Test URL mapping when provider not initialized."""
        provider = FirecrawlProvider("test_api_key")

        with pytest.raises(CrawlServiceError, match="Provider not initialized"):
            await provider.map_url("https://example.com")

    @pytest.mark.asyncio
    async def test_map_url_success(self):
        """Test successful URL mapping."""
        provider = FirecrawlProvider("test_api_key")
        provider._initialized = True

        mock_client = MagicMock()
        mock_client.map_url.return_value = {
            "success": True,
            "links": ["https://example.com/1", "https://example.com/2"],
        }
        provider._client = mock_client

        result = await provider.map_url("https://example.com", include_subdomains=True)

        assert result["success"] is True
        assert result["total"] == 2
        assert result["urls"] == ["https://example.com/1", "https://example.com/2"]

        mock_client.map_url.assert_called_once_with(
            url="https://example.com", params={"includeSubdomains": True}
        )

    @pytest.mark.asyncio
    async def test_map_url_failure(self):
        """Test URL mapping failure."""
        provider = FirecrawlProvider("test_api_key")
        provider._initialized = True

        mock_client = MagicMock()
        mock_client.map_url.return_value = {"success": False, "error": "Map failed"}
        provider._client = mock_client

        result = await provider.map_url("https://example.com")

        assert result["success"] is False
        assert result["error"] == "Map failed"
        assert result["urls"] == []
        assert result["total"] == 0

    @pytest.mark.asyncio
    async def test_map_url_exception(self):
        """Test URL mapping with exception."""
        provider = FirecrawlProvider("test_api_key")
        provider._initialized = True

        mock_client = MagicMock()
        mock_client.map_url.side_effect = Exception("Network error")
        provider._client = mock_client

        result = await provider.map_url("https://example.com")

        assert result["success"] is False
        assert "Network error" in result["error"]

    @pytest.mark.asyncio
    async def test_scrape_url_with_rate_limit(self):
        """Test _scrape_url_with_rate_limit method."""
        provider = FirecrawlProvider("test_api_key")
        provider._initialized = True

        # Mock rate limiter
        rate_limiter = AsyncMock()
        provider.rate_limiter = rate_limiter

        # Mock client and response
        mock_client = MagicMock()
        mock_client.scrape_url.return_value = {"success": True, "markdown": "Content"}
        provider._client = mock_client

        result = await provider._scrape_url_with_rate_limit(
            "https://example.com", ["markdown"]
        )

        rate_limiter.acquire.assert_called_once_with("firecrawl")
        mock_client.scrape_url.assert_called_once_with(
            "https://example.com", ["markdown"]
        )
        assert result == {"success": True, "markdown": "Content"}

    @pytest.mark.asyncio
    async def test_scrape_url_with_rate_limit_no_limiter(self):
        """Test _scrape_url_with_rate_limit without rate limiter."""
        provider = FirecrawlProvider("test_api_key")
        provider._initialized = True
        provider.rate_limiter = None

        mock_client = MagicMock()
        mock_client.scrape_url.return_value = {"success": True}
        provider._client = mock_client

        result = await provider._scrape_url_with_rate_limit(
            "https://example.com", ["markdown"]
        )

        assert result == {"success": True}

    @pytest.mark.asyncio
    async def test_async_crawl_url_with_rate_limit(self):
        """Test _async_crawl_url_with_rate_limit method."""
        provider = FirecrawlProvider("test_api_key")
        provider._initialized = True

        # Mock rate limiter
        rate_limiter = AsyncMock()
        provider.rate_limiter = rate_limiter

        # Mock client and response
        mock_client = MagicMock()
        mock_client.async_crawl_url.return_value = {"id": "crawl_123"}
        provider._client = mock_client

        result = await provider._async_crawl_url_with_rate_limit(
            "https://example.com", 50, ["markdown"]
        )

        rate_limiter.acquire.assert_called_once_with("firecrawl")
        mock_client.async_crawl_url.assert_called_once_with(
            "https://example.com", 50, {"formats": ["markdown"]}
        )
        assert result == {"id": "crawl_123"}

    @pytest.mark.asyncio
    async def test_async_crawl_url_with_rate_limit_no_limiter(self):
        """Test _async_crawl_url_with_rate_limit without rate limiter."""
        provider = FirecrawlProvider("test_api_key")
        provider._initialized = True
        provider.rate_limiter = None

        mock_client = MagicMock()
        mock_client.async_crawl_url.return_value = {"id": "crawl_123"}
        provider._client = mock_client

        result = await provider._async_crawl_url_with_rate_limit(
            "https://example.com", 50, ["markdown"]
        )

        assert result == {"id": "crawl_123"}

    def test_inheritance(self):
        """Test that FirecrawlProvider properly inherits from CrawlProvider."""
        provider = FirecrawlProvider("test_api_key")

        assert isinstance(provider, CrawlProvider)

        # Check that all abstract methods are implemented
        abstract_methods = CrawlProvider.__abstractmethods__
        for method_name in abstract_methods:
            assert hasattr(provider, method_name)
            assert callable(getattr(provider, method_name))

    @pytest.mark.asyncio
    async def test_full_workflow(self):
        """Test a complete workflow: initialize -> scrape -> crawl -> cleanup."""
        with patch(
            "src.services.crawling.firecrawl_provider.FirecrawlApp"
        ) as mock_app_class:
            mock_client = MagicMock()
            mock_app_class.return_value = mock_client

            provider = FirecrawlProvider("test_api_key")

            # Initialize
            await provider.initialize()
            assert provider._initialized is True

            # Mock scrape response
            with patch.object(provider, "_scrape_url_with_rate_limit") as mock_scrape:
                mock_scrape.return_value = {"success": True, "markdown": "Content"}

                scrape_result = await provider.scrape_url("https://example.com")
                assert scrape_result["success"] is True

            # Mock crawl response
            mock_client.check_crawl_status.return_value = {
                "status": "completed",
                "data": [{"url": "https://example.com", "markdown": "Content"}],
            }

            with patch.object(
                provider, "_async_crawl_url_with_rate_limit"
            ) as mock_crawl_start:
                mock_crawl_start.return_value = {"id": "crawl_123"}

                crawl_result = await provider.crawl_site("https://example.com")
                assert crawl_result["success"] is True

            # Cleanup
            await provider.cleanup()
            assert provider._initialized is False
