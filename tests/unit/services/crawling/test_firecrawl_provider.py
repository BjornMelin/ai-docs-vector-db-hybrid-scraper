"""Comprehensive tests for Firecrawl provider with Pydantic configuration."""

import asyncio
from unittest.mock import AsyncMock
from unittest.mock import MagicMock
from unittest.mock import patch

import pytest
from src.config.models import FirecrawlConfig
from src.services.base import BaseService
from src.services.crawling.base import CrawlProvider
from src.services.crawling.firecrawl_provider import FirecrawlProvider
from src.services.errors import CrawlServiceError


@pytest.fixture
def basic_config():
    """Create basic Firecrawl configuration."""
    return FirecrawlConfig(
        api_key="fc-test-key-12345",
        api_url="https://api.firecrawl.dev",
        timeout=30.0,
    )


@pytest.fixture
def mock_rate_limiter():
    """Create mock rate limiter."""
    return AsyncMock()


@pytest.fixture
def mock_firecrawl_app():
    """Create mock FirecrawlApp."""
    return MagicMock()


class TestFirecrawlProvider:
    """Test FirecrawlProvider class with Pydantic configuration."""

    def test_init_basic(self, basic_config):
        """Test basic initialization with Pydantic config."""
        provider = FirecrawlProvider(basic_config)

        assert isinstance(provider, BaseService)
        assert isinstance(provider, CrawlProvider)
        assert provider.config == basic_config
        assert provider._client is None
        assert provider._initialized is False
        assert provider.rate_limiter is None

    def test_init_with_rate_limiter(self, basic_config, mock_rate_limiter):
        """Test initialization with custom rate limiter."""
        provider = FirecrawlProvider(basic_config, mock_rate_limiter)
        assert provider.rate_limiter == mock_rate_limiter

    @pytest.mark.asyncio
    @patch("src.services.crawling.firecrawl_provider.FirecrawlApp")
    async def test_initialize_success(self, mock_app_class, basic_config):
        """Test successful provider initialization."""
        mock_app = MagicMock()
        mock_app_class.return_value = mock_app

        provider = FirecrawlProvider(basic_config)
        await provider.initialize()

        assert provider._initialized is True
        assert provider._client == mock_app
        mock_app_class.assert_called_once_with(
            api_key=basic_config.api_key, api_url=basic_config.api_url
        )

    @pytest.mark.asyncio
    @patch("src.services.crawling.firecrawl_provider.FirecrawlApp")
    async def test_initialize_already_initialized(self, mock_app_class, basic_config):
        """Test initialization when already initialized."""
        provider = FirecrawlProvider(basic_config)
        provider._initialized = True

        await provider.initialize()

        # Should not create new client
        mock_app_class.assert_not_called()

    @pytest.mark.asyncio
    @patch("src.services.crawling.firecrawl_provider.FirecrawlApp")
    async def test_initialize_failure(self, mock_app_class, basic_config):
        """Test initialization failure."""
        mock_app_class.side_effect = Exception("API key invalid")

        provider = FirecrawlProvider(basic_config)

        with pytest.raises(CrawlServiceError, match="Failed to initialize Firecrawl"):
            await provider.initialize()

    @pytest.mark.asyncio
    async def test_cleanup(self, basic_config):
        """Test cleanup functionality."""
        provider = FirecrawlProvider(basic_config)
        provider._client = MagicMock()
        provider._initialized = True

        await provider.cleanup()

        assert provider._client is None
        assert provider._initialized is False

    @pytest.mark.asyncio
    async def test_scrape_url_not_initialized(self, basic_config):
        """Test scraping when provider not initialized."""
        provider = FirecrawlProvider(basic_config)

        with pytest.raises(CrawlServiceError, match="Provider not initialized"):
            await provider.scrape_url("https://example.com")

    @pytest.mark.asyncio
    async def test_scrape_url_success(self, basic_config):
        """Test successful URL scraping."""
        provider = FirecrawlProvider(basic_config)
        mock_client = MagicMock()
        provider._client = mock_client
        provider._initialized = True

        # Mock successful response
        mock_firecrawl_result = {
            "success": True,
            "markdown": "# Test Content",
            "html": "<h1>Test Content</h1>",
            "metadata": {"title": "Test Page"},
        }

        with patch.object(provider, "_scrape_url_with_rate_limit") as mock_scrape:
            mock_scrape.return_value = mock_firecrawl_result

            result = await provider.scrape_url("https://example.com")

            assert result["success"] is True
            assert result["content"] == "# Test Content"
            assert result["html"] == "<h1>Test Content</h1>"
            assert result["metadata"]["title"] == "Test Page"
            assert result["url"] == "https://example.com"

    @pytest.mark.asyncio
    async def test_scrape_url_with_formats(self, basic_config):
        """Test URL scraping with custom formats."""
        provider = FirecrawlProvider(basic_config)
        mock_client = MagicMock()
        provider._client = mock_client
        provider._initialized = True

        mock_firecrawl_result = {"success": True, "markdown": "Content"}

        with patch.object(provider, "_scrape_url_with_rate_limit") as mock_scrape:
            mock_scrape.return_value = mock_firecrawl_result

            await provider.scrape_url(
                "https://example.com", formats=["markdown", "html"]
            )

            mock_scrape.assert_called_once_with(
                "https://example.com", ["markdown", "html"]
            )

    @pytest.mark.asyncio
    async def test_scrape_url_failure_response(self, basic_config):
        """Test URL scraping with failure response."""
        provider = FirecrawlProvider(basic_config)
        mock_client = MagicMock()
        provider._client = mock_client
        provider._initialized = True

        # Mock failure response
        mock_firecrawl_result = {
            "success": False,
            "error": "Page not found",
        }

        with patch.object(provider, "_scrape_url_with_rate_limit") as mock_scrape:
            mock_scrape.return_value = mock_firecrawl_result

            result = await provider.scrape_url("https://example.com")

            assert result["success"] is False
            assert result["error"] == "Page not found"
            assert result["content"] == ""
            assert result["url"] == "https://example.com"

    @pytest.mark.asyncio
    async def test_scrape_url_rate_limit_error(self, basic_config):
        """Test URL scraping with rate limit error."""
        provider = FirecrawlProvider(basic_config)
        mock_client = MagicMock()
        provider._client = mock_client
        provider._initialized = True

        with patch.object(provider, "_scrape_url_with_rate_limit") as mock_scrape:
            mock_scrape.side_effect = Exception("Rate limit exceeded")

            result = await provider.scrape_url("https://example.com")

            assert result["success"] is False
            assert "Rate limit exceeded" in result["error"]
            assert result["url"] == "https://example.com"

    @pytest.mark.asyncio
    async def test_scrape_url_auth_error(self, basic_config):
        """Test URL scraping with authentication error."""
        provider = FirecrawlProvider(basic_config)
        mock_client = MagicMock()
        provider._client = mock_client
        provider._initialized = True

        with patch.object(provider, "_scrape_url_with_rate_limit") as mock_scrape:
            mock_scrape.side_effect = Exception("Invalid API key")

            result = await provider.scrape_url("https://example.com")

            assert result["success"] is False
            assert "Invalid API key" in result["error"]

    @pytest.mark.asyncio
    async def test_scrape_url_timeout_error(self, basic_config):
        """Test URL scraping with timeout error."""
        provider = FirecrawlProvider(basic_config)
        mock_client = MagicMock()
        provider._client = mock_client
        provider._initialized = True

        with patch.object(provider, "_scrape_url_with_rate_limit") as mock_scrape:
            mock_scrape.side_effect = Exception("Request timeout")

            result = await provider.scrape_url("https://example.com")

            assert result["success"] is False
            assert "Request timed out" in result["error"]

    @pytest.mark.asyncio
    async def test_scrape_url_not_found_error(self, basic_config):
        """Test URL scraping with 404 error."""
        provider = FirecrawlProvider(basic_config)
        mock_client = MagicMock()
        provider._client = mock_client
        provider._initialized = True

        with patch.object(provider, "_scrape_url_with_rate_limit") as mock_scrape:
            mock_scrape.side_effect = Exception("404 not found")

            result = await provider.scrape_url("https://example.com")

            assert result["success"] is False
            assert "Page not found (404)" in result["error"]

    @pytest.mark.asyncio
    async def test_crawl_site_not_initialized(self, basic_config):
        """Test site crawling when not initialized."""
        provider = FirecrawlProvider(basic_config)

        with pytest.raises(CrawlServiceError, match="Provider not initialized"):
            await provider.crawl_site("https://example.com")

    @pytest.mark.asyncio
    async def test_crawl_site_success(self, basic_config):
        """Test successful site crawling."""
        provider = FirecrawlProvider(basic_config)
        mock_client = MagicMock()
        provider._client = mock_client
        provider._initialized = True

        # Mock crawl start
        with patch.object(provider, "_async_crawl_url_with_rate_limit") as mock_start:
            mock_start.return_value = {"id": "crawl-123"}

            # Mock status checks
            mock_client.check_crawl_status.side_effect = [
                {"status": "running"},
                {
                    "status": "completed",
                    "data": [
                        {
                            "url": "https://example.com",
                            "markdown": "Home content",
                            "html": "<h1>Home</h1>",
                            "metadata": {"title": "Home"},
                        },
                        {
                            "url": "https://example.com/about",
                            "markdown": "About content",
                            "html": "<h1>About</h1>",
                            "metadata": {"title": "About"},
                        },
                    ],
                },
            ]

            result = await provider.crawl_site("https://example.com", max_pages=10)

            assert result["success"] is True
            assert len(result["pages"]) == 2
            assert result["total"] == 2
            assert result["crawl_id"] == "crawl-123"
            assert result["pages"][0]["url"] == "https://example.com"
            assert result["pages"][0]["content"] == "Home content"

    @pytest.mark.asyncio
    async def test_crawl_site_failed(self, basic_config):
        """Test site crawling failure."""
        provider = FirecrawlProvider(basic_config)
        mock_client = MagicMock()
        provider._client = mock_client
        provider._initialized = True

        # Mock crawl start
        with patch.object(provider, "_async_crawl_url_with_rate_limit") as mock_start:
            mock_start.return_value = {"id": "crawl-123"}

            # Mock status check showing failure
            mock_client.check_crawl_status.return_value = {
                "status": "failed",
                "error": "Crawl failed due to network error",
            }

            result = await provider.crawl_site("https://example.com")

            assert result["success"] is False
            assert result["error"] == "Crawl failed due to network error"
            assert result["pages"] == []
            assert result["crawl_id"] == "crawl-123"

    @pytest.mark.asyncio
    async def test_crawl_site_timeout(self, basic_config):
        """Test site crawling timeout."""
        provider = FirecrawlProvider(basic_config)
        mock_client = MagicMock()
        provider._client = mock_client
        provider._initialized = True

        # Mock crawl start
        with patch.object(provider, "_async_crawl_url_with_rate_limit") as mock_start:
            mock_start.return_value = {"id": "crawl-123"}

            # Mock status check always showing running (simulating timeout)
            mock_client.check_crawl_status.return_value = {"status": "running"}

            with patch("asyncio.sleep"):  # Speed up the test
                result = await provider.crawl_site("https://example.com")

            assert result["success"] is False
            assert "Crawl timed out" in result["error"]
            assert result["crawl_id"] == "crawl-123"

    @pytest.mark.asyncio
    async def test_crawl_site_no_crawl_id(self, basic_config):
        """Test site crawling when no crawl ID returned."""
        provider = FirecrawlProvider(basic_config)
        mock_client = MagicMock()
        provider._client = mock_client
        provider._initialized = True

        # Mock crawl start with no ID
        with patch.object(provider, "_async_crawl_url_with_rate_limit") as mock_start:
            mock_start.return_value = {}  # No ID returned

            result = await provider.crawl_site("https://example.com")

            assert result["success"] is False
            assert "No crawl ID returned" in result["error"]

    @pytest.mark.asyncio
    async def test_crawl_site_exception(self, basic_config):
        """Test site crawling exception handling."""
        provider = FirecrawlProvider(basic_config)
        mock_client = MagicMock()
        provider._client = mock_client
        provider._initialized = True

        # Mock crawl start exception
        with patch.object(provider, "_async_crawl_url_with_rate_limit") as mock_start:
            mock_start.side_effect = Exception("Network error")

            result = await provider.crawl_site("https://example.com")

            assert result["success"] is False
            assert "Network error" in result["error"]

    @pytest.mark.asyncio
    async def test_cancel_crawl_not_initialized(self, basic_config):
        """Test canceling crawl when not initialized."""
        provider = FirecrawlProvider(basic_config)

        with pytest.raises(CrawlServiceError, match="Provider not initialized"):
            await provider.cancel_crawl("crawl-123")

    @pytest.mark.asyncio
    async def test_cancel_crawl_success(self, basic_config):
        """Test successful crawl cancellation."""
        provider = FirecrawlProvider(basic_config)
        mock_client = MagicMock()
        mock_client.cancel_crawl.return_value = {"success": True}
        provider._client = mock_client
        provider._initialized = True

        result = await provider.cancel_crawl("crawl-123")

        assert result is True
        mock_client.cancel_crawl.assert_called_once_with("crawl-123")

    @pytest.mark.asyncio
    async def test_cancel_crawl_failure(self, basic_config):
        """Test crawl cancellation failure."""
        provider = FirecrawlProvider(basic_config)
        mock_client = MagicMock()
        mock_client.cancel_crawl.return_value = {"success": False}
        provider._client = mock_client
        provider._initialized = True

        result = await provider.cancel_crawl("crawl-123")

        assert result is False

    @pytest.mark.asyncio
    async def test_cancel_crawl_exception(self, basic_config):
        """Test crawl cancellation exception handling."""
        provider = FirecrawlProvider(basic_config)
        mock_client = MagicMock()
        mock_client.cancel_crawl.side_effect = Exception("Cancel failed")
        provider._client = mock_client
        provider._initialized = True

        result = await provider.cancel_crawl("crawl-123")

        assert result is False

    @pytest.mark.asyncio
    async def test_map_url_not_initialized(self, basic_config):
        """Test URL mapping when not initialized."""
        provider = FirecrawlProvider(basic_config)

        with pytest.raises(CrawlServiceError, match="Provider not initialized"):
            await provider.map_url("https://example.com")

    @pytest.mark.asyncio
    async def test_map_url_success(self, basic_config):
        """Test successful URL mapping."""
        provider = FirecrawlProvider(basic_config)
        mock_client = MagicMock()
        mock_client.map_url.return_value = {
            "success": True,
            "links": [
                "https://example.com",
                "https://example.com/about",
                "https://example.com/contact",
            ],
        }
        provider._client = mock_client
        provider._initialized = True

        result = await provider.map_url("https://example.com", include_subdomains=True)

        assert result["success"] is True
        assert len(result["urls"]) == 3
        assert result["total"] == 3
        mock_client.map_url.assert_called_once_with(
            url="https://example.com", params={"includeSubdomains": True}
        )

    @pytest.mark.asyncio
    async def test_map_url_failure(self, basic_config):
        """Test URL mapping failure."""
        provider = FirecrawlProvider(basic_config)
        mock_client = MagicMock()
        mock_client.map_url.return_value = {"success": False, "error": "Mapping failed"}
        provider._client = mock_client
        provider._initialized = True

        result = await provider.map_url("https://example.com")

        assert result["success"] is False
        assert result["error"] == "Mapping failed"
        assert result["urls"] == []
        assert result["total"] == 0

    @pytest.mark.asyncio
    async def test_map_url_exception(self, basic_config):
        """Test URL mapping exception handling."""
        provider = FirecrawlProvider(basic_config)
        mock_client = MagicMock()
        mock_client.map_url.side_effect = Exception("Network error")
        provider._client = mock_client
        provider._initialized = True

        result = await provider.map_url("https://example.com")

        assert result["success"] is False
        assert "Network error" in result["error"]

    @pytest.mark.asyncio
    async def test_scrape_url_with_rate_limit(self, basic_config, mock_rate_limiter):
        """Test rate-limited scraping."""
        provider = FirecrawlProvider(basic_config, mock_rate_limiter)
        mock_client = MagicMock()
        mock_client.scrape_url.return_value = {"success": True}
        provider._client = mock_client

        with patch("asyncio.get_event_loop") as mock_loop:
            mock_executor = AsyncMock()
            mock_executor.return_value = {"success": True}
            mock_loop.return_value.run_in_executor = mock_executor

            result = await provider._scrape_url_with_rate_limit(
                "https://example.com", ["markdown"]
            )

            assert result["success"] is True
            mock_rate_limiter.acquire.assert_called_once_with("firecrawl")

    @pytest.mark.asyncio
    async def test_async_crawl_url_with_rate_limit(
        self, basic_config, mock_rate_limiter
    ):
        """Test rate-limited async crawl."""
        provider = FirecrawlProvider(basic_config, mock_rate_limiter)
        mock_client = MagicMock()
        mock_client.async_crawl_url.return_value = {"id": "crawl-123"}
        provider._client = mock_client

        with patch("asyncio.get_event_loop") as mock_loop:
            mock_executor = AsyncMock()
            mock_executor.return_value = {"id": "crawl-123"}
            mock_loop.return_value.run_in_executor = mock_executor

            result = await provider._async_crawl_url_with_rate_limit(
                "https://example.com", 10, ["markdown"]
            )

            assert result["id"] == "crawl-123"
            mock_rate_limiter.acquire.assert_called_once_with("firecrawl")


class TestPydanticConfigIntegration:
    """Test integration with Pydantic configuration models."""

    def test_config_validation(self):
        """Test that Pydantic config validation works."""
        # Valid config
        config = FirecrawlConfig(
            api_key="fc-test-key",
            api_url="https://api.firecrawl.dev",
            timeout=30.0,
        )

        provider = FirecrawlProvider(config)
        assert provider.config == config

    def test_config_field_access(self, basic_config):
        """Test accessing Pydantic config fields."""
        provider = FirecrawlProvider(basic_config)

        assert provider.config.api_key == "fc-test-key-12345"
        assert provider.config.api_url == "https://api.firecrawl.dev"
        assert provider.config.timeout == 30.0

    def test_config_defaults(self):
        """Test Pydantic config defaults."""
        config = FirecrawlConfig()  # Use all defaults
        provider = FirecrawlProvider(config)

        assert provider.config.api_key is None
        assert provider.config.api_url == "https://api.firecrawl.dev"
        assert provider.config.timeout == 30.0

    @pytest.mark.asyncio
    async def test_config_used_in_initialization(self, basic_config):
        """Test that Pydantic config is used in client initialization."""
        with patch(
            "src.services.crawling.firecrawl_provider.FirecrawlApp"
        ) as mock_app_class:
            mock_app = MagicMock()
            mock_app_class.return_value = mock_app

            provider = FirecrawlProvider(basic_config)
            await provider.initialize()

            # Verify config fields are passed to client
            mock_app_class.assert_called_once_with(
                api_key=basic_config.api_key, api_url=basic_config.api_url
            )

    def test_config_validation_api_key_format(self):
        """Test API key format validation."""
        # Valid key with fc- prefix
        config = FirecrawlConfig(api_key="fc-valid-key-123")
        provider = FirecrawlProvider(config)
        assert provider.config.api_key == "fc-valid-key-123"

    def test_config_inheritance(self, basic_config):
        """Test that provider inherits from BaseService with config."""
        provider = FirecrawlProvider(basic_config)

        # Should inherit from BaseService
        assert isinstance(provider, BaseService)
        # Config should be passed to parent
        assert hasattr(provider, "config")
        assert provider.config == basic_config


class TestErrorHandling:
    """Test comprehensive error handling scenarios."""

    @pytest.mark.asyncio
    async def test_multiple_error_types(self, basic_config):
        """Test handling of various error types."""
        provider = FirecrawlProvider(basic_config)
        provider._initialized = True

        error_scenarios = [
            ("rate limit exceeded", "Rate limit exceeded"),
            ("unauthorized", "Invalid API key"),
            ("timeout", "Request timed out"),
            ("404 not found", "Page not found (404)"),
            ("generic error", "Scraping failed: generic error"),
        ]

        for error_msg, expected_detail in error_scenarios:
            with patch.object(provider, "_scrape_url_with_rate_limit") as mock_scrape:
                mock_scrape.side_effect = Exception(error_msg)

                result = await provider.scrape_url("https://example.com")

                assert result["success"] is False
                assert expected_detail in result["error"]


class TestConcurrency:
    """Test concurrent operations."""

    @pytest.mark.asyncio
    async def test_concurrent_scraping(self, basic_config):
        """Test concurrent scraping operations."""
        provider = FirecrawlProvider(basic_config)
        provider._initialized = True

        urls = [f"https://example.com/{i}" for i in range(3)]

        with patch.object(provider, "_scrape_url_with_rate_limit") as mock_scrape:
            mock_scrape.return_value = {"success": True, "content": "test"}

            # Run concurrent scrapes
            tasks = [provider.scrape_url(url) for url in urls]
            results = await asyncio.gather(*tasks)

            assert len(results) == 3
            assert all(r["success"] for r in results)
            assert mock_scrape.call_count == 3
