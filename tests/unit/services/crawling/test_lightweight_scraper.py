import respx


"""Tests for the lightweight HTTP scraper."""

from unittest.mock import AsyncMock, MagicMock, patch

import httpx
import pytest
from bs4 import BeautifulSoup

# LightweightScraperConfig not in simplified config, use Config instead
from src.config import Config
from src.services.crawling.lightweight_scraper import (
    LightweightScraper,
    TierRecommendation,
)
from src.services.errors import CrawlServiceError


@pytest.fixture
def config():
    """Create test configuration."""
    return Config(
        enable_lightweight_tier=True,
        use_head_analysis=True,
        content_threshold=100,
        max_lightweight_size=1_000_000,
        timeout=5.0,
        head_timeout=2.0,
        simple_url_patterns=[
            r".*\.md$",
            r".*/raw/.*",
            r".*\.(txt|json|xml)$",
        ],
        known_simple_sites={
            "docs.python.org": {"selector": ".document"},
            "test.example.com": {"selector": ".content"},
        },
    )


@pytest.fixture
def scraper(config):
    """Create test scraper instance."""
    return LightweightScraper(config)


class TestLightweightScraper:
    """Test cases for LightweightScraper."""

    @pytest.mark.asyncio
    async def test_initialize(self, scraper):
        """Test scraper initialization."""
        assert not scraper._initialized
        assert scraper._http_client is None

        await scraper.initialize()

        assert scraper._initialized
        assert scraper._http_client is not None
        assert isinstance(scraper._http_client, httpx.AsyncClient)

        # Test idempotency
        await scraper.initialize()
        assert scraper._initialized

    @pytest.mark.asyncio
    async def test_cleanup(self, scraper):
        """Test scraper cleanup."""
        await scraper.initialize()
        assert scraper._initialized

        await scraper.cleanup()

        assert not scraper._initialized
        assert scraper._http_client is None

    @pytest.mark.asyncio
    async def test_can_handle_disabled(self, scraper):
        """Test can_handle when lightweight tier is disabled."""
        scraper.config.enable_lightweight_tier = False
        assert not await scraper.can_handle("https://example.com/test.md")

    @pytest.mark.asyncio
    async def test_can_handle_url_patterns(self, scraper):
        """Test can_handle with URL pattern matching."""
        # Should match .md files
        assert await scraper.can_handle("https://example.com/readme.md")

        # Should match raw content
        assert await scraper.can_handle(
            "https://github.com/user/repo/raw/main/file.txt"
        )

        # Should match JSON files
        assert await scraper.can_handle("https://api.example.com/data.json")

        # Should not match regular HTML
        assert not await scraper.can_handle("https://example.com/index.html")

    @pytest.mark.asyncio
    async def test_can_handle_known_sites(self, scraper):
        """Test can_handle with known simple sites."""
        assert await scraper.can_handle("https://docs.python.org/3/tutorial/index.html")
        assert await scraper.can_handle("https://test.example.com/page")
        assert not await scraper.can_handle("https://unknown.com/page")

    @pytest.mark.asyncio
    async def test_can_handle_with_head_analysis(self, scraper):
        """Test can_handle with HEAD request analysis."""
        scraper.config.use_head_analysis = True

        with patch.object(scraper, "_analyze_url") as mock_analyze:
            mock_analyze.return_value = TierRecommendation.LIGHTWEIGHT_OK
            assert await scraper.can_handle("https://example.com/page")

            mock_analyze.return_value = TierRecommendation.BROWSER_REQUIRED
            assert not await scraper.can_handle("https://example.com/spa")

    @pytest.mark.asyncio
    async def test_analyze_url_content_type(self, scraper):
        """Test URL analysis based on content type."""
        await scraper.initialize()

        # Mock HEAD response for non-HTML content
        with patch.object(scraper._http_client, "head") as mock_head:
            mock_response = MagicMock()
            mock_response.headers = {"content-type": "application/json"}
            mock_head.return_value = mock_response

            result = await scraper._analyze_url("https://api.example.com/data")
            assert result == TierRecommendation.LIGHTWEIGHT_OK

    @pytest.mark.asyncio
    async def test_analyze_url_spa_detection(self, scraper):
        """Test SPA detection in URL analysis."""
        await scraper.initialize()

        # Mock HEAD response with SPA indicators
        with patch.object(scraper._http_client, "head") as mock_head:
            mock_response = MagicMock()
            mock_response.headers = {
                "content-type": "text/html",
                "x-powered-by": "React",
            }
            mock_head.return_value = mock_response

            result = await scraper._analyze_url("https://spa.example.com")
            assert result == TierRecommendation.BROWSER_REQUIRED

    @pytest.mark.asyncio
    async def test_analyze_url_size_limit(self, scraper):
        """Test URL analysis with content size limit."""
        await scraper.initialize()

        # Mock HEAD response with large content
        with patch.object(scraper._http_client, "head") as mock_head:
            mock_response = MagicMock()
            mock_response.headers = {
                "content-type": "text/html",
                "content-length": "2000000",  # 2MB
            }
            mock_head.return_value = mock_response

            result = await scraper._analyze_url("https://large.example.com")
            assert result == TierRecommendation.STREAMING_REQUIRED

    @pytest.mark.asyncio
    async def test_analyze_url_csp_javascript(self, scraper):
        """Test JavaScript detection via CSP headers."""
        await scraper.initialize()

        # Mock HEAD response with CSP indicating heavy JS
        with patch.object(scraper._http_client, "head") as mock_head:
            mock_response = MagicMock()
            mock_response.headers = {
                "content-type": "text/html",
                "content-security-policy": "script-src 'unsafe-inline' 'self'",
            }
            mock_head.return_value = mock_response

            result = await scraper._analyze_url("https://js-heavy.example.com")
            assert result == TierRecommendation.BROWSER_REQUIRED

    @pytest.mark.asyncio
    async def test_scrape_url_not_initialized(self, scraper):
        """Test scraping without initialization."""
        with pytest.raises(CrawlServiceError, match="Scraper not initialized"):
            await scraper.scrape_url("https://example.com")

    @pytest.mark.asyncio
    async def test_scrape_url_success(self, scraper):
        """Test successful URL scraping."""
        await scraper.initialize()

        html_content = """
        <html>
            <head><title>Test Page</title></head>
            <body>
                <div class="content">
                    <h1>Main Title</h1>
                    <p>This is a test paragraph with enough content to pass the threshold.</p>
                    <p>Another paragraph to ensure we have sufficient text content for extraction.</p>
                </div>
            </body>
        </html>
        """

        with patch.object(scraper._http_client, "get") as mock_get:
            mock_response = MagicMock()
            mock_response.text = html_content
            mock_response.raise_for_status = MagicMock()
            mock_get.return_value = mock_response

            result = await scraper.scrape_url("https://test.example.com/page")

            assert result["success"] is True
            assert result["url"] == "https://test.example.com/page"
            assert "markdown" in result["content"]
            assert "Main Title" in result["content"]["markdown"]
            assert result["metadata"]["title"] == "Test Page"
            assert result["metadata"]["tier"] == "lightweight"

    @pytest.mark.asyncio
    async def test_scrape_url_insufficient_content(self, scraper):
        """Test scraping with insufficient content."""
        await scraper.initialize()

        html_content = """
        <html>
            <body>
                <div class="content">
                    <p>Too short.</p>
                </div>
            </body>
        </html>
        """

        with patch.object(scraper._http_client, "get") as mock_get:
            mock_response = MagicMock()
            mock_response.text = html_content
            mock_response.raise_for_status = MagicMock()
            mock_get.return_value = mock_response

            result = await scraper.scrape_url("https://example.com/short")

            assert result["success"] is False
            assert result["should_escalate"] is True
            assert "Insufficient content" in result["error"]

    @pytest.mark.asyncio
    async def test_scrape_url_http_error(self, scraper):
        """Test scraping with HTTP error."""
        await scraper.initialize()

        with patch.object(scraper._http_client, "get") as mock_get:
            mock_response = MagicMock()
            mock_response.status_code = 404
            mock_response.raise_for_status.side_effect = httpx.HTTPStatusError(
                "Not Found", request=MagicMock(), response=mock_response
            )
            mock_get.return_value = mock_response

            result = await scraper.scrape_url("https://example.com/404")

            assert result["success"] is False
            assert result["should_escalate"] is False  # 404 should not escalate
            assert "404" in result["error"]

    @pytest.mark.asyncio
    async def test_scrape_url_with_rate_limiter(self, scraper):
        """Test scraping with rate limiter."""
        await scraper.initialize()

        rate_limiter = AsyncMock()
        scraper.rate_limiter = rate_limiter

        html_content = (
            "<html><body><div class='content'>" + "x" * 200 + "</div></body></html>"
        )

        with patch.object(scraper._http_client, "get") as mock_get:
            mock_response = MagicMock()
            mock_response.text = html_content
            mock_response.raise_for_status = MagicMock()
            mock_get.return_value = mock_response

            await scraper.scrape_url("https://example.com/rate-limited")

            rate_limiter.acquire.assert_called_once_with(
                "https://example.com/rate-limited"
            )

    @pytest.mark.asyncio
    async def test_extract_content_known_selector(self, scraper):
        """Test content extraction with known selector."""
        html = """
        <html>
            <body>
                <div class="sidebar">Navigation</div>
                <div class="content">
                    <h1>Main Content</h1>
                    <p>This is the main content area.</p>
                </div>
            </body>
        </html>
        """

        soup = BeautifulSoup(html, "lxml")
        result = await scraper._extract_content(soup, "https://test.example.com/page")

        assert result is not None
        assert "Main Content" in result["text"]
        assert "Navigation" not in result["text"]

    @pytest.mark.asyncio
    async def test_extract_content_common_selectors(self, scraper):
        """Test content extraction with common selectors."""
        html = """
        <html>
            <body>
                <nav>Navigation</nav>
                <main>
                    <h1>Article Title</h1>
                    <p>Article content goes here.</p>
                </main>
            </body>
        </html>
        """

        soup = BeautifulSoup(html, "lxml")
        result = await scraper._extract_content(soup, "https://unknown.com/page")

        assert result is not None
        assert "Article Title" in result["text"]
        assert "Navigation" not in result["text"]

    @pytest.mark.asyncio
    async def test_extract_content_metadata(self, scraper):
        """Test metadata extraction."""
        html = """
        <html lang="fr">
            <head>
                <title>Page Title</title>
                <meta name="description" content="Page description">
            </head>
            <body>
                <article>
                    <h1>Content Title</h1>
                    <p>Content text.</p>
                </article>
            </body>
        </html>
        """

        soup = BeautifulSoup(html, "lxml")
        result = await scraper._extract_content(soup, "https://example.com")

        assert result["title"] == "Page Title"
        assert result["description"] == "Page description"
        assert result["language"] == "fr"

    def test_find_main_content(self, scraper):
        """Test main content detection algorithm."""
        html = """
        <html>
            <body>
                <div class="header">
                    <a href="/">Home</a>
                    <a href="/about">About</a>
                </div>
                <div class="sidebar">
                    <a href="/link1">Link 1</a>
                    <a href="/link2">Link 2</a>
                </div>
                <article class="main-content">
                    <h1>Main Article</h1>
                    <p>This is a long paragraph with lots of text content that should be identified as the main content area of the page.</p>
                    <p>Another paragraph to add more content.</p>
                </article>
            </body>
        </html>
        """

        soup = BeautifulSoup(html, "lxml")
        content = scraper._find_main_content(soup)

        assert content is not None
        assert content.name == "article"
        assert "Main Article" in content.get_text()

    def test_convert_to_markdown(self, scraper):
        """Test HTML to Markdown conversion."""
        html = """
        <div>
            <h1>Title</h1>
            <h2>Subtitle</h2>
            <p>Paragraph text.</p>
            <ul>
                <li>Item 1</li>
                <li>Item 2</li>
            </ul>
            <ol>
                <li>First</li>
                <li>Second</li>
            </ol>
            <p>Text with <strong>bold</strong> and <em>italic</em>.</p>
            <p>A <a href="https://example.com">link</a>.</p>
            <pre><code>Code block</code></pre>
            <blockquote>Quote text</blockquote>
        </div>
        """

        soup = BeautifulSoup(html, "lxml")
        content_element = soup.find("div")

        markdown = scraper._convert_to_markdown(content_element, "Page Title")

        assert "# Page Title" in markdown
        assert "# Title" in markdown
        assert "## Subtitle" in markdown
        assert "- Item 1" in markdown
        assert "1. First" in markdown
        assert "**bold**" in markdown
        assert "*italic*" in markdown
        assert "[link](https://example.com)" in markdown
        assert "```\nCode block\n```" in markdown
        assert "> Quote text" in markdown

    @pytest.mark.asyncio
    async def test_crawl_site_not_supported(self, scraper):
        """Test that site crawling is not supported."""
        result = await scraper.crawl_site("https://example.com")

        assert result["success"] is False
        assert "does not support site crawling" in result["error"]
        assert result["should_escalate"] is True

    @pytest.mark.asyncio
    async def test_scrape_url_multiple_formats(self, scraper):
        """Test scraping with multiple output formats."""
        await scraper.initialize()

        html_content = """
        <html>
            <head><title>Test Page</title></head>
            <body>
                <div class="content">
                    <h1>Test</h1>
                    <p>Content with enough text to pass the threshold check for extraction. This paragraph contains sufficient content to ensure we meet the minimum character requirement for successful extraction.</p>
                    <p>Additional content to ensure we have enough text for the content threshold validation.</p>
                </div>
            </body>
        </html>
        """

        with patch.object(scraper._http_client, "get") as mock_get:
            mock_response = MagicMock()
            mock_response.text = html_content
            mock_response.raise_for_status = MagicMock()
            mock_get.return_value = mock_response

            result = await scraper.scrape_url(
                "https://example.com", formats=["markdown", "html", "text"]
            )

            assert result["success"] is True
            assert "markdown" in result["content"]
            assert "html" in result["content"]
            assert "text" in result["content"]
            assert "# Test" in result["content"]["markdown"]
            assert "<h1>Test</h1>" in result["content"]["html"]
            assert "Test" in result["content"]["text"]

    @pytest.mark.asyncio
    async def test_scrape_url_exception_handling(self, scraper):
        """Test exception handling during scraping."""
        await scraper.initialize()

        with patch.object(scraper._http_client, "get") as mock_get:
            mock_get.side_effect = Exception("Network error")

            result = await scraper.scrape_url("https://example.com")

            assert result["success"] is False
            assert "Network error" in result["error"]
            assert result["should_escalate"] is True
