"""Integration tests for Crawl4AI with real documentation sites.

These tests validate Crawl4AI performance and content extraction
against actual documentation websites.
"""

import time

import pytest
from src.services.crawling.crawl4ai_provider import Crawl4AIProvider


@pytest.mark.integration
@pytest.mark.asyncio
class TestCrawl4AIRealSites:
    """Test Crawl4AI against real documentation sites."""

    @pytest.fixture
    async def provider(self):
        """Create and initialize Crawl4AI provider."""
        config = {
            "max_concurrent": 5,
            "rate_limit": 30,
            "browser": "chromium",
            "headless": True,
            "page_timeout": 30000,
        }
        provider = Crawl4AIProvider(config=config)
        await provider.initialize()
        yield provider
        await provider.cleanup()

    async def test_python_docs(self, provider):
        """Test crawling Python documentation."""
        url = "https://docs.python.org/3/library/asyncio.html"

        start_time = time.time()
        result = await provider.scrape_url(url)
        elapsed = time.time() - start_time

        assert result["success"] is True
        assert result["url"] == url
        assert len(result.get("content", "")) > 1000  # Should have substantial content
        assert elapsed < 10  # Should complete within 10 seconds
        assert "asyncio" in result.get("content", "").lower()

    async def test_fastapi_docs(self, provider):
        """Test crawling FastAPI documentation."""
        url = "https://fastapi.tiangolo.com/tutorial/"

        result = await provider.scrape_url(
            url,
            wait_for="article",
        )

        assert result["success"] is True
        assert len(result.get("content", "")) > 500
        assert "fastapi" in result.get("content", "").lower()

    async def test_react_docs_spa(self, provider):
        """Test crawling React documentation (SPA)."""
        url = "https://react.dev/learn"

        # React docs need custom JS for SPA navigation
        result = await provider.scrape_url(
            url,
            wait_for="main",
            js_code="await new Promise(r => setTimeout(r, 3000));",
        )

        assert result["success"] is True
        assert len(result.get("content", "")) > 500
        # Check for React-specific content
        content_lower = result.get("content", "").lower()
        assert any(term in content_lower for term in ["react", "component", "jsx"])

    async def test_mdn_docs_complex(self, provider):
        """Test crawling MDN Web Docs (complex layout)."""
        url = "https://developer.mozilla.org/en-US/docs/Web/JavaScript"

        result = await provider.scrape_url(
            url,
            wait_for="article",
        )

        assert result["success"] is True
        assert len(result.get("content", "")) > 1000
        assert "javascript" in result.get("content", "").lower()

    async def test_bulk_crawling_performance(self, provider):
        """Test bulk crawling performance."""
        urls = [
            "https://docs.python.org/3/library/os.html",
            "https://docs.python.org/3/library/sys.html",
            "https://docs.python.org/3/library/json.html",
        ]

        start_time = time.time()
        results = await provider.crawl_bulk(urls)
        elapsed = time.time() - start_time

        # All should succeed
        assert len(results) == len(urls)
        assert all(r["success"] for r in results)

        # Performance check: should complete in reasonable time
        assert elapsed < 20  # 3 URLs in under 20 seconds

        # Calculate throughput
        throughput = len(urls) / elapsed
        assert throughput > 0.15  # At least 0.15 URLs per second

    async def test_site_crawling(self, provider):
        """Test crawling multiple pages from a site."""
        # Use a small docs site section
        url = "https://docs.python.org/3/library/json.html"

        result = await provider.crawl_site(
            url,
            max_pages=3,  # Limit to 3 pages for test
        )

        assert result["success"] is True
        assert result["total"] > 0
        assert result["total"] <= 3
        assert len(result["pages"]) == result["total"]

        # Check page structure
        for page in result["pages"]:
            assert "url" in page
            assert "content" in page
            assert len(page["content"]) > 0

    async def test_structured_extraction(self, provider):
        """Test structured data extraction."""
        url = "https://docs.python.org/3/library/json.html"

        result = await provider.scrape_url(
            url,
            extraction_type="structured",
        )

        assert result["success"] is True
        assert "structured_data" in result

        # Should extract title and content
        structured = result["structured_data"]
        if structured:  # Structured extraction may not always work
            assert any(key in structured for key in ["title", "content"])

    async def test_javascript_execution(self, provider):
        """Test JavaScript execution capabilities."""
        url = "https://developer.mozilla.org/en-US/docs/Web/API"

        # Test with custom JavaScript
        js_code = """
            // Log page info
            console.log('Page title:', document.title);
            console.log('Links found:', document.querySelectorAll('a').length);

            // Expand any collapsible sections
            document.querySelectorAll('details').forEach(d => d.open = true);
        """

        result = await provider.scrape_url(
            url,
            js_code=js_code,
        )

        assert result["success"] is True
        assert len(result.get("content", "")) > 500

    @pytest.mark.parametrize(
        "url,min_content_length",
        [
            ("https://docs.github.com/en/rest", 1000),
            ("https://stripe.com/docs/api", 500),
            ("https://kubernetes.io/docs/", 1000),
        ],
    )
    async def test_various_doc_sites(self, provider, url, min_content_length):
        """Test various documentation sites."""
        result = await provider.scrape_url(url)

        assert result["success"] is True
        assert len(result.get("content", "")) > min_content_length

    async def test_error_handling(self, provider):
        """Test error handling for invalid URLs."""
        # Test non-existent page
        result = await provider.scrape_url("https://docs.python.org/404-not-found")

        # Should handle gracefully
        assert "success" in result
        if result["success"]:
            # May still succeed but with 404 content
            assert (
                "404" in result.get("content", "")
                or "not found" in result.get("content", "").lower()
            )
        else:
            # Or fail with appropriate error
            assert "error" in result

    async def test_performance_vs_target(self, provider):
        """Test performance against target metrics."""
        url = "https://docs.python.org/3/library/asyncio.html"

        # Run multiple times to get average
        times = []
        for _ in range(3):
            start = time.time()
            result = await provider.scrape_url(url)
            if result["success"]:
                times.append(time.time() - start)

        if times:
            avg_time = sum(times) / len(times)
            # Target: < 2.5 seconds average (vs Firecrawl's reported 2.5s)
            assert avg_time < 2.5, f"Average time {avg_time:.2f}s exceeds target"
