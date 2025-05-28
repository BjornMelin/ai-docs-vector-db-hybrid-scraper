"""Enhanced Crawl4AI provider with advanced features for high-performance web crawling."""

import asyncio
import hashlib
import logging
import time
from typing import Any
from urllib.parse import urlparse

import numpy as np
from crawl4ai import AsyncWebCrawler
from crawl4ai import BrowserConfig
from crawl4ai import CrawlerRunConfig
from crawl4ai.extraction_strategy import JsonCssExtractionStrategy
from crawl4ai.extraction_strategy import LLMExtractionStrategy

from ..base import BaseService
from ..errors import CrawlServiceError
from ..utilities.rate_limiter import RateLimiter
from .base import CrawlProvider

logger = logging.getLogger(__name__)


class JavaScriptExecutor:
    """Handle complex JavaScript execution for dynamic content."""

    def __init__(self):
        self.common_patterns = {
            "spa_navigation": """
                // Wait for SPA navigation
                await new Promise(resolve => {
                    const observer = new MutationObserver(() => {
                        if (document.querySelector('.content-loaded')) {
                            observer.disconnect();
                            resolve();
                        }
                    });
                    observer.observe(document.body, {childList: true, subtree: true});
                    setTimeout(resolve, 5000); // Timeout fallback
                });
            """,
            "infinite_scroll": """
                // Load all content via infinite scroll
                let lastHeight = 0;
                while (true) {
                    window.scrollTo(0, document.body.scrollHeight);
                    await new Promise(r => setTimeout(r, 1000));
                    let newHeight = document.body.scrollHeight;
                    if (newHeight === lastHeight) break;
                    lastHeight = newHeight;
                }
            """,
            "click_show_more": """
                // Click all "show more" buttons
                const buttons = document.querySelectorAll('[class*="show-more"], [class*="load-more"]');
                for (const button of buttons) {
                    button.click();
                    await new Promise(r => setTimeout(r, 500));
                }
            """,
        }

    def get_js_for_site(self, url: str) -> str | None:
        """Get custom JavaScript for specific documentation sites."""
        domain = urlparse(url).netloc

        # Site-specific JavaScript
        site_js = {
            "docs.python.org": self.common_patterns["spa_navigation"],
            "reactjs.org": self.common_patterns["spa_navigation"],
            "react.dev": self.common_patterns["spa_navigation"],
            "developer.mozilla.org": self.common_patterns["click_show_more"],
            "stackoverflow.com": self.common_patterns["infinite_scroll"],
        }

        return site_js.get(domain)


class DocumentationExtractor:
    """Optimized extraction for technical documentation."""

    def __init__(self):
        self.selectors = {
            # Common documentation selectors
            "content": [
                "main",
                "article",
                ".content",
                ".documentation",
                "#main-content",
                ".markdown-body",
                ".doc-content",
            ],
            # Code blocks
            "code": [
                "pre code",
                ".highlight",
                ".code-block",
                ".language-*",
            ],
            # Navigation (to extract structure)
            "nav": [
                ".sidebar",
                ".toc",
                "nav",
                ".navigation",
            ],
            # Metadata
            "metadata": {
                "title": ["h1", ".title", "title"],
                "description": ["meta[name='description']", ".description"],
                "author": [".author", "meta[name='author']"],
                "version": [".version", ".release"],
                "last_updated": ["time", ".last-updated", ".modified"],
            },
        }

    def create_extraction_schema(self, doc_type: str = "general") -> dict:
        """Create extraction schema based on documentation type."""
        schemas = {
            "api_reference": {
                "endpoints": "section.endpoint",
                "parameters": ".parameter",
                "responses": ".response",
                "examples": ".example",
            },
            "tutorial": {
                "steps": ".step, .tutorial-step",
                "code_examples": "pre code",
                "prerequisites": ".prerequisites",
                "objectives": ".objectives",
            },
            "guide": {
                "sections": "h2, h3",
                "content": "p, ul, ol",
                "callouts": ".note, .warning, .tip",
                "related": ".related-links",
            },
        }

        base_schema = {
            "title": self.selectors["metadata"]["title"],
            "content": self.selectors["content"],
            "code_blocks": self.selectors["code"],
        }

        return {**base_schema, **schemas.get(doc_type, {})}


class Crawl4AIProvider(BaseService, CrawlProvider):
    """High-performance web crawling with Crawl4AI."""

    def __init__(self, config: dict[str, Any] | None = None, rate_limiter: Any = None):
        """Initialize Crawl4AI provider with advanced configuration."""
        super().__init__(config or {})
        self.logger = logger
        self.rate_limiter = rate_limiter or RateLimiter(
            max_calls=self.config.get("rate_limit", 60), time_window=60
        )

        # Browser configuration
        self.browser_config = BrowserConfig(
            browser_type=self.config.get("browser", "chromium"),
            headless=self.config.get("headless", True),
            viewport_width=self.config.get("viewport_width", 1920),
            viewport_height=self.config.get("viewport_height", 1080),
            user_agent=self.config.get(
                "user_agent",
                "Mozilla/5.0 (compatible; AIDocs/1.0; +https://github.com/ai-docs)",
            ),
        )

        # Concurrent crawling settings
        self.max_concurrent = self.config.get("max_concurrent", 10)
        self.semaphore = asyncio.Semaphore(self.max_concurrent)

        # Initialize helpers
        self.js_executor = JavaScriptExecutor()
        self.doc_extractor = DocumentationExtractor()

        self._crawler: AsyncWebCrawler | None = None
        self._initialized = False

    async def initialize(self) -> None:
        """Initialize Crawl4AI crawler."""
        if self._initialized:
            return

        try:
            self._crawler = AsyncWebCrawler(config=self.browser_config)
            await self._crawler.start()
            self._initialized = True
            self.logger.info("Crawl4AI crawler initialized with advanced configuration")
        except Exception as e:
            raise CrawlServiceError(f"Failed to initialize Crawl4AI: {e}") from e

    async def cleanup(self) -> None:
        """Cleanup Crawl4AI resources."""
        if self._crawler:
            await self._crawler.close()
            self._crawler = None
            self._initialized = False
            self.logger.info("Crawl4AI resources cleaned up")

    async def scrape_url(
        self,
        url: str,
        formats: list[str] | None = None,
        extraction_type: str = "markdown",
        wait_for: str | None = None,
        js_code: str | None = None,
    ) -> dict[str, Any]:
        """Scrape single URL with advanced extraction options.

        Args:
            url: URL to scrape
            formats: Output formats (ignored, always returns markdown + html)
            extraction_type: Type of extraction ("markdown", "structured", "llm")
            wait_for: CSS selector to wait for before extraction
            js_code: Custom JavaScript to execute

        Returns:
            Scrape result with content, metadata, and extraction details
        """
        if not self._initialized:
            raise CrawlServiceError("Provider not initialized")

        async with self.semaphore:
            if self.rate_limiter:
                await self.rate_limiter.acquire()

            try:
                # Get site-specific JavaScript if not provided
                if not js_code:
                    js_code = self.js_executor.get_js_for_site(url)

                # Configure extraction strategy
                extraction_strategy = None
                if extraction_type == "structured":
                    extraction_strategy = JsonCssExtractionStrategy(
                        schema=self.doc_extractor.create_extraction_schema()
                    )
                elif extraction_type == "llm":
                    extraction_strategy = LLMExtractionStrategy(
                        provider="ollama/llama2",
                        api_token=self.config.get("llm_api_token"),
                        instruction="Extract technical documentation with code examples",
                    )

                # Configure crawler run
                run_config = CrawlerRunConfig(
                    word_count_threshold=10,
                    css_selector=", ".join(self.doc_extractor.selectors["content"]),
                    excluded_tags=[
                        "nav",
                        "footer",
                        "header",
                        "aside",
                        "script",
                        "style",
                    ],
                    wait_for=wait_for,
                    js_code=js_code,
                    extraction_strategy=extraction_strategy,
                    cache_mode="enabled",
                    page_timeout=self.config.get("page_timeout", 30000),
                    wait_until="networkidle",
                )

                # Crawl the URL
                result = await self._crawler.arun(url=url, config=run_config)

                if result.success:
                    # Extract structured data if available
                    structured_data = {}
                    if result.extracted_content:
                        structured_data = result.extracted_content

                    return {
                        "success": True,
                        "url": url,
                        "title": result.metadata.get("title", ""),
                        "content": result.markdown or "",
                        "html": result.html or "",
                        "metadata": {
                            **result.metadata,
                            "extraction_type": extraction_type,
                            "word_count": len((result.markdown or "").split()),
                            "has_structured_data": bool(structured_data),
                        },
                        "structured_data": structured_data,
                        "links": result.links or [],
                        "media": result.media or {},
                        "provider": "crawl4ai",
                    }
                else:
                    return {
                        "success": False,
                        "error": getattr(result, "error_message", "Crawl failed"),
                        "content": "",
                        "metadata": {},
                        "url": url,
                        "provider": "crawl4ai",
                    }

            except Exception as e:
                # Get additional context for better error reporting
                rate_limit_status = "unknown"
                if hasattr(self.rate_limiter, "current_calls"):
                    rate_limit_status = f"{self.rate_limiter.current_calls}/{self.rate_limiter.max_calls}"

                error_context = {
                    "url": url,
                    "extraction_type": extraction_type,
                    "rate_limit_status": rate_limit_status,
                    "semaphore_available": self.semaphore._value
                    if hasattr(self.semaphore, "_value")
                    else "unknown",
                }

                self.logger.error(
                    f"Failed to scrape {url}: {e} | Context: {error_context}"
                )
                return {
                    "success": False,
                    "error": str(e),
                    "error_context": error_context,
                    "content": "",
                    "metadata": {},
                    "url": url,
                    "provider": "crawl4ai",
                }

    async def crawl_bulk(
        self, urls: list[str], extraction_type: str = "markdown"
    ) -> list[dict[str, Any]]:
        """Crawl multiple URLs concurrently.

        Args:
            urls: List of URLs to crawl
            extraction_type: Type of extraction to use

        Returns:
            List of crawl results
        """
        tasks = [self.scrape_url(url, extraction_type=extraction_type) for url in urls]

        results = await asyncio.gather(*tasks, return_exceptions=True)

        # Process results
        successful = []
        failed = []

        for url, result in zip(urls, results, strict=False):
            if isinstance(result, Exception):
                failed.append({"url": url, "error": str(result)})
                self.logger.error(f"Failed to crawl {url}: {result}")
            else:
                successful.append(result)

        if failed:
            self.logger.warning(
                f"Failed to crawl {len(failed)} URLs out of {len(urls)}"
            )

        return successful

    async def crawl_site(
        self,
        url: str,
        max_pages: int = 50,
        formats: list[str] | None = None,
    ) -> dict[str, Any]:
        """Crawl entire site using recursive URL discovery.

        Args:
            url: Starting URL
            max_pages: Maximum pages to crawl
            formats: Output formats (ignored)

        Returns:
            Crawl result with all pages
        """
        if not self._initialized:
            raise CrawlServiceError("Provider not initialized")

        pages = []
        visited_urls = set()
        to_visit = [url]
        base_domain = urlparse(url).netloc

        # Memory optimization: limit visited_urls size for very large crawls
        max_visited_urls = max(max_pages * 3, 1000)  # 3x safety margin, min 1000

        try:
            while to_visit and len(pages) < max_pages:
                # Crawl batch of URLs
                batch_size = min(10, max_pages - len(pages))
                batch_urls = []

                while to_visit and len(batch_urls) < batch_size:
                    next_url = to_visit.pop(0)
                    if next_url not in visited_urls:
                        batch_urls.append(next_url)
                        visited_urls.add(next_url)

                        # Memory optimization: trim visited_urls if it gets too large
                        if len(visited_urls) > max_visited_urls:
                            # Keep only the most recent 80% of URLs (simple LRU approximation)
                            keep_count = int(max_visited_urls * 0.8)
                            visited_urls = set(list(visited_urls)[-keep_count:])
                            self.logger.debug(
                                f"Trimmed visited_urls from {max_visited_urls} to {keep_count} for memory optimization"
                            )

                if not batch_urls:
                    break

                # Crawl batch concurrently
                batch_results = await self.crawl_bulk(batch_urls)

                for result in batch_results:
                    if result["success"]:
                        pages.append(
                            {
                                "url": result["url"],
                                "content": result["content"],
                                "html": result["html"],
                                "metadata": result["metadata"],
                                "title": result.get("title", ""),
                            }
                        )

                        # Extract and filter links
                        for link in result.get("links", []):
                            link_url = link.get("href", "")
                            if link_url and link_url.startswith("http"):
                                link_domain = urlparse(link_url).netloc
                                if (
                                    link_domain == base_domain
                                    and link_url not in visited_urls
                                ):
                                    to_visit.append(link_url)

                self.logger.info(f"Crawled {len(pages)}/{max_pages} pages from {url}")

            return {
                "success": True,
                "pages": pages,
                "total": len(pages),
                "provider": "crawl4ai",
            }

        except Exception as e:
            # Enhanced error context for site crawling
            error_context = {
                "starting_url": url,
                "pages_crawled": len(pages),
                "urls_visited": len(visited_urls),
                "urls_remaining": len(to_visit),
                "max_pages_target": max_pages,
            }

            self.logger.error(
                f"Failed to crawl site {url}: {e} | Context: {error_context}"
            )
            return {
                "success": False,
                "error": str(e),
                "error_context": error_context,
                "pages": pages,
                "total": len(pages),
                "provider": "crawl4ai",
            }


class CrawlCache:
    """Intelligent caching for crawled content."""

    def __init__(self, cache_manager: Any):
        self.cache = cache_manager
        self.ttl = 86400  # 24 hours default

    async def get_or_crawl(
        self,
        url: str,
        crawler: Crawl4AIProvider,
        force_refresh: bool = False,
    ) -> dict[str, Any]:
        """Get from cache or crawl if needed."""
        # Generate cache key
        cache_key = f"crawl:{hashlib.md5(url.encode()).hexdigest()}"

        # Check cache unless forced refresh
        if not force_refresh:
            cached = await self.cache.get(cache_key)
            if cached:
                return cached

        # Crawl and cache
        result = await crawler.scrape_url(url)

        # Determine TTL based on content
        ttl = self.calculate_ttl(result)

        await self.cache.set(cache_key, result, ttl=ttl)

        return result

    def calculate_ttl(self, result: dict[str, Any]) -> int:
        """Dynamic TTL based on content characteristics.

        Args:
            result: Crawl result dictionary containing URL and content

        Returns:
            TTL in seconds based on content type:
            - API docs: 7 days (604800s)
            - Blog posts: 30 days (2592000s)
            - Other content: 3 days (259200s)
        """
        url = result.get("url", "").lower()

        # API docs change less frequently
        if "api" in url:
            return 604800  # 7 days

        # Blog posts are static
        if "blog" in url:
            return 2592000  # 30 days

        # Tutorials moderate change frequency
        return 259200  # 3 days


class CrawlBenchmark:
    """Benchmark Crawl4AI performance."""

    def __init__(self, crawl4ai: Crawl4AIProvider, firecrawl: Any = None):
        self.crawl4ai = crawl4ai
        self.firecrawl = firecrawl

    async def run_comparison(self, urls: list[str]) -> dict[str, dict[str, Any]]:
        """Compare performance of both crawlers.

        Args:
            urls: List of URLs to test against both crawlers

        Returns:
            Dictionary with performance statistics for each crawler including:
            - success/failed counts
            - timing statistics (avg, p95, min, max)
        """
        results = {
            "crawl4ai": {"times": [], "success": 0, "failed": 0},
            "firecrawl": {"times": [], "success": 0, "failed": 0},
        }

        # Test Crawl4AI
        for url in urls:
            start = time.time()
            try:
                result = await self.crawl4ai.scrape_url(url)
                if result["success"]:
                    results["crawl4ai"]["success"] += 1
                else:
                    results["crawl4ai"]["failed"] += 1
                results["crawl4ai"]["times"].append(time.time() - start)
            except Exception:
                results["crawl4ai"]["failed"] += 1

        # Test Firecrawl if available
        if self.firecrawl:
            for url in urls:
                start = time.time()
                try:
                    result = await self.firecrawl.scrape_url(url)
                    if result.get("success"):
                        results["firecrawl"]["success"] += 1
                    else:
                        results["firecrawl"]["failed"] += 1
                    results["firecrawl"]["times"].append(time.time() - start)
                except Exception:
                    results["firecrawl"]["failed"] += 1

        # Calculate statistics
        for _crawler, data in results.items():
            times = data["times"]
            if times:
                data["avg_time"] = float(np.mean(times))
                data["p95_time"] = float(np.percentile(times, 95))
                data["min_time"] = float(np.min(times))
                data["max_time"] = float(np.max(times))

        return results
