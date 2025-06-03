"""Enhanced Crawl4AI provider with advanced features for high-performance web crawling."""

import asyncio
import logging
from urllib.parse import urlparse

from crawl4ai import AsyncWebCrawler
from crawl4ai import BrowserConfig
from crawl4ai import CrawlerRunConfig
from crawl4ai.async_configs import LLMConfig
from crawl4ai.extraction_strategy import JsonCssExtractionStrategy
from crawl4ai.extraction_strategy import LLMExtractionStrategy

from ...config.models import Crawl4AIConfig
from ..base import BaseService
from ..errors import CrawlServiceError
from ..utilities.rate_limiter import RateLimiter
from .base import CrawlProvider
from .extractors import DocumentationExtractor
from .extractors import JavaScriptExecutor

logger = logging.getLogger(__name__)


class Crawl4AIProvider(BaseService, CrawlProvider):
    """High-performance web crawling with Crawl4AI."""

    def __init__(self, config: Crawl4AIConfig, rate_limiter: object = None):
        """Initialize Crawl4AI provider with advanced configuration."""
        super().__init__(config)
        self.config = config
        self.logger = logger
        self.rate_limiter = rate_limiter or RateLimiter(
            max_calls=50,
            time_window=60,  # Default rate limit for Crawl4AI
        )

        # Browser configuration from Pydantic model
        self.browser_config = BrowserConfig(
            browser_type=self.config.browser_type,
            headless=self.config.headless,
            viewport_width=self.config.viewport["width"],
            viewport_height=self.config.viewport["height"],
            user_agent="Mozilla/5.0 (compatible; AIDocs/1.0; +https://github.com/ai-docs)",
        )

        # Concurrent crawling settings
        self.max_concurrent = self.config.max_concurrent_crawls
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
            try:
                await self._crawler.close()
            except Exception as e:
                self.logger.error(f"Error closing crawler: {e}")
            finally:
                # Always reset state even if close() fails
                self._crawler = None
                self._initialized = False
                self.logger.info("Crawl4AI resources cleaned up")

    def _create_extraction_strategy(self, extraction_type: str) -> object | None:
        """Create extraction strategy based on type.

        Args:
            extraction_type: Type of extraction ("structured", "llm", or "markdown")

        Returns:
            Extraction strategy instance or None for markdown extraction
        """
        if extraction_type == "structured":
            return JsonCssExtractionStrategy(
                schema=self.doc_extractor.create_extraction_schema()
            )
        elif extraction_type == "llm":
            llm_config = LLMConfig(provider="ollama/llama2")
            return LLMExtractionStrategy(
                llm_config=llm_config,
                instruction="Extract technical documentation with code examples",
            )
        return None

    def _create_run_config(
        self,
        wait_for: str | None,
        js_code: str | None,
        extraction_strategy: object | None,
    ) -> CrawlerRunConfig:
        """Create crawler run configuration.

        Args:
            wait_for: CSS selector to wait for
            js_code: JavaScript code to execute
            extraction_strategy: Extraction strategy instance

        Returns:
            CrawlerRunConfig: Configured crawler run settings
        """
        return CrawlerRunConfig(
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
            page_timeout=int(
                self.config.page_timeout * 1000
            ),  # Convert seconds to milliseconds
            wait_until="networkidle",
        )

    def _build_success_result(
        self,
        url: str,
        result: object,
        extraction_type: str,
    ) -> dict[str, object]:
        """Build success result dictionary.

        Args:
            url: The scraped URL
            result: Crawl result object
            extraction_type: Type of extraction used

        Returns:
            dict[str, object]: Formatted success result
        """
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

    def _build_error_result(
        self,
        url: str,
        error: str | Exception,
        extraction_type: str | None = None,
    ) -> dict[str, object]:
        """Build error result dictionary.

        Args:
            url: The URL that failed
            error: Error message or exception
            extraction_type: Type of extraction attempted

        Returns:
            dict[str, object]: Formatted error result
        """
        # Get additional context for better error reporting
        rate_limit_status = "unknown"
        if hasattr(self.rate_limiter, "current_calls"):
            rate_limit_status = (
                f"{self.rate_limiter.current_calls}/{self.rate_limiter.max_calls}"
            )

        error_context = {
            "url": url,
            "extraction_type": extraction_type,
            "rate_limit_status": rate_limit_status,
            "semaphore_available": self.semaphore._value
            if hasattr(self.semaphore, "_value")
            else "unknown",
        }

        self.logger.error(f"Failed to scrape {url}: {error} | Context: {error_context}")

        return {
            "success": False,
            "error": str(error),
            "error_context": error_context,
            "content": "",
            "metadata": {},
            "url": url,
            "provider": "crawl4ai",
        }

    async def scrape_url(
        self,
        url: str,
        formats: list[str] | None = None,
        extraction_type: str = "markdown",
        wait_for: str | None = None,
        js_code: str | None = None,
    ) -> dict[str, object]:
        """Scrape single URL with advanced extraction options.

        Args:
            url: URL to scrape
            formats: Output formats (ignored, always returns markdown + html)
            extraction_type: Type of extraction ("markdown", "structured", "llm")
            wait_for: CSS selector to wait for before extraction
            js_code: Custom JavaScript to execute

        Returns:
            dict[str, object]: Scrape result with:
                - success: Whether scraping succeeded
                - content: Extracted content in markdown format
                - html: Raw HTML content
                - metadata: Additional information
                - structured_data: Structured extraction results
                - error: Error message if failed
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

                # Create extraction strategy and run configuration
                extraction_strategy = self._create_extraction_strategy(extraction_type)
                run_config = self._create_run_config(
                    wait_for, js_code, extraction_strategy
                )

                # Crawl the URL
                result = await self._crawler.arun(url=url, config=run_config)

                if result.success:
                    return self._build_success_result(url, result, extraction_type)
                else:
                    error_msg = getattr(result, "error_message", "Crawl failed")
                    return self._build_error_result(url, error_msg, extraction_type)

            except Exception as e:
                return self._build_error_result(url, e, extraction_type)

    async def crawl_bulk(
        self, urls: list[str], extraction_type: str = "markdown"
    ) -> list[dict[str, object]]:
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
    ) -> dict[str, object]:
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


# NOTE: CrawlCache and CrawlBenchmark classes have been removed as they are redundant:
#
# CrawlCache functionality is superseded by the main CacheManager in
# src/services/cache/manager.py which provides:
# - Proper CRAWL cache type support with configurable TTL
# - Two-tier caching (local + DragonflyDB)
# - Better memory management and compression
#
# CrawlBenchmark functionality should be moved to scripts/benchmark_crawl4ai_performance.py
# for standalone performance testing.
