"""Enhanced Crawl4AI provider with Memory-Adaptive Dispatcher for intelligent concurrency control."""

import asyncio
import logging
import subprocess
from collections.abc import AsyncIterator
from urllib.parse import urlparse

from crawl4ai import AsyncWebCrawler, BrowserConfig, CrawlerRunConfig
from crawl4ai.async_configs import LLMConfig
from crawl4ai.extraction_strategy import (
    JsonCssExtractionStrategy,
    LLMExtractionStrategy,
)


try:
    from crawl4ai import (
        CrawlerMonitor,
        LXMLWebScrapingStrategy,
        MemoryAdaptiveDispatcher,
        RateLimiter as Crawl4AIRateLimiter,
    )

    MEMORY_ADAPTIVE_AVAILABLE = True
except ImportError:
    MEMORY_ADAPTIVE_AVAILABLE = False

from src.config import Crawl4AIConfig
from src.services.base import BaseService
from src.services.errors import CrawlServiceError
from src.services.utilities.rate_limiter import RateLimiter

from .base import CrawlProvider
from .extractors import DocumentationExtractor, JavaScriptExecutor


logger = logging.getLogger(__name__)


class Crawl4AIProvider(BaseService, CrawlProvider):
    """High-performance web crawling with Memory-Adaptive Dispatcher for intelligent concurrency control."""

    def __init__(self, config: Crawl4AIConfig, rate_limiter: object = None):
        """Initialize Crawl4AI provider with Memory-Adaptive Dispatcher."""
        super().__init__(config)
        self.config = config
        self.logger = logger

        # Legacy rate limiter (kept for compatibility)
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

        # Memory-Adaptive Dispatcher or fallback to semaphore
        if MEMORY_ADAPTIVE_AVAILABLE and self.config.enable_memory_adaptive_dispatcher:
            self.dispatcher = self._create_memory_dispatcher()
            self.use_memory_dispatcher = True
            self.logger.info(
                "Using MemoryAdaptiveDispatcher for intelligent concurrency control"
            )
        else:
            # Fallback to traditional semaphore approach
            if (
                self.config.enable_memory_adaptive_dispatcher
                and not MEMORY_ADAPTIVE_AVAILABLE
            ):
                self.logger.warning(
                    "Memory-Adaptive Dispatcher requested but not available, "
                    "falling back to semaphore-based concurrency"
                )
            self.max_concurrent = self.config.max_concurrent_crawls
            self.semaphore = asyncio.Semaphore(self.max_concurrent)
            self.dispatcher = None
            self.use_memory_dispatcher = False
            self.logger.info(
                f"Using semaphore-based concurrency control (max: {self.max_concurrent})"
            )

        # Initialize helpers
        self.js_executor = JavaScriptExecutor()
        self.doc_extractor = DocumentationExtractor()

        self._crawler: AsyncWebCrawler | None = None
        self._initialized = False

    def _create_memory_dispatcher(self) -> "MemoryAdaptiveDispatcher":
        """Create Memory-Adaptive Dispatcher with configuration."""
        # Create Crawl4AI rate limiter with exponential backoff
        crawl4ai_rate_limiter = Crawl4AIRateLimiter(
            base_delay=(
                self.config.rate_limit_base_delay_min,
                self.config.rate_limit_base_delay_max,
            ),
            max_delay=self.config.rate_limit_max_delay,
            max_retries=self.config.rate_limit_max_retries,
        )

        # Create performance monitor
        monitor = CrawlerMonitor(refresh_rate=1.0, enable_ui=False)

        # Create Memory-Adaptive Dispatcher
        dispatcher = MemoryAdaptiveDispatcher(
            memory_threshold_percent=self.config.memory_threshold_percent,
            check_interval=self.config.dispatcher_check_interval,
            max_session_permit=self.config.max_session_permit,
            rate_limiter=crawl4ai_rate_limiter,
            monitor=monitor,
        )

        self.logger.info(
            f"Created MemoryAdaptiveDispatcher: "
            f"memory_threshold={self.config.memory_threshold_percent}%, "
            f"max_sessions={self.config.max_session_permit}, "
            f"check_interval={self.config.dispatcher_check_interval}s"
        )

        return dispatcher

    async def initialize(self) -> None:
        """Initialize Crawl4AI crawler with Memory-Adaptive Dispatcher."""
        if self._initialized:
            return

        try:
            # Initialize Memory-Adaptive Dispatcher if enabled
            if self.use_memory_dispatcher and self.dispatcher:
                # Check if dispatcher has initialize method
                if hasattr(self.dispatcher, "initialize"):
                    await self.dispatcher.initialize()
                    self.logger.info(
                        "MemoryAdaptiveDispatcher initialized successfully"
                    )
                else:
                    self.logger.info(
                        "MemoryAdaptiveDispatcher ready (no initialize method)"
                    )

            # Create crawler with enhanced configuration
            crawler_config = self.browser_config

            # Add LXMLWebScrapingStrategy for performance improvement if available
            if MEMORY_ADAPTIVE_AVAILABLE:
                try:
                    crawler_config.web_scraping_strategy = LXMLWebScrapingStrategy()
                    self.logger.info(
                        "Using LXMLWebScrapingStrategy for enhanced performance"
                    )
                except (
                    AttributeError,
                    ConnectionError,
                    ImportError,
                    RuntimeError,
                ) as e:
                    self.logger.warning("Could not set LXMLWebScrapingStrategy")

            self._crawler = AsyncWebCrawler(config=crawler_config)
            await self._crawler.start()
            self._initialized = True

            strategy_info = (
                "with LXMLWebScrapingStrategy"
                if MEMORY_ADAPTIVE_AVAILABLE
                else "with default strategy"
            )
            dispatcher_info = (
                "MemoryAdaptiveDispatcher"
                if self.use_memory_dispatcher
                else "semaphore"
            )
            self.logger.info(
                f"Crawl4AI crawler initialized {strategy_info} and {dispatcher_info}"
            )

        except Exception as e:
            msg = "Failed to initialize Crawl4AI"
            raise CrawlServiceError(msg) from e

    async def cleanup(self) -> None:
        """Cleanup Crawl4AI resources and Memory-Adaptive Dispatcher."""
        # Cleanup Memory-Adaptive Dispatcher first
        if self.use_memory_dispatcher and self.dispatcher:
            try:
                # MemoryAdaptiveDispatcher doesn't have cleanup method, just reset reference
                self.dispatcher = None
                self.logger.info("MemoryAdaptiveDispatcher reference cleared")
            except (ConnectionError, OSError, PermissionError) as e:
                self.logger.exception("Error cleaning up MemoryAdaptiveDispatcher")

        # Cleanup crawler
        if self._crawler:
            try:
                await self._crawler.close()
            except (OSError, AttributeError, ConnectionError, ImportError) as e:
                self.logger.exception("Error closing crawler")
            finally:
                # Always reset state even if close() fails
                self._crawler = None
                self._initialized = False

                cleanup_info = (
                    "with MemoryAdaptiveDispatcher"
                    if self.use_memory_dispatcher
                    else "with semaphore"
                )
                self.logger.info(
                    f"Crawl4AI resources cleaned up {cleanup_info}"
                )  # TODO: Convert f-string to logging format

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
        if extraction_type == "llm":
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
            "semaphore_available": (
                getattr(self, "semaphore", None)
                and hasattr(self.semaphore, "_value")
                and self.semaphore._value
            )
            or "unknown",
        }

        self.logger.error("Failed to scrape {url}: {error} | Context")

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
        _formats: list[str] | None = None,
        extraction_type: str = "markdown",
        wait_for: str | None = None,
        js_code: str | None = None,
        stream: bool | None = None,
    ) -> dict[str, object]:
        """Scrape single URL with Memory-Adaptive Dispatcher and optional streaming.

        Args:
            url: URL to scrape
            formats: Output formats (ignored, always returns markdown + html)
            extraction_type: Type of extraction ("markdown", "structured", "llm")
            wait_for: CSS selector to wait for before extraction
            js_code: Custom JavaScript to execute
            stream: Enable streaming mode (uses config default if None)

        Returns:
            dict[str, object]: Scrape result with:
                - success: Whether scraping succeeded
                - content: Extracted content in markdown format
                - html: Raw HTML content
                - metadata: Additional information
                - structured_data: Structured extraction results
                - error: Error message if failed
                - stream_data: Streaming iterator if streaming enabled

        """
        if not self._initialized:
            msg = "Provider not initialized"
            raise CrawlServiceError(msg)

        # Determine if streaming should be used
        enable_streaming = (
            stream if stream is not None else self.config.enable_streaming
        )

        # Use Memory-Adaptive Dispatcher or fallback to semaphore
        if self.use_memory_dispatcher:
            return await self._scrape_with_dispatcher(
                url, extraction_type, wait_for, js_code, enable_streaming
            )
        return await self._scrape_with_semaphore(
            url, extraction_type, wait_for, js_code
        )

    async def _scrape_with_dispatcher(
        self,
        url: str,
        extraction_type: str,
        wait_for: str | None,
        js_code: str | None,
        enable_streaming: bool,
    ) -> dict[str, object]:
        """Scrape URL using Memory-Adaptive Dispatcher."""
        try:
            # Get site-specific JavaScript if not provided
            if not js_code:
                js_code = self.js_executor.get_js_for_site(url)

            # Create extraction strategy and run configuration
            extraction_strategy = self._create_extraction_strategy(extraction_type)
            run_config = self._create_run_config(wait_for, js_code, extraction_strategy)

            # Enable streaming in run config if requested
            if enable_streaming:
                run_config.stream = True

            # Crawl the URL (MemoryAdaptiveDispatcher handles concurrency internally)
            if enable_streaming:
                # Stream processing for real-time results
                return await self._process_streaming_crawl(
                    url, run_config, extraction_type, None
                )
            # Standard crawl
            result = await self._crawler.arun(url=url, config=run_config)

            if result.success:
                success_result = self._build_success_result(
                    url, result, extraction_type
                )
                success_result["metadata"]["dispatcher_stats"] = (
                    self._get_dispatcher_stats()
                )
                return success_result
            error_msg = getattr(result, "error_message", "Crawl failed")
            return self._build_error_result(url, error_msg, extraction_type)

        except (RuntimeError, ValueError, TypeError) as e:
            error_result = self._build_error_result(url, e, extraction_type)
            error_result["metadata"]["dispatcher_stats"] = self._get_dispatcher_stats()
            return error_result

    async def _scrape_with_semaphore(
        self,
        url: str,
        extraction_type: str,
        wait_for: str | None,
        js_code: str | None,
    ) -> dict[str, object]:
        """Scrape URL using traditional semaphore approach (fallback)."""
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
                error_msg = getattr(result, "error_message", "Crawl failed")
                return self._build_error_result(url, error_msg, extraction_type)

            except (subprocess.SubprocessError, OSError, TimeoutError) as e:
                return self._build_error_result(url, e, extraction_type)

    async def _process_streaming_crawl(
        self,
        url: str,
        run_config: CrawlerRunConfig,
        extraction_type: str,
        _session: object | None,
    ) -> dict[str, object]:
        """Process crawl with streaming for real-time results."""
        try:
            # Create async iterator for streaming results
            # Note: AsyncWebCrawler doesn't have arun_stream, using arun instead
            result = await self._crawler.arun(url=url, config=run_config)

            # Collect streaming results
            streaming_results = []
            final_result = None

            async for chunk in stream_iterator:
                if chunk:
                    streaming_results.append(chunk)
                    # Store the final complete result
                    if hasattr(chunk, "success") and chunk.success:
                        final_result = chunk

            if final_result and final_result.success:
                success_result = self._build_success_result(
                    url, final_result, extraction_type
                )
                success_result["metadata"]["streaming"] = {
                    "enabled": True,
                    "chunks_received": len(streaming_results),
                    "real_time_processing": True,
                }
                success_result["metadata"]["dispatcher_stats"] = (
                    self._get_dispatcher_stats()
                )
                return success_result
            error_msg = "Streaming crawl failed to produce valid results"
            return self._build_error_result(url, error_msg, extraction_type)

        except Exception as e:
            self.logger.exception("Streaming crawl failed for {url}")
            return self._build_error_result(url, e, extraction_type)

    def _get_dispatcher_stats(self) -> dict[str, object]:
        """Get Memory-Adaptive Dispatcher performance statistics."""
        if not self.use_memory_dispatcher or not self.dispatcher:
            return {"dispatcher_type": "semaphore", "active_sessions": "unknown"}

        try:
            stats = {
                "dispatcher_type": "memory_adaptive",
                "memory_threshold_percent": self.config.memory_threshold_percent,
                "max_session_permit": self.config.max_session_permit,
                "check_interval": self.config.dispatcher_check_interval,
            }

            # Add runtime stats if available
            if hasattr(self.dispatcher, "get_stats"):
                runtime_stats = self.dispatcher.get_stats()
                stats.update(runtime_stats)

            return stats
        except (asyncio.CancelledError, TimeoutError, RuntimeError) as e:
            self.logger.warning("Could not get dispatcher stats")
            return {"dispatcher_type": "memory_adaptive", "stats_error": str(e)}

    async def scrape_url_stream(
        self,
        url: str,
        extraction_type: str = "markdown",
        wait_for: str | None = None,
        js_code: str | None = None,
    ) -> AsyncIterator[dict[str, object]]:
        """Stream scrape results in real-time using Memory-Adaptive Dispatcher.

        Args:
            url: URL to scrape
            extraction_type: Type of extraction to use
            wait_for: CSS selector to wait for
            js_code: Custom JavaScript to execute

        Yields:
            dict[str, object]: Streaming chunks of scrape results

        """
        if not self._initialized:
            msg = "Provider not initialized"
            raise CrawlServiceError(msg)

        if not self.use_memory_dispatcher:
            msg = "Streaming requires Memory-Adaptive Dispatcher"
            raise CrawlServiceError(msg)

        try:
            # Get site-specific JavaScript if not provided
            if not js_code:
                js_code = self.js_executor.get_js_for_site(url)

            # Create extraction strategy and run configuration
            extraction_strategy = self._create_extraction_strategy(extraction_type)
            run_config = self._create_run_config(wait_for, js_code, extraction_strategy)
            run_config.stream = True

            # Stream crawl results (MemoryAdaptiveDispatcher handles concurrency internally)
            # Note: AsyncWebCrawler doesn't have arun_stream, using arun instead
            result = await self._crawler.arun(url=url, config=run_config)
            if result:
                yield {
                    "url": url,
                        "chunk": result,
                        "timestamp": asyncio.get_event_loop().time(),
                        "extraction_type": extraction_type,
                        "provider": "crawl4ai",
                        "streaming": True,
                    }

        except (asyncio.CancelledError, TimeoutError, RuntimeError) as e:
            yield self._build_error_result(url, e, extraction_type)

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
                self.logger.error("Failed to crawl {url}")
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
        _formats: list[str] | None = None,
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
            msg = "Provider not initialized"
            raise CrawlServiceError(msg)

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

                self.logger.info(
                    f"Crawled {len(pages)}/{max_pages} pages from {url}"
                )  # TODO: Convert f-string to logging format

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

            self.logger.exception("Failed to crawl site {url}: {e} | Context")
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
