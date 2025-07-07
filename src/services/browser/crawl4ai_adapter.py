"""Crawl4AI adapter for browser automation router."""

import asyncio
import logging
import time
from typing import Any

from src.config import Crawl4AIConfig
from src.services.base import BaseService
from src.services.crawling.crawl4ai_provider import Crawl4AIProvider
from src.services.errors import CrawlServiceError


logger = logging.getLogger(__name__)


class Crawl4AIAdapter(BaseService):
    """Adapter for Crawl4AI to work with automation router."""

    def __init__(self, config: Crawl4AIConfig):
        """Initialize Crawl4AI adapter.

        Args:
            config: Crawl4AI configuration model

        """
        super().__init__(config)
        self.config = config
        self.logger = logger

        # Pass Pydantic config directly to the provider
        self._provider = Crawl4AIProvider(
            config=config,
            rate_limiter=None,  # Rate limiting handled by router
        )
        self._initialized = False

    async def initialize(self) -> None:
        """Initialize Crawl4AI provider."""
        if self._initialized:
            return

        try:
            await self._provider.initialize()
            self._initialized = True
            self.logger.info("Crawl4AI adapter initialized successfully")
        except Exception as e:
            msg = "Failed to initialize Crawl4AI adapter"
            raise CrawlServiceError(msg) from e

    async def cleanup(self) -> None:
        """Cleanup Crawl4AI resources."""
        if self._provider:
            try:
                await self._provider.cleanup()
                self.logger.info("Crawl4AI adapter cleaned up")
            except (OSError, AttributeError, ConnectionError, ImportError):
                self.logger.exception("Error cleaning up Crawl4AI adapter")
            finally:
                self._initialized = False

    async def scrape(
        self,
        url: str,
        wait_for_selector: str | None = None,
        js_code: str | None = None,
        _timeout: int = 30000,
    ) -> dict[str, Any]:
        """Scrape URL using Crawl4AI.

        Args:
            url: URL to scrape
            wait_for_selector: CSS selector to wait for
            js_code: JavaScript code to execute
            timeout: Timeout in milliseconds

        Returns:
            Scraping result with standardized format

        """
        if not self._initialized:
            msg = "Adapter not initialized"
            raise CrawlServiceError(msg)

        start_time = time.time()

        try:
            result = await self._provider.scrape_url(
                url=url,
                formats=["markdown"],  # Standard format
                extraction_type="markdown",
                wait_for=wait_for_selector,
                js_code=js_code,
            )
            return self._build_scrape_response(
                result, url, js_code, wait_for_selector, start_time
            )

        except Exception as e:
            self.logger.exception("Crawl4AI adapter error for %s", url)
            return {
                "success": False,
                "url": url,
                "error": str(e),
                "content": "",
                "metadata": {
                    "extraction_method": "crawl4ai",
                    "processing_time_ms": (time.time() - start_time) * 1000,
                },
            }

    async def crawl_bulk(
        self,
        urls: list[str],
        extraction_type: str = "markdown",
    ) -> list[dict[str, Any]]:
        """Crawl multiple URLs concurrently.

        Args:
            urls: List of URLs to crawl
            extraction_type: Type of extraction

        Returns:
            List of scraping results

        """
        if not self._initialized:
            msg = "Adapter not initialized"
            raise CrawlServiceError(msg)

        # Use provider's bulk crawling capability
        results = await self._provider.crawl_bulk(urls, extraction_type)

        # Standardize response format
        standardized_results = []
        for result in results:
            if result.get("success", False):
                standardized_results.append(
                    {
                        "success": True,
                        "url": result.get("url", ""),
                        "content": result.get("content", ""),
                        "html": result.get("html", ""),
                        "title": result.get("title", ""),
                        "metadata": {
                            **result.get("metadata", {}),
                            "extraction_method": "crawl4ai_bulk",
                        },
                        "links": result.get("links", []),
                        "structured_data": result.get("structured_data", {}),
                    }
                )
            else:
                standardized_results.append(
                    {
                        "success": False,
                        "url": result.get("url", ""),
                        "error": result.get("error", "Unknown error"),
                        "content": "",
                        "metadata": {"extraction_method": "crawl4ai_bulk"},
                    }
                )

        return standardized_results

    def _build_scrape_response(
        self,
        result: dict[str, Any],
        url: str,
        js_code: str | None,
        wait_for_selector: str | None,
        start_time: float,
    ) -> dict[str, Any]:
        """Build standardized scrape response from provider result."""
        processing_time_ms = (time.time() - start_time) * 1000

        if result.get("success", False):
            return {
                "success": True,
                "url": url,
                "content": result.get("content", ""),
                "html": result.get("html", ""),
                "title": result.get("title", ""),
                "metadata": {
                    **result.get("metadata", {}),
                    "extraction_method": "crawl4ai",
                    "js_executed": bool(js_code),
                    "wait_selector": wait_for_selector,
                    "processing_time_ms": processing_time_ms,
                },
                "links": result.get("links", []),
                "structured_data": result.get("structured_data", {}),
            }
        return {
            "success": False,
            "url": url,
            "error": result.get("error", "Unknown Crawl4AI error"),
            "content": "",
            "metadata": {
                "extraction_method": "crawl4ai",
                "processing_time_ms": processing_time_ms,
            },
        }

    def get_capabilities(self) -> dict[str, Any]:
        """Get adapter capabilities and limitations.

        Returns:
            Capabilities dictionary

        """
        return {
            "name": "crawl4ai",
            "description": "High-performance web crawling with basic JavaScript support",
            "advantages": [
                "4-6x faster than alternatives",
                "Zero cost",
                "Excellent for static content",
                "Good parallel processing",
                "Advanced content extraction",
            ],
            "limitations": [
                "Limited complex JavaScript interaction",
                "No AI-powered automation",
                "Basic dynamic content handling",
            ],
            "best_for": [
                "Documentation sites",
                "Static content",
                "Bulk crawling",
                "API documentation",
                "Blog posts",
            ],
            "performance": {
                "avg_speed": "0.4s per page",
                "concurrency": "10-50 pages",
                "success_rate": "98% for static sites",
            },
            "javascript_support": "basic",
            "dynamic_content": "limited",
            "authentication": False,
            "cost": 0,
        }

    async def health_check(self) -> dict[str, Any]:
        """Check adapter health and availability.

        Returns:
            Health status dictionary

        """
        try:
            if not self._initialized:
                return {
                    "healthy": False,
                    "status": "not_initialized",
                    "message": "Adapter not initialized",
                }

            # Test with a simple URL
            test_url = "https://httpbin.org/html"
            start_time = time.time()

            result = await asyncio.wait_for(self.scrape(test_url), timeout=10.0)

            response_time = time.time() - start_time

            return {
                "healthy": result.get("success", False),
                "status": "operational" if result.get("success") else "degraded",
                "message": "Health check passed"
                if result.get("success")
                else result.get("error", "Health check failed"),
                "response_time_ms": response_time * 1000,
                "test_url": test_url,
                "capabilities": self.get_capabilities(),
            }

        except TimeoutError:
            return {
                "healthy": False,
                "status": "timeout",
                "message": "Health check timed out",
                "response_time_ms": 10000,
            }
        except (AttributeError, RuntimeError, ValueError):
            return {
                "healthy": False,
                "status": "error",
                "message": "Health check failed",
            }

    async def get_performance_metrics(self) -> dict[str, Any]:
        """Get performance metrics from the underlying provider.

        Returns:
            Performance metrics dictionary

        """
        # Access provider's internal metrics if available
        if hasattr(self._provider, "metrics"):
            return self._provider.metrics

        return {
            "total_requests": 0,
            "successful_requests": 0,
            "failed_requests": 0,
            "avg_response_time": 0.0,
            "cache_hits": 0,
            "cache_misses": 0,
        }
