"""Crawl4AI adapter for browser automation router."""

import asyncio  # noqa: PLC0415
import logging  # noqa: PLC0415
import time  # noqa: PLC0415
from typing import Any

from src.config import Crawl4AIConfig

from ..base import BaseService
from ..crawling.crawl4ai_provider import Crawl4AIProvider
from ..errors import CrawlServiceError


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
            raise CrawlServiceError("Failed to initialize Crawl4AI adapter") from e

    async def cleanup(self) -> None:
        """Cleanup Crawl4AI resources."""
        if self._provider:
            try:
                await self._provider.cleanup()
                self.logger.info("Crawl4AI adapter cleaned up")
            except Exception as e:
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
            raise CrawlServiceError("Adapter not initialized")

        start_time = time.time()

        try:
            # Use Crawl4AI's enhanced scraping capabilities
            result = await self._provider.scrape_url(
                url=url,
                formats=["markdown"],  # Standard format
                extraction_type="markdown",
                wait_for=wait_for_selector,
                js_code=js_code,
            )

            # Standardize the response format for automation router
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
                        "processing_time_ms": (time.time() - start_time) * 1000,
                    },
                    "links": result.get("links", []),
                    "structured_data": result.get("structured_data", {}),
                }
            else:
                return {
                    "success": False,
                    "url": url,
                    "error": result.get("error", "Unknown Crawl4AI error"),
                    "content": "",
                    "metadata": {
                        "extraction_method": "crawl4ai",
                        "processing_time_ms": (time.time() - start_time) * 1000,
                    },
                }

        except Exception as e:
            self.logger.exception("Crawl4AI adapter error for {url}")
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
            raise CrawlServiceError("Adapter not initialized")

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
        except Exception as e:
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
