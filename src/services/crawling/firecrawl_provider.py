"""Firecrawl provider using direct SDK."""

import asyncio
import logging
from typing import Any

from firecrawl import FirecrawlApp

from src.config import FirecrawlConfig

from ..base import BaseService
from ..errors import CrawlServiceError
from ..utilities.rate_limiter import RateLimitManager
from .base import CrawlProvider


logger = logging.getLogger(__name__)


class FirecrawlProvider(BaseService, CrawlProvider):
    """Firecrawl provider for web crawling."""

    def __init__(
        self, config: FirecrawlConfig, rate_limiter: RateLimitManager | None = None
    ):
        """Initialize Firecrawl provider.

        Args:
            config: Firecrawl configuration model
            rate_limiter: Optional rate limiter
        """
        super().__init__(config)
        self.config = config
        self._client: FirecrawlApp | None = None
        self._initialized = False
        self.rate_limiter = rate_limiter

    async def initialize(self) -> None:
        """Initialize Firecrawl client."""
        if self._initialized:
            return

        try:
            self._client = FirecrawlApp(
                api_key=self.config.api_key, api_url=self.config.api_url
            )
            self._initialized = True
            logger.info("Firecrawl client initialized")
        except Exception:
            raise CrawlServiceError("Failed to initialize Firecrawl") from e

    async def cleanup(self) -> None:
        """Cleanup Firecrawl resources."""
        self._client = None
        self._initialized = False
        logger.info("Firecrawl resources cleaned up")

    async def scrape_url(
        self, url: str, formats: list[str] | None = None
    ) -> dict[str, Any]:
        """Scrape single URL.

        Args:
            url: URL to scrape
            formats: Output formats (default: ['markdown'])

        Returns:
            Scrape result
        """
        if not self._initialized:
            raise CrawlServiceError("Provider not initialized")

        formats = formats or ["markdown"]

        try:
            # Firecrawl SDK is synchronous, but we're in async context
            result = await self._scrape_url_with_rate_limit(url, formats)

            # Process result
            if result.get("success", False):
                return {
                    "success": True,
                    "content": result.get("markdown", ""),
                    "html": result.get("html", ""),
                    "metadata": result.get("metadata", {}),
                    "url": url,
                }
            else:
                return {
                    "success": False,
                    "error": result.get("error", "Unknown error"),
                    "content": "",
                    "metadata": {},
                    "url": url,
                }

        except Exception:
            logger.error("Failed to scrape {url}", exc_info=True)

            error_msg = str(e).lower()
            if "rate limit" in error_msg:
                logger.warning(f"Firecrawl rate limit hit for {url}")
                error_detail = "Rate limit exceeded. Please try again later."
            elif "invalid api key" in error_msg or "unauthorized" in error_msg:
                logger.exception("Invalid Firecrawl API key")
                error_detail = (
                    "Invalid API key. Please check your Firecrawl configuration."
                )
            elif "timeout" in error_msg:
                logger.warning(f"Timeout while scraping {url}")
                error_detail = (
                    "Request timed out. The page may be too large or slow to load."
                )
            elif "not found" in error_msg or "404" in error_msg:
                logger.info("Page not found")
                error_detail = "Page not found (404)."
            else:
                error_detail = "Scraping failed"

            return {
                "success": False,
                "error": error_detail,
                "content": "",
                "metadata": {},
                "url": url,
            }

    async def crawl_site(
        self,
        url: str,
        max_pages: int = 50,
        formats: list[str] | None = None,
    ) -> dict[str, Any]:
        """Crawl entire site.

        Args:
            url: Starting URL
            max_pages: Maximum pages to crawl
            formats: Output formats

        Returns:
            Crawl result
        """
        if not self._initialized:
            raise CrawlServiceError("Provider not initialized")

        formats = formats or ["markdown"]

        try:
            # Start async crawl
            crawl_result = await self._async_crawl_url_with_rate_limit(
                url, max_pages, formats
            )

            # Get crawl ID
            crawl_id = crawl_result.get("id")
            if not crawl_id:
                raise CrawlServiceError("No crawl ID returned")

            logger.info(f"Started crawl job {crawl_id} for {url}")

            # Poll for completion

            max_attempts = 120  # 10 minutes with 5 second intervals
            for _ in range(max_attempts):
                status = self._client.check_crawl_status(crawl_id)

                if status.get("status") == "completed":
                    # Get results
                    data = status.get("data", [])
                    return {
                        "success": True,
                        "pages": [
                            {
                                "url": page.get("url", ""),
                                "content": page.get("markdown", ""),
                                "html": page.get("html", ""),
                                "metadata": page.get("metadata", {}),
                            }
                            for page in data
                        ],
                        "total": len(data),
                        "crawl_id": crawl_id,
                    }
                elif status.get("status") == "failed":
                    return {
                        "success": False,
                        "error": status.get("error", "Crawl failed"),
                        "pages": [],
                        "total": 0,
                        "crawl_id": crawl_id,
                    }

                # Wait before next check
                await asyncio.sleep(5)

            # Timeout
            return {
                "success": False,
                "error": "Crawl timed out",
                "pages": [],
                "total": 0,
                "crawl_id": crawl_id,
            }

        except Exception:
            logger.exception("Failed to crawl {url}")
            return {
                "success": False,
                "error": str(e),
                "pages": [],
                "total": 0,
            }

    async def cancel_crawl(self, crawl_id: str) -> bool:
        """Cancel a crawl job.

        Args:
            crawl_id: Crawl job ID

        Returns:
            Success status
        """
        if not self._initialized:
            raise CrawlServiceError("Provider not initialized")

        try:
            result = self._client.cancel_crawl(crawl_id)
            return result.get("success", False)
        except Exception:
            logger.exception("Failed to cancel crawl {crawl_id}")
            return False

    async def map_url(
        self, url: str, include_subdomains: bool = False
    ) -> dict[str, Any]:
        """Map a website to get list of URLs.

        Args:
            url: URL to map
            include_subdomains: Include subdomains

        Returns:
            Map result with URLs
        """
        if not self._initialized:
            raise CrawlServiceError("Provider not initialized")

        try:
            result = self._client.map_url(
                url=url,
                params={"includeSubdomains": include_subdomains},
            )

            if result.get("success", False):
                return {
                    "success": True,
                    "urls": result.get("links", []),
                    "total": len(result.get("links", [])),
                }
            else:
                return {
                    "success": False,
                    "error": result.get("error", "Map failed"),
                    "urls": [],
                    "total": 0,
                }

        except Exception:
            logger.exception("Failed to map {url}")
            return {
                "success": False,
                "error": str(e),
                "urls": [],
                "total": 0,
            }

    async def _scrape_url_with_rate_limit(
        self, url: str, formats: list[str]
    ) -> dict[str, Any]:
        """Scrape URL with rate limiting.

        Args:
            url: URL to scrape
            formats: Output formats

        Returns:
            Firecrawl response
        """
        if self.rate_limiter:
            await self.rate_limiter.acquire("firecrawl")

        # Run synchronous method in thread pool to avoid blocking

        loop = asyncio.get_event_loop()
        return await loop.run_in_executor(None, self._client.scrape_url, url, formats)

    async def _async_crawl_url_with_rate_limit(
        self, url: str, max_pages: int, formats: list[str]
    ) -> dict[str, Any]:
        """Start crawl with rate limiting.

        Args:
            url: Starting URL
            max_pages: Maximum pages
            formats: Output formats

        Returns:
            Crawl job info
        """
        if self.rate_limiter:
            await self.rate_limiter.acquire("firecrawl")

        import asyncio

        loop = asyncio.get_event_loop()
        return await loop.run_in_executor(
            None,
            self._client.async_crawl_url,
            url,
            max_pages,
            {"formats": formats},
        )
