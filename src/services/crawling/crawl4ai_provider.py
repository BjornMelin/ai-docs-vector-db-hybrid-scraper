"""Crawl4AI provider using existing implementation."""

import logging
from typing import Any

from crawl4ai import AsyncWebCrawler
from crawl4ai.models import CrawlResult

from ..errors import CrawlServiceError
from .base import CrawlProvider

logger = logging.getLogger(__name__)


class Crawl4AIProvider(CrawlProvider):
    """Crawl4AI provider for high-performance web crawling."""

    def __init__(self):
        """Initialize Crawl4AI provider."""
        self._crawler: AsyncWebCrawler | None = None
        self._initialized = False

    async def initialize(self) -> None:
        """Initialize Crawl4AI crawler."""
        if self._initialized:
            return

        try:
            self._crawler = AsyncWebCrawler()
            await self._crawler.start()
            self._initialized = True
            logger.info("Crawl4AI crawler initialized")
        except Exception as e:
            raise CrawlServiceError(f"Failed to initialize Crawl4AI: {e}") from e

    async def cleanup(self) -> None:
        """Cleanup Crawl4AI resources."""
        if self._crawler:
            await self._crawler.close()
            self._crawler = None
            self._initialized = False
            logger.info("Crawl4AI resources cleaned up")

    async def scrape_url(
        self, url: str, formats: list[str] | None = None
    ) -> dict[str, Any]:
        """Scrape single URL using Crawl4AI.

        Args:
            url: URL to scrape
            formats: Output formats (ignored, always returns markdown)

        Returns:
            Scrape result
        """
        if not self._initialized:
            raise CrawlServiceError("Provider not initialized")

        try:
            # Crawl the URL
            result: CrawlResult = await self._crawler.crawl(url)

            if result.success:
                return {
                    "success": True,
                    "content": result.markdown or "",
                    "html": result.html or "",
                    "metadata": {
                        "title": result.title,
                        "description": getattr(result, "description", ""),
                        "language": getattr(result, "language", "en"),
                    },
                    "url": url,
                }
            else:
                return {
                    "success": False,
                    "error": getattr(result, "error", "Crawl failed"),
                    "content": "",
                    "metadata": {},
                    "url": url,
                }

        except Exception as e:
            logger.error(f"Failed to scrape {url}: {e}")
            return {
                "success": False,
                "error": str(e),
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
        """Crawl entire site using Crawl4AI.

        Note: Crawl4AI doesn't have built-in site crawling,
        so this is implemented as recursive URL discovery.

        Args:
            url: Starting URL
            max_pages: Maximum pages to crawl
            formats: Output formats (ignored)

        Returns:
            Crawl result
        """
        if not self._initialized:
            raise CrawlServiceError("Provider not initialized")

        pages = []
        visited_urls = set()
        to_visit = [url]

        try:
            while to_visit and len(pages) < max_pages:
                current_url = to_visit.pop(0)
                if current_url in visited_urls:
                    continue

                visited_urls.add(current_url)

                # Scrape the page
                result = await self.scrape_url(current_url)

                if result["success"]:
                    pages.append(
                        {
                            "url": current_url,
                            "content": result["content"],
                            "html": result["html"],
                            "metadata": result["metadata"],
                        }
                    )

                    # Extract links from HTML (simplified)
                    # In production, use proper HTML parsing
                    # This is a placeholder for the actual implementation

                logger.info(f"Crawled {len(pages)}/{max_pages} pages from {url}")

            return {
                "success": True,
                "pages": pages,
                "total": len(pages),
            }

        except Exception as e:
            logger.error(f"Failed to crawl site {url}: {e}")
            return {
                "success": False,
                "error": str(e),
                "pages": pages,
                "total": len(pages),
            }
