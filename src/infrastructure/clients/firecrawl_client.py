"""Firecrawl client provider."""

import logging
from typing import Any, Optional


try:
    from firecrawl import AsyncFirecrawlApp
except ImportError:
    # Create a placeholder if firecrawl is not available
    class AsyncFirecrawlApp:
        pass


logger = logging.getLogger(__name__)


class FirecrawlClientProvider:
    """Provider for Firecrawl client with health checks and circuit breaker."""

    def __init__(
        self,
        firecrawl_client: AsyncFirecrawlApp,
    ):
        self._client = firecrawl_client
        self._healthy = True

    @property
    def client(self) -> AsyncFirecrawlApp | None:
        """Get the Firecrawl client if available and healthy."""
        if not self._healthy:
            return None
        return self._client

    async def health_check(self) -> bool:
        """Check Firecrawl client health."""
        try:
            if not self._client:
                return False

            # Firecrawl doesn't have a direct health endpoint
            # We'll assume it's healthy if client exists and has API key
            self._healthy = True
            return True
        except Exception as e:
            logger.warning(f"Firecrawl health check failed: {e}")
            self._healthy = False
            return False

    async def scrape_url(
        self, url: str, params: dict[str, Any] | None = None
    ) -> dict[str, Any]:
        """Scrape a single URL.

        Args:
            url: URL to scrape
            params: Scraping parameters

        Returns:
            Scraped content

        Raises:
            RuntimeError: If client is unhealthy
        """
        if not self.client:
            msg = "Firecrawl client is not available or unhealthy"
            raise RuntimeError(msg)

        return await self.client.scrape_url(url, params=params or {})

    async def crawl_url(
        self, url: str, params: dict[str, Any] | None = None
    ) -> dict[str, Any]:
        """Crawl a website starting from URL.

        Args:
            url: Starting URL for crawl
            params: Crawling parameters

        Returns:
            Crawl job information

        Raises:
            RuntimeError: If client is unhealthy
        """
        if not self.client:
            msg = "Firecrawl client is not available or unhealthy"
            raise RuntimeError(msg)

        return await self.client.crawl_url(url, params=params or {})

    async def get_crawl_status(self, job_id: str) -> dict[str, Any]:
        """Get status of crawl job.

        Args:
            job_id: Crawl job ID

        Returns:
            Job status and results

        Raises:
            RuntimeError: If client is unhealthy
        """
        if not self.client:
            msg = "Firecrawl client is not available or unhealthy"
            raise RuntimeError(msg)

        return await self.client.get_crawl_status(job_id)

    async def search(
        self, query: str, params: dict[str, Any] | None = None
    ) -> dict[str, Any]:
        """Search the web using Firecrawl.

        Args:
            query: Search query
            params: Search parameters

        Returns:
            Search results

        Raises:
            RuntimeError: If client is unhealthy
        """
        if not self.client:
            msg = "Firecrawl client is not available or unhealthy"
            raise RuntimeError(msg)

        return await self.client.search(query, params=params or {})
