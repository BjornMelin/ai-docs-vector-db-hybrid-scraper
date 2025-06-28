"""Crawling manager service coordinator."""

import logging
from typing import TYPE_CHECKING, Any, Optional

from dependency_injector.wiring import Provide, inject

from src.infrastructure.container import ApplicationContainer
from src.services.errors import CrawlServiceError


if TYPE_CHECKING:
    from src.config import Config
    from src.services.crawling.manager import CrawlManager as CoreCrawlManager

logger = logging.getLogger(__name__)


class CrawlingManager:
    """Focused manager for browser automation and scraping services.

    Wraps and coordinates the core CrawlManager with
    5-tier browser automation and content extraction.
    """

    def __init__(self):
        """Initialize crawling manager."""
        self._core_manager: CoreCrawlManager | None = None
        self._initialized = False

    @inject
    async def initialize(
        self,
        config: "Config" = Provide[ApplicationContainer.config],
    ) -> None:
        """Initialize crawling manager using dependency injection.

        Args:
            config: Configuration from DI container
        """
        if self._initialized:
            return

        try:
            from src.services.crawling.manager import CrawlManager as CoreManager

            self._core_manager = CoreManager(
                config=config,
                rate_limiter=None,  # TODO: Add rate limiter injection
            )
            await self._core_manager.initialize()

            self._initialized = True
            logger.info("CrawlingManager service initialized with 5-tier automation")

        except Exception as e:
            logger.error(f"Failed to initialize CrawlingManager: {e}")  # TODO: Convert f-string to logging format
            raise CrawlServiceError(
                f"Failed to initialize crawling manager: {e}"
            ) from e

    async def cleanup(self) -> None:
        """Cleanup crawling manager resources."""
        if self._core_manager:
            await self._core_manager.cleanup()
            self._core_manager = None

        self._initialized = False
        logger.info("CrawlingManager service cleaned up")

    async def scrape_url(
        self,
        url: str,
        preferred_provider: str | None = None,
    ) -> dict[str, Any]:
        """Scrape URL with intelligent 5-tier automation.

        Uses AutomationRouter's 5-tier approach:
        - Tier 0: Lightweight HTTP (httpx + BeautifulSoup)
        - Tier 1: Crawl4AI Basic (90% of sites)
        - Tier 2: Crawl4AI Enhanced (Interactive content)
        - Tier 3: browser-use (Complex interactions)
        - Tier 4: Playwright + Firecrawl (Maximum control)

        Args:
            url: URL to scrape
            preferred_provider: Specific tool to force use

        Returns:
            Scraping result with content, metadata, and performance info

        Raises:
            CrawlServiceError: If manager not initialized or scraping fails
        """
        if not self._initialized or not self._core_manager:
            raise CrawlServiceError("Crawling manager not initialized")

        try:
            return await self._core_manager.scrape_url(url, preferred_provider)
        except Exception as e:
            logger.error(f"URL scraping failed for {url}: {e}")  # TODO: Convert f-string to logging format
            return {
                "success": False,
                "error": f"Scraping failed: {e}",
                "content": "",
                "url": url,
                "title": "",
                "metadata": {},
                "tier_used": "none",
                "automation_time_ms": 0,
                "quality_score": 0.0,
            }

    async def crawl_site(
        self,
        url: str,
        max_pages: int = 50,
        preferred_provider: str | None = None,
    ) -> dict[str, Any]:
        """Crawl entire website from starting URL.

        Args:
            url: Starting URL for crawl
            max_pages: Maximum pages to crawl
            preferred_provider: Specific tool to use for crawling

        Returns:
            Crawl results with pages list and metadata

        Raises:
            CrawlServiceError: If manager not initialized
        """
        if not self._initialized or not self._core_manager:
            raise CrawlServiceError("Crawling manager not initialized")

        try:
            return await self._core_manager.crawl_site(
                url, max_pages, preferred_provider
            )
        except Exception as e:
            logger.error(f"Site crawling failed for {url}: {e}")  # TODO: Convert f-string to logging format
            return {
                "success": False,
                "error": f"Site crawl failed: {e}",
                "pages": [],
                "total_pages": 0,
                "provider": "none",
            }

    async def get_recommended_tool(self, url: str) -> str:
        """Get recommended tool for a URL based on performance metrics.

        Args:
            url: URL to analyze

        Returns:
            Recommended tool name based on analysis

        Raises:
            CrawlServiceError: If manager not initialized
        """
        if not self._initialized or not self._core_manager:
            raise CrawlServiceError("Crawling manager not initialized")

        try:
            return await self._core_manager.get_recommended_tool(url)
        except Exception as e:
            logger.warning(f"Tool recommendation failed for {url}: {e}")  # TODO: Convert f-string to logging format
            return "crawl4ai"  # Default fallback

    async def map_url(
        self, url: str, include_subdomains: bool = False
    ) -> dict[str, Any]:
        """Map a website to get list of URLs.

        Note: Only Firecrawl supports this feature.

        Args:
            url: URL to map
            include_subdomains: Include subdomains in the mapping

        Returns:
            Map result with discovered URLs

        Raises:
            CrawlServiceError: If manager not initialized
        """
        if not self._initialized or not self._core_manager:
            raise CrawlServiceError("Crawling manager not initialized")

        try:
            return await self._core_manager.map_url(url, include_subdomains)
        except Exception as e:
            logger.warning(f"URL mapping failed for {url}: {e}")  # TODO: Convert f-string to logging format
            return {
                "success": False,
                "error": f"URL mapping failed: {e}",
                "urls": [],
                "total": 0,
            }

    def get_metrics(self) -> dict[str, dict]:
        """Get performance metrics for all tiers.

        Returns:
            Dictionary with metrics for each tier including success rates and timing

        Raises:
            CrawlServiceError: If manager not initialized
        """
        if not self._initialized or not self._core_manager:
            return {}

        try:
            return self._core_manager.get_metrics()
        except Exception as e:
            logger.warning(f"Failed to get crawling metrics: {e}")  # TODO: Convert f-string to logging format
            return {}

    def get_provider_info(self) -> dict[str, dict]:
        """Get information about available automation tools.

        Returns:
            Tool information including tier assignments and metrics

        Raises:
            CrawlServiceError: If manager not initialized
        """
        if not self._initialized or not self._core_manager:
            return {}

        try:
            return self._core_manager.get_provider_info()
        except Exception as e:
            logger.warning(f"Failed to get provider info: {e}")  # TODO: Convert f-string to logging format
            return {}

    def get_tier_metrics(self) -> dict[str, dict]:
        """Get performance metrics for each tier.

        Returns:
            Tier performance metrics for all 5 tiers

        Raises:
            CrawlServiceError: If manager not initialized
        """
        if not self._initialized or not self._core_manager:
            return {}

        try:
            return self._core_manager.get_tier_metrics()
        except Exception as e:
            logger.warning(f"Failed to get tier metrics: {e}")  # TODO: Convert f-string to logging format
            return {}

    async def get_status(self) -> dict[str, Any]:
        """Get crawling manager status.

        Returns:
            Status information for providers and metrics
        """
        status = {
            "initialized": self._initialized,
            "providers": {},
            "metrics": {},
            "tier_metrics": {},
        }

        if self._core_manager:
            try:
                status["providers"] = self.get_provider_info()
                status["metrics"] = self.get_metrics()
                status["tier_metrics"] = self.get_tier_metrics()
            except Exception as e:
                logger.warning(f"Failed to get crawling status: {e}")  # TODO: Convert f-string to logging format
                status["error"] = str(e)

        return status

    def get_core_manager(self) -> Optional["CoreCrawlManager"]:
        """Get core crawl manager instance.

        Returns:
            Core CrawlManager instance or None
        """
        return self._core_manager

    # Content Extraction Helpers
    async def extract_content(
        self,
        url: str,
        content_type: str = "text",
        extraction_rules: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        """Extract specific content from URL.

        Args:
            url: URL to extract content from
            content_type: Type of content to extract (text, links, images, etc.)
            extraction_rules: Custom extraction rules

        Returns:
            Extracted content with metadata
        """
        if not self._initialized or not self._core_manager:
            raise CrawlServiceError("Crawling manager not initialized")

        try:
            # Use scrape_url as base and extract specific content
            result = await self.scrape_url(url)

            if not result.get("success"):
                return result

            # Basic content extraction based on type
            content = result.get("content", "")
            metadata = result.get("metadata", {})

            extracted = {
                "success": True,
                "url": url,
                "content_type": content_type,
                "extraction_rules": extraction_rules or {},
                "tier_used": result.get("tier_used", "unknown"),
                "automation_time_ms": result.get("automation_time_ms", 0),
            }

            if content_type == "text":
                extracted["content"] = content
            elif content_type == "links":
                extracted["content"] = metadata.get("links", [])
            elif content_type == "images":
                extracted["content"] = metadata.get("images", [])
            else:
                extracted["content"] = {
                    "text": content,
                    "metadata": metadata,
                }

            return extracted

        except Exception as e:
            logger.error(f"Content extraction failed for {url}: {e}")  # TODO: Convert f-string to logging format
            return {
                "success": False,
                "error": f"Content extraction failed: {e}",
                "url": url,
                "content_type": content_type,
                "content": None,
            }

    async def bulk_scrape(
        self,
        urls: list[str],
        preferred_provider: str | None = None,
        max_concurrent: int = 5,
    ) -> list[dict[str, Any]]:
        """Scrape multiple URLs concurrently.

        Args:
            urls: List of URLs to scrape
            preferred_provider: Specific tool to use for all URLs
            max_concurrent: Maximum concurrent operations

        Returns:
            List of scraping results for each URL
        """
        if not self._initialized or not self._core_manager:
            raise CrawlServiceError("Crawling manager not initialized")

        import asyncio

        semaphore = asyncio.Semaphore(max_concurrent)

        async def scrape_with_semaphore(url: str) -> dict[str, Any]:
            async with semaphore:
                return await self.scrape_url(url, preferred_provider)

        try:
            tasks = [scrape_with_semaphore(url) for url in urls]
            results = await asyncio.gather(*tasks, return_exceptions=True)

            # Convert exceptions to error results
            processed_results = []
            for i, result in enumerate(results):
                if isinstance(result, Exception):
                    processed_results.append(
                        {
                            "success": False,
                            "error": f"Scraping failed: {result}",
                            "url": urls[i],
                            "content": "",
                            "metadata": {},
                        }
                    )
                else:
                    processed_results.append(result)

            return processed_results

        except Exception as e:
            logger.error(f"Bulk scraping failed: {e}")  # TODO: Convert f-string to logging format
            # Return error results for all URLs
            return [
                {
                    "success": False,
                    "error": f"Bulk scraping failed: {e}",
                    "url": url,
                    "content": "",
                    "metadata": {},
                }
                for url in urls
            ]