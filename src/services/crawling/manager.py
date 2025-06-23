import typing
"""Crawl manager with 5-tier browser automation via UnifiedBrowserManager."""

import logging
from typing import Any

from src.config import Config

from ..errors import CrawlServiceError

logger = logging.getLogger(__name__)


class CrawlManager:
    """Manager for crawling using 5-tier UnifiedBrowserManager architecture.

    Provides a high-level interface for crawling that internally uses
    the UnifiedBrowserManager to intelligently select the best tier for each URL
    and provides consistent response formatting across all tiers.
    """

    def __init__(self, config: Config, rate_limiter: object = None):
        """Initialize crawl manager.

        Args:
            config: Unified configuration with crawl provider settings
            rate_limiter: Optional RateLimitManager instance for rate limiting
        """
        self.config = config
        self.rate_limiter = rate_limiter
        self._unified_browser_manager: Any = None
        self._initialized = False

    async def initialize(self) -> None:
        """Initialize the 5-tier UnifiedBrowserManager.

        Initializes all 5 tiers through UnifiedBrowserManager:
        - Tier 0: Lightweight HTTP (httpx + BeautifulSoup)
        - Tier 1: Crawl4AI Basic
        - Tier 2: Crawl4AI Enhanced
        - Tier 3: Browser-use AI
        - Tier 4: Playwright + Firecrawl

        Raises:
            CrawlServiceError: If UnifiedBrowserManager initialization fails
        """
        if self._initialized:
            return

        try:
            from ..browser.unified_manager import UnifiedBrowserManager

            self._unified_browser_manager = UnifiedBrowserManager(self.config)
            await self._unified_browser_manager.initialize()

            self._initialized = True
            logger.info("CrawlManager initialized with 5-tier UnifiedBrowserManager")

        except Exception as e:
            logger.exception(f"Failed to initialize UnifiedBrowserManager: {e}")
            raise CrawlServiceError(f"Failed to initialize crawl manager: {e}") from e

    async def cleanup(self) -> None:
        """Cleanup the UnifiedBrowserManager."""
        if self._unified_browser_manager:
            try:
                await self._unified_browser_manager.cleanup()
                logger.info("Cleaned up UnifiedBrowserManager")
            except Exception as e:
                logger.exception(f"Error cleaning up UnifiedBrowserManager: {e}")

            self._unified_browser_manager = None

        self._initialized = False

    async def scrape_url(
        self,
        url: str,
        preferred_provider: str | None = None,
    ) -> dict[str, object]:
        """Scrape URL with intelligent 5-tier AutomationRouter selection.

        Uses AutomationRouter's 5-tier approach:
        - Tier 0: Lightweight HTTP (httpx + BeautifulSoup) - 5-10x faster for static content
        - Tier 1: Crawl4AI Basic (90% of sites) - 4-6x faster, browser automation
        - Tier 2: Crawl4AI Enhanced (Interactive content) - Custom JavaScript execution
        - Tier 3: browser-use (Complex interactions) - AI-powered automation with multi-LLM support
        - Tier 4: Playwright + Firecrawl (Maximum control) - Full programmatic control + API fallback

        Args:
            url: URL to scrape
            preferred_provider: Specific tool to force use (overrides selection logic)

        Returns:
            dict[str, object]: Scraping result with:
                - success: Whether scraping succeeded
                - content: Scraped content
                - metadata: Additional information (title, description, etc.)
                - automation_time_ms: Time taken for automation in milliseconds
                - url: Source URL
                - error: Error message if scraping failed

        Raises:
            CrawlServiceError: If manager not initialized
        """
        if not self._initialized:
            raise CrawlServiceError("Manager not initialized")

        try:
            # Create unified request
            from ..browser.unified_manager import UnifiedScrapingRequest

            request = UnifiedScrapingRequest(
                url=url,
                tier=preferred_provider if preferred_provider else "auto",
                interaction_required=False,
                timeout=30000,
            )

            # Use UnifiedBrowserManager for intelligent tool selection and execution
            result = await self._unified_browser_manager.scrape(request)

            # Return response in standardized format
            return {
                "success": result.success,
                "content": result.content,
                "url": result.url,
                "title": result.title,
                "metadata": result.metadata,
                "tier_used": result.tier_used,
                "automation_time_ms": result.execution_time_ms,
                "quality_score": result.quality_score,
                "error": result.error,
                "fallback_attempted": result.fallback_attempted,
                "failed_tiers": result.failed_tiers,
            }

        except Exception as e:
            logger.exception(f"UnifiedBrowserManager failed for {url}: {e}")
            return {
                "success": False,
                "error": f"UnifiedBrowserManager error: {e!s}",
                "content": "",
                "url": url,
                "title": "",
                "metadata": {},
                "tier_used": "none",
                "automation_time_ms": 0,
                "quality_score": 0.0,
            }

    def get_metrics(self) -> dict[str, dict]:
        """Get performance metrics for all tiers.

        Returns:
            Dictionary with metrics for each tier including success rates and timing
        """
        if not self._unified_browser_manager:
            return {}

        # Get tier metrics from UnifiedBrowserManager
        tier_metrics = self._unified_browser_manager.get_tier_metrics()
        return {tier: metrics.dict() for tier, metrics in tier_metrics.items()}

    async def get_recommended_tool(self, url: str) -> str:
        """Get recommended tool for a URL based on performance metrics.

        Args:
            url: URL to analyze

        Returns:
            Recommended tool name based on UnifiedBrowserManager analysis
        """
        if not self._unified_browser_manager:
            return "crawl4ai"  # Default fallback

        # Use UnifiedBrowserManager's URL analysis
        analysis = await self._unified_browser_manager.analyze_url(url)
        return analysis.get("recommended_tier", "crawl4ai")

    async def crawl_site(
        self,
        url: str,
        max_pages: int = 50,
        preferred_provider: str | None = None,
    ) -> dict[str, object]:
        """Crawl entire website from starting URL using AutomationRouter.

        Args:
            url: Starting URL for crawl
            max_pages: Maximum pages to crawl (default: 50)
            preferred_provider: Specific tool to use for crawling

        Returns:
            dict[str, object]: Crawl results with:
                - success: Whether crawl succeeded
                - pages: List of crawled page data
                - total_pages: Number of pages crawled
                - provider: Name of tool used
                - error: Error message if crawl failed

        Raises:
            CrawlServiceError: If manager not initialized
        """
        if not self._initialized:
            raise CrawlServiceError("Manager not initialized")

        logger.info(f"Starting site crawl of {url} with max {max_pages} pages")

        try:
            # Simple site crawling implementation using AutomationRouter
            # Start with the initial URL
            pages = []
            crawled_urls = set()
            urls_to_crawl = [url]

            tool_used = None

            while urls_to_crawl and len(pages) < max_pages:
                current_url = urls_to_crawl.pop(0)

                if current_url in crawled_urls:
                    continue

                crawled_urls.add(current_url)

                # Scrape the current page
                result = await self.scrape_url(current_url, preferred_provider)

                if result.get("success"):
                    pages.append(
                        {
                            "url": current_url,
                            "content": result.get("content", ""),
                            "metadata": result.get("metadata", {}),
                            "tier_used": result.get("tier_used", "unknown"),
                        }
                    )

                    if not tool_used:
                        tool_used = result.get("tier_used", "unknown")

                    # Extract links for further crawling (simple implementation)
                    # This is a basic approach - more sophisticated crawling would
                    # require dedicated crawling logic
                    links = result.get("metadata", {}).get("links", [])
                    if isinstance(links, list):
                        for link in links:
                            if isinstance(link, dict) and "url" in link:
                                link_url = link["url"]
                                if (
                                    link_url.startswith(url)
                                    and link_url not in crawled_urls
                                    and link_url not in urls_to_crawl
                                ):
                                    urls_to_crawl.append(link_url)
                else:
                    logger.warning(
                        f"Failed to scrape {current_url}: {result.get('error', 'Unknown error')}"
                    )

            return {
                "success": len(pages) > 0,
                "pages": pages,
                "total_pages": len(pages),
                "provider": tool_used or "none",
                "error": None if pages else "No pages could be crawled",
            }

        except Exception as e:
            logger.exception(f"Site crawl failed for {url}: {e}")
            return {
                "success": False,
                "error": f"Site crawl error: {e!s}",
                "pages": [],
                "total_pages": 0,
                "provider": "none",
            }

    def get_provider_info(self) -> dict[str, dict]:
        """Get information about available automation tools in 5-tier system.

        Returns:
            Tool information including tier assignments and metrics
        """
        if not self._unified_browser_manager:
            return {}

        # Get metrics from UnifiedBrowserManager
        tier_metrics = self._unified_browser_manager.get_tier_metrics()
        metrics = {tier: metrics.dict() for tier, metrics in tier_metrics.items()}

        # Enhanced tier mapping for 5-tier system
        tier_mapping = {
            "lightweight": 0,
            "crawl4ai": 1,
            "crawl4ai_enhanced": 2,
            "browser_use": 3,
            "playwright": 4,
            "firecrawl": 4,  # Also tier 4 as fallback
        }

        info = {}
        for tool_name, tool_metrics in metrics.items():
            success_rate = tool_metrics.get("success_rate", 0.0) * 100
            avg_time_ms = tool_metrics.get("avg_time", 0.0) * 1000

            info[tool_name] = {
                "type": f"{tool_name.title()}Adapter",
                "available": tool_metrics.get("available", False),
                "tier": tier_mapping.get(tool_name, -1),
                "is_preferred": False,  # AutomationRouter handles selection
                "has_api_key": tool_name == "firecrawl"
                and bool(
                    getattr(self.config, "firecrawl", None)
                    and getattr(self.config.firecrawl, "api_key", None)
                ),
                "metrics": {
                    "attempts": tool_metrics.get("total_attempts", 0),
                    "successes": tool_metrics.get("success", 0),
                    "success_rate": round(success_rate, 1),
                    "avg_time_ms": round(avg_time_ms, 0),
                },
            }

        return info

    def get_tier_metrics(self) -> dict[str, dict]:
        """Get performance metrics for each tier from UnifiedBrowserManager.

        Returns:
            Tier performance metrics for all 5 tiers
        """
        if not self._unified_browser_manager:
            return {}

        # Get tier metrics and convert to legacy format
        tier_metrics = self._unified_browser_manager.get_tier_metrics()
        return {tier: metrics.dict() for tier, metrics in tier_metrics.items()}

    async def map_url(
        self, url: str, include_subdomains: bool = False
    ) -> dict[str, object]:
        """Map a website to get list of URLs.

        Note: Only Firecrawl supports this feature.

        Args:
            url: URL to map
            include_subdomains: Include subdomains in the mapping

        Returns:
            dict[str, object]: Map result with:
                - success: Whether mapping succeeded
                - urls: List of discovered URLs
                - total: Total number of URLs found
                - error: Error message if mapping failed
        """
        if not self._initialized:
            raise CrawlServiceError("Manager not initialized")

        # URL mapping feature is not currently supported through AutomationRouter
        # This would require direct access to Firecrawl adapter
        # For now, use crawl_site as an alternative approach
        logger.warning(
            "URL mapping not supported through AutomationRouter. Use crawl_site() instead."
        )

        return {
            "success": False,
            "error": "URL mapping not supported through 5-tier AutomationRouter. Use crawl_site() method instead.",
            "urls": [],
            "total": 0,
        }
