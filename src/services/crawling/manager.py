"""Crawl manager with provider abstraction and fallback."""

import logging
import time

from ...config import UnifiedConfig
from ..errors import CrawlServiceError
from .base import CrawlProvider
from .crawl4ai_provider import Crawl4AIProvider
from .firecrawl_provider import FirecrawlProvider
from .lightweight_scraper import LightweightScraper

logger = logging.getLogger(__name__)


class CrawlManager:
    """Manager for crawling with multiple providers and tier-based optimization."""

    def __init__(self, config: UnifiedConfig, rate_limiter: object = None):
        """Initialize crawl manager.

        Args:
            config: Unified configuration with crawl provider settings
            rate_limiter: Optional RateLimitManager instance for rate limiting
        """
        self.config = config
        self.providers: dict[str, CrawlProvider] = {}
        self._initialized = False
        self.rate_limiter = rate_limiter

        # Tier metrics tracking
        self._tier_metrics = {
            "lightweight": {"attempts": 0, "successes": 0, "total_time": 0.0},
            "crawl4ai": {"attempts": 0, "successes": 0, "total_time": 0.0},
            "firecrawl": {"attempts": 0, "successes": 0, "total_time": 0.0},
        }

    async def initialize(self) -> None:
        """Initialize available providers in tier order.

        Initializes:
        - Tier 0: Lightweight HTTP scraper (httpx + BeautifulSoup)
        - Tier 1: Crawl4AI as primary browser-based provider
        - Tier 2: Firecrawl as fallback if API key available

        At least one provider must initialize successfully.

        Raises:
            CrawlServiceError: If no providers can be initialized
        """
        if self._initialized:
            return

        # Initialize Tier 0: Lightweight HTTP scraper
        if self.config.lightweight_scraper.enable_lightweight_tier:
            try:
                provider = LightweightScraper(
                    config=self.config.lightweight_scraper,
                    rate_limiter=self.rate_limiter,
                )
                await provider.initialize()
                self.providers["lightweight"] = provider
                logger.info("Initialized Lightweight HTTP scraper as Tier 0")
            except Exception as e:
                logger.warning(f"Failed to initialize Lightweight scraper: {e}")

        # Initialize Tier 1: Crawl4AI as primary browser provider
        try:
            crawl4ai_config = self.config.crawl4ai
            provider = Crawl4AIProvider(
                config=crawl4ai_config,
                rate_limiter=self.rate_limiter,
            )
            await provider.initialize()
            self.providers["crawl4ai"] = provider
            logger.info("Initialized Crawl4AI provider as Tier 1")
        except Exception as e:
            logger.warning(f"Failed to initialize Crawl4AI provider: {e}")

        # Initialize Tier 2: Firecrawl if API key available
        if self.config.firecrawl.api_key:
            try:
                provider = FirecrawlProvider(
                    config=self.config.firecrawl,
                    rate_limiter=self.rate_limiter,
                )
                await provider.initialize()
                self.providers["firecrawl"] = provider
                logger.info("Initialized Firecrawl provider as Tier 2")
            except Exception as e:
                logger.warning(f"Failed to initialize Firecrawl provider: {e}")

        if not self.providers:
            raise CrawlServiceError("No crawling providers available")

        self._initialized = True
        logger.info(
            f"Crawl manager initialized with {len(self.providers)} providers: "
            f"{list(self.providers.keys())}"
        )

    async def cleanup(self) -> None:
        """Cleanup all providers."""
        for name, provider in self.providers.items():
            try:
                await provider.cleanup()
                logger.info(f"Cleaned up {name} provider")
            except Exception as e:
                logger.error(f"Error cleaning up {name} provider: {e}")

        self.providers.clear()
        self._initialized = False

    async def scrape_url(
        self,
        url: str,
        formats: list[str] | None = None,
        preferred_provider: str | None = None,
    ) -> dict[str, object]:
        """Scrape URL with intelligent tier-based provider selection.

        Uses a three-tier approach:
        - Tier 0: Lightweight HTTP (5-10x faster for simple pages)
        - Tier 1: Crawl4AI (browser-based with optimizations)
        - Tier 2: Firecrawl (fallback with API)

        Args:
            url: URL to scrape
            formats: Output formats (["markdown"], ["html"], ["text"])
            preferred_provider: Provider to try first (overrides tier selection)

        Returns:
            dict[str, object]: Scraping result with:
                - success: Whether scraping succeeded
                - content: Scraped content in requested formats
                - metadata: Additional information (title, description, etc.)
                - provider: Name of provider that succeeded
                - tier: Tier level used (0, 1, or 2)
                - performance_ms: Time taken in milliseconds
                - error: Error message if all providers failed

        Raises:
            CrawlServiceError: If manager not initialized
        """
        if not self._initialized:
            raise CrawlServiceError("Manager not initialized")

        start_time = time.time()

        # If preferred provider specified, use it first
        if preferred_provider and preferred_provider in self.providers:
            result = await self._try_provider(
                preferred_provider, url, formats, start_time
            )
            if result["success"]:
                return result

        # Try Tier 0: Lightweight scraper first (if available and URL is suitable)
        if "lightweight" in self.providers and not preferred_provider:
            lightweight_provider = self.providers["lightweight"]

            # Check if URL can be handled by lightweight tier
            if await lightweight_provider.can_handle(url):
                logger.info(f"URL {url} eligible for lightweight tier")
                result = await self._try_provider(
                    "lightweight", url, formats, start_time
                )

                if result["success"]:
                    result["tier"] = 0
                    return result
                elif result.get("should_escalate", True):
                    logger.info(f"Escalating from lightweight tier for {url}")
                else:
                    # Don't escalate if explicitly told not to
                    return result

        # Determine remaining provider order (Tier 1 and 2)
        provider_order = []

        # Add Crawl4AI as Tier 1
        if "crawl4ai" in self.providers:
            provider_order.append("crawl4ai")

        # Add Firecrawl as Tier 2
        if "firecrawl" in self.providers:
            provider_order.append("firecrawl")

        # Try remaining providers in tier order
        last_error = None
        for idx, provider_name in enumerate(provider_order):
            result = await self._try_provider(provider_name, url, formats, start_time)

            if result["success"]:
                result["tier"] = idx + 1  # Tier 1 or 2
                return result
            else:
                last_error = result.get("error", "Unknown error")
                if not result.get("should_escalate", True):
                    return result

        # All providers failed
        return {
            "success": False,
            "error": f"All tiers failed. Last error: {last_error}",
            "content": "",
            "metadata": {},
            "url": url,
            "performance_ms": int((time.time() - start_time) * 1000),
        }

    async def _try_provider(
        self,
        provider_name: str,
        url: str,
        formats: list[str] | None,
        start_time: float,
    ) -> dict[str, object]:
        """Try a specific provider and track metrics.

        Args:
            provider_name: Name of the provider to try
            url: URL to scrape
            formats: Output formats
            start_time: Start time for performance tracking

        Returns:
            Result dictionary with success status and content
        """
        provider = self.providers[provider_name]
        provider_start = time.time()

        # Update attempt metrics
        self._tier_metrics[provider_name]["attempts"] += 1

        try:
            logger.info(f"Attempting to scrape {url} with {provider_name}")
            result = await provider.scrape_url(url, formats)

            # Track timing
            elapsed = time.time() - provider_start
            self._tier_metrics[provider_name]["total_time"] += elapsed

            if result.get("success", False):
                self._tier_metrics[provider_name]["successes"] += 1
                result["provider"] = provider_name
                result["performance_ms"] = int((time.time() - start_time) * 1000)
                logger.info(
                    f"Successfully scraped {url} with {provider_name} "
                    f"in {result['performance_ms']}ms"
                )
                return result
            else:
                logger.warning(
                    f"Provider {provider_name} failed: {result.get('error', 'Unknown')}"
                )
                return result

        except Exception as e:
            logger.error(f"Error with {provider_name}: {e}")
            return {
                "success": False,
                "error": str(e),
                "should_escalate": True,
            }

    async def crawl_site(
        self,
        url: str,
        max_pages: int = 50,
        formats: list[str] | None = None,
        preferred_provider: str | None = None,
    ) -> dict[str, object]:
        """Crawl entire website from starting URL.

        Args:
            url: Starting URL for crawl
            max_pages: Maximum pages to crawl (default: 50)
            formats: Output formats for each page
            preferred_provider: Provider to use (prefers Crawl4AI for sites)

        Returns:
            dict[str, object]: Crawl results with:
                - success: Whether crawl succeeded
                - pages: List of crawled page data
                - total_pages: Number of pages crawled
                - provider: Name of provider used
                - error: Error message if crawl failed

        Raises:
            CrawlServiceError: If manager not initialized
        """
        if not self._initialized:
            raise CrawlServiceError("Manager not initialized")

        # For site crawling, prefer Crawl4AI for better performance
        if not preferred_provider and "crawl4ai" in self.providers:
            preferred_provider = "crawl4ai"

        # Select provider
        provider_name = preferred_provider or self.config.crawl_provider
        if provider_name not in self.providers:
            provider_name = next(iter(self.providers.keys()))

        provider = self.providers[provider_name]
        logger.info(f"Crawling {url} with {provider_name}")

        try:
            result = await provider.crawl_site(url, max_pages, formats)
            result["provider"] = provider_name
            return result
        except Exception as e:
            logger.error(f"Failed to crawl {url} with {provider_name}: {e}")

            # Try fallback if available
            fallback_providers = [p for p in self.providers if p != provider_name]

            if fallback_providers:
                fallback_name = fallback_providers[0]
                logger.info(f"Trying fallback provider {fallback_name}")

                try:
                    provider = self.providers[fallback_name]
                    result = await provider.crawl_site(url, max_pages, formats)
                    result["provider"] = fallback_name
                    return result
                except Exception as fallback_error:
                    logger.error(
                        f"Fallback provider {fallback_name} also failed: "
                        f"{fallback_error}"
                    )

            return {
                "success": False,
                "error": str(e),
                "pages": [],
                "total": 0,
                "provider": provider_name,
            }

    def get_provider_info(self) -> dict[str, dict]:
        """Get information about available providers.

        Returns:
            Provider information including tier assignments
        """
        info = {}
        tier_mapping = {"lightweight": 0, "crawl4ai": 1, "firecrawl": 2}

        for name, provider in self.providers.items():
            metrics = self._tier_metrics.get(name, {})
            success_rate = 0.0
            avg_time_ms = 0.0

            if metrics.get("attempts", 0) > 0:
                success_rate = metrics["successes"] / metrics["attempts"] * 100
                avg_time_ms = metrics["total_time"] / metrics["attempts"] * 1000

            info[name] = {
                "type": provider.__class__.__name__,
                "available": True,
                "tier": tier_mapping.get(name, -1),
                "is_preferred": name == self.config.crawl_provider,
                "has_api_key": name == "firecrawl"
                and bool(self.config.firecrawl.api_key),
                "metrics": {
                    "attempts": metrics.get("attempts", 0),
                    "successes": metrics.get("successes", 0),
                    "success_rate": round(success_rate, 1),
                    "avg_time_ms": round(avg_time_ms, 0),
                },
            }
        return info

    def get_tier_metrics(self) -> dict[str, dict]:
        """Get performance metrics for each tier.

        Returns:
            Tier performance metrics
        """
        return self._tier_metrics.copy()

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

        if "firecrawl" not in self.providers:
            return {
                "success": False,
                "error": "URL mapping requires Firecrawl provider",
                "urls": [],
                "total": 0,
            }

        provider = self.providers["firecrawl"]
        return await provider.map_url(url, include_subdomains)
