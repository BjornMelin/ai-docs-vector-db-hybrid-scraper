"""Crawl manager with provider abstraction and fallback."""

import logging
from typing import Any

from ...config import UnifiedConfig
from ..errors import CrawlServiceError
from .base import CrawlProvider
from .crawl4ai_provider import Crawl4AIProvider
from .firecrawl_provider import FirecrawlProvider

logger = logging.getLogger(__name__)


class CrawlManager:
    """Manager for crawling with multiple providers."""

    def __init__(self, config: UnifiedConfig):
        """Initialize crawl manager.

        Args:
            config: Unified configuration
        """
        self.config = config
        self.providers: dict[str, CrawlProvider] = {}
        self._initialized = False

    async def initialize(self) -> None:
        """Initialize available providers."""
        if self._initialized:
            return

        # Initialize Firecrawl if API key available
        if self.config.firecrawl.api_key:
            try:
                provider = FirecrawlProvider(
                    api_key=self.config.firecrawl.api_key
                )
                await provider.initialize()
                self.providers["firecrawl"] = provider
                logger.info("Initialized Firecrawl provider")
            except Exception as e:
                logger.warning(f"Failed to initialize Firecrawl provider: {e}")

        # Always initialize Crawl4AI as fallback
        try:
            provider = Crawl4AIProvider()
            await provider.initialize()
            self.providers["crawl4ai"] = provider
            logger.info("Initialized Crawl4AI provider")
        except Exception as e:
            logger.warning(f"Failed to initialize Crawl4AI provider: {e}")

        if not self.providers:
            raise CrawlServiceError("No crawling providers available")

        self._initialized = True
        logger.info(f"Crawl manager initialized with {len(self.providers)} providers")

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
    ) -> dict[str, Any]:
        """Scrape URL with automatic provider fallback.

        Args:
            url: URL to scrape
            formats: Output formats (["markdown"], ["html"], ["text"])
            preferred_provider: Provider to try first ("firecrawl" or "crawl4ai")

        Returns:
            Scraping result with success status, content, metadata, and provider

        Raises:
            CrawlServiceError: If manager not initialized
        """
        if not self._initialized:
            raise CrawlServiceError("Manager not initialized")

        # Determine provider order
        if preferred_provider:
            if preferred_provider not in self.providers:
                logger.warning(
                    f"Preferred provider '{preferred_provider}' not available"
                )
                provider_order = list(self.providers.keys())
            else:
                # Preferred provider first, then others
                provider_order = [preferred_provider] + [
                    p for p in self.providers if p != preferred_provider
                ]
        else:
            # Use configured preference
            provider_order = []
            if self.config.crawling.preferred_provider in self.providers:
                provider_order.append(self.config.crawling.preferred_provider)
            provider_order.extend(p for p in self.providers if p not in provider_order)

        # Try providers in order
        last_error = None
        for provider_name in provider_order:
            provider = self.providers[provider_name]
            try:
                logger.info(f"Attempting to scrape {url} with {provider_name}")
                result = await provider.scrape_url(url, formats)

                if result.get("success", False):
                    result["provider"] = provider_name
                    return result
                else:
                    last_error = result.get("error", "Unknown error")
                    logger.warning(f"Provider {provider_name} failed: {last_error}")

            except Exception as e:
                last_error = str(e)
                logger.error(f"Error with {provider_name}: {e}")

        # All providers failed
        return {
            "success": False,
            "error": f"All providers failed. Last error: {last_error}",
            "content": "",
            "metadata": {},
            "url": url,
        }

    async def crawl_site(
        self,
        url: str,
        max_pages: int = 50,
        formats: list[str] | None = None,
        preferred_provider: str | None = None,
    ) -> dict[str, Any]:
        """Crawl entire website from starting URL.

        Args:
            url: Starting URL for crawl
            max_pages: Maximum pages to crawl (default: 50)
            formats: Output formats for each page
            preferred_provider: Provider to use (prefers Firecrawl for sites)

        Returns:
            Crawl results with pages list, total count, and provider used

        Raises:
            CrawlServiceError: If manager not initialized
        """
        if not self._initialized:
            raise CrawlServiceError("Manager not initialized")

        # For site crawling, prefer Firecrawl if available
        if not preferred_provider and "firecrawl" in self.providers:
            preferred_provider = "firecrawl"

        # Select provider
        provider_name = preferred_provider or self.config.crawling.preferred_provider
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
            Provider information
        """
        info = {}
        for name, provider in self.providers.items():
            info[name] = {
                "type": provider.__class__.__name__,
                "available": True,
                "is_preferred": name == self.config.crawling.preferred_provider,
                "has_api_key": name == "firecrawl"
                and bool(self.config.crawling.firecrawl_api_key),
            }
        return info

    async def map_url(
        self, url: str, include_subdomains: bool = False
    ) -> dict[str, Any]:
        """Map a website to get list of URLs.

        Note: Only Firecrawl supports this feature.

        Args:
            url: URL to map
            include_subdomains: Include subdomains

        Returns:
            Map result with URLs
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
