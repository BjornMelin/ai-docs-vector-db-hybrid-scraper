"""Unified browser manager built on the BrowserRouter."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Any

from src.config.browser import BrowserAutomationConfig
from src.config.loader import Settings
from src.services.browser.errors import BrowserRouterError
from src.services.browser.models import BrowserResult, ProviderKind, ScrapeRequest
from src.services.browser.providers import (
    BrowserUseProvider,
    Crawl4AIProvider,
    FirecrawlProvider,
    LightweightProvider,
    PlaywrightProvider,
)
from src.services.browser.providers.base import ProviderContext
from src.services.browser.router import BrowserRouter


class UnifiedBrowserManager:
    """Facade exposing high-level crawling operations."""

    def __init__(self, settings: Settings):
        """Initialize the manager with application settings.

        Args:
            settings: Fully-populated application settings.
        """
        browser_cfg: BrowserAutomationConfig = settings.browser
        self._router = BrowserRouter(
            settings=browser_cfg.router,
            lightweight=LightweightProvider(
                ProviderContext(ProviderKind.LIGHTWEIGHT), browser_cfg.lightweight
            ),
            crawl4ai=Crawl4AIProvider(
                ProviderContext(ProviderKind.CRAWL4AI), browser_cfg.crawl4ai
            ),
            playwright=PlaywrightProvider(
                ProviderContext(ProviderKind.PLAYWRIGHT), browser_cfg.playwright
            ),
            browser_use=BrowserUseProvider(
                ProviderContext(ProviderKind.BROWSER_USE), browser_cfg.browser_use
            ),
            firecrawl=FirecrawlProvider(
                ProviderContext(ProviderKind.FIRECRAWL), browser_cfg.firecrawl
            ),
        )

    async def initialize(self) -> None:
        """Initialize all underlying providers."""
        await self._router.initialize()

    async def cleanup(self) -> None:
        """Release provider resources."""
        await self._router.cleanup()

    def is_initialized(self) -> bool:
        """Return True when providers have been initialized."""
        return self._router.is_initialized()

    async def scrape_url(
        self,
        url: str,
        preferred_provider: str | None = None,
        *,
        metadata: Mapping[str, Any] | None = None,
        require_interaction: bool = False,
        timeout_ms: int | None = None,
        actions: Sequence[Mapping[str, Any]] | None = None,
        instructions: Sequence[Mapping[str, Any]] | None = None,
    ) -> BrowserResult:
        # pylint: disable=too-many-arguments
        """Scrape a single URL using the router.

        Args:
            url: Target URL to fetch.
            preferred_provider: Optional provider identifier to force.
            metadata: Optional metadata forwarded to providers.
            require_interaction: Whether scripted interaction is required.
            timeout_ms: Optional deadline override.
            actions: Optional scripted actions forwarded to interactive providers.
            instructions: Optional instruction payload for agentic providers.

        Returns:
            BrowserResult produced by the winning provider.
        """
        provider = None
        if preferred_provider:
            try:
                provider = ProviderKind(preferred_provider)
            except ValueError as exc:
                raise BrowserRouterError(
                    f"Unsupported provider: {preferred_provider}",
                    attempted_providers=[],
                ) from exc
        request = ScrapeRequest(
            url=url,
            provider=provider,
            metadata=metadata or {},
            require_interaction=require_interaction,
            timeout_ms=timeout_ms,
            actions=actions,
            instructions=instructions,
        )
        return await self._router.scrape(request)

    async def crawl_site(
        self,
        url: str,
        max_pages: int | None = None,
        preferred_provider: str | None = None,
    ) -> dict[str, Any]:
        """Execute a site-wide crawl.

        Args:
            url: Seed URL for the crawl.
            max_pages: Optional page limit.
            preferred_provider: Optional provider identifier.

        Returns:
            Crawl job payload produced by Firecrawl.
        """
        provider = preferred_provider or ProviderKind.FIRECRAWL.value
        if provider != ProviderKind.FIRECRAWL.value:
            raise BrowserRouterError(
                "Site crawling is only supported by the firecrawl provider",
                attempted_providers=[provider],
            )
        firecrawl = self._router.get_provider(ProviderKind.FIRECRAWL)
        if firecrawl is None or not isinstance(firecrawl, FirecrawlProvider):
            raise BrowserRouterError(
                "Firecrawl provider is unavailable",
                attempted_providers=[provider],
            )
        return await firecrawl.crawl_site(url, limit=max_pages)


__all__ = ["UnifiedBrowserManager"]
