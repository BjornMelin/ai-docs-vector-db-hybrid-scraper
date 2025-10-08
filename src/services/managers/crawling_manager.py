"""Crawling manager facade for the unified 5-tier stack."""

from __future__ import annotations

import logging
from typing import Any, Literal

from dependency_injector.wiring import Provide, inject

from src.config import Settings
from src.infrastructure.container import ApplicationContainer
from src.services.browser.unified_manager import (
    UnifiedBrowserManager,
    UnifiedScrapingRequest,
)
from src.services.errors import CrawlServiceError


logger = logging.getLogger(__name__)


class CrawlingManager:
    """Coordinates browser automation and scraping via UnifiedBrowserManager."""

    def __init__(self) -> None:
        self._mgr: UnifiedBrowserManager | None = None
        self._initialized = False

    @inject
    async def initialize(
        self, config: Settings = Provide[ApplicationContainer.config]
    ) -> None:
        if self._initialized:
            return
        self._mgr = UnifiedBrowserManager(config)
        await self._mgr.initialize()
        self._initialized = True
        logger.info("CrawlingManager initialized (unified stack)")

    async def cleanup(self) -> None:
        if self._mgr:
            await self._mgr.cleanup()
            self._mgr = None
        self._initialized = False
        logger.info("CrawlingManager cleaned up")

    async def scrape_url(
        self,
        url: str,
        *,
        tier: Literal[
            "auto",
            "lightweight",
            "crawl4ai",
            "browser_use",
            "playwright",
            "firecrawl",
        ]
        | None = None,
    ) -> dict[str, Any]:
        if not self._initialized or self._mgr is None:
            raise CrawlServiceError("CrawlingManager not initialized")
        req = UnifiedScrapingRequest(url=url, tier=(tier or "auto"))
        return await self._mgr.scrape_url(req)

    async def crawl_site(self, url: str, *, max_pages: int = 50) -> dict[str, Any]:
        if not self._initialized or self._mgr is None:
            raise CrawlServiceError("CrawlingManager not initialized")
        return await self._mgr.crawl_site(url, max_pages=max_pages)
