"""
UnifiedBrowserManager: single entry for the 5-tier automation stack.
"""

from __future__ import annotations

import logging
from typing import Any, Literal

from pydantic import BaseModel, Field

from src.config import Settings
from src.services.base import BaseService
from src.services.errors import CrawlServiceError

from .firecrawl_adapter import FirecrawlAdapter, FirecrawlAdapterConfig
from .router import AutomationRouter, ScrapeRequest


logger = logging.getLogger(__name__)


class UnifiedScrapingRequest(BaseModel):
    """Request model for unified scraping."""

    url: str = Field(...)
    tier: Literal[
        "auto", "lightweight", "crawl4ai", "browser_use", "playwright", "firecrawl"
    ] = Field(default="auto")
    interaction_required: bool = Field(default=False)
    custom_actions: list[dict] | None = Field(default=None)
    timeout: int = Field(default=30000)


class UnifiedBrowserManager(BaseService):
    """Single orchestration manager for scraping and crawling."""

    def __init__(self, config: Settings) -> None:
        super().__init__()
        self._config = config
        self._router = AutomationRouter(config)
        self._fc = FirecrawlAdapter(
            FirecrawlAdapterConfig(
                api_key=getattr(getattr(config, "firecrawl", object()), "api_key", ""),
                api_url=getattr(
                    getattr(config, "firecrawl", object()), "api_url", None
                ),
            )
        )
        self._initialized = False

    async def initialize(self) -> None:
        if self._initialized:
            return
        await self._router.initialize()
        await self._fc.initialize()
        self._initialized = True
        logger.info("UnifiedBrowserManager initialized")

    async def cleanup(self) -> None:
        await self._router.cleanup()
        await self._fc.cleanup()
        self._initialized = False
        logger.info("UnifiedBrowserManager cleaned up")

    async def scrape_url(self, request: UnifiedScrapingRequest) -> dict[str, Any]:
        """Scrape a URL with intelligent tier selection."""
        if not self._initialized:
            raise CrawlServiceError("UnifiedBrowserManager not initialized")

        sr = ScrapeRequest(
            url=request.url,
            timeout_ms=request.timeout,
            tier=request.tier,  # type: ignore[arg-type]
            interaction_required=request.interaction_required,
            custom_actions=request.custom_actions,
        )
        return await self._router.scrape(sr)

    async def crawl_site(
        self,
        url: str,
        *,
        max_pages: int = 50,
        prefer: Literal["firecrawl", "crawl4ai"] = "firecrawl",
    ) -> dict[str, Any]:
        """Crawl a site via Firecrawl waiter or Crawl4AI deep-crawl."""
        if not self._initialized:
            raise CrawlServiceError("UnifiedBrowserManager not initialized")

        if prefer == "firecrawl":
            return await self._fc.crawl(url=url, limit=max_pages, formats=["markdown"])
        # Fallback: single-page hop expansion using router (conservative)
        pages: list[dict[str, Any]] = []
        frontier = [url]
        seen: set[str] = set()
        while frontier and len(pages) < max_pages:
            cur = frontier.pop(0)
            if cur in seen:
                continue
            seen.add(cur)
            res = await self.scrape_url(UnifiedScrapingRequest(url=cur))
            if res.get("success"):
                pages.append(
                    {
                        "url": cur,
                        "content": res.get("content", ""),
                        "meta": res.get("metadata", {}),
                    }
                )
        return {"success": bool(pages), "total_pages": len(pages), "pages": pages}
