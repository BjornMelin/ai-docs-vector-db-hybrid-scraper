"""Crawl4AI provider implementation."""

from __future__ import annotations

from typing import Any, cast

from crawl4ai import (  # pyright: ignore[reportMissingTypeStubs]
    AsyncWebCrawler,
    BrowserConfig,
    CacheMode,
    CrawlerRunConfig,
)
from crawl4ai.async_dispatcher import (  # pyright: ignore[reportMissingTypeStubs]
    MemoryAdaptiveDispatcher,
)
from crawl4ai.models import CrawlResult  # pyright: ignore[reportMissingTypeStubs]

from src.config.browser import Crawl4AISettings
from src.services.browser.models import BrowserResult, ProviderKind, ScrapeRequest

from .base import BrowserProvider, ProviderContext


def _cache_mode(setting: str) -> CacheMode:
    return CacheMode.ENABLED if setting == "enabled" else CacheMode.BYPASS


class Crawl4AIProvider(BrowserProvider):
    """Adapter around Crawl4AI AsyncWebCrawler."""

    kind = ProviderKind.CRAWL4AI

    def __init__(self, context: ProviderContext, settings: Crawl4AISettings) -> None:
        super().__init__(context)
        self._settings = settings
        self._crawler: AsyncWebCrawler | None = None
        self._base_run_config = CrawlerRunConfig(
            cache_mode=_cache_mode(settings.cache_mode),
            session_id=settings.session_id or "",
            verbose=settings.verbose,
        )
        self._dispatcher = MemoryAdaptiveDispatcher(
            max_session_permit=settings.concurrency
        )

    async def initialize(self) -> None:
        """Instantiate and start the Crawl4AI crawler."""
        browser_config = BrowserConfig(
            browser_type=self._settings.browser_type,
            headless=self._settings.headless,
            viewport=self._settings.viewport,
            verbose=self._settings.verbose,
        )
        self._crawler = AsyncWebCrawler(config=browser_config)
        await self._crawler.start()

    async def close(self) -> None:
        """Shutdown crawler resources."""
        if self._crawler:
            await self._crawler.close()
            self._crawler = None

    def _build_run_config(self, request: ScrapeRequest) -> CrawlerRunConfig:
        overrides: dict[str, Any] = {}
        provider_meta = request.metadata.get("crawl4ai") if request.metadata else None
        if isinstance(provider_meta, dict):
            overrides.update(provider_meta)
        if request.timeout_ms:
            overrides.setdefault("page_timeout", request.timeout_ms)
        return self._base_run_config.clone(**overrides)

    def _to_result(self, payload: CrawlResult) -> BrowserResult:
        markdown = ""
        if payload.markdown is not None:
            markdown = getattr(payload.markdown, "raw_markdown", "") or ""
            if not markdown:
                markdown = getattr(payload.markdown, "fit_markdown", "") or ""
        content = markdown or payload.cleaned_html or payload.html or ""
        md = payload.model_dump()
        return BrowserResult(
            success=payload.success,
            url=payload.redirected_url or payload.url,
            title=(md.get("metadata") or {}).get("title", ""),
            content=content,
            html=payload.html or "",
            metadata=md,
            provider=self.kind,
            links=payload.links or {},
            assets=payload.media or {},
            elapsed_ms=None,
        )

    async def scrape(self, request: ScrapeRequest) -> BrowserResult:
        """Execute a Crawl4AI scrape."""
        if self._crawler is None:  # pragma: no cover - guarded by lifecycle
            raise RuntimeError("Provider not initialized")

        run_config = self._build_run_config(request)
        result = cast(
            CrawlResult,
            await self._crawler.arun(
                url=request.url, config=run_config, dispatcher=self._dispatcher
            ),
        )
        return self._to_result(result)
