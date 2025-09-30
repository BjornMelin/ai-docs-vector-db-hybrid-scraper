"""Integration tests for Crawl4AI Adapter against local server."""

from __future__ import annotations

import pytest


pytest.importorskip(
    "playwright.async_api", reason="Playwright is required for integration tests"
)

from collections.abc import AsyncIterator
from typing import cast

from crawl4ai import AsyncWebCrawler, BrowserConfig, CacheMode
from crawl4ai.models import CrawlResult

from src.config.models import Crawl4AIConfig
from src.services.browser.crawl4ai_adapter import Crawl4AIAdapter
from src.services.crawling.c4a_presets import (
    base_run_config,
    best_first_run_config,
    preset_browser_config,
)


@pytest.fixture(scope="session")
def browser_config() -> BrowserConfig:
    """Provide a headless browser configuration for integration tests."""

    config = preset_browser_config()
    config.headless = True
    return config


@pytest.mark.integration
@pytest.mark.browser
@pytest.mark.asyncio
async def test_async_webcrawler_static_page(
    integration_server: str, browser_config: BrowserConfig
) -> None:
    """AsyncWebCrawler should fetch static HTML and produce markdown."""

    run_cfg = base_run_config(cache_mode=CacheMode.BYPASS)
    async with AsyncWebCrawler(
        config=browser_config
    ) as crawler:  # pragma: no cover - integration
        result = cast(
            CrawlResult,
            await crawler.arun(url=f"{integration_server}/static", config=run_cfg),
        )
    assert result.success is True
    assert result.markdown is not None
    assert "Integration Static" in result.markdown.raw_markdown


@pytest.mark.integration
@pytest.mark.browser
@pytest.mark.asyncio
async def test_async_webcrawler_executes_js(
    integration_server: str, browser_config: BrowserConfig
) -> None:
    """Crawl4AI should wait for JS-driven content when wait_for selector is provided."""

    run_cfg = base_run_config(cache_mode=CacheMode.BYPASS, wait_for="#output")
    async with AsyncWebCrawler(
        config=browser_config
    ) as crawler:  # pragma: no cover - integration
        result = cast(
            CrawlResult,
            await crawler.arun(url=f"{integration_server}/js", config=run_cfg),
        )
    assert result.success is True
    assert result.markdown is not None
    assert "Hello from API" in result.markdown.raw_markdown


@pytest.mark.integration
@pytest.mark.browser
@pytest.mark.asyncio
async def test_crawl_best_first_streaming(
    integration_server: str, browser_config: BrowserConfig
) -> None:
    """Best-first crawl should stream results for relevant links."""

    run_cfg = best_first_run_config(["topic"], stream=True)
    async with AsyncWebCrawler(
        config=browser_config
    ) as crawler:  # pragma: no cover - integration
        raw_stream = await crawler.arun(
            url=f"{integration_server}/links", config=run_cfg
        )
        stream = cast(AsyncIterator[CrawlResult], raw_stream)
        collected = [item async for item in stream]
    assert collected, "Best-first crawl returned no results"
    assert any(
        item.markdown is not None
        and item.markdown.fit_markdown is not None
        and "Topic" in item.markdown.fit_markdown
        for item in collected
    )


@pytest.mark.integration
@pytest.mark.browser
@pytest.mark.asyncio
async def test_adapter_bulk_crawl(
    integration_server: str, browser_config: BrowserConfig
) -> None:
    """Crawl4AI adapter should normalize bulk results across multiple URLs."""

    adapter = Crawl4AIAdapter(Crawl4AIConfig())
    async with AsyncWebCrawler(
        config=browser_config
    ) as crawler:  # pragma: no cover - integration
        adapter._crawler = crawler  # use shared crawler within context
        adapter._mark_initialized()
        payloads = await adapter.crawl_bulk(
            [f"{integration_server}/static", f"{integration_server}/static"],
            extraction_type="integration",
        )
        adapter._mark_uninitialized()
    assert all(payload["success"] for payload in payloads)
    assert all(
        payload["metadata"]["extraction_type"] == "integration" for payload in payloads
    )
