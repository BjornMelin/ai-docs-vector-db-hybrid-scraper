"""Unit tests for c4a_provider entry points."""

from __future__ import annotations

from collections.abc import AsyncIterator
from types import SimpleNamespace
from typing import cast
from unittest.mock import AsyncMock

import pytest

from src.services.crawling.c4a_presets import (
    base_run_config,
    best_first_run_config,
    preset_browser_config,
)
from src.services.crawling.c4a_provider import (
    crawl_best_first,
    crawl_deep_bfs,
    crawl_page,
)


def _fake_result(
    url: str = "https://example.com", success: bool = True
) -> SimpleNamespace:
    """Create a fake CrawlResult for testing."""

    markdown = SimpleNamespace(raw_markdown="# Title", fit_markdown="# Title")
    return SimpleNamespace(
        success=success,
        url=url,
        html="<html></html>",
        markdown=markdown,
        metadata={"title": "Title"},
        extracted_content=None,
        links={},
        media={},
        error_message=None,
    )


@pytest.mark.asyncio
async def test_crawl_page_single_url_uses_arun() -> None:
    """Simulate crawling a single URL."""

    crawler = AsyncMock()
    crawler.arun.return_value = _fake_result()
    browser_cfg = preset_browser_config()
    run_cfg = base_run_config()

    result = await crawl_page(
        "https://example.com", run_cfg, browser_cfg, crawler=crawler
    )

    crawler.arun.assert_awaited_once()
    assert isinstance(result, SimpleNamespace)
    assert result.success is True


@pytest.mark.asyncio
async def test_crawl_page_multiple_urls_calls_arun_many() -> None:
    crawler = AsyncMock()
    """Simulate crawling multiple URLs."""

    crawler.arun_many.return_value = [
        _fake_result("https://example.com/a"),
        _fake_result("https://example.com/b"),
    ]
    browser_cfg = preset_browser_config()
    run_cfg = base_run_config()

    results = await crawl_page(
        ["https://example.com/a", "https://example.com/b"],
        run_cfg,
        browser_cfg,
        crawler=crawler,
    )

    crawler.arun_many.assert_awaited_once()
    assert isinstance(results, list)
    assert len(results) == 2


@pytest.mark.asyncio
async def test_crawl_page_with_dispatcher_uses_run_urls() -> None:
    crawler = AsyncMock()
    dispatcher = AsyncMock()
    crawler.arun_many.return_value = [_fake_result("https://example.com/1")]
    browser_cfg = preset_browser_config()
    run_cfg = base_run_config()

    results = await crawl_page(
        ["https://example.com/1"],
        run_cfg,
        browser_cfg,
        crawler=crawler,
        dispatcher=dispatcher,
    )

    crawler.arun_many.assert_awaited_once()
    assert isinstance(results, list)
    assert len(results) == 1
    assert results[0].url == "https://example.com/1"


@pytest.mark.asyncio
async def test_crawl_deep_bfs_applies_strategy_when_missing() -> None:
    """Simulate crawling a deep URL with BFS strategy."""

    crawler = AsyncMock()
    crawler.arun.return_value = _fake_result("https://example.com/deep")
    browser_cfg = preset_browser_config()
    run_cfg = base_run_config()

    results = await crawl_deep_bfs(
        "https://example.com", 2, run_cfg, browser_cfg, crawler=crawler
    )

    crawler.arun.assert_awaited_once()
    assert isinstance(results, list)
    assert len(results) == 1


@pytest.mark.asyncio
async def test_crawl_best_first_returns_list_when_not_streaming() -> None:
    """Simulate crawling a URL with Best-First Search strategy."""

    crawler = AsyncMock()
    run_cfg = best_first_run_config(["crawler"], stream=False)
    crawler.arun.return_value = [_fake_result("https://example.com/best")]
    browser_cfg = preset_browser_config()

    results = await crawl_best_first(
        "https://example.com", ["crawler"], run_cfg, browser_cfg, crawler=crawler
    )

    crawler.arun.assert_awaited_once()
    assert isinstance(results, list)
    assert results[0].url == "https://example.com/best"


@pytest.mark.asyncio
async def test_crawl_best_first_streaming_returns_async_iterator() -> None:
    """Simulate crawling a URL with Best-First Search strategy in streaming mode."""

    async def iterator() -> AsyncIterator[SimpleNamespace]:
        yield _fake_result("https://example.com/stream")

    crawler = AsyncMock()
    crawler.arun.return_value = iterator()
    browser_cfg = preset_browser_config()
    run_cfg = best_first_run_config(["crawler"], stream=True)

    stream = await crawl_best_first(
        "https://example.com", ["crawler"], run_cfg, browser_cfg, crawler=crawler
    )

    assert isinstance(stream, AsyncIterator)
    items = [item async for item in stream]
    assert len(items) == 1
    assert items[0].url == "https://example.com/stream"


@pytest.mark.asyncio
async def test_crawl_page_supports_existing_session() -> None:
    """Simulate crawling multiple pages in the same session."""

    crawler = AsyncMock()
    crawler.arun.side_effect = [
        _fake_result(),
        _fake_result("https://example.com/next"),
    ]
    browser_cfg = preset_browser_config()
    run_cfg = base_run_config(session_id="session-1")

    first = await crawl_page(
        "https://example.com", run_cfg, browser_cfg, crawler=crawler
    )
    second = await crawl_page(
        "https://example.com/next", run_cfg, browser_cfg, crawler=crawler
    )

    first_ns = cast(SimpleNamespace, first)
    second_ns = cast(SimpleNamespace, second)
    assert first_ns.url == "https://example.com"
    assert second_ns.url == "https://example.com/next"
    assert crawler.arun.await_count == 2
