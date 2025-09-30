"""Unit tests for the Crawl4AI adapter thin wrapper.

Verifies initialization, scraping, bulk crawling, health checks, and metrics collection.
"""

from __future__ import annotations

from collections.abc import AsyncGenerator
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock

import pytest

from src.config.models import Crawl4AIConfig
from src.services.browser.crawl4ai_adapter import Crawl4AIAdapter


def _fake_result(url: str = "https://example.com") -> SimpleNamespace:
    """Create a mock Crawl4AI result object."""

    markdown = SimpleNamespace(raw_markdown="# Title", fit_markdown="# Title")
    return SimpleNamespace(
        success=True,
        url=url,
        html="<html></html>",
        markdown=markdown,
        metadata={"title": "Title"},
        extracted_content=None,
        links={},
        media={},
        error_message=None,
    )


@pytest.fixture
async def initialized_adapter(
    monkeypatch: pytest.MonkeyPatch,
) -> AsyncGenerator[tuple[Crawl4AIAdapter, AsyncMock], None]:
    """Provide an initialized adapter with mocked crawler."""

    crawler_mock = AsyncMock()
    crawler_mock.start = AsyncMock()
    crawler_mock.close = AsyncMock()
    async_webcrawler_cls = MagicMock(return_value=crawler_mock)
    monkeypatch.setattr(
        "src.services.browser.crawl4ai_adapter.AsyncWebCrawler",
        async_webcrawler_cls,
    )

    adapter = Crawl4AIAdapter(Crawl4AIConfig())
    await adapter.initialize()
    try:
        yield adapter, crawler_mock
    finally:
        await adapter.cleanup()


@pytest.mark.asyncio
async def test_initialize_starts_crawler(monkeypatch: pytest.MonkeyPatch) -> None:
    """Verify adapter initialization starts the underlying crawler."""

    crawler_mock = AsyncMock()
    crawler_mock.start = AsyncMock()
    monkeypatch.setattr(
        "src.services.browser.crawl4ai_adapter.AsyncWebCrawler",
        MagicMock(return_value=crawler_mock),
    )
    adapter = Crawl4AIAdapter(Crawl4AIConfig())
    await adapter.initialize()
    crawler_mock.start.assert_awaited_once()


@pytest.mark.asyncio
async def test_scrape_success(
    monkeypatch: pytest.MonkeyPatch,
    initialized_adapter: tuple[Crawl4AIAdapter, AsyncMock],
) -> None:
    """Verify successful scrape returns expected payload structure."""

    adapter, _ = initialized_adapter
    monkeypatch.setattr(
        "src.services.browser.crawl4ai_adapter.crawl_page",
        AsyncMock(return_value=_fake_result()),
    )
    payload = await adapter.scrape("https://example.com")
    assert payload["success"] is True
    assert payload["metadata"]["extraction_type"] == "crawl4ai"
    assert payload["metadata"]["fit_markdown"] == "# Title"


@pytest.mark.asyncio
async def test_scrape_failure(
    monkeypatch: pytest.MonkeyPatch,
    initialized_adapter: tuple[Crawl4AIAdapter, AsyncMock],
) -> None:
    """Verify scrape handles exceptions and returns error payload."""

    adapter, _ = initialized_adapter
    monkeypatch.setattr(
        "src.services.browser.crawl4ai_adapter.crawl_page",
        AsyncMock(side_effect=RuntimeError("boom")),
    )
    payload = await adapter.scrape("https://example.com")
    assert payload["success"] is False
    assert "boom" in payload["error"]


@pytest.mark.asyncio
async def test_crawl_bulk(
    monkeypatch: pytest.MonkeyPatch,
    initialized_adapter: tuple[Crawl4AIAdapter, AsyncMock],
) -> None:
    """Verify bulk crawling returns payloads for multiple URLs."""

    adapter, _ = initialized_adapter
    monkeypatch.setattr(
        "src.services.browser.crawl4ai_adapter.crawl_page",
        AsyncMock(
            return_value=[
                _fake_result("https://example.com/a"),
                _fake_result("https://example.com/b"),
            ]
        ),
    )
    payloads = await adapter.crawl_bulk(
        ["https://example.com/a", "https://example.com/b"]
    )
    assert len(payloads) == 2
    assert all(item["success"] for item in payloads)


@pytest.mark.asyncio
async def test_health_check(
    monkeypatch: pytest.MonkeyPatch,
    initialized_adapter: tuple[Crawl4AIAdapter, AsyncMock],
) -> None:
    """Verify health check reports adapter status correctly."""

    adapter, _ = initialized_adapter
    monkeypatch.setattr(adapter, "scrape", AsyncMock(return_value={"success": True}))
    status = await adapter.health_check()
    assert status["healthy"] is True


@pytest.mark.asyncio
async def test_get_performance_metrics(
    monkeypatch: pytest.MonkeyPatch,
    initialized_adapter: tuple[Crawl4AIAdapter, AsyncMock],
) -> None:
    """Verify performance metrics track request counts accurately."""

    adapter, _ = initialized_adapter
    monkeypatch.setattr(
        "src.services.browser.crawl4ai_adapter.crawl_page",
        AsyncMock(return_value=_fake_result()),
    )
    await adapter.scrape("https://example.com")
    metrics = await adapter.get_performance_metrics()
    assert metrics["total_requests"] == 1
    assert metrics["successful_requests"] == 1
