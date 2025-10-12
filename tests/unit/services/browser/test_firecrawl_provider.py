from __future__ import annotations

from typing import Any

import pytest

from src.config.browser import FirecrawlSettings
from src.services.browser.models import BrowserResult, ProviderKind, ScrapeRequest
from src.services.browser.providers.base import ProviderContext
from src.services.browser.providers.firecrawl import FirecrawlProvider


class _StubAsyncFirecrawl:
    def __init__(self, *_, **__):  # ignore api_url/api_key
        """Stub implementation of AsyncFirecrawl client."""

        self.scrape_called = False
        self.search_called = False

    async def scrape(self, url: str, **kwargs: Any) -> dict[str, Any]:
        """Simulate a scrape call."""

        self.scrape_called = True
        return {
            "url": url,
            "data": {
                "metadata": {"title": "Example", "url": url},
                "markdown": "hello",
                "html": "<p>hello</p>",
            },
        }

    async def crawl(self, url: str, **kwargs: Any) -> dict[str, Any]:
        """Simulate a crawl call."""
        return {"status": "completed", "data": [{"url": url}]}

    async def search(self, query: str, **kwargs: Any) -> dict[str, Any]:
        """Simulate a search call."""
        self.search_called = True
        return {"query": query, "web": []}

    async def start_crawl(self, url: str, **kwargs: Any) -> dict[str, Any]:
        """Simulate starting a crawl job."""
        return {"id": "job123", "url": url}

    async def get_crawl_status(self, job_id: str, **kwargs: Any) -> dict[str, Any]:
        """Simulate checking crawl job status."""
        return {"id": job_id, "status": "completed"}

    async def batch_scrape(self, urls: list[str], **kwargs: Any) -> dict[str, Any]:
        """Simulate a batch scrape call."""
        return {"status": "completed", "total": len(urls)}


@pytest.mark.asyncio
async def test_firecrawl_scrape_and_search(monkeypatch: pytest.MonkeyPatch) -> None:
    """Test FirecrawlProvider scrape and search methods with stubbed client."""

    # Patch AsyncFirecrawl class used by provider
    import src.services.browser.providers.firecrawl as mod

    monkeypatch.setattr(mod, "AsyncFirecrawl", _StubAsyncFirecrawl, raising=True)

    settings = FirecrawlSettings(api_key="fc-test")
    provider = FirecrawlProvider(ProviderContext(ProviderKind.FIRECRAWL), settings)
    await provider.initialize()
    try:
        # scrape
        result = await provider.scrape(ScrapeRequest(url="https://example.com"))
        assert isinstance(result, BrowserResult)
        assert result.success is True
        assert result.content == "hello"

        # search
        raw = await provider.search("query", limit=1)
        assert raw.get("query") == "query"
    finally:
        await provider.close()
