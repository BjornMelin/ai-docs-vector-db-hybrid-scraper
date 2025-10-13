"""Unit tests for Firecrawl provider timeout and metadata handling."""
# pylint: disable=import-error

from __future__ import annotations

from typing import Any, cast

import pytest  # pylint: disable=import-error

from src.config.browser import FirecrawlSettings
from src.services.browser.errors import BrowserProviderError
from src.services.browser.models import BrowserResult, ProviderKind, ScrapeRequest
from src.services.browser.providers import firecrawl as firecrawl_module
from src.services.browser.providers.base import ProviderContext
from src.services.browser.providers.firecrawl import FirecrawlProvider


async def _passthrough_retry(**kwargs: Any) -> Any:
    """Invoke the supplied callable immediately, emulating execute_with_retry."""

    func = kwargs["func"]
    return await func()


class _StubAsyncFirecrawl:
    """Test double simulating AsyncFirecrawl while ignoring connection kwargs."""

    def __init__(self, *_, **__):  # ignore api_url/api_key
        """Stub implementation of AsyncFirecrawl client."""

        self.scrape_called = False
        self.search_called = False
        self.last_kwargs: dict[str, Any] | None = None

    async def scrape(self, url: str, **kwargs: Any) -> dict[str, Any]:
        """Simulate a scrape call."""

        self.scrape_called = True
        self.last_kwargs = kwargs
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
    monkeypatch.setattr(
        firecrawl_module, "AsyncFirecrawlApp", _StubAsyncFirecrawl, raising=True
    )
    calls: list[dict[str, Any]] = []

    async def fake_execute_with_retry(**kwargs: Any) -> Any:
        calls.append(kwargs)
        return await _passthrough_retry(**kwargs)

    monkeypatch.setattr(
        firecrawl_module, "execute_with_retry", fake_execute_with_retry, raising=True
    )

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

    assert [call["operation"] for call in calls] == ["scrape", "search"]


@pytest.mark.asyncio
async def test_firecrawl_scrape_respects_request_timeout(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Provider should forward request timeout in seconds to Firecrawl SDK."""

    monkeypatch.setattr(
        firecrawl_module, "AsyncFirecrawlApp", _StubAsyncFirecrawl, raising=True
    )
    monkeypatch.setattr(
        firecrawl_module, "execute_with_retry", _passthrough_retry, raising=True
    )

    settings = FirecrawlSettings(api_key="fc-test")
    provider = FirecrawlProvider(ProviderContext(ProviderKind.FIRECRAWL), settings)
    await provider.initialize()
    try:
        request = ScrapeRequest(url="https://example.com", timeout_ms=5000)
        result = await provider.scrape(request)
        assert result.success is True

        stub = cast(_StubAsyncFirecrawl, provider._client)
        # pylint: disable=no-member
        assert stub.last_kwargs is not None
        assert stub.last_kwargs.get("timeout") == pytest.approx(5.0)
    finally:
        await provider.close()


@pytest.mark.asyncio
async def test_firecrawl_scrape_prefers_smaller_metadata_timeout(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Metadata timeout should be clamped by router request budget."""

    monkeypatch.setattr(
        firecrawl_module, "AsyncFirecrawlApp", _StubAsyncFirecrawl, raising=True
    )
    monkeypatch.setattr(
        firecrawl_module, "execute_with_retry", _passthrough_retry, raising=True
    )

    settings = FirecrawlSettings(api_key="fc-test")
    provider = FirecrawlProvider(ProviderContext(ProviderKind.FIRECRAWL), settings)
    await provider.initialize()
    try:
        request = ScrapeRequest(
            url="https://example.com",
            timeout_ms=9000,
            metadata={"firecrawl": {"timeout": 3}},
        )
        await provider.scrape(request)

        stub = cast(_StubAsyncFirecrawl, provider._client)
        # pylint: disable=no-member
        assert stub.last_kwargs is not None
        # min(request timeout=9s, metadata timeout=3s) == 3s
        assert stub.last_kwargs.get("timeout") == pytest.approx(3.0)
    finally:
        await provider.close()


@pytest.mark.asyncio
async def test_firecrawl_scrape_rejects_invalid_metadata_timeout(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Non-numeric metadata timeout should raise a provider error."""

    monkeypatch.setattr(
        firecrawl_module, "AsyncFirecrawlApp", _StubAsyncFirecrawl, raising=True
    )
    monkeypatch.setattr(
        firecrawl_module, "execute_with_retry", _passthrough_retry, raising=True
    )

    settings = FirecrawlSettings(api_key="fc-test")
    provider = FirecrawlProvider(ProviderContext(ProviderKind.FIRECRAWL), settings)
    await provider.initialize()
    try:
        with pytest.raises(BrowserProviderError):
            await provider.scrape(
                ScrapeRequest(
                    url="https://example.com",
                    metadata={"firecrawl": {"timeout": "fast"}},
                )
            )
    finally:
        await provider.close()
