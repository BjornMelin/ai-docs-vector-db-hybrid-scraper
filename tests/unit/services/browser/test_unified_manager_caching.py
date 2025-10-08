"""Error-handling and reuse tests for UnifiedBrowserManager (no caching layer)."""

from __future__ import annotations

from typing import Any

import pytest

import src.services.browser.unified_manager as unified_manager_module
from src.config import Environment, Settings
from src.services.browser.unified_manager import (
    UnifiedBrowserManager,
    UnifiedScrapingRequest,
)
from src.services.errors import CrawlServiceError


@pytest.fixture
def manager_with_toggleable_stubs(
    monkeypatch: pytest.MonkeyPatch,
) -> tuple[UnifiedBrowserManager, dict[str, Any]]:
    """Create a UnifiedBrowserManager whose dependencies can simulate failures."""

    stubs: dict[str, Any] = {}

    class RouterStub:
        def __init__(self, *_args: Any, **_kwargs: Any) -> None:
            stubs["router"] = self
            self.initialize_calls = 0
            self.cleanup_calls = 0
            self.scrape_calls: list[Any] = []
            self.result: dict[str, Any] = {
                "success": True,
                "content": "fresh-content",
                "metadata": {},
            }
            self.side_effect: Exception | None = None

        async def initialize(self) -> None:
            self.initialize_calls += 1

        async def cleanup(self) -> None:
            self.cleanup_calls += 1

        async def scrape(self, request: Any) -> dict[str, Any]:
            self.scrape_calls.append(request)
            if self.side_effect is not None:
                raise self.side_effect
            return dict(self.result)

    class FirecrawlStub:
        def __init__(self, *_args: Any, **_kwargs: Any) -> None:
            stubs["firecrawl"] = self
            self.initialize_calls = 0
            self.cleanup_calls = 0
            self.crawl_calls: list[tuple[tuple[Any, ...], dict[str, Any]]] = []
            self.result: dict[str, Any] = {
                "success": True,
                "provider": "firecrawl",
                "pages": [{"url": "https://example.com", "content": "cached"}],
            }

        async def initialize(self) -> None:
            self.initialize_calls += 1

        async def cleanup(self) -> None:
            self.cleanup_calls += 1

        async def crawl(self, *args: Any, **kwargs: Any) -> dict[str, Any]:
            self.crawl_calls.append((args, kwargs))
            return dict(self.result)

    monkeypatch.setattr(unified_manager_module, "AutomationRouter", RouterStub)
    monkeypatch.setattr(unified_manager_module, "FirecrawlAdapter", FirecrawlStub)

    settings = Settings(environment=Environment.TESTING)
    manager = UnifiedBrowserManager(settings)

    return manager, stubs


@pytest.mark.asyncio
async def test_scrape_url_returns_failure_payload(
    manager_with_toggleable_stubs: tuple[UnifiedBrowserManager, dict[str, Any]],
) -> None:
    """scrape_url should surface router failure payloads unchanged."""

    manager, stubs = manager_with_toggleable_stubs
    await manager.initialize()

    stubs["router"].result = {
        "success": False,
        "error": "Router error",
        "provider": None,
    }

    response = await manager.scrape_url(
        UnifiedScrapingRequest(url="https://example.com")
    )

    assert response["success"] is False
    assert response["error"] == "Router error"

    await manager.cleanup()


@pytest.mark.asyncio
async def test_scrape_url_propagates_exceptions(
    manager_with_toggleable_stubs: tuple[UnifiedBrowserManager, dict[str, Any]],
) -> None:
    """Exceptions from the router should bubble up to callers."""

    manager, stubs = manager_with_toggleable_stubs
    await manager.initialize()

    stubs["router"].side_effect = CrawlServiceError("router unavailable")

    with pytest.raises(CrawlServiceError, match="router unavailable"):
        await manager.scrape_url(UnifiedScrapingRequest(url="https://example.com"))

    await manager.cleanup()


@pytest.mark.asyncio
async def test_crawl_site_firecrawl_failure(
    manager_with_toggleable_stubs: tuple[UnifiedBrowserManager, dict[str, Any]],
) -> None:
    """crawl_site should return failure metadata when Firecrawl fails."""

    manager, stubs = manager_with_toggleable_stubs
    await manager.initialize()

    stubs["firecrawl"].result = {
        "success": False,
        "error": "firecrawl outage",
        "provider": "firecrawl",
    }

    result = await manager.crawl_site("https://example.com")

    assert result["success"] is False
    assert result["error"] == "firecrawl outage"

    await manager.cleanup()


@pytest.mark.asyncio
async def test_crawl_site_router_failure_path(
    manager_with_toggleable_stubs: tuple[UnifiedBrowserManager, dict[str, Any]],
) -> None:
    """crawl_site prefer='crawl4ai' should report failure if the router fails."""

    manager, stubs = manager_with_toggleable_stubs
    await manager.initialize()

    stubs["router"].result = {"success": False, "error": "tier failure"}

    result = await manager.crawl_site("https://example.com/docs", prefer="crawl4ai")

    assert result["success"] is False
    assert result["error"] == "tier failure"
    assert result["total_pages"] == 0

    await manager.cleanup()
