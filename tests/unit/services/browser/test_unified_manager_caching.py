"""Error-handling and reuse tests for UnifiedBrowserManager (no caching layer)."""

from __future__ import annotations

from typing import Any

import pytest

from src.config import Environment, Settings
from src.services.browser.unified_manager import (
    UnifiedBrowserManager,
    UnifiedScrapingRequest,
)
from src.services.errors import CrawlServiceError

from ._manager_stub_utils import install_manager_stubs


@pytest.fixture
def manager_with_toggleable_stubs(
    monkeypatch: pytest.MonkeyPatch,
) -> tuple[UnifiedBrowserManager, dict[str, Any]]:
    """Create a UnifiedBrowserManager whose dependencies can simulate failures."""

    stubs: dict[str, Any] = {}

    install_manager_stubs(
        monkeypatch,
        stubs,
        router_default={
            "success": True,
            "content": "fresh-content",
            "metadata": {},
        },
        firecrawl_default={
            "success": True,
            "provider": "firecrawl",
            "pages": [{"url": "https://example.com", "content": "cached"}],
        },
        enable_router_side_effects=True,
    )

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
