"""Tests for the streamlined UnifiedBrowserManager implementation."""

from __future__ import annotations

from typing import Any

import pytest

from src.config.loader import Settings
from src.config.models import Environment
from src.services.browser.unified_manager import (
    UnifiedBrowserManager,
    UnifiedScrapingRequest,
)
from src.services.errors import CrawlServiceError

from ._manager_stub_utils import install_manager_stubs


@pytest.fixture
def manager_with_stubs(
    monkeypatch: pytest.MonkeyPatch,
) -> tuple[UnifiedBrowserManager, dict[str, Any]]:
    """Create a UnifiedBrowserManager with stubbed dependencies."""

    stubs: dict[str, Any] = {}

    install_manager_stubs(monkeypatch, stubs)

    settings = Settings(environment=Environment.TESTING)
    settings.firecrawl.api_key = "test-key"
    manager = UnifiedBrowserManager(settings)

    return manager, stubs


@pytest.mark.asyncio
async def test_initialize_invokes_router_and_firecrawl(
    manager_with_stubs: tuple[UnifiedBrowserManager, dict[str, Any]],
) -> None:
    """initialize should configure both underlying services."""

    manager, stubs = manager_with_stubs

    await manager.initialize()

    assert manager._initialized is True  # type: ignore[attr-defined]
    assert stubs["router"].initialize_calls == 1
    assert stubs["firecrawl"].initialize_calls == 1

    await manager.cleanup()


@pytest.mark.asyncio
async def test_initialize_is_idempotent(
    manager_with_stubs: tuple[UnifiedBrowserManager, dict[str, Any]],
) -> None:
    """Multiple initialize calls should not reinitialize dependencies."""

    manager, stubs = manager_with_stubs

    await manager.initialize()
    await manager.initialize()

    assert stubs["router"].initialize_calls == 1
    assert stubs["firecrawl"].initialize_calls == 1

    await manager.cleanup()


@pytest.mark.asyncio
async def test_cleanup_resets_state(
    manager_with_stubs: tuple[UnifiedBrowserManager, dict[str, Any]],
) -> None:
    """cleanup should tear down dependencies and reset the manager."""

    manager, stubs = manager_with_stubs

    await manager.initialize()
    await manager.cleanup()

    assert manager._initialized is False  # type: ignore[attr-defined]
    assert stubs["router"].cleanup_calls == 1
    assert stubs["firecrawl"].cleanup_calls == 1


@pytest.mark.asyncio
async def test_scrape_url_returns_router_payload(
    manager_with_stubs: tuple[UnifiedBrowserManager, dict[str, Any]],
) -> None:
    """scrape_url should proxy to the router and return its response."""

    manager, stubs = manager_with_stubs
    await manager.initialize()

    request = UnifiedScrapingRequest(url="https://example.com")
    response = await manager.scrape_url(request)

    assert response["success"] is True
    assert response["provider"] == "lightweight"
    assert stubs["router"].scrape_calls  # Router was invoked

    await manager.cleanup()


@pytest.mark.asyncio
async def test_scrape_url_requires_initialization(
    manager_with_stubs: tuple[UnifiedBrowserManager, dict[str, Any]],
) -> None:
    """scrape_url should validate initialization state."""

    manager, _ = manager_with_stubs

    with pytest.raises(CrawlServiceError, match="not initialized"):
        await manager.scrape_url(UnifiedScrapingRequest(url="https://example.com"))


@pytest.mark.asyncio
async def test_scrape_url_forwards_request_metadata(
    manager_with_stubs: tuple[UnifiedBrowserManager, dict[str, Any]],
) -> None:
    """scrape_url should propagate tier and interaction flags."""

    manager, stubs = manager_with_stubs
    await manager.initialize()

    request = UnifiedScrapingRequest(
        url="https://example.com/app",
        tier="browser_use",
        interaction_required=True,
        custom_actions=[{"type": "click", "selector": "#start"}],
        timeout=45_000,
    )

    await manager.scrape_url(request)

    assert len(stubs["router"].scrape_calls) == 1
    routed_request = stubs["router"].scrape_calls[0]
    assert routed_request.tier == "browser_use"
    assert routed_request.interaction_required is True
    assert routed_request.timeout_ms == 45_000
    assert routed_request.custom_actions == [{"type": "click", "selector": "#start"}]

    await manager.cleanup()


@pytest.mark.asyncio
async def test_crawl_site_prefers_firecrawl(
    manager_with_stubs: tuple[UnifiedBrowserManager, dict[str, Any]],
) -> None:
    """crawl_site should defer to Firecrawl by default."""

    manager, stubs = manager_with_stubs
    await manager.initialize()

    result = await manager.crawl_site("https://example.com/blog")

    assert result["success"] is True
    assert stubs["firecrawl"].crawl_calls
    assert stubs["router"].scrape_calls == []

    await manager.cleanup()


@pytest.mark.asyncio
async def test_crawl_site_uses_router_when_requested(
    manager_with_stubs: tuple[UnifiedBrowserManager, dict[str, Any]],
) -> None:
    """crawl_site with prefer='crawl4ai' should delegate to the router."""

    manager, stubs = manager_with_stubs
    await manager.initialize()

    stubs["router"].result = {
        "success": True,
        "content": "page",
        "metadata": {"title": "Docs"},
    }

    result = await manager.crawl_site("https://example.com/docs", prefer="crawl4ai")

    assert result["success"] is True
    assert stubs["router"].scrape_calls
    assert stubs["firecrawl"].crawl_calls == []
    assert result["total_pages"] == 1

    await manager.cleanup()


@pytest.mark.asyncio
async def test_crawl_site_requires_initialization(
    manager_with_stubs: tuple[UnifiedBrowserManager, dict[str, Any]],
) -> None:
    """crawl_site should enforce initialization."""

    manager, _ = manager_with_stubs

    with pytest.raises(CrawlServiceError, match="not initialized"):
        await manager.crawl_site("https://example.com")
