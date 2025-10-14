"""Tests for the streamlined UnifiedBrowserManager implementation."""

from dataclasses import dataclass

import pytest

from src.config.loader import Settings
from src.services.browser.models import BrowserResult, ProviderKind


@dataclass
class _StubResult:
    success: bool = True
    provider: ProviderKind = ProviderKind.LIGHTWEIGHT


class StubRouter:
    """Stub implementation of BrowserRouter for testing."""

    def __init__(self, *_, **__):
        """Initialize the stub router."""
        self.initialized = False

    async def initialize(self) -> None:
        """Initialize the stub router."""
        self.initialized = True

    async def cleanup(self) -> None:
        """Cleanup the stub router."""
        self.initialized = False

    def is_initialized(self) -> bool:
        """Check if the stub router is initialized."""
        return self.initialized

    async def scrape(self, request):  # pragma: no cover - stub intercept
        """Stub scrape method."""
        return BrowserResult(
            success=True,
            url=request.url,
            title="",
            content="stub",
            html="<p>stub</p>",
            metadata={},
            provider=ProviderKind.LIGHTWEIGHT,
            links=None,
            assets=None,
            elapsed_ms=None,
        )

    def get_provider(self, kind):  # pragma: no cover - manager doesn't call in test
        """Stub get_provider method."""
        return


@pytest.mark.asyncio
async def test_unified_manager_uses_router(monkeypatch):
    """Unified manager should use the router for scraping."""
    monkeypatch.setattr(
        "src.services.browser.unified_manager.BrowserRouter",
        StubRouter,
    )
    settings = Settings()
    manager_cls = __import__(
        "src.services.browser.unified_manager",
        fromlist=["UnifiedBrowserManager"],
    ).UnifiedBrowserManager
    manager = manager_cls(settings)

    await manager.initialize()
    payload = await manager.scrape_url("https://example.com")

    assert isinstance(payload, BrowserResult)
    assert payload.success is True
    assert payload.provider is ProviderKind.LIGHTWEIGHT
    await manager.cleanup()
    assert manager.is_initialized() is False
