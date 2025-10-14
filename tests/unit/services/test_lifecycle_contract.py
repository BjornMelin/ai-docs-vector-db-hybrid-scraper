"""Lifecycle tests for service abstractions and browser router."""

from __future__ import annotations

from typing import cast

import pytest

from src.config.browser import RateLimitConfig, RouterSettings
from src.services.base import BaseService
from src.services.browser.models import BrowserResult, ProviderKind, ScrapeRequest
from src.services.browser.providers import (
    BrowserUseProvider,
    Crawl4AIProvider,
    FirecrawlProvider,
    LightweightProvider,
    PlaywrightProvider,
)
from src.services.browser.providers.base import BrowserProvider, ProviderContext
from src.services.browser.router import BrowserRouter


class DummyService(BaseService):
    """Minimal concrete implementation used for lifecycle assertions."""

    def __init__(self) -> None:
        """Initialize dummy counters."""
        super().__init__(config=None)
        self.init_calls = 0
        self.cleanup_calls = 0

    async def initialize(self) -> None:
        """Simulate resource initialization and mark service ready."""
        self.init_calls += 1
        self._mark_initialized()

    async def cleanup(self) -> None:
        """Simulate resource cleanup and mark service uninitialized."""
        self.cleanup_calls += 1
        self._mark_uninitialized()


class DummyProvider(BrowserProvider):
    """No-op provider used to exercise router lifecycle behaviour."""

    kind = ProviderKind.LIGHTWEIGHT

    def __init__(self) -> None:
        super().__init__(ProviderContext(self.kind))
        self._initialized = False

    async def initialize(self) -> None:
        """Mark the provider as initialized."""
        self._initialized = True

    async def close(self) -> None:
        """Mark the provider as uninitialized."""
        self._initialized = False

    async def scrape(
        self, request: ScrapeRequest
    ) -> BrowserResult:  # pragma: no cover - not used
        """Raise to highlight this helper should not be invoked in tests."""
        raise NotImplementedError from None


@pytest.mark.asyncio
async def test_base_service_lifecycle() -> None:
    """Ensure BaseService lifecycle flags toggle during initialize/cleanup."""
    service = DummyService()
    assert not service.is_initialized()

    await service.initialize()
    assert service.is_initialized()

    await service.cleanup()
    assert not service.is_initialized()
    assert service.cleanup_calls == 1


@pytest.mark.asyncio
async def test_browser_router_lifecycle_marks_initialized() -> None:
    """Validate browser router propagates lifecycle markers to providers."""
    provider = DummyProvider()
    router = BrowserRouter(
        settings=RouterSettings(
            rate_limits={
                "lightweight": RateLimitConfig(max_requests=1, period_seconds=1.0)
            }
        ),
        lightweight=cast(LightweightProvider, provider),
        crawl4ai=cast(Crawl4AIProvider, provider),
        playwright=cast(PlaywrightProvider, provider),
        browser_use=cast(BrowserUseProvider, provider),
        firecrawl=cast(FirecrawlProvider, provider),
    )

    assert not router.is_initialized()
    await router.initialize()
    assert router.is_initialized()

    await router.cleanup()
    assert not router.is_initialized()
