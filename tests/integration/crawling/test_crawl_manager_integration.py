"""Integration-like tests for the CrawlManager using stubbed browser tiers."""

from __future__ import annotations

from types import SimpleNamespace
from typing import Any, cast

import pytest

from src.config import Config
from src.services.browser.unified_manager import UnifiedScrapingResponse
from src.services.crawling.manager import CrawlManager, CrawlServiceError


class _TierMetricsStub:
    def __init__(self, *, tier_name: str, success_rate: float) -> None:
        self._data = {
            "tier_name": tier_name,
            "total_requests": 1,
            "successful_requests": 1,
            "failed_requests": 0,
            "average_response_time_ms": 25.0,
            "success_rate": success_rate,
        }

    def dict(self) -> dict[str, float]:
        return self._data


class _StubbedUnifiedBrowserManager:
    """Stub for UnifiedBrowserManager used by CrawlManager integration tests."""

    def __init__(self, _config: Any) -> None:
        self.initialized = False
        self.cleanup_called = False
        self.requests: list[Any] = []
        self._response: UnifiedScrapingResponse | Exception | None = None
        self._tier_metrics = {
            "lightweight": _TierMetricsStub(
                tier_name="lightweight",
                success_rate=1.0,
            )
        }

    async def initialize(self) -> None:
        self.initialized = True

    async def cleanup(self) -> None:
        self.cleanup_called = True

    async def scrape(self, request) -> UnifiedScrapingResponse:
        self.requests.append(request)
        if isinstance(self._response, Exception):
            raise self._response
        if isinstance(self._response, UnifiedScrapingResponse):
            return self._response
        return UnifiedScrapingResponse(
            success=True,
            content="Example content",
            url=request.url,
            title="Example",
            metadata={"tier": request.tier},
            tier_used="lightweight",
            execution_time_ms=12.0,
            fallback_attempted=False,
            content_length=15,
            quality_score=0.9,
            error=None,
            failed_tiers=[],
        )

    def get_tier_metrics(self):
        return self._tier_metrics


def _build_config() -> SimpleNamespace:
    return SimpleNamespace(
        cache=SimpleNamespace(enable_browser_cache=False),
        performance=SimpleNamespace(enable_monitoring=False),
    )


@pytest.mark.asyncio
async def test_crawl_manager_returns_standard_payload(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """crawl_manager.scrape_url should forward to UnifiedBrowserManager."""

    stub_manager = _StubbedUnifiedBrowserManager(_build_config())
    monkeypatch.setattr(
        "src.services.crawling.manager.UnifiedBrowserManager",
        lambda config: stub_manager,
    )

    manager = CrawlManager(cast(Config, _build_config()))
    await manager.initialize()

    result = await manager.scrape_url("https://example.com")

    assert result["success"] is True
    assert result["tier_used"] == "lightweight"
    assert stub_manager.requests

    await manager.cleanup()
    assert stub_manager.cleanup_called is True


@pytest.mark.asyncio
async def test_crawl_manager_handles_scrape_errors(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """crawler errors should be translated into a standardized failure payload."""

    stub_manager = _StubbedUnifiedBrowserManager(_build_config())
    stub_manager._response = OSError("network error")  # type: ignore[attr-defined]
    monkeypatch.setattr(
        "src.services.crawling.manager.UnifiedBrowserManager",
        lambda config: stub_manager,
    )

    manager = CrawlManager(cast(Config, _build_config()))
    await manager.initialize()

    result = await manager.scrape_url("https://example.com/fail")

    assert result["success"] is False
    assert result["error"] == "UnifiedBrowserManager error"

    await manager.cleanup()


@pytest.mark.asyncio
async def test_crawl_manager_requires_initialization(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Calling scrape_url before initialize should raise a CrawlServiceError."""

    stub_manager = _StubbedUnifiedBrowserManager(_build_config())
    monkeypatch.setattr(
        "src.services.crawling.manager.UnifiedBrowserManager",
        lambda config: stub_manager,
    )

    manager = CrawlManager(cast(Config, _build_config()))

    with pytest.raises(CrawlServiceError, match="Manager not initialized"):
        await manager.scrape_url("https://example.com")


@pytest.mark.asyncio
async def test_crawl_manager_exposes_tier_metrics(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """get_metrics should proxy tier statistics from UnifiedBrowserManager."""

    stub_manager = _StubbedUnifiedBrowserManager(_build_config())
    monkeypatch.setattr(
        "src.services.crawling.manager.UnifiedBrowserManager",
        lambda config: stub_manager,
    )

    manager = CrawlManager(cast(Config, _build_config()))
    await manager.initialize()

    metrics = manager.get_metrics()
    assert metrics["lightweight"]["success_rate"] == 1.0

    await manager.cleanup()
