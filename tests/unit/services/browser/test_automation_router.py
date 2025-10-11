"""Focused tests for AutomationRouter with patched provider stubs."""

from __future__ import annotations

from typing import Any

import pytest

import src.services.browser.router as router_module
from src.config.loader import Settings
from src.config.models import Environment
from src.services.browser.router import AutomationRouter, ScrapeRequest
from src.services.errors import CrawlServiceError


@pytest.fixture
def router_with_stubs(
    monkeypatch: pytest.MonkeyPatch,
) -> tuple[AutomationRouter, dict[str, Any]]:
    """Create an AutomationRouter with lightweight provider stubs."""

    stubs: dict[str, Any] = {}

    class DummyScraped:
        """Simple object exposing model_dump for lightweight results."""

        def __init__(self, payload: dict[str, Any]) -> None:
            self._payload = payload

        def model_dump(self, **_: Any) -> dict[str, Any]:
            return dict(self._payload)

    class LightweightStub:
        """Stub for LightweightScraper avoiding network access."""

        def __init__(self, *_args: Any, **_kwargs: Any) -> None:
            stubs["lightweight"] = self
            self.calls: list[dict[str, Any]] = []
            self.should_fail = False
            self.return_payload: dict[str, Any] = {
                "success": True,
                "content": "lightweight-content",
            }

        async def initialize(self) -> None:
            return None

        async def cleanup(self) -> None:
            return None

        async def scrape(self, url: str, *, timeout_ms: int) -> DummyScraped | None:
            """Return deterministic payloads unless failure is requested."""

            self.calls.append({"url": url, "timeout_ms": timeout_ms})
            if self.should_fail:
                return None
            payload = dict(self.return_payload)
            payload.setdefault("url", url)
            payload.setdefault("success", True)
            return DummyScraped(payload)

    class AdapterStub:
        """Base stub for asynchronous automation adapters."""

        def __init__(self, name: str, *_args: Any, **_kwargs: Any) -> None:
            stubs[name] = self
            self.name = name
            self.calls: list[tuple[str, dict[str, Any]]] = []
            self.initialize_called = False
            self.cleanup_called = False
            self.side_effect: Exception | None = None
            self.return_payload: dict[str, Any] = {
                "success": True,
                "content": f"{name}-content",
            }

        async def initialize(self) -> None:
            self.initialize_called = True

        async def cleanup(self) -> None:
            self.cleanup_called = True

        async def scrape(self, url: str, **kwargs: Any) -> dict[str, Any]:
            """Record calls and optionally raise configured side effects."""

            self.calls.append((url, kwargs))
            if self.side_effect is not None:
                raise self.side_effect
            payload = dict(self.return_payload)
            payload.setdefault("success", True)
            payload.setdefault("provider", self.name)
            payload.setdefault("url", url)
            return payload

    class Crawl4AIStub(AdapterStub):
        def __init__(self, *_args: Any, **_kwargs: Any) -> None:
            super().__init__("crawl4ai", *_args, **_kwargs)

    class PlaywrightStub(AdapterStub):
        def __init__(self, *_args: Any, **_kwargs: Any) -> None:
            super().__init__("playwright", *_args, **_kwargs)

    class BrowserUseStub(AdapterStub):
        def __init__(self, *_args: Any, **_kwargs: Any) -> None:
            super().__init__("browser_use", *_args, **_kwargs)

    class FirecrawlStub(AdapterStub):
        def __init__(self, *_args: Any, **_kwargs: Any) -> None:
            super().__init__("firecrawl", *_args, **_kwargs)

        async def crawl(self, *_args: Any, **_kwargs: Any) -> dict[str, Any]:
            return {
                "success": True,
                "provider": self.name,
                "pages": [],
            }

    monkeypatch.setattr(router_module, "LightweightScraper", LightweightStub)
    monkeypatch.setattr(router_module, "Crawl4AIAdapter", Crawl4AIStub)
    monkeypatch.setattr(router_module, "PlaywrightAdapter", PlaywrightStub)
    monkeypatch.setattr(router_module, "BrowserUseAdapter", BrowserUseStub)
    monkeypatch.setattr(router_module, "FirecrawlAdapter", FirecrawlStub)

    settings = Settings(environment=Environment.TESTING)
    router = AutomationRouter(settings)

    return router, stubs


@pytest.mark.asyncio
async def test_initialize_invokes_firecrawl(
    router_with_stubs: tuple[AutomationRouter, dict[str, Any]],
) -> None:
    """initialize should set up the Firecrawl adapter."""

    router, stubs = router_with_stubs

    await router.initialize()

    firecrawl_stub = stubs["firecrawl"]
    assert firecrawl_stub.initialize_called is True

    await router.cleanup()


@pytest.mark.asyncio
async def test_scrape_returns_lightweight_result(
    router_with_stubs: tuple[AutomationRouter, dict[str, Any]],
) -> None:
    """scrape should use lightweight tier first and return its payload."""

    router, stubs = router_with_stubs
    request = ScrapeRequest(url="https://example.com/docs")

    result = await router.scrape(request)

    assert result["success"] is True
    assert result["provider"] == "lightweight"
    assert stubs["lightweight"].calls  # type: ignore[attr-defined]
    assert "automation_time_ms" in result

    await router.cleanup()


@pytest.mark.asyncio
async def test_scrape_falls_back_on_failure(
    router_with_stubs: tuple[AutomationRouter, dict[str, Any]],
) -> None:
    """Router should fall back to the next tier when the first tier fails."""

    router, stubs = router_with_stubs
    lightweight_stub = stubs["lightweight"]
    lightweight_stub.should_fail = True  # type: ignore[attr-defined]

    request = ScrapeRequest(url="https://example.com/guide")
    result = await router.scrape(request)

    assert result["provider"] == "crawl4ai"

    metrics = router.get_metrics_snapshot()
    assert metrics["lightweight"]["failure"] == 1
    assert metrics["crawl4ai"]["success"] == 1

    await router.cleanup()


@pytest.mark.asyncio
async def test_scrape_returns_error_when_all_tiers_fail(
    router_with_stubs: tuple[AutomationRouter, dict[str, Any]],
) -> None:
    """Router should surface an error when every tier fails."""

    router, stubs = router_with_stubs

    lightweight_stub = stubs["lightweight"]
    lightweight_stub.should_fail = True  # type: ignore[attr-defined]

    for name, stub in stubs.items():
        if name == "lightweight":
            continue
        stub.side_effect = CrawlServiceError(f"{name} failed")  # type: ignore[attr-defined]

    request = ScrapeRequest(url="https://example.com/failure")
    result = await router.scrape(request)

    assert result["success"] is False
    assert result["provider"] is None

    metrics = router.get_metrics_snapshot()
    assert metrics["firecrawl"]["failure"] >= 1

    await router.cleanup()


def test_choose_tiers_for_hard_domain(
    router_with_stubs: tuple[AutomationRouter, dict[str, Any]],
) -> None:
    """Hard domains should escalate to resilient tiers first."""

    router, _ = router_with_stubs
    router._hard_domains = frozenset({"github.com"})  # type: ignore[attr-defined]

    tiers = router._choose_tiers(ScrapeRequest(url="https://docs.github.com/start"))  # type: ignore[attr-defined]

    assert tiers[:3] == ["firecrawl", "playwright", "browser_use"]


def test_choose_tiers_with_interaction(
    router_with_stubs: tuple[AutomationRouter, dict[str, Any]],
) -> None:
    """Interaction-required requests should prioritize rich automation tiers."""

    router, _ = router_with_stubs

    tiers = router._choose_tiers(  # type: ignore[attr-defined]
        ScrapeRequest(url="https://example.com/form", interaction_required=True)
    )

    assert tiers[:3] == ["playwright", "browser_use", "crawl4ai"]


def test_choose_tiers_forced_tier(
    router_with_stubs: tuple[AutomationRouter, dict[str, Any]],
) -> None:
    """Forcing a tier should bypass automatic ordering."""

    router, _ = router_with_stubs

    tiers = router._choose_tiers(  # type: ignore[attr-defined]
        ScrapeRequest(url="https://example.com", tier="playwright")
    )

    assert tiers == ["playwright"]


def test_choose_tiers_static_domain(
    router_with_stubs: tuple[AutomationRouter, dict[str, Any]],
) -> None:
    """Static domains should prioritize lightweight tiers."""

    router, _ = router_with_stubs

    tiers = router._choose_tiers(  # type: ignore[attr-defined]
        ScrapeRequest(url="https://docs.example.com/guide")
    )

    assert tiers[:3] == ["lightweight", "crawl4ai", "playwright"]


def test_choose_tiers_dynamic_domain(
    router_with_stubs: tuple[AutomationRouter, dict[str, Any]],
) -> None:
    """Dynamic domains should escalate to scripted automation."""

    router, _ = router_with_stubs

    tiers = router._choose_tiers(  # type: ignore[attr-defined]
        ScrapeRequest(url="https://app.example.com/dashboard")
    )

    assert tiers[:3] == ["playwright", "crawl4ai", "browser_use"]


def test_choose_tiers_anti_bot_keyword(
    router_with_stubs: tuple[AutomationRouter, dict[str, Any]],
) -> None:
    """Anti-bot indicators should escalate to resilient cloud tiers."""

    router, _ = router_with_stubs

    tiers = router._choose_tiers(  # type: ignore[attr-defined]
        ScrapeRequest(url="https://secure.cloudflare-protected.example.com/login")
    )

    assert tiers[0] == "firecrawl"
