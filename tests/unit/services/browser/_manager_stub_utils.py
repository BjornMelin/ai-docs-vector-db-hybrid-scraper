"""Shared stubs for UnifiedBrowserManager dependency patching in tests."""

from __future__ import annotations

from collections.abc import Callable
from typing import Any

import src.services.browser.unified_manager as unified_manager_module


def install_manager_stubs(
    monkeypatch: Any,
    registry: dict[str, Any],
    *,
    router_default: dict[str, Any] | None = None,
    firecrawl_default: dict[str, Any] | None = None,
    enable_router_side_effects: bool = False,
) -> None:
    """Patch UnifiedBrowserManager dependencies with lightweight stubs.

    Args:
        monkeypatch: Pytest monkeypatch fixture.
        registry: Dictionary used to expose created stub instances.
        router_default: Default response dict for router.scrape.
        firecrawl_default: Default response dict for firecrawl.crawl.
        enable_router_side_effects: When True, the router stub exposes a
            ``side_effect`` attribute that, if set, is raised during scrape.
    """

    router_factory = _make_router_stub_factory(
        registry,
        default_result=router_default,
        enable_side_effects=enable_router_side_effects,
    )
    firecrawl_factory = _make_firecrawl_stub_factory(
        registry,
        default_result=firecrawl_default,
    )

    monkeypatch.setattr(unified_manager_module, "AutomationRouter", router_factory)
    monkeypatch.setattr(unified_manager_module, "FirecrawlAdapter", firecrawl_factory)


def _make_router_stub_factory(
    registry: dict[str, Any],
    *,
    default_result: dict[str, Any] | None,
    enable_side_effects: bool,
) -> Callable[..., Any]:
    payload = default_result or {
        "success": True,
        "provider": "lightweight",
        "content": "stub-content",
        "metadata": {},
    }

    class RouterStub:
        def __init__(self, *_args: Any, **_kwargs: Any) -> None:
            registry["router"] = self
            self.initialize_calls = 0
            self.cleanup_calls = 0
            self.scrape_calls: list[Any] = []
            self.result: dict[str, Any] = dict(payload)
            if enable_side_effects:
                self.side_effect: Exception | None = None

        async def initialize(self) -> None:
            self.initialize_calls += 1

        async def cleanup(self) -> None:
            self.cleanup_calls += 1

        async def scrape(self, request: Any) -> dict[str, Any]:
            self.scrape_calls.append(request)
            if enable_side_effects:
                error = getattr(self, "side_effect", None)
                if error is not None:
                    raise error
            return dict(self.result)

    return RouterStub


def _make_firecrawl_stub_factory(
    registry: dict[str, Any],
    *,
    default_result: dict[str, Any] | None,
) -> Callable[..., Any]:
    payload = default_result or {
        "success": True,
        "provider": "firecrawl",
        "pages": [{"url": "https://example.com", "content": "page"}],
    }

    class FirecrawlStub:
        def __init__(self, *_args: Any, **_kwargs: Any) -> None:
            registry["firecrawl"] = self
            self.initialize_calls = 0
            self.cleanup_calls = 0
            self.crawl_calls: list[tuple[tuple[Any, ...], dict[str, Any]]] = []
            self.result: dict[str, Any] = dict(payload)

        async def initialize(self) -> None:
            self.initialize_calls += 1

        async def cleanup(self) -> None:
            self.cleanup_calls += 1

        async def crawl(self, *args: Any, **kwargs: Any) -> dict[str, Any]:
            self.crawl_calls.append((args, kwargs))
            return dict(self.result)

    return FirecrawlStub
