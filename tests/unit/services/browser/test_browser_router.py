"""Unit tests for the AutomationRouter behaviour."""

from __future__ import annotations

import math
import sys
import types
from collections.abc import Callable, Iterator
from dataclasses import dataclass
from typing import Any, Protocol, cast
from unittest.mock import AsyncMock

import pytest
from aiolimiter import AsyncLimiter

from src.services.errors import CrawlServiceError


_STUBBED_NAMES: set[str] = set()
_PREVIOUS_MODULES: dict[str, types.ModuleType | None] = {}
_PARENT_ATTRS: dict[str, tuple[str, str]] = {}


def _register_stub(
    name: str, module: types.ModuleType, *, attach_to_parent: bool = False
) -> types.ModuleType:
    """Insert a stub module and optionally attach it to its parent module."""

    if name in sys.modules:
        _PREVIOUS_MODULES.setdefault(name, sys.modules[name])
        return sys.modules[name]

    _PREVIOUS_MODULES.setdefault(name, None)
    sys.modules[name] = module
    _STUBBED_NAMES.add(name)

    if attach_to_parent and "." in name:
        parent_name, attr = name.rsplit(".", 1)
        parent_module = sys.modules.get(parent_name)
        if parent_module is not None:
            setattr(parent_module, attr, module)
            _PARENT_ATTRS[name] = (parent_name, attr)

    return module


# pylint: disable=too-many-locals
def _install_optional_stubs() -> None:
    """Install lightweight stubs for optional browser dependencies."""

    if "firecrawl" not in sys.modules:
        firecrawl_module = types.ModuleType("firecrawl")

        class AsyncFirecrawl:  # pragma: no cover - minimal shim
            async def scrape(self, *_args: Any, **_kwargs: Any) -> dict[str, Any]:
                return {"success": True}

            async def crawl(self, *_args: Any, **_kwargs: Any) -> dict[str, Any]:
                return {"success": True}

        firecrawl_module.AsyncFirecrawl = AsyncFirecrawl  # type: ignore[attr-defined]
        _register_stub("firecrawl", firecrawl_module)

    if "crawl4ai" not in sys.modules:
        crawl4ai_module = types.ModuleType("crawl4ai")

        class AsyncWebCrawler:  # pragma: no cover - minimal shim
            async def start(self) -> None:
                return None

            async def close(self) -> None:
                return None

        class CacheMode:  # pragma: no cover - minimal shim
            BYPASS = "bypass"

        crawl4ai_module.AsyncWebCrawler = AsyncWebCrawler  # type: ignore[attr-defined]
        crawl4ai_module.CacheMode = CacheMode  # type: ignore[attr-defined]
        crawl4ai_module = _register_stub("crawl4ai", crawl4ai_module)

        models_module = types.ModuleType("crawl4ai.models")

        class CrawlResult:  # pragma: no cover - placeholder
            pass

        models_module.CrawlResult = CrawlResult  # type: ignore[attr-defined]
        _register_stub("crawl4ai.models", models_module, attach_to_parent=True)
    presets_name = "src.services.crawling.c4a_presets"
    crawling_pkg = "src.services.crawling"
    if crawling_pkg not in sys.modules:
        crawling_module = types.ModuleType(crawling_pkg)

        async def crawl_page(*_args: Any, **_kwargs: Any) -> dict[str, Any]:
            return {"success": True}

        crawling_module.crawl_page = crawl_page  # type: ignore[attr-defined]
        crawling_module = _register_stub(crawling_pkg, crawling_module)
    else:
        crawling_module = sys.modules[crawling_pkg]

    if presets_name not in sys.modules and crawling_pkg in sys.modules:
        presets_module = types.ModuleType(presets_name)

        class BrowserOptions:  # pragma: no cover - placeholder
            def __init__(self, **kwargs: Any) -> None:
                self.kwargs = kwargs

        def base_run_config(**_: Any) -> types.SimpleNamespace:
            return types.SimpleNamespace(clone=lambda **__: types.SimpleNamespace())

        def memory_dispatcher(**_: Any) -> types.SimpleNamespace:
            return types.SimpleNamespace()

        def preset_browser_config(_options: Any) -> types.SimpleNamespace:
            return types.SimpleNamespace()

        presets_module.BrowserOptions = BrowserOptions  # type: ignore[attr-defined]
        presets_module.base_run_config = base_run_config  # type: ignore[attr-defined]
        presets_module.memory_dispatcher = memory_dispatcher  # type: ignore[attr-defined]
        presets_module.preset_browser_config = preset_browser_config  # type: ignore[attr-defined]
        _register_stub(presets_name, presets_module, attach_to_parent=True)


# pylint: enable=too-many-locals

_install_optional_stubs()


@pytest.fixture(scope="module", autouse=True)
def _restore_optional_stubs() -> Iterator[None]:
    """Remove stubbed modules after the module-level tests complete."""

    try:
        yield
    finally:
        for name in sorted(_STUBBED_NAMES, key=len, reverse=True):
            previous = _PREVIOUS_MODULES.get(name)
            current = sys.modules.get(name)
            if previous is None:
                sys.modules.pop(name, None)
                parent_info = _PARENT_ATTRS.get(name)
                if parent_info and current is not None:
                    parent_name, attr = parent_info
                    parent_module = sys.modules.get(parent_name)
                    if (
                        parent_module is not None
                        and getattr(parent_module, attr, None) is current
                    ):
                        delattr(parent_module, attr)
            else:
                sys.modules[name] = previous
        _STUBBED_NAMES.clear()
        _PREVIOUS_MODULES.clear()
        _PARENT_ATTRS.clear()


# ruff: noqa: E402 - imports depend on stub installation above
# pylint: disable=wrong-import-position

from src.config.models import AutomationRouterConfig  # noqa: E402
from src.services.browser.router import AutomationRouter, ScrapeRequest  # noqa: E402


# pylint: enable=wrong-import-position


@dataclass(slots=True)
class ProviderCall:
    """Capture adapter invocation details for assertions."""

    args: tuple[Any, ...]
    kwargs: dict[str, Any]


class DummyScrapedContent:
    """Simple object exposing `model_dump` like Pydantic results."""

    def __init__(self, payload: dict[str, Any]) -> None:
        self._payload = payload

    def model_dump(self, **_: Any) -> dict[str, Any]:
        """Return a copy of the stored payload."""

        return dict(self._payload)


def _make_lightweight_stub(providers: dict[str, Any]) -> type[Any]:
    """Build a lightweight scraper stub class bound to the provider registry."""

    class LightweightStub:
        def __init__(self, *_args: Any, **_kwargs: Any) -> None:
            providers["lightweight"] = self
            self.result: DummyScrapedContent | None = DummyScrapedContent(
                {
                    "success": True,
                    "content": "lightweight",
                    "url": "https://example.com",
                }
            )
            self.side_effect: Exception | None = None
            self.calls: list[ProviderCall] = []

        async def scrape(self, url: str, **kwargs: Any) -> DummyScrapedContent | None:
            self.calls.append(ProviderCall(args=(url,), kwargs=dict(kwargs)))
            if self.side_effect is not None:
                raise self.side_effect
            return self.result

    return LightweightStub


def _make_dict_stub(
    providers: dict[str, Any],
    name: str,
    default: dict[str, Any],
    *,
    with_lifecycle: bool = False,
) -> type[Any]:
    """Build a provider stub class returning dictionary payloads."""

    class DictStub:
        def __init__(self, *_args: Any, **_kwargs: Any) -> None:
            providers[name] = self
            self.result: dict[str, Any] = dict(default)
            self.side_effect: Exception | None = None
            self.calls: list[ProviderCall] = []
            if with_lifecycle:
                self.initialize = AsyncMock()
                self.cleanup = AsyncMock()

        async def scrape(self, *args: Any, **kwargs: Any) -> dict[str, Any]:
            self.calls.append(ProviderCall(args=args, kwargs=kwargs))
            if self.side_effect is not None:
                raise self.side_effect
            return dict(self.result)

    return DictStub


def _patch_router_adapters(
    monkeypatch: pytest.MonkeyPatch, providers: dict[str, Any]
) -> None:
    """Patch router adapters with deterministic stub implementations."""

    patch_specs = [
        (
            "LightweightScraper",
            _make_lightweight_stub(providers),
        ),
        (
            "Crawl4AIAdapter",
            _make_dict_stub(
                providers,
                "crawl4ai",
                {"success": True, "content": "crawl4ai"},
            ),
        ),
        (
            "PlaywrightAdapter",
            _make_dict_stub(
                providers,
                "playwright",
                {"success": True, "content": "playwright"},
            ),
        ),
        (
            "BrowserUseAdapter",
            _make_dict_stub(
                providers,
                "browser_use",
                {"success": True, "content": "browser_use"},
            ),
        ),
        (
            "FirecrawlAdapter",
            _make_dict_stub(
                providers,
                "firecrawl",
                {"success": True, "content": "firecrawl"},
                with_lifecycle=True,
            ),
        ),
    ]
    for attr, stub in patch_specs:
        monkeypatch.setattr(f"src.services.browser.router.{attr}", stub)


class RouterFactory(Protocol):
    """Callable fixture returning a configured router and adapter stubs."""

    def __call__(
        self, automation_config: AutomationRouterConfig | None = None
    ) -> tuple[AutomationRouter, dict[str, Any]]: ...


@pytest.fixture
def router_factory(
    config_factory: Callable[..., Any], monkeypatch: pytest.MonkeyPatch
) -> RouterFactory:
    """Factory yielding an AutomationRouter with patched adapter dependencies."""

    def _create_router(
        automation_config: AutomationRouterConfig | None = None,
    ) -> tuple[AutomationRouter, dict[str, Any]]:
        providers: dict[str, Any] = {}
        config = config_factory()
        config.firecrawl = config.firecrawl.model_copy(
            update={"api_key": "fc-test-key"}
        )
        if automation_config is not None:
            config.automation_router = automation_config

        _patch_router_adapters(monkeypatch, providers)

        router = AutomationRouter(config)
        return router, providers

    return _create_router


@pytest.mark.asyncio
async def test_scrape_success_records_metrics(router_factory: RouterFactory) -> None:
    router, providers = router_factory()

    result = await router.scrape(ScrapeRequest(url="https://example.com"))

    assert result["success"] is True
    assert result["provider"] == "lightweight"
    assert "automation_time_ms" in result
    metrics = router.get_metrics_snapshot()
    assert metrics["lightweight"] == {"success": 1, "failure": 0, "rate_limited": 0}
    for tier in ("crawl4ai", "playwright", "browser_use", "firecrawl"):
        assert metrics[tier] == {"success": 0, "failure": 0, "rate_limited": 0}
        assert providers[tier].calls == []


@pytest.mark.asyncio
async def test_scrape_fallback_updates_metrics(router_factory: RouterFactory) -> None:
    router, providers = router_factory()
    providers["lightweight"].result = DummyScrapedContent(
        {"success": False, "error": "boom"}
    )

    payload = await router.scrape(ScrapeRequest(url="https://fallback.test"))

    assert payload["provider"] == "crawl4ai"
    metrics = router.get_metrics_snapshot()
    assert metrics["lightweight"]["failure"] == 1
    assert metrics["crawl4ai"]["success"] == 1


@pytest.mark.asyncio
async def test_lightweight_none_result_falls_back(
    router_factory: RouterFactory,
) -> None:
    router, providers = router_factory()
    providers["lightweight"].result = None

    payload = await router.scrape(ScrapeRequest(url="https://no-content.test"))

    assert payload["provider"] == "crawl4ai"
    metrics = router.get_metrics_snapshot()
    assert metrics["lightweight"]["failure"] == 1
    assert metrics["crawl4ai"]["success"] == 1


@pytest.mark.asyncio
async def test_scrape_force_tier_invokes_selected_adapter(
    router_factory: RouterFactory,
) -> None:
    router, providers = router_factory()

    result = await router.scrape(
        ScrapeRequest(url="https://play.test", tier="playwright")
    )

    assert result["provider"] == "playwright"
    assert providers["playwright"].calls, "Playwright not invoked"
    assert providers["lightweight"].calls == []
    assert providers["crawl4ai"].calls == []


@pytest.mark.asyncio
async def test_forced_browser_use_passes_task_and_instructions(
    router_factory: RouterFactory,
) -> None:
    router, providers = router_factory()
    actions = [{"type": "click", "selector": "#cta"}]

    await router.scrape(
        ScrapeRequest(
            url="https://browser.task",
            tier="browser_use",
            custom_actions=actions,
        )
    )

    call = providers["browser_use"].calls[0]
    assert call.kwargs["instructions"] == actions
    assert "Execute the provided browser automation instructions" in call.kwargs["task"]


@pytest.mark.parametrize(
    ("request_kwargs", "expected_order"),
    [
        (
            {"url": "https://docs.example"},
            ["lightweight", "crawl4ai", "playwright", "browser_use", "firecrawl"],
        ),
        (
            {"url": "https://docs.example", "interaction_required": True},
            ["playwright", "browser_use", "crawl4ai", "firecrawl", "lightweight"],
        ),
        (
            {
                "url": "https://docs.example",
                "custom_actions": [{"type": "click", "selector": "#cta"}],
            },
            ["playwright", "browser_use", "crawl4ai", "firecrawl", "lightweight"],
        ),
    ],
)
def test_choose_tiers_orders_adapters_for_context(
    router_factory: RouterFactory,
    request_kwargs: dict[str, Any],
    expected_order: list[str],
) -> None:
    router, _ = router_factory()
    tiers = router._choose_tiers(ScrapeRequest(**request_kwargs))  # noqa: SLF001
    assert tiers == expected_order


def test_choose_tiers_applies_hard_domain_override(
    router_factory: RouterFactory,
) -> None:
    automation_cfg = AutomationRouterConfig(hard_domains=["example.com"])
    router, _ = router_factory(automation_cfg)
    tiers = router._choose_tiers(  # noqa: SLF001
        ScrapeRequest(url="https://app.example.com/page")
    )
    assert tiers[0] == "firecrawl"


@pytest.mark.asyncio
async def test_failed_attempt_due_to_rate_limit_records_metric(
    router_factory: RouterFactory,
) -> None:
    router, _ = router_factory()

    class LimiterTimeoutStub:
        max_rate = 1
        time_period = 1.0

        async def acquire(self) -> None:
            raise TimeoutError

    router._limiter["lightweight"] = cast(AsyncLimiter, LimiterTimeoutStub())  # noqa: SLF001

    result = await router.scrape(ScrapeRequest(url="https://ratelimit.test"))

    assert result["provider"] == "crawl4ai"
    metrics = router.get_metrics_snapshot()
    assert metrics["lightweight"]["rate_limited"] == 1


@pytest.mark.asyncio
async def test_forced_firecrawl_uses_expected_formats(
    router_factory: RouterFactory,
) -> None:
    router, providers = router_factory()

    await router.scrape(ScrapeRequest(url="https://fire.test", tier="firecrawl"))

    call = providers["firecrawl"].calls[0]
    assert call.kwargs["formats"] == ["markdown", "html"]
    expected_timeout = max(
        1,
        math.ceil(router._router_config.per_attempt_cap_ms / 1000),  # noqa: SLF001
    )
    assert call.kwargs["timeout"] == expected_timeout


@pytest.mark.asyncio
async def test_zero_timeout_expires_before_attempt(
    router_factory: RouterFactory,
) -> None:
    router, providers = router_factory()

    result = await router.scrape(
        ScrapeRequest(url="https://timeout.test", timeout_ms=0)
    )

    assert result["success"] is False
    assert result["provider"] is None
    metrics = router.get_metrics_snapshot()
    assert all(
        stats == {"success": 0, "failure": 0, "rate_limited": 0}
        for stats in metrics.values()
    )
    for provider in providers.values():
        assert not provider.calls


@pytest.mark.asyncio
async def test_min_budget_failure_returns_error(router_factory: RouterFactory) -> None:
    automation_cfg = AutomationRouterConfig(min_attempt_ms=500)
    router, providers = router_factory(automation_cfg)
    for provider in providers.values():
        if hasattr(provider, "side_effect"):
            provider.side_effect = None  # type: ignore[attr-defined]
        if isinstance(provider.result, DummyScrapedContent):
            provider.result = DummyScrapedContent({"success": True})
        else:
            provider.result = {"success": True}

    result = await router.scrape(
        ScrapeRequest(url="https://budget.test", timeout_ms=100)
    )

    assert result["success"] is False
    assert "Remaining budget" in result["error"]


def test_limiter_override_applied(router_factory: RouterFactory) -> None:
    automation_cfg = AutomationRouterConfig(
        rate_limits={"lightweight": 3},
        limiter_period_seconds=2.5,
    )
    router, _ = router_factory(automation_cfg)

    limiter = router._limiter["lightweight"]  # noqa: SLF001
    assert limiter.max_rate == 3
    assert limiter.time_period == 2.5


@pytest.mark.asyncio
async def test_initialize_and_cleanup_delegate_to_firecrawl(
    router_factory: RouterFactory,
) -> None:
    router, providers = router_factory()

    await router.initialize()
    await router.cleanup()

    firecrawl = providers["firecrawl"]
    firecrawl.initialize.assert_awaited_once()
    firecrawl.cleanup.assert_awaited_once()


def test_browser_use_task_variants() -> None:
    base = ScrapeRequest(url="https://docs.example")
    assert (
        "extract comprehensive documentation content"
        in AutomationRouter._browser_use_task(base)
    )

    interaction = ScrapeRequest(url="https://docs.example", interaction_required=True)
    assert "Perform the necessary interactions" in AutomationRouter._browser_use_task(
        interaction
    )

    custom = ScrapeRequest(
        url="https://docs.example",
        custom_actions=[{"type": "click", "selector": "#cta"}],
    )
    assert "Execute the provided browser automation instructions" in (
        AutomationRouter._browser_use_task(custom)
    )


@pytest.mark.asyncio
async def test_crawl_service_error_from_provider_triggers_fallback(
    router_factory: RouterFactory,
) -> None:
    router, providers = router_factory()
    providers["lightweight"].result = DummyScrapedContent({"success": False})
    providers["crawl4ai"].side_effect = CrawlServiceError("adapter failed")
    providers["firecrawl"].result = {"success": True, "content": "firecrawl"}

    response = await router.scrape(ScrapeRequest(url="https://error.test"))

    assert response["provider"] == "playwright"
    metrics = router.get_metrics_snapshot()
    assert metrics["crawl4ai"]["failure"] == 1
