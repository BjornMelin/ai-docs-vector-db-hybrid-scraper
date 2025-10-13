"""Unit tests covering the browser router orchestration logic."""

from __future__ import annotations

import asyncio
from collections import deque
from collections.abc import Iterable
from typing import cast

import pytest  # pylint: disable=import-error

from src.config.browser import RateLimitConfig, RouterSettings
from src.services.browser.errors import BrowserProviderError, BrowserRouterError
from src.services.browser.models import BrowserResult, ProviderKind, ScrapeRequest
from src.services.browser.providers import (
    BrowserProvider,
    BrowserUseProvider,
    Crawl4AIProvider,
    FirecrawlProvider,
    LightweightProvider,
    PlaywrightProvider,
    ProviderContext,
)
from src.services.browser.router import BrowserRouter


MILLISECONDS_PER_SECOND = 1000


class StubProvider(BrowserProvider):
    """Test double that records invocations and yields configured responses."""

    def __init__(  # pylint: disable=too-many-arguments
        self,
        context: ProviderContext,
        *,
        responses: Iterable[object] | None = None,
        delay_ms: int = 0,
        init_failures: int = 0,
        call_log: list[ProviderKind] | None = None,
    ) -> None:
        """Initialize the stub provider."""

        super().__init__(context)
        self.kind = context.provider
        self._responses = deque(responses or [])
        self._delay = delay_ms / MILLISECONDS_PER_SECOND
        self._init_failures = init_failures
        self._call_log = call_log
        self.initialized = False
        self.calls = 0

    async def initialize(self) -> None:
        if self._init_failures > 0:
            self._init_failures -= 1
            raise BrowserProviderError("init failure", provider=self.kind.value)
        self.initialized = True

    async def close(self) -> None:
        self.initialized = False

    async def scrape(self, request: ScrapeRequest) -> BrowserResult:
        self.calls += 1
        if self._call_log is not None:
            self._call_log.append(self.kind)
        if self._delay:
            await asyncio.sleep(self._delay)

        if not self._responses:
            return BrowserResult.failure(
                url=request.url,
                provider=self.kind,
                error="no response configured",
            )

        outcome = self._responses.popleft()
        if isinstance(outcome, Exception):
            raise outcome
        if isinstance(outcome, BrowserResult):
            return outcome

        success = bool(outcome)
        if success:
            return BrowserResult(
                success=True,
                url=request.url,
                title="Example",
                content="example",
                html="<p>example</p>",
                metadata={},
                provider=self.kind,
                links=None,
                assets=None,
                elapsed_ms=5,
            )
        return BrowserResult.failure(
            url=request.url,
            provider=self.kind,
            error="configured failure",
        )


def _build_router(
    *,
    settings: RouterSettings | None = None,
    overrides: dict[ProviderKind, StubProvider] | None = None,
) -> BrowserRouter:
    """Helper to build a router with stub providers."""

    overrides = overrides or {}
    providers = {
        kind: overrides.get(
            kind,
            StubProvider(ProviderContext(kind), responses=[True]),
        )
        for kind in ProviderKind
    }
    return BrowserRouter(
        settings=settings or RouterSettings(),
        lightweight=cast(LightweightProvider, providers[ProviderKind.LIGHTWEIGHT]),
        crawl4ai=cast(Crawl4AIProvider, providers[ProviderKind.CRAWL4AI]),
        playwright=cast(PlaywrightProvider, providers[ProviderKind.PLAYWRIGHT]),
        browser_use=cast(BrowserUseProvider, providers[ProviderKind.BROWSER_USE]),
        firecrawl=cast(FirecrawlProvider, providers[ProviderKind.FIRECRAWL]),
    )


@pytest.mark.asyncio
async def test_router_returns_successful_result_and_records_metrics() -> None:
    """Test router fallback to successful provider and metrics recording."""

    failing_lightweight = StubProvider(
        ProviderContext(ProviderKind.LIGHTWEIGHT),
        responses=[False],
    )
    succeeding_crawl4ai = StubProvider(
        ProviderContext(ProviderKind.CRAWL4AI),
        responses=[True],
    )
    router = _build_router(
        overrides={
            ProviderKind.LIGHTWEIGHT: failing_lightweight,
            ProviderKind.CRAWL4AI: succeeding_crawl4ai,
        }
    )

    result = await router.scrape(ScrapeRequest(url="https://example.com"))

    assert result.provider is ProviderKind.CRAWL4AI
    assert failing_lightweight.calls == 1
    assert succeeding_crawl4ai.calls == 1
    # Router no longer exposes counters; success determined by provider result


@pytest.mark.asyncio
async def test_scrape_respects_explicit_provider_override() -> None:
    """Test explicit provider override in scrape request."""

    captured_calls: list[ProviderKind] = []
    browser_use = StubProvider(
        ProviderContext(ProviderKind.BROWSER_USE),
        responses=[True],
        call_log=captured_calls,
    )
    router = _build_router(overrides={ProviderKind.BROWSER_USE: browser_use})

    result = await router.scrape(
        ScrapeRequest(url="https://example.com", provider=ProviderKind.BROWSER_USE)
    )

    assert result.provider is ProviderKind.BROWSER_USE
    assert captured_calls == [ProviderKind.BROWSER_USE]


@pytest.mark.asyncio
async def test_require_interaction_skips_lightweight_and_crawl4ai() -> None:
    """Test interaction requirement skips lightweight and Crawl4AI providers."""

    call_log: list[ProviderKind] = []
    overrides = {
        kind: StubProvider(
            ProviderContext(kind),
            responses=[kind is ProviderKind.PLAYWRIGHT],
            call_log=call_log,
        )
        for kind in ProviderKind
    }
    router = _build_router(overrides=overrides)

    result = await router.scrape(
        ScrapeRequest(url="https://example.com", require_interaction=True)
    )

    assert result.provider is ProviderKind.PLAYWRIGHT
    assert ProviderKind.LIGHTWEIGHT not in call_log
    assert ProviderKind.CRAWL4AI not in call_log


@pytest.mark.asyncio
async def test_hard_domain_reorders_providers() -> None:
    """Test hard domain configuration reorders provider priority."""

    call_log: list[ProviderKind] = []
    overrides = {
        kind: StubProvider(
            ProviderContext(kind),
            responses=[kind is ProviderKind.PLAYWRIGHT],
            call_log=call_log,
        )
        for kind in ProviderKind
    }
    settings = RouterSettings(hard_domains=["example.com"])
    router = _build_router(settings=settings, overrides=overrides)

    result = await router.scrape(ScrapeRequest(url="https://news.example.com"))

    assert result.provider is ProviderKind.PLAYWRIGHT
    assert call_log[0] is ProviderKind.PLAYWRIGHT
    assert ProviderKind.LIGHTWEIGHT not in call_log


@pytest.mark.asyncio
async def test_router_skips_unavailable_providers_after_failed_init() -> None:
    """Test router skips providers that fail initialization."""

    unavailable_lightweight = StubProvider(
        ProviderContext(ProviderKind.LIGHTWEIGHT),
        responses=[True],
        init_failures=1,
    )
    succeeding_crawl4ai = StubProvider(
        ProviderContext(ProviderKind.CRAWL4AI),
        responses=[True],
    )
    router = _build_router(
        overrides={
            ProviderKind.LIGHTWEIGHT: unavailable_lightweight,
            ProviderKind.CRAWL4AI: succeeding_crawl4ai,
        }
    )

    await router.initialize()
    result = await router.scrape(ScrapeRequest(url="https://example.com"))

    assert result.provider is ProviderKind.CRAWL4AI
    assert unavailable_lightweight.calls == 0
    assert succeeding_crawl4ai.calls == 1


@pytest.mark.asyncio
async def test_router_reinitializes_provider_after_backoff() -> None:
    """Provider should recover once retry window elapses."""

    recovering_lightweight = StubProvider(
        ProviderContext(ProviderKind.LIGHTWEIGHT),
        responses=[True],
        init_failures=1,
    )
    backup_crawl4ai = StubProvider(
        ProviderContext(ProviderKind.CRAWL4AI),
        responses=[True, False],
    )
    router = _build_router(
        settings=RouterSettings(unavailable_retry_seconds=0.1),
        overrides={
            ProviderKind.LIGHTWEIGHT: recovering_lightweight,
            ProviderKind.CRAWL4AI: backup_crawl4ai,
        },
    )

    await router.initialize()

    first = await router.scrape(ScrapeRequest(url="https://example.com/first"))
    assert first.provider is ProviderKind.CRAWL4AI
    assert recovering_lightweight.calls == 0

    await asyncio.sleep(0.15)
    second = await router.scrape(ScrapeRequest(url="https://example.com/second"))
    assert second.provider is ProviderKind.LIGHTWEIGHT
    assert recovering_lightweight.calls == 1


@pytest.mark.asyncio
async def test_router_handles_rate_limited_provider(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Router should fall back when a provider is rate limited."""

    lightweight = StubProvider(
        ProviderContext(ProviderKind.LIGHTWEIGHT),
        responses=[True],
    )
    crawl4ai = StubProvider(
        ProviderContext(ProviderKind.CRAWL4AI),
        responses=[True],
    )
    settings = RouterSettings(
        rate_limits={
            "lightweight": RateLimitConfig(max_requests=1, period_seconds=10),
            "crawl4ai": RateLimitConfig(max_requests=1, period_seconds=10),
        }
    )
    router = _build_router(
        settings=settings,
        overrides={
            ProviderKind.LIGHTWEIGHT: lightweight,
            ProviderKind.CRAWL4AI: crawl4ai,
        },
    )

    async def fake_acquire(provider: ProviderKind, deadline: float) -> bool:  # noqa: ARG001
        return provider is not ProviderKind.LIGHTWEIGHT

    monkeypatch.setattr(router, "_acquire_limiter", fake_acquire)

    result = await router.scrape(ScrapeRequest(url="https://example.com"))

    assert result.provider is ProviderKind.CRAWL4AI
    # No metrics snapshot; validate provider selection only


@pytest.mark.asyncio
async def test_router_raises_after_all_providers_fail() -> None:
    """Test router raises error when all providers fail."""

    overrides = {
        kind: StubProvider(ProviderContext(kind), responses=[False])
        for kind in ProviderKind
    }
    router = _build_router(overrides=overrides)

    with pytest.raises(BrowserRouterError) as exc_info:
        await router.scrape(ScrapeRequest(url="https://example.com"))

    assert exc_info.value.context["attempted_providers"]


@pytest.mark.asyncio
async def test_router_continues_after_timeout_exception() -> None:
    """Providers raising TimeoutError should be treated as failures with fallback."""

    timing_out = StubProvider(
        ProviderContext(ProviderKind.LIGHTWEIGHT),
        responses=[TimeoutError("budget exceeded")],
    )
    succeeding = StubProvider(
        ProviderContext(ProviderKind.CRAWL4AI),
        responses=[True],
    )
    router = _build_router(
        overrides={
            ProviderKind.LIGHTWEIGHT: timing_out,
            ProviderKind.CRAWL4AI: succeeding,
        }
    )

    result = await router.scrape(ScrapeRequest(url="https://example.com"))

    assert result.provider is ProviderKind.CRAWL4AI
    assert timing_out.calls == 1
    assert succeeding.calls == 1
