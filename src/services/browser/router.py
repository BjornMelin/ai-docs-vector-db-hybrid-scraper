"""Browser router orchestrating provider execution."""

from __future__ import annotations

import asyncio
import logging
import time
from collections import OrderedDict
from collections.abc import Sequence
from urllib.parse import urlparse

from aiolimiter import AsyncLimiter

from src.config.browser import RouterSettings
from src.services.browser.errors import BrowserProviderError, BrowserRouterError
from src.services.browser.models import BrowserResult, ProviderKind, ScrapeRequest
from src.services.browser.providers import (
    BrowserProvider,
    BrowserUseProvider,
    Crawl4AIProvider,
    FirecrawlProvider,
    LightweightProvider,
    PlaywrightProvider,
)
from src.services.browser.telemetry import MetricsRecorder


DEFAULT_ORDER: Sequence[ProviderKind] = (
    ProviderKind.LIGHTWEIGHT,
    ProviderKind.CRAWL4AI,
    ProviderKind.PLAYWRIGHT,
    ProviderKind.BROWSER_USE,
    ProviderKind.FIRECRAWL,
)


class BrowserRouter:
    """Routes scrape requests across multiple providers."""

    def __init__(
        self,
        *,
        settings: RouterSettings,
        lightweight: LightweightProvider,
        crawl4ai: Crawl4AIProvider,
        playwright: PlaywrightProvider,
        browser_use: BrowserUseProvider,
        firecrawl: FirecrawlProvider,
    ) -> None:
        self._settings = settings
        self._providers: OrderedDict[ProviderKind, BrowserProvider] = OrderedDict(
            {
                ProviderKind.LIGHTWEIGHT: lightweight,
                ProviderKind.CRAWL4AI: crawl4ai,
                ProviderKind.PLAYWRIGHT: playwright,
                ProviderKind.BROWSER_USE: browser_use,
                ProviderKind.FIRECRAWL: firecrawl,
            }
        )
        self._limiters: dict[ProviderKind, AsyncLimiter] = {}
        for key, conf in self._settings.rate_limits.items():
            try:
                provider_kind = ProviderKind(key)
            except ValueError:
                continue
            self._limiters[provider_kind] = AsyncLimiter(
                conf.max_requests, conf.period_seconds
            )
        self._metrics = MetricsRecorder()
        self._initialized = False
        self._retry_backoff = settings.unavailable_retry_seconds
        self._unavailable: dict[ProviderKind, float] = {}
        self._logger = logging.getLogger(__name__)

    async def initialize(self) -> None:
        """Initialize all providers."""

        if self._initialized:
            return
        for provider in self._providers.values():
            try:
                await provider.initialize()
            except BrowserProviderError as exc:
                self._mark_unavailable(
                    provider.kind,
                    reason=f"initialization failed: {exc}",
                )
        self._initialized = True

    async def cleanup(self) -> None:
        """Cleanup all providers."""

        for provider in self._providers.values():
            await provider.close()
        self._initialized = False
        self._unavailable.clear()

    def get_metrics_snapshot(self) -> dict[str, dict[str, int]]:
        """Expose telemetry snapshot."""

        return self._metrics.snapshot()

    def is_initialized(self) -> bool:
        """Return True when providers are initialized."""

        return self._initialized

    def get_provider(self, kind: ProviderKind) -> BrowserProvider | None:
        """Return the provider instance for a given kind."""

        return self._providers.get(kind)

    def _deadline(self, request: ScrapeRequest) -> float:
        timeout_ms = request.timeout_ms or self._settings.per_attempt_cap_ms * len(
            DEFAULT_ORDER
        )
        return time.monotonic() + timeout_ms / 1000

    def _remaining_ms(self, deadline: float) -> int:
        return max(0, int((deadline - time.monotonic()) * 1000))

    def _choose_order(self, request: ScrapeRequest) -> list[ProviderKind]:
        if request.provider:
            return [request.provider]

        parsed = urlparse(request.url)
        domain = parsed.hostname or ""

        order = list(DEFAULT_ORDER)
        if request.require_interaction:
            order = [
                kind
                for kind in order
                if kind not in {ProviderKind.LIGHTWEIGHT, ProviderKind.CRAWL4AI}
            ]

        if domain and any(
            domain.endswith(hard) for hard in self._settings.hard_domains
        ):
            order = [
                kind
                for kind in order
                if kind not in {ProviderKind.LIGHTWEIGHT, ProviderKind.CRAWL4AI}
            ] + [ProviderKind.CRAWL4AI]

        return order

    async def _acquire_limiter(self, provider: ProviderKind, deadline: float) -> bool:
        limiter = self._limiters.get(provider)
        if limiter is None:
            return True
        remaining_ms = self._remaining_ms(deadline)
        timeout = min(
            remaining_ms / 1000,
            self._settings.per_attempt_cap_ms / 1000,
        )
        if timeout <= 0:
            return False
        try:
            await asyncio.wait_for(limiter.acquire(), timeout=timeout)
            return True
        except TimeoutError:
            self._metrics.record_rate_limited(provider)
            return False

    async def scrape(self, request: ScrapeRequest) -> BrowserResult:
        """Execute a scrape with tier fallbacks."""

        if not self._initialized:
            await self.initialize()

        deadline = self._deadline(request)
        attempted: list[str] = []
        for provider_kind in self._choose_order(request):
            provider = self._providers.get(provider_kind)
            if provider is None:
                continue
            if not await self._ensure_provider_available(provider_kind):
                continue
            if self._remaining_ms(deadline) < self._settings.min_budget_ms:
                break
            acquired = await self._acquire_limiter(provider_kind, deadline)
            if not acquired:
                continue

            attempted.append(provider_kind.value)
            try:
                result = await provider.run(request)
            except BrowserProviderError:
                self._metrics.record_failure(provider_kind)
                continue

            if result.success:
                self._metrics.record_success(provider_kind)
                return result

            self._metrics.record_failure(provider_kind)

        raise BrowserRouterError(
            "All providers failed or timed out",
            attempted_providers=attempted,
        )

    def _mark_unavailable(
        self, provider_kind: ProviderKind, *, reason: str | None = None
    ) -> None:
        next_retry = time.monotonic() + self._retry_backoff
        self._unavailable[provider_kind] = next_retry
        if reason:
            self._logger.warning(
                "Provider %s marked unavailable for %.1fs (%s)",
                provider_kind.value,
                self._retry_backoff,
                reason,
            )
        else:
            self._logger.warning(
                "Provider %s marked unavailable for %.1fs",
                provider_kind.value,
                self._retry_backoff,
            )

    async def _ensure_provider_available(self, provider_kind: ProviderKind) -> bool:
        retry_after = self._unavailable.get(provider_kind)
        if retry_after is None:
            return True
        now = time.monotonic()
        if retry_after > now:
            self._logger.debug(
                "Provider %s unavailable for another %.1fs",
                provider_kind.value,
                retry_after - now,
            )
            return False

        provider = self._providers.get(provider_kind)
        if provider is None:
            return False
        try:
            await provider.initialize()
        except BrowserProviderError as exc:
            self._mark_unavailable(
                provider_kind,
                reason=f"retry failed: {exc}",
            )
            return False

        self._unavailable.pop(provider_kind, None)
        self._logger.info("Provider %s recovered after retry", provider_kind.value)
        return True
