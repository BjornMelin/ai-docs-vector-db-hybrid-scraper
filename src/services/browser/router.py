"""
Single automation router for 5-tier browser automation.

Tiers:
  T0: Lightweight (httpx + Trafilatura)
  T1: Crawl4AI (static + light dynamic)
  T2: Playwright (dynamic, scripted)
  T3: browser-use (agentic actions)
  T4: Firecrawl v2 (cloud, anti-bot, site-wide)

Replaces legacy routers in this package.

This module exposes a single entry point (`AutomationRouter.scrape`) that
selects the optimal provider with fallbacks, bounded by a request-level
deadline and per-provider rate limiting.

Notes:
    - Rate limiting uses aiolimiter's `AsyncLimiter(max_rate, time_period)`
      API (positional arguments) for compatibility across versions.
    - Structural pattern matching is used for exhaustive tier dispatch.
"""

from __future__ import annotations

import asyncio
import logging
import math
import time
from dataclasses import dataclass
from typing import Any, Final, Literal, cast
from urllib.parse import urlparse


try:
    from pydantic import BaseModel, Field
except ModuleNotFoundError as exc:  # pragma: no cover - import guard
    raise ImportError(
        "pydantic is required for browser routing models. "
        "Install with `pip install pydantic`."
    ) from exc

from ...config.loader import Settings
from ...config.models import AutomationRouterConfig
from ..errors import CrawlServiceError
from .browser_use_adapter import BrowserUseAdapter
from .firecrawl_adapter import FirecrawlAdapter, FirecrawlAdapterConfig
from .lightweight_scraper import LightweightScraper
from .playwright_adapter import PlaywrightAdapter


try:  # pragma: no cover - optional dependency
    from .crawl4ai_adapter import Crawl4AIAdapter
except ImportError:  # pragma: no cover - optional dependency
    Crawl4AIAdapter = None  # type: ignore[assignment]


try:  # pragma: no cover - import guard
    from aiolimiter import AsyncLimiter
except ModuleNotFoundError as exc:  # pragma: no cover - import guard
    raise ImportError(
        "aiolimiter is required for router rate limiting. "
        "Install with `pip install aiolimiter`."
    ) from exc

logger = logging.getLogger(__name__)

TierName = Literal["lightweight", "crawl4ai", "playwright", "browser_use", "firecrawl"]

DEFAULT_RATE_LIMITS: Final[dict[TierName, int]] = {
    "lightweight": 10,
    "crawl4ai": 5,
    "playwright": 2,
    "browser_use": 1,
    "firecrawl": 5,
}


class ScrapeRequest(BaseModel):
    """Normalized request payload.

    Attributes:
        url: Target URL to fetch.
        timeout_ms: End-to-end time budget in milliseconds. Non-negative.
        tier: Specific tier to force, or "auto" for router selection.
        interaction_required: True if scripted interactions are needed.
        custom_actions: Optional provider-specific actions.
    """

    url: str
    timeout_ms: int = Field(default=30_000, ge=0)
    tier: TierName | Literal["auto"] = Field(default="auto")
    interaction_required: bool = Field(default=False)
    custom_actions: list[dict[str, Any]] | None = Field(default=None)


@dataclass(slots=True)
class _AttemptResult:
    """Result of a single provider attempt.

    Attributes:
        success: True if provider reported success.
        data: Raw provider payload.
        provider: Provider tier name.
        elapsed_ms: Time spent in milliseconds.
    """

    success: bool
    data: dict[str, Any]
    provider: TierName
    elapsed_ms: int


class AutomationRouter:  # pylint: disable=too-many-instance-attributes
    """Route scraping requests to the appropriate tier with fallback.

    The router enforces a global deadline per request and per-tier rate limits.
    """

    def __init__(self, settings: Settings) -> None:
        """Initialize the router and providers."""

        # Providers
        self._t0 = LightweightScraper(settings)
        if Crawl4AIAdapter is None:
            self._t1 = None
            logger.warning("Crawl4AI adapter unavailable; disabling crawl4ai tier")
        else:
            self._t1 = Crawl4AIAdapter(settings.crawl4ai)
        self._t2 = PlaywrightAdapter(settings.playwright)
        self._t3 = BrowserUseAdapter(settings.browser_use)
        self._t4 = FirecrawlAdapter(
            FirecrawlAdapterConfig(
                api_key=getattr(
                    getattr(settings, "firecrawl", object()), "api_key", ""
                ),
                api_url=getattr(
                    getattr(settings, "firecrawl", object()), "api_url", None
                ),
            )
        )

        router_cfg = getattr(settings, "automation_router", AutomationRouterConfig())
        if not isinstance(router_cfg, AutomationRouterConfig):
            router_cfg = AutomationRouterConfig()
        self._router_config = router_cfg
        self._hard_domains: Final[frozenset[str]] = frozenset(router_cfg.hard_domains)

        limiter_period = router_cfg.limiter_period_seconds
        rate_limits: dict[TierName, int] = DEFAULT_RATE_LIMITS.copy()
        for tier_name, rate in router_cfg.rate_limits.items():
            if tier_name not in rate_limits:
                logger.warning(
                    "Ignoring unknown automation tier %s in rate_limits", tier_name
                )
                continue
            if rate > 0:
                rate_limits[cast(TierName, tier_name)] = rate

        # Simple per-tier limiters (req/s). Use positional args for compatibility.
        self._limiter: dict[TierName, AsyncLimiter] = {
            tier: AsyncLimiter(rate, limiter_period)
            for tier, rate in rate_limits.items()
        }
        self._metrics: dict[TierName, dict[str, int]] = {
            tier: {"success": 0, "failure": 0, "rate_limited": 0}
            for tier in rate_limits
        }

    async def initialize(self) -> None:
        """Initialize underlying providers that require setup."""

        await self._t4.initialize()

    async def cleanup(self) -> None:
        """Cleanup underlying providers that require teardown."""

        await self._t4.cleanup()

    # Public entry -------------------------------------------------------

    async def scrape(self, req: ScrapeRequest) -> dict[str, Any]:
        """Scrape with tier selection and fallback.

        Args:
            req: Normalized scrape request.

        Returns:
            Provider payload on success, or error structure on failure.
        """

        url = req.url
        deadline = self._deadline(req.timeout_ms)
        tiers = self._choose_tiers(req)

        last_error: str | None = None
        for name in tiers:
            if self._expired(deadline):
                logger.debug("Deadline expired before tier %s for %s", name, url)
                break
            try:
                result = await self._attempt(name, req, deadline)
            except CrawlServiceError as err:
                last_error = str(err)
                logger.debug("Tier %s failed for %s: %s", name, url, err)
                self._metrics[name]["failure"] += 1
                continue

            if result.success:
                self._metrics[name]["success"] += 1
                payload = dict(result.data)
                payload["provider"] = name
                payload["automation_time_ms"] = result.elapsed_ms
                return payload
            self._metrics[name]["failure"] += 1
            last_error = result.data.get("error") or last_error

        return {
            "success": False,
            "url": url,
            "error": last_error or "All tiers failed or timed out",
            "provider": None,
        }

    def get_metrics_snapshot(self) -> dict[str, dict[str, int]]:
        """Return a copy of per-tier attempt metrics."""

        return {tier: stats.copy() for tier, stats in self._metrics.items()}

    # Internals ----------------------------------------------------------

    def _choose_tiers(self, req: ScrapeRequest) -> list[TierName]:
        """Compute an ordered list of tiers to try.

        Args:
            req: Normalized scrape request.

        Returns:
            Ordered list of tier names to attempt.
        """

        if req.tier != "auto":
            return [cast(TierName, req.tier)]

        domain = urlparse(req.url).hostname or ""

        if any(domain.endswith(d) for d in self._hard_domains):
            return ["firecrawl", "crawl4ai", "playwright", "lightweight", "browser_use"]

        if req.interaction_required or (req.custom_actions is not None):
            return ["playwright", "browser_use", "crawl4ai", "firecrawl", "lightweight"]

        # Default fast path
        return ["lightweight", "crawl4ai", "firecrawl", "playwright", "browser_use"]

    async def _attempt(
        self, name: TierName, req: ScrapeRequest, deadline: float
    ) -> _AttemptResult:
        """Attempt one provider within the remaining budget.

        Args:
            name: Provider tier to invoke.
            req: Normalized scrape request.
            deadline: Absolute deadline (monotonic seconds).

        Returns:
            Attempt result with timing.
        """

        remaining_before = self._remaining_ms(deadline)
        if remaining_before <= 0:
            raise CrawlServiceError(f"Deadline exceeded before invoking {name}")

        if remaining_before < self._router_config.min_attempt_ms:
            msg = (
                f"Remaining budget {remaining_before} ms is below minimum "
                f"{self._router_config.min_attempt_ms} ms for {name}"
            )
            raise CrawlServiceError(msg)

        per_try_ms = min(self._router_config.per_attempt_cap_ms, remaining_before)
        limiter_timeout_ms = min(
            remaining_before, self._router_config.limiter_acquire_timeout_ms
        )
        if limiter_timeout_ms <= 0:
            raise CrawlServiceError(
                f"No budget available to acquire limiter for {name}"
            )

        limiter = self._limiter[name]
        start = time.perf_counter()
        try:
            await asyncio.wait_for(limiter.acquire(), timeout=limiter_timeout_ms / 1000)
            remaining_after = self._remaining_ms(deadline)
            if remaining_after <= 0:
                raise CrawlServiceError(f"Deadline expired before invoking {name}")
            per_try_ms = min(per_try_ms, remaining_after)
            if per_try_ms <= 0:
                raise CrawlServiceError(
                    f"Insufficient budget after limiter acquisition for {name}"
                )
            data = await self._invoke_provider(name, req, per_try_ms)
        except TimeoutError as exc:
            self._metrics[name]["rate_limited"] += 1
            raise CrawlServiceError(f"Rate limit wait exceeded for {name}") from exc
        except CrawlServiceError:
            raise
        except Exception as exc:  # pragma: no cover - defensive trace
            raise CrawlServiceError(str(exc)) from exc

        elapsed = int((time.perf_counter() - start) * 1000)
        ok = bool(data.get("success"))
        return _AttemptResult(ok, data, name, elapsed)

    async def _invoke_provider(
        self, name: TierName, req: ScrapeRequest, timeout_ms: int
    ) -> dict[str, Any]:
        """Dispatch the request to the desired provider."""

        match name:
            case "lightweight":
                content = await self._t0.scrape(req.url, timeout_ms=timeout_ms)
                if content is None:
                    return {
                        "success": False,
                        "url": req.url,
                        "error": "Lightweight scraper returned no content",
                    }
                payload = content.model_dump()
                payload.setdefault("success", True)
                return payload
            case "crawl4ai":
                if self._t1 is None:
                    raise CrawlServiceError(
                        "Crawl4AI adapter not available; "
                        "install optional crawling extras"
                    )
                return cast(
                    dict[str, Any],
                    await self._t1.scrape(
                        req.url,
                        _timeout=timeout_ms,
                    ),
                )
            case "playwright":
                timeout_s = max(1, math.ceil(timeout_ms / 1000))
                return cast(
                    dict[str, Any],
                    await self._t2.scrape(
                        req.url,
                        actions=req.custom_actions or [],
                        timeout=timeout_s,
                    ),
                )
            case "browser_use":
                return cast(
                    dict[str, Any],
                    await self._t3.scrape(
                        url=req.url,
                        task=self._browser_use_task(req),
                        timeout=timeout_ms,
                        instructions=req.custom_actions or None,
                    ),
                )
            case "firecrawl":
                timeout_s = max(1, math.ceil(timeout_ms / 1000))
                return cast(
                    dict[str, Any],
                    await self._t4.scrape(
                        req.url,
                        formats=["markdown", "html"],
                        timeout=timeout_s,
                    ),
                )
        msg = f"Unknown automation tier: {name}"
        raise CrawlServiceError(msg)

    @staticmethod
    def _browser_use_task(req: ScrapeRequest) -> str:
        """Build a baseline task prompt for browser-use automation."""

        base_task = (
            "Navigate to the page and extract comprehensive documentation "
            "content, including code examples and metadata."
        )
        if req.custom_actions:
            return (
                "Execute the provided browser automation instructions and "
                "return comprehensive documentation content."
            )
        if req.interaction_required:
            return (
                "Perform the necessary interactions to surface dynamic content "
                "and extract comprehensive documentation details."
            )
        return base_task

    @staticmethod
    def _deadline(timeout_ms: int) -> float:
        """Compute absolute deadline in monotonic seconds."""

        return time.monotonic() + max(timeout_ms, 0) / 1000

    @staticmethod
    def _remaining_ms(deadline: float) -> int:
        """Milliseconds remaining until deadline."""

        return max(0, int((deadline - time.monotonic()) * 1000))

    @staticmethod
    def _expired(deadline: float) -> bool:
        """True if the deadline has passed."""

        return time.monotonic() >= deadline
