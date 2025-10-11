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

from dependency_injector import containers, providers

from ...config.loader import Settings
from ...config.models import AutomationRouterConfig
from ..errors import CrawlServiceError
from ..observability.tracing import get_tracer
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

        self._container = _AutomationRouterContainer(settings=settings)
        self._router_config = self._container.router_config()
        self._hard_domains: Final[frozenset[str]] = frozenset(
            self._router_config.hard_domains
        )

        # Providers
        self._t0 = self._container.lightweight_scraper()
        self._t1 = self._container.crawl4ai_adapter()
        if self._t1 is None:
            logger.warning("Crawl4AI adapter unavailable; disabling crawl4ai tier")
        self._t2 = self._container.playwright_adapter()
        self._t3 = self._container.browser_use_adapter()
        self._t4 = self._container.firecrawl_adapter()

        self._tracer = get_tracer("ai-docs.automation-router")

        limiter_period = self._router_config.limiter_period_seconds
        rate_limits: dict[TierName, int] = DEFAULT_RATE_LIMITS.copy()
        for tier_name, rate in self._router_config.rate_limits.items():
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
        with self._tracer.start_as_current_span(
            "automation_router.scrape",
            attributes={
                "automation_router.url": url,
                "automation_router.requested_tier": req.tier,
            },
        ) as span:
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
                    if span.is_recording():
                        span.set_attribute("automation_router.final_tier", name)
                    return payload
                self._metrics[name]["failure"] += 1
                last_error = result.data.get("error") or last_error

            failure_payload = {
                "success": False,
                "url": url,
                "error": last_error or "All tiers failed or timed out",
                "provider": None,
            }
            if span.is_recording():
                span.set_attribute("automation_router.final_tier", "none")
                span.set_attribute("automation_router.failed_tiers", ",".join(tiers))
            return failure_payload

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

        parsed = urlparse(req.url)
        domain = parsed.hostname or ""
        path = parsed.path or ""

        with self._tracer.start_as_current_span(
            "automation_router.choose_tiers",
            attributes={
                "automation_router.domain": domain or "unknown",
                "automation_router.request.tier": "auto",
            },
        ) as span:
            domain_complexity = self._analyze_domain_complexity(domain, path)
            js_required = self._predict_js_requirements(domain, path, parsed.query)
            hard_domain = any(domain.endswith(d) for d in self._hard_domains)
            anti_detection = self._assess_anti_detection_needs(
                domain, path, parsed.query
            )
            analysis: dict[str, Any] = {
                "domain": domain,
                "path": path,
                "domain_complexity": domain_complexity,
                "js_required": js_required,
                "anti_detection": anti_detection,
                "hard_domain": hard_domain,
                "interaction_required": req.interaction_required,
                "custom_actions": bool(req.custom_actions),
            }

            tiers = self._select_optimal_tier(analysis)

            if span.is_recording():
                span.set_attribute(
                    "automation_router.domain_complexity", domain_complexity
                )
                span.set_attribute("automation_router.js_required", js_required)
                span.set_attribute("automation_router.hard_domain", hard_domain)
                span.set_attribute("automation_router.anti_detection", anti_detection)
                span.set_attribute(
                    "automation_router.interaction_required",
                    req.interaction_required,
                )
                span.set_attribute(
                    "automation_router.custom_actions", bool(req.custom_actions)
                )
                span.set_attribute("automation_router.tier_sequence", ",".join(tiers))

            return tiers

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

    def _analyze_domain_complexity(self, domain: str, path: str) -> str:
        """Estimate domain complexity using lightweight heuristics."""

        if not domain:
            return "low"

        normalized_domain = domain.lower()
        domain_tokens = normalized_domain.split(".")
        path_tokens = [segment for segment in path.lower().split("/") if segment]

        complexity_score = 0
        complex_indicators = {
            "auth",
            "login",
            "admin",
            "portal",
            "dashboard",
            "api",
        }
        if any(indicator in domain_tokens for indicator in complex_indicators):
            complexity_score += 2
        if len(domain_tokens) > 3:
            complexity_score += 1
        if any(segment in complex_indicators for segment in path_tokens):
            complexity_score += 1
        if any(
            segment in {"checkout", "settings", "account"} for segment in path_tokens
        ):
            complexity_score += 1

        if complexity_score >= 3:
            return "high"
        if complexity_score >= 1:
            return "medium"
        return "low"

    @staticmethod
    def _predict_js_requirements(domain: str, path: str, query: str) -> bool:
        """Predict whether JavaScript execution is likely required."""

        haystack = f"{domain.lower()} {path.lower()} {query.lower()}"
        js_indicators = {
            "app",
            "spa",
            "dashboard",
            "portal",
            "react",
            "vue",
            "next",
            "angular",
        }
        return any(indicator in haystack for indicator in js_indicators)

    def _assess_anti_detection_needs(self, domain: str, path: str, query: str) -> bool:
        """Determine if enhanced anti-detection handling is recommended."""

        if not domain:
            return False

        haystack = f"{domain.lower()} {path.lower()} {query.lower()}"
        anti_bot_terms = {
            "cloudflare",
            "captcha",
            "akamai",
            "incapsula",
            "bot",
            "secure",
        }

        return any(term in haystack for term in anti_bot_terms)

    def _select_optimal_tier(self, analysis: dict[str, Any]) -> list[TierName]:
        """Select the preferred tier ordering based on analysis results."""

        base_order: list[TierName] = [
            "lightweight",
            "crawl4ai",
            "playwright",
            "browser_use",
            "firecrawl",
        ]

        if analysis["anti_detection"] or analysis["hard_domain"]:
            preferred: list[TierName] = [
                "firecrawl",
                "playwright",
                "browser_use",
                "crawl4ai",
                "lightweight",
            ]
        elif analysis["custom_actions"] or analysis["interaction_required"]:
            preferred = [
                "playwright",
                "browser_use",
                "crawl4ai",
                "firecrawl",
                "lightweight",
            ]
        elif analysis["js_required"] and analysis["domain_complexity"] in {"high", "medium"}:
            preferred = [
                "playwright",
                "crawl4ai",
                "browser_use",
                "firecrawl",
                "lightweight",
            ]
        elif analysis["js_required"]:
            preferred = [
                "crawl4ai",
                "playwright",
                "browser_use",
                "firecrawl",
                "lightweight",
            ]
        elif analysis["domain_complexity"] == "medium":
            preferred = [
                "crawl4ai",
                "lightweight",
                "playwright",
                "firecrawl",
                "browser_use",
            ]
        else:
            preferred = base_order

        seen: set[TierName] = set()
        ordered: list[TierName] = []
        for tier in preferred:
            if tier not in seen:
                ordered.append(tier)
                seen.add(tier)
        for tier in base_order:
            if tier not in seen:
                ordered.append(tier)
                seen.add(tier)
        return ordered

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


def _resolve_router_config(settings: Settings) -> AutomationRouterConfig:
    """Resolve router configuration from settings."""

    candidate = getattr(settings, "automation_router", AutomationRouterConfig())
    if isinstance(candidate, AutomationRouterConfig):
        return candidate
    if hasattr(candidate, "model_dump"):
        try:
            return AutomationRouterConfig.model_validate(candidate.model_dump())
        except ValueError:  # pragma: no cover - defensive validation
            logger.warning("Invalid automation router config; falling back to defaults")
            return AutomationRouterConfig()
    if isinstance(candidate, dict):
        try:
            return AutomationRouterConfig.model_validate(candidate)
        except ValueError:  # pragma: no cover - defensive validation
            logger.warning("Invalid automation router config; falling back to defaults")
            return AutomationRouterConfig()
    logger.warning("Unexpected automation router config type; using defaults")
    return AutomationRouterConfig()


def _create_crawl4ai_adapter(settings: Settings) -> Crawl4AIAdapter | None:
    """Instantiate the Crawl4AI adapter when available."""

    if Crawl4AIAdapter is None:
        return None
    crawl4ai_config = getattr(settings, "crawl4ai", None)
    if crawl4ai_config is None:
        logger.warning("Crawl4AI settings missing; adapter disabled")
        return None
    return Crawl4AIAdapter(crawl4ai_config)


def _create_firecrawl_adapter(settings: Settings) -> FirecrawlAdapter:
    """Instantiate the Firecrawl adapter from configuration."""

    firecrawl_settings = getattr(settings, "firecrawl", object())
    adapter_config = FirecrawlAdapterConfig(
        api_key=getattr(firecrawl_settings, "api_key", ""),
        api_url=getattr(firecrawl_settings, "api_url", None),
    )
    return FirecrawlAdapter(adapter_config)


def _create_playwright_adapter(settings: Settings) -> PlaywrightAdapter:
    """Instantiate the Playwright adapter using provided settings."""

    playwright_settings = getattr(settings, "playwright", None)
    if playwright_settings is None:
        msg = "Playwright settings missing; cannot initialize automation router"
        raise ValueError(msg)
    return PlaywrightAdapter(playwright_settings)


def _create_browser_use_adapter(settings: Settings) -> BrowserUseAdapter:
    """Instantiate the browser-use adapter using provided settings."""

    browser_use_settings = getattr(settings, "browser_use", None)
    if browser_use_settings is None:
        msg = "Browser-use settings missing; cannot initialize automation router"
        raise ValueError(msg)
    return BrowserUseAdapter(browser_use_settings)


class _AutomationRouterContainer(containers.DeclarativeContainer):  # type: ignore
    """Dependency-injector container for automation router components."""

    settings = providers.Dependency(instance_of=Settings)

    router_config = providers.Singleton(
        _resolve_router_config,
        settings=settings,
    )
    lightweight_scraper = providers.Singleton(
        LightweightScraper,
        settings=settings,
    )
    crawl4ai_adapter = providers.Singleton(
        _create_crawl4ai_adapter,
        settings=settings,
    )
    playwright_adapter = providers.Singleton(
        _create_playwright_adapter,
        settings=settings,
    )
    browser_use_adapter = providers.Singleton(
        _create_browser_use_adapter,
        settings=settings,
    )
    firecrawl_adapter = providers.Singleton(
        _create_firecrawl_adapter,
        settings=settings,
    )
