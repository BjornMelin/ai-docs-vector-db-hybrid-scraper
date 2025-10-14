"""Browser automation configuration models."""

from __future__ import annotations

from collections.abc import Sequence
from enum import Enum
from typing import Literal

from pydantic import BaseModel, Field, HttpUrl, model_validator


class StealthMode(str, Enum):
    """Stealth behaviour options for the Playwright provider."""

    DISABLED = "disabled"
    PLAYWRIGHT_STEALTH = "playwright_stealth"
    REBROWSER = "rebrowser"


class Crawl4AISettings(BaseModel):
    """Settings passed to the Crawl4AI provider."""

    headless: bool = Field(default=True, description="Run chromium in headless mode")
    browser_type: str = Field(default="chromium", description="Playwright browser type")
    viewport: dict[str, int] = Field(
        default_factory=lambda: {"width": 1280, "height": 1024},
        description="Viewport size forwarded to AsyncWebCrawler",
    )
    cache_mode: Literal["bypass", "enabled"] = Field(
        default="bypass", description="Crawler cache behaviour"
    )
    session_id: str | None = Field(
        default=None, description="Optional session identifier for Crawl4AI"
    )
    concurrency: int = Field(
        default=6, ge=1, le=32, description="Concurrent sessions tracked by dispatcher"
    )
    verbose: bool = Field(default=False, description="Enable Crawl4AI verbose logging")


class FirecrawlSettings(BaseModel):
    """Settings for the Firecrawl cloud provider."""

    api_key: str = Field(default="", description="Firecrawl API key (override env)")
    api_url: HttpUrl | None = Field(
        default=None, description="Optional custom Firecrawl endpoint"
    )
    default_formats: Sequence[str] = Field(
        default=("markdown", "html"),
        description="Default formats requested when none specified",
    )
    timeout_seconds: int = Field(
        default=120, gt=0, description="Timeout applied to Firecrawl waiter endpoints"
    )

    @model_validator(mode="after")
    def validate_api_key(self) -> FirecrawlSettings:
        api_key = self.model_dump().get("api_key")
        if isinstance(api_key, str) and api_key and not api_key.startswith("fc-"):
            msg = "Firecrawl API key must start with 'fc-'"
            raise ValueError(msg)
        return self


class CaptchaProvider(str, Enum):
    """Supported captcha solving providers."""

    NONE = "none"
    CAPMONSTER = "capmonster"


class PlaywrightSettings(BaseModel):
    """Settings for the Playwright provider."""

    browser: str = Field(default="chromium", description="Browser type")
    headless: bool = Field(default=True, description="Launch headless browser")
    timeout_ms: int = Field(default=30000, gt=0, description="Navigation timeout")
    stealth: StealthMode = Field(
        default=StealthMode.PLAYWRIGHT_STEALTH, description="Stealth strategy"
    )
    captcha_provider: CaptchaProvider = Field(
        default=CaptchaProvider.NONE, description="Captcha solving provider"
    )
    captcha_api_key: str | None = Field(
        default=None, description="API key for captcha provider if enabled"
    )
    challenges: Sequence[str] = Field(
        default=("captcha", "verify you are human", "access denied"),
        description="Keywords indicating bot-detection challenges",
    )

    @model_validator(mode="after")
    def validate_captcha(self) -> PlaywrightSettings:
        if (
            self.captcha_provider is not CaptchaProvider.NONE
            and not self.captcha_api_key
        ):
            msg = "captcha_api_key must be provided when captcha_provider != none"
            raise ValueError(msg)
        return self


class BrowserUseSettings(BaseModel):
    """Settings for the Browser-use provider."""

    llm_provider: Literal["openai", "anthropic", "gemini"] = Field(
        default="openai", description="Underlying LLM provider"
    )
    model: str = Field(
        default="gpt-4o-mini", description="Model identifier forwarded to provider"
    )
    timeout_ms: int = Field(default=60000, gt=0, description="Task timeout")
    max_retries: int = Field(default=2, ge=0, le=5, description="Retry attempts")
    headless: bool = Field(default=True, description="Run chromium headless")


class LightweightSettings(BaseModel):
    """Settings for the lightweight HTTP scraper."""

    timeout_seconds: float = Field(
        default=8.0, gt=0.1, description="HTTP client timeout"
    )
    allow_redirects: bool = Field(
        default=True, description="Follow redirects when scraping"
    )


class RateLimitConfig(BaseModel):
    """Per-provider rate limiting configuration."""

    max_requests: int = Field(default=5, gt=0)
    period_seconds: float = Field(default=1.0, gt=0)


class RouterSettings(BaseModel):
    """Settings driving tier ordering and rate limiting."""

    rate_limits: dict[str, RateLimitConfig] = Field(
        default_factory=lambda: {
            "lightweight": RateLimitConfig(max_requests=10, period_seconds=1),
            "crawl4ai": RateLimitConfig(max_requests=4, period_seconds=1),
            "playwright": RateLimitConfig(max_requests=2, period_seconds=1),
            "browser_use": RateLimitConfig(max_requests=1, period_seconds=1),
            "firecrawl": RateLimitConfig(max_requests=3, period_seconds=1),
        },
        description="Per-provider rate limiting buckets",
    )
    per_attempt_cap_ms: int = Field(
        default=20000,
        ge=500,
        description="Maximum duration allowed for a single provider attempt",
    )
    min_budget_ms: int = Field(
        default=500,
        ge=0,
        description="Minimum remaining deadline budget required to invoke a provider",
    )
    hard_domains: Sequence[str] = Field(
        default=(),
        description=(
            "Domains that should skip lightweight tiers and escalate immediately"
        ),
    )
    monitor_failures: bool = Field(
        default=True, description="Enable failure metrics tracking"
    )
    unavailable_retry_seconds: float = Field(
        default=60.0,
        ge=0.1,
        description="Backoff before retrying failed provider initialization",
    )


class BrowserAutomationConfig(BaseModel):
    """Aggregate configuration applied to browser orchestration."""

    lightweight: LightweightSettings = Field(
        default_factory=LightweightSettings,
        description="Lightweight HTTP scraper settings",
    )
    crawl4ai: Crawl4AISettings = Field(
        default_factory=Crawl4AISettings,
        description="Crawl4AI provider settings",
    )
    playwright: PlaywrightSettings = Field(
        default_factory=PlaywrightSettings,
        description="Playwright provider settings",
    )
    browser_use: BrowserUseSettings = Field(
        default_factory=BrowserUseSettings,
        description="Browser-use provider settings",
    )
    firecrawl: FirecrawlSettings = Field(
        default_factory=FirecrawlSettings,
        description="Firecrawl provider settings",
    )
    router: RouterSettings = Field(
        default_factory=RouterSettings,
        description="Router behaviour for tier selection and rate limits",
    )


__all__ = [
    "BrowserAutomationConfig",
    "BrowserUseSettings",
    "CaptchaProvider",
    "Crawl4AISettings",
    "FirecrawlSettings",
    "LightweightSettings",
    "PlaywrightSettings",
    "RateLimitConfig",
    "RouterSettings",
    "StealthMode",
]
