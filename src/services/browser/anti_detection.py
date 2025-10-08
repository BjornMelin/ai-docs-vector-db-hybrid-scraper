"""Enhanced Anti-Detection System for browser automation.

This module provides sophisticated anti-detection capabilities for browser automation,
including fingerprint management, behavioral patterns, and success rate monitoring.
Designed to achieve 95%+ success rate on challenging sites while maintaining
performance.

Note: Uses standard random module for anti-detection purposes
(timing, user agents, etc.)
This is intentional and not cryptographically sensitive.
"""
# Standard random is acceptable for anti-detection purposes

import logging
import random
from typing import Any, cast

from pydantic import BaseModel, Field

from src.config.models import PlaywrightConfig


logger = logging.getLogger(__name__)


class UserAgentPool(BaseModel):
    """Pool of realistic user agents with browser fingerprint matching."""

    chrome_agents: list[str] = Field(
        default_factory=lambda: [
            "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
            "(KHTML, like Gecko) Chrome/122.0.0.0 Safari/537.36",
            "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 "
            "(KHTML, like Gecko) Chrome/122.0.0.0 Safari/537.36",
            "Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 "
            "(KHTML, like Gecko) Chrome/122.0.0.0 Safari/537.36",
            "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
            "(KHTML, like Gecko) Chrome/121.0.0.0 Safari/537.36",
            "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 "
            "(KHTML, like Gecko) Chrome/121.0.0.0 Safari/537.36",
        ]
    )

    firefox_agents: list[str] = Field(
        default_factory=lambda: [
            "Mozilla/5.0 (Windows NT 10.0; Win64; x64; rv:122.0) "
            "Gecko/20100101 Firefox/122.0",
            "Mozilla/5.0 (Macintosh; Intel Mac OS X 10.15; rv:122.0) "
            "Gecko/20100101 Firefox/122.0",
            "Mozilla/5.0 (X11; Linux x86_64; rv:122.0) Gecko/20100101 Firefox/122.0",
            "Mozilla/5.0 (Windows NT 10.0; Win64; x64; rv:121.0) "
            "Gecko/20100101 Firefox/121.0",
        ]
    )

    safari_agents: list[str] = Field(
        default_factory=lambda: [
            "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/605.1.15 "
            "(KHTML, like Gecko) Version/17.2.1 Safari/605.1.15",
            "Mozilla/5.0 (iPhone; CPU iPhone OS 17_2_1 like Mac OS X) "
            "AppleWebKit/605.1.15 (KHTML, like Gecko) Version/17.2 "
            "Mobile/15E148 Safari/604.1",
        ]
    )


class ViewportProfile(BaseModel):
    """Viewport profile with common resolution patterns."""

    width: int = Field(ge=320, le=1920, description="Viewport width")
    height: int = Field(ge=568, le=1200, description="Viewport height")
    device_scale_factor: float = Field(default=1.0, ge=0.5, le=3.0)
    is_mobile: bool = Field(default=False)

    @classmethod
    def get_common_profiles(cls) -> list["ViewportProfile"]:
        """Get list of common viewport profiles."""
        return [
            cls(width=1366, height=768, device_scale_factor=1.0),  # Most common laptop
            cls(width=1920, height=1080, device_scale_factor=1.0),  # Full HD
            cls(width=1440, height=900, device_scale_factor=1.0),  # MacBook Pro 13"
            cls(width=1536, height=864, device_scale_factor=1.0),  # Surface Laptop
            cls(width=1280, height=720, device_scale_factor=1.0),  # HD
            cls(width=1600, height=900, device_scale_factor=1.0),  # 16:9 widescreen
            cls(
                width=375, height=667, device_scale_factor=2.0, is_mobile=True
            ),  # iPhone SE
            cls(
                width=414, height=896, device_scale_factor=2.0, is_mobile=True
            ),  # iPhone 11
        ]


class SiteProfile(BaseModel):
    """Site-specific anti-detection profile."""

    domain: str = Field(description="Domain or pattern this profile applies to")
    risk_level: str = Field(default="medium", pattern="^(low|medium|high|extreme)$")
    required_delay_ms: tuple[int, int] = Field(
        default=(1000, 3000), description="Min/max delay range"
    )
    stealth_level: str = Field(
        default="standard", pattern="^(basic|standard|advanced|maximum)$"
    )
    user_agent_rotation: bool = Field(default=True)
    viewport_randomization: bool = Field(default=True)
    behavioral_timing: bool = Field(default=True)
    canvas_fingerprint_protection: bool = Field(default=False)
    webgl_fingerprint_protection: bool = Field(default=False)


class TimingPattern(BaseModel):
    """Human-like timing patterns for interactions."""

    mouse_movement_delay: tuple[int, int] = Field(
        default=(50, 200), description="Mouse movement delay range (ms)"
    )
    click_delay: tuple[int, int] = Field(
        default=(100, 300), description="Click delay range (ms)"
    )
    typing_speed_wpm: tuple[int, int] = Field(
        default=(40, 80), description="Typing speed range (WPM)"
    )
    page_reading_time: tuple[int, int] = Field(
        default=(2000, 8000), description="Page reading time range (ms)"
    )
    scroll_speed_px_s: tuple[int, int] = Field(
        default=(100, 400), description="Scroll speed range (px/s)"
    )


class BrowserStealthConfig(BaseModel):
    """Enhanced browser configuration with anti-detection settings."""

    user_agent: str = Field(description="User agent string")
    viewport: ViewportProfile = Field(description="Viewport configuration")
    headers: dict[str, str] = Field(description="HTTP headers")
    extra_args: list[str] = Field(description="Additional browser arguments")
    timing: TimingPattern = Field(description="Timing patterns")
    javascript_enabled: bool = Field(default=True)
    images_enabled: bool = Field(default=True)
    css_enabled: bool = Field(default=True)
    languages: list[str] = Field(
        default_factory=lambda: ["en-US", "en"],
        description="Ordered list of preferred languages",
    )
    locale: str = Field(default="en-US", description="Primary locale identifier")
    timezone: str = Field(default="America/Los_Angeles", description="IANA time zone")
    platform: str = Field(default="Win32", description="Navigator platform value")
    client_hints_platform: str = Field(
        default="Windows", description="Sec-CH-UA-Platform header value"
    )
    webgl_vendor: str | None = Field(
        default=None, description="Override for UNMASKED_VENDOR_WEBGL"
    )
    webgl_renderer: str | None = Field(
        default=None, description="Override for UNMASKED_RENDERER_WEBGL"
    )


class SuccessRateMonitor(BaseModel):
    """Success rate monitoring and adaptive strategy adjustment."""

    total_attempts: int = Field(default=0)
    successful_attempts: int = Field(default=0)
    recent_successes: list[bool] = Field(default_factory=list, max_length=50)
    strategy_performance: dict[str, dict[str, Any]] = Field(default_factory=dict)

    @property
    def _total_attempts(self) -> int:
        """Backwards-compatible alias for total attempts."""
        return self.total_attempts

    @_total_attempts.setter
    def _total_attempts(self, value: int) -> None:
        self.total_attempts = value

    def record_attempt(self, success: bool, strategy: str = "default") -> None:
        """Record an attempt and update statistics."""
        self.total_attempts += 1
        if success:
            self.successful_attempts += 1

        # Maintain rolling window using pydantic-provided container.
        # pylint: disable=no-member
        recent_successes = cast(list[bool], self.recent_successes)
        recent_successes.append(success)
        if len(recent_successes) > 50:
            recent_successes.pop(0)

        # Update strategy-specific performance
        performance = cast(dict[str, dict[str, Any]], self.strategy_performance)
        if strategy not in performance:
            performance[strategy] = {
                "attempts": 0,
                "successes": 0,
                "recent_performance": [],
            }

        metrics = performance[strategy]
        metrics["attempts"] += 1
        if success:
            metrics["successes"] += 1

        # Keep recent performance window
        history = cast(list[bool], metrics["recent_performance"])
        history.append(success)
        if len(history) > 20:
            history.pop(0)

    def get_overall_success_rate(self) -> float:
        """Get overall success rate."""
        if self.total_attempts == 0:
            return 0.0
        return self.successful_attempts / self.total_attempts

    def get_recent_success_rate(self) -> float:
        """Get recent success rate (last 50 attempts)."""
        if not self.recent_successes:
            return 0.0
        return sum(self.recent_successes) / len(self.recent_successes)

    def get_strategy_success_rate(self, strategy: str) -> float:
        """Get success rate for specific strategy."""
        if strategy not in self.strategy_performance:
            return 0.0

        perf = self.strategy_performance[strategy]
        if perf["attempts"] == 0:
            return 0.0

        return perf["successes"] / perf["attempts"]

    def needs_strategy_adjustment(self) -> bool:
        """Check if strategy adjustment is needed based on recent performance."""
        recent_rate = self.get_recent_success_rate()
        return recent_rate < 0.8 and len(self.recent_successes) >= 10


class EnhancedAntiDetection:
    """Enhanced anti-detection system for browser automation."""

    def __init__(self) -> None:
        """Initialize the enhanced anti-detection system."""
        self.user_agents = UserAgentPool()
        self.viewport_profiles = ViewportProfile.get_common_profiles()
        self.success_monitor = SuccessRateMonitor()
        self.site_profiles: dict[str, SiteProfile] = {}
        self._load_default_site_profiles()

    def _load_default_site_profiles(self) -> None:
        """Load default site-specific profiles."""
        self.site_profiles.update(
            {
                "github.com": SiteProfile(
                    domain="github.com",
                    risk_level="high",
                    required_delay_ms=(2000, 5000),
                    stealth_level="advanced",
                    canvas_fingerprint_protection=True,
                ),
                "linkedin.com": SiteProfile(
                    domain="linkedin.com",
                    risk_level="extreme",
                    required_delay_ms=(3000, 8000),
                    stealth_level="maximum",
                    canvas_fingerprint_protection=True,
                    webgl_fingerprint_protection=True,
                ),
                "cloudflare.com": SiteProfile(
                    domain="cloudflare.com",
                    risk_level="extreme",
                    required_delay_ms=(5000, 10000),
                    stealth_level="maximum",
                    canvas_fingerprint_protection=True,
                    webgl_fingerprint_protection=True,
                ),
                # Default for other sites
                "default": SiteProfile(
                    domain="*",
                    risk_level="medium",
                    required_delay_ms=(1000, 3000),
                    stealth_level="standard",
                ),
            }
        )

    def get_stealth_config(self, site_profile: str = "default") -> BrowserStealthConfig:
        """Get stealth configuration for specified site profile.

        Args:
            site_profile: Site profile name or domain

        Returns:
            BrowserStealthConfig with anti-detection settings

        """
        # Get site profile or use default
        profile = self.site_profiles.get(site_profile, self.site_profiles["default"])

        user_agent, platform, ch_platform = self._rotate_user_agents()
        language_header, languages, locale = self._select_language_profile()
        headers = self._generate_realistic_headers(language_header, ch_platform)
        timezone = self._select_timezone(profile)
        viewport = self._randomize_viewport()
        vendor, renderer = self._select_webgl_overrides(profile, platform)

        return BrowserStealthConfig(
            user_agent=user_agent,
            viewport=viewport,
            headers=headers,
            extra_args=self._get_stealth_args(),
            timing=self._get_timing_pattern(profile),
            languages=languages,
            locale=locale,
            timezone=timezone,
            platform=platform,
            client_hints_platform=ch_platform,
            webgl_vendor=vendor,
            webgl_renderer=renderer,
        )

    def _rotate_user_agents(self) -> tuple[str, str, str]:
        """Rotate user agents with realistic browser signatures."""
        # Weighted selection favoring Chrome (most common)
        browser_weights = [0.65, 0.25, 0.10]  # Chrome, Firefox, Safari
        browser_pools = [
            self.user_agents.chrome_agents,
            self.user_agents.firefox_agents,
            self.user_agents.safari_agents,
        ]

        selected_pool = random.choices(browser_pools, weights=browser_weights)[  # noqa: S311
            0
        ]  # Anti-detection randomization
        user_agent = random.choice(selected_pool)  # noqa: S311  # Anti-detection randomization
        platform, ch_platform = self._derive_platform(user_agent)
        return user_agent, platform, ch_platform

    def _randomize_viewport(self) -> ViewportProfile:
        """Randomize viewport with common resolution patterns."""
        # Filter to non-mobile profiles for better compatibility
        desktop_profiles = [p for p in self.viewport_profiles if not p.is_mobile]
        profile = random.choice(desktop_profiles)  # noqa: S311  # Anti-detection randomization

        # Add slight randomization to avoid exact pattern matching
        width_variance = random.randint(-50, 50)  # noqa: S311  # Anti-detection randomization
        height_variance = random.randint(-30, 30)  # noqa: S311  # Anti-detection randomization

        return ViewportProfile(
            width=max(1200, min(1920, profile.width + width_variance)),
            height=max(720, min(1200, profile.height + height_variance)),
            device_scale_factor=profile.device_scale_factor,
            is_mobile=profile.is_mobile,
        )

    def _generate_realistic_headers(
        self, language_header: str, client_hints_platform: str
    ) -> dict[str, str]:
        """Generate realistic HTTP headers."""
        headers = {
            "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,"
            "image/avif,image/webp,image/apng,*/*;q=0.8,"
            "application/signed-exchange;v=b3;q=0.7",
            "Accept-Encoding": "gzip, deflate, br",
            "Accept-Language": language_header,
            "Cache-Control": "max-age=0",
            "Sec-Ch-Ua": '"Chromium";v="122", "Not(A:Brand";v="24", '
            '"Google Chrome";v="122"',
            "Sec-Ch-Ua-Mobile": "?0",
            "Sec-Ch-Ua-Platform": f'"{client_hints_platform}"',
            "Sec-Fetch-Dest": "document",
            "Sec-Fetch-Mode": "navigate",
            "Sec-Fetch-Site": "none",
            "Sec-Fetch-User": "?1",
            "Upgrade-Insecure-Requests": "1",
        }

        # Add DNT header occasionally (privacy-conscious users)
        if random.random() < 0.3:  # noqa: S311  # Anti-detection randomization
            headers["DNT"] = "1"

        return headers

    def _get_stealth_args(self) -> list[str]:
        """Return a conservative list of Chromium flags for stealth."""
        return ["--disable-blink-features=AutomationControlled"]

    def resolve_profile_for_domain(self, domain: str | None) -> str:
        """Resolve the most appropriate site profile name for a domain."""
        if not domain:
            return "default"
        lowered = domain.lower()
        for profile_name, profile in self.site_profiles.items():
            if profile_name == "default":
                continue
            if profile.domain in lowered or lowered.endswith(profile.domain):
                return profile_name
        return "default"

    def _select_language_profile(self) -> tuple[str, list[str], str]:
        """Return Accept-Language header, languages list, and locale."""

        templates = [
            ("en-US,en;q=0.9", ["en-US", "en"], "en-US"),
            ("en-GB,en;q=0.9", ["en-GB", "en"], "en-GB"),
            (
                "en-US,en;q=0.9,fr;q=0.8",
                ["en-US", "en", "fr"],
                "en-US",
            ),
            (
                "en-US,en;q=0.9,de;q=0.8",
                ["en-US", "en", "de"],
                "en-US",
            ),
            (
                "en-US,en;q=0.9,es;q=0.8",
                ["en-US", "en", "es"],
                "en-US",
            ),
        ]
        return random.choice(templates)  # noqa: S311  # Anti-detection randomization

    def _select_timezone(self, profile: SiteProfile) -> str:
        """Return a suitable time zone for the supplied profile."""

        pool = [
            "America/Los_Angeles",
            "America/New_York",
            "Europe/Berlin",
            "Europe/Paris",
            "Asia/Tokyo",
            "Asia/Singapore",
        ]
        if profile.domain == "linkedin.com":
            pool.extend(["America/Chicago", "Europe/London"])
        if profile.domain == "cloudflare.com":
            pool.extend(["America/Denver", "Europe/Amsterdam"])
        return random.choice(pool)  # noqa: S311  # Anti-detection randomization

    def _derive_platform(self, user_agent: str) -> tuple[str, str]:
        """Derive navigator platform and client hints platform."""

        lowered = user_agent.lower()
        if "windows" in lowered:
            return "Win32", "Windows"
        if "macintosh" in lowered or "mac os" in lowered:
            return "MacIntel", "macOS"
        if "x11" in lowered or "linux" in lowered:
            return "Linux x86_64", "Linux"
        return "Win32", "Windows"

    def _select_webgl_overrides(
        self, profile: SiteProfile, platform: str
    ) -> tuple[str | None, str | None]:
        """Return WebGL vendor/renderer overrides for high-risk profiles."""

        if profile.stealth_level not in {"advanced", "maximum"}:
            return None, None

        if platform == "MacIntel":
            return "Apple Inc.", "Apple GPU"
        if platform == "Linux x86_64":
            return (
                "Google Inc.",
                "ANGLE (AMD Radeon RX 6600 XT Direct3D11 vs_5_0 ps_5_0)",
            )
        return (
            "Google Inc.",
            "ANGLE (Intel(R) UHD Graphics 630 Direct3D11 vs_5_0 ps_5_0)",
        )

    def _get_timing_pattern(self, profile: SiteProfile) -> TimingPattern:
        """Get timing patterns based on site profile."""
        if profile.risk_level == "low":
            return TimingPattern(
                mouse_movement_delay=(30, 100),
                click_delay=(50, 150),
                typing_speed_wpm=(60, 100),
                page_reading_time=(1000, 3000),
            )
        if profile.risk_level == "high":
            return TimingPattern(
                mouse_movement_delay=(100, 300),
                click_delay=(200, 500),
                typing_speed_wpm=(30, 60),
                page_reading_time=(5000, 12000),
            )
        if profile.risk_level == "extreme":
            return TimingPattern(
                mouse_movement_delay=(200, 500),
                click_delay=(300, 800),
                typing_speed_wpm=(20, 45),
                page_reading_time=(8000, 20000),
            )
        # medium
        return TimingPattern()

    async def apply_stealth_to_playwright_config(
        self, config: PlaywrightConfig, site_profile: str = "default"
    ) -> PlaywrightConfig:
        """Apply anti-detection enhancements to PlaywrightConfig.

        Args:
            config: Original PlaywrightConfig
            site_profile: Site profile name

        Returns:
            Enhanced PlaywrightConfig with anti-detection settings

        """
        stealth_config = self.get_stealth_config(site_profile)

        # Create enhanced config
        viewport_model = cast(ViewportProfile, stealth_config.viewport)
        return PlaywrightConfig(
            browser=config.browser,
            headless=config.headless,
            viewport={
                "width": viewport_model.width,  # pylint: disable=no-member
                "height": viewport_model.height,  # pylint: disable=no-member
            },
            user_agent=stealth_config.user_agent,
            timeout=config.timeout,
        )

    async def get_human_like_delay(self, site_profile: str = "default") -> float:
        """Get human-like delay for interactions.

        Args:
            site_profile: Site profile name

        Returns:
            Delay in seconds

        """
        profile = self.site_profiles.get(site_profile, self.site_profiles["default"])
        min_delay, max_delay = profile.required_delay_ms

        # Add some randomness to avoid pattern detection
        base_delay = (
            random.uniform(min_delay, max_delay) / 1000.0  # noqa: S311
        )  # Anti-detection randomization
        jitter = random.uniform(0.8, 1.2)  # noqa: S311  # Anti-detection randomization

        return base_delay * jitter

    def record_attempt(self, success: bool, site_profile: str = "default") -> None:
        """Record attempt result for success rate monitoring."""
        self.success_monitor.record_attempt(success, site_profile)

    def get_success_metrics(self) -> dict[str, Any]:
        """Get current success rate metrics."""
        return {
            "overall_success_rate": self.success_monitor.get_overall_success_rate(),
            "recent_success_rate": self.success_monitor.get_recent_success_rate(),
            "_total_attempts": self.success_monitor.total_attempts,
            "total_attempts": self.success_monitor.total_attempts,
            "successful_attempts": self.success_monitor.successful_attempts,
            "needs_adjustment": self.success_monitor.needs_strategy_adjustment(),
            "strategy_performance": self.success_monitor.strategy_performance,
        }

    def get_recommended_strategy(self, domain: str) -> str:
        """Get recommended strategy based on domain and success rates."""
        resolved = self.resolve_profile_for_domain(domain)
        if resolved != "default":
            return resolved

        # Check recent performance for auto-adjustment
        if self.success_monitor.needs_strategy_adjustment():
            # If recent performance is poor, escalate to more aggressive strategy
            current_rate = self.success_monitor.get_recent_success_rate()
            if current_rate < 0.6:
                return "cloudflare.com"  # Maximum stealth
            if current_rate < 0.8:
                return "linkedin.com"  # High stealth

        return "default"
