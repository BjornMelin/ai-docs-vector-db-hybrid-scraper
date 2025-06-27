"""Enhanced Anti-Detection System for browser automation.

This module provides sophisticated anti-detection capabilities for browser automation,
including fingerprint management, behavioral patterns, and success rate monitoring.
Designed to achieve 95%+ success rate on challenging sites while maintaining performance.

Note: Uses standard random module for anti-detection purposes (timing, user agents, etc.)
This is intentional and not cryptographically sensitive.
"""
# Standard random is acceptable for anti-detection purposes

import json
import logging
import random
import secrets
import time
from pathlib import Path
from typing import Any

from pydantic import BaseModel, Field


logger = logging.getLogger(__name__)

from src.config import PlaywrightConfig


class UserAgentPool(BaseModel):
    """Pool of realistic user agents with browser fingerprint matching."""

    chrome_agents: list[str] = Field(
        default_factory=lambda: [
            "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/122.0.0.0 Safari/537.36",
            "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/122.0.0.0 Safari/537.36",
            "Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/122.0.0.0 Safari/537.36",
            "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/121.0.0.0 Safari/537.36",
            "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/121.0.0.0 Safari/537.36",
        ]
    )

    firefox_agents: list[str] = Field(
        default_factory=lambda: [
            "Mozilla/5.0 (Windows NT 10.0; Win64; x64; rv:122.0) Gecko/20100101 Firefox/122.0",
            "Mozilla/5.0 (Macintosh; Intel Mac OS X 10.15; rv:122.0) Gecko/20100101 Firefox/122.0",
            "Mozilla/5.0 (X11; Linux x86_64; rv:122.0) Gecko/20100101 Firefox/122.0",
            "Mozilla/5.0 (Windows NT 10.0; Win64; x64; rv:121.0) Gecko/20100101 Firefox/121.0",
        ]
    )

    safari_agents: list[str] = Field(
        default_factory=lambda: [
            "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/605.1.15 (KHTML, like Gecko) Version/17.2.1 Safari/605.1.15",
            "Mozilla/5.0 (iPhone; CPU iPhone OS 17_2_1 like Mac OS X) AppleWebKit/605.1.15 (KHTML, like Gecko) Version/17.2 Mobile/15E148 Safari/604.1",
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


class SessionData(BaseModel):
    """Browser session data for persistence."""

    session_id: str = Field(description="Unique session identifier")
    domain: str = Field(description="Domain this session belongs to")
    cookies: list[dict[str, Any]] = Field(
        default_factory=list, description="Cookie data"
    )
    local_storage: dict[str, str] = Field(
        default_factory=dict, description="localStorage data"
    )
    session_storage: dict[str, str] = Field(
        default_factory=dict, description="sessionStorage data"
    )
    user_agent: str = Field(description="User agent used for this session")
    viewport: dict[str, int] = Field(description="Viewport used for this session")
    created_at: float = Field(description="Session creation timestamp")
    last_used: float = Field(description="Last usage timestamp")
    success_count: int = Field(default=0, description="Number of successful requests")

    def is_expired(self, max_age_hours: int = 24) -> bool:
        """Check if session has expired."""
        age_hours = (time.time() - self.created_at) / 3600
        return age_hours > max_age_hours

    def update_usage(self, success: bool = True) -> None:
        """Update session usage statistics."""
        self.last_used = time.time()
        if success:
            self.success_count += 1


class SessionManager:
    """Manages browser sessions for persistent browsing behavior."""

    def __init__(self, session_dir: str | None = None):
        """Initialize session manager.

        Args:
            session_dir: Directory to store session data
        """
        self.session_dir = Path(session_dir or "./.anti_detection_sessions")
        self.session_dir.mkdir(exist_ok=True)
        self.active_sessions: dict[str, SessionData] = {}

    def _get_session_file(self, session_id: str) -> Path:
        """Get session file path."""
        return self.session_dir / f"{session_id}.json"

    def create_session(
        self, domain: str, user_agent: str, viewport: dict[str, int]
    ) -> SessionData:
        """Create a new session for a domain."""
        session_id = f"{domain}_{int(time.time())}_{secrets.randbelow(9000) + 1000}"

        session = SessionData(
            session_id=session_id,
            domain=domain,
            user_agent=user_agent,
            viewport=viewport,
            created_at=time.time(),
            last_used=time.time(),
        )

        self.active_sessions[session_id] = session
        self.save_session(session)
        return session

    def get_session(self, domain: str) -> SessionData | None:
        """Get an existing session for a domain."""
        # Look for active sessions first
        for session in self.active_sessions.values():
            if session.domain == domain and not session.is_expired():
                return session

        # Look for stored sessions
        for session_file in self.session_dir.glob(f"{domain}_*.json"):
            try:
                with session_file.open() as f:
                    session_data = json.load(f)
                session = SessionData(**session_data)

                if not session.is_expired():
                    self.active_sessions[session.session_id] = session
                    return session
                else:
                    # Clean up expired session
                    session_file.unlink(missing_ok=True)
            except Exception:
                # Clean up corrupted session file
                session_file.unlink(missing_ok=True)

        return None

    def save_session(self, session: SessionData) -> None:
        """Save session to disk."""
        session_file = self._get_session_file(session.session_id)
        try:
            with session_file.open("w") as f:
                json.dump(session.model_dump(), f, indent=2)
        except Exception as e:
            logger.debug(f"Failed to save session {session.session_id}: {e}")

    def update_session_data(
        self,
        session_id: str,
        cookies: list[dict[str, Any]] | None = None,
        local_storage: dict[str, str] | None = None,
        session_storage: dict[str, str] | None = None,
    ) -> None:
        """Update session data."""
        if session_id in self.active_sessions:
            session = self.active_sessions[session_id]

            if cookies is not None:
                session.cookies = cookies
            if local_storage is not None:
                session.local_storage = local_storage
            if session_storage is not None:
                session.session_storage = session_storage

            session.update_usage()
            self.save_session(session)

    def cleanup_expired_sessions(self) -> None:
        """Clean up expired sessions."""
        # Clean up active sessions
        expired_ids = [
            sid for sid, session in self.active_sessions.items() if session.is_expired()
        ]
        for sid in expired_ids:
            del self.active_sessions[sid]

        # Clean up session files
        for session_file in self.session_dir.glob("*.json"):
            try:
                with session_file.open() as f:
                    session_data = json.load(f)
                session = SessionData(**session_data)

                if session.is_expired():
                    session_file.unlink()
            except Exception:
                session_file.unlink(missing_ok=True)


class SuccessRateMonitor(BaseModel):
    """Success rate monitoring and adaptive strategy adjustment."""

    total_attempts: int = Field(default=0)
    successful_attempts: int = Field(default=0)
    recent_successes: list[bool] = Field(default_factory=list, max_length=50)
    strategy_performance: dict[str, dict[str, Any]] = Field(default_factory=dict)

    def record_attempt(self, success: bool, strategy: str = "default") -> None:
        """Record an attempt and update statistics."""
        self.total_attempts += 1
        if success:
            self.successful_attempts += 1

        # Maintain rolling window of recent attempts
        self.recent_successes.append(success)
        if len(self.recent_successes) > 50:
            self.recent_successes.pop(0)

        # Update strategy-specific performance
        if strategy not in self.strategy_performance:
            self.strategy_performance[strategy] = {
                "attempts": 0,
                "successes": 0,
                "recent_performance": [],
            }

        self.strategy_performance[strategy]["attempts"] += 1
        if success:
            self.strategy_performance[strategy]["successes"] += 1

        # Keep recent performance window
        self.strategy_performance[strategy]["recent_performance"].append(success)
        if len(self.strategy_performance[strategy]["recent_performance"]) > 20:
            self.strategy_performance[strategy]["recent_performance"].pop(0)

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
    """Enhanced anti-detection system for browser automation.

    Provides sophisticated fingerprint management, behavioral patterns,
    and adaptive strategy adjustment to achieve 95%+ success rates.
    """

    def __init__(self, enable_session_management: bool = True):
        """Initialize the enhanced anti-detection system."""
        self.user_agents = UserAgentPool()
        self.viewport_profiles = ViewportProfile.get_common_profiles()
        self.success_monitor = SuccessRateMonitor()
        self.site_profiles: dict[str, SiteProfile] = {}
        self.session_manager = SessionManager() if enable_session_management else None
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

        return BrowserStealthConfig(
            user_agent=self._rotate_user_agents(),
            viewport=self._randomize_viewport(),
            headers=self._generate_realistic_headers(profile),
            extra_args=self._get_stealth_args(profile),
            timing=self._get_timing_pattern(profile),
        )

    def _rotate_user_agents(self) -> str:
        """Rotate user agents with realistic browser signatures."""
        # Weighted selection favoring Chrome (most common)
        browser_weights = [0.65, 0.25, 0.10]  # Chrome, Firefox, Safari
        browser_pools = [
            self.user_agents.chrome_agents,
            self.user_agents.firefox_agents,
            self.user_agents.safari_agents,
        ]

        selected_pool = random.choices(browser_pools, weights=browser_weights)[0]
        return random.choice(selected_pool)

    def _randomize_viewport(self) -> ViewportProfile:
        """Randomize viewport with common resolution patterns."""
        # Filter to non-mobile profiles for better compatibility
        desktop_profiles = [p for p in self.viewport_profiles if not p.is_mobile]
        profile = random.choice(desktop_profiles)

        # Add slight randomization to avoid exact pattern matching
        width_variance = random.randint(-50, 50)
        height_variance = random.randint(-30, 30)

        return ViewportProfile(
            width=max(1200, min(1920, profile.width + width_variance)),
            height=max(720, min(1200, profile.height + height_variance)),
            device_scale_factor=profile.device_scale_factor,
            is_mobile=profile.is_mobile,
        )

    def _generate_realistic_headers(self, _profile: SiteProfile) -> dict[str, str]:
        """Generate realistic HTTP headers."""
        # Language preferences weighted by global usage
        languages = [
            "en-US,en;q=0.9",
            "en-US,en;q=0.9,es;q=0.8",
            "en-US,en;q=0.9,fr;q=0.8",
            "en-US,en;q=0.9,de;q=0.8",
            "en-GB,en;q=0.9",
        ]

        headers = {
            "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,image/avif,image/webp,image/apng,*/*;q=0.8,application/signed-exchange;v=b3;q=0.7",
            "Accept-Encoding": "gzip, deflate, br",
            "Accept-Language": random.choice(languages),
            "Cache-Control": "max-age=0",
            "Sec-Ch-Ua": '"Chromium";v="122", "Not(A:Brand";v="24", "Google Chrome";v="122"',
            "Sec-Ch-Ua-Mobile": "?0",
            "Sec-Ch-Ua-Platform": '"Windows"',
            "Sec-Fetch-Dest": "document",
            "Sec-Fetch-Mode": "navigate",
            "Sec-Fetch-Site": "none",
            "Sec-Fetch-User": "?1",
            "Upgrade-Insecure-Requests": "1",
        }

        # Add DNT header occasionally (privacy-conscious users)
        if random.random() < 0.3:
            headers["DNT"] = "1"

        return headers

    def _get_stealth_args(self, profile: SiteProfile) -> list[str]:
        """Get browser arguments for stealth based on profile."""
        base_args = [
            "--disable-blink-features=AutomationControlled",
            "--disable-web-security",
            "--disable-features=VizDisplayCompositor",
            "--no-sandbox",
            "--disable-dev-shm-usage",
            "--disable-background-timer-throttling",
            "--disable-backgrounding-occluded-windows",
            "--disable-renderer-backgrounding",
            "--disable-field-trial-config",
        ]

        if profile.stealth_level in ["advanced", "maximum"]:
            base_args.extend(
                [
                    "--disable-ipc-flooding-protection",
                    "--disable-default-apps",
                    "--disable-sync",
                    "--no-first-run",
                    "--no-default-browser-check",
                    "--disable-client-side-phishing-detection",
                    "--disable-component-update",
                    "--disable-hang-monitor",
                    "--disable-prompt-on-repost",
                    "--disable-background-networking",
                ]
            )

        if profile.stealth_level == "maximum":
            base_args.extend(
                [
                    "--disable-extensions",
                    "--disable-plugins-discovery",
                    "--disable-preconnect",
                    "--disable-software-rasterizer",
                    "--media-cache-size=0",
                    "--disk-cache-size=0",
                ]
            )

        return base_args

    def _get_timing_pattern(self, profile: SiteProfile) -> TimingPattern:
        """Get timing patterns based on site profile."""
        if profile.risk_level == "low":
            return TimingPattern(
                mouse_movement_delay=(30, 100),
                click_delay=(50, 150),
                typing_speed_wpm=(60, 100),
                page_reading_time=(1000, 3000),
            )
        elif profile.risk_level == "high":
            return TimingPattern(
                mouse_movement_delay=(100, 300),
                click_delay=(200, 500),
                typing_speed_wpm=(30, 60),
                page_reading_time=(5000, 12000),
            )
        elif profile.risk_level == "extreme":
            return TimingPattern(
                mouse_movement_delay=(200, 500),
                click_delay=(300, 800),
                typing_speed_wpm=(20, 45),
                page_reading_time=(8000, 20000),
            )
        else:  # medium
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
        enhanced_config = PlaywrightConfig(
            browser=config.browser,
            headless=config.headless,
            viewport={
                "width": stealth_config.viewport.width,
                "height": stealth_config.viewport.height,
            },
            user_agent=stealth_config.user_agent,
            timeout=config.timeout,
        )

        return enhanced_config

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
        base_delay = random.uniform(min_delay, max_delay) / 1000.0
        jitter = random.uniform(0.8, 1.2)

        return base_delay * jitter

    def record_attempt(self, success: bool, site_profile: str = "default") -> None:
        """Record attempt result for success rate monitoring."""
        self.success_monitor.record_attempt(success, site_profile)

    def get_success_metrics(self) -> dict[str, Any]:
        """Get current success rate metrics."""
        return {
            "overall_success_rate": self.success_monitor.get_overall_success_rate(),
            "recent_success_rate": self.success_monitor.get_recent_success_rate(),
            "total_attempts": self.success_monitor.total_attempts,
            "successful_attempts": self.success_monitor.successful_attempts,
            "needs_adjustment": self.success_monitor.needs_strategy_adjustment(),
            "strategy_performance": self.success_monitor.strategy_performance,
        }

    def get_recommended_strategy(self, domain: str) -> str:
        """Get recommended strategy based on domain and success rates."""
        # Check if domain has specific profile
        for profile_domain in self.site_profiles:
            if profile_domain in domain or domain in profile_domain:
                return profile_domain

        # Check recent performance for auto-adjustment
        if self.success_monitor.needs_strategy_adjustment():
            # If recent performance is poor, escalate to more aggressive strategy
            current_rate = self.success_monitor.get_recent_success_rate()
            if current_rate < 0.6:
                return "cloudflare.com"  # Maximum stealth
            elif current_rate < 0.8:
                return "linkedin.com"  # High stealth

        return "default"
