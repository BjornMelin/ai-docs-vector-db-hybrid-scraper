"""Configuration models for browser automation tiers."""

import logging
from dataclasses import dataclass
from dataclasses import field
from enum import Enum

logger = logging.getLogger(__name__)


class TierType(str, Enum):
    """Supported automation tier types."""

    CRAWL4AI = "crawl4ai"
    PLAYWRIGHT = "playwright"
    BROWSER_USE = "browser_use"


class CircuitBreakerState(str, Enum):
    """Circuit breaker states for tier health."""

    CLOSED = "closed"
    OPEN = "open"
    HALF_OPEN = "half_open"


@dataclass
class URLPattern:
    """URL pattern configuration for tier routing."""

    pattern: str
    priority: int = 50
    reason: str = ""


@dataclass
class DomainPreference:
    """Domain-specific tier preferences."""

    domain: str
    preferred_tier: str
    fallback_tiers: list[str] = field(default_factory=list)
    reason: str = ""


@dataclass
class PerformanceHistoryEntry:
    """Single performance measurement entry."""

    timestamp: float
    tier_name: str
    url: str
    success: bool
    duration_ms: float
    error_type: str | None = None
    memory_usage_mb: float | None = None


@dataclass
class TierPerformanceAnalysis:
    """Performance analysis for a tier."""

    tier_name: str
    total_requests: int = 0
    successful_requests: int = 0
    failed_requests: int = 0
    average_duration_ms: float = 0.0
    success_rate: float = 0.0
    last_success_time: float | None = None
    last_failure_time: float | None = None
    circuit_breaker_state: CircuitBreakerState = CircuitBreakerState.CLOSED

    @property
    def failure_rate(self) -> float:
        """Calculate failure rate."""
        if self.total_requests == 0:
            return 0.0
        return self.failed_requests / self.total_requests


@dataclass
class TierConfiguration:
    """Configuration for an automation tier."""

    tier_name: str
    tier_level: int
    description: str = ""
    enabled: bool = True
    max_concurrent_requests: int = 5
    timeout_ms: int = 30000
    retry_count: int = 1
    estimated_cost_per_request: float = 0.0
    memory_usage_mb: int = 100
    cpu_intensity: str = "medium"
    preferred_url_patterns: list[URLPattern] = field(default_factory=list)
    supports_javascript: bool = True
    supports_interaction: bool = False
    fallback_tiers: list[str] = field(default_factory=list)
    circuit_breaker_threshold: int = 5
    circuit_breaker_timeout_seconds: int = 60

    def __post_init__(self):
        """Validate configuration after initialization."""
        if self.tier_name not in [tier.value for tier in TierType]:
            logger.warning(f"Unknown tier type: {self.tier_name}")


@dataclass
class EnhancedRoutingConfig:
    """Enhanced routing configuration."""

    tiers: dict[str, TierConfiguration] = field(default_factory=dict)
    domain_preferences: list[DomainPreference] = field(default_factory=list)
    global_timeout_ms: int = 120000
    enable_performance_tracking: bool = True
    performance_history_size: int = 1000
    circuit_breaker_enabled: bool = True
    fallback_strategy: str = "next_tier"  # next_tier, lowest_cost, fastest

    def get_tier_config(self, tier_name: str) -> TierConfiguration | None:
        """Get configuration for a specific tier."""
        return self.tiers.get(tier_name)

    def get_domain_preference(self, domain: str) -> DomainPreference | None:
        """Get domain preference for a domain."""
        for pref in self.domain_preferences:
            if pref.domain == domain or domain.endswith(f".{pref.domain}"):
                return pref
        return None

    def get_enabled_tiers(self) -> list[str]:
        """Get list of enabled tier names."""
        return [name for name, config in self.tiers.items() if config.enabled]


def create_default_routing_config() -> EnhancedRoutingConfig:
    """Create default routing configuration."""
    return EnhancedRoutingConfig(
        tiers={
            TierType.CRAWL4AI.value: TierConfiguration(
                tier_name=TierType.CRAWL4AI.value,
                tier_level=1,
                description="Fast web crawling with basic JS support",
                max_concurrent_requests=10,
                timeout_ms=30000,
                supports_javascript=True,
                supports_interaction=False,
                fallback_tiers=[TierType.PLAYWRIGHT.value],
            ),
            TierType.PLAYWRIGHT.value: TierConfiguration(
                tier_name=TierType.PLAYWRIGHT.value,
                tier_level=2,
                description="Full browser automation with interaction support",
                max_concurrent_requests=3,
                timeout_ms=60000,
                supports_javascript=True,
                supports_interaction=True,
                fallback_tiers=[TierType.BROWSER_USE.value],
            ),
            TierType.BROWSER_USE.value: TierConfiguration(
                tier_name=TierType.BROWSER_USE.value,
                tier_level=3,
                description="AI-powered browser automation",
                max_concurrent_requests=1,
                timeout_ms=120000,
                estimated_cost_per_request=0.05,
                supports_javascript=True,
                supports_interaction=True,
                fallback_tiers=[],
            ),
        }
    )
