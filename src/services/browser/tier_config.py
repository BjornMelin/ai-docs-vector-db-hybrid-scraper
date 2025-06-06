"""Tier-specific configuration for 5-tier browser automation system.

This module provides enhanced configuration for each tier with performance
thresholds, fallback strategies, and URL pattern matching.
"""

import re
from typing import Any
from typing import Literal

from pydantic import BaseModel
from pydantic import Field


class TierPerformanceThresholds(BaseModel):
    """Performance thresholds for tier health monitoring."""

    min_success_rate: float = Field(
        default=0.8, description="Minimum success rate before considering degraded"
    )
    max_avg_response_time_ms: float = Field(
        default=5000, description="Maximum average response time before degraded"
    )
    max_consecutive_failures: int = Field(
        default=3, description="Maximum consecutive failures before circuit break"
    )
    circuit_break_duration_seconds: int = Field(
        default=300, description="Duration to keep circuit open"
    )


class URLPattern(BaseModel):
    """URL pattern matching configuration."""

    pattern: str = Field(description="Regex pattern to match URLs")
    priority: int = Field(
        default=0, description="Priority when multiple patterns match (higher wins)"
    )
    reason: str = Field(description="Reason for this pattern preference")

    def matches(self, url: str) -> bool:
        """Check if URL matches this pattern."""
        try:
            return bool(re.search(self.pattern, url, re.IGNORECASE))
        except Exception:
            return False


class TierConfiguration(BaseModel):
    """Configuration for a specific browser automation tier."""

    tier_name: Literal[
        "lightweight", "crawl4ai", "crawl4ai_enhanced", "browser_use", "playwright", "firecrawl"
    ]
    tier_level: int = Field(description="Tier level (0-4)")
    description: str = Field(description="Description of this tier's capabilities")

    # Performance settings
    enabled: bool = Field(default=True, description="Whether this tier is enabled")
    max_concurrent_requests: int = Field(default=5, description="Maximum concurrent requests")
    requests_per_minute: int = Field(
        default=60, description="Maximum requests per minute (0 for unlimited)"
    )
    timeout_ms: int = Field(default=30000, description="Default timeout in milliseconds")
    retry_count: int = Field(default=1, description="Number of retries on failure")

    # Cost and resource settings
    estimated_cost_per_request: float = Field(
        default=0.0, description="Estimated cost per request in USD"
    )
    memory_usage_mb: int = Field(
        default=0, description="Estimated memory usage per request"
    )
    cpu_intensity: Literal["low", "medium", "high"] = Field(
        default="medium", description="CPU intensity level"
    )

    # URL patterns this tier excels at
    preferred_url_patterns: list[URLPattern] = Field(
        default_factory=list, description="URL patterns this tier handles well"
    )

    # Performance thresholds
    performance_thresholds: TierPerformanceThresholds = Field(
        default_factory=TierPerformanceThresholds
    )

    # Capabilities
    supports_javascript: bool = Field(
        default=False, description="Whether tier can execute JavaScript"
    )
    supports_interaction: bool = Field(
        default=False, description="Whether tier supports page interactions"
    )
    supports_screenshots: bool = Field(
        default=False, description="Whether tier can take screenshots"
    )
    supports_cookies: bool = Field(
        default=False, description="Whether tier maintains cookies"
    )

    # Fallback preferences
    fallback_tiers: list[str] = Field(
        default_factory=list, description="Preferred fallback tiers in order"
    )


class DomainPreference(BaseModel):
    """Domain-specific tier preferences."""

    domain: str = Field(description="Domain pattern (can use wildcards)")
    preferred_tier: str = Field(description="Preferred tier for this domain")
    required_tier: bool = Field(
        default=False, description="Whether to force this tier (no fallback)"
    )
    reason: str = Field(description="Reason for this preference")

    def matches(self, domain: str) -> bool:
        """Check if domain matches this preference."""
        pattern = self.domain.replace("*", ".*")
        try:
            return bool(re.match(pattern, domain, re.IGNORECASE))
        except Exception:
            return False


class EnhancedRoutingConfig(BaseModel):
    """Enhanced routing configuration for AutomationRouter."""

    # Tier configurations
    tier_configs: dict[str, TierConfiguration] = Field(
        default_factory=dict, description="Configuration for each tier"
    )

    # Domain preferences
    domain_preferences: list[DomainPreference] = Field(
        default_factory=list, description="Domain-specific tier preferences"
    )

    # Global settings
    enable_performance_routing: bool = Field(
        default=True, description="Route based on historical performance"
    )
    enable_cost_optimization: bool = Field(
        default=False, description="Optimize for cost when possible"
    )
    enable_intelligent_fallback: bool = Field(
        default=True, description="Use intelligent fallback strategies"
    )

    # Performance analysis settings
    performance_window_hours: int = Field(
        default=24, description="Hours of history to consider for performance"
    )
    min_samples_for_analysis: int = Field(
        default=10, description="Minimum samples before using performance data"
    )

    # Circuit breaker settings
    global_circuit_breaker_enabled: bool = Field(
        default=True, description="Enable circuit breakers for all tiers"
    )

    @classmethod
    def get_default_config(cls) -> "EnhancedRoutingConfig":
        """Get default tier configurations."""
        return cls(
            tier_configs={
                "lightweight": TierConfiguration(
                    tier_name="lightweight",
                    tier_level=0,
                    description="Lightweight HTTP client with BeautifulSoup parsing",
                    max_concurrent_requests=20,
                    requests_per_minute=300,  # High limit for simple requests
                    timeout_ms=10000,
                    estimated_cost_per_request=0.0,
                    memory_usage_mb=50,
                    cpu_intensity="low",
                    preferred_url_patterns=[
                        URLPattern(
                            pattern=r"\.(txt|json|xml|csv)$",
                            priority=100,
                            reason="Static file formats"
                        ),
                        URLPattern(
                            pattern=r"api\.|/api/|\.api\.",
                            priority=90,
                            reason="API endpoints often return structured data"
                        ),
                        URLPattern(
                            pattern=r"raw\.githubusercontent\.com",
                            priority=95,
                            reason="GitHub raw content is always static"
                        ),
                    ],
                    supports_javascript=False,
                    supports_interaction=False,
                    fallback_tiers=["crawl4ai", "playwright"],
                ),
                "crawl4ai": TierConfiguration(
                    tier_name="crawl4ai",
                    tier_level=1,
                    description="Crawl4AI basic mode for standard browser automation",
                    max_concurrent_requests=10,
                    requests_per_minute=120,  # Moderate limit for browser automation
                    timeout_ms=30000,
                    estimated_cost_per_request=0.0,
                    memory_usage_mb=200,
                    cpu_intensity="medium",
                    preferred_url_patterns=[
                        URLPattern(
                            pattern=r"blog|article|news|post",
                            priority=70,
                            reason="Content-heavy pages"
                        ),
                        URLPattern(
                            pattern=r"wikipedia\.org|wikimedia\.org",
                            priority=80,
                            reason="Well-structured content sites"
                        ),
                    ],
                    supports_javascript=True,
                    supports_interaction=False,
                    supports_screenshots=True,
                    supports_cookies=True,
                    fallback_tiers=["crawl4ai_enhanced", "browser_use", "playwright"],
                ),
                "crawl4ai_enhanced": TierConfiguration(
                    tier_name="crawl4ai_enhanced",
                    tier_level=2,
                    description="Crawl4AI with custom JavaScript for dynamic content",
                    max_concurrent_requests=5,
                    requests_per_minute=60,  # Lower limit for complex operations
                    timeout_ms=45000,
                    estimated_cost_per_request=0.0,
                    memory_usage_mb=300,
                    cpu_intensity="high",
                    preferred_url_patterns=[
                        URLPattern(
                            pattern=r"spa|app|dashboard|console",
                            priority=75,
                            reason="Single-page applications"
                        ),
                        URLPattern(
                            pattern=r"react|vue|angular",
                            priority=75,
                            reason="Modern JS frameworks"
                        ),
                    ],
                    supports_javascript=True,
                    supports_interaction=True,
                    supports_screenshots=True,
                    supports_cookies=True,
                    fallback_tiers=["browser_use", "playwright"],
                ),
                "browser_use": TierConfiguration(
                    tier_name="browser_use",
                    tier_level=3,
                    description="AI-powered browser automation for complex interactions",
                    max_concurrent_requests=3,
                    requests_per_minute=30,  # Low limit due to AI resource usage
                    timeout_ms=60000,
                    estimated_cost_per_request=0.01,  # Estimated LLM cost
                    memory_usage_mb=500,
                    cpu_intensity="high",
                    preferred_url_patterns=[
                        URLPattern(
                            pattern=r"login|signin|auth|oauth",
                            priority=85,
                            reason="Authentication flows"
                        ),
                        URLPattern(
                            pattern=r"checkout|payment|cart",
                            priority=85,
                            reason="Complex multi-step processes"
                        ),
                        URLPattern(
                            pattern=r"captcha|recaptcha|hcaptcha",
                            priority=90,
                            reason="Requires intelligent interaction"
                        ),
                    ],
                    supports_javascript=True,
                    supports_interaction=True,
                    supports_screenshots=True,
                    supports_cookies=True,
                    fallback_tiers=["playwright"],
                ),
                "playwright": TierConfiguration(
                    tier_name="playwright",
                    tier_level=4,
                    description="Full programmatic browser control",
                    max_concurrent_requests=5,
                    requests_per_minute=60,  # Moderate limit for full browser control
                    timeout_ms=45000,
                    estimated_cost_per_request=0.0,
                    memory_usage_mb=400,
                    cpu_intensity="high",
                    preferred_url_patterns=[
                        URLPattern(
                            pattern=r"github\.com|gitlab\.com|bitbucket\.org",
                            priority=70,
                            reason="Code repository sites"
                        ),
                        URLPattern(
                            pattern=r"stackoverflow\.com|reddit\.com",
                            priority=70,
                            reason="Community sites with complex layouts"
                        ),
                    ],
                    supports_javascript=True,
                    supports_interaction=True,
                    supports_screenshots=True,
                    supports_cookies=True,
                    fallback_tiers=["firecrawl"],
                ),
                "firecrawl": TierConfiguration(
                    tier_name="firecrawl",
                    tier_level=4,
                    description="API-based fallback scraping service",
                    max_concurrent_requests=10,
                    requests_per_minute=100,  # API rate limit
                    timeout_ms=60000,
                    estimated_cost_per_request=0.02,  # API cost
                    memory_usage_mb=100,
                    cpu_intensity="low",
                    preferred_url_patterns=[
                        URLPattern(
                            pattern=r".*",  # Can handle anything as last resort
                            priority=0,
                            reason="Universal fallback"
                        ),
                    ],
                    supports_javascript=True,
                    supports_interaction=False,
                    supports_screenshots=False,
                    supports_cookies=False,
                    fallback_tiers=[],  # No fallback from tier 4
                ),
            },
            domain_preferences=[
                DomainPreference(
                    domain="*.anthropic.com",
                    preferred_tier="browser_use",
                    reason="Complex documentation site with dynamic content"
                ),
                DomainPreference(
                    domain="github.com",
                    preferred_tier="playwright",
                    reason="Requires authentication and has anti-bot measures"
                ),
                DomainPreference(
                    domain="*.openai.com",
                    preferred_tier="browser_use",
                    reason="Dynamic content with authentication"
                ),
                DomainPreference(
                    domain="raw.githubusercontent.com",
                    preferred_tier="lightweight",
                    required_tier=True,
                    reason="Static content, no need for browser"
                ),
            ],
        )


class PerformanceHistoryEntry(BaseModel):
    """Single performance history entry."""

    timestamp: float = Field(description="Unix timestamp")
    tier: str = Field(description="Tier used")
    url: str = Field(description="URL scraped")
    domain: str = Field(description="Domain of URL")
    success: bool = Field(description="Whether scraping succeeded")
    response_time_ms: float = Field(description="Response time in milliseconds")
    content_length: int = Field(description="Length of extracted content")
    error_type: str | None = Field(default=None, description="Error type if failed")


class TierPerformanceAnalysis(BaseModel):
    """Analysis of tier performance over time."""

    tier: str = Field(description="Tier name")
    total_requests: int = Field(default=0)
    successful_requests: int = Field(default=0)
    average_response_time_ms: float = Field(default=0.0)
    success_rate: float = Field(default=0.0)
    p95_response_time_ms: float = Field(default=0.0)

    # Domain-specific performance
    domain_performance: dict[str, dict[str, Any]] = Field(
        default_factory=dict, description="Performance by domain"
    )

    # Recent performance trend
    trend_direction: Literal["improving", "stable", "degrading"] = Field(
        default="stable"
    )
    trend_confidence: float = Field(
        default=0.0, description="Confidence in trend assessment (0-1)"
    )

    # Health assessment
    health_status: Literal["healthy", "degraded", "unhealthy"] = Field(
        default="healthy"
    )
    health_reasons: list[str] = Field(default_factory=list)
