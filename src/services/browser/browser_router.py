"""Enhanced automation router with performance tracking and intelligent routing.

This module extends the basic AutomationRouter with:
- Historical performance tracking
- Intelligent fallback strategies
- URL pattern-based tier recommendations
- Domain-specific tier preferences
- Circuit breaker pattern for tier health
"""

import asyncio
import logging
import time
from collections import defaultdict, deque
from typing import Any
from urllib.parse import urlparse

from src.config import Config
from src.services.errors import CrawlServiceError

from .automation_router import AutomationRouter
from .tier_config import (
    PerformanceHistoryEntry,
    TierConfiguration,
    TierPerformanceAnalysis,
    create_default_routing_config,
)
from .tier_rate_limiter import RateLimitContext, TierRateLimiter


logger = logging.getLogger(__name__)


class TierCircuitBreaker:
    """Circuit breaker state for a tier."""

    def __init__(self, tier: str, config: TierConfiguration):
        self.tier = tier
        self.config = config
        self.consecutive_failures = 0
        self.last_failure_time: float | None = None
        self.is_open = False

    def record_success(self) -> None:
        """Record a successful request."""
        self.consecutive_failures = 0
        self.is_open = False
        self.last_failure_time = None

    def record_failure(self) -> None:
        """Record a failed request."""
        self.consecutive_failures += 1
        self.last_failure_time = time.time()

        threshold = self.config.circuit_breaker_threshold
        if self.consecutive_failures >= threshold:
            self.is_open = True
            logger.warning(
                "Circuit breaker opened for %s after %d consecutive failures",
                self.tier,
                self.consecutive_failures,
            )

    def can_attempt(self) -> bool:
        """Check if we can attempt a request."""
        if not self.is_open:
            return True

        # Check if circuit should be reset
        if self.last_failure_time:
            elapsed = time.time() - self.last_failure_time
            duration = self.config.circuit_breaker_timeout_seconds
            if elapsed > duration:
                logger.info(
                    "Circuit breaker reset for %s after %.1fs", self.tier, elapsed
                )
                self.is_open = False
                self.consecutive_failures = 0
                return True

        return False


class EnhancedAutomationRouter(AutomationRouter):
    """Automation router with routing capabilities."""

    def __init__(self, config: Config):
        """Initialize enhanced router with configuration."""
        super().__init__(config)

        # Load enhanced configuration

        self.routing_config = create_default_routing_config()

        # Performance history tracking
        self.performance_history: deque[PerformanceHistoryEntry] = deque(maxlen=10000)
        self.performance_lock = asyncio.Lock()

        # Circuit breakers for each tier
        self.circuit_breakers: dict[str, TierCircuitBreaker] = {}
        for tier_name, tier_config in self.routing_config.tiers.items():
            self.circuit_breakers[tier_name] = TierCircuitBreaker(
                tier_name, tier_config
            )

        # Domain performance cache
        self.domain_tier_success: dict[str, dict[str, list[bool]]] = defaultdict(
            lambda: defaultdict(lambda: deque(maxlen=100))
        )

        # Rate limiter for tier requests
        self.rate_limiter = TierRateLimiter(self.routing_config.tiers)

    async def scrape(
        self,
        url: str,
        interaction_required: bool = False,
        custom_actions: list[dict] | None = None,
        force_tool: str | None = None,
        request_timeout_ms: int = 30000,
    ) -> dict[str, Any]:
        """Enhanced scraping with performance tracking and intelligent routing."""
        if not self._initialized:
            msg = "Router not initialized"
            raise CrawlServiceError(msg)

        domain = urlparse(url).netloc.lower()
        start_time = time.time()

        # Determine which tier to use with enhanced logic
        if force_tool:
            if force_tool not in self._adapters and force_tool not in [
                "crawl4ai_enhanced",
                "firecrawl",
            ]:
                msg = f"Forced tool '{force_tool}' not available"
                raise CrawlServiceError(msg)
            selected_tier = force_tool
        else:
            selected_tier = await self._enhanced_select_tier(
                url, domain, interaction_required, custom_actions
            )

        logger.info("Enhanced router selected %s for %s", selected_tier, url)

        # Check circuit breaker
        breaker = self.circuit_breakers.get(selected_tier)
        if breaker and not breaker.can_attempt():
            logger.warning("Circuit breaker open for %s, using fallback", selected_tier)
            # Get first available fallback tier
            tier_config = self.routing_config.tier_configs.get(selected_tier)
            if tier_config and tier_config.fallback_tiers:
                for fallback in tier_config.fallback_tiers:
                    fallback_breaker = self.circuit_breakers.get(fallback)
                    if fallback in self._adapters and (
                        not fallback_breaker or fallback_breaker.can_attempt()
                    ):
                        selected_tier = fallback
                        break
                else:
                    # No available fallback, will fail
                    msg = (
                        f"Circuit breaker open for {selected_tier} and "
                        "no fallbacks available"
                    )
                    raise CrawlServiceError(msg)
            else:
                msg = (
                    f"Circuit breaker open for {selected_tier} with "
                    "no fallback configured"
                )
                raise CrawlServiceError(msg)

        # Execute scraping with rate limiting
        try:
            result = await self._execute_scraping_with_rate_limiting(
                selected_tier,
                url,
                domain,
                custom_actions,
                request_timeout_ms,
                start_time,
                breaker,
            )

        except Exception as e:
            elapsed = time.time() - start_time
            logger.exception("{selected_tier} failed for {url}")

            # Record failure
            await self._record_performance(
                url,
                domain,
                selected_tier,
                False,
                elapsed * 1000,
                0,
                str(type(e).__name__),
            )

            if breaker:
                breaker.record_failure()

            # Try intelligent fallback
            return await self._intelligent_fallback(
                url, domain, selected_tier, custom_actions, request_timeout_ms, str(e)
            )

        else:
            return result

    async def _enhanced_select_tier(
        self,
        url: str,
        domain: str,
        interaction_required: bool,
        custom_actions: list[dict] | None,
    ) -> str:
        """Select tier using enhanced logic with performance data."""
        # Check domain preferences first
        domain_tier = self._check_domain_preferences(domain)
        if domain_tier:
            return domain_tier

        # Check URL patterns across all tiers
        pattern_tier = self._check_url_patterns(url)
        if pattern_tier:
            return pattern_tier

        # Check interaction requirements
        if interaction_required or custom_actions:
            return self._get_best_interaction_tier()

        # Use performance-based selection if enabled
        if self.routing_config.enable_performance_routing:
            perf_tier = await self._get_performance_based_tier(domain)
            if perf_tier:
                return perf_tier

        # Fall back to original selection logic
        return await self._select_tool(url, interaction_required, custom_actions)

    def _check_domain_preferences(self, domain: str) -> str | None:
        """Check if domain has specific tier preference."""
        for pref in self.routing_config.domain_preferences:
            if pref.matches(domain):
                tier = pref.preferred_tier
                if tier in self._adapters or tier in ["crawl4ai_enhanced", "firecrawl"]:
                    logger.debug(
                        "Domain preference for %s: %s (%s)", domain, tier, pref.reason
                    )
                    return tier
        return None

    def _check_url_patterns(self, url: str) -> str | None:
        """Check URL patterns across all tiers to find best match."""
        best_match = None
        best_priority = -1

        for tier_name, tier_config in self.routing_config.tier_configs.items():
            # Skip disabled or unavailable tiers
            if not tier_config.enabled:
                continue
            if tier_name not in self._adapters and tier_name not in [
                "crawl4ai_enhanced",
                "firecrawl",
            ]:
                continue

            for pattern in tier_config.preferred_url_patterns:
                if pattern.matches(url) and pattern.priority > best_priority:
                    best_match = tier_name
                    best_priority = pattern.priority
                    logger.debug(
                        "URL pattern match for %s: %s (priority=%d, reason=%s)",
                        url,
                        tier_name,
                        pattern.priority,
                        pattern.reason,
                    )

        return best_match

    def _get_best_interaction_tier(self) -> str:
        """Get best tier for interaction requirements."""
        # Prefer tiers that support interaction
        interaction_tiers = []
        for tier_name, tier_config in self.routing_config.tier_configs.items():
            if tier_config.supports_interaction and tier_name in self._adapters:
                interaction_tiers.append((tier_name, tier_config.tier_level))

        if interaction_tiers:
            # Sort by tier level (higher is better for interaction)
            interaction_tiers.sort(key=lambda x: x[1], reverse=True)
            return interaction_tiers[0][0]

        # Fallback to browser_use or playwright
        return self._get_interaction_tool()

    async def _get_performance_based_tier(self, domain: str) -> str | None:
        """Select tier based on historical performance for domain."""
        async with self.performance_lock:
            # Get performance analysis for each tier
            analyses = await self._analyze_tier_performance(domain)

            if not analyses:
                return None

            # Find best performing tier
            best_tier = None
            best_score = -1.0

            for tier, analysis in analyses.items():
                # Skip if not enough data
                if (
                    analysis.total_requests
                    < self.routing_config.min_samples_for_analysis
                ):
                    continue

                # Skip if unhealthy
                if analysis.health_status == "unhealthy":
                    continue

                # Calculate performance score
                score = self._calculate_performance_score(analysis)
                if score > best_score:
                    best_score = score
                    best_tier = tier

            if best_tier:
                logger.info(
                    "Performance-based selection for %s: %s (score=%.2f)",
                    domain,
                    best_tier,
                    best_score,
                )

            return best_tier

    def _calculate_performance_score(self, analysis: TierPerformanceAnalysis) -> float:
        """Calculate performance score for a tier."""
        # Weighted scoring: success rate (60%), response time (30%), trend (10%)
        success_score = analysis.success_rate * 0.6

        # Normalize response time (faster is better, max 10s)
        time_score = max(0, 1 - (analysis.average_response_time_ms / 10000)) * 0.3

        # Trend bonus/penalty
        trend_score = 0.1
        if analysis.trend_direction == "improving":
            trend_score *= 1.2
        elif analysis.trend_direction == "degrading":
            trend_score *= 0.8

        return success_score + time_score + trend_score

    async def _intelligent_fallback(
        self,
        url: str,
        domain: str,
        failed_tier: str,
        custom_actions: list[dict] | None,
        request_timeout_ms: int,
        error_message: str,
    ) -> dict[str, Any]:
        """Intelligent fallback with performance awareness."""
        if not self.routing_config.enable_intelligent_fallback:
            # Use standard fallback
            return await self._fallback_scrape(
                url, failed_tier, custom_actions, request_timeout_ms
            )

        # Get tier configuration
        tier_config = self.routing_config.tier_configs.get(failed_tier)
        if not tier_config:
            return await self._fallback_scrape(
                url, failed_tier, custom_actions, request_timeout_ms
            )

        # Build fallback order based on configuration and performance
        fallback_order = await self._build_intelligent_fallback_order(
            failed_tier, tier_config, domain, error_message
        )

        failed_tiers = [failed_tier]

        for fallback_tier in fallback_order:
            start_time = time.time()
            # Check circuit breaker
            breaker = self.circuit_breakers.get(fallback_tier)
            if breaker and not breaker.can_attempt():
                logger.debug("Skipping %s due to open circuit breaker", fallback_tier)
                continue

            try:
                result = await self._execute_fallback_attempt(
                    fallback_tier,
                    url,
                    domain,
                    custom_actions,
                    request_timeout_ms,
                    failed_tier,
                    failed_tiers,
                    breaker,
                )

            except Exception as e:
                elapsed = time.time() - start_time
                logger.exception("Fallback {fallback_tier} also failed")

                # Record failure
                await self._record_performance(
                    url,
                    domain,
                    fallback_tier,
                    False,
                    elapsed * 1000,
                    0,
                    str(type(e).__name__),
                )

                if breaker:
                    breaker.record_failure()

                failed_tiers.append(fallback_tier)

            else:
                return result
        # All tiers failed
        return {
            "success": False,
            "error": "All tiers failed for {url}. Original error",
            "content": "",
            "metadata": {},
            "url": url,
            "tier_used": "none",
            "provider": "none",
            "failed_tiers": failed_tiers,
        }

    async def _build_intelligent_fallback_order(
        self,
        _failed_tier: str,
        tier_config: TierConfiguration,
        domain: str,
        error_message: str,
    ) -> list[str]:
        """Build intelligent fallback order based on configuration and performance."""
        # Start with configured fallbacks
        fallback_order = [
            tier for tier in tier_config.fallback_tiers if tier in self._adapters
        ]

        # If error suggests we need more capabilities, adjust order
        if any(
            keyword in error_message.lower()
            for keyword in ["javascript", "dynamic", "timeout", "interactive"]
        ):
            # Prioritize higher-tier tools
            higher_tiers = []
            current_level = tier_config.tier_level

            for tier_name, config in self.routing_config.tier_configs.items():
                if (
                    config.tier_level > current_level
                    and tier_name in self._adapters
                    and tier_name not in fallback_order
                ):
                    higher_tiers.append((tier_name, config.tier_level))

            # Sort by tier level and prepend to fallback order
            higher_tiers.sort(key=lambda x: x[1], reverse=True)
            fallback_order = [tier[0] for tier in higher_tiers] + fallback_order

        # Add performance-based ordering for domain
        if self.routing_config.enable_performance_routing:
            analyses = await self._analyze_tier_performance(domain)
            tier_scores = {}

            for tier in fallback_order:
                if tier in analyses:
                    tier_scores[tier] = self._calculate_performance_score(
                        analyses[tier]
                    )

            # Sort by performance score
            fallback_order.sort(key=lambda t: tier_scores.get(t, 0), reverse=True)

        # Remove duplicates while preserving order
        seen = set()
        unique_order = []
        for tier in fallback_order:
            if tier not in seen:
                seen.add(tier)
                unique_order.append(tier)

        return unique_order

    async def _execute_tier_scraping(
        self,
        tier: str,
        url: str,
        custom_actions: list[dict] | None,
        request_timeout_ms: int,
    ) -> dict[str, Any]:
        """Execute scraping for a specific tier."""
        # Use parent class methods for actual execution
        if tier == "lightweight":
            return await self._try_lightweight(url, request_timeout_ms)
        if tier == "crawl4ai":
            return await self._try_crawl4ai(url, custom_actions, request_timeout_ms)
        if tier == "crawl4ai_enhanced":
            return await self._try_crawl4ai_enhanced(
                url, custom_actions, request_timeout_ms
            )
        if tier == "browser_use":
            return await self._try_browser_use(url, custom_actions, request_timeout_ms)
        if tier == "playwright":
            return await self._try_playwright(url, custom_actions, request_timeout_ms)
        if tier == "firecrawl":
            return await self._try_firecrawl(url, request_timeout_ms)
        msg = "Unknown tier"
        raise CrawlServiceError(msg)

    async def _record_performance(
        self,
        url: str,
        domain: str,
        tier: str,
        success: bool,
        response_time_ms: float,
        content_length: int,
        error_type: str | None = None,
    ) -> None:
        """Record performance history entry."""
        async with self.performance_lock:
            entry = PerformanceHistoryEntry(
                timestamp=time.time(),
                tier=tier,
                url=url,
                domain=domain,
                success=success,
                response_time_ms=response_time_ms,
                content_length=content_length,
                error_type=error_type,
            )
            self.performance_history.append(entry)

            # Update domain-tier success tracking
            self.domain_tier_success[domain][tier].append(success)

        # Also update parent class metrics
        self._update_metrics(tier, success, response_time_ms / 1000)

    async def _analyze_tier_performance(
        self, domain: str | None = None
    ) -> dict[str, TierPerformanceAnalysis]:
        """Analyze tier performance from history."""
        analyses = {}

        # Filter history by time window
        cutoff_time = time.time() - (
            self.routing_config.performance_window_hours * 3600
        )
        recent_entries = [
            e for e in self.performance_history if e.timestamp > cutoff_time
        ]

        # Group by tier
        tier_entries = defaultdict(list)
        for entry in recent_entries:
            if domain is None or entry.domain == domain:
                tier_entries[entry.tier].append(entry)

        # Analyze each tier
        for tier, entries in tier_entries.items():
            if not entries:
                continue

            analysis = TierPerformanceAnalysis(tier=tier)
            analysis.total_requests = len(entries)
            analysis.successful_requests = sum(1 for e in entries if e.success)
            analysis.success_rate = (
                analysis.successful_requests / analysis.total_requests
                if analysis.total_requests > 0
                else 0
            )

            # Calculate response times
            success_times = [e.response_time_ms for e in entries if e.success]
            if success_times:
                analysis.average_response_time_ms = sum(success_times) / len(
                    success_times
                )
                sorted_times = sorted(success_times)
                p95_index = int(len(sorted_times) * 0.95)
                analysis.p95_response_time_ms = sorted_times[
                    min(p95_index, len(sorted_times) - 1)
                ]

            # Analyze trend (compare first half vs second half)
            if len(entries) >= 10:
                half = len(entries) // 2
                first_half_success = sum(1 for e in entries[:half] if e.success) / half
                second_half_success = sum(1 for e in entries[half:] if e.success) / (
                    len(entries) - half
                )

                if second_half_success > first_half_success + 0.1:
                    analysis.trend_direction = "improving"
                    analysis.trend_confidence = min(
                        (second_half_success - first_half_success) * 2, 1.0
                    )
                elif second_half_success < first_half_success - 0.1:
                    analysis.trend_direction = "degrading"
                    analysis.trend_confidence = min(
                        (first_half_success - second_half_success) * 2, 1.0
                    )

            # Health assessment
            tier_config = self.routing_config.tier_configs.get(tier)
            if tier_config:
                thresholds = tier_config.performance_thresholds

                if analysis.success_rate < thresholds.min_success_rate:
                    analysis.health_status = "unhealthy"
                    analysis.health_reasons.append(
                        f"Success rate {analysis.success_rate:.1%} below threshold "
                        f"{thresholds.min_success_rate:.1%}"
                    )
                elif (
                    analysis.average_response_time_ms
                    > thresholds.max_avg_response_time_ms
                ):
                    analysis.health_status = "degraded"
                    analysis.health_reasons.append(
                        f"Average response time "
                        f"{analysis.average_response_time_ms:.0f}ms "
                        f"above threshold {thresholds.max_avg_response_time_ms:.0f}ms"
                    )

            analyses[tier] = analysis

        return analyses

    async def get_recommended_tool(self, url: str) -> str:
        """Get recommended tool using enhanced logic."""
        domain = urlparse(url).netloc.lower()
        return await self._enhanced_select_tier(url, domain, False, None)

    async def get_performance_report(self) -> dict[str, Any]:
        """Get comprehensive performance report."""
        async with self.performance_lock:
            # Overall analysis
            overall_analysis = await self._analyze_tier_performance()

            # Domain-specific analysis
            domain_analyses = {}
            domains = {entry.domain for entry in self.performance_history}
            for domain in domains:
                domain_analyses[domain] = await self._analyze_tier_performance(domain)

            # Circuit breaker status
            circuit_status = {
                tier: {
                    "is_open": breaker.is_open,
                    "consecutive_failures": breaker.consecutive_failures,
                    "can_attempt": breaker.can_attempt(),
                }
                for tier, breaker in self.circuit_breakers.items()
            }

            return {
                "overall_performance": {
                    tier: analysis.dict() for tier, analysis in overall_analysis.items()
                },
                "domain_performance": {
                    domain: {
                        tier: analysis.dict() for tier, analysis in analyses.items()
                    }
                    for domain, analyses in domain_analyses.items()
                },
                "circuit_breakers": circuit_status,
                "history_size": len(self.performance_history),
                "config": {
                    "performance_routing_enabled": (
                        self.routing_config.enable_performance_routing
                    ),
                    "intelligent_fallback_enabled": (
                        self.routing_config.enable_intelligent_fallback
                    ),
                    "cost_optimization_enabled": (
                        self.routing_config.enable_cost_optimization
                    ),
                },
            }

    async def _execute_scraping_with_rate_limiting(
        self,
        selected_tier: str,
        url: str,
        domain: str,
        custom_actions: list[dict] | None,
        request_timeout_ms: int,
        start_time: float,
        breaker: TierCircuitBreaker | None,
    ) -> dict[str, Any]:
        """Execute scraping with rate limiting and circuit breaker tracking."""
        async with RateLimitContext(
            self.rate_limiter, selected_tier, timeout=5.0
        ) as allowed:
            if not allowed:
                # Rate limited, try fallback
                logger.warning(
                    "Rate limit exceeded for %s, attempting fallback", selected_tier
                )
                return await self._intelligent_fallback(
                    url,
                    domain,
                    selected_tier,
                    custom_actions,
                    request_timeout_ms,
                    "Rate limit exceeded",
                )

            result = await self._execute_tier_scraping(
                selected_tier, url, custom_actions, request_timeout_ms
            )

        # Record success
        elapsed = time.time() - start_time
        await self._record_performance(
            url,
            domain,
            selected_tier,
            True,
            elapsed * 1000,
            len(result.get("content", "")),
        )

        if breaker:
            breaker.record_success()

        result["tier_used"] = selected_tier
        result["provider"] = selected_tier
        result["automation_time_ms"] = elapsed * 1000

        return result

    async def _execute_fallback_attempt(
        self,
        fallback_tier: str,
        url: str,
        domain: str,
        custom_actions: list[dict] | None,
        request_timeout_ms: int,
        failed_tier: str,
        failed_tiers: list[str],
        breaker: TierCircuitBreaker | None,
    ) -> dict[str, Any]:
        """Execute fallback attempt with proper tracking."""
        logger.info("Intelligent fallback to %s for %s", fallback_tier, url)
        start_time = time.time()

        result = await self._execute_tier_scraping(
            fallback_tier, url, custom_actions, request_timeout_ms
        )

        # Record success
        elapsed = time.time() - start_time
        await self._record_performance(
            url,
            domain,
            fallback_tier,
            True,
            elapsed * 1000,
            len(result.get("content", "")),
        )

        if breaker:
            breaker.record_success()

        result["tier_used"] = fallback_tier
        result["provider"] = fallback_tier
        result["fallback_from"] = failed_tier
        result["failed_tiers"] = failed_tiers
        result["automation_time_ms"] = elapsed * 1000

        return result
