"""Rate limiting for browser automation tiers.

This module provides tier-specific rate limiting to prevent overloading
browser automation providers and ensure fair usage across tiers.
"""

import asyncio
import logging
import time
from collections import defaultdict, deque
from typing import Any

from .tier_config import TierConfiguration


logger = logging.getLogger(__name__)


class TierRateLimiter:
    """Rate limiter for browser automation tiers.

    Implements a sliding window rate limiter with tier-specific limits.
    Supports both request-per-minute and concurrent request limits.
    """

    def __init__(self, tier_configs: dict[str, TierConfiguration]):
        """Initialize rate limiter with tier configurations.

        Args:
            tier_configs: Dictionary mapping tier names to configurations
        """
        self.tier_configs = tier_configs

        # Track request timestamps per tier (sliding window)
        self.request_history: dict[str, deque[float]] = defaultdict(
            lambda: deque(maxlen=1000)
        )

        # Track concurrent requests per tier
        self.concurrent_requests: dict[str, int] = defaultdict(int)

        # Semaphores for concurrent request limiting
        self.tier_semaphores: dict[str, asyncio.Semaphore] = {}
        for tier_name, config in tier_configs.items():
            if config.enabled:
                self.tier_semaphores[tier_name] = asyncio.Semaphore(
                    config.max_concurrent_requests
                )

        # Track rate limit hits
        self.rate_limit_hits: dict[str, int] = defaultdict(int)

        # Lock for thread safety
        self.lock = asyncio.Lock()

    async def acquire(self, tier: str, timeout: float | None = None) -> bool:
        """Acquire permission to make a request for a tier.

        Args:
            tier: Tier name
            timeout: Maximum time to wait for permission (seconds)

        Returns:
            True if permission granted, False if rate limited
        """
        if tier not in self.tier_configs:
            logger.warning(f"Unknown tier {tier}, allowing request")
            return True

        config = self.tier_configs[tier]
        if not config.enabled:
            logger.warning(f"Tier {tier} is disabled")
            return False

        # Check rate limit
        if not await self._check_rate_limit(tier):
            self.rate_limit_hits[tier] += 1
            logger.warning(
                f"Rate limit exceeded for {tier} "
                f"(total hits: {self.rate_limit_hits[tier]})"
            )
            return False

        # Acquire semaphore for concurrent limit
        semaphore = self.tier_semaphores.get(tier)
        if semaphore:
            try:
                if timeout is not None:
                    acquired = await asyncio.wait_for(
                        semaphore.acquire(), timeout=timeout
                    )
                else:
                    await semaphore.acquire()
                    acquired = True

                if acquired:
                    async with self.lock:
                        self.concurrent_requests[tier] += 1
                        self.request_history[tier].append(time.time())
                    return True
            except TimeoutError:
                logger.warning(
                    f"Timeout waiting for {tier} rate limiter "
                    f"(concurrent: {self.concurrent_requests[tier]})"
                )
                return False

        return True

    async def release(self, tier: str) -> None:
        """Release a request slot for a tier.

        Args:
            tier: Tier name
        """
        semaphore = self.tier_semaphores.get(tier)
        if semaphore:
            semaphore.release()
            async with self.lock:
                self.concurrent_requests[tier] = max(
                    0, self.concurrent_requests[tier] - 1
                )

    async def _check_rate_limit(self, tier: str) -> bool:
        """Check if request is within rate limit.

        Args:
            tier: Tier name

        Returns:
            True if within limit, False otherwise
        """
        config = self.tier_configs[tier]
        if (
            not hasattr(config, "requests_per_minute")
            or config.requests_per_minute <= 0
        ):
            return True  # No rate limit configured

        async with self.lock:
            # Clean old requests from history
            current_time = time.time()
            cutoff_time = current_time - 60  # 1 minute window

            history = self.request_history[tier]
            while history and history[0] < cutoff_time:
                history.popleft()

            # Check if we're within limit
            return len(history) < config.requests_per_minute

    def get_wait_time(self, tier: str) -> float:
        """Get estimated wait time until next request allowed.

        Args:
            tier: Tier name

        Returns:
            Seconds to wait (0 if no wait needed)
        """
        config = self.tier_configs.get(tier)
        if not config or not hasattr(config, "requests_per_minute"):
            return 0.0

        if config.requests_per_minute <= 0:
            return 0.0

        history = self.request_history[tier]
        if len(history) < config.requests_per_minute:
            return 0.0

        # Calculate when oldest request will expire
        oldest_request = history[0]
        current_time = time.time()
        time_until_expire = max(0, (oldest_request + 60) - current_time)

        return time_until_expire

    def get_status(self, tier: str | None = None) -> dict[str, Any]:
        """Get rate limiter status.

        Args:
            tier: Specific tier to check (None for all)

        Returns:
            Status information
        """
        if tier:
            return self._get_tier_status(tier)

        # Get status for all tiers
        status = {
            "tiers": {},
            "total_rate_limit_hits": sum(self.rate_limit_hits.values()),
        }

        for tier_name in self.tier_configs:
            status["tiers"][tier_name] = self._get_tier_status(tier_name)

        return status

    def _get_tier_status(self, tier: str) -> dict[str, Any]:
        """Get status for a specific tier.

        Args:
            tier: Tier name

        Returns:
            Tier status information
        """
        config = self.tier_configs.get(tier)
        if not config:
            return {"error": "Unknown tier"}

        history = self.request_history[tier]
        current_time = time.time()
        recent_requests = sum(1 for t in history if t > current_time - 60)

        status = {
            "enabled": config.enabled,
            "concurrent_requests": self.concurrent_requests[tier],
            "max_concurrent": config.max_concurrent_requests,
            "recent_requests": recent_requests,
            "rate_limit_hits": self.rate_limit_hits[tier],
        }

        if hasattr(config, "requests_per_minute") and config.requests_per_minute > 0:
            status["requests_per_minute"] = config.requests_per_minute
            status["remaining_capacity"] = max(
                0, config.requests_per_minute - recent_requests
            )
            status["wait_time_seconds"] = self.get_wait_time(tier)

        return status

    async def wait_if_needed(self, tier: str) -> None:
        """Wait if rate limit requires it.

        Args:
            tier: Tier name
        """
        wait_time = self.get_wait_time(tier)
        if wait_time > 0:
            logger.info(f"Rate limit reached for {tier}, waiting {wait_time:.1f}s")
            await asyncio.sleep(wait_time)

    def reset_tier(self, tier: str) -> None:
        """Reset rate limit tracking for a tier.

        Args:
            tier: Tier name
        """
        self.request_history[tier].clear()
        self.concurrent_requests[tier] = 0
        self.rate_limit_hits[tier] = 0
        logger.info(f"Reset rate limiter for tier {tier}")

    def reset_all(self) -> None:
        """Reset all rate limit tracking."""
        self.request_history.clear()
        self.concurrent_requests.clear()
        self.rate_limit_hits.clear()
        logger.info("Reset all rate limiters")


class RateLimitContext:
    """Context manager for rate-limited tier operations."""

    def __init__(
        self, rate_limiter: TierRateLimiter, tier: str, timeout: float | None = None
    ):
        """Initialize rate limit context.

        Args:
            rate_limiter: Rate limiter instance
            tier: Tier name
            timeout: Timeout for acquiring permission
        """
        self.rate_limiter = rate_limiter
        self.tier = tier
        self.timeout = timeout
        self.acquired = False

    async def __aenter__(self) -> bool:
        """Acquire rate limit permission.

        Returns:
            True if acquired, False if rate limited
        """
        self.acquired = await self.rate_limiter.acquire(self.tier, self.timeout)
        return self.acquired

    async def __aexit__(self, exc_type, exc_val, exc_tb) -> None:
        """Release rate limit permission."""
        if self.acquired:
            await self.rate_limiter.release(self.tier)
