"""Rate limiting utilities for API calls."""

import asyncio
import logging
import time
from typing import TypeVar

from src.config import Config
from src.services.errors import APIError


logger = logging.getLogger(__name__)

# Type variable for generic function return type
T = TypeVar("T")


class RateLimiter:
    """Token bucket rate limiter for API calls.

    Implements a token bucket algorithm to limit the rate of API calls.
    Supports both per-second and per-minute rate limits with burst capacity.

    Attributes:
        max_calls: Maximum number of calls allowed
        time_window: Time window in seconds
        burst_multiplier: Multiplier for burst capacity
    """

    def __init__(
        self,
        max_calls: int,
        time_window: int = 60,
        burst_multiplier: float = 1.5,
    ):
        """Initialize rate limiter.

        Args:
            max_calls: Maximum calls allowed in time window
            time_window: Time window in seconds (default: 60)
            burst_multiplier: Burst capacity multiplier (default: 1.5)
        """

        self.max_calls = max_calls
        self.time_window = time_window
        self.burst_multiplier = burst_multiplier

        # Token bucket parameters
        self.max_tokens = int(max_calls * burst_multiplier)
        self.tokens = self.max_tokens
        self.refill_rate = max_calls / time_window
        self.last_refill = time.time()

        # Lock for thread safety
        self._lock = asyncio.Lock()

    async def acquire(self, tokens: int = 1) -> None:
        """Acquire tokens from the bucket, waiting if necessary.

        Args:
            tokens: Number of tokens to acquire (default: 1)
        """

        if tokens > self.max_tokens:
            msg = f"Requested {tokens} tokens exceeds bucket capacity {self.max_tokens}"
            raise APIError(msg)

        async with self._lock:
            # Refill tokens based on elapsed time
            now = time.time()
            elapsed = now - self.last_refill
            self.tokens = min(self.max_tokens, self.tokens + elapsed * self.refill_rate)
            self.last_refill = now

            # Wait if not enough tokens
            if self.tokens < tokens:
                wait_time = (tokens - self.tokens) / self.refill_rate
                logger.warning(
                    "Rate limit reached. Waiting %.2fs for %d tokens", wait_time, tokens
                )
                await asyncio.sleep(wait_time)

                # Refill after wait
                self.tokens = min(
                    self.max_tokens, self.tokens + wait_time * self.refill_rate
                )

            # Consume tokens
            self.tokens -= tokens
            logger.debug("Acquired %d tokens. %.1f remaining", tokens, self.tokens)


class RateLimitManager:
    """Manager for multiple rate limiters by provider/endpoint.

    Provides centralized rate limiting for different API providers
    with configurable limits per provider and endpoint.
    """

    def __init__(self, config: Config):
        """Initialize rate limit manager.

        Args:
            config: Config instance with rate limiting configuration.

        """
        self.limiters: dict[str, RateLimiter] = {}
        self.default_limits = config.performance.default_rate_limits.copy()

    def get_limiter(self, provider: str, endpoint: str | None = None) -> RateLimiter:
        """Get or create rate limiter for provider/endpoint.

        Args:
            provider: Provider name (e.g., "openai", "firecrawl")
            endpoint: Optional specific endpoint

        Returns:
            RateLimiter instance
        """

        key = f"{provider}:{endpoint}" if endpoint else provider

        if key not in self.limiters:
            # Get limits for provider or raise error if not configured
            if provider not in self.default_limits:
                msg = (
                    f"No rate limits configured for provider '{provider}'. "
                    f"Available providers: {list(self.default_limits.keys())}"
                )
                raise ValueError(msg)

            limits = self.default_limits[provider]

            self.limiters[key] = RateLimiter(
                max_calls=limits["max_calls"],
                time_window=limits["time_window"],
            )

            logger.info(
                "Created rate limiter for %s: %d calls per %ds",
                key,
                limits["max_calls"],
                limits["time_window"],
            )

        return self.limiters[key]

    async def acquire(
        self,
        provider: str,
        endpoint: str | None = None,
        tokens: int = 1,
    ) -> None:
        """Acquire tokens from provider's rate limiter.

        Args:
            provider: Provider name
            endpoint: Optional endpoint name
            tokens: Number of tokens to acquire
        """

        limiter = self.get_limiter(provider, endpoint)
        await limiter.acquire(tokens)


class AdaptiveRateLimiter(RateLimiter):
    """Adaptive rate limiter that adjusts based on API responses.

    Monitors API responses for rate limit headers and errors,
    automatically adjusting the rate limit to stay within bounds.
    """

    def __init__(
        self,
        initial_max_calls: int = 60,
        time_window: int = 60,
        min_rate: float = 0.1,
        max_rate: float = 2.0,
    ):
        """Initialize adaptive rate limiter.

        Args:
            initial_max_calls: Initial rate limit estimate
            time_window: Time window in seconds
            min_rate: Minimum rate multiplier (default: 0.1)
            max_rate: Maximum rate multiplier (default: 2.0)
        """

        super().__init__(initial_max_calls, time_window)
        self.initial_max_calls = initial_max_calls
        self.min_rate = min_rate
        self.max_rate = max_rate
        self.adjustment_factor = 1.0

    async def handle_response(
        self,
        status_code: int,
        headers: dict[str, str] | None = None,
    ) -> None:
        """Adjust rate limit based on API response.

        Args:
            status_code: HTTP status code
            headers: Response headers containing rate limit info
        """

        async with self._lock:
            if status_code == 429:  # Rate limited
                # Reduce rate by 50%
                self.adjustment_factor = max(
                    self.min_rate, self.adjustment_factor * 0.5
                )
                logger.warning(
                    "Rate limit hit. Reducing rate to %.2fx", self.adjustment_factor
                )
            elif status_code < 300:  # Success
                # Slowly increase rate by 5%
                self.adjustment_factor = min(
                    self.max_rate, self.adjustment_factor * 1.05
                )

            # Parse rate limit headers if available
            if headers:
                # Common rate limit headers
                remaining = headers.get("x-ratelimit-remaining")
                limit = headers.get("x-ratelimit-limit")
                headers.get("x-ratelimit-reset")

                if remaining and limit:
                    try:
                        remaining_calls = int(remaining)
                        limit_calls = int(limit)

                        # Adjust max_calls based on actual limits
                        self.max_calls = int(limit_calls * self.adjustment_factor)
                        self.max_tokens = int(self.max_calls * self.burst_multiplier)

                        logger.debug(
                            "Adjusted rate limit: %d calls, %d remaining",
                            self.max_calls,
                            remaining_calls,
                        )
                    except ValueError:
                        pass  # Invalid header values

            # Update refill rate
            self.refill_rate = self.max_calls / self.time_window
