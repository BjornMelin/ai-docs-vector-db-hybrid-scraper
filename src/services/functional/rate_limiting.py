"""Function-based rate limiting with dependency injection.

Transforms the complex RateLimitManager class into pure functions with
dependency injection. Maintains all functionality while improving testability.
"""

import asyncio
import logging
import time
from typing import Annotated, Any

from fastapi import Depends, HTTPException

from src.config import Config

from .circuit_breaker import CircuitBreakerConfig, circuit_breaker
from .dependencies import get_config


logger = logging.getLogger(__name__)


class TokenBucket:
    """Simple token bucket for rate limiting."""

    def __init__(
        self, max_calls: int, time_window: int = 60, burst_multiplier: float = 1.5
    ):
        self.max_calls = max_calls
        self.time_window = time_window
        self.max_tokens = int(max_calls * burst_multiplier)
        self.tokens = self.max_tokens
        self.refill_rate = max_calls / time_window
        self.last_refill = time.time()
        self._lock = asyncio.Lock()

    async def acquire(self, tokens: int = 1) -> None:
        """Acquire tokens, waiting if necessary."""
        if tokens > self.max_tokens:
            raise HTTPException(
                status_code=429, detail=f"Request size {tokens} exceeds bucket capacity"
            )

        async with self._lock:
            now = time.time()
            elapsed = now - self.last_refill
            self.tokens = min(self.max_tokens, self.tokens + elapsed * self.refill_rate)
            self.last_refill = now

            if self.tokens < tokens:
                wait_time = (tokens - self.tokens) / self.refill_rate
                logger.warning(f"Rate limit reached. Waiting {wait_time:.2f}s")
                await asyncio.sleep(wait_time)
                self.tokens = min(
                    self.max_tokens, self.tokens + wait_time * self.refill_rate
                )

            self.tokens -= tokens


# Global bucket storage (simple in-memory implementation)
_rate_limiters: dict[str, TokenBucket] = {}
_limiters_lock = asyncio.Lock()


async def get_rate_limiter(
    provider: str,
    endpoint: str | None = None,
    config: Annotated[Config, Depends(get_config)] = None,
) -> TokenBucket:
    """Get or create rate limiter for provider/endpoint.

    Pure function replacement for RateLimitManager.get_limiter().

    Args:
        provider: Provider name (e.g., "openai", "firecrawl")
        endpoint: Optional specific endpoint
        config: Injected configuration

    Returns:
        TokenBucket instance
    """
    key = f"{provider}:{endpoint}" if endpoint else provider

    async with _limiters_lock:
        if key not in _rate_limiters:
            # Get limits from config
            limits = config.performance.default_rate_limits.get(provider)
            if not limits:
                raise HTTPException(
                    status_code=500,
                    detail=f"No rate limits configured for provider '{provider}'",
                )

            _rate_limiters[key] = TokenBucket(
                max_calls=limits["max_calls"],
                time_window=limits["time_window"],
            )

            logger.info(
                f"Created rate limiter for {key}: "
                f"{limits['max_calls']} calls per {limits['time_window']}s"
            )

        return _rate_limiters[key]


@circuit_breaker(CircuitBreakerConfig.simple_mode())
async def acquire_rate_limit(
    provider: str,
    endpoint: str | None = None,
    tokens: int = 1,
    config: Annotated[Config, Depends(get_config)] = None,
) -> None:
    """Acquire rate limit tokens for a provider.

    Pure function replacement for RateLimitManager.acquire().

    Args:
        provider: Provider name
        endpoint: Optional endpoint name
        tokens: Number of tokens to acquire
        config: Injected configuration

    Raises:
        HTTPException: If rate limiting fails
    """
    try:
        limiter = await get_rate_limiter(provider, endpoint, config)
        await limiter.acquire(tokens)
        logger.debug(f"Acquired {tokens} tokens for {provider}")

    except Exception as e:
        logger.exception(f"Rate limit acquisition failed: {e}")
        raise HTTPException(status_code=429, detail=f"Rate limiting failed: {e!s}")


async def get_rate_limit_status(
    provider: str,
    endpoint: str | None = None,
    config: Annotated[Config, Depends(get_config)] = None,
) -> dict[str, Any]:
    """Get current rate limit status for a provider.

    New function providing visibility into rate limit state.

    Args:
        provider: Provider name
        endpoint: Optional endpoint name
        config: Injected configuration

    Returns:
        Rate limit status information
    """
    try:
        limiter = await get_rate_limiter(provider, endpoint, config)

        return {
            "provider": provider,
            "endpoint": endpoint,
            "max_tokens": limiter.max_tokens,
            "current_tokens": limiter.tokens,
            "refill_rate": limiter.refill_rate,
            "utilization": 1.0 - (limiter.tokens / limiter.max_tokens),
        }

    except Exception as e:
        logger.exception(f"Rate limit status check failed: {e}")
        return {
            "provider": provider,
            "endpoint": endpoint,
            "error": str(e),
        }


class AdaptiveTokenBucket(TokenBucket):
    """Adaptive token bucket that adjusts based on API responses."""

    def __init__(self, initial_max_calls: int = 60, time_window: int = 60):
        super().__init__(initial_max_calls, time_window)
        self.adjustment_factor = 1.0
        self.min_rate = 0.1
        self.max_rate = 2.0

    async def handle_response(
        self, status_code: int, headers: dict[str, str] | None = None
    ) -> None:
        """Adjust rate limit based on API response."""
        async with self._lock:
            if status_code == 429:  # Rate limited
                self.adjustment_factor = max(
                    self.min_rate, self.adjustment_factor * 0.5
                )
                logger.warning(
                    f"Rate limit hit. Reducing rate to {self.adjustment_factor:.2f}x"
                )
            elif status_code < 300:  # Success
                self.adjustment_factor = min(
                    self.max_rate, self.adjustment_factor * 1.05
                )

            # Update rate based on adjustment
            self.max_calls = int(self.max_calls * self.adjustment_factor)
            self.refill_rate = self.max_calls / self.time_window


async def handle_api_response(
    provider: str,
    status_code: int,
    response_headers: dict[str, str] | None = None,
    endpoint: str | None = None,
    config: Annotated[Config, Depends(get_config)] = None,
) -> None:
    """Handle API response for adaptive rate limiting.

    New function that provides feedback-based rate limit adjustment.

    Args:
        provider: Provider name
        status_code: HTTP status code from API response
        response_headers: Response headers containing rate limit info
        endpoint: Optional endpoint name
        config: Injected configuration
    """
    try:
        limiter = await get_rate_limiter(provider, endpoint, config)

        # Only handle adaptive adjustment if it's an adaptive bucket
        if isinstance(limiter, AdaptiveTokenBucket):
            await limiter.handle_response(status_code, response_headers)

        logger.debug(f"Processed response for {provider}: {status_code}")

    except Exception as e:
        logger.exception(f"Response handling failed: {e}")
        # Non-critical, don't raise exception


# Rate limiting context manager for easy usage
class RateLimitContext:
    """Context manager for rate-limited operations."""

    def __init__(self, provider: str, endpoint: str | None = None, tokens: int = 1):
        self.provider = provider
        self.endpoint = endpoint
        self.tokens = tokens

    async def __aenter__(self):
        await acquire_rate_limit(self.provider, self.endpoint, self.tokens)
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb):
        # Could add response handling here if needed
        pass


def rate_limited(provider: str, endpoint: str | None = None, tokens: int = 1):
    """Decorator for rate-limited functions.

    Usage:
        @rate_limited("openai", "embeddings")
        async def generate_embeddings(...):
            # Function automatically rate limited
    """

    def decorator(func):
        async def wrapper(*args, **kwargs):
            async with RateLimitContext(provider, endpoint, tokens):
                return await func(*args, **kwargs)

        return wrapper

    return decorator
