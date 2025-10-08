"""Modern rate limiting implementation using slowapi.

This module provides a modernized rate limiting implementation using
the battle-tested slowapi library for distributed rate limiting with Redis.
"""

import logging
from collections.abc import Callable
from typing import Any

from fastapi import FastAPI, Request, Response  # type: ignore
from slowapi import Limiter
from slowapi.errors import RateLimitExceeded
from slowapi.middleware import SlowAPIMiddleware
from slowapi.util import get_remote_address

from src.config.loader import Settings


logger = logging.getLogger(__name__)


class ModernRateLimiter:
    """Modern rate limiter using slowapi with Redis backend.

    Provides distributed rate limiting with automatic key generation,
    configurable limits, and integration with FastAPI applications.
    """

    def __init__(
        self,
        app: FastAPI,
        redis_url: str = "redis://localhost:6379",
        key_func: Callable | None = None,
        default_limits: list[str] | None = None,
        config: Settings | None = None,
    ):
        """Initialize modern rate limiter.

        Args:
            app: FastAPI application instance
            redis_url: Redis URL for distributed rate limiting
            key_func: Function to generate rate limit keys (default: IP-based)
            default_limits: Default rate limits to apply
            config: Application configuration

        """
        self.app = app
        self.redis_url = redis_url
        self.config = config

        # Use provided key function or default to remote address
        self.key_func = key_func or get_remote_address

        # Create limiter instance with Redis storage
        self.limiter = Limiter(
            key_func=self.key_func,
            storage_uri=redis_url,
            default_limits=default_limits or ["1000/hour", "100/minute"],  # type: ignore[arg-type]
        )

        # Attach limiter to app state for access in route handlers
        app.state.limiter = self.limiter

        # Add exception handler for rate limit exceeded
        app.add_exception_handler(RateLimitExceeded, self._rate_limit_handler)

        # Add middleware for automatic rate limiting
        app.add_middleware(SlowAPIMiddleware)

        logger.info(
            "ModernRateLimiter initialized with Redis: %s, default_limits: %s",
            redis_url,
            default_limits,
        )

    def limit(self, rate: str, per_method: bool = False):
        """Create a rate limit decorator for specific endpoints.

        Args:
            rate: Rate limit string (e.g., "10/minute", "100/hour")
            per_method: Whether to apply limit per HTTP method

        Returns:
            Rate limit decorator

        """
        return self.limiter.limit(rate, per_method=per_method)

    def shared_limit(self, rate: str, scope: str):
        """Create a shared rate limit across multiple endpoints.

        Args:
            rate: Rate limit string (e.g., "10/minute", "100/hour")
            scope: Shared scope identifier

        Returns:
            Shared rate limit decorator

        """
        return self.limiter.shared_limit(rate, scope)

    def exempt(self, request: Request) -> bool:
        """Check if a request should be exempted from rate limiting.

        Args:
            request: FastAPI request object

        Returns:
            True if request should be exempted, False otherwise

        """
        # Example exemption logic - customize as needed
        if hasattr(self.config, "rate_limiting") and self.config.rate_limiting:  # type: ignore[attr-defined]
            exempt_ips = getattr(self.config.rate_limiting, "exempt_ips", [])  # type: ignore[attr-defined]
            client_ip = get_remote_address(request)
            return client_ip in exempt_ips

        return False

    async def _rate_limit_handler(
        self,
        request: Request,
        exc: RateLimitExceeded,
    ) -> Response:
        """Handle rate limit exceeded exceptions.

        Args:
            request: FastAPI request object
            exc: Rate limit exceeded exception

        Returns:
            HTTP 429 response with rate limit information

        """
        response = Response(
            content=f"Rate limit exceeded: {exc.detail}",
            status_code=429,
        )

        # Add rate limit headers if available
        if hasattr(exc, "retry_after"):
            response.headers["Retry-After"] = str(exc.retry_after)  # type: ignore[attr-defined]

        if hasattr(exc, "limit"):
            response.headers["X-RateLimit-Limit"] = str(exc.limit)

        if hasattr(exc, "remaining"):
            response.headers["X-RateLimit-Remaining"] = str(exc.remaining)  # type: ignore[attr-defined]

        if hasattr(exc, "reset"):
            response.headers["X-RateLimit-Reset"] = str(exc.reset)  # type: ignore[attr-defined]

        # Log rate limit exceeded for monitoring
        client_ip = get_remote_address(request)
        logger.warning(
            "Rate limit exceeded for %s on %s: %s",
            client_ip,
            request.url.path,
            exc.detail,
        )

        return response

    def get_current_limits(self, request: Request) -> dict[str, Any]:
        """Get current rate limit status for a request.

        Args:
            request: FastAPI request object

        Returns:
            Dictionary with current rate limit information

        """
        try:
            key = self.key_func(request)
            # This would require extending slowapi or accessing internal state
            # For now, return basic information
        except Exception as e:
            logger.exception("Error getting current limits")
            return {"error": str(e)}

        else:
            return {
                "key": key,
                "limiter_storage": self.redis_url,
                "default_limits": getattr(
                    self.limiter,
                    "default_limits",
                    getattr(self.limiter, "_default_limits", None),
                ),
            }

    async def reset_limits(self, _key: str) -> bool:
        """Reset rate limits for a specific key.

        Args:
            _key: Rate limit key to reset

        Returns:
            True if successful, False otherwise

        """
        try:
            # Rate limit reset requires direct storage access which is not exposed
            # by slowapi's public interface. This is intentionally not supported
            # to maintain proper encapsulation of the rate limiting library.
            logger.warning("Rate limit reset not supported by current implementation")
        except Exception:
            logger.exception("Error resetting limits for {key}")
            return False

        else:
            return False

    async def get_stats(self) -> dict[str, Any]:
        """Get rate limiter statistics.

        Returns:
            Dictionary with rate limiter statistics

        """
        try:
            return {
                "limiter": {
                    "redis_url": self.redis_url,
                    "default_limits": getattr(
                        self.limiter,
                        "default_limits",
                        getattr(self.limiter, "_default_limits", None),
                    ),
                    "key_function": self.key_func.__name__,
                },
                "middleware": {
                    "enabled": True,
                    "exception_handler": "custom",
                },
            }
        except Exception as e:
            logger.exception("Error getting rate limiter stats")
            return {"error": str(e)}


# Additional rate limiting utilities


def create_api_key_limiter(redis_url: str) -> Limiter:
    """Create a rate limiter based on API keys.

    Args:
        redis_url: Redis URL for storage

    Returns:
        Limiter instance configured for API key-based limiting

    """

    def get_api_key(request: Request) -> str:
        """Extract API key from request headers."""
        api_key = request.headers.get("X-API-Key") or request.headers.get(
            "Authorization",
        )
        if api_key and api_key.startswith("Bearer "):
            api_key = api_key[7:]  # Remove "Bearer " prefix
        return api_key or get_remote_address(request)

    return Limiter(
        key_func=get_api_key,
        storage_uri=redis_url,
        default_limits=["10000/hour", "1000/minute"],  # Higher limits for API keys
    )


def create_user_based_limiter(redis_url: str) -> Limiter:
    """Create a rate limiter based on authenticated users.

    Args:
        redis_url: Redis URL for storage

    Returns:
        Limiter instance configured for user-based limiting

    """

    def get_user_id(request: Request) -> str:
        """Extract user ID from request context."""
        # This would need integration with your authentication system
        user = getattr(request.state, "user", None)
        if user and hasattr(user, "id"):
            return f"user:{user.id}"
        return get_remote_address(request)  # Fallback to IP

    return Limiter(
        key_func=get_user_id,
        storage_uri=redis_url,
        default_limits=[
            "5000/hour",
            "500/minute",
        ],  # Higher limits for authenticated users
    )


def setup_rate_limiting(
    app: FastAPI,
    redis_url: str = "redis://localhost:6379",
    config: Settings | None = None,
) -> ModernRateLimiter:
    """Set up rate limiting for a FastAPI application.

    Args:
        app: FastAPI application instance
        redis_url: Redis URL for distributed rate limiting
        config: Application configuration

    Returns:
        ModernRateLimiter instance

    """
    # Determine rate limits from config
    default_limits = ["1000/hour", "100/minute"]
    if config and hasattr(config, "rate_limiting"):
        default_limits = getattr(config.rate_limiting, "default_limits", default_limits)  # type: ignore[attr-defined]

    return ModernRateLimiter(
        app=app,
        redis_url=redis_url,
        default_limits=default_limits,
        config=config,
    )
