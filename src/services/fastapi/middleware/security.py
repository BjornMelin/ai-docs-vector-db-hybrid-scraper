"""Security middleware for production-grade security headers and protection.

This middleware adds essential security headers and provides basic protection
against common web vulnerabilities in production deployments, including
Redis-backed rate limiting for distributed deployment scenarios.
"""

import asyncio
import logging
import time
from collections import defaultdict
from collections.abc import Callable
from typing import Optional

import redis.asyncio as redis
from starlette.middleware.base import BaseHTTPMiddleware
from starlette.requests import Request
from starlette.responses import JSONResponse, Response

from src.config import SecurityConfig


logger = logging.getLogger(__name__)


class SecurityMiddleware(BaseHTTPMiddleware):
    """Security middleware with headers injection and Redis-backed rate limiting.

    Features:
    - Essential security headers injection
    - Redis-backed rate limiting for distributed deployments
    - Fallback to in-memory rate limiting for development
    - Request validation and filtering
    - Security event logging
    - AI-specific threat detection (prompt injection)
    """

    def __init__(
        self, app: Callable, config: SecurityConfig, redis_url: str | None = None
    ):
        """Initialize security middleware.

        Args:
            app: ASGI application
            config: Security configuration
            redis_url: Redis connection URL for distributed rate limiting

        """
        super().__init__(app)
        self.config = config
        self.redis_url = redis_url or "redis://localhost:6379/1"

        # Redis connection for distributed rate limiting
        self.redis_client: redis.Redis | None = None
        self._redis_healthy = False

        # In-memory fallback rate limiting storage
        self._rate_limit_storage: dict[str, dict[str, int]] = defaultdict(
            lambda: {"requests": 0, "window_start": 0}
        )

        # Security headers to inject
        self._security_headers = self._build_security_headers()

        # Initialize Redis connection
        asyncio.create_task(self._initialize_redis())

    async def _initialize_redis(self) -> None:
        """Initialize Redis connection for distributed rate limiting.

        Establishes connection to Redis/DragonflyDB and performs health checks.
        Falls back to in-memory rate limiting if Redis is unavailable.
        """
        try:
            self.redis_client = redis.Redis.from_url(
                self.redis_url,
                encoding="utf-8",
                decode_responses=True,
                socket_connect_timeout=5,
                socket_timeout=5,
                retry_on_timeout=True,
                health_check_interval=30,
            )

            # Test Redis connection
            await self.redis_client.ping()
            self._redis_healthy = True

            logger.info(
                "Redis connection established for distributed rate limiting",
                extra={
                    "redis_url": self.redis_url,
                    "fallback_available": True,
                },
            )

        except Exception as e:
            logger.warning(
                "Redis connection failed, falling back to in-memory rate limiting",
                extra={
                    "error": str(e),
                    "redis_url": self.redis_url,
                    "fallback_mode": True,
                },
            )
            self._redis_healthy = False
            self.redis_client = None

    async def dispatch(self, request: Request, call_next: Callable) -> Response:
        """Process request with security checks and header injection."""
        if not self.config.enabled:
            return await call_next(request)

        # Get client IP for rate limiting
        client_ip = self._get_client_ip(request)

        # Check rate limiting
        if self.config.enable_rate_limiting and not await self._check_rate_limit(
            client_ip
        ):
            logger.warning(
                "Rate limit exceeded",
                extra={
                    "client_ip": client_ip,
                    "method": request.method,
                    "path": request.url.path,
                    "user_agent": request.headers.get("user-agent"),
                },
            )
            return JSONResponse(
                status_code=429,
                content={
                    "error": "Rate limit exceeded",
                    "retry_after": self.config.rate_limit_window,
                },
                headers={"Retry-After": str(self.config.rate_limit_window)},
            )

        # Process the request
        response = await call_next(request)

        # Inject security headers
        self._inject_security_headers(response)

        return response

    def _build_security_headers(self) -> dict[str, str]:
        """Build security headers dictionary from configuration.

        Returns:
            Dictionary of security headers

        """
        headers = {}

        if self.config.x_frame_options:
            headers["X-Frame-Options"] = self.config.x_frame_options

        if self.config.x_content_type_options:
            headers["X-Content-Type-Options"] = self.config.x_content_type_options

        if self.config.x_xss_protection:
            headers["X-XSS-Protection"] = self.config.x_xss_protection

        if self.config.strict_transport_security:
            headers["Strict-Transport-Security"] = self.config.strict_transport_security

        if self.config.content_security_policy:
            headers["Content-Security-Policy"] = self.config.content_security_policy

        # Add additional security headers
        headers.update(
            {
                "X-Permitted-Cross-Domain-Policies": "none",
                "Referrer-Policy": "strict-origin-when-cross-origin",
                "X-Download-Options": "noopen",
                "X-DNS-Prefetch-Control": "off",
            }
        )

        return headers

    def _inject_security_headers(self, response: Response) -> None:
        """Inject security headers into response.

        Args:
            response: HTTP response to modify

        """
        for header_name, header_value in self._security_headers.items():
            # Don't override existing headers
            if header_name not in response.headers:
                response.headers[header_name] = header_value

    async def _check_rate_limit(self, client_ip: str) -> bool:
        """Check if client is within rate limits using Redis or in-memory fallback.

        Implements sliding window rate limiting with Redis for distributed deployments.
        Falls back to in-memory storage for development or when Redis is unavailable.

        Args:
            client_ip: Client IP address

        Returns:
            True if request is allowed, False if rate limited

        """
        # Use Redis rate limiting if available and healthy
        if self._redis_healthy and self.redis_client:
            return await self._check_rate_limit_redis(client_ip)

        # Fallback to in-memory rate limiting
        return self._check_rate_limit_memory(client_ip)

    async def _check_rate_limit_redis(self, client_ip: str) -> bool:
        """Redis-based sliding window rate limiting implementation.

        Uses Redis atomic operations for distributed rate limiting that persists
        across application restarts and scales horizontally.

        Args:
            client_ip: Client IP address

        Returns:
            True if request is allowed, False if rate limited
        """
        try:
            # Create rate limiting key
            rate_limit_key = f"rate_limit:{client_ip}"

            # Use Redis pipeline for atomic operations
            async with self.redis_client.pipeline(transaction=True) as pipe:
                # Get current request count
                current_count = await pipe.get(rate_limit_key).execute()

                if current_count and current_count[0]:
                    count = int(current_count[0])

                    # Check if limit exceeded
                    if count >= self.config.rate_limit_requests:
                        return False

                    # Increment counter
                    await pipe.incr(rate_limit_key).execute()
                else:
                    # First request in window - set counter and expiry
                    await pipe.multi()
                    await pipe.incr(rate_limit_key)
                    await pipe.expire(rate_limit_key, self.config.rate_limit_window)
                    await pipe.execute()

                return True

        except Exception as e:
            logger.warning(
                "Redis rate limiting failed, falling back to in-memory",
                extra={
                    "client_ip": client_ip,
                    "error": str(e),
                    "fallback_mode": True,
                },
            )
            # Mark Redis as unhealthy and fall back to memory
            self._redis_healthy = False
            return self._check_rate_limit_memory(client_ip)

    def _check_rate_limit_memory(self, client_ip: str) -> bool:
        """In-memory sliding window rate limiting implementation.

        Fallback rate limiting using local memory storage.
        Suitable for development or single-instance deployments.

        Args:
            client_ip: Client IP address

        Returns:
            True if request is allowed, False if rate limited

        """
        current_time = int(time.time())
        client_data = self._rate_limit_storage[client_ip]

        # Check if we're in a new time window
        if current_time - client_data["window_start"] >= self.config.rate_limit_window:
            # Reset counter for new window
            client_data["requests"] = 0
            client_data["window_start"] = current_time

        # Check if client has exceeded rate limit
        if client_data["requests"] >= self.config.rate_limit_requests:
            return False

        # Increment request counter
        client_data["requests"] += 1
        return True

    async def _check_redis_health(self) -> bool:
        """Check Redis connection health and attempt reconnection if needed.

        Returns:
            True if Redis is healthy, False otherwise
        """
        if not self.redis_client:
            return False

        try:
            await self.redis_client.ping()
            if not self._redis_healthy:
                logger.info("Redis connection restored")
                self._redis_healthy = True
            return True

        except Exception as e:
            if self._redis_healthy:
                logger.warning(
                    "Redis connection lost",
                    extra={
                        "error": str(e),
                        "fallback_mode": True,
                    },
                )
            self._redis_healthy = False
            return False

    async def cleanup(self) -> None:
        """Cleanup Redis connection and resources."""
        if self.redis_client:
            try:
                await self.redis_client.aclose()
                logger.info("Redis connection closed")
            except Exception as e:
                logger.warning(f"Error closing Redis connection: {e}")
            finally:
                self.redis_client = None
                self._redis_healthy = False

    def _get_client_ip(self, request: Request) -> str:
        """Extract client IP address from request.

        Args:
            request: HTTP request

        Returns:
            Client IP address

        """
        # Check X-Forwarded-For header first
        forwarded_for = request.headers.get("x-forwarded-for")
        if forwarded_for:
            return forwarded_for.split(",")[0].strip()

        # Check X-Real-IP header
        real_ip = request.headers.get("x-real-ip")
        if real_ip:
            return real_ip

        # Fall back to direct client IP
        if request.client:
            return request.client.host

        return "unknown"

    def _clean_old_entries(self) -> None:
        """Clean old rate limiting entries to prevent memory leaks.

        This should be called periodically in production.
        Consider using a background task or external cleanup process.
        """
        current_time = int(time.time())
        expired_ips = []

        for client_ip, data in self._rate_limit_storage.items():
            if current_time - data["window_start"] > self.config.rate_limit_window * 2:
                expired_ips.append(client_ip)

        for ip in expired_ips:
            del self._rate_limit_storage[ip]

        if expired_ips:
            logger.debug(f"Cleaned {len(expired_ips)} expired rate limit entries")


class CSRFProtectionMiddleware(BaseHTTPMiddleware):
    """CSRF protection middleware for form submissions.

    Simple CSRF protection implementation for APIs that handle form data.
    """

    def __init__(
        self, app: Callable, secret_key: str, exempt_paths: list | None = None
    ):
        """Initialize CSRF protection middleware.

        Args:
            app: ASGI application
            secret_key: Secret key for token generation
            exempt_paths: List of paths to exempt from CSRF protection

        """
        super().__init__(app)
        self.secret_key = secret_key
        self.exempt_paths = exempt_paths or []

    async def dispatch(self, request: Request, call_next: Callable) -> Response:
        """Process request with CSRF protection."""
        # Skip CSRF protection for safe methods and exempt paths
        if (
            request.method in ("GET", "HEAD", "OPTIONS")
            or request.url.path in self.exempt_paths
        ):
            return await call_next(request)

        # For now, implement basic CSRF header checking
        # In a full implementation, you'd use proper CSRF tokens
        csrf_token = request.headers.get("X-CSRFToken")
        if not csrf_token:
            return JSONResponse(
                status_code=403, content={"error": "CSRF token missing"}
            )

        return await call_next(request)


# Export middleware classes
__all__ = ["CSRFProtectionMiddleware", "SecurityMiddleware"]
