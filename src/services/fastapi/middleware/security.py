"""Security middleware for production-grade security headers and protection.

This middleware adds essential security headers and provides basic protection
against common web vulnerabilities in production deployments.
"""

import logging
import time
from collections import defaultdict
from collections.abc import Callable

from starlette.middleware.base import BaseHTTPMiddleware
from starlette.requests import Request
from starlette.responses import JSONResponse, Response

from src.config import SecurityConfig


logger = logging.getLogger(__name__)


class SecurityMiddleware(BaseHTTPMiddleware):
    """Security middleware with headers injection and rate limiting.

    Features:
    - Essential security headers injection
    - Simple in-memory rate limiting
    - Request validation and filtering
    - Security event logging
    """

    def __init__(self, app: Callable, config: SecurityConfig):
        """Initialize security middleware.

        Args:
            app: ASGI application
            config: Security configuration
        """
        super().__init__(app)
        self.config = config

        # In-memory rate limiting storage
        # Note: In production, use Redis or similar distributed storage
        self._rate_limit_storage: dict[str, dict[str, int]] = defaultdict(
            lambda: {"requests": 0, "window_start": 0}
        )

        # Security headers to inject
        self._security_headers = self._build_security_headers()

    async def dispatch(self, request: Request, call_next: Callable) -> Response:
        """Process request with security checks and header injection."""
        if not self.config.enabled:
            return await call_next(request)

        # Get client IP for rate limiting
        client_ip = self._get_client_ip(request)

        # Check rate limiting
        if self.config.enable_rate_limiting and not self._check_rate_limit(client_ip):
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

    def _check_rate_limit(self, client_ip: str) -> bool:
        """Check if client is within rate limits.

        Simple sliding window rate limiting implementation.
        In production, consider using Redis with atomic operations.

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
