"""Modern error handling middleware for FastAPI.

This module provides comprehensive error handling middleware that integrates
with circuit breakers, rate limiting, and observability while using FastAPI
native patterns.
"""

import logging
import time
from typing import Any

from fastapi import Request, Response, status
from fastapi.responses import JSONResponse
from starlette.middleware.base import BaseHTTPMiddleware

from .exceptions import (
    APIException,
    CircuitBreakerException,
    RateLimitedException,
)


logger = logging.getLogger(__name__)


class ErrorHandlingMiddleware(BaseHTTPMiddleware):
    """Comprehensive error handling middleware.

    Provides:
    - Request/response error tracking
    - Circuit breaker integration
    - Rate limiting enforcement
    - Performance monitoring
    - Security error sanitization
    """

    def __init__(
        self,
        app,
        *,
        enable_detailed_errors: bool = False,
        enable_performance_tracking: bool = True,
    ):
        """Initialize error handling middleware.

        Args:
            app: FastAPI application
            enable_detailed_errors: Whether to include detailed error info
            enable_performance_tracking: Whether to track performance metrics
        """
        super().__init__(app)
        self.enable_detailed_errors = enable_detailed_errors
        self.enable_performance_tracking = enable_performance_tracking
        self._request_count = 0
        self._error_count = 0
        self._total_response_time = 0.0

    async def dispatch(self, request: Request, call_next) -> Response:
        """Process request with comprehensive error handling."""
        start_time = time.time()
        self._request_count += 1

        try:
            # Add request context for tracing
            request.state.request_id = f"req_{int(start_time * 1000)}"
            request.state.start_time = start_time

            # Process the request
            response = await call_next(request)

            # Track performance metrics
            if self.enable_performance_tracking:
                response_time = time.time() - start_time
                self._total_response_time += response_time

                # Add performance headers
                response.headers["X-Response-Time"] = f"{response_time:.3f}s"
                response.headers["X-Request-ID"] = request.state.request_id

                # Log slow requests
                if response_time > 1.0:  # Log requests taking more than 1 second
                    logger.warning(
                        f"Slow request: {request.method} {request.url.path} "
                        f"took {response_time:.3f}s"
                    )

            return response

        except APIException as e:
            # Handle our custom API exceptions
            self._error_count += 1
            return await self._handle_api_exception(request, e)

        except TimeoutError:
            # Handle timeout errors
            self._error_count += 1
            logger.warning(f"Request timeout: {request.method} {request.url.path}")

            return JSONResponse(
                status_code=status.HTTP_504_GATEWAY_TIMEOUT,
                content={
                    "error": "Request timeout",
                    "status_code": status.HTTP_504_GATEWAY_TIMEOUT,
                    "timestamp": time.time(),
                },
            )

        except ConnectionError:
            # Handle connection errors
            self._error_count += 1
            logger.exception("Connection error")

            return JSONResponse(
                status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
                content={
                    "error": "Service temporarily unavailable",
                    "status_code": status.HTTP_503_SERVICE_UNAVAILABLE,
                    "timestamp": time.time(),
                },
            )

        except Exception as e:
            # Handle unexpected errors
            self._error_count += 1
            logger.error(
                f"Unexpected error in {request.method} {request.url.path}: {e}",
                exc_info=True,
            )

            # Return sanitized error response
            error_detail = (
                str(e) if self.enable_detailed_errors else "Internal server error"
            )

            return JSONResponse(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                content={
                    "error": error_detail,
                    "status_code": status.HTTP_500_INTERNAL_SERVER_ERROR,
                    "timestamp": time.time(),
                },
            )

    async def _handle_api_exception(
        self, request: Request, exc: APIException
    ) -> JSONResponse:
        """Handle API exceptions with enhanced context."""
        content = {
            "error": exc.detail,
            "status_code": exc.status_code,
            "timestamp": exc.timestamp,
        }

        # Add request context if available
        if hasattr(request.state, "request_id"):
            content["request_id"] = request.state.request_id

        # Add safe context (filtered for security)
        if exc.context and self.enable_detailed_errors:
            safe_context = {
                k: v
                for k, v in exc.context.items()
                if not any(
                    sensitive in k.lower()
                    for sensitive in ["key", "token", "password", "secret"]
                )
            }
            if safe_context:
                content["context"] = safe_context

        return JSONResponse(
            status_code=exc.status_code,
            content=content,
            headers=exc.headers,
        )

    def get_metrics(self) -> dict[str, Any]:
        """Get error handling metrics."""
        avg_response_time = (
            self._total_response_time / self._request_count
            if self._request_count > 0
            else 0.0
        )

        error_rate = (
            self._error_count / self._request_count if self._request_count > 0 else 0.0
        )

        return {
            "total_requests": self._request_count,
            "total_errors": self._error_count,
            "error_rate": error_rate,
            "avg_response_time": avg_response_time,
        }


class CircuitBreakerMiddleware(BaseHTTPMiddleware):
    """Circuit breaker middleware for service protection.

    Integrates with the existing circuit breaker registry to protect
    against cascading failures at the HTTP layer.
    """

    def __init__(self, app, *, enable_circuit_breaker: bool = True):
        """Initialize circuit breaker middleware.

        Args:
            app: FastAPI application
            enable_circuit_breaker: Whether to enable circuit breaker protection
        """
        super().__init__(app)
        self.enable_circuit_breaker = enable_circuit_breaker

    async def dispatch(self, request: Request, call_next) -> Response:
        """Process request with circuit breaker protection."""
        if not self.enable_circuit_breaker:
            return await call_next(request)

        try:
            # Check if any critical services are in circuit breaker state
            # This would integrate with the existing CircuitBreakerRegistry
            from ..services.errors import CircuitBreakerRegistry

            # Get status of critical services
            all_status = CircuitBreakerRegistry.get_all_status()

            # Check for open circuits in critical services
            critical_services = ["vector_db", "embedding_service", "crawling_service"]
            for service_name in critical_services:
                if service_name in all_status:
                    service_status = all_status[service_name]
                    if service_status.get("state") == "open":
                        # Service is unavailable
                        retry_after = service_status.get("adaptive_timeout", 60)
                        raise CircuitBreakerException(
                            service_name=service_name,
                            retry_after=int(retry_after),
                            context={"circuit_state": service_status},
                        )

            # Process request normally
            return await call_next(request)

        except CircuitBreakerException:
            # Re-raise circuit breaker exceptions
            raise
        except Exception as e:
            # Log and continue with normal error handling
            logger.debug(f"Circuit breaker check failed: {e}")
            return await call_next(request)


class RateLimitingMiddleware(BaseHTTPMiddleware):
    """Rate limiting middleware for API protection.

    Provides basic rate limiting at the request level while integrating
    with the existing rate limiting infrastructure for AI services.
    """

    def __init__(
        self,
        app,
        *,
        requests_per_minute: int = 100,
        enable_rate_limiting: bool = True,
    ):
        """Initialize rate limiting middleware.

        Args:
            app: FastAPI application
            requests_per_minute: Maximum requests per minute per IP
            enable_rate_limiting: Whether to enable rate limiting
        """
        super().__init__(app)
        self.requests_per_minute = requests_per_minute
        self.enable_rate_limiting = enable_rate_limiting
        self._request_history: dict[str, list[float]] = {}

    async def dispatch(self, request: Request, call_next) -> Response:
        """Process request with rate limiting."""
        if not self.enable_rate_limiting:
            return await call_next(request)

        # Get client IP
        client_ip = self._get_client_ip(request)
        current_time = time.time()

        # Clean old requests and check rate limit
        if self._is_rate_limited(client_ip, current_time):
            raise RateLimitedException(
                detail=f"Rate limit exceeded: {self.requests_per_minute} requests per minute",
                retry_after=60,
                context={"client_ip": client_ip},
            )

        # Record this request
        self._record_request(client_ip, current_time)

        return await call_next(request)

    def _get_client_ip(self, request: Request) -> str:
        """Get client IP address from request."""
        # Check for forwarded headers first
        forwarded_for = request.headers.get("X-Forwarded-For")
        if forwarded_for:
            return forwarded_for.split(",")[0].strip()

        real_ip = request.headers.get("X-Real-IP")
        if real_ip:
            return real_ip

        # Fall back to client host
        if request.client:
            return request.client.host

        return "unknown"

    def _is_rate_limited(self, client_ip: str, current_time: float) -> bool:
        """Check if client is rate limited."""
        if client_ip not in self._request_history:
            return False

        # Remove requests older than 1 minute
        minute_ago = current_time - 60
        self._request_history[client_ip] = [
            req_time
            for req_time in self._request_history[client_ip]
            if req_time > minute_ago
        ]

        # Check if rate limit exceeded
        return len(self._request_history[client_ip]) >= self.requests_per_minute

    def _record_request(self, client_ip: str, request_time: float) -> None:
        """Record a request for rate limiting tracking."""
        if client_ip not in self._request_history:
            self._request_history[client_ip] = []

        self._request_history[client_ip].append(request_time)

        # Keep only recent requests to prevent memory growth
        minute_ago = request_time - 60
        self._request_history[client_ip] = [
            req_time
            for req_time in self._request_history[client_ip]
            if req_time > minute_ago
        ]


class SecurityMiddleware(BaseHTTPMiddleware):
    """Security middleware for request validation and sanitization."""

    def __init__(self, app, *, enable_security_headers: bool = True):
        """Initialize security middleware.

        Args:
            app: FastAPI application
            enable_security_headers: Whether to add security headers
        """
        super().__init__(app)
        self.enable_security_headers = enable_security_headers

    async def dispatch(self, request: Request, call_next) -> Response:
        """Process request with security enhancements."""
        # Validate request size (prevent large payload attacks)
        content_length = request.headers.get("content-length")
        if content_length and int(content_length) > 10 * 1024 * 1024:  # 10MB limit
            return JSONResponse(
                status_code=status.HTTP_413_REQUEST_ENTITY_TOO_LARGE,
                content={
                    "error": "Request payload too large",
                    "status_code": status.HTTP_413_REQUEST_ENTITY_TOO_LARGE,
                    "timestamp": time.time(),
                },
            )

        # Process request
        response = await call_next(request)

        # Add security headers
        if self.enable_security_headers:
            response.headers["X-Content-Type-Options"] = "nosniff"
            response.headers["X-Frame-Options"] = "DENY"
            response.headers["X-XSS-Protection"] = "1; mode=block"
            response.headers["Referrer-Policy"] = "strict-origin-when-cross-origin"

        return response
