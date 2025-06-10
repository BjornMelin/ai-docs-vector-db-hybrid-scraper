"""Request tracing middleware for correlation ID tracking and observability.

This middleware provides request correlation ID generation and injection,
essential for distributed tracing and log correlation in production environments.
"""

import logging
import time
import uuid
from collections.abc import Callable

from src.config.fastapi import TracingConfig
from starlette.middleware.base import BaseHTTPMiddleware
from starlette.requests import Request
from starlette.responses import Response

logger = logging.getLogger(__name__)


class TracingMiddleware(BaseHTTPMiddleware):
    """Request tracing middleware with correlation ID and performance monitoring.

    Features:
    - Automatic correlation ID generation and injection
    - Request/response logging with structured format
    - Request timing and performance monitoring
    - Thread-safe correlation ID context
    """

    def __init__(self, app: Callable, config: TracingConfig):
        """Initialize tracing middleware.

        Args:
            app: ASGI application
            config: Tracing configuration
        """
        super().__init__(app)
        self.config = config

        # Setup structured logger for requests
        self.request_logger = logging.getLogger(f"{__name__}.requests")

    async def dispatch(self, request: Request, call_next: Callable) -> Response:
        """Process request with tracing and correlation ID injection."""
        if not self.config.enabled:
            return await call_next(request)

        # Generate or extract correlation ID
        correlation_id = self._get_or_generate_correlation_id(request)

        # Add correlation ID to request state for access in route handlers
        request.state.correlation_id = correlation_id

        # Start timing
        start_time = time.perf_counter()

        # Log request details
        if self.config.log_requests:
            self._log_request(request, correlation_id)

        try:
            # Process request
            response = await call_next(request)

            # Calculate processing time
            process_time = time.perf_counter() - start_time

            # Add correlation ID and timing to response headers
            response.headers[self.config.correlation_id_header] = correlation_id
            response.headers["X-Process-Time"] = f"{process_time:.4f}"

            # Log response details
            if self.config.log_responses:
                self._log_response(response, correlation_id, process_time)

            return response

        except Exception as e:
            # Calculate processing time for failed requests
            process_time = time.perf_counter() - start_time

            # Log error with correlation ID
            logger.error(
                "Request failed",
                extra={
                    "correlation_id": correlation_id,
                    "method": request.method,
                    "url": str(request.url),
                    "process_time": process_time,
                    "error": str(e),
                    "error_type": type(e).__name__,
                },
            )

            # Re-raise the exception
            raise

    def _get_or_generate_correlation_id(self, request: Request) -> str:
        """Get existing correlation ID from headers or generate new one.

        Args:
            request: HTTP request

        Returns:
            Correlation ID string
        """
        # Try to get existing correlation ID from headers
        correlation_id = request.headers.get(self.config.correlation_id_header)

        if not correlation_id and self.config.generate_correlation_id:
            # Generate new UUID-based correlation ID
            correlation_id = str(uuid.uuid4())

        return correlation_id or "unknown"

    def _log_request(self, request: Request, correlation_id: str) -> None:
        """Log structured request information.

        Args:
            request: HTTP request
            correlation_id: Request correlation ID
        """
        # Build request log data
        log_data = {
            "event": "request_started",
            "correlation_id": correlation_id,
            "method": request.method,
            "url": str(request.url),
            "path": request.url.path,
            "query_params": dict(request.query_params),
            "user_agent": request.headers.get("user-agent"),
            "client_ip": self._get_client_ip(request),
        }

        # Add content type and length for POST/PUT requests
        if request.method in ("POST", "PUT", "PATCH"):
            log_data.update(
                {
                    "content_type": request.headers.get("content-type"),
                    "content_length": request.headers.get("content-length"),
                }
            )

        self.request_logger.info("Request started", extra=log_data)

    def _log_response(
        self, response: Response, correlation_id: str, process_time: float
    ) -> None:
        """Log structured response information.

        Args:
            response: HTTP response
            correlation_id: Request correlation ID
            process_time: Request processing time in seconds
        """
        log_data = {
            "event": "request_completed",
            "correlation_id": correlation_id,
            "status_code": response.status_code,
            "process_time": process_time,
            "content_type": response.headers.get("content-type"),
            "content_length": response.headers.get("content-length"),
        }

        # Determine log level based on status code
        if response.status_code >= 500:
            log_level = logging.ERROR
        elif response.status_code >= 400:
            log_level = logging.WARNING
        else:
            log_level = logging.INFO

        self.request_logger.log(log_level, "Request completed", extra=log_data)

    def _get_client_ip(self, request: Request) -> str:
        """Extract client IP address from request headers.

        Handles common proxy headers like X-Forwarded-For.

        Args:
            request: HTTP request

        Returns:
            Client IP address
        """
        # Check X-Forwarded-For header first (common in load balancers)
        forwarded_for = request.headers.get("x-forwarded-for")
        if forwarded_for:
            # Take the first IP in the chain
            return forwarded_for.split(",")[0].strip()

        # Check X-Real-IP header (nginx)
        real_ip = request.headers.get("x-real-ip")
        if real_ip:
            return real_ip

        # Fall back to direct client IP
        client_host = (
            getattr(request.client, "host", "unknown") if request.client else "unknown"
        )
        return client_host


def get_correlation_id(request: Request) -> str:
    """Get correlation ID from request state.

    Utility function to access correlation ID in route handlers.

    Args:
        request: HTTP request

    Returns:
        Correlation ID string
    """
    return getattr(request.state, "correlation_id", "unknown")


# Export middleware class and utility function
__all__ = ["TracingMiddleware", "get_correlation_id"]
