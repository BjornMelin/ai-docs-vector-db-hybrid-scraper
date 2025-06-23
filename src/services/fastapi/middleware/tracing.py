import typing

"""Request tracing and correlation middleware for production observability.

This middleware provides request correlation IDs, distributed tracing support,
and comprehensive request/response logging for production monitoring.
"""

import logging
import time
import uuid
from collections.abc import Callable

from starlette.middleware.base import BaseHTTPMiddleware
from starlette.requests import Request
from starlette.responses import Response

logger = logging.getLogger(__name__)


def get_correlation_id(request: Request) -> str:
    """Get correlation ID from request headers or generate a new one.

    Args:
        request: HTTP request

    Returns:
        Correlation ID string
    """
    # Check for existing correlation ID in headers
    correlation_id = request.headers.get("x-correlation-id")
    if correlation_id:
        return correlation_id

    # Check for request ID in headers (common alternative)
    request_id = request.headers.get("x-request-id")
    if request_id:
        return request_id

    # Check if already set in request state
    if hasattr(request.state, "correlation_id"):
        return request.state.correlation_id

    # Generate a new correlation ID
    new_id = str(uuid.uuid4())
    request.state.correlation_id = new_id
    return new_id


class TracingMiddleware(BaseHTTPMiddleware):
    """Request tracing middleware with correlation IDs and observability.

    Features:
    - Automatic correlation ID generation and propagation
    - Request/response logging with timing
    - Trace context preservation
    - Configurable log levels and formats
    """

    def __init__(
        self,
        app: Callable,
        enable_request_logging: bool = True,
        enable_response_logging: bool = True,
        log_request_body: bool = False,
        log_response_body: bool = False,
        max_body_size: int = 1024,
    ):
        """Initialize tracing middleware.

        Args:
            app: ASGI application
            enable_request_logging: Enable request logging
            enable_response_logging: Enable response logging
            log_request_body: Log request body content
            log_response_body: Log response body content
            max_body_size: Maximum body size to log (bytes)
        """
        super().__init__(app)
        self.enable_request_logging = enable_request_logging
        self.enable_response_logging = enable_response_logging
        self.log_request_body = log_request_body
        self.log_response_body = log_response_body
        self.max_body_size = max_body_size

    async def dispatch(self, request: Request, call_next: Callable) -> Response:
        """Process request with tracing and correlation ID injection."""
        # Generate or extract correlation ID
        correlation_id = get_correlation_id(request)
        request.state.correlation_id = correlation_id

        # Start timing
        start_time = time.perf_counter()

        # Log incoming request
        if self.enable_request_logging:
            await self._log_request(request, correlation_id)

        try:
            # Process the request
            response = await call_next(request)

            # Calculate processing time
            end_time = time.perf_counter()
            duration = end_time - start_time

            # Add correlation ID to response headers
            response.headers["x-correlation-id"] = correlation_id
            response.headers["x-request-duration"] = f"{duration:.4f}"

            # Log response
            if self.enable_response_logging:
                await self._log_response(request, response, correlation_id, duration)

            return response

        except Exception as e:
            # Calculate processing time for error case
            end_time = time.perf_counter()
            duration = end_time - start_time

            # Log error
            logger.exception(
                "Request failed",
                extra={
                    "correlation_id": correlation_id,
                    "method": request.method,
                    "path": request.url.path,
                    "duration": duration,
                    "error": str(e),
                    "error_type": type(e).__name__,
                },
            )

            # Re-raise the exception
            raise

    async def _log_request(self, request: Request, correlation_id: str) -> None:
        """Log incoming request details.

        Args:
            request: HTTP request
            correlation_id: Request correlation ID
        """
        log_data = {
            "correlation_id": correlation_id,
            "method": request.method,
            "path": request.url.path,
            "query_params": str(request.query_params) if request.query_params else None,
            "client_ip": self._get_client_ip(request),
            "user_agent": request.headers.get("user-agent"),
            "content_type": request.headers.get("content-type"),
            "content_length": request.headers.get("content-length"),
        }

        # Log request body if enabled
        if self.log_request_body and request.method in ("POST", "PUT", "PATCH"):
            try:
                body = await self._get_request_body(request)
                if body:
                    log_data["request_body"] = body
            except Exception as e:
                log_data["request_body_error"] = str(e)

        logger.info("Incoming request", extra=log_data)

    async def _log_response(
        self,
        request: Request,
        response: Response,
        correlation_id: str,
        duration: float,
    ) -> None:
        """Log response details.

        Args:
            request: HTTP request
            response: HTTP response
            correlation_id: Request correlation ID
            duration: Request processing duration
        """
        log_data = {
            "correlation_id": correlation_id,
            "method": request.method,
            "path": request.url.path,
            "status_code": response.status_code,
            "duration": duration,
            "response_size": len(response.body) if hasattr(response, "body") else None,
            "content_type": response.headers.get("content-type"),
        }

        # Log response body if enabled
        if self.log_response_body:
            try:
                body = self._get_response_body(response)
                if body:
                    log_data["response_body"] = body
            except Exception as e:
                log_data["response_body_error"] = str(e)

        # Choose log level based on status code
        if 200 <= response.status_code < 400:
            logger.info("Request completed", extra=log_data)
        elif 400 <= response.status_code < 500:
            logger.warning("Client error response", extra=log_data)
        else:
            logger.error("Server error response", extra=log_data)

    async def _get_request_body(self, request: Request) -> str | None:
        """Get request body for logging.

        Args:
            request: HTTP request

        Returns:
            Request body as string or None
        """
        try:
            # Read body
            body = await request.body()
            if not body:
                return None

            # Limit body size for logging
            if len(body) > self.max_body_size:
                body = body[: self.max_body_size]
                truncated = True
            else:
                truncated = False

            # Decode body
            try:
                body_str = body.decode("utf-8")
                if truncated:
                    body_str += f"... (truncated at {self.max_body_size} bytes)"
                return body_str
            except UnicodeDecodeError:
                return f"<binary data: {len(body)} bytes>"

        except Exception:
            return None

    def _get_response_body(self, response: Response) -> str | None:
        """Get response body for logging.

        Args:
            response: HTTP response

        Returns:
            Response body as string or None
        """
        try:
            if not hasattr(response, "body"):
                return None

            body = response.body
            if not body:
                return None

            # Limit body size for logging
            if len(body) > self.max_body_size:
                body = body[: self.max_body_size]
                truncated = True
            else:
                truncated = False

            # Decode body
            try:
                body_str = body.decode("utf-8")
                if truncated:
                    body_str += f"... (truncated at {self.max_body_size} bytes)"
                return body_str
            except UnicodeDecodeError:
                return f"<binary data: {len(body)} bytes>"

        except Exception:
            return None

    def _get_client_ip(self, request: Request) -> str:
        """Get client IP address from request.

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


class DistributedTracingMiddleware(BaseHTTPMiddleware):
    """Distributed tracing middleware for OpenTelemetry integration.

    This middleware integrates with OpenTelemetry for distributed tracing
    across microservices.
    """

    def __init__(self, app: Callable, service_name: str = "fastapi-service"):
        """Initialize distributed tracing middleware.

        Args:
            app: ASGI application
            service_name: Name of the service for tracing
        """
        super().__init__(app)
        self.service_name = service_name

        # Try to import OpenTelemetry
        try:
            from opentelemetry import trace
            from opentelemetry.trace import Status
            from opentelemetry.trace import StatusCode

            self.tracer = trace.get_tracer(__name__)
            self.Status = Status
            self.StatusCode = StatusCode
            self.otel_available = True
        except ImportError:
            logger.warning("OpenTelemetry not available, distributed tracing disabled")
            self.otel_available = False

    async def dispatch(self, request: Request, call_next: Callable) -> Response:
        """Process request with distributed tracing."""
        if not self.otel_available:
            return await call_next(request)

        # Extract trace context from headers
        correlation_id = get_correlation_id(request)

        # Create span for this request
        span_name = f"{request.method} {request.url.path}"

        with self.tracer.start_as_current_span(span_name) as span:
            # Set span attributes
            span.set_attribute("http.method", request.method)
            span.set_attribute("http.url", str(request.url))
            span.set_attribute("http.scheme", request.url.scheme)
            span.set_attribute("http.host", request.url.hostname or "")
            span.set_attribute("http.target", request.url.path)
            span.set_attribute("correlation.id", correlation_id)
            span.set_attribute("service.name", self.service_name)

            # Add client information
            client_ip = self._get_client_ip(request)
            if client_ip:
                span.set_attribute("http.client_ip", client_ip)

            user_agent = request.headers.get("user-agent")
            if user_agent:
                span.set_attribute("http.user_agent", user_agent)

            try:
                # Process the request
                response = await call_next(request)

                # Set response attributes
                span.set_attribute("http.status_code", response.status_code)

                # Set span status based on response
                if 400 <= response.status_code < 600:
                    span.set_status(
                        self.Status(
                            self.StatusCode.ERROR,
                            f"HTTP {response.status_code}",
                        )
                    )
                else:
                    span.set_status(self.Status(self.StatusCode.OK))

                return response

            except Exception as e:
                # Record exception in span
                span.record_exception(e)
                span.set_status(self.Status(self.StatusCode.ERROR, str(e)))
                raise

    def _get_client_ip(self, request: Request) -> str | None:
        """Get client IP address from request."""
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

        return None


# Export middleware classes and utility functions
__all__ = [
    "DistributedTracingMiddleware",
    "TracingMiddleware",
    "get_correlation_id",
]
