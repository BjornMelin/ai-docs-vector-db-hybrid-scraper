"""Request tracing and correlation middleware for production observability.

This middleware provides request correlation IDs, distributed tracing support,
and comprehensive request/response logging for production monitoring.
"""

import asyncio
import html
import logging
import time
import uuid
from collections.abc import Callable


try:
    import httpx
except ImportError:
    httpx = None

from starlette.middleware.base import BaseHTTPMiddleware
from starlette.requests import Request
from starlette.responses import Response


try:
    from opentelemetry import trace
    from opentelemetry.trace import Status, StatusCode
except ImportError:
    trace = None
    Status = None
    StatusCode = None


logger = logging.getLogger(__name__)


def _raise_opentelemetry_not_available() -> None:
    """Raise ImportError for opentelemetry not available."""
    msg = "opentelemetry not available"
    raise ImportError(msg)


def _safe_escape_for_logging(value: str | None) -> str | None:
    """Safely escape user input for logging to prevent XSS in log viewers.

    Args:
        value: String value that may contain user input

    Returns:
        HTML-escaped string or None if input was None
    """
    if value is None:
        return None
    return html.escape(str(value))


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
        *,
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
            response = await call_next(request)
        except Exception as e:
            self._log_request_error(e, correlation_id, request, start_time)
            raise
        else:
            return self._finalize_response(
                response, correlation_id, request, start_time
            )

    def _log_request_error(
        self, error: Exception, correlation_id: str, request: Request, start_time: float
    ) -> None:
        """Log request processing error.

        Args:
            error: Exception that occurred
            correlation_id: Request correlation ID
            request: HTTP request
            start_time: Request start time
        """
        end_time = time.perf_counter()
        duration = end_time - start_time

        logger.exception(
            "Request failed",
            extra={
                "correlation_id": correlation_id,
                "method": request.method,
                "path": request.url.path,
                "duration": duration,
                "error": str(error),
                "error_type": type(error).__name__,
            },
        )

    async def _finalize_response(
        self,
        response: Response,
        correlation_id: str,
        request: Request,
        start_time: float,
    ) -> Response:
        """Finalize response with headers and logging.

        Args:
            response: HTTP response
            correlation_id: Request correlation ID
            request: HTTP request
            start_time: Request start time

        Returns:
            Response with correlation headers
        """
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
            "query_params": _safe_escape_for_logging(str(request.query_params))
            if request.query_params
            else None,
            "client_ip": self._get_client_ip(request),
            "user_agent": _safe_escape_for_logging(request.headers.get("user-agent")),
            "content_type": request.headers.get("content-type"),
            "content_length": request.headers.get("content-length"),
        }

        # Log request body if enabled
        if self.log_request_body and request.method in ("POST", "PUT", "PATCH"):
            try:
                body = await self._get_request_body(request)
            except (asyncio.CancelledError, TimeoutError, RuntimeError) as e:
                log_data["request_body_error"] = _safe_escape_for_logging(str(e))
            else:
                if body:
                    log_data["request_body"] = _safe_escape_for_logging(body)

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
            except (httpx.HTTPError, httpx.ResponseNotRead, ValueError) as e:
                log_data["response_body_error"] = _safe_escape_for_logging(str(e))
            else:
                if body:
                    log_data["response_body"] = _safe_escape_for_logging(body)

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
            except UnicodeDecodeError:
                return f"<binary data: {len(body)} bytes>"

            else:
                return body_str
        except (RuntimeError, TypeError, UnicodeDecodeError, ValueError):
            return None

    def _get_response_body(self, response: Response) -> str | None:
        """Get response body for logging.

        Args:
            response: HTTP response

        Returns:
            Response body as string or None

        """
        try:
            body = self._extract_response_body(response)
        except (ConnectionError, OSError, RuntimeError, TimeoutError):
            return None

        if not body:
            return None

        return self._format_body_for_logging(body)

    def _extract_response_body(self, response: Response) -> bytes | None:
        """Extract response body safely.

        Args:
            response: HTTP response

        Returns:
            Response body as bytes or None
        """
        if not hasattr(response, "body"):
            return None

        return response.body

    def _format_body_for_logging(self, body: bytes) -> str:
        """Format response body for logging.

        Args:
            body: Response body as bytes

        Returns:
            Formatted body string
        """
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
        except UnicodeDecodeError:
            return f"<binary data: {len(body)} bytes>"
        else:
            return body_str

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
            self._initialize_tracing()
        except ImportError:
            logger.warning("OpenTelemetry not available, distributed tracing disabled")
            self.otel_available = False

    def _initialize_tracing(self) -> None:
        """Initialize OpenTelemetry tracing components.

        Raises:
            ImportError: If OpenTelemetry is not available
        """
        if trace is None:
            _raise_opentelemetry_not_available()

        self.tracer = trace.get_tracer(__name__)
        self.Status = Status
        self.StatusCode = StatusCode
        self.otel_available = True

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
                response = await call_next(request)
            except Exception as e:
                self._handle_request_exception(span, e)
                raise
            else:
                self._set_response_attributes(span, response)
                return response

    def _handle_request_exception(self, span, exception: Exception) -> None:
        """Handle request exception in tracing span.

        Args:
            span: OpenTelemetry span
            exception: Exception that occurred
        """
        span.record_exception(exception)
        span.set_status(self.Status(self.StatusCode.ERROR, str(exception)))

    def _set_response_attributes(self, span, response: Response) -> None:
        """Set response attributes on tracing span.

        Args:
            span: OpenTelemetry span
            response: HTTP response
        """
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
