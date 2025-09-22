"""FastAPI middleware for OpenTelemetry observability integration.

Provides middleware that enhances the existing tracing middleware with
OpenTelemetry integration while maintaining compatibility with existing patterns.
"""

import logging
import time
from collections.abc import Callable

import httpx
from starlette.middleware.base import BaseHTTPMiddleware
from starlette.requests import Request
from starlette.responses import Response

from .tracking import get_meter, get_tracer


# Optional OpenTelemetry imports
try:
    from opentelemetry.trace import Status, StatusCode
except ImportError:
    Status = None
    StatusCode = None


logger = logging.getLogger(__name__)


class FastAPIObservabilityMiddleware(BaseHTTPMiddleware):
    """FastAPI middleware for enhanced OpenTelemetry observability.

    Integrates with the existing TracingMiddleware to add OpenTelemetry
    distributed tracing with AI-specific context and attributes.
    """

    def __init__(
        self,
        app: Callable,
        service_name: str = "ai-docs-vector-db",
        record_request_metrics: bool = True,
        record_ai_context: bool = True,
    ):
        """Initialize observability middleware.

        Args:
            app: ASGI application
            service_name: Service name for tracing
            record_request_metrics: Whether to record request metrics
            record_ai_context: Whether to record AI-specific context

        """
        super().__init__(app)
        self.service_name = service_name
        self.record_request_metrics = record_request_metrics
        self.record_ai_context = record_ai_context
        self.tracer = get_tracer(f"{service_name}.middleware")

        # Initialize meter to None by default
        self.meter = None

        # Initialize request metrics
        if self.record_request_metrics:
            self._initialize_metrics()

    def _initialize_metrics(self) -> None:
        """Initialize request-level metrics."""
        try:
            self.meter = get_meter(f"{self.service_name}.requests")

            self.request_duration = self.meter.create_histogram(
                "http_request_duration_seconds",
                description="Duration of HTTP requests",
                unit="s",
            )

            self.request_counter = self.meter.create_counter(
                "http_requests_total",
                description="Total number of HTTP requests",
            )

            self.active_requests = self.meter.create_up_down_counter(
                "http_requests_active",
                description="Number of active HTTP requests",
            )

        except (httpx.HTTPError, httpx.TimeoutException, ConnectionError, Exception) as e:
            logger.warning(
                f"Failed to initialize request metrics: {e}"
            )  # TODO: Convert f-string to logging format
            self.meter = None

    async def dispatch(self, request: Request, call_next: Callable) -> Response:
        """Process request with OpenTelemetry observability."""
        # Start timing
        start_time = time.perf_counter()

        # Get correlation ID from existing middleware
        correlation_id = getattr(request.state, "correlation_id", None)

        # Create span name from method and path
        span_name = f"{request.method} {request.url.path}"

        with self.tracer.start_as_current_span(span_name) as span:
            # Set standard HTTP attributes
            span.set_attribute("http.method", request.method)
            span.set_attribute("http.url", str(request.url))
            span.set_attribute("http.scheme", request.url.scheme)
            span.set_attribute("http.host", request.url.hostname or "")
            span.set_attribute("http.target", request.url.path)
            span.set_attribute("http.user_agent", request.headers.get("user-agent", ""))

            # Add service context
            span.set_attribute("service.name", self.service_name)

            # Add correlation ID if available
            if correlation_id:
                span.set_attribute("correlation.id", correlation_id)

            # Add AI-specific context if enabled
            if self.record_ai_context:
                self._add_ai_context(request, span)

            # Track active requests
            if self.meter:
                self.active_requests.add(1)

            try:
                # Process the request
                response = await call_next(request)

                # Record response attributes
                span.set_attribute("http.status_code", response.status_code)

                # Set span status based on response
                if 400 <= response.status_code < 600 and Status and StatusCode:
                    span.set_status(
                        Status(StatusCode.ERROR, f"HTTP {response.status_code}")
                    )

                # Record metrics
                if self.meter:
                    self._record_request_metrics(request, response, start_time)

            except Exception as e:
                # Record exception
                span.record_exception(e)
                if Status and StatusCode:
                    span.set_status(Status(StatusCode.ERROR, str(e)))

                # Record error metrics
                if self.meter:
                    self._record_error_metrics(request, e, start_time)

                return Response(status_code=500)

            else:
                return response
            finally:
                # Track active requests
                if self.meter:
                    self.active_requests.add(-1)

    def _add_ai_context(self, request: Request, span) -> None:
        """Add AI-specific context to the span.

        Args:
            request: HTTP request
            span: OpenTelemetry span

        """
        try:
            # Detect AI operation types from URL path
            path = request.url.path.lower()

            if "/search" in path or "/query" in path:
                span.set_attribute("ai.operation.type", "search")
                span.set_attribute("ai.operation.category", "vector_search")

            elif "/embed" in path:
                span.set_attribute("ai.operation.type", "embedding")
                span.set_attribute("ai.operation.category", "text_embedding")

            elif "/rag" in path or "/generate" in path:
                span.set_attribute("ai.operation.type", "generation")
                span.set_attribute("ai.operation.category", "rag")

            elif "/crawl" in path or "/scrape" in path:
                span.set_attribute("ai.operation.type", "crawling")
                span.set_attribute("ai.operation.category", "web_scraping")

            # Add query parameters that might indicate AI operations
            if request.query_params:
                if "model" in request.query_params:
                    span.set_attribute("ai.model", request.query_params["model"])

                if "provider" in request.query_params:
                    span.set_attribute("ai.provider", request.query_params["provider"])

                if "query" in request.query_params:
                    # Only record first 100 chars of query for privacy
                    query = request.query_params["query"][:100]
                    span.set_attribute(
                        "ai.query.length", len(request.query_params["query"])
                    )
                    span.set_attribute("ai.query.preview", query)

            # Add content type for AI operations
            content_type = request.headers.get("content-type", "")
            if content_type:
                span.set_attribute("http.request.content_type", content_type)

        except (ValueError, TypeError, UnicodeDecodeError, Exception) as e:
            logger.debug(
                f"Failed to add AI context: {e}"
            )  # TODO: Convert f-string to logging format

    def _record_request_metrics(
        self, request: Request, response: Response, start_time: float
    ) -> None:
        """Record request metrics.

        Args:
            request: HTTP request
            response: HTTP response
            start_time: Request start time

        """
        try:
            duration = time.perf_counter() - start_time

            attributes = {
                "method": request.method,
                "status_code": str(response.status_code),
                "endpoint": request.url.path,
            }

            # Add status category
            if 200 <= response.status_code < 300:
                attributes["status_category"] = "success"
            elif 400 <= response.status_code < 500:
                attributes["status_category"] = "client_error"
            elif 500 <= response.status_code < 600:
                attributes["status_category"] = "server_error"
            else:
                attributes["status_category"] = "other"

            # Record metrics
            self.request_duration.record(duration, attributes)
            self.request_counter.add(1, attributes)

        except (ValueError, TypeError, UnicodeDecodeError, Exception) as e:
            logger.warning(
                f"Failed to record request metrics: {e}"
            )  # TODO: Convert f-string to logging format

    def _record_error_metrics(
        self, request: Request, error: Exception, start_time: float
    ) -> None:
        """Record error metrics.

        Args:
            request: HTTP request
            error: Exception that occurred
            start_time: Request start time

        """
        try:
            duration = time.perf_counter() - start_time

            attributes = {
                "method": request.method,
                "status_code": "500",  # Assume server error for exceptions
                "status_category": "server_error",
                "endpoint": request.url.path,
                "error_type": type(error).__name__,
            }

            self.request_duration.record(duration, attributes)
            self.request_counter.add(1, attributes)

        except (OSError, FileNotFoundError, PermissionError, Exception) as e:
            logger.warning(
                f"Failed to record error metrics: {e}"
            )  # TODO: Convert f-string to logging format
