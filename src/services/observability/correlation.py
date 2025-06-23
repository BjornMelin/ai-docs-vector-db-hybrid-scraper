import typing
"""Trace correlation and context propagation for distributed observability.

This module provides advanced correlation capabilities including request ID propagation,
user context tracking, business context baggage, and cross-service trace correlation
for comprehensive distributed tracing in AI/ML pipelines.
"""

import logging
import uuid
from contextlib import contextmanager
from typing import Any

from opentelemetry import baggage
from opentelemetry import context
from opentelemetry import trace
from opentelemetry.propagate import extract
from opentelemetry.propagate import inject
from opentelemetry.trace import Status
from opentelemetry.trace import StatusCode

logger = logging.getLogger(__name__)


class TraceCorrelationManager:
    """Manages trace correlation and context propagation across operations."""

    def __init__(self):
        """Initialize trace correlation manager."""
        self.tracer = trace.get_tracer(__name__)

    def generate_request_id(self) -> str:
        """Generate a unique request ID.

        Returns:
            Unique request ID string
        """
        return str(uuid.uuid4())

    def set_request_context(
        self,
        request_id: typing.Optional[str] = None,
        user_id: typing.Optional[str] = None,
        session_id: typing.Optional[str] = None,
        tenant_id: typing.Optional[str] = None,
    ) -> str:
        """Set request-level context in baggage and span attributes.

        Args:
            request_id: Unique request identifier (generated if not provided)
            user_id: User identifier
            session_id: Session identifier
            tenant_id: Tenant identifier for multi-tenant systems

        Returns:
            Request ID that was set
        """
        if request_id is None:
            request_id = self.generate_request_id()

        # Set baggage for propagation across services
        baggage.set_baggage("request.id", request_id)

        if user_id:
            baggage.set_baggage("user.id", user_id)
        if session_id:
            baggage.set_baggage("session.id", session_id)
        if tenant_id:
            baggage.set_baggage("tenant.id", tenant_id)

        # Set span attributes on current span if available
        current_span = trace.get_current_span()
        if current_span and current_span.is_recording():
            current_span.set_attribute("request.id", request_id)
            if user_id:
                current_span.set_attribute("user.id", user_id)
            if session_id:
                current_span.set_attribute("session.id", session_id)
            if tenant_id:
                current_span.set_attribute("tenant.id", tenant_id)

        logger.debug(f"Set request context: request_id={request_id}, user_id={user_id}")
        return request_id

    def set_business_context(
        self,
        operation_type: str,
        query_type: typing.Optional[str] = None,
        search_method: typing.Optional[str] = None,
        ai_provider: typing.Optional[str] = None,
        model_name: typing.Optional[str] = None,
        cache_strategy: typing.Optional[str] = None,
    ) -> None:
        """Set business-specific context for operation tracking.

        Args:
            operation_type: Type of business operation
            query_type: Type of query being processed
            search_method: Search method used (semantic, hybrid, keyword)
            ai_provider: AI/ML provider being used
            model_name: Model name being used
            cache_strategy: Caching strategy employed
        """
        # Set baggage for cross-service propagation
        baggage.set_baggage("business.operation_type", operation_type)

        if query_type:
            baggage.set_baggage("business.query_type", query_type)
        if search_method:
            baggage.set_baggage("business.search_method", search_method)
        if ai_provider:
            baggage.set_baggage("ai.provider", ai_provider)
        if model_name:
            baggage.set_baggage("ai.model", model_name)
        if cache_strategy:
            baggage.set_baggage("cache.strategy", cache_strategy)

        # Set span attributes on current span
        current_span = trace.get_current_span()
        if current_span and current_span.is_recording():
            current_span.set_attribute("business.operation_type", operation_type)
            if query_type:
                current_span.set_attribute("business.query_type", query_type)
            if search_method:
                current_span.set_attribute("business.search_method", search_method)
            if ai_provider:
                current_span.set_attribute("ai.provider", ai_provider)
            if model_name:
                current_span.set_attribute("ai.model", model_name)
            if cache_strategy:
                current_span.set_attribute("cache.strategy", cache_strategy)

    def set_performance_context(
        self,
        priority: str = "normal",
        timeout_ms: typing.Optional[int] = None,
        retry_count: int = 0,
        circuit_breaker_state: typing.Optional[str] = None,
    ) -> None:
        """Set performance-related context for monitoring.

        Args:
            priority: Operation priority (low, normal, high, critical)
            timeout_ms: Operation timeout in milliseconds
            retry_count: Current retry attempt count
            circuit_breaker_state: Circuit breaker state (closed, open, half-open)
        """
        # Set baggage
        baggage.set_baggage("performance.priority", priority)
        baggage.set_baggage("performance.retry_count", str(retry_count))

        if timeout_ms:
            baggage.set_baggage("performance.timeout_ms", str(timeout_ms))
        if circuit_breaker_state:
            baggage.set_baggage(
                "performance.circuit_breaker_state", circuit_breaker_state
            )

        # Set span attributes
        current_span = trace.get_current_span()
        if current_span and current_span.is_recording():
            current_span.set_attribute("performance.priority", priority)
            current_span.set_attribute("performance.retry_count", retry_count)
            if timeout_ms:
                current_span.set_attribute("performance.timeout_ms", timeout_ms)
            if circuit_breaker_state:
                current_span.set_attribute(
                    "performance.circuit_breaker_state", circuit_breaker_state
                )

    def get_current_context(self) -> dict[str, Any]:
        """Get current trace and baggage context.

        Returns:
            Dictionary containing current context information
        """
        context_info = {}

        # Get trace information
        current_span = trace.get_current_span()
        if current_span and current_span.is_recording():
            span_context = current_span.get_span_context()
            context_info["trace_id"] = format(span_context.trace_id, "032x")
            context_info["span_id"] = format(span_context.span_id, "016x")

        # Get baggage information
        current_baggage = baggage.get_all()
        if current_baggage:
            context_info["baggage"] = dict(current_baggage)

        return context_info

    def create_correlation_id(self, operation_name: str) -> str:
        """Create a correlation ID for linking related operations.

        Args:
            operation_name: Name of the operation

        Returns:
            Correlation ID
        """
        correlation_id = f"{operation_name}_{uuid.uuid4().hex[:8]}"

        # Set in baggage for propagation
        baggage.set_baggage("correlation.id", correlation_id)

        # Set in current span
        current_span = trace.get_current_span()
        if current_span and current_span.is_recording():
            current_span.set_attribute("correlation.id", correlation_id)

        return correlation_id

    @contextmanager
    def correlated_operation(
        self,
        operation_name: str,
        correlation_id: typing.Optional[str] = None,
        **additional_context,
    ):
        """Context manager for correlated operations.

        Args:
            operation_name: Name of the operation
            correlation_id: Existing correlation ID or None to generate
            **additional_context: Additional context to set

        Yields:
            Correlation ID for the operation
        """
        if correlation_id is None:
            correlation_id = self.create_correlation_id(operation_name)
        else:
            baggage.set_baggage("correlation.id", correlation_id)

        with self.tracer.start_as_current_span(operation_name) as span:
            # Set correlation ID in span
            span.set_attribute("correlation.id", correlation_id)
            span.set_attribute("operation.name", operation_name)

            # Set additional context
            for key, value in additional_context.items():
                span.set_attribute(f"operation.{key}", str(value))
                baggage.set_baggage(f"operation.{key}", str(value))

            try:
                yield correlation_id
                span.set_status(Status(StatusCode.OK))
            except Exception as e:
                span.record_exception(e)
                span.set_status(Status(StatusCode.ERROR, str(e)))
                raise

    def link_operations(self, parent_correlation_id: str, child_operation: str) -> str:
        """Create a linked operation with parent-child relationship.

        Args:
            parent_correlation_id: Correlation ID of parent operation
            child_operation: Name of child operation

        Returns:
            Correlation ID for child operation
        """
        child_correlation_id = f"{child_operation}_{uuid.uuid4().hex[:8]}"

        # Set parent-child relationship in baggage
        baggage.set_baggage("correlation.parent_id", parent_correlation_id)
        baggage.set_baggage("correlation.id", child_correlation_id)

        # Set in current span
        current_span = trace.get_current_span()
        if current_span and current_span.is_recording():
            current_span.set_attribute("correlation.parent_id", parent_correlation_id)
            current_span.set_attribute("correlation.id", child_correlation_id)
            current_span.add_event(
                "operation_linked",
                {
                    "parent_correlation_id": parent_correlation_id,
                    "child_correlation_id": child_correlation_id,
                },
            )

        return child_correlation_id

    def extract_context_from_headers(self, headers: dict[str, str]) -> context.Context:
        """Extract OpenTelemetry context from HTTP headers.

        Args:
            headers: HTTP headers dictionary

        Returns:
            OpenTelemetry context object
        """
        return extract(headers)

    def inject_context_to_headers(self, headers: dict[str, str]) -> None:
        """Inject current OpenTelemetry context into HTTP headers.

        Args:
            headers: HTTP headers dictionary to inject into
        """
        inject(headers)

    def propagate_context_to_background_task(self) -> context.Context:
        """Get current context for propagation to background tasks.

        Returns:
            Current OpenTelemetry context
        """
        return context.get_current()

    def run_with_context(self, ctx: context.Context, func, *args, **kwargs):
        """Run function with specific OpenTelemetry context.

        Args:
            ctx: OpenTelemetry context to use
            func: Function to run
            *args: Function arguments
            **kwargs: Function keyword arguments

        Returns:
            Function result
        """
        # Use context.attach and detach pattern for compatibility
        token = context.attach(ctx)
        try:
            return func(*args, **kwargs)
        finally:
            context.detach(token)


class ErrorCorrelationTracker:
    """Tracks and correlates errors across distributed operations."""

    def __init__(self, correlation_manager: TraceCorrelationManager):
        """Initialize error correlation tracker.

        Args:
            correlation_manager: Trace correlation manager instance
        """
        self.correlation_manager = correlation_manager
        self.tracer = trace.get_tracer(__name__)

    def record_error(
        self,
        error: Exception,
        error_type: str = "application_error",
        severity: str = "error",
        user_impact: str = "medium",
        recovery_action: typing.Optional[str] = None,
    ) -> str:
        """Record an error with correlation context.

        Args:
            error: Exception that occurred
            error_type: Type of error (validation, network, ai_model, etc.)
            severity: Error severity (low, medium, high, critical)
            user_impact: Impact on user experience
            recovery_action: Recommended recovery action

        Returns:
            Error correlation ID
        """
        error_id = str(uuid.uuid4())
        context_info = self.correlation_manager.get_current_context()

        # Get current span for error recording
        current_span = trace.get_current_span()
        if current_span and current_span.is_recording():
            # Record exception in span
            current_span.record_exception(error)
            current_span.set_status(Status(StatusCode.ERROR, str(error)))

            # Set error correlation attributes
            current_span.set_attribute("error.id", error_id)
            current_span.set_attribute("error.type", error_type)
            current_span.set_attribute("error.severity", severity)
            current_span.set_attribute("error.user_impact", user_impact)

            if recovery_action:
                current_span.set_attribute("error.recovery_action", recovery_action)

            # Add correlation context to error
            if "trace_id" in context_info:
                current_span.set_attribute("error.trace_id", context_info["trace_id"])
            if "baggage" in context_info:
                for key, value in context_info["baggage"].items():
                    current_span.set_attribute(f"error.context.{key}", value)

            # Add error event
            current_span.add_event(
                "error_occurred",
                {
                    "error.id": error_id,
                    "error.message": str(error),
                    "error.type": error_type,
                    "error.severity": severity,
                },
            )

        # Log error with correlation information
        logger.error(
            f"Correlated error occurred: {error}",
            extra={
                "error_id": error_id,
                "error_type": error_type,
                "severity": severity,
                "user_impact": user_impact,
                "recovery_action": recovery_action,
                "correlation_context": context_info,
            },
        )

        return error_id

    def create_error_span(
        self,
        error_name: str,
        error_details: dict[str, Any],
        parent_correlation_id: typing.Optional[str] = None,
    ):
        """Create a dedicated span for error analysis.

        Args:
            error_name: Name of the error for span naming
            error_details: Detailed error information
            parent_correlation_id: Parent operation correlation ID

        Returns:
            Context manager for error span
        """

        @contextmanager
        def error_span_context():
            with self.tracer.start_as_current_span(f"error.{error_name}") as span:
                # Set error span attributes
                span.set_attribute("span.kind", "error")
                span.set_attribute("error.name", error_name)

                if parent_correlation_id:
                    span.set_attribute(
                        "error.parent_correlation_id", parent_correlation_id
                    )

                # Set detailed error information
                for key, value in error_details.items():
                    span.set_attribute(f"error.details.{key}", str(value))

                try:
                    yield span
                except Exception as e:
                    span.record_exception(e)
                    span.set_status(Status(StatusCode.ERROR, str(e)))
                    raise

        return error_span_context()


# Global instances
_correlation_manager: typing.Optional[TraceCorrelationManager] = None
_error_tracker: typing.Optional[ErrorCorrelationTracker] = None


def get_correlation_manager() -> TraceCorrelationManager:
    """Get global trace correlation manager.

    Returns:
        Global TraceCorrelationManager instance
    """
    global _correlation_manager
    if _correlation_manager is None:
        _correlation_manager = TraceCorrelationManager()
    return _correlation_manager


def get_error_tracker() -> ErrorCorrelationTracker:
    """Get global error correlation tracker.

    Returns:
        Global ErrorCorrelationTracker instance
    """
    global _error_tracker
    if _error_tracker is None:
        correlation_manager = get_correlation_manager()
        _error_tracker = ErrorCorrelationTracker(correlation_manager)
    return _error_tracker


# Convenience functions
def set_request_context(
    request_id: typing.Optional[str] = None,
    user_id: typing.Optional[str] = None,
    session_id: typing.Optional[str] = None,
    tenant_id: typing.Optional[str] = None,
) -> str:
    """Set request context using global correlation manager."""
    manager = get_correlation_manager()
    return manager.set_request_context(request_id, user_id, session_id, tenant_id)


def set_business_context(
    operation_type: str,
    query_type: typing.Optional[str] = None,
    search_method: typing.Optional[str] = None,
    ai_provider: typing.Optional[str] = None,
    model_name: typing.Optional[str] = None,
    cache_strategy: typing.Optional[str] = None,
) -> None:
    """Set business context using global correlation manager."""
    manager = get_correlation_manager()
    manager.set_business_context(
        operation_type,
        query_type,
        search_method,
        ai_provider,
        model_name,
        cache_strategy,
    )


def correlated_operation(
    operation_name: str, correlation_id: typing.Optional[str] = None, **additional_context
):
    """Create correlated operation using global correlation manager."""
    manager = get_correlation_manager()
    return manager.correlated_operation(
        operation_name, correlation_id, **additional_context
    )


def record_error(
    error: Exception,
    error_type: str = "application_error",
    severity: str = "error",
    user_impact: str = "medium",
    recovery_action: typing.Optional[str] = None,
) -> str:
    """Record error using global error tracker."""
    tracker = get_error_tracker()
    return tracker.record_error(
        error, error_type, severity, user_impact, recovery_action
    )


def get_current_trace_context() -> dict[str, Any]:
    """Get current trace context information."""
    manager = get_correlation_manager()
    return manager.get_current_context()
