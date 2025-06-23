
"""Correlation ID utilities for request tracking.

This module provides utilities for managing correlation IDs across requests
for improved observability and debugging in production environments.
"""

import uuid

from starlette.requests import Request


def get_correlation_id(request: Request) -> str:
    """Get correlation ID from request headers or generate a new one.

    This function checks for correlation ID in various header formats
    and generates a new UUID if none is found.

    Args:
        request: HTTP request object

    Returns:
        Correlation ID string
    """
    # Check for existing correlation ID in headers (most common)
    correlation_id = request.headers.get("x-correlation-id")
    if correlation_id:
        return correlation_id

    # Check for request ID in headers (alternative format)
    request_id = request.headers.get("x-request-id")
    if request_id:
        return request_id

    # Check for trace ID (OpenTelemetry format)
    trace_id = request.headers.get("x-trace-id")
    if trace_id:
        return trace_id

    # Check if already set in request state
    if hasattr(request.state, "correlation_id"):
        return request.state.correlation_id

    # Generate a new correlation ID
    new_id = str(uuid.uuid4())
    request.state.correlation_id = new_id
    return new_id


def set_correlation_id(request: Request, correlation_id: str) -> None:
    """Set correlation ID in request state.

    Args:
        request: HTTP request object
        correlation_id: Correlation ID to set
    """
    request.state.correlation_id = correlation_id


def generate_correlation_id() -> str:
    """Generate a new correlation ID.

    Returns:
        New correlation ID as UUID string
    """
    return str(uuid.uuid4())


# Export utility functions
__all__ = [
    "generate_correlation_id",
    "get_correlation_id",
    "set_correlation_id",
]
