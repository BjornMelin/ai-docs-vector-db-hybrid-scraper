"""Modern FastAPI exception handling for AI Docs Vector DB Hybrid Scraper.

This module provides FastAPI-native exception handling while maintaining
compatibility with existing service patterns and preserving critical features.
"""

import json
import logging
import time
from typing import Any

from fastapi import HTTPException, Request, status
from fastapi.exception_handlers import http_exception_handler
from fastapi.responses import JSONResponse
from pydantic import ValidationError
from starlette.exceptions import HTTPException as StarletteHTTPException


logger = logging.getLogger(__name__)


class APIException(HTTPException):
    """Enhanced HTTPException with additional context and logging.

    Maintains compatibility with existing error patterns while using
    FastAPI's native exception system.
    """

    def __init__(
        self,
        status_code: int,
        detail: str,
        *,
        headers: dict[str, str] | None = None,
        context: dict[str, Any] | None = None,
        log_level: str = "warning",
    ):
        """Initialize API exception with enhanced context.

        Args:
            status_code: HTTP status code
            detail: Error message
            headers: Optional HTTP headers
            context: Additional error context for logging
            log_level: Logging level (debug, info, warning, error)
        """
        super().__init__(status_code=status_code, detail=detail, headers=headers)
        self.context = context or {}
        self.timestamp = time.time()

        # Log the error with context
        log_method = getattr(logger, log_level, logger.warning)
        context_str = ", ".join(f"{k}={v}" for k, v in self.context.items())
        log_method(
            f"API Exception [{status_code}]: {detail}"
            + (f" (Context: {context_str})" if context_str else "")
        )


# Service-specific exceptions using FastAPI patterns
class VectorDBException(APIException):
    """Vector database operation errors."""

    def __init__(self, detail: str, *, context: dict[str, Any] | None = None):
        super().__init__(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail=f"Vector database error: {detail}",
            context=context,
        )


class EmbeddingException(APIException):
    """Embedding service errors."""

    def __init__(self, detail: str, *, context: dict[str, Any] | None = None):
        super().__init__(
            status_code=status.HTTP_502_BAD_GATEWAY,
            detail=f"Embedding service error: {detail}",
            context=context,
        )


class CrawlingException(APIException):
    """Web crawling service errors."""

    def __init__(self, detail: str, *, context: dict[str, Any] | None = None):
        super().__init__(
            status_code=status.HTTP_502_BAD_GATEWAY,
            detail=f"Crawling service error: {detail}",
            context=context,
        )


class RateLimitedException(APIException):
    """Rate limiting errors with retry information."""

    def __init__(
        self,
        detail: str = "Rate limit exceeded",
        *,
        retry_after: int | None = None,
        context: dict[str, Any] | None = None,
    ):
        headers = {"Retry-After": str(retry_after)} if retry_after else None
        context = context or {}
        if retry_after:
            context["retry_after"] = retry_after

        super().__init__(
            status_code=status.HTTP_429_TOO_MANY_REQUESTS,
            detail=detail,
            headers=headers,
            context=context,
        )


class ConfigurationException(APIException):
    """Configuration and environment errors."""

    def __init__(self, detail: str, *, context: dict[str, Any] | None = None):
        super().__init__(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Configuration error: {detail}",
            context=context,
            log_level="error",
        )


class CircuitBreakerException(APIException):
    """Circuit breaker is open - service unavailable."""

    def __init__(
        self,
        service_name: str,
        *,
        retry_after: int | None = None,
        context: dict[str, Any] | None = None,
    ):
        headers = {"Retry-After": str(retry_after)} if retry_after else None
        context = context or {}
        context["service"] = service_name
        if retry_after:
            context["retry_after"] = retry_after

        super().__init__(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail=f"Service '{service_name}' is temporarily unavailable",
            headers=headers,
            context=context,
        )


# Exception handlers
async def api_exception_handler(request: Request, exc: APIException) -> JSONResponse:
    """Handle APIException with enhanced error response."""
    content = {
        "error": exc.detail,
        "status_code": exc.status_code,
        "timestamp": exc.timestamp,
    }

    # Add context if present (filtered for security)
    if exc.context:
        # Filter out sensitive information
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


async def validation_exception_handler(
    request: Request, exc: ValidationError
) -> JSONResponse:
    """Handle Pydantic validation errors."""
    errors = []
    for error in exc.errors():
        error_detail = {
            "field": ".".join(str(loc) for loc in error["loc"]),
            "message": error["msg"],
            "type": error["type"],
        }

        # Mask sensitive input values
        if "input" in error:
            input_val = error["input"]
            if any(
                sensitive in str(error["loc"]).lower()
                for sensitive in ["key", "token", "password", "secret"]
            ):
                input_val = "***MASKED***"
            error_detail["input"] = input_val

        errors.append(error_detail)

    logger.warning(f"Validation error: {len(errors)} validation errors occurred")

    return JSONResponse(
        status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
        content={
            "error": "Validation failed",
            "status_code": status.HTTP_422_UNPROCESSABLE_ENTITY,
            "errors": errors,
            "timestamp": time.time(),
        },
    )


async def generic_exception_handler(request: Request, exc: Exception) -> JSONResponse:
    """Handle unexpected exceptions safely."""
    # Log the full exception for debugging
    logger.error(f"Unexpected error in {request.url.path}: {exc}", exc_info=True)

    # Return generic error message to avoid information leakage
    return JSONResponse(
        status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
        content={
            "error": "Internal server error",
            "status_code": status.HTTP_500_INTERNAL_SERVER_ERROR,
            "timestamp": time.time(),
        },
    )


async def http_exception_handler_override(
    request: Request, exc: StarletteHTTPException
) -> JSONResponse:
    """Enhanced HTTP exception handler with consistent response format."""
    # Use FastAPI's default handler for standard behavior
    response = await http_exception_handler(request, exc)

    # If it's a JSONResponse, enhance the content format
    if isinstance(response, JSONResponse):
        try:
            content = response.body.decode() if response.body else "{}"
            original_content = json.loads(content)

            # Enhance with timestamp if not already present
            if (
                isinstance(original_content, dict)
                and "timestamp" not in original_content
            ):
                original_content["timestamp"] = time.time()
                original_content["status_code"] = exc.status_code

                response = JSONResponse(
                    status_code=exc.status_code,
                    content=original_content,
                    headers=response.headers,
                )
        except (json.JSONDecodeError, UnicodeDecodeError):
            # If we can't parse the response, just add timestamp
            response = JSONResponse(
                status_code=exc.status_code,
                content={
                    "detail": exc.detail,
                    "status_code": exc.status_code,
                    "timestamp": time.time(),
                },
                headers=response.headers,
            )

    return response


# Utility functions for service error handling
def handle_service_error(
    operation: str,
    error: Exception,
    *,
    context: dict[str, Any] | None = None,
    default_status: int = status.HTTP_500_INTERNAL_SERVER_ERROR,
) -> APIException:
    """Convert service errors to appropriate API exceptions.

    Args:
        operation: Operation that failed
        error: Original exception
        context: Additional error context
        default_status: Default HTTP status code

    Returns:
        APIException: Appropriate API exception
    """
    context = context or {}
    context["operation"] = operation
    context["original_error"] = type(error).__name__

    error_msg = str(error)

    # Map common error patterns to appropriate exceptions
    if "connection" in error_msg.lower() or "timeout" in error_msg.lower():
        return APIException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail=f"{operation} failed due to connectivity issues",
            context=context,
        )
    elif "rate limit" in error_msg.lower():
        return RateLimitedException(
            detail=f"{operation} failed due to rate limiting",
            context=context,
        )
    elif "validation" in error_msg.lower() or "invalid" in error_msg.lower():
        return APIException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"{operation} failed due to invalid input: {error_msg}",
            context=context,
        )
    else:
        return APIException(
            status_code=default_status,
            detail=f"{operation} failed: {error_msg}",
            context=context,
            log_level="error",
        )


def safe_error_response(
    success: bool,
    error: Exception | str | None = None,
    **kwargs: Any,
) -> dict[str, Any]:
    """Create safe error responses for MCP and internal tools.

    Maintains compatibility with existing MCP patterns while using
    modern error handling internally.

    Args:
        success: Whether operation succeeded
        error: Error message or exception
        **kwargs: Additional response data

    Returns:
        Safe response dictionary
    """
    response = {"success": success, "timestamp": time.time()}

    if success:
        response.update(kwargs)
    else:
        # Sanitize error message
        if isinstance(error, Exception):
            error_msg = str(error)
        else:
            error_msg = error or "Unknown error"

        # Remove sensitive information
        error_msg = error_msg.replace("/home/", "/****/")
        for sensitive in ["api_key", "token", "password", "secret"]:
            error_msg = error_msg.replace(sensitive, "***")

        response["error"] = error_msg
        response["error_type"] = kwargs.get("error_type", "general")

    return response
