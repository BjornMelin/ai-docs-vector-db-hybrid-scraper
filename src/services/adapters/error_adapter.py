"""Error handling adapter for transitioning from custom exceptions to FastAPI patterns.

This module provides compatibility functions to help services transition from
the custom exception hierarchy to FastAPI's native error handling while
preserving critical functionality like circuit breakers and rate limiting.
"""

import asyncio
import functools
import logging
from collections.abc import Callable
from typing import Any, TypeVar

from fastapi import status

from src.api.exceptions import (
    APIException,
    CircuitBreakerException,
    ConfigurationException,
    CrawlingException,
    EmbeddingException,
    RateLimitedException,
    VectorDBException,
    handle_service_error,
    safe_error_response,
)

from ..errors import (
    BaseError,
    ConfigurationError,
    CrawlServiceError,
    EmbeddingServiceError,
    ExternalServiceError,
    MCPError,
    NetworkError,
    QdrantServiceError,
    RateLimitError,
    ResourceError,
    ServiceError,
    ToolError,
    ValidationError,
)


logger = logging.getLogger(__name__)

F = TypeVar("F", bound=Callable[..., Any])


def convert_legacy_exception(exc: Exception) -> APIException:
    """Convert legacy custom exceptions to FastAPI-compatible exceptions.

    This function provides a bridge between the old exception hierarchy
    and the new FastAPI-native error handling system.

    Args:
        exc: Legacy exception to convert

    Returns:
        APIException: Converted FastAPI exception
    """
    context = {}

    # Extract context from legacy exceptions if available
    if hasattr(exc, "context"):
        context = exc.context or {}
    if hasattr(exc, "error_code"):
        context["error_code"] = exc.error_code

    # Map legacy exceptions to new API exceptions
    if isinstance(exc, QdrantServiceError):
        return VectorDBException(str(exc), context=context)

    elif isinstance(exc, EmbeddingServiceError):
        return EmbeddingException(str(exc), context=context)

    elif isinstance(exc, CrawlServiceError):
        return CrawlingException(str(exc), context=context)

    elif isinstance(exc, RateLimitError):
        retry_after = getattr(exc, "retry_after", None)
        return RateLimitedException(
            str(exc),
            retry_after=retry_after,
            context=context,
        )

    elif isinstance(exc, ConfigurationError):
        return ConfigurationException(str(exc), context=context)

    elif isinstance(exc, NetworkError):
        return APIException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail=f"Network error: {exc}",
            context=context,
        )

    elif isinstance(exc, ExternalServiceError):
        status_code = getattr(exc, "status_code", status.HTTP_502_BAD_GATEWAY)
        return APIException(
            status_code=status_code,
            detail=f"External service error: {exc}",
            context=context,
        )

    elif isinstance(exc, ValidationError):
        return APIException(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            detail=f"Validation error: {exc}",
            context=context,
        )

    elif isinstance(exc, ToolError | ResourceError):
        # MCP errors - these should be handled by MCP patterns
        return APIException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"MCP error: {exc}",
            context=context,
        )

    elif isinstance(exc, ServiceError | BaseError):
        # Generic service errors
        return APIException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Service error: {exc}",
            context=context,
        )

    else:
        # Unknown exception - use generic handling
        return handle_service_error(
            operation="unknown operation",
            error=exc,
            context=context,
        )


def legacy_error_handler(
    *,
    operation: str = "service operation",
    reraise_as_api_exception: bool = True,
) -> Callable[[F], F]:
    """Decorator to handle legacy exceptions in service methods.

    This decorator converts legacy custom exceptions to FastAPI-compatible
    exceptions while preserving all the original functionality.

    Args:
        operation: Name of the operation for error context
        reraise_as_api_exception: Whether to convert to APIException

    Returns:
        Decorated function
    """

    def decorator(func: F) -> F:
        @functools.wraps(func)
        async def async_wrapper(*args, **kwargs):
            try:
                return await func(*args, **kwargs)
            except APIException:
                # Already a FastAPI exception, re-raise as-is
                raise
            except Exception as e:
                logger.warning(f"Legacy exception in {operation}: {e}")

                if reraise_as_api_exception:
                    # Convert to APIException
                    api_exc = convert_legacy_exception(e)
                    # Add operation context
                    api_exc.context["operation"] = operation
                    api_exc.context["legacy_exception"] = type(e).__name__
                    raise api_exc from e
                else:
                    # Re-raise original exception
                    raise

        @functools.wraps(func)
        def sync_wrapper(*args, **kwargs):
            try:
                return func(*args, **kwargs)
            except APIException:
                # Already a FastAPI exception, re-raise as-is
                raise
            except Exception as e:
                logger.warning(f"Legacy exception in {operation}: {e}")

                if reraise_as_api_exception:
                    # Convert to APIException
                    api_exc = convert_legacy_exception(e)
                    # Add operation context
                    api_exc.context["operation"] = operation
                    api_exc.context["legacy_exception"] = type(e).__name__
                    raise api_exc from e
                else:
                    # Re-raise original exception
                    raise

        # Return appropriate wrapper based on function type
        if asyncio.iscoroutinefunction(func):
            return async_wrapper  # type: ignore
        else:
            return sync_wrapper  # type: ignore

    return decorator


def mcp_error_handler[F: Callable[..., Any]](func: F) -> F:
    """Decorator for MCP tool error handling with FastAPI integration.

    This preserves the existing MCP error handling patterns while
    integrating with the new FastAPI error system.

    Args:
        func: Function to decorate

    Returns:
        Decorated function that returns safe responses
    """

    @functools.wraps(func)
    async def wrapper(*args, **kwargs) -> dict[str, Any]:
        try:
            result = await func(*args, **kwargs)
            if isinstance(result, dict) and "success" in result:
                return result
            return safe_error_response(True, result=result)

        except (ToolError, ResourceError) as e:
            # MCP errors are sent to clients
            logger.warning(f"MCP error in {func.__name__}: {e}")
            return safe_error_response(
                False,
                error=str(e),
                error_type=getattr(e, "error_code", "mcp_error"),
            )

        except APIException as e:
            # Convert FastAPI exceptions to MCP-safe responses
            logger.warning(f"API error in {func.__name__}: {e}")
            return safe_error_response(
                False,
                error=e.detail,
                error_type="api_error",
            )

        except Exception as e:
            # Convert legacy exceptions and handle safely
            try:
                api_exc = convert_legacy_exception(e)
                logger.warning(f"Legacy error converted in {func.__name__}: {e}")
                return safe_error_response(
                    False,
                    error=api_exc.detail,
                    error_type="service_error",
                )
            except Exception:
                # Final fallback for unexpected errors
                logger.error(f"Unexpected error in {func.__name__}: {e}", exc_info=True)
                return safe_error_response(
                    False,
                    error="Internal server error",
                    error_type="internal",
                )

    return wrapper  # type: ignore


class CircuitBreakerAdapter:
    """Adapter to integrate existing circuit breaker with FastAPI exceptions."""

    @staticmethod
    def handle_circuit_breaker_error(
        service_name: str,
        error: Exception,
    ) -> CircuitBreakerException:
        """Convert circuit breaker errors to FastAPI exceptions.

        Args:
            service_name: Name of the service
            error: Original circuit breaker error

        Returns:
            CircuitBreakerException: FastAPI-compatible exception
        """
        # Extract retry information if available
        retry_after = None
        context = {"service": service_name}

        if hasattr(error, "retry_after"):
            retry_after = int(error.retry_after)
        elif hasattr(error, "adaptive_timeout"):
            retry_after = int(error.adaptive_timeout)

        if hasattr(error, "context"):
            context.update(error.context)

        return CircuitBreakerException(
            service_name=service_name,
            retry_after=retry_after,
            context=context,
        )


class RateLimitAdapter:
    """Adapter to integrate existing rate limiter with FastAPI exceptions."""

    @staticmethod
    def handle_rate_limit_error(
        error: Exception,
        *,
        retry_after: int | None = None,
    ) -> RateLimitedException:
        """Convert rate limit errors to FastAPI exceptions.

        Args:
            error: Original rate limit error
            retry_after: Optional retry delay

        Returns:
            RateLimitedException: FastAPI-compatible exception
        """
        context = {}

        # Extract retry information
        if retry_after is None and hasattr(error, "retry_after"):
            retry_after = int(error.retry_after)

        if hasattr(error, "context"):
            context.update(error.context)

        return RateLimitedException(
            detail=str(error),
            retry_after=retry_after,
            context=context,
        )


# Backward compatibility functions
def create_service_error(message: str, **kwargs) -> APIException:
    """Create a service error using the new FastAPI patterns.

    Provides backward compatibility for code that creates ServiceError instances.
    """
    return APIException(
        status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
        detail=message,
        context=kwargs,
    )


def create_validation_error(message: str, **kwargs) -> APIException:
    """Create a validation error using the new FastAPI patterns.

    Provides backward compatibility for code that creates ValidationError instances.
    """
    return APIException(
        status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
        detail=message,
        context=kwargs,
    )


def create_configuration_error(message: str, **kwargs) -> ConfigurationException:
    """Create a configuration error using the new FastAPI patterns.

    Provides backward compatibility for code that creates ConfigurationError instances.
    """
    return ConfigurationException(detail=message, context=kwargs)


# Migration utilities
def is_legacy_exception(exc: Exception) -> bool:
    """Check if an exception is from the legacy error hierarchy."""
    return isinstance(exc, BaseError | ServiceError | MCPError)


def get_error_context(exc: Exception) -> dict[str, Any]:
    """Extract error context from any exception type."""
    context = {}

    if hasattr(exc, "context") and exc.context:
        context.update(exc.context)

    if hasattr(exc, "error_code"):
        context["error_code"] = exc.error_code

    if hasattr(exc, "status_code"):
        context["status_code"] = exc.status_code

    # Add exception type information
    context["exception_type"] = type(exc).__name__

    return context
