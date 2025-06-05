"""Consolidated error classes for all services and MCP server.

This module provides a comprehensive error hierarchy for the AI Documentation Vector DB project,
following best practices from Pydantic 2.0 and FastMCP 2.0.

Error Hierarchy:
    BaseError: Root exception with error_code, message, and context
    ├── ServiceError: Base for service-layer errors
    │   ├── QdrantServiceError: Vector database errors
    │   ├── EmbeddingServiceError: Embedding generation errors
    │   ├── CrawlServiceError: Web crawling errors
    │   └── CacheServiceError: Caching layer errors
    ├── ValidationError: Input validation errors (Pydantic integration)
    ├── MCPError: MCP server-specific errors
    │   ├── ToolError: MCP tool execution errors
    │   └── ResourceError: MCP resource access errors
    ├── APIError: External API integration errors
    │   ├── RateLimitError: Rate limiting errors
    │   ├── NetworkError: Network connectivity errors
    │   └── ExternalServiceError: General external service errors
    └── ConfigurationError: Configuration and environment errors
"""

import asyncio
import functools
import logging
import time
from collections.abc import Callable
from typing import Any
from typing import TypeVar

from pydantic import ValidationError as PydanticValidationError
from pydantic_core import PydanticCustomError

logger = logging.getLogger(__name__)

F = TypeVar("F", bound=Callable[..., Any])


# Base Error Classes
class BaseError(Exception):
    """Base error class with enhanced context support."""

    def __init__(
        self,
        message: str,
        error_code: str | None = None,
        context: dict[str, Any] | None = None,
    ):
        """Initialize base error.

        Args:
            message: Human-readable error message
            error_code: Machine-readable error code for categorization
            context: Additional context information
        """
        self.message = message
        self.error_code = error_code or self.__class__.__name__
        self.context = context or {}
        super().__init__(message)

    def to_dict(self) -> dict[str, Any]:
        """Convert error to dictionary for API responses."""
        return {
            "error": self.message,
            "error_code": self.error_code,
            "error_type": self.__class__.__name__,
            "context": self.context,
        }


# Service Layer Errors
class ServiceError(BaseError):
    """Base class for service-layer errors."""

    pass


class QdrantServiceError(ServiceError):
    """Qdrant vector database errors."""

    pass


class EmbeddingServiceError(ServiceError):
    """Embedding service errors."""

    pass


class CrawlServiceError(ServiceError):
    """Web crawling service errors."""

    pass


class CacheServiceError(ServiceError):
    """Cache service errors."""

    pass


# Validation Errors
class ValidationError(BaseError):
    """Input validation error with Pydantic integration."""

    @classmethod
    def from_pydantic(cls, exc: PydanticValidationError) -> "ValidationError":
        """Create ValidationError from Pydantic ValidationError."""
        errors = exc.errors()
        if len(errors) == 1:
            error = errors[0]
            return cls(
                message=error["msg"],
                error_code="validation_error",
                context={
                    "field": ".".join(str(loc) for loc in error["loc"]),
                    "type": error["type"],
                    "input": error.get("input"),
                },
            )
        else:
            return cls(
                message="Multiple validation errors occurred",
                error_code="validation_error",
                context={"errors": errors},
            )


# MCP Server Errors (FastMCP pattern)
class MCPError(BaseError):
    """Base MCP server error."""

    pass


class ToolError(MCPError):
    """MCP tool execution error.

    Following FastMCP pattern where ToolError contents are sent to clients.
    """

    pass


class ResourceError(MCPError):
    """MCP resource access error.

    Following FastMCP pattern where ResourceError contents are sent to clients.
    """

    pass


# API Integration Errors
class APIError(BaseError):
    """Base error for external API integrations."""

    def __init__(
        self,
        message: str,
        status_code: int | None = None,
        error_code: str | None = None,
        context: dict[str, Any] | None = None,
    ):
        """Initialize API error with HTTP status code."""
        super().__init__(message, error_code, context)
        self.status_code = status_code


class ExternalServiceError(APIError):
    """General external service error."""

    pass


class RateLimitError(ExternalServiceError):
    """Rate limiting error with retry information."""

    def __init__(
        self,
        message: str,
        retry_after: float | None = None,
        error_code: str | None = None,
        context: dict[str, Any] | None = None,
    ):
        """Initialize rate limit error.

        Args:
            message: Error message
            retry_after: Seconds to wait before retry
            error_code: Error code
            context: Additional context
        """
        super().__init__(message, 429, error_code, context)
        self.retry_after = retry_after
        if retry_after:
            self.context["retry_after"] = retry_after


class NetworkError(ExternalServiceError):
    """Network connectivity error."""

    def __init__(
        self,
        message: str,
        error_code: str | None = None,
        context: dict[str, Any] | None = None,
    ):
        """Initialize network error."""
        super().__init__(message, 503, error_code, context)


class ConfigurationError(BaseError):
    """Configuration or environment error."""

    pass


# Utility Functions and Decorators
def safe_response(success: bool, **kwargs) -> dict[str, Any]:
    """Create a safe response dictionary for MCP tools.

    Args:
        success: Whether the operation was successful
        **kwargs: Additional response data

    Returns:
        Safe response dictionary with sanitized error messages
    """
    response = {"success": success, "timestamp": time.time()}

    if success:
        response.update(kwargs)
    else:
        # Sanitize error messages
        error = kwargs.get("error", "Unknown error")
        if isinstance(error, Exception):
            error = str(error)

        # Don't expose internal paths or sensitive info
        error = error.replace("/home/", "/****/")
        error = error.replace("api_key", "***")
        error = error.replace("token", "***")
        error = error.replace("password", "***")
        error = error.replace("secret", "***")

        response["error"] = error
        response["error_type"] = kwargs.get("error_type", "general")

    return response


def retry_async(
    max_attempts: int = 3,
    base_delay: float = 1.0,
    max_delay: float = 60.0,
    backoff_factor: float = 2.0,
    exceptions: tuple = (Exception,),
) -> Callable[[F], F]:
    """Async retry decorator with exponential backoff.

    Args:
        max_attempts: Maximum number of retry attempts
        base_delay: Base delay between attempts in seconds
        max_delay: Maximum delay between attempts in seconds
        backoff_factor: Factor to increase delay by each attempt
        exceptions: Exception types to retry on

    Returns:
        Decorated function
    """

    def decorator(func: F) -> F:
        @functools.wraps(func)
        async def wrapper(*args, **kwargs):
            last_exception = None

            for attempt in range(max_attempts):
                try:
                    return await func(*args, **kwargs)
                except exceptions as e:
                    last_exception = e

                    if attempt == max_attempts - 1:
                        logger.error(f"Final attempt failed for {func.__name__}: {e}")
                        break

                    delay = min(base_delay * (backoff_factor**attempt), max_delay)

                    logger.warning(
                        f"Attempt {attempt + 1}/{max_attempts} failed for {func.__name__}: {e}. "
                        f"Retrying in {delay:.1f}s..."
                    )

                    await asyncio.sleep(delay)
                except Exception as e:
                    # Non-retryable error
                    logger.error(f"Non-retryable error in {func.__name__}: {e}")
                    raise

            raise last_exception or Exception("All retry attempts failed")

        return wrapper  # type: ignore

    return decorator


def circuit_breaker(
    failure_threshold: int = 5,
    recovery_timeout: float = 60.0,
    expected_exception: type = Exception,
) -> Callable[[F], F]:
    """Circuit breaker pattern decorator.

    Args:
        failure_threshold: Number of failures before opening circuit
        recovery_timeout: Time to wait before attempting recovery
        expected_exception: Exception type that triggers circuit breaker

    Returns:
        Decorated function
    """

    def decorator(func: F) -> F:
        failure_count = 0
        last_failure_time: float | None = None
        state = "closed"  # closed, open, half-open

        @functools.wraps(func)
        async def wrapper(*args, **kwargs):
            nonlocal failure_count, last_failure_time, state

            # Check if circuit should be half-open
            if state == "open":
                if (
                    last_failure_time
                    and time.time() - last_failure_time > recovery_timeout
                ):
                    state = "half-open"
                    logger.info(f"Circuit breaker for {func.__name__} is half-open")
                else:
                    wait_time = (
                        recovery_timeout - (time.time() - last_failure_time)
                        if last_failure_time
                        else recovery_timeout
                    )
                    raise ExternalServiceError(
                        f"Circuit breaker is open for {func.__name__}. "
                        f"Try again in {wait_time:.1f}s"
                    )

            try:
                result = await func(*args, **kwargs)

                # Reset on success
                if state == "half-open":
                    state = "closed"
                    failure_count = 0
                    logger.info(
                        f"Circuit breaker for {func.__name__} closed (recovered)"
                    )

                return result

            except expected_exception as e:
                failure_count += 1
                last_failure_time = time.time()

                logger.warning(
                    f"Circuit breaker failure {failure_count}/{failure_threshold} "
                    f"for {func.__name__}: {e}"
                )

                if failure_count >= failure_threshold:
                    state = "open"
                    logger.error(f"Circuit breaker opened for {func.__name__}")

                raise

        return wrapper  # type: ignore

    return decorator


def handle_mcp_errors(func: F) -> F:
    """Decorator to handle MCP tool errors safely.

    Following FastMCP patterns where ToolError and ResourceError
    contents are sent to clients while other exceptions are masked.

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
            return safe_response(True, result=result)
        except (ToolError, ResourceError) as e:
            # These errors are meant to be sent to clients
            logger.warning(f"MCP error in {func.__name__}: {e}")
            return safe_response(
                False, error=str(e), error_type=e.error_code or "mcp_error"
            )
        except (
            ValidationError,
            RateLimitError,
            NetworkError,
            ExternalServiceError,
            ConfigurationError,
        ) as e:
            error_type_map = {
                ValidationError: "validation",
                RateLimitError: "rate_limit",
                NetworkError: "network",
                ExternalServiceError: "external_service",
                ConfigurationError: "configuration",
            }
            error_type = error_type_map.get(type(e), "general")
            logger.warning(f"{error_type.capitalize()} error in {func.__name__}: {e}")
            return safe_response(False, error=str(e), error_type=error_type)
        except Exception as e:
            # Mask internal errors for security
            logger.error(f"Unexpected error in {func.__name__}: {e}", exc_info=True)
            return safe_response(
                False, error="Internal server error", error_type="internal"
            )

    return wrapper  # type: ignore


def validate_input(**validators) -> Callable[[F], F]:
    """Decorator to validate function inputs using Pydantic-style validation.

    Args:
        **validators: Keyword arguments mapping parameter names to validator functions

    Returns:
        Decorated function
    """

    def decorator(func: F) -> F:
        @functools.wraps(func)
        async def wrapper(*args, **kwargs):
            # Get function signature
            import inspect

            sig = inspect.signature(func)
            bound_args = sig.bind(*args, **kwargs)
            bound_args.apply_defaults()

            # Validate arguments
            for param_name, validator in validators.items():
                if param_name in bound_args.arguments:
                    value = bound_args.arguments[param_name]
                    try:
                        validated_value = validator(value)
                        bound_args.arguments[param_name] = validated_value
                    except Exception as e:
                        raise ValidationError(
                            f"Invalid {param_name}: {e}",
                            error_code="invalid_input",
                            context={"field": param_name, "value": value},
                        ) from e

            return await func(*bound_args.args, **bound_args.kwargs)

        return wrapper  # type: ignore

    return decorator


# NOTE: Rate limiting has been consolidated to use the advanced RateLimitManager
# from src.services.utilities.rate_limiter.py, which provides:
# - Token bucket algorithm with burst capacity
# - Adaptive rate limiting based on API responses
# - Centralized configuration through UnifiedConfig
#
# For rate limiting in your services, use:
# from ..utilities.rate_limiter import RateLimitManager
# rate_limiter = RateLimitManager(config)
# await rate_limiter.acquire(provider="openai", tokens=1)


# Custom Pydantic errors following Pydantic 2.0 patterns
def create_validation_error(
    field: str, message: str, error_type: str = "value_error", **context
) -> PydanticCustomError:
    """Create a custom Pydantic validation error.

    Args:
        field: Field name that failed validation
        message: Error message
        error_type: Type of validation error
        **context: Additional context for the error

    Returns:
        PydanticCustomError instance
    """
    return PydanticCustomError(
        error_type,
        message,
        {"field": field, **context},
    )
