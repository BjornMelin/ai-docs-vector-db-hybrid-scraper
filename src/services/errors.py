"""Error classes for all services and MCP server.

This module provides an error hierarchy for the AI Documentation Vector DB project,
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
import inspect
import logging
import time
from collections.abc import Callable
from typing import Any, LiteralString, TypeVar


try:  # pragma: no cover - optional purgatory integration
    from purgatory.domain.model import OpenedState
except ModuleNotFoundError:  # pragma: no cover
    OpenedState = type(
        "OpenedState",
        (Exception,),
        {
            "__doc__": "Fallback OpenedState when purgatory is unavailable.",
        },
    )


from pydantic import ValidationError as PydanticValidationError
from pydantic_core import PydanticCustomError
from tenacity import (
    AsyncRetrying,
    retry_if_exception_type,
    stop_after_attempt,
    wait_exponential,
)


logger = logging.getLogger(__name__)

F = TypeVar("F", bound=Callable[..., Any])


# Base Error Classes
class BaseError(Exception):
    """Base error class with context support."""

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


class QdrantServiceError(ServiceError):
    """Qdrant vector database errors."""


class EmbeddingServiceError(ServiceError):
    """Embedding service errors."""


class CrawlServiceError(ServiceError):
    """Web crawling service errors."""


class CacheServiceError(ServiceError):
    """Cache service errors."""


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
        return cls(
            message="Multiple validation errors occurred",
            error_code="validation_error",
            context={"errors": errors},
        )


# MCP Server Errors (FastMCP pattern)
class MCPError(BaseError):
    """Base MCP server error."""


class ToolError(MCPError):
    """MCP tool execution error.

    Following FastMCP pattern where ToolError contents are sent to clients.
    """


class ResourceError(MCPError):
    """MCP resource access error.

    Following FastMCP pattern where ResourceError contents are sent to clients.
    """


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
                        logger.exception("Final attempt failed for %s", func.__name__)
                        break

                    delay = min(base_delay * (backoff_factor**attempt), max_delay)

                    logger.warning(
                        "Attempt %d/%d failed for %s: %s. Retrying in %.1fs...",
                        attempt + 1,
                        max_attempts,
                        func.__name__,
                        e,
                        delay,
                    )

                    await asyncio.sleep(delay)
                except (TimeoutError, OSError, PermissionError):
                    # Non-retryable error
                    logger.exception("Non-retryable error in %s", func.__name__)
                    raise

            raise last_exception or Exception("All retry attempts failed")

        return wrapper  # type: ignore[misc]

    return decorator


def _build_breaker_kwargs(
    failure_threshold: int,
    recovery_timeout: float,
) -> dict[str, object]:
    """Translate legacy decorator arguments into purgatory parameters."""

    return {
        "threshold": failure_threshold,
        "ttl": recovery_timeout,
    }


async def _call_with_circuit_breaker(
    service_name: str,
    breaker_kwargs: dict[str, object],
    func,
    *args,
    **kwargs,
):
    """Execute an async callable inside a purgatory circuit breaker."""

    manager = await _get_circuit_breaker_manager()
    breaker = await manager.get_breaker(service_name, **breaker_kwargs)
    try:
        async with breaker:
            return await func(*args, **kwargs)
    except OpenedState as exc:
        msg = f"Circuit breaker is open for {service_name}"
        raise ExternalServiceError(
            msg,
            context={"service": service_name, "state": "open"},
        ) from exc


def circuit_breaker(  # pylint: disable=too-many-arguments
    service_name: str | None = None,
    failure_threshold: int = 5,
    recovery_timeout: float = 60.0,
    half_open_max_calls: int = 3,  # Retained for backward compatibility
    expected_exceptions: tuple[type[Exception], ...] | None = None,
    *,
    expected_exception: type[Exception] | None = None,
    enable_adaptive_timeout: bool = True,
    enable_metrics: bool = True,
) -> Callable[[F], F]:
    """Wrap an async function with the shared purgatory circuit breaker."""

    _ = (
        half_open_max_calls,
        expected_exceptions,
        expected_exception,
        enable_adaptive_timeout,
        enable_metrics,
    )

    breaker_kwargs = _build_breaker_kwargs(failure_threshold, recovery_timeout)

    def decorator(func: F) -> F:
        breaker_name = service_name or f"{func.__module__}.{func.__name__}"

        @functools.wraps(func)
        async def wrapper(*args, **kwargs):
            async def _invoke(*inner_args, **inner_kwargs):
                return await func(*inner_args, **inner_kwargs)

            return await _call_with_circuit_breaker(
                breaker_name,
                breaker_kwargs,
                _invoke,
                *args,
                **kwargs,
            )

        async def _status() -> dict[str, Any]:
            manager = await _get_circuit_breaker_manager()
            return await manager.get_breaker_status(breaker_name)

        async def _reset() -> bool:
            manager = await _get_circuit_breaker_manager()
            return await manager.reset_breaker(breaker_name)

        wrapper.circuit_breaker_name = breaker_name  # type: ignore[attr-defined]
        wrapper.get_circuit_status = _status  # type: ignore[attr-defined]
        wrapper.reset_circuit = _reset  # type: ignore[attr-defined]
        return wrapper  # type: ignore[misc]

    return decorator


def tenacity_circuit_breaker(  # pylint: disable=too-many-arguments,too-many-positional-arguments
    service_name: str | None = None,
    max_attempts: int = 3,
    wait_multiplier: float = 1.0,
    wait_max: float = 10.0,
    failure_threshold: int = 5,
    recovery_timeout: float = 60.0,
    expected_exceptions: tuple[type[Exception], ...] = (
        ExternalServiceError,
        NetworkError,
        RateLimitError,
    ),
) -> Callable[[F], F]:
    """Combine Tenacity retries with the shared purgatory circuit breaker."""

    breaker_kwargs = _build_breaker_kwargs(failure_threshold, recovery_timeout)

    def decorator(func: F) -> F:
        breaker_name = service_name or f"{func.__module__}.{func.__name__}"

        @functools.wraps(func)
        async def wrapper(*args, **kwargs):
            async def _invoke(*inner_args, **inner_kwargs):
                return await func(*inner_args, **inner_kwargs)

            async for attempt in AsyncRetrying(
                stop=stop_after_attempt(max_attempts),
                wait=wait_exponential(multiplier=wait_multiplier, max=wait_max),
                retry=retry_if_exception_type(expected_exceptions),
                reraise=True,
            ):
                with attempt:
                    return await _call_with_circuit_breaker(
                        breaker_name,
                        breaker_kwargs,
                        _invoke,
                        *args,
                        **kwargs,
                    )

        async def _status() -> dict[str, Any]:
            manager = await _get_circuit_breaker_manager()
            return await manager.get_breaker_status(breaker_name)

        async def _reset() -> bool:
            manager = await _get_circuit_breaker_manager()
            return await manager.reset_breaker(breaker_name)

        wrapper.circuit_breaker_name = breaker_name  # type: ignore[attr-defined]
        wrapper.get_circuit_status = _status  # type: ignore[attr-defined]
        wrapper.reset_circuit = _reset  # type: ignore[attr-defined]
        return wrapper  # type: ignore[misc]

    return decorator


def handle_mcp_errors(func: Callable[..., Any]) -> Callable[..., Any]:
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
            logger.warning("MCP error in %s: %s", func.__name__, e)
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
            logger.warning(
                "%s error in %s: %s", error_type.capitalize(), func.__name__, e
            )
            return safe_response(False, error=str(e), error_type=error_type)
        except (ConnectionError, OSError, PermissionError):
            # Mask internal errors for security
            logger.exception("Unexpected error in %s", func.__name__)
            return safe_response(
                False, error="Internal server error", error_type="internal"
            )
        except Exception:
            logger.exception("Unexpected error in %s", func.__name__)
            return safe_response(
                False, error="Internal server error", error_type="internal"
            )

    return wrapper  # type: ignore[misc]


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
                    except Exception as e:  # pragma: no cover - validator bug
                        msg = f"Invalid {param_name}: {e}"
                        raise ValidationError(
                            msg,
                            error_code="invalid_input",
                            context={"field": param_name, "value": value},
                        ) from e

            return await func(*bound_args.args, **bound_args.kwargs)

        return wrapper  # type: ignore[misc]

    return decorator


# Custom Pydantic errors following Pydantic 2.0 patterns
def create_validation_error(
    field: str,
    message: LiteralString,
    error_type: LiteralString = "value_error",
    **context: Any,
) -> PydanticCustomError:
    """Create a custom Pydantic validation error.

    Args:
        field: Field name that failed validation
        message: Error message (must be a literal string)
        error_type: Type of validation error (must be a literal string)
        **context: Additional context for the error

    Returns:
        PydanticCustomError instance
    """
    return PydanticCustomError(
        error_type,
        message,
        {"field": field, **context},
    )


async def _get_circuit_breaker_manager():
    """Resolve the circuit breaker manager lazily to avoid import cycles."""

    from src.services.circuit_breaker.provider import (  # pylint: disable=import-outside-toplevel
        get_circuit_breaker_manager,
    )

    return await get_circuit_breaker_manager()
