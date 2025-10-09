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
    │   ├── NetworkError: Network connectivity errors
    │   └── ExternalServiceError: General external service errors
    └── ConfigurationError: Configuration and environment errors
"""

import asyncio
import functools
import inspect
import logging
import re
import time
from collections.abc import Callable
from typing import Any, LiteralString, TypeVar

from pydantic import ValidationError as PydanticValidationError
from pydantic_core import PydanticCustomError


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
    def from_pydantic(cls, exc: PydanticValidationError) -> "ValidationError":  # type: ignore[name-defined]
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
        raw_error = kwargs.get("error", "Unknown error")
        error = str(raw_error)

        # Don't expose internal paths or sensitive info
        error = re.sub(r"([A-Za-z]:)?[/\\][^\s]+", "/****/", error)
        for pattern in (r"api[_-]?key", r"token", r"password", r"secret"):
            error = re.sub(pattern, "***", error, flags=re.IGNORECASE)

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
        """Decorator for async retry logic."""

        @functools.wraps(func)
        async def wrapper(*args, **kwargs):
            """Wrapper function for async retry logic."""

            last_exception = None

            for attempt in range(max_attempts):
                try:
                    return await func(*args, **kwargs)
                except (TimeoutError, OSError, PermissionError):
                    # Non-retryable error
                    logger.exception("Non-retryable error in %s", func.__name__)
                    raise
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

            raise last_exception or Exception("All retry attempts failed")

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
            NetworkError,
            ExternalServiceError,
            ConfigurationError,
        ) as e:
            error_type_map = {
                ValidationError: "validation",
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
