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

import time
from typing import Any

from pydantic import ValidationError as PydanticValidationError
from pydantic_core import PydanticCustomError


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


# Utility Functions
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


__all__ = [
    "APIError",
    "BaseError",
    "CacheServiceError",
    "ConfigurationError",
    "CrawlServiceError",
    "EmbeddingServiceError",
    "ExternalServiceError",
    "MCPError",
    "NetworkError",
    "QdrantServiceError",
    "RateLimitError",
    "ResourceError",
    "ServiceError",
    "ToolError",
    "ValidationError",
    "create_validation_error",
    "safe_response",
]
