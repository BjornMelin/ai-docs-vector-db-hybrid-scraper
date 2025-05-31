"""Core utilities package for the AI Documentation Vector DB system.

This package provides shared utilities, decorators, error handling,
and common patterns used throughout the application.
"""

# Error handling
# Constants and enums
from .constants import DEFAULT_CACHE_TTL
from .constants import DEFAULT_CHUNK_SIZE
from .constants import DEFAULT_REQUEST_TIMEOUT
from .constants import EMBEDDING_BATCH_SIZE
from .constants import MAX_RETRIES
from .constants import RATE_LIMITS

# Decorators and patterns
from .decorators import RateLimiter
from .decorators import circuit_breaker
from .decorators import handle_mcp_errors
from .decorators import retry_async
from .decorators import validate_input
from .errors import APIError
from .errors import BaseError
from .errors import CacheServiceError
from .errors import ConfigurationError
from .errors import CrawlServiceError
from .errors import EmbeddingServiceError
from .errors import ExternalServiceError
from .errors import MCPError
from .errors import NetworkError
from .errors import QdrantServiceError
from .errors import RateLimitError
from .errors import ResourceError
from .errors import ServiceError
from .errors import ToolError
from .errors import ValidationError
from .errors import create_validation_error
from .errors import safe_response

# Utility functions
from .utils import async_command
from .utils import async_to_sync_click

__all__ = [
    # Error classes
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
    # Decorators and patterns
    "RateLimiter",
    "circuit_breaker",
    "handle_mcp_errors",
    "retry_async",
    "validate_input",
    # Constants
    "DEFAULT_CACHE_TTL",
    "DEFAULT_CHUNK_SIZE",
    "DEFAULT_REQUEST_TIMEOUT",
    "EMBEDDING_BATCH_SIZE",
    "MAX_RETRIES",
    "RATE_LIMITS",
    # Utilities
    "async_command",
    "async_to_sync_click",
]
