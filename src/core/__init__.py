"""Core utilities package for the AI Documentation Vector DB system.

This package provides shared utilities, constants, and common patterns
used throughout the application.

Note: Error classes and decorators have been consolidated into src.services.errors
for better organization and to eliminate duplication.
"""

# Basic application constants (non-configurable)
from .constants import (
    DEFAULT_CACHE_TTL,
    DEFAULT_CHUNK_SIZE,
    DEFAULT_REQUEST_TIMEOUT,
    EMBEDDING_BATCH_SIZE,
    MAX_RETRIES,
)


__all__ = [
    # Basic application constants
    "DEFAULT_CACHE_TTL",
    "DEFAULT_CHUNK_SIZE",
    "DEFAULT_REQUEST_TIMEOUT",
    "EMBEDDING_BATCH_SIZE",
    "MAX_RETRIES",
]
