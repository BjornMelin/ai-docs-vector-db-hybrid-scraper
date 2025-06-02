"""Core utilities package for the AI Documentation Vector DB system.

This package provides shared utilities, constants, and common patterns
used throughout the application.

Note: Error classes and decorators have been consolidated into src.services.errors
for better organization and to eliminate duplication.
"""

# Constants and enums
from .constants import DEFAULT_CACHE_TTL
from .constants import DEFAULT_CHUNK_SIZE
from .constants import DEFAULT_REQUEST_TIMEOUT
from .constants import EMBEDDING_BATCH_SIZE
from .constants import MAX_RETRIES
from .constants import RATE_LIMITS

__all__ = [
    # Constants
    "DEFAULT_CACHE_TTL",
    "DEFAULT_CHUNK_SIZE",
    "DEFAULT_REQUEST_TIMEOUT",
    "EMBEDDING_BATCH_SIZE",
    "MAX_RETRIES",
    "RATE_LIMITS",
]
