"""Qdrant manager module - compatibility wrapper for the main QdrantService.

This module provides a compatibility interface for imports expecting qdrant_manager.
The actual implementation is in service.py with the QdrantService class.
"""

from .service import QdrantService


# Compatibility aliases
QdrantManager = QdrantService
qdrant_service = QdrantService


# Legacy compatibility
def get_qdrant_manager(*args, **kwargs):
    """Get a QdrantService instance (legacy compatibility function)."""
    return QdrantService(*args, **kwargs)


__all__ = [
    "QdrantManager",
    "QdrantService",
    "get_qdrant_manager",
    "qdrant_service",
]
