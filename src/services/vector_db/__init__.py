import typing

"""Vector database services with modular Qdrant implementation.

This module provides a clean, modular architecture for Qdrant operations:
- QdrantService: Unified facade over all functionality (uses ClientManager for connections)
- QdrantCollections: Collection CRUD and optimization
- QdrantSearch: Advanced search operations (hybrid, multi-stage, HyDE)
- QdrantIndexing: Payload indexing and optimization
- QdrantDocuments: Document/point CRUD operations
"""

from .collections import QdrantCollections
from .documents import QdrantDocuments
from .indexing import QdrantIndexing
from .search import QdrantSearch
from .service import QdrantService

__all__ = [
    "QdrantCollections",
    "QdrantDocuments",
    "QdrantIndexing",
    "QdrantSearch",
    "QdrantService",
]
