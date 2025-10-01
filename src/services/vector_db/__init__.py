"""Vector store adapters and shared data structures."""

from .adapter import QdrantVectorAdapter
from .adapter_base import CollectionSchema, TextDocument, VectorMatch, VectorRecord


__all__ = [
    "CollectionSchema",
    "QdrantVectorAdapter",
    "TextDocument",
    "VectorMatch",
    "VectorRecord",
]
