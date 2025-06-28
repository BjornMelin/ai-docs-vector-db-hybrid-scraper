"""Services package for direct API/SDK integration."""

from . import deployment, processing, vector_db
from .base import BaseService
from .errors import (
    APIError,
    CrawlServiceError,
    EmbeddingServiceError,
    QdrantServiceError,
)


__all__ = [
    "APIError",
    "BaseService",
    "CrawlServiceError",
    "EmbeddingServiceError",
    "QdrantServiceError",
    "deployment",
    "processing",
    "vector_db",
]
