"""Services package for direct API/SDK integration."""

from . import core, deployment, functional, hyde, processing, vector_db
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
    "core",
    "deployment",
    "functional",
    "hyde",
    "processing",
    "vector_db",
]
