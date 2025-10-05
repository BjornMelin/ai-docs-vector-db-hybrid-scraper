"""Public API for service-layer modules."""

from __future__ import annotations

from . import (
    circuit_breaker,
    core,
    deployment,
    functional,
    hyde,
    middleware,
    processing,
    task_queue,
    vector_db,
)
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
    "circuit_breaker",
    "core",
    "deployment",
    "functional",
    "hyde",
    "middleware",
    "processing",
    "task_queue",
    "vector_db",
]
