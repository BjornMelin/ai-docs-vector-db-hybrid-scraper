"""Unit tests for FastAPI dependency helpers."""

# pylint: disable=duplicate-code,import-error

from typing import Any, cast

import pytest

from src.services.fastapi.dependencies.core import ServiceHealthChecker


class _StubVectorService:
    """Provide a healthy vector service stub."""

    async def list_collections(self) -> list[str]:
        """Return the default collection for health checks."""

        return ["default"]


class _FailingVectorService:
    """Simulate a vector service failure."""

    async def list_collections(self) -> list[str]:  # pragma: no cover - negative path
        """Raise to mimic an unavailable vector service."""

        raise RuntimeError("vector unavailable")


class _StubEmbeddingManager:
    """Provide embedding manager stubs."""

    def get_provider_info(self) -> dict[str, object]:
        """Return provider information for the embedding service."""

        return {"fastembed": {"status": "available"}}


class _StubCacheManager:
    """Provide cache manager statistics stubs."""

    async def get_stats(self) -> dict[str, object]:
        """Return a perfect cache hit rate to indicate health."""

        return {"hit_rate": 1.0}


class _StubContainer:
    """Bundle the stubbed service dependencies."""

    def __init__(
        self, vector_service, embedding_manager, cache_manager, initialized=True
    ):
        """Initialize the container with dependency stubs."""

        self._vector_service = vector_service
        self._embedding_manager = embedding_manager
        self._cache_manager = cache_manager
        self.is_initialized = initialized

    @property
    def vector_service(self):
        """Return the vector service stub or raise when missing."""

        if self._vector_service is None:
            raise RuntimeError("vector not initialized")
        return self._vector_service

    @property
    def embedding_manager(self):
        """Return the embedding manager stub or raise when missing."""

        if self._embedding_manager is None:
            raise RuntimeError("embedding not initialized")
        return self._embedding_manager

    @property
    def cache_manager(self):
        """Return the cache manager stub or raise when missing."""

        if self._cache_manager is None:
            raise RuntimeError("cache not initialized")
        return self._cache_manager


@pytest.mark.asyncio
async def test_service_health_checker_reports_healthy():
    """Verify health checker reports a healthy status when all services respond."""

    container = _StubContainer(
        vector_service=_StubVectorService(),
        embedding_manager=_StubEmbeddingManager(),
        cache_manager=_StubCacheManager(),
    )

    # Cast stub to Any for DependencyContainer type expected by ServiceHealthChecker.
    checker = ServiceHealthChecker(cast(Any, container))

    result = await checker.check_health()

    assert result["status"] == "healthy"
    assert result["services"]["vector_db"]["status"] == "healthy"
    assert result["services"]["embeddings"]["status"] == "healthy"
    assert result["services"]["cache"]["status"] == "healthy"


@pytest.mark.asyncio
async def test_service_health_checker_degrades_on_vector_failure():
    """Verify health checker degrades when the vector service fails."""

    container = _StubContainer(
        vector_service=_FailingVectorService(),
        embedding_manager=_StubEmbeddingManager(),
        cache_manager=_StubCacheManager(),
    )

    # Cast stub to Any for DependencyContainer type expected by ServiceHealthChecker.
    checker = ServiceHealthChecker(cast(Any, container))

    result = await checker.check_health()

    assert result["status"] == "degraded"
    assert result["services"]["vector_db"]["status"] == "unhealthy"
