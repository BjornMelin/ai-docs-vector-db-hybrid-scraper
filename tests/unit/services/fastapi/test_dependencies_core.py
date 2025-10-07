"""Unit tests for FastAPI dependency helpers."""

from __future__ import annotations

from typing import Any

import pytest

from src.services.fastapi.dependencies import core as dependencies
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


@pytest.mark.asyncio
async def test_service_health_checker_reports_healthy(monkeypatch: Any) -> None:
    """Verify health checker reports a healthy status when all services respond."""

    async def _vector_service() -> _StubVectorService:
        return _StubVectorService()

    async def _cache_manager() -> _StubCacheManager:
        return _StubCacheManager()

    async def _embedding_manager() -> _StubEmbeddingManager:
        return _StubEmbeddingManager()

    monkeypatch.setattr(
        dependencies, "get_vector_store_service", _vector_service, raising=True
    )
    monkeypatch.setattr(dependencies, "get_cache_manager", _cache_manager, raising=True)
    monkeypatch.setattr(
        dependencies, "get_embedding_manager", _embedding_manager, raising=True
    )

    checker = ServiceHealthChecker()
    result = await checker.check_health()

    assert result["status"] == "healthy"
    assert result["services"]["vector_db"]["status"] == "healthy"
    assert result["services"]["embeddings"]["status"] == "healthy"
    assert result["services"]["cache"]["status"] == "healthy"


@pytest.mark.asyncio
async def test_service_health_checker_degrades_on_vector_failure(monkeypatch: Any):
    """Verify health checker degrades when the vector service fails."""

    async def _failing_vector_service() -> _FailingVectorService:
        return _FailingVectorService()

    async def _cache_manager() -> _StubCacheManager:
        return _StubCacheManager()

    async def _embedding_manager() -> _StubEmbeddingManager:
        return _StubEmbeddingManager()

    monkeypatch.setattr(
        dependencies, "get_vector_store_service", _failing_vector_service, raising=True
    )
    monkeypatch.setattr(dependencies, "get_cache_manager", _cache_manager, raising=True)
    monkeypatch.setattr(
        dependencies, "get_embedding_manager", _embedding_manager, raising=True
    )

    checker = ServiceHealthChecker()
    result = await checker.check_health()

    assert result["status"] == "degraded"
    assert result["services"]["vector_db"]["status"] == "unhealthy"
