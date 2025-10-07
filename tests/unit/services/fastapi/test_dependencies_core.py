"""Unit tests for FastAPI dependency helpers."""

from __future__ import annotations

from typing import Any

import pytest
from fastapi import HTTPException

from src.services import dependencies as service_dependencies
from src.services.dependencies import EmbeddingRequest, EmbeddingServiceError
from src.services.fastapi import dependencies as fastapi_dependencies
from src.services.fastapi.dependencies import ServiceHealthChecker


class _StubVectorService:
    """Provide a healthy vector service stub."""

    async def list_collections(self) -> list[str]:
        """Return the default collection for health checks."""

        return ["default"]


class _FailingVectorService:
    """Simulate a vector service failure."""

    async def list_collections(self) -> list[str]:
        """Raise to mimic an unavailable vector service."""

        raise RuntimeError("vector unavailable")


class _StubEmbeddingManager:
    """Provide embedding manager stubs."""

    def get_provider_info(self) -> dict[str, object]:
        """Return provider information for the embedding service."""

        return {"fastembed": {"status": "available"}}


class _RecordingEmbeddingManager:
    """Capture requests passed into the dependencies wrapper."""

    def __init__(self, *, fail: bool = False) -> None:
        self.fail = fail
        self.calls: list[dict[str, Any]] = []

    async def generate_embeddings(self, **kwargs: Any) -> dict[str, Any]:
        """Record invocation arguments and optionally simulate failure."""

        self.calls.append(kwargs)
        if self.fail:  # pragma: no cover - failure exercised in dedicated test
            raise RuntimeError("generation failed")

        return {
            "embeddings": [[0.1, 0.2]],
            "provider": "fastembed",
            "model": "test-model",
            "cost": 0.0,
            "latency_ms": 5.0,
            "tokens": 16,
            "reasoning": "test",
            "quality_tier": "balanced",
            "cache_hit": False,
        }


class _StubCacheManager:
    """Provide cache manager statistics stubs."""

    async def get_stats(self) -> dict[str, object]:
        """Return a perfect cache hit rate to indicate health."""

        return {"hit_rate": 1.0}


class _FailingCacheManager:
    """Simulate cache manager failure surfaced via HTTPException."""

    async def get_stats(self) -> dict[str, object]:
        """Raise HTTPException to mimic degraded cache behaviour."""

        raise HTTPException(status_code=503, detail="cache unavailable")


class _TimeoutCacheManager:
    """Simulate cache stats timing out."""

    async def get_stats(self) -> dict[str, object]:
        """Raise asyncio timeout to exercise timeout handling."""

        raise TimeoutError("cache stats timed out")


class _StubClientManager:
    """Return preconfigured service stubs for dependency helpers."""

    def __init__(
        self,
        *,
        vector_service: Any,
        cache_manager: Any,
        embedding_manager: Any,
    ) -> None:
        self._vector_service = vector_service
        self._cache_manager = cache_manager
        self._embedding_manager = embedding_manager

    async def get_vector_store_service(self) -> Any:
        """Return the injected vector service stub."""

        return self._vector_service

    async def get_cache_manager(self) -> Any:
        """Return the injected cache manager stub."""

        return self._cache_manager

    async def get_embedding_manager(self) -> Any:
        """Return the injected embedding manager stub."""

        return self._embedding_manager


async def _patch_client_manager(
    monkeypatch: Any,
    *,
    vector_service: Any,
    cache_manager: Any,
    embedding_manager: Any,
) -> None:
    """Patch FastAPI dependency to use a stubbed client manager."""

    async def _client_manager() -> _StubClientManager:
        return _StubClientManager(
            vector_service=vector_service,
            cache_manager=cache_manager,
            embedding_manager=embedding_manager,
        )

    monkeypatch.setattr(
        fastapi_dependencies,
        "get_client_manager",
        _client_manager,
        raising=True,
    )


@pytest.mark.asyncio
async def test_service_health_checker_reports_healthy(monkeypatch: Any) -> None:
    """Verify health checker reports a healthy status when all services respond."""

    await _patch_client_manager(
        monkeypatch,
        vector_service=_StubVectorService(),
        cache_manager=_StubCacheManager(),
        embedding_manager=_StubEmbeddingManager(),
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

    await _patch_client_manager(
        monkeypatch,
        vector_service=_FailingVectorService(),
        cache_manager=_StubCacheManager(),
        embedding_manager=_StubEmbeddingManager(),
    )

    checker = ServiceHealthChecker()
    result = await checker.check_health()

    assert result["status"] == "degraded"
    assert result["services"]["vector_db"]["status"] == "unhealthy"


@pytest.mark.asyncio
async def test_service_health_checker_degrades_on_cache_http_exception(
    monkeypatch: Any,
) -> None:
    """Verify cache HTTPExceptions degrade health instead of bubbling up."""

    await _patch_client_manager(
        monkeypatch,
        vector_service=_StubVectorService(),
        cache_manager=_FailingCacheManager(),
        embedding_manager=_StubEmbeddingManager(),
    )

    checker = ServiceHealthChecker()
    result = await checker.check_health()

    assert result["status"] == "degraded"
    assert result["services"]["cache"]["status"] == "unhealthy"
    assert result["services"]["cache"]["error_type"] == "HTTPException"


@pytest.mark.asyncio
async def test_service_health_checker_degrades_on_cache_timeout(
    monkeypatch: Any,
) -> None:
    """Ensure cache timeouts degrade health with timeout metadata."""

    await _patch_client_manager(
        monkeypatch,
        vector_service=_StubVectorService(),
        cache_manager=_TimeoutCacheManager(),
        embedding_manager=_StubEmbeddingManager(),
    )

    checker = ServiceHealthChecker()
    result = await checker.check_health()

    assert result["status"] == "degraded"
    assert result["services"]["cache"]["status"] == "unhealthy"
    assert result["services"]["cache"]["error_type"] == "TimeoutError"


@pytest.mark.asyncio
async def test_generate_embeddings_dependency_success() -> None:
    """Ensure the embedding dependency returns a serialisable response."""

    manager = _RecordingEmbeddingManager()
    request = EmbeddingRequest(
        texts=["hello world"],
        quality_tier="balanced",
        provider_name="fastembed",
        max_cost=1.0,
        speed_priority=True,
        auto_select=True,
        generate_sparse=False,
    )

    response = await service_dependencies.generate_embeddings(request, manager)

    assert response.provider == "fastembed"
    assert response.embeddings and response.embeddings[0] == [0.1, 0.2]
    assert manager.calls[0]["texts"] == ["hello world"]
    assert manager.calls[0]["quality_tier"].value == "balanced"


@pytest.mark.asyncio
async def test_generate_embeddings_dependency_failure() -> None:
    """Verify the dependency wraps underlying failures with service errors."""

    manager = _RecordingEmbeddingManager(fail=True)
    request = EmbeddingRequest(texts=["oops"])

    with pytest.raises(EmbeddingServiceError):
        await service_dependencies.generate_embeddings(request, manager)


@pytest.mark.asyncio
async def test_get_embedding_manager_maps_failure_to_503(monkeypatch: Any) -> None:
    """Ensure embedding manager failures convert to HTTP 503."""

    async def _ensure_failure(*_args: Any, **_kwargs: Any) -> None:
        raise RuntimeError("registry failure")

    monkeypatch.setattr(
        fastapi_dependencies,
        "ensure_service_registry",
        _ensure_failure,
        raising=True,
    )

    with pytest.raises(HTTPException) as err:
        await fastapi_dependencies.get_embedding_manager()

    assert err.value.status_code == 503
    assert "Embedding manager not available" in err.value.detail


@pytest.mark.asyncio
async def test_get_cache_manager_maps_failure_to_503(monkeypatch: Any) -> None:
    """Ensure cache manager failures convert to HTTP 503."""

    async def _ensure_failure(*_args: Any, **_kwargs: Any) -> None:
        raise RuntimeError("registry failure")

    monkeypatch.setattr(
        fastapi_dependencies,
        "ensure_service_registry",
        _ensure_failure,
        raising=True,
    )

    with pytest.raises(HTTPException) as err:
        await fastapi_dependencies.get_cache_manager()

    assert err.value.status_code == 503
    assert "Cache manager not available" in err.value.detail
