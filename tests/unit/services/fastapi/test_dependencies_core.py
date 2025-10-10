"""Tests for FastAPI dependency helpers."""

from __future__ import annotations

from typing import Any, cast
from unittest.mock import MagicMock

import pytest
from fastapi import HTTPException

from src.services import dependencies as service_dependencies
from src.services.dependencies import (
    EmbeddingManagerDep,
    EmbeddingRequest,
    EmbeddingServiceError,
)
from src.services.fastapi import dependencies as fastapi_dependencies
from src.services.fastapi.dependencies import HealthCheckManager


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


@pytest.mark.asyncio()
async def test_generate_embeddings_dependency_success() -> None:
    """Ensure the embedding dependency returns a serialisable response."""

    manager = cast(EmbeddingManagerDep, _RecordingEmbeddingManager())
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
    assert cast(_RecordingEmbeddingManager, manager).calls[0]["texts"] == [
        "hello world"
    ]
    assert (
        cast(_RecordingEmbeddingManager, manager).calls[0]["quality_tier"].value
        == "balanced"
    )


@pytest.mark.asyncio()
async def test_generate_embeddings_dependency_failure() -> None:
    """Verify the dependency wraps underlying failures with service errors."""

    manager = cast(EmbeddingManagerDep, _RecordingEmbeddingManager(fail=True))
    request = EmbeddingRequest(texts=["oops"])

    with pytest.raises(EmbeddingServiceError):
        await service_dependencies.generate_embeddings(request, manager)


@pytest.mark.asyncio()
async def test_get_embedding_manager_maps_failure_to_503(monkeypatch: Any) -> None:
    """Ensure embedding manager failures convert to HTTP 503."""

    async def _ensure_failure(*_args: Any, **_kwargs: Any) -> None:
        raise RuntimeError("registry failure")

    monkeypatch.setattr(
        fastapi_dependencies,
        "core_get_embedding_manager",
        _ensure_failure,
        raising=True,
    )

    with pytest.raises(HTTPException) as err:
        await fastapi_dependencies.get_embedding_manager()

    assert err.value.status_code == 503
    assert "Embedding manager not available" in err.value.detail


@pytest.mark.asyncio()
async def test_get_cache_manager_maps_failure_to_503(monkeypatch: Any) -> None:
    """Ensure cache manager failures convert to HTTP 503."""

    async def _ensure_failure(*_args: Any, **_kwargs: Any) -> None:
        raise RuntimeError("registry failure")

    monkeypatch.setattr(
        fastapi_dependencies,
        "core_get_cache_manager",
        _ensure_failure,
        raising=True,
    )

    with pytest.raises(HTTPException) as err:
        await fastapi_dependencies.get_cache_manager()

    assert err.value.status_code == 503
    assert "Cache manager not available" in err.value.detail


@pytest.mark.asyncio()
async def test_get_vector_service_maps_failure_to_503(monkeypatch: Any) -> None:
    """Ensure vector service dependencies map failures to HTTP 503."""

    async def _client_manager_failure() -> Any:
        raise RuntimeError("client manager failure")

    monkeypatch.setattr(
        fastapi_dependencies,
        "core_get_vector_store_service",
        _client_manager_failure,
        raising=True,
    )

    with pytest.raises(HTTPException) as err:
        await fastapi_dependencies.get_vector_service()

    assert err.value.status_code == 503
    assert "Vector service not available" in err.value.detail


@pytest.mark.asyncio()
async def test_get_health_checker_returns_singleton(monkeypatch: Any) -> None:
    """`get_health_checker` should memoise the constructed manager."""

    manager = MagicMock(spec=HealthCheckManager)
    build_mock = MagicMock(return_value=manager)
    monkeypatch.setattr(
        fastapi_dependencies, "build_health_manager", build_mock, raising=True
    )
    monkeypatch.setattr(fastapi_dependencies, "get_settings", object(), raising=True)
    monkeypatch.setattr(fastapi_dependencies, "_health_manager", None, raising=False)

    first = await fastapi_dependencies.get_health_checker()
    second = await fastapi_dependencies.get_health_checker()

    assert first is manager
    assert second is manager
    build_mock.assert_called_once()


@pytest.mark.asyncio()
async def test_get_health_checker_handles_client_errors(monkeypatch: Any) -> None:
    """The health checker should still be constructed when client lookup fails."""

    manager = MagicMock(spec=HealthCheckManager)

    monkeypatch.setattr(
        fastapi_dependencies,
        "build_health_manager",
        lambda *_args, **_kwargs: manager,
        raising=True,
    )
    monkeypatch.setattr(fastapi_dependencies, "get_settings", object(), raising=True)
    monkeypatch.setattr(fastapi_dependencies, "_health_manager", None, raising=False)

    result = await fastapi_dependencies.get_health_checker()

    assert result is manager
