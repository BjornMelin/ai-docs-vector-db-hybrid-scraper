"""Tests for FastAPI dependency resolution via the centralized service resolver."""

from __future__ import annotations

from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock

import pytest
from fastapi import HTTPException

from src.services import service_resolver
from src.services.fastapi import dependencies as fastapi_dependencies
from src.services.fastapi.dependencies import (
    DatabaseSessionContext,
    HealthCheckManager,
)


class _InitialisableService:
    """Simple service stub that tracks initialization calls."""

    def __init__(self) -> None:
        self.initialized = False

    def initialize(self) -> None:
        self.initialized = True

    def is_initialized(self) -> bool:
        return self.initialized


@pytest.mark.asyncio()
async def test_get_vector_service_uses_container(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Vector service should be resolved from the application container."""
    service = _InitialisableService()

    class _Container:
        def vector_store_service(self) -> _InitialisableService:
            return service

    monkeypatch.setattr(service_resolver, "get_container", _Container)

    resolved = await fastapi_dependencies.get_vector_service()

    assert resolved is service
    assert service.initialized is True


@pytest.mark.asyncio()
async def test_get_vector_service_maps_failure(monkeypatch: pytest.MonkeyPatch) -> None:
    """Vector service resolution failures should surface as HTTP 503 errors."""

    class _Container:
        def vector_store_service(self) -> _InitialisableService:
            raise RuntimeError("boom")

    monkeypatch.setattr(service_resolver, "get_container", _Container)

    with pytest.raises(HTTPException) as err:
        await fastapi_dependencies.get_vector_service()

    assert err.value.status_code == 503
    assert "Vector service not available" in err.value.detail


@pytest.mark.asyncio()
async def test_get_cache_manager_maps_failure(monkeypatch: pytest.MonkeyPatch) -> None:
    """Cache manager failures should raise HTTP 503 errors."""

    async def _fail() -> None:
        raise RuntimeError("cache failure")

    monkeypatch.setattr(
        fastapi_dependencies,
        "resolve_cache_manager",
        AsyncMock(side_effect=_fail),
    )

    with pytest.raises(HTTPException) as err:
        await fastapi_dependencies.get_cache_manager()

    assert err.value.status_code == 503
    assert "Cache manager not available" in err.value.detail


@pytest.mark.asyncio()
async def test_database_session_provides_context(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Database session should expose cache, vector, and dragonfly clients."""

    class _Dragonfly:
        def __init__(self) -> None:
            self.closed = False

        async def close(self) -> None:
            self.closed = True

    dragonfly = _Dragonfly()

    class _Container:
        def dragonfly_client(self) -> _Dragonfly:
            return dragonfly

    monkeypatch.setattr(fastapi_dependencies, "get_container", _Container)
    monkeypatch.setattr(
        fastapi_dependencies,
        "resolve_cache_manager",
        AsyncMock(return_value="cache"),
    )
    monkeypatch.setattr(
        fastapi_dependencies,
        "resolve_vector_store_service",
        AsyncMock(return_value="vector"),
    )

    async with fastapi_dependencies.database_session() as context:
        assert isinstance(context, DatabaseSessionContext)
        assert context.cache_manager == "cache"
        assert context.vector_service == "vector"
        assert context.dragonfly is dragonfly

    assert dragonfly.closed is True


@pytest.mark.asyncio()
async def test_database_session_requires_container(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Missing container should surface as HTTP 503."""

    def _get_none_container() -> None:
        return None

    monkeypatch.setattr(fastapi_dependencies, "get_container", _get_none_container)

    with pytest.raises(HTTPException) as err:
        async with fastapi_dependencies.database_session():
            pass

    assert err.value.status_code == 503
    assert "Service container unavailable" in err.value.detail


@pytest.mark.asyncio()
async def test_get_rag_generator_maps_failure(monkeypatch: pytest.MonkeyPatch) -> None:
    """RAG generator errors should map to HTTP 503."""

    async def _fail() -> None:
        raise RuntimeError("rag failure")

    monkeypatch.setattr(
        fastapi_dependencies,
        "resolve_rag_generator",
        AsyncMock(side_effect=_fail),
    )

    with pytest.raises(HTTPException) as err:
        await fastapi_dependencies.get_rag_generator()

    assert err.value.status_code == 503
    assert "RAG generator not available" in err.value.detail


@pytest.mark.asyncio()
async def test_get_health_checker_returns_singleton(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """`get_health_checker` should memoise the constructed manager."""
    manager = MagicMock(spec=HealthCheckManager)
    build_mock = MagicMock(return_value=manager)
    monkeypatch.setattr(
        fastapi_dependencies, "build_health_manager", build_mock, raising=True
    )
    monkeypatch.setattr(
        fastapi_dependencies,
        "get_settings",
        MagicMock(return_value=SimpleNamespace()),
        raising=True,
    )
    monkeypatch.setattr(fastapi_dependencies, "_health_manager", None, raising=False)

    first = await fastapi_dependencies.get_health_checker()
    second = await fastapi_dependencies.get_health_checker()

    assert first is manager
    assert second is manager
    build_mock.assert_called_once()


@pytest.mark.asyncio()
async def test_get_health_checker_handles_reinitialization(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Subsequent calls should reuse cached manager even after failure."""
    manager = MagicMock(spec=HealthCheckManager)

    def _build_manager(*_args: object, **_kwargs: object) -> HealthCheckManager:
        return manager

    monkeypatch.setattr(
        fastapi_dependencies,
        "build_health_manager",
        _build_manager,
        raising=True,
    )
    monkeypatch.setattr(
        fastapi_dependencies,
        "get_settings",
        MagicMock(return_value=SimpleNamespace()),
        raising=True,
    )
    monkeypatch.setattr(fastapi_dependencies, "_health_manager", None, raising=False)

    result = await fastapi_dependencies.get_health_checker()

    assert result is manager
