"""Deterministic tests for the lightweight client manager facade."""

from __future__ import annotations

from collections.abc import AsyncGenerator
from types import SimpleNamespace
from typing import Any

import pytest

from src.infrastructure.client_manager import ClientManager
from src.services.errors import APIError


@pytest.fixture()
async def client_manager(
    monkeypatch: pytest.MonkeyPatch,
) -> AsyncGenerator[ClientManager, None]:
    """Provide an initialized `ClientManager` wired to stub providers."""

    providers = {
        "openai": SimpleNamespace(client="openai"),
        "qdrant": SimpleNamespace(client="qdrant"),
        "redis": SimpleNamespace(client="redis"),
        "firecrawl": SimpleNamespace(client="firecrawl"),
        "http": SimpleNamespace(client="http"),
    }

    class _StubContainer:
        def __init__(self) -> None:
            self.modules: tuple[Any, ...] = ()

        def wire(self, modules: list[Any]) -> None:  # noqa: D401 - simple stub
            self.modules = tuple(modules)  # pragma: no cover

    def _get_container_stub() -> _StubContainer:
        return _StubContainer()

    def _initialize_providers_stub(self, *args: Any, **kwargs: Any) -> None:
        self._providers = providers.copy()

    async def _initialize_parallel_stub(self) -> None:
        self._parallel_processing_system = "parallel-system"

    monkeypatch.setattr(
        "src.infrastructure.client_manager.get_container", _get_container_stub
    )
    monkeypatch.setattr(
        ClientManager, "initialize_providers", _initialize_providers_stub
    )
    monkeypatch.setattr(
        ClientManager,
        "_initialize_parallel_processing_system",
        _initialize_parallel_stub,
    )

    ClientManager.reset_singleton()
    manager = ClientManager()
    await manager.initialize()
    yield manager
    await manager.cleanup()
    ClientManager.reset_singleton()


@pytest.mark.anyio
async def test_initialize_populates_providers(client_manager: ClientManager) -> None:
    """Initialization should populate all configured client providers."""
    assert client_manager.is_initialized is True
    assert await client_manager.get_qdrant_client() == "qdrant"
    assert await client_manager.get_openai_client() == "openai"
    assert await client_manager.get_firecrawl_client() == "firecrawl"
    assert await client_manager.get_http_client() == "http"


@pytest.mark.anyio
async def test_managed_client_yields_provider(client_manager: ClientManager) -> None:
    """The managed client context should yield the requested provider client."""
    async with client_manager.managed_client("qdrant") as client:
        assert client == "qdrant"


@pytest.mark.anyio
async def test_managed_client_unknown_type(client_manager: ClientManager) -> None:
    """Unknown client types should raise a descriptive error."""
    with pytest.raises(ValueError):
        async with client_manager.managed_client("unknown"):
            pass


@pytest.mark.anyio
async def test_missing_provider_raises_api_error(client_manager: ClientManager) -> None:
    """Accessing a missing provider surfaces an API error."""
    client_manager._providers.pop("qdrant")  # pylint: disable=protected-access
    with pytest.raises(APIError):
        await client_manager.get_qdrant_client()
