"""Targeted tests for the infrastructure client manager."""

from __future__ import annotations

import importlib.util
import sys
import types
from pathlib import Path
from typing import Any, cast
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from tests.unit.stub_factories import register_rag_dependency_stubs


register_rag_dependency_stubs()

crawl4ai_module = cast(Any, sys.modules["crawl4ai"])


class _StubAsyncCrawler:
    async def __aenter__(self) -> _StubAsyncCrawler:
        """Support async context management for crawler stubs."""

        return self

    async def __aexit__(self, exc_type, exc, tb) -> None:
        """Handle async context exit for crawler stubs."""

        return None

    async def start(self) -> None:
        """Simulate crawler startup."""

        return None

    async def close(self) -> None:
        """Simulate crawler shutdown."""

        return None


crawl4ai_module.AsyncWebCrawler = _StubAsyncCrawler

ROOT = Path(__file__).resolve().parents[3]
MODULE_PATH = ROOT / "src/infrastructure/client_manager.py"
_spec = importlib.util.spec_from_file_location("client_manager_under_test", MODULE_PATH)
assert _spec and _spec.loader
client_manager_module = importlib.util.module_from_spec(_spec)
sys.modules[_spec.name] = client_manager_module
_spec.loader.exec_module(client_manager_module)  # type: ignore[arg-type]

ClientManager = client_manager_module.ClientManager


@pytest.fixture()
async def client_manager(monkeypatch: pytest.MonkeyPatch) -> ClientManager:
    """Create a client manager instance with stubbed dependencies."""

    monkeypatch.setattr(
        client_manager_module,
        "get_config",
        lambda: types.SimpleNamespace(
            fastembed=types.SimpleNamespace(model="stub"),
            qdrant=types.SimpleNamespace(collection_name="documents"),
            rag=types.SimpleNamespace(
                model="stub-model",
                temperature=0.1,
                max_tokens=256,
                include_sources=True,
                max_results_for_context=5,
                include_confidence_score=True,
            ),
        ),
    )
    manager = ClientManager()
    await manager.initialize()
    yield manager
    await manager.cleanup()
    ClientManager.reset_singleton()


@pytest.mark.asyncio()
async def test_get_rag_generator_caches_instance(client_manager: ClientManager) -> None:
    """get_rag_generator should initialise once and reuse the instance."""

    vector_store = MagicMock()
    rag_generator = MagicMock()
    rag_generator.cleanup = AsyncMock()
    with (
        patch.object(
            client_manager,
            "get_vector_store_service",
            AsyncMock(return_value=vector_store),
        ) as get_vector_store,
        patch.object(
            client_manager_module,
            "initialise_rag_generator",
            AsyncMock(return_value=(rag_generator, MagicMock())),
        ) as initialise,
    ):
        first = await client_manager.get_rag_generator()
        second = await client_manager.get_rag_generator()

    assert first is rag_generator
    assert second is rag_generator
    get_vector_store.assert_awaited_once()
    initialise.assert_awaited_once_with(
        client_manager_module.get_config(), vector_store
    )


@pytest.mark.asyncio()
async def test_cleanup_resets_cached_services(client_manager: ClientManager) -> None:
    """cleanup should dispose of cached rag generator and vector store services."""

    vector_store = MagicMock()
    vector_store.cleanup = AsyncMock()
    rag_generator = MagicMock()
    rag_generator.cleanup = AsyncMock()

    client_manager._vector_store_service = vector_store
    client_manager._rag_generator = rag_generator

    await client_manager.cleanup()

    vector_store.cleanup.assert_awaited_once()
    rag_generator.cleanup.assert_awaited_once()
    assert client_manager._vector_store_service is None
    assert client_manager._rag_generator is None


@pytest.mark.asyncio()
async def test_get_rag_generator_returns_initialised_manager(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """get_rag_generator should return a cached generator when preset."""

    manager = ClientManager()
    await manager.initialize()

    rag_generator = MagicMock()
    rag_generator.cleanup = AsyncMock()
    manager._rag_generator = rag_generator

    result = await manager.get_rag_generator()

    assert result is rag_generator
    await manager.cleanup()
    ClientManager.reset_singleton()
