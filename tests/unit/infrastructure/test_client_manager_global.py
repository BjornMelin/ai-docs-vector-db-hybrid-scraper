"""Tests for the ClientManager global lifecycle helpers."""

from __future__ import annotations

import pytest

from src.infrastructure import client_manager as client_manager_module
from src.infrastructure.client_manager import ClientManager


@pytest.fixture(autouse=True)
async def reset_global_manager() -> None:
    """Ensure the global ClientManager state is reset around each test."""

    await client_manager_module.shutdown_client_manager()
    ClientManager.reset_singleton()
    yield
    await client_manager_module.shutdown_client_manager()
    ClientManager.reset_singleton()


@pytest.mark.asyncio
async def test_ensure_client_manager_initializes_singleton(mocker) -> None:
    """The ensure helper should initialize the manager only once."""

    fake_manager = mocker.Mock(spec=ClientManager)
    fake_manager.initialize = mocker.AsyncMock()
    fake_manager.cleanup = mocker.AsyncMock()

    mocker.patch.object(
        ClientManager,
        "from_unified_config",
        return_value=fake_manager,
    )

    first = await client_manager_module.ensure_client_manager(force=True)
    second = await client_manager_module.ensure_client_manager()

    assert first is second is fake_manager
    fake_manager.initialize.assert_awaited_once()


@pytest.mark.asyncio
async def test_shutdown_client_manager_invokes_cleanup(mocker) -> None:
    """Global shutdown should delegate to the manager cleanup."""

    fake_manager = mocker.Mock(spec=ClientManager)
    fake_manager.initialize = mocker.AsyncMock()
    fake_manager.cleanup = mocker.AsyncMock()

    mocker.patch.object(
        ClientManager,
        "from_unified_config",
        return_value=fake_manager,
    )

    await client_manager_module.ensure_client_manager(force=True)
    await client_manager_module.shutdown_client_manager()

    fake_manager.cleanup.assert_awaited_once()


@pytest.mark.asyncio
async def test_get_client_manager_requires_initialization() -> None:
    """Accessing the global manager before initialization should fail."""

    with pytest.raises(RuntimeError):
        client_manager_module.get_client_manager()
