"""Tests for container bootstrap helpers."""

from __future__ import annotations

from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock

import pytest

from src.infrastructure import bootstrap


@pytest.mark.asyncio
async def test_ensure_container_reuses_existing(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """`ensure_container` should return an existing container without reinitializing."""

    existing = SimpleNamespace()
    monkeypatch.setattr(bootstrap, "get_container", lambda: existing)
    initialize_mock = AsyncMock()
    monkeypatch.setattr(bootstrap, "initialize_container", initialize_mock)

    container = await bootstrap.ensure_container(
        settings=MagicMock(), force_reload=False
    )

    assert container is existing
    initialize_mock.assert_not_awaited()


@pytest.mark.asyncio
async def test_container_session_initializes_and_shuts_down(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """`container_session` should initialize and then shut down the container."""

    created = SimpleNamespace()
    get_container_mock = MagicMock(return_value=None)
    initialize_mock = AsyncMock(return_value=created)
    shutdown_mock = AsyncMock()

    monkeypatch.setattr(bootstrap, "get_container", get_container_mock)
    monkeypatch.setattr(bootstrap, "initialize_container", initialize_mock)
    monkeypatch.setattr(bootstrap, "shutdown_container", shutdown_mock)

    async with bootstrap.container_session(
        settings="config", force_reload=True
    ) as container:
        assert container is created

    initialize_mock.assert_awaited_once_with("config")
    shutdown_mock.assert_awaited_once()


@pytest.mark.asyncio
async def test_container_session_shutdown_on_context_error(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """`container_session` should shut down even when the context body raises."""

    created = SimpleNamespace()
    monkeypatch.setattr(bootstrap, "get_container", MagicMock(return_value=None))
    initialize_mock = AsyncMock(return_value=created)
    shutdown_mock = AsyncMock()
    monkeypatch.setattr(bootstrap, "initialize_container", initialize_mock)
    monkeypatch.setattr(bootstrap, "shutdown_container", shutdown_mock)

    with pytest.raises(RuntimeError, match="boom"):
        async with bootstrap.container_session(
            settings="config", force_reload=True
        ) as container:
            assert container is created
            raise RuntimeError("boom")

    initialize_mock.assert_awaited_once_with("config")
    shutdown_mock.assert_awaited_once()
