"""Tests for container bootstrap helpers."""

from __future__ import annotations

from types import SimpleNamespace
from unittest.mock import AsyncMock

import pytest

from src.config import Settings
from src.config.models import Environment
from src.infrastructure import bootstrap


@pytest.mark.asyncio
async def test_ensure_container_reuses_existing(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """`ensure_container` should return an existing container without reinitializing."""
    existing = SimpleNamespace()
    settings = Settings(environment=Environment.TESTING)
    monkeypatch.setattr(bootstrap, "get_container", lambda: existing)
    initialize_mock = AsyncMock()
    monkeypatch.setattr(bootstrap, "initialize_container", initialize_mock)

    container = await bootstrap.ensure_container(settings=settings, force_reload=False)

    assert container is existing
    initialize_mock.assert_not_awaited()


@pytest.mark.asyncio
async def test_container_session_initializes_and_shuts_down(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """`container_session` should initialize and then shut down the container."""
    created = SimpleNamespace()
    settings = Settings(environment=Environment.TESTING)
    lease = SimpleNamespace(container=created)
    acquire_mock = AsyncMock(return_value=lease)
    release_mock = AsyncMock()

    monkeypatch.setattr(bootstrap, "acquire_container", acquire_mock)
    monkeypatch.setattr(bootstrap, "release_container", release_mock)

    async with bootstrap.container_session(
        settings=settings, force_reload=True
    ) as container:
        assert container is created

    acquire_mock.assert_awaited_once_with(settings, force_reload=True)
    release_mock.assert_awaited_once_with(lease)


@pytest.mark.asyncio
async def test_container_session_shutdown_on_context_error(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """`container_session` should shut down even when the context body raises."""
    created = SimpleNamespace()
    settings = Settings(environment=Environment.TESTING)
    lease = SimpleNamespace(container=created)
    acquire_mock = AsyncMock(return_value=lease)
    release_mock = AsyncMock()
    monkeypatch.setattr(bootstrap, "acquire_container", acquire_mock)
    monkeypatch.setattr(bootstrap, "release_container", release_mock)

    with pytest.raises(RuntimeError, match="boom"):
        async with bootstrap.container_session(
            settings=settings, force_reload=True
        ) as container:
            assert container is created
            raise RuntimeError("boom")

    acquire_mock.assert_awaited_once_with(settings, force_reload=True)
    release_mock.assert_awaited_once_with(lease)
