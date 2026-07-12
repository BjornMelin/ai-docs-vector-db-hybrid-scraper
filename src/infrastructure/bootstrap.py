"""Shared helpers for initializing and shutting down the DI container."""

from __future__ import annotations

from collections.abc import AsyncIterator
from contextlib import asynccontextmanager

from src.config.loader import Settings, get_settings
from src.infrastructure.container import (
    ApplicationContainer,
    acquire_container,
    get_container,
    initialize_container,
    release_container,
    shutdown_container,
)


async def ensure_container(
    *,
    settings: Settings | None = None,
    force_reload: bool = False,
) -> ApplicationContainer:
    """Ensure a container instance is available and return it.

    Args:
        settings: Optional configuration override. Defaults to the global settings.
        force_reload: Whether to shutdown any existing container before initialization.

    Returns:
        The active :class:`ApplicationContainer` instance.
    """
    container = get_container()
    if container is not None and not force_reload:
        return container

    if container is not None and force_reload:
        await shutdown_container()

    config = settings or get_settings()
    return await initialize_container(config)


@asynccontextmanager
async def container_session(
    *,
    settings: Settings | None = None,
    force_reload: bool = False,
) -> AsyncIterator[ApplicationContainer]:
    """Context manager that yields a container and handles lifecycle cleanup.

    Args:
        settings: Optional configuration override. Defaults to the global settings.
        force_reload: Whether to shutdown any existing container before initialization.

    Yields:
        The active :class:`ApplicationContainer` instance.
    """
    lease = await acquire_container(
        settings or get_settings(),
        force_reload=force_reload,
    )
    try:
        yield lease.container
    finally:
        await release_container(lease)
