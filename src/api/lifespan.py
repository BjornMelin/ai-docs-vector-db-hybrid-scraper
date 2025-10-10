"""Reusable FastAPI lifespan helpers."""

from __future__ import annotations

from collections.abc import AsyncIterator
from contextlib import asynccontextmanager

from fastapi import FastAPI

from src.config.loader import get_settings
from src.infrastructure.container import initialize_container, shutdown_container


@asynccontextmanager
async def client_manager_lifespan(app: FastAPI) -> AsyncIterator[None]:
    """Manage the DI container lifecycle for FastAPI."""

    container = await initialize_container(get_settings())
    app.state.container = container
    try:
        yield
    finally:
        await shutdown_container()
        app.state.container = None
