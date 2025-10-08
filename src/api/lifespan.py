"""Reusable FastAPI lifespan helpers."""

from __future__ import annotations

from collections.abc import AsyncIterator
from contextlib import asynccontextmanager

from fastapi import FastAPI

from src.infrastructure.client_manager import (
    ensure_client_manager,
    shutdown_client_manager,
)


@asynccontextmanager
async def client_manager_lifespan(app: FastAPI) -> AsyncIterator[None]:
    """Manage the global ClientManager lifecycle for FastAPI."""

    client_manager = await ensure_client_manager()
    app.state.client_manager = client_manager
    try:
        yield
    finally:
        await shutdown_client_manager()
        app.state.client_manager = None
