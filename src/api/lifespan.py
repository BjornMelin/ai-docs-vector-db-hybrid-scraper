"""Reusable FastAPI lifespan helpers."""

from __future__ import annotations

from collections.abc import AsyncIterator
from contextlib import asynccontextmanager

from fastapi import FastAPI

from src.services.registry import ensure_service_registry, shutdown_service_registry


@asynccontextmanager
async def service_registry_lifespan(app: FastAPI) -> AsyncIterator[None]:
    """Manage the application service registry lifecycle."""

    service_registry = await ensure_service_registry()
    app.state.service_registry = service_registry
    try:
        yield
    finally:
        await shutdown_service_registry()
        app.state.service_registry = None
