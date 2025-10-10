"""Reusable FastAPI lifespan helpers."""

from __future__ import annotations

from collections.abc import AsyncIterator
from contextlib import asynccontextmanager

from fastapi import FastAPI

from src.config.loader import get_settings
from src.infrastructure.bootstrap import container_session


@asynccontextmanager
async def container_lifespan(app: FastAPI) -> AsyncIterator[None]:
    """Manage the dependency-injector container lifecycle for FastAPI."""

    async with container_session(
        settings=get_settings(), force_reload=True
    ) as container:
        app.state.container = container
        try:
            yield
        finally:
            app.state.container = None
