"""Shared FastAPI dependency helpers for API routers."""

from __future__ import annotations

from fastapi import HTTPException, Request
from starlette import status

from src.api.app_factory import get_app_container
from src.architecture.modes import ModeConfig
from src.infrastructure.container import ApplicationContainer
from src.services.dependencies import (
    get_vector_store_service as core_get_vector_store_service,
)
from src.services.vector_db.service import VectorStoreService


def get_mode_config_from_request(request: Request) -> ModeConfig:
    """Return the active mode configuration from the FastAPI application."""

    mode_config = getattr(request.app.state, "mode_config", None)
    if mode_config is None:
        msg = "Mode configuration is not attached to application state"
        raise RuntimeError(msg)
    if not isinstance(mode_config, ModeConfig):
        msg = "Application state mode_config is not a ModeConfig instance"
        raise TypeError(msg)
    return mode_config


def get_app_container_from_request(request: Request) -> ApplicationContainer:
    """Return the DI container attached to the FastAPI application."""

    container = get_app_container(request.app)
    return container


async def get_vector_service_dependency(request: Request) -> VectorStoreService:
    """Resolve vector store service, ensuring it is enabled for the mode."""

    mode_config = get_mode_config_from_request(request)
    if "vector_db_service" not in mode_config.enabled_services:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Vector service is disabled in the current mode",
        )
    return await core_get_vector_store_service()
