"""Shared FastAPI dependency helpers for API routers."""

from __future__ import annotations

from fastapi import HTTPException, Request
from starlette import status

from src.api.app_factory import get_app_client_manager
from src.architecture.modes import ModeConfig
from src.infrastructure.client_manager import ClientManager


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


def get_vector_client_manager(request: Request) -> ClientManager:
    """Return a ClientManager when vector services are enabled."""

    mode_config = get_mode_config_from_request(request)
    if "vector_db_service" not in mode_config.enabled_services:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Vector service is disabled in the current mode",
        )
    return get_app_client_manager(request.app)
