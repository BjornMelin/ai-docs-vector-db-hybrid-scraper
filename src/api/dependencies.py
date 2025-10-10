"""Shared FastAPI dependency helpers for API routers."""

from __future__ import annotations

from fastapi import Request

from src.api.app_factory import get_app_container
from src.infrastructure.container import ApplicationContainer
from src.services.dependencies import (
    get_vector_store_service as core_get_vector_store_service,
)
from src.services.vector_db.service import VectorStoreService


def get_app_container_from_request(request: Request) -> ApplicationContainer:
    """Return the DI container attached to the FastAPI application."""

    container = get_app_container(request.app)
    return container


async def get_vector_service_dependency() -> VectorStoreService:
    """Resolve the vector store service from the dependency container."""

    return await core_get_vector_store_service()
