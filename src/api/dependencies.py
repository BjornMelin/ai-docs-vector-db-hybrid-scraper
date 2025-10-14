"""Shared FastAPI dependency helpers for API routers."""

from __future__ import annotations

from fastapi import Request

from src.api.app_factory import get_app_container
from src.infrastructure.container import ApplicationContainer
from src.services.service_resolver import (
    get_vector_store_service as resolve_vector_store_service,
)
from src.services.vector_db.service import VectorStoreService


def get_app_container_from_request(request: Request) -> ApplicationContainer:
    """Return the DI container attached to the FastAPI application.

    Args:
        request: Active FastAPI request containing the application instance.

    Returns:
        ApplicationContainer: The dependency injector container bound to the app.
    """
    return get_app_container(request.app)


async def get_vector_service_dependency() -> VectorStoreService:
    """Resolve the vector store service from the dependency container.

    Returns:
        VectorStoreService: Vector store service resolved from the
        application container.
    """
    return await resolve_vector_store_service()
