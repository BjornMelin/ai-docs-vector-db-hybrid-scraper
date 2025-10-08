"""Shared MCP tool helpers for vector service access and conversions."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

from src.services.vector_db.service import VectorStoreService


if TYPE_CHECKING:  # pragma: no cover - typing only
    from src.infrastructure.client_manager import ClientManager
else:  # pragma: no cover - runtime alias for tooling
    ClientManager = Any


async def ensure_vector_service(client_manager: ClientManager) -> VectorStoreService:
    """Return an initialized VectorStoreService instance."""

    service = await client_manager.get_vector_store_service()
    if not service.is_initialized():
        await service.initialize()
    return service
