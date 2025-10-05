"""Shared MCP tool helpers for vector service access and conversions."""

from __future__ import annotations

from typing import Any

from src.infrastructure.client_manager import ClientManager
from src.mcp_tools.models.responses import SearchResult
from src.services.vector_db.service import VectorStoreService
from src.services.vector_db.types import VectorMatch


async def ensure_vector_service(client_manager: ClientManager) -> VectorStoreService:
    """Return an initialized VectorStoreService instance."""

    service = await client_manager.get_vector_store_service()
    if not service.is_initialized():
        await service.initialize()
    return service


def match_to_result(match: VectorMatch, *, include_metadata: bool) -> SearchResult:
    """Convert a vector match into the MCP ``SearchResult`` schema."""

    payload: dict[str, Any] = dict(match.payload or {})
    metadata: dict[str, Any] | None = payload if include_metadata else None
    url_value = payload.get("url")
    title_value = payload.get("title")
    return SearchResult(
        id=match.id,
        content=str(payload.get("content") or payload.get("text") or ""),
        score=float(match.score),
        url=str(url_value) if isinstance(url_value, str) else None,
        title=str(title_value) if isinstance(title_value, str) else None,
        metadata=metadata,
    )
