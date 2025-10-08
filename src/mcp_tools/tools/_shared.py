"""Shared MCP tool helpers for vector service access and conversions."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

from src.contracts.retrieval import SearchRecord
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


def search_record_to_dict(
    record: SearchRecord,
    *,
    include_metadata: bool = True,
) -> dict[str, Any]:
    """Serialize a :class:`SearchRecord` into a plain dictionary."""

    payload: dict[str, Any] = {
        "id": record.id,
        "score": record.score,
    }

    if record.content:
        payload["content"] = record.content
    if record.title is not None:
        payload["title"] = record.title
    if record.url is not None:
        payload["url"] = record.url
    if record.collection is not None:
        payload["collection"] = record.collection
    if record.normalized_score is not None:
        payload["normalized_score"] = record.normalized_score
    if record.raw_score is not None:
        payload["raw_score"] = record.raw_score
    if record.group_id is not None:
        payload["group_id"] = record.group_id
    if record.group_rank is not None:
        payload["group_rank"] = record.group_rank
    if record.grouping_applied is not None:
        payload["grouping_applied"] = record.grouping_applied
    if record.content_type is not None:
        payload["content_type"] = record.content_type

    if include_metadata:
        payload["metadata"] = dict(record.metadata or {})

    return payload
