"""Simple mode search API router.

Simplified search endpoints optimized for solo developers.
"""

import logging
from time import perf_counter
from typing import Annotated, Any, cast

from fastapi import APIRouter, Depends, HTTPException, Query
from pydantic import BaseModel, Field

from src.api.dependencies import get_vector_client_manager
from src.infrastructure.client_manager import ClientManager
from src.services.vector_db import VectorMatch
from src.services.vector_db.service import VectorStoreService


logger = logging.getLogger(__name__)

router = APIRouter()


class SimpleSearchRequest(BaseModel):
    """Simplified search request for simple mode."""

    query: str = Field(..., min_length=1, max_length=500, description="Search query")
    limit: int = Field(default=10, ge=1, le=25, description="Maximum results to return")
    collection: str = Field(default="documents", description="Collection to search")


class SimpleSearchResponse(BaseModel):
    """Simplified search response for simple mode."""

    query: str = Field(..., description="Original search query")
    results: list[dict[str, Any]] = Field(..., description="Search results")
    total_count: int = Field(..., description="Total number of results")
    processing_time_ms: float = Field(
        ..., description="Processing time in milliseconds"
    )


ClientManagerDependency = Annotated[ClientManager, Depends(get_vector_client_manager)]


@router.post("/search", response_model=SimpleSearchResponse)
async def search_documents(
    request: SimpleSearchRequest,
    client_manager: ClientManagerDependency,
) -> SimpleSearchResponse:
    """Search documents using simple vector search.

    This endpoint provides basic vector search functionality without advanced features
    like reranking, query expansion, or hybrid search.
    """
    try:
        return await _perform_search(request, client_manager)
    except Exception as e:
        logger.exception("Search failed")
        # Return generic error message to prevent information disclosure
        raise HTTPException(
            status_code=500,
            detail="An internal error occurred while processing your search request",
        ) from e


async def _perform_search(
    request: SimpleSearchRequest, client_manager: ClientManager
) -> SimpleSearchResponse:
    """Perform search with service lookup and response conversion."""
    vector_service = await _get_vector_store_service(client_manager)
    started = perf_counter()
    matches = await vector_service.search_documents(
        request.collection,
        request.query,
        limit=request.limit,
        group_by="doc_id",
        group_size=1,
        normalize_scores=True,
    )
    processing_time_ms = (perf_counter() - started) * 1000
    results = [_vector_match_to_result(match) for match in matches]
    return SimpleSearchResponse(
        query=request.query,
        results=results,
        total_count=len(results),
        processing_time_ms=processing_time_ms,
    )


@router.get("/search", response_model=SimpleSearchResponse)
async def search_documents_get(
    client_manager: ClientManagerDependency,
    q: str = Query(..., min_length=1, max_length=500, description="Search query"),
    limit: int = Query(default=10, ge=1, le=25, description="Maximum results"),
    collection: str = Query(default="documents", description="Collection to search"),
) -> SimpleSearchResponse:
    """Search documents using GET request (simplified interface)."""
    request = SimpleSearchRequest(
        query=q,
        limit=limit,
        collection=collection,
    )
    return await search_documents(request, client_manager)


@router.get("/search/health")
async def search_health(
    client_manager: ClientManagerDependency,
) -> dict[str, Any]:
    """Get search service health status."""
    try:
        stats = await _get_search_stats(client_manager)
    except Exception as e:
        logger.exception("Search health check failed")
        return {
            "status": "unhealthy",
            "error": str(e),
        }

    return {
        "status": "healthy",
        "service_type": "simple",
        "stats": stats,
    }


async def _get_search_stats(client_manager: ClientManager) -> dict[str, Any]:
    """Get search service statistics."""
    vector_service = await _get_vector_store_service(client_manager)
    collections = await vector_service.list_collections()
    stats: dict[str, Any] = {"collections": collections}
    qdrant_config = getattr(vector_service.config, "qdrant", None)
    primary_collection = getattr(qdrant_config, "collection_name", None) or "documents"
    if primary_collection in collections:
        try:
            stats["primary_collection"] = primary_collection
            stats["primary_collection_stats"] = await vector_service.collection_stats(
                primary_collection
            )
        except Exception as exc:  # pragma: no cover - defensive logging branch
            stats["primary_collection_error"] = str(exc)
    return stats


async def _get_vector_store_service(
    client_manager: ClientManager,
) -> VectorStoreService:
    """Resolve the initialized vector store service from the factory."""

    service = await client_manager.get_vector_store_service()
    if not isinstance(service, VectorStoreService):  # pragma: no cover - safety net
        msg = "Vector DB service is not a VectorStoreService instance"
        raise TypeError(msg)
    return cast(VectorStoreService, service)


def _vector_match_to_result(match: VectorMatch) -> dict[str, Any]:
    """Normalize a vector match into the simple search response format."""

    payload = dict(match.payload or {})
    grouping = payload.get("_grouping") or {}
    collection = (
        payload.get("collection") or payload.get("_collection") or match.collection
    )
    result: dict[str, Any] = {
        "id": match.id,
        "score": match.raw_score if match.raw_score is not None else match.score,
        "raw_score": match.raw_score,
        "normalized_score": match.normalized_score,
        "collection": collection,
        "group": grouping if grouping else None,
        "payload": payload,
    }
    if match.vector is not None:
        result["vector"] = list(match.vector)
    if result["group"] is None:
        result.pop("group")
    return result
