"""Simple mode search API router.

Simplified search endpoints optimized for solo developers.
"""

import logging
from time import perf_counter
from typing import Any, cast

from fastapi import APIRouter, HTTPException, Query
from pydantic import BaseModel, Field

from src.architecture.service_factory import get_service
from src.services.vector_db import VectorMatch, VectorStoreService


logger = logging.getLogger(__name__)

router = APIRouter()


class SimpleSearchRequest(BaseModel):
    """Simplified search request for simple mode."""

    query: str = Field(..., min_length=1, max_length=500, description="Search query")
    limit: int = Field(default=10, ge=1, le=25, description="Maximum results to return")
    collection_name: str = Field(
        default="documents", description="Collection to search"
    )


class SimpleSearchResponse(BaseModel):
    """Simplified search response for simple mode."""

    query: str = Field(..., description="Original search query")
    results: list[dict[str, Any]] = Field(..., description="Search results")
    total_count: int = Field(..., description="Total number of results")
    processing_time_ms: float = Field(
        ..., description="Processing time in milliseconds"
    )


@router.post("/search", response_model=SimpleSearchResponse)
async def search_documents(request: SimpleSearchRequest) -> SimpleSearchResponse:
    """Search documents using simple vector search.

    This endpoint provides basic vector search functionality without advanced features
    like reranking, query expansion, or hybrid search.
    """
    try:
        return await _perform_search(request)
    except Exception as e:
        logger.exception("Search failed")
        # Return generic error message to prevent information disclosure
        raise HTTPException(
            status_code=500,
            detail="An internal error occurred while processing your search request",
        ) from e


async def _perform_search(request: SimpleSearchRequest) -> SimpleSearchResponse:
    """Perform search with service lookup and response conversion."""
    vector_service = await _get_vector_store_service()
    started = perf_counter()
    matches = await vector_service.search_documents(
        request.collection_name,
        request.query,
        limit=request.limit,
    )
    processing_time_ms = (perf_counter() - started) * 1000
    results = [_vector_match_to_result(match) for match in matches]
    return SimpleSearchResponse(
        query=request.query,
        results=results,
        total_count=len(results),
        processing_time_ms=processing_time_ms,
    )


@router.get("/search")
async def search_documents_get(
    q: str = Query(..., min_length=1, max_length=500, description="Search query"),
    limit: int = Query(default=10, ge=1, le=25, description="Maximum results"),
    collection: str = Query(default="documents", description="Collection to search"),
) -> SimpleSearchResponse:
    """Search documents using GET request (simplified interface)."""
    request = SimpleSearchRequest(
        query=q,
        limit=limit,
        collection_name=collection,
    )
    return await search_documents(request)


@router.get("/search/health")
async def search_health() -> dict[str, Any]:
    """Get search service health status."""
    try:
        stats = await _get_search_stats()
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


async def _get_search_stats() -> dict[str, Any]:
    """Get search service statistics."""
    vector_service = await _get_vector_store_service()
    collections = await vector_service.list_collections()
    stats: dict[str, Any] = {"collections": collections}
    default_collection = "documents"
    if default_collection in collections:
        try:
            stats["default_collection"] = default_collection
            stats["default_collection_stats"] = (
                await vector_service.collection_stats(default_collection)
            )
        except Exception as exc:  # pragma: no cover - defensive logging branch
            stats["default_collection_error"] = str(exc)
    return stats


async def _get_vector_store_service() -> VectorStoreService:
    """Resolve the initialized vector store service from the factory."""

    service = await get_service("vector_db_service")
    if not isinstance(service, VectorStoreService):  # pragma: no cover - safety net
        msg = "Vector DB service is not a VectorStoreService instance"
        raise TypeError(msg)
    return cast(VectorStoreService, service)


def _vector_match_to_result(match: VectorMatch) -> dict[str, Any]:
    """Normalize a vector match into the simple search response format."""

    payload = dict(match.payload or {})
    result: dict[str, Any] = {
        "id": match.id,
        "score": match.score,
        "payload": payload,
    }
    if match.vector is not None:
        result["vector"] = list(match.vector)
    return result
