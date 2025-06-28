"""Simple mode search API router.

Simplified search endpoints optimized for solo developers.
"""

import logging
from typing import Any
from fastapi import APIRouter, Depends, HTTPException, Query
from pydantic import BaseModel, Field

from src.architecture.service_factory import get_service


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
        # Get search service
        search_service = await get_service("search_service")

        # Convert to internal search request
        internal_request = SearchRequest(
            query=request.query,
            limit=request.limit,
            collection_name=request.collection_name,
        )

        # Perform search
        response = await search_service.search(internal_request)

        # Convert to simple response
        return SimpleSearchResponse(
            query=response.query,
            results=response.results,
            total_count=response.total_count,
            processing_time_ms=response.processing_time_ms,
        )

    except Exception as e:
        logger.exception(f"Search failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))


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
        search_service = await get_service("search_service")
        stats = search_service.get_search_stats()

        return {
            "status": "healthy",
            "service_type": "simple",
            "stats": stats,
        }

    except Exception as e:
        logger.exception(f"Search health check failed: {e}")
        return {
            "status": "unhealthy",
            "error": str(e),
        }
