"""Simple mode search API router.

Simplified search endpoints optimized for solo developers.
"""

import logging
from time import perf_counter
from typing import Annotated

from fastapi import APIRouter, Depends, HTTPException, Query

from src.api.dependencies import get_vector_service_dependency
from src.contracts.retrieval import SearchRecord
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
    results: list[SearchRecord] = Field(..., description="Search results")
    total_count: int = Field(..., description="Total number of results")
    processing_time_ms: float = Field(
        ..., description="Processing time in milliseconds"
    )


VectorServiceDependency = Annotated[
    VectorStoreService, Depends(get_vector_service_dependency)
]


@router.post("/search", response_model=SearchResponse)
async def search_documents(
    request: SimpleSearchRequest,
    vector_service: VectorServiceDependency,
) -> SimpleSearchResponse:
    """Search documents using simple vector search.

    This endpoint provides basic vector search functionality without advanced features
    like reranking, query expansion, or hybrid search.
    """
    try:
        return await _perform_search(request, vector_service)
    except Exception as e:
        logger.exception("Search failed")
        # Return generic error message to prevent information disclosure
        raise HTTPException(
            status_code=500,
            detail="An internal error occurred while processing your search request",
        ) from e


async def _perform_search(
    request: SimpleSearchRequest, vector_service: VectorStoreService
) -> SimpleSearchResponse:
    """Perform search with service lookup and response conversion."""
    started = perf_counter()
    records = await vector_service.search_documents(
        request.collection or "documentation",
        request.query,
        limit=request.limit,
        filters=request.filters,
        group_by=request.group_by,
        group_size=request.group_size,
        overfetch_multiplier=request.overfetch_multiplier,
        normalize_scores=request.normalize_scores,
    )
    processing_time_ms = (perf_counter() - started) * 1000
    return SearchResponse(
        query=request.query,
        records=records,
        total_results=len(records),
        processing_time_ms=processing_time_ms,
    )


@router.get("/search", response_model=SearchResponse)
async def search_documents_get(
    vector_service: VectorServiceDependency,
    q: str = Query(..., min_length=1, max_length=500, description="Search query"),
    limit: int = Query(default=10, ge=1, le=25, description="Maximum results"),
    collection: str = Query(default="documents", description="Collection to search"),
) -> SearchResponse:
    """Search documents using GET request (simplified interface)."""
    request = SearchRequest.from_input(
        q,
        limit=limit,
        collection=collection,
    )
    return await search_documents(request, vector_service)
