"""Simple mode search API router.

Simplified search endpoints optimized for solo developers.
"""

import logging
from time import perf_counter
from typing import Annotated, cast

from fastapi import APIRouter, Depends, HTTPException, Query

from src.api.dependencies import get_vector_client_manager
from src.contracts.retrieval import SearchResponse
from src.infrastructure.client_manager import ClientManager
from src.models.search import SearchRequest
from src.services.vector_db.service import VectorStoreService


logger = logging.getLogger(__name__)

router = APIRouter()

ClientManagerDependency = Annotated[ClientManager, Depends(get_vector_client_manager)]


@router.post("/search", response_model=SearchResponse)
async def search_documents(
    request: SearchRequest,
    client_manager: ClientManagerDependency,
) -> SearchResponse:
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
    request: SearchRequest, client_manager: ClientManager
) -> SearchResponse:
    """Perform search with service lookup and response conversion."""
    vector_service = await _get_vector_store_service(client_manager)
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
    client_manager: ClientManagerDependency,
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
    return await search_documents(request, client_manager)


async def _get_vector_store_service(
    client_manager: ClientManager,
) -> VectorStoreService:
    """Resolve the initialized vector store service from the factory."""

    service = await client_manager.get_vector_store_service()
    if not isinstance(service, VectorStoreService):  # pragma: no cover - safety net
        msg = "Vector DB service is not a VectorStoreService instance"
        raise TypeError(msg)
    return cast(VectorStoreService, service)
