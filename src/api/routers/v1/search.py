"""Versioned search API router backed by canonical search contracts."""

from __future__ import annotations

import logging
from collections.abc import Sequence
from time import perf_counter
from typing import Annotated, Any

from fastapi import APIRouter, Depends, HTTPException, Query

from src.api.dependencies import get_vector_service_dependency
from src.api.routers.v1.service_helpers import execute_service_call
from src.contracts.retrieval import SearchRecord, SearchResponse
from src.models.search import SearchRequest
from src.services.vector_db.service import VectorStoreService


logger = logging.getLogger(__name__)

router = APIRouter()


VectorServiceDependency = Annotated[
    VectorStoreService,
    Depends(get_vector_service_dependency),
]


@router.post("/search", response_model=SearchResponse)
async def search_documents(
    request: SearchRequest,
    vector_service: VectorServiceDependency,
) -> SearchResponse:
    """Execute a search request using the canonical search contracts.

    Args:
        request: Validated search payload containing query parameters.
        vector_service: Vector store dependency resolved from the DI container.

    Returns:
        Canonical search response composed of ``SearchRecord`` entries.
    """
    return await _perform_search(request, vector_service)


@router.get("/search", response_model=SearchResponse)
async def search_documents_get(
    vector_service: VectorServiceDependency,
    q: str = Query(..., min_length=1, description="Search query text."),
    limit: int = Query(
        default=10, ge=1, le=1000, description="Maximum number of records to return."
    ),
    collection: str = Query(
        default="documentation", description="Collection to search against."
    ),
    offset: int = Query(
        default=0, ge=0, description="Number of records to skip for pagination."
    ),
) -> SearchResponse:
    """Convenience GET endpoint that emits a ``SearchRequest`` internally.

    Args:
        vector_service: Vector store dependency resolved from the DI container.
        q: Text query submitted via query parameters.
        limit: Maximum number of results to return.
        collection: Collection identifier to search within.
        offset: Pagination offset expressed as record count.

    Returns:
        Canonical search response payload.
    """
    request = SearchRequest(query=q, limit=limit, collection=collection, offset=offset)
    return await search_documents(request, vector_service)


async def _perform_search(
    request: SearchRequest,
    vector_service: VectorStoreService,
) -> SearchResponse:
    """Dispatch search execution based on the request contents."""
    started = perf_counter()
    collection = request.collection or "documentation"

    if request.query_vector:
        records = await _search_by_vector(
            vector_service,
            collection=collection,
            vector=request.query_vector,
            limit=request.limit,
            filters=request.filters,
        )
    else:
        records = await _search_by_query(
            vector_service,
            request=request,
            collection=collection,
        )

    elapsed_ms = (perf_counter() - started) * 1000
    grouping_applied = any(record.grouping_applied for record in records)

    return SearchResponse(
        query=request.query,
        records=records,
        total_results=len(records),
        processing_time_ms=elapsed_ms,
        grouping_applied=grouping_applied,
    )


async def _search_by_query(
    vector_service: VectorStoreService,
    *,
    request: SearchRequest,
    collection: str,
) -> list[SearchRecord]:
    """Execute a dense similarity search."""
    return await execute_service_call(
        operation="search.query",
        logger=logger,
        coroutine_factory=lambda: vector_service.search_documents(
            collection=collection,
            query=request.query,
            limit=request.limit,
            filters=request.filters,
            group_by=request.group_by,
            group_size=request.group_size,
            overfetch_multiplier=request.overfetch_multiplier,
            normalize_scores=request.normalize_scores,
        ),
        error_detail="Search request failed due to an internal error.",
        extra={"collection": collection},
    )


async def _search_by_vector(
    vector_service: VectorStoreService,
    *,
    collection: str,
    vector: Sequence[float],
    limit: int,
    filters: dict[str, Any] | None,
) -> list[SearchRecord]:
    """Execute a vector-only search."""
    if not vector:
        raise HTTPException(
            status_code=400,
            detail="query_vector must contain at least one value.",
        )
    return await execute_service_call(
        operation="search.vector",
        logger=logger,
        coroutine_factory=lambda: vector_service.search_vector(
            collection=collection,
            vector=vector,
            limit=limit,
            filters=filters,
        ),
        error_detail="Vector search failed due to an internal error.",
        extra={"collection": collection},
    )
