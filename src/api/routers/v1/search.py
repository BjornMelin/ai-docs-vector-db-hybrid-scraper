"""Versioned search API router backed by canonical search contracts."""

from __future__ import annotations

import logging
from collections.abc import Sequence
from time import perf_counter
from typing import Annotated, Any

from fastapi import APIRouter, Depends, HTTPException, Query

from src.api.dependencies import get_vector_service_dependency
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
    """Execute a search request using the canonical SearchRequest contract."""

    try:
        return await _perform_search(request, vector_service)
    except HTTPException:
        raise
    except Exception as exc:  # pragma: no cover - defensive logging
        logger.exception("Search execution failed")
        raise HTTPException(
            status_code=500,
            detail="Search request failed due to an internal error.",
        ) from exc


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
    """Convenience GET endpoint that maps query parameters to SearchRequest."""

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
            collection=collection,
            query=request.query,
            limit=request.limit,
            filters=request.filters,
            group_by=request.group_by,
            group_size=request.group_size,
            overfetch_multiplier=request.overfetch_multiplier,
            normalize_scores=request.normalize_scores,
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
    collection: str,
    query: str,
    limit: int,
    filters: dict[str, Any] | None,
    group_by: str | None,
    group_size: int | None,
    overfetch_multiplier: float | None,
    normalize_scores: bool | None,
) -> list[SearchRecord]:
    """Execute a dense similarity search."""

    if not query:
        raise HTTPException(
            status_code=400,
            detail="Search query must not be empty when query_vector is omitted.",
        )
    return await vector_service.search_documents(
        collection=collection,
        query=query,
        limit=limit,
        filters=filters,
        group_by=group_by,
        group_size=group_size,
        overfetch_multiplier=overfetch_multiplier,
        normalize_scores=normalize_scores,
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
    return await vector_service.search_vector(
        collection=collection,
        vector=vector,
        limit=limit,
        filters=filters,
    )
