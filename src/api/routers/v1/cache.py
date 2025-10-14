"""API endpoints for cache management operations."""

from __future__ import annotations

import logging
from collections.abc import Awaitable, Callable
from typing import Any

from fastapi import APIRouter
from pydantic import BaseModel, Field

from src.services.cache.warmup import warm_caches
from src.services.circuit_breaker.decorators import CircuitOpenError
from src.services.fastapi.dependencies import CacheManagerDep, EmbeddingManagerDep
from src.services.service_resolver import (
    get_vector_store_service as resolve_vector_store_service,
)
from src.services.vector_db.service import VectorStoreService


router = APIRouter(prefix="/cache", tags=["Cache"])


logger = logging.getLogger(__name__)


class CacheWarmRequest(BaseModel):
    """Request payload for cache warmup operations.

    Attributes:
        embedding_queries: Queries used to prime embedding cache.
        search_queries: Queries used to prime vector search results.
        search_collection: Collection name used for vector store warmup.
    """

    embedding_queries: list[str] = Field(
        default_factory=list,
        description="Embedding queries to prime in the distributed cache.",
    )
    search_queries: list[str] = Field(
        default_factory=list,
        description="Popular search queries to warm, when supported.",
    )
    search_collection: str = Field(
        default="default",
        description="Collection name used when warming search results.",
    )


class CacheWarmResponse(BaseModel):
    """Response summarising cache warmup activity.

    Attributes:
        embeddings: Summary of embedding warmup results.
        search: Summary of search warmup results.
    """

    embeddings: dict[str, Any]
    search: dict[str, Any]


@router.post("/warm", response_model=CacheWarmResponse)
async def warm_cache(
    request: CacheWarmRequest,
    cache_manager: CacheManagerDep,
    embedding_manager: EmbeddingManagerDep,
) -> CacheWarmResponse:
    """Warm embedding and search caches using configured services.

    Args:
        request: Payload describing cache warmup parameters.
        cache_manager: Cache manager resolved from dependency injection.
        embedding_manager: Embedding manager used to pre-compute embeddings.

    Returns:
        CacheWarmResponse: Summary of embedding and search warmup activity.
    """

    search_executor: Callable[[str, str], Awaitable[list[dict[str, Any]]]] | None = None
    vector_service: VectorStoreService | None = None
    try:
        vector_service = await resolve_vector_store_service()
    except (CircuitOpenError, RuntimeError) as exc:
        logger.debug("Vector store unavailable for cache warmup: %s", exc)

    if vector_service is not None:

        async def _execute(query: str, collection: str) -> list[dict[str, Any]]:
            records = await vector_service.search_documents(
                collection=collection,
                query=query,
                limit=10,
            )
            return [record.model_dump() for record in records]

        search_executor = _execute

    summary = await warm_caches(
        cache_manager,
        embedding_manager=embedding_manager,
        embedding_queries=request.embedding_queries,
        search_queries=request.search_queries,
        search_collection=request.search_collection,
        search_executor=search_executor,
    )
    return CacheWarmResponse(**summary)
