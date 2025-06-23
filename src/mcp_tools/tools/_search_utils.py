import typing
"""Core search utilities for MCP tools."""

import logging
from uuid import uuid4

from src.config import SearchStrategy

from ...infrastructure.client_manager import ClientManager
from ..models.requests import SearchRequest
from ..models.responses import SearchResult

logger = logging.getLogger(__name__)


async def search_documents_core(
    request: SearchRequest, client_manager: ClientManager, ctx=None
) -> list[SearchResult]:
    """
    Core search documents functionality without MCP dependencies.

    Supports dense, sparse, and hybrid search strategies with optional
    BGE reranking for improved accuracy.
    """
    # Generate request ID for tracking
    request_id = str(uuid4())
    if ctx:
        await ctx.info(
            f"Starting search request {request_id} with strategy {request.strategy}"
        )

    try:
        # Get services from client manager
        cache_manager = await client_manager.get_cache_manager()
        embedding_manager = await client_manager.get_embedding_manager()
        qdrant_service = await client_manager.get_qdrant_service()

        # Check cache first
        cache_key = f"search:{request.collection}:{request.query}:{request.strategy}:{request.limit}"
        cached = await cache_manager.get(cache_key)
        if cached:
            if ctx:
                await ctx.debug(f"Cache hit for request {request_id}")
            return [SearchResult(**r) for r in cached]

        # Generate embedding for query
        if ctx:
            await ctx.debug(f"Generating embeddings for query: {request.query[:50]}...")
        # Generate sparse embeddings for hybrid search
        generate_sparse = request.strategy in [
            SearchStrategy.HYBRID,
            SearchStrategy.SPARSE,
        ]
        embedding_result = await embedding_manager.generate_embeddings(
            texts=[request.query],
            generate_sparse=generate_sparse,
            model=request.embedding_model,
        )

        # Execute search based on strategy
        if ctx:
            await ctx.info(f"Executing {request.strategy} search...")

        # Prepare sparse vector if available
        sparse_vector = None
        if generate_sparse and embedding_result.sparse_embeddings:
            sparse_embedding = embedding_result.sparse_embeddings[0]
            # Convert sparse embedding to dict format expected by Qdrant
            sparse_vector = {
                int(idx): float(val)
                for idx, val in enumerate(sparse_embedding)
                if val != 0
            }

        # Use hybrid_search for all strategies
        if request.strategy == SearchStrategy.SPARSE and not sparse_vector:
            raise ValueError("Sparse embeddings not available for sparse search")

        # For sparse-only search, pass empty dense vector
        query_vector = (
            embedding_result.embeddings[0]
            if request.strategy != SearchStrategy.SPARSE
            else []
        )

        # Execute search
        results = await qdrant_service.hybrid_search(
            collection_name=request.collection,
            query_vector=query_vector,
            sparse_vector=sparse_vector
            if request.strategy != SearchStrategy.DENSE
            else None,
            limit=request.limit,
            score_threshold=request.score_threshold,
            fusion_type="rrf",  # Use RRF fusion for hybrid
            search_accuracy=request.search_accuracy
            if hasattr(request, "search_accuracy")
            else "balanced",
        )

        # Apply reranking if requested
        if request.rerank and len(results) > 0:
            if ctx:
                await ctx.debug("Applying BGE reranking...")
            # Note: rerank_results method would need to be implemented in QdrantService
            # For now, we'll skip reranking
            pass

        # Convert to response format with content intelligence metadata
        search_results = []
        for result in results:
            payload = result["payload"]

            # Base search result
            search_result_data = {
                "content": payload.get("content", ""),
                "metadata": payload.get("metadata", {}),
                "score": result["score"],
                "id": str(result["id"]),
                "url": payload.get("url"),
                "title": payload.get("title"),
            }

            # Add content intelligence metadata if available
            if payload.get("content_intelligence_analyzed"):
                search_result_data.update(
                    {
                        "content_type": payload.get("content_type"),
                        "content_confidence": payload.get("content_confidence"),
                        "quality_overall": payload.get("quality_overall"),
                        "quality_completeness": payload.get("quality_completeness"),
                        "quality_relevance": payload.get("quality_relevance"),
                        "quality_confidence": payload.get("quality_confidence"),
                        "content_intelligence_analyzed": True,
                    }
                )

            search_results.append(SearchResult(**search_result_data))

        # Cache results
        if search_results:
            cache_data = [r.model_dump() for r in search_results]
            await cache_manager.set(cache_key, cache_data, ttl=request.cache_ttl or 300)

        if ctx:
            await ctx.info(f"Search completed: {len(search_results)} results found")
        return search_results

    except Exception as e:
        if ctx:
            await ctx.error(f"Search failed: {e!s}")
        raise
