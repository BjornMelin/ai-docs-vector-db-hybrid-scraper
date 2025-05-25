"""Search and retrieval tools for MCP server."""

import logging
from uuid import uuid4

from fastmcp import Context

from ...config.enums import SearchStrategy
from ..models.requests import SearchRequest
from ..models.responses import SearchResult
from ..service_manager import UnifiedServiceManager

logger = logging.getLogger(__name__)


def register_tools(mcp, service_manager: UnifiedServiceManager):
    """Register search tools with the MCP server."""

    @mcp.tool()
    async def search_documents(
        request: SearchRequest, ctx: Context
    ) -> list[SearchResult]:
        """
        Search documents with advanced hybrid search and reranking.

        Supports dense, sparse, and hybrid search strategies with optional
        BGE reranking for improved accuracy.
        """
        await service_manager.initialize()

        # Generate request ID for tracking
        request_id = str(uuid4())
        await ctx.info(
            f"Starting search request {request_id} with strategy {request.strategy}"
        )

        try:
            # Check cache first
            cache_key = f"search:{request.collection}:{request.query}:{request.strategy}:{request.limit}"
            cached = await service_manager.cache_manager.get(cache_key)
            if cached:
                await ctx.debug(f"Cache hit for request {request_id}")
                return [SearchResult(**r) for r in cached]

            # Generate embedding for query
            await ctx.debug(f"Generating embeddings for query: {request.query[:50]}...")
            # Generate sparse embeddings for hybrid search
            generate_sparse = request.strategy == SearchStrategy.HYBRID
            embedding_result = (
                await service_manager.embedding_manager.generate_embeddings(
                    [request.query], generate_sparse=generate_sparse
                )
            )

            query_vector = embedding_result["embeddings"][0]
            sparse_vector = None
            if embedding_result.get("sparse_embeddings"):
                sparse_vector = embedding_result["sparse_embeddings"][0]
                await ctx.debug(
                    f"Generated sparse vector with {len(sparse_vector['indices'])} non-zero elements"
                )

            await ctx.debug(f"Generated embedding with dimension {len(query_vector)}")

            # Perform search based on strategy
            if request.strategy == SearchStrategy.HYBRID:
                # For hybrid search, use both dense and sparse vectors
                results = await service_manager.qdrant_service.hybrid_search(
                    collection_name=request.collection,
                    query_vector=query_vector,
                    sparse_vector=sparse_vector,  # Now using actual sparse vector
                    limit=request.limit * 3
                    if request.enable_reranking
                    else request.limit,
                    score_threshold=0.0,
                    fusion_type="rrf",
                )
            else:
                # For dense search, use direct vector search
                results = await service_manager.qdrant_service.hybrid_search(
                    collection_name=request.collection,
                    query_vector=query_vector,
                    sparse_vector=None,  # Dense search only
                    limit=request.limit * 3
                    if request.enable_reranking
                    else request.limit,
                    score_threshold=0.0,
                )

            # Convert to search results
            search_results = []
            for point in results:
                result = SearchResult(
                    id=str(point.id),
                    content=point.payload.get("content", ""),
                    score=point.score,
                    url=point.payload.get("url"),
                    title=point.payload.get("title"),
                    metadata=point.payload if request.include_metadata else None,
                )
                search_results.append(result)

            # Apply reranking if enabled
            if request.enable_reranking and search_results:
                await ctx.debug(
                    f"Applying BGE reranking to {len(search_results)} results"
                )
                # Convert to format expected by reranker
                results_for_reranking = [
                    {"content": r.content, "original": r} for r in search_results
                ]

                # Rerank results
                reranked = await service_manager.embedding_manager.rerank_results(
                    query=request.query, results=results_for_reranking
                )

                # Extract reranked SearchResult objects
                search_results = [r["original"] for r in reranked[: request.limit]]
                await ctx.debug(
                    f"Reranking complete, returning top {len(search_results)} results"
                )
            else:
                # Without reranking, just limit to requested number
                search_results = search_results[: request.limit]

            # Cache results
            cache_data = [r.model_dump() for r in search_results]
            await service_manager.cache_manager.set(cache_key, cache_data, ttl=3600)

            await ctx.info(
                f"Search request {request_id} completed: {len(search_results)} results found"
            )

            return search_results

        except Exception as e:
            await ctx.error(f"Search request {request_id} failed: {e}")
            logger.error(f"Search failed: {e}")
            raise

    @mcp.tool()
    async def search_similar(
        content: str,
        collection: str = "documentation",
        limit: int = 10,
        threshold: float = 0.7,
        ctx: Context = None,
    ) -> list[SearchResult]:
        """
        Find similar documents using vector similarity search.

        Uses pure vector similarity without keyword matching, ideal for
        finding conceptually similar content.
        """
        await service_manager.initialize()

        request = SearchRequest(
            query=content,
            collection=collection,
            limit=limit,
            strategy=SearchStrategy.DENSE,
            enable_reranking=False,
        )

        # Use the main search function with dense strategy
        results = await search_documents(request, ctx or Context())

        # Filter by similarity threshold
        filtered = [r for r in results if r.score >= threshold]

        return filtered
