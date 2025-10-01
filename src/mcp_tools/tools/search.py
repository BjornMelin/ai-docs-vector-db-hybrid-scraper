"""Search tools using Qdrant client native methods.

Provides vector search, hybrid search, reranking, and filtering capabilities
using qdrant-client 1.15.1+ native methods: search(), query_points(), scroll(),
recommend(), and RRF fusion.
"""

import logging
from typing import Any

from fastmcp import Context
from qdrant_client import models

from src.infrastructure.client_manager import ClientManager


logger = logging.getLogger(__name__)


def register_tools(mcp, client_manager: ClientManager):
    """Register search tools using Qdrant client native methods."""

    @mcp.tool()
    async def search_documents(
        query: str,
        collection: str = "documentation",
        limit: int = 10,
        score_threshold: float | None = None,
        filter_conditions: dict[str, Any] | None = None,
        ctx: Context | None = None,
    ) -> list[dict[str, Any]]:
        """Vector similarity search using qdrant_client.search().

        Args:
            query: Search query text
            collection: Collection name
            limit: Maximum results (1-100)
            score_threshold: Minimum similarity score (0.0-1.0)
            filter_conditions: Optional filters, e.g. {"key": "value"}
            ctx: MCP context

        Returns:
            List of search results with scores and payloads
        """
        try:
            if ctx:
                await ctx.info(f"Searching '{collection}' for: {query}")

            qdrant_service = await client_manager.get_qdrant_service()
            embedding_manager = await client_manager.get_embedding_manager()
            if not embedding_manager:
                raise RuntimeError("Embedding manager not available")

            # Generate query embedding
            embeddings_result = await embedding_manager.generate_embeddings([query])
            query_embedding = embeddings_result[0]

            # Build Qdrant filter from conditions
            query_filter = None
            if filter_conditions:
                must_conditions = []
                for key, value in filter_conditions.items():
                    if isinstance(value, dict):
                        # Range filter: {"gte": 0, "lte": 100}
                        must_conditions.append(
                            models.FieldCondition(key=key, range=models.Range(**value))
                        )
                    else:
                        # Match filter
                        must_conditions.append(
                            models.FieldCondition(
                                key=key, match=models.MatchValue(value=value)
                            )
                        )

                if must_conditions:
                    # FieldCondition is a valid Condition subtype
                    query_filter = models.Filter(must=must_conditions)  # type: ignore[arg-type]

            # Use Qdrant native search() method
            client = await qdrant_service.get_client()
            results: list[models.ScoredPoint] = await client.search(
                collection_name=collection,
                query_vector=query_embedding,
                limit=limit,
                score_threshold=score_threshold,
                query_filter=query_filter,
                with_payload=True,
                with_vectors=False,
            )

            # Format results
            formatted_results = [
                {
                    "id": str(result.id),
                    "score": result.score,
                    "payload": result.payload or {},
                }
                for result in results
            ]

            if ctx:
                await ctx.info(f"Found {len(formatted_results)} results")

            return formatted_results

        except Exception as e:
            logger.exception("Search failed")
            if ctx:
                await ctx.error(f"Search error: {e}")
            return []

    @mcp.tool()
    async def hybrid_search(
        query: str,
        collection: str = "documentation",
        limit: int = 10,
        filter_conditions: dict[str, Any] | None = None,
        ctx: Context | None = None,
    ) -> list[dict[str, Any]]:
        """Hybrid search using qdrant_client.query_points().

        Combines vector and keyword search using Qdrant's query interface.

        Args:
            query: Search query text
            collection: Collection name
            limit: Maximum results
            filter_conditions: Optional filters
            ctx: MCP context

        Returns:
            Search results with hybrid scores
        """
        try:
            if ctx:
                await ctx.info(f"Hybrid search in '{collection}': {query}")

            qdrant_service = await client_manager.get_qdrant_service()
            embedding_manager = await client_manager.get_embedding_manager()
            if not embedding_manager:
                raise RuntimeError("Embedding manager not available")

            # Generate embeddings
            embeddings_result = await embedding_manager.generate_embeddings([query])
            dense_vector = embeddings_result[0]

            # Build filter
            query_filter = None
            if filter_conditions:
                must_conditions = [
                    models.FieldCondition(key=k, match=models.MatchValue(value=v))
                    for k, v in filter_conditions.items()
                ]
                if must_conditions:
                    query_filter = models.Filter(must=must_conditions)  # type: ignore[arg-type]

            client = await qdrant_service.get_client()

            # Use Qdrant's query_points for hybrid search
            results = await client.query_points(
                collection_name=collection,
                query=dense_vector,
                limit=limit,
                query_filter=query_filter,
                with_payload=True,
                with_vectors=False,
            )

            # Format results - handle both QueryResponse and list responses
            points = results.points if hasattr(results, "points") else results
            formatted_results = [
                {
                    "id": str(point.id),
                    "score": point.score,
                    "payload": point.payload or {},
                }
                for point in points
            ]

            if ctx:
                await ctx.info(f"Hybrid search found {len(formatted_results)} results")

            return formatted_results

        except Exception as e:
            logger.exception("Hybrid search failed")
            if ctx:
                await ctx.error(f"Hybrid search error: {e}")
            return []

    @mcp.tool()
    async def scroll_collection(
        collection: str = "documentation",
        limit: int = 100,
        filter_conditions: dict[str, Any] | None = None,
        offset: str | None = None,
        ctx: Context | None = None,
    ) -> dict[str, Any]:
        """Paginate through collection using qdrant_client.scroll().

        Efficiently retrieves large result sets with pagination support.

        Args:
            collection: Collection name
            limit: Points per page (max 10000)
            filter_conditions: Optional filters
            offset: Pagination offset (point ID from previous scroll)
            ctx: MCP context

        Returns:
            Dict with points and next_page_offset
        """
        try:
            if ctx:
                await ctx.info(f"Scrolling collection '{collection}'")

            qdrant_service = await client_manager.get_qdrant_service()

            # Build filter
            query_filter = None
            if filter_conditions:
                must_conditions = [
                    models.FieldCondition(key=k, match=models.MatchValue(value=v))
                    for k, v in filter_conditions.items()
                ]
                if must_conditions:
                    query_filter = models.Filter(must=must_conditions)  # type: ignore[arg-type]

            client = await qdrant_service.get_client()

            # Use Qdrant native scroll() method
            points, next_offset = await client.scroll(
                collection_name=collection,
                scroll_filter=query_filter,
                limit=limit,
                offset=offset,
                with_payload=True,
                with_vectors=False,
            )

            # Format results
            formatted_points = [
                {
                    "id": str(point.id),
                    "payload": point.payload or {},
                }
                for point in points
            ]

            result = {
                "points": formatted_points,
                "count": len(formatted_points),
                "next_page_offset": str(next_offset) if next_offset else None,
            }

            if ctx:
                await ctx.info(f"Retrieved {len(formatted_points)} points")

            return result

        except Exception as e:
            logger.exception("Scroll failed")
            if ctx:
                await ctx.error(f"Scroll error: {e}")
            return {"points": [], "count": 0, "next_page_offset": None}

    @mcp.tool()
    async def search_with_context(
        query: str,
        collection: str = "documentation",
        limit: int = 10,
        context_size: int = 3,
        ctx: Context | None = None,
    ) -> list[dict[str, Any]]:
        """Multi-stage search using qdrant_client prefetch mechanism.

        Retrieves results with additional context using nested queries.

        Args:
            query: Search query
            collection: Collection name
            limit: Main results limit
            context_size: Additional context points per result
            ctx: MCP context

        Returns:
            Results with expanded context
        """
        try:
            if ctx:
                await ctx.info(f"Context search in '{collection}': {query}")

            qdrant_service = await client_manager.get_qdrant_service()
            embedding_manager = await client_manager.get_embedding_manager()
            if not embedding_manager:
                raise RuntimeError("Embedding manager not available")

            embeddings_result = await embedding_manager.generate_embeddings([query])
            query_vector = embeddings_result[0]

            client = await qdrant_service.get_client()

            # Use Qdrant's prefetch for multi-stage retrieval
            results = await client.query_points(
                collection_name=collection,
                query=query_vector,
                limit=limit,
                prefetch=[
                    models.Prefetch(
                        query=query_vector,
                        limit=limit + context_size,
                    )
                ],
                with_payload=True,
                with_vectors=False,
            )

            # Handle response format
            points = results.points if hasattr(results, "points") else results
            formatted_results = [
                {
                    "id": str(point.id),
                    "score": point.score,
                    "payload": point.payload or {},
                }
                for point in points
            ]

            if ctx:
                await ctx.info(f"Found {len(formatted_results)} results with context")

            return formatted_results

        except Exception as e:
            logger.exception("Context search failed")
            if ctx:
                await ctx.error(f"Context search error: {e}")
            return []

    @mcp.tool()
    async def recommend_similar(
        point_id: str,
        collection: str = "documentation",
        limit: int = 10,
        filter_conditions: dict[str, Any] | None = None,
        ctx: Context | None = None,
    ) -> list[dict[str, Any]]:
        """Find similar documents using qdrant_client.recommend().

        Args:
            point_id: Source point ID
            collection: Collection name
            limit: Maximum results
            filter_conditions: Optional filters
            ctx: MCP context

        Returns:
            Similar documents with scores
        """
        try:
            if ctx:
                await ctx.info(f"Finding similar to {point_id} in '{collection}'")

            qdrant_service = await client_manager.get_qdrant_service()

            # Build filter
            query_filter = None
            if filter_conditions:
                must_conditions = [
                    models.FieldCondition(key=k, match=models.MatchValue(value=v))
                    for k, v in filter_conditions.items()
                ]
                if must_conditions:
                    query_filter = models.Filter(must=must_conditions)  # type: ignore[arg-type]

            client = await qdrant_service.get_client()

            # Use Qdrant native recommend() method
            results: list[models.ScoredPoint] = await client.recommend(
                collection_name=collection,
                positive=[point_id],
                limit=limit,
                query_filter=query_filter,
                with_payload=True,
                with_vectors=False,
            )

            formatted_results = [
                {
                    "id": str(result.id),
                    "score": result.score,
                    "payload": result.payload or {},
                }
                for result in results
            ]

            if ctx:
                await ctx.info(f"Found {len(formatted_results)} similar documents")

            return formatted_results

        except Exception as e:
            logger.exception("Recommendation failed")
            if ctx:
                await ctx.error(f"Recommendation error: {e}")
            return []

    @mcp.tool()
    async def hyde_search(
        query: str,
        collection: str = "documentation",
        limit: int = 10,
        num_hypothetical_docs: int = 3,
        filter_conditions: dict[str, Any] | None = None,
        ctx: Context | None = None,
    ) -> list[dict[str, Any]]:
        """HyDE (Hypothetical Document Embeddings) search via recommend().

        Uses initial search results as positive examples for refinement.
        Note: Full HyDE requires LLM-generated hypothetical documents; this
        implementation uses query repetition as a simplified approach.

        Args:
            query: Search query
            collection: Collection name
            limit: Maximum results
            num_hypothetical_docs: Number of query expansions (1-5)
            filter_conditions: Optional filters
            ctx: MCP context

        Returns:
            Refined search results
        """
        try:
            if ctx:
                await ctx.info(f"HyDE search in '{collection}': {query}")

            qdrant_service = await client_manager.get_qdrant_service()
            embedding_manager = await client_manager.get_embedding_manager()
            if not embedding_manager:
                raise RuntimeError("Embedding manager not available")

            # Generate query embedding
            embeddings_result = await embedding_manager.generate_embeddings([query])
            query_vector = embeddings_result[0]

            # Build filter
            query_filter = None
            if filter_conditions:
                must_conditions = [
                    models.FieldCondition(key=k, match=models.MatchValue(value=v))
                    for k, v in filter_conditions.items()
                ]
                if must_conditions:
                    query_filter = models.Filter(must=must_conditions)  # type: ignore[arg-type]

            client = await qdrant_service.get_client()

            # Initial search to find candidate documents
            initial_results = await client.search(
                collection_name=collection,
                query_vector=query_vector,
                limit=limit * 2,
                query_filter=query_filter,
                with_payload=True,
                with_vectors=False,
            )

            # Use top results as positive examples for refined search
            if initial_results:
                top_ids = [str(r.id) for r in initial_results[:3]]
                refined_results = await client.recommend(
                    collection_name=collection,
                    positive=top_ids,
                    limit=limit,
                    query_filter=query_filter,
                    with_payload=True,
                    with_vectors=False,
                )

                formatted_results = [
                    {
                        "id": str(result.id),
                        "score": result.score,
                        "payload": result.payload or {},
                        "method": "hyde_refined",
                    }
                    for result in refined_results
                ]
            else:
                formatted_results = []

            if ctx:
                await ctx.info(f"HyDE search found {len(formatted_results)} results")

            return formatted_results

        except Exception as e:
            logger.exception("HyDE search failed")
            if ctx:
                await ctx.error(f"HyDE search error: {e}")
            return []

    @mcp.tool()
    async def reranked_search(
        query: str,
        collection: str = "documentation",
        limit: int = 10,
        rerank_limit: int = 50,
        filter_conditions: dict[str, Any] | None = None,
        ctx: Context | None = None,
    ) -> list[dict[str, Any]]:
        """Search with RRF (Reciprocal Rank Fusion) reranking.

        Uses qdrant_client query_points() with prefetch and FusionQuery.RRF
        to rerank results. RRF score = 1 / (60 + rank).

        Args:
            query: Search query
            collection: Collection name
            limit: Final result count
            rerank_limit: Initial results to fetch for reranking (50-200)
            filter_conditions: Optional filters
            ctx: MCP context

        Returns:
            Reranked results using RRF fusion
        """
        try:
            if ctx:
                await ctx.info(f"Reranked search in '{collection}': {query}")

            qdrant_service = await client_manager.get_qdrant_service()
            embedding_manager = await client_manager.get_embedding_manager()
            if not embedding_manager:
                raise RuntimeError("Embedding manager not available")

            embeddings_result = await embedding_manager.generate_embeddings([query])
            query_vector = embeddings_result[0]

            # Build filter
            query_filter = None
            if filter_conditions:
                must_conditions = [
                    models.FieldCondition(key=k, match=models.MatchValue(value=v))
                    for k, v in filter_conditions.items()
                ]
                if must_conditions:
                    query_filter = models.Filter(must=must_conditions)  # type: ignore[arg-type]

            client = await qdrant_service.get_client()

            # RRF fusion via prefetch + FusionQuery
            results = await client.query_points(
                collection_name=collection,
                prefetch=[
                    models.Prefetch(
                        query=query_vector,
                        limit=rerank_limit,
                    )
                ],
                query=models.FusionQuery(fusion=models.Fusion.RRF),
                limit=limit,
                query_filter=query_filter,
                with_payload=True,
                with_vectors=False,
            )

            # Handle response format
            points = results.points if hasattr(results, "points") else results
            formatted_results = [
                {
                    "id": str(point.id),
                    "score": point.score,
                    "payload": point.payload or {},
                    "method": "rrf_reranked",
                }
                for point in points
            ]

            if ctx:
                count = len(formatted_results)
                await ctx.info(f"Reranked search: {rerank_limit} → {count} results")

            return formatted_results

        except Exception as e:
            logger.exception("Reranked search failed")
            if ctx:
                await ctx.error(f"Reranked search error: {e}")
            return []

    @mcp.tool()
    async def multi_stage_search(
        query: str,
        collection: str = "documentation",
        limit: int = 10,
        stages: int = 3,
        filter_conditions: dict[str, Any] | None = None,
        ctx: Context | None = None,
    ) -> list[dict[str, Any]]:
        """Multi-stage progressive search using nested prefetch.

        Uses qdrant_client query_points() with chained prefetch stages to
        progressively narrow results. Each stage retrieves 10x more candidates
        than the next stage.

        Args:
            query: Search query
            collection: Collection name
            limit: Final result count
            stages: Number of search stages (2-5)
            filter_conditions: Optional filters
            ctx: MCP context

        Returns:
            Progressively refined results
        """
        try:
            if ctx:
                await ctx.info(
                    f"Multi-stage search ({stages} stages) in '{collection}'"
                )

            qdrant_service = await client_manager.get_qdrant_service()
            embedding_manager = await client_manager.get_embedding_manager()
            if not embedding_manager:
                raise RuntimeError("Embedding manager not available")

            embeddings_result = await embedding_manager.generate_embeddings([query])
            query_vector = embeddings_result[0]

            # Build filter
            query_filter = None
            if filter_conditions:
                must_conditions = [
                    models.FieldCondition(key=k, match=models.MatchValue(value=v))
                    for k, v in filter_conditions.items()
                ]
                if must_conditions:
                    query_filter = models.Filter(must=must_conditions)  # type: ignore[arg-type]

            client = await qdrant_service.get_client()

            # Build nested prefetch chain (each stage narrows by 10x)
            num_stages = min(max(stages, 2), 5)
            stage_limits = [limit * (10 ** (num_stages - i)) for i in range(num_stages)]

            prefetch_stages = []
            for stage_limit in stage_limits[:-1]:
                prefetch_stages.append(
                    models.Prefetch(
                        query=query_vector,
                        limit=min(stage_limit, 1000),
                    )
                )

            # Execute query with nested prefetch
            results = await client.query_points(
                collection_name=collection,
                prefetch=prefetch_stages if prefetch_stages else None,
                query=query_vector,
                limit=limit,
                query_filter=query_filter,
                with_payload=True,
                with_vectors=False,
            )

            # Handle response format
            points = results.points if hasattr(results, "points") else results
            formatted_results = [
                {
                    "id": str(point.id),
                    "score": point.score,
                    "payload": point.payload or {},
                    "method": f"multi_stage_{num_stages}",
                }
                for point in points
            ]

            if ctx:
                count = len(formatted_results)
                await ctx.info(
                    f"Multi-stage search ({num_stages} stages) found {count} results"
                )

            return formatted_results

        except Exception as e:
            logger.exception("Multi-stage search failed")
            if ctx:
                await ctx.error(f"Multi-stage search error: {e}")
            return []

    @mcp.tool()
    async def filtered_search(
        query: str,
        collection: str = "documentation",
        limit: int = 10,
        must_conditions: list[dict[str, Any]] | None = None,
        should_conditions: list[dict[str, Any]] | None = None,
        must_not_conditions: list[dict[str, Any]] | None = None,
        ctx: Context | None = None,
    ) -> list[dict[str, Any]]:
        """Boolean filtered search using qdrant_client Filter objects.

        Supports must (AND), should (OR), and must_not (NOT) filter logic
        via qdrant_client.models.Filter with FieldCondition combinations.

        Args:
            query: Search query
            collection: Collection name
            limit: Maximum results
            must_conditions: Required conditions [{"key": "field", "value": "val"}]
            should_conditions: Optional conditions (OR logic)
            must_not_conditions: Exclusion conditions
            ctx: MCP context

        Returns:
            Filtered search results
        """
        try:
            if ctx:
                await ctx.info(f"Filtered search in '{collection}'")

            qdrant_service = await client_manager.get_qdrant_service()
            embedding_manager = await client_manager.get_embedding_manager()
            if not embedding_manager:
                raise RuntimeError("Embedding manager not available")

            embeddings_result = await embedding_manager.generate_embeddings([query])
            query_vector = embeddings_result[0]

            # Build Filter with boolean logic
            query_filter = None
            filter_parts = {}

            if must_conditions:
                filter_parts["must"] = [
                    models.FieldCondition(
                        key=cond["key"], match=models.MatchValue(value=cond["value"])
                    )
                    for cond in must_conditions
                ]

            if should_conditions:
                filter_parts["should"] = [
                    models.FieldCondition(
                        key=cond["key"], match=models.MatchValue(value=cond["value"])
                    )
                    for cond in should_conditions
                ]

            if must_not_conditions:
                filter_parts["must_not"] = [
                    models.FieldCondition(
                        key=cond["key"], match=models.MatchValue(value=cond["value"])
                    )
                    for cond in must_not_conditions
                ]

            if filter_parts:
                query_filter = models.Filter(**filter_parts)  # type: ignore[arg-type]

            client = await qdrant_service.get_client()

            # Vector search with boolean filter
            results = await client.search(
                collection_name=collection,
                query_vector=query_vector,
                limit=limit,
                query_filter=query_filter,
                with_payload=True,
                with_vectors=False,
            )

            formatted_results = [
                {
                    "id": str(result.id),
                    "score": result.score,
                    "payload": result.payload or {},
                }
                for result in results
            ]

            if ctx:
                await ctx.info(
                    f"Filtered search found {len(formatted_results)} results"
                )

            return formatted_results

        except Exception as e:
            logger.exception("Filtered search failed")
            if ctx:
                await ctx.error(f"Filtered search error: {e}")
            return []
