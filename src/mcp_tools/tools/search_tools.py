"""Advanced search tools for MCP server."""

import logging
from typing import TYPE_CHECKING
from uuid import uuid4

if TYPE_CHECKING:
    from fastmcp import Context
else:
    # Use a protocol for testing to avoid FastMCP import issues
    from typing import Protocol

    class Context(Protocol):
        async def info(self, msg: str) -> None: ...
        async def debug(self, msg: str) -> None: ...
        async def warning(self, msg: str) -> None: ...
        async def error(self, msg: str) -> None: ...


from src.config import SearchStrategy

from ...infrastructure.client_manager import ClientManager
from ...security import MLSecurityValidator as SecurityValidator
from ..models.requests import FilteredSearchRequest
from ..models.requests import HyDESearchRequest
from ..models.requests import MultiStageSearchRequest
from ..models.requests import SearchRequest
from ..models.responses import HyDEAdvancedResponse
from ..models.responses import SearchResult

logger = logging.getLogger(__name__)


async def _perform_ab_test_search(
    query: str,
    collection: str,
    limit: int,
    domain: "str | None",
    use_cache: bool,
    client_manager: ClientManager,
    ctx: "Context | None",
) -> tuple[list, dict]:
    """Perform A/B test comparing HyDE vs regular search."""
    import asyncio

    # Get services
    hyde_engine = await client_manager.get_hyde_engine()
    qdrant_service = await client_manager.get_qdrant_service()

    # HyDE search
    hyde_task = hyde_engine.enhanced_search(
        query=query,
        collection_name=collection,
        limit=limit * 2,  # Get more for comparison
        domain=domain,
        use_cache=use_cache,
        force_hyde=True,
    )

    # Regular search for comparison
    regular_task = qdrant_service.hybrid_search(
        collection_name=collection,
        query_vector=None,  # Will be generated inside
        sparse_vector=None,
        limit=limit * 2,
        score_threshold=0.0,
        fusion_type="rrf",
        search_accuracy="balanced",
    )

    # Wait for both to complete
    hyde_results, regular_results = await asyncio.gather(
        hyde_task, regular_task, return_exceptions=True
    )

    # Handle any exceptions
    if isinstance(hyde_results, Exception):
        if ctx:
            await ctx.warning(f"HyDE search failed in A/B test: {hyde_results}")
        hyde_results = []
    if isinstance(regular_results, Exception):
        if ctx:
            await ctx.warning(f"Regular search failed in A/B test: {regular_results}")
        regular_results = []

    # Compare results
    ab_test_results = {
        "hyde_count": len(hyde_results) if hyde_results else 0,
        "regular_count": len(regular_results) if regular_results else 0,
        "hyde_avg_score": sum(
            r.get("score", 0) if isinstance(r, dict) else r.score for r in hyde_results
        )
        / len(hyde_results)
        if hyde_results
        else 0,
        "regular_avg_score": sum(r.score for r in regular_results)
        / len(regular_results)
        if regular_results
        else 0,
    }

    # Use HyDE results as primary
    search_results = hyde_results[:limit] if hyde_results else regular_results[:limit]

    return search_results, ab_test_results


def register_tools(mcp, client_manager: ClientManager):
    """Register advanced search tools with the MCP server."""

    async def _search_documents_direct(
        request: SearchRequest, ctx: Context
    ) -> list[SearchResult]:
        """Direct access to search_documents functionality without mock MCP."""
        from ._search_utils import search_documents_core

        return await search_documents_core(request, client_manager, ctx)

    @mcp.tool()
    async def multi_stage_search(
        request: MultiStageSearchRequest, ctx: Context
    ) -> list[SearchResult]:
        """
        Perform multi-stage retrieval with Matryoshka embeddings.

        Implements advanced Query API patterns for complex retrieval strategies
        with optimized prefetch and fusion algorithms.
        """
        # Generate request ID for tracking
        request_id = str(uuid4())
        await ctx.info(
            f"Starting multi-stage search {request_id} with {len(request.stages)} stages"
        )

        try:
            # Validate collection name
            security_validator = SecurityValidator.from_unified_config()
            request.collection = security_validator.validate_collection_name(
                request.collection
            )

            # Get services
            qdrant_service = await client_manager.get_qdrant_service()

            # Convert stage configuration to QdrantService format
            stages = []
            for stage_req in request.stages:
                # Handle both dict and SearchStageRequest object formats
                if isinstance(stage_req, dict):
                    stage = {
                        "query_vector": stage_req["query_vector"],
                        "vector_name": stage_req["vector_name"],
                        "vector_type": stage_req["vector_type"],
                        "limit": stage_req["limit"],
                        "filter": stage_req.get("filters"),
                    }
                else:
                    # SearchStageRequest object
                    stage = {
                        "query_vector": stage_req.query_vector,
                        "vector_name": stage_req.vector_name,
                        "vector_type": stage_req.vector_type.value,
                        "limit": stage_req.limit,
                        "filter": stage_req.filters,
                    }
                stages.append(stage)

            # Perform multi-stage search
            results = await qdrant_service.multi_stage_search(
                collection_name=request.collection,
                stages=stages,
                limit=request.limit,
                fusion_algorithm=request.fusion_algorithm.value,
                search_accuracy=request.search_accuracy.value,
            )

            # Convert to search results
            search_results = [
                SearchResult(
                    id=str(point["id"]),
                    content=point["payload"].get("content", ""),
                    score=point["score"],
                    url=point["payload"].get("url"),
                    title=point["payload"].get("title"),
                    metadata=point["payload"],
                )
                for point in results
            ]

            await ctx.info(
                f"Multi-stage search {request_id} completed: {len(search_results)} results"
            )
            return search_results

        except Exception as e:
            await ctx.error(f"Multi-stage search {request_id} failed: {e}")
            logger.exception(f"Multi-stage search failed: {e}")
            raise

    @mcp.tool()
    async def hyde_search(
        request: HyDESearchRequest, ctx: Context
    ) -> list[SearchResult]:
        """
        Search using HyDE (Hypothetical Document Embeddings).

        Generates hypothetical documents to improve retrieval accuracy by 15-25%
        using advanced Query API with optimized prefetch patterns and LLM generation.
        """
        # Generate request ID for tracking
        request_id = str(uuid4())
        await ctx.info(
            f"Starting HyDE search {request_id} for query: {request.query[:50]}..."
        )

        try:
            # Validate collection name and query
            security_validator = SecurityValidator.from_unified_config()
            request.collection = security_validator.validate_collection_name(
                request.collection
            )
            request.query = security_validator.validate_query_string(request.query)

            # Get services
            try:
                hyde_engine = await client_manager.get_hyde_engine()
            except Exception:
                await ctx.warning(
                    "HyDE engine not available, falling back to regular search"
                )
                # Fallback to regular search
                fallback_request = SearchRequest(
                    query=request.query,
                    collection=request.collection,
                    limit=request.limit,
                    strategy=SearchStrategy.HYBRID,
                    enable_reranking=request.enable_reranking,
                    fusion_algorithm=request.fusion_algorithm,
                    search_accuracy=request.search_accuracy,
                )
                return await _search_documents_direct(fallback_request, ctx)

            # Use HyDE engine for enhanced search
            await ctx.debug(
                f"Using HyDE engine with {request.num_generations} generations"
            )

            # Determine search accuracy level
            search_accuracy = (
                request.search_accuracy.value
                if hasattr(request.search_accuracy, "value")
                else request.search_accuracy
            )

            # Get embedding manager for reranking
            embedding_manager = await client_manager.get_embedding_manager()

            # Perform HyDE search with all optimizations
            results = await hyde_engine.enhanced_search(
                query=request.query,
                collection_name=request.collection,
                limit=request.limit * 3
                if request.enable_reranking
                else request.limit,  # Get more for reranking
                domain=request.domain,
                search_accuracy=search_accuracy,
                use_cache=True,
                force_hyde=True,
            )

            await ctx.debug(f"HyDE engine returned {len(results)} initial results")

            # Convert to SearchResult objects
            search_results = []
            for result in results:
                # Handle both dict and object formats
                if isinstance(result, dict):
                    search_result = SearchResult(
                        id=result.get("id", str(uuid4())),
                        content=result.get("content", ""),
                        score=result.get("score", 0.0),
                        url=result.get("url"),
                        title=result.get("title"),
                        metadata=result.get("metadata")
                        if request.include_metadata
                        else None,
                    )
                else:
                    # Assume it's a Qdrant point object
                    search_result = SearchResult(
                        id=str(result.id),
                        content=result.payload.get("content", ""),
                        score=result.score,
                        url=result.payload.get("url"),
                        title=result.payload.get("title"),
                        metadata=result.payload if request.include_metadata else None,
                    )
                search_results.append(search_result)

            # Apply reranking if enabled and we have results
            if request.enable_reranking and search_results:
                await ctx.debug(
                    f"Applying BGE reranking to {len(search_results)} results"
                )
                # Convert to format expected by reranker
                results_for_reranking = [
                    {"content": r.content, "original": r} for r in search_results
                ]

                # Rerank results
                reranked = await embedding_manager.rerank_results(
                    query=request.query, results=results_for_reranking
                )

                # Extract reranked SearchResult objects and limit to requested number
                search_results = [r["original"] for r in reranked[: request.limit]]
                await ctx.debug(
                    f"Reranking complete, returning top {len(search_results)} results"
                )
            else:
                # Without reranking, just limit to requested number
                search_results = search_results[: request.limit]

            await ctx.info(
                f"HyDE search {request_id} completed: {len(search_results)} results found"
            )
            return search_results

        except Exception as e:
            await ctx.error(f"HyDE search {request_id} failed: {e}")
            logger.exception(f"HyDE search failed: {e}")
            # Fallback to regular search on error
            try:
                await ctx.warning(
                    "HyDE search failed, attempting fallback to regular search"
                )
                fallback_request = SearchRequest(
                    query=request.query,
                    collection=request.collection,
                    limit=request.limit,
                    strategy=SearchStrategy.HYBRID,
                    enable_reranking=request.enable_reranking,
                )
                return await _search_documents_direct(fallback_request, ctx)
            except Exception as fallback_error:
                await ctx.error(f"Fallback search also failed: {fallback_error}")
                raise e from fallback_error

    @mcp.tool()
    async def hyde_search_advanced(
        query: str,
        collection: str = "documentation",
        domain: "str | None" = None,
        num_generations: int = 5,
        generation_temperature: float = 0.7,
        limit: int = 10,
        enable_reranking: bool = True,
        enable_ab_testing: bool = False,
        use_cache: bool = True,
        ctx: "Context | None" = None,
    ) -> HyDEAdvancedResponse:
        """
        Advanced HyDE search with full configuration control and A/B testing.

        Provides comprehensive HyDE search capabilities with detailed metrics,
        A/B testing comparison, and fine-grained control over generation parameters.
        """
        # Generate request ID for tracking
        request_id = str(uuid4())
        if ctx:
            await ctx.info(
                f"Starting advanced HyDE search {request_id} for query: {query[:50]}..."
            )

        try:
            # Validate inputs
            security_validator = SecurityValidator.from_unified_config()
            collection = security_validator.validate_collection_name(collection)
            query = security_validator.validate_query_string(query)

            # Get services
            try:
                hyde_engine = await client_manager.get_hyde_engine()
                embedding_manager = await client_manager.get_embedding_manager()
            except Exception as e:
                if ctx:
                    await ctx.error("HyDE engine not available")
                raise ValueError("HyDE engine not initialized") from e

            result_dict = {
                "request_id": request_id,
                "query": query,
                "collection": collection,
                "hyde_config": {
                    "domain": domain,
                    "num_generations": num_generations,
                    "temperature": generation_temperature,
                    "enable_ab_testing": enable_ab_testing,
                    "use_cache": use_cache,
                },
                "results": [],
                "metrics": {},
                "ab_test_results": None,
            }

            import time

            start_time = time.time()

            # Perform search based on configuration
            if enable_ab_testing:
                if ctx:
                    await ctx.debug("Running A/B test: HyDE vs Regular search")

                try:
                    search_results, ab_test_results = await _perform_ab_test_search(
                        query, collection, limit, domain, use_cache, client_manager, ctx
                    )
                    result_dict["ab_test_results"] = ab_test_results
                except Exception as ab_error:
                    if ctx:
                        await ctx.warning(f"A/B testing failed: {ab_error}")
                    # Fallback to regular HyDE search
                    search_results = await hyde_engine.enhanced_search(
                        query=query,
                        collection_name=collection,
                        limit=limit,
                        domain=domain,
                        use_cache=use_cache,
                        force_hyde=True,
                    )
            else:
                # Regular HyDE search
                search_results = await hyde_engine.enhanced_search(
                    query=query,
                    collection_name=collection,
                    limit=limit * 3 if enable_reranking else limit,
                    domain=domain,
                    use_cache=use_cache,
                    force_hyde=True,
                )

            search_time = (time.time() - start_time) * 1000  # Convert to milliseconds

            # Convert results to consistent format
            formatted_results = []
            for res in search_results:
                if isinstance(res, dict):
                    formatted_result = {
                        "id": res.get("id", str(uuid4())),
                        "content": res.get("content", ""),
                        "score": res.get("score", 0.0),
                        "url": res.get("url"),
                        "title": res.get("title"),
                        "metadata": res.get("metadata"),
                    }
                else:
                    # Qdrant point object
                    formatted_result = {
                        "id": str(res.id),
                        "content": res.payload.get("content", ""),
                        "score": res.score,
                        "url": res.payload.get("url"),
                        "title": res.payload.get("title"),
                        "metadata": res.payload,
                    }
                formatted_results.append(formatted_result)

            # Apply reranking if enabled
            if enable_reranking and formatted_results:
                if ctx:
                    await ctx.debug(
                        f"Applying BGE reranking to {len(formatted_results)} results"
                    )

                # Convert to format expected by reranker
                results_for_reranking = [
                    {"content": r["content"], "original": r} for r in formatted_results
                ]

                # Rerank results
                reranked = await embedding_manager.rerank_results(
                    query=query, results=results_for_reranking
                )

                # Extract reranked results and limit to requested number
                formatted_results = [r["original"] for r in reranked[:limit]]
                if ctx:
                    await ctx.debug(
                        f"Reranking complete, returning top {len(formatted_results)} results"
                    )

            # Collect metrics
            result_dict["metrics"] = {
                "search_time_ms": round(search_time, 2),
                "results_found": len(formatted_results),
                "reranking_applied": enable_reranking,
                "cache_used": use_cache,
                "generation_parameters": {
                    "num_generations": num_generations,
                    "temperature": generation_temperature,
                    "domain": domain,
                },
            }

            result_dict["results"] = formatted_results

            if ctx:
                await ctx.info(
                    f"Advanced HyDE search {request_id} completed: {len(formatted_results)} results in {search_time:.2f}ms"
                )

            return HyDEAdvancedResponse(**result_dict)

        except Exception as e:
            if ctx:
                await ctx.error(f"Advanced HyDE search {request_id} failed: {e}")
            logger.exception(f"Advanced HyDE search failed: {e}")
            raise

    @mcp.tool()
    async def filtered_search(
        request: FilteredSearchRequest, ctx: Context
    ) -> list[SearchResult]:
        """
        Optimized filtered search using indexed payload fields.

        Performs efficient filtered search with Query API optimizations
        for high-performance filtering on indexed fields.
        """
        # Generate request ID for tracking
        request_id = str(uuid4())
        await ctx.info(
            f"Starting filtered search {request_id} with filters: {request.filters}"
        )

        try:
            # Validate collection name and query
            security_validator = SecurityValidator.from_unified_config()
            request.collection = security_validator.validate_collection_name(
                request.collection
            )
            request.query = security_validator.validate_query_string(request.query)

            # Get services
            embedding_manager = await client_manager.get_embedding_manager()
            qdrant_service = await client_manager.get_qdrant_service()

            # Generate embedding for query
            embedding_result = await embedding_manager.generate_embeddings(
                [request.query], generate_sparse=False
            )
            query_vector = embedding_result.embeddings[0]

            await ctx.debug(f"Generated embedding with dimension {len(query_vector)}")

            # Perform filtered search
            results = await qdrant_service.filtered_search(
                collection_name=request.collection,
                query_vector=query_vector,
                filters=request.filters,
                limit=request.limit,
                search_accuracy=request.search_accuracy.value,
            )

            # Convert to search results
            search_results = [
                SearchResult(
                    id=str(point["id"]),
                    content=point["payload"].get("content", ""),
                    score=point["score"],
                    url=point["payload"].get("url"),
                    title=point["payload"].get("title"),
                    metadata=point["payload"] if request.include_metadata else None,
                )
                for point in results
            ]

            await ctx.info(
                f"Filtered search {request_id} completed: {len(search_results)} results"
            )
            return search_results

        except Exception as e:
            await ctx.error(f"Filtered search {request_id} failed: {e}")
            logger.exception(f"Filtered search failed: {e}")
            raise
