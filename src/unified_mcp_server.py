#!/usr/bin/env python3
"""
Unified MCP Server for AI Documentation Vector DB

This server consolidates all functionality from the refactored servers and exposes
advanced features via FastMCP 2.0 tools while using the service layer for implementation.

Features:
- Hybrid search with dense+sparse vectors and reranking
- Multi-provider embedding generation with smart model selection
- Advanced chunking strategies (Basic, Enhanced, AST-based)
- Project-based document management
- Two-tier caching with metrics
- Batch processing and streaming support
- Cost estimation and optimization
- Analytics and monitoring
"""

import asyncio
import logging
from datetime import UTC
from datetime import datetime
from typing import Any
from uuid import uuid4

from fastmcp import Context
from fastmcp import FastMCP

# Handle both module and script imports
try:
    from chunking import ChunkingConfig
    from chunking import EnhancedChunker
    from config.enums import ChunkingStrategy
    from config.enums import SearchStrategy
    from mcp.models.requests import AnalyticsRequest
    from mcp.models.requests import BatchRequest
    from mcp.models.requests import DocumentRequest
    from mcp.models.requests import EmbeddingRequest
    from mcp.models.requests import ProjectRequest
    from mcp.models.requests import SearchRequest
    from mcp.models.responses import SearchResult
    from mcp.service_manager import UnifiedServiceManager
    from security import SecurityValidator
    from services.logging_config import configure_logging
except ImportError:
    from .chunking import ChunkingConfig
    from .chunking import EnhancedChunker
    from .config.enums import ChunkingStrategy
    from .config.enums import SearchStrategy
    from .mcp.models.requests import AnalyticsRequest
    from .mcp.models.requests import BatchRequest
    from .mcp.models.requests import DocumentRequest
    from .mcp.models.requests import EmbeddingRequest
    from .mcp.models.requests import FilteredSearchRequest
    from .mcp.models.requests import HyDESearchRequest
    from .mcp.models.requests import MultiStageSearchRequest
    from .mcp.models.requests import ProjectRequest
    from .mcp.models.requests import SearchRequest
    from .mcp.models.responses import SearchResult
    from .mcp.service_manager import UnifiedServiceManager
    from .security import SecurityValidator
    from .services.logging_config import configure_logging

# Initialize logging
configure_logging()
logger = logging.getLogger(__name__)

# Initialize FastMCP server
mcp = FastMCP("ai-docs-vector-db-unified")

# Initialize service manager
service_manager = UnifiedServiceManager()


# Search and Retrieval Tools
@mcp.tool()
async def search_documents(request: SearchRequest, ctx: Context) -> list[SearchResult]:
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
        # Validate collection name and query
        security_validator = SecurityValidator.from_unified_config()
        request.collection = security_validator.validate_collection_name(
            request.collection
        )
        request.query = security_validator.validate_query_string(request.query)

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
        embedding_result = await service_manager.embedding_manager.generate_embeddings(
            [request.query], generate_sparse=generate_sparse
        )

        query_vector = embedding_result["embeddings"][0]
        sparse_vector = None
        if embedding_result.get("sparse_embeddings"):
            sparse_vector = embedding_result["sparse_embeddings"][0]
            await ctx.debug(
                f"Generated sparse vector with {len(sparse_vector['indices'])} non-zero elements"
            )

        await ctx.debug(f"Generated embedding with dimension {len(query_vector)}")

        # Perform search based on strategy using enhanced hybrid_search
        if request.strategy == SearchStrategy.HYBRID:
            # For hybrid search, use both dense and sparse vectors
            results = await service_manager.qdrant_service.hybrid_search(
                collection_name=request.collection,
                query_vector=query_vector,
                sparse_vector=sparse_vector,  # Now using actual sparse vector
                limit=request.limit * 3 if request.enable_reranking else request.limit,
                score_threshold=0.0,
                fusion_type=request.fusion_algorithm.value,
                search_accuracy=request.search_accuracy.value,
            )
        else:
            # For dense search, use direct vector search
            results = await service_manager.qdrant_service.hybrid_search(
                collection_name=request.collection,
                query_vector=query_vector,
                sparse_vector=None,  # Dense search only
                limit=request.limit * 3 if request.enable_reranking else request.limit,
                score_threshold=0.0,
                fusion_type=request.fusion_algorithm.value,
                search_accuracy=request.search_accuracy.value,
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
            await ctx.debug(f"Applying BGE reranking to {len(search_results)} results")
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

    # Validate collection name
    security_validator = SecurityValidator.from_unified_config()
    collection = security_validator.validate_collection_name(collection)

    # Log if context available
    if ctx:
        await ctx.info(f"Starting similarity search in collection {collection}")

    try:
        # Generate embedding for content
        if ctx:
            await ctx.debug(f"Generating embeddings for content: {content[:50]}...")
        embeddings = await service_manager.embedding_manager.generate_embeddings(
            [content]
        )
        embedding = embeddings[0]

        # Search by vector using hybrid_search with dense only
        results = await service_manager.qdrant_service.hybrid_search(
            collection_name=collection,
            query_vector=embedding,
            sparse_vector=None,
            limit=limit,
            score_threshold=threshold,
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
                metadata=point.payload,
            )
            search_results.append(result)

        if ctx:
            await ctx.info(
                f"Similarity search completed: {len(search_results)} results found"
            )

        return search_results

    except Exception as e:
        if ctx:
            await ctx.error(f"Similarity search failed: {e}")
        logger.error(f"Similarity search failed: {e}")
        raise


# Advanced Search Tools for Query API


@mcp.tool()
async def multi_stage_search(
    request: MultiStageSearchRequest, ctx: Context
) -> list[SearchResult]:
    """
    Perform multi-stage retrieval with Matryoshka embeddings.

    Implements advanced Query API patterns for complex retrieval strategies
    with optimized prefetch and fusion algorithms.
    """
    await service_manager.initialize()

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

        # Convert SearchStageRequest objects to dictionaries for QdrantService
        stages = []
        for stage_req in request.stages:
            stage = {
                "query_vector": stage_req.query_vector,
                "vector_name": stage_req.vector_name,
                "vector_type": stage_req.vector_type.value,
                "limit": stage_req.limit,
                "filter": stage_req.filters,
            }
            stages.append(stage)

        # Perform multi-stage search
        results = await service_manager.qdrant_service.multi_stage_search(
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
        logger.error(f"Multi-stage search failed: {e}")
        raise


@mcp.tool()
async def hyde_search(request: HyDESearchRequest, ctx: Context) -> list[SearchResult]:
    """
    Search using HyDE (Hypothetical Document Embeddings).

    Generates hypothetical documents to improve retrieval accuracy by 15-25%
    using advanced Query API with optimized prefetch patterns and LLM generation.
    """
    await service_manager.initialize()

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

        # Check if HyDE engine is available
        if (
            not hasattr(service_manager, "hyde_engine")
            or service_manager.hyde_engine is None
        ):
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
            return await search_documents(fallback_request, ctx)

        # Use HyDE engine for enhanced search
        await ctx.debug(f"Using HyDE engine with {request.num_generations} generations")

        # Determine search accuracy level
        search_accuracy = (
            request.search_accuracy.value
            if hasattr(request.search_accuracy, "value")
            else request.search_accuracy
        )

        # Perform HyDE search with all optimizations
        results = await service_manager.hyde_engine.enhanced_search(
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
            await ctx.debug(f"Applying BGE reranking to {len(search_results)} results")
            # Convert to format expected by reranker
            results_for_reranking = [
                {"content": r.content, "original": r} for r in search_results
            ]

            # Rerank results
            reranked = await service_manager.embedding_manager.rerank_results(
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
        logger.error(f"HyDE search failed: {e}")
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
            return await search_documents(fallback_request, ctx)
        except Exception as fallback_error:
            await ctx.error(f"Fallback search also failed: {fallback_error}")
            raise e from fallback_error


async def _perform_ab_test_search(
    query: str,
    collection: str,
    limit: int,
    domain: str | None,
    use_cache: bool,
    ctx: Context | None,
) -> tuple[list, dict]:
    """Perform A/B test comparing HyDE vs regular search."""
    import asyncio

    # HyDE search
    hyde_task = service_manager.hyde_engine.enhanced_search(
        query=query,
        collection_name=collection,
        limit=limit * 2,  # Get more for comparison
        domain=domain,
        use_cache=use_cache,
        force_hyde=True,
    )

    # Regular search for comparison
    regular_task = service_manager.qdrant_service.hybrid_search(
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


@mcp.tool()
async def hyde_search_advanced(  # noqa: PLR0912, PLR0915
    query: str,
    collection: str = "documentation",
    domain: str | None = None,
    num_generations: int = 5,
    generation_temperature: float = 0.7,
    limit: int = 10,
    enable_reranking: bool = True,
    enable_ab_testing: bool = False,
    use_cache: bool = True,
    ctx: Context = None,
) -> dict[str, Any]:
    """
    Advanced HyDE search with full configuration control and A/B testing.

    Provides comprehensive HyDE search capabilities with detailed metrics,
    A/B testing comparison, and fine-grained control over generation parameters.
    """
    await service_manager.initialize()

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

        # Check if HyDE engine is available
        if (
            not hasattr(service_manager, "hyde_engine")
            or service_manager.hyde_engine is None
        ):
            if ctx:
                await ctx.error("HyDE engine not available")
            raise ValueError("HyDE engine not initialized")

        result = {
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
                    query, collection, limit, domain, use_cache, ctx
                )
                result["ab_test_results"] = ab_test_results
            except Exception as ab_error:
                if ctx:
                    await ctx.warning(f"A/B testing failed: {ab_error}")
                # Fallback to regular HyDE search
                search_results = await service_manager.hyde_engine.enhanced_search(
                    query=query,
                    collection_name=collection,
                    limit=limit,
                    domain=domain,
                    use_cache=use_cache,
                    force_hyde=True,
                )
        else:
            # Regular HyDE search
            search_results = await service_manager.hyde_engine.enhanced_search(
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
            reranked = await service_manager.embedding_manager.rerank_results(
                query=query, results=results_for_reranking
            )

            # Extract reranked results and limit to requested number
            formatted_results = [r["original"] for r in reranked[:limit]]
            if ctx:
                await ctx.debug(
                    f"Reranking complete, returning top {len(formatted_results)} results"
                )

        # Collect metrics
        result["metrics"] = {
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

        result["results"] = formatted_results

        if ctx:
            await ctx.info(
                f"Advanced HyDE search {request_id} completed: {len(formatted_results)} results in {search_time:.2f}ms"
            )

        return result

    except Exception as e:
        if ctx:
            await ctx.error(f"Advanced HyDE search {request_id} failed: {e}")
        logger.error(f"Advanced HyDE search failed: {e}")
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
    await service_manager.initialize()

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

        # Generate embedding for query
        embedding_result = await service_manager.embedding_manager.generate_embeddings(
            [request.query], generate_sparse=False
        )
        query_vector = embedding_result["embeddings"][0]

        await ctx.debug(f"Generated embedding with dimension {len(query_vector)}")

        # Perform filtered search
        results = await service_manager.qdrant_service.filtered_search(
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
        logger.error(f"Filtered search failed: {e}")
        raise


# Embedding Management Tools
@mcp.tool()
async def generate_embeddings(request: EmbeddingRequest) -> dict[str, Any]:
    """
    Generate embeddings using the optimal provider.

    Automatically selects the best embedding model based on cost,
    performance, and availability.
    """
    await service_manager.initialize()

    try:
        # Generate embeddings
        embeddings = await service_manager.embedding_manager.generate_embeddings(
            texts=request.texts,
            model_name=request.model,
            batch_size=request.batch_size,
        )

        # Get provider info
        provider_info = service_manager.embedding_manager.get_current_provider_info()

        return {
            "embeddings": embeddings,
            "count": len(embeddings),
            "dimensions": len(embeddings[0]) if embeddings else 0,
            "provider": provider_info.get("name", "unknown"),
            "model": provider_info.get("model", "unknown"),
            "cost_estimate": len(request.texts) * 0.00002,  # Rough estimate
        }

    except Exception as e:
        logger.error(f"Embedding generation failed: {e}")
        raise


@mcp.tool()
async def list_embedding_providers() -> list[dict[str, Any]]:
    """
    List available embedding providers and their capabilities.

    Shows all configured providers with their models, dimensions,
    and current availability status.
    """
    await service_manager.initialize()

    providers = []

    # Get provider information
    for provider_name in ["openai", "fastembed"]:
        info = {
            "name": provider_name,
            "available": False,
            "models": [],
            "default_model": None,
            "dimensions": None,
        }

        # Check if provider is available
        try:
            if provider_name == "openai" and service_manager.config.openai_api_key:
                info["available"] = True
                info["models"] = ["text-embedding-3-small", "text-embedding-3-large"]
                info["default_model"] = "text-embedding-3-small"
                info["dimensions"] = 1536
            elif provider_name == "fastembed":
                info["available"] = True
                info["models"] = [
                    "BAAI/bge-small-en-v1.5",
                    "sentence-transformers/all-MiniLM-L6-v2",
                ]
                info["default_model"] = "BAAI/bge-small-en-v1.5"
                info["dimensions"] = 384
        except Exception as e:
            logger.warning(f"Failed to check provider {provider_name}: {e}")

        providers.append(info)

    return providers


# Document Management Tools
@mcp.tool()
async def add_document(request: DocumentRequest, ctx: Context) -> dict[str, Any]:
    """
    Add a document to the vector database with smart chunking.

    Crawls the URL, applies the selected chunking strategy, generates
    embeddings, and stores in the specified collection.
    """
    await service_manager.initialize()

    doc_id = str(uuid4())
    await ctx.info(f"Processing document {doc_id}: {request.url}")

    try:
        # Validate URL using SecurityValidator
        security_validator = SecurityValidator.from_unified_config()
        validated_url = security_validator.validate_url(request.url)
        request.url = validated_url
        # Check cache for existing document
        cache_key = f"doc:{request.url}"
        cached = await service_manager.cache_manager.get(cache_key)
        if cached:
            await ctx.debug(f"Document {doc_id} found in cache")
            return cached

        # Crawl the URL
        await ctx.debug(f"Crawling URL for document {doc_id}")
        crawl_result = await service_manager.crawl_manager.crawl_single(request.url)
        if not crawl_result or not crawl_result.markdown:
            await ctx.error(f"Failed to crawl {request.url}")
            raise ValueError(f"Failed to crawl {request.url}")

        # Configure chunking
        chunk_config = ChunkingConfig(
            strategy=request.chunk_strategy,
            chunk_size=request.chunk_size,
            chunk_overlap=request.chunk_overlap,
        )

        # Chunk the document
        await ctx.debug(
            f"Chunking document {doc_id} with strategy {request.chunk_strategy}"
        )
        chunker = EnhancedChunker(chunk_config)
        chunks = chunker.chunk_content(
            content=crawl_result.markdown,
            title=crawl_result.metadata.get("title", ""),
            url=crawl_result.metadata.get("url", request.url),
        )
        await ctx.debug(f"Created {len(chunks)} chunks for document {doc_id}")

        # Generate embeddings for chunks
        texts = [chunk["content"] for chunk in chunks]
        await ctx.debug(f"Generating embeddings for {len(texts)} chunks")
        embeddings = await service_manager.embedding_manager.generate_embeddings(texts)

        # Prepare points for insertion
        points = []
        for i, (chunk, embedding) in enumerate(zip(chunks, embeddings, strict=False)):
            point = {
                "id": str(uuid4()),
                "vector": embedding,
                "payload": {
                    "content": chunk["content"],
                    "url": request.url,
                    "title": crawl_result.metadata.get("title", ""),
                    "chunk_index": i,
                    "total_chunks": len(chunks),
                    **chunk.get("metadata", {}),
                },
            }
            points.append(point)

        # Ensure collection exists
        await service_manager.qdrant_service.create_collection(
            collection_name=request.collection,
            vector_size=len(embeddings[0]),
            distance="Cosine",
            sparse_vector_name="sparse"
            if request.chunk_strategy != ChunkingStrategy.BASIC
            else None,
            enable_quantization=True,
        )

        # Insert points
        await service_manager.qdrant_service.upsert_points(
            collection_name=request.collection,
            points=points,
        )

        # Prepare response
        result = {
            "url": request.url,
            "title": crawl_result.metadata.get("title", ""),
            "chunks_created": len(chunks),
            "collection": request.collection,
            "chunking_strategy": request.chunk_strategy.value,
            "embedding_dimensions": len(embeddings[0]),
        }

        # Cache result
        await service_manager.cache_manager.set(cache_key, result, ttl=86400)

        await ctx.info(
            f"Document {doc_id} processed successfully: "
            f"{len(chunks)} chunks created in collection {request.collection}"
        )

        return result

    except Exception as e:
        await ctx.error(f"Failed to process document {doc_id}: {e}")
        logger.error(f"Failed to add document: {e}")
        raise


@mcp.tool()
async def add_documents_batch(request: BatchRequest) -> dict[str, Any]:
    """
    Add multiple documents in batch with optimized processing.

    Processes multiple URLs concurrently with rate limiting and
    progress tracking.
    """
    await service_manager.initialize()

    results = {
        "successful": [],
        "failed": [],
        "total": len(request.urls),
    }

    # Process URLs in batches
    semaphore = asyncio.Semaphore(request.max_concurrent)

    async def process_url(url: str):
        async with semaphore:
            try:
                # Validate URL first
                security_validator = SecurityValidator.from_unified_config()
                validated_url = security_validator.validate_url(url)

                doc_request = DocumentRequest(
                    url=validated_url,
                    collection=request.collection,
                )
                result = await add_document(doc_request)
                results["successful"].append(result)
            except Exception as e:
                results["failed"].append(
                    {
                        "url": url,
                        "error": str(e),
                    }
                )

    # Process all URLs concurrently
    await asyncio.gather(
        *[process_url(url) for url in request.urls],
        return_exceptions=True,
    )

    return results


# Project Management Tools
@mcp.tool()
async def create_project(request: ProjectRequest) -> dict[str, Any]:
    """
    Create a new documentation project.

    Projects allow grouping related documents with shared configuration
    and quality settings.
    """
    await service_manager.initialize()

    project_id = str(uuid4())
    project = {
        "id": project_id,
        "name": request.name,
        "description": request.description,
        "created_at": datetime.now(UTC).isoformat(),
        "quality_tier": request.quality_tier,
        "collection": f"project_{project_id}",
        "document_count": 0,
        "urls": [],
    }

    # Store project in both memory and persistent storage
    service_manager.projects[project_id] = project
    await service_manager.project_storage.save_project(project_id, project)

    # Create collection with quality-based config
    vector_size = 1536 if request.quality_tier == "premium" else 384
    enable_hybrid = request.quality_tier in ["balanced", "premium"]

    await service_manager.qdrant_service.create_collection(
        collection_name=project["collection"],
        vector_size=vector_size,
        distance="Cosine",
        sparse_vector_name="sparse" if enable_hybrid else None,
        enable_quantization=request.quality_tier != "premium",
    )

    # Add initial URLs if provided
    if request.urls:
        batch_request = BatchRequest(
            urls=request.urls,
            collection=project["collection"],
        )
        batch_result = await add_documents_batch(batch_request)
        project["urls"] = request.urls
        project["document_count"] = len(batch_result["successful"])

        # Update persistent storage
        await service_manager.project_storage.update_project(
            project_id,
            {"urls": project["urls"], "document_count": project["document_count"]},
        )

    return project


@mcp.tool()
async def list_projects() -> list[dict[str, Any]]:
    """
    List all documentation projects.

    Returns all projects with their metadata and statistics.
    """
    await service_manager.initialize()

    projects = []
    for project in service_manager.projects.values():
        # Get collection stats
        try:
            info = await service_manager.qdrant_service.get_collection_info(
                project["collection"]
            )
            project["vector_count"] = info.vectors_count
            project["indexed_count"] = info.indexed_vectors_count
        except Exception:
            project["vector_count"] = 0
            project["indexed_count"] = 0

        projects.append(project)

    return projects


@mcp.tool()
async def search_project(
    project_id: str,
    query: str,
    limit: int = 10,
    strategy: SearchStrategy = SearchStrategy.HYBRID,
) -> list[SearchResult]:
    """
    Search within a specific project.

    Uses project-specific quality settings and collection.
    """
    await service_manager.initialize()

    project = service_manager.projects.get(project_id)
    if not project:
        raise ValueError(f"Project {project_id} not found")

    request = SearchRequest(
        query=query,
        collection=project["collection"],
        limit=limit,
        strategy=strategy,
        enable_reranking=project["quality_tier"] == "premium",
    )

    return await search_documents(request)


@mcp.tool()
async def update_project(
    project_id: str,
    name: str | None = None,
    description: str | None = None,
) -> dict[str, Any]:
    """
    Update project metadata.

    Updates the name and/or description of an existing project.
    """
    await service_manager.initialize()

    project = service_manager.projects.get(project_id)
    if not project:
        raise ValueError(f"Project {project_id} not found")

    updates = {}
    if name is not None:
        project["name"] = name
        updates["name"] = name
    if description is not None:
        project["description"] = description
        updates["description"] = description

    if updates:
        await service_manager.project_storage.update_project(project_id, updates)

    return project


@mcp.tool()
async def delete_project(
    project_id: str, delete_collection: bool = True
) -> dict[str, str]:
    """
    Delete a project and optionally its collection.

    Args:
        project_id: Project ID to delete
        delete_collection: Whether to delete the associated Qdrant collection

    Returns:
        Status message
    """
    await service_manager.initialize()

    project = service_manager.projects.get(project_id)
    if not project:
        raise ValueError(f"Project {project_id} not found")

    # Delete collection if requested
    if delete_collection:
        try:
            await service_manager.qdrant_service.delete_collection(
                project["collection"]
            )
        except Exception as e:
            logger.warning(f"Failed to delete collection {project['collection']}: {e}")

    # Remove from in-memory storage
    del service_manager.projects[project_id]

    # Remove from persistent storage
    await service_manager.project_storage.delete_project(project_id)

    return {
        "status": "deleted",
        "project_id": project_id,
        "collection_deleted": str(delete_collection),
    }


# Collection Management Tools
@mcp.tool()
async def list_collections() -> list[dict[str, Any]]:
    """
    List all vector database collections.

    Returns collection names with their configuration and statistics.
    """
    await service_manager.initialize()

    try:
        # Get collections with details using service method
        collection_info = (
            await service_manager.qdrant_service.list_collections_details()
        )
        return collection_info
    except Exception as e:
        logger.error(f"Failed to list collections: {e}")
        raise


@mcp.tool()
async def delete_collection(name: str) -> dict[str, str]:
    """
    Delete a vector database collection.

    Permanently removes the collection and all its data.
    """
    await service_manager.initialize()

    try:
        await service_manager.qdrant_service.delete_collection(name)

        # Remove from projects if it's a project collection
        for project_id, project in list(service_manager.projects.items()):
            if project["collection"] == name:
                del service_manager.projects[project_id]

        return {"status": "success", "message": f"Collection '{name}' deleted"}
    except Exception as e:
        logger.error(f"Failed to delete collection: {e}")
        raise


@mcp.tool()
async def optimize_collection(name: str) -> dict[str, Any]:
    """
    Optimize a collection for better performance.

    Runs indexing optimization and updates collection configuration
    for improved search performance.
    """
    await service_manager.initialize()

    try:
        # Get current collection info
        info = await service_manager.qdrant_service.get_collection_info(name)

        # Trigger collection optimization
        await service_manager.qdrant_service.trigger_collection_optimization(name)

        # Get updated info
        new_info = await service_manager.qdrant_service.get_collection_info(name)

        return {
            "collection": name,
            "status": "optimized",
            "vectors_before": info.get("vectors_count", 0),
            "vectors_after": new_info.get("vectors_count", 0),
            "indexed_before": info.get("points_count", 0),
            "indexed_after": new_info.get("points_count", 0),
        }

    except Exception as e:
        logger.error(f"Failed to optimize collection: {e}")
        raise


# Payload Indexing Management Tools for Issue #56


@mcp.tool()
async def create_payload_indexes(collection_name: str, ctx: Context) -> dict[str, Any]:
    """
    Create payload indexes on a collection for 10-100x faster filtering.

    Creates indexes on key metadata fields like site_name, embedding_model,
    title, word_count, scraped_at, etc. for dramatic performance improvements.
    """
    await service_manager.initialize()

    # Generate request ID for tracking
    request_id = str(uuid4())
    await ctx.info(
        f"Creating payload indexes for collection: {collection_name} (Request: {request_id})"
    )

    try:
        # Validate collection name
        security_validator = SecurityValidator.from_unified_config()
        collection_name = security_validator.validate_collection_name(collection_name)

        # Check if collection exists
        collections = await service_manager.qdrant_service.list_collections()
        if collection_name not in collections:
            raise ValueError(f"Collection '{collection_name}' not found")

        # Create payload indexes
        await service_manager.qdrant_service.create_payload_indexes(collection_name)

        # Get index statistics
        stats = await service_manager.qdrant_service.get_payload_index_stats(
            collection_name
        )

        await ctx.info(
            f"Successfully created {stats['indexed_fields_count']} payload indexes for {collection_name}"
        )

        return {
            "collection_name": collection_name,
            "status": "success",
            "indexes_created": stats["indexed_fields_count"],
            "indexed_fields": stats["indexed_fields"],
            "total_points": stats["total_points"],
            "request_id": request_id,
        }

    except Exception as e:
        await ctx.error(f"Failed to create payload indexes for {collection_name}: {e}")
        logger.error(f"Failed to create payload indexes: {e}")
        raise


@mcp.tool()
async def list_payload_indexes(collection_name: str, ctx: Context) -> dict[str, Any]:
    """
    List all payload indexes in a collection.

    Shows which fields are indexed and their types for performance monitoring.
    """
    await service_manager.initialize()

    await ctx.info(f"Listing payload indexes for collection: {collection_name}")

    try:
        # Validate collection name
        security_validator = SecurityValidator.from_unified_config()
        collection_name = security_validator.validate_collection_name(collection_name)

        # Get index statistics
        stats = await service_manager.qdrant_service.get_payload_index_stats(
            collection_name
        )

        await ctx.info(
            f"Found {stats['indexed_fields_count']} indexed fields in {collection_name}"
        )

        return stats

    except Exception as e:
        await ctx.error(f"Failed to list payload indexes for {collection_name}: {e}")
        logger.error(f"Failed to list payload indexes: {e}")
        raise


@mcp.tool()
async def reindex_collection(collection_name: str, ctx: Context) -> dict[str, Any]:
    """
    Reindex all payload fields in a collection.

    Drops existing indexes and recreates them. Useful after bulk updates
    or when index performance degrades.
    """
    await service_manager.initialize()

    # Generate request ID for tracking
    request_id = str(uuid4())
    await ctx.info(
        f"Starting full reindex for collection: {collection_name} (Request: {request_id})"
    )

    try:
        # Validate collection name
        security_validator = SecurityValidator.from_unified_config()
        collection_name = security_validator.validate_collection_name(collection_name)

        # Get stats before reindexing
        stats_before = await service_manager.qdrant_service.get_payload_index_stats(
            collection_name
        )

        # Perform reindexing
        await service_manager.qdrant_service.reindex_collection(collection_name)

        # Get stats after reindexing
        stats_after = await service_manager.qdrant_service.get_payload_index_stats(
            collection_name
        )

        await ctx.info(f"Successfully reindexed collection: {collection_name}")

        return {
            "collection_name": collection_name,
            "status": "success",
            "indexes_before": stats_before["indexed_fields_count"],
            "indexes_after": stats_after["indexed_fields_count"],
            "indexed_fields": stats_after["indexed_fields"],
            "total_points": stats_after["total_points"],
            "request_id": request_id,
        }

    except Exception as e:
        await ctx.error(f"Failed to reindex collection {collection_name}: {e}")
        logger.error(f"Failed to reindex collection: {e}")
        raise


@mcp.tool()
async def benchmark_filtered_search(
    collection_name: str,
    test_filters: dict[str, Any],
    query: str = "documentation search test",
    ctx: Context = None,
) -> dict[str, Any]:
    """
    Benchmark filtered search performance to demonstrate indexing improvements.

    Compares performance of filtered searches and provides metrics on the
    effectiveness of payload indexing.
    """
    await service_manager.initialize()

    if ctx:
        await ctx.info(f"Benchmarking filtered search on collection: {collection_name}")

    try:
        # Validate collection name and filters
        security_validator = SecurityValidator.from_unified_config()
        collection_name = security_validator.validate_collection_name(collection_name)
        query = security_validator.validate_query_string(query)

        # Generate embedding for test query
        embedding_result = await service_manager.embedding_manager.generate_embeddings(
            [query], generate_sparse=False
        )
        query_vector = embedding_result["embeddings"][0]

        # Run filtered search with timing
        import time

        start_time = time.time()

        results = await service_manager.qdrant_service.filtered_search(
            collection_name=collection_name,
            query_vector=query_vector,
            filters=test_filters,
            limit=10,
            search_accuracy="balanced",
        )

        search_time = (time.time() - start_time) * 1000  # Convert to milliseconds

        # Get collection and index stats
        collection_stats = await service_manager.qdrant_service.get_collection_info(
            collection_name
        )
        index_stats = await service_manager.qdrant_service.get_payload_index_stats(
            collection_name
        )

        if ctx:
            await ctx.info(
                f"Filtered search completed in {search_time:.2f}ms with {len(results)} results"
            )

        return {
            "collection_name": collection_name,
            "query": query,
            "filters_applied": test_filters,
            "search_time_ms": round(search_time, 2),
            "results_found": len(results),
            "total_points": collection_stats.get("points_count", 0),
            "indexed_fields": index_stats["indexed_fields"],
            "performance_estimate": "10-100x faster than unindexed"
            if index_stats["indexed_fields"]
            else "No indexes detected",
            "benchmark_timestamp": datetime.now(UTC).isoformat(),
        }

    except Exception as e:
        if ctx:
            await ctx.error(f"Failed to benchmark filtered search: {e}")
        logger.error(f"Failed to benchmark filtered search: {e}")
        raise


# Analytics and Monitoring Tools
@mcp.tool()
async def get_analytics(request: AnalyticsRequest) -> dict[str, Any]:
    """
    Get analytics and metrics for collections and operations.

    Provides performance metrics, cost analysis, and usage statistics.
    """
    await service_manager.initialize()

    analytics = {
        "timestamp": datetime.now(UTC).isoformat(),
        "collections": {},
        "cache_metrics": {},
        "performance": {},
        "costs": {},
    }

    # Get collection analytics
    if request.collection:
        collections = [request.collection]
    else:
        # Get collections using service method
        collections = await service_manager.qdrant_service.list_collections()

    for collection in collections:
        try:
            info = await service_manager.qdrant_service.get_collection_info(collection)
            analytics["collections"][collection] = {
                "vector_count": info.get("vectors_count", 0),
                "indexed_count": info.get("points_count", 0),
                "status": info.get("status", "unknown"),
            }
        except Exception as e:
            logger.warning(f"Failed to get analytics for {collection}: {e}")

    # Get cache metrics
    if request.include_performance:
        cache_stats = await service_manager.cache_manager.get_stats()
        analytics["cache_metrics"] = cache_stats

    # Estimate costs
    if request.include_costs:
        total_vectors = sum(
            c.get("vector_count", 0) for c in analytics["collections"].values()
        )

        # Rough cost estimates
        analytics["costs"] = {
            "storage_gb": total_vectors * 1536 * 4 / 1e9,  # 4 bytes per dimension
            "monthly_estimate": total_vectors * 0.00001,  # Rough estimate
            "embedding_calls": cache_stats.get("total_requests", 0),
            "cache_savings": cache_stats.get("hit_rate", 0) * 0.02,  # Saved API calls
        }

    return analytics


@mcp.tool()
async def get_system_health() -> dict[str, Any]:
    """
    Get system health and status information.

    Checks all services and returns their health status.
    """
    await service_manager.initialize()

    health = {
        "status": "healthy",
        "timestamp": datetime.now(UTC).isoformat(),
        "services": {},
    }

    # Check Qdrant
    try:
        # Get collections using service method
        collections = await service_manager.qdrant_service.list_collections()
        health["services"]["qdrant"] = {
            "status": "healthy",
            "collections": len(collections),
        }
    except Exception as e:
        health["services"]["qdrant"] = {
            "status": "unhealthy",
            "error": str(e),
        }
        health["status"] = "degraded"

    # Check embedding service
    try:
        provider_info = service_manager.embedding_manager.get_current_provider_info()
        health["services"]["embeddings"] = {
            "status": "healthy",
            "provider": provider_info.get("name", "unknown"),
        }
    except Exception as e:
        health["services"]["embeddings"] = {
            "status": "unhealthy",
            "error": str(e),
        }
        health["status"] = "degraded"

    # Check cache
    try:
        cache_stats = await service_manager.cache_manager.get_stats()
        health["services"]["cache"] = {
            "status": "healthy",
            "hit_rate": cache_stats.get("hit_rate", 0),
        }
    except Exception as e:
        health["services"]["cache"] = {
            "status": "unhealthy",
            "error": str(e),
        }

    return health


# Cache Management Tools
@mcp.tool()
async def clear_cache(pattern: str | None = None) -> dict[str, Any]:
    """
    Clear cache entries.

    Clears all cache entries or those matching a specific pattern.
    """
    await service_manager.initialize()

    try:
        if pattern:
            cleared = await service_manager.cache_manager.clear_pattern(pattern)
        else:
            cleared = await service_manager.cache_manager.clear_all()

        return {
            "status": "success",
            "cleared_count": cleared,
            "pattern": pattern,
        }

    except Exception as e:
        logger.error(f"Failed to clear cache: {e}")
        raise


@mcp.tool()
async def get_cache_stats() -> dict[str, Any]:
    """
    Get cache statistics and metrics.

    Returns hit rate, size, and performance metrics for the cache.
    """
    await service_manager.initialize()

    return await service_manager.cache_manager.get_stats()


# Utility Tools
@mcp.tool()
async def estimate_costs(
    text_count: int, average_length: int = 1000, include_storage: bool = True
) -> dict[str, Any]:
    """
    Estimate costs for processing documents.

    Calculates embedding generation and storage costs based on
    current pricing models.
    """
    # Estimate tokens (rough approximation)
    total_chars = text_count * average_length
    estimated_tokens = total_chars / 4  # Rough char-to-token ratio

    # Calculate costs
    embedding_cost = estimated_tokens * 0.00002 / 1000  # $0.02 per 1M tokens

    costs = {
        "text_count": text_count,
        "estimated_tokens": int(estimated_tokens),
        "embedding_cost": round(embedding_cost, 4),
        "provider": "openai/text-embedding-3-small",
    }

    if include_storage:
        # Assume 1536 dimensions, 4 bytes per float
        storage_bytes = text_count * 1536 * 4
        storage_gb = storage_bytes / 1e9
        storage_cost = storage_gb * 0.20  # $0.20 per GB/month estimate

        costs["storage_gb"] = round(storage_gb, 4)
        costs["storage_cost_monthly"] = round(storage_cost, 4)
        costs["total_cost"] = round(embedding_cost + storage_cost, 4)

    return costs


@mcp.tool()
async def validate_configuration() -> dict[str, Any]:
    """
    Validate system configuration and API keys.

    Checks all required configuration and returns validation results.
    """
    config_status = {
        "valid": True,
        "errors": [],
        "warnings": [],
        "config": {},
    }

    # Check Qdrant
    config_status["config"]["qdrant_url"] = service_manager.config.qdrant_url

    # Check API keys
    if not service_manager.config.openai_api_key:
        config_status["warnings"].append("OpenAI API key not configured")
    else:
        config_status["config"]["openai"] = "configured"

    if not service_manager.config.firecrawl_api_key:
        config_status["warnings"].append("Firecrawl API key not configured")
    else:
        config_status["config"]["firecrawl"] = "configured"

    # Check cache configuration
    config_status["config"]["cache"] = {
        "l1_enabled": True,
        "l1_max_items": service_manager.config.cache_config.max_items,
        "l2_enabled": service_manager.config.redis_url is not None,
    }

    # Determine overall validity
    if config_status["errors"]:
        config_status["valid"] = False

    return config_status


# Deployment and Alias Management Tools
@mcp.tool()
async def search_with_alias(
    query: str,
    alias: str = "documentation",
    limit: int = 10,
    strategy: SearchStrategy = SearchStrategy.HYBRID,
    enable_reranking: bool = False,
    ctx: Context = None,
) -> list[SearchResult]:
    """
    Search using collection alias for zero-downtime updates.

    Aliases allow instant switching between collection versions without
    affecting search availability.
    """
    await service_manager.initialize()

    # Get actual collection from alias
    collection = await service_manager.alias_manager.get_collection_for_alias(alias)
    if not collection:
        raise ValueError(f"Alias {alias} not found")

    # Perform search on actual collection
    request = SearchRequest(
        query=query,
        collection=collection,
        limit=limit,
        strategy=strategy,
        enable_reranking=enable_reranking,
    )

    return await search_documents(request, ctx)


@mcp.tool()
async def list_aliases() -> dict[str, str]:
    """
    List all collection aliases and their targets.

    Returns a mapping of alias names to collection names.
    """
    await service_manager.initialize()

    return await service_manager.alias_manager.list_aliases()


@mcp.tool()
async def create_alias(
    alias_name: str,
    collection_name: str,
    force: bool = False,
) -> dict[str, Any]:
    """
    Create or update an alias to point to a collection.

    Args:
        alias_name: Name of the alias
        collection_name: Collection to point to
        force: If True, overwrite existing alias

    Returns:
        Status information
    """
    await service_manager.initialize()

    success = await service_manager.alias_manager.create_alias(
        alias_name=alias_name,
        collection_name=collection_name,
        force=force,
    )

    return {
        "success": success,
        "alias": alias_name,
        "collection": collection_name,
    }


@mcp.tool()
async def deploy_new_index(
    alias: str,
    source: str,
    validation_queries: list[str] | None = None,
    rollback_on_failure: bool = True,
    ctx: Context = None,
) -> dict[str, Any]:
    """
    Deploy new index version with zero downtime using blue-green deployment.

    Creates a new collection, populates it, validates, and atomically switches
    the alias. Includes automatic rollback on failure.

    Args:
        alias: Alias to update
        source: Data source (e.g., "collection:docs_v1" or "crawl:new")
        validation_queries: Queries to validate deployment
        rollback_on_failure: Whether to rollback on validation failure

    Returns:
        Deployment status with details
    """
    await service_manager.initialize()

    if ctx:
        await ctx.info(f"Starting blue-green deployment for alias {alias}")

    # Default validation queries if none provided
    if not validation_queries:
        validation_queries = [
            "python asyncio",
            "react hooks",
            "fastapi authentication",
        ]

    result = await service_manager.blue_green.deploy_new_version(
        alias_name=alias,
        data_source=source,
        validation_queries=validation_queries,
        rollback_on_failure=rollback_on_failure,
    )

    if ctx:
        await ctx.info(
            f"Deployment completed successfully. "
            f"Alias {alias} now points to {result['new_collection']}"
        )

    return result


@mcp.tool()
async def start_ab_test(
    experiment_name: str,
    control_collection: str,
    treatment_collection: str,
    traffic_split: float = 0.5,
    metrics: list[str] | None = None,
) -> dict[str, str]:
    """
    Start A/B test between two collections.

    Enables testing new embeddings, chunking strategies, or configurations
    on live traffic with automatic metrics collection.

    Args:
        experiment_name: Name of the experiment
        control_collection: Control (baseline) collection
        treatment_collection: Treatment (test) collection
        traffic_split: Percentage of traffic to treatment (0-1)
        metrics: Metrics to track (default: latency, relevance, clicks)

    Returns:
        Experiment ID and status
    """
    await service_manager.initialize()

    experiment_id = await service_manager.ab_testing.create_experiment(
        experiment_name=experiment_name,
        control_collection=control_collection,
        treatment_collection=treatment_collection,
        traffic_split=traffic_split,
        metrics_to_track=metrics,
    )

    return {
        "experiment_id": experiment_id,
        "status": "started",
        "control": control_collection,
        "treatment": treatment_collection,
        "traffic_split": traffic_split,
    }


@mcp.tool()
async def analyze_ab_test(experiment_id: str) -> dict[str, Any]:
    """
    Analyze results of an A/B test experiment.

    Returns statistical analysis including p-values, confidence intervals,
    and improvement metrics for each tracked metric.

    Args:
        experiment_id: ID of the experiment to analyze

    Returns:
        Detailed analysis results
    """
    await service_manager.initialize()

    return service_manager.ab_testing.analyze_experiment(experiment_id)


@mcp.tool()
async def start_canary_deployment(
    alias: str,
    new_collection: str,
    stages: list[dict] | None = None,
    auto_rollback: bool = True,
    ctx: Context = None,
) -> dict[str, Any]:
    """
    Start canary deployment with gradual traffic rollout.

    Progressively shifts traffic to new collection with health monitoring
    and automatic rollback on errors.

    Args:
        alias: Alias to update
        new_collection: New collection to deploy
        stages: Custom deployment stages (default: 5% -> 25% -> 50% -> 100%)
        auto_rollback: Whether to auto-rollback on failure

    Returns:
        Deployment ID and status
    """
    await service_manager.initialize()

    if ctx:
        await ctx.info(f"Starting canary deployment for alias {alias}")

    # Validate stages if provided
    if stages:
        for i, stage in enumerate(stages):
            if not isinstance(stage, dict):
                raise ValueError(f"Stage {i} must be a dictionary")
            if "percentage" not in stage:
                raise ValueError(f"Stage {i} missing required 'percentage' field")
            if "duration_minutes" not in stage:
                raise ValueError(f"Stage {i} missing required 'duration_minutes' field")

            percentage = stage["percentage"]
            if (
                not isinstance(percentage, int | float)
                or percentage < 0
                or percentage > 100
            ):
                raise ValueError(f"Stage {i} percentage must be between 0 and 100")

            duration = stage["duration_minutes"]
            if not isinstance(duration, int | float) or duration <= 0:
                raise ValueError(f"Stage {i} duration_minutes must be positive")

    deployment_id = await service_manager.canary.start_canary(
        alias_name=alias,
        new_collection=new_collection,
        stages=stages,
        auto_rollback=auto_rollback,
    )

    if ctx:
        await ctx.info(f"Canary deployment started with ID: {deployment_id}")

    return {
        "deployment_id": deployment_id,
        "status": "started",
        "alias": alias,
        "new_collection": new_collection,
    }


@mcp.tool()
async def get_canary_status(deployment_id: str) -> dict[str, Any]:
    """
    Get current status of a canary deployment.

    Shows current stage, traffic percentage, and health metrics.

    Args:
        deployment_id: ID of the deployment

    Returns:
        Current deployment status
    """
    await service_manager.initialize()

    return await service_manager.canary.get_deployment_status(deployment_id)


@mcp.tool()
async def pause_canary(deployment_id: str) -> dict[str, str]:
    """
    Pause a canary deployment.

    Stops progression through stages but maintains current traffic split.

    Args:
        deployment_id: ID of the deployment to pause

    Returns:
        Status message
    """
    await service_manager.initialize()

    success = await service_manager.canary.pause_deployment(deployment_id)

    return {
        "status": "paused" if success else "failed",
        "deployment_id": deployment_id,
    }


@mcp.tool()
async def resume_canary(deployment_id: str) -> dict[str, str]:
    """
    Resume a paused canary deployment.

    Continues progression through remaining stages.

    Args:
        deployment_id: ID of the deployment to resume

    Returns:
        Status message
    """
    await service_manager.initialize()

    success = await service_manager.canary.resume_deployment(deployment_id)

    return {
        "status": "resumed" if success else "failed",
        "deployment_id": deployment_id,
    }


# Server lifecycle - FastMCP doesn't support lifecycle decorators
# Service initialization is handled within each tool via service_manager.initialize()


# Run the server
if __name__ == "__main__":
    mcp.run()
