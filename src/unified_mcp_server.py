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
from pydantic import BaseModel
from pydantic import Field

# Handle both module and script imports
try:
    from chunking import ChunkingConfig
    from chunking import chunk_content
    from config.enums import ChunkingStrategy
    from config.enums import SearchStrategy
    from services.base import BaseService
    from services.cache.manager import CacheManager
    from services.config import APIConfig
    from services.crawling.manager import CrawlManager
    from services.embeddings.manager import EmbeddingManager
    from services.qdrant_service import QdrantService
except ImportError:
    from .chunking import ChunkingConfig
    from .chunking import chunk_content
    from .config.enums import ChunkingStrategy
    from .config.enums import SearchStrategy
    from .services.base import BaseService
    from .services.cache.manager import CacheManager
    from .services.config import APIConfig
    from .services.crawling.manager import CrawlManager
    from .services.embeddings.manager import EmbeddingManager
    from .services.qdrant_service import QdrantService

# Initialize logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Initialize FastMCP server
mcp = FastMCP("ai-docs-vector-db-unified")


# Request/Response Models
class SearchRequest(BaseModel):
    """Search request with advanced options"""

    query: str = Field(..., description="Search query")
    collection: str = Field(default="documentation", description="Collection to search")
    limit: int = Field(default=10, ge=1, le=100, description="Number of results")
    strategy: SearchStrategy = Field(
        default=SearchStrategy.HYBRID, description="Search strategy"
    )
    enable_reranking: bool = Field(default=True, description="Enable BGE reranking")
    include_metadata: bool = Field(
        default=True, description="Include metadata in results"
    )
    filters: dict[str, Any] | None = Field(default=None, description="Metadata filters")


class SearchResult(BaseModel):
    """Search result with metadata"""

    id: str
    content: str
    score: float
    url: str | None = None
    title: str | None = None
    metadata: dict[str, Any] | None = None


class EmbeddingRequest(BaseModel):
    """Embedding generation request"""

    texts: list[str] = Field(..., description="Texts to embed")
    model: str | None = Field(default=None, description="Specific model to use")
    batch_size: int = Field(default=32, ge=1, le=100, description="Batch size")


class DocumentRequest(BaseModel):
    """Document processing request"""

    url: str = Field(..., description="URL to process")
    collection: str = Field(default="documentation", description="Target collection")
    chunking_strategy: ChunkingStrategy = Field(
        default=ChunkingStrategy.ENHANCED, description="Chunking strategy"
    )
    chunk_size: int = Field(default=1600, ge=100, le=4000, description="Chunk size")
    chunk_overlap: int = Field(default=200, ge=0, le=500, description="Chunk overlap")


class ProjectRequest(BaseModel):
    """Project management request"""

    name: str = Field(..., description="Project name")
    description: str | None = Field(default=None, description="Project description")
    urls: list[str] = Field(default_factory=list, description="Initial URLs")
    quality_tier: str = Field(default="balanced", description="Quality tier")


class BatchRequest(BaseModel):
    """Batch processing request"""

    urls: list[str] = Field(..., description="URLs to process")
    collection: str = Field(default="documentation", description="Target collection")
    parallel_limit: int = Field(default=5, ge=1, le=20, description="Parallel limit")


class AnalyticsRequest(BaseModel):
    """Analytics request"""

    collection: str | None = Field(default=None, description="Specific collection")
    include_performance: bool = Field(
        default=True, description="Include performance metrics"
    )
    include_costs: bool = Field(default=True, description="Include cost analysis")


# Service Manager
class UnifiedServiceManager(BaseService):
    """Manages all services for the unified MCP server"""

    def __init__(self):
        super().__init__()
        self.config = APIConfig()
        self.embedding_manager: EmbeddingManager | None = None
        self.crawl_manager: CrawlManager | None = None
        self.qdrant_service: QdrantService | None = None
        self.cache_manager: CacheManager | None = None
        self.projects: dict[str, dict[str, Any]] = {}
        self._initialized = False

    async def initialize(self):
        """Initialize all services"""
        if self._initialized:
            return

        try:
            # Initialize services
            self.embedding_manager = EmbeddingManager(self.config)
            self.crawl_manager = CrawlManager(self.config)
            self.qdrant_service = QdrantService(url=self.config.qdrant_url)
            self.cache_manager = CacheManager(self.config)

            # Initialize each service
            await self.embedding_manager.initialize()
            await self.crawl_manager.initialize()
            await self.qdrant_service.initialize()
            await self.cache_manager.initialize()

            self._initialized = True
            logger.info("All services initialized successfully")

        except Exception as e:
            logger.error(f"Failed to initialize services: {e}")
            raise

    async def cleanup(self):
        """Cleanup all services"""
        if self.embedding_manager:
            await self.embedding_manager.cleanup()
        if self.crawl_manager:
            await self.crawl_manager.cleanup()
        if self.qdrant_service:
            await self.qdrant_service.cleanup()
        if self.cache_manager:
            await self.cache_manager.cleanup()

        self._initialized = False
        logger.info("All services cleaned up")


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
        # Check cache first
        cache_key = f"search:{request.collection}:{request.query}:{request.strategy}:{request.limit}"
        cached = await service_manager.cache_manager.get(cache_key)
        if cached:
            await ctx.debug(f"Cache hit for request {request_id}")
            return [SearchResult(**r) for r in cached]

        # Generate embedding for query
        await ctx.debug(f"Generating embeddings for query: {request.query[:50]}...")
        query_embeddings = await service_manager.embedding_manager.generate_embeddings(
            [request.query]
        )
        query_vector = query_embeddings[0]
        await ctx.debug(f"Generated embedding with dimension {len(query_vector)}")

        # Perform search based on strategy
        if request.strategy == SearchStrategy.HYBRID:
            # For hybrid search, we need both dense and sparse vectors
            # For now, we'll use dense only (sparse would require SPLADE encoding)
            results = await service_manager.qdrant_service.hybrid_search(
                collection_name=request.collection,
                query_vector=query_vector,
                sparse_vector=None,  # TODO: Add sparse vector generation
                limit=request.limit * 3 if request.enable_reranking else request.limit,
                score_threshold=0.0,
                fusion_type="rrf",
            )
        else:
            # For dense search, use direct vector search
            # Since QdrantService doesn't have a simple search method, use hybrid_search with dense only
            results = await service_manager.qdrant_service.hybrid_search(
                collection_name=request.collection,
                query_vector=query_vector,
                sparse_vector=None,
                limit=request.limit * 3 if request.enable_reranking else request.limit,
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
            # TODO: Implement BGE reranking
            # For now, just limit to requested number
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
            strategy=request.chunking_strategy,
            chunk_size=request.chunk_size,
            chunk_overlap=request.chunk_overlap,
        )

        # Chunk the document
        await ctx.debug(
            f"Chunking document {doc_id} with strategy {request.chunking_strategy}"
        )
        chunks = chunk_content(
            crawl_result.markdown,
            crawl_result.metadata,
            chunk_config,
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
            if request.chunking_strategy != ChunkingStrategy.BASIC
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
            "chunking_strategy": request.chunking_strategy.value,
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
    semaphore = asyncio.Semaphore(request.parallel_limit)

    async def process_url(url: str):
        async with semaphore:
            try:
                doc_request = DocumentRequest(
                    url=url,
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

    # Store project
    service_manager.projects[project_id] = project

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


# Server lifecycle - FastMCP doesn't support lifecycle decorators
# Service initialization is handled within each tool via service_manager.initialize()


# Run the server
if __name__ == "__main__":
    mcp.run()
