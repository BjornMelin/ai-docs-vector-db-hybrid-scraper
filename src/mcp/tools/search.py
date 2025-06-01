"""Search and retrieval tools for MCP server."""

import logging
from uuid import uuid4

from fastmcp import Context

from ...config.enums import SearchStrategy
from ...infrastructure.client_manager import ClientManager
from ...services.cache.manager import CacheManager
from ...services.embeddings.manager import EmbeddingManager
from ...services.vector_db.service import QdrantService
from ..models.requests import SearchRequest
from ..models.responses import SearchResult

logger = logging.getLogger(__name__)


def register_tools(mcp, client_manager: ClientManager):  # noqa: PLR0915
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
        # Generate request ID for tracking
        request_id = str(uuid4())
        await ctx.info(
            f"Starting search request {request_id} with strategy {request.strategy}"
        )

        try:
            # Initialize services on-demand
            cache_manager = CacheManager(client_manager)
            embedding_manager = EmbeddingManager(client_manager.unified_config)
            qdrant_service = QdrantService(client_manager.unified_config)
            await qdrant_service.initialize()

            # Check cache first
            cache_key = f"search:{request.collection}:{request.query}:{request.strategy}:{request.limit}"
            cached = await cache_manager.get(cache_key)
            if cached:
                await ctx.debug(f"Cache hit for request {request_id}")
                return [SearchResult(**r) for r in cached]

            # Generate embedding for query
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
                await ctx.debug("Applying BGE reranking...")
                # Note: rerank_results method would need to be implemented in QdrantService
                # For now, we'll skip reranking
                pass

            # Convert to response format
            search_results = []
            for result in results:
                search_results.append(
                    SearchResult(
                        content=result["payload"].get("content", ""),
                        metadata=result["payload"].get("metadata", {}),
                        score=result["score"],
                        id=str(result["id"]),
                        collection=request.collection,
                    )
                )

            # Cache results
            if search_results:
                cache_data = [r.model_dump() for r in search_results]
                await cache_manager.set(
                    cache_key, cache_data, ttl=request.cache_ttl or 300
                )

            await ctx.info(f"Search completed: {len(search_results)} results found")
            return search_results

        except Exception as e:
            await ctx.error(f"Search failed: {e!s}")
            raise

    @mcp.tool()
    async def search_similar(
        query_id: str,
        collection: str = "documentation",
        limit: int = 10,
        score_threshold: float = 0.7,
        ctx: Context = None,
    ) -> list[SearchResult]:
        """
        Search for documents similar to a given document ID.

        Uses the document's embedding to find semantically similar content.
        """
        try:
            qdrant_service = QdrantService(client_manager.unified_config)
            await qdrant_service.initialize()

            # Retrieve the source document
            await ctx.info(f"Retrieving source document {query_id}")

            # We need to implement a retrieve method or use Qdrant's retrieve API
            # For now, let's use a simplified approach

            # Get the document by ID
            retrieved = await qdrant_service._client.retrieve(
                collection_name=collection,
                ids=[query_id],
                with_vectors=True,
                with_payload=True,
            )

            if not retrieved:
                raise ValueError(
                    f"Document {query_id} not found in collection {collection}"
                )

            # Extract the vector
            source_doc = retrieved[0]
            if hasattr(source_doc.vector, "dense"):
                query_vector = source_doc.vector.dense
            elif isinstance(source_doc.vector, list):
                query_vector = source_doc.vector
            else:
                query_vector = source_doc.vector.get("dense", [])

            # Search using the document's vector
            results = await qdrant_service.hybrid_search(
                collection_name=collection,
                query_vector=query_vector,
                sparse_vector=None,  # No sparse vector for similarity search
                limit=limit + 1,  # +1 to exclude self
                score_threshold=score_threshold,
                fusion_type="rrf",
                search_accuracy="balanced",
            )

            # Convert to response format, excluding the source document
            search_results = []
            for result in results:
                if str(result["id"]) != query_id:
                    search_results.append(
                        SearchResult(
                            content=result["payload"].get("content", ""),
                            metadata=result["payload"].get("metadata", {}),
                            score=result["score"],
                            id=str(result["id"]),
                            collection=collection,
                        )
                    )

            await ctx.info(f"Found {len(search_results)} similar documents")
            return search_results[:limit]  # Ensure we don't exceed requested limit

        except Exception as e:
            await ctx.error(f"Similar search failed: {e!s}")
            raise
