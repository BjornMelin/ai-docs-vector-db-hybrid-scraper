"""Search and retrieval tools for MCP server."""

import logging

from fastmcp import Context

from ...infrastructure.client_manager import ClientManager
from ..models.requests import SearchRequest
from ..models.responses import SearchResult

logger = logging.getLogger(__name__)


def register_tools(mcp, client_manager: ClientManager):
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
        from ._search_utils import search_documents_core

        return await search_documents_core(request, client_manager, ctx)

    @mcp.tool()
    async def search_similar(
        query_id: str,
        collection: str = "documentation",
        limit: int = 10,
        score_threshold: float = 0.7,
        ctx=None,
    ) -> list[SearchResult]:
        """
        Search for documents similar to a given document ID.

        Uses the document's embedding to find semantically similar content.
        """
        try:
            qdrant_service = await client_manager.get_qdrant_service()

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
