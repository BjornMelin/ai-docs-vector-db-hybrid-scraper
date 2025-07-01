"""Search and retrieval tools for MCP server."""

import asyncio
import logging
from typing import TYPE_CHECKING

from ._search_utils import search_documents_core


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


from src.infrastructure.client_manager import ClientManager
from src.mcp_tools.models.requests import SearchRequest
from src.mcp_tools.models.responses import SearchResult


logger = logging.getLogger(__name__)


def register_tools(mcp, client_manager: ClientManager):
    """Register search tools with the MCP server."""

    @mcp.tool()
    async def search_documents(
        request: SearchRequest, ctx: Context
    ) -> list[SearchResult]:
        """Search documents with advanced hybrid search and reranking.

        Supports dense, sparse, and hybrid search strategies with optional
        BGE reranking for improved accuracy.
        """
        return await search_documents_core(request, client_manager, ctx)

    @mcp.tool()
    async def search_similar(
        query_id: str,
        collection: str = "documentation",
        limit: int = 10,
        score_threshold: float = 0.7,
        ctx: Context = None,
    ) -> list[SearchResult]:
        """Search for documents similar to a given document ID.

        Uses the document's embedding to find semantically similar content.
        """
        if ctx:
            await ctx.info(
                f"Starting similarity search for document {query_id} in collection {collection}"
            )

        try:
            qdrant_service = await client_manager.get_qdrant_service()

            # Retrieve the source document
            if ctx:
                await ctx.debug(f"Retrieving source document {query_id}")

            # Get the document by ID
            retrieved = await qdrant_service._client.retrieve(
                collection_name=collection,
                ids=[query_id],
                with_vectors=True,
                with_payload=True,
            )

            if not retrieved:
                if ctx:
                    await ctx.error(
                        f"Document {query_id} not found in collection {collection}"
                    )
                msg = f"Document {query_id} not found in collection {collection}"
                raise ValueError(msg)

            # Extract the vector
            source_doc = retrieved[0]
            if hasattr(source_doc.vector, "dense"):
                query_vector = source_doc.vector.dense
            elif isinstance(source_doc.vector, list):
                query_vector = source_doc.vector
            else:
                query_vector = source_doc.vector.get("dense", [])

            if ctx:
                await ctx.debug(f"Extracted vector with {len(query_vector)} dimensions")

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

            if ctx:
                await ctx.debug(f"Hybrid search returned {len(results)} results")

            # Convert to response format, excluding the source document
            search_results = [
                SearchResult(
                    id=str(result["id"]),
                    content=result["payload"].get("content", ""),
                    score=result["score"],
                    url=result["payload"].get("url"),
                    title=result["payload"].get("title"),
                    metadata=result["payload"],
                )
                for result in results
                if str(result["id"]) != query_id
            ]

            final_results = search_results[
                :limit
            ]  # Ensure we don't exceed requested limit
            if ctx:
                await ctx.info(
                    f"Found {len(final_results)} similar documents for {query_id}"
                )
            return final_results

        except (TimeoutError, OSError, PermissionError) as e:
            if ctx:
                await ctx.error("Similar search failed")
            logger.exception("Similar search failed")
            raise
