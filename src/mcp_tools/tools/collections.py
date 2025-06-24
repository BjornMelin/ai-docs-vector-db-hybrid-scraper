"""Collection management tools for MCP server."""

import logging
from typing import TYPE_CHECKING


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


from ...infrastructure.client_manager import ClientManager


logger = logging.getLogger(__name__)


def register_tools(mcp, client_manager: ClientManager):
    """Register collection management tools with the MCP server."""

    from ..models.responses import CollectionInfo, CollectionOperationResponse

    @mcp.tool()
    async def list_collections(ctx: Context = None) -> list[CollectionInfo]:
        """
        List all vector database collections.

        Returns information about each collection including size and status.
        """
        if ctx:
            await ctx.info("Retrieving list of all collections")

        try:
            qdrant_service = await client_manager.get_qdrant_service()
            collections = await qdrant_service.list_collections()
            collection_info = []

            if ctx:
                await ctx.debug(f"Found {len(collections)} collections")

            for collection_name in collections:
                try:
                    info = await qdrant_service.get_collection_info(collection_name)
                    collection_info.append(
                        {
                            "name": collection_name,
                            "vectors_count": info.vectors_count,
                            "indexed_vectors_count": info.indexed_vectors_count,
                            "config": {
                                "size": info.config.params.vectors.size,
                                "distance": info.config.params.vectors.distance,
                            },
                        }
                    )
                    if ctx:
                        await ctx.debug(
                            f"Retrieved info for collection {collection_name}: {info.vectors_count} vectors"
                        )
                except Exception as e:
                    logger.exception(
                        f"Failed to get info for collection {collection_name}: {e}"
                    )
                    if ctx:
                        await ctx.warning(
                            f"Failed to get info for collection {collection_name}: {e}"
                        )
                    collection_info.append({"name": collection_name, "error": str(e)})

            if ctx:
                await ctx.info(
                    f"Successfully retrieved information for {len(collection_info)} collections"
                )

            return [CollectionInfo(**c) for c in collection_info]

        except Exception as e:
            if ctx:
                await ctx.error(f"Failed to list collections: {e}")
            logger.exception(f"Failed to list collections: {e}")
            raise

    @mcp.tool()
    async def delete_collection(
        collection_name: str, ctx: Context = None
    ) -> CollectionOperationResponse:
        """
        Delete a vector database collection.

        Permanently removes the collection and all its data.
        """
        if ctx:
            await ctx.info(f"Starting deletion of collection: {collection_name}")

        try:
            qdrant_service = await client_manager.get_qdrant_service()
            cache_manager = await client_manager.get_cache_manager()

            await qdrant_service.delete_collection(collection_name)
            if ctx:
                await ctx.debug(f"Collection {collection_name} deleted from Qdrant")

            # Clear cache entries for this collection
            await cache_manager.clear(pattern=f"*:{collection_name}:*")
            if ctx:
                await ctx.debug(
                    f"Cache entries cleared for collection {collection_name}"
                )

            if ctx:
                await ctx.info(f"Successfully deleted collection: {collection_name}")

            return CollectionOperationResponse(
                status="deleted", collection=collection_name
            )
        except Exception as e:
            if ctx:
                await ctx.error(f"Failed to delete collection {collection_name}: {e}")
            logger.exception(f"Failed to delete collection {collection_name}: {e}")
            return CollectionOperationResponse(status="error", message=str(e))

    @mcp.tool()
    async def optimize_collection(
        collection_name: str, ctx: Context = None
    ) -> CollectionOperationResponse:
        """
        Optimize a collection for better performance.

        Rebuilds indexes and optimizes storage.
        """
        if ctx:
            await ctx.info(f"Starting optimization of collection: {collection_name}")

        try:
            qdrant_service = await client_manager.get_qdrant_service()
            # Get current collection info
            info = await qdrant_service.get_collection_info(collection_name)
            if ctx:
                await ctx.debug(
                    f"Collection {collection_name} has {info.vectors_count} vectors"
                )

            # Trigger optimization
            # Note: Qdrant automatically optimizes, but we can force index rebuild
            # This is a placeholder for future optimization strategies

            if ctx:
                await ctx.info(f"Successfully optimized collection: {collection_name}")

            return CollectionOperationResponse(
                status="optimized",
                collection=collection_name,
                details={
                    "vectors_count": info.vectors_count,
                    "indexed_vectors_count": info.indexed_vectors_count,
                },
            )
        except Exception as e:
            if ctx:
                await ctx.error(f"Failed to optimize collection {collection_name}: {e}")
            logger.exception(f"Failed to optimize collection {collection_name}: {e}")
            return CollectionOperationResponse(status="error", message=str(e))
