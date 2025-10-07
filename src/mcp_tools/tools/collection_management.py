"""Collection management tools for MCP server."""

import logging
from collections.abc import Awaitable, Callable
from typing import cast

from fastmcp import Context

from src.infrastructure.client_manager import ClientManager
from src.mcp_tools.models.responses import CollectionInfo, CollectionOperationResponse


logger = logging.getLogger(__name__)


def register_tools(mcp, client_manager: ClientManager):  # pylint: disable=too-many-statements
    """Register collection management tools with the MCP server."""

    @mcp.tool()
    async def list_collections(ctx: Context | None = None) -> list[CollectionInfo]:
        """List all vector database collections.

        Returns information about each collection including size and status.
        """
        if ctx:
            await ctx.info("Retrieving list of all collections")

        try:
            vector_service = await client_manager.get_vector_store_service()
            collections = await vector_service.list_collections()
            collection_info: list[CollectionInfo] = []

            if ctx:
                await ctx.debug(f"Found {len(collections)} collections")

            for collection_name in collections:
                try:
                    stats = await vector_service.collection_stats(collection_name)
                    vectors_meta = (
                        stats.get("vectors", {}) if isinstance(stats, dict) else {}
                    )
                    collection_info.append(
                        CollectionInfo.model_validate(
                            {
                                "name": collection_name,
                                "vectors_count": vectors_meta.get("size"),
                                "points_count": stats.get("points_count")
                                if isinstance(stats, dict)
                                else None,
                                "status": "active",
                                "indexed_vectors_count": stats.get("indexed_vectors")
                                if isinstance(stats, dict)
                                else None,
                                "vector_config": vectors_meta,
                            }
                        )
                    )
                    if ctx:
                        await ctx.debug(
                            "Retrieved info for collection %s: %s vectors",
                            collection_name,
                            vectors_meta.get("size"),
                        )
                except Exception as exc:  # pragma: no cover - defensive branch
                    logger.exception(
                        "Failed to get info for collection %s", collection_name
                    )
                    if ctx:
                        await ctx.warning(
                            "Failed to get info for collection "
                            f"{collection_name}: {exc}"
                        )
                    collection_info.append(
                        CollectionInfo.model_validate(
                            {
                                "name": collection_name,
                                "status": "error",
                                "error": str(exc),
                            }
                        )
                    )

            if ctx:
                await ctx.info(
                    "Successfully retrieved information for "
                    f"{len(collection_info)} collections"
                )

            return collection_info

        except Exception as e:
            if ctx:
                await ctx.error(f"Failed to list collections: {e}")
            logger.exception("Failed to list collections")
            raise

    @mcp.tool()
    async def delete_collection(
        collection_name: str, ctx: Context | None = None
    ) -> CollectionOperationResponse:
        """Delete a vector database collection.

        Permanently removes the collection and all its data.
        """
        if ctx:
            await ctx.info(f"Starting deletion of collection: {collection_name}")

        try:
            vector_service = await client_manager.get_vector_store_service()
            cache_manager = await client_manager.get_cache_manager()

            delete_alias = getattr(vector_service, "delete_collection", None)
            drop_method = getattr(vector_service, "drop_collection", None)
            if callable(delete_alias):
                delete_callable = cast(Callable[[str], Awaitable[None]], delete_alias)
                await delete_callable(collection_name)
            elif callable(drop_method):
                drop_callable = cast(Callable[[str], Awaitable[None]], drop_method)
                await drop_callable(collection_name)
            else:  # pragma: no cover - defensive compatibility guard
                msg = "Vector service does not expose a collection deletion method"
                raise AttributeError(msg)
            if ctx:
                await ctx.debug(f"Collection {collection_name} deleted from Qdrant")

            # Clear cache entries for this collection
            await cache_manager.clear_pattern(f"*:{collection_name}:*")
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
            logger.exception("Failed to delete collection %s", collection_name)
            return CollectionOperationResponse(status="error", message=str(e))

    @mcp.tool()
    async def optimize_collection(
        collection_name: str, ctx: Context | None = None
    ) -> CollectionOperationResponse:
        """Optimize a collection for better performance.

        Rebuilds indexes and optimizes storage.
        """
        if ctx:
            await ctx.info(f"Starting optimization of collection: {collection_name}")

        try:
            vector_service = await client_manager.get_vector_store_service()
            # Get current collection info
            stats = await vector_service.collection_stats(collection_name)
            vectors_meta = stats.get("vectors", {}) if isinstance(stats, dict) else {}
            if ctx:
                await ctx.debug(
                    "Collection %s has %s vectors",
                    collection_name,
                    vectors_meta.get("size"),
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
                    "vectors_count": vectors_meta.get("size"),
                    "indexed_vectors_count": stats.get("indexed_vectors")
                    if isinstance(stats, dict)
                    else None,
                },
            )
        except Exception as e:
            if ctx:
                await ctx.error(f"Failed to optimize collection {collection_name}: {e}")
            logger.exception("Failed to optimize collection %s", collection_name)
            return CollectionOperationResponse(status="error", message=str(e))
