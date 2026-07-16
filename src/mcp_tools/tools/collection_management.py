"""Collection management tools for MCP server."""

from __future__ import annotations

import logging

from fastmcp import Context

from src.mcp_tools.models.responses import CollectionInfo, CollectionOperationResponse
from src.services.cache.manager import CacheManager
from src.services.vector_db.service import VectorStoreService, dense_vector_config


logger = logging.getLogger(__name__)


def register_tools(  # pylint: disable=too-many-statements
    mcp,
    *,
    vector_service: VectorStoreService,
    cache_manager: CacheManager,
) -> None:
    """Register collection management tools with the MCP server."""

    @mcp.tool()
    async def list_collections(ctx: Context | None = None) -> list[CollectionInfo]:
        """List all vector database collections.

        Returns information about each collection including size and status.
        """
        if ctx:
            await ctx.info("Retrieving list of all collections")

        try:
            service = vector_service
            collections = await service.list_collections()
            collection_info: list[CollectionInfo] = []

            if ctx:
                await ctx.debug(f"Found {len(collections)} collections")

            for collection_name in collections:
                try:
                    stats = await service.collection_stats(collection_name)
                    vectors_meta = dense_vector_config(stats)
                    collection_info.append(
                        CollectionInfo.model_validate(
                            {
                                "name": collection_name,
                                "vectors_count": stats.get("points_count"),
                                "points_count": stats.get("points_count"),
                                "status": "active",
                                "vector_dimension": vectors_meta.get("size"),
                                "vector_config": vectors_meta,
                            }
                        )
                    )
                    if ctx:
                        await ctx.debug(
                            "Retrieved info for collection %s: %s vectors",
                            collection_name,
                            stats.get("points_count"),
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
            service = vector_service
            cache = cache_manager

            await service.drop_collection(collection_name)
            if ctx:
                await ctx.debug(
                    "Collection %s deleted from vector store", collection_name
                )

            # Clear cache entries for this collection
            await cache.clear_pattern(f"*:{collection_name}:*")
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
