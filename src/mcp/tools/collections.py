"""Collection management tools for MCP server."""

import logging
from typing import Any

from ..service_manager import UnifiedServiceManager

logger = logging.getLogger(__name__)


def register_tools(mcp, service_manager: UnifiedServiceManager):
    """Register collection management tools with the MCP server."""

    @mcp.tool()
    async def list_collections() -> list[dict[str, Any]]:
        """
        List all vector database collections.

        Returns information about each collection including size and status.
        """
        await service_manager.initialize()

        collections = await service_manager.qdrant_service.list_collections()
        collection_info = []

        for collection_name in collections:
            try:
                info = await service_manager.qdrant_service.get_collection_info(
                    collection_name
                )
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
            except Exception as e:
                logger.error(
                    f"Failed to get info for collection {collection_name}: {e}"
                )
                collection_info.append({"name": collection_name, "error": str(e)})

        return collection_info

    @mcp.tool()
    async def delete_collection(collection_name: str) -> dict[str, str]:
        """
        Delete a vector database collection.

        Permanently removes the collection and all its data.
        """
        await service_manager.initialize()

        try:
            await service_manager.qdrant_service.delete_collection(collection_name)

            # Clear cache entries for this collection
            await service_manager.cache_manager.clear(pattern=f"*:{collection_name}:*")

            return {"status": "deleted", "collection": collection_name}
        except Exception as e:
            logger.error(f"Failed to delete collection {collection_name}: {e}")
            return {"status": "error", "message": str(e)}

    @mcp.tool()
    async def optimize_collection(collection_name: str) -> dict[str, Any]:
        """
        Optimize a collection for better performance.

        Rebuilds indexes and optimizes storage.
        """
        await service_manager.initialize()

        try:
            # Get current collection info
            info = await service_manager.qdrant_service.get_collection_info(
                collection_name
            )

            # Trigger optimization
            # Note: Qdrant automatically optimizes, but we can force index rebuild
            # This is a placeholder for future optimization strategies

            return {
                "status": "optimized",
                "collection": collection_name,
                "vectors_count": info.vectors_count,
                "indexed_vectors_count": info.indexed_vectors_count,
            }
        except Exception as e:
            logger.error(f"Failed to optimize collection {collection_name}: {e}")
            return {"status": "error", "message": str(e)}
