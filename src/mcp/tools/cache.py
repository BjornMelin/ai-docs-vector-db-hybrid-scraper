"""Cache management tools for MCP server."""

import logging
from typing import Any

from ...infrastructure.client_manager import ClientManager

logger = logging.getLogger(__name__)


def register_tools(mcp, client_manager: ClientManager):
    """Register cache management tools with the MCP server."""

    @mcp.tool()
    async def clear_cache(pattern: str | None = None) -> dict[str, Any]:
        """
        Clear cache entries.

        Clears all cache entries or those matching a specific pattern.
        """
        try:
            if pattern:
                cleared = await client_manager.cache_manager.clear_pattern(pattern)
            else:
                cleared = await client_manager.cache_manager.clear_all()

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
        return await client_manager.cache_manager.get_stats()
