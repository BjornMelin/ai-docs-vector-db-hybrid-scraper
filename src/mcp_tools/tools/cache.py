"""Cache management tools for MCP server."""

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
    """Register cache management tools with the MCP server."""

    from ..models.responses import CacheClearResponse, CacheStatsResponse

    @mcp.tool()
    async def clear_cache(
        pattern: str | None = None, ctx: Context = None
    ) -> CacheClearResponse:
        """
        Clear cache entries.

        Clears all cache entries or those matching a specific pattern.
        """
        if ctx:
            if pattern:
                await ctx.info(f"Starting cache clear with pattern: {pattern}")
            else:
                await ctx.info("Starting full cache clear")

        try:
            cache_manager = await client_manager.get_cache_manager()
            if pattern:
                cleared = await cache_manager.clear_pattern(pattern)
                if ctx:
                    await ctx.info(
                        f"Cleared {cleared} cache entries matching pattern: {pattern}"
                    )
            else:
                cleared = await cache_manager.clear_all()
                if ctx:
                    await ctx.info(f"Cleared all {cleared} cache entries")

            return CacheClearResponse(
                status="success",
                cleared_count=cleared,
                pattern=pattern,
            )

        except Exception as e:
            if ctx:
                await ctx.error(f"Failed to clear cache: {e}")
            logger.exception("Failed to clear cache")
            raise

    @mcp.tool()
    async def get_cache_stats(ctx: Context = None) -> CacheStatsResponse:
        """
        Get cache statistics and metrics.

        Returns hit rate, size, and performance metrics for the cache.
        """
        if ctx:
            await ctx.info("Retrieving cache statistics")

        try:
            cache_manager = await client_manager.get_cache_manager()
            stats = await cache_manager.get_stats()

            if ctx:
                await ctx.info(
                    f"Cache statistics retrieved: hit_rate={stats.get('hit_rate', 0)}, size={stats.get('size', 0)}"
                )

            # Cache manager may return additional fields; allow them via **stats
            return CacheStatsResponse(**stats)

        except Exception as e:
            if ctx:
                await ctx.error(f"Failed to retrieve cache statistics: {e}")
            logger.exception("Failed to retrieve cache statistics")
            raise
