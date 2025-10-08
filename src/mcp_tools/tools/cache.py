"""Cache management tools for MCP server."""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING

from fastmcp import Context

from src.mcp_tools.models.responses import CacheClearResponse, CacheStatsResponse


logger = logging.getLogger(__name__)

if TYPE_CHECKING:
    from src.infrastructure.client_manager import ClientManager


def register_tools(mcp, client_manager: ClientManager):
    """Register cache management tools with the MCP server."""

    @mcp.tool()
    async def clear_cache(
        pattern: str | None = None, ctx: Context | None = None
    ) -> CacheClearResponse:
        """Clear cache entries.

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
    async def get_cache_stats(ctx: Context | None = None) -> CacheStatsResponse:
        """Get cache statistics and metrics.

        Returns hit rate, size, and performance metrics for the cache.
        """
        if ctx:
            await ctx.info("Retrieving cache statistics")

        try:
            cache_manager = await client_manager.get_cache_manager()
            stats = await cache_manager.get_stats()

            hit_rate_value = stats.get("hit_rate")
            hit_rate = (
                float(hit_rate_value)
                if isinstance(hit_rate_value, int | float)
                else None
            )

            size_value = stats.get("size")
            size = int(size_value) if isinstance(size_value, int) else None

            total_requests_value = stats.get("total_requests")
            total_requests = (
                int(total_requests_value)
                if isinstance(total_requests_value, int)
                else None
            )

            if ctx:
                hit_rate_display = hit_rate if hit_rate is not None else "n/a"
                size_display = size if size is not None else "n/a"
                await ctx.info(
                    f"Cache statistics retrieved: hit_rate={hit_rate_display}, "
                    f"size={size_display}"
                )

            extra_payload = {
                key: value
                for key, value in stats.items()
                if key not in {"hit_rate", "size", "total_requests"}
            }

            return CacheStatsResponse(
                hit_rate=hit_rate,
                size=size,
                total_requests=total_requests,
                **extra_payload,
            )

        except Exception as e:
            if ctx:
                await ctx.error(f"Failed to retrieve cache statistics: {e}")
            logger.exception("Failed to retrieve cache statistics")
            raise
