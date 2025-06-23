
"""Analytics and monitoring tools for MCP server."""

import logging
from datetime import UTC
from datetime import datetime
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
from ..models.requests import AnalyticsRequest

logger = logging.getLogger(__name__)


def register_tools(mcp, client_manager: ClientManager):
    """Register analytics and monitoring tools with the MCP server."""

    from ..models.responses import AnalyticsResponse
    from ..models.responses import SystemHealthResponse

    @mcp.tool()
    async def get_analytics(
        request: AnalyticsRequest, ctx: Context
    ) -> AnalyticsResponse:
        """
        Get analytics and metrics for collections and operations.

        Provides performance metrics, cost analysis, and usage statistics.
        """
        await ctx.info("Starting analytics collection")

        analytics = {
            "timestamp": datetime.now(UTC).isoformat(),
            "collections": {},
            "cache_metrics": {},
            "performance": {},
            "costs": {},
        }

        try:
            # Get collection analytics
            if request.collection:
                collections = [request.collection]
                await ctx.debug(f"Analyzing specific collection: {request.collection}")
            else:
                # Get collections using service method
                qdrant_service = await client_manager.get_qdrant_service()
                collections = await qdrant_service.list_collections()
                await ctx.debug(f"Analyzing all collections: {len(collections)} found")

            qdrant_service = await client_manager.get_qdrant_service()
            for collection in collections:
                try:
                    info = await qdrant_service.get_collection_info(collection)
                    analytics["collections"][collection] = {
                        "vector_count": info.get("vectors_count", 0),
                        "indexed_count": info.get("points_count", 0),
                        "status": info.get("status", "unknown"),
                    }
                    await ctx.debug(f"Collected analytics for collection {collection}")
                except Exception as e:
                    await ctx.warning(f"Failed to get analytics for {collection}: {e}")
                    logger.warning(f"Failed to get analytics for {collection}: {e}")

            # Get cache metrics
            if request.include_performance:
                await ctx.debug("Including performance metrics")
                cache_manager = await client_manager.get_cache_manager()
                cache_stats = await cache_manager.get_stats()
                analytics["cache_metrics"] = cache_stats

            # Estimate costs
            if request.include_costs:
                await ctx.debug("Including cost estimates")
                total_vectors = sum(
                    c.get("vector_count", 0) for c in analytics["collections"].values()
                )

                # Rough cost estimates
                analytics["costs"] = {
                    "storage_gb": total_vectors
                    * 1536
                    * 4
                    / 1e9,  # 4 bytes per dimension
                    "monthly_estimate": total_vectors * 0.00001,  # Rough estimate
                    "embedding_calls": cache_stats.get("total_requests", 0)
                    if "cache_stats" in locals()
                    else 0,
                    "cache_savings": cache_stats.get("hit_rate", 0) * 0.02
                    if "cache_stats" in locals()
                    else 0,  # Saved API calls
                }

            await ctx.info(
                f"Analytics collection completed for {len(analytics['collections'])} collections"
            )
            return AnalyticsResponse(**analytics)

        except Exception as e:
            await ctx.error(f"Failed to collect analytics: {e}")
            logger.exception(f"Failed to collect analytics: {e}")
            raise

    @mcp.tool()
    async def get_system_health(ctx: Context) -> SystemHealthResponse:
        """
        Get system health and status information.

        Checks all services and returns their health status.
        """
        await ctx.info("Starting system health check")

        health = {
            "status": "healthy",
            "timestamp": datetime.now(UTC).isoformat(),
            "services": {},
        }

        # Check Qdrant
        try:
            await ctx.debug("Checking Qdrant service health")
            # Get collections using service method
            qdrant_service = await client_manager.get_qdrant_service()
            collections = await qdrant_service.list_collections()
            health["services"]["qdrant"] = {
                "status": "healthy",
                "collections": len(collections),
            }
            await ctx.debug(f"Qdrant healthy with {len(collections)} collections")
        except Exception as e:
            health["services"]["qdrant"] = {
                "status": "unhealthy",
                "error": str(e),
            }
            health["status"] = "degraded"
            await ctx.warning(f"Qdrant service unhealthy: {e}")

        # Check embedding service
        try:
            await ctx.debug("Checking embedding service health")
            embedding_manager = await client_manager.get_embedding_manager()
            provider_info = embedding_manager.get_current_provider_info()
            health["services"]["embeddings"] = {
                "status": "healthy",
                "provider": provider_info.get("name", "unknown"),
            }
            await ctx.debug(
                f"Embeddings healthy with provider: {provider_info.get('name', 'unknown')}"
            )
        except Exception as e:
            health["services"]["embeddings"] = {
                "status": "unhealthy",
                "error": str(e),
            }
            health["status"] = "degraded"
            await ctx.warning(f"Embedding service unhealthy: {e}")

        # Check cache
        try:
            await ctx.debug("Checking cache service health")
            cache_manager = await client_manager.get_cache_manager()
            cache_stats = await cache_manager.get_stats()
            health["services"]["cache"] = {
                "status": "healthy",
                "hit_rate": cache_stats.get("hit_rate", 0),
            }
            await ctx.debug(
                f"Cache healthy with hit rate: {cache_stats.get('hit_rate', 0)}"
            )
        except Exception as e:
            health["services"]["cache"] = {
                "status": "unhealthy",
                "error": str(e),
            }
            await ctx.warning(f"Cache service unhealthy: {e}")

        await ctx.info(
            f"System health check completed. Overall status: {health['status']}"
        )
        return SystemHealthResponse(**health)
