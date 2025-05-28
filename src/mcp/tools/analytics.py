"""Analytics and monitoring tools for MCP server."""

import logging
from datetime import UTC
from datetime import datetime
from typing import Any

from ...infrastructure.client_manager import ClientManager
from ..models.requests import AnalyticsRequest

logger = logging.getLogger(__name__)


def register_tools(mcp, client_manager: ClientManager):
    """Register analytics and monitoring tools with the MCP server."""

    @mcp.tool()
    async def get_analytics(request: AnalyticsRequest) -> dict[str, Any]:
        """
        Get analytics and metrics for collections and operations.

        Provides performance metrics, cost analysis, and usage statistics.
        """
        analytics = {
            "timestamp": datetime.now(UTC).isoformat(),
            "collections": {},
            "cache_metrics": {},
            "performance": {},
            "costs": {},
        }

        # Get collection analytics
        if request.collection:
            collections = [request.collection]
        else:
            # Get collections using service method
            collections = await client_manager.qdrant_service.list_collections()

        for collection in collections:
            try:
                info = await client_manager.qdrant_service.get_collection_info(
                    collection
                )
                analytics["collections"][collection] = {
                    "vector_count": info.get("vectors_count", 0),
                    "indexed_count": info.get("points_count", 0),
                    "status": info.get("status", "unknown"),
                }
            except Exception as e:
                logger.warning(f"Failed to get analytics for {collection}: {e}")

        # Get cache metrics
        if request.include_performance:
            cache_stats = await client_manager.cache_manager.get_stats()
            analytics["cache_metrics"] = cache_stats

        # Estimate costs
        if request.include_costs:
            total_vectors = sum(
                c.get("vector_count", 0) for c in analytics["collections"].values()
            )

            # Rough cost estimates
            analytics["costs"] = {
                "storage_gb": total_vectors * 1536 * 4 / 1e9,  # 4 bytes per dimension
                "monthly_estimate": total_vectors * 0.00001,  # Rough estimate
                "embedding_calls": cache_stats.get("total_requests", 0),
                "cache_savings": cache_stats.get("hit_rate", 0)
                * 0.02,  # Saved API calls
            }

        return analytics

    @mcp.tool()
    async def get_system_health() -> dict[str, Any]:
        """
        Get system health and status information.

        Checks all services and returns their health status.
        """
        health = {
            "status": "healthy",
            "timestamp": datetime.now(UTC).isoformat(),
            "services": {},
        }

        # Check Qdrant
        try:
            # Get collections using service method
            collections = await client_manager.qdrant_service.list_collections()
            health["services"]["qdrant"] = {
                "status": "healthy",
                "collections": len(collections),
            }
        except Exception as e:
            health["services"]["qdrant"] = {
                "status": "unhealthy",
                "error": str(e),
            }
            health["status"] = "degraded"

        # Check embedding service
        try:
            provider_info = client_manager.embedding_manager.get_current_provider_info()
            health["services"]["embeddings"] = {
                "status": "healthy",
                "provider": provider_info.get("name", "unknown"),
            }
        except Exception as e:
            health["services"]["embeddings"] = {
                "status": "unhealthy",
                "error": str(e),
            }
            health["status"] = "degraded"

        # Check cache
        try:
            cache_stats = await client_manager.cache_manager.get_stats()
            health["services"]["cache"] = {
                "status": "healthy",
                "hit_rate": cache_stats.get("hit_rate", 0),
            }
        except Exception as e:
            health["services"]["cache"] = {
                "status": "unhealthy",
                "error": str(e),
            }

        return health
