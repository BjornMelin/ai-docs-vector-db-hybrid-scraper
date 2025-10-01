"""Analytics MCP tools backed by the consolidated vector service."""

from __future__ import annotations

import asyncio
import logging
from datetime import datetime, timezone
from typing import Any

from fastmcp import Context

from src.infrastructure.client_manager import ClientManager
from src.mcp_tools.models.requests import AnalyticsRequest
from src.mcp_tools.models.responses import (
    AnalyticsResponse,
    SystemHealthResponse,
    SystemHealthServiceStatus,
)
from src.services.vector_db.service import VectorStoreService


logger = logging.getLogger(__name__)

_VECTOR_SERVICE_INIT_LOCK = asyncio.Lock()
_ESTIMATE_DIMENSIONS = 1536
_BYTES_PER_FLOAT32 = 4


async def _get_vector_service(client_manager: ClientManager) -> VectorStoreService:
    service = await client_manager.get_vector_store_service()
    if not service.is_initialized():
        async with _VECTOR_SERVICE_INIT_LOCK:
            if not service.is_initialized():
                await service.initialize()
    return service


def _timestamp() -> str:
    ts = datetime.now(tz=timezone.utc).isoformat()  # noqa: UP017 (Python 3.10 support)
    return ts.replace("+00:00", "Z")


def _normalize_stats(stats: Any) -> dict[str, Any]:
    """Best-effort normalization of collection stats to a plain dict."""

    if isinstance(stats, dict):
        return stats
    if hasattr(stats, "model_dump"):
        return stats.model_dump()  # type: ignore[no-untyped-call]
    if hasattr(stats, "dict"):
        return stats.dict()  # type: ignore[no-untyped-call]
    if hasattr(stats, "__dict__"):
        return dict(vars(stats))
    msg = f"Unsupported stats payload type: {type(stats)!r}"
    raise TypeError(msg)


def register_tools(mcp, client_manager: ClientManager) -> None:
    """Register analytics and health-check tools."""

    @mcp.tool()
    async def get_analytics(
        request: AnalyticsRequest, ctx: Context | None = None
    ) -> AnalyticsResponse:
        """Return lightweight analytics for vector collections."""

        vector_service = await _get_vector_service(client_manager)
        if request.collection:
            collections = [request.collection]
        else:
            collections = await vector_service.list_collections()

        collection_stats: dict[str, dict[str, Any]] = {}

        async def _fetch_stats(name: str) -> None:
            try:
                stats = await vector_service.collection_stats(name)
                collection_stats[name] = _normalize_stats(stats)
            except Exception as exc:  # pragma: no cover - runtime guard
                logger.warning("Failed to fetch analytics for %s: %s", name, exc)
                logger.debug("Collection stats failure", exc_info=True)
                if ctx:
                    await ctx.warning(
                        f"Analytics unavailable for collection '{name}': {exc}"
                    )

        await asyncio.gather(*(_fetch_stats(collection) for collection in collections))

        total_vectors = sum(
            stats.get("points_count", 0) for stats in collection_stats.values()
        )

        analytics_payload = {
            "timestamp": _timestamp(),
            "collections": collection_stats,
        }

        if request.include_performance:
            analytics_payload["performance"] = {
                "total_vectors": total_vectors,
                "collection_count": len(collection_stats),
            }

        if request.include_costs:
            estimated_bytes = total_vectors * _ESTIMATE_DIMENSIONS * _BYTES_PER_FLOAT32
            analytics_payload["costs"] = {
                "estimated_storage_gb": round(estimated_bytes / 1e9, 3),
                "placeholder_note": (
                    "Derived from vector count using default dimensions"
                ),
            }

        if ctx:
            await ctx.info(
                f"Analytics generated for {len(collection_stats)} collections"
            )

        return AnalyticsResponse(**analytics_payload)

    @mcp.tool()
    async def get_system_health(ctx: Context | None = None) -> SystemHealthResponse:
        """Return a concise system health report."""

        vector_service = await _get_vector_service(client_manager)
        services: dict[str, SystemHealthServiceStatus] = {}

        try:
            collections = await vector_service.list_collections()
            services["vector_store"] = SystemHealthServiceStatus.model_validate(
                {"status": "healthy", "collections": len(collections)}
            )
        except Exception as exc:  # pragma: no cover - runtime guard
            logger.debug("Vector store health check failed", exc_info=True)
            services["vector_store"] = SystemHealthServiceStatus.model_validate(
                {"status": "unhealthy", "error": str(exc)}
            )

        overall_status = "healthy"
        if any(service.status == "unhealthy" for service in services.values()):
            overall_status = "unhealthy"

        if ctx:
            await ctx.info(f"System health status: {overall_status}")

        return SystemHealthResponse(
            status=overall_status,
            timestamp=_timestamp(),
            services=services,
        )
