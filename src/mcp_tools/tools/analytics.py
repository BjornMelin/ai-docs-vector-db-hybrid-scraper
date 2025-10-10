"""Analytics MCP tools backed by VectorStoreService."""

from __future__ import annotations

import asyncio
import logging
from datetime import UTC, datetime
from typing import Any

from fastmcp import Context

from src.mcp_tools.models.requests import AnalyticsRequest
from src.mcp_tools.models.responses import (
    AnalyticsResponse,
    SystemHealthResponse,
    SystemHealthServiceStatus,
)
from src.services.dependencies import get_vector_store_service
from src.services.vector_db.service import VectorStoreService


logger = logging.getLogger(__name__)
_ESTIMATE_DIMENSIONS = 1536
_BYTES_PER_FLOAT32 = 4


async def _resolve_vector_service(
    vector_service: VectorStoreService | None = None,
) -> VectorStoreService:
    """Return an initialized vector service, resolving from container when needed."""

    if vector_service is not None:
        if (
            hasattr(vector_service, "is_initialized")
            and not vector_service.is_initialized()
        ):
            initializer = getattr(vector_service, "initialize", None)
            if callable(initializer):
                result = initializer()
                if asyncio.iscoroutine(result):
                    await result
        return vector_service
    return await get_vector_store_service()


def _timestamp() -> str:
    """Return the current UTC timestamp in ISO 8601 format."""

    ts = datetime.now(tz=UTC).isoformat()
    return ts.replace("+00:00", "Z")


def _normalize_stats(stats: Any) -> dict[str, Any]:
    """Normalize stats payload into a plain dictionary."""

    if isinstance(stats, dict):
        return stats
    for attr in ("model_dump", "dict"):
        fn = getattr(stats, attr, None)
        if callable(fn):
            return fn()  # type: ignore[no-any-return]
    if hasattr(stats, "__dict__"):
        return dict(vars(stats))
    raise TypeError(f"Unsupported stats payload type: {type(stats)!r}")


def register_tools(
    mcp,
    vector_service: VectorStoreService | None = None,
) -> None:
    """Register analytics tools.

    Args:
        mcp: FastMCP server instance used for tool registration.
        vector_service: Optional pre-resolved vector service. When omitted, the
            service is resolved from the dependency-injector container.
    """

    @mcp.tool()
    async def get_analytics(
        request: AnalyticsRequest, ctx: Context | None = None
    ) -> AnalyticsResponse:
        """Return lightweight analytics for vector collections."""

        service = await _resolve_vector_service(vector_service)
        collections = (
            [request.collection]
            if request.collection
            else await service.list_collections()
        )

        collection_stats: dict[str, dict[str, Any]] = {}

        async def _fetch(name: str) -> None:
            try:
                stats = await service.collection_stats(name)
                collection_stats[name] = _normalize_stats(stats)
            except (ValueError, RuntimeError) as exc:
                logger.warning("Analytics failed for %s: %s", name, exc)
                if ctx:
                    await ctx.warning(
                        f"Analytics unavailable for collection '{name}': {exc}"
                    )

        await asyncio.gather(*(_fetch(c) for c in collections))

        total_vectors = sum(
            stats.get("points_count", 0) for stats in collection_stats.values()
        )
        payload: dict[str, Any] = {
            "timestamp": _timestamp(),
            "collections": collection_stats,
        }  # noqa: E501

        if request.include_performance:
            payload["performance"] = {
                "total_vectors": total_vectors,
                "collection_count": len(collection_stats),
            }
        if request.include_costs:
            est_bytes = total_vectors * _ESTIMATE_DIMENSIONS * _BYTES_PER_FLOAT32
            payload["costs"] = {
                "estimated_storage_gb": round(est_bytes / 1e9, 3),
                "note": "Derived from vector count with default dimensions",
            }

        if ctx:
            await ctx.info(
                f"Analytics generated for {len(collection_stats)} collections"
            )  # noqa: E501
        return AnalyticsResponse(**payload)

    @mcp.tool()
    async def get_system_health(ctx: Context | None = None) -> SystemHealthResponse:
        """Report high-level system health for analytics dependencies."""

        vector_status = SystemHealthServiceStatus(status="healthy", error=None)
        overall_status = "healthy"

        try:
            service = await _resolve_vector_service(vector_service)
            await service.list_collections()
            if ctx:
                await ctx.info("Vector store service responded successfully")
        except Exception as exc:  # pragma: no cover - defensive guard
            overall_status = "unhealthy"
            vector_status = SystemHealthServiceStatus(
                status="unhealthy", error=str(exc)
            )
            if ctx:
                await ctx.info(f"Vector store reported unhealthy: {exc}")

        return SystemHealthResponse(
            status=overall_status,
            timestamp=_timestamp(),
            services={"vector_store": vector_status},
        )
