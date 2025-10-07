"""Analytics MCP tools backed by VectorStoreService."""

from __future__ import annotations

import asyncio
import logging
from datetime import UTC, datetime
from typing import Any

from fastmcp import Context

from src.infrastructure.client_manager import ClientManager
from src.mcp_tools.models.requests import AnalyticsRequest
from src.mcp_tools.models.responses import AnalyticsResponse
from src.services.vector_db.service import VectorStoreService


logger = logging.getLogger(__name__)
_VECTOR_SERVICE_INIT_LOCK = asyncio.Lock()
_ESTIMATE_DIMENSIONS = 1536
_BYTES_PER_FLOAT32 = 4


async def _get_vector_service(client_manager: ClientManager) -> VectorStoreService:
    svc = await client_manager.get_vector_store_service()
    if not svc.is_initialized():
        async with _VECTOR_SERVICE_INIT_LOCK:
            if not svc.is_initialized():
                await svc.initialize()
    return svc


def _timestamp() -> str:
    ts = datetime.now(tz=UTC).isoformat()
    return ts.replace("+00:00", "Z")


def _normalize_stats(stats: Any) -> dict[str, Any]:
    if isinstance(stats, dict):
        return stats
    for attr in ("model_dump", "dict"):
        fn = getattr(stats, attr, None)
        if callable(fn):
            return fn()  # type: ignore[no-any-return]
    if hasattr(stats, "__dict__"):
        return dict(vars(stats))
    raise TypeError(f"Unsupported stats payload type: {type(stats)!r}")


def register_tools(mcp, client_manager: ClientManager) -> None:
    """Register analytics tools."""

    @mcp.tool()
    async def get_analytics(
        request: AnalyticsRequest, ctx: Context | None = None
    ) -> AnalyticsResponse:
        """Return lightweight analytics for vector collections."""
        vector_service = await _get_vector_service(client_manager)
        collections = (
            [request.collection]
            if request.collection
            else await vector_service.list_collections()
        )

        collection_stats: dict[str, dict[str, Any]] = {}

        async def _fetch(name: str) -> None:
            try:
                stats = await vector_service.collection_stats(name)
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
