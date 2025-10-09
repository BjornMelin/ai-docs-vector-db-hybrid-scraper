"""MCP tools for querying centralized system health state."""

from __future__ import annotations

import gc
import logging
from typing import TYPE_CHECKING, Any

from src.infrastructure.client_manager import ClientManager
from src.services.health.manager import HealthCheckManager, HealthStatus


try:  # pragma: no cover - optional dependency
    import psutil  # type: ignore[import-not-found]
except ImportError:  # pragma: no cover - optional dependency
    psutil = None  # type: ignore[assignment]

try:  # pragma: no cover - optional dependency
    from fastmcp import Context  # type: ignore[reportMissingImports]
except ImportError:  # pragma: no cover - fallback for linting environments
    Context = Any  # type: ignore[assignment]

if TYPE_CHECKING:  # pragma: no cover - typing only
    from fastmcp import Context as ContextType  # type: ignore[reportMissingImports]
else:  # pragma: no cover - runtime fallback
    ContextType = Any  # type: ignore[assignment]


logger = logging.getLogger(__name__)


def _bytes_to_mb(value: float | int) -> float:
    """Convert raw byte counts to mebibytes."""

    return float(value) / (1024 * 1024)


def _collect_resource_snapshot() -> dict[str, Any]:
    """Collect process and system resource utilisation statistics."""

    if psutil is None:
        return {"psutil_available": False}

    snapshot: dict[str, Any] = {"psutil_available": True}

    try:
        process = psutil.Process()
        with process.oneshot():
            cpu_percent = process.cpu_percent(interval=None)
            memory_info = process.memory_info()
            memory_percent = process.memory_percent()
            num_threads = process.num_threads()
            open_file_count = len(process.open_files())
            connection_count = len(process.connections(kind="inet"))
        snapshot["process"] = {
            "cpu_percent": cpu_percent,
            "memory_percent": memory_percent,
            "rss_memory_mb": _bytes_to_mb(memory_info.rss),
            "num_threads": num_threads,
            "open_file_count": open_file_count,
            "connection_count": connection_count,
        }
    except (psutil.Error, OSError):  # pragma: no cover - best effort only
        logger.debug("Failed to collect process metrics", exc_info=True)

    try:
        virtual_memory = psutil.virtual_memory()
        swap = psutil.swap_memory()
        disk_usage = psutil.disk_usage("/")
        snapshot["system"] = {
            "cpu_count_logical": psutil.cpu_count(logical=True),
            "memory_total_mb": _bytes_to_mb(virtual_memory.total),
            "memory_available_mb": _bytes_to_mb(virtual_memory.available),
            "memory_usage_percent": virtual_memory.percent,
            "swap_usage_percent": swap.percent,
            "disk_usage_percent": disk_usage.percent,
            "disk_total_gb": disk_usage.total / (1024**3),
            "disk_free_gb": disk_usage.free / (1024**3),
        }
    except (psutil.Error, OSError):  # pragma: no cover - best effort only
        logger.debug("Failed to collect system metrics", exc_info=True)

    try:
        disk_io = psutil.disk_io_counters()
        net_io = psutil.net_io_counters()
    except (psutil.Error, OSError):  # pragma: no cover - best effort only
        logger.debug("Failed to collect IO counters", exc_info=True)
    else:
        if disk_io is not None and net_io is not None:
            snapshot["io"] = {
                "disk_read_mb": _bytes_to_mb(disk_io.read_bytes),
                "disk_write_mb": _bytes_to_mb(disk_io.write_bytes),
                "net_sent_mb": _bytes_to_mb(net_io.bytes_sent),
                "net_received_mb": _bytes_to_mb(net_io.bytes_recv),
            }

    gc_stats: dict[str, Any] = {
        "counts": list(gc.get_count()),
    }
    get_stats = getattr(gc, "get_stats", None)
    if callable(get_stats):
        raw_stats = get_stats()
        generation_stats: list[dict[str, int | None]] = []
        if isinstance(raw_stats, list):
            for stat in raw_stats:
                if isinstance(stat, dict):
                    generation_stats.append(
                        {
                            "collections": stat.get("collections"),
                            "collected": stat.get("collected"),
                            "uncollectable": stat.get("uncollectable"),
                        }
                    )
        if generation_stats:
            gc_stats["generations"] = generation_stats
    snapshot["garbage_collector"] = gc_stats

    return snapshot


def _resolve_health_manager(client_manager: ClientManager) -> HealthCheckManager:
    """Return the configured health manager instance."""

    try:
        return client_manager.get_health_manager()
    except RuntimeError as error:  # pragma: no cover - defensive
        logger.error("Health manager unavailable: %s", error)
        raise


def register_tools(mcp, client_manager: ClientManager):
    """Register system health monitoring tools."""

    @mcp.tool()
    async def get_system_health(
        ctx: ContextType | None = None,
    ) -> dict[str, Any]:
        """Return overall health information from the central manager.

        Args:
            ctx: Optional MCP context for streaming updates.

        Returns:
            Aggregated health summary generated by :class:`HealthCheckManager`.
        """

        try:
            manager = _resolve_health_manager(client_manager)
        except RuntimeError as error:
            message = f"Health manager unavailable: {error}"
            logger.exception(message)
            if ctx:
                await ctx.error(message)
            return {"status": HealthStatus.UNKNOWN.value, "error": message}
        summary = await manager.get_overall_health()

        if ctx:
            await ctx.info(f"System health: {summary['overall_status']}")

        return summary

    @mcp.tool()
    async def get_process_info(
        ctx: ContextType | None = None,
    ) -> dict[str, Any]:
        """Return the latest system resource check metadata.

        Args:
            ctx: Optional MCP context for streaming updates.

        Returns:
            Structured metadata for the ``system_resources`` health check.
        """

        try:
            manager = _resolve_health_manager(client_manager)
        except RuntimeError as error:
            message = f"Health manager unavailable: {error}"
            logger.exception(message)
            if ctx:
                await ctx.error(message)
            return {"status": HealthStatus.UNKNOWN.value, "error": message}
        result = await manager.check_single("system_resources")
        if result is None:
            message = (
                "System resource health check is not registered; "
                "enable monitoring to collect metrics."
            )
            if ctx:
                await ctx.info(message)
            return {"status": HealthStatus.UNKNOWN.value, "message": message}

        resource_snapshot = _collect_resource_snapshot()
        combined_metrics = dict(result.metadata)
        process_metrics = resource_snapshot.get("process", {})
        for key in ("cpu_percent", "memory_percent", "rss_memory_mb"):
            value = process_metrics.get(key)
            if key not in combined_metrics and value is not None:
                combined_metrics[key] = value

        response = {
            "status": result.status.value,
            "message": result.message,
            "timestamp": result.timestamp,
            "duration_ms": result.duration_ms,
            "metrics": combined_metrics,
            "resource_snapshot": resource_snapshot,
        }

        if ctx:
            await ctx.info(
                "System resources %s: CPU %.1f%%, memory %.1f%%",
                result.status.value,
                combined_metrics.get("cpu_percent", 0.0),
                combined_metrics.get("memory_percent", 0.0),
            )

        return response
