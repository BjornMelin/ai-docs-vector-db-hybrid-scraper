"""Minimal system health monitoring using psutil."""

import logging
from typing import Any

from fastmcp import Context

from src.infrastructure.client_manager import ClientManager


logger = logging.getLogger(__name__)


def register_tools(mcp, client_manager: ClientManager):
    """Register system health monitoring tools."""

    @mcp.tool()
    async def get_system_health(
        ctx: Context | None = None,
    ) -> dict[str, Any]:
        """Get current system health metrics.

        Args:
            ctx: MCP context

        Returns:
            System health status with CPU, memory, disk metrics
        """

        try:
            # Lazy import to avoid hard dependency
            import psutil  # pylint: disable=import-outside-toplevel

            cpu_percent = psutil.cpu_percent(interval=1)
            memory = psutil.virtual_memory()
            disk = psutil.disk_usage("/")

            health = {
                "status": "healthy"
                if cpu_percent < 80 and memory.percent < 80
                else "degraded",
                "cpu": {
                    "percent": cpu_percent,
                    "count": psutil.cpu_count(),
                },
                "memory": {
                    "percent": memory.percent,
                    "available_gb": round(memory.available / (1024**3), 2),
                    "total_gb": round(memory.total / (1024**3), 2),
                },
                "disk": {
                    "percent": disk.percent,
                    "free_gb": round(disk.free / (1024**3), 2),
                    "total_gb": round(disk.total / (1024**3), 2),
                },
            }

            if ctx:
                await ctx.info(f"System health: {health['status']}")

            return health

        except ImportError:
            logger.warning("psutil not installed - install with: pip install psutil")
            return {
                "error": "psutil not installed",
                "install_command": "pip install psutil",
            }
        except Exception as e:
            logger.exception("Failed to get system health")
            if ctx:
                await ctx.error(f"Health check failed: {e}")
            return {"error": str(e)}

    @mcp.tool()
    async def get_process_info(
        ctx: Context | None = None,
    ) -> dict[str, Any]:
        """Get current process information.

        Args:
            ctx: MCP context

        Returns:
            Process metrics (CPU, memory usage)
        """

        try:
            import psutil  # pylint: disable=import-outside-toplevel

            process = psutil.Process()

            info = {
                "pid": process.pid,
                "cpu_percent": process.cpu_percent(interval=1),
                "memory_mb": round(process.memory_info().rss / (1024**2), 2),
                "num_threads": process.num_threads(),
                "status": process.status(),
            }

            if ctx:
                await ctx.info(
                    f"Process {info['pid']}: {info['cpu_percent']}% CPU, "
                    f"{info['memory_mb']}MB RAM"
                )

            return info

        except ImportError:
            return {
                "error": "psutil not installed",
                "install_command": "pip install psutil",
            }
        except Exception as e:
            logger.exception("Failed to get process info")
            if ctx:
                await ctx.error(f"Process info failed: {e}")
            return {"error": str(e)}
