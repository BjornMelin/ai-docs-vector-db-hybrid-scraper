"""Monitoring initialization for FastMCP applications using OTel instrumentation."""

from __future__ import annotations

import asyncio
import logging
import time
from typing import Any

from fastapi.responses import JSONResponse

from src.config.loader import Settings
from src.config.models import MonitoringConfig
from src.services.health.manager import (
    HealthCheckConfig,
    HealthCheckManager,
    HealthStatus,
    build_health_manager,
)


logger = logging.getLogger(__name__)


async def initialize_monitoring(
    config: MonitoringConfig,
) -> HealthCheckManager | None:
    """Initialize health monitoring from a MonitoringConfig instance."""

    if not config.enabled:
        logger.info("Monitoring disabled by configuration")
        return None

    health_config = HealthCheckConfig(
        enabled=config.enabled,
        interval=config.system_metrics_interval,
        timeout=config.health_check_timeout,
    )
    logger.info("Monitoring initialized with health checks only")
    return HealthCheckManager(health_config)


async def cleanup_monitoring(
    health_manager: HealthCheckManager | None,
) -> None:
    """Clean up health monitoring resources (placeholder for symmetry)."""

    if health_manager:
        logger.info("Cleaning up health manager resources")
    logger.info("Monitoring cleanup complete")


async def start_background_monitoring_tasks(
    health_manager: HealthCheckManager | None,
) -> list[asyncio.Task[Any]]:
    """Start periodic health checks when monitoring is enabled."""

    if not health_manager or not health_manager.config.enabled:
        return []

    interval = health_manager.config.interval
    task = asyncio.create_task(
        run_periodic_health_checks(health_manager, interval_seconds=interval)
    )
    logger.info("Started periodic health checks task")
    return [task]


async def stop_background_monitoring_tasks(tasks: list[asyncio.Task[Any]]) -> None:
    """Stop background monitoring tasks."""

    if not tasks:
        return

    logger.info("Stopping %d background monitoring tasks...", len(tasks))
    for task in tasks:
        task.cancel()
    await asyncio.gather(*tasks, return_exceptions=True)
    logger.info("Background monitoring tasks stopped")


def initialize_monitoring_system(
    config: Settings,
    qdrant_client: Any | None = None,
    redis_url: str | None = None,
) -> HealthCheckManager | None:
    """Initialize monitoring stack for the unified MCP server."""

    if not config.monitoring.enabled:
        logger.info("Monitoring disabled by configuration")
        return None

    logger.info("Initializing monitoring system (health checks only)...")
    health_manager = build_health_manager(
        config,
        qdrant_client=qdrant_client,
        redis_url=redis_url,
    )
    logger.info("Monitoring system initialization complete")
    return health_manager


def setup_fastmcp_monitoring(
    mcp_app: Any,
    config: Settings,
    health_manager: HealthCheckManager | None,
) -> None:
    """Attach health endpoints to the FastMCP FastAPI app."""

    if not config.monitoring.enabled or health_manager is None:
        logger.info("Skipping FastMCP monitoring setup (disabled or missing manager)")
        return

    if not hasattr(mcp_app, "app"):
        logger.warning(
            "FastMCP app does not expose the embedded FastAPI app; health endpoints "
            "disabled"
        )
        return

    fastapi_app = mcp_app.app

    @fastapi_app.get(
        config.monitoring.health_path,
        include_in_schema=False,
        tags=["monitoring"],
    )
    async def health_endpoint():
        """Return aggregated health status."""

        await health_manager.check_all()
        overall_status = health_manager.get_overall_status()
        status_code = (
            200
            if overall_status in (HealthStatus.HEALTHY, HealthStatus.DEGRADED)
            else 503
        )
        return JSONResponse(
            content=health_manager.get_health_summary(),
            status_code=status_code,
        )

    @fastapi_app.get(
        f"{config.monitoring.health_path}/live",
        include_in_schema=False,
        tags=["monitoring"],
    )
    async def liveness_endpoint():
        """Kubernetes liveness probe endpoint."""

        return JSONResponse(
            content={"status": "alive", "timestamp": time.time()},
            status_code=200,
        )

    @fastapi_app.get(
        f"{config.monitoring.health_path}/ready",
        include_in_schema=False,
        tags=["monitoring"],
    )
    async def readiness_endpoint():
        """Kubernetes readiness probe endpoint."""

        await health_manager.check_all()
        overall_status = health_manager.get_overall_status()
        ready = overall_status in (HealthStatus.HEALTHY, HealthStatus.DEGRADED)
        status_code = 200 if ready else 503
        status_value = "ready" if ready else "not_ready"
        return JSONResponse(
            content={
                "status": status_value,
                "health_summary": health_manager.get_health_summary(),
                "timestamp": time.time(),
            },
            status_code=status_code,
        )

    logger.info(
        "FastMCP health endpoints registered at %s", config.monitoring.health_path
    )


async def run_periodic_health_checks(
    health_manager: HealthCheckManager,
    interval_seconds: float = 60.0,
) -> None:
    """Run periodic health checks in the background."""

    logger.info("Starting periodic health checks (interval: %ss)", interval_seconds)
    while True:
        try:
            await health_manager.check_all()
            logger.debug("Completed periodic health check cycle")
        except Exception:  # pragma: no cover - defensive logging
            logger.exception("Error executing periodic health check")
        await asyncio.sleep(interval_seconds)
