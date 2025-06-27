"""Monitoring system initialization for FastMCP applications.

This module provides utilities to initialize and configure monitoring
for FastMCP-based applications.
"""

import asyncio  # noqa: PLC0415
import logging  # noqa: PLC0415
import time  # noqa: PLC0415

from src.config import Config, MonitoringConfig

from .health import HealthCheckConfig, HealthCheckManager, HealthStatus
from .metrics import MetricsConfig, MetricsRegistry, initialize_metrics


try:
    from fastapi.responses import JSONResponse
except ImportError:
    JSONResponse = None  # type: ignore


logger = logging.getLogger(__name__)


async def initialize_monitoring(
    config: MonitoringConfig,
) -> tuple[MetricsRegistry | None, HealthCheckManager | None]:
    """Initialize monitoring system.

    Args:
        config: Monitoring configuration

    Returns:
        Tuple of (MetricsRegistry, HealthCheckManager) or (None, None) if disabled
    """
    if not config.enabled:
        logger.info("Monitoring disabled by configuration")
        return None, None

    logger.info("Initializing monitoring system...")

    # Create metrics config from monitoring config
    metrics_config = MetricsConfig(
        enabled=config.enabled,
        export_port=config.metrics_port,
        namespace=config.namespace,
        include_system_metrics=config.include_system_metrics,
        collection_interval=config.system_metrics_interval,
    )

    # Initialize metrics registry
    metrics_registry = MetricsRegistry(metrics_config)

    # Create health check config from monitoring config
    health_config = HealthCheckConfig(
        enabled=config.enabled, timeout=config.health_check_timeout
    )

    # Initialize health check manager
    health_manager = HealthCheckManager(health_config)

    logger.info("Monitoring system initialization complete")
    return metrics_registry, health_manager


async def cleanup_monitoring(
    metrics_registry: MetricsRegistry | None, health_manager: HealthCheckManager | None
) -> None:
    """Clean up monitoring resources.

    Args:
        metrics_registry: Metrics registry to clean up
        health_manager: Health manager to clean up
    """
    if metrics_registry:
        logger.info("Cleaning up metrics registry...")
        # Any cleanup needed for metrics

    if health_manager:
        logger.info("Cleaning up health manager...")
        # Any cleanup needed for health checks

    logger.info("Monitoring cleanup complete")


async def start_background_monitoring_tasks(
    metrics_registry: MetricsRegistry | None, health_manager: HealthCheckManager | None
) -> list[asyncio.Task]:
    """Start background monitoring tasks.

    Args:
        metrics_registry: Metrics registry
        health_manager: Health manager

    Returns:
        List of started tasks
    """
    tasks = []

    # Start system metrics collection if enabled
    if (
        metrics_registry
        and metrics_registry.config.enabled
        and metrics_registry.config.include_system_metrics
    ):
        task = asyncio.create_task(
            update_system_metrics_periodically(
                metrics_registry, metrics_registry.config.collection_interval
            )
        )
        tasks.append(task)
        logger.info("Started system metrics collection task")

    # Start health checks if enabled
    if health_manager and health_manager.config.enabled:
        task = asyncio.create_task(
            run_periodic_health_checks(health_manager, health_manager.config.interval)
        )
        tasks.append(task)
        logger.info("Started periodic health checks task")

    return tasks


async def stop_background_monitoring_tasks(tasks: list[asyncio.Task]) -> None:
    """Stop background monitoring tasks.

    Args:
        tasks: List of tasks to stop
    """
    if not tasks:
        return

    logger.info(f"Stopping {len(tasks)} background monitoring tasks...")

    for task in tasks:
        task.cancel()

    # Wait for tasks to be cancelled
    await asyncio.gather(*tasks, return_exceptions=True)

    logger.info("Background monitoring tasks stopped")


def initialize_monitoring_system(
    config: Config, qdrant_client=None, redis_url: str | None = None
) -> tuple[MetricsRegistry, HealthCheckManager]:
    """Initialize the complete monitoring system.

    Args:
        config: Application configuration
        qdrant_client: Optional Qdrant client for health checks
        redis_url: Optional Redis URL for health checks

    Returns:
        Tuple of (MetricsRegistry, HealthCheckManager)
    """
    if not config.monitoring.enabled:
        logger.info("Monitoring disabled by configuration")
        return None, None

    logger.info("Initializing monitoring system...")

    # Create metrics configuration
    metrics_config = MetricsConfig(
        enabled=config.monitoring.enabled,
        export_port=config.monitoring.metrics_port,
        namespace=config.monitoring.namespace,
        include_system_metrics=config.monitoring.include_system_metrics,
        collection_interval=config.monitoring.system_metrics_interval,
    )

    # Initialize metrics registry
    metrics_registry = initialize_metrics(metrics_config)

    # Create health check manager
    from .health import HealthCheckConfig  # noqa: PLC0415

    health_config = HealthCheckConfig(
        enabled=config.monitoring.enabled,
        timeout=config.monitoring.health_check_timeout,
    )
    health_manager = HealthCheckManager(health_config, metrics_registry)

    # Add system resource checks
    health_manager.add_system_resource_check(
        cpu_threshold=config.monitoring.cpu_threshold,
        memory_threshold=config.monitoring.memory_threshold,
        disk_threshold=config.monitoring.disk_threshold,
        timeout_seconds=config.monitoring.health_check_timeout,
    )

    # Add Qdrant health check if client provided
    if qdrant_client:
        health_manager.add_qdrant_check(
            qdrant_client, timeout_seconds=config.monitoring.health_check_timeout
        )
        logger.info("Added Qdrant health check")

    # Add Redis health check if cache is enabled
    if config.cache.enable_dragonfly_cache:
        redis_url = redis_url or config.cache.dragonfly_url
        health_manager.add_redis_check(
            redis_url, timeout_seconds=config.monitoring.health_check_timeout
        )
        logger.info("Added Redis/DragonflyDB health check")

    # Add external service health checks
    for service_name, service_url in config.monitoring.external_services.items():
        health_manager.add_http_check(
            url=service_url,
            name=service_name,
            timeout_seconds=config.monitoring.health_check_timeout,
        )
        logger.info("Added external service health check")

    # Start metrics server
    if metrics_config.enabled:
        try:
            metrics_registry.start_metrics_server()
            logger.info(
                f"Prometheus metrics server started on port {metrics_config.export_port}"
            )
        except Exception as e:
            logger.exception("Failed to start metrics server")

    logger.info("Monitoring system initialization complete")
    return metrics_registry, health_manager


def setup_fastmcp_monitoring(
    mcp_app,
    config: Config,
    metrics_registry: MetricsRegistry,
    health_manager: HealthCheckManager,
) -> None:
    """Set up monitoring for FastMCP application.

    Args:
        mcp_app: FastMCP application instance
        config: Application configuration
        metrics_registry: Initialized metrics registry
        health_manager: Initialized health check manager
    """
    if not config.monitoring.enabled or not metrics_registry or not health_manager:
        logger.info("Monitoring not enabled or not initialized")
        return

    logger.info("Setting up FastMCP monitoring integration...")

    try:
        # Check if FastMCP exposes the underlying FastAPI app
        if hasattr(mcp_app, "app"):
            fastapi_app = mcp_app.app

            # Add health endpoint
            @fastapi_app.get(
                config.monitoring.health_path,
                include_in_schema=False,
                tags=["monitoring"],
            )
            async def health_endpoint():
                """Health check endpoint for FastMCP."""

                if not health_manager:
                    return JSONResponse(
                        content={
                            "status": "healthy",
                            "message": "No health checks configured",
                            "timestamp": time.time(),
                        },
                        status_code=200,
                    )

                # Run all health checks
                await health_manager.check_all()
                overall_status = health_manager.get_overall_status()

                # Determine HTTP status code based on health
                if overall_status == HealthStatus.HEALTHY:
                    status_code = 200
                elif overall_status == HealthStatus.DEGRADED:
                    status_code = 200  # Still OK but with warnings
                else:
                    status_code = 503  # Service unavailable

                return JSONResponse(
                    content=health_manager.get_health_summary(), status_code=status_code
                )

            # Add liveness endpoint
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

            # Add readiness endpoint
            @fastapi_app.get(
                f"{config.monitoring.health_path}/ready",
                include_in_schema=False,
                tags=["monitoring"],
            )
            async def readiness_endpoint():
                """Kubernetes readiness probe endpoint."""

                if not health_manager:
                    return JSONResponse(
                        content={
                            "status": "ready",
                            "message": "No dependencies to check",
                            "timestamp": time.time(),
                        },
                        status_code=200,
                    )

                # Check critical dependencies only
                await health_manager.check_all()
                overall_status = health_manager.get_overall_status()

                # Ready only if all critical dependencies are healthy
                if overall_status in [HealthStatus.HEALTHY, HealthStatus.DEGRADED]:
                    return JSONResponse(
                        content={
                            "status": "ready",
                            "health_summary": health_manager.get_health_summary(),
                            "timestamp": time.time(),
                        },
                        status_code=200,
                    )
                else:
                    return JSONResponse(
                        content={
                            "status": "not_ready",
                            "health_summary": health_manager.get_health_summary(),
                            "timestamp": time.time(),
                        },
                        status_code=503,
                    )

            logger.info(f"Added health endpoints at {config.monitoring.health_path}")

        else:
            logger.warning(
                "FastMCP app does not expose underlying FastAPI app - health endpoints not added"
            )

    except Exception as e:
        logger.exception("Failed to set up FastMCP monitoring")


async def run_periodic_health_checks(
    health_manager: HealthCheckManager, interval_seconds: float = 60.0
) -> None:
    """Run periodic health checks in the background.

    Args:
        health_manager: Health check manager
        interval_seconds: Interval between health checks
    """
    if not health_manager:
        return

    logger.info(f"Starting periodic health checks (interval: {interval_seconds}s)")

    while True:
        try:
            await health_manager.check_all()
            logger.debug("Completed periodic health check")
        except Exception as e:
            logger.exception("Error in periodic health check")

        await asyncio.sleep(interval_seconds)


async def update_system_metrics_periodically(
    metrics_registry: MetricsRegistry, interval_seconds: float = 30.0
) -> None:
    """Update system metrics periodically.

    Args:
        metrics_registry: Metrics registry
        interval_seconds: Interval between updates
    """
    if not metrics_registry or not metrics_registry.config.include_system_metrics:
        return

    logger.info(
        f"Starting periodic system metrics updates (interval: {interval_seconds}s)"
    )

    while True:
        try:
            metrics_registry.update_system_metrics()
            logger.debug("Updated system metrics")
        except Exception as e:
            logger.exception("Error updating system metrics")

        await asyncio.sleep(interval_seconds)


async def update_cache_metrics_periodically(
    metrics_registry: MetricsRegistry, cache_manager, interval_seconds: float = 30.0
) -> None:
    """Update cache metrics periodically.

    Args:
        metrics_registry: Metrics registry
        cache_manager: Cache manager instance to collect stats from
        interval_seconds: Interval between updates
    """
    if not metrics_registry or not cache_manager:
        return

    logger.info(
        f"Starting periodic cache metrics updates (interval: {interval_seconds}s)"
    )

    while True:
        try:
            metrics_registry.update_cache_stats(cache_manager)
            logger.debug("Updated cache metrics")
        except Exception as e:
            logger.exception("Error updating cache metrics")

        await asyncio.sleep(interval_seconds)
