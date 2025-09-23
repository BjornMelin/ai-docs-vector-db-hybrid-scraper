"""FastAPI middleware for Prometheus metrics integration.

This module provides middleware components for automatic HTTP request monitoring,
health check endpoints, and Prometheus metrics exposure.
"""

import logging
import time
from collections.abc import Callable

from fastapi import FastAPI, Request, Response
from fastapi.responses import JSONResponse
from prometheus_client import CONTENT_TYPE_LATEST, generate_latest
from prometheus_fastapi_instrumentator import Instrumentator, metrics
from starlette.middleware.base import BaseHTTPMiddleware

from .health import HealthCheckManager, HealthStatus
from .metrics import MetricsRegistry, get_metrics_registry


logger = logging.getLogger(__name__)


class PrometheusMiddleware:
    """Middleware for integrating Prometheus metrics with FastAPI applications."""

    def __init__(
        self,
        app: FastAPI,
        metrics_registry: MetricsRegistry,
        *,
        health_manager: HealthCheckManager | None = None,
        metrics_path: str = "/metrics",
        health_path: str = "/health",
        enable_default_metrics: bool = True,
    ):
        """Initialize Prometheus middleware.

        Args:
            app: FastAPI application instance
            metrics_registry: Metrics registry for custom metrics
            health_manager: Optional health check manager
            metrics_path: Path for Prometheus metrics endpoint
            health_path: Path for health check endpoint
            enable_default_metrics: Enable default HTTP metrics

        """
        self.app = app
        self.metrics_registry = metrics_registry
        self.health_manager = health_manager
        self.metrics_path = metrics_path
        self.health_path = health_path

        # Set up instrumentator for default metrics
        if enable_default_metrics:
            self.instrumentator = Instrumentator(
                should_group_status_codes=True,
                should_ignore_untemplated=True,
                should_respect_env_var=True,
                should_instrument_requests_inprogress=True,
                excluded_handlers=[metrics_path, health_path],
                env_var_name="ENABLE_METRICS",
                inprogress_name="fastapi_inprogress",
                inprogress_labels=True,
            )

            # Add default metrics (using available metrics)
            try:
                # Try the new API first
                if hasattr(metrics, "request_size_and_response_size"):
                    self.instrumentator.add(
                        metrics.request_size_and_response_size(
                            should_include_handler=True,
                            should_include_method=True,
                            should_include_status=True,
                        )
                    )
                else:
                    # Fall back to individual metrics if available
                    if hasattr(metrics, "request_size"):
                        self.instrumentator.add(metrics.request_size())
                    if hasattr(metrics, "response_size"):
                        self.instrumentator.add(metrics.response_size())
                    if hasattr(metrics, "requests"):
                        self.instrumentator.add(metrics.requests())
            except AttributeError as e:
                logger.warning(
                    f"Could not add request/response size metrics: {e}"
                )  # TODO: Convert f-string to logging format
                # Add basic request count metric as fallback
                try:
                    if hasattr(metrics, "requests"):
                        self.instrumentator.add(metrics.requests())
                except AttributeError:
                    logger.warning(
                        "No compatible metrics found, continuing with "
                        "basic instrumentator"
                    )
            self.instrumentator.add(
                metrics.latency(
                    should_include_handler=True,
                    should_include_method=True,
                    should_include_status=True,
                )
            )
            self.instrumentator.add(
                metrics.requests(
                    should_include_handler=True,
                    should_include_method=True,
                    should_include_status=True,
                )
            )

            # Instrument the app
            self.instrumentator.instrument(app)
        else:
            self.instrumentator = None

        # Add custom endpoints
        self._add_endpoints()

    def _add_endpoints(self) -> None:
        """Add metrics and health endpoints to the FastAPI app."""

        @self.app.get(
            self.metrics_path,
            include_in_schema=False,
            response_class=Response,
            tags=["monitoring"],
        )
        async def metrics_endpoint():
            """Prometheus metrics endpoint."""
            # Update system metrics before generating output
            if self.metrics_registry.config.include_system_metrics:
                self.metrics_registry.update_system_metrics()

            # Generate Prometheus metrics
            return Response(
                generate_latest(self.metrics_registry.registry),
                media_type=CONTENT_TYPE_LATEST,
            )

        @self.app.get(
            self.health_path,
            include_in_schema=False,
            response_class=JSONResponse,
            tags=["monitoring"],
        )
        async def health_endpoint():
            """Health check endpoint."""
            if not self.health_manager:
                return JSONResponse(
                    content={
                        "status": "healthy",
                        "message": "No health checks configured",
                        "timestamp": time.time(),
                    },
                    status_code=200,
                )

            # Run all health checks
            await self.health_manager.check_all()
            overall_status = self.health_manager.get_overall_status()

            # Determine HTTP status code based on health
            if overall_status == HealthStatus.HEALTHY:
                status_code = 200
            elif overall_status == HealthStatus.DEGRADED:
                status_code = 200  # Still OK but with warnings
            else:
                status_code = 503  # Service unavailable

            return JSONResponse(
                content=self.health_manager.get_health_summary(),
                status_code=status_code,
            )

        @self.app.get(
            f"{self.health_path}/live",
            include_in_schema=False,
            response_class=JSONResponse,
            tags=["monitoring"],
        )
        async def liveness_endpoint():
            """Kubernetes liveness probe endpoint."""
            return JSONResponse(
                content={"status": "alive", "timestamp": time.time()}, status_code=200
            )

        @self.app.get(
            f"{self.health_path}/ready",
            include_in_schema=False,
            response_class=JSONResponse,
            tags=["monitoring"],
        )
        async def readiness_endpoint():
            """Kubernetes readiness probe endpoint."""
            if not self.health_manager:
                return JSONResponse(
                    content={
                        "status": "ready",
                        "message": "No dependencies to check",
                        "timestamp": time.time(),
                    },
                    status_code=200,
                )

            # Check critical dependencies only
            await self.health_manager.check_all()
            overall_status = self.health_manager.get_overall_status()

            # Ready only if all critical dependencies are healthy
            if overall_status in [HealthStatus.HEALTHY, HealthStatus.DEGRADED]:
                return JSONResponse(
                    content={
                        "status": "ready",
                        "health_summary": self.health_manager.get_health_summary(),
                        "timestamp": time.time(),
                    },
                    status_code=200,
                )
            return JSONResponse(
                content={
                    "status": "not_ready",
                    "health_summary": self.health_manager.get_health_summary(),
                    "timestamp": time.time(),
                },
                status_code=503,
            )

    def expose_metrics(self) -> None:
        """Expose Prometheus metrics (if using instrumentator)."""
        if self.instrumentator:
            self.instrumentator.expose(self.app, endpoint=self.metrics_path)


class CustomMetricsMiddleware(BaseHTTPMiddleware):
    """Custom middleware for additional application-specific metrics."""

    def __init__(self, app, metrics_registry: MetricsRegistry | None = None):
        """Initialize custom metrics middleware.

        Args:
            app: ASGI application
            metrics_registry: Optional metrics registry

        """
        super().__init__(app)
        self.metrics_registry = metrics_registry or get_metrics_registry()

    async def dispatch(self, request: Request, call_next: Callable) -> Response:
        """Process request and collect custom metrics.

        Args:
            request: HTTP request
            call_next: Next middleware in chain

        Returns:
            HTTP response

        """
        start_time = time.time()

        # Extract request metadata
        method = request.method
        path = request.url.path

        # Process request
        response = await call_next(request)

        # Calculate duration
        duration = time.time() - start_time

        # Record custom metrics (if enabled)
        if self.metrics_registry:
            custom_duration_metric = self.metrics_registry.get_metric(
                "custom_request_duration"
            )
            if custom_duration_metric:
                # Record request duration by endpoint
                custom_duration_metric.labels(
                    method=method, endpoint=path, status_code=response.status_code
                ).observe(duration)

        return response


def setup_monitoring(
    app: FastAPI,
    metrics_registry: MetricsRegistry,
    *,
    health_manager: HealthCheckManager | None = None,
    enable_default_metrics: bool = True,
    enable_custom_metrics: bool = True,
    metrics_path: str = "/metrics",
    health_path: str = "/health",
) -> PrometheusMiddleware:
    """Set up comprehensive monitoring for FastAPI application.

    Args:
        app: FastAPI application instance
        metrics_registry: Metrics registry
        health_manager: Optional health check manager
        enable_default_metrics: Enable default HTTP metrics
        enable_custom_metrics: Enable custom metrics middleware
        metrics_path: Path for metrics endpoint
        health_path: Path for health endpoint

    Returns:
        Configured PrometheusMiddleware instance

    """
    # Set up Prometheus middleware
    prometheus_middleware = PrometheusMiddleware(
        app=app,
        metrics_registry=metrics_registry,
        health_manager=health_manager,
        metrics_path=metrics_path,
        health_path=health_path,
        enable_default_metrics=enable_default_metrics,
    )

    # Add custom metrics middleware if enabled
    if enable_custom_metrics:
        app.add_middleware(CustomMetricsMiddleware, metrics_registry=metrics_registry)

    return prometheus_middleware


def create_health_manager_with_defaults(
    qdrant_client=None,
    redis_url: str | None = None,
    external_services: dict[str, str] | None = None,
    metrics_registry: MetricsRegistry | None = None,
) -> HealthCheckManager:
    """Create health manager with common health checks.

    Args:
        qdrant_client: Optional Qdrant client for vector DB health checks
        redis_url: Optional Redis URL for cache health checks
        external_services: Optional dict of service name -> URL mappings
        metrics_registry: Optional metrics registry

    Returns:
        Configured HealthCheckManager

    """
    health_manager = HealthCheckManager(metrics_registry)

    # Add system resource check
    health_manager.add_system_resource_check()

    # Add Qdrant check if client provided
    if qdrant_client:
        health_manager.add_qdrant_check(qdrant_client)

    # Add Redis check if URL provided
    if redis_url:
        health_manager.add_redis_check(redis_url)

    # Add external service checks
    if external_services:
        for service_name, service_url in external_services.items():
            health_manager.add_http_check(url=service_url, name=service_name)

    return health_manager
