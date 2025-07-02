"""Monitoring manager for observability coordination."""

import asyncio
import contextlib
import logging
import time
from typing import TYPE_CHECKING, Any

from dependency_injector.wiring import Provide, inject

from src.infrastructure.container import ApplicationContainer
from src.infrastructure.shared import ClientHealth, ClientState
from src.services.monitoring.metrics import get_metrics_registry
from src.services.monitoring.performance_monitor import RealTimePerformanceMonitor


if TYPE_CHECKING:
    from src.config import Config

logger = logging.getLogger(__name__)


class MonitoringManager:
    """Focused manager for observability coordination.

    Handles health checks, metrics collection, performance monitoring,
    and observability coordination with single responsibility.
    """

    def __init__(self):
        """Initialize monitoring manager."""
        self._health_checks: dict[str, ClientHealth] = {}
        self._metrics_registry: Any | None = None
        self._performance_monitor: Any | None = None
        self._health_check_task: asyncio.Task | None = None
        self._initialized = False
        self._config: Config | None = None

    @inject
    async def initialize(
        self,
        config: "Config" = Provide[ApplicationContainer.config],
    ) -> None:
        """Initialize monitoring manager using dependency injection.

        Args:
            config: Configuration from DI container
        """
        if self._initialized:
            return

        self._config = config

        # Initialize metrics registry
        try:
            self._metrics_registry = get_metrics_registry()
            logger.info("Monitoring metrics registry initialized")
        except ImportError:
            logger.warning("Monitoring metrics registry not available")

        # Initialize performance monitor
        try:
            self._performance_monitor = RealTimePerformanceMonitor()
            logger.info("Performance monitor initialized")
        except ImportError:
            logger.warning("Performance monitor not available")

        # Start health check loop
        self._health_check_task = asyncio.create_task(self._health_check_loop())

        self._initialized = True
        logger.info("MonitoringManager initialized with observability components")

    async def cleanup(self) -> None:
        """Cleanup monitoring manager resources."""
        # Cancel health check task
        if self._health_check_task and not self._health_check_task.done():
            self._health_check_task.cancel()
            with contextlib.suppress(asyncio.CancelledError):
                await self._health_check_task

        # Cleanup performance monitor
        if self._performance_monitor and hasattr(self._performance_monitor, "cleanup"):
            await self._performance_monitor.cleanup()

        self._health_checks.clear()
        self._metrics_registry = None
        self._performance_monitor = None
        self._health_check_task = None
        self._initialized = False
        logger.info("MonitoringManager cleaned up")

    # Health Check Operations
    def register_health_check(
        self, service_name: str, check_function: callable, check_interval: int = 30
    ) -> None:
        """Register a health check for a service.

        Args:
            service_name: Name of the service
            check_function: Async function that returns bool for health status
            check_interval: Check interval in seconds
        """
        self._health_checks[service_name] = ClientHealth(
            state=ClientState.HEALTHY,
            last_check=time.time(),
            check_function=check_function,
            check_interval=check_interval,
        )
        logger.info(
            f"Registered health check for {service_name}"
        )  # TODO: Convert f-string to logging format

    def unregister_health_check(self, service_name: str) -> None:
        """Unregister a health check for a service.

        Args:
            service_name: Name of the service
        """
        if service_name in self._health_checks:
            del self._health_checks[service_name]
            logger.info(
                f"Unregistered health check for {service_name}"
            )  # TODO: Convert f-string to logging format

    async def check_service_health(self, service_name: str) -> bool:
        """Check health of a specific service.

        Args:
            service_name: Name of the service to check

        Returns:
            True if healthy, False otherwise
        """
        if service_name not in self._health_checks:
            logger.warning(
                f"No health check registered for {service_name}"
            )  # TODO: Convert f-string to logging format
            return False

        health = self._health_checks[service_name]

        try:
            if hasattr(health, "check_function") and health.check_function:
                is_healthy = await health.check_function()

                health.last_check = time.time()
                if is_healthy:
                    health.state = ClientState.HEALTHY
                    health.consecutive_failures = 0
                    health.last_error = None
                else:
                    health.consecutive_failures += 1
                    health.state = (
                        ClientState.DEGRADED
                        if health.consecutive_failures < 3
                        else ClientState.FAILED
                    )
                    health.last_error = "Health check returned false"

                return is_healthy
        except Exception as e:
            logger.exception(
                "Health check failed for : {e}"
            )  # TODO: Convert f-string to logging format
            health.last_check = time.time()
            health.last_error = str(e)
            health.consecutive_failures += 1
            health.state = ClientState.FAILED
            return False

        return False

    async def get_health_status(self) -> dict[str, dict[str, Any]]:
        """Get health status of all monitored services.

        Returns:
            Dictionary mapping service names to health information
        """
        status = {}

        for service_name, health in self._health_checks.items():
            status[service_name] = {
                "state": health.state.value,
                "last_check": health.last_check,
                "last_error": health.last_error,
                "consecutive_failures": health.consecutive_failures,
                "is_healthy": health.state == ClientState.HEALTHY,
            }

        return status

    async def get_overall_health(self) -> dict[str, Any]:
        """Get overall system health summary.

        Returns:
            Overall health status with summary metrics
        """
        health_status = await self.get_health_status()

        total_services = len(health_status)
        healthy_services = sum(
            1 for status in health_status.values() if status["is_healthy"]
        )
        failed_services = sum(
            1 for status in health_status.values() if status["state"] == "failed"
        )

        overall_healthy = failed_services == 0 and healthy_services == total_services

        return {
            "overall_healthy": overall_healthy,
            "total_services": total_services,
            "healthy_services": healthy_services,
            "failed_services": failed_services,
            "health_percentage": (healthy_services / max(total_services, 1)) * 100,
            "services": health_status,
        }

    # Metrics Operations
    def record_metric(
        self, metric_name: str, value: float, labels: dict[str, str] | None = None
    ) -> None:
        """Record a metric value.

        Args:
            metric_name: Name of the metric
            value: Metric value
            labels: Optional labels for the metric
        """
        if not self._metrics_registry:
            return

        try:
            if hasattr(self._metrics_registry, "record_metric"):
                self._metrics_registry.record_metric(metric_name, value, labels or {})
        except (ValueError, TypeError, UnicodeDecodeError) as e:
            logger.warning(
                f"Failed to record metric {metric_name}: {e}"
            )  # TODO: Convert f-string to logging format

    def increment_counter(
        self, counter_name: str, labels: dict[str, str] | None = None
    ) -> None:
        """Increment a counter metric.

        Args:
            counter_name: Name of the counter
            labels: Optional labels for the counter
        """
        if not self._metrics_registry:
            return

        try:
            if hasattr(self._metrics_registry, "increment_counter"):
                self._metrics_registry.increment_counter(counter_name, labels or {})
        except (ValueError, TypeError, UnicodeDecodeError) as e:
            logger.warning(
                f"Failed to increment counter {counter_name}: {e}"
            )  # TODO: Convert f-string to logging format

    def record_histogram(
        self, histogram_name: str, value: float, labels: dict[str, str] | None = None
    ) -> None:
        """Record a histogram value.

        Args:
            histogram_name: Name of the histogram
            value: Value to record
            labels: Optional labels for the histogram
        """
        if not self._metrics_registry:
            return

        try:
            if hasattr(self._metrics_registry, "record_histogram"):
                self._metrics_registry.record_histogram(
                    histogram_name, value, labels or {}
                )
        except (ValueError, TypeError, UnicodeDecodeError) as e:
            logger.warning(
                f"Failed to record histogram {histogram_name}: {e}"
            )  # TODO: Convert f-string to logging format

    # Performance Monitoring
    async def track_performance(
        self, operation_name: str, operation_func: callable, *args, **kwargs
    ) -> Any:
        """Track performance of an operation.

        Args:
            operation_name: Name of the operation
            operation_func: Function to execute and track
            *args: Arguments for the function
            **kwargs: Keyword arguments for the function

        Returns:
            Result of the operation function
        """
        start_time = time.time()

        try:
            result = await operation_func(*args, **kwargs)

            # Record success metrics
            duration_ms = (time.time() - start_time) * 1000
            self.record_histogram(f"{operation_name}_duration_ms", duration_ms)
            self.increment_counter(f"{operation_name}_total", {"status": "success"})

        except Exception:
            # Record failure metrics
            duration_ms = (time.time() - start_time) * 1000
            self.record_histogram(f"{operation_name}_duration_ms", duration_ms)
            self.increment_counter(f"{operation_name}_total", {"status": "error"})
            self.increment_counter(f"{operation_name}_errors")

            logger.exception(
                "Operation  failed: {e}"
            )  # TODO: Convert f-string to logging format
            raise

        else:
            return result

    def get_performance_metrics(self) -> dict[str, Any]:
        """Get performance metrics summary.

        Returns:
            Performance metrics data
        """
        if not self._performance_monitor:
            return {}

        try:
            if hasattr(self._performance_monitor, "get_metrics"):
                return self._performance_monitor.get_metrics()
        except (ValueError, TypeError, UnicodeDecodeError) as e:
            logger.warning(
                f"Failed to get performance metrics: {e}"
            )  # TODO: Convert f-string to logging format

        return {}

    # Observability Operations
    async def log_operation(
        self, operation: str, details: dict[str, Any], level: str = "info"
    ) -> None:
        """Log an operation with structured details.

        Args:
            operation: Name of the operation
            details: Operation details
            level: Log level (debug, info, warning, error)
        """
        log_data = {"operation": operation, "timestamp": time.time(), **details}

        if level == "debug":
            logger.debug(
                f"Operation: {operation}", extra=log_data
            )  # TODO: Convert f-string to logging format
        elif level == "info":
            logger.info(
                f"Operation: {operation}", extra=log_data
            )  # TODO: Convert f-string to logging format
        elif level == "warning":
            logger.warning(
                f"Operation: {operation}", extra=log_data
            )  # TODO: Convert f-string to logging format
        elif level == "error":
            logger.error(
                f"Operation: {operation}", extra=log_data
            )  # TODO: Convert f-string to logging format

    def create_span(self, span_name: str) -> Any:
        """Create a tracing span for distributed tracing.

        Args:
            span_name: Name of the span

        Returns:
            Span context manager
        """

        # Basic span implementation - can be extended with actual tracing
        class BasicSpan:
            def __init__(self, name: str):
                self.name = name
                self.start_time = time.time()

            def __enter__(self):
                logger.debug(
                    f"Starting span: {self.name}"
                )  # TODO: Convert f-string to logging format
                return self

            def __exit__(self, exc_type, exc_val, exc_tb):
                duration_ms = (time.time() - self.start_time) * 1000
                logger.debug(
                    f"Completed span: {self.name} in {duration_ms:.2f}ms"
                )  # TODO: Convert f-string to logging format

                if exc_type:
                    logger.error(
                        f"Span {self.name} failed: {exc_val}"
                    )  # TODO: Convert f-string to logging format

        return BasicSpan(span_name)

    async def get_status(self) -> dict[str, Any]:
        """Get monitoring manager status.

        Returns:
            Status information for all monitoring components
        """
        status = {
            "initialized": self._initialized,
            "health_checks": {
                "registered": len(self._health_checks),
                "services": list(self._health_checks.keys()),
            },
            "metrics_registry": {
                "available": self._metrics_registry is not None,
            },
            "performance_monitor": {
                "available": self._performance_monitor is not None,
            },
            "health_check_task": {
                "running": (
                    self._health_check_task is not None
                    and not self._health_check_task.done()
                ),
            },
        }

        # Get overall health
        try:
            status["overall_health"] = await self.get_overall_health()
        except (asyncio.CancelledError, TimeoutError, RuntimeError) as e:
            logger.warning(
                f"Failed to get overall health: {e}"
            )  # TODO: Convert f-string to logging format
            status["overall_health"] = {"error": str(e)}

        # Get performance metrics
        try:
            status["performance_metrics"] = self.get_performance_metrics()
        except (asyncio.CancelledError, TimeoutError, RuntimeError) as e:
            logger.warning(
                f"Failed to get performance metrics: {e}"
            )  # TODO: Convert f-string to logging format
            status["performance_metrics"] = {"error": str(e)}

        return status

    # Private Methods
    async def _health_check_loop(self) -> None:
        """Background task to periodically check service health."""
        while True:
            try:
                # Default health check interval from config or 30 seconds
                interval = 30
                if self._config and hasattr(self._config, "performance"):
                    interval = getattr(
                        self._config.performance, "health_check_interval", 30
                    )

                await asyncio.sleep(interval)

                # Run health checks for all registered services
                tasks = []
                for service_name in self._health_checks:
                    task = asyncio.create_task(
                        self.check_service_health(service_name),
                        name=f"health_check_{service_name}",
                    )
                    tasks.append(task)

                if tasks:
                    await asyncio.gather(*tasks, return_exceptions=True)

            except asyncio.CancelledError:
                logger.info("Health check loop cancelled")
                break
            except Exception:
                logger.exception("Error in health check loop")
                await asyncio.sleep(10)  # Brief pause before retry
