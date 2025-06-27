"""Health checking system for auto-detected services with monitoring integration.

Provides comprehensive health monitoring for:
- Service availability and response times
- Connection pool health and utilization
- Circuit breaker status and recovery
- Performance metrics and alerting thresholds
"""

import asyncio
import contextlib
import logging
import time
from typing import Any

from pydantic import BaseModel

from src.config.auto_detect import AutoDetectionConfig, DetectedService
from src.services.errors import circuit_breaker


logger = logging.getLogger(__name__)


class HealthCheckResult(BaseModel):
    """Result of a health check operation."""

    service_name: str
    is_healthy: bool
    response_time_ms: float
    status_code: int | None = None
    error_message: str | None = None
    metadata: dict[str, Any] = {}
    timestamp: float


class HealthSummary(BaseModel):
    """Summary of all health checks."""

    total_services: int
    healthy_services: int
    unhealthy_services: int
    overall_health_score: float
    average_response_time_ms: float
    last_check_timestamp: float
    service_results: list[HealthCheckResult]


class HealthChecker:
    """Performs health checks on auto-detected services."""

    def __init__(self, config: AutoDetectionConfig):
        self.config = config
        self.logger = logger.getChild("health")
        self._check_history: dict[str, list[HealthCheckResult]] = {}
        self._check_intervals = {
            "redis": 30,  # seconds
            "qdrant": 30,
            "postgresql": 60,
        }
        self._health_check_task: asyncio.Task | None = None
        self._running = False

    async def start_monitoring(self, services: list[DetectedService]) -> None:
        """Start continuous health monitoring for services."""
        if self._running:
            return

        self._services = services
        self._running = True

        # Initialize check history
        for service in services:
            self._check_history[service.service_name] = []

        # Start background monitoring task
        self._health_check_task = asyncio.create_task(self._monitor_loop())

        self.logger.info(f"Health monitoring started for {len(services)} services")

    async def stop_monitoring(self) -> None:
        """Stop health monitoring."""
        self._running = False

        if self._health_check_task:
            self._health_check_task.cancel()
            with contextlib.suppress(asyncio.CancelledError):
                await self._health_check_task

        self.logger.info("Health monitoring stopped")

    async def check_all_services(self) -> HealthSummary:
        """Perform health checks on all services."""
        if not hasattr(self, "_services"):
            return HealthSummary(
                total_services=0,
                healthy_services=0,
                unhealthy_services=0,
                overall_health_score=0.0,
                average_response_time_ms=0.0,
                last_check_timestamp=time.time(),
                service_results=[],
            )

        results = []
        start_time = time.time()

        # Run health checks in parallel
        check_tasks = [self.check_service(service) for service in self._services]

        if check_tasks:
            results = await asyncio.gather(*check_tasks, return_exceptions=True)

            # Filter out exceptions and convert to HealthCheckResult
            valid_results = []
            for result in results:
                if isinstance(result, HealthCheckResult):
                    valid_results.append(result)
                    # Store in history
                    self._store_check_result(result)

        # Calculate summary metrics
        healthy_count = sum(1 for r in valid_results if r.is_healthy)
        total_count = len(valid_results)

        overall_health_score = healthy_count / total_count if total_count > 0 else 0.0

        avg_response_time = (
            sum(r.response_time_ms for r in valid_results) / total_count
            if total_count > 0
            else 0.0
        )

        summary = HealthSummary(
            total_services=total_count,
            healthy_services=healthy_count,
            unhealthy_services=total_count - healthy_count,
            overall_health_score=overall_health_score,
            average_response_time_ms=avg_response_time,
            last_check_timestamp=time.time(),
            service_results=valid_results,
        )

        check_duration = (time.time() - start_time) * 1000
        self.logger.info(
            f"Health check completed: {healthy_count}/{total_count} healthy "
            f"(score: {overall_health_score:.2f}) in {check_duration:.1f}ms"
        )

        return summary

    @circuit_breaker(
        service_name="health_check",
        failure_threshold=5,
        recovery_timeout=60.0,
    )
    async def check_service(self, service: DetectedService) -> HealthCheckResult:
        """Perform health check on a specific service."""
        start_time = time.time()

        try:
            if service.service_type == "redis":
                return await self._check_redis_health(service, start_time)
            elif service.service_type == "qdrant":
                return await self._check_qdrant_health(service, start_time)
            elif service.service_type == "postgresql":
                return await self._check_postgresql_health(service, start_time)
            else:
                return await self._check_generic_health(service, start_time)

        except Exception as e:
            response_time_ms = (time.time() - start_time) * 1000

            self.logger.warning("Health check failed for {service.service_name}")

            return HealthCheckResult(
                service_name=service.service_name,
                is_healthy=False,
                response_time_ms=response_time_ms,
                error_message=str(e),
                timestamp=time.time(),
                metadata={"service_type": service.service_type},
            )

    async def _check_redis_health(
        self, service: DetectedService, start_time: float
    ) -> HealthCheckResult:
        """Check Redis service health."""
        try:
            import redis.asyncio as redis

            client = redis.Redis(host=service.host, port=service.port, socket_timeout=5)

            # Perform PING command
            await client.ping()

            # Get basic info
            info = await client.info()

            await client.aclose()

            response_time_ms = (time.time() - start_time) * 1000

            return HealthCheckResult(
                service_name=service.service_name,
                is_healthy=True,
                response_time_ms=response_time_ms,
                timestamp=time.time(),
                metadata={
                    "service_type": "redis",
                    "connected_clients": info.get("connected_clients", 0),
                    "used_memory_human": info.get("used_memory_human", "unknown"),
                    "redis_version": info.get("redis_version", "unknown"),
                },
            )

        except ImportError:
            raise RuntimeError("redis package not available")
        except Exception as e:
            raise RuntimeError("Redis health check failed")

    async def _check_qdrant_health(
        self, service: DetectedService, start_time: float
    ) -> HealthCheckResult:
        """Check Qdrant service health."""
        try:
            import httpx

            async with httpx.AsyncClient(timeout=5.0) as client:
                # Use HTTP health endpoint
                health_url = (
                    service.health_check_url
                    or f"http://{service.host}:{service.port}/health"
                )
                response = await client.get(health_url)

                is_healthy = response.status_code == 200
                response_time_ms = (time.time() - start_time) * 1000

                metadata = {
                    "service_type": "qdrant",
                    "status_code": response.status_code,
                }

                if is_healthy:
                    try:
                        health_data = response.json()
                        metadata.update(health_data)
                    except Exception as e:
                        logger.debug(
                            "Failed to parse health data from {service.service_name}"
                        )

                return HealthCheckResult(
                    service_name=service.service_name,
                    is_healthy=is_healthy,
                    response_time_ms=response_time_ms,
                    status_code=response.status_code,
                    timestamp=time.time(),
                    metadata=metadata,
                )

        except Exception as e:
            raise RuntimeError("Qdrant health check failed")

    async def _check_postgresql_health(
        self, service: DetectedService, start_time: float
    ) -> HealthCheckResult:
        """Check PostgreSQL service health."""
        try:
            # Simple TCP connection test for now
            reader, writer = await asyncio.wait_for(
                asyncio.open_connection(service.host, service.port), timeout=5.0
            )

            writer.close()
            await writer.wait_closed()

            response_time_ms = (time.time() - start_time) * 1000

            return HealthCheckResult(
                service_name=service.service_name,
                is_healthy=True,
                response_time_ms=response_time_ms,
                timestamp=time.time(),
                metadata={
                    "service_type": "postgresql",
                    "connection_test": "tcp_success",
                },
            )

        except Exception as e:
            raise RuntimeError("PostgreSQL health check failed")

    async def _check_generic_health(
        self, service: DetectedService, start_time: float
    ) -> HealthCheckResult:
        """Generic health check using HTTP or TCP."""
        try:
            if service.health_check_url:
                # HTTP health check
                import httpx

                async with httpx.AsyncClient(timeout=5.0) as client:
                    response = await client.get(service.health_check_url)

                    is_healthy = response.status_code == 200
                    response_time_ms = (time.time() - start_time) * 1000

                    return HealthCheckResult(
                        service_name=service.service_name,
                        is_healthy=is_healthy,
                        response_time_ms=response_time_ms,
                        status_code=response.status_code,
                        timestamp=time.time(),
                        metadata={"service_type": service.service_type},
                    )
            else:
                # TCP connection test
                reader, writer = await asyncio.wait_for(
                    asyncio.open_connection(service.host, service.port), timeout=5.0
                )

                writer.close()
                await writer.wait_closed()

                response_time_ms = (time.time() - start_time) * 1000

                return HealthCheckResult(
                    service_name=service.service_name,
                    is_healthy=True,
                    response_time_ms=response_time_ms,
                    timestamp=time.time(),
                    metadata={
                        "service_type": service.service_type,
                        "check_type": "tcp",
                    },
                )

        except Exception as e:
            raise RuntimeError("Generic health check failed")

    async def _monitor_loop(self) -> None:
        """Background monitoring loop."""
        while self._running:
            try:
                # Perform health checks
                summary = await self.check_all_services()

                # Log health status
                if summary.overall_health_score < 0.8:
                    self.logger.warning(
                        f"Health degraded: {summary.healthy_services}/{summary.total_services} "
                        f"services healthy (score: {summary.overall_health_score:.2f})"
                    )

                # Wait for next check interval
                await asyncio.sleep(30)  # Check every 30 seconds

            except asyncio.CancelledError:
                break
            except Exception as e:
                self.logger.exception("Health monitoring error")
                await asyncio.sleep(60)  # Wait longer on error

    def _store_check_result(self, result: HealthCheckResult) -> None:
        """Store health check result in history."""
        service_history = self._check_history.get(result.service_name, [])
        service_history.append(result)

        # Keep only last 100 results per service
        if len(service_history) > 100:
            service_history.pop(0)

        self._check_history[result.service_name] = service_history

    def get_service_history(
        self, service_name: str, limit: int = 10
    ) -> list[HealthCheckResult]:
        """Get recent health check history for a service."""
        history = self._check_history.get(service_name, [])
        return history[-limit:] if history else []

    def get_service_uptime(
        self, service_name: str, time_window_seconds: int = 3600
    ) -> float:
        """Calculate service uptime percentage over a time window."""
        history = self._check_history.get(service_name, [])
        if not history:
            return 0.0

        cutoff_time = time.time() - time_window_seconds
        recent_checks = [r for r in history if r.timestamp >= cutoff_time]

        if not recent_checks:
            return 0.0

        healthy_checks = sum(1 for r in recent_checks if r.is_healthy)
        return healthy_checks / len(recent_checks)

    def get_health_trends(self) -> dict[str, Any]:
        """Get health trends and statistics."""
        trends = {}

        for service_name, history in self._check_history.items():
            if not history:
                continue

            recent_history = history[-20:]  # Last 20 checks

            avg_response_time = sum(r.response_time_ms for r in recent_history) / len(
                recent_history
            )
            uptime_1h = self.get_service_uptime(service_name, 3600)  # 1 hour
            uptime_24h = self.get_service_uptime(service_name, 86400)  # 24 hours

            trends[service_name] = {
                "average_response_time_ms": avg_response_time,
                "uptime_1h": uptime_1h,
                "uptime_24h": uptime_24h,
                "total_checks": len(history),
                "last_check": recent_history[-1].timestamp if recent_history else None,
            }

        return trends
