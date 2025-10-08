"""Health check system for monitoring service dependencies and application health.

This module provides health checking for all system dependencies
including Qdrant, Redis, external APIs, and internal services.
"""

import asyncio
import logging
import time
from abc import ABC, abstractmethod
from enum import Enum
from typing import TYPE_CHECKING, Any
from urllib.parse import urlparse

import aiohttp
import redis.asyncio as redis
from pydantic import BaseModel, Field
from qdrant_client import AsyncQdrantClient
from qdrant_client.http.exceptions import UnexpectedResponse

from .metrics import MetricsRegistry


if TYPE_CHECKING:
    from src.config.loader import Settings


# Optional dependencies
try:
    import psutil
except ImportError:
    psutil = None


logger = logging.getLogger(__name__)


class HealthCheckConfig(BaseModel):
    """Configuration for health check system."""

    enabled: bool = Field(default=True, description="Enable health checks")
    interval: float = Field(
        default=30.0, description="Health check interval in seconds"
    )
    timeout: float = Field(default=10.0, description="Health check timeout in seconds")
    max_retries: int = Field(default=3, description="Maximum retry attempts")
    qdrant_url: str | None = Field(
        default=None, description="Qdrant URL for health checks"
    )
    redis_url: str | None = Field(
        default=None, description="Redis URL for health checks"
    )

    @classmethod
    def from_unified_config(cls, settings: "Settings") -> "HealthCheckConfig":
        """Build health-check configuration from the unified settings model."""

        redis_enabled = getattr(settings.cache, "enable_redis_cache", False) or getattr(  # type: ignore[attr-defined]
            settings.cache, "enable_dragonfly_cache", False
        )
        redis_url: str | None = None
        if redis_enabled:
            redis_url = getattr(settings.cache, "redis_url", None)
            dragonfly_url = getattr(settings.cache, "dragonfly_url", None)
            if (
                getattr(settings.cache, "enable_dragonfly_cache", False)
                and dragonfly_url
            ):
                redis_url = dragonfly_url

        return cls(
            enabled=settings.monitoring.enable_health_checks,
            interval=settings.monitoring.system_metrics_interval,
            timeout=settings.monitoring.health_check_timeout,
            max_retries=3,
            qdrant_url=settings.qdrant.url,
            redis_url=redis_url,
        )


class HealthStatus(str, Enum):
    """Health check status enumeration."""

    HEALTHY = "healthy"
    UNHEALTHY = "unhealthy"
    DEGRADED = "degraded"
    UNKNOWN = "unknown"


class HealthCheckResult(BaseModel):
    """Result of a health check operation."""

    name: str = Field(description="Name of the health check")
    status: HealthStatus = Field(description="Health status")
    message: str = Field(description="Human-readable status message")
    timestamp: float = Field(default_factory=time.time, description="Check timestamp")
    duration_ms: float = Field(description="Check duration in milliseconds")
    metadata: dict[str, Any] = Field(
        default_factory=dict, description="Additional metadata"
    )


class HealthCheck(ABC):
    """Abstract base class for health checks."""

    def __init__(self, name: str, timeout_seconds: float = 5.0):
        """Initialize health check.

        Args:
            name: Name of the health check
            timeout_seconds: Timeout for health check operation
        """

        self.name = name
        self.timeout_seconds = timeout_seconds

    @abstractmethod
    async def check(self) -> HealthCheckResult:
        """Perform the health check."""

    async def _execute_with_timeout(self, coro) -> HealthCheckResult:
        """Execute health check with timeout.

        Args:
            coro: Coroutine to execute

        Returns:
            Health check result
        """

        start_time = time.time()

        try:
            result = await asyncio.wait_for(coro, timeout=self.timeout_seconds)
        except TimeoutError:
            duration_ms = (time.time() - start_time) * 1000
            return HealthCheckResult(
                name=self.name,
                status=HealthStatus.UNHEALTHY,
                message=f"Health check timed out after {self.timeout_seconds}s",
                duration_ms=duration_ms,
            )
        except (ValueError, TypeError, AttributeError) as e:
            duration_ms = (time.time() - start_time) * 1000
            return HealthCheckResult(
                name=self.name,
                status=HealthStatus.UNHEALTHY,
                message=f"Health check failed: {e!s}",
                duration_ms=duration_ms,
            )

        duration_ms = (time.time() - start_time) * 1000
        result.duration_ms = duration_ms
        return result


class QdrantHealthCheck(HealthCheck):
    """Health check for Qdrant vector database."""

    def __init__(
        self,
        client: AsyncQdrantClient,
        name: str = "qdrant",
        timeout_seconds: float = 5.0,
    ):
        """Initialize Qdrant health check.

        Args:
            client: Qdrant client instance
            name: Name of the health check
            timeout_seconds: Timeout for health check
        """

        super().__init__(name, timeout_seconds)
        self.client = client

    async def check(self) -> HealthCheckResult:
        """Check Qdrant health by retrieving collection metadata."""

        return await self._execute_with_timeout(self._perform_qdrant_check())

    async def _perform_qdrant_check(self) -> HealthCheckResult:
        """Perform the actual Qdrant health check."""

        try:
            cluster_info = await self._get_cluster_info()
            return self._evaluate_cluster_health(cluster_info)
        except UnexpectedResponse as e:
            return self._create_error_result(f"Qdrant API error: {e!s}")
        except (ValueError, ConnectionError, TimeoutError, RuntimeError) as e:
            return self._create_error_result(f"Qdrant connection failed: {e!s}")

    async def _get_cluster_info(self):
        """Fetch collection overview from Qdrant.

        Returns:
            Collections response containing registered collections
        """

        return await self.client.get_collections()

    def _evaluate_cluster_health(self, cluster_info) -> HealthCheckResult:
        """Evaluate Qdrant health based on retrieved metadata.

        Args:
            cluster_info: Collections response from Qdrant

        Returns:
            Health check result
        """

        collections = getattr(cluster_info, "collections", None)
        if collections is not None:
            collection_names = [
                getattr(collection, "name", "unknown") for collection in collections
            ]
            return HealthCheckResult(
                name=self.name,
                status=HealthStatus.HEALTHY,
                message="Qdrant service is operational",
                duration_ms=0.0,
                metadata={
                    "collection_count": len(collection_names),
                    "collections": collection_names,
                },
            )
        return HealthCheckResult(
            name=self.name,
            status=HealthStatus.DEGRADED,
            message="Qdrant responded without collection metadata",
            duration_ms=0.0,
        )

    def _create_error_result(self, message: str) -> HealthCheckResult:
        """Create an error health check result.

        Args:
            message: Error message

        Returns:
            Unhealthy health check result
        """

        return HealthCheckResult(
            name=self.name,
            status=HealthStatus.UNHEALTHY,
            message=message,
            duration_ms=0.0,
        )


class RedisHealthCheck(HealthCheck):
    """Health check for Redis cache."""

    def __init__(
        self, redis_url: str, name: str = "redis", timeout_seconds: float = 5.0
    ):
        """Initialize Redis health check.

        Args:
            redis_url: Redis connection URL
            name: Name of the health check
            timeout_seconds: Timeout for health check
        """

        super().__init__(name, timeout_seconds)
        self.redis_url = redis_url

    async def check(self) -> HealthCheckResult:
        """Check Redis health by pinging the server."""

        return await self._execute_with_timeout(self._perform_redis_check())

    async def _perform_redis_check(self) -> HealthCheckResult:
        """Perform the actual Redis health check."""

        redis_client = None
        try:
            redis_client = await self._create_redis_client()
            return await self._check_redis_connectivity(redis_client)
        except redis.ConnectionError as e:
            return self._create_redis_error_result(f"Redis connection failed: {e!s}")
        except (redis.RedisError, ConnectionError, TimeoutError, ValueError) as e:
            return self._create_redis_error_result(f"Redis health check error: {e!s}")
        finally:
            await self._cleanup_redis_client(redis_client)

    async def _create_redis_client(self):
        """Create Redis client.

        Returns:
            Redis client instance
        """

        return redis.from_url(self.redis_url)

    async def _check_redis_connectivity(self, redis_client) -> HealthCheckResult:
        """Check Redis connectivity and gather info.

        Args:
            redis_client: Redis client instance

        Returns:
            Health check result
        """

        pong = await redis_client.ping()
        if not pong:
            return self._create_redis_error_result("Redis ping failed")

        info = await redis_client.info()
        return HealthCheckResult(
            name=self.name,
            status=HealthStatus.HEALTHY,
            message="Redis server is responding",
            duration_ms=0.0,
            metadata={
                "redis_version": info.get("redis_version", "unknown"),
                "connected_clients": info.get("connected_clients", 0),
                "used_memory_human": info.get("used_memory_human", "unknown"),
            },
        )

    def _create_redis_error_result(self, message: str) -> HealthCheckResult:
        """Create Redis error result.

        Args:
            message: Error message

        Returns:
            Unhealthy health check result
        """

        return HealthCheckResult(
            name=self.name,
            status=HealthStatus.UNHEALTHY,
            message=message,
            duration_ms=0.0,
        )

    async def _cleanup_redis_client(self, redis_client) -> None:
        """Clean up Redis client.

        Args:
            redis_client: Redis client to clean up
        """

        if redis_client:
            await redis_client.aclose()


class HTTPHealthCheck(HealthCheck):
    """Health check for HTTP endpoints."""

    def __init__(
        self,
        url: str,
        *,
        expected_status: int = 200,
        name: str | None = None,
        timeout_seconds: float = 5.0,
        headers: dict[str, str] | None = None,
    ):
        """Initialize HTTP health check.

        Args:
            url: URL to check
            expected_status: Expected HTTP status code
            name: Name of the health check (defaults to URL host)
            timeout_seconds: Timeout for health check
            headers: Optional HTTP headers
        """
        # pylint: disable=too-many-arguments

        if name is None:
            parsed = urlparse(url)
            name = f"http_{parsed.netloc}"

        super().__init__(name, timeout_seconds)
        self.url = url
        self.expected_status = expected_status
        self.headers = headers or {}

    async def check(self) -> HealthCheckResult:
        """Check HTTP endpoint health."""

        return await self._execute_with_timeout(self._perform_http_check())

    async def _perform_http_check(self) -> HealthCheckResult:
        """Perform the actual HTTP health check."""

        try:
            return await self._make_http_request()
        except aiohttp.ClientError as e:
            return self._create_http_error_result(f"HTTP request failed: {e!s}")
        except ConnectionError as e:
            return self._create_http_error_result(f"HTTP health check error: {e!s}")

    async def _make_http_request(self) -> HealthCheckResult:
        """Make HTTP request and evaluate response."""

        timeout = aiohttp.ClientTimeout(total=self.timeout_seconds)
        async with (
            aiohttp.ClientSession(timeout=timeout) as session,
            session.get(self.url, headers=self.headers) as response,
        ):
            return self._evaluate_http_response(response)

    def _evaluate_http_response(self, response) -> HealthCheckResult:
        """Evaluate HTTP response status.

        Args:
            response: HTTP response object

        Returns:
            Health check result
        """

        if response.status == self.expected_status:
            return HealthCheckResult(
                name=self.name,
                status=HealthStatus.HEALTHY,
                message=f"HTTP endpoint responding with status {response.status}",
                duration_ms=0.0,
                metadata={
                    "status_code": response.status,
                    "content_type": response.headers.get("content-type", "unknown"),
                },
            )
        return HealthCheckResult(
            name=self.name,
            status=HealthStatus.UNHEALTHY,
            message=(
                f"HTTP endpoint returned status {response.status}, "
                f"expected {self.expected_status}"
            ),
            duration_ms=0.0,
            metadata={"status_code": response.status},
        )

    def _create_http_error_result(self, message: str) -> HealthCheckResult:
        """Create HTTP error result.

        Args:
            message: Error message

        Returns:
            Unhealthy health check result
        """

        return HealthCheckResult(
            name=self.name,
            status=HealthStatus.UNHEALTHY,
            message=message,
            duration_ms=0.0,
        )


class SystemResourceHealthCheck(HealthCheck):
    """Health check for system resources (CPU, memory, disk)."""

    def __init__(
        self,
        *,
        name: str = "system_resources",
        cpu_threshold: float = 90.0,
        memory_threshold: float = 90.0,
        disk_threshold: float = 90.0,
        timeout_seconds: float = 5.0,
    ):
        """Initialize system resource health check.

        Args:
            name: Name of the health check
            cpu_threshold: CPU usage threshold percentage
            memory_threshold: Memory usage threshold percentage
            disk_threshold: Disk usage threshold percentage
            timeout_seconds: Timeout for health check
        """
        # pylint: disable=too-many-arguments

        super().__init__(name, timeout_seconds)
        self.cpu_threshold = cpu_threshold
        self.memory_threshold = memory_threshold
        self.disk_threshold = disk_threshold

    async def check(self) -> HealthCheckResult:
        """Check system resource health."""

        return await self._execute_with_timeout(self._perform_system_check())

    async def _perform_system_check(self) -> HealthCheckResult:
        """Perform the actual system resource check."""

        if psutil is None:
            return self._create_psutil_unavailable_result()

        try:
            resource_metrics = await self._gather_resource_metrics()
            return self._evaluate_system_health(resource_metrics)
        except (ValueError, TypeError, AttributeError) as e:
            return self._create_system_error_result(
                f"System resource check failed: {e!s}"
            )

    def _create_psutil_unavailable_result(self) -> HealthCheckResult:
        """Create result when psutil is not available.

        Returns:
            Degraded health check result
        """

        return HealthCheckResult(
            name=self.name,
            status=HealthStatus.DEGRADED,
            message="psutil not available",
            duration_ms=0.0,
            metadata={"error": "psutil not available"},
        )

    async def _gather_resource_metrics(self) -> dict:
        """Gather system resource metrics.

        Returns:
            Dictionary of resource metrics
        """

        if psutil is None:
            msg = "psutil is required to gather system resource metrics"
            raise RuntimeError(msg)

        assert psutil is not None
        cpu_percent = psutil.cpu_percent(interval=1)
        memory = psutil.virtual_memory()
        disk = psutil.disk_usage("/")

        return {
            "cpu_percent": cpu_percent,
            "memory_percent": memory.percent,
            "disk_percent": (disk.used / disk.total) * 100,
            "memory": memory,
            "disk": disk,
        }

    def _evaluate_system_health(self, metrics: dict) -> HealthCheckResult:
        """Evaluate system health based on metrics.

        Args:
            metrics: System resource metrics

        Returns:
            Health check result
        """

        issues = []
        status = HealthStatus.HEALTHY

        # Check each resource threshold
        if metrics["cpu_percent"] > self.cpu_threshold:
            issues.append(f"High CPU usage: {metrics['cpu_percent']:.1f}%")
            status = HealthStatus.DEGRADED

        if metrics["memory_percent"] > self.memory_threshold:
            issues.append(f"High memory usage: {metrics['memory_percent']:.1f}%")
            status = HealthStatus.DEGRADED

        if metrics["disk_percent"] > self.disk_threshold:
            issues.append(f"High disk usage: {metrics['disk_percent']:.1f}%")
            status = HealthStatus.DEGRADED

        # Multiple issues indicate unhealthy system
        if len(issues) >= 2:
            status = HealthStatus.UNHEALTHY

        message = "System resources healthy" if not issues else "; ".join(issues)

        return HealthCheckResult(
            name=self.name,
            status=status,
            message=message,
            duration_ms=0.0,
            metadata={
                "cpu_percent": metrics["cpu_percent"],
                "memory_percent": metrics["memory_percent"],
                "disk_percent": metrics["disk_percent"],
                "memory_available_gb": metrics["memory"].available / (1024**3),
                "disk_free_gb": metrics["disk"].free / (1024**3),
            },
        )

    def _create_system_error_result(self, message: str) -> HealthCheckResult:
        """Create system error result.

        Args:
            message: Error message

        Returns:
            Unhealthy health check result
        """

        return HealthCheckResult(
            name=self.name,
            status=HealthStatus.UNHEALTHY,
            message=message,
            duration_ms=0.0,
        )


class HealthCheckManager:
    """Manager for coordinating multiple health checks."""

    def __init__(
        self, config: HealthCheckConfig, metrics_registry: MetricsRegistry | None = None
    ):
        """Initialize health check manager.

        Args:
            config: Health check configuration
            metrics_registry: Optional metrics registry for recording health status

        """
        self.config = config
        self.health_checks: list[HealthCheck] = []
        self.metrics_registry = metrics_registry
        self._last_results: dict[str, HealthCheckResult] = {}

        # Default health checks are added explicitly via helper methods.

    def add_health_check(self, health_check: HealthCheck) -> None:
        """Add a health check to the manager.

        Args:
            health_check: Health check to add
        """

        self.health_checks.append(health_check)

    def add_qdrant_check(
        self, client: AsyncQdrantClient, timeout_seconds: float = 5.0
    ) -> None:
        """Add Qdrant health check.

        Args:
            client: Qdrant client instance
            timeout_seconds: Timeout for health check
        """

        self.add_health_check(
            QdrantHealthCheck(client, timeout_seconds=timeout_seconds)
        )

    def add_redis_check(self, redis_url: str, timeout_seconds: float = 5.0) -> None:
        """Add Redis health check.

        Args:
            redis_url: Redis connection URL
            timeout_seconds: Timeout for health check
        """

        self.add_health_check(
            RedisHealthCheck(redis_url, timeout_seconds=timeout_seconds)
        )

    def add_http_check(
        self,
        url: str,
        expected_status: int = 200,
        name: str | None = None,
        timeout_seconds: float = 5.0,
    ) -> None:
        """Add HTTP endpoint health check.

        Args:
            url: URL to check
            expected_status: Expected HTTP status code
            name: Optional name for the check
            timeout_seconds: Timeout for health check
        """

        self.add_health_check(
            HTTPHealthCheck(
                url,
                expected_status=expected_status,
                name=name,
                timeout_seconds=timeout_seconds,
            )
        )

    def add_system_resource_check(
        self,
        cpu_threshold: float = 90.0,
        memory_threshold: float = 90.0,
        disk_threshold: float = 90.0,
        timeout_seconds: float = 5.0,
    ) -> None:
        """Add system resource health check.

        Args:
            cpu_threshold: CPU usage threshold percentage
            memory_threshold: Memory usage threshold percentage
            disk_threshold: Disk usage threshold percentage
            timeout_seconds: Timeout for health check
        """

        self.add_health_check(
            SystemResourceHealthCheck(
                cpu_threshold=cpu_threshold,
                memory_threshold=memory_threshold,
                disk_threshold=disk_threshold,
                timeout_seconds=timeout_seconds,
            )
        )

    async def check_all(self) -> dict[str, HealthCheckResult]:
        """Run all health checks concurrently.

        Returns:
            Dictionary mapping check names to results
        """

        if not self.health_checks:
            return {}

        # Run all health checks concurrently
        tasks = [check.check() for check in self.health_checks]
        results = await asyncio.gather(*tasks, return_exceptions=True)

        # Process results
        health_results: dict[str, HealthCheckResult] = {}
        for check, result in zip(self.health_checks, results, strict=False):
            if isinstance(result, BaseException):
                # Handle unexpected exceptions
                health_result = HealthCheckResult(
                    name=check.name,
                    status=HealthStatus.UNHEALTHY,
                    message=f"Health check failed with exception: {result!s}",
                    duration_ms=0,
                )
            else:
                assert isinstance(result, HealthCheckResult)
                health_result = result

            health_results[check.name] = health_result
            self._last_results[check.name] = health_result

            # Update metrics if registry is available
            if self.metrics_registry:
                is_healthy = health_result.status == HealthStatus.HEALTHY
                self.metrics_registry.update_dependency_health(check.name, is_healthy)

        return health_results

    async def check_single(self, check_name: str) -> HealthCheckResult | None:
        """Run a single health check by name.

        Args:
            check_name: Name of the health check to run

        Returns:
            Health check result or None if check not found
        """

        for check in self.health_checks:
            if check.name == check_name:
                result = await check.check()
                self._last_results[check_name] = result

                # Update metrics
                if self.metrics_registry:
                    is_healthy = result.status == HealthStatus.HEALTHY
                    self.metrics_registry.update_dependency_health(
                        check_name, is_healthy
                    )

                return result

        return None

    def get_last_results(self) -> dict[str, HealthCheckResult]:
        """Get the last health check results.

        Returns:
            Dictionary of last health check results
        """

        return self._last_results.copy()

    def get_overall_status(self) -> HealthStatus:
        """Get overall system health status.

        Returns:
            Overall health status based on all checks
        """

        if not self._last_results:
            return HealthStatus.UNKNOWN

        statuses = [result.status for result in self._last_results.values()]

        if any(status == HealthStatus.UNHEALTHY for status in statuses):
            return HealthStatus.UNHEALTHY
        if any(status == HealthStatus.DEGRADED for status in statuses):
            return HealthStatus.DEGRADED
        if all(status == HealthStatus.HEALTHY for status in statuses):
            return HealthStatus.HEALTHY
        return HealthStatus.UNKNOWN

    def get_health_summary(self) -> dict[str, Any]:
        """Get comprehensive health summary.

        Returns:
            Health summary including overall status and individual check results
        """

        return {
            "overall_status": self.get_overall_status(),
            "timestamp": time.time(),
            "checks": self.get_last_results(),
            "healthy_count": sum(
                1
                for result in self._last_results.values()
                if result.status == HealthStatus.HEALTHY
            ),
            "total_count": len(self._last_results),
        }
