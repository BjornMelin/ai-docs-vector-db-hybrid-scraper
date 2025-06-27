"""Health check system for monitoring service dependencies and application health.

This module provides comprehensive health checking for all system dependencies
including Qdrant, Redis, external APIs, and internal services.
"""

import asyncio
import logging
import time
from abc import ABC, abstractmethod
from enum import Enum
from typing import Any

import aiohttp
import redis.asyncio as redis
from pydantic import BaseModel, Field
from qdrant_client import AsyncQdrantClient
from qdrant_client.http.exceptions import UnexpectedResponse

from .metrics import MetricsRegistry


logger = logging.getLogger(__name__)


class HealthCheckConfig(BaseModel):
    """Configuration for health check system."""

    enabled: bool = Field(default=True, description="Enable health checks")
    interval: float = Field(
        default=30.0, description="Health check interval in seconds"
    )
    timeout: float = Field(default=10.0, description="Health check timeout in seconds")
    max_retries: int = Field(default=3, description="Maximum retry attempts")
    qdrant_url: str = Field(
        default="http://localhost:6333", description="Qdrant URL for health checks"
    )
    redis_url: str = Field(
        default="redis://localhost:6379", description="Redis URL for health checks"
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
        """Perform the health check.

        Returns:
            Health check result
        """
        pass

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
            duration_ms = (time.time() - start_time) * 1000
            result.duration_ms = duration_ms
            return result

        except TimeoutError:
            duration_ms = (time.time() - start_time) * 1000
            return HealthCheckResult(
                name=self.name,
                status=HealthStatus.UNHEALTHY,
                message=f"Health check timed out after {self.timeout_seconds}s",
                duration_ms=duration_ms,
            )

        except Exception:
            duration_ms = (time.time() - start_time) * 1000
            return HealthCheckResult(
                name=self.name,
                status=HealthStatus.UNHEALTHY,
                message=f"Health check failed: {e!s}",
                duration_ms=duration_ms,
            )


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
        """Check Qdrant health by retrieving cluster info.

        Returns:
            Health check result
        """

        async def _check():
            try:
                # Try to get cluster info to verify connectivity
                cluster_info = await self.client.get_cluster_info()

                # Check if cluster is operational
                if cluster_info and hasattr(cluster_info, "status"):
                    return HealthCheckResult(
                        name=self.name,
                        status=HealthStatus.HEALTHY,
                        message="Qdrant cluster is operational",
                        duration_ms=0.0,  # Will be updated by _execute_with_timeout
                        metadata={
                            "cluster_status": str(cluster_info.status),
                            "peer_count": len(cluster_info.peers)
                            if hasattr(cluster_info, "peers")
                            else 0,
                        },
                    )
                else:
                    return HealthCheckResult(
                        name=self.name,
                        status=HealthStatus.DEGRADED,
                        message="Qdrant responding but cluster info unavailable",
                        duration_ms=0.0,  # Will be updated by _execute_with_timeout
                    )

            except UnexpectedResponse as e:
                return HealthCheckResult(
                    name=self.name,
                    status=HealthStatus.UNHEALTHY,
                    message=f"Qdrant API error: {e!s}",
                    duration_ms=0.0,  # Will be updated by _execute_with_timeout
                )

            except Exception:
                return HealthCheckResult(
                    name=self.name,
                    status=HealthStatus.UNHEALTHY,
                    message=f"Qdrant connection failed: {e!s}",
                    duration_ms=0.0,  # Will be updated by _execute_with_timeout
                )

        return await self._execute_with_timeout(_check())


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
        """Check Redis health by pinging the server.

        Returns:
            Health check result
        """

        async def _check():
            redis_client = None
            try:
                redis_client = redis.from_url(self.redis_url)

                # Ping Redis to verify connectivity
                pong = await redis_client.ping()
                if pong:
                    # Get server info for additional metadata
                    info = await redis_client.info()

                    return HealthCheckResult(
                        name=self.name,
                        status=HealthStatus.HEALTHY,
                        message="Redis server is responding",
                        duration_ms=0.0,  # Will be updated by _execute_with_timeout
                        metadata={
                            "redis_version": info.get("redis_version", "unknown"),
                            "connected_clients": info.get("connected_clients", 0),
                            "used_memory_human": info.get(
                                "used_memory_human", "unknown"
                            ),
                        },
                    )
                else:
                    return HealthCheckResult(
                        name=self.name,
                        status=HealthStatus.UNHEALTHY,
                        message="Redis ping failed",
                        duration_ms=0.0,  # Will be updated by _execute_with_timeout
                    )

            except redis.ConnectionError as e:
                return HealthCheckResult(
                    name=self.name,
                    status=HealthStatus.UNHEALTHY,
                    message=f"Redis connection failed: {e!s}",
                    duration_ms=0.0,  # Will be updated by _execute_with_timeout
                )

            except Exception:
                return HealthCheckResult(
                    name=self.name,
                    status=HealthStatus.UNHEALTHY,
                    message=f"Redis health check error: {e!s}",
                    duration_ms=0.0,  # Will be updated by _execute_with_timeout
                )

            finally:
                if redis_client:
                    await redis_client.aclose()

        return await self._execute_with_timeout(_check())


class HTTPHealthCheck(HealthCheck):
    """Health check for HTTP endpoints."""

    def __init__(
        self,
        url: str,
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
        if name is None:
            from urllib.parse import urlparse

            parsed = urlparse(url)
            name = f"http_{parsed.netloc}"

        super().__init__(name, timeout_seconds)
        self.url = url
        self.expected_status = expected_status
        self.headers = headers or {}

    async def check(self) -> HealthCheckResult:
        """Check HTTP endpoint health.

        Returns:
            Health check result
        """

        async def _check():
            try:
                timeout = aiohttp.ClientTimeout(total=self.timeout_seconds)
                async with (
                    aiohttp.ClientSession(timeout=timeout) as session,
                    session.get(self.url, headers=self.headers) as response,
                ):
                    if response.status == self.expected_status:
                        return HealthCheckResult(
                            name=self.name,
                            status=HealthStatus.HEALTHY,
                            message=f"HTTP endpoint responding with status {response.status}",
                            duration_ms=0.0,  # Will be updated by _execute_with_timeout
                            metadata={
                                "status_code": response.status,
                                "content_type": response.headers.get(
                                    "content-type", "unknown"
                                ),
                            },
                        )
                    else:
                        return HealthCheckResult(
                            name=self.name,
                            status=HealthStatus.UNHEALTHY,
                            message=f"HTTP endpoint returned status {response.status}, expected {self.expected_status}",
                            duration_ms=0.0,  # Will be updated by _execute_with_timeout
                            metadata={"status_code": response.status},
                        )

            except aiohttp.ClientError as e:
                return HealthCheckResult(
                    name=self.name,
                    status=HealthStatus.UNHEALTHY,
                    message=f"HTTP request failed: {e!s}",
                    duration_ms=0.0,  # Will be updated by _execute_with_timeout
                )

            except Exception:
                return HealthCheckResult(
                    name=self.name,
                    status=HealthStatus.UNHEALTHY,
                    message=f"HTTP health check error: {e!s}",
                    duration_ms=0.0,  # Will be updated by _execute_with_timeout
                )

        return await self._execute_with_timeout(_check())


class SystemResourceHealthCheck(HealthCheck):
    """Health check for system resources (CPU, memory, disk)."""

    def __init__(
        self,
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
        super().__init__(name, timeout_seconds)
        self.cpu_threshold = cpu_threshold
        self.memory_threshold = memory_threshold
        self.disk_threshold = disk_threshold

    async def check(self) -> HealthCheckResult:
        """Check system resource health.

        Returns:
            Health check result
        """

        async def _check():
            try:
                import psutil

                # Check CPU usage
                cpu_percent = psutil.cpu_percent(interval=1)

                # Check memory usage
                memory = psutil.virtual_memory()
                memory_percent = memory.percent

                # Check disk usage for root
                disk = psutil.disk_usage("/")
                disk_percent = (disk.used / disk.total) * 100

                # Determine overall status
                issues = []
                status = HealthStatus.HEALTHY

                if cpu_percent > self.cpu_threshold:
                    issues.append(f"High CPU usage: {cpu_percent:.1f}%")
                    status = HealthStatus.DEGRADED

                if memory_percent > self.memory_threshold:
                    issues.append(f"High memory usage: {memory_percent:.1f}%")
                    status = HealthStatus.DEGRADED

                if disk_percent > self.disk_threshold:
                    issues.append(f"High disk usage: {disk_percent:.1f}%")
                    status = HealthStatus.DEGRADED

                if len(issues) >= 2:
                    status = HealthStatus.UNHEALTHY

                message = (
                    "System resources healthy" if not issues else "; ".join(issues)
                )

                return HealthCheckResult(
                    name=self.name,
                    status=status,
                    message=message,
                    duration_ms=0.0,  # Will be updated by _execute_with_timeout
                    metadata={
                        "cpu_percent": cpu_percent,
                        "memory_percent": memory_percent,
                        "disk_percent": disk_percent,
                        "memory_available_gb": memory.available / (1024**3),
                        "disk_free_gb": disk.free / (1024**3),
                    },
                )

            except Exception:
                return HealthCheckResult(
                    name=self.name,
                    status=HealthStatus.UNHEALTHY,
                    message=f"System resource check failed: {e!s}",
                    duration_ms=0.0,  # Will be updated by _execute_with_timeout
                )

        return await self._execute_with_timeout(_check())


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

        # Add default health checks if enabled
        if config.enabled:
            # Create Qdrant client for health check
            qdrant_client = AsyncQdrantClient(url=config.qdrant_url)
            self.health_checks.extend(
                [
                    QdrantHealthCheck(qdrant_client),
                    RedisHealthCheck(config.redis_url),
                    SystemResourceHealthCheck(),
                ]
            )

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
            HTTPHealthCheck(url, expected_status, name, timeout_seconds)
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
        health_results = {}
        for i, result in enumerate(results):
            check_name = self.health_checks[i].name

            if isinstance(result, Exception):
                # Handle unexpected exceptions
                health_result = HealthCheckResult(
                    name=check_name,
                    status=HealthStatus.UNHEALTHY,
                    message=f"Health check failed with exception: {result!s}",
                    duration_ms=0,
                )
            else:
                health_result = result

            health_results[check_name] = health_result
            self._last_results[check_name] = health_result

            # Update metrics if registry is available
            if self.metrics_registry:
                is_healthy = health_result.status == HealthStatus.HEALTHY
                self.metrics_registry.update_dependency_health(check_name, is_healthy)

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
        elif any(status == HealthStatus.DEGRADED for status in statuses):
            return HealthStatus.DEGRADED
        elif all(status == HealthStatus.HEALTHY for status in statuses):
            return HealthStatus.HEALTHY
        else:
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
