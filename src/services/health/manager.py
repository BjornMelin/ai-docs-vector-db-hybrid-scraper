"""Centralized health monitoring for application dependencies."""

from __future__ import annotations

import asyncio
import logging
import time
from abc import ABC, abstractmethod
from collections.abc import Iterable
from enum import Enum
from typing import TYPE_CHECKING, Any
from urllib.parse import urlparse

import aiohttp
import httpx
import redis.asyncio as redis
from openai import AsyncOpenAI, OpenAIError as OpenAIAPIError
from pydantic import BaseModel, Field
from qdrant_client import AsyncQdrantClient
from qdrant_client.http.exceptions import UnexpectedResponse

from src.config.models import CrawlProvider, EmbeddingProvider
from src.services.monitoring.metrics import MetricsRegistry


if TYPE_CHECKING:  # pragma: no cover - typing only
    from src.config.loader import Settings


try:  # pragma: no cover - optional dependency
    import psutil
except ImportError:  # pragma: no cover - graceful fallback
    psutil = None


logger = logging.getLogger(__name__)


class HealthCheckConfig(BaseModel):
    """Configuration for the health check system."""

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
    def from_unified_config(cls, settings: Settings) -> HealthCheckConfig:
        """Create configuration from application settings.

        Args:
            settings: Loaded application configuration.

        Returns:
            Parsed health check configuration.
        """

        redis_enabled = bool(
            getattr(settings.cache, "enable_redis_cache", False)
            or getattr(settings.cache, "enable_dragonfly_cache", False)
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
    """Result status reported by a health check."""

    HEALTHY = "healthy"
    UNHEALTHY = "unhealthy"
    DEGRADED = "degraded"
    UNKNOWN = "unknown"
    SKIPPED = "skipped"


class HealthCheckResult(BaseModel):
    """Structured response from a health check."""

    name: str = Field(description="Name of the health check")
    status: HealthStatus = Field(description="Health status")
    message: str = Field(description="Human-readable status message")
    timestamp: float = Field(default_factory=time.time, description="Check timestamp")
    duration_ms: float = Field(description="Check duration in milliseconds")
    metadata: dict[str, Any] = Field(
        default_factory=dict, description="Additional metadata"
    )


class HealthCheck(ABC):
    """Base class for asynchronous health checks."""

    def __init__(self, name: str, timeout_seconds: float = 5.0) -> None:
        """Initialize the health check descriptor.

        Args:
            name: Unique identifier for the health probe.
            timeout_seconds: Maximum duration allowed for the probe.
        """

        self.name = name
        self.timeout_seconds = timeout_seconds

    @abstractmethod
    async def check(self) -> HealthCheckResult:
        """Execute the health probe."""

    async def _execute_with_timeout(self, coro: Any) -> HealthCheckResult:
        """Run a coroutine with a timeout safeguard.

        Args:
            coro: Coroutine encapsulating the health check implementation.

        Returns:
            Completed health check result or timeout failure.
        """

        start_time = time.time()
        try:
            result = await asyncio.wait_for(coro, timeout=self.timeout_seconds)
        except TimeoutError:
            duration_ms = (time.time() - start_time) * 1000
            return HealthCheckResult(
                name=self.name,
                status=HealthStatus.UNHEALTHY,
                message=f"Health check timed out after {self.timeout_seconds:.1f}s",
                duration_ms=duration_ms,
            )
        except (ValueError, TypeError, AttributeError) as exc:
            duration_ms = (time.time() - start_time) * 1000
            return HealthCheckResult(
                name=self.name,
                status=HealthStatus.UNHEALTHY,
                message=f"Health check failed: {exc!s}",
                duration_ms=duration_ms,
            )

        duration_ms = (time.time() - start_time) * 1000
        result.duration_ms = duration_ms
        return result


class QdrantHealthCheck(HealthCheck):
    """Health check for the Qdrant vector database."""

    def __init__(
        self,
        client: AsyncQdrantClient,
        *,
        name: str = "qdrant",
        timeout_seconds: float = 5.0,
    ) -> None:
        """Initialize the Qdrant health probe.

        Args:
            client: Connected asynchronous Qdrant client.
            name: Unique health check name.
            timeout_seconds: Probe timeout budget in seconds.
        """

        super().__init__(name, timeout_seconds)
        self._client = client

    async def check(self) -> HealthCheckResult:
        """Confirm Qdrant availability by listing collections."""

        return await self._execute_with_timeout(self._perform_qdrant_check())

    async def _perform_qdrant_check(self) -> HealthCheckResult:
        """Execute the Qdrant health probe."""

        try:
            cluster_info = await self._client.get_collections()
            collections = getattr(cluster_info, "collections", [])
        except UnexpectedResponse as exc:
            return HealthCheckResult(
                name=self.name,
                status=HealthStatus.UNHEALTHY,
                message=f"Qdrant API error: {exc!s}",
                duration_ms=0.0,
            )
        except (ValueError, ConnectionError, TimeoutError, RuntimeError) as exc:
            return HealthCheckResult(
                name=self.name,
                status=HealthStatus.UNHEALTHY,
                message=f"Qdrant connectivity failed: {exc!s}",
                duration_ms=0.0,
            )

        collection_names = [
            getattr(collection, "name", "unknown") for collection in collections
        ]
        metadata = {
            "collection_count": len(collection_names),
            "collections": collection_names,
        }
        if not collection_names:
            return HealthCheckResult(
                name=self.name,
                status=HealthStatus.DEGRADED,
                message="Qdrant responded without collection metadata",
                duration_ms=0.0,
                metadata=metadata,
            )
        return HealthCheckResult(
            name=self.name,
            status=HealthStatus.HEALTHY,
            message="Qdrant service is operational",
            duration_ms=0.0,
            metadata=metadata,
        )


class RedisHealthCheck(HealthCheck):
    """Health check for Redis or Dragonfly cache backends."""

    def __init__(
        self,
        redis_url: str,
        *,
        name: str = "redis",
        timeout_seconds: float = 5.0,
    ) -> None:
        """Initialize the Redis health probe."""

        super().__init__(name, timeout_seconds)
        self._redis_url = redis_url

    async def check(self) -> HealthCheckResult:
        """Ping the cache service and gather key metrics."""

        return await self._execute_with_timeout(self._perform_redis_check())

    async def _perform_redis_check(self) -> HealthCheckResult:
        """Execute the Redis health probe."""

        redis_client = None
        try:
            redis_client = redis.from_url(self._redis_url)
            pong = await redis_client.ping()
            if not pong:
                return HealthCheckResult(
                    name=self.name,
                    status=HealthStatus.UNHEALTHY,
                    message="Redis ping failed",
                    duration_ms=0.0,
                )
            info = await redis_client.info()
        except redis.ConnectionError as exc:
            return HealthCheckResult(
                name=self.name,
                status=HealthStatus.UNHEALTHY,
                message=f"Redis connection failed: {exc!s}",
                duration_ms=0.0,
            )
        except (redis.RedisError, ConnectionError, TimeoutError, ValueError) as exc:
            return HealthCheckResult(
                name=self.name,
                status=HealthStatus.UNHEALTHY,
                message=f"Redis health check error: {exc!s}",
                duration_ms=0.0,
            )
        finally:
            if redis_client is not None:
                await redis_client.aclose()

        metadata = {
            "redis_version": info.get("redis_version", "unknown"),
            "connected_clients": info.get("connected_clients", 0),
            "used_memory_human": info.get("used_memory_human", "unknown"),
        }
        return HealthCheckResult(
            name=self.name,
            status=HealthStatus.HEALTHY,
            message="Redis server is responding",
            duration_ms=0.0,
            metadata=metadata,
        )


class OpenAIHealthCheck(HealthCheck):
    """Health check for OpenAI API connectivity."""

    def __init__(
        self,
        client: AsyncOpenAI,
        *,
        name: str = "openai",
        timeout_seconds: float = 5.0,
    ) -> None:
        """Initialize the OpenAI health probe."""

        super().__init__(name, timeout_seconds)
        self._client = client

    async def check(self) -> HealthCheckResult:
        """List available models to validate API access."""

        return await self._execute_with_timeout(self._perform_openai_check())

    async def _perform_openai_check(self) -> HealthCheckResult:
        """Execute the OpenAI health probe."""

        try:
            models = await self._client.models.list()
        except (ConnectionError, TimeoutError, RuntimeError, httpx.HTTPError) as exc:
            return HealthCheckResult(
                name=self.name,
                status=HealthStatus.UNHEALTHY,
                message=f"OpenAI connectivity failed: {exc!s}",
                duration_ms=0.0,
            )
        except OpenAIAPIError as exc:
            return HealthCheckResult(
                name=self.name,
                status=HealthStatus.UNHEALTHY,
                message=f"OpenAI API error: {exc!s}",
                duration_ms=0.0,
            )

        return HealthCheckResult(
            name=self.name,
            status=HealthStatus.HEALTHY,
            message="OpenAI API is reachable",
            duration_ms=0.0,
            metadata={"model_count": len(models.data)},
        )


class HTTPHealthCheck(HealthCheck):
    """Generic HTTP health probe."""

    # pylint: disable=too-many-arguments
    def __init__(
        self,
        url: str,
        *,
        expected_status: int = 200,
        name: str | None = None,
        timeout_seconds: float = 5.0,
        headers: dict[str, str] | None = None,
    ) -> None:
        """Initialize the HTTP health probe."""

        if name is None:
            parsed = urlparse(url)
            name = f"http_{parsed.netloc}"

        super().__init__(name, timeout_seconds)
        self._url = url
        self._expected_status = expected_status
        self._headers = headers or {}

    async def check(self) -> HealthCheckResult:
        """Perform an HTTP GET request and evaluate the status."""

        return await self._execute_with_timeout(self._perform_http_check())

    async def _perform_http_check(self) -> HealthCheckResult:
        """Execute the HTTP health probe."""

        timeout = aiohttp.ClientTimeout(total=self.timeout_seconds)
        try:
            async with (
                aiohttp.ClientSession(timeout=timeout) as session,
                session.get(self._url, headers=self._headers) as response,
            ):
                if response.status == self._expected_status:
                    return HealthCheckResult(
                        name=self.name,
                        status=HealthStatus.HEALTHY,
                        message=(
                            f"HTTP endpoint responding with status {response.status}"
                        ),
                        duration_ms=0.0,
                        metadata={
                            "status_code": response.status,
                            "content_type": response.headers.get(
                                "content-type", "unknown"
                            ),
                        },
                    )
                return HealthCheckResult(
                    name=self.name,
                    status=HealthStatus.UNHEALTHY,
                    message=(
                        f"HTTP endpoint returned status {response.status}, "
                        f"expected {self._expected_status}"
                    ),
                    duration_ms=0.0,
                    metadata={"status_code": response.status},
                )
        except aiohttp.ClientError as exc:
            return HealthCheckResult(
                name=self.name,
                status=HealthStatus.UNHEALTHY,
                message=f"HTTP request failed: {exc!s}",
                duration_ms=0.0,
            )
        except ConnectionError as exc:
            return HealthCheckResult(
                name=self.name,
                status=HealthStatus.UNHEALTHY,
                message=f"HTTP connection error: {exc!s}",
                duration_ms=0.0,
            )


class FirecrawlHealthCheck(HTTPHealthCheck):
    """Health check for the Firecrawl API."""

    def __init__(
        self,
        base_url: str,
        api_key: str | None,
        *,
        timeout_seconds: float = 5.0,
    ) -> None:
        """Initialize the Firecrawl health probe."""

        headers: dict[str, str] | None = None
        if api_key:
            headers = {"Authorization": f"Bearer {api_key}"}
        super().__init__(
            url=f"{base_url.rstrip('/')}/health",
            expected_status=200,
            name="firecrawl",
            timeout_seconds=timeout_seconds,
            headers=headers,
        )

    async def check(self) -> HealthCheckResult:
        """Skip the probe when no API key is configured."""

        if not self._headers:
            return HealthCheckResult(
                name=self.name,
                status=HealthStatus.UNHEALTHY,
                message="Firecrawl API key missing",
                duration_ms=0.0,
            )
        return await super().check()


class SystemResourceHealthCheck(HealthCheck):
    """Health check for host system resources."""

    # pylint: disable=too-many-arguments
    def __init__(
        self,
        *,
        name: str = "system_resources",
        cpu_threshold: float = 90.0,
        memory_threshold: float = 90.0,
        disk_threshold: float = 90.0,
        timeout_seconds: float = 5.0,
    ) -> None:
        """Initialize the system resource probe."""

        super().__init__(name, timeout_seconds)
        self._cpu_threshold = cpu_threshold
        self._memory_threshold = memory_threshold
        self._disk_threshold = disk_threshold

    async def check(self) -> HealthCheckResult:
        """Inspect CPU, memory, and disk utilisation."""

        return await self._execute_with_timeout(self._perform_system_check())

    async def _perform_system_check(self) -> HealthCheckResult:
        """Execute the system resource probe."""

        if psutil is None:
            return HealthCheckResult(
                name=self.name,
                status=HealthStatus.DEGRADED,
                message="psutil not available",
                duration_ms=0.0,
                metadata={"error": "psutil not installed"},
            )

        cpu_percent = psutil.cpu_percent(interval=1)  # type: ignore[call-arg]
        memory = psutil.virtual_memory()  # type: ignore[attr-defined]
        disk = psutil.disk_usage("/")  # type: ignore[attr-defined]

        metrics = {
            "cpu_percent": cpu_percent,
            "memory_percent": memory.percent,
            "disk_percent": (disk.used / disk.total) * 100,
            "memory_available_gb": memory.available / (1024**3),
            "disk_free_gb": disk.free / (1024**3),
        }

        issues: list[str] = []
        status = HealthStatus.HEALTHY
        if metrics["cpu_percent"] > self._cpu_threshold:
            issues.append(f"High CPU usage: {metrics['cpu_percent']:.1f}%")
            status = HealthStatus.DEGRADED
        if metrics["memory_percent"] > self._memory_threshold:
            issues.append(f"High memory usage: {metrics['memory_percent']:.1f}%")
            status = HealthStatus.DEGRADED
        if metrics["disk_percent"] > self._disk_threshold:
            issues.append(f"High disk usage: {metrics['disk_percent']:.1f}%")
            status = HealthStatus.DEGRADED
        if len(issues) >= 2:
            status = HealthStatus.UNHEALTHY

        message = "System resources healthy" if not issues else "; ".join(issues)
        return HealthCheckResult(
            name=self.name,
            status=status,
            message=message,
            duration_ms=0.0,
            metadata=metrics,
        )


class HealthCheckManager:
    """Coordinator for registered health checks."""

    def __init__(
        self,
        config: HealthCheckConfig,
        metrics_registry: MetricsRegistry | None = None,
    ) -> None:
        """Initialize the health check manager."""

        self.config = config
        self.metrics_registry = metrics_registry
        self._health_checks: list[HealthCheck] = []
        self._last_results: dict[str, HealthCheckResult] = {}

    def add_health_check(self, health_check: HealthCheck) -> None:
        """Register an additional health check."""

        self._health_checks.append(health_check)

    def add_checks(self, checks: Iterable[HealthCheck]) -> None:
        """Register multiple health checks at once."""

        for check in checks:
            self.add_health_check(check)

    async def check_all(self) -> dict[str, HealthCheckResult]:
        """Execute all registered checks concurrently."""

        if not self._health_checks:
            return {}

        tasks = [check.check() for check in self._health_checks]
        results = await asyncio.gather(*tasks, return_exceptions=True)

        health_results: dict[str, HealthCheckResult] = {}
        for check, result in zip(self._health_checks, results, strict=False):
            if isinstance(result, BaseException):
                health_result = HealthCheckResult(
                    name=check.name,
                    status=HealthStatus.UNHEALTHY,
                    message=f"Health check failed with exception: {result!s}",
                    duration_ms=0.0,
                )
            else:
                health_result = result

            health_results[check.name] = health_result
            self._last_results[check.name] = health_result

            if self.metrics_registry:
                is_healthy = health_result.status == HealthStatus.HEALTHY
                self.metrics_registry.update_dependency_health(check.name, is_healthy)

        return health_results

    async def check_single(self, check_name: str) -> HealthCheckResult | None:
        """Run a single health check by name."""

        for check in self._health_checks:
            if check.name == check_name:
                result = await check.check()
                self._last_results[check_name] = result
                if self.metrics_registry:
                    is_healthy = result.status == HealthStatus.HEALTHY
                    self.metrics_registry.update_dependency_health(
                        check.name, is_healthy
                    )
                return result
        return None

    def get_last_results(self) -> dict[str, HealthCheckResult]:
        """Return the most recent health check results."""

        return self._last_results.copy()

    def get_overall_status(self) -> HealthStatus:
        """Compute an aggregate health status."""

        if not self._last_results:
            return HealthStatus.UNKNOWN

        statuses = [result.status for result in self._last_results.values()]
        if any(status == HealthStatus.UNHEALTHY for status in statuses):
            return HealthStatus.UNHEALTHY
        if any(status == HealthStatus.DEGRADED for status in statuses):
            return HealthStatus.DEGRADED
        if all(status == HealthStatus.HEALTHY for status in statuses):
            return HealthStatus.HEALTHY
        if all(status == HealthStatus.SKIPPED for status in statuses):
            return HealthStatus.SKIPPED
        return HealthStatus.UNKNOWN

    def get_health_summary(self) -> dict[str, Any]:
        """Return a structured summary of recent health checks."""

        return {
            "overall_status": self.get_overall_status().value,
            "timestamp": time.time(),
            "checks": {
                name: result.model_dump(mode="json")
                for name, result in self._last_results.items()
            },
            "healthy_count": sum(
                1
                for result in self._last_results.values()
                if result.status == HealthStatus.HEALTHY
            ),
            "total_count": len(self._last_results),
        }


def build_health_manager(
    settings: Settings,
    *,
    metrics_registry: MetricsRegistry | None = None,
    qdrant_client: AsyncQdrantClient | None = None,
    redis_url: str | None = None,
) -> HealthCheckManager:
    """Create a health manager configured for application settings.

    Args:
        settings: Loaded application configuration.
        metrics_registry: Optional metrics registry for reporting results.
        qdrant_client: Optional pre-configured Qdrant client instance.
        redis_url: Optional override for the Redis connection URL.

    Returns:
        Initialized health check manager with configured probes.
    """

    config = HealthCheckConfig.from_unified_config(settings)
    manager = HealthCheckManager(config, metrics_registry)

    if settings.monitoring.include_system_metrics:
        manager.add_health_check(
            SystemResourceHealthCheck(
                cpu_threshold=settings.monitoring.cpu_threshold,
                memory_threshold=settings.monitoring.memory_threshold,
                disk_threshold=settings.monitoring.disk_threshold,
                timeout_seconds=settings.monitoring.health_check_timeout,
            )
        )

    if config.qdrant_url:
        client = qdrant_client or AsyncQdrantClient(
            url=config.qdrant_url,
            api_key=getattr(settings.qdrant, "api_key", None),
            timeout=int(settings.qdrant.timeout),
        )
        manager.add_health_check(
            QdrantHealthCheck(
                client,
                timeout_seconds=settings.monitoring.health_check_timeout,
            )
        )

    redis_source = redis_url or config.redis_url
    if redis_source:
        manager.add_health_check(
            RedisHealthCheck(
                redis_source,
                timeout_seconds=settings.monitoring.health_check_timeout,
            )
        )

    if settings.embedding_provider is EmbeddingProvider.OPENAI:
        api_key = getattr(settings.openai, "api_key", None)
        if api_key:
            manager.add_health_check(
                OpenAIHealthCheck(
                    AsyncOpenAI(api_key=api_key),
                    timeout_seconds=settings.monitoring.health_check_timeout,
                )
            )
        else:
            manager.add_health_check(
                HTTPHealthCheck(
                    url="https://api.openai.com/v1/models",
                    name="openai",
                    headers={"Authorization": "Bearer"},
                    timeout_seconds=settings.monitoring.health_check_timeout,
                )
            )

    if settings.crawl_provider is CrawlProvider.FIRECRAWL:
        manager.add_health_check(
            FirecrawlHealthCheck(
                base_url=settings.firecrawl.api_url,
                api_key=getattr(settings.firecrawl, "api_key", None),
                timeout_seconds=settings.monitoring.health_check_timeout,
            )
        )

    for service_name, service_url in settings.monitoring.external_services.items():
        manager.add_health_check(
            HTTPHealthCheck(
                url=service_url,
                name=service_name,
                timeout_seconds=settings.monitoring.health_check_timeout,
            )
        )

    return manager


__all__ = [
    "FirecrawlHealthCheck",
    "HealthCheck",
    "HealthCheckConfig",
    "HealthCheckManager",
    "HealthCheckResult",
    "HealthStatus",
    "HTTPHealthCheck",
    "OpenAIHealthCheck",
    "QdrantHealthCheck",
    "RedisHealthCheck",
    "SystemResourceHealthCheck",
    "build_health_manager",
]
