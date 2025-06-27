"""Service discovery with connection pooling optimization for Redis 8.2, Qdrant, and PostgreSQL.

Implements modern async service discovery patterns with:
- Circuit breaker resilience patterns
- Connection pooling with health checks
- Performance metrics and monitoring
- Graceful fallback to manual configuration
"""

import asyncio
import logging
import time
from typing import Any
from urllib.parse import urlparse

import httpx
from pydantic import BaseModel

from src.config.auto_detect import AutoDetectionConfig, DetectedService
from src.services.errors import circuit_breaker


logger = logging.getLogger(__name__)


class ServiceDiscoveryResult(BaseModel):
    """Result container for service discovery operations."""

    services: list[DetectedService]
    discovery_time_ms: float
    errors: list[str]
    metadata: dict[str, Any]


class ServiceDiscovery:
    """Discovers available services with health checks and connection pooling."""

    def __init__(self, config: AutoDetectionConfig):
        self.config = config
        self.logger = logger.getChild("discovery")
        self._discovery_cache: dict[str, tuple[DetectedService, float]] = {}

    async def discover_all_services(self) -> ServiceDiscoveryResult:
        """Discover all enabled services in parallel."""
        start_time = time.time()
        services = []
        errors = []

        # Build list of discovery tasks
        discovery_tasks = []

        if self.config.redis_discovery_enabled:
            discovery_tasks.append(("redis", self._discover_redis()))

        if self.config.qdrant_discovery_enabled:
            discovery_tasks.append(("qdrant", self._discover_qdrant()))

        if self.config.postgresql_discovery_enabled:
            discovery_tasks.append(("postgresql", self._discover_postgresql()))

        # Run discoveries in parallel with semaphore for concurrency control
        semaphore = asyncio.Semaphore(self.config.max_concurrent_detections)

        async def run_discovery(service_type: str, task):
            async with semaphore:
                try:
                    result = await task
                    if result:
                        services.append(result)
                        self.logger.info(
                            f"Discovered {service_type}: {result.host}:{result.port}"
                        )
                except Exception as e:
                    error_msg = f"Failed to discover {service_type}: {e}"
                    errors.append(error_msg)
                    self.logger.warning(error_msg)

        # Execute all discovery tasks
        await asyncio.gather(
            *[
                run_discovery(service_type, task)
                for service_type, task in discovery_tasks
            ]
        )

        discovery_time_ms = (time.time() - start_time) * 1000

        self.logger.info(
            f"Service discovery completed: {len(services)} services found "
            f"in {discovery_time_ms:.1f}ms"
        )

        return ServiceDiscoveryResult(
            services=services,
            discovery_time_ms=discovery_time_ms,
            errors=errors,
            metadata={
                "total_attempted": len(discovery_tasks),
                "successful": len(services),
                "failed": len(errors),
            },
        )

    @circuit_breaker(
        service_name="redis_discovery",
        failure_threshold=3,
        recovery_timeout=30.0,
    )
    async def _discover_redis(self) -> DetectedService | None:
        """Discover Redis service with Redis 8.2 optimizations."""
        start_time = time.time()

        # Check cache first
        cached = self._get_cached_service("redis")
        if cached:
            return cached

        # Common Redis ports and locations
        redis_candidates = [
            ("localhost", 6379),
            ("127.0.0.1", 6379),
            ("redis", 6379),  # Docker compose service name
            ("redis-server", 6379),
            ("dragonfly", 6379),  # DragonflyDB (Redis-compatible)
        ]

        # Add environment-specific candidates
        redis_candidates.extend(self._get_env_specific_redis_candidates())

        for host, port in redis_candidates:
            try:
                # Test TCP connection first
                if not await self._test_tcp_connection(host, port, timeout=2.0):
                    continue

                # Test Redis-specific protocol
                redis_info = await self._test_redis_connection(host, port)
                if redis_info:
                    detection_time_ms = (time.time() - start_time) * 1000

                    service = DetectedService(
                        service_name="redis",
                        service_type="redis",
                        host=host,
                        port=port,
                        is_available=True,
                        connection_string=f"redis://{host}:{port}",
                        version=redis_info.get("version"),
                        supports_pooling=True,
                        pool_config=self._get_redis_pool_config(redis_info),
                        health_check_url=f"redis://{host}:{port}/ping",
                        detection_time_ms=detection_time_ms,
                        metadata=redis_info,
                    )

                    # Cache the result
                    self._cache_service("redis", service)
                    return service

            except Exception as e:
                self.logger.debug(f"Redis discovery failed for {host}:{port}: {e}")
                continue

        return None

    @circuit_breaker(
        service_name="qdrant_discovery",
        failure_threshold=3,
        recovery_timeout=30.0,
    )
    async def _discover_qdrant(self) -> DetectedService | None:
        """Discover Qdrant service using AsyncQdrantClient connection testing."""
        start_time = time.time()

        # Check cache first
        cached = self._get_cached_service("qdrant")
        if cached:
            return cached

        # Common Qdrant locations (prioritize HTTP API)
        qdrant_candidates = [
            ("localhost", 6333),  # HTTP API
            ("127.0.0.1", 6333),
            ("qdrant", 6333),  # Docker compose service name
        ]

        for host, port in qdrant_candidates:
            try:
                # Use AsyncQdrantClient for direct connection testing
                qdrant_info = await self._test_qdrant_connection(host, port)

                if qdrant_info:
                    detection_time_ms = (time.time() - start_time) * 1000

                    # Test gRPC availability (6334) for performance optimization
                    grpc_port = 6334
                    grpc_available = await self._test_qdrant_grpc_availability(
                        host, grpc_port
                    )

                    service = DetectedService(
                        service_name="qdrant",
                        service_type="qdrant",
                        host=host,
                        port=port,
                        is_available=True,
                        connection_string=f"http://{host}:{port}",
                        version=qdrant_info.get("version"),
                        supports_pooling=True,
                        pool_config=self._get_qdrant_pool_config(
                            qdrant_info, grpc_available
                        ),
                        health_check_url=f"http://{host}:{port}/health",
                        detection_time_ms=detection_time_ms,
                        metadata={
                            **qdrant_info,
                            "grpc_available": grpc_available,
                            "grpc_port": grpc_port,
                        },
                    )

                    # Cache the result
                    self._cache_service("qdrant", service)
                    return service

            except Exception as e:
                self.logger.debug(f"Qdrant discovery failed for {host}:{port}: {e}")
                continue

        return None

    @circuit_breaker(
        service_name="postgresql_discovery",
        failure_threshold=3,
        recovery_timeout=30.0,
    )
    async def _discover_postgresql(self) -> DetectedService | None:
        """Discover PostgreSQL service using asyncpg connection testing."""
        start_time = time.time()

        # Check cache first
        cached = self._get_cached_service("postgresql")
        if cached:
            return cached

        # Common PostgreSQL ports and locations
        pg_candidates = [
            ("localhost", 5432),
            ("127.0.0.1", 5432),
            ("postgres", 5432),  # Docker compose service name
            ("postgresql", 5432),
            ("db", 5432),
        ]

        for host, port in pg_candidates:
            try:
                # Use asyncpg for direct connection testing (eliminates TCP + protocol redundancy)
                pg_info = await self._test_postgresql_connection(host, port)
                if pg_info:
                    detection_time_ms = (time.time() - start_time) * 1000

                    service = DetectedService(
                        service_name="postgresql",
                        service_type="postgresql",
                        host=host,
                        port=port,
                        is_available=True,
                        connection_string=f"postgresql://user:pass@{host}:{port}/db",
                        version=pg_info.get("version"),
                        supports_pooling=True,
                        pool_config=self._get_postgresql_pool_config(pg_info),
                        health_check_url=None,  # No HTTP health check for PostgreSQL
                        detection_time_ms=detection_time_ms,
                        metadata=pg_info,
                    )

                    # Cache the result
                    self._cache_service("postgresql", service)
                    return service

            except Exception as e:
                self.logger.debug(f"PostgreSQL discovery failed for {host}:{port}: {e}")
                continue

        return None

    async def _test_tcp_connection(
        self,
        host: str,
        port: int,
        timeout: float = 5.0,  # timeout used in asyncio.wait_for
    ) -> bool:
        """Test basic TCP connectivity to a service.

        Note: Only used for Redis discovery now. Qdrant and PostgreSQL use
        their respective client libraries for direct connection testing.
        """
        try:
            reader, writer = await asyncio.wait_for(
                asyncio.open_connection(host, port), timeout=timeout
            )
            writer.close()
            await writer.wait_closed()
            return True
        except Exception:
            return False

    async def _test_redis_connection(
        self, host: str, port: int
    ) -> dict[str, Any] | None:
        """Test Redis connection and get server info using redis-py."""
        try:
            import redis.asyncio as redis

            # Use redis-py for reliable connection testing
            client = redis.Redis(
                host=host,
                port=port,
                socket_connect_timeout=3.0,
                socket_timeout=3.0,
                decode_responses=True,
            )

            # Test connection with built-in ping
            await client.ping()

            # Get server info using built-in info command
            info = await client.info()
            await client.aclose()

            return {
                "version": info.get("redis_version", "unknown"),
                "mode": info.get("redis_mode", "standalone"),
                "memory_usage": info.get("used_memory_human"),
                "connected_clients": str(info.get("connected_clients", 0)),
                "server_info": info,
            }

        except ImportError:
            self.logger.warning("redis package not available for connection testing")
            return None
        except Exception as e:
            self.logger.debug(f"Redis connection test failed: {e}")
            return None

    async def _test_qdrant_connection(
        self, host: str, port: int
    ) -> dict[str, Any] | None:
        """Test Qdrant connection using AsyncQdrantClient directly."""
        try:
            from qdrant_client import AsyncQdrantClient

            # Use AsyncQdrantClient for native connection testing
            client = AsyncQdrantClient(url=f"http://{host}:{port}", timeout=3.0)

            # Test connection by checking collection existence (lightweight operation)
            collections = await client.get_collections()

            # Get version info from the root endpoint
            async with httpx.AsyncClient(timeout=3.0) as http_client:
                response = await http_client.get(f"http://{host}:{port}/")
                version_data = response.json() if response.status_code == 200 else {}

            await client.close()

            return {
                "version": version_data.get("version", "unknown"),
                "collections_count": len(collections.collections),
                "api_available": True,
                "version_data": version_data,
            }

        except ImportError:
            self.logger.warning(
                "qdrant-client package not available for connection testing"
            )
            return None
        except Exception as e:
            self.logger.debug(f"Qdrant connection test failed: {e}")
            return None

    async def _test_qdrant_grpc_availability(self, host: str, port: int) -> bool:
        """Test Qdrant gRPC availability for performance optimization."""
        try:
            # Quick TCP test for gRPC port availability
            reader, writer = await asyncio.wait_for(
                asyncio.open_connection(host, port), timeout=1.0
            )
            writer.close()
            await writer.wait_closed()
            return True
        except Exception:
            return False

    async def _test_postgresql_connection(
        self, host: str, port: int
    ) -> dict[str, Any] | None:
        """Test PostgreSQL connection using asyncpg directly."""
        try:
            import asyncpg

            # Use asyncpg for native connection testing with common credentials
            connection_params = [
                {"user": "postgres", "password": "", "database": "postgres"},
                {"user": "postgres", "password": "postgres", "database": "postgres"},
                {"user": "user", "password": "password", "database": "postgres"},
                {"user": "postgres", "database": "postgres"},  # No password
            ]

            for params in connection_params:
                try:
                    # Attempt connection with timeout
                    conn = await asyncio.wait_for(
                        asyncpg.connect(host=host, port=port, **params), timeout=3.0
                    )

                    # Get server version
                    version = await conn.fetchval("SELECT version()")
                    server_info = await conn.fetch(
                        "SELECT name, setting FROM pg_settings WHERE name IN ('server_version', 'max_connections')"
                    )

                    await conn.close()

                    # Parse version info
                    version_parts = version.split() if version else []
                    pg_version = (
                        version_parts[1] if len(version_parts) > 1 else "unknown"
                    )

                    settings = {row["name"]: row["setting"] for row in server_info}

                    return {
                        "version": pg_version,
                        "full_version": version,
                        "max_connections": settings.get("max_connections"),
                        "connection_successful": True,
                        "auth_params": {
                            k: v for k, v in params.items() if k != "password"
                        },
                    }

                except (TimeoutError, Exception) as e:
                    self.logger.debug(f"Database connection failed: {e}")
                    continue  # Try next credential set

        except ImportError:
            self.logger.warning(
                "asyncpg package not available for PostgreSQL connection testing"
            )
            return None
        except Exception as e:
            self.logger.debug(f"PostgreSQL connection test failed: {e}")

        return None

    def _get_env_specific_redis_candidates(self) -> list[tuple[str, int]]:
        """Get environment-specific Redis candidates."""
        candidates = []

        # Check for Redis URL in environment
        import os

        redis_url = os.getenv("REDIS_URL")
        if redis_url:
            try:
                parsed = urlparse(redis_url)
                if parsed.hostname and parsed.port:
                    candidates.append((parsed.hostname, parsed.port))
            except Exception as e:
                logger.debug(f"Failed to parse Redis URL '{redis_url}': {e}")

        return candidates

    def _get_redis_pool_config(self, redis_info: dict[str, Any]) -> dict[str, Any]:
        """Get optimized Redis 8.2 connection pool configuration."""
        return {
            "max_connections": 20,
            "retry_on_timeout": True,
            "health_check_interval": 30,
            "socket_keepalive": True,
            "socket_keepalive_options": {
                "TCP_KEEPIDLE": 1,
                "TCP_KEEPINTVL": 3,
                "TCP_KEEPCNT": 5,
            },
            "decode_responses": True,
            "protocol": 3,  # Redis 8.2 RESP3 protocol
            "version_info": redis_info,
        }

    def _get_qdrant_pool_config(
        self, qdrant_info: dict[str, Any], grpc_available: bool
    ) -> dict[str, Any]:
        """Get optimized Qdrant connection pool configuration."""
        config = {
            "timeout": 10.0,
            "prefer_grpc": grpc_available,
            "max_retries": 3,
            "grpc_options": {
                "grpc.keepalive_time_ms": 30000,
                "grpc.keepalive_timeout_ms": 5000,
                "grpc.keepalive_permit_without_calls": True,
                "grpc.http2.max_pings_without_data": 0,
            }
            if grpc_available
            else None,
            "version_info": qdrant_info,
        }

        return config

    def _get_postgresql_pool_config(self, pg_info: dict[str, Any]) -> dict[str, Any]:
        """Get optimized PostgreSQL connection pool configuration."""
        return {
            "min_size": 2,
            "max_size": 20,
            "command_timeout": 10,
            "server_settings": {
                "jit": "off",  # Disable JIT for consistent performance
                "application_name": "ai_docs_vector_db",
            },
            "version_info": pg_info,
        }

    def _get_cached_service(self, service_type: str) -> DetectedService | None:
        """Get cached service if still valid."""
        if service_type in self._discovery_cache:
            service, cache_time = self._discovery_cache[service_type]
            if time.time() - cache_time < self.config.cache_ttl_seconds:
                self.logger.debug(f"Using cached {service_type} service")
                return service

        return None

    def _cache_service(self, service_type: str, service: DetectedService) -> None:
        """Cache discovered service."""
        self._discovery_cache[service_type] = (service, time.time())

    def clear_cache(self) -> None:
        """Clear discovery cache."""
        self._discovery_cache.clear()
        self.logger.info("Service discovery cache cleared")
