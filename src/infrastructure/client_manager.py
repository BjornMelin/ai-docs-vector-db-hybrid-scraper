"""Centralized API client management with singleton pattern and health checks."""

import asyncio
import contextlib
import logging
import time
from contextlib import asynccontextmanager
from dataclasses import dataclass
from enum import Enum
from typing import Any
from typing import Optional

import redis.asyncio as redis
from firecrawl import AsyncFirecrawlApp
from openai import AsyncOpenAI
from pydantic import BaseModel
from pydantic import Field
from pydantic import field_validator
from qdrant_client import AsyncQdrantClient

from ..services.errors import APIError

logger = logging.getLogger(__name__)


class ClientState(Enum):
    """Client connection state."""

    UNINITIALIZED = "uninitialized"
    HEALTHY = "healthy"
    DEGRADED = "degraded"
    FAILED = "failed"


@dataclass
class ClientHealth:
    """Client health status."""

    state: ClientState
    last_check: float
    last_error: str | None = None
    consecutive_failures: int = 0


class ClientManagerConfig(BaseModel):
    """Configuration for client manager."""

    # Qdrant settings
    qdrant_url: str = Field(default="http://localhost:6333")
    qdrant_api_key: str | None = None
    qdrant_timeout: float = Field(default=30.0, gt=0)
    qdrant_prefer_grpc: bool = Field(default=False)

    # OpenAI settings
    openai_api_key: str | None = None
    openai_timeout: float = Field(default=30.0, gt=0)
    openai_max_retries: int = Field(default=3, ge=0, le=10)

    # Firecrawl settings
    firecrawl_api_key: str | None = None
    firecrawl_timeout: float = Field(default=60.0, gt=0)

    # Redis settings
    redis_url: str = Field(default="redis://localhost:6379")
    redis_max_connections: int = Field(default=10, gt=0)
    redis_decode_responses: bool = Field(default=True)

    # Health check settings
    health_check_interval: float = Field(default=30.0, gt=0)
    health_check_timeout: float = Field(default=5.0, gt=0)
    max_consecutive_failures: int = Field(default=3, gt=0)

    # Circuit breaker settings
    circuit_breaker_failure_threshold: int = Field(default=5, gt=0)
    circuit_breaker_recovery_timeout: float = Field(default=60.0, gt=0)
    circuit_breaker_half_open_requests: int = Field(default=1, gt=0)

    @field_validator("qdrant_url")
    @classmethod
    def validate_qdrant_url(cls, v):
        """Validate Qdrant URL format."""
        if not v.startswith(("http://", "https://")):
            raise ValueError("Qdrant URL must start with http:// or https://")
        return v

    @field_validator("redis_url")
    @classmethod
    def validate_redis_url(cls, v):
        """Validate Redis URL format."""
        if not v.startswith(("redis://", "rediss://")):
            raise ValueError("Redis URL must start with redis:// or rediss://")
        return v


class CircuitBreaker:
    """Circuit breaker implementation for fault tolerance."""

    def __init__(
        self,
        failure_threshold: int = 5,
        recovery_timeout: float = 60.0,
        half_open_requests: int = 1,
    ):
        """Initialize circuit breaker.

        Args:
            failure_threshold: Number of failures before opening circuit
            recovery_timeout: Time in seconds before attempting recovery
            half_open_requests: Number of test requests in half-open state
        """
        self.failure_threshold = failure_threshold
        self.recovery_timeout = recovery_timeout
        self.half_open_requests = half_open_requests

        self._failure_count = 0
        self._last_failure_time: float | None = None
        self._state = ClientState.HEALTHY
        self._half_open_attempts = 0
        self._lock = asyncio.Lock()

    @property
    def state(self) -> ClientState:
        """Get current circuit state."""
        if (
            self._state == ClientState.FAILED
            and self._last_failure_time
            and time.time() - self._last_failure_time > self.recovery_timeout
        ):
            return ClientState.DEGRADED  # Half-open
        return self._state

    async def call(self, func, *args, **kwargs):
        """Execute function with circuit breaker protection."""
        async with self._lock:
            current_state = self.state

            if current_state == ClientState.FAILED:
                raise APIError("Circuit breaker is open")

            if current_state == ClientState.DEGRADED:
                if self._half_open_attempts >= self.half_open_requests:
                    self._state = ClientState.FAILED
                    raise APIError("Circuit breaker is open (half-open test failed)")
                self._half_open_attempts += 1

        try:
            result = await func(*args, **kwargs)
            # Success - reset the circuit
            async with self._lock:
                self._failure_count = 0
                self._half_open_attempts = 0
                self._state = ClientState.HEALTHY
            return result

        except Exception as e:
            async with self._lock:
                self._failure_count += 1
                self._last_failure_time = time.time()

                if self._failure_count >= self.failure_threshold:
                    self._state = ClientState.FAILED
                    logger.error(
                        f"Circuit breaker opened after {self._failure_count} failures"
                    )

            raise e


class ClientManager:
    """Centralized API client management with singleton pattern."""

    _instance: Optional["ClientManager"] = None
    _lock = asyncio.Lock()

    def __new__(cls, config: ClientManagerConfig | None = None):
        """Ensure singleton instance."""
        if cls._instance is None:
            cls._instance = super().__new__(cls)
        return cls._instance

    def __init__(self, config: ClientManagerConfig | None = None):
        """Initialize client manager.

        Args:
            config: Client manager configuration
        """
        # Skip re-initialization if already initialized
        if hasattr(self, "_initialized") and self._initialized:
            if config and config != self.config:
                raise ValueError(
                    "ClientManager already initialized with different config. "
                    "Use cleanup() and reinitialize or create a new instance."
                )
            return

        self.config = config or ClientManagerConfig()
        self._clients: dict[str, Any] = {}
        self._locks: dict[str, asyncio.Lock] = {}
        self._health: dict[str, ClientHealth] = {}
        self._circuit_breakers: dict[str, CircuitBreaker] = {}
        self._initialized = False
        self._health_check_task: asyncio.Task | None = None

    async def initialize(self) -> None:
        """Initialize client manager and start health checks."""
        if self._initialized:
            return

        self._initialized = True
        # Start background health check task
        self._health_check_task = asyncio.create_task(self._health_check_loop())
        logger.info("ClientManager initialized")

    async def cleanup(self) -> None:
        """Cleanup all clients and resources."""
        # Cancel health check task
        if self._health_check_task and not self._health_check_task.done():
            self._health_check_task.cancel()
            with contextlib.suppress(asyncio.CancelledError):
                await self._health_check_task

        # Close all clients
        for name, client in self._clients.items():
            try:
                if hasattr(client, "close"):
                    await client.close()
                elif hasattr(client, "aclose"):
                    await client.aclose()
                logger.info(f"Closed {name} client")
            except Exception as e:
                logger.error(f"Error closing {name} client: {e}")

        self._clients.clear()
        self._health.clear()
        self._circuit_breakers.clear()
        self._initialized = False
        logger.info("ClientManager cleaned up")

    async def get_qdrant_client(self) -> AsyncQdrantClient:
        """Get or create Qdrant client with health checks.

        Returns:
            AsyncQdrantClient instance

        Raises:
            APIError: If client is unhealthy or initialization fails
        """
        return await self._get_or_create_client(
            "qdrant",
            self._create_qdrant_client,
            self._check_qdrant_health,
        )

    async def get_openai_client(self) -> AsyncOpenAI | None:
        """Get or create OpenAI client.

        Returns:
            AsyncOpenAI instance or None if no API key

        Raises:
            APIError: If client is unhealthy
        """
        if not self.config.openai_api_key:
            return None

        return await self._get_or_create_client(
            "openai",
            self._create_openai_client,
            self._check_openai_health,
        )

    async def get_firecrawl_client(self) -> AsyncFirecrawlApp | None:
        """Get or create Firecrawl client.

        Returns:
            AsyncFirecrawlApp instance or None if no API key

        Raises:
            APIError: If client is unhealthy
        """
        if not self.config.firecrawl_api_key:
            return None

        return await self._get_or_create_client(
            "firecrawl",
            self._create_firecrawl_client,
            self._check_firecrawl_health,
        )

    async def get_redis_client(self) -> redis.Redis:
        """Get or create Redis client with connection pooling.

        Returns:
            Redis client instance

        Raises:
            APIError: If client is unhealthy
        """
        return await self._get_or_create_client(
            "redis",
            self._create_redis_client,
            self._check_redis_health,
        )

    async def _get_or_create_client(
        self,
        name: str,
        create_func,
        health_check_func,
    ) -> Any:
        """Get existing client or create new one with health checks.

        Args:
            name: Client name
            create_func: Function to create client
            health_check_func: Function to check client health

        Returns:
            Client instance

        Raises:
            APIError: If client is unhealthy or creation fails
        """
        # Check circuit breaker state
        if name in self._circuit_breakers:
            breaker = self._circuit_breakers[name]
            if breaker.state == ClientState.FAILED:
                raise APIError(f"{name} client circuit breaker is open")

        # Get or create client
        if name not in self._clients:
            if name not in self._locks:
                self._locks[name] = asyncio.Lock()

            async with self._locks[name]:
                # Double-check after acquiring lock
                if name not in self._clients:
                    try:
                        # Create circuit breaker if not exists
                        if name not in self._circuit_breakers:
                            self._circuit_breakers[name] = CircuitBreaker(
                                failure_threshold=self.config.circuit_breaker_failure_threshold,
                                recovery_timeout=self.config.circuit_breaker_recovery_timeout,
                                half_open_requests=self.config.circuit_breaker_half_open_requests,
                            )

                        # Create client with circuit breaker
                        breaker = self._circuit_breakers[name]
                        client = await breaker.call(create_func)
                        self._clients[name] = client

                        # Initialize health status
                        self._health[name] = ClientHealth(
                            state=ClientState.HEALTHY,
                            last_check=time.time(),
                        )

                        logger.info(f"Created {name} client")

                    except Exception as e:
                        logger.error(f"Failed to create {name} client: {e}")
                        raise APIError(f"Failed to create {name} client: {e}") from e

        # Check health status
        if name in self._health:
            health = self._health[name]
            if (
                health.state == ClientState.FAILED
                and health.consecutive_failures >= self.config.max_consecutive_failures
            ):
                raise APIError(f"{name} client is unhealthy: {health.last_error}")

        return self._clients[name]

    async def _create_qdrant_client(self) -> AsyncQdrantClient:
        """Create Qdrant client instance."""
        client = AsyncQdrantClient(
            url=self.config.qdrant_url,
            api_key=self.config.qdrant_api_key,
            timeout=self.config.qdrant_timeout,
            prefer_grpc=self.config.qdrant_prefer_grpc,
        )

        # Validate connection
        await client.get_collections()
        return client

    async def _create_openai_client(self) -> AsyncOpenAI:
        """Create OpenAI client instance."""
        return AsyncOpenAI(
            api_key=self.config.openai_api_key,
            timeout=self.config.openai_timeout,
            max_retries=self.config.openai_max_retries,
        )

    async def _create_firecrawl_client(self) -> AsyncFirecrawlApp:
        """Create Firecrawl client instance."""
        return AsyncFirecrawlApp(
            api_key=self.config.firecrawl_api_key,
        )

    async def _create_redis_client(self) -> redis.Redis:
        """Create Redis client with connection pooling."""
        return redis.from_url(
            self.config.redis_url,
            max_connections=self.config.redis_max_connections,
            decode_responses=self.config.redis_decode_responses,
        )

    async def _check_qdrant_health(self) -> bool:
        """Check Qdrant client health."""
        try:
            client = self._clients.get("qdrant")
            if not client:
                return False

            # Use circuit breaker for health check
            breaker = self._circuit_breakers.get("qdrant")
            if breaker:
                await breaker.call(client.get_collections)
            else:
                await client.get_collections()

            return True
        except Exception as e:
            logger.warning(f"Qdrant health check failed: {e}")
            return False

    async def _check_openai_health(self) -> bool:
        """Check OpenAI client health."""
        try:
            client = self._clients.get("openai")
            if not client:
                return False

            # Simple API call to check connectivity
            breaker = self._circuit_breakers.get("openai")
            if breaker:
                await breaker.call(client.models.list)
            else:
                await client.models.list()

            return True
        except Exception as e:
            logger.warning(f"OpenAI health check failed: {e}")
            return False

    async def _check_firecrawl_health(self) -> bool:
        """Check Firecrawl client health."""
        try:
            client = self._clients.get("firecrawl")
            # Firecrawl doesn't have a direct health endpoint
            # We'll assume it's healthy if client exists
            return bool(client)
        except Exception as e:
            logger.warning(f"Firecrawl health check failed: {e}")
            return False

    async def _check_redis_health(self) -> bool:
        """Check Redis client health."""
        try:
            client = self._clients.get("redis")
            if not client:
                return False

            # Simple ping to check connectivity
            breaker = self._circuit_breakers.get("redis")
            if breaker:
                await breaker.call(client.ping)
            else:
                await client.ping()

            return True
        except Exception as e:
            logger.warning(f"Redis health check failed: {e}")
            return False

    async def _run_single_health_check(self, name: str, check_func) -> None:
        """Run a single health check and update client status."""
        try:
            # Run health check with timeout
            is_healthy = await asyncio.wait_for(
                check_func(),
                timeout=self.config.health_check_timeout,
            )

            # Update health status
            if name not in self._health:
                self._health[name] = ClientHealth(
                    state=ClientState.HEALTHY,
                    last_check=time.time(),
                )

            health = self._health[name]
            previous_state = health.state
            health.last_check = time.time()

            if is_healthy:
                # Check if client was previously failed and should be recreated
                if previous_state == ClientState.FAILED:
                    await self._recreate_client_if_needed(name)
                    logger.info(f"{name} client recovered and recreated")

                health.state = ClientState.HEALTHY
                health.consecutive_failures = 0
                health.last_error = None
            else:
                health.consecutive_failures += 1
                health.state = (
                    ClientState.DEGRADED
                    if health.consecutive_failures
                    < self.config.max_consecutive_failures
                    else ClientState.FAILED
                )
                health.last_error = "Health check returned false"

        except TimeoutError:
            logger.error(f"{name} health check timed out")
            self._update_health_failure(name, "Health check timeout")
        except Exception as e:
            logger.error(f"{name} health check error: {e}")
            self._update_health_failure(name, str(e))

    async def _recreate_client_if_needed(self, name: str) -> None:
        """Recreate a client if it was previously failed."""
        if name not in self._clients:
            return

        try:
            # Remove old client
            old_client = self._clients[name]
            if hasattr(old_client, "close"):
                await old_client.close()
            elif hasattr(old_client, "aclose"):
                await old_client.aclose()

            # Remove from clients dict to force recreation on next access
            del self._clients[name]

            # Reset circuit breaker
            if name in self._circuit_breakers:
                breaker = self._circuit_breakers[name]
                breaker._failure_count = 0
                breaker._state = ClientState.HEALTHY
                breaker._half_open_attempts = 0

            logger.info(f"Recreated {name} client after recovery")

        except Exception as e:
            logger.warning(f"Failed to recreate {name} client: {e}")

    async def _health_check_loop(self) -> None:
        """Background task to periodically check client health."""
        health_checks = {
            "qdrant": self._check_qdrant_health,
            "openai": self._check_openai_health,
            "firecrawl": self._check_firecrawl_health,
            "redis": self._check_redis_health,
        }

        while True:
            try:
                await asyncio.sleep(self.config.health_check_interval)

                # Run health checks in parallel for better performance
                active_clients = [
                    (name, check_func)
                    for name, check_func in health_checks.items()
                    if name in self._clients
                ]

                if not active_clients:
                    continue

                # Create tasks for parallel execution
                tasks = []
                for name, check_func in active_clients:
                    task = asyncio.create_task(
                        self._run_single_health_check(name, check_func),
                        name=f"health_check_{name}",
                    )
                    tasks.append(task)

                # Wait for all health checks to complete
                await asyncio.gather(*tasks, return_exceptions=True)

            except asyncio.CancelledError:
                logger.info("Health check loop cancelled")
                break
            except Exception as e:
                logger.error(f"Error in health check loop: {e}")
                await asyncio.sleep(10)  # Brief pause before retry

    def _update_health_failure(self, name: str, error: str) -> None:
        """Update health status for a failure."""
        if name not in self._health:
            self._health[name] = ClientHealth(
                state=ClientState.FAILED,
                last_check=time.time(),
                last_error=error,
                consecutive_failures=1,
            )
        else:
            health = self._health[name]
            health.last_check = time.time()
            health.last_error = error
            health.consecutive_failures += 1
            health.state = (
                ClientState.DEGRADED
                if health.consecutive_failures < self.config.max_consecutive_failures
                else ClientState.FAILED
            )

    @asynccontextmanager
    async def managed_client(self, client_type: str):
        """Context manager for automatic client lifecycle management.

        Args:
            client_type: Type of client ("qdrant", "openai", "firecrawl", "redis")

        Yields:
            Client instance

        Example:
            async with client_manager.managed_client("qdrant") as client:
                await client.get_collections()
        """
        client_getters = {
            "qdrant": self.get_qdrant_client,
            "openai": self.get_openai_client,
            "firecrawl": self.get_firecrawl_client,
            "redis": self.get_redis_client,
        }

        if client_type not in client_getters:
            raise ValueError(f"Unknown client type: {client_type}")

        try:
            client = await client_getters[client_type]()
            yield client
        except Exception as e:
            logger.error(f"Error using {client_type} client: {e}")
            raise

    async def get_health_status(self) -> dict[str, dict[str, Any]]:
        """Get health status of all clients.

        Returns:
            Dictionary mapping client names to health information
        """
        status = {}

        for name, health in self._health.items():
            status[name] = {
                "state": health.state.value,
                "last_check": health.last_check,
                "last_error": health.last_error,
                "consecutive_failures": health.consecutive_failures,
                "circuit_breaker_state": (
                    self._circuit_breakers[name].state.value
                    if name in self._circuit_breakers
                    else None
                ),
            }

        return status

    async def __aenter__(self):
        """Async context manager entry."""
        await self.initialize()
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb):
        """Async context manager exit."""
        await self.cleanup()
        return False
