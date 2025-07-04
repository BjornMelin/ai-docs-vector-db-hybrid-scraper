"""Modern circuit breaker implementation using purgatory-circuitbreaker.

This module provides a modernized circuit breaker implementation that replaces
the custom circuit breaker with the battle-tested purgatory-circuitbreaker library.
Provides distributed state management and improved reliability.
"""

import asyncio
import logging
from collections.abc import Awaitable, Callable
from typing import Any, TypeVar

from purgatory import AsyncCircuitBreakerFactory, AsyncRedisUnitOfWork

from src.config import Config


logger = logging.getLogger(__name__)

T = TypeVar("T")


class ModernCircuitBreakerManager:
    """Modern circuit breaker manager using purgatory-circuitbreaker.

    Provides distributed circuit breaker functionality with Redis storage
    for state persistence across multiple instances.
    """

    def __init__(self, redis_url: str, config: Config | None = None):
        """Initialize modern circuit breaker manager.

        Args:
            redis_url: Redis URL for distributed state storage
            config: Application configuration for circuit breaker settings
        """
        self.redis_url = redis_url
        self.config = config

        # Create Redis unit of work for distributed state
        self.redis_storage = AsyncRedisUnitOfWork(redis_url)

        # Get circuit breaker settings from config or use defaults
        if config and hasattr(config, "performance"):
            default_threshold = getattr(
                config.performance, "circuit_breaker_failure_threshold", 5
            )
            default_recovery_timeout = getattr(
                config.performance, "circuit_breaker_recovery_timeout", 60
            )
        else:
            default_threshold = 5
            default_recovery_timeout = 60

        # Create circuit breaker factory with distributed storage
        self.factory = AsyncCircuitBreakerFactory(
            default_threshold=default_threshold,
            default_recovery_timeout=default_recovery_timeout,
            storage=self.redis_storage,
        )

        # Cache for circuit breaker instances
        self._breakers: dict[str, Any] = {}
        self._lock = asyncio.Lock()

        logger.info(
            f"ModernCircuitBreakerManager initialized with Redis: {redis_url}, "
            f"threshold={default_threshold}, recovery_timeout={default_recovery_timeout}s"
        )

    async def get_breaker(self, service_name: str, **kwargs: Any):
        """Get or create a circuit breaker for a service.

        Args:
            service_name: Name of the service
            **kwargs: Additional circuit breaker configuration

        Returns:
            Circuit breaker instance for the service
        """
        if service_name not in self._breakers:
            async with self._lock:
                if service_name not in self._breakers:
                    self._breakers[service_name] = await self.factory.get_breaker(
                        service_name, **kwargs
                    )
                    logger.debug(
                        f"Created circuit breaker for service: {service_name}"
                    )  # TODO: Convert f-string to logging format

        return self._breakers[service_name]

    async def protected_call(
        self,
        service_name: str,
        func: Callable[..., Awaitable[T]],
        *args: Any,
        **kwargs: Any,
    ) -> T:
        """Execute a function with circuit breaker protection.

        Args:
            service_name: Name of the service being called
            func: Async function to execute
            *args: Positional arguments for the function
            **kwargs: Keyword arguments for the function

        Returns:
            Result of the function call

        Raises:
            CircuitBreakerOpenError: If the circuit breaker is open
            Exception: Any exception raised by the function
        """
        breaker = await self.get_breaker(service_name)

        async with breaker:
            return await func(*args, **kwargs)

    def decorator(self, service_name: str, **_breaker_kwargs: Any):
        """Decorator for circuit breaker protection.

        Args:
            service_name: Name of the service
            **breaker_kwargs: Additional circuit breaker configuration

        Returns:
            Decorator function
        """

        def decorator_func(
            func: Callable[..., Awaitable[T]],
        ) -> Callable[..., Awaitable[T]]:
            async def wrapper(*args: Any, **kwargs: Any) -> T:
                return await self.protected_call(service_name, func, *args, **kwargs)

            wrapper.__name__ = func.__name__
            wrapper.__doc__ = func.__doc__
            return wrapper

        return decorator_func

    async def get_breaker_status(self, service_name: str) -> dict[str, Any]:
        """Get the status of a circuit breaker.

        Args:
            service_name: Name of the service

        Returns:
            Dictionary with circuit breaker status information
        """
        if service_name not in self._breakers:
            return {"status": "not_initialized"}

        breaker = self._breakers[service_name]

        # Get status information from the circuit breaker
        try:
            return self._extract_breaker_status(service_name, breaker)
        except (OSError, PermissionError, ValueError) as e:
            logger.warning(
                f"Failed to get status for circuit breaker {service_name}: {e}"
            )
            return {"status": "error", "error": str(e)}

    def _extract_breaker_status(
        self, service_name: str, breaker: Any
    ) -> dict[str, Any]:
        """Extract status information from circuit breaker."""
        return {
            "service_name": service_name,
            "state": str(breaker.state),
            "failure_count": getattr(breaker, "failure_count", 0),
            "last_failure_time": getattr(breaker, "last_failure_time", None),
            "is_open": getattr(breaker, "is_open", False),
            "is_half_open": getattr(breaker, "is_half_open", False),
        }

    async def reset_breaker(self, service_name: str) -> bool:
        """Reset a circuit breaker to closed state.

        Args:
            service_name: Name of the service

        Returns:
            True if reset was successful, False otherwise
        """
        try:
            return await self._attempt_breaker_reset(service_name)
        except Exception:
            logger.exception("Failed to reset circuit breaker for {service_name}")
            return False

    async def _attempt_breaker_reset(self, service_name: str) -> bool:
        """Attempt to reset circuit breaker for service."""
        breaker = await self.get_breaker(service_name)
        if not hasattr(breaker, "reset"):
            logger.warning(
                "Circuit breaker for %s does not support reset", service_name
            )
            return False

        await breaker.reset()
        logger.info(
            f"Reset circuit breaker for service: {service_name}"
        )  # TODO: Convert f-string to logging format
        return True

    async def get_all_statuses(self) -> dict[str, dict[str, Any]]:
        """Get status of all circuit breakers.

        Returns:
            Dictionary mapping service names to their circuit breaker status
        """
        statuses = {}
        for service_name in self._breakers:
            statuses[service_name] = await self.get_breaker_status(service_name)
        return statuses

    async def close(self) -> None:
        """Clean up resources.

        Closes the Redis storage connection and clears cached breakers.
        """
        try:
            await self._cleanup_resources()
        except Exception:
            logger.exception("Error closing ModernCircuitBreakerManager")

    async def _cleanup_resources(self) -> None:
        """Clean up circuit breaker manager resources."""
        if hasattr(self.redis_storage, "close"):
            await self.redis_storage.close()
        self._breakers.clear()
        logger.info("ModernCircuitBreakerManager closed successfully")


# Convenience function for creating circuit breaker manager
def create_modern_circuit_breaker_manager(
    redis_url: str, config: Config | None = None
) -> ModernCircuitBreakerManager:
    """Create a modern circuit breaker manager instance.

    Args:
        redis_url: Redis URL for distributed state storage
        config: Application configuration

    Returns:
        ModernCircuitBreakerManager instance
    """
    return ModernCircuitBreakerManager(redis_url, config)
