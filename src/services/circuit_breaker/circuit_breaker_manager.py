"""Circuit breaker implementation using purgatory-circuitbreaker.

This module provides a circuit breaker implementation that replaces
the custom circuit breaker with the purgatory-circuitbreaker library.
Provides distributed state management and reliability.
"""

import asyncio
import logging
from collections.abc import Awaitable, Callable
from typing import Any, TypeVar

from purgatory import (
    AsyncCircuitBreakerFactory,
    AsyncRedisUnitOfWork,
)
from purgatory.domain.model import ClosedState
from purgatory.service._async.unit_of_work import AsyncAbstractUnitOfWork

from src.config import Config


logger = logging.getLogger(__name__)

T = TypeVar("T")


class CircuitBreakerManager:
    """Circuit breaker manager using purgatory-circuitbreaker.

    Provides distributed circuit breaker functionality with Redis storage
    for state persistence across multiple instances.
    """

    def __init__(
        self,
        redis_url: str,
        config: Config | None = None,
        unit_of_work: AsyncAbstractUnitOfWork | None = None,
    ):
        """Initialize circuit breaker manager.

        Args:
            redis_url: Redis URL for distributed state storage
            config: Application configuration for circuit breaker settings
        """

        self.redis_url = redis_url
        self.config = config

        # Create Redis unit of work for distributed state
        self.redis_url = redis_url
        self.redis_storage = unit_of_work or AsyncRedisUnitOfWork(redis_url)

        # Get circuit breaker settings from config or use defaults
        if config and hasattr(config, "performance"):
            default_threshold = getattr(
                config.performance, "circuit_breaker_failure_threshold", 5
            )
            default_ttl = getattr(
                config.performance, "circuit_breaker_recovery_timeout", 60
            )
        else:
            default_threshold = 5
            default_ttl = 60

        # Create circuit breaker factory with distributed storage
        self.factory = AsyncCircuitBreakerFactory(
            default_threshold=default_threshold,
            default_ttl=default_ttl,
            uow=self.redis_storage,
        )

        # Cache for circuit breaker instances
        self._breakers: dict[str, Any] = {}
        self._lock = asyncio.Lock()

        logger.info(
            "CircuitBreakerManager initialized with Redis: %s, "
            "threshold=%s, recovery_timeout=%ss",
            redis_url,
            default_threshold,
            default_ttl,
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
                        "Created circuit breaker for service: %s", service_name
                    )

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

        breaker = self._breakers.get(service_name)
        if breaker is None:
            return {"service_name": service_name, "state": "not_initialized"}

        try:
            context = breaker.context
            return {
                "service_name": service_name,
                "state": context.state,
                "failure_count": context.failure_count or 0,
                "opened_at": context.opened_at,
                "threshold": context.threshold,
                "ttl": context.ttl,
            }
        except (OSError, PermissionError, ValueError) as exc:
            logger.warning(
                "Failed to get status for circuit breaker %s: %s", service_name, exc
            )
            return {"service_name": service_name, "state": "error", "error": str(exc)}

    async def reset_breaker(self, service_name: str) -> bool:
        """Reset a circuit breaker to closed state.

        Args:
            service_name: Name of the service

        Returns:
            True if reset was successful, False otherwise
        """

        try:
            breaker = await self.get_breaker(service_name)
            context = breaker.context
            context.set_state(ClosedState())
            context.recover_failure()
            await self._flush_messages(breaker)
            logger.info("Reset circuit breaker for service: %s", service_name)
            return True
        except Exception:
            logger.exception("Failed to reset circuit breaker for %s", service_name)
            return False

    async def get_all_statuses(self) -> dict[str, dict[str, Any]]:
        """Get status of all circuit breakers.

        Returns:
            Dictionary mapping service names to their circuit breaker status
        """

        return {
            service_name: await self.get_breaker_status(service_name)
            for service_name in self._breakers
        }

    def list_services(self) -> list[str]:
        """Return the list of service names with initialized breakers."""

        return sorted(self._breakers.keys())

    async def close(self) -> None:
        """Clean up resources.

        Clears cached breakers.
        """
        try:
            await self._cleanup_resources()
        except Exception:
            logger.exception("Error closing CircuitBreakerManager")

    async def _cleanup_resources(self) -> None:
        """Clean up circuit breaker manager resources."""

        self._breakers.clear()
        logger.info("CircuitBreakerManager closed successfully")

    async def _flush_messages(self, breaker: Any) -> None:
        """Persist pending domain events emitted by the breaker context."""

        while breaker.context.messages:
            await breaker.messagebus.handle(
                breaker.context.messages.pop(0),
                breaker.uow,
            )


# Convenience function for creating circuit breaker manager
def create_circuit_breaker_manager(
    redis_url: str, config: Config | None = None
) -> CircuitBreakerManager:
    """Create a circuit breaker manager instance.

    Args:
        redis_url: Redis URL for distributed state storage
        config: Application configuration

    Returns:
        CircuitBreakerManager instance
    """

    return CircuitBreakerManager(redis_url, config)
