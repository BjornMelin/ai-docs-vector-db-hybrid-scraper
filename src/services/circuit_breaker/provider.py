"""Helpers for accessing the shared circuit breaker manager."""

from __future__ import annotations

import logging


try:
    from purgatory import AsyncInMemoryUnitOfWork
except ModuleNotFoundError:  # pragma: no cover - optional dependency
    AsyncInMemoryUnitOfWork = None  # type: ignore[assignment]

from src.infrastructure.container import get_container
from src.services.circuit_breaker import CircuitBreakerManager


logger = logging.getLogger(__name__)

_fallback_manager: CircuitBreakerManager | None = None


def _build_fallback_manager() -> CircuitBreakerManager:
    global _fallback_manager  # pylint: disable=global-statement
    if _fallback_manager is None:
        if AsyncInMemoryUnitOfWork is None:
            raise RuntimeError(
                "purgatory AsyncInMemoryUnitOfWork is unavailable; cannot "
                "construct fallback circuit breaker manager."
            )
        _fallback_manager = CircuitBreakerManager(
            redis_url="memory://local",
            config=None,
            unit_of_work=AsyncInMemoryUnitOfWork(),
        )
        logger.warning(
            "Using in-memory fallback circuit breaker manager until registry is "
            "available."
        )
    return _fallback_manager


async def get_circuit_breaker_manager() -> CircuitBreakerManager:
    """Return the shared CircuitBreakerManager instance from the DI container."""

    container = get_container()
    if container is None:
        logger.debug("DI container not initialized; using fallback circuit breaker")
        return _build_fallback_manager()

    manager = container.circuit_breaker_manager()
    if manager is None:
        logger.debug("CircuitBreakerManager provider returned None; using fallback")
        return _build_fallback_manager()
    return manager
