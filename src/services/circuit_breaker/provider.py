"""Helpers for accessing the shared circuit breaker manager."""

from __future__ import annotations

import asyncio

from purgatory import AsyncInMemoryUnitOfWork

from src.services.circuit_breaker import CircuitBreakerManager
from src.services.registry import ensure_service_registry, get_service_registry


_manager: CircuitBreakerManager | None = None
_manager_lock = asyncio.Lock()


async def get_circuit_breaker_manager() -> CircuitBreakerManager:
    """Return the shared CircuitBreakerManager instance from the registry."""

    global _manager  # pylint: disable=global-statement
    if _manager is not None:
        return _manager

    async with _manager_lock:
        if _manager is None:
            try:
                registry = get_service_registry()
            except RuntimeError:
                try:
                    registry = await ensure_service_registry()
                except Exception:
                    _manager = CircuitBreakerManager(
                        redis_url="memory://local",
                        config=None,
                        unit_of_work=AsyncInMemoryUnitOfWork(),
                    )
                else:
                    _manager = registry.circuit_breaker_manager
            else:
                _manager = registry.circuit_breaker_manager
    return _manager
