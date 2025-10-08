"""Helpers for accessing the shared circuit breaker manager."""

from __future__ import annotations

import asyncio

from purgatory import AsyncInMemoryUnitOfWork

from src.infrastructure.client_manager import ensure_client_manager
from src.services.circuit_breaker import CircuitBreakerManager


_manager: CircuitBreakerManager | None = None
_manager_lock = asyncio.Lock()


async def get_circuit_breaker_manager() -> CircuitBreakerManager:
    """Return the shared CircuitBreakerManager instance from the ClientManager."""

    global _manager  # pylint: disable=global-statement
    if _manager is not None:
        return _manager

    async with _manager_lock:
        if _manager is None:
            try:
                client_manager = await ensure_client_manager()
                _manager = await client_manager.get_circuit_breaker_manager()
            except Exception:
                _manager = CircuitBreakerManager(
                    redis_url="memory://local",
                    config=None,
                    unit_of_work=AsyncInMemoryUnitOfWork(),
                )
    return _manager
