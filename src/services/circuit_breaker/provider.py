"""Helpers for accessing the shared circuit breaker manager."""

from __future__ import annotations

import asyncio
import logging


try:
    from purgatory import AsyncInMemoryUnitOfWork
except ModuleNotFoundError:  # pragma: no cover - optional dependency
    AsyncInMemoryUnitOfWork = None  # type: ignore[assignment]

from src.infrastructure.client_manager import ensure_client_manager
from src.services.circuit_breaker import CircuitBreakerManager


logger = logging.getLogger(__name__)

_manager: CircuitBreakerManager | None = None
_fallback_manager: CircuitBreakerManager | None = None
_manager_lock = asyncio.Lock()


async def _resolve_registry_manager() -> CircuitBreakerManager | None:
    try:
        registry = get_service_registry()
    except RuntimeError:
        try:
            registry = await ensure_service_registry()
        except Exception as exc:  # pragma: no cover - startup race conditions
            logger.debug(
                "Service registry unavailable for circuit breaker manager: %s",
                exc,
                exc_info=True,
            )
            return None
    return registry.circuit_breaker_manager


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
    """Return the shared CircuitBreakerManager instance from the ClientManager."""

    global _manager  # pylint: disable=global-statement
    if _manager is not None:
        if _manager is _fallback_manager:
            # Allow subsequent calls to retry registry resolution.
            async with _manager_lock:
                _manager = await _resolve_registry_manager() or _manager
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
