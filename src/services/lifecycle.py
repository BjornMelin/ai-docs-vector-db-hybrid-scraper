"""Shared service lifecycle protocol and helpers."""

# pylint: disable=unnecessary-ellipsis

from __future__ import annotations

from typing import Protocol, runtime_checkable


@runtime_checkable
class ServiceLifecycle(Protocol):
    """Protocol defining the minimal lifecycle every service must expose."""

    async def initialize(self) -> None:
        """Bring the service into a ready state."""
        ...

    async def cleanup(self) -> None:
        """Release resources and return to an uninitialized state."""
        ...

    def is_initialized(self) -> bool:
        """Return True when the service is ready to handle work."""
        ...


class LifecycleTracker:
    """Mixin that tracks initialized state for lifecycle-aware services."""

    def __init__(self) -> None:
        self._initialized = False

    def _mark_initialized(self) -> None:
        """Mark the service as initialized."""

        self._initialized = True

    def _mark_uninitialized(self) -> None:
        """Mark the service as uninitialized."""

        self._initialized = False

    def is_initialized(self) -> bool:
        """Implement :meth:`ServiceLifecycle.is_initialized`."""

        return self._initialized
