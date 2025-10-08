"""Base service class for all services."""

import logging
from abc import ABC, abstractmethod
from collections.abc import Mapping
from contextlib import asynccontextmanager
from typing import Any

from src.config import Settings

from .errors import APIError
from .lifecycle import LifecycleTracker, ServiceLifecycle


logger = logging.getLogger(__name__)


class BaseService(ABC, LifecycleTracker, ServiceLifecycle):
    """Abstract base class for all services."""

    def __init__(self, config: Settings | None = None):
        """Initialize base service.

        Args:
            config: Unified configuration
        """

        self.config = config
        self._client: object | None = None
        LifecycleTracker.__init__(self)

    @abstractmethod
    async def initialize(self) -> None:
        """Initialize service resources."""

    @abstractmethod
    async def cleanup(self) -> None:
        """Cleanup service resources."""

    async def health_check(self) -> Mapping[str, Any] | None:
        """Return health data for the service.

        Subclasses may override to provide richer diagnostics.
        """

        return None

    @asynccontextmanager
    async def context(self):
        """Context manager for service lifecycle."""
        try:
            await self.initialize()
            yield self
        finally:
            await self.cleanup()

    def _validate_initialized(self) -> None:
        """Ensure service is initialized."""
        if not self.is_initialized():
            msg = (
                f"{self.__class__.__name__} not initialized. "
                "Call initialize() or use context manager."
            )
            raise APIError(msg)

    async def __aenter__(self):
        """Async context manager entry."""

        await self.initialize()
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb):
        """Async context manager exit."""

        await self.cleanup()
        return False
