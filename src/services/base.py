"""Base service class for all services."""

import logging
from abc import ABC, abstractmethod
from contextlib import asynccontextmanager

from src.config import Config

from .errors import APIError


logger = logging.getLogger(__name__)


class BaseService(ABC):
    """Abstract base class for all services."""

    def __init__(self, config: Config | None = None):
        """Initialize base service.

        Args:
            config: Unified configuration
        """
        self.config = config
        self._client: object | None = None
        self._initialized = False

    @abstractmethod
    async def initialize(self) -> None:
        """Initialize service resources."""
        pass

    @abstractmethod
    async def cleanup(self) -> None:
        """Cleanup service resources."""
        pass

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
        if not self._initialized:
            raise APIError(
                f"{self.__class__.__name__} not initialized. "
                "Call initialize() or use context manager."
            )

    async def __aenter__(self):
        """Async context manager entry."""
        await self.initialize()
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb):
        """Async context manager exit."""
        await self.cleanup()
        return False
