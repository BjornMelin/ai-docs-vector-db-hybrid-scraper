"""Base service class for all services."""

import asyncio
import logging
from abc import ABC
from abc import abstractmethod
from contextlib import asynccontextmanager
from typing import Any

from pydantic import BaseModel

from .errors import APIError

logger = logging.getLogger(__name__)


class BaseService(ABC):
    """Abstract base class for all services."""

    def __init__(self, config: BaseModel | None = None):
        """Initialize base service.

        Args:
            config: Service configuration
        """
        self.config = config
        self._client: Any | None = None
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

    async def _retry_with_backoff(
        self,
        func,
        *args,
        max_retries: int = 3,
        base_delay: float = 1.0,
        **kwargs,
    ):
        """Execute function with exponential backoff retry.

        Args:
            func: Function to execute
            max_retries: Maximum number of retries
            base_delay: Base delay in seconds
            *args: Function arguments
            **kwargs: Function keyword arguments

        Returns:
            Function result

        Raises:
            APIError: If all retries fail
        """
        last_error = None

        for attempt in range(max_retries):
            try:
                return await func(*args, **kwargs)
            except Exception as e:
                last_error = e
                if attempt == max_retries - 1:
                    logger.error(f"API call failed after {max_retries} attempts: {e}")
                    raise APIError(
                        f"API call failed after {max_retries} attempts: {e}"
                    ) from e

                delay = base_delay * (2**attempt)
                logger.warning(
                    f"API call failed (attempt {attempt + 1}/{max_retries}), "
                    f"retrying in {delay}s: {e}"
                )
                await asyncio.sleep(delay)

        raise APIError(f"API call failed: {last_error}")

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
