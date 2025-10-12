"""Provider base classes and utilities."""

from __future__ import annotations

import abc
import contextlib
import time
from collections.abc import AsyncIterator
from dataclasses import dataclass

from src.services.browser.errors import BrowserProviderError
from src.services.browser.models import BrowserResult, ProviderKind, ScrapeRequest


@dataclass(slots=True)
class ProviderContext:
    """Shared provider runtime context."""

    provider: ProviderKind


class BrowserProvider(abc.ABC):
    """Interface implemented by all providers."""

    kind: ProviderKind

    def __init__(self, context: ProviderContext) -> None:
        self._context = context

    @abc.abstractmethod
    async def initialize(self) -> None:
        """Initialize resources for the provider."""

    @abc.abstractmethod
    async def close(self) -> None:
        """Release provider resources."""

    @abc.abstractmethod
    async def scrape(self, request: ScrapeRequest) -> BrowserResult:
        """Execute a scrape request."""

    @contextlib.asynccontextmanager
    async def lifecycle(self) -> AsyncIterator[BrowserProvider]:
        """Async context manager managing provider lifecycle."""

        await self.initialize()
        try:
            yield self
        finally:
            await self.close()

    def _failure(
        self, request: ScrapeRequest, error: Exception, *, elapsed_ms: int | None = None
    ) -> BrowserResult:
        """Create a standardized failure result and re-raise BrowserProviderError."""

        message = str(error)
        return BrowserResult.failure(
            url=request.url,
            provider=self._context.provider,
            error=message,
            metadata={"exception": error.__class__.__name__},
            elapsed_ms=elapsed_ms,
        )

    async def run(self, request: ScrapeRequest) -> BrowserResult:
        """Measure execution and normalize errors."""

        start = time.perf_counter()
        try:
            result = await self.scrape(request)
            if result.elapsed_ms is None:
                result.elapsed_ms = int((time.perf_counter() - start) * 1000)
            return result
        except BrowserProviderError:
            raise
        except Exception as exc:  # pragma: no cover - escalated as provider error
            elapsed = int((time.perf_counter() - start) * 1000)
            failure = self._failure(request, exc, elapsed_ms=elapsed)
            raise BrowserProviderError(
                failure.metadata.get("error", "provider failure"),
                provider=self._context.provider.value,
                context={"elapsed_ms": elapsed},
            ) from exc
