"""Error types for the browser orchestration stack."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from src.services.errors import CrawlServiceError


class BrowserProviderError(CrawlServiceError):
    """Raised when a provider fails to complete a scrape attempt."""

    def __init__(
        self, message: str, *, provider: str, context: dict[str, Any] | None = None
    ):
        """Initialize provider error.

        Args:
            message: Human-readable description.
            provider: Provider identifier that triggered the failure.
            context: Optional context payload.
        """

        super().__init__(message, context={"provider": provider, **(context or {})})


class BrowserRouterError(CrawlServiceError):
    """Raised when the router exhausts all providers."""

    def __init__(self, message: str, *, attempted_providers: list[str]):
        """Initialize router error.

        Args:
            message: Error description.
            attempted_providers: Providers attempted before failure.
        """

        super().__init__(message, context={"attempted_providers": attempted_providers})


@dataclass(slots=True)
class ProviderFailure:
    """Captured failure metadata for observability."""

    provider: str
    error: str
