"""Minimal telemetry helpers for browser orchestration."""

from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass

from src.services.browser.models import ProviderKind


@dataclass(slots=True)
class ProviderStats:
    """Aggregated statistics per provider."""

    success: int = 0
    failure: int = 0
    rate_limited: int = 0

    def as_dict(self) -> dict[str, int]:
        """Render stats as plain dict."""

        return {
            "success": self.success,
            "failure": self.failure,
            "rate_limited": self.rate_limited,
        }


class MetricsRecorder:
    """In-memory recorder used by the router."""

    def __init__(self) -> None:
        self._stats: dict[ProviderKind, ProviderStats] = defaultdict(ProviderStats)

    def record_success(self, provider: ProviderKind) -> None:
        """Increment success counter."""

        self._stats[provider].success += 1

    def record_failure(self, provider: ProviderKind) -> None:
        """Increment failure counter."""

        self._stats[provider].failure += 1

    def record_rate_limited(self, provider: ProviderKind) -> None:
        """Increment rate-limited counter."""

        self._stats[provider].rate_limited += 1

    def snapshot(self) -> dict[str, dict[str, int]]:
        """Return a serializable snapshot."""

        return {
            provider.value: stats.as_dict() for provider, stats in self._stats.items()
        }
