"""Usage tracking and budget guardrails for embedding services."""

from __future__ import annotations

import logging
from collections import defaultdict
from dataclasses import dataclass, field
from datetime import date
from typing import Any


logger = logging.getLogger(__name__)


@dataclass
class UsageStats:
    """Tracks cumulative and daily usage statistics for embedding requests."""

    _total_requests: int = 0
    _total_tokens: int = 0
    _total_cost: float = 0.0
    requests_by_tier: defaultdict[str, int] = field(
        default_factory=lambda: defaultdict(int)
    )
    requests_by_provider: defaultdict[str, int] = field(
        default_factory=lambda: defaultdict(int)
    )
    daily_cost: float = 0.0
    last_reset_date: date = field(default_factory=date.today)

    def ensure_current_day(self) -> None:
        """Reset daily counters when the calendar day changes."""
        today = date.today()
        if self.last_reset_date != today:
            self.daily_cost = 0.0
            self.last_reset_date = today


@dataclass(frozen=True)
class UsageRecord:
    """Represents a single usage event."""

    provider: str
    model: str
    tokens: int
    cost: float
    tier: str


class UsageTracker:
    """Encapsulates usage accounting and budget enforcement."""

    def __init__(
        self, smart_config: Any | None = None, budget_limit: float | None = None
    ) -> None:
        """Initialize usage tracker with optional budget constraints."""
        self._stats = UsageStats()
        self._smart_config = smart_config
        self._budget_limit = budget_limit

    @property
    def stats(self) -> UsageStats:
        """Expose underlying usage statistics."""
        return self._stats

    def set_budget_limit(self, budget_limit: float | None) -> None:
        """Update budget limit at runtime."""
        self._budget_limit = budget_limit

    def record(self, record: UsageRecord) -> None:
        """Record usage for a completed embedding request."""
        stats = self._stats
        stats.ensure_current_day()

        safe_cost = max(float(record.cost), 0.0)
        safe_tokens = max(int(record.tokens), 0)

        stats._total_requests += 1
        stats._total_tokens += safe_tokens
        stats._total_cost += safe_cost
        stats.daily_cost += safe_cost
        stats.requests_by_tier[record.tier] += 1
        stats.requests_by_provider[record.provider] += 1

        logger.debug(
            "Usage recorded provider=%s model=%s tokens=%s cost=%.6f",
            record.provider,
            record.model,
            safe_tokens,
            safe_cost,
        )

    def check_budget(self, estimated_cost: float) -> dict[str, Any]:
        """Evaluate projected spend against configured budget limits."""
        projected_cost = max(float(estimated_cost), 0.0)
        stats = self._stats
        stats.ensure_current_day()

        daily_usage = stats.daily_cost
        projected_total = daily_usage + projected_cost

        budget_limit = self._budget_limit or getattr(
            self._smart_config, "daily_budget_limit", None
        )
        result = {
            "within_budget": True,
            "warnings": [],
            "daily_usage": daily_usage,
            "estimated__total": projected_total,
            "budget_limit": budget_limit,
        }

        if not budget_limit or budget_limit <= 0:
            return result

        utilization = projected_total / budget_limit

        if projected_total > budget_limit:
            result["within_budget"] = False
            message = (
                f"Estimated cost ${projected_total:.4f} would exceed daily budget "
                f"${budget_limit:.4f}"
            )
            result["warnings"].append(message)
            return result

        warning_threshold = getattr(self._smart_config, "budget_warning_threshold", 0.8)
        critical_threshold = getattr(
            self._smart_config, "budget_critical_threshold", 0.95
        )

        if utilization >= critical_threshold:
            message = (
                f"Projected spend ${projected_total:.4f} exceeds "
                f"{critical_threshold:.0%} of the daily budget "
                f"(${budget_limit:.4f})"
            )
            result["warnings"].append(message)
        elif utilization >= warning_threshold:
            message = (
                f"Projected spend ${projected_total:.4f} exceeds "
                f"{warning_threshold:.0%} of the daily budget "
                f"(${budget_limit:.4f})"
            )
            result["warnings"].append(message)

        return result

    def report(self) -> dict[str, Any]:
        """Build a snapshot of usage aggregates for monitoring and budgeting."""
        stats = self._stats
        stats.ensure_current_day()

        return {
            "summary": {
                "_total_requests": stats._total_requests,
                "_total_tokens": stats._total_tokens,
                "_total_cost": stats._total_cost,
                "daily_cost": stats.daily_cost,
                "last_reset_date": stats.last_reset_date.isoformat(),
            },
            "by_tier": dict(stats.requests_by_tier),
            "by_provider": dict(stats.requests_by_provider),
            "budget": {
                "limit": self._budget_limit,
                "daily_usage": stats.daily_cost,
            },
        }

    def set_smart_config(self, smart_config: Any | None) -> None:
        """Replace smart selection configuration for future calculations."""
        self._smart_config = smart_config
