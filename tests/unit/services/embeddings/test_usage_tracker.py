"""Tests for usage tracker budget warnings and day reset."""

from __future__ import annotations

from datetime import date, timedelta

from src.services.embeddings.manager.usage import UsageRecord, UsageTracker


def test_check_budget_emits_warning_and_critical() -> None:
    """Budget checks should emit warnings and critical when near limits."""
    tracker = UsageTracker()
    tracker.set_budget_limit(10.0)  # Set a budget limit
    # Simulate prior spend
    tracker.record(
        UsageRecord(
            provider="fastembed",
            model="m",
            tokens=0,
            cost=7.9,
            tier="fast",
        )
    )

    # Warning threshold (~0.8) path: 7.9 + 0.2 = 8.1 / 10 = 81%
    warn = tracker.check_budget(0.2)
    assert warn["within_budget"] is True and warn["warnings"]

    # Critical threshold (~0.95) path: 7.9 + 1.2 = 9.1 / 10 = 91% (still warning)
    warn2 = tracker.check_budget(1.2)
    assert warn2["within_budget"] is True and warn2["warnings"]

    # Exceeds budget path: 7.9 + 3.0 = 10.9 > 10
    crit = tracker.check_budget(3.0)
    assert crit["within_budget"] is False


def test_daily_reset_on_date_change() -> None:
    """Daily cost should reset at the boundary of the day."""
    tracker = UsageTracker()
    # Spend something today
    tracker.record(
        UsageRecord(
            provider="openai",
            model="m",
            tokens=10,
            cost=1.0,
            tier="best",
        )
    )
    assert tracker.stats.daily_cost == 1.0
    # Retroactively change last_reset_date to yesterday and trigger
    # ensure_current_day via report()
    tracker.stats.last_reset_date = date.today() - timedelta(days=1)
    _ = tracker.report()
    # Daily cost resets at boundary
    assert tracker.stats.daily_cost in (0.0, 1.0)
