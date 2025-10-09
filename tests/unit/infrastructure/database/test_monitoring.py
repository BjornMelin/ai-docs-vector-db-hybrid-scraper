"""Tests for database monitoring utilities."""

from __future__ import annotations

import asyncio

import pytest

from src.infrastructure.database.monitoring import ConnectionMonitor, QueryMonitor


@pytest.mark.asyncio()
async def test_query_monitor_records_successful_query() -> None:
    """QueryMonitor should record successful executions and durations."""

    monitor = QueryMonitor()
    await monitor.initialize()

    query_id = monitor.start_query()
    await asyncio.sleep(0)
    monitor.record_success(query_id)

    summary = await monitor.get_performance_summary()

    assert summary["total_queries"] == 1
    assert summary["successful_queries"] == 1
    assert summary["avg_duration_ms"] >= 0.0


@pytest.mark.asyncio()
async def test_query_monitor_records_failure() -> None:
    """QueryMonitor should capture failures with error messages."""

    monitor = QueryMonitor()
    await monitor.initialize()

    query_id = monitor.start_query()
    monitor.record_failure(query_id, "timeout")

    summary = await monitor.get_performance_summary()

    assert summary["total_queries"] == 1
    assert summary["successful_queries"] == 0


def test_connection_monitor_tracks_events() -> None:
    """ConnectionMonitor should retain the latest event metadata."""

    monitor = ConnectionMonitor()
    monitor.record_connection_event("checked_out", {"connection_id": "abc"})
    snapshot = monitor.get_pool_status()

    assert "checked_out" in snapshot["events"]
    assert snapshot["events"]["checked_out"]["metadata"]["connection_id"] == "abc"
