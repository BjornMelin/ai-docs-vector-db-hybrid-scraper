"""Tests for lightweight tracking helpers."""

from collections.abc import Iterator
from concurrent.futures import ThreadPoolExecutor

import pytest

from src.services.observability.tracking import (
    AIOperationTracker,
    PerformanceTracker,
    TraceCorrelationManager,
    get_ai_tracker,
    get_correlation_manager,
    record_ai_operation,
    track_cost,
)


@pytest.fixture(autouse=True)
def _reset_singletons() -> Iterator[None]:
    """Ensure global trackers are clean for each test."""
    get_ai_tracker().reset()
    get_correlation_manager().clear()
    yield
    get_ai_tracker().reset()
    get_correlation_manager().clear()


class TestAIOperationTracker:
    """Tests for AI operation tracking and metrics aggregation."""

    def test_records_operation(self) -> None:
        """Verify tracker aggregates operation counts, tokens, and costs."""
        tracker = AIOperationTracker()
        tracker.record_operation(
            operation="llm",
            provider="openai",
            model="gpt",
            duration_s=0.1,
            tokens=10,
            cost_usd=0.02,
            success=True,
        )
        tracker.record_operation(
            operation="llm",
            provider="openai",
            model="gpt",
            duration_s=0.2,
            tokens=5,
            cost_usd=0.01,
            success=False,
        )

        snapshot = tracker.snapshot()
        assert snapshot["llm"]["count"] == 2
        assert snapshot["llm"]["success_count"] == 1
        assert snapshot["llm"]["total_tokens"] == 15
        assert snapshot["llm"]["total_cost_usd"] == 0.03

        tracker.reset()
        assert tracker.snapshot() == {}

    def test_singleton_helpers(self) -> None:
        """Verify singleton access and recording helper work correctly."""
        tracker = get_ai_tracker()
        record_ai_operation(
            operation_type="embedding",
            provider="openai",
            model="ada",
            duration_s=0.2,
        )
        assert "embedding" in tracker.snapshot()

    def test_thread_safe_recording(self) -> None:
        """Verify concurrent recording maintains accurate counts."""
        tracker = AIOperationTracker()

        def _worker(success: bool) -> None:
            for _ in range(100):
                tracker.record_operation(
                    operation="parallel",
                    provider="provider",
                    model="model",
                    duration_s=0.01,
                    success=success,
                )

        with ThreadPoolExecutor(max_workers=4) as executor:
            futures = [executor.submit(_worker, idx % 2 == 0) for idx in range(4)]
            for future in futures:
                future.result()

        snapshot = tracker.snapshot()["parallel"]
        assert snapshot["count"] == 400
        assert snapshot["success_count"] == 200

    def test_track_cost_records_cost_only(self) -> None:
        """Verify cost helper records cost without token counts."""
        tracker = get_ai_tracker()
        track_cost(
            operation_type="billing",
            provider="openai",
            model="gpt",
            cost_usd=0.55,
        )
        snapshot = tracker.snapshot()["billing"]
        assert snapshot["total_cost_usd"] == pytest.approx(0.55)
        assert snapshot["total_tokens"] == 0


class TestPerformanceTracker:
    """Tests for performance timing context manager."""

    def test_track_context_manager(self) -> None:
        """Verify performance tracker records elapsed time for operations."""
        tracker = PerformanceTracker()
        with tracker.track("operation"):
            pass
        summary = tracker.summary()
        assert "operation" in summary


class TestTraceCorrelationManager:
    """Tests for trace correlation context management."""

    def test_context_management(self) -> None:
        """Verify context can be set, retrieved, and nested."""
        manager = TraceCorrelationManager()
        manager.set_context(request_id="abc123")
        assert manager.get_context()["request_id"] == "abc123"

        with manager.correlated_operation(agent="analytics"):
            inner = manager.get_context()
            assert inner["agent"] == "analytics"
            assert inner["request_id"] == "abc123"

        assert "agent" not in manager.get_context()

    def test_singleton(self) -> None:
        """Verify singleton instance is consistent across calls."""
        manager = get_correlation_manager()
        manager.set_context(flow="testing")
        assert get_correlation_manager().get_context()["flow"] == "testing"

    def test_correlated_operation_restores_context_on_exception(self) -> None:
        """Verify context is restored after an exception in correlated operation."""
        manager = TraceCorrelationManager()
        manager.set_context(request_id="req-1")

        with pytest.raises(RuntimeError), manager.correlated_operation(agent="failing"):
            raise RuntimeError("boom")

        assert manager.get_context() == {"request_id": "req-1"}
