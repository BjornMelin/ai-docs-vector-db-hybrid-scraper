"""Tests for lightweight tracking helpers."""

from src.services.observability.tracking import (
    AIOperationTracker,
    PerformanceTracker,
    TraceCorrelationManager,
    get_ai_tracker,
    get_correlation_manager,
    record_ai_operation,
)


class TestAIOperationTracker:
    def test_records_operation(self) -> None:
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
        tracker = get_ai_tracker()
        record_ai_operation(
            operation_type="embedding",
            provider="openai",
            model="ada",
            duration_s=0.2,
        )
        assert "embedding" in tracker.snapshot()


class TestPerformanceTracker:
    def test_track_context_manager(self) -> None:
        tracker = PerformanceTracker()
        with tracker.track("operation"):
            pass
        summary = tracker.summary()
        assert "operation" in summary


class TestTraceCorrelationManager:
    def test_context_management(self) -> None:
        manager = TraceCorrelationManager()
        manager.set_context(request_id="abc123")
        assert manager.get_context()["request_id"] == "abc123"

        with manager.correlated_operation(agent="analytics"):
            inner = manager.get_context()
            assert inner["agent"] == "analytics"
            assert inner["request_id"] == "abc123"

        assert "agent" not in manager.get_context()

    def test_singleton(self) -> None:
        manager = get_correlation_manager()
        manager.set_context(flow="testing")
        assert get_correlation_manager().get_context()["flow"] == "testing"
