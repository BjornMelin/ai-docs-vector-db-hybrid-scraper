"""Test concurrent access patterns for drift detection.

This module tests the thread safety of the drift detection system,
ensuring that concurrent operations don't cause race conditions or
data corruption when multiple threads access shared state.
"""

import asyncio
import concurrent.futures
import json
import tempfile
import time
from datetime import UTC, datetime, timezone
from pathlib import Path
from threading import Thread
from typing import Any

import pytest

from src.config.drift_detection import (
    ConfigDriftDetector,
    DriftDetectionConfig,
    DriftSeverity,
)


class TestDriftDetectionConcurrency:
    """Test thread safety of drift detection operations."""

    @pytest.fixture
    def config(self) -> DriftDetectionConfig:
        """Create test configuration."""
        return DriftDetectionConfig(
            enabled=True,
            snapshot_interval_minutes=1,
            comparison_interval_minutes=1,
            monitored_paths=["test_config.json"],
            alert_on_severity=[DriftSeverity.HIGH, DriftSeverity.CRITICAL],
            integrate_with_task20_anomaly=False,
            use_performance_monitoring=False,
        )

    @pytest.fixture
    def detector(self, config: DriftDetectionConfig) -> ConfigDriftDetector:
        """Create test detector instance."""
        return ConfigDriftDetector(config)

    @pytest.fixture
    def temp_config_file(self) -> Path:
        """Create temporary config file."""
        with tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False) as f:
            json.dump({"test_key": "test_value", "version": 1}, f)
            return Path(f.name)

    def test_concurrent_snapshot_creation(
        self, detector: ConfigDriftDetector, temp_config_file: Path
    ):
        """Test concurrent snapshot creation doesn't cause race conditions."""
        num_threads = 10
        iterations_per_thread = 5

        def create_snapshots():
            """Worker function to create snapshots."""
            for _i in range(iterations_per_thread):
                detector.take_snapshot(str(temp_config_file))
                time.sleep(0.01)  # Small delay to encourage race conditions

        # Run concurrent snapshot creation
        threads = []
        for _ in range(num_threads):
            t = Thread(target=create_snapshots)
            threads.append(t)
            t.start()

        # Wait for all threads
        for t in threads:
            t.join()

        # Verify snapshots were created correctly
        with detector._snapshots_lock:
            assert str(temp_config_file) in detector._snapshots
            snapshots = detector._snapshots[str(temp_config_file)]
            # Should have at least some snapshots (not necessarily all due to deduplication)
            assert len(snapshots) > 0
            # Verify all snapshots have valid structure
            for snapshot in snapshots:
                assert snapshot.config_hash is not None
                assert snapshot.timestamp is not None
                assert snapshot.config_data is not None

    def test_concurrent_drift_detection(
        self, detector: ConfigDriftDetector, temp_config_file: Path
    ):
        """Test concurrent drift detection and event recording."""
        # Create initial snapshots
        detector.take_snapshot(str(temp_config_file))

        # Modify the config file
        with open(temp_config_file, "w") as f:
            json.dump({"test_key": "modified_value", "version": 2}, f)

        detector.take_snapshot(str(temp_config_file))

        num_threads = 5
        detected_events = []

        def detect_drift():
            """Worker function to detect drift."""
            events = detector.compare_snapshots(str(temp_config_file))
            detected_events.extend(events)

        # Run concurrent drift detection
        threads = []
        for _ in range(num_threads):
            t = Thread(target=detect_drift)
            threads.append(t)
            t.start()

        # Wait for all threads
        for t in threads:
            t.join()

        # Verify drift events were recorded correctly
        with detector._events_lock:
            assert len(detector._drift_events) > 0
            # Verify event structure
            for event in detector._drift_events:
                assert event.id is not None
                assert event.timestamp is not None
                assert event.drift_type is not None
                assert event.severity is not None

    def test_concurrent_alert_rate_limiting(self, detector: ConfigDriftDetector):
        """Test concurrent access to alert rate limiting."""
        from src.config.drift_detection import DriftEvent, DriftType

        # Create a test event
        test_event = DriftEvent(
            id="test_event",
            timestamp=datetime.now(tz=UTC),
            drift_type=DriftType.MANUAL_CHANGE,
            severity=DriftSeverity.HIGH,
            source="test_source",
            description="Test drift",
            old_value="old",
            new_value="new",
            diff_details={},
        )

        num_threads = 10
        alert_results = []

        def check_alert():
            """Worker function to check if alert should be sent."""
            result = detector.should_alert(test_event)
            alert_results.append(result)

        # Run concurrent alert checks
        threads = []
        for _ in range(num_threads):
            t = Thread(target=check_alert)
            threads.append(t)
            t.start()

        # Wait for all threads
        for t in threads:
            t.join()

        # Verify alert tracking is consistent
        with detector._alerts_lock:
            alert_key = f"{test_event.source}_{test_event.drift_type.value}"
            assert alert_key in detector._last_alert_times
            # Should have recorded the alert time
            assert isinstance(detector._last_alert_times[alert_key], datetime)

    def test_concurrent_cleanup_operations(self, detector: ConfigDriftDetector):
        """Test concurrent cleanup operations don't corrupt data."""
        from src.config.drift_detection import ConfigSnapshot, DriftEvent, DriftType

        # Add test data
        test_source = "test_cleanup"
        with detector._snapshots_lock:
            detector._snapshots[test_source] = [
                ConfigSnapshot(
                    timestamp=datetime.now(tz=UTC),
                    config_hash=f"hash_{i}",
                    config_data={"test": i},
                    source=test_source,
                )
                for i in range(20)
            ]

        with detector._events_lock:
            detector._drift_events = [
                DriftEvent(
                    id=f"event_{i}",
                    timestamp=datetime.now(tz=UTC),
                    drift_type=DriftType.MANUAL_CHANGE,
                    severity=DriftSeverity.LOW,
                    source=test_source,
                    description=f"Event {i}",
                    old_value=i,
                    new_value=i + 1,
                    diff_details={},
                )
                for i in range(20)
            ]

        num_threads = 5

        def cleanup_operations():
            """Worker function to perform cleanup."""
            detector._cleanup_old_snapshots(test_source)
            detector._cleanup_old_events()

        # Run concurrent cleanup
        threads = []
        for _ in range(num_threads):
            t = Thread(target=cleanup_operations)
            threads.append(t)
            t.start()

        # Wait for all threads
        for t in threads:
            t.join()

        # Verify data integrity after cleanup
        with detector._snapshots_lock:
            if test_source in detector._snapshots:
                # All remaining snapshots should be valid
                for snapshot in detector._snapshots[test_source]:
                    assert snapshot.config_hash is not None
                    assert snapshot.timestamp is not None

        with detector._events_lock:
            # All remaining events should be valid
            for event in detector._drift_events:
                assert event.id is not None
                assert event.timestamp is not None

    def test_concurrent_summary_generation(self, detector: ConfigDriftDetector):
        """Test concurrent access to drift summary doesn't cause issues."""
        from src.config.drift_detection import DriftEvent, DriftType

        # Add some test events
        with detector._events_lock:
            detector._drift_events = [
                DriftEvent(
                    id=f"event_{i}",
                    timestamp=datetime.now(tz=UTC),
                    drift_type=DriftType.MANUAL_CHANGE,
                    severity=DriftSeverity.HIGH if i % 2 == 0 else DriftSeverity.LOW,
                    source=f"source_{i}",
                    description=f"Event {i}",
                    old_value=i,
                    new_value=i + 1,
                    diff_details={},
                    auto_remediable=i % 3 == 0,
                )
                for i in range(10)
            ]

        num_threads = 10
        summaries = []

        def get_summary():
            """Worker function to get drift summary."""
            summary = detector.get_drift_summary()
            summaries.append(summary)

        # Run concurrent summary generation
        threads = []
        for _ in range(num_threads):
            t = Thread(target=get_summary)
            threads.append(t)
            t.start()

        # Wait for all threads
        for t in threads:
            t.join()

        # Verify all summaries are valid and consistent
        assert len(summaries) == num_threads
        for summary in summaries:
            assert "detection_enabled" in summary
            assert "monitored_sources" in summary
            assert "snapshots_stored" in summary
            assert "recent_events_24h" in summary
            assert "severity_breakdown" in summary
            assert isinstance(summary["severity_breakdown"], dict)

    @pytest.mark.asyncio
    async def test_async_operations_compatibility(self, detector: ConfigDriftDetector):
        """Test that async operations can work with the drift detector."""
        # This test ensures the detector can be used in async contexts
        # even though it's primarily synchronous

        async def async_snapshot_operation():
            """Async wrapper for snapshot operations."""
            # In a real async scenario, we might need to use asyncio locks
            # For now, the threading locks work fine with async code
            await asyncio.sleep(0.01)
            summary = detector.get_drift_summary()
            return summary

        # Run multiple async operations
        tasks = [async_snapshot_operation() for _ in range(5)]
        results = await asyncio.gather(*tasks)

        # Verify all operations completed successfully
        assert len(results) == 5
        for result in results:
            assert isinstance(result, dict)
            assert "detection_enabled" in result

    def test_thread_pool_executor_compatibility(
        self, detector: ConfigDriftDetector, temp_config_file: Path
    ):
        """Test drift detector works correctly with ThreadPoolExecutor."""

        def worker_task(task_id: int) -> dict[str, Any]:
            """Worker task for thread pool."""
            # Take snapshot
            detector.take_snapshot(str(temp_config_file))

            # Get summary
            summary = detector.get_drift_summary()
            summary["task_id"] = task_id

            return summary

        # Use ThreadPoolExecutor for concurrent operations
        with concurrent.futures.ThreadPoolExecutor(max_workers=5) as executor:
            futures = [executor.submit(worker_task, i) for i in range(10)]
            results = [f.result() for f in concurrent.futures.as_completed(futures)]

        # Verify all tasks completed successfully
        assert len(results) == 10
        task_ids = {r["task_id"] for r in results}
        assert len(task_ids) == 10  # All unique task IDs

    def test_high_contention_scenario(
        self, detector: ConfigDriftDetector, temp_config_file: Path
    ):
        """Test behavior under high contention with many threads."""
        num_threads = 20
        operations_per_thread = 10

        def high_contention_worker():
            """Worker that performs many operations rapidly."""
            for _i in range(operations_per_thread):
                # Snapshot operation
                detector.take_snapshot(str(temp_config_file))

                # Comparison operation (might not find drift)
                detector.compare_snapshots(str(temp_config_file))

                # Summary operation
                detector.get_drift_summary()

                # Minimal delay to stress test locks
                time.sleep(0.001)

        # Run high contention test
        start_time = time.time()
        threads = []
        for _ in range(num_threads):
            t = Thread(target=high_contention_worker)
            threads.append(t)
            t.start()

        # Wait for completion
        for t in threads:
            t.join()

        duration = time.time() - start_time

        # Verify system remained stable
        with detector._snapshots_lock:
            assert str(temp_config_file) in detector._snapshots
            assert len(detector._snapshots[str(temp_config_file)]) > 0

        # Performance check - should complete in reasonable time even with locks
        assert duration < 30  # 30 seconds for stress test

        # Get final summary to ensure data integrity
        summary = detector.get_drift_summary()
        assert isinstance(summary, dict)
        assert all(key in summary for key in ["detection_enabled", "snapshots_stored"])
