"""Tests for configuration drift detection system."""

import json
import tempfile
import time
from datetime import UTC, datetime, timedelta
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from src.config.drift_detection import (
    ConfigDriftDetector,
    ConfigSnapshot,
    DriftDetectionConfig,
    DriftEvent,
    DriftSeverity,
    DriftType,
    get_drift_detector,
    get_drift_summary,
    initialize_drift_detector,
    run_drift_detection,
)


@pytest.fixture
def temp_config_file():
    """Create a temporary configuration file for testing."""
    with tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False) as f:
        config_data = {
            "database_url": "postgresql://localhost:5432/testdb",
            "api_key": "test_api_key_123",
            "debug": False,
            "max_connections": 10,
        }
        json.dump(config_data, f)
        f.flush()  # Ensure data is written to disk
        temp_path = Path(f.name)

    yield temp_path

    # Cleanup
    temp_path.unlink(missing_ok=True)


@pytest.fixture
def drift_config():
    """Create drift detection configuration for testing."""
    return DriftDetectionConfig(
        enabled=True,
        snapshot_interval_minutes=1,
        comparison_interval_minutes=1,
        monitored_paths=["test_config.json"],
        snapshot_retention_days=1,
        events_retention_days=1,
        integrate_with_task20_anomaly=False,  # Disable for unit tests
        use_performance_monitoring=False,  # Disable for unit tests
    )


@pytest.fixture
def mock_performance_monitor():
    """Mock performance monitor for testing."""
    with patch("src.config.drift_detection.get_performance_monitor") as mock:
        mock_monitor = MagicMock()
        mock_monitor.monitor_operation.return_value.__enter__ = MagicMock()
        mock_monitor.monitor_operation.return_value.__exit__ = MagicMock(
            return_value=None
        )
        mock.return_value = mock_monitor
        yield mock_monitor


@pytest.fixture
def mock_metrics_bridge():
    """Mock metrics bridge for testing."""
    with patch("src.config.drift_detection.get_metrics_bridge") as mock:
        mock_bridge = MagicMock()
        mock_bridge.create_custom_counter.return_value = MagicMock()
        mock_bridge.create_custom_gauge.return_value = MagicMock()
        mock_bridge.create_custom_histogram.return_value = MagicMock()
        mock.return_value = mock_bridge
        yield mock_bridge


class TestConfigSnapshot:
    """Test ConfigSnapshot functionality."""

    def test_config_snapshot_creation(self):
        """Test creating a configuration snapshot."""
        config_data = {"key": "value", "number": 123}
        snapshot = ConfigSnapshot(
            timestamp=datetime.now(tz=UTC),
            config_hash="test_hash",
            config_data=config_data,
            source="test.json",
        )

        assert snapshot.config_data == config_data
        assert snapshot.source == "test.json"
        assert snapshot.config_hash == "test_hash"
        assert isinstance(snapshot.timestamp, datetime)


class TestDriftEvent:
    """Test DriftEvent functionality."""

    def test_drift_event_creation(self):
        """Test creating a drift event."""
        event = DriftEvent(
            id="test_event_1",
            timestamp=datetime.now(tz=UTC),
            drift_type=DriftType.MANUAL_CHANGE,
            severity=DriftSeverity.MEDIUM,
            source="test.json",
            description="Test configuration change",
            old_value="old",
            new_value="new",
            diff_details={"path": "test_key"},
        )

        assert event.drift_type == DriftType.MANUAL_CHANGE
        assert event.severity == DriftSeverity.MEDIUM
        assert event.description == "Test configuration change"
        assert event.old_value == "old"
        assert event.new_value == "new"


class TestDriftDetectionConfig:
    """Test DriftDetectionConfig validation."""

    def test_default_config(self):
        """Test default configuration values."""
        config = DriftDetectionConfig()

        assert config.enabled is True
        assert config.snapshot_interval_minutes == 15
        assert config.comparison_interval_minutes == 5
        assert "src/config/" in config.monitored_paths
        assert "**/__pycache__/" in config.excluded_paths
        assert DriftSeverity.HIGH in config.alert_on_severity
        assert config.max_alerts_per_hour == 10

    def test_custom_config(self):
        """Test custom configuration values."""
        config = DriftDetectionConfig(
            enabled=False,
            snapshot_interval_minutes=30,
            max_alerts_per_hour=5,
            alert_on_severity=[DriftSeverity.CRITICAL],
        )

        assert config.enabled is False
        assert config.snapshot_interval_minutes == 30
        assert config.max_alerts_per_hour == 5
        assert config.alert_on_severity == [DriftSeverity.CRITICAL]


class TestConfigDriftDetector:
    """Test ConfigDriftDetector functionality."""

    def test_detector_initialization(self, drift_config, _mock_metrics_bridge):
        """Test drift detector initialization."""
        detector = ConfigDriftDetector(drift_config)

        assert detector.config == drift_config
        assert isinstance(detector._snapshots, dict)
        assert isinstance(detector._drift_events, list)
        assert isinstance(detector._last_alert_times, dict)

    def test_config_hash_calculation(self, drift_config, _mock_metrics_bridge):
        """Test configuration hash calculation."""
        detector = ConfigDriftDetector(drift_config)

        config1 = {"key1": "value1", "key2": "value2"}
        config2 = {"key2": "value2", "key1": "value1"}  # Same content, different order
        config3 = {"key1": "value1", "key2": "value3"}  # Different content

        hash1 = detector._calculate_config_hash(config1)
        hash2 = detector._calculate_config_hash(config2)
        hash3 = detector._calculate_config_hash(config3)

        assert hash1 == hash2  # Order shouldn't matter
        assert hash1 != hash3  # Different content should produce different hashes
        assert len(hash1) == 64  # SHA256 produces 64-character hex string

    def test_load_config_json(
        self, drift_config, temp_config_file, _mock_metrics_bridge
    ):
        """Test loading JSON configuration."""
        detector = ConfigDriftDetector(drift_config)

        config = detector._load_current_config(str(temp_config_file))

        assert config["database_url"] == "postgresql://localhost:5432/testdb"
        assert config["api_key"] == "test_api_key_123"
        assert config["debug"] is False
        assert config["max_connections"] == 10

    def test_load_config_env_file(self, drift_config, _mock_metrics_bridge):
        """Test loading environment file configuration."""
        detector = ConfigDriftDetector(drift_config)

        with tempfile.NamedTemporaryFile(mode="w", suffix=".env", delete=False) as f:
            f.write("DATABASE_URL=postgresql://localhost:5432/testdb\n")
            f.write("API_KEY=test_key_123\n")
            f.write("# This is a comment\n")
            f.write("DEBUG=false\n")
            env_file = Path(f.name)

        try:
            config = detector._load_current_config(str(env_file))

            assert config["DATABASE_URL"] == "postgresql://localhost:5432/testdb"
            assert config["API_KEY"] == "test_key_123"
            assert config["DEBUG"] == "false"
            assert "# This is a comment" not in config
        finally:
            env_file.unlink(missing_ok=True)

    def test_take_snapshot(self, drift_config, temp_config_file, _mock_metrics_bridge):
        """Test taking a configuration snapshot."""
        detector = ConfigDriftDetector(drift_config)

        snapshot = detector.take_snapshot(str(temp_config_file))

        assert isinstance(snapshot, ConfigSnapshot)
        assert snapshot.source == str(temp_config_file)
        assert len(snapshot.config_hash) == 64
        assert "database_url" in snapshot.config_data
        assert isinstance(snapshot.timestamp, datetime)

        # Verify snapshot is stored
        assert str(temp_config_file) in detector._snapshots
        assert len(detector._snapshots[str(temp_config_file)]) == 1

    def test_compare_snapshots_no_change(
        self, drift_config, temp_config_file, _mock_metrics_bridge
    ):
        """Test comparing snapshots with no changes."""
        detector = ConfigDriftDetector(drift_config)

        # Take two identical snapshots
        detector.take_snapshot(str(temp_config_file))
        time.sleep(0.01)  # Small delay to ensure different timestamps
        detector.take_snapshot(str(temp_config_file))

        events = detector.compare_snapshots(str(temp_config_file))

        assert len(events) == 0  # No changes should result in no events

    def test_compare_snapshots_with_changes(self, drift_config, _mock_metrics_bridge):
        """Test comparing snapshots with configuration changes."""
        detector = ConfigDriftDetector(drift_config)

        # Create two different configurations
        config1 = {"key1": "value1", "key2": "value2"}
        config2 = {
            "key1": "value1_modified",
            "key3": "value3",
        }  # Modified and added key

        # Manually create snapshots
        snapshot1 = ConfigSnapshot(
            timestamp=datetime.now(tz=UTC),
            config_hash=detector._calculate_config_hash(config1),
            config_data=config1,
            source="test.json",
        )
        snapshot2 = ConfigSnapshot(
            timestamp=datetime.now(tz=UTC),
            config_hash=detector._calculate_config_hash(config2),
            config_data=config2,
            source="test.json",
        )

        detector._snapshots["test.json"] = [snapshot1, snapshot2]

        events = detector.compare_snapshots("test.json")

        assert len(events) == 3  # Modified, removed, and added

        # Check event types
        event_types = {event.diff_details["type"] for event in events}
        assert "modified" in event_types
        assert "removed" in event_types
        assert "added" in event_types

    def test_classify_drift_type_security(self, drift_config, _mock_metrics_bridge):
        """Test drift type classification for security changes."""
        detector = ConfigDriftDetector(drift_config)

        security_change = {
            "type": "modified",
            "path": "api_key",
            "old_value": "old_key",
            "new_value": "new_key",
        }

        drift_type = detector._classify_drift_type(security_change, "test.json")
        assert drift_type == DriftType.SECURITY_DEGRADATION

    def test_classify_drift_type_environment(self, drift_config, _mock_metrics_bridge):
        """Test drift type classification for environment changes."""
        detector = ConfigDriftDetector(drift_config)

        env_change = {
            "type": "modified",
            "path": "environment",
            "old_value": "development",
            "new_value": "production",
        }

        drift_type = detector._classify_drift_type(env_change, "test.json")
        assert drift_type == DriftType.ENVIRONMENT_MISMATCH

    def test_calculate_drift_severity_critical(
        self, drift_config, _mock_metrics_bridge
    ):
        """Test drift severity calculation for critical changes."""
        detector = ConfigDriftDetector(drift_config)

        critical_change = {
            "type": "modified",
            "path": "secret_key",
            "old_value": "old_secret",
            "new_value": "new_secret",
        }

        severity = detector._calculate_drift_severity(critical_change, "test.json")
        assert severity == DriftSeverity.CRITICAL

    def test_calculate_drift_severity_high(self, drift_config, _mock_metrics_bridge):
        """Test drift severity calculation for high severity changes."""
        detector = ConfigDriftDetector(drift_config)

        high_change = {
            "type": "modified",
            "path": "database_url",
            "old_value": "old_url",
            "new_value": "new_url",
        }

        severity = detector._calculate_drift_severity(high_change, "test.json")
        assert severity == DriftSeverity.HIGH

    def test_calculate_drift_severity_low(self, drift_config, _mock_metrics_bridge):
        """Test drift severity calculation for low severity changes."""
        detector = ConfigDriftDetector(drift_config)

        low_change = {
            "type": "modified",
            "path": "log_level",
            "old_value": "INFO",
            "new_value": "DEBUG",
        }

        severity = detector._calculate_drift_severity(low_change, "test.json")
        assert severity == DriftSeverity.LOW

    def test_auto_remediation_detection(self, drift_config, _mock_metrics_bridge):
        """Test auto-remediation detection."""
        detector = ConfigDriftDetector(drift_config)

        # Security change should not be auto-remediable
        security_change = {
            "type": "modified",
            "path": "password",
            "old_value": "old_pass",
            "new_value": "new_pass",
        }
        assert not detector._is_auto_remediable(security_change)

        # Simple value change should be auto-remediable
        simple_change = {
            "type": "modified",
            "path": "timeout",
            "old_value": 30,
            "new_value": 60,
        }
        assert detector._is_auto_remediable(simple_change)

    def test_remediation_suggestions(self, drift_config, _mock_metrics_bridge):
        """Test remediation suggestion generation."""
        detector = ConfigDriftDetector(drift_config)

        modified_change = {
            "type": "modified",
            "path": "timeout",
            "old_value": 30,
            "new_value": 60,
        }
        suggestion = detector._generate_remediation_suggestion(modified_change)
        assert suggestion == "Revert 'timeout' to previous value: 30"

        added_change = {
            "type": "added",
            "path": "new_key",
            "old_value": None,
            "new_value": "new_value",
        }
        suggestion = detector._generate_remediation_suggestion(added_change)
        assert suggestion == "Remove newly added key: 'new_key'"

    def test_should_alert_severity_threshold(self, drift_config, _mock_metrics_bridge):
        """Test alert threshold based on severity."""
        detector = ConfigDriftDetector(drift_config)

        high_severity_event = DriftEvent(
            id="test_1",
            timestamp=datetime.now(tz=UTC),
            drift_type=DriftType.MANUAL_CHANGE,
            severity=DriftSeverity.HIGH,
            source="test.json",
            description="High severity change",
            old_value="old",
            new_value="new",
            diff_details={},
        )

        low_severity_event = DriftEvent(
            id="test_2",
            timestamp=datetime.now(tz=UTC),
            drift_type=DriftType.MANUAL_CHANGE,
            severity=DriftSeverity.LOW,
            source="test.json",
            description="Low severity change",
            old_value="old",
            new_value="new",
            diff_details={},
        )

        assert detector.should_alert(high_severity_event)
        assert not detector.should_alert(low_severity_event)

    def test_should_alert_rate_limiting(self, drift_config, _mock_metrics_bridge):
        """Test alert rate limiting."""
        # Set low rate limit for testing
        drift_config.max_alerts_per_hour = 1
        detector = ConfigDriftDetector(drift_config)

        event = DriftEvent(
            id="test_1",
            timestamp=datetime.now(tz=UTC),
            drift_type=DriftType.MANUAL_CHANGE,
            severity=DriftSeverity.HIGH,
            source="test.json",
            description="High severity change",
            old_value="old",
            new_value="new",
            diff_details={},
        )

        # First alert should pass
        assert detector.should_alert(event)

        # Second alert should be rate limited
        event.id = "test_2"
        assert not detector.should_alert(event)

    def test_cleanup_old_snapshots(self, drift_config, _mock_metrics_bridge):
        """Test cleanup of old snapshots."""
        # Set short retention for testing - use 0.5 days so recent snapshot stays
        drift_config.snapshot_retention_days = 0.5
        detector = ConfigDriftDetector(drift_config)

        # Create old snapshots
        old_snapshot = ConfigSnapshot(
            timestamp=datetime.now(tz=UTC)
            - timedelta(days=1),  # 1 day old - should be removed
            config_hash="old_hash",
            config_data={"old": "data"},
            source="test.json",
        )
        recent_snapshot = ConfigSnapshot(
            timestamp=datetime.now(tz=UTC)
            - timedelta(minutes=1),  # 1 minute old - should stay
            config_hash="new_hash",
            config_data={"new": "data"},
            source="test.json",
        )

        detector._snapshots["test.json"] = [old_snapshot, recent_snapshot]

        detector._cleanup_old_snapshots("test.json")

        # Only recent snapshot should remain
        assert len(detector._snapshots["test.json"]) == 1
        assert detector._snapshots["test.json"][0].config_hash == "new_hash"

    def test_get_drift_summary(self, drift_config, _mock_metrics_bridge):
        """Test drift summary generation."""
        detector = ConfigDriftDetector(drift_config)

        # Add some test events
        recent_event = DriftEvent(
            id="test_1",
            timestamp=datetime.now(tz=UTC),
            drift_type=DriftType.MANUAL_CHANGE,
            severity=DriftSeverity.HIGH,
            source="test.json",
            description="Recent change",
            old_value="old",
            new_value="new",
            diff_details={},
            auto_remediable=True,
        )
        detector._drift_events.append(recent_event)

        # Add some snapshots
        detector._snapshots["test1.json"] = [
            ConfigSnapshot(
                timestamp=datetime.now(tz=UTC),
                config_hash="hash1",
                config_data={},
                source="test1.json",
            )
        ]
        detector._snapshots["test2.json"] = [
            ConfigSnapshot(
                timestamp=datetime.now(tz=UTC),
                config_hash="hash2",
                config_data={},
                source="test2.json",
            )
        ]

        summary = detector.get_drift_summary()

        assert summary["detection_enabled"] is True
        assert summary["monitored_sources"] == len(drift_config.monitored_paths)
        assert summary["snapshots_stored"] == 2
        assert summary["recent_events_24h"] == 1
        assert summary["auto_remediable_events"] == 1
        assert summary["severity_breakdown"]["high"] == 1


class TestGlobalFunctions:
    """Test global drift detection functions."""

    def test_initialize_and_get_detector(self, drift_config, _mock_metrics_bridge):
        """Test global detector initialization and retrieval."""
        detector = initialize_drift_detector(drift_config)

        assert isinstance(detector, ConfigDriftDetector)
        assert detector.config == drift_config

        # Test retrieval
        retrieved_detector = get_drift_detector()
        assert retrieved_detector is detector

    def test_get_detector_not_initialized(self):
        """Test getting detector when not initialized."""
        # Reset global state
        import src.config.drift_detection

        src.config.drift_detection._drift_detector = None

        with pytest.raises(RuntimeError, match="not initialized"):
            get_drift_detector()

    @patch("src.config.drift_detection.get_drift_detector")
    def test_run_drift_detection(
        self, mock_get_detector, _drift_config, _mock_metrics_bridge
    ):
        """Test global drift detection run."""
        mock_detector = MagicMock()
        mock_detector.run_detection_cycle.return_value = []
        mock_get_detector.return_value = mock_detector

        events = run_drift_detection()

        assert events == []
        mock_detector.run_detection_cycle.assert_called_once()

    @patch("src.config.drift_detection.get_drift_detector")
    def test_get_drift_summary_global(
        self, mock_get_detector, _drift_config, _mock_metrics_bridge
    ):
        """Test global drift summary."""
        mock_detector = MagicMock()
        mock_detector.get_drift_summary.return_value = {"test": "summary"}
        mock_get_detector.return_value = mock_detector

        summary = get_drift_summary()

        assert summary == {"test": "summary"}
        mock_detector.get_drift_summary.assert_called_once()


class TestIntegrationWithTask20:
    """Test integration with existing Task 20 infrastructure."""

    def test_performance_monitor_integration(
        self, drift_config, mock_performance_monitor, _mock_metrics_bridge
    ):
        """Test integration with performance monitoring."""
        drift_config.use_performance_monitoring = True

        with patch(
            "src.config.drift_detection.get_performance_monitor",
            return_value=mock_performance_monitor,
        ):
            detector = ConfigDriftDetector(drift_config)

            # Take a snapshot which should use performance monitoring
            with tempfile.NamedTemporaryFile(
                mode="w", suffix=".json", delete=False
            ) as f:
                json.dump({"test": "data"}, f)
                temp_file = Path(f.name)

            try:
                detector.take_snapshot(str(temp_file))

                # Verify performance monitor was used
                mock_performance_monitor.monitor_operation.assert_called()
            finally:
                temp_file.unlink(missing_ok=True)

    def test_metrics_bridge_integration(self, drift_config, mock_metrics_bridge):
        """Test integration with metrics bridge."""
        _detector = ConfigDriftDetector(drift_config)

        # Verify custom metrics were created
        assert mock_metrics_bridge.create_custom_counter.called
        assert mock_metrics_bridge.create_custom_gauge.called
        assert mock_metrics_bridge.create_custom_histogram.called

        # Verify metrics creation calls
        counter_calls = [
            call[0][0]
            for call in mock_metrics_bridge.create_custom_counter.call_args_list
        ]
        assert "config_drift_events_total" in counter_calls

        gauge_calls = [
            call[0][0]
            for call in mock_metrics_bridge.create_custom_gauge.call_args_list
        ]
        assert "config_drift_severity_current" in gauge_calls

        histogram_calls = [
            call[0][0]
            for call in mock_metrics_bridge.create_custom_histogram.call_args_list
        ]
        assert "config_comparison_duration_ms" in histogram_calls
