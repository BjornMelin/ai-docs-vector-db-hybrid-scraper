"""Configuration drift detection and alerting system.

This module provides configuration drift detection by leveraging existing Task 20
anomaly detection infrastructure and observability systems to monitor configuration
changes, detect unauthorized modifications, and alert on compliance violations.
"""

import asyncio  # noqa: PLC0415
import hashlib
import json  # noqa: PLC0415
import logging  # noqa: PLC0415
import threading
import time  # noqa: PLC0415
from dataclasses import dataclass, field
from datetime import UTC, datetime, timedelta
from enum import Enum
from pathlib import Path
from typing import Any

import yaml
from pydantic import BaseModel, Field

from ..services.observability.metrics_bridge import get_metrics_bridge
from ..services.observability.performance import get_performance_monitor


logger = logging.getLogger(__name__)


class DriftSeverity(str, Enum):
    """Configuration drift severity levels."""

    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    CRITICAL = "critical"


class DriftType(str, Enum):
    """Types of configuration drift."""

    MANUAL_CHANGE = "manual_change"
    SCHEMA_VIOLATION = "schema_violation"
    ENVIRONMENT_MISMATCH = "environment_mismatch"
    UNAUTHORIZED_ACCESS = "unauthorized_access"
    COMPLIANCE_VIOLATION = "compliance_violation"
    SECURITY_DEGRADATION = "security_degradation"


@dataclass
class ConfigSnapshot:
    """Configuration state snapshot for comparison."""

    timestamp: datetime
    config_hash: str
    config_data: dict[str, Any]
    source: str  # file path, API endpoint, etc.
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass
class DriftEvent:
    """Configuration drift detection event."""

    id: str
    timestamp: datetime
    drift_type: DriftType
    severity: DriftSeverity
    source: str
    description: str
    old_value: Any
    new_value: Any
    diff_details: dict[str, Any]
    auto_remediable: bool = False
    remediation_suggestion: str | None = None
    metadata: dict[str, Any] = field(default_factory=dict)


class DriftDetectionConfig(BaseModel):
    """Configuration for drift detection system."""

    # Detection settings
    enabled: bool = Field(default=True)
    snapshot_interval_minutes: int = Field(default=15, gt=0, le=1440)
    comparison_interval_minutes: int = Field(default=5, gt=0, le=60)

    # Monitoring configuration
    monitored_paths: list[str] = Field(
        default_factory=lambda: [
            "src/config/",
            ".env",
            "config.yaml",
            "config.json",
            "docker-compose.yml",
            "docker-compose.yaml",
        ]
    )
    excluded_paths: list[str] = Field(
        default_factory=lambda: [
            "**/__pycache__/",
            "**/*.pyc",
            "**/logs/",
            "**/cache/",
            "**/tmp/",
        ]
    )

    # Alerting thresholds
    alert_on_severity: list[DriftSeverity] = Field(
        default_factory=lambda: [DriftSeverity.HIGH, DriftSeverity.CRITICAL]
    )
    max_alerts_per_hour: int = Field(default=10, gt=0)

    # Storage configuration
    snapshot_retention_days: int = Field(default=30, gt=0)
    events_retention_days: int = Field(default=90, gt=0)

    # Integration settings
    integrate_with_task20_anomaly: bool = Field(default=True)
    use_performance_monitoring: bool = Field(default=True)

    # Auto-remediation
    enable_auto_remediation: bool = Field(default=False)
    auto_remediation_severity_threshold: DriftSeverity = Field(
        default=DriftSeverity.HIGH
    )


class ConfigDriftDetector:
    """Configuration drift detection engine using existing observability infrastructure."""

    def __init__(self, config: DriftDetectionConfig):
        """Initialize configuration drift detector.

        Args:
            config: Drift detection configuration
        """
        self.config = config
        self._snapshots: dict[str, list[ConfigSnapshot]] = {}
        self._drift_events: list[DriftEvent] = []
        self._last_alert_times: dict[str, datetime] = {}

        # Thread safety locks
        self._snapshots_lock = threading.RLock()  # Re-entrant lock for snapshots
        self._events_lock = threading.RLock()  # Re-entrant lock for drift events
        self._alerts_lock = threading.RLock()  # Re-entrant lock for alert tracking

        # Async lock for async operations (if needed in future)
        self._async_lock = asyncio.Lock()

        # Initialize integrations with existing infrastructure
        self._setup_integrations()

    def _setup_integrations(self) -> None:
        """Set up integrations with existing monitoring infrastructure."""
        try:
            # Integrate with performance monitoring from Task 20
            if self.config.use_performance_monitoring:
                self.performance_monitor = get_performance_monitor()
                logger.info("Integrated with existing performance monitoring system")
            else:
                self.performance_monitor = None
        except Exception:
            logger.warning("Failed to initialize performance monitor")
            self.performance_monitor = None

        try:
            # Integrate with metrics bridge for alerting
            self.metrics_bridge = get_metrics_bridge()

            # Create custom metrics for drift detection
            self._setup_custom_metrics()
            logger.info("Integrated with existing metrics bridge system")
        except Exception:
            logger.warning("Failed to initialize metrics bridge")
            self.metrics_bridge = None

    def _setup_custom_metrics(self) -> None:
        """Set up custom metrics for configuration drift detection."""
        if not self.metrics_bridge:
            return

        # Create drift detection metrics
        self.drift_counter = self.metrics_bridge.create_custom_counter(
            "config_drift_events_total",
            "Total number of configuration drift events detected",
        )

        self.drift_severity_gauge = self.metrics_bridge.create_custom_gauge(
            "config_drift_severity_current",
            "Current maximum configuration drift severity",
        )

        self.snapshot_age_gauge = self.metrics_bridge.create_custom_gauge(
            "config_snapshot_age_seconds",
            "Age of last configuration snapshot in seconds",
        )

        self.comparison_duration = self.metrics_bridge.create_custom_histogram(
            "config_comparison_duration_ms",
            "Time taken to compare configuration snapshots",
            "ms",
        )

    def _calculate_config_hash(self, config_data: dict[str, Any]) -> str:
        """Calculate deterministic hash of configuration data.

        Args:
            config_data: Configuration data to hash

        Returns:
            SHA256 hash of configuration
        """
        # Sort keys to ensure deterministic hashing
        config_json = json.dumps(config_data, sort_keys=True, separators=(",", ":"))
        return hashlib.sha256(config_json.encode()).hexdigest()

    def _load_current_config(self, source: str) -> dict[str, Any]:
        """Load current configuration from source.

        Args:
            source: Configuration source (file path, etc.)

        Returns:
            Current configuration data
        """
        source_path = Path(source)

        if not source_path.exists():
            return {}

        try:
            if source_path.suffix in [".json"]:
                with source_path.open() as f:
                    return json.load(f)
            elif source_path.suffix in [".yaml", ".yml"]:
                with source_path.open() as f:
                    return yaml.safe_load(f) or {}
            elif source_path.suffix in [".env"]:
                # Parse environment file
                config = {}
                with source_path.open() as f:
                    for line in f:
                        line = line.strip()
                        if line and not line.startswith("#") and "=" in line:
                            key, value = line.split("=", 1)
                            config[key.strip()] = value.strip()
                return config
            else:
                # For other files, just check existence and modification time
                stat = source_path.stat()
                return {
                    "size": stat.st_size,
                    "mtime": stat.st_mtime,
                    "permissions": oct(stat.st_mode),
                }
        except Exception:
            logger.warning("Failed to load config from {source}")
            return {}

    def take_snapshot(self, source: str) -> ConfigSnapshot:
        """Take a configuration snapshot.

        Args:
            source: Configuration source identifier

        Returns:
            Configuration snapshot
        """
        start_time = time.time()

        try:
            # Load current configuration
            config_data = self._load_current_config(source)

            # Calculate hash
            config_hash = self._calculate_config_hash(config_data)

            # Create snapshot
            snapshot = ConfigSnapshot(
                timestamp=datetime.now(tz=UTC),
                config_hash=config_hash,
                config_data=config_data,
                source=source,
                metadata={
                    "snapshot_method": "file_based",
                    "data_size": len(json.dumps(config_data)),
                },
            )

            # Store snapshot with thread safety
            with self._snapshots_lock:
                if source not in self._snapshots:
                    self._snapshots[source] = []
                self._snapshots[source].append(snapshot)

            # Clean up old snapshots
            self._cleanup_old_snapshots(source)

            # Record metrics
            if self.metrics_bridge:
                _duration_ms = (
                    time.time() - start_time
                ) * 1000  # Recorded but not used in current implementation
                self.snapshot_age_gauge.set(0, {"source": source})

                # Record with performance monitor if available
                if self.performance_monitor:
                    with self.performance_monitor.monitor_operation(
                        "config_snapshot", category="config_drift"
                    ) as perf_data:
                        perf_data["custom_metrics"]["snapshot_size"] = len(
                            json.dumps(config_data)
                        )
                        perf_data["custom_metrics"]["source"] = source

            logger.debug("Created config snapshot for {source}")
            return snapshot

        except Exception:
            logger.exception("Failed to take snapshot of {source}")
            raise

    def compare_snapshots(self, source: str) -> list[DriftEvent]:
        """Compare recent snapshots to detect drift.

        Args:
            source: Configuration source to compare

        Returns:
            List of detected drift events
        """
        # Thread-safe snapshot access
        with self._snapshots_lock:
            if source not in self._snapshots or len(self._snapshots[source]) < 2:
                return []

            # Make copies of snapshots to avoid holding lock during comparison
            snapshots = self._snapshots[source].copy()
            current = snapshots[-1]
            previous = snapshots[-2]

        start_time = time.time()
        events = []

        try:
            # Quick hash comparison
            if current.config_hash == previous.config_hash:
                return []

            # Detailed comparison for drift analysis
            drift_details = self._analyze_config_changes(
                previous.config_data, current.config_data, source
            )

            # Create drift events
            for change in drift_details:
                drift_event = DriftEvent(
                    id=f"{source}_{int(time.time())}_{change['path']}",
                    timestamp=current.timestamp,
                    drift_type=self._classify_drift_type(change, source),
                    severity=self._calculate_drift_severity(change, source),
                    source=source,
                    description=change["description"],
                    old_value=change["old_value"],
                    new_value=change["new_value"],
                    diff_details=change,
                    auto_remediable=self._is_auto_remediable(change),
                    remediation_suggestion=self._generate_remediation_suggestion(
                        change
                    ),
                    metadata={
                        "snapshot_comparison": True,
                        "change_path": change["path"],
                        "change_type": change["type"],
                    },
                )
                events.append(drift_event)

                # Thread-safe event storage
                with self._events_lock:
                    self._drift_events.append(drift_event)

            # Record metrics
            if self.metrics_bridge and events:
                duration_ms = (time.time() - start_time) * 1000
                self.comparison_duration.record(duration_ms, {"source": source})

                for event in events:
                    self.drift_counter.add(
                        1,
                        {
                            "source": source,
                            "drift_type": event.drift_type.value,
                            "severity": event.severity.value,
                        },
                    )

                # Update severity gauge with highest severity
                max_severity = max(
                    (self._severity_to_int(event.severity) for event in events),
                    default=0,
                )
                self.drift_severity_gauge.set(max_severity, {"source": source})

            logger.info(f"Detected {len(events)} drift events for {source}")
            return events

        except Exception:
            logger.exception("Failed to compare snapshots for {source}")
            return []

    def _analyze_config_changes(
        self, old_config: dict[str, Any], new_config: dict[str, Any], _source: str
    ) -> list[dict[str, Any]]:
        """Analyze configuration changes between two configurations.

        Args:
            old_config: Previous configuration
            new_config: Current configuration
            source: Configuration source

        Returns:
            List of detected changes
        """
        changes = []

        # Find added keys
        changes.extend(
            [
                {
                    "type": "added",
                    "path": key,
                    "old_value": None,
                    "new_value": new_config[key],
                    "description": f"New configuration key '{key}' added",
                }
                for key in new_config.keys() - old_config.keys()
            ]
        )

        # Find removed keys
        changes.extend(
            [
                {
                    "type": "removed",
                    "path": key,
                    "old_value": old_config[key],
                    "new_value": None,
                    "description": f"Configuration key '{key}' removed",
                }
                for key in old_config.keys() - new_config.keys()
            ]
        )

        # Find modified keys
        changes.extend(
            [
                {
                    "type": "modified",
                    "path": key,
                    "old_value": old_config[key],
                    "new_value": new_config[key],
                    "description": f"Configuration key '{key}' changed from '{old_config[key]}' to '{new_config[key]}'",
                }
                for key in old_config.keys() & new_config.keys()
                if old_config[key] != new_config[key]
            ]
        )

        return changes

    def _classify_drift_type(self, change: dict[str, Any], _source: str) -> DriftType:
        """Classify the type of configuration drift.

        Args:
            change: Configuration change details
            source: Configuration source

        Returns:
            Drift type classification
        """
        change_type = change.get("type", "")
        path = change.get("path", "")

        # Security-related changes
        if any(
            security_key in path.lower()
            for security_key in [
                "password",
                "key",
                "secret",
                "token",
                "auth",
                "security",
                "ssl",
                "tls",
            ]
        ):
            return DriftType.SECURITY_DEGRADATION

        # Environment-specific changes
        if any(
            env_key in path.lower()
            for env_key in ["env", "environment", "stage", "tier", "debug"]
        ):
            return DriftType.ENVIRONMENT_MISMATCH

        # Schema violations (unexpected structure changes)
        if change_type in ["added", "removed"] and "required" in str(
            change.get("old_value", "")
        ):
            return DriftType.SCHEMA_VIOLATION

        # Default to manual change
        return DriftType.MANUAL_CHANGE

    def _calculate_drift_severity(
        self, change: dict[str, Any], _source: str
    ) -> DriftSeverity:
        """Calculate severity of configuration drift.

        Args:
            change: Configuration change details
            source: Configuration source

        Returns:
            Drift severity level
        """
        change_type = change.get("type", "")
        path = change.get("path", "").lower()

        # Critical severity triggers
        if any(
            critical_key in path
            for critical_key in ["password", "secret", "private_key", "token"]
        ):
            return DriftSeverity.CRITICAL

        # High severity triggers
        if any(
            high_key in path
            for high_key in ["security", "auth", "ssl", "database_url", "api_key"]
        ):
            return DriftSeverity.HIGH

        # Medium severity triggers
        if change_type in ["removed"] or any(
            medium_key in path
            for medium_key in ["url", "endpoint", "host", "port", "timeout"]
        ):
            return DriftSeverity.MEDIUM

        # Default to low severity
        return DriftSeverity.LOW

    def _is_auto_remediable(self, change: dict[str, Any]) -> bool:
        """Determine if a configuration change can be automatically remediated.

        Args:
            change: Configuration change details

        Returns:
            True if auto-remediable
        """
        # Simple heuristics for auto-remediation
        change_type = change.get("type", "")
        path = change.get("path", "").lower()

        # Never auto-remediate security changes
        if any(
            security_key in path
            for security_key in ["password", "key", "secret", "token", "auth"]
        ):
            return False

        # Auto-remediate simple value reversions
        return bool(
            change_type == "modified"
            and isinstance(change.get("old_value"), str | int | bool)
        )

    def _generate_remediation_suggestion(self, change: dict[str, Any]) -> str | None:
        """Generate remediation suggestion for configuration drift.

        Args:
            change: Configuration change details

        Returns:
            Remediation suggestion or None
        """
        change_type = change.get("type", "")
        path = change.get("path", "")
        old_value = change.get("old_value")

        if change_type == "modified" and old_value is not None:
            return "Revert '{path}' to previous value"
        elif change_type == "added":
            return f"Remove newly added key: '{path}'"
        elif change_type == "removed":
            return f"Restore removed key: '{path}'"

        return "Manual review and remediation required"

    def _severity_to_int(self, severity: DriftSeverity) -> int:
        """Convert severity enum to integer for comparison.

        Args:
            severity: Drift severity

        Returns:
            Integer representation
        """
        return {
            DriftSeverity.LOW: 1,
            DriftSeverity.MEDIUM: 2,
            DriftSeverity.HIGH: 3,
            DriftSeverity.CRITICAL: 4,
        }[severity]

    def _cleanup_old_snapshots(self, source: str) -> None:
        """Clean up old snapshots based on retention policy.

        Args:
            source: Configuration source
        """
        with self._snapshots_lock:
            if source not in self._snapshots:
                return

            cutoff_time = datetime.now(tz=UTC) - timedelta(
                days=self.config.snapshot_retention_days
            )
            original_count = len(self._snapshots[source])

            self._snapshots[source] = [
                snapshot
                for snapshot in self._snapshots[source]
                if snapshot.timestamp > cutoff_time
            ]

            cleaned_count = original_count - len(self._snapshots[source])
            if cleaned_count > 0:
                logger.debug(f"Cleaned up {cleaned_count} old snapshots for {source}")

    def _cleanup_old_events(self) -> None:
        """Clean up old drift events based on retention policy."""
        with self._events_lock:
            cutoff_time = datetime.now(tz=UTC) - timedelta(
                days=self.config.events_retention_days
            )
            original_count = len(self._drift_events)

            self._drift_events = [
                event for event in self._drift_events if event.timestamp > cutoff_time
            ]

            cleaned_count = original_count - len(self._drift_events)
            if cleaned_count > 0:
                logger.debug(f"Cleaned up {cleaned_count} old drift events")

    def should_alert(self, event: DriftEvent) -> bool:
        """Determine if an alert should be sent for a drift event.

        Args:
            event: Drift event to evaluate

        Returns:
            True if alert should be sent
        """
        # Check severity threshold
        if event.severity not in self.config.alert_on_severity:
            return False

        # Check rate limiting with thread safety
        alert_key = f"{event.source}_{event.drift_type.value}"
        now = datetime.now(tz=UTC)

        with self._alerts_lock:
            if alert_key in self._last_alert_times:
                time_since_last = now - self._last_alert_times[alert_key]
                if time_since_last < timedelta(hours=1):
                    # Count recent alerts
                    recent_alerts = sum(
                        1
                        for alert_time in self._last_alert_times.values()
                        if now - alert_time < timedelta(hours=1)
                    )
                    if recent_alerts >= self.config.max_alerts_per_hour:
                        return False

            self._last_alert_times[alert_key] = now
            return True

    def send_alert(self, event: DriftEvent) -> None:
        """Send alert for configuration drift event.

        Args:
            event: Drift event to alert on
        """
        # Use existing performance monitoring for alerting
        if self.performance_monitor:
            with self.performance_monitor.monitor_operation(
                "config_drift_alert", category="security"
            ) as perf_data:
                perf_data["custom_metrics"]["drift_type"] = event.drift_type.value
                perf_data["custom_metrics"]["severity"] = event.severity.value
                perf_data["custom_metrics"]["source"] = event.source

        # Log alert (could be extended to send to external systems)
        logger.warning(
            f"CONFIGURATION DRIFT ALERT: {event.severity.value.upper()} - "
            f"{event.description} in {event.source}"
        )

        # Record alert metrics
        if self.metrics_bridge:
            alert_counter = self.metrics_bridge.create_custom_counter(
                "config_drift_alerts_total", "Total configuration drift alerts sent"
            )
            alert_counter.add(
                1,
                {
                    "source": event.source,
                    "drift_type": event.drift_type.value,
                    "severity": event.severity.value,
                },
            )

    def run_detection_cycle(self) -> list[DriftEvent]:
        """Run a complete drift detection cycle for all monitored sources.

        Returns:
            List of all detected drift events
        """
        if not self.config.enabled:
            return []

        all_events = []

        # Take snapshots and compare for each monitored path
        for source in self.config.monitored_paths:
            try:
                # Take current snapshot
                self.take_snapshot(source)

                # Compare with previous snapshots
                events = self.compare_snapshots(source)
                all_events.extend(events)

                # Send alerts for qualifying events
                for event in events:
                    if self.should_alert(event):
                        self.send_alert(event)

            except Exception:
                logger.exception("Failed drift detection for {source}")

        # Clean up old data
        self._cleanup_old_events()

        return all_events

    def get_drift_summary(self) -> dict[str, Any]:
        """Get summary of configuration drift status.

        Returns:
            Drift status summary
        """
        # Thread-safe access to drift events
        with self._events_lock:
            recent_events = [
                event
                for event in self._drift_events
                if event.timestamp > datetime.now(tz=UTC) - timedelta(hours=24)
            ]

        severity_counts = {}
        for severity in DriftSeverity:
            severity_counts[severity.value] = sum(
                1 for event in recent_events if event.severity == severity
            )

        # Thread-safe access to snapshots
        with self._snapshots_lock:
            snapshots_count = sum(
                len(snapshots) for snapshots in self._snapshots.values()
            )

        return {
            "detection_enabled": self.config.enabled,
            "monitored_sources": len(self.config.monitored_paths),
            "snapshots_stored": snapshots_count,
            "recent_events_24h": len(recent_events),
            "severity_breakdown": severity_counts,
            "auto_remediable_events": sum(
                1 for event in recent_events if event.auto_remediable
            ),
            "last_detection_run": max(
                (event.timestamp for event in recent_events), default=None
            ),
        }


# Global drift detector instance
_drift_detector: ConfigDriftDetector | None = None


def initialize_drift_detector(config: DriftDetectionConfig) -> ConfigDriftDetector:
    """Initialize global configuration drift detector.

    Args:
        config: Drift detection configuration

    Returns:
        Initialized drift detector
    """
    global _drift_detector
    _drift_detector = ConfigDriftDetector(config)
    return _drift_detector


def get_drift_detector() -> ConfigDriftDetector:
    """Get global drift detector instance.

    Returns:
        Global drift detector

    Raises:
        RuntimeError: If detector not initialized
    """
    if _drift_detector is None:
        raise RuntimeError(
            "Configuration drift detector not initialized. "
            "Call initialize_drift_detector() first."
        )
    return _drift_detector


# Convenience functions
def run_drift_detection() -> list[DriftEvent]:
    """Run drift detection using global detector."""
    detector = get_drift_detector()
    return detector.run_detection_cycle()


def get_drift_summary() -> dict[str, Any]:
    """Get drift summary using global detector."""
    detector = get_drift_detector()
    return detector.get_drift_summary()
