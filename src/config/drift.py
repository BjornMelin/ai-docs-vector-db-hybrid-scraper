#!/usr/bin/env python3
"""Configuration drift detection system."""

from datetime import UTC, datetime
from enum import Enum
from pathlib import Path
from typing import Any

from pydantic import BaseModel, Field


class DriftType(str, Enum):
    """Types of configuration drift."""

    ADDED = "added"
    REMOVED = "removed"
    MODIFIED = "modified"
    PERMISSIONS = "permissions"
    STRUCTURE = "structure"


class DriftSeverity(str, Enum):
    """Severity levels for drift events."""

    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    CRITICAL = "critical"


class DriftEvent(BaseModel):
    """Represents a configuration drift event."""

    drift_type: DriftType
    severity: DriftSeverity
    path: str
    old_value: Any | None = None
    new_value: Any | None = None
    timestamp: datetime = Field(default_factory=lambda: datetime.now(UTC))
    description: str = ""


class ConfigSnapshot(BaseModel):
    """Snapshot of configuration at a point in time."""

    timestamp: datetime = Field(default_factory=lambda: datetime.now(UTC))
    config_data: dict[str, Any] = Field(default_factory=dict)
    checksum: str = ""
    version: str = "1.0.0"


class ConfigDriftDetector:
    """Detects configuration drift over time."""

    def __init__(self, config_dir: Path, snapshot_interval: int = 3600):
        """Initialize the drift detector."""
        self.config_dir = config_dir
        self.snapshot_interval = snapshot_interval
        self.snapshots: list[ConfigSnapshot] = []
        self.drift_events: list[DriftEvent] = []

    def take_snapshot(self) -> ConfigSnapshot:
        """Take a snapshot of current configuration."""
        snapshot = ConfigSnapshot(
            config_data={"placeholder": "data"},
            checksum="mock_checksum",
        )
        self.snapshots.append(snapshot)
        return snapshot

    def detect_drift(self) -> list[DriftEvent]:
        """Detect configuration drift."""
        # Placeholder implementation for testing
        return self.drift_events

    def get_latest_snapshot(self) -> ConfigSnapshot | None:
        """Get the most recent configuration snapshot."""
        return self.snapshots[-1] if self.snapshots else None


# Global drift detector instance
_drift_detector: ConfigDriftDetector | None = None


def initialize_drift_detector(
    config_dir: Path, snapshot_interval: int = 3600
) -> ConfigDriftDetector:
    """Initialize the global drift detector."""
    global _drift_detector
    _drift_detector = ConfigDriftDetector(config_dir, snapshot_interval)
    return _drift_detector


def get_drift_detector() -> ConfigDriftDetector | None:
    """Get the global drift detector instance."""
    return _drift_detector


def run_drift_detection() -> list[DriftEvent]:
    """Run drift detection and return events."""
    detector = get_drift_detector()
    if detector:
        return detector.detect_drift()
    return []


def get_drift_summary() -> dict[str, Any]:
    """Get a summary of drift detection results."""
    detector = get_drift_detector()
    if not detector:
        return {"error": "Drift detector not initialized"}

    events = detector.drift_events
    return {
        "total_events": len(events),
        "by_severity": {
            severity.value: len([e for e in events if e.severity == severity])
            for severity in DriftSeverity
        },
        "by_type": {
            drift_type.value: len([e for e in events if e.drift_type == drift_type])
            for drift_type in DriftType
        },
        "latest_snapshot": detector.get_latest_snapshot().model_dump()
        if detector.get_latest_snapshot()
        else None,
    }
