#!/usr/bin/env python3
"""Configuration drift detection system backed by filesystem hashes."""

from __future__ import annotations

import json
import logging
from collections.abc import Mapping
from datetime import UTC, datetime
from enum import Enum
from hashlib import sha256
from pathlib import Path
from typing import Any

from pydantic import BaseModel, Field


logger = logging.getLogger(__name__)


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

    id: str = Field(default_factory=lambda: f"drift_{datetime.now(UTC).timestamp()}")
    drift_type: DriftType
    severity: DriftSeverity
    source: str = Field(default="", description="Source of the drift")
    path: str
    old_value: Any | None = None
    new_value: Any | None = None
    timestamp: datetime = Field(default_factory=lambda: datetime.now(UTC))
    description: str = ""
    auto_remediable: bool = Field(
        default=False, description="Whether drift can be auto-remediated"
    )
    remediation_suggestion: str | None = Field(
        default=None, description="Suggested remediation steps"
    )


class ConfigSnapshot(BaseModel):
    """Snapshot of configuration at a point in time."""

    timestamp: datetime = Field(default_factory=lambda: datetime.now(UTC))
    config_data: dict[str, Any] = Field(default_factory=dict)
    config_hash: str = Field(default="", description="Stable hash of the snapshot data")
    checksum: str = Field(
        default="", description="Digest for snapshot integrity validation"
    )
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
        """Capture a snapshot of the monitored configuration directory."""

        state = self._collect_state()
        digest = _hash_mapping(state)
        snapshot = ConfigSnapshot(
            config_data=state,
            config_hash=digest,
            checksum=digest,
        )
        self.snapshots.append(snapshot)
        logger.debug(
            "Captured configuration snapshot %s with %d entries",
            snapshot.config_hash,
            len(state),
        )
        return snapshot

    def detect_drift(self) -> list[DriftEvent]:
        """Compare the latest snapshots and record drift events."""

        if len(self.snapshots) < 2:
            logger.debug("Not enough snapshots to detect drift")
            return []

        previous = self.snapshots[-2].config_data
        current = self.snapshots[-1].config_data

        prev_keys = set(previous.keys())
        curr_keys = set(current.keys())

        events: list[DriftEvent] = []

        for added in sorted(curr_keys - prev_keys):
            events.append(
                DriftEvent(
                    drift_type=DriftType.ADDED,
                    severity=DriftSeverity.MEDIUM,
                    source="filesystem",
                    path=added,
                    new_value=current[added],
                    description="Configuration entry added",
                )
            )

        for removed in sorted(prev_keys - curr_keys):
            events.append(
                DriftEvent(
                    drift_type=DriftType.REMOVED,
                    severity=DriftSeverity.HIGH,
                    source="filesystem",
                    path=removed,
                    old_value=previous[removed],
                    description="Configuration entry removed",
                    auto_remediable=True,
                    remediation_suggestion="Restore the missing configuration file",
                )
            )

        for shared in sorted(prev_keys & curr_keys):
            prev_entry = previous[shared]
            curr_entry = current[shared]
            if prev_entry["hash"] != curr_entry["hash"]:
                events.append(
                    DriftEvent(
                        drift_type=DriftType.MODIFIED,
                        severity=DriftSeverity.MEDIUM,
                        source="filesystem",
                        path=shared,
                        old_value=prev_entry,
                        new_value=curr_entry,
                        description="Configuration file contents changed",
                    )
                )
            elif prev_entry["mode"] != curr_entry["mode"]:
                events.append(
                    DriftEvent(
                        drift_type=DriftType.PERMISSIONS,
                        severity=DriftSeverity.LOW,
                        source="filesystem",
                        path=shared,
                        old_value=prev_entry["mode"],
                        new_value=curr_entry["mode"],
                        description="Configuration file permissions changed",
                        auto_remediable=True,
                        remediation_suggestion=(
                            "Reset file permissions to the previous value"
                        ),
                    )
                )

        if events:
            self.drift_events.extend(events)
            logger.info("Detected %d configuration drift events", len(events))
        else:
            logger.debug("No configuration drift detected between latest snapshots")

        return events

    def get_latest_snapshot(self) -> ConfigSnapshot | None:
        """Get the most recent configuration snapshot."""
        return self.snapshots[-1] if self.snapshots else None

    def get_drift_summary(self) -> dict[str, Any]:
        """Get a summary of drift detection results."""

        events = self.drift_events
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
            "latest_snapshot": (
                latest_snapshot.model_dump()
                if (latest_snapshot := self.get_latest_snapshot())
                else None
            ),
            "total_snapshots": len(self.snapshots),
        }

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _collect_state(self) -> dict[str, dict[str, Any]]:
        state: dict[str, dict[str, Any]] = {}

        target = self.config_dir
        if not target.exists():
            logger.debug(
                "Config directory %s does not exist; snapshot is empty", target
            )
            return state

        iterator = (
            [target]
            if target.is_file()
            else sorted(p for p in target.rglob("*") if p.is_file())
        )

        for path in iterator:
            try:
                content_hash = _hash_bytes(path.read_bytes())
                stat_result = path.stat()
            except (OSError, PermissionError) as exc:
                logger.debug("Skipping %s during snapshot: %s", path, exc)
                continue

            relative = path.name if target.is_file() else str(path.relative_to(target))
            state[relative] = {
                "hash": content_hash,
                "size": stat_result.st_size,
                "mode": stat_result.st_mode,
                "modified_at": stat_result.st_mtime,
            }

        return state


# Global drift detector singleton


class _DriftDetectorSingleton:
    """Singleton holder for the configuration drift detector."""

    _instance: ConfigDriftDetector | None = None

    @classmethod
    def initialize(
        cls, config_dir: Path | Any, snapshot_interval: int = 3600
    ) -> ConfigDriftDetector:
        """Create or replace the global drift detector instance."""

        actual_dir: Path
        if isinstance(config_dir, Path):
            actual_dir = config_dir
        elif hasattr(config_dir, "data_dir"):
            actual_dir = Path(str(config_dir.data_dir))
        else:
            actual_dir = Path(str(config_dir))

        cls._instance = ConfigDriftDetector(actual_dir.expanduser(), snapshot_interval)
        return cls._instance

    @classmethod
    def get(cls) -> ConfigDriftDetector | None:
        """Return the current drift detector instance."""

        return cls._instance


def initialize_drift_detector(
    config_dir: Path | Any, snapshot_interval: int = 3600
) -> ConfigDriftDetector:
    """Initialize the global drift detector.

    Args:
        config_dir: Path to config directory or drift config object
        snapshot_interval: Snapshot interval in seconds

    Returns:
        ConfigDriftDetector instance
    """
    return _DriftDetectorSingleton.initialize(config_dir, snapshot_interval)


def _hash_bytes(data: bytes) -> str:
    """Return a deterministic SHA-256 hash for the provided bytes."""

    return sha256(data).hexdigest()


def _hash_mapping(payload: Mapping[str, Any]) -> str:
    """Hash a mapping by serialising it in a stable form."""

    serialised = json.dumps(
        payload,
        sort_keys=True,
        separators=(",", ":"),
        default=str,
    ).encode("utf-8")
    return _hash_bytes(serialised)


def get_drift_detector() -> ConfigDriftDetector | None:
    """Get the global drift detector instance."""
    return _DriftDetectorSingleton.get()


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
        "latest_snapshot": (
            latest_snapshot.model_dump()
            if (latest_snapshot := detector.get_latest_snapshot())
            else None
        ),
    }
