"""Thin proxy exposing configuration drift detection utilities."""

# Import everything from the unified drift system
from .drift import (
    ConfigDriftDetector,
    ConfigSnapshot,
    DriftEvent,
    DriftSeverity,
    DriftType,
    get_drift_detector,
    get_drift_summary,
    initialize_drift_detector,
    run_drift_detection,
)

# Import DriftDetectionConfig from models
from .models import DriftDetectionConfig


# Public re-exports to maintain a stable import surface
__all__ = [
    "ConfigDriftDetector",
    "ConfigSnapshot",
    "DriftDetectionConfig",
    "DriftEvent",
    "DriftSeverity",
    "DriftType",
    "get_drift_detector",
    "get_drift_summary",
    "initialize_drift_detector",
    "run_drift_detection",
]
