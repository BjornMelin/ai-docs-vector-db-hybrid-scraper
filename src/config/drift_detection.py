"""Configuration drift detection module.

This module provides backward compatibility for the drift detection system
that is now unified in the main config system.
"""

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

# Import DriftDetectionConfig from settings
from .settings import DriftDetectionConfig


# Export everything for backward compatibility
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
