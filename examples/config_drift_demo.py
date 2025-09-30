#!/usr/bin/env python3
"""Configuration drift detection example.

Demonstrates configuration drift detection capabilities:
1. Initialize the drift detection system
2. Take configuration snapshots
3. Detect drift between configurations
4. Generate alerts and remediation suggestions
5. Monitor configuration changes over time

Usage:
    python examples/config_drift_demo.py
"""

import asyncio
import json
import tempfile
from pathlib import Path
from typing import Any

import aiofiles

from src.config import (
    ConfigDriftDetector,
    DriftDetectionConfig,
    DriftSeverity,
    get_drift_summary,
    initialize_drift_detector,
    run_drift_detection,
)


def create_sample_configs():
    """Create sample configuration files for demonstration."""
    configs = {}

    # Original configuration
    configs["original"] = {
        "database_url": "postgresql://localhost:5432/ai_docs",
        "api_key": "sk-1234567890abcdef",
        "debug": False,
        "max_connections": 20,
        "cache_size": 1000,
        "environment": "development",
        "log_level": "INFO",
    }

    # Modified configuration with security change
    configs["security_drift"] = {
        "database_url": "postgresql://localhost:5432/ai_docs",
        "api_key": "sk-different123456789",  # Security change
        "debug": False,
        "max_connections": 20,
        "cache_size": 1000,
        "environment": "development",
        "log_level": "INFO",
    }

    # Modified configuration with environment drift
    configs["env_drift"] = {
        "database_url": "postgresql://prod-server:5432/ai_docs_prod",  # DB change
        "api_key": "sk-1234567890abcdef",
        "debug": False,
        "max_connections": 100,  # Performance change
        "cache_size": 5000,  # Performance change
        "environment": "production",  # Environment change
        "log_level": "WARNING",  # Log level change
    }

    # Configuration with schema violation
    configs["schema_drift"] = {
        "database_url": "postgresql://localhost:5432/ai_docs",
        "api_key": "sk-1234567890abcdef",
        "debug": False,
        "max_connections": 20,
        "cache_size": 1000,
        "environment": "development",
        "log_level": "INFO",
        "new_experimental_feature": True,  # Added field
        # "cache_size" removed would be detected as removal
    }

    return configs


def _setup_demo_environment() -> tuple[dict[str, Any], ConfigDriftDetector]:
    """Set up the demo environment and return configs and detector."""
    print("=" * 60)
    print("Configuration Drift Detection Demo")
    print("=" * 60)

    configs = create_sample_configs()

    drift_config = DriftDetectionConfig(
        enabled=True,
        snapshot_interval_minutes=1,
        comparison_interval_minutes=1,
        monitored_paths=["demo_config.json"],
        alert_on_severity=[
            DriftSeverity.MEDIUM,
            DriftSeverity.HIGH,
            DriftSeverity.CRITICAL,
        ],
        max_alerts_per_hour=10,
        snapshot_retention_days=1,
        integrate_with_task20_anomaly=False,
        use_performance_monitoring=False,
    )

    detector = initialize_drift_detector(drift_config)
    paths_count = len(drift_config.monitored_paths)
    print(f"Initialized drift detector with {paths_count} monitored paths")

    return configs, detector


def _create_temp_config_files(configs: dict[str, Any]) -> list[Path]:
    """Create temporary configuration files for demo."""
    temp_files = []

    for config_name, config_data in configs.items():
        with tempfile.NamedTemporaryFile(
            mode="w", suffix=".json", delete=False, prefix=f"{config_name}_"
        ) as temp_file:
            json.dump(config_data, temp_file, indent=2)
            temp_file.flush()
            temp_files.append(Path(temp_file.name))
            print(f"📝 Created sample config: {config_name} -> {temp_file.name}")

    return temp_files


def _take_initial_snapshot(detector: ConfigDriftDetector, config_file: Path) -> None:
    """Take and display initial configuration snapshot."""
    print("\n" + "=" * 60)
    print("Taking Initial Configuration Snapshot")
    print("=" * 60)

    snapshot = detector.take_snapshot()
    print(f"Snapshot taken for {config_file.name}")
    print(f"   Config hash: {snapshot.config_hash[:16]}...")
    print(f"   Timestamp: {snapshot.timestamp.strftime('%H:%M:%S')}")
    print(f"   Keys: {list(snapshot.config_data.keys())}")


async def _simulate_drift_scenarios(
    detector: ConfigDriftDetector, configs: dict[str, Any], original_file: Path
) -> None:
    """Simulate different drift scenarios and detect changes."""
    print("\n" + "=" * 60)
    print("Simulating Configuration Changes")
    print("=" * 60)

    drift_scenarios = [
        (configs["security_drift"], "Security API Key Change"),
        (configs["env_drift"], "Environment Drift"),
        (configs["schema_drift"], "Schema Violation"),
    ]

    for config_data, scenario_name in drift_scenarios:
        print(f"\n{scenario_name}")
        print("-" * 40)

        async with aiofiles.open(original_file, "w", encoding="utf-8") as f:
            await f.write(json.dumps(config_data, indent=2))

        detector.take_snapshot()
        events = detector.detect_drift()

        _display_drift_events(detector, events)


def _display_drift_events(detector: ConfigDriftDetector, events: list) -> None:
    """Display detected drift events with formatting."""
    if events:
        print(f"Detected {len(events)} drift events:")
        for event in events:
            severity_text = event.severity.value.upper()
            print(f"   {severity_text}: {event.description}")
            print(f"      Type: {event.drift_type.value}")
            print(f"      Auto-remediable: {'Yes' if event.auto_remediable else 'No'}")

            if event.remediation_suggestion:
                print(f"      Suggestion: {event.remediation_suggestion}")
    else:
        print("No drift detected")


def _display_drift_summary(detector: ConfigDriftDetector) -> None:
    """Display comprehensive drift detection summary."""
    print("\n" + "=" * 60)
    print("Drift Detection Summary")
    print("=" * 60)

    summary = detector.get_drift_summary()

    print(f"Total Events: {summary['total_events']}")
    print(f"Total Snapshots: {summary['total_snapshots']}")

    print("\nSeverity Breakdown:")
    for severity, count in summary["by_severity"].items():
        if count > 0:
            print(f"   {severity.title()}: {count}")

    print("\nType Breakdown:")
    for drift_type, count in summary["by_type"].items():
        if count > 0:
            print(f"   {drift_type.title()}: {count}")


def _demonstrate_global_functions() -> None:
    """Demonstrate global drift detection functions."""
    print("\n" + "=" * 60)
    print("Global Detection Functions Demo")
    print("=" * 60)

    print("Running global drift detection...")
    global_events = run_drift_detection()
    print(f"   Found {len(global_events)} events via global function")

    global_summary = get_drift_summary()
    print(f"   Global summary: {global_summary['total_events']} total events")


def _cleanup_and_summary(temp_files: list[Path]) -> None:
    """Clean up temporary files and display demo summary."""
    print(f"\nCleaning up {len(temp_files)} temporary files...")
    for temp_file in temp_files:
        temp_file.unlink(missing_ok=True)

    print("\n" + "=" * 60)
    print("Configuration Drift Detection Demo Complete")
    print("=" * 60)
    print("\nKey Features Demonstrated:")
    print("   • Configuration snapshot management")
    print("   • Drift detection across snapshots")
    print("   • Severity-based classification")
    print("   • Auto-remediation suggestions")
    print("   • Comprehensive drift reporting")


async def demo_basic_drift_detection():
    """Demonstrate basic configuration drift detection."""
    configs, detector = _setup_demo_environment()
    temp_files = []

    try:
        temp_files = _create_temp_config_files(configs)
        original_file = temp_files[0]

        _take_initial_snapshot(detector, original_file)
        await asyncio.sleep(0.1)  # Wait for timestamp difference

        await _simulate_drift_scenarios(detector, configs, original_file)
        _display_drift_summary(detector)
        _demonstrate_global_functions()

    finally:
        _cleanup_and_summary(temp_files)


if __name__ == "__main__":
    # Run the demo
    asyncio.run(demo_basic_drift_detection())
