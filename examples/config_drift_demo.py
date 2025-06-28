#!/usr/bin/env python3
"""
Configuration Drift Detection Demo

This script demonstrates the configuration drift detection capabilities
implemented in the AI Documentation Vector DB system. It shows how to:
1. Initialize the drift detection system
2. Take configuration snapshots
3. Detect drift between configurations
4. Generate alerts and remediation suggestions
5. Integrate with existing Task 20 monitoring infrastructure

Usage:
    python examples/config_drift_demo.py
"""

import asyncio
import json
import tempfile
from pathlib import Path

import aiofiles

from src.config.drift_detection import (
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


def _setup_demo_environment() -> tuple[dict[str, Any], DriftDetector]:
    """Set up the demo environment and return configs and detector."""
    print("=" * 60)
    print("🔍 Configuration Drift Detection Demo")
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
    print(
        f"✅ Initialized drift detector with {len(drift_config.monitored_paths)} monitored paths"
    )

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


def _take_initial_snapshot(detector: DriftDetector, config_file: Path) -> None:
    """Take and display initial configuration snapshot."""
    print("\n" + "=" * 60)
    print("🔄 Taking Initial Configuration Snapshots")
    print("=" * 60)

    snapshot = detector.take_snapshot(str(config_file))
    print(f"📸 Snapshot taken for {config_file.name}")
    print(f"   📊 Config hash: {snapshot.config_hash[:16]}...")
    print(f"   🕐 Timestamp: {snapshot.timestamp.strftime('%H:%M:%S')}")
    print(f"   📂 Keys: {list(snapshot.config_data.keys())}")


async def _simulate_drift_scenarios(
    detector: DriftDetector, configs: dict[str, Any], original_file: Path
) -> None:
    """Simulate different drift scenarios and detect changes."""
    print("\n" + "=" * 60)
    print("🚨 Simulating Configuration Changes")
    print("=" * 60)

    drift_scenarios = [
        (configs["security_drift"], "Security API Key Change", "🔐"),
        (configs["env_drift"], "Environment Drift", "🌍"),
        (configs["schema_drift"], "Schema Violation", "⚠️"),
    ]

    for config_data, scenario_name, emoji in drift_scenarios:
        print(f"\n{emoji} {scenario_name}")
        print("-" * 40)

        async with aiofiles.open(original_file, "w") as f:
            await f.write(json.dumps(config_data, indent=2))

        detector.take_snapshot(str(original_file))
        events = detector.compare_snapshots(str(original_file))

        _display_drift_events(detector, events)


def _display_drift_events(detector: DriftDetector, events: list) -> None:
    """Display detected drift events with formatting."""
    if events:
        print(f"🔍 Detected {len(events)} drift events:")
        for event in events:
            severity_emoji = {
                "low": "💚",
                "medium": "💛",
                "high": "🧡",
                "critical": "🔴",
            }.get(event.severity.value, "❔")

            print(
                f"   {severity_emoji} {event.severity.value.upper()}: {event.description}"
            )
            print(f"      📍 Type: {event.drift_type.value}")
            print(
                f"      🔧 Auto-remediable: {'Yes' if event.auto_remediable else 'No'}"
            )

            if event.remediation_suggestion:
                print(f"      💡 Suggestion: {event.remediation_suggestion}")

            should_alert = detector.should_alert(event)
            alert_msg = (
                "🚨 Alert triggered!"
                if should_alert
                else "🔇 Alert suppressed (severity/rate limit)"
            )
            print(f"      {alert_msg}")
    else:
        print("✅ No drift detected")


def _display_drift_summary(detector: DriftDetector) -> None:
    """Display comprehensive drift detection summary."""
    print("\n" + "=" * 60)
    print("📊 Drift Detection Summary")
    print("=" * 60)

    summary = detector.get_drift_summary()

    print(
        f"🟢 Detection Status: {'Enabled' if summary['detection_enabled'] else 'Disabled'}"
    )
    print(f"📁 Monitored Sources: {summary['monitored_sources']}")
    print(f"📸 Snapshots Stored: {summary['snapshots_stored']}")
    print(f"⚡ Recent Events (24h): {summary['recent_events_24h']}")
    print(f"🔧 Auto-remediable Events: {summary['auto_remediable_events']}")

    print("\n📈 Severity Breakdown:")
    for severity, count in summary["severity_breakdown"].items():
        if count > 0:
            severity_emoji = {
                "low": "💚",
                "medium": "💛",
                "high": "🧡",
                "critical": "🔴",
            }.get(severity, "❔")
            print(f"   {severity_emoji} {severity.title()}: {count}")

    if summary.get("recent_alerts"):
        print("\n🚨 Recent Alerts:")
        for alert in summary["recent_alerts"][-3:]:
            print(f"   • {alert}")
    else:
        print("\n🔇 No recent alerts")


def _demonstrate_global_functions() -> None:
    """Demonstrate global drift detection functions."""
    print("\n" + "=" * 60)
    print("🔄 Global Detection Functions Demo")
    print("=" * 60)

    print("🌍 Running global drift detection...")
    global_events = run_drift_detection()
    print(f"   📊 Found {len(global_events)} events via global function")

    global_summary = get_drift_summary()
    print(f"   📈 Global summary: {global_summary['recent_events_24h']} events in 24h")


def _cleanup_and_summary(temp_files: list[Path]) -> None:
    """Clean up temporary files and display demo summary."""
    print(f"\n🧹 Cleaning up {len(temp_files)} temporary files...")
    for temp_file in temp_files:
        temp_file.unlink(missing_ok=True)

    print("\n" + "=" * 60)
    print("✅ Configuration Drift Detection Demo Complete!")
    print("=" * 60)
    print("\n💡 Key Features Demonstrated:")
    print("   • Configuration snapshot management")
    print("   • Multi-type drift detection (security, environment, schema)")
    print("   • Severity-based classification")
    print("   • Auto-remediation suggestions")
    print("   • Alert rate limiting")
    print("   • Integration with existing Task 20 infrastructure")
    print("   • Comprehensive drift reporting")
    print("\n🔗 Integration Points:")
    print("   • Task 20 Performance Monitoring")
    print("   • Task 20 Anomaly Detection")
    print("   • Observability Metrics Bridge")
    print("   • RESTful API endpoints")
    print("   • Background task processing")


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
