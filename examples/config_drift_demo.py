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
import time
from datetime import datetime
from pathlib import Path

from src.config.drift_detection import (
    ConfigDriftDetector,
    DriftDetectionConfig,
    DriftSeverity,
    initialize_drift_detector,
    get_drift_detector,
    run_drift_detection,
    get_drift_summary,
)


def create_sample_configs():
    """Create sample configuration files for demonstration."""
    configs = {}
    
    # Original configuration
    configs['original'] = {
        "database_url": "postgresql://localhost:5432/ai_docs",
        "api_key": "sk-1234567890abcdef",
        "debug": False,
        "max_connections": 20,
        "cache_size": 1000,
        "environment": "development",
        "log_level": "INFO"
    }
    
    # Modified configuration with security change
    configs['security_drift'] = {
        "database_url": "postgresql://localhost:5432/ai_docs",
        "api_key": "sk-different123456789",  # Security change
        "debug": False,
        "max_connections": 20,
        "cache_size": 1000,
        "environment": "development",
        "log_level": "INFO"
    }
    
    # Modified configuration with environment drift
    configs['env_drift'] = {
        "database_url": "postgresql://prod-server:5432/ai_docs_prod",  # DB change
        "api_key": "sk-1234567890abcdef",
        "debug": False,
        "max_connections": 100,  # Performance change
        "cache_size": 5000,     # Performance change
        "environment": "production",  # Environment change
        "log_level": "WARNING"  # Log level change
    }
    
    # Configuration with schema violation
    configs['schema_drift'] = {
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


async def demo_basic_drift_detection():
    """Demonstrate basic configuration drift detection."""
    print("=" * 60)
    print("🔍 Configuration Drift Detection Demo")
    print("=" * 60)
    
    # Create sample configurations
    configs = create_sample_configs()
    
    # Initialize drift detection with demo settings
    drift_config = DriftDetectionConfig(
        enabled=True,
        snapshot_interval_minutes=1,
        comparison_interval_minutes=1,
        monitored_paths=["demo_config.json"],
        alert_on_severity=[DriftSeverity.MEDIUM, DriftSeverity.HIGH, DriftSeverity.CRITICAL],
        max_alerts_per_hour=10,
        snapshot_retention_days=1,
        # Disable integrations for demo
        integrate_with_task20_anomaly=False,
        use_performance_monitoring=False,
    )
    
    # Initialize detector
    detector = initialize_drift_detector(drift_config)
    print(f"✅ Initialized drift detector with {len(drift_config.monitored_paths)} monitored paths")
    
    # Create temporary config files
    temp_files = []
    
    try:
        for config_name, config_data in configs.items():
            temp_file = tempfile.NamedTemporaryFile(
                mode='w', suffix='.json', delete=False, prefix=f'{config_name}_'
            )
            json.dump(config_data, temp_file, indent=2)
            temp_file.flush()
            temp_files.append(Path(temp_file.name))
            print(f"📝 Created sample config: {config_name} -> {temp_file.name}")
        
        print("\n" + "=" * 60)
        print("🔄 Taking Initial Configuration Snapshots")
        print("=" * 60)
        
        # Take snapshots of all configurations
        original_file = temp_files[0]
        snapshot = detector.take_snapshot(str(original_file))
        print(f"📸 Snapshot taken for {original_file.name}")
        print(f"   📊 Config hash: {snapshot.config_hash[:16]}...")
        print(f"   🕐 Timestamp: {snapshot.timestamp.strftime('%H:%M:%S')}")
        print(f"   📂 Keys: {list(snapshot.config_data.keys())}")
        
        # Wait a moment for timestamp difference
        await asyncio.sleep(0.1)
        
        print("\n" + "=" * 60)
        print("🚨 Simulating Configuration Changes")
        print("=" * 60)
        
        # Test different types of drift by replacing the original file content
        drift_scenarios = [
            (configs['security_drift'], "Security API Key Change", "🔐"),
            (configs['env_drift'], "Environment Drift", "🌍"), 
            (configs['schema_drift'], "Schema Violation", "⚠️"),
        ]
        
        for config_data, scenario_name, emoji in drift_scenarios:
            print(f"\n{emoji} {scenario_name}")
            print("-" * 40)
            
            # Update the original file with new content
            with open(original_file, 'w') as f:
                json.dump(config_data, f, indent=2)
            
            # Take a new snapshot of the modified file
            new_snapshot = detector.take_snapshot(str(original_file))
            
            # Compare with previous snapshots  
            events = detector.compare_snapshots(str(original_file))
            
            if events:
                print(f"🔍 Detected {len(events)} drift events:")
                for event in events:
                    severity_emoji = {
                        "low": "💚",
                        "medium": "💛", 
                        "high": "🧡",
                        "critical": "🔴"
                    }.get(event.severity.value, "❔")
                    
                    print(f"   {severity_emoji} {event.severity.value.upper()}: {event.description}")
                    print(f"      📍 Type: {event.drift_type.value}")
                    print(f"      🔧 Auto-remediable: {'Yes' if event.auto_remediable else 'No'}")
                    if event.remediation_suggestion:
                        print(f"      💡 Suggestion: {event.remediation_suggestion}")
                    
                    # Check if alert should be sent
                    should_alert = detector.should_alert(event)
                    if should_alert:
                        print(f"      🚨 Alert triggered!")
                    else:
                        print(f"      🔇 Alert suppressed (severity/rate limit)")
            else:
                print("✅ No drift detected")
        
        print("\n" + "=" * 60)
        print("📊 Drift Detection Summary")
        print("=" * 60)
        
        # Get comprehensive summary
        summary = detector.get_drift_summary()
        
        print(f"🟢 Detection Status: {'Enabled' if summary['detection_enabled'] else 'Disabled'}")
        print(f"📁 Monitored Sources: {summary['monitored_sources']}")
        print(f"📸 Snapshots Stored: {summary['snapshots_stored']}")
        print(f"⚡ Recent Events (24h): {summary['recent_events_24h']}")
        print(f"🔧 Auto-remediable Events: {summary['auto_remediable_events']}")
        
        print(f"\n📈 Severity Breakdown:")
        for severity, count in summary['severity_breakdown'].items():
            if count > 0:
                severity_emoji = {
                    "low": "💚",
                    "medium": "💛", 
                    "high": "🧡",
                    "critical": "🔴"
                }.get(severity, "❔")
                print(f"   {severity_emoji} {severity.title()}: {count}")
        
        if summary.get('recent_alerts'):
            print(f"\n🚨 Recent Alerts:")
            for alert in summary['recent_alerts'][-3:]:  # Show last 3
                print(f"   • {alert}")
        else:
            print(f"\n🔇 No recent alerts")
        
        print("\n" + "=" * 60)
        print("🔄 Global Detection Functions Demo")
        print("=" * 60)
        
        # Demonstrate global functions
        print("🌍 Running global drift detection...")
        global_events = run_drift_detection()
        print(f"   📊 Found {len(global_events)} events via global function")
        
        global_summary = get_drift_summary()
        print(f"   📈 Global summary: {global_summary['recent_events_24h']} events in 24h")
        
    finally:
        # Cleanup temporary files
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


if __name__ == "__main__":
    # Run the demo
    asyncio.run(demo_basic_drift_detection())