"""Configuration Observability Automation System.

A comprehensive enterprise-level automation system for configuration management
that provides intelligent monitoring, validation, and optimization capabilities.

This module demonstrates sophisticated automation patterns:
- Real-time configuration drift detection and remediation
- Multi-environment configuration consistency validation
- Automated performance optimization based on metrics
- Intelligent configuration recommendations and auto-tuning
- Comprehensive configuration health monitoring
- Zero-downtime configuration updates with rollback capability

Key Components:
- automation.py: Core automation engine with drift detection and remediation
- cli.py: Command-line interface for system management
- dashboard.py: Real-time Streamlit dashboard for monitoring
- api.py: FastAPI integration with WebSocket support

Usage Examples:

    # Start automation system programmatically
    from src.config.observability import start_automation_system

    automation = await start_automation_system(
        enable_auto_remediation=True,
        enable_performance_optimization=True,
    )

    # Check for configuration drift
    drifts = await automation.detect_configuration_drift()

    # Validate configuration health
    validation_results = await automation.validate_configuration_health()

    # Generate optimization recommendations
    recommendations = await automation.generate_optimization_recommendations()

    # Command-line usage
    python -m src.config.observability.cli start --auto-remediation
    python -m src.config.observability.cli drift-check --auto-fix
    python -m src.config.observability.cli validate
    python -m src.config.observability.cli optimize --apply

    # Dashboard usage
    streamlit run src/config/observability/dashboard.py

Features:
- Environment-aware configuration monitoring
- Automated drift detection and correction
- Performance-based configuration optimization
- Configuration compliance validation
- Real-time alerting and remediation
- Configuration version control and rollback
- Multi-tier deployment validation
- WebSocket-based real-time updates
- Interactive CLI and web dashboard
- FastAPI integration for enterprise systems

Architecture:
- Event-driven automation with file system monitoring
- Configurable automation policies and thresholds
- Pluggable validation and optimization rules
- Multi-environment baseline management
- Performance metrics integration
- Comprehensive audit logging
"""

from .automation import (
    ConfigDrift,
    ConfigDriftSeverity,
    ConfigObservabilityAutomation,
    ConfigValidationResult,
    ConfigValidationStatus,
    OptimizationRecommendation,
    PerformanceMetric,
    get_automation_system,
    start_automation_system,
    stop_automation_system,
)


__all__ = [
    "ConfigDrift",
    "ConfigDriftSeverity",
    # Core automation classes
    "ConfigObservabilityAutomation",
    "ConfigValidationResult",
    "ConfigValidationStatus",
    "OptimizationRecommendation",
    "PerformanceMetric",
    # System management functions
    "get_automation_system",
    "start_automation_system",
    "stop_automation_system",
]

__version__ = "1.0.0"
__author__ = "AI Documentation System"
__description__ = "Enterprise Configuration Observability Automation"
