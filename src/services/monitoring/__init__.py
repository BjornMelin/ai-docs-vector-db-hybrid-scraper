"""Monitoring and observability services for ML/vector search application.

This package provides comprehensive monitoring infrastructure including:
- Prometheus metrics collection
- Health check systems
- Performance monitoring decorators
- Grafana dashboard configurations
- FastMCP integration
"""

from src.services.health.manager import HealthCheckManager
from src.services.monitoring.initialization import (
    initialize_monitoring_system,
    setup_fastmcp_monitoring,
)
from src.services.monitoring.metrics import MetricsRegistry
from src.services.monitoring.middleware import PrometheusMiddleware
from src.services.monitoring.telemetry_repository import get_telemetry_repository


__all__ = [
    "HealthCheckManager",
    "MetricsRegistry",
    "get_telemetry_repository",
    "PrometheusMiddleware",
    "initialize_monitoring_system",
    "setup_fastmcp_monitoring",
]
