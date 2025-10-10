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


__all__ = [
    "HealthCheckManager",
    "initialize_monitoring_system",
    "setup_fastmcp_monitoring",
]
