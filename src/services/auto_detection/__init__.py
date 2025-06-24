"""Auto-detection services for environment profiling and service discovery.

This package provides:
- Environment detection (Docker, Kubernetes, cloud providers)
- Service discovery with connection pooling optimization
- Health checks and monitoring integration
- Circuit breaker resilience patterns

Key Components:
- EnvironmentDetector: Detects runtime environment
- ServiceDiscovery: Discovers available services with health checks
- ConnectionPoolManager: Manages optimized connection pools
- HealthChecker: Validates service availability and performance
"""

from src.config.auto_detect import (
    AutoDetectedServices,
    AutoDetectionConfig,
    DetectedEnvironment,
    DetectedService,
    EnvironmentDetector,
)

from .connection_pools import ConnectionPoolManager, PoolHealthMetrics
from .health_checks import HealthChecker, HealthCheckResult, HealthSummary
from .service_discovery import ServiceDiscovery, ServiceDiscoveryResult


__all__ = [
    "AutoDetectedServices",
    "AutoDetectionConfig",
    "ConnectionPoolManager",
    "DetectedEnvironment",
    "DetectedService",
    "EnvironmentDetector",
    "HealthCheckResult",
    "HealthChecker",
    "HealthSummary",
    "PoolHealthMetrics",
    "ServiceDiscovery",
    "ServiceDiscoveryResult",
]

# Version info
__version__ = "1.0.0"
__author__ = "AI Docs Vector DB Team"
__description__ = "Service auto-detection and environment profiling"
