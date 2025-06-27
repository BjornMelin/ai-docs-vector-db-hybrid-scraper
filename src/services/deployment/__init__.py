"""Enterprise Deployment Services.

This module provides advanced deployment capabilities for enterprise use:
- A/B Testing with traffic splitting and statistical analysis
- Blue-Green Deployments for zero-downtime releases
- Canary Deployments with progressive traffic rollout
- Feature Flag integration for tier-based access control

These services are designed for portfolio showcase and enterprise environments,
controlled by feature flags to maintain simplicity for personal use.
"""

from .ab_testing import ABTestingManager, ABTestResult, ABTestStatus
from .blue_green import BlueGreenDeployment, BlueGreenStatus
from .canary import CanaryDeployment, CanaryMetrics, CanaryStatus
from .feature_flags import DeploymentTier, FeatureFlagManager
from .models import (
    DeploymentEnvironment,
    DeploymentHealth,
    DeploymentMetrics,
    DeploymentStatus,
)


__all__ = [
    "ABTestResult",
    "ABTestStatus",
    # A/B Testing
    "ABTestingManager",
    # Blue-Green Deployment
    "BlueGreenDeployment",
    "BlueGreenStatus",
    # Canary Deployment
    "CanaryDeployment",
    "CanaryMetrics",
    "CanaryStatus",
    # Models
    "DeploymentEnvironment",
    "DeploymentHealth",
    "DeploymentMetrics",
    "DeploymentStatus",
    # Feature Flags
    "DeploymentTier",
    "FeatureFlagManager",
]
