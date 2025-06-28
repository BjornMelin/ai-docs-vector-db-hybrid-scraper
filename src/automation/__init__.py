"""Zero-maintenance automation system.

This module provides automated systems for:
- Self-healing infrastructure
- Configuration management
- Dependency updates
- Performance optimization
- Error recovery
"""

from .config_automation import AutoConfigManager, ConfigDriftHealer
from .dependency_automation import DependencyUpdateAgent, SecurityUpdateManager
from .infrastructure_automation import AutoScalingManager, SelfHealingManager
from .monitoring_automation import AdaptiveAlertManager, PredictiveMaintenanceSystem
from .test_automation import AutomatedQualityGate, SelfMaintainingTestSuite


__all__ = [
    "AdaptiveAlertManager",
    "AutoConfigManager",
    "AutoScalingManager",
    "AutomatedQualityGate",
    "ConfigDriftHealer",
    "DependencyUpdateAgent",
    "PredictiveMaintenanceSystem",
    "SecurityUpdateManager",
    "SelfHealingManager",
    "SelfMaintainingTestSuite",
]
