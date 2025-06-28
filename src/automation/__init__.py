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
from .infrastructure_automation import SelfHealingManager, AutoScalingManager
from .monitoring_automation import AdaptiveAlertManager, PredictiveMaintenanceSystem
from .test_automation import SelfMaintainingTestSuite, AutomatedQualityGate

__all__ = [
    "AutoConfigManager",
    "ConfigDriftHealer", 
    "DependencyUpdateAgent",
    "SecurityUpdateManager",
    "SelfHealingManager",
    "AutoScalingManager",
    "AdaptiveAlertManager",
    "PredictiveMaintenanceSystem",
    "SelfMaintainingTestSuite",
    "AutomatedQualityGate",
]