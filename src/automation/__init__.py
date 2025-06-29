"""Zero-maintenance automation system.

This module provides automated systems for:
- Self-healing infrastructure
- Configuration management
- Infrastructure automation
"""

from .config_automation import AutoConfigManager
from .infrastructure_automation import SelfHealingDatabaseManager
from .self_healing.auto_remediation_engine import AutoRemediationEngine

# Self-healing infrastructure components
from .self_healing.autonomous_health_monitor import AutonomousHealthMonitor
from .self_healing.intelligent_chaos_orchestrator import IntelligentChaosOrchestrator
from .self_healing.predictive_maintenance import PredictiveMaintenanceScheduler


__all__ = [
    "AutoConfigManager",
    "AutoRemediationEngine",
    "AutonomousHealthMonitor",
    "IntelligentChaosOrchestrator",
    "PredictiveMaintenanceScheduler",
    "SelfHealingDatabaseManager",
]
