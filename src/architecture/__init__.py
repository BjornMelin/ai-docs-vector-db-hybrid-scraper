"""Architecture module for dual-mode application management.

This module implements the Enterprise Paradox solution by providing:
- Simple mode (25K lines) for daily development use
- Enterprise mode (70K lines) for portfolio demonstrations
- Feature flag system for gradual migration
- Service factory pattern for mode-specific implementations
"""

from .features import FeatureFlag, conditional_feature, enterprise_only
from .modes import ApplicationMode, ModeConfig, get_mode_config


__all__ = [
    "ApplicationMode",
    "FeatureFlag",
    "ModeConfig",
    "conditional_feature",
    "enterprise_only",
    "get_mode_config",
]
