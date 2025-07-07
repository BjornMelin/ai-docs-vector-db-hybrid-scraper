"""Modern configuration module.

This module provides backward compatibility for the modern config system
that is now unified in the main config system.
"""

# Import everything from the unified settings system
from .settings import (
    Config,
    Settings,
    create_enterprise_config,
    create_settings_from_env,
    create_simple_config,
    get_config,
    get_settings,
    reset_config,
    reset_settings,
    set_config,
    set_settings,
)


# Export everything for backward compatibility
__all__ = [
    "Config",
    "Settings",
    "create_enterprise_config",
    "create_settings_from_env",
    "create_simple_config",
    "get_config",
    "get_settings",
    "reset_config",
    "reset_settings",
    "set_config",
    "set_settings",
]
