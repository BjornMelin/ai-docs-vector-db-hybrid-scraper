"""Helper utilities for configuration management."""

import os
from pathlib import Path
from typing import Any, Optional


def get_env_bool(key: str, default: bool = False) -> bool:
    """Get boolean value from environment variable.

    Args:
        key: Environment variable name
        default: Default value if not set

    Returns:
        Boolean value
    """
    value = os.getenv(key, "").lower()
    if not value:
        return default
    return value in ("true", "1", "yes", "on")


def get_env_list(key: str, separator: str = ",") -> list[str]:
    """Get list from comma-separated environment variable.

    Args:
        key: Environment variable name
        separator: String separator (default: comma)

    Returns:
        List of strings
    """
    value = os.getenv(key, "")
    if not value:
        return []
    return [item.strip() for item in value.split(separator) if item.strip()]


def mask_secret(value: str, visible_chars: int = 4) -> str:
    """Mask a secret value for safe display.

    Args:
        value: Secret value to mask
        visible_chars: Number of characters to show at start

    Returns:
        Masked string
    """
    if not value or len(value) <= visible_chars:
        return "***"
    return f"{value[:visible_chars]}{'*' * (len(value) - visible_chars)}"


def ensure_path_exists(path: Path) -> Path:
    """Ensure a directory path exists.

    Args:
        path: Path to create if needed

    Returns:
        The path object
    """
    path.mkdir(parents=True, exist_ok=True)
    return path


def merge_config_dicts(
    base: dict[str, Any], override: dict[str, Any]
) -> dict[str, Any]:
    """Deep merge configuration dictionaries.

    Args:
        base: Base configuration
        override: Override values

    Returns:
        Merged configuration
    """
    result = base.copy()

    for key, value in override.items():
        if key in result and isinstance(result[key], dict) and isinstance(value, dict):
            result[key] = merge_config_dicts(result[key], value)
        else:
            result[key] = value

    return result


__all__ = [
    "ensure_path_exists",
    "get_env_bool",
    "get_env_list",
    "mask_secret",
    "merge_config_dicts",
]
