"""Cross-platform filesystem helpers built on ``platformdirs``.

The helpers return resolved :class:`pathlib.Path` instances so callers can use
native filesystem APIs without additional coercion. No legacy fallbacks are
retained—platform-specific logic lives entirely inside ``platformdirs``.
"""

from __future__ import annotations

from pathlib import Path

from platformdirs import user_cache_dir, user_config_dir, user_data_dir, user_log_dir


__all__ = [
    "DEFAULT_APP_NAME",
    "get_cache_dir",
    "get_config_dir",
    "get_data_dir",
    "get_log_dir",
    "normalize_path",
]

DEFAULT_APP_NAME = "ai-docs-scraper"


def get_cache_dir(app_name: str = DEFAULT_APP_NAME) -> Path:
    """Return the platform-appropriate cache directory."""
    return Path(user_cache_dir(app_name, appauthor=None)).expanduser().resolve()


def get_config_dir(app_name: str = DEFAULT_APP_NAME) -> Path:
    """Return the platform-appropriate configuration directory."""
    return Path(user_config_dir(app_name, appauthor=None)).expanduser().resolve()


def get_data_dir(app_name: str = DEFAULT_APP_NAME) -> Path:
    """Return the platform-appropriate data directory."""
    return Path(user_data_dir(app_name, appauthor=None)).expanduser().resolve()


def get_log_dir(app_name: str = DEFAULT_APP_NAME) -> Path:
    """Return the platform-appropriate log directory."""
    return Path(user_log_dir(app_name, appauthor=None)).expanduser().resolve()


def normalize_path(path: Path | str) -> Path:
    """Return a resolved absolute path for ``path``."""
    return Path(path).expanduser().resolve()
