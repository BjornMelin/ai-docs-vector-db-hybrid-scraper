"""Factories for configuration objects used across unit tests."""

from __future__ import annotations

from pathlib import Path
from typing import Any

from src.config.settings import ApplicationMode, Environment, Settings


def make_test_settings(tmp_path: Path, **overrides: Any) -> Settings:
    """Create a `Settings` instance isolated to a temporary directory.

    The helper ensures that configuration-dependent tests never mutate the
    repository tree while still exercising the real configuration model. The
    default mode targets simple deployments, but callers can override any
    field to exercise enterprise behaviour.
    """

    base_dirs = {
        "data_dir": overrides.pop("data_dir", tmp_path / "data"),
        "cache_dir": overrides.pop("cache_dir", tmp_path / "cache"),
        "logs_dir": overrides.pop("logs_dir", tmp_path / "logs"),
    }

    settings_kwargs: dict[str, Any] = {
        "mode": overrides.pop("mode", ApplicationMode.SIMPLE),
        "environment": overrides.pop("environment", Environment.TESTING),
        **base_dirs,
        **overrides,
    }

    return Settings(**settings_kwargs)
