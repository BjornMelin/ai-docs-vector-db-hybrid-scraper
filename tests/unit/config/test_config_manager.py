"""Unit tests for the configuration manager backup and failure handling."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from src.config import Config, ConfigLoadError, ConfigManager


def _write_config(path: Path, api_key: str) -> None:
    """Persist a minimal configuration payload to ``path``."""

    payload = {"openai": {"api_key": api_key}, "qdrant": {"url": "http://localhost"}}
    path.write_text(json.dumps(payload), encoding="utf-8")


def test_restore_from_backup(tmp_path: Path) -> None:
    """ConfigManager.restore_from_backup() should reinstate an older snapshot."""

    config_file = tmp_path / "config.json"
    _write_config(config_file, api_key="sk-initial")

    manager = ConfigManager(
        config_class=Config,
        config_file=config_file,
        enable_file_watching=False,
        backup_limit=5,
    )
    assert manager.get_config().openai.api_key == "sk-initial"

    _write_config(config_file, api_key="sk-second")
    assert manager.reload_config() is True

    _write_config(config_file, api_key="sk-third")
    assert manager.reload_config() is True

    assert manager.restore_from_backup(1) is True
    restored_config = manager.get_config()
    assert restored_config.openai.api_key == "sk-second"


def test_reload_failure_records_recent_failure(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Failed reloads should populate the recent failure log."""

    config_file = tmp_path / "invalid.json"
    _write_config(config_file, api_key="sk-before-failure")

    manager = ConfigManager(
        config_class=Config,
        config_file=config_file,
        enable_file_watching=False,
    )

    def raise_load_error(_path: Path) -> dict[str, str]:
        raise ConfigLoadError("simulated failure during load")

    monkeypatch.setattr(manager.safe_loader, "load_from_file", raise_load_error)

    assert manager.reload_config() is False
    recent_failures = manager.get_status()["recent_failures"]
    assert recent_failures, "Expected the failure list to contain an entry"
    failure_entry = recent_failures[0]
    assert failure_entry["operation"] == "reload_config"
    assert "simulated failure" in failure_entry["error"]
