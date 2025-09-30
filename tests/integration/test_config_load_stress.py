"""Stress-oriented integration tests for the configuration manager."""

from __future__ import annotations

import asyncio
import json
from collections.abc import Iterator
from pathlib import Path

import pytest

from src.config import ConfigManager, ConfigReloader
from src.config.reloader import ReloadTrigger


@pytest.fixture
def config_path(tmp_path: Path) -> Path:
    path = tmp_path / "settings.json"
    path.write_text(
        json.dumps({"openai": {"api_key": "sk-initial"}}),
        encoding="utf-8",
    )
    return path


@pytest.fixture
def manager(config_path: Path) -> Iterator[ConfigManager]:
    manager = ConfigManager(
        config_file=config_path,
        enable_file_watching=False,
        backup_limit=3,
    )
    yield manager


def test_restore_from_backup(manager: ConfigManager, config_path: Path) -> None:
    """Reloading multiple times should persist backup history and allow restoration."""

    for idx in range(4):
        payload = {"openai": {"api_key": f"sk-{idx}"}}
        config_path.write_text(json.dumps(payload), encoding="utf-8")
        assert manager.reload_config() is True

    assert manager.restore_from_backup(1) is True
    assert manager.get_config().openai.api_key == "sk-2"


@pytest.mark.asyncio
async def test_reload_failure_records_recent_failure(
    manager: ConfigManager, monkeypatch: pytest.MonkeyPatch, config_path: Path
) -> None:
    """Errors loading configuration should populate the recent failure log."""

    def raise_error(_: Path) -> dict[str, str]:
        raise OSError("unreadable config")

    monkeypatch.setattr(manager.safe_loader, "load_from_file", raise_error)
    assert await manager.reload_config_async() is False
    failures = manager.get_status()["recent_failures"]
    assert failures and failures[0]["error"]


@pytest.mark.asyncio
async def test_reloader_round_trip(config_path: Path) -> None:
    """The ConfigReloader should reload and expose history metadata."""

    reloader = ConfigReloader()

    async def update_and_reload(idx: int) -> None:
        payload = {"openai": {"api_key": f"sk-{idx}"}}
        config_path.write_text(json.dumps(payload), encoding="utf-8")
        await reloader.reload_config(
            trigger=ReloadTrigger.MANUAL,
            config_source=config_path,
            force=True,
        )

    await asyncio.gather(*(update_and_reload(i) for i in range(3)))
    history = reloader.get_reload_history(limit=3)
    assert len(history) == 3
