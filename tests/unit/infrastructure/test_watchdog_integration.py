"""Tests for ConfigReloader integration and file watching."""

from __future__ import annotations

import json
from collections.abc import Iterator
from pathlib import Path

import pytest

from src.config import Config
from src.config.loader import get_config, reset_config, set_config
from src.config.models import Environment
from src.config.reloader import (
    ConfigError,
    ConfigLoadError,
    ConfigReloader,
    ReloadTrigger,
)
from src.services.monitoring.file_integrity import (
    FileChangeAction,
    FileChangeEvent,
    StubFileIntegrityProvider,
)


@pytest.fixture(autouse=True)
def clean_config_state() -> Iterator[None]:
    """Reset global Config cache before and after each test."""

    reset_config()
    yield
    reset_config()


class TestConfigReloaderFileWatching:
    """Tests covering file watching behaviour for ConfigReloader."""

    @pytest.mark.asyncio
    async def test_enable_file_watching_triggers_reload(self, tmp_path: Path) -> None:
        """Test that enabling file watching triggers config reload on file change."""
        config_path = tmp_path / "config.json"
        config_path.write_text("{}", encoding="utf-8")

        reloader = ConfigReloader(config_source=config_path)
        provider = StubFileIntegrityProvider()
        await reloader.attach_file_integrity_provider(provider)

        reload_calls: list[tuple[ReloadTrigger, Path | None]] = []

        async def fake_reload_config(
            *, trigger: ReloadTrigger, config_source: Path | None, force: bool = False
        ) -> None:
            reload_calls.append((trigger, config_source))

        reloader.reload_config = fake_reload_config  # type: ignore[assignment]

        await reloader.enable_file_watching(poll_interval=0)

        await provider.emit(
            FileChangeEvent(path=config_path, action=FileChangeAction.UPDATED)
        )

        assert reload_calls
        trigger, source = reload_calls[0]
        assert trigger == ReloadTrigger.FILE_WATCH
        assert source == config_path
        assert reloader.is_file_watch_enabled() is True

    @pytest.mark.asyncio
    async def test_enable_file_watching_requires_provider(self, tmp_path: Path) -> None:
        """Test that enabling file watching raises error when provider missing."""
        config_path = tmp_path / "config.json"
        config_path.write_text("{}", encoding="utf-8")

        reloader = ConfigReloader(config_source=config_path)

        with pytest.raises(ConfigError):
            await reloader.enable_file_watching()

    @pytest.mark.asyncio
    async def test_enable_file_watching_missing_source(self, tmp_path: Path) -> None:
        """Test that enabling file watching raises error for missing config source."""
        missing_path = tmp_path / "missing.json"
        reloader = ConfigReloader(config_source=missing_path)
        provider = StubFileIntegrityProvider()
        await reloader.attach_file_integrity_provider(provider)

        with pytest.raises(ConfigLoadError):
            await reloader.enable_file_watching()

    @pytest.mark.asyncio
    async def test_disable_file_watching_cancels_task(
        self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
    ) -> None:
        """Test that disabling file watching cancels the watch task."""
        config_path = tmp_path / "config.json"
        config_path.write_text("{}", encoding="utf-8")

        reloader = ConfigReloader(config_source=config_path)
        provider = StubFileIntegrityProvider()
        await reloader.attach_file_integrity_provider(provider)

        await reloader.enable_file_watching(poll_interval=0)
        assert reloader.is_file_watch_enabled() is True
        assert provider._running is True  # type: ignore[attr-defined]

        await reloader.disable_file_watching()
        assert reloader.is_file_watch_enabled() is False
        assert provider._running is False  # type: ignore[attr-defined]


class TestConfigReloaderOperations:
    """Integration tests for reload and rollback flows."""

    @pytest.mark.asyncio
    async def test_reload_and_rollback_flow(self, tmp_path: Path) -> None:
        """Test reload and rollback config operations with backup verification."""
        initial = Config(environment=Environment.TESTING, app_name="initial")
        set_config(initial)

        config_path = tmp_path / "config.json"
        config_path.write_text(
            json.dumps({"environment": "testing", "app_name": "updated"}),
            encoding="utf-8",
        )

        reloader = ConfigReloader(config_source=config_path)
        operation = await reloader.reload_config(config_source=config_path)

        assert operation.success is True
        assert get_config().app_name == "updated"

        backups = list(reloader.get_config_backups())
        assert len(backups) == 1
        backup_hash, _backup = backups[0]

        rollback_operation = await reloader.rollback_config(target_hash=backup_hash)
        assert rollback_operation.success is True
        assert get_config().app_name == "initial"

        stats = reloader.get_reload_stats()
        assert stats["total_operations"] == 2
        assert stats["successful_operations"] == 2
        assert stats["backups_available"] == 1
