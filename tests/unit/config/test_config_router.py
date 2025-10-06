"""Unit tests for the configuration management router."""

from __future__ import annotations

import importlib.util
import warnings
from pathlib import Path
from types import SimpleNamespace
from typing import Any, cast

import pytest
from fastapi import FastAPI, status
from fastapi.testclient import TestClient

from src.config.reloader import ConfigError, ConfigLoadError
from src.services.monitoring.file_integrity import StubFileIntegrityProvider


warnings.filterwarnings("ignore", category=DeprecationWarning)

BASE_DIR = Path(__file__).resolve().parents[3]
MODULE_PATH = BASE_DIR / "src" / "api" / "routers" / "config.py"
CONFIG_SPEC = importlib.util.spec_from_file_location("config_router", MODULE_PATH)
assert CONFIG_SPEC is not None and CONFIG_SPEC.loader is not None
config_router = importlib.util.module_from_spec(CONFIG_SPEC)
CONFIG_SPEC.loader.exec_module(config_router)


class DummyBackupConfig:
    """Simple container representing a configuration backup."""

    def __init__(self, *, created_at: str | None, environment: str | None) -> None:
        self.created_at = created_at
        self.environment = environment


class DummyReloadOperation:  # pylint: disable=too-many-instance-attributes
    """Reload operation with the attributes the router expects."""

    def __init__(
        self, *, success: bool = True, error_message: str | None = None
    ) -> None:
        self.operation_id = "op-1"
        self.status = SimpleNamespace(value="completed" if success else "failed")
        self.success = success
        self.error_message = error_message
        self.total_duration_ms = 12.3
        self.validation_duration_ms = 4.2
        self.apply_duration_ms = 8.1
        self.previous_config_hash = "hash-old"
        self.new_config_hash = "hash-new"
        self.changes_applied: list[str] = ["reload"]
        self.services_notified: list[str] = ["api"]
        self.validation_errors: list[str] = []
        self.validation_warnings: list[str] = []


class DummyReloader:  # pylint: disable=too-many-instance-attributes
    """In-memory stub used to drive the router in tests."""

    def __init__(self) -> None:
        self.backup_count = 1
        self._provider: StubFileIntegrityProvider | None = None
        self._file_watch_enabled = False
        self.raise_reload_error = False
        self.raise_rollback_error = False
        self.fail_set_default = False
        self.operation_success = True
        self.operation_error_message = "operation failed"
        self.raise_watch_error = False
        self._default_source: Path | None = Path("/tmp/config.json")

    async def reload_config(
        self,
        *,
        trigger: Any,
        config_source: Path | None = None,
        force: bool = False,
    ) -> DummyReloadOperation:
        del trigger, config_source, force
        if self.raise_reload_error:
            raise RuntimeError("reload failure")
        if self.operation_success:
            return DummyReloadOperation()
        return DummyReloadOperation(
            success=False, error_message=self.operation_error_message
        )

    async def rollback_config(
        self,
        *,
        target_hash: str | None = None,
    ) -> DummyReloadOperation:
        del target_hash
        if self.raise_rollback_error:
            return DummyReloadOperation(success=False, error_message="rollback failed")
        return DummyReloadOperation()

    def get_reload_history(self, limit: int) -> list[DummyReloadOperation]:
        del limit
        return [DummyReloadOperation()]

    def get_reload_stats(self) -> dict[str, Any]:
        return {
            "total_operations": 1,
            "successful_operations": 1,
            "failed_operations": 0,
            "success_rate": 1.0,
            "average_duration_ms": 12.3,
            "backups_available": 1,
            "current_config_hash": "hash-new",
        }

    def get_default_config_source(self) -> Path | None:
        return self._default_source

    def set_default_config_source(self, config_source: Path | str | None) -> None:
        if self.fail_set_default:
            raise ConfigLoadError("invalid path requested")
        if config_source is None:
            self._default_source = None
        else:
            self._default_source = Path(config_source)

    def is_file_watch_enabled(self) -> bool:  # noqa: D401 - matches router expectation
        return self._file_watch_enabled

    async def attach_file_integrity_provider(self, provider: Any) -> None:
        self._provider = provider

    async def enable_file_watching(self, *, poll_interval: float | None = None) -> None:
        del poll_interval
        if self.raise_watch_error:
            raise ConfigError("watch error")
        self._file_watch_enabled = True
        if self._provider is not None and hasattr(self._provider, "start"):
            await self._provider.start()

    async def disable_file_watching(self) -> None:
        self._file_watch_enabled = False
        if self._provider is not None and hasattr(self._provider, "stop"):
            await self._provider.stop()

    def get_file_integrity_provider(self) -> Any:
        return self._provider

    def get_config_backups(self) -> list[tuple[str, DummyBackupConfig]]:
        return [
            (
                "hash-new",
                DummyBackupConfig(
                    created_at="2024-05-01T10:00:00Z", environment="prod"
                ),
            )
        ]


ClientWithReloader = TestClient  # Alias for readability


@pytest.fixture()
def app(monkeypatch: pytest.MonkeyPatch) -> ClientWithReloader:
    """Provide a FastAPI test client with the configuration router mounted."""

    reloader = DummyReloader()
    monkeypatch.setattr(config_router, "get_config_reloader", lambda: reloader)
    fastapi_app = FastAPI()  # type: ignore[operator]
    fastapi_app.include_router(config_router.router)
    client = TestClient(fastapi_app)
    cast(Any, client).reloader = reloader
    return cast(ClientWithReloader, client)


def test_reload_configuration_success(app: ClientWithReloader) -> None:
    """Ensure a successful reload returns success metadata."""

    response = app.post("/config/reload", json={"force": True})
    assert response.status_code == 200
    payload = response.json()
    assert payload["success"] is True
    assert payload["changes_applied"] == ["reload"]


def test_reload_configuration_error(app: ClientWithReloader) -> None:
    """Ensure unexpected reloader errors surface as HTTP 500."""

    reloader = cast(DummyReloader, cast(Any, app).reloader)
    reloader.raise_reload_error = True
    response = app.post("/config/reload", json={})
    assert response.status_code == status.HTTP_500_INTERNAL_SERVER_ERROR


def test_reload_configuration_operation_failure_returns_400(
    app: ClientWithReloader,
) -> None:
    """Operator errors should map to HTTP 400 responses."""

    reloader = cast(DummyReloader, cast(Any, app).reloader)
    reloader.operation_success = False
    reloader.operation_error_message = "invalid config payload"
    response = app.post("/config/reload", json={})
    assert response.status_code == status.HTTP_400_BAD_REQUEST
    assert "invalid config" in response.json()["detail"]


def test_reload_configuration_invalid_path_returns_400(
    app: ClientWithReloader,
) -> None:
    """Invalid config_source input should be rejected with HTTP 400."""

    reloader = cast(DummyReloader, cast(Any, app).reloader)
    reloader.fail_set_default = True
    response = app.post("/config/reload", json={"config_source": "/tmp/missing.yml"})
    assert response.status_code == status.HTTP_400_BAD_REQUEST


def test_rollback_configuration_failure(app: ClientWithReloader) -> None:
    """Rollback failures should surface as HTTP 400."""

    reloader = cast(DummyReloader, cast(Any, app).reloader)
    reloader.raise_rollback_error = True
    response = app.post("/config/rollback", json={})
    assert response.status_code == status.HTTP_400_BAD_REQUEST


def test_history_endpoint(app: ClientWithReloader) -> None:
    """History endpoint returns serialized operations."""

    response = app.get("/config/history", params={"limit": 5})
    assert response.status_code == 200
    body = response.json()
    assert body["total_count"] == 1
    assert len(body["operations"]) == 1


def test_stats_endpoint(app: ClientWithReloader) -> None:
    """Statistics endpoint returns aggregate counters."""

    response = app.get("/config/stats")
    assert response.status_code == 200
    assert response.json()["total_operations"] == 1


def test_status_endpoint(app: ClientWithReloader) -> None:
    """Status endpoint returns a health snapshot."""

    response = app.get("/config/status")
    assert response.status_code == 200
    body = response.json()
    assert body["config_reloader_enabled"] is True
    assert body["reload_statistics"]["total_operations"] == 1


def test_enable_file_watching_requires_config_source(
    app: ClientWithReloader,
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    """Enabling file watching without a configured source should fail."""

    reloader = cast(DummyReloader, cast(Any, app).reloader)
    reloader.set_default_config_source(None)
    log_path = tmp_path / "osqueryd.results.log"
    log_path.write_text("", encoding="utf-8")
    monkeypatch.setenv("OSQUERY_RESULTS_LOG", str(log_path))

    response = app.post("/config/file-watch/enable")
    assert response.status_code == status.HTTP_400_BAD_REQUEST


def test_enable_file_watching_success(
    app: ClientWithReloader,
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    """File watching endpoint should report readiness when provider is ready."""

    reloader = cast(DummyReloader, cast(Any, app).reloader)
    config_file = tmp_path / "config.yaml"
    config_file.write_text("{}", encoding="utf-8")
    reloader.set_default_config_source(config_file)

    log_path = tmp_path / "osqueryd.results.log"
    log_path.write_text("", encoding="utf-8")
    monkeypatch.setenv("OSQUERY_RESULTS_LOG", str(log_path))

    class ReadyProvider(StubFileIntegrityProvider):
        """Stub osquery provider that matches the production signature."""

        def __init__(self, *, results_log: Path, poll_interval: float) -> None:
            super().__init__()
            self.results_log = results_log
            self.poll_interval = poll_interval

    monkeypatch.setattr(config_router, "OsqueryFileIntegrityProvider", ReadyProvider)

    response = app.post("/config/file-watch/enable")
    assert response.status_code == status.HTTP_200_OK
    payload = response.json()
    assert payload["file_watching_enabled"] is True
    assert payload["file_watching_ready"] is True
    assert payload["config_source"].endswith("config.yaml")


def test_file_watch_toggle(
    app: ClientWithReloader,
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    """Enabling and disabling file watching should succeed."""

    reloader = cast(DummyReloader, cast(Any, app).reloader)
    config_file = tmp_path / "config.yaml"
    config_file.write_text("{}", encoding="utf-8")
    reloader.set_default_config_source(config_file)

    log_path = tmp_path / "osqueryd.results.log"
    log_path.write_text("", encoding="utf-8")
    monkeypatch.setenv("OSQUERY_RESULTS_LOG", str(log_path))

    class ReadyProvider(StubFileIntegrityProvider):
        def __init__(self, *, results_log: Path, poll_interval: float) -> None:
            super().__init__()
            self.results_log = results_log
            self.poll_interval = poll_interval

    monkeypatch.setattr(config_router, "OsqueryFileIntegrityProvider", ReadyProvider)

    enable_response = app.post(
        "/config/file-watch/enable", params={"poll_interval": 0.5}
    )
    assert enable_response.status_code == 200
    disable_response = app.post("/config/file-watch/disable")
    assert disable_response.status_code == 200
    assert disable_response.json()["file_watching_enabled"] is False


def test_backups_endpoint(app: ClientWithReloader) -> None:
    """Backups endpoint returns metadata for available snapshots."""

    response = app.get("/config/backups")
    assert response.status_code == 200
    body = response.json()
    assert body["available_backups"] == 1
    assert body["backups"][0]["hash"] == "hash-new"
