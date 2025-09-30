"""Unit tests for the configuration management router."""

from __future__ import annotations

import warnings
from types import SimpleNamespace
from typing import Any, cast

import pytest
from fastapi import FastAPI, status
from fastapi.testclient import TestClient

from src.api.routers import config as config_router


warnings.filterwarnings("ignore", category=DeprecationWarning)


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
        self.config_source = "config.yaml"
        self.backup_count = 1
        self._file_watch_enabled = False
        self.raise_reload_error = False
        self.raise_rollback_error = False

    async def reload_config(self, *args: Any, **_kwargs: Any) -> DummyReloadOperation:
        if self.raise_reload_error:
            raise RuntimeError("reload failure")
        return DummyReloadOperation()

    async def rollback_config(self, *args: Any, **_kwargs: Any) -> DummyReloadOperation:
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
            "listeners_registered": 2,
            "backups_available": 1,
            "current_config_hash": "hash-new",
        }

    def is_file_watch_enabled(self) -> bool:  # noqa: D401 - matches router expectation
        return self._file_watch_enabled

    async def enable_file_watching(self, *args: Any, **_kwargs: Any) -> None:
        del args, _kwargs
        self._file_watch_enabled = True

    async def disable_file_watching(self) -> None:
        self._file_watch_enabled = False

    def get_config_backups(self) -> list[tuple[str, DummyBackupConfig]]:
        return [
            (
                "hash-new",
                DummyBackupConfig(
                    created_at="2024-05-01T10:00:00Z", environment="prod"
                ),
            )
        ]

    async def enable_signal_handler(
        self,
    ) -> None:  # pragma: no cover - unused but present
        return None


class ClientWithReloader(TestClient):
    """Typed test client exposing the mocked config reloader."""

    reloader: DummyReloader


@pytest.fixture()
def app(monkeypatch: pytest.MonkeyPatch) -> ClientWithReloader:
    """Provide a FastAPI test client with the configuration router mounted."""

    reloader = DummyReloader()
    monkeypatch.setattr(config_router, "get_config_reloader", lambda: reloader)
    fastapi_app = FastAPI()
    fastapi_app.include_router(config_router.router)
    client = cast(ClientWithReloader, TestClient(fastapi_app))
    client.reloader = reloader
    return client


def test_reload_configuration_success(app: ClientWithReloader) -> None:
    """Ensure a successful reload returns success metadata."""

    response = app.post("/config/reload", json={"force": True})
    assert response.status_code == 200
    payload = response.json()
    assert payload["success"] is True
    assert payload["changes_applied"] == ["reload"]


def test_reload_configuration_error(app: ClientWithReloader) -> None:
    """Verify reload errors propagate as HTTP 500 responses."""

    app.reloader.raise_reload_error = True
    response = app.post("/config/reload", json={})
    assert response.status_code == status.HTTP_500_INTERNAL_SERVER_ERROR


def test_rollback_configuration_failure(app: ClientWithReloader) -> None:
    """Check rollback failures surface descriptive client errors."""

    app.reloader.raise_rollback_error = True
    response = app.post("/config/rollback", json={})
    assert response.status_code == status.HTTP_400_BAD_REQUEST


def test_history_endpoint(app: ClientWithReloader) -> None:
    """Confirm history endpoint returns operation metadata."""

    response = app.get("/config/history", params={"limit": 5})
    assert response.status_code == 200
    body = response.json()
    assert body["total_count"] == 1
    assert len(body["operations"]) == 1


def test_stats_endpoint(app: ClientWithReloader) -> None:
    """Ensure stats endpoint exposes aggregated counts."""

    response = app.get("/config/stats")
    assert response.status_code == 200
    assert response.json()["total_operations"] == 1


def test_status_endpoint(app: ClientWithReloader) -> None:
    """Validate status endpoint exposes reloader health data."""

    response = app.get("/config/status")
    assert response.status_code == 200
    body = response.json()
    assert body["config_reloader_enabled"] is True
    assert body["reload_statistics"]["total_operations"] == 1


def test_file_watch_toggle(app: ClientWithReloader) -> None:
    """Ensure file watching can be enabled and disabled."""

    enable_response = app.post(
        "/config/file-watch/enable", params={"poll_interval": 0.5}
    )
    assert enable_response.status_code == 200
    disable_response = app.post("/config/file-watch/disable")
    assert disable_response.status_code == 200
    assert disable_response.json()["file_watching_enabled"] is False


def test_backups_endpoint(app: ClientWithReloader) -> None:
    """Verify backups endpoint surfaces the stored metadata."""

    response = app.get("/config/backups")
    assert response.status_code == 200
    body = response.json()
    assert body["available_backups"] == 1
    assert body["backups"][0]["hash"] == "hash-new"
