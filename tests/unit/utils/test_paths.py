"""Tests for the path utilities."""

from __future__ import annotations

from collections.abc import Callable
from pathlib import Path

import pytest

from src.utils import paths


@pytest.fixture
def resolver(tmp_path: Path) -> Callable[[str], Path]:
    """Provide a helper to convert string paths into resolved directories."""

    def _resolve(relative: str) -> Path:
        target = tmp_path / relative
        target.mkdir(parents=True, exist_ok=True)
        return target

    return _resolve


def test_get_cache_dir_returns_path(
    monkeypatch: pytest.MonkeyPatch,
    resolver: Callable[[str], Path],
) -> None:
    """User cache dir is derived from platformdirs and resolves to a path."""

    expected = resolver("cache")

    def fake_user_cache_dir(app_name: str, appauthor: None) -> str:
        assert app_name == "example-app"
        assert appauthor is None
        return str(expected)

    monkeypatch.setattr(paths, "user_cache_dir", fake_user_cache_dir)
    result = paths.get_cache_dir("example-app")

    assert result == expected.resolve()


def test_get_config_dir_uses_platformdirs(
    monkeypatch: pytest.MonkeyPatch,
    resolver: Callable[[str], Path],
) -> None:
    """Config dir helper proxies through platformdirs."""

    expected = resolver("config")

    def fake_user_config_dir(app_name: str, appauthor: None) -> str:
        assert app_name == "example-app"
        assert appauthor is None
        return str(expected)

    monkeypatch.setattr(paths, "user_config_dir", fake_user_config_dir)
    result = paths.get_config_dir("example-app")

    assert result == expected.resolve()


def test_get_data_dir_uses_platformdirs(
    monkeypatch: pytest.MonkeyPatch,
    resolver: Callable[[str], Path],
) -> None:
    """Data dir helper proxies through platformdirs."""

    expected = resolver("data")

    def fake_user_data_dir(app_name: str, appauthor: None) -> str:
        assert app_name == "example-app"
        assert appauthor is None
        return str(expected)

    monkeypatch.setattr(paths, "user_data_dir", fake_user_data_dir)
    result = paths.get_data_dir("example-app")

    assert result == expected.resolve()


def test_get_log_dir_uses_platformdirs(
    monkeypatch: pytest.MonkeyPatch,
    resolver: Callable[[str], Path],
) -> None:
    """Log dir helper proxies through platformdirs."""

    expected = resolver("logs")

    def fake_user_log_dir(app_name: str, appauthor: None) -> str:
        assert app_name == "example-app"
        assert appauthor is None
        return str(expected)

    monkeypatch.setattr(paths, "user_log_dir", fake_user_log_dir)
    result = paths.get_log_dir("example-app")

    assert result == expected.resolve()


def test_normalize_path_expands_user(
    monkeypatch: pytest.MonkeyPatch,
    resolver: Callable[[str], Path],
) -> None:
    """normalize_path resolves user home tokens."""

    home = resolver("home")
    monkeypatch.setenv("HOME", str(home))
    path = paths.normalize_path("~/data")

    assert path == home / "data"


def test_normalize_path_accepts_path_instances(tmp_path: Path) -> None:
    """Passing a Path instance returns its resolved equivalent."""

    target = (tmp_path / "store").resolve()
    assert paths.normalize_path(target) == target
