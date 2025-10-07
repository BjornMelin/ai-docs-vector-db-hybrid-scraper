"""Tests for the path utilities."""

from pathlib import Path

from src.utils import paths


def test_get_cache_dir_returns_path(tmp_path, monkeypatch):
    """User cache dir is derived from platformdirs and resolves to a path."""

    monkeypatch.setenv("LOCALAPPDATA", str(tmp_path))
    result = paths.get_cache_dir("example-app")
    assert isinstance(result, Path)
    assert result.is_absolute()


def test_normalize_path_expands_user(tmp_path, monkeypatch):
    """normalize_path resolves user home tokens."""

    home = tmp_path / "home"
    home.mkdir()
    monkeypatch.setenv("HOME", str(home))
    path = paths.normalize_path("~/data")
    assert path == home / "data"
