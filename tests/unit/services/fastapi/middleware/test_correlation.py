"""Tests for correlation ID helpers."""

from __future__ import annotations

import uuid

import pytest
from starlette.requests import Request

from src.services.fastapi.middleware import correlation


class _DummyCorrelation:
    """Simple contextvar-like stub used in tests."""

    def __init__(self) -> None:
        """Initialize the dummy correlation context."""

        self.value: str | None = None

    def get(self) -> str | None:
        """Get the current correlation value."""

        return self.value

    def set(self, value: str) -> None:
        """Set the correlation value."""

        self.value = value


@pytest.fixture(name="dummy_correlation")
def _dummy_correlation(monkeypatch: pytest.MonkeyPatch) -> _DummyCorrelation:
    """Fixture to patch correlation context with a dummy."""

    dummy = _DummyCorrelation()
    monkeypatch.setattr(correlation, "_cid", dummy)
    return dummy


def _make_request() -> Request:
    """Create a test request for correlation tests."""

    scope = {
        "type": "http",
        "method": "GET",
        "path": "/",
        "headers": [],
    }
    return Request(scope)


def test_generate_correlation_id_returns_uuid() -> None:
    """Test that generate_correlation_id returns a valid UUID."""

    cid = correlation.generate_correlation_id()
    uuid.UUID(cid)  # Should not raise


def test_get_correlation_id_uses_stub(dummy_correlation: _DummyCorrelation) -> None:
    """Test that get_correlation_id uses the stub when available."""

    dummy_correlation.set("abc123")
    assert correlation.get_correlation_id(_make_request()) == "abc123"


def test_get_correlation_id_falls_back_to_uuid(monkeypatch: pytest.MonkeyPatch) -> None:
    """Test that get_correlation_id falls back to generating a UUID."""

    monkeypatch.setattr(correlation, "_cid", None)
    cid = correlation.get_correlation_id(_make_request())
    uuid.UUID(cid)


def test_set_correlation_id_noop_without_library(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Test that set_correlation_id is a no-op without library."""

    monkeypatch.setattr(correlation, "_cid", None)
    # Should not raise
    correlation.set_correlation_id(_make_request(), "ignored")


def test_set_correlation_id_updates_stub(dummy_correlation: _DummyCorrelation) -> None:
    """Test that set_correlation_id updates the stub."""

    correlation.set_correlation_id(_make_request(), "value")
    assert dummy_correlation.get() == "value"
