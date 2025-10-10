"""Tests for lifecycle protocol helpers."""

from __future__ import annotations

import pytest

from src.services.lifecycle import LifecycleTracker, ServiceLifecycle


class DummyService(LifecycleTracker):
    """Concrete class using LifecycleTracker mixin."""

    async def initialize(self) -> None:
        self._mark_initialized()

    async def cleanup(self) -> None:
        self._mark_uninitialized()

    async def health_check(self):  # pragma: no cover - not used in test
        return None


@pytest.mark.asyncio
async def test_lifecycle_tracker_marks_state() -> None:
    service = DummyService()

    assert service.is_initialized() is False

    await service.initialize()
    assert service.is_initialized() is True

    await service.cleanup()
    assert service.is_initialized() is False


def test_dummy_service_implements_protocol() -> None:
    assert isinstance(DummyService(), ServiceLifecycle)
