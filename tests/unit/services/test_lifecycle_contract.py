"""Contract tests for the unified service lifecycle."""

from __future__ import annotations

import pytest

from src.architecture.service_factory import BaseService as FactoryBaseService
from src.config.models import Crawl4AIConfig
from src.services.base import BaseService
from src.services.browser.crawl4ai_adapter import Crawl4AIAdapter
from src.services.lifecycle import ServiceLifecycle


class DummyService(BaseService):
    """Minimal BaseService implementation for lifecycle validation."""

    def __init__(self) -> None:
        super().__init__(config=None)
        self.init_calls = 0
        self.cleanup_calls = 0

    async def initialize(self) -> None:
        self.init_calls += 1
        self._mark_initialized()

    async def cleanup(self) -> None:
        self.cleanup_calls += 1
        self._mark_uninitialized()

    async def health_check(self):  # type: ignore[override]
        return {"ok": True}


class DummyFactoryService(FactoryBaseService):
    """Minimal factory BaseService implementation."""

    def __init__(self) -> None:
        super().__init__()
        self.init_calls = 0
        self.cleanup_calls = 0

    async def initialize(self) -> None:
        self.init_calls += 1
        self._mark_initialized()

    async def cleanup(self) -> None:
        self.cleanup_calls += 1
        self._mark_cleanup()

    def get_service_name(self) -> str:
        return "dummy_factory_service"

    async def health_check(self):  # type: ignore[override]
        return {"ok": True}


@pytest.mark.asyncio
async def test_base_service_lifecycle_marks_initialized() -> None:
    """BaseService should reflect initialized state through lifecycle helpers."""

    service = DummyService()
    assert isinstance(service, ServiceLifecycle)
    assert not service.is_initialized()

    await service.initialize()
    assert service.is_initialized()
    assert service.init_calls == 1

    await service.cleanup()
    assert not service.is_initialized()
    assert service.cleanup_calls == 1


@pytest.mark.asyncio
async def test_factory_service_lifecycle_marks_initialized() -> None:
    """Factory BaseService should also follow unified lifecycle semantics."""

    service = DummyFactoryService()
    assert isinstance(service, ServiceLifecycle)
    assert not service.is_initialized()

    await service.initialize()
    assert service.is_initialized()

    await service.cleanup()
    assert not service.is_initialized()


@pytest.mark.asyncio
async def test_crawl4ai_adapter_implements_service_lifecycle(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Crawl4AIAdapter should comply with ServiceLifecycle protocol."""

    async def start(self) -> None:  # type: ignore[no-self-use]
        return None

    async def close(self) -> None:  # type: ignore[no-self-use]
        return None

    crawler_mock = type("_Crawler", (), {"start": start, "close": close})
    monkeypatch.setattr(
        "src.services.browser.crawl4ai_adapter.AsyncWebCrawler",
        lambda config: crawler_mock(),
    )

    adapter = Crawl4AIAdapter(Crawl4AIConfig())
    assert isinstance(adapter, ServiceLifecycle)
    assert not adapter.is_initialized()

    await adapter.initialize()
    assert adapter.is_initialized()

    await adapter.cleanup()
    assert not adapter.is_initialized()
