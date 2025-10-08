"""Unit tests for FirecrawlAdapter using a stubbed SDK."""

from __future__ import annotations

import sys
import types
from importlib import import_module
from typing import Any

import pytest


def _load_firecrawl_adapter_module() -> types.ModuleType:
    try:
        return import_module("src.services.browser.firecrawl_adapter")
    except ModuleNotFoundError as exc:
        if exc.name not in {
            "firecrawl",
            "src.services.crawling",
            "src.services.crawling.c4a_presets",
        }:
            raise

        if "firecrawl" not in sys.modules:
            firecrawl_module = types.ModuleType("firecrawl")

            class _AsyncFirecrawlPlaceholder:
                async def scrape(self, *_args: Any, **_kwargs: Any) -> dict[str, Any]:
                    return {"success": True}

                async def crawl(self, *_args: Any, **_kwargs: Any) -> dict[str, Any]:
                    return {"success": True}

            firecrawl_module.AsyncFirecrawl = _AsyncFirecrawlPlaceholder  # type: ignore[attr-defined]
            sys.modules["firecrawl"] = firecrawl_module

        if "src.services.crawling" not in sys.modules:
            crawling_module = types.ModuleType("src.services.crawling")

            async def crawl_page(*_args: Any, **_kwargs: Any) -> dict[str, Any]:
                return {}

            crawling_module.crawl_page = crawl_page  # type: ignore[attr-defined]
            sys.modules["src.services.crawling"] = crawling_module

        presets_name = "src.services.crawling.c4a_presets"
        if presets_name not in sys.modules:
            presets_module = types.ModuleType(presets_name)

            class BrowserOptions:  # pragma: no cover - placeholder
                def __init__(self, *args: Any, **kwargs: Any) -> None:
                    self.kwargs = kwargs

            def base_run_config(**_: Any) -> Any:
                return types.SimpleNamespace(clone=lambda **__: types.SimpleNamespace())

            def memory_dispatcher(**_: Any) -> Any:
                return types.SimpleNamespace()

            def preset_browser_config(_options: Any) -> Any:
                return types.SimpleNamespace()

            presets_module.BrowserOptions = BrowserOptions  # type: ignore[attr-defined]
            presets_module.base_run_config = base_run_config  # type: ignore[attr-defined]
            presets_module.memory_dispatcher = memory_dispatcher  # type: ignore[attr-defined]
            presets_module.preset_browser_config = preset_browser_config  # type: ignore[attr-defined]
            sys.modules[presets_name] = presets_module

        return import_module("src.services.browser.firecrawl_adapter")


adapter_module = _load_firecrawl_adapter_module()
FirecrawlAdapter: Any = adapter_module.FirecrawlAdapter
FirecrawlAdapterConfig: Any = adapter_module.FirecrawlAdapterConfig
FirecrawlCrawlOptions: Any = adapter_module.FirecrawlCrawlOptions


@pytest.fixture
def async_firecrawl_stub(monkeypatch):
    """Patch the Firecrawl SDK with a configurable async stub."""

    instances: list[Any] = []

    class StubClient:
        def __init__(self, **kwargs: Any) -> None:
            self.kwargs = kwargs
            self.scrape_payload: dict[str, Any] = {
                "markdown": "# Stub",
                "html": "<h1>Stub</h1>",
                "metadata": {"title": "Stub", "url": "https://stub.local"},
            }
            self.crawl_payload: dict[str, Any] = {
                "status": "completed",
                "completed": 1,
                "total": 1,
                "data": [
                    {
                        "markdown": "Page",
                        "metadata": {"url": "https://stub.local/page"},
                    }
                ],
            }
            self.scrape_call: dict[str, Any] | None = None
            self.crawl_call: dict[str, Any] | None = None
            instances.append(self)

        async def scrape(self, url: str, **kwargs: Any) -> dict[str, Any]:
            self.scrape_call = {"url": url, "kwargs": kwargs}
            return self.scrape_payload

        async def crawl(self, url: str, **kwargs: Any) -> dict[str, Any]:
            self.crawl_call = {"url": url, "kwargs": kwargs}
            return self.crawl_payload

    monkeypatch.setattr(
        "src.services.browser.firecrawl_adapter.AsyncFirecrawl", StubClient
    )
    return instances


@pytest.mark.asyncio
async def test_initialize_requires_api_key(monkeypatch):
    """Adapter should validate that an API key is available."""

    monkeypatch.delenv("AI_DOCS__FIRECRAWL__API_KEY", raising=False)
    adapter = FirecrawlAdapter(FirecrawlAdapterConfig(api_key=""))

    with pytest.raises(ValueError, match="Firecrawl API key missing"):
        await adapter.initialize()


@pytest.mark.asyncio
async def test_initialize_uses_config_key(async_firecrawl_stub):
    """Adapter should instantiate the SDK with the provided config key."""

    config = FirecrawlAdapterConfig(api_key="fc-test-key")
    adapter = FirecrawlAdapter(config)

    await adapter.initialize()
    try:
        assert async_firecrawl_stub
        stub = async_firecrawl_stub[0]
        assert stub.kwargs["api_key"] == "fc-test-key"
    finally:
        await adapter.cleanup()


@pytest.mark.asyncio
async def test_initialize_uses_env_key(monkeypatch, async_firecrawl_stub):
    """Environment variable should supply the API key when config omits it."""

    monkeypatch.setenv("AI_DOCS__FIRECRAWL__API_KEY", "fc-env-key")
    config = FirecrawlAdapterConfig(api_key="")
    adapter = FirecrawlAdapter(config)

    await adapter.initialize()
    try:
        stub = async_firecrawl_stub[0]
        assert stub.kwargs["api_key"] == "fc-env-key"
    finally:
        await adapter.cleanup()
        monkeypatch.delenv("AI_DOCS__FIRECRAWL__API_KEY")


@pytest.mark.asyncio
async def test_scrape_success_normalizes_payload(async_firecrawl_stub):
    """Scrape responses should be normalized to the adapter schema."""

    adapter = FirecrawlAdapter(FirecrawlAdapterConfig(api_key="fc-key"))
    await adapter.initialize()
    try:
        stub = async_firecrawl_stub[0]
        stub.scrape_payload = {
            "markdown": "## Hello",
            "html": "<h2>Hello</h2>",
            "metadata": {"title": "Hello", "url": "https://example.com"},
        }

        result = await adapter.scrape("https://example.com", formats=["markdown"])

        assert result["success"] is True
        assert result["content"] == "## Hello"
        assert result["html"] == "<h2>Hello</h2>"
        assert result["url"] == "https://example.com"
        assert result["title"] == "Hello"
        assert result["provider"] == "firecrawl"
        assert stub.scrape_call["kwargs"]["formats"] == ["markdown"]
    finally:
        await adapter.cleanup()


@pytest.mark.asyncio
async def test_scrape_invalid_formats_returns_failure(async_firecrawl_stub):
    """Invalid format inputs should be reported as errors."""

    adapter = FirecrawlAdapter(FirecrawlAdapterConfig(api_key="fc-key"))
    await adapter.initialize()
    try:
        result = await adapter.scrape("https://example.com", formats=object())  # type: ignore[arg-type]
        assert result["success"] is False
        assert "formats must be a string" in result["error"]
        assert result["provider"] == "firecrawl"
    finally:
        await adapter.cleanup()


@pytest.mark.asyncio
async def test_crawl_normalizes_pages(async_firecrawl_stub):
    """Crawl responses should be flattened and normalized."""

    adapter = FirecrawlAdapter(FirecrawlAdapterConfig(api_key="fc-key"))
    await adapter.initialize()
    try:
        stub = async_firecrawl_stub[0]
        stub.crawl_payload = {
            "status": "completed",
            "completed": 2,
            "total": 2,
            "data": [
                {
                    "markdown": "Page 1",
                    "metadata": {"url": "https://example.com/p1"},
                },
                {
                    "markdown": "Page 2",
                    "metadata": {"url": "https://example.com/p2"},
                },
            ],
        }

        result = await adapter.crawl(
            "https://example.com",
            options=FirecrawlCrawlOptions(limit=5),
            limit=3,
        )

        assert result["success"] is True
        assert result["total_pages"] == 2
        assert result["pages"][0]["url"] == "https://example.com/p1"
        assert stub.crawl_call["kwargs"]["limit"] == 3
    finally:
        await adapter.cleanup()
