"""Tests for crawler normalization helpers."""

from __future__ import annotations

import sys
from dataclasses import dataclass
from enum import Enum
from importlib import import_module
from types import ModuleType
from typing import Any, cast

import pytest


class ProviderKind(str, Enum):
    """Minimal provider enum for browser results."""

    CRAWL4AI = "crawl4ai"


@dataclass(slots=True)
class BrowserResult:
    """Lightweight BrowserResult stub mirroring the production shape."""

    success: bool
    url: str
    title: str
    content: str
    html: str
    metadata: dict[str, object]
    provider: ProviderKind


def _load_normalization() -> ModuleType:
    """Load the normalization module with lightweight browser stubs."""

    module_name = "src.services.crawling.normalization"

    browser_pkg = ModuleType("src.services.browser")
    browser_pkg.__path__ = []  # type: ignore[attr-defined]
    models_module = ModuleType("src.services.browser.models")
    models_module.BrowserResult = BrowserResult  # type: ignore[attr-defined]
    models_module.ProviderKind = ProviderKind  # type: ignore[attr-defined]

    backups: dict[str, ModuleType] = {}
    injected_modules: dict[str, ModuleType] = {
        "src": ModuleType("src"),
        "src.services": ModuleType("src.services"),
        "src.services.browser": browser_pkg,
        "src.services.browser.models": models_module,
    }

    sys.modules.pop(module_name, None)
    for name, module in injected_modules.items():
        if name in sys.modules:
            backups[name] = sys.modules[name]  # type: ignore[assignment]
        sys.modules[name] = module

    try:
        return import_module(module_name)
    finally:
        for name in injected_modules:
            if name in backups:
                sys.modules[name] = backups[name]
            else:
                sys.modules.pop(name, None)


@pytest.fixture(scope="module")
def normalization_module() -> ModuleType:
    """Provide the normalization module with browser stubs injected."""

    return _load_normalization()


@pytest.fixture()
def sample_browser_result() -> BrowserResult:
    metadata = {
        "title": "Sample Page",
        "content": {"markdown": "**hello**"},
        "language": "en",
        "raw_html": "<p>hello</p>",
    }
    return BrowserResult(
        success=True,
        url="https://example.com/page",
        title="Sample Page",
        content="Plain text fallback",
        html="<p>hello</p>",
        metadata=metadata,
        provider=ProviderKind.CRAWL4AI,
    )


def test_normalize_crawler_output_from_browser_result(
    sample_browser_result: BrowserResult,
    normalization_module: ModuleType,
) -> None:
    payload = normalization_module.normalize_crawler_output(
        cast(Any, sample_browser_result),
        fallback_url="https://fallback.local",
    )

    assert payload["success"] is True
    assert payload["url"] == "https://example.com/page"
    assert payload["metadata"]["provider"] == ProviderKind.CRAWL4AI.value
    assert payload["content"]["markdown"] == "**hello**"
    assert payload["content"]["html"] == "<p>hello</p>"
    assert payload["content"]["text"] == "Plain text fallback"
    assert payload["raw_html"] == "<p>hello</p>"


def test_normalize_crawler_output_from_mapping(
    normalization_module: ModuleType,
) -> None:
    payload = normalization_module.normalize_crawler_output(
        {
            "title": "Mapped",
            "content": {"html": "<article>content</article>"},
            "metadata": {"content_type": "text/html"},
        },
        fallback_url="https://fallback.local",
    )

    assert payload["success"] is True
    assert payload["url"] == "https://fallback.local"
    assert payload["metadata"]["content_type"] == "text/html"
    assert payload["raw_html"] == "<article>content</article>"
    assert payload["content"]["html"] == "<article>content</article>"


def test_resolve_chunk_inputs_prefers_structured_content(
    sample_browser_result: BrowserResult,
    normalization_module: ModuleType,
) -> None:
    payload = normalization_module.normalize_crawler_output(
        cast(Any, sample_browser_result),
        fallback_url="https://fallback.local",
    )

    raw, metadata, kind = normalization_module.resolve_chunk_inputs(
        payload,
        fallback_url="https://fallback.local",
    )

    assert raw == "**hello**"
    assert metadata["source"] == "https://example.com/page"
    assert metadata["title"] == "Sample Page"
    assert metadata["language"] == "en"
    assert kind == "markdown"


def test_resolve_chunk_inputs_raises_when_missing_content(
    normalization_module: ModuleType,
) -> None:
    payload = {"metadata": {"title": "Missing"}}

    with pytest.raises(ValueError):
        normalization_module.resolve_chunk_inputs(
            payload, fallback_url="https://fallback.local"
        )


def test_resolve_chunk_inputs_falls_back_to_raw_html(
    normalization_module: ModuleType,
) -> None:
    payload = {
        "url": "https://example.com/page",
        "metadata": {"title": "Sample"},
        "raw_html": "<div>hi</div>",
    }

    raw, metadata, kind = normalization_module.resolve_chunk_inputs(
        payload, fallback_url="https://fallback.local"
    )

    assert raw == "<div>hi</div>"
    assert metadata["source"] == "https://example.com/page"
    assert kind == "html"
