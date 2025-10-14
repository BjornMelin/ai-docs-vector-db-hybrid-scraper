"""Tests for crawler normalization helpers."""

from __future__ import annotations

import pytest

from src.services.browser.models import BrowserResult, ProviderKind
from src.services.crawling import normalization as normalization_module


@pytest.fixture()
def sample_browser_result() -> BrowserResult:
    """Provide a sample BrowserResult for testing."""
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
) -> None:
    """Verify normalization of BrowserResult into standard payload."""
    payload = normalization_module.normalize_crawler_output(
        sample_browser_result,
        fallback_url="https://fallback.local",
    )

    assert payload["success"] is True
    assert payload["url"] == "https://example.com/page"
    assert payload["metadata"]["provider"] == ProviderKind.CRAWL4AI.value
    assert payload["content"]["markdown"] == "**hello**"
    assert payload["content"]["html"] == "<p>hello</p>"
    assert payload["content"]["text"] == "Plain text fallback"
    assert payload["raw_html"] == "<p>hello</p>"


def test_normalize_crawler_output_from_mapping() -> None:
    """Verify normalization of dict-based crawler output."""
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
) -> None:
    """Verify chunk input resolution prefers structured content."""
    payload = normalization_module.normalize_crawler_output(
        sample_browser_result,
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


def test_resolve_chunk_inputs_raises_when_missing_content() -> None:
    """Verify error raised when content is missing."""
    payload = {"metadata": {"title": "Missing"}}

    with pytest.raises(ValueError):
        normalization_module.resolve_chunk_inputs(
            payload, fallback_url="https://fallback.local"
        )


def test_resolve_chunk_inputs_falls_back_to_raw_html() -> None:
    """Verify fallback to raw HTML when structured content unavailable."""
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
