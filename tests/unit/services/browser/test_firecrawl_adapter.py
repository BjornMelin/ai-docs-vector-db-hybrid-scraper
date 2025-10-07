from __future__ import annotations

from types import SimpleNamespace
from unittest.mock import AsyncMock

import pytest
from firecrawl.v2.types import (
    CrawlJob,
    Document,
    DocumentMetadata,
    SearchData,
    SearchResultWeb,
)

from src.services.browser.firecrawl_adapter import (
    FirecrawlAdapter,
    FirecrawlAdapterConfig,
)


@pytest.fixture
def adapter() -> FirecrawlAdapter:
    """Fixture for FirecrawlAdapter with mocked client."""

    config = FirecrawlAdapterConfig(api_key="test-key")
    instance = FirecrawlAdapter(config)
    instance._initialized = True  # type: ignore[attr-defined]
    return instance


@pytest.mark.asyncio
async def test_scrape_normalizes_document(adapter: FirecrawlAdapter) -> None:
    """Test normalization of a scraped document from Firecrawl."""

    document = Document(
        markdown="**Hello**",
        html="<p>Hello</p>",
        metadata=DocumentMetadata(
            title="Hello",
            source_url="https://example.com/page",
            status_code=200,
        ),
    )
    adapter._client = SimpleNamespace(  # type: ignore[assignment]
        scrape=AsyncMock(return_value=document)
    )

    result = await adapter.scrape("https://example.com/page")

    assert result["success"] is True
    assert result["url"] == "https://example.com/page"
    assert result["title"] == "Hello"
    assert result["content"] == "**Hello**"
    assert result["html"] == "<p>Hello</p>"
    assert result["metadata"]["status_code"] == 200
    assert "provider" in result


@pytest.mark.asyncio
async def test_crawl_normalizes_job(adapter: FirecrawlAdapter) -> None:
    """Test normalization of crawl jobs from Firecrawl."""

    pages = [
        Document(
            markdown="Page 1",
            metadata=DocumentMetadata(url="https://example.com/a"),
        ),
        Document(
            markdown="Page 2",
            metadata=DocumentMetadata(url="https://example.com/b"),
        ),
    ]
    job = CrawlJob(status="completed", total=2, completed=2, credits_used=1, data=pages)

    adapter._client = SimpleNamespace(  # type: ignore[assignment]
        crawl=AsyncMock(return_value=job)
    )

    result = await adapter.crawl("https://example.com", limit=2)

    assert result["success"] is True
    assert result["status"] == "completed"
    assert result["seed_url"] == "https://example.com"
    assert result["total_pages"] == 2
    assert {page["url"] for page in result["pages"]} == {
        "https://example.com/a",
        "https://example.com/b",
    }


@pytest.mark.asyncio
async def test_search_normalizes_mixed_results(adapter: FirecrawlAdapter) -> None:
    """Test normalization of mixed search results from Firecrawl."""

    search_data = SearchData(
        web=[
            SearchResultWeb(
                url="https://news.example.com",
                title="Breaking",
                description="Something happened",
            ),
            Document(
                markdown="Result content",
                metadata=DocumentMetadata(url="https://docs.example.com"),
            ),
        ]
    )
    adapter._client = SimpleNamespace(  # type: ignore[assignment]
        search=AsyncMock(return_value=search_data)
    )

    result = await adapter.search("firecrawl", sources=["web"])

    assert result["success"] is True
    assert result["sources"] == ["web"]
    assert "web" in result["results"]
    web_results = result["results"]["web"]

    assert web_results[0]["url"] == "https://news.example.com"
    assert web_results[0]["title"] == "Breaking"

    document_result = web_results[1]
    assert document_result["url"] == "https://docs.example.com"
    assert document_result["content"] == "Result content"
    assert "provider" not in document_result
    assert "success" not in document_result
