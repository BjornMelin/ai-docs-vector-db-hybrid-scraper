"""Tests for translating LangChain documents into TextDocument payloads."""

from __future__ import annotations

from dataclasses import dataclass
from types import SimpleNamespace

from langchain_core.documents import Document

from src.services.vector_db.document_builder import (
    DocumentBuildParams,
    build_params_from_crawl,
    build_text_documents,
)


@dataclass
class DummyContentType:
    """Hashable content type stub for enrichment tests."""

    value: str

    def __hash__(self) -> int:
        return hash(self.value)


def test_build_params_from_crawl_extracts_defaults() -> None:
    crawl_result = {
        "url": "https://example.com/doc",
        "title": "Example Doc",
        "provider": "tier-1",
        "quality_score": 0.91,
        "metadata": {
            "title": "Example Doc",
            "content_type": "text/html",
            "language": "en",
            "custom": "value",
        },
    }

    params = build_params_from_crawl(
        crawl_result,
        fallback_url="https://fallback.local",
        tenant="documentation",
        doc_id="doc-123",
    )

    assert params.source_url == "https://example.com/doc"
    assert params.title == "Example Doc"
    assert params.provider == "tier-1"
    assert params.default_content_type == "text/html"
    assert params.language_hint == "en"
    assert params.base_metadata is not None
    assert params.base_metadata["custom"] == "value"


def test_build_text_documents_merges_metadata() -> None:
    params = DocumentBuildParams(
        doc_id="doc-123",
        tenant="documentation",
        source_url="https://example.com/doc",
        title="Example Doc",
        provider="tier-1",
        default_content_type="text/plain",
        language_hint="en",
    )
    chunks = [
        Document(
            page_content="alpha", metadata={"section": "intro", "chunk_id": "hash-1"}
        ),
        Document(page_content="beta", metadata={"section": "body", "start_index": 5}),
    ]

    documents = build_text_documents(chunks, params)

    assert len(documents) == 2
    first = documents[0]
    first_metadata: dict[str, object] = dict(first.metadata or {})
    assert first.id == "doc-123:0"
    assert first_metadata["section"] == "intro"
    assert first_metadata["chunk_index"] == 0
    assert first_metadata["chunk_id"] == 0
    assert first_metadata["chunk_hash"] == "hash-1"
    assert first_metadata["lang"] == "en"
    assert first_metadata["content_type"] == "text/plain"
    assert "start_char" not in first_metadata
    assert "end_char" not in first_metadata
    assert first_metadata["total_chunks"] == 2

    second = documents[1]
    second_metadata: dict[str, object] = dict(second.metadata or {})
    assert second_metadata["chunk_index"] == 1
    assert second_metadata["chunk_id"] == 1
    assert second_metadata["start_char"] == 5
    assert second_metadata["end_char"] == 5 + len("beta")
    assert second_metadata["doc_id"] == "doc-123"
    assert second_metadata["tenant"] == "documentation"


def test_build_text_documents_applies_enrichment_metadata() -> None:
    primary = DummyContentType("guide")
    secondary = DummyContentType("cheatsheet")
    enriched = SimpleNamespace(
        classification=SimpleNamespace(
            primary_type=primary,
            secondary_types=[secondary],
            confidence_scores={primary: 0.88},
        ),
        quality_score=SimpleNamespace(
            overall_score=0.91,
            completeness=0.93,
            relevance=0.9,
            confidence=0.89,
        ),
        metadata=SimpleNamespace(
            word_count=1200,
            char_count=6800,
            language="en",
            semantic_tags=["docs"],
        ),
    )

    params = DocumentBuildParams(
        doc_id="doc-456",
        tenant="documentation",
        source_url="https://example.com/doc",
        title="Example Doc",
        provider="tier-1",
        enriched_content=enriched,
    )
    chunks = [Document(page_content="alpha")]

    documents = build_text_documents(chunks, params)

    assert len(documents) == 1
    metadata: dict[str, object] = dict(documents[0].metadata or {})
    assert metadata["content_intelligence_analyzed"] is True
    assert metadata["content_type"] == "guide"
    assert metadata["quality_overall"] == 0.91
    assert metadata["quality_confidence"] == 0.89
    assert metadata["content_confidence"] == 0.88
    assert metadata["ci_word_count"] == 1200
    assert metadata["ci_semantic_tags"] == ["docs"]
    assert metadata["secondary_content_types"] == ["cheatsheet"]
