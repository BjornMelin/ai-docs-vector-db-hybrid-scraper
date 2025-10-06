"""Unit tests for MCP document ingestion tools backed by VectorStoreService."""

from __future__ import annotations

from collections.abc import Callable
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock, Mock

import pytest

from src.mcp_tools.models.requests import BatchRequest, DocumentRequest
from src.mcp_tools.models.responses import AddDocumentResponse
from src.mcp_tools.tools import documents
from src.services.vector_db.types import CollectionSchema, TextDocument


class DummyContentType:
    """Minimal content type record matching the enrichment interface."""

    def __init__(self, value: str):
        self.value = value

    def __hash__(self) -> int:  # pragma: no cover - hashing required for dict keys
        return hash(self.value)

    def __eq__(self, other: object) -> bool:  # pragma: no cover - defensive
        if not isinstance(other, DummyContentType):
            return NotImplemented
        return self.value == other.value


class VectorServiceStub:
    """Async stub for VectorStoreService interactions used in tests."""

    def __init__(self) -> None:
        self._initialized = False
        self.ensure_collection = AsyncMock()
        self.upsert_documents = AsyncMock()

    def is_initialized(self) -> bool:
        return self._initialized

    async def initialize(self) -> None:
        self._initialized = True

    @property
    def embedding_dimension(self) -> int:
        return 1536


class DummyValidator:
    """URL validator stub returning the original URL."""

    def validate_url(self, url: str) -> str:
        return url


class DummyChunker:
    """Chunker stub that returns deterministic segments."""

    def __init__(self, config):
        self.config = config

    def chunk_content(self, content: str, title: str, url: str):
        return [
            {"content": "chunk-0", "metadata": {"section": "intro"}},
            {"content": "chunk-1", "metadata": {"section": "body"}},
        ]


def _make_enriched_content() -> SimpleNamespace:
    primary = DummyContentType("guide")
    secondary = [DummyContentType("cheatsheet")]
    classification = SimpleNamespace(
        primary_type=primary,
        secondary_types=secondary,
        confidence_scores={primary: 0.92},
    )
    quality_score = SimpleNamespace(
        overall_score=0.87,
        completeness=0.82,
        relevance=0.91,
        confidence=0.85,
    )
    metadata = SimpleNamespace(
        word_count=1200,
        char_count=7800,
        language="en",
        semantic_tags=["documentation"],
    )
    return SimpleNamespace(
        classification=classification,
        quality_score=quality_score,
        metadata=metadata,
    )


@pytest.fixture()
def documents_env(monkeypatch) -> SimpleNamespace:
    """Provide registered document tools with mocked dependencies."""

    vector_service = VectorServiceStub()
    cache_manager = Mock()
    cache_manager.get = AsyncMock(return_value=None)
    cache_manager.set = AsyncMock()

    crawl_payload = {
        "success": True,
        "content": "Example document content",
        "title": "Example Title",
        "metadata": {"title": "Example Title"},
        "url": "https://example.com/doc",
        "tier_used": "tier-1",
        "quality_score": 0.78,
    }
    crawl_manager = Mock()
    crawl_manager.scrape_url = AsyncMock(return_value=crawl_payload)

    enriched_content = _make_enriched_content()
    content_intelligence = Mock()
    content_intelligence.analyze_content = AsyncMock(
        return_value=SimpleNamespace(success=True, enriched_content=enriched_content)
    )

    validator = DummyValidator()
    monkeypatch.setattr(
        documents.SecurityValidator,
        "from_unified_config",
        classmethod(lambda cls: validator),
    )
    monkeypatch.setattr(documents, "DocumentChunker", DummyChunker)

    cache_dependency = AsyncMock(return_value=cache_manager)
    crawl_dependency = AsyncMock(return_value=crawl_manager)
    content_dependency = AsyncMock(return_value=content_intelligence)

    monkeypatch.setattr("src.services.dependencies.get_cache_manager", cache_dependency)
    monkeypatch.setattr("src.services.dependencies.get_crawl_manager", crawl_dependency)
    monkeypatch.setattr(
        "src.services.dependencies.get_content_intelligence_service",
        content_dependency,
    )

    client_manager = Mock()
    client_manager.get_vector_store_service = AsyncMock(return_value=vector_service)

    mock_mcp = MagicMock()
    registered: dict[str, Callable] = {}

    def capture(func):
        registered[func.__name__] = func
        return func

    mock_mcp.tool.return_value = capture
    documents.register_tools(mock_mcp, client_manager)

    ctx = Mock()
    ctx.info = AsyncMock()
    ctx.debug = AsyncMock()
    ctx.warning = AsyncMock()
    ctx.error = AsyncMock()

    return SimpleNamespace(
        tools=registered,
        vector_service=vector_service,
        cache_manager=cache_manager,
        crawl_manager=crawl_manager,
        content_intelligence=content_intelligence,
        cache_dependency=cache_dependency,
        crawl_dependency=crawl_dependency,
        content_dependency=content_dependency,
        client_manager=client_manager,
        context=ctx,
        validator=validator,
    )


@pytest.mark.asyncio
async def test_add_document_ingests_chunks(documents_env: SimpleNamespace) -> None:
    request = DocumentRequest(url="https://example.com/doc")
    result = await documents_env.tools["add_document"](request, documents_env.context)

    assert isinstance(result, AddDocumentResponse)
    assert result.chunks_created == 2
    assert (
        result.embedding_dimensions == documents_env.vector_service.embedding_dimension
    )

    documents_env.vector_service.ensure_collection.assert_awaited_once()
    schema_arg = documents_env.vector_service.ensure_collection.await_args.args[0]
    assert isinstance(schema_arg, CollectionSchema)
    assert schema_arg.requires_sparse is True

    documents_env.vector_service.upsert_documents.assert_awaited_once()
    upsert_args = documents_env.vector_service.upsert_documents.await_args.args
    assert upsert_args[0] == request.collection
    documents_payload = upsert_args[1]
    assert len(documents_payload) == 2
    assert all(isinstance(doc, TextDocument) for doc in documents_payload)
    first_metadata = dict(documents_payload[0].metadata or {})
    assert first_metadata["chunk_index"] == 0
    assert first_metadata["content_intelligence_analyzed"] is True
    assert first_metadata["content_type"] == "guide"

    documents_env.cache_manager.set.assert_awaited_once()
    documents_env.context.info.assert_awaited()


@pytest.mark.asyncio
async def test_add_document_returns_cached_result(
    documents_env: SimpleNamespace,
) -> None:
    cached_response = AddDocumentResponse(
        url="https://example.com/doc",
        title="Cached Title",
        chunks_created=1,
        collection="documentation",
        chunking_strategy="enhanced",
        embedding_dimensions=1536,
    )
    documents_env.cache_manager.get.return_value = cached_response.model_dump()

    request = DocumentRequest(url="https://example.com/doc")
    result = await documents_env.tools["add_document"](request, documents_env.context)

    assert result == cached_response
    documents_env.vector_service.ensure_collection.assert_not_called()
    documents_env.vector_service.upsert_documents.assert_not_called()


@pytest.mark.asyncio
async def test_add_document_without_content_intelligence(
    documents_env: SimpleNamespace,
) -> None:
    documents_env.content_dependency.return_value = None

    request = DocumentRequest(url="https://example.com/doc")
    result = await documents_env.tools["add_document"](request, documents_env.context)

    assert isinstance(result, AddDocumentResponse)
    upsert_args = documents_env.vector_service.upsert_documents.await_args.args
    text_documents = upsert_args[1]
    metadata_flags = [
        bool(doc.metadata.get("content_intelligence_analyzed"))
        for doc in text_documents
    ]
    assert metadata_flags == [False, False]
    for index, document in enumerate(text_documents):
        metadata = document.metadata or {}
        assert metadata["doc_id"]
        assert metadata["chunk_id"] == index
        assert metadata["tenant"] == request.collection
        assert metadata["source"] == request.url


@pytest.mark.asyncio
async def test_add_documents_batch_captures_failures(
    documents_env: SimpleNamespace,
) -> None:
    success_payload = documents_env.crawl_manager.scrape_url.return_value
    documents_env.crawl_manager.scrape_url.side_effect = [
        success_payload,
        ConnectionError("unreachable"),
    ]

    request = BatchRequest(
        urls=["https://example.com/a", "https://example.com/b"],
        collection="documentation",
    )
    response = await documents_env.tools["add_documents_batch"](
        request,
        documents_env.context,
    )

    assert len(response.successful) == 1
    assert response.failed == ["https://example.com/b"]
    assert response.total == 2

    # Vector service should only be invoked for the successful document
    documents_env.vector_service.upsert_documents.assert_awaited_once()
