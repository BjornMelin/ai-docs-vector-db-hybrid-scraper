"""Unit tests for MCP document ingestion tools backed by VectorStoreService."""

from __future__ import annotations

from collections.abc import Callable
from types import SimpleNamespace
from typing import Any, cast
from unittest.mock import AsyncMock, MagicMock, Mock

import pytest
from langchain_core.documents import Document

from src.mcp_tools.models.requests import BatchRequest, DocumentRequest
from src.mcp_tools.models.responses import AddDocumentResponse
from src.mcp_tools.tools import documents
from src.services.vector_db.types import CollectionSchema, TextDocument


class DummyContentType:
    """Minimal content type record matching the enrichment interface."""

    def __init__(self, value: str):
        """Initialize content type with string value."""
        self.value = value

    def __hash__(self) -> int:  # pragma: no cover - hashing required for dict keys
        """Return hash of value for dict key usage."""
        return hash(self.value)

    def __eq__(self, other: object) -> bool:  # pragma: no cover - defensive
        """Compare equality based on value."""
        if not isinstance(other, DummyContentType):
            return NotImplemented
        return self.value == other.value


class VectorServiceStub:
    """Async stub for VectorStoreService interactions used in tests."""

    def __init__(self) -> None:
        """Initialize stub with mock methods."""
        self._initialized = False
        self.ensure_collection = AsyncMock()
        self.upsert_documents = AsyncMock()

    def is_initialized(self) -> bool:
        """Return initialization state."""
        return self._initialized

    async def initialize(self) -> None:
        """Mark service as initialized."""
        self._initialized = True

    @property
    def embedding_dimension(self) -> int:
        """Return fixed embedding dimension."""
        return 1536


class DummyValidator:
    """URL validator stub returning the original URL."""

    def validate_url(self, url: str) -> str:
        """Return URL without validation for testing."""
        return url


def _dummy_splitter(*_: Any, **__: Any) -> list[Document]:
    """Return deterministic LangChain documents with canonical metadata."""
    documents: list[Document] = []
    for index, section in enumerate(("intro", "body")):
        chunk_hash = f"feedbeefdead{index:02d}"
        documents.append(
            Document(
                page_content=f"chunk-{index}",
                metadata={
                    "section": section,
                    "chunk_index": index,
                    "chunk_id": chunk_hash,
                    "kind": "markdown",
                    "source": "https://example.com/doc",
                    "uri_or_path": "https://example.com/doc",
                    "title": "Example Title",
                    "provider": "tier-1",
                    "mime_type": "text/markdown",
                },
            )
        )
    return documents


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
def documents_env(monkeypatch) -> SimpleNamespace:  # pylint: disable=too-many-locals
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
        "provider": "tier-1",
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
        documents.MLSecurityValidator,
        "from_unified_config",
        classmethod(lambda cls: validator),
    )
    monkeypatch.setattr(
        "src.services.document_chunking.chunk_to_documents",
        lambda *args, **kwargs: _dummy_splitter(*args, **kwargs),
    )
    monkeypatch.setattr(documents, "chunk_to_documents", _dummy_splitter)

    mock_mcp = MagicMock()
    registered: dict[str, Callable] = {}

    def capture(func):
        registered[func.__name__] = func
        return func

    mock_mcp.tool.return_value = capture
    documents.register_tools(
        mock_mcp,
        vector_service=cast(Any, vector_service),
        cache_manager=cast(Any, cache_manager),
        crawl_manager=cast(Any, crawl_manager),
        content_intelligence_service=cast(Any, content_intelligence),
    )

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
        context=ctx,
        validator=validator,
    )


@pytest.mark.asyncio
async def test_add_document_ingests_chunks(documents_env: SimpleNamespace) -> None:
    """Verify add_document crawls URL, chunks content, and upserts with metadata."""
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
    first_document = documents_payload[0]
    first_metadata = dict(first_document.metadata or {})
    assert first_document.id.endswith(":0")
    assert first_metadata["chunk_index"] == 0
    assert first_metadata["chunk_id"] == 0
    assert first_metadata["chunk_hash"] == "feedbeefdead00"
    assert first_metadata["total_chunks"] == 2
    assert first_metadata["tenant"] == request.collection
    assert first_metadata["source"] == request.url
    assert first_metadata["uri_or_path"] == request.url
    assert first_metadata["title"] == "Example Title"
    assert first_metadata["provider"] == "tier-1"
    assert first_metadata["content_type"] == "guide"
    assert first_metadata["content_confidence"] == pytest.approx(0.92)
    assert first_metadata["quality_overall"] == pytest.approx(0.87)
    assert first_metadata["quality_completeness"] == pytest.approx(0.82)
    assert first_metadata["quality_relevance"] == pytest.approx(0.91)
    assert first_metadata["quality_confidence"] == pytest.approx(0.85)
    assert first_metadata["ci_word_count"] == 1200
    assert first_metadata["ci_char_count"] == 7800
    assert first_metadata["ci_language"] == "en"
    assert first_metadata["ci_semantic_tags"] == ["documentation"]
    assert first_metadata["secondary_content_types"] == ["cheatsheet"]
    assert first_metadata["content_intelligence_analyzed"] is True
    assert isinstance(first_metadata["created_at"], str)
    assert first_metadata["updated_at"] == first_metadata["created_at"]
    assert first_metadata["doc_id"]
    assert first_metadata["section"] == "intro"
    assert "lang" not in first_metadata or first_metadata["lang"] is None

    documents_env.cache_manager.set.assert_awaited_once()
    cache_args = documents_env.cache_manager.set.await_args.kwargs
    assert cache_args == {"ttl": 86400}
    cache_key, cache_payload = documents_env.cache_manager.set.await_args.args
    assert cache_key == "doc:https://example.com/doc"
    assert cache_payload["url"] == request.url
    documents_env.context.info.assert_awaited()


@pytest.mark.asyncio
async def test_add_document_returns_cached_result(
    documents_env: SimpleNamespace,
) -> None:
    """Verify add_document skips processing when cached result exists."""
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
    documents_env.cache_manager.get.assert_awaited_once_with(
        "doc:https://example.com/doc"
    )


@pytest.mark.asyncio
async def test_add_document_without_content_intelligence(
    documents_env: SimpleNamespace,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Verify document ingestion succeeds without content intelligence enrichment."""
    monkeypatch.setattr(
        documents,
        "_run_content_intelligence",
        AsyncMock(return_value=None),
    )

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
        assert metadata["uri_or_path"] == request.url
        assert metadata["chunk_index"] == index
        assert metadata["total_chunks"] == len(text_documents)
        assert isinstance(metadata["created_at"], str)


@pytest.mark.asyncio
async def test_add_documents_batch_captures_failures(
    documents_env: SimpleNamespace,
) -> None:
    """Verify batch ingestion records partial failures without aborting."""
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
