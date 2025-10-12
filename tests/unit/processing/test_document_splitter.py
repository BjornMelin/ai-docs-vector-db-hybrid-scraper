"""Tests for LangChain document splitting utilities."""

from collections.abc import Sequence

from langchain_core.documents import Document

from src.config.models import ChunkingConfig
from src.services.vector_db.document_splitter import split_content_into_documents


def _extract_overlap(documents: Sequence[Document]) -> list[int]:
    """Compute token overlap using simple whitespace heuristics."""

    overlaps: list[int] = []
    for first, second in zip(documents, documents[1:], strict=False):
        overlaps.append(
            len(set(first.page_content.split()) & set(second.page_content.split()))
        )
    return overlaps


def test_splitter_returns_documents_with_metadata() -> None:
    """The splitter should return LangChain Document instances."""

    config = ChunkingConfig(chunk_size=80, chunk_overlap=10)
    documents = split_content_into_documents(
        "Heading\n\nParagraph body content for testing.",
        config,
        metadata={"source": "https://example.com"},
    )

    assert documents
    assert all(isinstance(doc, Document) for doc in documents)
    assert documents[0].metadata.get("source") == "https://example.com"
    assert "start_index" in documents[0].metadata


def test_splitter_respects_overlap_configuration() -> None:
    """Chunk overlap configuration should produce shared tokens between chunks."""

    content = " ".join(str(i) for i in range(500))
    config = ChunkingConfig(chunk_size=120, chunk_overlap=40)
    documents = split_content_into_documents(content, config)

    assert len(documents) > 1
    overlaps = _extract_overlap(documents)
    assert all(value > 0 for value in overlaps)
