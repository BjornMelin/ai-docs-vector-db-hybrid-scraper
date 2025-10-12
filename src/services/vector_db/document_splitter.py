"""Utilities for splitting content into LangChain documents."""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter

from src.config.models import ChunkingConfig


def _looks_like_json(payload: str) -> bool:
    """Return ``True`` when the payload resembles JSON."""

    stripped = payload.lstrip()
    return stripped.startswith("{") or stripped.startswith("[")


def _segment_payload(content: str, config: ChunkingConfig) -> list[str]:
    """Create preprocessing windows prior to recursive splitting."""

    if _looks_like_json(content) and len(content) > config.json_max_chars:
        window = config.json_max_chars
        return [content[i : i + window] for i in range(0, len(content), window)]
    return [content]


def split_content_into_documents(
    content: str,
    config: ChunkingConfig,
    *,
    metadata: Mapping[str, Any] | None = None,
) -> list[Document]:
    """Split content into :class:`langchain_core.documents.Document` objects.

    Args:
        content: Raw text to split.
        config: Chunking configuration containing size and overlap settings.
        metadata: Optional metadata applied to each generated document.

    Returns:
        List of LangChain ``Document`` instances preserving base metadata.
    """

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=config.chunk_size,
        chunk_overlap=config.chunk_overlap,
        separators=["\n\n", "\n", " ", ""],
        add_start_index=True,
    )

    base_metadata = dict(metadata or {})
    segments = _segment_payload(content, config)
    documents = splitter.create_documents(
        segments, metadatas=[base_metadata for _ in segments]
    )
    return documents


__all__ = ["split_content_into_documents"]
