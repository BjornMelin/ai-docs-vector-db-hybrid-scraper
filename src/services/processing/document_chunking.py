"""LangChain-powered document chunking utilities."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Any

from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter

from src.config.models import ChunkingConfig, ChunkingStrategy
from src.models.validators import validate_chunk_parameters


# Default separators prioritise paragraph and sentence boundaries.
_BASIC_SEPARATORS: Sequence[str] = ("\n\n", "\n", " ", "")
# Enhanced separators include Markdown headings and code fences for smoother context.
_ENHANCED_SEPARATORS: Sequence[str] = (
    "\n```",
    "\n~~~",
    "\n### ",
    "\n## ",
    "\n# ",
    "\n\n",
    "\n",
    " ",
    "",
)


def _separators_for_strategy(strategy: ChunkingStrategy) -> Sequence[str]:
    """Return separators tuned for the requested chunking strategy."""

    if strategy is ChunkingStrategy.ENHANCED:
        return _ENHANCED_SEPARATORS
    return _BASIC_SEPARATORS


def build_text_splitter(config: ChunkingConfig) -> RecursiveCharacterTextSplitter:
    """Instantiate a text splitter configured for the current settings."""

    validate_chunk_parameters(config.chunk_size, config.chunk_overlap)
    return RecursiveCharacterTextSplitter(
        chunk_size=config.chunk_size,
        chunk_overlap=config.chunk_overlap,
        separators=list(_separators_for_strategy(config.strategy)),
        add_start_index=True,
    )


def chunk_content(
    content: str,
    *,
    config: ChunkingConfig,
    metadata: Mapping[str, Any] | None = None,
) -> list[Document]:
    """Split raw content into LangChain ``Document`` chunks."""

    base_metadata: dict[str, Any] = {}
    if metadata:
        base_metadata.update(dict(metadata))

    document = Document(page_content=content, metadata=base_metadata)
    splitter = build_text_splitter(config)
    return splitter.split_documents([document])


__all__ = [
    "build_text_splitter",
    "chunk_content",
]
