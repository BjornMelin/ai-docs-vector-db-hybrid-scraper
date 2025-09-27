"""Deterministic coverage for the document chunking pipeline."""

from __future__ import annotations

from typing import Any
from unittest.mock import patch

import pytest

from src.chunking import DocumentChunker
from src.config import ChunkingConfig, ChunkingStrategy


@pytest.fixture()
def small_chunker() -> DocumentChunker:
    """Build a chunker with tiny chunk size to force boundary splits."""
    return DocumentChunker(
        ChunkingConfig(chunk_size=50, chunk_overlap=10, min_chunk_size=10)
    )


def _extract_lengths(chunks: list[dict[str, Any]]) -> list[int]:
    """Return the character lengths for each chunk payload."""
    return [len(chunk["content"]) for chunk in chunks]


def test_chunk_content_produces_structured_records(
    small_chunker: DocumentChunker,
) -> None:
    """The chunker returns dictionaries with semantic metadata."""
    content = """\
    ## Heading
    Body line one.

    ```python
    def example() -> None:
        return None
    ```
    """.strip()

    chunks = small_chunker.chunk_content(content, title="Doc", url="https://example")

    assert all(isinstance(chunk, dict) for chunk in chunks)
    assert {"content", "chunk_index", "title", "url"} <= chunks[0].keys()
    assert chunks[0]["title"] == "Doc"
    assert chunks[0]["url"] == "https://example"


def test_chunk_content_respects_window_and_overlap(
    small_chunker: DocumentChunker,
) -> None:
    """Generated chunks respect the configured size and overlap windows."""
    content = " ".join(str(index) for index in range(200))
    chunks = small_chunker.chunk_content(content)

    # Force multiple segments and verify chunk indices are monotonic.
    assert len(chunks) > 1
    assert [chunk["chunk_index"] for chunk in chunks] == list(range(len(chunks)))

    lengths = _extract_lengths(chunks)
    assert max(lengths) <= small_chunker.config.chunk_size + 5

    overlaps = [
        len(set(chunks[i]["content"].split()) & set(chunks[i + 1]["content"].split()))
        for i in range(len(chunks) - 1)
    ]
    assert all(overlap > 0 for overlap in overlaps)


def test_ast_strategy_gracefully_falls_back() -> None:
    """AST chunking falls back to semantic mode when parsers are unavailable."""
    config = ChunkingConfig(
        strategy=ChunkingStrategy.AST_AWARE,
        enable_ast_chunking=True,
        supported_languages=["python"],
    )

    with (
        patch("src.chunking.TREE_SITTER_AVAILABLE", False),
        patch("src.chunking.Parser", None),
        patch("src.chunking.Language", None),
    ):
        chunker = DocumentChunker(config)

    result = chunker.chunk_content(
        "def fallback() -> None:\n    return None", url="test.py"
    )
    assert result
    assert all(isinstance(chunk, dict) for chunk in result)
