"""Deterministic coverage for the document chunking pipeline."""

from __future__ import annotations

from typing import Any

import pytest

from src.chunking import DocumentChunker
from src.config.models import ChunkingConfig


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


def test_html_segmentation_controls() -> None:
    """HTML segmentation can be toggled via configuration."""
    html = """
    <html><body>
        <section><h1>Title</h1><p>Paragraph A</p></section>
        <section><h2>Sub</h2><p>Paragraph B</p></section>
    </body></html>
    """
    config = ChunkingConfig(
        chunk_size=200,
        chunk_overlap=20,
        enable_semantic_html_segmentation=True,
    )
    chunker = DocumentChunker(config)
    segmented = chunker.chunk_content(html)

    assert segmented
    assert segmented[0]["content"].strip().startswith("Title")

    config_no_html = ChunkingConfig(
        chunk_size=200,
        chunk_overlap=20,
        enable_semantic_html_segmentation=False,
    )
    chunker_no_html = DocumentChunker(config_no_html)
    raw_segments = chunker_no_html.chunk_content(html)

    assert raw_segments
    assert len(segmented) <= len(raw_segments)
