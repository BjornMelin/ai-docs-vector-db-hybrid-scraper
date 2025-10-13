"""Unit tests for document chunking utilities."""

from __future__ import annotations

import json

import pytest
from langchain_core.documents import Document

from src.config.models import ChunkingConfig
from src.services.document_chunking import (
    chunk_to_documents,
    infer_document_kind,
    infer_extension,
    infer_language,
)


def _make_config(**overrides: int | bool | str) -> ChunkingConfig:
    base_kwargs: dict[str, int | bool | str] = {
        "chunk_size": 200,
        "chunk_overlap": 20,
        "token_chunk_size": 40,
        "token_chunk_overlap": 5,
        "json_max_chars": 400,
    }
    base_kwargs.update(overrides)
    return ChunkingConfig(**base_kwargs)  # type: ignore[arg-type]


def test_markdown_splitter_preserves_header_hierarchy() -> None:
    config = _make_config()
    markdown = (
        "# Title\n\n"
        "## Section\n\n"
        "Paragraph one.\n\n"
        "Another sentence in the same section."
    )

    documents = chunk_to_documents(
        markdown,
        {"uri_or_path": "guide.md"},
        "markdown",
        config,
    )

    assert len(documents) >= 1
    assert all(isinstance(doc, Document) for doc in documents)
    assert documents[0].metadata["kind"] == "markdown"
    headers = {
        key: value
        for key, value in documents[0].metadata.items()
        if key.startswith("Header")
    }
    assert headers == {"Header 1": "Title", "Header 2": "Section"}


def test_html_semantic_splitter_respects_sections() -> None:
    config = _make_config(enable_semantic_html_segmentation=True)
    html = (
        "<html><body><h1>Title</h1><p>Intro paragraph.</p>"
        "<h2>Details</h2><p>Second paragraph.</p></body></html>"
    )

    documents = chunk_to_documents(
        html,
        {"uri_or_path": "page.html", "mime_type": "text/html"},
        "html",
        config,
    )

    assert len(documents) >= 2
    first, second = documents[:2]
    assert first.metadata["chunk_index"] == 0
    assert second.metadata["chunk_index"] == 1
    assert "Intro" in first.page_content.lower()
    assert "second" in second.page_content.lower()


def test_code_language_inferred_from_extension() -> None:
    config = _make_config()
    code = """def add(a: int, b: int) -> int:\n    return a + b\n"""

    documents = chunk_to_documents(
        code,
        {"uri_or_path": "app/main.py"},
        "code",
        config,
    )

    assert len(documents) >= 1
    metadata = documents[0].metadata
    assert metadata["kind"] == "code"
    assert metadata["language"] == "python"


def test_json_splitter_creates_structured_chunks() -> None:
    config = _make_config(
        chunk_size=60,
        chunk_overlap=10,
        min_chunk_size=20,
        max_chunk_size=120,
        json_max_chars=60,
    )
    payload = json.dumps(
        {
            "users": [
                {"id": 1, "name": "Ada"},
                {"id": 2, "name": "Grace"},
                {"id": 3, "name": "Edsger"},
            ],
            "meta": {"count": 3},
        }
    )

    documents = chunk_to_documents(
        payload,
        {"uri_or_path": "data.json"},
        "json",
        config,
    )

    assert len(documents) >= 2
    assert all(doc.metadata["kind"] == "json" for doc in documents)
    assert any("Ada" in doc.page_content for doc in documents)


def test_token_splitter_respects_token_configuration() -> None:
    config = _make_config(token_chunk_size=10, token_chunk_overlap=0)
    text = " ".join(f"token{i}" for i in range(40))

    documents = chunk_to_documents(
        text,
        {"source": "https://example.com"},
        "token",
        config,
    )

    assert len(documents) > 1
    assert documents[0].metadata["kind"] == "token"
    assert documents[0].metadata["chunk_index"] == 0
    assert documents[1].metadata["chunk_index"] == 1


@pytest.mark.parametrize(
    "uri, expected",
    [
        ("https://example.com/file.md", "md"),
        ("/var/data/archive.JSON?query=1", "json"),
        (None, None),
    ],
)
def test_infer_extension_handles_paths(uri: str | None, expected: str | None) -> None:
    assert infer_extension(uri) == expected


def test_infer_document_kind_uses_metadata_hierarchy() -> None:
    metadata = {
        "mime_type": "text/markdown",
        "uri_or_path": "./guide.md",
        "metadata": {"content_type": "text/markdown"},
    }
    assert infer_document_kind(metadata) == "markdown"


def test_infer_language_falls_back_to_extension() -> None:
    assert infer_language("py", {}) == "python"
