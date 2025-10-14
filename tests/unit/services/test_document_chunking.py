"""Unit tests for document chunking utilities."""

from __future__ import annotations

import json
from collections.abc import Mapping
from typing import Any

import pytest
from langchain_core._api.beta_decorator import LangChainBetaWarning
from langchain_core.documents import Document
from langchain_text_splitters import Language

from src.config.models import ChunkingConfig
from src.services.document_chunking import (
    CodeChunkingStrategy,
    JsonChunkingStrategy,
    PlainTextChunkingStrategy,
    TokenAwareChunkingStrategy,
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
        "min_chunk_size": 20,
        "max_chunk_size": 800,
    }
    base_kwargs.update(overrides)
    return ChunkingConfig(**base_kwargs)  # type: ignore[arg-type]


def test_plain_text_strategy_chunks_multiple_sections() -> None:
    """Verify plain text splitter creates multiple chunks from paragraphs."""
    strategy = PlainTextChunkingStrategy()
    cfg = _make_config(chunk_size=20, chunk_overlap=0)
    raw = "First paragraph.\n\nSecond paragraph continues with more text."

    documents = strategy.chunk(raw, cfg, None)

    assert len(documents) >= 2
    assert all("start_index" in doc.metadata for doc in documents)


def test_token_strategy_chunks_using_token_window() -> None:
    """Verify token-aware strategy respects token window configuration."""
    strategy = TokenAwareChunkingStrategy()
    cfg = _make_config(token_chunk_size=5, token_chunk_overlap=0)
    raw = " ".join(f"token{i}" for i in range(12))

    documents = strategy.chunk(raw, cfg, None)

    assert len(documents) >= 2
    assert all(doc.metadata == {} for doc in documents)


def test_code_strategy_respects_language_hint(monkeypatch: pytest.MonkeyPatch) -> None:
    """Verify code strategy uses language metadata to configure splitter."""
    strategy = CodeChunkingStrategy()
    cfg = _make_config()
    captured: dict[str, Language] = {}

    def fake_from_language(
        *,
        language: Language,
        chunk_size: int,
        chunk_overlap: int,
        add_start_index: bool,
    ) -> Any:
        captured.update({"language": language})

        class _StubSplitter:
            """Stub splitter returning a single document for testing."""

            def create_documents(
                self,
                texts: list[str],
                metadatas: list[Mapping[str, Any]],
            ) -> list[Document]:
                return [Document(page_content=texts[0], metadata=metadatas[0])]

        _ = chunk_size
        _ = chunk_overlap
        _ = add_start_index
        return _StubSplitter()

    monkeypatch.setattr(
        "src.services.document_chunking.RecursiveCharacterTextSplitter.from_language",
        fake_from_language,
    )

    documents = strategy.chunk("print('hello world')", cfg, {"language": "python"})

    assert captured["language"] == Language.PYTHON
    assert len(documents) == 1


def test_json_strategy_raises_on_invalid_payload() -> None:
    """Verify JSON strategy raises error on malformed input."""
    strategy = JsonChunkingStrategy()
    cfg = _make_config()

    with pytest.raises(ValueError):
        strategy.chunk("not-json", cfg, None)


def test_markdown_splitter_preserves_header_hierarchy() -> None:
    """Verify markdown splitter retains header metadata in chunks."""
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
    first_metadata = documents[0].metadata
    assert first_metadata["kind"] == "markdown"
    assert isinstance(first_metadata["chunk_id"], str)
    assert len(first_metadata["chunk_id"]) == 16
    headers = {
        key: value
        for key, value in documents[0].metadata.items()
        if key.startswith("Header")
    }
    assert headers == {"Header 1": "Title", "Header 2": "Section"}


def test_html_semantic_splitter_respects_sections() -> None:
    """Verify HTML splitter uses semantic boundaries for chunking."""
    config = _make_config(enable_semantic_html_segmentation=True)
    html = (
        "<html><body><h1>Title</h1><p>Intro paragraph.</p>"
        "<h2>Details</h2><p>Second paragraph.</p></body></html>"
    )

    with pytest.warns(LangChainBetaWarning):
        documents = chunk_to_documents(
            html,
            {"uri_or_path": "page.html", "mime_type": "text/html"},
            "html",
            config,
        )

    assert len(documents) >= 2
    first_document = documents[0]
    second_document = documents[1]
    assert first_document.metadata["chunk_index"] == 0
    assert isinstance(first_document.metadata["chunk_id"], str)
    assert len(first_document.metadata["chunk_id"]) == 16
    assert second_document.metadata["chunk_index"] == 1
    assert "intro" in first_document.page_content.lower()
    assert "second" in second_document.page_content.lower()


def test_code_language_inferred_from_extension() -> None:
    """Verify code language detected from file extension."""
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
    assert isinstance(metadata["chunk_id"], str)
    assert len(metadata["chunk_id"]) == 16


def test_json_splitter_creates_structured_chunks() -> None:
    """Verify JSON splitter respects max_chars configuration."""
    config = _make_config(
        chunk_size=60,
        chunk_overlap=10,
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
    assert all(len(doc.metadata["chunk_id"]) == 16 for doc in documents)
    assert any("Ada" in doc.page_content for doc in documents)


def test_token_splitter_respects_token_configuration() -> None:
    """Verify token splitter honors chunk size and overlap settings."""
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


def test_metadata_token_hint_triggers_token_splitter() -> None:
    """Verify token_aware metadata flag activates token splitter."""
    config = _make_config(token_chunk_size=12, token_chunk_overlap=2)
    text = "This document should be split using token aware logic." * 2

    documents = chunk_to_documents(
        text,
        {"chunking_mode": "token", "uri_or_path": "https://example.com/tokens"},
        "auto",
        config,
    )

    assert len(documents) >= 2
    kinds = {doc.metadata["kind"] for doc in documents}
    assert kinds == {"token"}
    assert all(
        doc.metadata["chunk_index"] == index for index, doc in enumerate(documents)
    )


@pytest.mark.parametrize(
    "uri, expected",
    [
        ("https://example.com/file.md", "md"),
        ("/var/data/archive.JSON?query=1", "json"),
        (None, None),
    ],
)
def test_infer_extension_handles_paths(uri: str | None, expected: str | None) -> None:
    """Verify extension inference handles URLs, paths, and None."""
    assert infer_extension(uri) == expected


def test_chunk_to_documents_fallbacks_to_text_kind() -> None:
    """Verify unsupported document kind falls back to text chunking."""
    config = _make_config()

    documents = chunk_to_documents(
        "fallback sample",
        {"uri_or_path": "file.unknown"},
        "pdf",  # unsupported kind should fallback
        config,
    )

    assert documents
    assert all(doc.metadata["kind"] == "text" for doc in documents)


def test_infer_document_kind_uses_metadata_hierarchy() -> None:
    """Verify document kind inferred from mime_type and uri_or_path."""
    metadata = {
        "mime_type": "text/markdown",
        "uri_or_path": "./guide.md",
        "metadata": {"content_type": "text/markdown"},
    }
    assert infer_document_kind(metadata) == "markdown"


def test_infer_document_kind_detects_token_flag() -> None:
    """Verify token_aware boolean flag overrides default kind."""
    metadata = {"token_aware": True}
    assert infer_document_kind(metadata) == "token"


def test_infer_document_kind_handles_token_alias_string() -> None:
    """Verify token_aware string alias triggers token kind."""
    metadata = {"token_aware": "Token_Aware"}
    assert infer_document_kind(metadata) == "token"


def test_infer_language_falls_back_to_extension() -> None:
    """Verify language mapping uses file extension when metadata absent."""
    assert infer_language("py", {}) == "python"
