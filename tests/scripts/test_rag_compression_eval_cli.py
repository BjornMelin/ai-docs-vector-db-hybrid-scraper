"""Tests for the RAG compression evaluation CLI script."""

from __future__ import annotations

from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest
from langchain_core.documents import Document

from scripts.eval.rag_compression_eval import (
    _build_documents,
    _estimate_tokens,
    _evaluate,
    _load_vector_service,
)


class _StubVectorService:
    """Stub vector store service for testing."""

    def __init__(self) -> None:
        self.collection_name: str | None = None
        self.config = MagicMock()  # Add config attribute

    async def cleanup(self) -> None:
        pass

    def is_initialized(self) -> bool:
        return True

    async def initialize(self) -> None:
        pass


def _install_vector_service_stub(monkeypatch: pytest.MonkeyPatch) -> None:
    """Install stub implementations for vector service dependencies."""

    async def _fake_load_vector_service(
        collection_override: str | None,
    ) -> _StubVectorService:
        service = _StubVectorService()
        if collection_override:
            service.collection_name = collection_override
        return service

    monkeypatch.setattr(
        "scripts.eval.rag_compression_eval._load_vector_service",
        _fake_load_vector_service,
    )


@pytest.mark.asyncio
async def test_load_vector_service_no_override() -> None:
    """Test loading vector service without collection override."""

    with (
        patch("scripts.eval.rag_compression_eval.get_settings") as mock_settings,
        patch("scripts.eval.rag_compression_eval.get_container") as mock_get_container,
        patch(
            "scripts.eval.rag_compression_eval.initialize_container"
        ) as mock_init_container,
    ):
        # Setup mocks
        mock_config = MagicMock()
        mock_settings.return_value = mock_config
        mock_container = MagicMock()
        mock_get_container.return_value = None
        mock_init_container.return_value = mock_container

        mock_service = MagicMock()
        mock_service.collection_name = None
        mock_service.is_initialized.return_value = True
        mock_container.vector_store_service.return_value = mock_service

        # Test
        result = await _load_vector_service(None)

        assert result == mock_service
        mock_init_container.assert_called_once_with(mock_config)
        assert result.collection_name is None


@pytest.mark.asyncio
async def test_load_vector_service_with_override() -> None:
    """Test loading vector service with collection override."""

    with (
        patch("scripts.eval.rag_compression_eval.get_settings") as mock_settings,
        patch("scripts.eval.rag_compression_eval.get_container") as mock_get_container,
        patch(
            "scripts.eval.rag_compression_eval.initialize_container"
        ) as mock_init_container,
    ):
        # Setup mocks
        mock_config = MagicMock()
        mock_settings.return_value = mock_config
        mock_container = MagicMock()
        mock_get_container.return_value = None
        mock_init_container.return_value = mock_container

        mock_service = MagicMock()
        mock_service.collection_name = None
        mock_service.is_initialized.return_value = True
        mock_container.vector_store_service.return_value = mock_service

        # Test
        result = await _load_vector_service("test_collection")

        assert result == mock_service
        assert result.collection_name == "test_collection"


def test_build_documents() -> None:
    """Test building Document objects from raw data."""

    raw_documents = [
        {"content": "Test content 1", "metadata": {"key": "value1"}},
        {"content": "Test content 2", "metadata": None},
        {"text": "Test content 3"},  # Alternative content key
    ]

    documents = _build_documents(raw_documents)

    assert len(documents) == 3
    assert isinstance(documents[0], Document)
    assert documents[0].page_content == "Test content 1"
    assert documents[0].metadata == {"key": "value1"}
    assert documents[1].page_content == "Test content 2"
    assert documents[1].metadata == {}
    assert documents[2].page_content == "Test content 3"
    assert documents[2].metadata == {}


def test_estimate_tokens() -> None:
    """Test token estimation function."""

    assert _estimate_tokens("hello world") == 2
    assert _estimate_tokens("") == 1  # minimum
    assert _estimate_tokens("a b c d e f g") == 7


@pytest.mark.asyncio
async def test_evaluate_compression_disabled(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Test evaluation when compression is disabled."""

    _install_vector_service_stub(monkeypatch)

    # Mock settings with compression disabled
    mock_config = MagicMock()
    mock_config.rag.compression_enabled = False

    with patch(
        "scripts.eval.rag_compression_eval.get_settings", return_value=mock_config
    ):
        dataset_path = tmp_path / "empty.json"
        dataset_path.write_text("[]", encoding="utf-8")

        # Should not raise and should print message
        with patch("builtins.print") as mock_print:
            await _evaluate(dataset_path, None)

        mock_print.assert_called_with(
            "Compression is disabled in the active configuration; nothing to evaluate."
        )
