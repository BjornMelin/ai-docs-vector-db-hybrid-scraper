"""Integration tests for RAG compression evaluation script."""

from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from scripts.eval.rag_compression_eval import (
    _build_documents,
    _estimate_tokens,
    _evaluate,
)


@pytest.fixture
def compression_eval_dataset(tmp_path: Path) -> Path:
    """Create a realistic compression evaluation dataset."""

    dataset = tmp_path / "compression_eval.json"
    data = [
        {
            "query": "What is LangChain?",
            "documents": [
                {
                    "content": (
                        "LangChain is a framework for building applications "
                        "with large language models."
                    ),
                    "metadata": {"source": "docs/langchain.md", "chunk_id": 1},
                },
                {
                    "content": (
                        "LangChain provides components for "
                        "retrieval-augmented generation and "
                        "agent orchestration."
                    ),
                    "metadata": {"source": "docs/langchain.md", "chunk_id": 2},
                },
                {
                    "content": (
                        "LangChain is a framework for building applications "
                        "with large language models."
                    ),
                    "metadata": {"source": "docs/langchain.md", "chunk_id": 3},
                },  # duplicate content
                {
                    "content": (
                        "The framework supports multiple LLM providers including "
                        "OpenAI and Anthropic."
                    ),
                    "metadata": {"source": "docs/providers.md", "chunk_id": 1},
                },
            ],
            "relevant_phrases": [
                "framework for building applications",
                "large language models",
            ],
        },
        {
            "query": "How does vector search work?",
            "documents": [
                {
                    "content": (
                        "Vector search finds similar items by comparing their "
                        "vector representations in high-dimensional space."
                    ),
                    "metadata": {"source": "docs/vector_search.md", "chunk_id": 1},
                },
                {
                    "content": (
                        "Embeddings convert text into numerical vectors that "
                        "capture semantic meaning."
                    ),
                    "metadata": {"source": "docs/embeddings.md", "chunk_id": 1},
                },
            ],
            "relevant_phrases": [
                "comparing their vector representations",
                "high-dimensional space",
            ],
        },
    ]
    dataset.write_text(json.dumps(data, indent=2), encoding="utf-8")
    return dataset


@pytest.mark.integration
def test_cli_script_execution(compression_eval_dataset: Path, tmp_path: Path) -> None:
    """Test that the CLI script can be executed without errors."""

    # This test verifies the script can be invoked; may fail due to missing services
    # The important thing is that the script loads and parses arguments correctly

    script_path = Path("scripts/eval/rag_compression_eval.py")

    # Test argument parsing by running with --help
    result = subprocess.run(  # noqa: S603
        [sys.executable, str(script_path), "--help"],
        capture_output=True,
        text=True,
        cwd=Path.cwd(),
        check=False,
    )

    # Should exit with code 0 for --help
    assert result.returncode == 0
    assert (
        "Evaluate the LangChain-based contextual compression pipeline" in result.stdout
    )
    assert "--input" in result.stdout
    assert "--collection" in result.stdout


@pytest.mark.integration
@pytest.mark.asyncio
async def test_compression_evaluation_with_empty_dataset(tmp_path: Path) -> None:
    """Test evaluation with an empty dataset."""

    empty_dataset = tmp_path / "empty.json"
    empty_dataset.write_text("[]", encoding="utf-8")

    mock_config = MagicMock()
    mock_config.rag.compression_enabled = True

    with (
        patch(
            "scripts.eval.rag_compression_eval.get_settings", return_value=mock_config
        ),
        patch(
            "scripts.eval.rag_compression_eval._load_vector_service"
        ) as mock_load_service,
        patch("builtins.print") as mock_print,
        patch(
            "scripts.eval.rag_compression_eval.FastEmbedEmbeddings"
        ) as mock_embeddings,
        patch(
            "scripts.eval.rag_compression_eval.EmbeddingsRedundantFilter"
        ) as mock_redundant_filter,
        patch(
            "scripts.eval.rag_compression_eval.EmbeddingsFilter"
        ) as mock_embeddings_filter,
        patch(
            "scripts.eval.rag_compression_eval.DocumentCompressorPipeline"
        ) as mock_compressor,
    ):
        mock_service = MagicMock()
        mock_service.cleanup = AsyncMock()
        mock_load_service.return_value = mock_service
        mock_embeddings.return_value = MagicMock()
        mock_redundant_filter.return_value = MagicMock()
        mock_embeddings_filter.return_value = MagicMock()
        mock_compressor.return_value = MagicMock()

        await _evaluate(empty_dataset, None)

        # Should print no valid samples message
        print_calls = [call.args[0] for call in mock_print.call_args_list]
        assert any(
            "No valid samples found in the dataset" in call for call in print_calls
        )


@pytest.mark.integration
@pytest.mark.asyncio
async def test_compression_evaluation_with_malformed_data(tmp_path: Path) -> None:
    """Test evaluation handles malformed data gracefully."""

    malformed_dataset = tmp_path / "malformed.json"
    # Dataset with missing required fields
    malformed_dataset.write_text(
        json.dumps(
            [
                {"query": "test"},  # missing documents
                {"documents": []},  # missing query
            ]
        ),
        encoding="utf-8",
    )

    mock_config = MagicMock()
    mock_config.rag.compression_enabled = True

    with (
        patch(
            "scripts.eval.rag_compression_eval.get_settings", return_value=mock_config
        ),
        patch(
            "scripts.eval.rag_compression_eval._load_vector_service"
        ) as mock_load_service,
        patch("builtins.print") as mock_print,
        patch(
            "scripts.eval.rag_compression_eval.FastEmbedEmbeddings"
        ) as mock_embeddings,
        patch(
            "scripts.eval.rag_compression_eval.EmbeddingsRedundantFilter"
        ) as mock_redundant_filter,
        patch(
            "scripts.eval.rag_compression_eval.EmbeddingsFilter"
        ) as mock_embeddings_filter,
        patch(
            "scripts.eval.rag_compression_eval.DocumentCompressorPipeline"
        ) as mock_compressor,
    ):
        mock_service = MagicMock()
        mock_service.cleanup = AsyncMock()
        mock_load_service.return_value = mock_service
        mock_embeddings.return_value = MagicMock()
        mock_redundant_filter.return_value = MagicMock()
        mock_embeddings_filter.return_value = MagicMock()
        mock_compressor.return_value = MagicMock()

        await _evaluate(malformed_dataset, None)

        # Should handle gracefully and print no valid samples message
        print_calls = [call.args[0] for call in mock_print.call_args_list]
        assert any(
            "No valid samples found in the dataset" in call for call in print_calls
        )


@pytest.mark.integration
def test_token_estimation_realistic_cases() -> None:
    """Test token estimation with realistic text samples."""

    # Test various text lengths and complexities
    test_cases = [
        ("", 1),  # Empty string gets minimum 1
        ("hello", 1),
        ("hello world", 2),
        ("This is a longer sentence with multiple words.", 8),
        (
            "Technical documentation often contains complex terms like "
            "Retrieval-Augmented Generation and Large Language Models.",
            13,
        ),
        ("Multi\nline\ntext\nwith\nbreaks", 5),  # split() handles newlines
    ]

    for text, expected in test_cases:
        assert _estimate_tokens(text) == expected, f"Failed for text: {text!r}"


@pytest.mark.integration
def test_document_building_with_realistic_metadata() -> None:
    """Test document building with realistic metadata structures."""

    raw_docs = [
        {
            "content": (
                "FastAPI is a modern web framework for building APIs with Python."
            ),
            "metadata": {
                "source": "docs/fastapi.md",
                "chunk_id": 1,
                "title": "FastAPI Introduction",
                "url": "https://example.com/fastapi",
                "created_at": "2024-01-01T00:00:00Z",
            },
        },
        {
            # Alternative content key
            "text": "LangChain provides components for LLM applications.",
            "metadata": {
                "source": "docs/langchain.md",
                "chunk_id": 2,
                "tags": ["framework", "llm", "python"],
            },
        },
        {
            "content": (
                "Vector databases store high-dimensional vectors for similarity search."
            ),
            "metadata": None,  # Test None metadata
        },
    ]

    documents = _build_documents(raw_docs)

    assert len(documents) == 3

    # First document
    assert (
        documents[0].page_content
        == "FastAPI is a modern web framework for building APIs with Python."
    )
    assert documents[0].metadata["source"] == "docs/fastapi.md"
    assert documents[0].metadata["chunk_id"] == 1

    # Second document (using 'text' key)
    assert (
        documents[1].page_content
        == "LangChain provides components for LLM applications."
    )
    assert documents[1].metadata["source"] == "docs/langchain.md"

    # Third document (None metadata)
    assert (
        documents[2].page_content
        == "Vector databases store high-dimensional vectors for similarity search."
    )
    assert documents[2].metadata == {}
