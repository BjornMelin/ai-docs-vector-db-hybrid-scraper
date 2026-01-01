"""Unit tests for small, pure helpers in `src.services.rag.generator`."""

from __future__ import annotations

from typing import Any, cast

import pytest
from langchain_core.documents import Document

from src.services.errors import EmbeddingServiceError
from src.services.rag.generator import RAGGenerator, _normalize_score
from src.services.rag.models import RAGConfig, RAGRequest


class _AsyncRetriever:
    async def ainvoke(self, query: str, /) -> list[str]:
        return [query]


def test_normalize_score_maps_common_ranges() -> None:
    """Normalise retriever scores into the [0, 1] interval when possible."""
    assert _normalize_score(-1.0) == 0.0
    assert _normalize_score(0.0) == 0.5
    assert _normalize_score(1.0) == 1.0
    assert _normalize_score(0.2) == 0.6
    assert _normalize_score(2.0) is None
    assert _normalize_score("nope") is None


def test_build_metrics_extracts_token_usage() -> None:
    """Extract token usage fields when present in LLM response metadata."""
    metrics = RAGGenerator._build_metrics(
        12.3,
        {
            "token_usage": {
                "prompt_tokens": 4,
                "completion_tokens": 6,
                "total_tokens": 10,
            }
        },
    )
    assert metrics.generation_time_ms == 12.3
    assert metrics.prompt_tokens == 4
    assert metrics.completion_tokens == 6
    assert metrics.total_tokens == 10


def test_render_context_includes_url_when_requested() -> None:
    """Include URL lines only when sources are requested."""
    documents = [
        Document(page_content="content", metadata={"title": "T", "url": "https://e.com"})
    ]
    rendered = RAGGenerator._render_context(documents, include_sources=True)
    assert "URL: https://e.com" in rendered

    rendered_without_sources = RAGGenerator._render_context(
        documents, include_sources=False
    )
    assert "URL: https://e.com" not in rendered_without_sources


def test_build_sources_normalizes_metadata() -> None:
    """Build `SourceAttribution` entries and normalise score metadata."""
    documents = [
        Document(
            page_content="content",
            metadata={"source_id": "s1", "title": "T1", "score": -1.0, "excerpt": "x"},
        )
    ]
    sources = RAGGenerator._build_sources(documents)
    assert len(sources) == 1
    assert sources[0].source_id == "s1"
    assert sources[0].title == "T1"
    assert sources[0].score == 0.0
    assert sources[0].excerpt == "x"


def test_derive_confidence_returns_none_when_disabled() -> None:
    """Return `None` when confidence derivation is disabled."""
    config = RAGConfig(confidence_from_scores=False)
    generator = RAGGenerator(config, cast(Any, _AsyncRetriever()))
    docs = [Document(page_content="content", metadata={"score": 1.0})]
    assert generator._derive_confidence(docs) is None


def test_derive_confidence_averages_normalised_scores() -> None:
    """Average normalised scores into a heuristic confidence."""
    config = RAGConfig(confidence_from_scores=True)
    generator = RAGGenerator(config, cast(Any, _AsyncRetriever()))
    docs = [
        Document(page_content="c1", metadata={"score": -1.0}),
        Document(page_content="c2", metadata={"score": 1.0}),
    ]
    assert generator._derive_confidence(docs) == pytest.approx(0.5)


class _DocRetriever:
    async def ainvoke(self, _query: str, /) -> list[Document]:
        return [Document(page_content="content", metadata={"id": "doc-1"})]


@pytest.mark.asyncio
async def test_collect_documents_sets_source_id_when_include_sources() -> None:
    """Populate `source_id` from `id` for returned documents."""
    config = RAGConfig(retriever_top_k=1)
    generator = RAGGenerator(config, cast(Any, _DocRetriever()))
    request = RAGRequest(query="query", top_k=1)
    docs = await generator._collect_documents(request, include_sources=True)
    assert docs[0].metadata["source_id"] == "doc-1"


@pytest.mark.asyncio
async def test_initialize_creates_chat_model(monkeypatch: pytest.MonkeyPatch) -> None:
    """Create a chat model instance when none is provided."""
    created: dict[str, Any] = {}

    class _StubChatModel:
        def __init__(self, **kwargs: Any):
            created.update(kwargs)

    monkeypatch.setattr("src.services.rag.generator.ChatOpenAI", _StubChatModel)

    config = RAGConfig(model="stub-model", temperature=0.7, max_tokens=123)
    generator = RAGGenerator(config, cast(Any, _AsyncRetriever()))
    await generator.initialize()
    assert generator.is_initialized() is True
    assert created["model"] == "stub-model"
    assert created["temperature"] == 0.7
    assert created["max_tokens"] == 123


@pytest.mark.asyncio
async def test_cleanup_resets_chat_model() -> None:
    """Clear any cached chat model instance."""
    generator = RAGGenerator(
        RAGConfig(),
        cast(Any, _AsyncRetriever()),
        chat_model=cast(Any, object()),
    )
    assert generator.is_initialized() is True
    await generator.cleanup()
    assert generator.is_initialized() is False


@pytest.mark.asyncio
async def test_generate_answer_rejects_uninitialized_generator() -> None:
    """Reject generation requests until the generator is initialized."""
    generator = RAGGenerator(RAGConfig(), cast(Any, _AsyncRetriever()))
    with pytest.raises(EmbeddingServiceError):
        await generator.generate_answer(RAGRequest(query="query"))
