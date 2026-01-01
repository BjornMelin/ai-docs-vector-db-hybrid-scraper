"""Unit tests for the public RAG API helpers."""

from __future__ import annotations

from types import SimpleNamespace
from typing import Any

import pytest

from src.services.errors import ExternalServiceError, NetworkError
from src.services.rag import api as rag_api
from src.services.rag.models import AnswerMetrics, RAGResult, SourceAttribution


def test_convert_to_internal_request_defaults_top_k_from_search_results() -> None:
    """Derive `top_k` from the number of provided search results."""
    request = rag_api.RAGRequest(query="hello", search_results=[{"a": 1}, {"b": 2}])
    internal = rag_api._convert_to_internal_request(request)
    assert internal.query == "hello"
    assert internal.top_k == 2
    assert internal.filters is None


def test_convert_to_internal_request_respects_max_context_results() -> None:
    """Prefer explicit `max_context_results` over inferred values."""
    request = rag_api.RAGRequest(
        query="hello",
        search_results=[{"a": 1}, {"b": 2}, {"c": 3}],
        max_context_results=1,
        preferred_source_types=["docs"],
    )
    internal = rag_api._convert_to_internal_request(request)
    assert internal.top_k == 1
    assert internal.filters == {"source_types": ["docs"]}


def test_convert_to_internal_request_allows_no_search_results() -> None:
    """Allow requests without upstream search results."""
    request = rag_api.RAGRequest(query="hello", max_context_results=5)
    internal = rag_api._convert_to_internal_request(request)
    assert internal.top_k == 5


def test_format_sources_respects_include_sources_flag() -> None:
    """Skip source formatting when the request disables it explicitly."""
    request = rag_api.RAGRequest(query="hello", include_sources=False)
    result = SimpleNamespace(sources=[SimpleNamespace(source_id="1", title="t")])
    assert rag_api._format_sources(request, result) is None


def test_format_sources_formats_source_objects() -> None:
    """Format structured source models into plain dicts."""
    request = rag_api.RAGRequest(query="hello", include_sources=True)
    sources = [
        SourceAttribution(
            source_id="a",
            title="Title A",
            url="https://example.com/a",
            excerpt="excerpt",
            score=0.9,
        )
    ]
    result = SimpleNamespace(sources=sources)
    formatted = rag_api._format_sources(request, result)
    assert formatted == [
        {
            "source_id": "a",
            "title": "Title A",
            "url": "https://example.com/a",
            "relevance_score": 0.9,
            "excerpt": "excerpt",
        }
    ]


def test_format_metrics_prefers_model_dump() -> None:
    """Serialise pydantic metrics objects via `model_dump`."""
    metrics = AnswerMetrics(
        generation_time_ms=12.3,
        total_tokens=10,
        prompt_tokens=4,
        completion_tokens=6,
    )
    result = SimpleNamespace(metrics=metrics)
    formatted = rag_api._format_metrics(result)
    assert isinstance(formatted, dict)
    assert formatted["generation_time_ms"] == 12.3
    assert formatted["total_tokens"] == 10


def test_format_metrics_handles_plain_object() -> None:
    """Fallback to attribute reads when metrics are plain objects."""
    metrics = SimpleNamespace(
        generation_time_ms=12.3,
        total_tokens=10,
        prompt_tokens=4,
        completion_tokens=6,
    )
    result = SimpleNamespace(metrics=metrics)
    assert rag_api._format_metrics(result) == {
        "total_tokens": 10,
        "prompt_tokens": 4,
        "completion_tokens": 6,
        "generation_time_ms": 12.3,
    }


class _StubGenerator:
    def __init__(self, result: Any):
        self._result = result
        self._cleared = False

    async def generate_answer(self, request: Any) -> Any:
        return self._result

    def get_metrics(self) -> dict[str, Any]:
        return {"ok": True}

    def clear_cache(self) -> None:
        self._cleared = True


@pytest.mark.asyncio
async def test_generate_rag_answer_success() -> None:
    """Return a structured response when the generator succeeds."""
    result = RAGResult(
        answer="hi",
        confidence_score=0.8,
        sources=[
            SourceAttribution(
                source_id="s1",
                title="T1",
                url=None,
                excerpt=None,
                score=0.5,
            )
        ],
        generation_time_ms=1.5,
        metrics=AnswerMetrics(generation_time_ms=1.5),
    )
    generator = _StubGenerator(result)
    request = rag_api.RAGRequest(query="query", include_sources=True)
    response = await rag_api.generate_rag_answer(request, rag_generator=generator)
    assert response.answer == "hi"
    assert response.confidence_score == 0.8
    assert response.sources_used == 1
    assert response.generation_time_ms == 1.5
    assert response.sources == [
        {
            "source_id": "s1",
            "title": "T1",
            "url": None,
            "relevance_score": 0.5,
            "excerpt": None,
        }
    ]
    assert response.metrics is not None


@pytest.mark.asyncio
async def test_generate_rag_answer_wraps_unexpected_errors() -> None:
    """Wrap unknown exceptions into `ExternalServiceError`."""
    class BoomGenerator:
        async def generate_answer(self, _request: Any) -> Any:
            raise RuntimeError("boom")

    with pytest.raises(ExternalServiceError):
        await rag_api.generate_rag_answer(
            rag_api.RAGRequest(query="query"),
            rag_generator=BoomGenerator(),
        )


@pytest.mark.asyncio
async def test_generate_rag_answer_propagates_network_errors() -> None:
    """Propagate well-known network errors without wrapping."""
    class NetworkBoomGenerator:
        async def generate_answer(self, _request: Any) -> Any:
            raise NetworkError("nope")

    with pytest.raises(NetworkError):
        await rag_api.generate_rag_answer(
            rag_api.RAGRequest(query="q"),
            rag_generator=NetworkBoomGenerator(),
        )


@pytest.mark.asyncio
async def test_get_rag_metrics_and_clear_cache() -> None:
    """Return status payloads for management helpers."""
    generator = _StubGenerator(
        result=RAGResult(answer="answer", sources=[], generation_time_ms=0.1)
    )
    assert await rag_api.get_rag_metrics(rag_generator=generator) == {"ok": True}
    assert await rag_api.clear_rag_cache(rag_generator=generator) == {
        "status": "success",
        "message": "RAG cache cleared",
    }
    assert generator._cleared is True
