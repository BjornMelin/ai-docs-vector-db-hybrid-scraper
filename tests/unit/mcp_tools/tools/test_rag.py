"""Tests for the RAG MCP tool module."""

from __future__ import annotations

from collections.abc import Callable
from types import SimpleNamespace
from typing import Any, cast
from unittest.mock import AsyncMock, Mock

import pytest
from fastmcp import FastMCP

from src.mcp_tools.tools import rag as rag_module
from src.services.rag.generator import RAGGenerator
from src.services.rag.models import (
    AnswerMetrics,
    RAGResult,
    RAGServiceMetrics,
    SourceAttribution,
)


@pytest.fixture()
def configured_app(monkeypatch: pytest.MonkeyPatch) -> SimpleNamespace:
    """Provide a configuration object with RAG enabled for tests."""
    config = SimpleNamespace(
        rag=SimpleNamespace(
            enable_rag=True,
            include_sources=True,
            max_tokens=256,
            temperature=0.2,
            model="test-model",
            max_context_length=4096,
        )
    )
    monkeypatch.setattr(rag_module, "get_settings", lambda: config)
    return config


@pytest.fixture()
def mock_rag_generator() -> SimpleNamespace:
    """Return a stub RAG generator exposing the service contract."""
    generator = SimpleNamespace()
    generator.generate_answer = AsyncMock()
    generator.get_metrics = Mock(
        return_value=RAGServiceMetrics(
            generation_count=10,
            avg_generation_time_ms=25.0,
            total_generation_time_ms=250.0,
        )
    )
    generator.validate_configuration = AsyncMock()
    return generator


@pytest.fixture()
def registered_tools(
    configured_app: SimpleNamespace,
    mock_rag_generator: SimpleNamespace,
    monkeypatch: pytest.MonkeyPatch,
) -> dict[str, Callable[..., Any]]:
    """Register RAG tools against a mocked FastMCP instance."""
    del configured_app
    mock_app = SimpleNamespace()
    registry: dict[str, Callable[..., Any]] = {}

    def capture(func: Callable[..., Any]) -> Callable[..., Any]:
        registry[func.__name__] = func
        return func

    mock_app.tool = lambda *args, **kwargs: capture

    rag_module.register_tools(
        cast(FastMCP[Any], mock_app),
        rag_generator=cast(RAGGenerator, mock_rag_generator),
    )
    return registry


@pytest.mark.asyncio()
async def test_generate_rag_answer_uses_override(
    registered_tools: dict[str, Callable[..., Any]],
    mock_rag_generator: SimpleNamespace,
) -> None:
    """generate_rag_answer should reuse the injected generator override."""
    rag_result = RAGResult(
        answer="Answer",
        confidence_score=0.9,
        sources=[
            SourceAttribution(
                source_id="doc-1",
                title="Document 1",
                url="https://example.com/doc-1",
                excerpt="snippet",
                score=0.8,
            )
        ],
        generation_time_ms=42.0,
        metrics=AnswerMetrics(
            total_tokens=50,
            prompt_tokens=20,
            completion_tokens=30,
            generation_time_ms=42.0,
        ),
    )
    mock_rag_generator.generate_answer.return_value = rag_result

    request = rag_module.RAGAnswerRequest(
        query="What is RAG?",
        top_k=3,
        filters=None,
        max_tokens=None,
        temperature=None,
        include_sources=None,
    )
    response = await registered_tools["generate_rag_answer"](request)

    mock_rag_generator.generate_answer.assert_awaited_once()
    assert response.answer == "Answer"
    assert response.confidence_score == 0.9
    assert response.sources_used == 1
    assert response.sources is not None and response.sources[0]["source_id"] == "doc-1"


@pytest.mark.asyncio()
async def test_get_rag_metrics_surfaces_generator_metrics(
    registered_tools: dict[str, Callable[..., Any]],
    mock_rag_generator: SimpleNamespace,
) -> None:
    """get_rag_metrics should proxy generator metrics."""
    response = await registered_tools["get_rag_metrics"]()

    mock_rag_generator.get_metrics.assert_called_once()
    assert isinstance(response, rag_module.RAGMetricsResponse)
    assert response.generation_count == 10
    assert response.avg_generation_time_ms == 25.0


@pytest.mark.asyncio()
async def test_test_rag_configuration_confirms_connectivity(
    registered_tools: dict[str, Callable[..., Any]],
    mock_rag_generator: SimpleNamespace,
) -> None:
    """test_rag_configuration should validate generator availability."""
    results = await registered_tools["test_rag_configuration"]()

    mock_rag_generator.validate_configuration.assert_awaited_once()
    assert results["rag_enabled"] is True
    assert results["connectivity_test"] is True
