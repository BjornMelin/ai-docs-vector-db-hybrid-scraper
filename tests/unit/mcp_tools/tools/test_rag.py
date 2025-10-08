"""Tests for the RAG MCP tool module."""

from __future__ import annotations

import importlib.util
import sys
import types
from collections.abc import Callable
from pathlib import Path
from types import SimpleNamespace
from typing import Any, cast
from unittest.mock import AsyncMock, MagicMock

import pytest

from src.services.rag.models import (
    AnswerMetrics,
    RAGResult,
    RAGServiceMetrics,
    SourceAttribution,
)
from tests.unit.stub_factories import register_rag_dependency_stubs


register_rag_dependency_stubs()

client_manager_stub = cast(Any, types.ModuleType("src.infrastructure.client_manager"))
client_manager_stub.ClientManager = type("ClientManager", (), {})
sys.modules.setdefault("src.infrastructure.client_manager", client_manager_stub)

crawl4ai_stub = cast(Any, sys.modules["crawl4ai"])


class _StubAsyncCrawler:
    async def start(self) -> None:
        """Simulate asynchronous crawler start."""

        return None

    async def close(self) -> None:
        """Simulate asynchronous crawler shutdown."""

        return None


crawl4ai_stub.AsyncWebCrawler = _StubAsyncCrawler

ROOT = Path(__file__).resolve().parents[4]
MODULE_PATH = ROOT / "src/mcp_tools/tools/rag.py"
_spec = importlib.util.spec_from_file_location("rag_under_test", MODULE_PATH)
assert _spec and _spec.loader
rag_module = importlib.util.module_from_spec(_spec)
sys.modules[_spec.name] = rag_module
_spec.loader.exec_module(rag_module)  # type: ignore[arg-type]

RAGAnswerRequest = rag_module.RAGAnswerRequest
RAGMetricsResponse = rag_module.RAGMetricsResponse
register_tools = rag_module.register_tools

sys.modules.pop("src.infrastructure.client_manager", None)


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
def mock_client_manager() -> MagicMock:
    """Create a client manager mock exposing get_rag_generator."""

    manager = MagicMock()
    manager.get_rag_generator = AsyncMock()
    return manager


@pytest.fixture()
def registered_tools(
    configured_app: SimpleNamespace, mock_client_manager: MagicMock
) -> dict[str, Callable]:
    """Register RAG tools against a mocked FastMCP instance."""

    del configured_app
    mock_app = MagicMock()
    tools: dict[str, Callable] = {}

    def capture(func: Callable) -> Callable:
        tools[func.__name__] = func
        return func

    mock_app.tool.return_value = capture
    register_tools(mock_app, mock_client_manager)
    return tools


@pytest.mark.asyncio()
async def test_generate_rag_answer_uses_shared_generator(
    configured_app: SimpleNamespace,
    registered_tools: dict[str, Callable],
    mock_client_manager: MagicMock,
) -> None:
    """generate_rag_answer should reuse the shared client manager generator."""

    del configured_app
    rag_generator = MagicMock()
    rag_generator.generate_answer = AsyncMock(
        return_value=RAGResult(
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
    )
    rag_generator.cleanup = AsyncMock()
    mock_client_manager.get_rag_generator.return_value = rag_generator

    request = RAGAnswerRequest(query="What is RAG?", top_k=3)
    response = await registered_tools["generate_rag_answer"](request)

    mock_client_manager.get_rag_generator.assert_awaited_once()
    rag_generator.generate_answer.assert_awaited_once()
    rag_generator.cleanup.assert_not_called()
    assert response.answer == "Answer"
    assert response.confidence_score == 0.9
    assert response.sources_used == 1
    assert response.sources is not None and response.sources[0]["source_id"] == "doc-1"


@pytest.mark.asyncio()
async def test_get_rag_metrics_returns_service_metrics(
    configured_app: SimpleNamespace,
    registered_tools: dict[str, Callable],
    mock_client_manager: MagicMock,
) -> None:
    """get_rag_metrics should surface metrics from the shared generator."""

    del configured_app
    rag_generator = MagicMock()
    rag_generator.get_metrics.return_value = RAGServiceMetrics(
        generation_count=10,
        avg_generation_time_ms=25.0,
        total_generation_time_ms=250.0,
    )
    mock_client_manager.get_rag_generator.return_value = rag_generator

    response = await registered_tools["get_rag_metrics"]()

    mock_client_manager.get_rag_generator.assert_awaited_once()
    rag_generator.get_metrics.assert_called_once()
    assert isinstance(response, RAGMetricsResponse)
    assert response.generation_count == 10
    assert response.avg_generation_time_ms == 25.0


@pytest.mark.asyncio()
async def test_test_rag_configuration_confirms_connectivity(
    configured_app: SimpleNamespace,
    registered_tools: dict[str, Callable],
    mock_client_manager: MagicMock,
) -> None:
    """test_rag_configuration should validate connectivity without cleanup."""

    del configured_app
    rag_generator = MagicMock()
    rag_generator.cleanup = AsyncMock()
    mock_client_manager.get_rag_generator.return_value = rag_generator

    results = await registered_tools["test_rag_configuration"]()

    mock_client_manager.get_rag_generator.assert_awaited_once()
    rag_generator.cleanup.assert_not_called()
    assert results["rag_enabled"] is True
    assert results["connectivity_test"] is True
    assert results["error"] is None


@pytest.mark.asyncio()
async def test_generate_rag_answer_errors_when_rag_disabled(
    configured_app: SimpleNamespace,
    registered_tools: dict[str, Callable],
    mock_client_manager: MagicMock,
) -> None:
    """generate_rag_answer should fail fast when configuration disables RAG."""

    configured_app.rag.enable_rag = False

    request = RAGAnswerRequest(query="Is RAG enabled?")

    with pytest.raises(RuntimeError, match="RAG is not enabled"):
        await registered_tools["generate_rag_answer"](request)

    mock_client_manager.get_rag_generator.assert_not_called()
