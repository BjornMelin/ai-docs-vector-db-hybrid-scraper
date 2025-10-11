"""Focused unit tests for content intelligence MCP tool registration."""

from __future__ import annotations

from types import SimpleNamespace
from typing import Any
from unittest.mock import AsyncMock

import pytest

from src.mcp_tools.models.responses import ContentIntelligenceResult
from src.mcp_tools.tools.content_intelligence import (
    ContentAnalysisToolPayload,
    ContentClassificationToolPayload,
    ContentMetadataToolPayload,
    ContentQualityToolPayload,
    register_tools,
)
from src.services.content_intelligence.models import (
    AdaptationRecommendation,
    AdaptationStrategy,
    ContentAnalysisRequest,
    ContentAnalysisResponse,
    ContentClassification,
    ContentMetadata,
    ContentType,
    EnrichedContent,
    QualityScore,
)
from tests.unit.conftest import require_optional_dependency  # type: ignore[import]


require_optional_dependency("redis")


class MockMCP:
    """Minimal MCP stub capturing registered tool callables."""

    def __init__(self) -> None:
        self.tools: dict[str, Any] = {}

    def tool(self):
        def decorator(func):
            self.tools[func.__name__] = func
            return func

        return decorator


class MockContext:
    """Collects log messages emitted by tool implementations."""

    def __init__(self) -> None:
        self.logs: dict[str, list[str]] = {key: [] for key in ("info", "error")}

    async def info(self, message: str) -> None:
        self.logs["info"].append(message)

    async def error(self, message: str) -> None:
        self.logs["error"].append(message)


@pytest.fixture()
def content_service() -> Any:
    service = SimpleNamespace()
    enriched_metadata = ContentMetadata(
        url="https://example.com",
        title="Doc",
        description="Desc",
        word_count=100,
        char_count=500,
        tags=["docs"],
        topics=["topic"],
        language="en",
    )
    service.analyze_content = AsyncMock(
        return_value=ContentAnalysisResponse(
            success=True,
            enriched_content=EnrichedContent(
                url="https://example.com",
                content="Enhanced content",
                title="Example",
                success=True,
                content_classification=ContentClassification(
                    primary_type=ContentType.DOCUMENTATION,
                    secondary_types=[],
                    confidence_scores={ContentType.DOCUMENTATION: 0.9},
                    classification_reasoning="",
                ),
                quality_score=QualityScore(
                    overall_score=0.9,
                    completeness=0.9,
                    relevance=0.9,
                    confidence=0.9,
                    meets_threshold=True,
                ),
                enriched_metadata=enriched_metadata,
            ),
            processing_time_ms=42.0,
            cache_hit=False,
        )
    )
    service.classify_content_type = AsyncMock(
        return_value=ContentClassification(
            primary_type=ContentType.DOCUMENTATION,
            secondary_types=[],
            confidence_scores={ContentType.DOCUMENTATION: 0.9},
            classification_reasoning="",
        )
    )
    service.assess_extraction_quality = AsyncMock(
        return_value=QualityScore(
            overall_score=0.8,
            completeness=0.75,
            relevance=0.8,
            confidence=0.85,
            meets_threshold=True,
        )
    )
    service.extract_metadata = AsyncMock(return_value=enriched_metadata)
    service.recommend_adaptations = AsyncMock(
        return_value=[
            AdaptationRecommendation(
                strategy=AdaptationStrategy.EXTRACT_MAIN_CONTENT,
                priority="high",
                confidence=0.85,
                reasoning="Primary content available via main selector",
            )
        ]
    )
    return service


@pytest.fixture()
def mock_mcp(content_service: AsyncMock) -> MockMCP:
    registry = MockMCP()
    register_tools(registry, content_service=content_service)
    return registry


@pytest.fixture()
def mock_context() -> MockContext:
    return MockContext()


@pytest.mark.asyncio
async def test_analyze_content_success(
    mock_mcp: MockMCP, mock_context: MockContext
) -> None:
    tool = mock_mcp.tools["analyze_content_intelligence"]
    request = ContentAnalysisToolPayload(
        analysis=ContentAnalysisRequest(content="Hello", url="https://example.com")
    )

    result = await tool(request, mock_context)

    assert isinstance(result, ContentIntelligenceResult)
    assert result.success is True
    assert any(
        "Starting content intelligence analysis" in msg
        for msg in mock_context.logs["info"]
    )


@pytest.mark.asyncio
async def test_analyze_content_without_service_returns_failure(
    mock_context: MockContext,
) -> None:
    registry = MockMCP()
    register_tools(registry, content_service=None)
    tool = registry.tools["analyze_content_intelligence"]
    request = ContentAnalysisToolPayload(
        analysis=ContentAnalysisRequest(content="Hello", url="https://example.com")
    )

    result = await tool(request, mock_context)

    assert result.success is False
    assert result.error == "Content Intelligence Service not initialized"


@pytest.mark.asyncio
async def test_classify_content_type_handles_service_failure(
    content_service: AsyncMock, mock_context: MockContext
) -> None:
    content_service.classify_content_type.side_effect = RuntimeError("boom")

    registry = MockMCP()
    register_tools(registry, content_service=content_service)
    tool = registry.tools["classify_content_type"]
    payload = ContentClassificationToolPayload(
        content="Sample", url="https://example.com", title="Doc"
    )

    classification = await tool(payload, mock_context)

    assert classification.primary_type is ContentType.UNKNOWN
    assert "boom" in classification.classification_reasoning


@pytest.mark.asyncio
async def test_quality_assessment_exception_returns_fallback(
    content_service: AsyncMock, mock_context: MockContext
) -> None:
    content_service.assess_extraction_quality.side_effect = RuntimeError("fail")

    registry = MockMCP()
    register_tools(registry, content_service=content_service)
    tool = registry.tools["assess_content_quality"]
    payload = ContentQualityToolPayload(content="Body text", confidence_threshold=0.5)

    score = await tool(payload, mock_context)

    assert pytest.approx(score.overall_score, rel=0.01) == 0.1
    assert "fail" in mock_context.logs["error"][0]


@pytest.mark.asyncio
async def test_metadata_extraction_without_service_returns_word_counts(
    mock_context: MockContext,
) -> None:
    registry = MockMCP()
    register_tools(registry, content_service=None)
    tool = registry.tools["extract_content_metadata"]
    payload = ContentMetadataToolPayload(
        content="one two three",
        url="https://example.com",
    )

    metadata = await tool(payload, mock_context)

    assert metadata.word_count == 3
    assert metadata.char_count == len("one two three")


@pytest.mark.asyncio
async def test_get_adaptation_recommendations_returns_service_output(
    content_service: Any,
) -> None:
    registry = MockMCP()
    register_tools(registry, content_service=content_service)
    tool = registry.tools["get_adaptation_recommendations"]

    recommendations = await tool("https://example.com", ctx=None)

    assert recommendations[0]["strategy"] == (
        AdaptationStrategy.EXTRACT_MAIN_CONTENT.value
    )
    content_service.recommend_adaptations.assert_awaited_once()
