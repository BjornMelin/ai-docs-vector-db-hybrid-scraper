"""Comprehensive tests for Content Intelligence MCP tools.

- Real-world functionality focus
- Comprehensive coverage of all tools
- Zero flaky tests
- Modern pytest patterns
"""

from unittest.mock import AsyncMock, Mock

import pytest

from src.infrastructure.client_manager import ClientManager
from src.mcp_tools.models.requests import (
    ContentIntelligenceAnalysisRequest,
    ContentIntelligenceClassificationRequest,
    ContentIntelligenceMetadataRequest,
    ContentIntelligenceQualityRequest,
)
from src.mcp_tools.models.responses import ContentIntelligenceResult
from src.mcp_tools.tools.content_intelligence import register_tools
from src.services.content_intelligence.models import (
    ContentAnalysisResponse,
    ContentClassification,
    ContentMetadata,
    ContentType,
    EnrichedContent,
    QualityScore,
)


class MockContext:
    """Mock context for testing."""

    def __init__(self):
        self.logs = {"info": [], "debug": [], "warning": [], "error": []}

    async def info(self, msg: str):
        self.logs["info"].append(msg)

    async def debug(self, msg: str):
        self.logs["debug"].append(msg)

    async def warning(self, msg: str):
        self.logs["warning"].append(msg)

    async def error(self, msg: str):
        self.logs["error"].append(msg)


@pytest.fixture
def mock_content_intelligence_service():
    """Create mock content intelligence service."""
    service = AsyncMock()

    # Mock analyze_content method
    # Create mock enriched content
    enriched_content = EnrichedContent(
        url="https://example.com/docs/auth",
        content="Enhanced content with intelligence",
        title="Authentication Documentation",
        success=True,
        content_classification=ContentClassification(
            primary_type=ContentType.DOCUMENTATION,
            secondary_types=[ContentType.TUTORIAL],
            confidence_scores={
                ContentType.DOCUMENTATION: 0.85,
                ContentType.TUTORIAL: 0.65,
            },
            classification_reasoning="Well-structured documentation",
        ),
        quality_score=QualityScore(
            overall_score=0.88,
            completeness=0.90,
            relevance=0.85,
            confidence=0.92,
            meets_threshold=True,
            quality_issues=["Minor formatting inconsistencies"],
            improvement_suggestions=["Add more headings"],
        ),
        enriched_metadata=ContentMetadata(
            title="Authentication Documentation",
            description="Comprehensive authentication guide",
            word_count=500,
            char_count=2500,
            tags=["auth", "documentation", "security"],
            topics=["authentication", "security"],
            language="en",
        ),
    )

    analysis_result = ContentAnalysisResponse(
        success=True,
        enriched_content=enriched_content,
        processing_time_ms=125.5,
        cache_hit=False,
        error=None,
    )
    service.analyze_content = AsyncMock(return_value=analysis_result)

    # Mock classify_content_type method
    classification_result = ContentClassification(
        primary_type=ContentType.DOCUMENTATION,
        secondary_types=[ContentType.TUTORIAL],
        confidence_scores={ContentType.DOCUMENTATION: 0.85, ContentType.TUTORIAL: 0.65},
        classification_reasoning="Well-structured documentation with tutorial elements",
    )
    service.classify_content_type = AsyncMock(return_value=classification_result)

    # Mock assess_extraction_quality method
    quality_result = QualityScore(
        overall_score=0.88,
        completeness=0.90,
        relevance=0.85,
        confidence=0.92,
        meets_threshold=True,
        quality_issues=["Minor formatting inconsistencies"],
        improvement_suggestions=["Add more headings for better structure"],
    )
    service.assess_extraction_quality = AsyncMock(return_value=quality_result)

    # Mock extract_metadata method
    metadata_result = ContentMetadata(
        title="API Documentation",
        description="Comprehensive API documentation",
        word_count=1500,
        char_count=8500,
        tags=["api", "documentation", "web", "development"],
        topics=["authentication", "endpoints", "responses"],
        language="en",
        content_hash="abc123def456",
    )
    service.extract_metadata = AsyncMock(return_value=metadata_result)

    return service


@pytest.fixture
def mock_client_manager(mock_content_intelligence_service):
    """Create mock client manager."""
    manager = Mock(spec=ClientManager)
    manager.get_content_intelligence_service = AsyncMock(
        return_value=mock_content_intelligence_service
    )
    return manager


@pytest.fixture
def mock_mcp():
    """Create mock MCP server."""
    mcp = Mock()
    mcp.tools = {}

    def tool_decorator():
        def decorator(func):
            mcp.tools[func.__name__] = func
            return func

        return decorator

    mcp.tool = tool_decorator
    return mcp


@pytest.fixture
def mock_context():
    """Create mock context."""
    return MockContext()


@pytest.fixture(autouse=True)
def setup_tools(mock_mcp, mock_client_manager):
    """Register tools for testing."""
    register_tools(mock_mcp, mock_client_manager)


class TestContentIntelligenceToolRegistration:
    """Test tool registration functionality."""

    def test_all_tools_registered(self, mock_mcp):
        """Test that all expected tools are registered."""
        expected_tools = [
            "analyze_content_intelligence",
            "classify_content_type",
            "assess_content_quality",
            "extract_content_metadata",
            "get_adaptation_recommendations",
        ]

        for tool_name in expected_tools:
            assert tool_name in mock_mcp.tools
            assert callable(mock_mcp.tools[tool_name])


class TestAnalyzeContentIntelligence:
    """Test comprehensive content analysis tool."""

    @pytest.mark.asyncio
    async def test_successful_analysis(
        self, mock_mcp, mock_context, mock_content_intelligence_service
    ):
        """Test successful content intelligence analysis."""
        request = ContentIntelligenceAnalysisRequest(
            content="This is a comprehensive API documentation page explaining authentication.",
            url="https://example.com/docs/auth",
            title="Authentication Documentation",
            confidence_threshold=0.8,
            enable_classification=True,
            enable_quality_assessment=True,
            enable_metadata_extraction=True,
            enable_adaptations=True,
        )

        tool_func = mock_mcp.tools["analyze_content_intelligence"]
        result = await tool_func(request, mock_context)

        # Verify successful response
        assert isinstance(result, ContentIntelligenceResult)
        assert result.success is True
        assert result.enriched_content is not None
        assert isinstance(result.enriched_content, EnrichedContent)
        assert result.enriched_content.content == "Enhanced content with intelligence"
        assert result.processing_time_ms == 125.5
        assert result.cache_hit is False
        assert result.error is None

        # Verify service was called correctly
        mock_content_intelligence_service.analyze_content.assert_called_once()
        call_args = mock_content_intelligence_service.analyze_content.call_args[0][0]
        assert call_args.content == request.content
        assert call_args.url == request.url
        assert call_args.confidence_threshold == request.confidence_threshold

        # Verify logging
        assert len(mock_context.logs["info"]) >= 2
        assert any(
            "Starting content intelligence analysis" in msg
            for msg in mock_context.logs["info"]
        )
        assert any("completed in 125.5ms" in msg for msg in mock_context.logs["info"])

    @pytest.mark.asyncio
    async def test_service_unavailable(
        self, mock_mcp, mock_context, mock_client_manager
    ):
        """Test handling when content intelligence service is unavailable."""
        # Make service unavailable
        mock_client_manager.get_content_intelligence_service.return_value = None

        request = ContentIntelligenceAnalysisRequest(
            content="Test content", url="https://example.com/test"
        )

        tool_func = mock_mcp.tools["analyze_content_intelligence"]
        result = await tool_func(request, mock_context)

        # Verify error response
        assert result.success is False
        assert "Content Intelligence Service not initialized" in result.error

        # Verify error logging
        assert len(mock_context.logs["error"]) >= 1
        assert any("not available" in msg for msg in mock_context.logs["error"])

    @pytest.mark.asyncio
    async def test_analysis_exception_handling(
        self, mock_mcp, mock_context, mock_content_intelligence_service
    ):
        """Test exception handling during analysis."""
        # Make service raise exception
        mock_content_intelligence_service.analyze_content.side_effect = Exception(
            "Service error"
        )

        request = ContentIntelligenceAnalysisRequest(
            content="Test content", url="https://example.com/test"
        )

        tool_func = mock_mcp.tools["analyze_content_intelligence"]
        result = await tool_func(request, mock_context)

        # Verify error response
        assert result.success is False
        assert "Analysis failed: Service error" in result.error

        # Verify error logging
        assert any("failed: Service error" in msg for msg in mock_context.logs["error"])


class TestClassifyContentType:
    """Test content type classification tool."""

    @pytest.mark.asyncio
    async def test_successful_classification(
        self, mock_mcp, mock_context, mock_content_intelligence_service
    ):
        """Test successful content type classification."""
        request = ContentIntelligenceClassificationRequest(
            content="# API Reference\\n\\nThis document provides comprehensive API documentation.",
            url="https://example.com/api-docs",
            title="API Reference",
        )

        tool_func = mock_mcp.tools["classify_content_type"]
        result = await tool_func(request, mock_context)

        # Verify classification result
        assert isinstance(result, ContentClassification)
        assert result.primary_type == ContentType.DOCUMENTATION
        assert ContentType.TUTORIAL in result.secondary_types
        assert result.confidence_scores[ContentType.DOCUMENTATION] == 0.85
        assert "Well-structured documentation" in result.classification_reasoning

        # Verify service was called correctly
        mock_content_intelligence_service.classify_content_type.assert_called_once()

        # Verify logging
        assert any(
            "Classifying content type" in msg for msg in mock_context.logs["info"]
        )
        assert any(
            "classified as: documentation" in msg for msg in mock_context.logs["info"]
        )

    @pytest.mark.asyncio
    async def test_classification_service_unavailable(
        self, mock_mcp, mock_context, mock_client_manager
    ):
        """Test classification when service is unavailable."""
        mock_client_manager.get_content_intelligence_service.return_value = None

        request = ContentIntelligenceClassificationRequest(
            content="Test content", url="https://example.com/test"
        )

        tool_func = mock_mcp.tools["classify_content_type"]
        result = await tool_func(request, mock_context)

        # Verify fallback response
        assert result.primary_type == ContentType.UNKNOWN
        assert result.secondary_types == []
        assert "Service not available" in result.classification_reasoning

        # Verify error logging
        assert any("not available" in msg for msg in mock_context.logs["error"])

    @pytest.mark.asyncio
    async def test_classification_exception_handling(
        self, mock_mcp, mock_context, mock_content_intelligence_service
    ):
        """Test exception handling during classification."""
        mock_content_intelligence_service.classify_content_type.side_effect = Exception(
            "Classification error"
        )

        request = ContentIntelligenceClassificationRequest(
            content="Test content", url="https://example.com/test"
        )

        tool_func = mock_mcp.tools["classify_content_type"]
        result = await tool_func(request, mock_context)

        # Verify error response
        assert result.primary_type == ContentType.UNKNOWN
        assert (
            "Classification failed: Classification error"
            in result.classification_reasoning
        )


class TestAssessContentQuality:
    """Test content quality assessment tool."""

    @pytest.mark.asyncio
    async def test_successful_quality_assessment(
        self, mock_mcp, mock_context, mock_content_intelligence_service
    ):
        """Test successful content quality assessment."""
        request = ContentIntelligenceQualityRequest(
            content="Well-structured documentation with clear examples and good formatting.",
            confidence_threshold=0.8,
            query_context="API documentation quality",
            extraction_metadata={"method": "crawl4ai", "confidence": 0.9},
        )

        tool_func = mock_mcp.tools["assess_content_quality"]
        result = await tool_func(request, mock_context)

        # Verify quality assessment result
        assert isinstance(result, QualityScore)
        assert result.overall_score == 0.88
        assert result.completeness == 0.90
        assert result.relevance == 0.85
        assert result.confidence == 0.92
        assert result.meets_threshold is True
        assert "Minor formatting inconsistencies" in result.quality_issues
        assert len(result.improvement_suggestions) > 0

        # Verify service was called correctly
        mock_content_intelligence_service.assess_extraction_quality.assert_called_once()

        # Verify logging
        assert any(
            "Assessing content quality" in msg for msg in mock_context.logs["info"]
        )
        assert any("overall score 0.88" in msg for msg in mock_context.logs["info"])

    @pytest.mark.asyncio
    async def test_quality_assessment_service_unavailable(
        self, mock_mcp, mock_context, mock_client_manager
    ):
        """Test quality assessment when service is unavailable."""
        mock_client_manager.get_content_intelligence_service.return_value = None

        request = ContentIntelligenceQualityRequest(
            content="Test content", confidence_threshold=0.8
        )

        tool_func = mock_mcp.tools["assess_content_quality"]
        result = await tool_func(request, mock_context)

        # Verify fallback response
        assert result.overall_score == 0.1
        assert result.completeness == 0.1
        assert result.confidence == 0.1
        assert "Service not available" in result.quality_issues

        # Verify error logging
        assert any("not available" in msg for msg in mock_context.logs["error"])

    @pytest.mark.asyncio
    async def test_quality_assessment_exception_handling(
        self, mock_mcp, mock_context, mock_content_intelligence_service
    ):
        """Test exception handling during quality assessment."""
        mock_content_intelligence_service.assess_extraction_quality.side_effect = (
            Exception("Assessment error")
        )

        request = ContentIntelligenceQualityRequest(
            content="Test content", confidence_threshold=0.8
        )

        tool_func = mock_mcp.tools["assess_content_quality"]
        result = await tool_func(request, mock_context)

        # Verify error response
        assert result.overall_score == 0.1
        assert "Assessment failed: Assessment error" in result.quality_issues


class TestExtractContentMetadata:
    """Test metadata extraction tool."""

    @pytest.mark.asyncio
    async def test_successful_metadata_extraction(
        self, mock_mcp, mock_context, mock_content_intelligence_service
    ):
        """Test successful metadata extraction."""
        request = ContentIntelligenceMetadataRequest(
            content="# API Documentation\\n\\nComprehensive guide to our REST API with examples.",
            url="https://example.com/docs/api",
            raw_html="<html><head><title>API Docs</title></head><body>...</body></html>",
            extraction_metadata={"crawler": "crawl4ai", "timestamp": "2024-01-01"},
        )

        tool_func = mock_mcp.tools["extract_content_metadata"]
        result = await tool_func(request, mock_context)

        # Verify metadata extraction result
        assert isinstance(result, ContentMetadata)
        assert result.title == "API Documentation"
        assert result.description == "Comprehensive API documentation"
        assert result.word_count == 1500
        assert result.char_count == 8500
        assert "api" in result.tags
        assert "documentation" in result.tags
        assert "authentication" in result.topics
        assert result.language == "en"
        assert result.content_hash == "abc123def456"

        # Verify service was called correctly
        mock_content_intelligence_service.extract_metadata.assert_called_once()

        # Verify logging
        assert any(
            "Extracting metadata for URL" in msg for msg in mock_context.logs["info"]
        )
        assert any("1500 words, 4 tags" in msg for msg in mock_context.logs["info"])

    @pytest.mark.asyncio
    async def test_metadata_extraction_service_unavailable(
        self, mock_mcp, mock_context, mock_client_manager
    ):
        """Test metadata extraction when service is unavailable."""
        mock_client_manager.get_content_intelligence_service.return_value = None

        request = ContentIntelligenceMetadataRequest(
            content="Test content with multiple words here",
            url="https://example.com/test",
        )

        tool_func = mock_mcp.tools["extract_content_metadata"]
        result = await tool_func(request, mock_context)

        # Verify fallback response with basic metadata
        assert result.word_count == 6  # "Test content with multiple words here"
        assert result.char_count == len("Test content with multiple words here")

        # Verify error logging
        assert any("not available" in msg for msg in mock_context.logs["error"])

    @pytest.mark.asyncio
    async def test_metadata_extraction_exception_handling(
        self, mock_mcp, mock_context, mock_content_intelligence_service
    ):
        """Test exception handling during metadata extraction."""
        mock_content_intelligence_service.extract_metadata.side_effect = Exception(
            "Extraction error"
        )

        request = ContentIntelligenceMetadataRequest(
            content="Test content with several words", url="https://example.com/test"
        )

        tool_func = mock_mcp.tools["extract_content_metadata"]
        result = await tool_func(request, mock_context)

        # Verify fallback response
        assert result.word_count == 5  # Basic word count fallback
        assert result.char_count == len("Test content with several words")


class TestContentIntelligenceIntegration:
    """Test integration scenarios and real-world usage patterns."""

    @pytest.mark.asyncio
    async def test_minimal_request_handling(self, mock_mcp, mock_context):
        """Test tools handle minimal request data correctly."""
        # Test with minimal analysis request
        minimal_request = ContentIntelligenceAnalysisRequest(
            content="Basic content", url="https://example.com"
        )

        tool_func = mock_mcp.tools["analyze_content_intelligence"]
        result = await tool_func(minimal_request, mock_context)

        # Should complete successfully with minimal data
        assert isinstance(result, ContentIntelligenceResult)

    @pytest.mark.asyncio
    async def test_comprehensive_workflow(self, mock_mcp, mock_context):
        """Test a comprehensive workflow using multiple tools."""
        content = "# API Documentation\\n\\nComprehensive REST API guide with examples."
        url = "https://example.com/api-docs"

        # 1. Classify content type
        classification_request = ContentIntelligenceClassificationRequest(
            content=content, url=url, title="API Documentation"
        )

        classify_func = mock_mcp.tools["classify_content_type"]
        classification = await classify_func(classification_request, mock_context)

        # 2. Assess quality
        quality_request = ContentIntelligenceQualityRequest(
            content=content, confidence_threshold=0.8
        )

        quality_func = mock_mcp.tools["assess_content_quality"]
        quality = await quality_func(quality_request, mock_context)

        # 3. Extract metadata
        metadata_request = ContentIntelligenceMetadataRequest(content=content, url=url)

        metadata_func = mock_mcp.tools["extract_content_metadata"]
        metadata = await metadata_func(metadata_request, mock_context)

        # Verify all steps completed successfully
        assert classification.primary_type == ContentType.DOCUMENTATION
        assert quality.overall_score > 0.8
        assert metadata.word_count > 0

        # Verify appropriate logging throughout workflow
        assert len(mock_context.logs["info"]) >= 6  # At least 2 per tool
