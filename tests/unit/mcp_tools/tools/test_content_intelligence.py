"""Tests for Content Intelligence MCP tools."""

from unittest.mock import AsyncMock, MagicMock
import pytest

from src.mcp_tools.tools.content_intelligence import register_tools
from src.mcp_tools.models.requests import (
    ContentIntelligenceAnalysisRequest,
    ContentIntelligenceClassificationRequest,
    ContentIntelligenceQualityRequest,
    ContentIntelligenceMetadataRequest,
)
from src.mcp_tools.models.responses import ContentIntelligenceResult
from src.services.content_intelligence.models import (
    ContentAnalysisResponse,
    ContentClassification,
    ContentType,
    QualityScore,
    ContentMetadata,
    EnrichedContent,
    AdaptationRecommendation,
    AdaptationStrategy,
)


@pytest.fixture
def mock_mcp():
    """Create mock MCP server."""
    mcp = MagicMock()
    mcp.tool = MagicMock()
    return mcp


@pytest.fixture
def mock_content_service():
    """Create mock Content Intelligence Service."""
    service = AsyncMock()
    
    # Mock comprehensive analysis
    service.analyze_content = AsyncMock(return_value=ContentAnalysisResponse(
        success=True,
        enriched_content=EnrichedContent(
            url="https://example.com/test",
            content="Test content",
            title="Test Page",
            content_classification=ContentClassification(
                primary_type=ContentType.DOCUMENTATION,
                secondary_types=[],
                confidence_scores={ContentType.DOCUMENTATION: 0.9},
                classification_reasoning="Documentation patterns detected",
            ),
            quality_score=QualityScore(
                overall_score=0.85,
                completeness=0.9,
                relevance=0.8,
                confidence=0.85,
            ),
            enriched_metadata=ContentMetadata(
                title="Test Page",
                word_count=100,
                char_count=500,
            ),
        ),
        processing_time_ms=150.0,
        cache_hit=False,
    ))
    
    # Mock individual component methods
    service.classify_content_type = AsyncMock(return_value=ContentClassification(
        primary_type=ContentType.DOCUMENTATION,
        secondary_types=[],
        confidence_scores={ContentType.DOCUMENTATION: 0.9},
        classification_reasoning="Documentation patterns detected",
    ))
    
    service.assess_extraction_quality = AsyncMock(return_value=QualityScore(
        overall_score=0.85,
        completeness=0.9,
        relevance=0.8,
        confidence=0.85,
    ))
    
    service.extract_metadata = AsyncMock(return_value=ContentMetadata(
        title="Test Page",
        word_count=100,
        char_count=500,
    ))
    
    service.recommend_adaptations = AsyncMock(return_value=[
        AdaptationRecommendation(
            strategy=AdaptationStrategy.EXTRACT_MAIN_CONTENT,
            priority="high",
            confidence=0.9,
            reasoning="Main content extraction recommended",
            implementation_notes="Use .main-content selector",
            estimated_improvement=0.3,
        )
    ])
    
    service.get_performance_metrics = MagicMock(return_value={
        "total_analyses": 10,
        "average_processing_time_ms": 150.5,
        "cache_hit_rate": 0.25,
    })
    
    return service


@pytest.fixture
def mock_client_manager(mock_content_service):
    """Create mock client manager."""
    manager = AsyncMock()
    manager.get_content_intelligence_service = AsyncMock(return_value=mock_content_service)
    return manager


@pytest.fixture
def mock_context():
    """Create mock MCP context."""
    context = AsyncMock()
    context.info = AsyncMock()
    context.debug = AsyncMock()
    context.warning = AsyncMock()
    context.error = AsyncMock()
    return context


class TestContentIntelligenceMCPTools:
    """Test Content Intelligence MCP tools."""

    def test_register_tools(self, mock_mcp, mock_client_manager):
        """Test that tools are registered with MCP server."""
        register_tools(mock_mcp, mock_client_manager)
        
        # Should register all tools
        assert mock_mcp.tool.call_count == 6  # 6 tools total
        
        # Verify tool names (by checking decorator calls)
        tool_calls = mock_mcp.tool.call_args_list
        assert len(tool_calls) == 6

    async def test_analyze_content_intelligence_success(self, mock_mcp, mock_client_manager, mock_context, mock_content_service):
        """Test successful content intelligence analysis."""
        register_tools(mock_mcp, mock_client_manager)
        
        # Get the registered function
        analyze_func = mock_mcp.tool.call_args_list[0][1]['func']
        
        request = ContentIntelligenceAnalysisRequest(
            content="Test content for analysis",
            url="https://example.com/test",
            title="Test Page",
            confidence_threshold=0.8,
        )
        
        result = await analyze_func(request, mock_context)
        
        assert isinstance(result, ContentIntelligenceResult)
        assert result.success is True
        assert result.enriched_content is not None
        assert result.processing_time_ms == 150.0
        assert result.cache_hit is False
        assert result.error is None
        
        # Verify service was called correctly
        mock_content_service.analyze_content.assert_called_once()

    async def test_analyze_content_intelligence_service_unavailable(self, mock_mcp, mock_client_manager, mock_context):
        """Test analysis when service is unavailable."""
        mock_client_manager.get_content_intelligence_service.return_value = None
        
        register_tools(mock_mcp, mock_client_manager)
        analyze_func = mock_mcp.tool.call_args_list[0][1]['func']
        
        request = ContentIntelligenceAnalysisRequest(
            content="Test content",
            url="https://example.com/test",
        )
        
        result = await analyze_func(request, mock_context)
        
        assert result.success is False
        assert result.error == "Content Intelligence Service not initialized"
        mock_context.error.assert_called()

    async def test_analyze_content_intelligence_exception(self, mock_mcp, mock_client_manager, mock_context, mock_content_service):
        """Test analysis with service exception."""
        mock_content_service.analyze_content.side_effect = Exception("Service error")
        
        register_tools(mock_mcp, mock_client_manager)
        analyze_func = mock_mcp.tool.call_args_list[0][1]['func']
        
        request = ContentIntelligenceAnalysisRequest(
            content="Test content",
            url="https://example.com/test",
        )
        
        result = await analyze_func(request, mock_context)
        
        assert result.success is False
        assert "Service error" in result.error
        mock_context.error.assert_called()

    async def test_classify_content_type_success(self, mock_mcp, mock_client_manager, mock_context, mock_content_service):
        """Test successful content type classification."""
        register_tools(mock_mcp, mock_client_manager)
        
        # Get the classify function (2nd tool)
        classify_func = mock_mcp.tool.call_args_list[1][1]['func']
        
        request = ContentIntelligenceClassificationRequest(
            content="def hello(): print('Hello, World!')",
            url="https://github.com/user/repo/hello.py",
            title="Hello Script",
        )
        
        result = await classify_func(request, mock_context)
        
        assert isinstance(result, ContentClassification)
        assert result.primary_type == ContentType.DOCUMENTATION
        assert ContentType.DOCUMENTATION in result.confidence_scores
        
        mock_content_service.classify_content_type.assert_called_once_with(
            content=request.content,
            url=request.url,
            title=request.title,
        )

    async def test_classify_content_type_service_unavailable(self, mock_mcp, mock_client_manager, mock_context):
        """Test classification when service is unavailable."""
        mock_client_manager.get_content_intelligence_service.return_value = None
        
        register_tools(mock_mcp, mock_client_manager)
        classify_func = mock_mcp.tool.call_args_list[1][1]['func']
        
        request = ContentIntelligenceClassificationRequest(
            content="test content",
            url="https://example.com/test",
        )
        
        result = await classify_func(request, mock_context)
        
        assert result.primary_type == ContentType.UNKNOWN
        assert result.classification_reasoning == "Service not available"

    async def test_assess_content_quality_success(self, mock_mcp, mock_client_manager, mock_context, mock_content_service):
        """Test successful content quality assessment."""
        register_tools(mock_mcp, mock_client_manager)
        
        # Get the quality assessment function (3rd tool)
        assess_func = mock_mcp.tool.call_args_list[2][1]['func']
        
        request = ContentIntelligenceQualityRequest(
            content="High quality content with good structure and comprehensive information.",
            confidence_threshold=0.85,
            query_context="machine learning tutorial",
        )
        
        result = await assess_func(request, mock_context)
        
        assert isinstance(result, QualityScore)
        assert result.overall_score == 0.85
        assert result.completeness == 0.9
        
        mock_content_service.assess_extraction_quality.assert_called_once_with(
            content=request.content,
            confidence_threshold=request.confidence_threshold,
            query_context=request.query_context,
            extraction_metadata=request.extraction_metadata,
        )

    async def test_assess_content_quality_service_unavailable(self, mock_mcp, mock_client_manager, mock_context):
        """Test quality assessment when service is unavailable."""
        mock_client_manager.get_content_intelligence_service.return_value = None
        
        register_tools(mock_mcp, mock_client_manager)
        assess_func = mock_mcp.tool.call_args_list[2][1]['func']
        
        request = ContentIntelligenceQualityRequest(
            content="test content",
        )
        
        result = await assess_func(request, mock_context)
        
        assert result.overall_score == 0.1
        assert "Service not available" in result.quality_issues

    async def test_extract_content_metadata_success(self, mock_mcp, mock_client_manager, mock_context, mock_content_service):
        """Test successful metadata extraction."""
        register_tools(mock_mcp, mock_client_manager)
        
        # Get the metadata extraction function (4th tool)
        extract_func = mock_mcp.tool.call_args_list[3][1]['func']
        
        request = ContentIntelligenceMetadataRequest(
            content="Content with metadata to extract",
            url="https://example.com/page",
            raw_html="<html><head><title>Test Page</title></head></html>",
        )
        
        result = await extract_func(request, mock_context)
        
        assert isinstance(result, ContentMetadata)
        assert result.title == "Test Page"  # From mock
        assert result.word_count == 100
        
        mock_content_service.extract_metadata.assert_called_once_with(
            content=request.content,
            url=request.url,
            raw_html=request.raw_html,
            extraction_metadata=request.extraction_metadata,
        )

    async def test_extract_content_metadata_service_unavailable(self, mock_mcp, mock_client_manager, mock_context):
        """Test metadata extraction when service is unavailable."""
        mock_client_manager.get_content_intelligence_service.return_value = None
        
        register_tools(mock_mcp, mock_client_manager)
        extract_func = mock_mcp.tool.call_args_list[3][1]['func']
        
        request = ContentIntelligenceMetadataRequest(
            content="test content with multiple words here",
            url="https://example.com/test",
        )
        
        result = await extract_func(request, mock_context)
        
        assert result.title is None  # No title in fallback
        assert result.word_count == 6  # "test content with multiple words here"
        assert result.char_count == len(request.content)

    async def test_get_adaptation_recommendations_success(self, mock_mcp, mock_client_manager, mock_context, mock_content_service):
        """Test successful adaptation recommendations."""
        register_tools(mock_mcp, mock_client_manager)
        
        # Get the adaptation recommendations function (5th tool)
        adapt_func = mock_mcp.tool.call_args_list[4][1]['func']
        
        result = await adapt_func(
            url="https://github.com/user/repo",
            content_patterns=["known_site:github.com"],
            quality_issues=["incomplete content"],
            ctx=mock_context,
        )
        
        assert isinstance(result, list)
        assert len(result) == 1
        assert result[0]["strategy"] == "extract_main_content"
        assert result[0]["priority"] == "high"
        
        mock_content_service.recommend_adaptations.assert_called_once()

    async def test_get_adaptation_recommendations_service_unavailable(self, mock_mcp, mock_client_manager, mock_context):
        """Test adaptation recommendations when service is unavailable."""
        mock_client_manager.get_content_intelligence_service.return_value = None
        
        register_tools(mock_mcp, mock_client_manager)
        adapt_func = mock_mcp.tool.call_args_list[4][1]['func']
        
        result = await adapt_func(
            url="https://example.com/test",
            ctx=mock_context,
        )
        
        assert result == []
        mock_context.error.assert_called()

    async def test_get_content_intelligence_metrics_success(self, mock_mcp, mock_client_manager, mock_context, mock_content_service):
        """Test successful metrics retrieval."""
        register_tools(mock_mcp, mock_client_manager)
        
        # Get the metrics function (6th tool)
        metrics_func = mock_mcp.tool.call_args_list[5][1]['func']
        
        result = await metrics_func(ctx=mock_context)
        
        assert isinstance(result, dict)
        assert result["service_available"] is True
        assert result["total_analyses"] == 10
        assert result["average_processing_time_ms"] == 150.5
        assert result["cache_hit_rate"] == 0.25
        
        mock_content_service.get_performance_metrics.assert_called_once()

    async def test_get_content_intelligence_metrics_service_unavailable(self, mock_mcp, mock_client_manager, mock_context):
        """Test metrics retrieval when service is unavailable."""
        mock_client_manager.get_content_intelligence_service.return_value = None
        
        register_tools(mock_mcp, mock_client_manager)
        metrics_func = mock_mcp.tool.call_args_list[5][1]['func']
        
        result = await metrics_func(ctx=mock_context)
        
        assert result["service_available"] is False
        assert result["error"] == "Service not initialized"

    async def test_tool_error_handling(self, mock_mcp, mock_client_manager, mock_context, mock_content_service):
        """Test error handling in tools."""
        mock_content_service.classify_content_type.side_effect = Exception("Classification error")
        
        register_tools(mock_mcp, mock_client_manager)
        classify_func = mock_mcp.tool.call_args_list[1][1]['func']
        
        request = ContentIntelligenceClassificationRequest(
            content="test content",
            url="https://example.com/test",
        )
        
        result = await classify_func(request, mock_context)
        
        assert result.primary_type == ContentType.UNKNOWN
        assert "Classification error" in result.classification_reasoning
        mock_context.error.assert_called()

    async def test_context_logging(self, mock_mcp, mock_client_manager, mock_context, mock_content_service):
        """Test that tools properly log to context."""
        register_tools(mock_mcp, mock_client_manager)
        analyze_func = mock_mcp.tool.call_args_list[0][1]['func']
        
        request = ContentIntelligenceAnalysisRequest(
            content="Test content for logging",
            url="https://example.com/logging-test",
        )
        
        await analyze_func(request, mock_context)
        
        # Should have called info for start and completion
        assert mock_context.info.call_count >= 2
        
        # Check that URL was logged
        info_calls = [call[0][0] for call in mock_context.info.call_args_list]
        assert any("https://example.com/logging-test" in call for call in info_calls)

    async def test_tool_parameter_validation(self, mock_mcp, mock_client_manager, mock_context, mock_content_service):
        """Test that tools handle parameter validation correctly."""
        register_tools(mock_mcp, mock_client_manager)
        
        # Test each tool with valid parameters
        analyze_func = mock_mcp.tool.call_args_list[0][1]['func']
        classify_func = mock_mcp.tool.call_args_list[1][1]['func']
        assess_func = mock_mcp.tool.call_args_list[2][1]['func']
        extract_func = mock_mcp.tool.call_args_list[3][1]['func']
        
        # All should handle their respective request types
        analysis_request = ContentIntelligenceAnalysisRequest(
            content="test", url="https://example.com"
        )
        classification_request = ContentIntelligenceClassificationRequest(
            content="test", url="https://example.com"
        )
        quality_request = ContentIntelligenceQualityRequest(content="test")
        metadata_request = ContentIntelligenceMetadataRequest(
            content="test", url="https://example.com"
        )
        
        # Should not raise exceptions
        await analyze_func(analysis_request, mock_context)
        await classify_func(classification_request, mock_context)
        await assess_func(quality_request, mock_context)
        await extract_func(metadata_request, mock_context)