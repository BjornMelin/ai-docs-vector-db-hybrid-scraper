"""Simple tests for Content Intelligence MCP tools."""

from unittest.mock import MagicMock

import pytest
from src.mcp_tools.models.requests import ContentIntelligenceAnalysisRequest
from src.mcp_tools.tools.content_intelligence import register_tools


class TestContentIntelligenceMCPTools:
    """Simple tests for content intelligence MCP tools."""

    def test_register_tools_succeeds(self):
        """Test that register_tools completes without error."""
        mock_mcp = MagicMock()
        mock_client_manager = MagicMock()

        # Should not raise any exceptions
        register_tools(mock_mcp, mock_client_manager)

        # Verify tools were registered (mcp.tool decorator was called)
        assert mock_mcp.tool.called

    def test_content_intelligence_request_model(self):
        """Test ContentIntelligenceAnalysisRequest model validation."""
        request = ContentIntelligenceAnalysisRequest(
            content="Test content for analysis",
            url="https://example.com/test",
            title="Test Page",
            confidence_threshold=0.8,
        )

        assert request.content == "Test content for analysis"
        assert request.url == "https://example.com/test"
        assert request.title == "Test Page"
        assert request.confidence_threshold == 0.8

    def test_content_intelligence_request_validation(self):
        """Test ContentIntelligenceAnalysisRequest field validation."""
        # Test invalid confidence threshold
        with pytest.raises(ValueError):
            ContentIntelligenceAnalysisRequest(
                content="Test content",
                url="https://example.com/test",
                confidence_threshold=1.5,  # Invalid - must be <= 1.0
            )

        with pytest.raises(ValueError):
            ContentIntelligenceAnalysisRequest(
                content="Test content",
                url="https://example.com/test",
                confidence_threshold=-0.1,  # Invalid - must be >= 0.0
            )
