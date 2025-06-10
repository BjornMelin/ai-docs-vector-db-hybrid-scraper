"""Simple tests for Document MCP tools."""

from unittest.mock import MagicMock

from src.mcp_tools.models.requests import DocumentRequest
from src.mcp_tools.tools.documents import register_tools


class TestDocumentMCPTools:
    """Simple tests for document MCP tools."""

    def test_register_tools_succeeds(self):
        """Test that register_tools completes without error."""
        mock_mcp = MagicMock()
        mock_client_manager = MagicMock()

        # Should not raise any exceptions
        register_tools(mock_mcp, mock_client_manager)

        # Verify tools were registered (mcp.tool decorator was called)
        assert mock_mcp.tool.called

    def test_document_request_model(self):
        """Test DocumentRequest model validation."""
        request = DocumentRequest(
            url="https://example.com/test-doc", collection="test_collection"
        )

        assert request.url == "https://example.com/test-doc"
        assert request.collection == "test_collection"

    def test_document_request_validation(self):
        """Test DocumentRequest field validation."""
        # Test valid minimal request
        request = DocumentRequest(
            url="https://example.com/minimal", collection="minimal_test"
        )
        assert request.url == "https://example.com/minimal"
        assert request.collection == "minimal_test"
