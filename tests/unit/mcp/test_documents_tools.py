"""Comprehensive tests for documents tools module."""

from unittest.mock import AsyncMock
from unittest.mock import Mock
from unittest.mock import patch

import pytest
from fastmcp import Context
from src.config.enums import ChunkingStrategy
from src.infrastructure.client_manager import ClientManager
from src.mcp.models.requests import BatchRequest
from src.mcp.models.requests import DocumentRequest
from src.mcp.tools import documents


class TestDocumentsTools:
    """Test documents tool functions."""

    @pytest.fixture
    def mock_client_manager(self):
        """Create mock client manager."""
        cm = Mock(spec=ClientManager)

        # Mock services
        cm.crawl_manager = AsyncMock()
        cm.embedding_manager = AsyncMock()
        cm.qdrant_service = AsyncMock()
        cm.cache_manager = AsyncMock()

        return cm

    @pytest.fixture
    def mock_context(self):
        """Create mock MCP context."""
        ctx = Mock(spec=Context)
        ctx.info = AsyncMock()
        ctx.debug = AsyncMock()
        ctx.warning = AsyncMock()
        ctx.error = AsyncMock()
        return ctx

    @pytest.fixture
    def mock_mcp(self):
        """Create mock MCP instance that captures registered tools."""
        mcp = Mock()
        mcp._tools = {}

        def tool_decorator(func=None, **kwargs):
            def wrapper(f):
                mcp._tools[f.__name__] = f
                return f

            return wrapper if func is None else wrapper(func)

        mcp.tool = tool_decorator
        return mcp

    def test_register_tools(self, mock_mcp, mock_client_manager):
        """Test that document tools are registered correctly."""
        # Register tools
        documents.register_tools(mock_mcp, mock_client_manager)

        # Check that tools were registered
        assert "add_document" in mock_mcp._tools
        assert "add_documents_batch" in mock_mcp._tools

    @pytest.mark.asyncio
    async def test_add_document_basic(
        self, mock_mcp, mock_client_manager, mock_context
    ):
        """Test basic add_document functionality."""
        # Setup mocks
        with (
            patch("src.mcp.tools.documents.SecurityValidator") as MockSecurityValidator,
            patch("src.mcp.tools.documents.EnhancedChunker") as MockChunker,
            patch("src.mcp.tools.documents.uuid4", return_value="test-doc-id"),
        ):
            # Configure security validator
            mock_validator = Mock()
            mock_validator.validate_url = Mock(return_value="https://example.com")
            MockSecurityValidator.from_unified_config.return_value = mock_validator

            # Configure chunker
            mock_chunker = Mock()
            mock_chunker.chunk_content = Mock(
                return_value=[
                    {"content": "Chunk 1", "metadata": {"chunk_index": 0}},
                    {"content": "Chunk 2", "metadata": {"chunk_index": 1}},
                ]
            )
            MockChunker.return_value = mock_chunker

            # Configure cache (not found)
            mock_client_manager.cache_manager.get = AsyncMock(return_value=None)
            mock_client_manager.cache_manager.set = AsyncMock()

            # Configure crawl manager
            mock_client_manager.crawl_manager.crawl_single = AsyncMock(
                return_value=Mock(
                    markdown="Test content from example.com",
                    metadata={
                        "url": "https://example.com",
                        "title": "Example Page",
                    },
                )
            )

            # Configure embedding manager
            mock_client_manager.embedding_manager.generate_embeddings = AsyncMock(
                return_value=[[0.1, 0.2], [0.3, 0.4]]
            )

            # Configure qdrant service
            mock_client_manager.qdrant_service.create_collection = AsyncMock()
            mock_client_manager.qdrant_service.upsert_points = AsyncMock()

            # Register tools
            documents.register_tools(mock_mcp, mock_client_manager)

            # Get the registered function
            add_func = mock_mcp._tools["add_document"]

            # Create request
            request = DocumentRequest(
                url="https://example.com",
                collection="test_collection",
                chunk_strategy=ChunkingStrategy.ENHANCED,
            )

            # Call the function
            result = await add_func(request, mock_context)

            # Verify results
            assert result["url"] == "https://example.com"
            assert result["chunks_created"] == 2
            assert result["title"] == "Example Page"
            assert result["collection"] == "test_collection"
            assert result["chunking_strategy"] == "enhanced"
            assert result["embedding_dimensions"] == 2

    @pytest.mark.asyncio
    async def test_add_document_crawl_failure(
        self, mock_mcp, mock_client_manager, mock_context
    ):
        """Test add_document when crawl fails."""
        with patch(
            "src.mcp.tools.documents.SecurityValidator"
        ) as MockSecurityValidator:
            # Configure security validator
            mock_validator = Mock()
            mock_validator.validate_url = Mock(return_value="https://example.com")
            MockSecurityValidator.from_unified_config.return_value = mock_validator

            # Configure cache (not found)
            mock_client_manager.cache_manager.get = AsyncMock(return_value=None)

            # Configure crawl to fail
            mock_client_manager.crawl_manager.crawl_single = AsyncMock(
                return_value=None
            )

            # Register tools
            documents.register_tools(mock_mcp, mock_client_manager)

            # Get the registered function
            add_func = mock_mcp._tools["add_document"]

            # Create request
            request = DocumentRequest(
                url="https://example.com", collection="test_collection"
            )

            # Call the function - should raise exception
            with pytest.raises(ValueError, match="Failed to crawl"):
                await add_func(request, mock_context)

    @pytest.mark.asyncio
    async def test_add_documents_batch(
        self, mock_mcp, mock_client_manager, mock_context
    ):
        """Test add_documents_batch functionality."""
        with (
            patch("src.mcp.tools.documents.SecurityValidator") as MockSecurityValidator,
            patch("src.mcp.tools.documents.EnhancedChunker") as MockChunker,
        ):
            # Configure mocks
            mock_validator = Mock()
            mock_validator.validate_url = Mock(side_effect=lambda url: url)
            MockSecurityValidator.from_unified_config.return_value = mock_validator

            # Configure chunker
            mock_chunker = Mock()
            mock_chunker.chunk_content = Mock(
                return_value=[{"content": "Chunk", "metadata": {}}]
            )
            MockChunker.return_value = mock_chunker

            # Configure cache (not found)
            mock_client_manager.cache_manager.get = AsyncMock(return_value=None)
            mock_client_manager.cache_manager.set = AsyncMock()

            # Configure crawl results
            crawl_results = [
                Mock(
                    markdown="Content 1",
                    metadata={"url": "https://example1.com", "title": "Page 1"},
                ),
                None,  # Second URL fails to crawl
            ]
            mock_client_manager.crawl_manager.crawl_single = AsyncMock(
                side_effect=crawl_results
            )

            # Configure embedding manager
            mock_client_manager.embedding_manager.generate_embeddings = AsyncMock(
                return_value=[[0.1, 0.2]]
            )

            # Configure qdrant service
            mock_client_manager.qdrant_service.create_collection = AsyncMock()
            mock_client_manager.qdrant_service.upsert_points = AsyncMock()

            # Register tools
            documents.register_tools(mock_mcp, mock_client_manager)

            # Get the registered batch function
            batch_func = mock_mcp._tools["add_documents_batch"]

            # Create request
            request = BatchRequest(
                urls=["https://example1.com", "https://example2.com"],
                collection="test_collection",
                max_concurrent=2,
            )

            # Call the function with ctx parameter
            result = await batch_func(request, mock_context)

            # Verify results
            assert result["total"] == 2
            assert len(result["successful"]) == 1
            assert len(result["failed"]) == 1
            assert result["successful"][0]["url"] == "https://example1.com"
            assert result["failed"][0]["url"] == "https://example2.com"
            assert "Failed to crawl" in result["failed"][0]["error"]

    @pytest.mark.asyncio
    async def test_add_document_security_validation(
        self, mock_mcp, mock_client_manager, mock_context
    ):
        """Test add_document with security validation failure."""
        with patch(
            "src.mcp.tools.documents.SecurityValidator"
        ) as MockSecurityValidator:
            # Configure security validator to raise error
            mock_validator = Mock()
            mock_validator.validate_url = Mock(
                side_effect=Exception("Dangerous URL detected")
            )
            MockSecurityValidator.from_unified_config.return_value = mock_validator

            # Register tools
            documents.register_tools(mock_mcp, mock_client_manager)

            # Get the registered function
            add_func = mock_mcp._tools["add_document"]

            # Create request
            request = DocumentRequest(
                url="https://malicious.com", collection="test_collection"
            )

            # Call the function - should raise exception
            with pytest.raises(Exception, match="Dangerous URL detected"):
                await add_func(request, mock_context)

    @pytest.mark.asyncio
    async def test_add_document_with_cache_hit(
        self, mock_mcp, mock_client_manager, mock_context
    ):
        """Test add_document when document is in cache."""
        with patch(
            "src.mcp.tools.documents.SecurityValidator"
        ) as MockSecurityValidator:
            # Configure security validator
            mock_validator = Mock()
            mock_validator.validate_url = Mock(return_value="https://example.com")
            MockSecurityValidator.from_unified_config.return_value = mock_validator

            # Configure cache (found)
            cached_result = {
                "url": "https://example.com",
                "title": "Cached Page",
                "chunks_created": 3,
                "collection": "test_collection",
            }
            mock_client_manager.cache_manager.get = AsyncMock(
                return_value=cached_result
            )

            # Register tools
            documents.register_tools(mock_mcp, mock_client_manager)

            # Get the registered function
            add_func = mock_mcp._tools["add_document"]

            # Create request
            request = DocumentRequest(
                url="https://example.com", collection="test_collection"
            )

            # Call the function
            result = await add_func(request, mock_context)

            # Verify cached result returned
            assert result == cached_result

            # Verify crawl was not called
            mock_client_manager.crawl_manager.crawl_single.assert_not_called()
